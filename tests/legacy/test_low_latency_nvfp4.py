import argparse
import random
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import calc_diff
from deep_ep.utils.testing import bench_kineto


def wait(event, hook, async_finish: bool, return_recv_hook: bool):
    if return_recv_hook:
        hook()
    elif async_finish:
        event.current_stream_wait()


def pad(t: torch.Tensor, num_rows: int, value: int = 0):
    return torch.cat((t, torch.full((num_rows - t.size(0), *t.shape[1:]), value, dtype=t.dtype, device='cuda')))


def unswizzle_scales(sf: torch.Tensor, num_rows: int, hidden: int):
    # (32, 4, rm, 4, rk, l) view of physical (l, rm, rk, 32, 4, 4) -> (l, rows, hidden // 16)
    l = sf.size(-1)
    sf = sf.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8)
    sf = sf.view(l, -1, hidden // 64, 32, 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(l, -1, hidden // 16)
    return sf[:, :num_rows]


def make_inputs(num_tokens: int, hidden: int, num_experts: int, num_topk: int, rank: int, num_ranks: int, pattern: str):
    # Tag each token with its global index in the first packed word
    q = torch.randint(0, 256, (num_tokens, hidden // 2), dtype=torch.uint8, device='cuda')
    tags = torch.arange(num_tokens, dtype=torch.int32, device='cuda') + rank * num_tokens
    q[:, :4] = tags.view(torch.uint8).view(-1, 4)
    sf = torch.randint(0, 256, (num_tokens, hidden // 16), dtype=torch.uint8, device='cuda')
    # Arbitrary bit patterns (including NaN/Inf) must survive transport
    outer_bits = torch.randint(-(2**31), 2**31 - 1, (num_tokens, ), dtype=torch.int32, device='cuda')

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(deep_ep.topk_idx_t)
    if pattern == 'masked':
        for _ in range(min(10, num_tokens)):
            topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1
    elif pattern == 'single_expert':
        topk_idx.fill_(-1)
        topk_idx[:, 0] = 0
    elif pattern == 'invalid':
        topk_idx.fill_(-1)
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='cuda') + 0.1
    topk_weights.masked_fill_(topk_idx < 0, 0)
    return q, sf, outer_bits, tags, topk_idx, topk_weights


def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    num_local_experts = num_experts // num_ranks

    for pattern in ('masked', 'single_expert', 'invalid'):
        for count in (0, 1, num_tokens // 2, num_tokens):
            q, sf, outer_bits, tags, topk_idx, topk_weights = make_inputs(num_tokens, hidden, num_experts, num_topk, rank, num_ranks,
                                                                          pattern)
            # Ranks may send different numbers of tokens; rank 0 also covers the empty case
            n = 0 if rank == 0 and count == num_tokens // 2 else count
            q, sf, outer_bits, tags, topk_idx, topk_weights = (t[:n].contiguous()
                                                               for t in (q, sf, outer_bits, tags, topk_idx, topk_weights))

            # Gather references from all ranks
            all_q = [torch.empty((num_tokens, hidden // 2), dtype=torch.uint8, device='cuda') for _ in range(num_ranks)]
            all_sf = [torch.empty((num_tokens, hidden // 16), dtype=torch.uint8, device='cuda') for _ in range(num_ranks)]
            all_outer = [torch.empty((num_tokens, ), dtype=torch.int32, device='cuda') for _ in range(num_ranks)]
            all_idx = [torch.full((num_tokens, num_topk), -1, dtype=topk_idx.dtype, device='cuda') for _ in range(num_ranks)]
            dist.all_gather(all_q, pad(q, num_tokens), group=group)
            dist.all_gather(all_sf, pad(sf, num_tokens), group=group)
            dist.all_gather(all_outer, pad(outer_bits, num_tokens), group=group)
            dist.all_gather(all_idx, pad(topk_idx, num_tokens, -1), group=group)
            all_q, all_sf, all_outer, all_idx = (torch.cat(t) for t in (all_q, all_sf, all_outer, all_idx))

            for with_outer, sf_dtype, swizzle in ((True, torch.uint8, False), (False, torch.uint8, False),
                                                  (True, torch.float8_e4m3fn, False), (True, torch.uint8, True),
                                                  (False, torch.float8_e4m3fn, True)):
                for async_finish, return_recv_hook in ((False, False), (True, False), (False, True)):
                    (recv_q, recv_sf, recv_outer), recv_count, handle, event, hook = buffer.low_latency_dispatch_fp4(
                        q,
                        sf.view(sf_dtype),
                        topk_idx,
                        num_tokens,
                        num_experts,
                        per_token_scale=outer_bits.view(torch.float32) if with_outer else None,
                        async_finish=async_finish,
                        return_recv_hook=return_recv_hook,
                        swizzle_scales=swizzle)
                    wait(event, hook, async_finish, return_recv_hook)

                    num_rows = num_ranks * num_tokens
                    assert recv_q.shape == (num_local_experts, num_rows, hidden // 2)
                    assert (recv_outer is not None) == with_outer
                    assert recv_sf.dtype == sf_dtype
                    if swizzle:
                        num_tiles = (num_rows + 127) // 128
                        assert recv_sf.shape == (32, 4, num_tiles, 4, hidden // 64, num_local_experts)
                        recv_sf = unswizzle_scales(recv_sf, num_rows, hidden)
                    else:
                        assert recv_sf.shape == (num_local_experts, num_rows, hidden // 16)
                        recv_sf = recv_sf.view(torch.uint8)

                    # Check counts and bitwise payloads
                    for i in range(num_local_experts):
                        expert_id = rank * num_local_experts + i
                        num_valid = int(recv_count[i].item())
                        assert num_valid == int((all_idx == expert_id).sum().item()), f'{expert_id=}, {num_valid=}'
                        rows = recv_q[i, :num_valid, :4].contiguous().view(torch.int32).flatten().long()
                        assert rows.unique().numel() == num_valid
                        assert torch.equal(recv_q[i, :num_valid], all_q[rows]), 'Packed values corrupted'
                        assert torch.equal(recv_sf[i, :num_valid], all_sf[rows]), 'Block scales corrupted'
                        if with_outer:
                            assert torch.equal(recv_outer[i, :num_valid].view(torch.int32), all_outer[rows]), 'Per-token scales corrupted'

                    # The handle must work with BF16 combine
                    valid = torch.arange(num_ranks * num_tokens, device='cuda')[None, :] < recv_count[:, None]
                    recv_tags = recv_q[:, :, :4].contiguous().view(torch.int32).squeeze(-1)
                    out = (recv_tags % 64).masked_fill(~valid, 0).to(torch.bfloat16)
                    out = out.unsqueeze(-1).expand(-1, -1, hidden).contiguous()
                    combined_x, event, hook = buffer.low_latency_combine(out,
                                                                         topk_idx,
                                                                         topk_weights,
                                                                         handle,
                                                                         async_finish=async_finish,
                                                                         return_recv_hook=return_recv_hook)
                    wait(event, hook, async_finish, return_recv_hook)
                    ref = ((tags % 64).float() * topk_weights.sum(-1)).unsqueeze(-1).expand(-1, hidden)
                    if n > 0:
                        assert torch.isnan(combined_x).sum().item() == 0
                        diff = calc_diff(combined_x.float(), ref)
                        assert diff < 1e-5, f'Combine mismatch: {diff=}'
            if rank == 0:
                print(f' > pattern={pattern}, count={count}: passed', flush=True)

    # Compare dispatch kernel time with BF16, in-kernel NVFP4 cast and pre-quantized NVFP4
    q, sf, outer_bits, _, topk_idx, _ = make_inputs(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, 'masked')
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_global_scale = torch.ones((), dtype=torch.float32, device='cuda')
    funcs = {
        'BF16':
        lambda: buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts, use_fp8=False),
        'NVFP4 (cast)':
        lambda: buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, x_global_scale=x_global_scale, use_fp8=False, use_nvfp4=True),
        'NVFP4 (pre-quantized)':
        lambda: buffer.low_latency_dispatch_fp4(q, sf, topk_idx, num_tokens, num_experts, per_token_scale=outer_bits.view(torch.float32)),
        'NVFP4 (pre-quantized, swizzled scales)':
        lambda: buffer.low_latency_dispatch_fp4(
            q, sf, topk_idx, num_tokens, num_experts, per_token_scale=outer_bits.view(torch.float32), swizzle_scales=True),
    }
    for name, func in funcs.items():
        group.barrier()
        t = bench_kineto(func, kernel_names='dispatch', barrier_comm_profiling=True, suppress_kineto_output=True)
        if rank == 0:
            print(f' > {name} dispatch: {t * 1e6:.2f} us', flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=1)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test low-latency dispatch of pre-quantized NVFP4')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
