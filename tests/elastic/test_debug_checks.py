# MIT License
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Test the `EP_DEBUG_CHECKS` reports. The test compiles the kernels with the checks, then:
#   1. runs a clean dispatch and combine, and expects no report,
#   2. puts one NaN in the dispatch input (BF16 and FP8), and expects a report on each rank
#      that receives the token, with the correct source and element,
#   3. puts one NaN in the combine input, and expects one report on the rank that owns the token.
# The reports come from device printf, so the test captures file descriptor 1.

import argparse
import contextlib
import ctypes
import os
import re
import tempfile
import torch
import torch.distributed as dist

# The flag must be set before the first kernel compiles
os.environ['EP_DEBUG_CHECKS'] = '1'

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.math import per_token_cast_to_fp8

_REPORT = 'DeepEP debug check:'
_DISPATCH_REPORT = re.compile(r'non-finite token in dispatch, rank: (\d+), recv token: \d+/\d+, src rank: (\d+), '
                              r'src token: (\d+), local expert: -?\d+, bad elements: (\d+)/\d+, first bad: (\d+)')
_COMBINE_REPORT = re.compile(r'non-finite token in combine, rank: (\d+), token: (\d+)/\d+, sources: \d+, '
                             r'bad elements: (\d+)/\d+, first bad: (\d+)')


@contextlib.contextmanager
def capture_device_printf(lines: list):
    """Capture what this process writes to file descriptor 1, device printf included."""
    libc = ctypes.CDLL(None)
    torch.cuda.synchronize()
    libc.fflush(None)
    with tempfile.TemporaryFile(mode='w+') as f:
        saved_fd = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            yield
            torch.cuda.synchronize()
            libc.fflush(None)
        finally:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        f.seek(0)
        lines.extend(line for line in f.read().splitlines() if _REPORT in line)


def expect(condition: bool, message: str):
    if not condition:
        raise AssertionError(f'rank {dist.get_rank()}: {message}')


# noinspection PyShadowingNames
def test_debug_checks(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    rank, num_ranks = dist.get_rank(), dist.get_world_size()
    num_tokens, hidden, num_experts = args.num_tokens, args.hidden, args.num_experts
    num_local_experts = num_experts // num_ranks
    src_rank, src_token, bad_elem = 0, 7, 100

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda').abs()
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    # The ranks that receive the planted token
    planted_topk_idx = topk_idx[src_token].clone()
    dist.broadcast(planted_topk_idx, src=src_rank)
    receivers = set((planted_topk_idx[planted_topk_idx >= 0] // num_local_experts).tolist())

    def dispatch(x):
        return buffer.dispatch(x=x, topk_idx=topk_idx, topk_weights=topk_weights,
                               num_max_tokens_per_rank=num_tokens, num_experts=num_experts)

    def combine(x, handle, recv_topk_weights):
        return buffer.combine(x=x, handle=handle, topk_weights=recv_topk_weights)

    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    # Clean data: no report from any check
    dist_print(' > Clean dispatch and combine', once_in_node=True)
    lines = []
    with capture_device_printf(lines):
        recv_x, _, recv_topk_weights, handle, _ = dispatch(x)
        combine(recv_x, handle, recv_topk_weights)
        dispatch(per_token_cast_to_fp8(x))
    expect(not lines, f'clean data gave reports: {lines}')
    dist.barrier()

    # One NaN in the dispatch input, BF16 and FP8
    for name, planted_x, bits in (('BF16', x, '0x7fc0'), ('FP8', per_token_cast_to_fp8(x), '0x7f')):
        dist_print(f' > Dispatch with a NaN, {name}', once_in_node=True)
        planted = planted_x.clone() if name == 'BF16' else (planted_x[0].clone(), planted_x[1])
        if rank == src_rank:
            (planted if name == 'BF16' else planted[0])[src_token, bad_elem] = float('nan')
        lines = []
        with capture_device_printf(lines):
            dispatch(planted)
        reports = [_DISPATCH_REPORT.search(line) for line in lines]
        expect(all(reports), f'{name}: unexpected report: {lines}')
        if rank in receivers:
            expect(len(reports) == 1, f'{name}: expected 1 report, got {lines}')
            got = tuple(int(v) for v in reports[0].groups())
            expect(got == (rank, src_rank, src_token, 1, bad_elem),
                   f'{name}: expected (rank, src rank, src token, bad, first bad) = '
                   f'{(rank, src_rank, src_token, 1, bad_elem)}, got {got}')
            expect(f'bits {bits}' in lines[0], f'{name}: expected bits {bits}: {lines[0]}')
        else:
            expect(not reports, f'{name}: rank does not receive the token, but reported: {lines}')
        dist.barrier()

    # One NaN in the combine input, at the first received token of `src_rank`
    dist_print(' > Combine with a NaN', once_in_node=True)
    recv_x, _, recv_topk_weights, handle, _ = dispatch(x)
    num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    combine_x = torch.zeros((recv_x.shape[0], hidden), dtype=torch.bfloat16, device='cuda')
    combine_x[:num_recv_tokens] = recv_x[:num_recv_tokens]
    owner = torch.tensor([-1, -1], dtype=torch.long, device='cuda')
    if rank == src_rank:
        expect(num_recv_tokens > 0, 'source rank received no token')
        combine_x[0, bad_elem] = float('nan')
        global_idx = handle.recv_src_metadata[0, 0].item()
        owner = torch.tensor([global_idx // num_tokens, global_idx % num_tokens], dtype=torch.long, device='cuda')
    dist.broadcast(owner, src=src_rank)
    owner_rank, owner_token = owner.tolist()
    lines = []
    with capture_device_printf(lines):
        combine(combine_x, handle, recv_topk_weights)
    reports = [_COMBINE_REPORT.search(line) for line in lines]
    expect(all(reports), f'combine: unexpected report: {lines}')
    if rank == owner_rank:
        expect(len(reports) == 1, f'combine: expected 1 report, got {lines}')
        got = tuple(int(v) for v in reports[0].groups())
        expect(got == (rank, owner_token, 1, bad_elem),
               f'combine: expected (rank, token, bad, first bad) = {(rank, owner_token, 1, bad_elem)}, got {got}')
    else:
        expect(not reports, f'combine: rank does not own the token, but reported: {lines}')
    dist.barrier()
    dist_print(' > All debug checks passed', once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.ElasticBuffer(group, num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                   num_topk=args.num_topk, prefer_overlap_with_compute=False, sl_idx=0,
                                   explicitly_destroy=True)
    test_debug_checks(buffer, args)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the EP_DEBUG_CHECKS reports')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=1024, help='Number of tokens')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
