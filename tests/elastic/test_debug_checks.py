# MIT License
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Test the `token_integrity` and `token_sync` checks of `EP_DEBUG_CHECKS`. The test compiles the kernels
# with the checks. For `token_integrity`, it:
#   1. runs a clean dispatch and combine, and expects no report,
#   2. puts one NaN in the dispatch input (BF16 and FP8), and expects a report on each rank
#      that receives the token, with the correct source and element,
#   3. puts one NaN in the combine input, and expects one report on the rank that owns the token,
#   4. puts a NaN in every token, and expects the reports to stop at the rate limit, and the
#      next window to print how many reports were not printed.
# Before these steps, `test_corruption` corrupts tokens while dispatch and combine read them, and
# flips single bits, and compares the reports with a bitwise check against the clean data.
# For `token_sync`, `test_token_sync` runs dispatch and combine many times and expects
# no report. A bad signal traps the kernel, so a report also fails the run.
# The reports come from device printf, so the test captures file descriptor 1.

import argparse
import contextlib
import ctypes
import os
import re
import sys
import tempfile
import time
import torch
import torch.distributed as dist

# The flag must be set before the first kernel compiles
os.environ['EP_DEBUG_CHECKS'] = 'token_integrity,token_sync'

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.math import per_token_cast_to_fp8

# The rate limit in `deep_ep/include/deep_ep/common/diag.cuh`
_MAX_REPORTS_PER_WINDOW = 32
_REPORT_WINDOW_SECONDS = 2 ** 30 / 1e9

_REPORT = 'DeepEP debug check:'
_DISPATCH_REPORT = re.compile(r'non-finite token in dispatch, rank: (\d+), recv token: \d+/\d+, src rank: (\d+), '
                              r'src token: (\d+), local expert: -?\d+, bad elements: (\d+)/\d+, first bad: (\d+)')
_COMBINE_REPORT = re.compile(r'non-finite token in combine, rank: (\d+), token: (\d+)/\d+, sources: \d+, '
                             r'bad elements: (\d+)/\d+, first bad: (\d+)')
_NOT_PRINTED_REPORT = re.compile(r'(\d+) more reports were not printed')
_SIGNAL_REPORT = 'GIN signal skew'


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
        except BaseException:
            # A check that traps fails the step here: show what the kernels printed before the trap
            libc.fflush(None)
            f.seek(0)
            sys.stderr.write(f.read())
            raise
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
                               num_max_tokens_per_rank=num_tokens, num_experts=num_experts,
                               num_sms=args.num_sms)

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

    # A NaN in every token: the reports stop at the rate limit, and the next window prints how
    # many reports the earlier window did not print. When a new window starts during a burst,
    # the burst can print up to two windows of reports
    dist_print(' > Dispatch with a NaN in every token', once_in_node=True)
    all_bad_x = x.clone()
    all_bad_x[:, bad_elem] = float('nan')
    max_burst_reports = 2 * _MAX_REPORTS_PER_WINDOW
    num_not_printed = []
    for burst in range(2):
        if burst > 0:
            time.sleep(_REPORT_WINDOW_SECONDS * 1.2)
        lines = []
        with capture_device_printf(lines):
            _, _, _, handle, _ = dispatch(all_bad_x)
        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        num_reports = sum(1 for line in lines if _DISPATCH_REPORT.search(line))
        num_not_printed += [int(m.group(1)) for line in lines if (m := _NOT_PRINTED_REPORT.search(line))]
        expect(num_recv_tokens > max_burst_reports, f'too few received tokens for this test: {num_recv_tokens}')
        expect(0 < num_reports <= max_burst_reports,
               f'burst {burst}: expected 1 to {max_burst_reports} reports for {num_recv_tokens} bad tokens, '
               f'got {num_reports}')
    expect(num_not_printed and all(n > 0 for n in num_not_printed),
           f'expected a "more reports were not printed" line, got {num_not_printed}')
    dist.barrier()
    dist_print(' > All debug checks passed', once_in_node=True)


class ReportBudget:
    """Keeps the reports in each rate-limit window under the limit, so that a missing report is a real miss.
    Steps between two sleeps print at most half the limit, and a window can cover at most two such blocks."""

    def __init__(self):
        self.num_used = 0

    def reserve(self, num_reports: int):
        """`num_reports` is the most reports that any rank can print in the next step. All ranks pass the same value."""
        expect(num_reports <= _MAX_REPORTS_PER_WINDOW // 2, f'step can print too many reports: {num_reports}')
        if self.num_used + num_reports > _MAX_REPORTS_PER_WINDOW // 2:
            time.sleep(_REPORT_WINDOW_SECONDS * 1.1)
            self.num_used = 0
        self.num_used += num_reports


def nonfinite_rows(t: torch.Tensor) -> torch.Tensor:
    return ~torch.isfinite(t.float()).all(dim=-1)


# noinspection PyShadowingNames
def test_corruption(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    """Corrupt tokens and compare the debug check with a bitwise check against the clean data. Two parts:
      A. a second stream overwrites a few tokens in HBM after a random delay, while dispatch or combine
         reads them. The test does not know when the write lands, so it checks each received or combined
         token: the check must report exactly the tokens that differ from the clean data and hold a NaN
         or Inf. A changed token that holds only valid numbers is a miss that the check cannot see, and
         the test counts it.
      B. one bit flip in each of many tokens, for every bit of BF16 and FP8. The check must report
         exactly the flips that make a NaN or Inf (`torch.isfinite` is the reference)."""
    rank, num_ranks = dist.get_rank(), dist.get_world_size()
    num_tokens, hidden, num_experts = args.num_tokens, args.hidden, args.num_experts
    budget, side_stream = ReportBudget(), torch.cuda.Stream()

    # Inputs from a fixed seed, and a generator with the same seed on all ranks to pick the victims
    gen = torch.Generator(device='cuda')
    gen.manual_seed(1000 + rank)
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float, device='cuda', generator=gen)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda', generator=gen)
    victim_gen = torch.Generator(device='cpu')
    victim_gen.manual_seed(2000)

    def randint(high: int, size: int) -> list:
        return torch.randint(0, high, (size, ), generator=victim_gen, device='cpu').tolist()

    def pick_victims(high: int, size: int) -> list:
        """`size` different values in `[0, high)`, the same on all ranks"""
        return torch.randperm(high, generator=victim_gen, device='cpu')[:size].tolist()

    def dispatch(t):
        return buffer.dispatch(x=t, topk_idx=topk_idx, topk_weights=topk_weights,
                               num_max_tokens_per_rank=num_tokens, num_experts=num_experts, num_sms=args.num_sms)

    # One checksum for each token. A change of a token's bits changes it, except with a negligible chance
    hash_weights = torch.randint(1, 1 << 20, (hidden, ), generator=victim_gen, device='cpu').cuda()

    def checksums(t: torch.Tensor) -> torch.Tensor:
        return (t.view(torch.int16).to(torch.int64) * hash_weights).sum(-1)

    def overwrite_later(t: torch.Tensor, rows: list, values: list, delay_cycles: int):
        """On the side stream, after `delay_cycles`, write `values[i]` to row `rows[i]` of `t`"""
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            torch.cuda._sleep(delay_cycles)
            for row, value in zip(rows, values):
                t[row].copy_(value)
                # The caller drops `value` now: keep its memory until the side stream has used it
                value.record_stream(side_stream)
            t.record_stream(side_stream)

    def bad_value(pattern: int, inf_elem: int) -> torch.Tensor:
        """0: a NaN row, 1: one Inf element, 2: a row of other valid numbers"""
        if pattern == 2:
            return torch.randn((hidden, ), dtype=torch.bfloat16, device='cuda', generator=gen)
        value = torch.full((hidden, ), float('nan') if pattern == 0 else 0.0, dtype=torch.bfloat16, device='cuda')
        if pattern == 1:
            value[inf_elem] = float('inf')
        return value

    def reported_dispatch_sources(lines: list) -> set:
        reports = [_DISPATCH_REPORT.search(line) for line in lines]
        expect(all(reports), f'unexpected report: {lines}')
        return {int(m.group(2)) * num_tokens + int(m.group(3)) for m in reports}

    # A, dispatch: 2 victim tokens in each iteration, so no rank can print more than 2 reports
    dist_print(' > Corrupt the dispatch input while dispatch reads it', once_in_node=True)
    clean_checksums = torch.empty((num_ranks, num_tokens), dtype=torch.int64, device='cuda')
    dist.all_gather_into_tensor(clean_checksums, checksums(x))
    tally = torch.zeros(4, dtype=torch.int64, device='cuda')   # Received, read before the write, reported, missed
    for _ in range(args.num_corruption_iters):
        # All ranks draw the same values in the same order, so that the shared generator stays the same
        victims = pick_victims(num_ranks * num_tokens, 2)
        patterns, inf_elems, delay = randint(3, 2), randint(hidden, 2), randint(1 << 21, 1)[0]
        mine = [i for i, v in enumerate(victims) if v // num_tokens == rank]
        budget.reserve(2)
        x_run = x.clone()
        lines = []
        with capture_device_printf(lines):
            # Queue the write first: dispatch waits for the CPU, and the write must land while dispatch runs
            overwrite_later(x_run, [victims[i] % num_tokens for i in mine],
                            [bad_value(patterns[i], inf_elems[i]) for i in mine], delay)
            recv_x, _, _, handle, _ = dispatch(x_run)
            side_stream.synchronize()
        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        src = handle.recv_src_metadata[:num_recv_tokens, 0].long()
        changed = checksums(recv_x[:num_recv_tokens]) != clean_checksums.view(-1)[src]
        received = torch.isin(src, torch.tensor(victims, device='cuda'))
        expect(not (changed & ~received).any().item(), 'a token that was not corrupted changed')
        nonfinite = nonfinite_rows(recv_x[:num_recv_tokens])
        expected = set(src[changed & nonfinite].tolist())
        reported = reported_dispatch_sources(lines)
        expect(reported == expected, f'reported sources {sorted(reported)}, expected {sorted(expected)}')
        tally += torch.stack([received.sum(), (received & ~changed).sum(),
                              (changed & nonfinite).sum(), (changed & ~nonfinite).sum()])
    dist.all_reduce(tally)
    dist_print(f'   received copies of victim tokens: {tally[0].item()}, read before the write: {tally[1].item()}, '
               f'NaN/Inf and reported: {tally[2].item()}, changed but only valid numbers (cannot be seen): '
               f'{tally[3].item()}', once_in_node=True)

    # A, combine: 2 victim rows of the combine input in each iteration, so no rank owns more than 2 bad tokens
    dist_print(' > Corrupt the combine input while combine reads it', once_in_node=True)
    recv_x, _, recv_topk_weights, handle, _ = dispatch(x)
    num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    min_num_recv_tokens = torch.tensor([num_recv_tokens], device='cuda')
    dist.all_reduce(min_num_recv_tokens, op=dist.ReduceOp.MIN)
    min_num_recv_tokens = min_num_recv_tokens.item()

    def combine(t):
        return buffer.combine(x=t, handle=handle, topk_weights=recv_topk_weights)[0]

    clean_combined_x = combine(recv_x).clone()
    expect(torch.equal(combine(recv_x), clean_combined_x), 'two combines of the same data differ')
    tally = torch.zeros(3, dtype=torch.int64, device='cuda')   # Changed, reported, missed
    for _ in range(args.num_corruption_iters):
        victims = pick_victims(num_ranks * min_num_recv_tokens, 2)
        patterns, inf_elems, delay = randint(3, 2), randint(hidden, 2), randint(1 << 21, 1)[0]
        mine = [i for i, v in enumerate(victims) if v // min_num_recv_tokens == rank]
        budget.reserve(2)
        combine_x = recv_x.clone()
        lines = []
        with capture_device_printf(lines):
            overwrite_later(combine_x, [victims[i] % min_num_recv_tokens for i in mine],
                            [bad_value(patterns[i], inf_elems[i]) for i in mine], delay)
            combined_x = combine(combine_x)
            side_stream.synchronize()
        changed = (combined_x.view(torch.int16) != clean_combined_x.view(torch.int16)).any(dim=-1)
        nonfinite = nonfinite_rows(combined_x)
        reports = [_COMBINE_REPORT.search(line) for line in lines]
        expect(all(reports), f'combine: unexpected report: {lines}')
        reported = {int(m.group(2)) for m in reports}
        expected = set((changed & nonfinite).nonzero().flatten().tolist())
        expect(reported == expected, f'combine: reported tokens {sorted(reported)}, expected {sorted(expected)}')
        tally += torch.stack([changed.sum(), (changed & nonfinite).sum(), (changed & ~nonfinite).sum()])
    dist.all_reduce(tally)
    dist_print(f'   changed combined tokens: {tally[0].item()}, NaN/Inf and reported: {tally[1].item()}, '
               f'changed but only valid numbers (cannot be seen): {tally[2].item()}', once_in_node=True)

    # B: one bit flip in each of `num_flips` different tokens, for each bit of the element type
    for name, num_bits in (('BF16', 16), ('FP8', 8)):
        dist_print(f' > Flip each bit of {name}, {args.num_flips} tokens for each bit', once_in_node=True)
        base = x if name == 'BF16' else per_token_cast_to_fp8(x)
        table = []
        for bit in range(num_bits):
            victims = pick_victims(num_ranks * num_tokens, args.num_flips)
            elems = randint(hidden, args.num_flips)
            x_run = base.clone() if name == 'BF16' else (base[0].clone(), base[1])
            payload = x_run if name == 'BF16' else x_run[0]
            words = payload.view(torch.int16 if name == 'BF16' else torch.uint8)
            # Bit 15 of an int16 is its sign bit
            mask = (1 << bit) - (1 << 16 if name == 'BF16' and bit == 15 else 0)
            # Which flips make a NaN or Inf: known on the source ranks, then shared
            flips_nonfinite = torch.zeros(args.num_flips, dtype=torch.int64, device='cuda')
            for i, (v, e) in enumerate(zip(victims, elems)):
                if v // num_tokens == rank:
                    words[v % num_tokens, e] ^= mask
                    flips_nonfinite[i] = int(not torch.isfinite(payload[v % num_tokens, e].float()).item())
            dist.all_reduce(flips_nonfinite)
            budget.reserve(flips_nonfinite.sum().item())
            lines = []
            with capture_device_printf(lines):
                recv_x, _, _, handle, _ = dispatch(x_run)
            recv_payload = recv_x if name == 'BF16' else recv_x[0]
            num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
            src = handle.recv_src_metadata[:num_recv_tokens, 0].long()
            expected = set(src[nonfinite_rows(recv_payload[:num_recv_tokens])].tolist())
            reported = reported_dispatch_sources(lines)
            expect(reported == expected, f'{name} bit {bit}: reported {sorted(reported)}, expected {sorted(expected)}')
            # Each victim token counts once, on any rank that received it
            caught = torch.tensor([int(v in reported) for v in victims], dtype=torch.int64, device='cuda')
            dist.all_reduce(caught, op=dist.ReduceOp.MAX)
            expect(torch.equal(caught, flips_nonfinite), f'{name} bit {bit}: a flip to NaN/Inf was not reported')
            table.append(f'{bit}:{caught.sum().item()}')
        dist_print(f'   {name}, bit:caught of {args.num_flips}: {" ".join(table)}', once_in_node=True)

    # Start the next steps in a new rate-limit window, with no report left from this step
    time.sleep(_REPORT_WINDOW_SECONDS * 1.1)
    dist.barrier()
    dist_print(' > Corruption checks passed', once_in_node=True)


# noinspection PyShadowingNames
def test_token_sync(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    """The signals must be in the expected state at each entry of the unordered dispatch and combine"""
    num_tokens, hidden, num_experts = args.num_tokens, args.hidden, args.num_experts

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda').abs()
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def dispatch_and_combine():
        recv_x, _, recv_topk_weights, handle, _ = buffer.dispatch(
            x=x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_max_tokens_per_rank=num_tokens, num_experts=num_experts, num_sms=args.num_sms)
        buffer.combine(x=recv_x, handle=handle, topk_weights=recv_topk_weights)

    dist_print(' > Dispatch and combine', once_in_node=True)
    lines = []
    with capture_device_printf(lines):
        for _ in range(args.num_signal_iters):
            dispatch_and_combine()
    expect(not any(_SIGNAL_REPORT in line for line in lines), f'signal reports: {lines}')
    dist.barrier()
    dist_print(' > Token sync checks passed', once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.ElasticBuffer(group, num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                   num_topk=args.num_topk, prefer_overlap_with_compute=False, sl_idx=0,
                                   explicitly_destroy=True)
    # The corruption checks run first: the rate-limit step of `test_debug_checks` uses up the report budget
    test_corruption(buffer, args)
    test_debug_checks(buffer, args)
    test_token_sync(buffer, args)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the token_integrity and token_sync checks of EP_DEBUG_CHECKS')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=1024, help='Number of tokens')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    parser.add_argument('--num-sms', type=int, default=0,
                        help='Number of SMs to use (0 means auto, which probes the NIC set by `EP_NIC_NAME`)')
    parser.add_argument('--num-corruption-iters', type=int, default=48,
                        help='Iterations of each racing-write corruption check')
    parser.add_argument('--num-flips', type=int, default=24, help='Tokens with a bit flip, for each bit')
    parser.add_argument('--num-signal-iters', type=int, default=16, help='Dispatch and combine iterations of the signal check')
    args = parser.parse_args()

    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
