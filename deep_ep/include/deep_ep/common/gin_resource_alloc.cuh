#pragma once

// MIT License
//
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep::elastic::gin_alloc {

// NCCL GIN: provider resource budget, in QP units per GPU. A context costs 2 units (its data and
// PutValue endpoints) plus 1 per indexed signal, so n contexts of S signals fit when
// n * (2 + S) <= kTotalQPBudget. Every unit is one QP on the NIC the GIN connection puts the
// context on, and the context also binds 2 + 2 S of that NIC's completion counters, at most two
// per unit. Both limits are per NIC generation (read them with `fi_info -p efa -v`: cntr_cnt, and
// `ibv_devinfo -v`: max_qp); the generations measured so far report 512 counters and 256 or 768
// QPs, so 256 is the largest budget that fits every context count in [2, 41] on all of them when
// one NIC carries all of a GPU's contexts. `EP_GIN_QP_BUDGET` (read at buffer creation) widens it
// where the connection spreads the contexts round-robin over several NICs (ceil(n / rails) of them
// then share one NIC): with two NICs of 512 counters and >= 256 QPs each, 384 fits every context
// count in [2, 41] and 512 does not (an odd count puts the extra context on one NIC). Verify the
// rail count and the per-NIC limits of the platform before raising it. The kernels take the
// resulting signal count as a template argument, so an override is consistent end to end.
static constexpr int kTotalQPBudget              = 256;
// Context ceiling: one data QP per SM up to 40 SMs after the notify context (see `get_qp_mode`:
// an SM owns a QP only when the kernel runs on at most kNumQPs - 1 SMs, and the combine proxy's
// thread-exclusive put needs that ownership).
static constexpr int kMaxGinContextBudget        = 41;

// Design constant: ScaleOut warps per SM used by the auto-tuner's budget math.
// Matches the buffer's `!prefer_overlap_with_compute` cap on `num_channels_per_sm`
// (one ScaleOut warp per channel). Under-provisions vs. the runtime peak (which
// can go up to `kNumMaxChannelsPerSM = 8`); worst case just means heavier
// warps-per-context sharing, not a correctness issue.
static constexpr int kMaxWarpsPerSM              = 4;

// Upper bound on ScaleOut warps used by any single kernel: 55 SMs × 4 warps = 220.
//
// Each ScaleOut warp (== channel) needs its own dedicated signal: e.g. when
// warp0 (channel0) signals its peer channel0 warp on a remote node, it uses a
// signal reserved for that channel. That same signal is reused for the channel
// across all nodes in the rail group, so it is shared rail-wide — the signal
// count scales with the number of channels (warps), not with the number of
// nodes.
//
// Pinned rather than derived from the budget: (256 - 2 * 17) / 4 = 55 at the original 17-context
// ceiling, and the checks below are written against this SM count.
static constexpr int kMaxSM           = 55;
static constexpr int kMaxScaleoutWarps           = kMaxSM * kMaxWarpsPerSM;

struct GinResourceConfig {
    int gin_indexed_signals_cnt;      // NGinIndexedSignalsPerGinContext (per NCCL context)
    int gin_context_cnt;              // ScaleOut contexts + one Notify context
};

static constexpr int kMinGinContextCnt           = 2;
static constexpr int kMaxGinContextCnt           = kMaxGinContextBudget;

// Default context count (== default QP count). 13 contexts provide 17 signals per
// context and are the smallest allocation that gives 12 SMs one data QP each after
// reserving the notify context. The resulting 221 signals leave one above the
// 220-warp ceiling checked below.
static constexpr int kDefaultGinContextCnt       = 13;

// Per-context indexed-signal budget, workaround for current limitations in provider.
__forceinline__ __device__ __host__ constexpr int gin_indexed_signals_for(int gin_context_cnt,
                                                                          int qp_budget = kTotalQPBudget) {
    return (qp_budget - 2 * gin_context_cnt) / gin_context_cnt;
}

__forceinline__ __device__ __host__ constexpr GinResourceConfig make_gin_resources(int gin_context_cnt,
                                                                                   int qp_budget = kTotalQPBudget) {
    return GinResourceConfig{gin_indexed_signals_for(gin_context_cnt, qp_budget), gin_context_cnt};
}

// Each ScaleOut warp (== channel) needs its own dedicated indexed signal id, so the total
// signal budget (ctx * signals/ctx) must cover the worst-case warp count for every context count
// up to `kMaxGinContextCntFullWarps` (the ceiling before it was widened to one QP per SM), not
// just the default. The tightest points are ctx = 13 and ctx = 17, both at 221 against the
// 220-warp ceiling -- one signal of slack. Do not raise `kMaxSM` / `kMaxWarpsPerSM`, widen this
// range, or lower `kTotalQPBudget` without re-checking. Above 17 contexts the default budget leaves
// fewer signals per context; `compute_part_allocation` then lowers the parts and, at high SM
// counts, the channels per SM to fit (the second check, after it, is the guarantee that remains).
static constexpr int kMaxGinContextCntFullWarps  = 17;
__forceinline__ __host__ constexpr bool all_gin_context_counts_cover_warps() {
    for (int ctx = kMinGinContextCnt; ctx <= kMaxGinContextCntFullWarps; ++ ctx)
        if (ctx * gin_indexed_signals_for(ctx) < kMaxScaleoutWarps)
            return false;
    return true;
}
static_assert(all_gin_context_counts_cover_warps(),
              "GIN layout cannot give each ScaleOut warp a dedicated signal id "
              "for every context count up to kMaxGinContextCntFullWarps");

// Preferred (and workspace-sizing) maximum for per-part signalling.
static constexpr int kMaxParts = 4;

static constexpr int kScaleoutSlotRoundingReserve = kMaxParts;

struct GinPartAllocation {
    int num_parts;           // per-channel part count (>= 1)
    int num_channels_per_sm;
};

// Worst-case number of channels that land on a single GIN context (QP). MUST match the
// signal-id assignment in `channel_to_signal_id` (`qp_mapping.cuh`) exactly, because the
// per-channel signal id is its offset within its QP's block and the provisioned budget has
// to cover the largest such offset.
//
// QP assignment is two-level, so there are two regimes:
//   * num_sms <= num_available_qps: each SM owns its own block of QPs
//     (`num_qps_in_sm = avail / num_sms`, plus one for the first `avail % num_sms` SMs) and
//     balances only its OWN channels across them. The worst SM is one WITHOUT the remainder
//     bonus, so it hosts `ceil(channels_per_sm / (avail / num_sms))` channels per QP.
//   * num_sms >  num_available_qps: all SMs share all QPs, and the global balanced partition
//     gives `ceil(num_channels / avail)` channels per QP.
//
// NOTE: the first regime is NOT `ceil(num_sms * channels_per_sm / avail)`. That form
// under-counts, because spare QPs owned by OTHER SMs cannot absorb this SM's channels. Using
// it there provisions too few signals, and the kernel then signals ids outside the
// provisioned range -- which fails silently: no counts ever arrive and dispatch times out in
// the CPU wait with all-zero received counts.
__forceinline__ __device__ __host__ constexpr int channels_per_context(
        int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int avail = num_available_qps > 1 ? num_available_qps : 1;
    const int sms = num_sms > 1 ? num_sms : 1;
    if (sms <= avail) {
        const int num_qps_in_sm = avail / sms;
        return math::constexpr_ceil_div(num_channels_per_sm,
                                        num_qps_in_sm > 1 ? num_qps_in_sm : 1);
    }
    return math::constexpr_ceil_div(sms * num_channels_per_sm, avail);
}

// Per-part signal allocation: pick the largest num_parts (up to kMaxParts) that fits
//   channels_per_context(...) * num_parts <= gin_indexed_signals_cnt
// at the requested channels/SM, then reduce channels_per_sm until the budget holds.
__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int gin_signals = cfg.gin_indexed_signals_cnt;
    const int channels_per_ctx = channels_per_context(num_sms, num_available_qps, num_channels_per_sm);
    const int budget_parts = gin_signals / (channels_per_ctx > 1 ? channels_per_ctx : 1);
    GinPartAllocation alloc{};
    alloc.num_parts = budget_parts < kMaxParts ? budget_parts : kMaxParts;
    alloc.num_parts = alloc.num_parts > 1 ? alloc.num_parts : 1;
    alloc.num_channels_per_sm = num_channels_per_sm;
    while (alloc.num_channels_per_sm > 1 and
           static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                      alloc.num_channels_per_sm)) * alloc.num_parts > gin_signals)
        --alloc.num_channels_per_sm;
#ifndef __CUDA_ARCH__
    EP_HOST_ASSERT(static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                               alloc.num_channels_per_sm)) * alloc.num_parts <= gin_signals and
                   "GIN signal budget cannot host even 1 part-signal per channel "
                   "at 1 channel/SM. Reduce --num-sms or num_allocated_qps.");
#endif
    return alloc;
}

// The guarantee kept for every legal context count at the default budget: a kernel on up to
// `kMaxSM` SMs always fits at least one channel per SM (one part), so the host assert in
// `compute_part_allocation` cannot fire below that SM count. Beyond 17 contexts it is the parts
// and then the channels per SM that give, never the allocation itself.
__forceinline__ __host__ constexpr bool all_gin_context_counts_host_one_channel() {
    for (int ctx = kMinGinContextCnt; ctx <= kMaxGinContextCnt; ++ ctx)
        if (channels_per_context(kMaxSM, ctx - 1, 1) > gin_indexed_signals_for(ctx))
            return false;
    return true;
}
static_assert(all_gin_context_counts_host_one_channel(),
              "GIN layout cannot host one channel per SM at kMaxSM for every legal context count");

// Kernel-side entry points: derive the per-channel part count (and verify the launched
// channel count) as compile-time constants from the provisioned indexed-signal budget.
__forceinline__ __device__ __host__ constexpr int constexpr_num_parts(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_parts;
}

__forceinline__ __device__ __host__ constexpr int constexpr_channels_per_sm(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_channels_per_sm;
}

}  // namespace deep_ep::elastic::gin_alloc
