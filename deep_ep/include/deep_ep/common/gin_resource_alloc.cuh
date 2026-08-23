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

// NCCL GIN: provider resource budget.
static constexpr int kTotalQPBudget              = 256;
static constexpr int kMaxGinContextBudget        = 17;

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
static constexpr int kMaxSM           = (kTotalQPBudget - 2*kMaxGinContextBudget)/kMaxWarpsPerSM;
static constexpr int kMaxScaleoutWarps           = kMaxSM * kMaxWarpsPerSM;

struct GinResourceConfig {
    int gin_indexed_signals_cnt;      // NGinIndexedSignalsPerGinContext (per NCCL context)
    int gin_context_cnt;              // ScaleOut contexts + one Notify context
};

static constexpr int kMinGinContextCnt           = 2;
static constexpr int kMaxGinContextCnt           = kMaxGinContextBudget;

// Default context count (== default QP count). 11 contexts -> 21 signals/context.
// Contexts and signals-per-context are inversely coupled through
// `gin_indexed_signals_for`, so more QPs means a smaller per-context signal budget. The
// equivalent alternatives are {5, 6, 7, 8, 9, 14}; everything else loses a part somewhere.
// Notably 12, 15, 16 and 17 all drop to 3 parts at 12 SMs -- 17 (the provider maximum) leaves
// only 13 signals/context, and its per-SM QP split puts 4 channels on the busiest QP.
static constexpr int kDefaultGinContextCnt       = 11;

// Per-context indexed-signal budget, workaround for current limitations in provider.
__forceinline__ __device__ __host__ constexpr int gin_indexed_signals_for(int gin_context_cnt) {
    return (kTotalQPBudget - 2 * gin_context_cnt) / gin_context_cnt;
}

__forceinline__ __device__ __host__ constexpr GinResourceConfig make_gin_resources(int gin_context_cnt) {
    return GinResourceConfig{gin_indexed_signals_for(gin_context_cnt), gin_context_cnt};
}

// Indexed-signal ids reserved for the device barrier, taken off the bottom of EVERY
// context's id space. This is for the RAIL instantiation of `gin_barrier_wo_local_sync`,
// which is a counting barrier and needs exactly one slot whatever the team size -- but it
// must be an id no data channel can ever produce, which is what this reservation buys.
//
// The World instantiation keeps the per-peer barrier and is NOT covered by this
// reservation. It does not need to be: it is provisioned from the other arm of
// `NCCLSymmetricMemoryContext` (`reqs.ginSignalCount = num_ranks + 2 * 2`), and the
// unordered data path that `data_signal_id` shifts does not run on that arm at all --
// `get_qp_signal_id` / `get_per_part_signal_id` are called only from
// `hybrid_{dispatch,combine}_unordered.cuh`.
//
// This is a fix, not hardening. In `cached_mode` the unordered hybrid kernels run with zero
// notify warps (`dispatch.hpp`), which makes `kQPStartIdx = 0` (`qp_mapping.cuh`) and puts
// data channels on context 0 -- the barrier's own context. `channel_to_signal_id` is 0-based,
// so channel 0 part 0 produced id 0 there: exactly the barrier's slot. NCCL addresses a
// shadow by (context, signal) alone, so that was a shared counter, and each side inflated
// the other's arrival count.
//
// Only the barrier's own context (QP 0) strictly needs the hole, yet the reservation is
// applied uniformly, so the id derivation does not have to depend on which QP a channel
// landed on. That is NOT free -- measured with `compute_part_allocation` at the shipped
// context count, 16-17 SMs lose one part (3 -> 2) and 51-52 SMs lose a channel per SM
// (4 -> 3, with the host `[WARN]`). Weigh that before changing `kDefaultGinContextCnt`.
//
// ONE id, not two, because the Rail barrier is the only counting barrier and a GIN
// scale-up barrier and a GIN scale-out barrier can never be live at once -- `get_logical_domain_size` (`nccl.cu`) sets `num_scaleup_ranks` to
// `num_nvl_ranks` in hybrid mode (so `kIsScaleupNVLink` is true there by construction) and
// forces `num_scaleout_ranks` to 1 in direct mode. `gpu_barrier` asserts that, and
// `NCCLSymmetricMemoryContext` asserts it again on the host where it fails at init.
//
// NOTE: plain `int`, not an NCCL signal type -- this header is deliberately NCCL-free so it
// stays host-compilable, which is what lets the invariants below be `static_assert`s.
static constexpr int kNumReservedBarrierSignals = 1;
static constexpr int kBarrierSignalId = 0;
static_assert(kBarrierSignalId >= 0 and kBarrierSignalId < kNumReservedBarrierSignals,
              "the barrier's signal id must lie inside the reserved range");

// The single place the reservation offset is applied. Both id derivations in `comm.cuh`
// (per-channel and per-part) route through this, so the offset cannot be dropped from one
// and kept in the other.
__forceinline__ __device__ __host__ constexpr int data_signal_id(int raw_offset) {
    return kNumReservedBarrierSignals + raw_offset;
}

// For every legal context count the layout must remain SERVICEABLE at the worst-case launch
// (`kMaxSM` SMs x `kMaxWarpsPerSM` ScaleOut warps): `compute_part_allocation` must return at
// least one channel per SM and at least one part, without tripping its own host assert.
//
// This replaces an earlier aggregate check, `ctx * signals/ctx >= kMaxScaleoutWarps`, which
// claimed to prove "every ScaleOut warp gets a dedicated id". It never did: the ids are
// per-context, so an aggregate total says nothing about the busiest context, and at ctx = 13
// the worst-case launch already needed 19 ids against 17 available and was silently relying
// on the tuner to cut channels. The reservation made the old check additionally stale
// (usable is `ctx * (signals - 1)`), which is what surfaced the problem.
//
// Channel reduction at high SM counts is expected and is a graceful degradation -- the tuner
// warns and proceeds. What must never happen is an unserviceable configuration, and that is
// what this asserts. `compute_part_allocation` is defined below; the check sits with it.
__forceinline__ __host__ constexpr bool all_gin_context_counts_are_serviceable();

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
//   channels_per_context(...) * num_parts <= gin_indexed_signals_cnt - kNumReservedBarrierSignals
// at the requested channels/SM, then reduce channels_per_sm until the budget holds.
//
// Split in two: `_raw` is the pure math, free of diagnostics, so it can be evaluated inside a
// `static_assert` (a `printf` or a throwing assert reached during constant evaluation makes
// the expression non-constant). `compute_part_allocation` is the shipping entry point and
// adds the host-side warning and check on top. Keep the math in `_raw` only -- duplicating it
// into the invariant below is exactly the drift this split exists to prevent.
__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation_raw(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    // The data path may only use ids at or above `kNumReservedBarrierSignals`, so the
    // usable budget is that much smaller than the provisioned count.
    const int provisioned = cfg.gin_indexed_signals_cnt;
    const int gin_signals = provisioned > kNumReservedBarrierSignals
                          ? provisioned - kNumReservedBarrierSignals : 0;
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
    return alloc;
}

// True when the allocation actually fits the usable budget -- i.e. the reduction loop above
// converged rather than bottoming out at one channel per SM and still not fitting.
__forceinline__ __device__ __host__ constexpr bool part_allocation_fits(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, const GinPartAllocation& alloc) {
    const int provisioned = cfg.gin_indexed_signals_cnt;
    const int gin_signals = provisioned > kNumReservedBarrierSignals
                          ? provisioned - kNumReservedBarrierSignals : 0;
    return static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                       alloc.num_channels_per_sm)) * alloc.num_parts
           <= gin_signals;
}

__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    const GinPartAllocation alloc =
        compute_part_allocation_raw(cfg, num_sms, num_available_qps, num_channels_per_sm);
#ifndef __CUDA_ARCH__
    if (alloc.num_channels_per_sm < num_channels_per_sm)
        printf("[WARN] DeepEP GIN signal budget reduced the number of channels per SM "
               "from %d to %d\n", num_channels_per_sm, alloc.num_channels_per_sm);
    EP_HOST_ASSERT(part_allocation_fits(cfg, num_sms, num_available_qps, alloc) and
                   "GIN signal budget cannot host even 1 part-signal per channel "
                   "at 1 channel/SM. Reduce --num-sms or num_allocated_qps.");
#endif
    return alloc;
}

// The invariant declared above, now that the math it checks is in scope.
__forceinline__ __host__ constexpr bool all_gin_context_counts_are_serviceable() {
    for (int ctx = kMinGinContextCnt; ctx <= kMaxGinContextCnt; ++ ctx) {
        // The notify warp owns QP 0, so only `ctx - 1` contexts carry data channels.
        const int avail = ctx > 1 ? ctx - 1 : 1;
        const auto cfg = make_gin_resources(ctx);
        const auto alloc = compute_part_allocation_raw(cfg, kMaxSM, avail, kMaxWarpsPerSM);
        if (alloc.num_channels_per_sm < 1 or alloc.num_parts < 1)
            return false;
        if (not part_allocation_fits(cfg, kMaxSM, avail, alloc))
            return false;
    }
    return true;
}
static_assert(all_gin_context_counts_are_serviceable(),
              "some legal GIN context count cannot service the worst-case launch "
              "(kMaxSM x kMaxWarpsPerSM) once the barrier reservation is taken out");

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
