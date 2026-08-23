#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cooperative_groups.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/gin_resource_alloc.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/qp_mapping.cuh>

namespace deep_ep::elastic::comm {

static constexpr int64_t kNumOneSecCycles = 2000000000;  // An approximation of the GPU clock at 2000 MHz

// Some reserved tags
static constexpr int kDeviceBarrierTag = 0;
static constexpr int kKernelBarrierTag = 1;
static constexpr int kDispatchTag0 = 2;
static constexpr int kDispatchTag1 = 3;
static constexpr int kCombineTag0 = 4;
static constexpr int kCombineTag1 = 5;
static constexpr int kHybridDispatchTag0 = 6;
static constexpr int kHybridDispatchTag1 = 7;
static constexpr int kHybridCombineTag0 = 8;
static constexpr int kHybridCombineTag1 = 9;

// Some reserved count
static constexpr int kFlushAllAllocatedQPs = -1;

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const bool& condition, const func_t& func,
                                              int64_t start_clock = 0) {
    // User may share a start clock for multiple waits
    if (start_clock == 0)
        start_clock = clock64();

    while (condition) {
        const bool timeout = clock64() - start_clock >= kNumTimeoutCycles;
        if (func(timeout))
            break;

        if (timeout) {
            // Wait another 1 second to let all threads print information and trap
            start_clock = clock64();
            while (clock64() - start_clock < kNumOneSecCycles) {}
            ptx::trap();
        }
    }
}

template <int64_t kNumTimeoutCycles, typename func_t>
__device__ __forceinline__ void timeout_while(const func_t& func, const int64_t& start_clock = 0) {
    timeout_while<kNumTimeoutCycles, func_t>(true, func, start_clock);
}

// Channels map onto QPs with a BALANCED CONTIGUOUS (block) partition: fill a QP with
// consecutive channels before spilling to the next (so an SM's channels stay grouped
// on the same GIN context), and when the channels do not divide the QPs evenly,
// spread the remainder one-per-QP across the leading QPs instead of leaving a
// trailing QP idle. Used by all callers (dispatch, hybrid_dispatch, combine,
// hybrid_combine). The integer partition lives in `qp_mapping.cuh` (host-testable);
// this wrapper only adds the NCCL resource-sharing mode.
//   e.g. 48 channels / 4 QPs  -> {12,12,12,12}  (even; identical to before)
//        4  channels / 3 QPs  -> QP0<-2, QP1<-1, QP2<-1  (was {2,2,0}, QP2 idle)
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp = false) {
    constexpr auto kSharingCTA = NCCL_GIN_RESOURCE_SHARING_CTA;
    constexpr auto kSharingGrid = kNumSMs == 1 ? NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU;

    // Only one QP
    if constexpr (kNumQPs == 1)
        return {0, kSharingGrid};

    // The notify warp always use 1 SM and 1 QP
    if (is_notify_warp)
        return {0, kSharingCTA};

    // Data channels: QP index from the shared balanced partition, sharing mode from
    // the branch (CTA when SMs each own their QPs, GPU-wide when SMs share QPs).
    constexpr int kNumAvailableQPs = kNumQPs - static_cast<int>(kWithNotifyWarps);
    const int qp_idx = channel_to_qp<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
        sm_idx, channel_in_sm_idx, is_notify_warp);
    if constexpr (kNumSMs <= kNumAvailableQPs)
        return {qp_idx, kSharingCTA};
    else
        return {qp_idx, kSharingGrid};
}

// Companion to `get_qp_mode`: within a QP, gives a unique per-channel signal id
// (position of this channel among all channels sharing the same QP). Uses the same
// BALANCED CONTIGUOUS partition as `get_qp_mode` (see `qp_mapping.cuh`), so the id is
// the channel's offset within its QP's block:
//   kNumQPs == 1                : every channel on the single QP -> id = global_channel_idx
//   kNumSMs <= kNumAvailableQPs : offset within the SM's balanced local-QP block
//   kNumSMs  > kNumAvailableQPs : offset within the global balanced QP block
// In all cases id < ceil(channels / qps), so `(qp, id)` is unique per channel and the
// tuner's per-QP signal budget (sized for ceil(channels / qps)) is never exceeded.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_qp_signal_id(
    const int& sm_idx, const int& channel_in_sm_idx) {
    // Shifted past the barrier's reserved ids. `channel_to_signal_id` stays 0-based (pure
    // integer math); the reservation is applied here, at the boundary where a raw offset
    // becomes a signal id the data path will actually signal.
    return elastic::gin_alloc::data_signal_id(
        channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
            sm_idx, channel_in_sm_idx));
}

// Per-part indexed-signal id: kNumParts contiguous ids under the channel's base id, one
// per token part and shared by all sources. The tuner + channel cap guarantee
// ceil(num_channels / qps) * kNumParts <= gin_indexed_signals_cnt.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, int kNumParts,
          bool kWithNotifyWarps = false>
__device__ __forceinline__ int get_per_part_signal_id(
    const int& sm_idx, const int& channel_in_sm_idx, const int& part_idx) {
    // NOTE: the reservation is added AFTER the multiply, and `channel_to_signal_id` is called
    // directly rather than through `get_qp_signal_id`. Going through the latter would scale
    // the offset by `kNumParts` and burn `kNumReservedBarrierSignals * kNumParts` ids.
    return elastic::gin_alloc::data_signal_id(
        channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, kWithNotifyWarps>(
            sm_idx, channel_in_sm_idx) * kNumParts + part_idx);
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag>
__forceinline__ __device__ void nvlink_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    // This barrier only uses 1 SM
    if (kNumSMs > 1 and sm_idx > 0)
        return;

    // Read the current barrier phase first
    const int status = static_cast<int>((*workspace.get_nvl_barrier_counter_ptr()) & 3);
    const int phase = status & 1, sign = status >> 1;

    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (thread_idx < kNumRanks) {
        const auto dst_ptr =
            gin.get_sym_ptr<ncclTeamTagLsa>(workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
        ptx::red_add_rel_sys(dst_ptr, sign ? -1 : 1);
    }
    __syncthreads();

    // NOTES: we need `2^64 / 1e6 / 3600 / 24 / 365 = 571000` years to make the counter overflow (1 barrier per us)
    // Add the phase counter
    if (thread_idx == 0)
        atomicAdd(workspace.get_nvl_barrier_counter_ptr(), 1);

    // Check timeout
    const auto target = sign ? 0 : kNumRanks;
    timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));
        if (signal == target)
            return true;

        if (is_last_check) {
            printf("DeepEP NVLink barrier timeout, tag: %d, nvl: %d, thread: %d, "
                   "status: %d, signal: %d, phase: %d, target: %d, counter: %llu\n",
                   kTag, rank_idx, thread_idx, status, signal, phase, target,
                   *workspace.get_nvl_barrier_counter_ptr());
        }
        return false;
    });
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles,
          typename team_t, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true,
          int kNumWarps = kNumThreads / 32>
__forceinline__ __device__ void gin_barrier_wo_local_sync(
    const ncclDevComm_t& nccl_dev_comm,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx, 
    const int& sm_idx, const int& thread_idx) {
    const auto global_warp_idx = sm_idx * kNumWarps + (thread_idx / 32);
    const int& rank_idx = (std::is_same_v<team_t, ncclTeamTagWorld>) ? scaleup_rank_idx : scaleout_rank_idx;
    const int num_qps = kNumQPs == kFlushAllAllocatedQPs ? nccl_dev_comm.ginContextCount : kNumQPs;

    // Flush all QPs by all SMs (only needed for release semantics)
    if constexpr (kFlushStores) {
        for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {
            ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
        }

        // World-team barriers may mix direct NVLink writes with GIN signals, so a system-scope
        // threadfence is required to make the NVLink writes visible before publishing the barrier
        if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>)
            ptx::fence_acq_rel_sys();

        // NOTES: we can not use `kNumSMs` to judge, as maybe only part of the SMs will call this function
        (gridDim.x > 1) ? cooperative_groups::this_grid().sync() : __syncthreads();
    }

    if (sm_idx == 0) {
        // Use QP 0 to do barrier
        const auto team = (std::is_same_v<team_t, ncclTeamTagWorld>) ?
            ncclTeamWorld(nccl_dev_comm) : ncclTeamRail(nccl_dev_comm);
        const ncclGin gin(nccl_dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

        // The two team instantiations get DIFFERENT barrier protocols, and the split is
        // deliberate. Rail gets a counting barrier; World keeps the per-peer barrier
        // unchanged. Three facts force it:
        //
        //  1. Only Rail has a ceiling to remove. `NCCLSymmetricMemoryContext`
        //     (`csrc/kernels/backend/nccl.cu`) provisions the two paths from different
        //     arms: the unordered-hybrid arm asks for `gin_indexed_signals_cnt`, a
        //     per-context budget of `(kTotalQPBudget - 2c)/c` that does NOT grow with the
        //     team -- that is the budget a per-peer rail barrier overran at 23 domains.
        //     The direct / ordered arm asks for `num_ranks + 2 * 2`, explicitly commented
        //     "Customized RDMA barrier needs extra signals". The World barrier's per-peer
        //     slots are already budgeted there and scale with the team, so that path never
        //     had the ceiling and gains nothing from being converted.
        //
        //  2. Only World is used as a RELEASE barrier. Every Rail call site passes
        //     `kFlushStores = false` (`hybrid_{dispatch,combine}{,_unordered}.cuh`,
        //     `barrier.cuh`), so the Rail barrier never even issues the QP flush above --
        //     it is a pure synchronisation point. The two "ensure data arrival" barriers
        //     in the hybrid kernels explicitly pass `do_scaleout = false` and run over
        //     NVLink. By contrast `dispatch.cuh:398` and `combine.cuh:240` take
        //     `kFlushStores = true` on this World path and then read what peers wrote
        //     (they trigger the copy-epilogue kernel immediately after).
        //
        //  3. A counting barrier cannot carry release. Its counter is anonymous, so a peer
        //     one round ahead can supply an increment that stands in for a delayed
        //     current-round arrival: the count reaches its target without every distinct
        //     peer having arrived. "Everyone arrived" survives that (a rank can only be a
        //     round ahead if it already saw everyone), but "every peer's prior writes are
        //     visible to me" does not. Signal STRENGTH cannot repair it either -- strong
        //     signals order a sender's own prior puts, they do not say which sender
        //     incremented. Identity is the missing half, and only per-peer slots have it.
        //
        // So: convert the path that has the ceiling and does not need release; leave the
        // path that needs release and has no ceiling exactly as it was.
        if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>) {
            // UNCHANGED from the base commit apart from this block's four-space re-indent,
            // so this path can be reviewed as "not touched" (strip comments and whitespace
            // from both revisions and compare: 746 characters of code, identical). Compact signal indexing: (kNumRanks - 1) signal
            // slots per rank. Sender rank_idx writes to every peer i at the slot that
            // identifies *itself* in the peer's enumeration:
            //   sig = (rank_idx < i) ? rank_idx : (rank_idx - 1)
            // So on receiver R, each of the (kNumRanks - 1) slots gets exactly +1 from a
            // distinct sender, and the wait side just iterates all slots looking for one
            // increment per slot.
            for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
                if (i == rank_idx) continue;
                const auto sig = static_cast<ncclGinSignal_t>((rank_idx < i) ? rank_idx : (rank_idx - 1));
                gin.signal(team, i, ncclGin_SignalInc{sig});
            }

            for (int i = thread_idx; i < kNumRanks - 1; i += kNumThreads) {
                const auto signal_idx = static_cast<ncclGinSignal_t>(i);
                const auto shadow_ptr = gin.getSignalShadowPtr(signal_idx);
                const auto target = ++(*shadow_ptr);

                // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
                timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto signal = gin.readSignal(signal_idx, 64, cuda::memory_order_acquire);
                    if (signal >= target)
                        return true;

                    if (is_last_check) {
                        printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, thread: %d, "
                               "signal: %lu, target: %lu\n", kTag, scaleout_rank_idx, scaleup_rank_idx, thread_idx, signal, target);
                    }
                    return false;
                });
            }
        } else {
            // Counting signal barrier, for the RAIL team only. Every sender adds 1 to the
            // SAME signal id on every peer, and the waiter advances its shadow by the
            // number of increments one barrier round delivers, (kNumRanks - 1). A
            // synchronisation-only barrier is a counting predicate, so it does not need to
            // distinguish senders -- see the three-point argument above for why that is
            // true here and false on the World path.
            //
            // This costs ONE indexed-signal slot instead of (kNumRanks - 1), which is what
            // removes the team-size term from the per-context signal budget, and hence the
            // scale-out ceiling: measured on p6-b200, 22 NVLink domains complete and 23
            // refuse at the shipped context count.
            //
            // The pattern is the one the unordered data path already relies on: a single
            // signal accumulating increments from many remote senders, polled against a
            // shadow advanced by the expected delta (`hybrid_combine_unordered.cuh`, the
            // `num_expected_arrivals` wait; sends there use `ncclGin_SignalAdd{.., 1}`).
            // `SignalAdd{.., 1}` is used here rather than `SignalInc` to match that
            // precedent exactly.
            // Reserved id, off-limits to the data path -- see `kNumReservedBarrierSignals`.
            constexpr auto kBarrierSignal =
                static_cast<ncclGinSignal_t>(elastic::gin_alloc::kBarrierSignalId);
            for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
                if (i == rank_idx) continue;
                gin.signal(team, i, ncclGin_SignalAdd{kBarrierSignal, 1ull});
            }

            // Two block-wide syncs bracket the wait, and they do different jobs.
            //
            // BEFORE: every thread has issued its strided share of the sends above. Without
            // this, thread 0 enters the spin below while lanes 1..31 of its own warp still
            // have sends pending -- a divergent warp with one tight polling loop starves the
            // sending lanes, and at large `kNumRanks` that delays outbound sends against a
            // running timeout.
            //
            // AFTER: a single slot means a single waiter, so the other threads must not run
            // ahead of the barrier. The per-peer layout had every thread wait on its own
            // slot, which made "all threads have observed completion" implicit; it has to be
            // restored explicitly here.
            //
            // The enclosing `if (sm_idx == 0)` is uniform across the block (`sm_idx` is
            // derived from `blockIdx.x` at every call site, including the synthetic
            // `sm_idx - 1` in `gpu_barrier`'s hybrid split), so both syncs are reached by
            // all of the block's threads.
            __syncthreads();

            if (thread_idx == 0) {
                const auto shadow_ptr = gin.getSignalShadowPtr(kBarrierSignal);
                const auto target = (*shadow_ptr += static_cast<uint64_t>(kNumRanks - 1));

                // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
                timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto signal = gin.readSignal(kBarrierSignal, 64, cuda::memory_order_acquire);
                    if (signal >= target)
                        return true;

                    if (is_last_check) {
                        // Report the shortfall: with one counting slot the stalled *peer* is
                        // no longer identifiable, so print how many of the expected arrivals
                        // are missing, and the signal id.
                        printf("DeepEP Gin barrier timeout, tag: %d, scaleout: %d, scaleup: %d, "
                               "signal_id: %d, observed: %lu, target: %lu, missing: %lu of %d\n",
                               kTag, scaleout_rank_idx, scaleup_rank_idx,
                               static_cast<int>(kBarrierSignal), signal, target,
                               target - signal, kNumRanks - 1);
                    }
                    return false;
                });
            }
            __syncthreads();
        }
    }
}

template <bool kIsScaleupNVLink, int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag, bool kFlushStores = true>
__forceinline__ __device__ void scaleup_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const layout::WorkspaceLayout& workspace,
    const int& rank_idx, const int& sm_idx, const int& thread_idx) {
    if constexpr (kIsScaleupNVLink) {
        nvlink_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumTimeoutCycles, kTag>(
            gin, workspace, rank_idx, sm_idx, thread_idx);
    } else {
        gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagWorld, kTag, kFlushStores>(
            gin.nccl_dev_comm, 1, rank_idx, sm_idx, thread_idx);
    }
}

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true>
__forceinline__ __device__ void scaleout_barrier_wo_local_sync(
    const handle::NCCLGin& gin,
    const int& scaleout_rank_idx, const int& scaleup_rank_idx,
    const int& sm_idx, const int& thread_idx) {
    gin_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, ncclTeamTagRail, kTag, kFlushStores>(
        gin.nccl_dev_comm, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}

template <bool kIsScaleupNVLink,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumSMs, int kNumThreads, int kNumQPs,
          int64_t kNumTimeoutCycles, int kTag = kDeviceBarrierTag,
          bool kFlushStores = true, bool kSyncAtStart = true, bool kSyncAtEnd = true>
__forceinline__ __device__ void gpu_barrier(const handle::NCCLGin& gin,
                                            const layout::WorkspaceLayout& workspace,
                                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                            const int& sm_idx, const int& thread_idx,
                                            bool do_scaleout = true, bool do_scaleup = true) {
    // A general TMA store wait to prevent proxy memory issues
    if constexpr (kFlushStores) {
        ptx::tma_store_commit();
        ptx::tma_store_wait();
        __syncwarp();
    }

    // All the SMs should wait
    if constexpr (kSyncAtStart) {
        cooperative_groups::this_grid().sync();
    } else {
        EP_STATIC_ASSERT(not kFlushStores, "No data to be flushed");
    }

    // A GIN scale-up barrier (`ncclTeamTagWorld`) and a GIN scale-out barrier
    // (`ncclTeamTagRail`) would both land on the SAME shadow: NCCL addresses shadows by
    // (context, signal) only -- `_signalShadows = comm.ginSignalShadows + contextIndex *
    // comm.ginSignalCount` -- with no team or tag term. Both use context 0, and their id
    // ranges overlap even though the two now run different protocols: World's per-peer
    // slots start at 0 and Rail's counting slot IS 0. So if they were ever live
    // CONCURRENTLY they would corrupt each other -- World would read Rail's increments as
    // peer 0's arrival, and Rail's count would absorb World's. Rail is a subset of World,
    // so the two rank sets are not even disjoint.
    //
    // Three ways that cannot happen, any one of which is sufficient:
    //   * scale-up runs over NVLink, so it never touches a GIN signal at all; or
    //   * there is no scale-up team to synchronize (`do_scaleup` is masked off below); or
    //   * there is no scale-out team to synchronize (`do_scaleout` likewise).
    // Note this is about CONCURRENCY: `barrier.cuh`'s sequential path issues Rail and World
    // from two separate, globally ordered `gpu_barrier` calls and is safe for that reason,
    // which is why the condition below is a disjunction rather than the stricter
    // `kIsScaleupNVLink or kNumScaleoutRanks <= 1`.
    //
    // Today no instantiation can violate it: `NCCLSymmetricMemoryContext`
    // (`csrc/kernels/backend/nccl.cu`) sets `num_scaleup_ranks = num_nvl_ranks` in hybrid mode
    // -- making `is_scaleup_nvlink` true there by construction -- and `num_scaleout_ranks = 1`
    // in direct mode. This assert exists so that stops being an accident.
    //
    // CAVEAT on the safety net: these kernels are NVRTC-generated from runtime values
    // (`csrc/kernels/elastic/*.hpp` format the template arguments into the instantiation), so
    // a violation surfaces as a JIT compile exception on first launch, in production -- NOT
    // as a build failure. The host-side `EP_HOST_ASSERT` in `NCCLSymmetricMemoryContext` is
    // the gate that actually fails early; this one is the backstop for a caller that
    // constructs the template arguments some other way.
    EP_STATIC_ASSERT(kIsScaleupNVLink or kNumScaleupRanks <= 1 or kNumScaleoutRanks <= 1,
                     "A GIN scale-up barrier and a GIN scale-out barrier would share the "
                     "reserved barrier signal id; allocate a second reserved id before "
                     "allowing this combination");

    do_scaleout &= kNumScaleoutRanks > 1;
    do_scaleup &= kNumScaleupRanks > 1;
    if (do_scaleup and do_scaleout) {
        // Do scaleup and scaleout barrier in parallel
        EP_DEVICE_ASSERT(kNumSMs >= 2 and "At least 2 SMs for a hybrid barrier");
        if (sm_idx == 0) {
            // First SM do the scaleup barrier
            scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);

            // We need an extra grid sync, as the scaleout barrier will do a sync after flush, before the barrier
            // NOTES: this is kind of hacky
            if constexpr (kFlushStores) 
                cooperative_groups::this_grid().sync();
        } else {
            // The remaining SMs do the scaleout barrier
            scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
                gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);
        }
    } else if (do_scaleup) {
        // Scaleup only
        scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);
    } else if (do_scaleout) {
        // Scaleout only
        scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, kTag, kFlushStores>(
            gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }

    // All the SMs should wait
    if constexpr (kSyncAtEnd)
        cooperative_groups::this_grid().sync();
}

}  // namespace deep_ep::elastic::comm
