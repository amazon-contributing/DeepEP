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

// The `token_sync` and `hang_detector` checks of `EP_DEBUG_CHECKS`. The bits are in `compiled.cuh`, see `diag.cuh`

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::diag {

// Call after the report of a bad signal. As after a timeout, wait one second so that the other threads
// can print their reports, then trap
__device__ __forceinline__ void trap_after_report() {
    const auto stop = clock64() + comm::kNumOneSecCycles;
    while (clock64() < stop) {}
    ptx::trap();
}

// Each GIN indexed signal has a counter that the peers add to, and a shadow that the local kernel
// adds to when it waits. At the entry of the unordered kernels the two must be level, except in
// context 0. Context 0 is shared between phases: it holds the barrier signals of all kernels, so a
// signal is one ahead when a peer has already entered the next barrier. All other contexts are fully
// synced, because the previous kernel waited for every increment it expected.
//
// The check reads all (context, id) signals, so it also finds an increment on a signal that no channel
// waits for. All threads of the grid call it before the entry barrier. It prints each bad signal and
// traps
__device__ __forceinline__ void check_gin_signals(const ncclDevComm_t& nccl_dev_comm,
                                                  const int& global_thread_idx,
                                                  const int& num_global_threads) {
    const int num_signals = nccl_dev_comm.ginSignalCount;
    const int num_slots = static_cast<int>(nccl_dev_comm.ginContextCount) * num_signals;
    bool found_bad_signal = false;
    for (int slot = global_thread_idx; slot < num_slots; slot += num_global_threads) {
        const int context_idx = slot / num_signals, signal_idx = slot % num_signals;
        const auto signal_id = static_cast<ncclGinSignal_t>(signal_idx);
        const ncclGin gin(nccl_dev_comm, context_idx);

        // The counter can wrap, so compare only the low `comm::kNumSignalBits` bits, the same as the
        // waits of the kernels: `readSignal` returns these bits of the counter, the mask keeps these
        // bits of the shadow, and `signal_delta` takes the difference with wrap. Nothing is read after
        // the signal, so the read can be relaxed
        const auto shadow = *gin.getSignalShadowPtr(signal_id) & ((uint64_t{1} << comm::kNumSignalBits) - 1);
        const auto signal = gin.readSignal(signal_id, comm::kNumSignalBits, cuda::memory_order_relaxed);
        const auto skew = comm::signal_delta(signal, shadow);
        if (skew < 0 or skew > (context_idx == 0 ? 1 : 0)) {
            printf("DeepEP debug check: GIN signal skew at kernel entry, rank: %d, context: %d, signal: %d, "
                   "shadow: %llu, counter: %llu, skew: %lld\n",
                   nccl_dev_comm.rank, context_idx, signal_idx, static_cast<unsigned long long>(shadow),
                   static_cast<unsigned long long>(signal), static_cast<long long>(skew));
            found_bad_signal = true;
        }
    }
    if (found_bad_signal)
        trap_after_report();
}

namespace hang_detector {

constexpr bool kEnabled = (kDebugChecks & kDebugCheckHangDetector) != 0;

// With the check on, an idle lane of the warp watches the lanes that post, since `ncclGin::put` has no timeout.
// The words live in the last `kNumHangDetectorSmemBytes` of the dynamic shared memory, outside the TMA pool.
__device__ __forceinline__ unsigned* words() {
    extern __shared__ __align__(16) int8_t hang_detector_smem[];
    uint32_t num_smem_bytes;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(num_smem_bytes));
    return reinterpret_cast<unsigned*>(hang_detector_smem + (num_smem_bytes - kNumHangDetectorSmemBytes));
}

// The lowest lane that does not post, or -1 when every lane posts
__device__ __forceinline__ int watcher_lane(const unsigned& poster_mask) {
    return ptx::ffs(~poster_mask);
}

__device__ __noinline__ void report(const unsigned stuck_mask, const char* where,
                                    const int scaleout_rank_idx, const int scaleup_rank_idx) {
    printf("DeepEP GIN post stuck (%s), scale-out: %d, scale-up: %d, sm: %d, warp: %d, stuck lanes: 0x%08x\n",
           where, scaleout_rank_idx, scaleup_rank_idx, static_cast<int>(blockIdx.x), static_cast<int>(threadIdx.x / 32), stuck_mask);
    trap_after_report();
}

// For a converged warp. The posting lanes run `post`, and the lowest idle lane waits for them in `timeout_while`
// and prints the lanes that did not return. One word per warp of the CTA, so every warp of the kernel may call it.
template <int64_t kNumTimeoutCycles, int kNumThreads, typename post_t>
__device__ __forceinline__ void guarded_post(const bool& do_post, const post_t& post, const char* where,
                                             const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    EP_STATIC_ASSERT(kNumThreads / 32 <= kNumHangDetectorWords, "One word per warp of the CTA");
    if constexpr (not kEnabled) {
        if (do_post)
            post();
        return;
    }
    const unsigned post_mask = ptx::gather(do_post);
    if (post_mask == 0)
        return;
    auto* returned = words() + threadIdx.x / 32;
    const int lane_idx = static_cast<int>(threadIdx.x) % 32;
    const int watcher_lane_idx = watcher_lane(post_mask);
    if (lane_idx == watcher_lane_idx)
        *returned = 0;
    __syncwarp();
    if (do_post) {
        post();
        atomicOr_block(returned, 1u << lane_idx);
    } else if (lane_idx == watcher_lane_idx) {
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto returned_mask = *reinterpret_cast<volatile unsigned*>(returned);
            if (returned_mask == post_mask)
                return true;
            if (is_last_check)
                report(post_mask & ~returned_mask, where, scaleout_rank_idx, scaleup_rank_idx);
            return false;
        });
    }
    __syncwarp();
}

// For a warp whose lanes post on their own schedule (the combine proxy). Each posting lane stamps the clock into its
// word for the duration of its put, and the lowest idle lane watches the stamps until every posting lane has called
// `done`. One word per lane, so only `kNumLaneWarps` warps of the kernel may call these, the proxy warp alone in
// the combine. The watcher's own word holds the done mask.
static constexpr int kStampShift = 11;
static constexpr unsigned kWatchPollNanos = 2000;

__device__ __forceinline__ unsigned stamp() {
    return static_cast<unsigned>(clock64() >> kStampShift) | 1u;
}

template <int kNumLaneWarps>
__device__ __forceinline__ void init() {
    EP_STATIC_ASSERT(kNumLaneWarps * 32 <= kNumHangDetectorWords, "One word per lane of the warps that index by lane");
    if constexpr (kEnabled) {
        words()[threadIdx.x % 32] = 0;
        __syncwarp();
    }
}

__device__ __forceinline__ void post_begin() {
    if constexpr (kEnabled)
        reinterpret_cast<volatile unsigned*>(words())[threadIdx.x % 32] = stamp();
}

__device__ __forceinline__ void post_end() {
    if constexpr (kEnabled)
        reinterpret_cast<volatile unsigned*>(words())[threadIdx.x % 32] = 0;
}

__device__ __forceinline__ void done(const unsigned& poster_mask) {
    if constexpr (kEnabled)
        atomicOr_block(words() + watcher_lane(poster_mask), 1u << (threadIdx.x % 32));
}

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void watch(const unsigned& poster_mask, const char* where,
                                      const int& scaleout_rank_idx, const int& scaleup_rank_idx) {
    if constexpr (kEnabled) {
        const int lane_idx = static_cast<int>(threadIdx.x) % 32;
        const int watcher_lane_idx = watcher_lane(poster_mask);
        const bool every_lane_posts = poster_mask == 0xffffffffu;
        if (every_lane_posts or lane_idx != watcher_lane_idx)
            return;
        constexpr int kNumTimeoutStamps = static_cast<int>(kNumTimeoutCycles >> kStampShift);
        auto* stamps = reinterpret_cast<volatile unsigned*>(words());
        unsigned stuck_mask = 0;
        while (stuck_mask == 0 and (stamps[watcher_lane_idx] & poster_mask) != poster_mask) {
            const auto now = stamp();
            #pragma unroll 1
            for (auto pending = poster_mask; pending != 0; pending &= pending - 1) {
                const int lane = ptx::ffs(pending);
                const auto lane_stamp = stamps[lane];
                if (lane_stamp != 0 and static_cast<int>(now - lane_stamp) > kNumTimeoutStamps)
                    stuck_mask |= 1u << lane;
            }
            __nanosleep(kWatchPollNanos);
        }
        if (stuck_mask != 0)
            report(stuck_mask, where, scaleout_rank_idx, scaleup_rank_idx);
    }
}

}  // namespace hang_detector

}  // namespace deep_ep::elastic::diag
