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

// GIN signal check. The host compiles it in with `EP_DEBUG_CHECKS=1`, see
// `csrc/jit/compiler.hpp`. When the flag is off, this header compiles to nothing.

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/diag.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/gin_resource_alloc.cuh>

namespace deep_ep::elastic::diag {

// Each GIN indexed signal has a hardware counter and a software shadow. Remote peers
// add to the counter. The local kernel adds to the shadow the count it waited for. At
// kernel entry the two must be level, with one exception. A rail barrier slot can be
// one ahead when a peer has already entered the next barrier. It cannot be two ahead,
// because the peer needs our next increment first. A data slot cannot be ahead at all:
// the previous kernel waited for every increment it expected before it ended. The
// scale-up barrier of the unordered kernels runs over NVLink and uses no GIN signal.
//
// The check reads every (context, id) slot. It does not use the kernels' own index
// math, so it also sees an increment on a slot that no channel waits for. A bad skew
// does not heal by itself, because the counters never reset. So one read at each kernel
// entry is enough, and the check reports a slot when its skew changes, not on every
// launch. Each kernel variant is its own JIT module with its own state, so one bad slot
// gives one line per variant. The check is read-only and does not wait.
//
// A barrier slot with one extra increment shows a skew of 1 most of the time, which is
// legal. The check reports it at the first entry where the peer is also one barrier
// ahead, so the skew is 2. That can take many iterations.
static constexpr int64_t kMaxBarrierSlotSkew = 1;
static constexpr int64_t kMaxDataSlotSkew = 0;

// All contexts together have fewer signals than the QP budget, see `gin_alloc`
static constexpr int kNumMaxSlots = gin_alloc::kTotalQPBudget;

#if EP_DEBUG_CHECKS
// The skew last reported for each slot, `0` when the slot was clean at the last check
static __device__ int64_t last_reported_skew[kNumMaxSlots] = {};
#endif

// Check all GIN signals of this rank. All threads of the grid call this function before
// the entry barrier. The rail barrier uses context 0, ids `0 .. kNumScaleoutRanks - 2`,
// see `comm::gin_barrier_wo_local_sync`. `iteration` is `-1` when the kernel has no
// iteration count
template <int kNumScaleoutRanks>
__device__ __forceinline__ void check_gin_signals(
    const ncclDevComm_t& nccl_dev_comm,
    const char* kernel_name, const int& iteration, const int& rank_idx,
    const int& global_thread_idx, const int& num_global_threads) {
#if EP_DEBUG_CHECKS
    constexpr int kNumBarrierSlots = kNumScaleoutRanks - 1;
    const int num_signals = nccl_dev_comm.ginSignalCount;
    const int num_slots = static_cast<int>(nccl_dev_comm.ginContextCount) * num_signals;
    EP_DEVICE_ASSERT(num_slots <= kNumMaxSlots);
    for (int slot = global_thread_idx; slot < num_slots; slot += num_global_threads) {
        const int context_idx = slot / num_signals, signal_idx = slot % num_signals;
        const auto signal_id = static_cast<ncclGinSignal_t>(signal_idx);

        // The raw handle is enough to read a signal and its shadow
        const ncclGin gin(nccl_dev_comm, context_idx, NCCL_GIN_RESOURCE_SHARING_GPU);
        const auto shadow = *gin.getSignalShadowPtr(signal_id) & ((uint64_t{1} << comm::kNumSignalBits) - 1);
        // The hardware counter can be narrower than the shadow and wrap. Compare both modulo
        // `comm::kNumSignalBits`, the same as the kernels' own waits
        const auto signal = gin.readSignal(signal_id, comm::kNumSignalBits, cuda::memory_order_acquire);

        const bool is_barrier_slot = context_idx == 0 and signal_idx < kNumBarrierSlots;
        const auto max_skew = is_barrier_slot ? kMaxBarrierSlotSkew : kMaxDataSlotSkew;
        const auto skew = comm::signal_delta(signal, shadow);
        const auto bad_skew = (skew < 0 or skew > max_skew) ? skew : 0;
        if (bad_skew == last_reported_skew[slot])
            continue;
        last_reported_skew[slot] = bad_skew;
        if (bad_skew != 0) {
            EP_DIAG_REPORT("DeepEP debug check: GIN signal skew at %s entry, rank: %d, iteration: %d, "
                           "context: %d/%d, signal: %d/%d, slot: %s, shadow: %llu, signal: %llu, "
                           "skew: %lld, max legal: %lld\n",
                           kernel_name, rank_idx, iteration,
                           context_idx, static_cast<int>(nccl_dev_comm.ginContextCount), signal_idx, num_signals,
                           is_barrier_slot ? "barrier" : "data",
                           static_cast<unsigned long long>(shadow), static_cast<unsigned long long>(signal),
                           static_cast<long long>(skew), static_cast<long long>(max_skew));
        }
    }
#endif
}

}  // namespace deep_ep::elastic::diag
