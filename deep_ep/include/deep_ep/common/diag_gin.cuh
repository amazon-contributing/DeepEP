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

// The `token_sync` check of `EP_DEBUG_CHECKS`. The bit is in `compiled.cuh`, see `diag.cuh`

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
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

}  // namespace deep_ep::elastic::diag
