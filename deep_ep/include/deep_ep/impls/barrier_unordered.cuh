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

#include <deep_ep/common/comm_unordered.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          int kNumSMs, int kNumThreads,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int64_t kNumTimeoutCycles,
          bool kSequential>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* workspace,
             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);

    // Barrier only uses the first part of workspace, so making `num_experts` as 0 is fine
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    if constexpr (kSequential) {
        // Scaleout barrier
        if constexpr (kNumScaleoutRanks > 1)
            comm::unordered::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                                       kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, false>(
                    gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, true, false);

        // Scaleup barrier, and it needs to flush the RDMA requests issued by scaleout barrier 
        comm::unordered::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs, kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
                gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, false, true);
    } else {
        comm::unordered::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, false>(
                gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }
}

} // namespace deep_ep::elastic
