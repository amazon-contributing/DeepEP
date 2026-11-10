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
#include <cstdio>

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/gin_resource_alloc_unordered.cuh>

#include "../../utils/system.hpp"

namespace deep_ep::elastic {

// GIN requirements of the unordered hybrid kernels: shared contexts and weak signals only
static gin_alloc::GinResourceConfig configure_unordered_gin(const ncclComm_t& comm, ncclDevCommRequirements_t& reqs,
                                                            const int& num_ranks, const bool& with_gin,
                                                            int& num_allocated_qps) {
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 31, 2)
    EP_HOST_ASSERT(false and "EP_HYBRID_KERNEL=unordered requires NCCL 2.31.2 or later");
#endif

    const int requested = num_allocated_qps == 0 ? gin_alloc::kDefaultGinContextCnt : num_allocated_qps;
    const int ctx = std::clamp(requested, gin_alloc::kMinGinContextCnt, gin_alloc::kMaxGinContextCnt);
    if (ctx != requested)
        printf("[WARN] DeepEP clamped num_allocated_qps from %d to %d: the unordered "
               "GIN layout supports [%d, %d] contexts (one GIN context supplies one QP)\n",
               requested, ctx, gin_alloc::kMinGinContextCnt, gin_alloc::kMaxGinContextCnt);
    num_allocated_qps = ctx;

    gin_alloc::GinResourceConfig gin_config = {};
    gin_config.gin_context_cnt = ctx;
    if (not with_gin)
        return gin_config;

    const int num_rdma_ranks = ncclTeamRail(comm).nRanks;
    EP_HOST_ASSERT(num_ranks == ncclTeamLsa(comm).nRanks * num_rdma_ranks);

    const ncclDevCommRequirements_t default_reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginContextCount = ctx;
    reqs.ginExclusiveContexts = false;
    if (num_rdma_ranks > 1) {
        gin_config = gin_alloc::make_gin_resources(ctx);
        reqs.ginSignalCount = gin_config.gin_indexed_signals_cnt;
    } else {
        reqs.ginSignalCount = default_reqs.ginSignalCount;
    }
    EP_HOST_ASSERT(gin_config.gin_indexed_signals_cnt >= (num_rdma_ranks - 1) and
                   "GIN indexed-signal budget cannot give each peer rail team a dedicated "
                   "signal; reduce num_allocated_qps to raise the per-context signal count");

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 2)
    reqs.ginVaSignalsRequired = false;
    reqs.ginStrongSignalsRequired = false;
#endif

    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("GIN layout: gin_context_cnt=%d, gin_indexed_signals_cnt=%d\n",
               gin_config.gin_context_cnt, gin_config.gin_indexed_signals_cnt);
    return gin_config;
}

}  // namespace deep_ep::elastic
