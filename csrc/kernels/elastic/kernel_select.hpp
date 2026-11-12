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

#include <cstdio>
#include <string>

#include <nccl.h>

#include <deep_ep/common/exception.cuh>

#include "../../utils/system.hpp"

namespace deep_ep::elastic {

// `EP_HYBRID_KERNEL` selects the hybrid (scale-out) dispatch/combine kernel pair at
// JIT-generation time:
//
//   "unordered" (default) — per-part batched puts with in-band token headers and
//                           counting signals. Makes no assumption about network
//                           delivery order, so strong signal is not required, and
//                           weak signal is enough.
//
//   "ordered"             — the upstream kernels: the sender publishes a tail via a
//                           trailing signal and the receiver assumes all data
//                           preceding the tail has landed. Requires the backend to
//                           support strong signals and VA signals.
//
// The value is read once per process. It changes the JIT-generated source (header and
// kernel names differ), so the JIT cache distinguishes the two variants automatically.
static bool use_ordered_hybrid_kernel() {
    static const bool ordered = [] {
        const auto value = get_env<std::string>("EP_HYBRID_KERNEL");
        bool result = false;
        if (value.empty() or value == "unordered") {
            result = false;
        } else if (value == "ordered") {
            result = true;
        } else {
            EP_HOST_ASSERT(false and "EP_HYBRID_KERNEL must be `ordered` or `unordered`");
        }
        printf("DeepEP hybrid kernel selection: %s\n", result ? "ordered" : "unordered");
        return result;
    }();
    return ordered;
}

// The mask of `EP_DEBUG_CHECKS` (see `deep_ep/common/compiled.cuh`), read once per process. The JIT
// generators pass it to the kernels as the template argument `kDebugChecks`
static int get_debug_check_mask() {
    static const int mask = get_debug_checks();
    return mask;
}

// The unordered kernels come in two generations. The current one hands the combine's whole scale-out
// side to a single proxy lane per SM, which posts thread-exclusively (NCCL_GIN_RESOURCE_SHARING_THREAD)
// on a QP the SM owns; its dispatch records the forward list as two FIFOs (remote tokens first) for
// that combine to replay. The thread-exclusive put (NVIDIA/nccl ad35c997) is first released in NCCL 2.32.6.
// On an older NCCL the same lane would post on the CTA-shared path, serially, and the combine is
// slower than the previous kernels; so without the thread-exclusive put the JIT selects the previous
// generation of the dispatch, its copy epilogue and the combine (`*_legacy.cuh`, the kernels of DeepEP
// a209bc5, the main this branch is based on). The value is read once per process from the loaded library, the same one the JIT passes as `EP_NCCL_RUNTIME_VERSION`;
// the header names differ, so the JIT cache keeps the two generations apart.
static bool use_thread_put_unordered_kernels() {
    static const bool thread_put = [] {
        int version = 0;
        NCCL_CHECK(ncclGetVersion(&version));
        const bool result = version >= NCCL_VERSION(2, 32, 6);
        if (not result)
            printf("DeepEP unordered kernels: NCCL %d.%d.%d has no thread-exclusive GIN put, using the previous "
                   "dispatch/combine generation\n", version / 10000, (version % 10000) / 100, version % 100);
        return result;
    }();
    return thread_put;
}

} // namespace deep_ep::elastic
