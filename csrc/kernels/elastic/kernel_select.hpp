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

// `EP_HYBRID_KERNEL` selects the hybrid (scale-out) kernels: "unordered" (default) does not assume
// in-order delivery and uses GIN weak signals, "ordered" does and uses GIN strong signals
inline bool use_unordered_hybrid_kernel() {
    static const bool unordered = [] {
        const auto value = get_env<std::string>("EP_HYBRID_KERNEL");
        bool result = true;
        if (value.empty() or value == "unordered") {
            result = true;
        } else if (value == "ordered") {
            result = false;
        } else {
            EP_HOST_ASSERT(false and "EP_HYBRID_KERNEL must be `ordered` or `unordered`");
        }
        if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("DeepEP hybrid kernel selection: %s\n", result ? "unordered" : "ordered");
        return result;
    }();
    return unordered;
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
// 13bb6fe, the main this branch is based on). The GIN layout follows the generation: the previous
// kernels keep their original default of 11 contexts (`kLegacyGinContextCnt`). The value is read
// once per process from the loaded library, the same one the JIT passes as
// `EP_NCCL_RUNTIME_VERSION`; the header names differ, so the JIT cache keeps the two generations
// apart.
inline bool use_thread_put_unordered_kernels() {
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
