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
// `EP_COMBINE_GATE` selects how the unordered combine's return gate synchronizes:
//
//   "signal" (default) — counting GIN indexed signals (the existing path).
//
//   "flag"             — a 4-byte iteration number written by putValue into a
//                        symmetric workspace slot; the receiver polls it. The
//                        sender flushes its data puts before writing the flag.
//
// Read once per process; JIT-cached per variant. Only the unordered kernel has
// this gate, so "flag" rejects EP_HYBRID_KERNEL=ordered.
static bool use_flag_combine_gate() {
    static const bool flag = [] {
        const auto value = get_env<std::string>("EP_COMBINE_GATE");
        bool result = false;
        if (value.empty() or value == "signal") {
            result = false;
        } else if (value == "flag") {
            result = true;
        } else {
            EP_HOST_ASSERT(false and "EP_COMBINE_GATE must be `signal` or `flag`");
        }
        printf("DeepEP combine gate selection: %s\n", result ? "flag" : "signal");
        return result;
    }();
    return flag;
}

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

} // namespace deep_ep::elastic
