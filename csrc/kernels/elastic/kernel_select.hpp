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

// `EP_HYBRID_KERNEL` selects the hybrid (scale-out) kernels: "ordered" (default) assumes in-order
// delivery and uses GIN strong signals, "unordered" does not and uses GIN weak signals
inline bool use_unordered_hybrid_kernel() {
    static const bool unordered = [] {
        const auto value = get_env<std::string>("EP_HYBRID_KERNEL");
        bool result = false;
        if (value.empty() or value == "ordered") {
            result = false;
        } else if (value == "unordered") {
            result = true;
        } else {
            EP_HOST_ASSERT(false and "EP_HYBRID_KERNEL must be `ordered` or `unordered`");
        }
        if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("DeepEP hybrid kernel selection: %s\n", result ? "unordered" : "ordered");
        return result;
    }();
    return unordered;
}

} // namespace deep_ep::elastic
