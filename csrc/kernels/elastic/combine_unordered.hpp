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


namespace deep_ep::elastic {

// Combine and combine reduce epilogue arguments used only by the unordered hybrid kernels
struct UnorderedCombineExtendedArgs {
    // Captured by dispatch and used by combine to correlate the `gin.put` batch granularity to tokens,
    // so combine knows how many tokens each batch carries. Usually every batch carries the same token
    // count, and only the last batch is a bit smaller as it holds the remainder
    int* token_map_at_dispatch;
    // Number of local tokens, i.e. the valid rows of `token_map_at_dispatch` (allocated for
    // `num_max_tokens_per_rank`), used to bound the iteration over it
    int num_combined_tokens;
    // GIN channel count, needed by the combine epilogue to correlate the landed data with the token location:
    // the unordered TX and RX layouts are the same, so copying into the user's tensor requires knowing where
    // the combine kernel placed each partial, which is tightly coupled to the combine internals
    int num_channels;
};

}  // namespace deep_ep::elastic
