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

// Debug checks. The host compiles them in with `EP_DEBUG_CHECKS=1`, see
// `csrc/jit/compiler.hpp`. When the flag is off, this header compiles to nothing.

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>

#ifndef EP_DEBUG_CHECKS
#define EP_DEBUG_CHECKS 0
#endif

namespace deep_ep::elastic::diag {

// Element type of a token payload. The kernels see the payload as bytes
static constexpr int kPayloadBF16 = 0;
static constexpr int kPayloadFP8E4M3 = 1;

// Dispatch takes BF16, or FP8 E4M3 with scale factors (see `ElasticBuffer.dispatch`),
// so the scale factors give the type. Other dtypes are outside the API: the scan reads
// them as BF16, and can miss non-finite values (FP16) or give false reports (FP32)
constexpr int get_dispatch_payload_type(const int& num_sf_packs) {
    return num_sf_packs > 0 ? kPayloadFP8E4M3 : kPayloadBF16;
}

constexpr int get_payload_elem_bytes(const int& payload_type) {
    return payload_type == kPayloadBF16 ? 2 : 1;
}

// Maximum number of reports for each kernel and process. One bad buffer can
// contain many bad tokens, and device printf drops lines when its FIFO is full
static constexpr unsigned int kNumMaxReports = 32;

#if EP_DEBUG_CHECKS
static __device__ unsigned int num_reports = 0;
#define EP_DIAG_REPORT(...) \
    do { if (atomicAdd(&::deep_ep::elastic::diag::num_reports, 1u) < ::deep_ep::elastic::diag::kNumMaxReports) printf(__VA_ARGS__); } while (0)
#else
#define EP_DIAG_REPORT(...) do {} while (0)
#endif

// Result of the scan of one token. The values are the same in all lanes
struct NonFiniteScan {
    int num_bad;
    int first_bad_idx;          // -1 when the token is clean
    unsigned int first_bad_bits;
};

// Count the non-finite elements of one token and find the first one. A full warp
// calls this function. `ptr` is 16-byte aligned shared or global memory.
//
// A BF16 element `S EEEEEEEE MMMMMMM` is non-finite when its exponent bits are all
// ones: 0x7f80 is Inf, and larger values are NaN. An FP8 E4M3 element `S EEEE MMM`
// has no Inf and one NaN encoding: exponent and mantissa all ones. Both tests are
// "these bits are all ones". The test runs on all elements of a 32-bit word at
// once: mask the word to those bits, add one at their lowest bit, and keep the top
// bit of each element. The carry sets the top bit only when the bits were all ones.
// The mask drops the sign, so the sum is at most the top bit, and the carry cannot
// cross into the next element.
//
// BF16 example: NaN is 0x7fc0. 0x7fc0 & 0x7f80 = 0x7f80, + 0x0080 = 0x8000, so the
// top bit is set. 1.0 is 0x3f80. 0x3f80 & 0x7f80 = 0x3f80, + 0x0080 = 0x4000, so the
// top bit is clear
//
// The scan has two passes. The fast pass has no branches and no reductions. It
// only asks "did any lane see a hit". This is the full cost of a clean token. The
// slow pass counts and locates the bad elements. It runs only after a hit
template <int kNumBytes, int kPayloadType>
__device__ __forceinline__ NonFiniteScan scan_nonfinite(const void* ptr, const int& lane_idx) {
    EP_STATIC_ASSERT(kNumBytes % sizeof(int4) == 0, "Token must be a multiple of 16 bytes");
    EP_STATIC_ASSERT(kPayloadType == kPayloadBF16 or kPayloadType == kPayloadFP8E4M3, "Unsupported payload type");
    constexpr bool kIsBF16 = kPayloadType == kPayloadBF16;
    constexpr int kElemBits = kIsBF16 ? 16 : 8, kNumElemsPerWord = 32 / kElemBits;
    constexpr unsigned int kOnesMask = kIsBF16 ? 0x7f807f80u : 0x7f7f7f7fu;   // The bits that must all be ones
    constexpr unsigned int kOnesLsb  = kIsBF16 ? 0x00800080u : 0x01010101u;   // The lowest of these bits
    constexpr unsigned int kTopBit   = kIsBF16 ? 0x80008000u : 0x80808080u;   // The top bit of each element
    constexpr int kNumVecs = kNumBytes / static_cast<int>(sizeof(int4));
    const auto* vec_ptr = static_cast<const uint4*>(ptr);

    // One bit for each element, at the top bit of the element. The bit is set when
    // the element is non-finite
    const auto hits_in = [](const unsigned int& word) { return ((word & kOnesMask) + kOnesLsb) & kTopBit; };

    // Fast pass: fully unrolled, no branches, no reductions
    unsigned int any_hits = 0;
    #pragma unroll
    for (int it = 0; it < (kNumVecs + 31) / 32; ++ it) {
        const int v = it * 32 + lane_idx;
        if (kNumVecs % 32 == 0 or v < kNumVecs) {
            const auto vec = vec_ptr[v];
            any_hits |= hits_in(vec.x) | hits_in(vec.y) | hits_in(vec.z) | hits_in(vec.w);
        }
    }
    if (not __any_sync(0xffffffff, any_hits != 0))
        return {0, -1, 0};

    // Slow pass, only after a hit: count the bad elements, and find the first one and its bits
    constexpr unsigned int kNone = 0x7fffffffu;
    int num_bad = 0;
    unsigned int first_bad_idx = kNone, first_bad_bits = 0;
    for (int v = lane_idx; v < kNumVecs; v += 32) {
        const auto vec = vec_ptr[v];
        const unsigned int words[4] = {vec.x, vec.y, vec.z, vec.w};
        #pragma unroll
        for (int w = 0; w < 4; ++ w) {
            const auto hits = hits_in(words[w]);
            num_bad += __popc(hits);
            if (hits != 0 and first_bad_idx == kNone) {
                // Little-endian: the lowest set bit belongs to the element with the lowest index
                const int e = (__ffs(hits) - 1) / kElemBits;
                first_bad_idx = (v * 4 + w) * kNumElemsPerWord + e;
                first_bad_bits = (words[w] >> (e * kElemBits)) & ((1u << kElemBits) - 1);
            }
        }
    }
    // Each lane saw only its own vectors. Combine them: the total count, the smallest
    // index, and the bits from the lane that found that index
    num_bad = ptx::reduce_add(num_bad);
    const auto min_idx = __reduce_min_sync(0xffffffff, first_bad_idx);
    const auto owner_lane_idx = __ffs(__ballot_sync(0xffffffff, first_bad_idx == min_idx)) - 1;
    first_bad_bits = __shfl_sync(0xffffffff, first_bad_bits, owner_lane_idx);
    return {num_bad, min_idx == kNone ? -1 : static_cast<int>(min_idx), first_bad_bits};
}


}  // namespace deep_ep::elastic::diag
