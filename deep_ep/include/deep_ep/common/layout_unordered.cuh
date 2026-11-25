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


#include <cstdint>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep {

// Per-channel shared-memory TMA buffer count for the unordered hybrid dispatch kernel:
// 1 for the scale-out send warp + 2 for the forward warp (double-buffered
// loads). The host channel budget and the kernel pool must both use this.
constexpr int kNumDispatchSendBuffers = 1;
constexpr int kNumDispatchFwdBuffers = 2;
constexpr int kNumDispatchBuffersPerChannel = kNumDispatchSendBuffers + kNumDispatchFwdBuffers;

}  // namespace deep_ep

namespace deep_ep::elastic::layout {

// Maximum number of token parts per scale-out channel in the unordered dispatch
static constexpr int kNumMaxParts = 64;

// Scale-out token of the unordered hybrid kernels: the regular token layout followed by an
// 8-byte in-band header, placed after the metadata at the next 8-byte boundary.
struct TokenLayoutUnordered : TokenLayout {
    static constexpr int kHdrBytes = sizeof(int64_t);

    __forceinline__ __device__ __host__
    TokenLayoutUnordered(const int& num_hidden_bytes, const int& num_sf_bytes,
                         const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        TokenLayout(num_hidden_bytes, num_sf_bytes, num_topk, with_metadata, base) {
        num_metadata_bytes = math::align<int>(num_metadata_bytes, sizeof(int64_t)) + kHdrBytes;
    }

    __forceinline__ __device__ __host__ int64_t* get_hdr_ptr() const {
        return math::advance_ptr<int64_t>(get_metadata_ptr(), num_metadata_bytes - kHdrBytes);
    }
};

// `BufferLayout` over `TokenLayoutUnordered`, so that sub-buffers and tokens keep the header
template <bool kWithMBarrier>
struct BufferLayoutUnordered : BufferLayout<kWithMBarrier> {
    using base_t = BufferLayout<kWithMBarrier>;

    __forceinline__ __device__ __host__
    BufferLayoutUnordered(const TokenLayoutUnordered& token_layout,
                          const int& num_ranks,
                          const int& max_num_tokens_per_rank,
                          void* base = nullptr) :
        base_t(token_layout, num_ranks, max_num_tokens_per_rank, base) {}

    __forceinline__ __device__ __host__
    explicit BufferLayoutUnordered(const base_t& buffer) : base_t(buffer) {}

    __forceinline__ __device__ __host__
    BufferLayoutUnordered get_rank_buffer(const int& rank_idx) const {
        return BufferLayoutUnordered(base_t::get_rank_buffer(rank_idx));
    }

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayoutUnordered get_channel_buffer(const int& channel_idx) const {
        return BufferLayoutUnordered(base_t::template get_channel_buffer<kNumTokensPerChannel>(channel_idx));
    }

    __forceinline__ __device__ __host__
    TokenLayoutUnordered get_token_buffer(const int& token_idx, const bool& global = false) const {
        const auto& token_layout = base_t::token_layout;
        return TokenLayoutUnordered(token_layout.num_hidden_bytes, token_layout.num_sf_bytes,
                                   token_layout.num_topk, token_layout.with_metadata,
                                   base_t::get_token_buffer(token_idx, global).get_base_ptr());
    }
};

}  // namespace deep_ep::elastic::layout

namespace deep_ep::elastic {

// Packing for `token_map_at_dispatch[token][k]`. Written by dispatch, read by the combine epilogue.
// Layout (int32):
//   bits [30:26] — dst_scaleout_rank (5 bits, 0..31)
//   bits [25:12] — slot within `recv[dst_scaleout_rank][channel × slots_per_channel + slot]` (14 bits, 0..16383)
//   bits [11:0]  — channel index (12 bits, 0..4095)
// Field widths add up to 31, leaving bit 31 clear on any valid entry. The epilogue reuses the
// `-1`/`< 0` sentinel convention that `compute_topk_slots` produces for trailing (unused) `topk_slot_idx`
// entries — `combine_reduce` predicates loads via `ldg_with_gez_pred`'s sign check. Keeping bit 31
// clear on valid entries preserves this shared sentinel invariant so no extra masking is needed.
constexpr int kCombineRecvMapChannelBits = 12;
constexpr int kCombineRecvMapSlotBits    = 14;
constexpr int kCombineRecvMapRankBits    = 5;
constexpr int kCombineRecvMapChannelMask = (1 << kCombineRecvMapChannelBits) - 1;
constexpr int kCombineRecvMapSlotMask    = (1 << kCombineRecvMapSlotBits) - 1;
constexpr int kCombineRecvMapRankMask    = (1 << kCombineRecvMapRankBits) - 1;
constexpr int kCombineRecvMapSlotShift    = kCombineRecvMapChannelBits;
constexpr int kCombineRecvMapRankShift    = kCombineRecvMapChannelBits + kCombineRecvMapSlotBits;

__device__ __host__ __forceinline__
int pack_combine_recv_addr(const int& rank, const int& slot, const int& channel) {
    return (rank << kCombineRecvMapRankShift) | (slot << kCombineRecvMapSlotShift) | channel;
}

__device__ __host__ __forceinline__
void unpack_combine_recv_addr(const int& packed, int& rank, int& slot, int& channel) {
    rank    = (packed >> kCombineRecvMapRankShift) & kCombineRecvMapRankMask;
    slot    = (packed >> kCombineRecvMapSlotShift) & kCombineRecvMapSlotMask;
    channel = packed & kCombineRecvMapChannelMask;
}

}  // namespace deep_ep::elastic
