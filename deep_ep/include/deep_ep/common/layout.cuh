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

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::layout {

struct WorkspaceLayout {
    void* workspace;

    int num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_experts, num_experts_per_rank;

    // We want to fix the layout position for all settings,
    // so that one buffer can be reused for all cases
    static constexpr int kNumMaxRanks = 1024;
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxExpertsPerRank = 256;
    static constexpr int kNumMaxInflightAGRS = 32;
    static constexpr int kNumMaxParts = 64;

    static constexpr int64_t kNumBarrierSignalBytes = 16;

    // Rail barrier flag storage. One slot per (tag, phase, peer). Tags come from the
    // `k*Tag*` constants in `comm.cuh` and every instantiation of the rail barrier checks
    // its tag against this bound; phases alternate so that only one write per slot is
    // ever in flight (see the ordering argument at the rail branch of
    // `gin_barrier_wo_local_sync`). The flags and the round counters behind them are
    // monotonic for the life of the buffer: they are zeroed once at construction and must
    // never be reset, because a rank whose counters restart while its peers' flags do not
    // either passes a barrier without waiting or waits forever.
    static constexpr int kNumBarrierTags = 16;
    static constexpr int kNumBarrierPhases = 2;

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,
                    const int& num_scaleout_ranks,
                    const int& num_scaleup_ranks,
                    const int& num_experts):
        workspace(workspace),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
    }

    static int64_t get_num_bytes() {
        // Pure NVLink scaleup barrier signals
        int64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;

        // Notify reduction workspace
        num_bytes += (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t);

        // Scaleup notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;

        // Scaleup atomic sender count
        num_bytes += kNumMaxRanks * sizeof(int);

        // Scaleout notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int) * 2;

        // Scaleout channel metadata (finish flag and tails)
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int64_t);

        // Channel aggregated into the scaleup domains
        // Also reused for channel scaleup tail
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int);

        // Rank send/recv count, for PP prev/next ranks
        num_bytes += 2 * 2 * sizeof(int64_t);

        // AGRS signals
        num_bytes += (kNumMaxInflightAGRS + 1) * kNumMaxRanks * sizeof(int);

        // Rail barrier iteration flags, written by peers
        num_bytes += kNumBarrierTags * kNumBarrierPhases * kNumMaxRanks * sizeof(uint32_t);

        // Rail barrier local round counters, never written by a peer
        num_bytes += 2 * kNumBarrierTags * kNumMaxRanks * sizeof(uint32_t);

        return num_bytes;
    }

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() const {
        return static_cast<unsigned long long*>(workspace);
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) const {
        return math::advance_ptr<int>(workspace, (2 + phase) * sizeof(int));
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return math::advance_ptr<int64_t>(workspace, kNumBarrierSignalBytes);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();
        return base_ptr + scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks;
        return base_ptr + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_scaleout_rank_expert_count_ptr<true>(),
            (kNumMaxRanks + kNumMaxExperts) * sizeof(int) * 2);
        return base_ptr + (channel_idx * num_scaleout_ranks + scaleout_rank_idx);
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_channel_signaled_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int64_t));
        return base_ptr + (channel_idx * num_scaleup_ranks + scaleup_rank_idx);
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_channel_scaleup_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + slot * kNumMaxRanks + rank_idx;
    }

    __forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
        return base_ptr + rank_idx;
    }

    // Rail barrier flag written by a peer. Monotonic across rounds: zeroed once at
    // construction and never reset, which is a carve-out from the keep-workspace-zero
    // rule in `buffer.hpp`.
    __forceinline__ __device__ __host__ uint32_t* get_rail_barrier_flag_ptr(
        const int& tag, const int& phase, const int& slot_idx) const {
        const auto base_ptr = math::advance_ptr<uint32_t>(
            get_agrs_session_signal_ptr(0), kNumMaxRanks * sizeof(int));
        return base_ptr + ((tag * kNumBarrierPhases + phase) * kNumMaxRanks + slot_idx);
    }

    // Local round counters for the rail barrier, one pair per (tag, index). Never written
    // by a peer, so these carry no ordering requirement. Same monotonic carve-out.
    __forceinline__ __device__ __host__ uint32_t* get_rail_barrier_send_seq_ptr(
        const int& tag, const int& peer_idx) const {
        const auto base_ptr = math::advance_ptr<uint32_t>(
            get_rail_barrier_flag_ptr(0, 0, 0),
            kNumBarrierTags * kNumBarrierPhases * kNumMaxRanks * sizeof(uint32_t));
        return base_ptr + (tag * kNumMaxRanks + peer_idx);
    }

    __forceinline__ __device__ __host__ uint32_t* get_rail_barrier_recv_seq_ptr(
        const int& tag, const int& slot_idx) const {
        const auto base_ptr = math::advance_ptr<uint32_t>(
            get_rail_barrier_send_seq_ptr(0, 0), kNumBarrierTags * kNumMaxRanks * sizeof(uint32_t));
        return base_ptr + (tag * kNumMaxRanks + slot_idx);
    }
};

struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;
    // Per-token scale-out header present iff true. See kHdrBytes note below.
    bool with_scaleout_hdr;
    int num_topk, num_metadata_bytes;
    void* base;

    static constexpr int kHdrBytes = sizeof(int64_t);

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata,
                void* base = nullptr, const bool& with_scaleout_hdr = false) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        with_metadata(with_metadata),
        with_scaleout_hdr(with_scaleout_hdr),
        num_topk(num_topk),
        num_metadata_bytes((with_scaleout_hdr ? math::align<int>(hdrless_content_bytes(num_topk, with_metadata), sizeof(int64_t)) + kHdrBytes
                                              : hdrless_content_bytes(num_topk, with_metadata))),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    __forceinline__ __device__ __host__ static int hdrless_content_bytes(const int& num_topk, const bool& with_metadata) {
        return num_topk * (sizeof(int) + sizeof(float)) + (with_metadata ? (1 + num_topk) * sizeof(int) : 0);
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        return math::advance_ptr<float>(get_topk_idx_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ __host__ int64_t* get_hdr_ptr() const {
        const int hdr_offset = math::align<int>(hdrless_content_bytes(num_topk, with_metadata), sizeof(int64_t));
        return static_cast<int64_t*>(static_cast<void*>(
            math::advance_ptr<int8_t>(get_metadata_ptr(), hdr_offset)));
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;

    void* base;

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx,
                           token_layout.with_scaleout_hdr);
    }
};

}  // namespace deep_ep::elastic
