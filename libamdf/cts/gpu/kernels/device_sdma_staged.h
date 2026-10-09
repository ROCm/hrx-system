// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_DEVICE_SDMA_STAGED_H_
#define AMDF_CTS_GPU_KERNELS_DEVICE_SDMA_STAGED_H_

#include <cstddef>
#include <cstdint>

namespace kernels::device_sdma_staged {

enum class Phase : uint32_t {
  kUpload = 0,
  kDownload = 1,
};

// One immutable argument block per transfer dispatch. The preceding dispatch
// has completed before this dispatch starts; only one publisher owns SDMA.
struct alignas(16) TransferArguments {
  // GPU view of the byte-indexed SDMA ring.
  uint64_t ring;
  // Aligned 64-bit consumed command frontier.
  uint64_t read_index;
  // Aligned 64-bit published command frontier.
  uint64_t write_index;
  // Write-only aligned 64-bit queue notification address.
  uint64_t notification;
  // Separate 64-byte-aligned SDMA completion word.
  uint64_t completion;
  // Persistent frontier at byte 0 and selection state at byte 8.
  uint64_t state;
  // This job's 64-byte selection record, shared with its compute consumer.
  uint64_t selection;
  // Four payload word counts in [1, 262144], fitting the slot's guarded extent.
  uint64_t lengths;
  // This job's complete output-slot readback, including prefix and suffix.
  uint64_t readback;
  // Eight immutable source pages with payloads beginning at byte 64.
  uint64_t source_address;
  // Input slot allocation; each payload starts at byte 64 of its slot.
  uint64_t input_address;
  // Output slot allocation with the same guarded layout.
  uint64_t output_address;
  // Numeric SDMA address for the same allocation as readback.
  uint64_t readback_address;
  // Numeric SDMA address for the same word as completion.
  uint64_t completion_address;
  // Power-of-two ring byte capacity in [4096, 2^32].
  uint64_t capacity;
  // Aligned source-page, input-slot and output-slot byte stride.
  uint64_t slot_byte_length;
  // Zero-based job ordinal; upload/download use generations 2*job+1/2*job+2.
  uint32_t job_index;
  // Power-of-two number of reusable input/output slot pairs.
  uint32_t slot_count;
  // Upload selects the payload; download consumes the completed computation.
  Phase phase;
  // Family-selected scope fields in COPY_LINEAR DWORD 2.
  uint32_t copy_control;
  // Family-selected complete FENCE header.
  uint32_t fence_header;
  // Queried USER_GCR acquire (bit 0) and release (bit 1) for this direction.
  uint32_t cache_flags;
};
static_assert(offsetof(TransferArguments, job_index) == 128);
static_assert(offsetof(TransferArguments, cache_flags) == 148);

// The compute dispatch's argument block is immutable. Its GPU-produced
// selection is ordinary data read after the dispatch's SYSTEM acquire.
struct alignas(16) ConsumerArguments {
  // Complete input-slot allocation.
  uint64_t input;
  // Complete output-slot allocation.
  uint64_t output;
  // This job's descriptor, produced by the preceding upload dispatch.
  uint64_t selection;
  // Byte stride shared by both slot allocations.
  uint64_t slot_byte_length;
};
static_assert(sizeof(ConsumerArguments) == 32);

struct alignas(64) Selection {
  // Source page selected from the preceding returned result.
  uint32_t page;
  // Reusable input/output pair selected for this job.
  uint32_t slot;
  // Number of payload words uploaded and processed.
  uint32_t word_count;
  // Selection state read before this upload.
  uint32_t prior_state;
  // Per-job transform addend, derived from prior_state and the job ordinal.
  uint32_t addend;
  // State derived from this job's returned first/last processed words.
  uint32_t next_state;
  // Published SDMA frontier after upload, including ring padding.
  uint64_t upload_frontier;
  // Published SDMA frontier after download, including ring padding.
  uint64_t download_frontier;
  // Untouched suffix following the authored descriptor fields.
  uint32_t guards[6];
};
static_assert(sizeof(Selection) == 64);
static_assert(offsetof(Selection, upload_frontier) == 24);
static_assert(offsetof(Selection, download_frontier) == 32);

}  // namespace kernels::device_sdma_staged

#endif  // AMDF_CTS_GPU_KERNELS_DEVICE_SDMA_STAGED_H_
