// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_KERNELS_FILE_NOTIFICATION_H_
#define AMDF_CTS_GPU_KERNELS_FILE_NOTIFICATION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kernels::file_notification {

// Finite causal reads recycle both the payload and the native ring slots.
inline constexpr uint32_t kRoundCount = 8;
// The first source block; subsequent causes come from the returned payload.
inline constexpr uint32_t kInitialCause = 3;
// Terminal state shares the generation publication, so it cannot be observed
// early through a separate, not-yet-published status field.
inline constexpr uint64_t kTerminal = UINT64_C(1) << 63;

struct Round {
  // Native CQ identity consumed by the shader for this read.
  uint64_t ticket;
  // Source block selected before this read was issued.
  uint32_t block;
  // Sum modulo 2^32 of every returned word.
  uint32_t checksum;
};

struct Control {
  // Shader release publication; kTerminal marks final consumption atomically.
  uint64_t generation;
  // Host release request to end the resident wait before fixture teardown.
  uint32_t stop;
  // Last consumed native CQ result, including short reads and errors.
  int32_t result;
  // Number of completely consumed pages and initialized round records.
  uint32_t completed;
  // First word of the last fully consumed page, or the initial cause.
  uint32_t cause;
  // Number of SQEs published, including a request stopped before admission.
  uint32_t requests;
  // Exact consumer results; unfinished entries retain their initial guards.
  std::array<Round, kRoundCount> rounds;
};

struct alignas(16) Arguments {
  // GPU address of native SQEs, initialized to zero by the caller.
  uint64_t submission_entries;
  // GPU address of the native SQ tail, exclusively published by this shader.
  uint64_t submission_tail;
  // GPU address of native CQEs.
  uint64_t completion_entries;
  // GPU address of the native CQ head, exclusively advanced by this shader.
  uint64_t completion_head;
  // GPU address of the kernel-published CQ tail.
  uint64_t completion_tail;
  // GPU address of the single reusable, registered payload page.
  uint64_t payload;
  // GPU address of the guarded Control record.
  uint64_t control;
  // CPU virtual address of payload within fixed buffer zero.
  uint64_t host_payload;
  // Native event mailbox address supplied by libamdf.
  uint64_t mailbox;
  // Native event mailbox value, independent of application generations.
  uint64_t trigger;
  // Returned SQ entry-count mask.
  uint32_t submission_mask;
  // Returned CQ entry-count mask.
  uint32_t completion_mask;
  // Number of 32-bit words in each source and payload page.
  uint32_t word_count;
  // Mask of the power-of-two source page bank.
  uint32_t file_block_mask;
  // Maximum complete reads, bounded by the Control record capacity.
  uint32_t round_count;
  // Native shader interrupt operand supplied by libamdf.
  uint32_t interrupt_context;
};

inline constexpr std::array<uint32_t, 16> kArgumentByteOffsets = {
    offsetof(Arguments, submission_entries),
    offsetof(Arguments, submission_tail),
    offsetof(Arguments, completion_entries),
    offsetof(Arguments, completion_head),
    offsetof(Arguments, completion_tail),
    offsetof(Arguments, payload),
    offsetof(Arguments, control),
    offsetof(Arguments, host_payload),
    offsetof(Arguments, mailbox),
    offsetof(Arguments, trigger),
    offsetof(Arguments, submission_mask),
    offsetof(Arguments, completion_mask),
    offsetof(Arguments, word_count),
    offsetof(Arguments, file_block_mask),
    offsetof(Arguments, round_count),
    offsetof(Arguments, interrupt_context)};
inline constexpr std::array<uint32_t, 16> kArgumentByteLengths = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4};
inline constexpr std::array<std::string_view, 16> kArgumentValueKinds = {
    "global_buffer", "global_buffer", "global_buffer", "global_buffer",
    "global_buffer", "global_buffer", "global_buffer", "by_value",
    "global_buffer", "by_value",      "by_value",      "by_value",
    "by_value",      "by_value",      "by_value",      "by_value"};

static_assert(sizeof(Round) == 16);
static_assert(offsetof(Control, rounds) == 32);
static_assert(sizeof(Control) == 160);
static_assert(sizeof(Arguments) == 112);

}  // namespace kernels::file_notification

#endif  // AMDF_CTS_GPU_KERNELS_FILE_NOTIFICATION_H_
