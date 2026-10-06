// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

namespace {

// Native Linux io_uring submission entry fields used by this program. The
// untouched fields arrive zero initialized from the host.
struct SubmissionEntry {
  // Opcode, SQE flags, I/O priority, and reserved personality bits.
  unsigned operation;
  // Fixed-file table index.
  unsigned file_index;
  // File byte offset.
  unsigned long long offset;
  // CPU virtual address within the registered fixed buffer.
  unsigned long long address;
  // Remaining byte count.
  unsigned length;
  // Operation-specific flags, retained as zero here.
  unsigned operation_flags;
  // Submission ticket returned by the completion entry.
  unsigned long long user_data;
  // Unused native word ten.
  unsigned reserved_word_10;
  // Unused native word eleven.
  unsigned reserved_word_11;
  // Unused native word twelve.
  unsigned reserved_word_12;
  // Unused native word thirteen.
  unsigned reserved_word_13;
  // Unused native word fourteen.
  unsigned reserved_word_14;
  // Unused native word fifteen.
  unsigned reserved_word_15;
};

// Native Linux io_uring completion entry.
struct CompletionEntry {
  // Submission-owned correlation value.
  unsigned long long user_data;
  // Signed native result or error.
  int result;
  // Native completion flags.
  unsigned flags;
};

static_assert(sizeof(SubmissionEntry) == 64);
static_assert(sizeof(CompletionEntry) == 16);
static_assert(__builtin_offsetof(SubmissionEntry, offset) == 8);
static_assert(__builtin_offsetof(SubmissionEntry, address) == 16);
static_assert(__builtin_offsetof(SubmissionEntry, length) == 24);
static_assert(__builtin_offsetof(SubmissionEntry, user_data) == 32);

using loom::atomic::ordering;
using loom::atomic::scope;
using loom::view::atomic::load;
using loom::view::atomic::store;

}  // namespace

// One invocation owns an initially empty Linux SQ/CQ with one in-flight I/O.
// The host establishes fixed file/buffer tables and SQPOLL progress, but never
// publishes SQEs or consumes CQEs while this program runs.
[[loom::kernel, loom::symbol("file_exchange"), loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void file_exchange(
    [[loom::assume_aligned(64)]] SubmissionEntry* submission_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* submission_tail,
    [[loom::assume_aligned(8)]] const CompletionEntry* completion_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* completion_head,
    [[loom::assume_aligned(4)]] const volatile unsigned* completion_tail,
    [[loom::assume_aligned(4)]] unsigned* payload,
    [[loom::assume_aligned(4)]] unsigned* records,
    unsigned long long host_payload, unsigned submission_mask,
    unsigned completion_mask, unsigned round_count, unsigned word_count,
    unsigned file_block_mask, unsigned seed, unsigned payload_stride,
    unsigned file_index) {
  loom::assume(word_count >= 1u && word_count <= 16384u);

  constexpr unsigned kReadFixed = 260;
  constexpr unsigned kWriteFixed = 261;
  constexpr int kIncomplete = -61;
  constexpr unsigned kHeaderWords = 3;
  constexpr unsigned kSummaryWords = 4;

  const unsigned blocks = file_block_mask + 1u;
  const unsigned record_words = word_count + kHeaderWords;
  const unsigned length = word_count * 4u;
  unsigned* input = payload;
  unsigned char* payload_bytes = reinterpret_cast<unsigned char*>(payload);
  unsigned* output =
      reinterpret_cast<unsigned*>(payload_bytes + payload_stride);
  unsigned* reload =
      reinterpret_cast<unsigned*>(payload_bytes + 2u * payload_stride);

  unsigned round = 0;
  unsigned phase = 0;
  unsigned progress = 0;
  unsigned ticket = 0;
  unsigned cause = seed;
  int status = 0;
  while (round < round_count && status == 0) {
    const unsigned input_block = cause & file_block_mask;
    const unsigned output_ordinal = (input_block * 5u + 1u) & file_block_mask;
    const unsigned output_block = output_ordinal + blocks;
    const bool reading_input = phase == 0u;
    const bool writing = phase == 1u;
    const bool reloading = phase == 2u;
    const unsigned block = reading_input ? input_block : output_block;
    const unsigned long long file_offset =
        static_cast<unsigned long long>(block) * length + progress;
    const unsigned window_offset = phase * payload_stride;
    const unsigned long long io_address =
        host_payload + static_cast<unsigned long long>(window_offset) +
        progress;
    const unsigned remaining = length - progress;

    SubmissionEntry* submission =
        submission_entries + (ticket & submission_mask);
    submission->operation = writing ? kWriteFixed : kReadFixed;
    submission->file_index = file_index;
    submission->offset = file_offset;
    submission->address = io_address;
    submission->length = remaining;
    submission->user_data = ticket;

    const unsigned next_ticket = ticket + 1u;
    store<ordering::release, scope::system>(next_ticket, submission_tail);
    while (load<ordering::relaxed, scope::system>(completion_tail) == ticket) {
    }
    loom::buffer::fence<ordering::acquire, scope::system>();

    const CompletionEntry* completion =
        completion_entries + (ticket & completion_mask);
    const int result = completion->result;
    store<ordering::release, scope::system>(next_ticket, completion_head);

    const bool positive = result > 0;
    status = positive ? 0 : (result == 0 ? kIncomplete : result);
    const unsigned advanced = progress + static_cast<unsigned>(result);
    const bool complete = positive && advanced == length;
    if (complete) {
      if (reading_input) {
        for (unsigned element = 0; element < word_count; ++element) {
          output[element] = input[element] * 3u + cause + round;
        }
      }
      if (reloading) {
        unsigned* record = records + kSummaryWords + round * record_words;
        record[0] = input_block;
        record[1] = cause;
        record[2] = output_block;
        for (unsigned element = 0; element < word_count; ++element) {
          record[kHeaderWords + element] = reload[element];
        }
        cause = reload[0];
        ++round;
        phase = 0;
      } else {
        ++phase;
      }
      progress = 0;
    } else {
      progress = advanced;
    }
    ticket = next_ticket;
  }

  records[0] = round;
  records[1] = static_cast<unsigned>(status);
  records[2] = ticket;
  records[3] = cause;
}
