// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

namespace {

// Target providers select the reference-clock implementations while linking.
LOOM_TEMPLATE("completed_tick") unsigned completed_tick();
LOOM_TEMPLATE("completed_tick64") unsigned long long completed_tick64();

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
  // Submission ticket and owner slot returned by the completion entry.
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

// Full-run counters and reference-clock bounds.
struct Summary {
  // First native error, or zero after complete success.
  int status;
  // Number of release-published SQEs.
  unsigned submitted;
  // Number of acquired and consumed CQEs, including drain after failure.
  unsigned completed;
  // Low word of the reference-clock sample before the first submission.
  unsigned begin_tick;
  // Low word of the reference-clock sample after the final completion.
  unsigned end_tick;
  // High word paired with begin_tick.
  unsigned begin_tick_high;
  // High word paired with end_tick.
  unsigned end_tick_high;
  // Zero-filled padding aligning the following slot table.
  unsigned reserved;
};

// One reusable payload credit and at most one I/O belong to each slot.
struct Slot {
  // Response-derived cause selecting this consumer's input block.
  unsigned cause;
  // Completed consumer generations in this stream.
  unsigned round;
  // Read zero, write one, or reload two.
  unsigned phase;
  // Successfully transferred bytes within the current operation.
  unsigned progress;
};

// One logical operation record. Positive short retries retain the same row.
struct Record {
  // Stream owning the operation and its payload credit.
  unsigned slot;
  // Consumer generation within that stream.
  unsigned round;
  // Read zero, write one, or reload two.
  unsigned phase;
  // Immutable input block selected by the cause.
  unsigned key;
  // Cause before consuming this operation's returned bytes.
  unsigned cause;
  // Reference-clock sample before the first physical submission.
  unsigned begin_tick;
  // Reference-clock sample after successful payload observation.
  unsigned end_tick;
  // Completed logical byte length; zero until full retirement.
  int result;
  // First payload word; writes retain their zero initialization.
  unsigned first;
  // Cause-selected payload word; writes retain their zero initialization.
  unsigned selected;
  // Last payload word; writes retain their zero initialization.
  unsigned last;
  // Last physical submission ticket, including a positive short retry.
  unsigned ticket;
};

static_assert(sizeof(SubmissionEntry) == 64);
static_assert(sizeof(CompletionEntry) == 16);
static_assert(sizeof(Summary) == 32);
static_assert(sizeof(Slot) == 16);
static_assert(sizeof(Record) == 48);
static_assert(__builtin_offsetof(SubmissionEntry, offset) == 8);
static_assert(__builtin_offsetof(SubmissionEntry, address) == 16);
static_assert(__builtin_offsetof(SubmissionEntry, length) == 24);
static_assert(__builtin_offsetof(SubmissionEntry, user_data) == 32);

using loom::atomic::ordering;
using loom::atomic::scope;
using loom::view::atomic::load;
using loom::view::atomic::store;

}  // namespace

// A finite completion-driven owner measures response-dependent lookups or
// read/write/reload chains. Each stream owns two guard-separated payloads. A
// completion immediately admits its own successor, while failures stop new
// publication and drain every accepted operation before return.
[[loom::kernel, loom::symbol("file_latency"), loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void file_latency(
    [[loom::assume_aligned(64)]] SubmissionEntry* submission_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* submission_tail,
    [[loom::assume_aligned(8)]] const CompletionEntry* completion_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* completion_head,
    [[loom::assume_aligned(4)]] const volatile unsigned* completion_tail,
    [[loom::assume_aligned(4)]] unsigned* payload,
    [[loom::assume_aligned(4)]] Summary* summary,
    [[loom::assume_aligned(4)]] Record* records,
    unsigned long long host_payload, unsigned submission_mask,
    unsigned completion_mask, unsigned initial_position, unsigned depth,
    unsigned round_count, unsigned word_count, unsigned file_block_mask,
    unsigned seed, unsigned payload_stride, unsigned phase_count,
    unsigned gap_ticks, unsigned file_index) {
  loom::assume(word_count >= 1u && word_count <= 1048576u);

  constexpr unsigned kReadFixed = 260;
  constexpr unsigned kWriteFixed = 261;
  constexpr int kIncomplete = -61;

  unsigned char* state_bytes = reinterpret_cast<unsigned char*>(summary);
  Slot* slots = reinterpret_cast<Slot*>(state_bytes + sizeof(Summary));
  unsigned char* payload_bytes = reinterpret_cast<unsigned char*>(payload);
  const unsigned length = word_count * 4u;
  const unsigned word_mask = word_count - 1u;
  const unsigned blocks = file_block_mask + 1u;

  for (unsigned slot_index = 0; slot_index < depth; ++slot_index) {
    Slot* owner = slots + slot_index;
    owner->cause = seed + slot_index;
    owner->round = 0;
    owner->phase = 0;
    owner->progress = 0;
  }

  const unsigned long long begin = completed_tick64();
  summary->begin_tick = static_cast<unsigned>(begin);
  summary->begin_tick_high = static_cast<unsigned>(begin >> 32u);

  unsigned tail = initial_position;
  unsigned head = initial_position;
  unsigned next_slot = 0;
  int status = 0;
  while (tail != head || (next_slot < depth && status == 0)) {
    const bool healthy = status == 0;
    if (next_slot < depth && healthy) {
      Slot* owner = slots + next_slot;
      const unsigned cause = owner->cause;
      const unsigned round = owner->round;
      const unsigned phase = owner->phase;
      const unsigned progress = owner->progress;
      const unsigned key = cause & file_block_mask;
      const bool reading = phase == 0u;
      const bool writing = phase == 1u;
      const bool reloading = phase == 2u;
      const unsigned write_block = blocks + next_slot * 3u;
      const unsigned block = reading ? key : write_block;
      const unsigned reload_slot = next_slot + depth;
      const unsigned window = reloading ? reload_slot : next_slot;
      const unsigned window_offset = window * payload_stride;
      const unsigned long long address =
          host_payload + static_cast<unsigned long long>(window_offset) +
          progress;
      const unsigned long long file_offset =
          static_cast<unsigned long long>(block) * length + progress;
      const unsigned remaining = length - progress;
      const unsigned record_number =
          (next_slot * round_count + round) * phase_count + phase;
      Record* record = records + record_number;
      record->slot = next_slot;
      record->round = round;
      record->phase = phase;
      record->key = key;
      record->cause = cause;
      record->ticket = tail;

      SubmissionEntry* submission =
          submission_entries + (tail & submission_mask);
      submission->operation = writing ? kWriteFixed : kReadFixed;
      submission->file_index = file_index;
      submission->offset = file_offset;
      submission->address = address;
      submission->length = remaining;
      submission->user_data =
          (static_cast<unsigned long long>(tail) << 32u) | next_slot;
      if (progress == 0u) {
        record->begin_tick = completed_tick();
      }

      ++tail;
      store<ordering::release, scope::system>(tail, submission_tail);
    }

    const unsigned issued = tail - initial_position;
    if (issued < depth) {
      next_slot = issued;
    } else {
      while (load<ordering::relaxed, scope::system>(completion_tail) == head) {
      }
      loom::buffer::fence<ordering::acquire, scope::system>();

      const CompletionEntry* completion =
          completion_entries + (head & completion_mask);
      const unsigned slot_index = static_cast<unsigned>(completion->user_data);
      const int result = completion->result;
      ++head;
      store<ordering::release, scope::system>(head, completion_head);

      const bool positive = result > 0;
      if (healthy && !positive) {
        status = result == 0 ? kIncomplete : result;
      }
      if (healthy && positive) {
        Slot* owner = slots + slot_index;
        const unsigned cause = owner->cause;
        const unsigned round = owner->round;
        const unsigned phase = owner->phase;
        const unsigned advanced =
            owner->progress + static_cast<unsigned>(result);
        unsigned after_round = round;
        if (advanced == length) {
          const unsigned key = cause & file_block_mask;
          const unsigned record_number =
              (slot_index * round_count + round) * phase_count + phase;
          Record* record = records + record_number;
          record->result = static_cast<int>(advanced);

          unsigned value = cause;
          if (phase != 1u) {
            const unsigned window =
                phase == 2u ? slot_index + depth : slot_index;
            unsigned* words = reinterpret_cast<unsigned*>(
                payload_bytes + window * payload_stride);
            const unsigned first = words[0];
            const unsigned selected = words[key & word_mask];
            const unsigned last = words[word_mask];
            record->first = first;
            record->selected = selected;
            record->last = last;
            const unsigned based = (first ^ selected ^ last) + cause;
            const unsigned hashed1 = based ^ (based << 13u);
            const unsigned hashed2 = hashed1 ^ (hashed1 >> 17u);
            value = hashed2 ^ (hashed2 << 5u);
          }

          const unsigned end = completed_tick();
          record->end_tick = end;
          const unsigned next_phase = phase + 1u;
          const bool finished = next_phase == phase_count;
          const unsigned final_round = finished ? round + 1u : round;
          owner->round = final_round;
          owner->phase = finished ? 0u : next_phase;
          owner->cause = finished ? value : cause;
          owner->progress = 0;
          if (finished && final_round < round_count && gap_ticks != 0u) {
            while (completed_tick() - end < gap_ticks) {
            }
          }
          after_round = final_round;
        } else {
          owner->progress = advanced;
        }
        next_slot = after_round < round_count ? slot_index : depth;
      } else {
        next_slot = depth;
      }
    }
  }

  summary->status = status;
  summary->submitted = tail - initial_position;
  summary->completed = head - initial_position;
  const unsigned long long end = completed_tick64();
  summary->end_tick = static_cast<unsigned>(end);
  summary->end_tick_high = static_cast<unsigned>(end >> 32u);
}
