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

// Device counters describe ownership rather than elapsed time.
struct Summary {
  // First negative result, -ENODATA for EOF, or -EOVERFLOW for a full journal.
  int status;
  // Number of release-published SQEs.
  unsigned submitted;
  // Number of acquired and consumed CQEs, including drain after failure.
  unsigned completed;
  // Maximum published-minus-consumed request count.
  unsigned peak_outstanding;
  // First submissions of unique input reads, excluding positive short retries.
  unsigned unique_reads;
  // Requests joined to an existing key without issuing another read.
  unsigned deduplicated;
  // Peer reloads after their first round while the held source stays live.
  unsigned held_peer_reloads;
  // Complete peer round trips; both peers finish before the delayed reader.
  unsigned peer_completed;
  // Immutable SQ tail when the first error was observed; zero on success.
  unsigned failure_tail;
  // CQ position when failure stops useful processing; zero on success.
  unsigned failure_completion;
  // Set when the first duplicate reader finishes and leaves one reference.
  unsigned held_ready;
  // Slot found by the device key lookup for the duplicate request.
  unsigned duplicate_slot;
};

// One source credit and at most one I/O belong to each slot.
struct Slot {
  // Ready/in-flight read 0/1, write 2/3, reload 4/5, held 6, or retired 7.
  unsigned phase;
  // Logical consumer within this stream; the held stream has three consumers.
  unsigned consumer;
  // Cause selecting the source and biasing its consumer's arithmetic.
  unsigned cause;
  // Immutable input file block retained until the final reader releases it.
  unsigned key;
  // Pending source readers, independent of native CQ ownership.
  unsigned references;
  // Completed bytes of the current logical read, write, or reload.
  unsigned progress;
  // Latest submitted ticket, paired with slot identity in native user_data.
  unsigned ticket;
  // Generation of input backing; the duplicate shares generation zero.
  unsigned generation;
};

// One retirement-oracle entry, indexed by submission ticket.
struct Request {
  // Owner slot encoded in native user_data's low word.
  unsigned slot;
  // Ready phase at publication: read 0, write 2, or reload 4.
  unsigned phase;
  // Logical consumer borrowing this operation.
  unsigned consumer;
  // Cause at publication, independent of completion order.
  unsigned cause;
  // File block before adding progress to the byte offset.
  unsigned block;
  // Byte offset into the logical block and payload window.
  unsigned progress;
  // Remaining bytes requested by this SQE.
  unsigned length;
  // Exact native result, including short, zero, and negative completions.
  int result;
  // Zero-based CQ position at consumption, not the submission ticket.
  unsigned completion;
  // Published tail at consumption, used to verify stop-and-drain ordering.
  unsigned submitted_at_completion;
  // Source references before this operation's payload preparation.
  unsigned references;
  // Exact native CQ flags.
  unsigned flags;
  // Number of consumed CQEs before publication, proving dependency order.
  unsigned completion_frontier;
};

static_assert(sizeof(SubmissionEntry) == 64);
static_assert(sizeof(CompletionEntry) == 16);
static_assert(sizeof(Summary) == 48);
static_assert(sizeof(Slot) == 32);
static_assert(sizeof(Request) == 52);
static_assert(__builtin_offsetof(SubmissionEntry, offset) == 8);
static_assert(__builtin_offsetof(SubmissionEntry, address) == 16);
static_assert(__builtin_offsetof(SubmissionEntry, length) == 24);
static_assert(__builtin_offsetof(SubmissionEntry, user_data) == 32);

using loom::atomic::ordering;
using loom::atomic::scope;
using loom::view::atomic::load;
using loom::view::atomic::store;

}  // namespace

// One invocation owns a native SQ/CQ and three reusable I/O credits. A fourth
// demand joins a published input key and keeps that source alive while two
// peers finish their causal round trips. Every failure drains accepted I/O
// before releasing logical readers and returning.
[[loom::kernel, loom::symbol("file_gather"), loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void file_gather(
    [[loom::assume_aligned(64)]] SubmissionEntry* submission_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* submission_tail,
    [[loom::assume_aligned(8)]] const CompletionEntry* completion_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* completion_head,
    [[loom::assume_aligned(4)]] const volatile unsigned* completion_tail,
    [[loom::assume_aligned(4)]] unsigned* payload,
    [[loom::assume_aligned(4)]] Summary* summary,
    [[loom::assume_aligned(4)]] unsigned* records,
    [[loom::assume_aligned(4)]] Request* requests,
    unsigned long long host_payload, unsigned submission_mask,
    unsigned completion_mask, unsigned peer_round_count, unsigned word_count,
    unsigned seed, unsigned payload_stride, unsigned held_slot, unsigned fault,
    unsigned request_capacity) {
  loom::assume(word_count >= 1u && word_count <= 16384u);

  constexpr unsigned kSlotCount = 3;
  constexpr unsigned kReadFixed = 260;
  constexpr unsigned kWriteFixed = 261;
  constexpr unsigned kOutputMask = 127;
  constexpr unsigned kPartialBlock = 144;
  constexpr unsigned kRecordScale = 257;
  constexpr int kIncomplete = -61;
  constexpr int kFullJournal = -75;
  constexpr unsigned kRecordHeaderWords = 8;

  unsigned char* state_bytes = reinterpret_cast<unsigned char*>(summary);
  Slot* slots = reinterpret_cast<Slot*>(state_bytes + sizeof(Summary));
  unsigned char* payload_bytes = reinterpret_cast<unsigned char*>(payload);
  const unsigned length = word_count * 4u;
  const unsigned record_words = word_count + kRecordHeaderWords;
  const unsigned peer_total = peer_round_count * 2u;
  const unsigned duplicate_cause = seed + held_slot * 17u;
  const unsigned duplicate_key = (duplicate_cause & 3u) * 4u + held_slot;
  const unsigned fault_slot = (held_slot + 1u) % kSlotCount;

  summary->status = 0;
  summary->submitted = 0;
  summary->completed = 0;
  summary->peak_outstanding = 0;
  summary->unique_reads = 0;
  summary->deduplicated = 0;
  summary->held_peer_reloads = 0;
  summary->peer_completed = 0;
  summary->failure_tail = 0;
  summary->failure_completion = 0;
  summary->held_ready = 0;
  summary->duplicate_slot = 0;
  for (unsigned slot_index = 0; slot_index < kSlotCount; ++slot_index) {
    Slot* owner = slots + slot_index;
    owner->phase = 0;
    owner->consumer = 0;
    owner->cause = seed + slot_index * 17u;
    owner->key = 0;
    owner->references = 0;
    owner->progress = 0;
    owner->ticket = 0;
    owner->generation = 0;
  }

  unsigned submitted = 0;
  unsigned completed = 0;
  int status = 0;
  unsigned finished = 0;
  while ((status == 0 && finished < kSlotCount) || submitted != completed) {
    const unsigned selected_held =
        summary->deduplicated != 0u ? summary->duplicate_slot : held_slot;
    Slot* held = slots + selected_held;

    for (unsigned slot_index = 0; slot_index < kSlotCount; ++slot_index) {
      Slot* owner = slots + slot_index;
      const bool is_held = slot_index == selected_held;
      const bool healthy = status == 0;
      const bool waiting = owner->phase == 6u;
      const bool peers_done = summary->peer_completed == peer_total;
      if (waiting && peers_done && healthy) {
        owner->phase = 2;
      }

      const unsigned phase = owner->phase;
      const unsigned consumer = owner->consumer;
      const unsigned cause = owner->cause;
      const unsigned progress = owner->progress;
      const bool reading = phase == 0u;
      const bool writing = phase == 2u;
      const bool reloading = phase == 4u;
      const bool ready = reading || writing || reloading;
      const bool read_credit = owner->references == 0u || progress != 0u;
      const bool source_available = !reading || read_credit;
      const bool initial_or_held = consumer == 0u || is_held;
      const bool admitted = initial_or_held || summary->held_ready != 0u;
      const bool issue = ready && source_available && admitted && healthy;
      if (issue) {
        if (submitted < request_capacity) {
          const bool begin = progress == 0u;
          if (reading && begin) {
            const unsigned normal_key = (cause & 3u) * 4u + slot_index;
            const bool short_input =
                fault == 2u && slot_index == fault_slot && consumer == 0u;
            owner->key = short_input ? kPartialBlock : normal_key;
            owner->references = 1;
            ++summary->unique_reads;
          }

          const unsigned references = owner->references;
          const unsigned key = owner->key;
          const unsigned peer_rank =
              slot_index < selected_held ? slot_index : slot_index - 1u;
          const unsigned peer_record =
              peer_rank * peer_round_count + 3u + consumer;
          const unsigned record = is_held ? consumer : peer_record;
          const unsigned output_block =
              ((record * 5u + 1u) & kOutputMask) + 16u;
          const unsigned write_slot = 5u - slot_index;
          const unsigned reload_slot = (slot_index + 1u) % kSlotCount + 6u;
          unsigned* output = reinterpret_cast<unsigned*>(
              payload_bytes + write_slot * payload_stride);
          if (writing && begin) {
            unsigned* input = reinterpret_cast<unsigned*>(
                payload_bytes + slot_index * payload_stride);
            const unsigned bias = cause + record * kRecordScale;
            for (unsigned element = 0; element < word_count; ++element) {
              output[element] = input[element] * 3u + bias;
            }
            owner->references = references - 1u;
          }

          const unsigned other_window = writing ? write_slot : reload_slot;
          const unsigned window = reading ? slot_index : other_window;
          const unsigned long long address =
              host_payload +
              static_cast<unsigned long long>(window * payload_stride) +
              progress;
          const unsigned block = reading ? key : output_block;
          const unsigned long long file_offset =
              static_cast<unsigned long long>(block) * length + progress;
          const unsigned remaining = length - progress;
          const bool bad_operation = (fault == 1u && reading) ||
                                     (fault == 3u && writing) ||
                                     (fault == 4u && reloading);
          const bool bad_file =
              bad_operation && slot_index == fault_slot && consumer == 0u;

          SubmissionEntry* submission =
              submission_entries + (submitted & submission_mask);
          submission->operation = writing ? kWriteFixed : kReadFixed;
          submission->file_index = bad_file ? 1u : 0u;
          submission->offset = file_offset;
          submission->address = address;
          submission->length = remaining;
          submission->user_data =
              (static_cast<unsigned long long>(submitted) << 32u) | slot_index;

          Request* request = requests + submitted;
          request->slot = slot_index;
          request->phase = phase;
          request->consumer = consumer;
          request->cause = cause;
          request->block = block;
          request->progress = progress;
          request->length = remaining;
          request->references = references;
          request->completion_frontier = completed;

          owner->phase = phase + 1u;
          owner->ticket = submitted;
          ++submitted;
          store<ordering::release, scope::system>(submitted, submission_tail);
        } else {
          summary->failure_tail = submitted;
          summary->failure_completion = completed;
          status = kFullJournal;
        }
      }
    }

    const unsigned outstanding = submitted - completed;
    if (outstanding > summary->peak_outstanding) {
      summary->peak_outstanding = outstanding;
    }

    // Join the fourth demand against live ownership after all initial reads
    // have been published and before any completion is reaped.
    if (summary->deduplicated == 0u) {
      for (unsigned slot_index = 0; slot_index < kSlotCount; ++slot_index) {
        Slot* owner = slots + slot_index;
        if (owner->key == duplicate_key) {
          ++owner->references;
          summary->deduplicated = 1;
          summary->duplicate_slot = slot_index;
        }
      }
    }

    if (submitted != completed) {
      while (load<ordering::relaxed, scope::system>(completion_tail) ==
             completed) {
      }
      loom::buffer::fence<ordering::acquire, scope::system>();

      const CompletionEntry* completion =
          completion_entries + (completed & completion_mask);
      const unsigned slot_index = static_cast<unsigned>(completion->user_data);
      const unsigned ticket =
          static_cast<unsigned>(completion->user_data >> 32u);
      const int result = completion->result;
      const unsigned flags = completion->flags;
      Request* request = requests + ticket;
      request->result = result;
      request->completion = completed;
      request->submitted_at_completion = submitted;
      request->flags = flags;

      const unsigned consumed = completed + 1u;
      store<ordering::release, scope::system>(consumed, completion_head);
      const bool positive = result > 0;
      const bool healthy = status == 0;
      const bool first_failure = healthy && result <= 0;
      if (first_failure) {
        status = result == 0 ? kIncomplete : result;
        summary->failure_tail = submitted;
        summary->failure_completion = completed;
      }

      if (healthy && positive) {
        Slot* owner = slots + slot_index;
        const unsigned phase = owner->phase - 1u;
        const unsigned advanced =
            owner->progress + static_cast<unsigned>(result);
        if (advanced == length) {
          owner->progress = 0;
          if (phase == 4u) {
            const unsigned consumer = owner->consumer;
            const unsigned cause = owner->cause;
            const unsigned key = owner->key;
            const unsigned generation = owner->generation;
            const bool is_held = slot_index == selected_held;
            const unsigned peer_rank =
                slot_index < selected_held ? slot_index : slot_index - 1u;
            const unsigned peer_record =
                peer_rank * peer_round_count + 3u + consumer;
            const unsigned record_number = is_held ? consumer : peer_record;
            unsigned* record = records + record_number * record_words;
            const unsigned write_slot = 5u - slot_index;
            const unsigned reload_slot = (slot_index + 1u) % kSlotCount + 6u;
            unsigned* reload = reinterpret_cast<unsigned*>(
                payload_bytes + reload_slot * payload_stride);
            const bool is_peer = slot_index != selected_held;
            const bool held_witness = is_peer && consumer != 0u;
            const unsigned retained = held_witness ? held->references : 0u;

            record[0] = key;
            record[1] = cause;
            record[2] = consumer;
            record[3] = generation;
            record[4] = slot_index;
            record[5] = write_slot;
            record[6] = reload_slot;
            record[7] = retained;
            for (unsigned element = 0; element < word_count; ++element) {
              record[kRecordHeaderWords + element] = reload[element];
            }

            if (is_peer) {
              ++summary->peer_completed;
              if (held_witness) {
                ++summary->held_peer_reloads;
              }
            }
            const bool hold = is_held && consumer == 0u;
            const unsigned next_consumer = consumer + 1u;
            owner->consumer = next_consumer;
            const unsigned limit = is_held ? kSlotCount : peer_round_count;
            const bool last = next_consumer == limit;
            owner->phase = hold ? 6u : (last ? 7u : 0u);
            if (hold) {
              summary->held_ready = 1;
            } else {
              owner->cause = reload[0];
              owner->generation = generation + 1u;
            }
            if (last) {
              ++finished;
            }
          } else {
            owner->phase = phase + 2u;
          }
        } else {
          owner->phase = phase;
          owner->progress = advanced;
        }
      }
      completed = consumed;
    }
  }

  // Accepted I/O is retired before abandoned readers release their backing.
  if (status != 0) {
    for (unsigned slot_index = 0; slot_index < kSlotCount; ++slot_index) {
      slots[slot_index].references = 0;
      slots[slot_index].phase = 7;
    }
  }
  summary->status = status;
  summary->submitted = submitted;
  summary->completed = completed;
}
