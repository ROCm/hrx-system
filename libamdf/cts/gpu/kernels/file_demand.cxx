// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

namespace {

constexpr unsigned kMaximumCredits = 256;
constexpr unsigned kMaximumFileBlocks = 1024;
constexpr unsigned kMaximumDemands = 1024;
constexpr unsigned kSummaryWords = 20;
constexpr unsigned kSlotWords = 16;
constexpr unsigned kTableWords = kSummaryWords + kMaximumCredits * kSlotWords;

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

// Private owner counters copied to global state after complete retirement.
struct Summary {
  // First native error, or zero on complete success.
  int status;
  // Number of physically published SQEs, including short retries.
  unsigned submitted;
  // Number of native CQEs consumed before returning.
  unsigned completed;
  // Reference-clock origin of the offered schedule.
  unsigned begin_tick;
  // Reference-clock time after every accepted ownership path retires.
  unsigned end_tick;
  // Maximum native submissions awaiting GPU consumption.
  unsigned peak_outstanding;
  // Logical demands admitted into an allocated or shared credit.
  unsigned admitted;
  // Logical consumers that released their final payload reference.
  unsigned consumed;
  // Admitted duplicate readers that issued no new physical read.
  unsigned deduplicated;
  // Maximum live payload credits, including retained completed reads.
  unsigned peak_credits;
  // Free-list head, encoded as slot plus one; zero means empty.
  unsigned free_head;
  // Ready-consumer queue head, encoded as slot plus one.
  unsigned ready_head;
  // Ready-consumer queue tail, encoded as slot plus one.
  unsigned ready_tail;
  // Number of currently allocated payload credits.
  unsigned live_credits;
  // Index of the next offered demand.
  unsigned next_demand;
  // Next credit to publish, encoded as slot plus one.
  unsigned issue;
  // Upper word of the sample that produced begin_tick.
  unsigned begin_tick_high;
  // Upper word of the sample that produced end_tick.
  unsigned end_tick_high;
  // First zero word aligning the following slot table.
  unsigned reserved_0;
  // Second zero word aligning the following slot table.
  unsigned reserved_1;
};

// One workgroup-private payload credit and its reader ownership.
struct Slot {
  // Immutable input block selected by the GPU hash.
  unsigned key;
  // First pending consumer, encoded as record index plus one.
  unsigned first_reader;
  // Last pending consumer, encoded as record index plus one.
  unsigned last_reader;
  // Free-list or ready-queue successor, encoded as slot plus one.
  unsigned next;
  // Current native operation: read zero, write one, or reload two.
  unsigned phase;
  // Positive bytes already completed within that operation.
  unsigned progress;
  // First publication timestamp of the current physical operation.
  unsigned submit_tick;
  // Final I/O completion timestamp before consuming logical readers.
  unsigned ready_tick;
  // Payload generation incremented on each new ownership.
  unsigned generation;
  // Logical references that have not released this payload.
  unsigned readers;
  // Last physical submission ticket for the retirement oracle.
  unsigned ticket;
  // Latest full-operation completion timestamp.
  unsigned phase_end_tick;
  // Stable first record retaining physical-operation timestamps.
  unsigned leader;
  // Final reader release of the preceding generation, or zero initially.
  unsigned last_release;
  // First zero word preserving sixteen-byte slot alignment.
  unsigned reserved_0;
  // Second zero word preserving sixteen-byte slot alignment.
  unsigned reserved_1;
};

// One offered logical demand and its exported result.
struct Record {
  // Offered time relative to Summary.begin_tick.
  unsigned arrival_ticks;
  // Input to the GPU key hash; equal values are duplicate demand.
  unsigned key_seed;
  // Deliberate retained-reader interval after native readiness.
  unsigned hold_ticks;
  // Next reader of the same backing, encoded as record index plus one.
  unsigned next;
  // Reference clock when admission acquired or shared a credit.
  unsigned admitted_tick;
  // Reference clock when this consumer first probed returned data.
  unsigned ready_tick;
  // Reference clock after final observation and reference release.
  unsigned end_tick;
  // Physical payload credit used by this consumer.
  unsigned slot;
  // Payload generation observed by this consumer.
  unsigned generation;
  // File key selected by the GPU hash.
  unsigned key;
  // First payload word observed on final consumption.
  unsigned first;
  // Key-selected payload word observed on final consumption.
  unsigned selected;
  // Last payload word observed on final consumption.
  unsigned last;
  // Combined numerical output from the three payload words.
  unsigned result;
  // Whether admission shared an existing key.
  unsigned shared;
  // First read publication time, shared by joined readers.
  unsigned read_begin;
  // First read native completion time.
  unsigned read_end;
  // KV write publication time; zero for immutable lookups.
  unsigned write_begin;
  // KV write native completion time; zero for immutable lookups.
  unsigned write_end;
  // KV reload publication time; zero for immutable lookups.
  unsigned reload_begin;
  // KV reload native completion time; zero for immutable lookups.
  unsigned reload_end;
  // Final physical submission ticket observed by this consumer.
  unsigned ticket;
  // Final release of the preceding credit owner, or zero.
  unsigned previous_release;
  // Whether the first payload probe and ready time were recorded.
  unsigned observed;
};

// Private progress for one reader while its public record stays immutable.
struct ReaderState {
  // Next reader of the same backing, encoded as record index plus one.
  unsigned next;
  // Input retained for the final numerical probe.
  unsigned seed;
  // Required retention interval after the first probe.
  unsigned hold;
  // Admission timestamp.
  unsigned admitted;
  // First successful payload-probe timestamp.
  unsigned ready;
  // Final release timestamp.
  unsigned released;
  // Whether the first probe completed.
  unsigned observed;
  // Whether this reader joined an existing credit.
  unsigned shared;
};

// Cross-dispatch coordination with independent arithmetic work.
struct Background {
  // Independent GPU instance has started and retained its arguments.
  unsigned started;
  // I/O owner established the common clock origin.
  unsigned run;
  // I/O owner completed native and consumer retirement.
  unsigned stop;
  // Completed independent arithmetic iterations.
  unsigned iterations;
  // Independently checkable modulo-32-bit arithmetic result.
  unsigned result;
};

static_assert(sizeof(SubmissionEntry) == 64);
static_assert(sizeof(CompletionEntry) == 16);
static_assert(sizeof(Summary) == 80);
static_assert(sizeof(Slot) == 64);
static_assert(sizeof(Record) == 96);
static_assert(sizeof(ReaderState) == 32);
static_assert(sizeof(Background) == 20);
static_assert(__builtin_offsetof(SubmissionEntry, offset) == 8);
static_assert(__builtin_offsetof(SubmissionEntry, address) == 16);
static_assert(__builtin_offsetof(SubmissionEntry, length) == 24);
static_assert(__builtin_offsetof(SubmissionEntry, user_data) == 32);

using loom::atomic::ordering;
using loom::atomic::scope;
using loom::view::atomic::load;
using loom::view::atomic::store;

LOOM_FORCE_INLINE unsigned demand_hash(unsigned input) {
  unsigned first = input ^ (input << 13u);
  unsigned second = first ^ (first >> 17u);
  return second ^ (second << 5u);
}

// Probes the immutable payload, refreshes the public numerical witness, and
// returns a clock sample taken after those observations and stores.
LOOM_FORCE_INLINE unsigned demand_probe(unsigned* data, Record* record,
                                        unsigned key, unsigned seed,
                                        unsigned word_count) {
  const unsigned word_mask = word_count - 1u;
  const unsigned first = data[0];
  const unsigned selected = data[key & word_mask];
  const unsigned last = data[word_mask];
  record->first = first;
  record->selected = selected;
  record->last = last;
  record->result = demand_hash((first ^ selected ^ last) + seed);
  return completed_tick();
}

}  // namespace

// One finite owner services a sorted offered schedule, native completions, and
// ready readers without a batch join. A second role performs independent GPU
// arithmetic while the owner executes I/O. Private reader bookkeeping remains
// in workgroup memory and is exported only after every ownership path retires.
[[loom::kernel, loom::symbol("file_demand"), loom::workgroup_size(1, 1, 1),
  loom::workgroup_count(1, 1, 1)]]
void file_demand(
    [[loom::assume_aligned(64)]] SubmissionEntry* submission_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* submission_tail,
    [[loom::assume_aligned(8)]] const CompletionEntry* completion_entries,
    [[loom::assume_aligned(4)]] volatile unsigned* completion_head,
    [[loom::assume_aligned(4)]] const volatile unsigned* completion_tail,
    [[loom::assume_aligned(4)]] unsigned* payload,
    [[loom::assume_aligned(4)]] unsigned* state,
    [[loom::assume_aligned(4)]] Record* records,
    [[loom::assume_aligned(4)]] unsigned* keys,
    [[loom::assume_aligned(4)]] Background* background,
    unsigned long long host_payload, unsigned submission_mask,
    unsigned completion_mask, unsigned initial_position, unsigned credit_count,
    unsigned demand_count, unsigned word_count, unsigned file_block_mask,
    unsigned payload_stride, unsigned phase_count, unsigned file_index,
    unsigned role, unsigned background_enabled) {
  loom::assume(credit_count >= 1u && credit_count <= kMaximumCredits);
  loom::assume(demand_count <= kMaximumDemands);
  loom::assume(file_block_mask < kMaximumFileBlocks);
  loom::assume(word_count >= 1u && word_count <= 1048576u);

  constexpr unsigned kReadFixed = 260;
  constexpr unsigned kWriteFixed = 261;
  constexpr int kIncomplete = -61;

  if (role != 0u) {
    store<ordering::release, scope::system>(1u, &background->started);
    while (load<ordering::acquire, scope::system>(&background->run) == 0u) {
    }
    unsigned iterations = 0;
    unsigned value = 19088743u;
    unsigned stop = 0;
    while (stop == 0u) {
      value = demand_hash(value + iterations);
      ++iterations;
      stop = load<ordering::acquire, scope::system>(&background->stop);
    }
    background->iterations = iterations;
    background->result = value;
  } else {
    LOOM_WORKGROUP unsigned table_storage[kTableWords];
    LOOM_WORKGROUP unsigned key_map[kMaximumFileBlocks];
    LOOM_WORKGROUP ReaderState reader_states[kMaximumDemands];

    Summary* summary = reinterpret_cast<Summary*>(table_storage);
    Slot* slots = reinterpret_cast<Slot*>(table_storage + kSummaryWords);
    unsigned char* payload_bytes = reinterpret_cast<unsigned char*>(payload);
    const unsigned block_count = file_block_mask + 1u;
    const unsigned table_word_count = kSummaryWords + credit_count * kSlotWords;
    const unsigned length = word_count * 4u;

    for (unsigned word = 0; word < table_word_count; ++word) {
      table_storage[word] = 0;
    }
    for (unsigned key = 0; key < block_count; ++key) {
      key_map[key] = 0;
    }
    for (unsigned slot_index = 0; slot_index < credit_count; ++slot_index) {
      const unsigned encoded = slot_index + 2u;
      slots[slot_index].next = encoded <= credit_count ? encoded : 0u;
    }
    summary->free_head = 1;

    if (background_enabled != 0u) {
      while (load<ordering::acquire, scope::system>(&background->started) ==
             0u) {
      }
    }
    const unsigned long long begin_wide = completed_tick64();
    const unsigned begin = static_cast<unsigned>(begin_wide);
    summary->begin_tick = begin;
    summary->begin_tick_high = static_cast<unsigned>(begin_wide >> 32u);
    store<ordering::release, scope::system>(1u, &background->run);

    unsigned tail = initial_position;
    unsigned head = initial_position;
    int status = 0;
    while ((tail != head) |
           ((summary->consumed < demand_count) & (status == 0))) {
      const bool healthy = status == 0;

      if (summary->issue == 0u && summary->next_demand < demand_count &&
          healthy) {
        const unsigned record_number = summary->next_demand;
        Record* record = records + record_number;
        const unsigned elapsed = completed_tick() - begin;
        if (elapsed >= record->arrival_ticks) {
          const unsigned seed = record->key_seed;
          const unsigned key = demand_hash(seed) & file_block_mask;
          const bool read_only = phase_count == 1u;
          const unsigned shared = read_only ? key_map[key] : 0u;
          const bool join = shared != 0u;
          const unsigned selected = join ? shared : summary->free_head;
          if (selected != 0u) {
            ReaderState* reader = reader_states + record_number;
            reader->next = 0;
            reader->seed = seed;
            reader->hold = record->hold_ticks;
            reader->ready = 0;
            reader->released = 0;
            reader->observed = 0;
            reader->shared = join ? 1u : 0u;

            const unsigned slot_index = selected - 1u;
            Slot* owner = slots + slot_index;
            const unsigned encoded_record = record_number + 1u;
            if (join) {
              ReaderState* previous = reader_states + (owner->last_reader - 1u);
              previous->next = encoded_record;
              ++owner->readers;
              ++summary->deduplicated;
            } else {
              summary->free_head = owner->next;
              owner->key = key;
              owner->first_reader = encoded_record;
              owner->next = 0;
              owner->phase = 0;
              owner->progress = 0;
              owner->ready_tick = 0;
              owner->readers = 1;
              owner->leader = record_number;
              ++owner->generation;
              ++summary->live_credits;
              if (summary->live_credits > summary->peak_credits) {
                summary->peak_credits = summary->live_credits;
              }
              summary->issue = selected;
              if (read_only) {
                key_map[key] = selected;
              }
            }
            owner->last_reader = encoded_record;
            record->slot = slot_index;
            record->generation = owner->generation;
            record->key = key;
            record->previous_release = owner->last_release;
            reader->admitted = completed_tick();
            summary->admitted = encoded_record;
            summary->next_demand = encoded_record;
          }
        }
      }

      const unsigned selected = summary->issue;
      if (selected != 0u && healthy) {
        const unsigned slot_index = selected - 1u;
        Slot* owner = slots + slot_index;
        const unsigned phase = owner->phase;
        const unsigned progress = owner->progress;
        const bool reading = phase == 0u;
        const bool writing = phase == 1u;
        const bool reloading = phase == 2u;
        const unsigned write_block = block_count + slot_index * 3u;
        const unsigned block = reading ? owner->key : write_block;
        const unsigned window =
            reloading ? slot_index + credit_count : slot_index;
        const unsigned window_offset = window * payload_stride;
        const unsigned long long address =
            host_payload + static_cast<unsigned long long>(window_offset) +
            progress;
        const unsigned long long file_offset =
            static_cast<unsigned long long>(block) * length + progress;
        const unsigned remaining = length - progress;

        SubmissionEntry* submission =
            submission_entries + (tail & submission_mask);
        submission->operation = writing ? kWriteFixed : kReadFixed;
        submission->file_index = file_index;
        submission->offset = file_offset;
        submission->address = address;
        submission->length = remaining;
        submission->user_data =
            (static_cast<unsigned long long>(tail) << 32u) | slot_index;
        owner->ticket = tail;
        if (progress == 0u) {
          Record* leader = records + owner->leader;
          const unsigned tick = completed_tick();
          if (phase == 0u) {
            leader->read_begin = tick;
          } else if (phase == 1u) {
            leader->write_begin = tick;
          } else {
            leader->reload_begin = tick;
          }
          owner->submit_tick = tick;
        }

        ++tail;
        store<ordering::release, scope::system>(tail, submission_tail);
        summary->issue = 0;
        const unsigned outstanding = tail - head;
        if (outstanding > summary->peak_outstanding) {
          summary->peak_outstanding = outstanding;
        }
      }

      const unsigned completion_end =
          load<ordering::relaxed, scope::system>(completion_tail);
      if (completion_end != head) {
        loom::buffer::fence<ordering::acquire, scope::system>();
        const CompletionEntry* completion =
            completion_entries + (head & completion_mask);
        const unsigned slot_index =
            static_cast<unsigned>(completion->user_data);
        const int result = completion->result;
        ++head;
        store<ordering::release, scope::system>(head, completion_head);

        const bool positive = result > 0;
        if (healthy && !positive) {
          status = result == 0 ? kIncomplete : result;
        }
        if (healthy && positive) {
          Slot* owner = slots + slot_index;
          const unsigned advanced =
              owner->progress + static_cast<unsigned>(result);
          const unsigned encoded_slot = slot_index + 1u;
          if (advanced == length) {
            const unsigned tick = completed_tick();
            Record* leader = records + owner->leader;
            if (owner->phase == 0u) {
              leader->read_end = tick;
            } else if (owner->phase == 1u) {
              leader->write_end = tick;
            } else {
              leader->reload_end = tick;
            }
            owner->phase_end_tick = tick;
            owner->progress = 0;
            const unsigned next_phase = owner->phase + 1u;
            if (next_phase == phase_count) {
              owner->ready_tick = tick;
              if (summary->ready_tail == 0u) {
                summary->ready_head = encoded_slot;
              } else {
                slots[summary->ready_tail - 1u].next = encoded_slot;
              }
              owner->next = 0;
              summary->ready_tail = encoded_slot;
            } else {
              owner->phase = next_phase;
              summary->issue = encoded_slot;
            }
          } else {
            owner->progress = advanced;
            summary->issue = encoded_slot;
          }
        }
      }

      if (summary->ready_head != 0u && status == 0) {
        const unsigned encoded_slot = summary->ready_head;
        const unsigned slot_index = encoded_slot - 1u;
        Slot* owner = slots + slot_index;
        summary->ready_head = owner->next;
        if (summary->ready_head == 0u) {
          summary->ready_tail = 0;
        }

        const unsigned encoded_reader = owner->first_reader;
        const unsigned record_number = encoded_reader - 1u;
        Record* record = records + record_number;
        ReaderState* reader = reader_states + record_number;
        const unsigned window =
            phase_count == 3u ? slot_index + credit_count : slot_index;
        unsigned* data = reinterpret_cast<unsigned*>(payload_bytes +
                                                     window * payload_stride);
        if (reader->observed == 0u) {
          reader->ready =
              demand_probe(data, record, owner->key, reader->seed, word_count);
          reader->observed = 1;
          if (phase_count == 1u) {
            record->read_begin = owner->submit_tick;
            record->read_end = owner->phase_end_tick;
          }
          record->ticket = owner->ticket;
        }

        const unsigned age = completed_tick() - reader->ready;
        const bool release = age >= reader->hold;
        const unsigned next_reader = reader->next;
        if (release) {
          static_cast<void>(
              demand_probe(data, record, owner->key, reader->seed, word_count));
          owner->first_reader = next_reader;
          ++summary->consumed;
          --owner->readers;
          reader->released = completed_tick();
          if (owner->readers == 0u) {
            owner->last_release = reader->released;
          }
        } else if (owner->readers > 1u) {
          ReaderState* last = reader_states + (owner->last_reader - 1u);
          last->next = encoded_reader;
          reader->next = 0;
          owner->first_reader = next_reader;
          owner->last_reader = encoded_reader;
        }

        if (owner->readers != 0u) {
          if (summary->ready_tail == 0u) {
            summary->ready_head = encoded_slot;
          } else {
            slots[summary->ready_tail - 1u].next = encoded_slot;
          }
          owner->next = 0;
          summary->ready_tail = encoded_slot;
        } else {
          key_map[owner->key] = 0;
          owner->last_reader = 0;
          owner->next = summary->free_head;
          summary->free_head = encoded_slot;
          --summary->live_credits;
        }
      }
    }

    // Failure relinquishes logical readers only after all native I/O drains.
    if (status != 0) {
      for (unsigned slot_index = 0; slot_index < credit_count; ++slot_index) {
        slots[slot_index].first_reader = 0;
        slots[slot_index].last_reader = 0;
        slots[slot_index].readers = 0;
      }
      summary->ready_head = 0;
      summary->ready_tail = 0;
      summary->live_credits = 0;
      summary->issue = 0;
    }

    summary->status = status;
    summary->submitted = tail - initial_position;
    summary->completed = head - initial_position;
    const unsigned long long end = completed_tick64();
    summary->end_tick = static_cast<unsigned>(end);
    summary->end_tick_high = static_cast<unsigned>(end >> 32u);
    store<ordering::release, scope::system>(1u, &background->stop);

    for (unsigned record_number = 0; record_number < summary->admitted;
         ++record_number) {
      ReaderState* reader = reader_states + record_number;
      Record* record = records + record_number;
      record->next = reader->next;
      record->admitted_tick = reader->admitted;
      record->ready_tick = reader->ready;
      record->end_tick = reader->released;
      record->shared = reader->shared;
      record->observed = reader->observed;
    }
    for (unsigned word = 0; word < table_word_count; ++word) {
      state[word] = table_storage[word];
    }
    for (unsigned key = 0; key < block_count; ++key) {
      keys[key] = key_map[key];
    }
  }
}
