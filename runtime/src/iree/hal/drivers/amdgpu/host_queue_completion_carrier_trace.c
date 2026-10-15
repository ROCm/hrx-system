// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_completion_carrier_trace.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/time.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

enum {
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY = 32768,
  IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY = 8192,
};

typedef struct iree_hal_amdgpu_completion_carrier_slot_t {
  // Absolute packet ID currently owning this physical slot.
  uint64_t packet_id;
  // Reservation containing |packet_id|.
  uint64_t reservation_id;
  // Absolute packet ID of the slot's preceding owner.
  uint64_t previous_packet_id;
  // Completion signal observed for the preceding owner.
  uint64_t previous_completion_signal;
  // Completion signal observed for the current owner.
  uint64_t completion_signal;
  // Number of packets in the current reservation.
  uint32_t reservation_count;
  // Offset of this packet within the current reservation.
  uint32_t reservation_offset;
} iree_hal_amdgpu_completion_carrier_slot_t;

typedef struct iree_hal_amdgpu_completion_carrier_packet_t {
  // Absolute AQL packet ID.
  uint64_t packet_id;
  // Reservation identity assigned when the packet span was claimed.
  uint64_t reservation_id;
  // Absolute packet ID of the preceding physical-slot owner.
  uint64_t previous_packet_id;
  // Completion signal carried by the preceding physical-slot owner.
  uint64_t previous_completion_signal;
  // Completion signal present immediately before ringing the doorbell.
  uint64_t completion_signal;
  // First notification epoch possibly assigned by this doorbell batch.
  uint64_t first_epoch;
  // Last notification epoch assigned by this doorbell batch.
  uint64_t last_epoch;
  // Host publication epoch immediately before the doorbell.
  uint64_t published_epoch;
  // Monotonic doorbell sequence containing this packet.
  uint64_t doorbell_sequence;
  // Number of packets in the owning reservation.
  uint32_t reservation_count;
  // Offset of this packet within the owning reservation.
  uint32_t reservation_offset;
  // Physical AQL ring slot used by this packet.
  uint32_t slot;
  // Number of epochs assigned by this doorbell batch.
  uint32_t batch_epoch_count;
  // Number of queue-epoch signal bodies found in this doorbell batch.
  uint32_t batch_carrier_count;
  // PM4 sidecar record index, or UINT32_MAX when not applicable.
  uint32_t pm4_record_index;
  // Header observed immediately before ringing the doorbell.
  uint16_t header;
  // Setup or vendor format observed with |header|.
  uint16_t setup;
  // True when reservation tracking owned the physical slot.
  bool has_reservation;
  // Complete packet bytes sampled immediately after ringing the doorbell.
  uint8_t raw[64];
} iree_hal_amdgpu_completion_carrier_packet_t;

typedef struct iree_hal_amdgpu_completion_carrier_pm4_t {
  // Complete queue-private PM4 IB slot bytes.
  uint32_t dwords[IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY];
} iree_hal_amdgpu_completion_carrier_pm4_t;

typedef struct iree_hal_amdgpu_completion_carrier_trace_t {
  // Queue whose reservation and doorbell boundaries are observed.
  iree_hal_amdgpu_host_queue_t* queue;
  // Allocator owning all trace storage.
  iree_allocator_t allocator;
  // Pre-opened trace output, never touched from an observer callback.
  FILE* output_file;
  // Per-physical-slot reservation and reuse identity.
  iree_hal_amdgpu_completion_carrier_slot_t* slots;
  // Bounded chronological packet ledger.
  iree_hal_amdgpu_completion_carrier_packet_t* packets;
  // Number of populated packet records.
  uint32_t packet_count;
  // Bounded PM4 snapshots referenced by packet records.
  iree_hal_amdgpu_completion_carrier_pm4_t* pm4_records;
  // Number of populated PM4 records.
  uint32_t pm4_record_count;
  // Monotonic reservation identity.
  uint64_t reservation_count;
  // Total number of packets claimed by observed reservations.
  uint64_t reserved_packet_count;
  // Total number of packet IDs covered by observed doorbells.
  uint64_t doorbelled_packet_count;
  // Total notification epochs assigned before observed doorbells.
  uint64_t assigned_epoch_count;
  // Total queue-epoch signal bodies observed before doorbells.
  uint64_t carrier_count;
  // Number of doorbell batches whose carrier and epoch counts differ.
  uint64_t carrier_mismatch_count;
  // First packet ID expected in the next reservation.
  uint64_t next_reservation_packet_id;
  // First packet ID expected in the next doorbell batch.
  uint64_t next_doorbell_packet_id;
  // Notification epoch observed at the preceding doorbell.
  uint64_t last_doorbell_epoch;
  // Number of doorbell operations observed.
  uint64_t doorbell_count;
  // First packet record populated by the active doorbell callback pair.
  uint32_t active_record_begin;
  // Number of packet records populated by the active callback pair.
  uint32_t active_record_count;
  // Number of packets in the active doorbell batch.
  uint64_t active_packet_count;
  // True after the first reservation establishes the expected packet ID.
  bool has_reservation_position;
  // True after the first reservation establishes the doorbell position.
  bool has_doorbell_position;
  // True when reservation IDs were not contiguous.
  bool has_reservation_gap;
  // True when a doorbelled slot lacked the expected reservation identity.
  bool has_unowned_packet;
  // True when the chronological packet ledger filled.
  bool packet_overflow;
  // True when the PM4 sidecar ledger filled.
  bool pm4_overflow;
  // True after the trace has been exported.
  bool has_exported;
} iree_hal_amdgpu_completion_carrier_trace_t;

static bool iree_hal_amdgpu_completion_carrier_trace_is_enabled(void) {
  const char* override = getenv("IREE_AMDGPU_COMPLETION_CARRIER_TRACE");
  if (override && override[0] != '\0' && strcmp(override, "0") != 0) {
    return true;
  }
  const char* test_target = getenv("TEST_TARGET");
  return test_target &&
         strstr(test_target,
                "indirection_access_test_execute_amdgpu_access_test") != NULL;
}

static uint64_t iree_hal_amdgpu_completion_carrier_load_signal(
    const iree_hal_amdgpu_aql_packet_t* packet) {
  uint64_t signal = 0;
  memcpy(&signal, &packet->raw[sizeof(packet->raw) - sizeof(signal)],
         sizeof(signal));
  return signal;
}

static void iree_hal_amdgpu_completion_carrier_trace_reserve(
    void* user_data, uint64_t first_packet_id, uint32_t count) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  const uint64_t reservation_id = ++trace->reservation_count;
  if (!trace->has_reservation_position) {
    trace->next_reservation_packet_id = first_packet_id;
    trace->next_doorbell_packet_id = first_packet_id;
    trace->has_reservation_position = true;
    trace->has_doorbell_position = true;
  }
  if (first_packet_id != trace->next_reservation_packet_id) {
    trace->has_reservation_gap = true;
  }
  trace->next_reservation_packet_id = first_packet_id + count;
  trace->reserved_packet_count += count;

  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t packet_id = first_packet_id + i;
    iree_hal_amdgpu_completion_carrier_slot_t* slot =
        &trace->slots[packet_id & trace->queue->aql_ring.mask];
    slot->previous_packet_id = slot->packet_id;
    slot->previous_completion_signal = slot->completion_signal;
    slot->packet_id = packet_id;
    slot->reservation_id = reservation_id;
    slot->completion_signal = 0;
    slot->reservation_count = count;
    slot->reservation_offset = i;
  }
}

static void iree_hal_amdgpu_completion_carrier_trace_before_doorbell(
    void* user_data, uint64_t final_packet_id) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  ++trace->doorbell_count;
  trace->active_record_begin = trace->packet_count;
  trace->active_record_count = 0;
  trace->active_packet_count = 0;

  const uint64_t current_epoch =
      trace->queue->notification_ring.epoch.next_submission;
  const uint64_t published_epoch = (uint64_t)iree_atomic_load(
      &trace->queue->notification_ring.epoch.last_published,
      iree_memory_order_acquire);
  const uint64_t batch_epoch_count = current_epoch - trace->last_doorbell_epoch;
  trace->assigned_epoch_count += batch_epoch_count;

  if (!trace->has_doorbell_position ||
      final_packet_id < trace->next_doorbell_packet_id) {
    if (batch_epoch_count != 0) {
      ++trace->carrier_mismatch_count;
    }
    trace->last_doorbell_epoch = current_epoch;
    return;
  }

  const uint64_t first_packet_id = trace->next_doorbell_packet_id;
  const uint64_t packet_count = final_packet_id - first_packet_id + 1;
  trace->active_packet_count = packet_count;
  trace->doorbelled_packet_count += packet_count;

  uint32_t batch_carrier_count = 0;
  const uint64_t epoch_signal =
      trace->queue->notification_ring.epoch.signal.handle;
  for (uint64_t i = 0; i < packet_count; ++i) {
    const uint64_t packet_id = first_packet_id + i;
    const uint32_t slot_index =
        (uint32_t)(packet_id & trace->queue->aql_ring.mask);
    iree_hal_amdgpu_completion_carrier_slot_t* slot = &trace->slots[slot_index];
    iree_hal_amdgpu_aql_packet_t* packet =
        &trace->queue->aql_ring.base[slot_index];
    uint32_t header_setup = 0;
    memcpy(&header_setup, packet->raw, sizeof(header_setup));
    const uint64_t completion_signal =
        iree_hal_amdgpu_completion_carrier_load_signal(packet);
    if (completion_signal == epoch_signal) {
      ++batch_carrier_count;
    }

    const bool has_reservation = slot->packet_id == packet_id;
    if (!has_reservation) {
      trace->has_unowned_packet = true;
    }
    slot->completion_signal = completion_signal;

    if (trace->packet_count >=
        IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY) {
      trace->packet_overflow = true;
      continue;
    }
    iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[trace->packet_count++];
    memset(record, 0, sizeof(*record));
    record->packet_id = packet_id;
    record->reservation_id = has_reservation ? slot->reservation_id : 0;
    record->previous_packet_id =
        has_reservation ? slot->previous_packet_id : UINT64_MAX;
    record->previous_completion_signal =
        has_reservation ? slot->previous_completion_signal : 0;
    record->completion_signal = completion_signal;
    record->first_epoch =
        batch_epoch_count != 0 ? trace->last_doorbell_epoch + 1 : 0;
    record->last_epoch = batch_epoch_count != 0 ? current_epoch : 0;
    record->published_epoch = published_epoch;
    record->doorbell_sequence = trace->doorbell_count;
    record->reservation_count = has_reservation ? slot->reservation_count : 0;
    record->reservation_offset = has_reservation ? slot->reservation_offset : 0;
    record->slot = slot_index;
    record->batch_epoch_count = batch_epoch_count <= UINT32_MAX
                                    ? (uint32_t)batch_epoch_count
                                    : UINT32_MAX;
    record->pm4_record_index = UINT32_MAX;
    record->header = (uint16_t)header_setup;
    record->setup = (uint16_t)(header_setup >> 16);
    record->has_reservation = has_reservation;
    ++trace->active_record_count;
  }

  trace->carrier_count += batch_carrier_count;
  if (batch_carrier_count != batch_epoch_count) {
    ++trace->carrier_mismatch_count;
  }
  for (uint32_t i = 0; i < trace->active_record_count; ++i) {
    trace->packets[trace->active_record_begin + i].batch_carrier_count =
        batch_carrier_count;
  }
  trace->last_doorbell_epoch = current_epoch;
}

static void iree_hal_amdgpu_completion_carrier_trace_after_doorbell(
    void* user_data, uint64_t final_packet_id) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)user_data;
  for (uint32_t i = 0; i < trace->active_record_count; ++i) {
    const uint32_t record_index = trace->active_record_begin + i;
    iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[record_index];
    iree_hal_amdgpu_aql_packet_t* packet =
        &trace->queue->aql_ring.base[record->slot];
    memcpy(record->raw, packet->raw, sizeof(record->raw));

    const uint8_t packet_type =
        (uint8_t)(record->header >> IREE_HSA_PACKET_HEADER_TYPE);
    const uint8_t vendor_format = (uint8_t)record->setup;
    if (packet_type != IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC ||
        vendor_format != IREE_HSA_AMD_AQL_FORMAT_PM4_IB ||
        !trace->queue->pm4_ib_slots) {
      continue;
    }
    if (trace->pm4_record_count >=
        IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY) {
      trace->pm4_overflow = true;
      continue;
    }
    const uint32_t pm4_record_index = trace->pm4_record_count++;
    iree_hal_amdgpu_completion_carrier_pm4_t* pm4_record =
        &trace->pm4_records[pm4_record_index];
    memcpy(pm4_record->dwords, trace->queue->pm4_ib_slots[record->slot].dwords,
           sizeof(pm4_record->dwords));
    record->pm4_record_index = pm4_record_index;
  }
  if (trace->active_packet_count != 0) {
    trace->next_doorbell_packet_id = final_packet_id + 1;
  }
  trace->active_record_count = 0;
  trace->active_packet_count = 0;
}

static void iree_hal_amdgpu_completion_carrier_trace_write_hex(
    FILE* file, const void* data, iree_host_size_t data_length) {
  static const char kHex[] = "0123456789abcdef";
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < data_length; ++i) {
    fputc(kHex[bytes[i] >> 4], file);
    fputc(kHex[bytes[i] & 0x0F], file);
  }
}

static void iree_hal_amdgpu_completion_carrier_trace_export(
    iree_hal_amdgpu_completion_carrier_trace_t* trace, const char* reason,
    uint64_t completed_epoch, uint64_t published_epoch,
    uint64_t submitted_epoch, uint64_t drained_epoch) {
  if (trace->has_exported) {
    return;
  }
  trace->has_exported = true;

  FILE* file = trace->output_file;
  fprintf(file,
          "{\"kind\":\"summary\",\"reason\":\"%s\","
          "\"device\":%" PRIhsz
          ",\"queue\":%u,"
          "\"ring_capacity\":%u,\"epoch_signal\":\"0x%016" PRIx64
          "\",\"completed\":%" PRIu64 ",\"submitted\":%" PRIu64
          ",\"published\":%" PRIu64 ",\"drained\":%" PRIu64
          ",\"reservations\":%" PRIu64 ",\"reserved_packets\":%" PRIu64
          ",\"doorbells\":%" PRIu64 ",\"doorbelled_packets\":%" PRIu64
          ",\"assigned_epochs\":%" PRIu64 ",\"carriers\":%" PRIu64
          ",\"carrier_mismatches\":%" PRIu64
          ",\"reservation_gap\":%s,\"unowned_packet\":%s,"
          "\"packet_overflow\":%s,\"pm4_overflow\":%s}\n",
          reason, trace->queue->device_ordinal,
          (uint32_t)trace->queue->physical_queue_ordinal,
          trace->queue->aql_ring.mask + 1,
          trace->queue->notification_ring.epoch.signal.handle, completed_epoch,
          submitted_epoch, published_epoch, drained_epoch,
          trace->reservation_count, trace->reserved_packet_count,
          trace->doorbell_count, trace->doorbelled_packet_count,
          trace->assigned_epoch_count, trace->carrier_count,
          trace->carrier_mismatch_count,
          trace->has_reservation_gap ? "true" : "false",
          trace->has_unowned_packet ? "true" : "false",
          trace->packet_overflow ? "true" : "false",
          trace->pm4_overflow ? "true" : "false");

  for (uint32_t i = 0; i < trace->packet_count; ++i) {
    const iree_hal_amdgpu_completion_carrier_packet_t* record =
        &trace->packets[i];
    fprintf(file,
            "{\"kind\":\"packet\",\"packet_id\":%" PRIu64
            ",\"slot\":%u,\"reservation_id\":%" PRIu64
            ",\"reservation_offset\":%u,\"reservation_count\":%u,"
            "\"previous_packet_id\":",
            record->packet_id, record->slot, record->reservation_id,
            record->reservation_offset, record->reservation_count);
    if (record->previous_packet_id == UINT64_MAX) {
      fputs("null", file);
    } else {
      fprintf(file, "%" PRIu64, record->previous_packet_id);
    }
    fprintf(file,
            ",\"previous_completion\":\"0x%016" PRIx64
            "\",\"completion\":\"0x%016" PRIx64
            "\",\"header\":\"0x%04x\",\"setup\":\"0x%04x\","
            "\"first_epoch\":%" PRIu64 ",\"last_epoch\":%" PRIu64
            ",\"published_epoch\":%" PRIu64 ",\"doorbell\":%" PRIu64
            ",\"batch_epochs\":%u,"
            "\"batch_carriers\":%u,\"has_reservation\":%s,\"raw\":\"",
            record->previous_completion_signal, record->completion_signal,
            record->header, record->setup, record->first_epoch,
            record->last_epoch, record->published_epoch,
            record->doorbell_sequence, record->batch_epoch_count,
            record->batch_carrier_count,
            record->has_reservation ? "true" : "false");
    iree_hal_amdgpu_completion_carrier_trace_write_hex(file, record->raw,
                                                       sizeof(record->raw));
    fputs("\"", file);
    if (record->pm4_record_index != UINT32_MAX) {
      fputs(",\"pm4\":\"", file);
      const iree_hal_amdgpu_completion_carrier_pm4_t* pm4_record =
          &trace->pm4_records[record->pm4_record_index];
      iree_hal_amdgpu_completion_carrier_trace_write_hex(
          file, pm4_record->dwords, sizeof(pm4_record->dwords));
      fputs("\"", file);
    }
    fputs("}\n", file);
  }

  if (fflush(file) != 0 || ferror(file)) {
    fprintf(stderr, "AMDGPU completion-carrier trace export failed: %s\n",
            strerror(errno));
  } else {
    fprintf(stderr,
            "AMDGPU completion-carrier trace: %s, packets=%u, epochs=%" PRIu64
            ", carriers=%" PRIu64 ", mismatches=%" PRIu64 ", overflow=%s\n",
            reason, trace->packet_count, trace->assigned_epoch_count,
            trace->carrier_count, trace->carrier_mismatch_count,
            (trace->packet_overflow || trace->pm4_overflow) ? "yes" : "no");
  }
}

iree_status_t iree_hal_amdgpu_host_queue_completion_carrier_trace_initialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!iree_hal_amdgpu_completion_carrier_trace_is_enabled()) {
    return iree_ok_status();
  }
  const char* output_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (!output_dir || output_dir[0] == '\0') {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU completion-carrier trace requires "
                            "TEST_UNDECLARED_OUTPUTS_DIR");
  }

  iree_hal_amdgpu_completion_carrier_trace_t* trace = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             sizeof(*trace), (void**)&trace));
  memset(trace, 0, sizeof(*trace));
  trace->queue = queue;
  trace->allocator = queue->host_allocator;

  iree_status_t status =
      iree_allocator_malloc_array(trace->allocator, queue->aql_ring.mask + 1,
                                  sizeof(*trace->slots), (void**)&trace->slots);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        trace->allocator, IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY,
        sizeof(*trace->packets), (void**)&trace->packets);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        trace->allocator, IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY,
        sizeof(*trace->pm4_records), (void**)&trace->pm4_records);
  }
  if (iree_status_is_ok(status)) {
    memset(trace->slots, 0, (queue->aql_ring.mask + 1) * sizeof(*trace->slots));
    for (uint32_t i = 0; i <= queue->aql_ring.mask; ++i) {
      trace->slots[i].packet_id = UINT64_MAX;
      trace->slots[i].previous_packet_id = UINT64_MAX;
    }
    memset(trace->packets, 0,
           IREE_HAL_AMDGPU_COMPLETION_CARRIER_PACKET_CAPACITY *
               sizeof(*trace->packets));
    memset(trace->pm4_records, 0,
           IREE_HAL_AMDGPU_COMPLETION_CARRIER_PM4_CAPACITY *
               sizeof(*trace->pm4_records));

    char output_path[IREE_MAX_PATH + 1];
    const int output_path_length = snprintf(
        output_path, sizeof(output_path),
        "%s/amdgpu-completion-carriers-%" PRId64 "-d%" PRIhsz "q%u.jsonl",
        output_dir, iree_time_now(), queue->device_ordinal,
        (uint32_t)queue->physical_queue_ordinal);
    if (output_path_length < 0 ||
        (iree_host_size_t)output_path_length >= sizeof(output_path)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "completion-carrier trace path is too long");
    } else {
      trace->output_file = fopen(output_path, "wb");
      if (!trace->output_file) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "failed to open completion-carrier trace '%s': %s",
                             output_path, strerror(errno));
      }
    }
  }

  if (iree_status_is_ok(status)) {
    queue->aql_ring.observer = (iree_hal_amdgpu_aql_ring_observer_t){
        .user_data = trace,
        .reserve = iree_hal_amdgpu_completion_carrier_trace_reserve,
        .before_doorbell =
            iree_hal_amdgpu_completion_carrier_trace_before_doorbell,
        .after_doorbell =
            iree_hal_amdgpu_completion_carrier_trace_after_doorbell,
    };
    return iree_ok_status();
  }

  if (trace->output_file) {
    fclose(trace->output_file);
  }
  iree_allocator_free(trace->allocator, trace->pm4_records);
  iree_allocator_free(trace->allocator, trace->packets);
  iree_allocator_free(trace->allocator, trace->slots);
  iree_allocator_free(trace->allocator, trace);
  return status;
}

void iree_hal_amdgpu_host_queue_completion_carrier_trace_record_overrun(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t completed_epoch,
    uint64_t published_epoch, uint64_t submitted_epoch,
    uint64_t drained_epoch) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)
          queue->aql_ring.observer.user_data;
  if (!trace) {
    return;
  }
  iree_hal_amdgpu_completion_carrier_trace_export(
      trace, "overrun", completed_epoch, published_epoch, submitted_epoch,
      drained_epoch);
}

void iree_hal_amdgpu_host_queue_completion_carrier_trace_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_hal_amdgpu_completion_carrier_trace_t* trace =
      (iree_hal_amdgpu_completion_carrier_trace_t*)
          queue->aql_ring.observer.user_data;
  if (!trace) {
    return;
  }
  queue->aql_ring.observer = (iree_hal_amdgpu_aql_ring_observer_t){0};

  if (!trace->has_exported) {
    const iree_hal_amdgpu_notification_epoch_snapshot_t snapshot =
        iree_hal_amdgpu_notification_ring_query_epoch_snapshot(
            &queue->notification_ring);
    const uint64_t drained_epoch =
        (uint64_t)iree_atomic_load(&queue->notification_ring.epoch.last_drained,
                                   iree_memory_order_acquire);
    iree_hal_amdgpu_completion_carrier_trace_export(
        trace, "clean", snapshot.completed_epoch, snapshot.published_epoch,
        queue->notification_ring.epoch.next_submission, drained_epoch);
  }

  fclose(trace->output_file);
  iree_allocator_free(trace->allocator, trace->pm4_records);
  iree_allocator_free(trace->allocator, trace->packets);
  iree_allocator_free(trace->allocator, trace->slots);
  iree_allocator_free(trace->allocator, trace);
}
