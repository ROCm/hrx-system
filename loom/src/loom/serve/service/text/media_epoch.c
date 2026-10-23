// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_epoch.h"

#include <string.h>

#include "loom/serve/service/text/model.h"

enum {
  MEDIA_EPOCH_RECORD_BYTES =
      LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT * sizeof(uint32_t),
};

typedef struct loom_serve_text_media_epoch_layout_t {
  // Projected rows intersecting the scheduled text partition.
  iree_host_size_t feature_count;
  // Image intersections requiring metadata and producer readiness.
  iree_host_size_t image_count;
  // Source-private bytes copied once per image intersection.
  iree_host_size_t metadata_length;
  // Exact coalesced device-copy descriptor count.
  iree_host_size_t copy_count;
  // Checked total target extent, including packet records and row alignment.
  uint32_t device_length;
  // Previous physical range, extended while adjacent rows can share a copy.
  iree_hal_buffer_ref_t previous;
  // Destination byte offset of the previous coalesced range.
  uint32_t previous_offset;
} loom_serve_text_media_epoch_layout_t;

// Arbitrary chunks can begin after a retained prefix or in the middle of an
// image. Sorted source facts locate that window without rescanning its prefix.
static iree_host_size_t loom_serve_text_media_epoch_first(
    const loom_serve_text_span_t* span) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = span->media->span_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_serve_text_media_input_span_t* image =
        &span->media->spans[middle];
    if ((iree_host_size_t)image->input_offset + image->input_count <=
        span->input_offset) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static bool loom_serve_text_media_epoch_layout(
    iree_host_size_t count, const loom_serve_text_span_t* spans,
    loom_serve_text_media_epoch_layout_t* layout) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    if (!span->media) {
      continue;
    }
    const iree_host_size_t end = span->input_offset + span->token_count;
    for (iree_host_size_t j = loom_serve_text_media_epoch_first(span);
         j < span->media->span_count &&
         span->media->spans[j].input_offset < end;
         ++j) {
      const loom_serve_text_media_input_span_t* image = &span->media->spans[j];
      const iree_host_size_t first =
          iree_max(span->input_offset, image->input_offset);
      const iree_host_size_t last = iree_min(
          end, (iree_host_size_t)image->input_offset + image->input_count);
      layout->feature_count += last - first;
      ++layout->image_count;
      if (!iree_host_size_checked_add(layout->metadata_length,
                                      image->metadata.data_length,
                                      &layout->metadata_length)) {
        return false;
      }
      for (iree_host_size_t input = first; input < last; ++input) {
        const iree_hal_buffer_ref_t row = loom_serve_finite_result_row(
            image->result, image->request_index, input - image->input_offset);
        const uint64_t offset =
            iree_align_uint64(layout->device_length,
                              LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT);
        if (offset > UINT32_MAX || row.length > UINT32_MAX - offset) {
          return false;
        }
        if (layout->previous.buffer == row.buffer &&
            layout->previous.offset + layout->previous.length == row.offset &&
            layout->previous_offset + layout->previous.length == offset) {
          layout->previous.length += row.length;
        } else {
          ++layout->copy_count;
          layout->previous = row;
          layout->previous_offset = (uint32_t)offset;
        }
        layout->device_length = (uint32_t)(offset + row.length);
      }
    }
  }
  return true;
}

// Source and destination adjacency permit one copy for a dense image chunk.
// Ragged/reordered result plans preserve their exact individual byte ranges.
static void loom_serve_text_media_epoch_copy(
    loom_serve_text_media_epoch_t* epoch, iree_hal_buffer_ref_t row,
    uint32_t offset) {
  if (epoch->transfer.operation_count) {
    iree_hal_transfer_operation_t* previous =
        &epoch->transfer.operations[epoch->transfer.operation_count - 1];
    if (previous->copy.source_buffer == row.buffer &&
        previous->copy.source_offset + previous->copy.length == row.offset &&
        previous->copy.target_offset + previous->copy.length == offset) {
      previous->copy.length += row.length;
      return;
    }
  }
  epoch->transfer.operations[epoch->transfer.operation_count++] =
      (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy = {.source_buffer = row.buffer,
                   .source_offset = row.offset,
                   .target_offset = offset,
                   .length = row.length},
      };
}

static void loom_serve_text_media_epoch_store(
    iree_host_size_t count, const loom_serve_text_span_t* spans,
    uint32_t plan_begin, uint8_t* bytes, iree_host_size_t* record_index,
    iree_host_size_t* metadata_offset, loom_serve_text_media_epoch_t* epoch) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    if (!span->media) {
      continue;
    }
    const iree_host_size_t end = span->input_offset + span->token_count;
    for (iree_host_size_t j = loom_serve_text_media_epoch_first(span);
         j < span->media->span_count &&
         span->media->spans[j].input_offset < end;
         ++j) {
      const loom_serve_text_media_input_span_t* image = &span->media->spans[j];
      const iree_host_size_t first =
          iree_max(span->input_offset, image->input_offset);
      const iree_host_size_t last = iree_min(
          end, (iree_host_size_t)image->input_offset + image->input_count);
      epoch->transfer.results[epoch->transfer.result_count++] = image->result;
      if (image->metadata.data_length) {
        memcpy(bytes + *metadata_offset, image->metadata.data,
               image->metadata.data_length);
      }
      for (iree_host_size_t input = first; input < last; ++input) {
        const uint32_t feature = (uint32_t)(input - image->input_offset);
        const iree_hal_buffer_ref_t row = loom_serve_finite_result_row(
            image->result, image->request_index, feature);
        const uint32_t offset = (uint32_t)iree_align_uint64(
            epoch->device_length,
            LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT);
        const uint32_t fields[LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT] = {
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX] = plan_begin + i,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET] =
                (uint32_t)(input - span->input_offset),
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET] =
                (uint32_t)offset,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH] =
                (uint32_t)row.length,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET] =
                (uint32_t)*metadata_offset,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH] =
                (uint32_t)image->metadata.data_length,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_INDEX] = feature,
        };
        uint8_t* record = bytes + (*record_index)++ * MEDIA_EPOCH_RECORD_BYTES;
        for (iree_host_size_t field = 0; field < IREE_ARRAYSIZE(fields);
             ++field) {
          iree_unaligned_store_le_u32(record + field * sizeof(uint32_t),
                                      fields[field]);
        }
        loom_serve_text_media_epoch_copy(epoch, row, (uint32_t)offset);
        epoch->device_length = (uint32_t)(offset + row.length);
      }
      *metadata_offset += image->metadata.data_length;
    }
  }
}

iree_status_t loom_serve_text_media_epoch_initialize(
    iree_host_size_t first_span_count,
    const loom_serve_text_span_t* first_spans,
    iree_host_size_t first_packet_capacity, iree_host_size_t next_span_count,
    const loom_serve_text_span_t* next_spans,
    iree_host_size_t next_packet_capacity, iree_allocator_t host_allocator,
    loom_serve_text_media_epoch_t* out_epoch) {
  memset(out_epoch, 0, sizeof(*out_epoch));
  const iree_host_size_t packet_length =
      (first_packet_capacity + next_packet_capacity) *
      LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES;
  loom_serve_text_media_epoch_layout_t layout = {.device_length =
                                                     (uint32_t)packet_length};
  if (!loom_serve_text_media_epoch_layout(first_span_count, first_spans,
                                          &layout)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input layout exceeds 32-bit extent");
  }
  const uint32_t first_feature_count = (uint32_t)layout.feature_count;
  if (!loom_serve_text_media_epoch_layout(next_span_count, next_spans,
                                          &layout)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input layout exceeds 32-bit extent");
  }
  const iree_host_size_t feature_count = layout.feature_count;
  if (!feature_count) {
    return iree_ok_status();
  }
  const iree_host_size_t record_length =
      feature_count * MEDIA_EPOCH_RECORD_BYTES;
  iree_host_size_t plan_length = 0;
  if (!iree_host_size_checked_add(record_length, layout.metadata_length,
                                  &plan_length) ||
      plan_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input metadata exceeds 32-bit extent");
  }
  const iree_host_size_t operation_bytes =
      layout.copy_count * sizeof(iree_hal_transfer_operation_t);
  const iree_host_size_t result_bytes =
      layout.image_count * sizeof(loom_serve_finite_result_t*);
  loom_serve_text_media_epoch_t epoch = {
      .host_allocator = host_allocator,
      .first_feature_count = first_feature_count,
      .next_feature_count = (uint32_t)feature_count - first_feature_count,
      .first_packet_capacity = (uint32_t)first_packet_capacity,
      .next_packet_capacity = (uint32_t)next_packet_capacity,
      .device_length = (uint32_t)packet_length,
  };
  iree_status_t status = iree_vm_buffer_create(
      plan_length, iree_alignof(uint32_t), host_allocator, &epoch.plan);
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_create(packet_length, iree_alignof(uint32_t),
                                   host_allocator, &epoch.packet_records);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(epoch.packet_records, 0, packet_length,
                                      &epoch.packet_record_bytes);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, operation_bytes + result_bytes,
                              (void**)&epoch.transfer.operations);
  }
  iree_byte_span_t plan = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    epoch.transfer.results =
        (const loom_serve_finite_result_t**)((uint8_t*)
                                                 epoch.transfer.operations +
                                             operation_bytes);
    status = iree_vm_buffer_map_write(epoch.plan, 0, plan_length, &plan);
  }
  iree_host_size_t record_index = 0;
  iree_host_size_t metadata_offset = record_length;
  if (iree_status_is_ok(status)) {
    loom_serve_text_media_epoch_store(first_span_count, first_spans, 0,
                                      plan.data, &record_index,
                                      &metadata_offset, &epoch);
    loom_serve_text_media_epoch_store(next_span_count, next_spans,
                                      LOOM_SERVE_TEXT_ROW_CAPACITY, plan.data,
                                      &record_index, &metadata_offset, &epoch);
  }
  if (iree_status_is_ok(status)) {
    *out_epoch = epoch;
  } else {
    loom_serve_text_media_epoch_deinitialize(&epoch);
  }
  return status;
}

iree_status_t loom_serve_text_media_epoch_enqueue(
    loom_serve_text_media_epoch_t* epoch, loom_serve_execution_t* execution,
    iree_hal_buffer_t* target, uint64_t* out_value) {
  // Cohorts from one component share timelines. Merge their values while
  // preserving distinct execution generations after component reconstruction.
  loom_serve_execution_frontier_t ready =
      loom_serve_finite_result_ready(epoch->transfer.results[0]);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 1;
       i < epoch->transfer.result_count && iree_status_is_ok(status); ++i) {
    const loom_serve_execution_frontier_t next =
        loom_serve_finite_result_ready(epoch->transfer.results[i]);
    if (ready.semaphores[0] == next.semaphores[0] &&
        ready.semaphores[1] == next.semaphores[1]) {
      ready.values[0] = iree_max(ready.values[0], next.values[0]);
      ready.values[1] = iree_max(ready.values[1], next.values[1]);
    } else {
      const iree_hal_semaphore_list_t waits = {2, ready.semaphores,
                                               ready.values};
      status = loom_serve_execution_barrier(execution, waits);
      ready = next;
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t waits = {2, ready.semaphores, ready.values};
    status = loom_serve_execution_barrier(execution, waits);
  }
  iree_hal_transfer_operation_t* operations = epoch->transfer.operations;
  for (iree_host_size_t i = 0; i < epoch->transfer.operation_count; ++i) {
    operations[i].copy.target_buffer = target;
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_transfer(
        execution, epoch->transfer.operation_count, operations, out_value);
  }
  iree_allocator_free(epoch->host_allocator, epoch->transfer.operations);
  memset(&epoch->transfer, 0, sizeof(epoch->transfer));
  return status;
}

void loom_serve_text_media_epoch_deinitialize(
    loom_serve_text_media_epoch_t* epoch) {
  iree_allocator_free(epoch->host_allocator, epoch->transfer.operations);
  iree_vm_buffer_release(epoch->packet_records);
  iree_vm_buffer_release(epoch->plan);
  memset(epoch, 0, sizeof(*epoch));
}
