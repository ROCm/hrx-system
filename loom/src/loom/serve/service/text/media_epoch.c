// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_epoch.h"

#include <string.h>

enum {
  MEDIA_EPOCH_RECORD_BYTES =
      LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT * sizeof(uint32_t),
};

typedef struct media_epoch_layout_t {
  // Number of projected feature rows intersecting scheduled spans.
  iree_host_size_t feature_count;
  // Source-private metadata bytes copied once per intersected image span.
  iree_host_size_t metadata_length;
} media_epoch_layout_t;

static bool loom_serve_text_media_epoch_add_span_layout(
    const loom_serve_text_media_epoch_span_t* span,
    media_epoch_layout_t* layout) {
  if (!span->media) {
    return true;
  }
  const iree_host_size_t span_end = span->input_offset + span->input_count;
  for (iree_host_size_t i = 0; i < span->media->span_count; ++i) {
    const loom_serve_text_media_input_span_t* media_span =
        &span->media->spans[i];
    const iree_host_size_t media_end =
        (iree_host_size_t)media_span->input_offset + media_span->input_count;
    const iree_host_size_t begin =
        iree_max(span->input_offset, media_span->input_offset);
    const iree_host_size_t end = iree_min(span_end, media_end);
    if (begin >= end) {
      continue;
    }
    if (!iree_host_size_checked_add(layout->feature_count, end - begin,
                                    &layout->feature_count) ||
        !iree_host_size_checked_add(layout->metadata_length,
                                    media_span->metadata_length,
                                    &layout->metadata_length)) {
      return false;
    }
  }
  return true;
}

static bool loom_serve_text_media_epoch_add_layout(
    iree_host_size_t span_count,
    const loom_serve_text_media_epoch_span_t* spans,
    media_epoch_layout_t* layout) {
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    if (!loom_serve_text_media_epoch_add_span_layout(&spans[i], layout)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_serve_text_media_epoch_store_spans(
    iree_host_size_t span_count,
    const loom_serve_text_media_epoch_span_t* spans, uint8_t* plan_bytes,
    iree_host_size_t* record_index, iree_host_size_t* metadata_offset,
    iree_host_size_t* feature_offset,
    loom_serve_text_media_epoch_feature_t* features) {
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_media_epoch_span_t* span = &spans[i];
    if (!span->media) {
      continue;
    }
    const iree_host_size_t span_end = span->input_offset + span->input_count;
    for (iree_host_size_t j = 0; j < span->media->span_count; ++j) {
      const loom_serve_text_media_input_span_t* media_span =
          &span->media->spans[j];
      const iree_host_size_t media_end =
          (iree_host_size_t)media_span->input_offset + media_span->input_count;
      const iree_host_size_t begin =
          iree_max(span->input_offset, media_span->input_offset);
      const iree_host_size_t end = iree_min(span_end, media_end);
      if (begin >= end) {
        continue;
      }
      if (media_span->metadata_length) {
        memcpy(plan_bytes + *metadata_offset,
               span->media->metadata.data + media_span->metadata_offset,
               media_span->metadata_length);
      }
      for (iree_host_size_t input = begin; input < end; ++input) {
        const iree_host_size_t media_offset = input - media_span->input_offset;
        const iree_host_size_t row_index =
            media_span->row_offset + media_offset;
        const iree_const_byte_span_t row = span->media->rows[row_index];
        iree_host_size_t aligned_offset = 0;
        if (!iree_host_size_checked_add(
                *feature_offset,
                LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT - 1,
                &aligned_offset)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "mixed input packet size overflow");
        }
        aligned_offset &= ~(LOOM_SERVE_TEXT_MEDIA_PACKET_FEATURE_ALIGNMENT - 1);
        iree_host_size_t next_offset = 0;
        if (!iree_host_size_checked_add(aligned_offset, row.data_length,
                                        &next_offset) ||
            next_offset > UINT32_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "mixed input packet exceeds 32-bit extent");
        }
        const uint32_t fields[LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT] = {
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX] = span->plan_index,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET] =
                (uint32_t)(input - span->input_offset),
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET] =
                (uint32_t)aligned_offset,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH] =
                (uint32_t)row.data_length,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET] =
                (uint32_t)*metadata_offset,
            [LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH] =
                media_span->metadata_length,
        };
        uint8_t* record = plan_bytes + *record_index * MEDIA_EPOCH_RECORD_BYTES;
        for (iree_host_size_t field = 0;
             field < LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT; ++field) {
          iree_unaligned_store_le_u32(record + field * sizeof(uint32_t),
                                      fields[field]);
        }
        features[*record_index] = (loom_serve_text_media_epoch_feature_t){
            .source = row,
            .target_offset = (uint32_t)aligned_offset,
        };
        ++*record_index;
        *feature_offset = next_offset;
      }
      *metadata_offset += media_span->metadata_length;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_media_epoch_initialize(
    iree_host_size_t first_span_count,
    const loom_serve_text_media_epoch_span_t* first_spans,
    iree_host_size_t first_packet_capacity, iree_host_size_t next_span_count,
    const loom_serve_text_media_epoch_span_t* next_spans,
    iree_host_size_t next_packet_capacity, iree_allocator_t host_allocator,
    loom_serve_text_media_epoch_t* out_epoch) {
  memset(out_epoch, 0, sizeof(*out_epoch));
  media_epoch_layout_t first_layout = {0};
  media_epoch_layout_t next_layout = {0};
  if (!loom_serve_text_media_epoch_add_layout(first_span_count, first_spans,
                                              &first_layout) ||
      !loom_serve_text_media_epoch_add_layout(next_span_count, next_spans,
                                              &next_layout)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input epoch layout overflow");
  }
  iree_host_size_t feature_count = 0;
  iree_host_size_t metadata_length = 0;
  iree_host_size_t packet_capacity = 0;
  if (!iree_host_size_checked_add(first_layout.feature_count,
                                  next_layout.feature_count, &feature_count) ||
      !iree_host_size_checked_add(first_layout.metadata_length,
                                  next_layout.metadata_length,
                                  &metadata_length) ||
      !iree_host_size_checked_add(first_packet_capacity, next_packet_capacity,
                                  &packet_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input epoch layout overflow");
  }
  if (!feature_count) {
    return iree_ok_status();
  }
  iree_host_size_t plan_record_bytes = 0;
  iree_host_size_t plan_length = 0;
  iree_host_size_t packet_record_bytes = 0;
  if (first_layout.feature_count > UINT32_MAX ||
      next_layout.feature_count > UINT32_MAX ||
      first_packet_capacity > UINT32_MAX || next_packet_capacity > UINT32_MAX ||
      !iree_host_size_checked_mul(feature_count, MEDIA_EPOCH_RECORD_BYTES,
                                  &plan_record_bytes) ||
      !iree_host_size_checked_add(plan_record_bytes, metadata_length,
                                  &plan_length) ||
      !iree_host_size_checked_mul(packet_capacity,
                                  LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES,
                                  &packet_record_bytes) ||
      plan_length > UINT32_MAX || packet_record_bytes > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mixed input epoch exceeds 32-bit extent");
  }

  loom_serve_text_media_epoch_t epoch = {
      .host_allocator = host_allocator,
      .first_feature_count = (uint32_t)first_layout.feature_count,
      .next_feature_count = (uint32_t)next_layout.feature_count,
      .first_packet_capacity = (uint32_t)first_packet_capacity,
      .next_packet_capacity = (uint32_t)next_packet_capacity,
  };
  iree_status_t status = iree_vm_buffer_create(
      plan_length, iree_alignof(uint32_t), host_allocator, &epoch.plan);
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_create(packet_record_bytes, iree_alignof(uint32_t),
                                   host_allocator, &epoch.packet_records);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_map_write(epoch.packet_records, 0, packet_record_bytes,
                                 &epoch.packet_record_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, feature_count,
                                         sizeof(*epoch.features),
                                         (void**)&epoch.features);
  }
  iree_byte_span_t plan_bytes = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(epoch.plan, 0, plan_length, &plan_bytes);
  }
  iree_host_size_t record_index = 0;
  iree_host_size_t metadata_offset = plan_record_bytes;
  iree_host_size_t feature_offset = packet_record_bytes;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_epoch_store_spans(
        first_span_count, first_spans, plan_bytes.data, &record_index,
        &metadata_offset, &feature_offset, epoch.features);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_media_epoch_store_spans(
        next_span_count, next_spans, plan_bytes.data, &record_index,
        &metadata_offset, &feature_offset, epoch.features);
  }
  if (iree_status_is_ok(status)) {
    epoch.device_length = (uint32_t)feature_offset;
    *out_epoch = epoch;
  } else {
    loom_serve_text_media_epoch_deinitialize(&epoch);
  }
  return status;
}

iree_status_t loom_serve_text_media_epoch_upload(
    const loom_serve_text_media_epoch_t* epoch,
    loom_serve_execution_t* execution, iree_const_byte_span_t metadata,
    iree_hal_buffer_t* metadata_target, iree_hal_buffer_t* target,
    uint64_t* out_value) {
  const iree_host_size_t feature_count =
      (iree_host_size_t)epoch->first_feature_count + epoch->next_feature_count;
  iree_hal_transfer_operation_t* operations = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(epoch->host_allocator, feature_count + 2,
                                  sizeof(*operations), (void**)&operations));
  operations[0] = (iree_hal_transfer_operation_t){
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {.source = metadata.data,
                 .target_buffer = metadata_target,
                 .length = metadata.data_length},
  };
  operations[1] = (iree_hal_transfer_operation_t){
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {.source = epoch->packet_record_bytes.data,
                 .target_buffer = target,
                 .length = epoch->packet_record_bytes.data_length},
  };
  for (iree_host_size_t i = 0; i < feature_count; ++i) {
    operations[i + 2] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = epoch->features[i].source.data,
                   .target_buffer = target,
                   .target_offset = epoch->features[i].target_offset,
                   .length = epoch->features[i].source.data_length},
    };
  }
  const iree_status_t status = loom_serve_execution_transfer(
      execution, feature_count + 2, operations, out_value);
  iree_allocator_free(epoch->host_allocator, operations);
  return status;
}

void loom_serve_text_media_epoch_deinitialize(
    loom_serve_text_media_epoch_t* epoch) {
  if (epoch->features) {
    iree_allocator_free(epoch->host_allocator, epoch->features);
  }
  iree_vm_buffer_release(epoch->packet_records);
  iree_vm_buffer_release(epoch->plan);
  memset(epoch, 0, sizeof(*epoch));
}
