// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/capture.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/tooling/filesystem.h"

#define ID4_TOOLING_CAPTURE_NPY_PREFIX_LENGTH 10

typedef struct id4_tooling_capture_host_bytes_t {
  // Number of valid bytes in data.
  iree_host_size_t length;
  // Owned host allocation containing copied tensor bytes.
  uint8_t* data;
} id4_tooling_capture_host_bytes_t;

static void id4_tooling_capture_host_bytes_deinitialize(
    id4_tooling_capture_host_bytes_t* bytes, iree_allocator_t host_allocator) {
  if (!bytes) return;
  iree_allocator_free(host_allocator, bytes->data);
  memset(bytes, 0, sizeof(*bytes));
}

static iree_status_t id4_tooling_capture_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_tooling_capture_dup_cstring(
    iree_string_view_t value, iree_allocator_t host_allocator,
    char** out_string) {
  *out_string = NULL;
  if (value.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "capture path is too large");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, value.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, value.data, value.size);
  storage[value.size] = 0;
  *out_string = storage;
  return iree_ok_status();
}

static iree_status_t id4_tooling_capture_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_tooling_capture_validate_options(
    const id4_tooling_capture_execution_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture options are required");
  }
  IREE_RETURN_IF_ERROR(id4_tooling_capture_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("capture")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "capture extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->run_id)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture run_id is required");
  }
  if (iree_string_view_is_empty(options->output_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture output directory is required");
  }
  if (!options->plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture plan is required");
  }
  if (!options->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture device is required");
  }
  if (iree_allocator_is_null(options->host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture host allocator is required");
  }
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(options->plan);
  if (options->boundary_binding_count != boundary_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture boundary binding count %" PRIhsz
                            " does not match plan boundary count %" PRIhsz,
                            options->boundary_binding_count, boundary_count);
  }
  if (boundary_count != 0 && !options->boundary_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture boundary bindings are required");
  }
  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(options->plan);
  if (options->diagnostic_tap_binding_count != diagnostic_tap_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "capture diagnostic tap binding count %" PRIhsz
        " does not match plan diagnostic tap count %" PRIhsz,
        options->diagnostic_tap_binding_count, diagnostic_tap_count);
  }
  if (diagnostic_tap_count != 0 && !options->diagnostic_tap_bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture diagnostic tap bindings are required");
  }
  return id4_tooling_capture_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("capture wait"));
}

static iree_status_t id4_tooling_capture_write_file(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_allocator_t host_allocator) {
  char* path_string = NULL;
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_dup_cstring(path, host_allocator, &path_string));
  FILE* file = fopen(path_string, "wb");
  const int open_errno = errno;
  iree_allocator_free(host_allocator, path_string);
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(open_errno),
                            "failed to open capture file (%d)", open_errno);
  }
  iree_status_t status = iree_ok_status();
  if (contents.data_length != 0) {
    const size_t written = fwrite(contents.data, 1, contents.data_length, file);
    if (written != contents.data_length) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "failed to write capture file contents");
    }
  }
  if (fclose(file) != 0 && iree_status_is_ok(status)) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "failed to close capture file (%d)", errno);
  }
  return status;
}

static iree_status_t id4_tooling_capture_append_json_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char c = (unsigned char)value.data[i];
    switch (c) {
      case '\"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\""));
        break;
      }
      case '\\': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\\"));
        break;
      }
      case '\b': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\b"));
        break;
      }
      case '\f': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\f"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\t"));
        break;
      }
      default: {
        if (c < 0x20) {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_format(builder, "\\u%04x", c));
        } else {
          char* storage = NULL;
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_inline(builder, 1, &storage));
          storage[0] = (char)c;
        }
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t id4_tooling_capture_append_shape_json(
    iree_string_builder_t* builder, id4_pipeline_tensor_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t id4_tooling_capture_npy_descriptor(
    id4_pipeline_tensor_dtype_t dtype, iree_string_view_t* out_descriptor) {
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
      *out_descriptor = IREE_SV("<f4");
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_F16:
      *out_descriptor = IREE_SV("<f2");
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      // NumPy has no portable bfloat16 descriptor. The manifest carries the
      // logical dtype while the NPY payload stores the raw 16-bit lanes.
      *out_descriptor = IREE_SV("<u2");
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_I32:
      *out_descriptor = IREE_SV("<i4");
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_U32:
      *out_descriptor = IREE_SV("<u4");
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "capture only supports f32, f16, bf16, i32, and u32 tensor payloads");
  }
}

static iree_status_t id4_tooling_capture_append_npy_shape(
    iree_string_builder_t* builder, id4_pipeline_tensor_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "("));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  if (shape.rank == 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  }
  return iree_string_builder_append_cstring(builder, ")");
}

static iree_status_t id4_tooling_capture_write_npy(
    iree_string_view_t path, const id4_pipeline_tensor_layout_t* layout,
    iree_const_byte_span_t payload, iree_allocator_t host_allocator) {
  iree_string_view_t descriptor = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_npy_descriptor(layout->dtype, &descriptor));

  iree_string_builder_t header_builder;
  iree_string_builder_initialize(host_allocator, &header_builder);
  iree_status_t status =
      iree_string_builder_append_cstring(&header_builder, "{'descr': '");
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&header_builder, descriptor);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &header_builder, "', 'fortran_order': False, 'shape': ");
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_tooling_capture_append_npy_shape(&header_builder, layout->shape);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&header_builder, ", }");
  }
  iree_host_size_t padding = 0;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t unpadded_length =
        ID4_TOOLING_CAPTURE_NPY_PREFIX_LENGTH +
        iree_string_builder_size(&header_builder) + 1;
    padding = 16 - (unpadded_length % 16);
    if (padding == 16) padding = 0;
  }

  iree_string_builder_t file_builder;
  iree_string_builder_initialize(host_allocator, &file_builder);
  if (iree_status_is_ok(status)) {
    const uint8_t prefix[] = {
        0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00, 0x00, 0x00,
    };
    status = iree_string_builder_append_string(
        &file_builder,
        iree_make_string_view((const char*)prefix, sizeof(prefix)));
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t header_length =
        iree_string_builder_size(&header_builder) + padding + 1;
    if (header_length > UINT16_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "capture NPY header is too large");
    } else {
      char* file_storage = (char*)iree_string_builder_buffer(&file_builder);
      file_storage[8] = (char)(header_length & 0xFFu);
      file_storage[9] = (char)((header_length >> 8) & 0xFFu);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        &file_builder, iree_string_builder_view(&header_builder));
  }
  for (iree_host_size_t i = 0; i < padding && iree_status_is_ok(status); ++i) {
    status = iree_string_builder_append_cstring(&file_builder, " ");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&file_builder, "\n");
  }
  if (iree_status_is_ok(status) && payload.data_length != 0) {
    status = iree_string_builder_append_string(
        &file_builder,
        iree_make_string_view((const char*)payload.data, payload.data_length));
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_capture_write_file(
        path,
        iree_make_const_byte_span(iree_string_builder_buffer(&file_builder),
                                  iree_string_builder_size(&file_builder)),
        host_allocator);
  }
  iree_string_builder_deinitialize(&file_builder);
  iree_string_builder_deinitialize(&header_builder);
  return status;
}

static iree_status_t id4_tooling_capture_readback_tensor(
    const id4_tooling_capture_execution_options_t* options,
    const id4_pipeline_tensor_layout_t* layout,
    const iree_hal_buffer_binding_t* binding,
    id4_tooling_capture_host_bytes_t* out_bytes) {
  memset(out_bytes, 0, sizeof(*out_bytes));
  if (layout->byte_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "capture tensor %.*s byte length %" PRIu64
                            " exceeds host capacity",
                            (int)layout->name.size, layout->name.data,
                            (uint64_t)layout->byte_length);
  }

  iree_hal_buffer_params_t readback_params;
  memset(&readback_params, 0, sizeof(readback_params));
  readback_params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  readback_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  readback_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  readback_params.queue_affinity = options->queue_affinity;

  iree_hal_buffer_t* readback_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(options->device), readback_params,
      layout->byte_length, &readback_buffer));

  iree_hal_semaphore_t* copy_semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      options->device, options->queue_affinity, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &copy_semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_t* signal_semaphores[] = {copy_semaphore};
    uint64_t signal_values[] = {1};
    iree_hal_semaphore_list_t signal_list = {
        IREE_ARRAYSIZE(signal_semaphores),
        signal_semaphores,
        signal_values,
    };
    status = iree_hal_device_queue_copy(
        options->device, options->queue_affinity, options->wait_semaphore_list,
        signal_list, binding->buffer, binding->offset, readback_buffer,
        /*target_offset=*/0, layout->byte_length, IREE_HAL_COPY_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(
          signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status)) {
    out_bytes->length = (iree_host_size_t)layout->byte_length;
    status = iree_allocator_malloc_array(
        options->host_allocator, out_bytes->length, sizeof(out_bytes->data[0]),
        (void**)&out_bytes->data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(readback_buffer, /*source_offset=*/0,
                                      out_bytes->data, out_bytes->length);
  }
  iree_hal_semaphore_release(copy_semaphore);
  iree_hal_buffer_release(readback_buffer);
  if (!iree_status_is_ok(status)) {
    id4_tooling_capture_host_bytes_deinitialize(out_bytes,
                                                options->host_allocator);
  }
  return status;
}

static iree_status_t id4_tooling_capture_append_record_json(
    iree_string_builder_t* manifest_builder, iree_string_view_t stage_name,
    const id4_pipeline_tensor_layout_t* layout, iree_string_view_t file_name,
    iree_host_size_t record_ordinal) {
  if (record_ordinal != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(manifest_builder, ",\n"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(manifest_builder, "    {\"dtype\":"));
  IREE_RETURN_IF_ERROR(id4_tooling_capture_append_json_string(
      manifest_builder, id4_pipeline_tensor_dtype_format(layout->dtype)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(manifest_builder, ",\"file\":"));
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_append_json_string(manifest_builder, file_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      manifest_builder, ",\"kind\":\"tensor\""));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(manifest_builder, ",\"name\":"));
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_append_json_string(manifest_builder, layout->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      manifest_builder, ",\"role\":\"actual\""));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(manifest_builder, ",\"shape\":"));
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_append_shape_json(manifest_builder, layout->shape));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(manifest_builder, ",\"stage\":"));
  IREE_RETURN_IF_ERROR(
      id4_tooling_capture_append_json_string(manifest_builder, stage_name));
  return iree_string_builder_append_cstring(manifest_builder, "}");
}

static iree_status_t id4_tooling_capture_capture_tensor(
    const id4_tooling_capture_execution_options_t* options,
    iree_string_builder_t* manifest_builder, iree_string_view_t stage_name,
    iree_string_view_t file_prefix, iree_host_size_t file_ordinal,
    const id4_pipeline_tensor_layout_t* layout,
    const iree_hal_buffer_binding_t* binding, iree_host_size_t* record_count) {
  char file_name_storage[64];
  snprintf(file_name_storage, sizeof(file_name_storage),
           "%.*s_%04" PRIhsz ".npy", (int)file_prefix.size, file_prefix.data,
           file_ordinal);
  iree_string_view_t file_name = iree_make_cstring_view(file_name_storage);
  iree_string_view_t file_path = iree_string_view_empty();
  iree_status_t status =
      id4_tooling_format_child_path(options->output_directory, file_name,
                                    options->host_allocator, &file_path);
  id4_tooling_capture_host_bytes_t bytes;
  memset(&bytes, 0, sizeof(bytes));
  if (iree_status_is_ok(status)) {
    status =
        id4_tooling_capture_readback_tensor(options, layout, binding, &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_capture_write_npy(
        file_path, layout, iree_make_const_byte_span(bytes.data, bytes.length),
        options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_capture_append_record_json(
        manifest_builder, stage_name, layout, file_name, *record_count);
  }
  id4_tooling_capture_host_bytes_deinitialize(&bytes, options->host_allocator);
  id4_tooling_free_path(&file_path, options->host_allocator);
  if (iree_status_is_ok(status)) ++*record_count;
  return status;
}

iree_status_t id4_tooling_capture_execution(
    const id4_tooling_capture_execution_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_tooling_capture_validate_options(options));
  IREE_RETURN_IF_ERROR(id4_tooling_ensure_directory(options->output_directory,
                                                    options->host_allocator));

  iree_string_builder_t manifest_builder;
  iree_string_builder_initialize(options->host_allocator, &manifest_builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &manifest_builder, "{\n  \"records\": [\n");

  const iree_string_view_t stage_name =
      id4_pipeline_plan_stage_name(options->plan);
  iree_host_size_t record_count = 0;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(options->plan);
  for (iree_host_size_t i = 0; i < boundary_count && iree_status_is_ok(status);
       ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(options->plan, i);
    if (!boundary ||
        !iree_all_bits_set(boundary->flags,
                           ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
      continue;
    }
    status = id4_tooling_capture_capture_tensor(
        options, &manifest_builder, stage_name, IREE_SV("boundary"), i,
        &boundary->layout, &options->boundary_bindings[i], &record_count);
  }

  const iree_host_size_t diagnostic_tap_count =
      id4_pipeline_plan_diagnostic_tap_count(options->plan);
  for (iree_host_size_t i = 0;
       i < diagnostic_tap_count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(options->plan, i);
    status = id4_tooling_capture_capture_tensor(
        options, &manifest_builder, stage_name, IREE_SV("tap"), i, &tap->layout,
        &options->diagnostic_tap_bindings[i], &record_count);
  }

  if (iree_status_is_ok(status) && record_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "capture plan has no exported boundary tensors "
                              "or diagnostic taps");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&manifest_builder,
                                                "\n  ],\n  \"run_id\": ");
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_capture_append_json_string(&manifest_builder,
                                                    options->run_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        &manifest_builder, ",\n  \"schema_version\": 1\n}\n");
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t manifest_path = iree_string_view_empty();
    status = id4_tooling_format_child_path(
        options->output_directory, IREE_SV("manifest.json"),
        options->host_allocator, &manifest_path);
    if (iree_status_is_ok(status)) {
      status = id4_tooling_capture_write_file(
          manifest_path,
          iree_make_const_byte_span(
              iree_string_builder_buffer(&manifest_builder),
              iree_string_builder_size(&manifest_builder)),
          options->host_allocator);
    }
    id4_tooling_free_path(&manifest_path, options->host_allocator);
  }
  iree_string_builder_deinitialize(&manifest_builder);
  return status;
}
