// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/layer_data.h"

#include <math.h>
#include <string.h>

#include "iree/io/file_contents.h"

static iree_status_t qwen_tooling_layer_data_map(
    iree_string_view_t path, iree_host_size_t expected_byte_length,
    iree_allocator_t host_allocator, iree_io_file_contents_t** out_contents) {
  *out_contents = NULL;
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "layer data path must not be empty");
  }

  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 host_allocator, out_contents));
  if ((*out_contents)->const_buffer.data_length != expected_byte_length) {
    const iree_host_size_t actual_byte_length =
        (*out_contents)->const_buffer.data_length;
    iree_io_file_contents_free(*out_contents);
    *out_contents = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "layer data file '%.*s' has %" PRIhsz
                            " bytes; expected exactly "
                            "%" PRIhsz " raw F32 bytes",
                            (int)path.size, path.data, actual_byte_length,
                            expected_byte_length);
  }
  return iree_ok_status();
}

iree_status_t qwen_tooling_layer_hidden_state_byte_length(
    iree_host_size_t token_count, iree_host_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  if (token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "layer token count must be nonzero");
  }

  iree_host_size_t element_count = 0;
  if (!iree_host_size_checked_mul(token_count, QWEN_TOOLING_LAYER_HIDDEN_SIZE,
                                  &element_count) ||
      !iree_host_size_checked_mul(element_count, sizeof(float),
                                  out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "layer hidden-state byte length overflows");
  }
  return iree_ok_status();
}

iree_status_t qwen_tooling_layer_data_initialize(
    iree_string_view_t input_path, iree_string_view_t expected_path,
    iree_host_size_t token_count, iree_allocator_t host_allocator,
    qwen_tooling_layer_data_t* out_data) {
  IREE_ASSERT_ARGUMENT(out_data);
  memset(out_data, 0, sizeof(*out_data));

  iree_status_t status = qwen_tooling_layer_hidden_state_byte_length(
      token_count, &out_data->byte_length);
  if (iree_status_is_ok(status)) {
    status =
        qwen_tooling_layer_data_map(input_path, out_data->byte_length,
                                    host_allocator, &out_data->input_contents);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_layer_data_map(expected_path, out_data->byte_length,
                                         host_allocator,
                                         &out_data->expected_contents);
  }
  if (!iree_status_is_ok(status)) {
    qwen_tooling_layer_data_deinitialize(out_data);
  }
  return status;
}

void qwen_tooling_layer_data_deinitialize(qwen_tooling_layer_data_t* data) {
  if (!data) return;
  iree_io_file_contents_free(data->expected_contents);
  iree_io_file_contents_free(data->input_contents);
  memset(data, 0, sizeof(*data));
}

iree_const_byte_span_t qwen_tooling_layer_data_input(
    const qwen_tooling_layer_data_t* data) {
  IREE_ASSERT_ARGUMENT(data);
  IREE_ASSERT_ARGUMENT(data->input_contents);
  return data->input_contents->const_buffer;
}

iree_status_t qwen_tooling_layer_data_compare(
    const qwen_tooling_layer_data_t* data, iree_const_byte_span_t actual_data,
    float absolute_tolerance, float relative_tolerance,
    qwen_tooling_layer_comparison_t* out_comparison) {
  IREE_ASSERT_ARGUMENT(data);
  IREE_ASSERT_ARGUMENT(out_comparison);
  memset(out_comparison, 0, sizeof(*out_comparison));

  if (absolute_tolerance < 0.0f || relative_tolerance < 0.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "layer comparison tolerances must be nonnegative");
  }
  if (actual_data.data_length != data->byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "actual layer output has %" PRIhsz
                            " bytes; expected exactly %" PRIhsz,
                            actual_data.data_length, data->byte_length);
  }

  const iree_const_byte_span_t expected_data =
      data->expected_contents->const_buffer;
  const iree_host_size_t element_count = data->byte_length / sizeof(float);
  out_comparison->element_count = element_count;
  out_comparison->first_mismatch_index = element_count;
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    float actual = 0.0f;
    float expected = 0.0f;
    memcpy(&actual, actual_data.data + i * sizeof(float), sizeof(actual));
    memcpy(&expected, expected_data.data + i * sizeof(float), sizeof(expected));

    bool matches = actual == expected;
    if (!matches && isnan(actual) && isnan(expected)) {
      matches = true;
    }

    float absolute_error = 0.0f;
    float relative_error = 0.0f;
    if (!matches && isfinite(actual) && isfinite(expected)) {
      absolute_error = fabsf(actual - expected);
      if (expected != 0.0f) {
        relative_error = absolute_error / fabsf(expected);
      }
      matches = absolute_error <=
                absolute_tolerance + relative_tolerance * fabsf(expected);
    }
    out_comparison->maximum_absolute_error =
        fmaxf(out_comparison->maximum_absolute_error, absolute_error);
    out_comparison->maximum_relative_error =
        fmaxf(out_comparison->maximum_relative_error, relative_error);

    if (!matches) {
      if (out_comparison->mismatch_count == 0) {
        out_comparison->first_mismatch_index = i;
        out_comparison->first_actual_value = actual;
        out_comparison->first_expected_value = expected;
      }
      ++out_comparison->mismatch_count;
    }
  }

  if (out_comparison->mismatch_count != 0) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "%" PRIhsz " of %" PRIhsz
        " layer output elements differ; first mismatch at index %" PRIhsz
        " (actual=%g expected=%g), max_abs=%g max_rel=%g",
        out_comparison->mismatch_count, out_comparison->element_count,
        out_comparison->first_mismatch_index,
        out_comparison->first_actual_value,
        out_comparison->first_expected_value,
        out_comparison->maximum_absolute_error,
        out_comparison->maximum_relative_error);
  }
  return iree_ok_status();
}
