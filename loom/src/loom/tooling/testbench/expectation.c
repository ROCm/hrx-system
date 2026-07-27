// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/expectation.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/tooling/buffer_view_matchers.h"
#include "loom/util/json.h"

const char* loom_testbench_expectation_kind_name(
    loom_testbench_expectation_kind_t kind) {
  switch (kind) {
    case LOOM_TESTBENCH_EXPECTATION_NONE:
      return "none";
    case LOOM_TESTBENCH_EXPECTATION_EQUAL:
      return "equal";
    case LOOM_TESTBENCH_EXPECTATION_BITWISE:
      return "bitwise";
    case LOOM_TESTBENCH_EXPECTATION_CLOSE:
      return "close";
    case LOOM_TESTBENCH_EXPECTATION_SHAPE:
      return "shape";
    case LOOM_TESTBENCH_EXPECTATION_CUSTOM:
      return "custom";
    case LOOM_TESTBENCH_EXPECTATION_EVENT:
      return "event";
  }
  return "unknown";
}

void loom_testbench_expectation_options_initialize(
    loom_testbench_expectation_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->providers = loom_testbench_expectation_provider_list_empty();
}

static bool loom_testbench_find_expectation_provider(
    const loom_testbench_expectation_options_t* options,
    iree_string_view_t name,
    loom_testbench_expectation_callback_t* out_evaluate) {
  for (iree_host_size_t provider_index = 0;
       provider_index < options->providers.count; ++provider_index) {
    const loom_testbench_expectation_provider_t* provider =
        &options->providers.values[provider_index];
    if (iree_string_view_equal(provider->name, name)) {
      *out_evaluate = provider->evaluate;
      return true;
    }
  }
  memset(out_evaluate, 0, sizeof(*out_evaluate));
  return false;
}

iree_status_t loom_testbench_prepare_case_expectations(
    const loom_testbench_expectation_options_t* options,
    const loom_testbench_case_plan_t* case_plan, iree_arena_allocator_t* arena,
    loom_testbench_expectation_schedule_t* out_schedule) {
  memset(out_schedule, 0, sizeof(*out_schedule));

  loom_testbench_prepared_expectation_t* prepared_expectations = NULL;
  if (case_plan->expectation_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, case_plan->expectation_count, sizeof(*prepared_expectations),
        (void**)&prepared_expectations));
    memset(prepared_expectations, 0,
           case_plan->expectation_count * sizeof(*prepared_expectations));
  }

  for (iree_host_size_t expectation_index = 0;
       expectation_index < case_plan->expectation_count; ++expectation_index) {
    const loom_testbench_expectation_plan_t* expectation =
        &case_plan->expectations[expectation_index];
    prepared_expectations[expectation_index].plan = expectation;
    if (expectation->kind != LOOM_TESTBENCH_EXPECTATION_CUSTOM) {
      continue;
    }

    loom_testbench_expectation_callback_t evaluate = {0};
    if (!loom_testbench_find_expectation_provider(
            options, expectation->custom.provider, &evaluate)) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "expectation provider `%.*s` is not configured",
                              (int)expectation->custom.provider.size,
                              expectation->custom.provider.data);
    }
    if (!evaluate.fn) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expectation provider `%.*s` has no callback",
                              (int)expectation->custom.provider.size,
                              expectation->custom.provider.data);
    }
    prepared_expectations[expectation_index].custom_evaluate = evaluate;
  }

  out_schedule->expectations = prepared_expectations;
  out_schedule->expectation_count = case_plan->expectation_count;
  return iree_ok_status();
}

iree_status_t loom_testbench_expectation_report_initialize(
    iree_host_size_t failure_capacity, iree_allocator_t host_allocator,
    loom_testbench_expectation_report_t* out_report) {
  memset(out_report, 0, sizeof(*out_report));
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  out_report->host_allocator = host_allocator;
  out_report->failure_capacity = failure_capacity;
  iree_string_builder_initialize(host_allocator, &out_report->detail_builder);
  if (failure_capacity == 0) {
    return iree_ok_status();
  }
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, failure_capacity, sizeof(*out_report->failures),
      (void**)&out_report->failures);
  if (!iree_status_is_ok(status)) {
    loom_testbench_expectation_report_deinitialize(out_report);
  }
  return status;
}

void loom_testbench_expectation_report_reset(
    loom_testbench_expectation_report_t* report) {
  report->expectation_count = 0;
  report->passed_count = 0;
  report->failure_count = 0;
  iree_string_builder_reset(&report->detail_builder);
  if (report->failures) {
    memset(report->failures, 0,
           report->failure_capacity * sizeof(*report->failures));
  }
}

void loom_testbench_expectation_report_deinitialize(
    loom_testbench_expectation_report_t* report) {
  if (!report) {
    return;
  }
  iree_string_builder_deinitialize(&report->detail_builder);
  iree_allocator_free(report->host_allocator, report->failures);
  memset(report, 0, sizeof(*report));
}

iree_string_view_t loom_testbench_expectation_failure_detail(
    const loom_testbench_expectation_report_t* report,
    const loom_testbench_expectation_failure_t* failure) {
  iree_string_view_t details =
      iree_string_builder_view(&report->detail_builder);
  if (failure->detail_offset > details.size ||
      failure->detail_length > details.size - failure->detail_offset) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(details.data + failure->detail_offset,
                               failure->detail_length);
}

static const char* loom_testbench_tooling_value_kind_name(
    iree_tooling_value_kind_t kind) {
  switch (kind) {
    case IREE_TOOLING_VALUE_KIND_NONE:
      return "none";
    case IREE_TOOLING_VALUE_KIND_I32:
      return "i32";
    case IREE_TOOLING_VALUE_KIND_U32:
      return "u32";
    case IREE_TOOLING_VALUE_KIND_I64:
      return "i64";
    case IREE_TOOLING_VALUE_KIND_U64:
      return "u64";
    case IREE_TOOLING_VALUE_KIND_F32:
      return "f32";
    case IREE_TOOLING_VALUE_KIND_F64:
      return "f64";
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      return "raw_u32";
    default:
      return "unknown";
  }
}

static iree_host_size_t loom_testbench_tooling_value_storage_size(
    iree_tooling_value_kind_t kind) {
  switch (kind) {
    case IREE_TOOLING_VALUE_KIND_I32:
    case IREE_TOOLING_VALUE_KIND_U32:
    case IREE_TOOLING_VALUE_KIND_F32:
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      return sizeof(uint32_t);
    case IREE_TOOLING_VALUE_KIND_I64:
    case IREE_TOOLING_VALUE_KIND_U64:
    case IREE_TOOLING_VALUE_KIND_F64:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

static iree_status_t loom_testbench_append_tooling_value(
    const iree_tooling_value_t* value, iree_string_builder_t* detail_builder) {
  switch (value->kind) {
    case IREE_TOOLING_VALUE_KIND_I32:
      return iree_string_builder_append_format(detail_builder, "%" PRIi32,
                                               value->storage.i32);
    case IREE_TOOLING_VALUE_KIND_U32:
      return iree_string_builder_append_format(detail_builder, "%" PRIu32,
                                               value->storage.u32);
    case IREE_TOOLING_VALUE_KIND_I64:
      return iree_string_builder_append_format(detail_builder, "%" PRIi64,
                                               value->storage.i64);
    case IREE_TOOLING_VALUE_KIND_U64:
      return iree_string_builder_append_format(detail_builder, "%" PRIu64,
                                               value->storage.u64);
    case IREE_TOOLING_VALUE_KIND_F32:
      return iree_string_builder_append_format(detail_builder, "%g",
                                               (double)value->storage.f32);
    case IREE_TOOLING_VALUE_KIND_F64:
      return iree_string_builder_append_format(detail_builder, "%g",
                                               value->storage.f64);
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      return iree_string_builder_append_format(detail_builder, "0x%08" PRIx32,
                                               value->storage.u32);
    default:
      return iree_string_builder_append_cstring(detail_builder, "<unknown>");
  }
}

static iree_status_t loom_testbench_append_value_kind(
    const loom_testbench_value_t* value,
    iree_string_builder_t* detail_builder) {
  if (value == NULL || value->kind == LOOM_TESTBENCH_VALUE_KIND_NONE) {
    return iree_string_builder_append_string(detail_builder, IREE_SV("empty"));
  }
  if (loom_testbench_value_is_scalar(value)) {
    return iree_string_builder_append_format(
        detail_builder, "scalar %s",
        loom_testbench_tooling_value_kind_name(value->scalar.kind));
  }
  if (loom_testbench_value_buffer_view(value) != NULL) {
    return iree_string_builder_append_string(detail_builder,
                                             IREE_SV("buffer_view"));
  }
  if (loom_testbench_value_is_buffer(value)) {
    return iree_string_builder_append_string(detail_builder,
                                             IREE_SV("storage_buffer"));
  }
  return iree_string_builder_append_string(detail_builder, IREE_SV("unknown"));
}

static iree_status_t loom_testbench_append_value_kind_mismatch(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(detail_builder, IREE_SV("actual ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_value_kind(actual, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" cannot be compared with expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_value_kind(expected, detail_builder));
  return iree_ok_status();
}

static iree_status_t loom_testbench_compare_scalar_exact(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  const iree_tooling_value_t* actual_value = &actual->scalar;
  const iree_tooling_value_t* expected_value = &expected->scalar;
  if (actual_value->kind != expected_value->kind) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_tooling_value_kind_name(actual_value->kind),
        loom_testbench_tooling_value_kind_name(expected_value->kind));
  }
  iree_host_size_t value_size =
      loom_testbench_tooling_value_storage_size(actual_value->kind);
  if (value_size == 0) {
    return iree_string_builder_append_format(detail_builder,
                                             "unsupported scalar type %u",
                                             (unsigned)actual_value->kind);
  }
  if (memcmp(&actual_value->storage, &expected_value->storage, value_size) ==
      0) {
    *out_matched = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      detail_builder, "actual %s value ",
      loom_testbench_tooling_value_kind_name(actual_value->kind)));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_tooling_value(actual_value, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" does not match "
                              "expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_tooling_value(expected_value, detail_builder));
  return iree_ok_status();
}

static iree_status_t loom_testbench_compare_scalar_equal(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  const iree_tooling_value_t* actual_value = &actual->scalar;
  const iree_tooling_value_t* expected_value = &expected->scalar;
  if (actual_value->kind != expected_value->kind) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_tooling_value_kind_name(actual_value->kind),
        loom_testbench_tooling_value_kind_name(expected_value->kind));
  }

  switch (actual_value->kind) {
    case IREE_TOOLING_VALUE_KIND_I32:
      *out_matched = actual_value->storage.i32 == expected_value->storage.i32;
      break;
    case IREE_TOOLING_VALUE_KIND_U32:
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      *out_matched = actual_value->storage.u32 == expected_value->storage.u32;
      break;
    case IREE_TOOLING_VALUE_KIND_I64:
      *out_matched = actual_value->storage.i64 == expected_value->storage.i64;
      break;
    case IREE_TOOLING_VALUE_KIND_U64:
      *out_matched = actual_value->storage.u64 == expected_value->storage.u64;
      break;
    case IREE_TOOLING_VALUE_KIND_F32:
      *out_matched = actual_value->storage.f32 == expected_value->storage.f32;
      break;
    case IREE_TOOLING_VALUE_KIND_F64:
      *out_matched = actual_value->storage.f64 == expected_value->storage.f64;
      break;
    default:
      return iree_string_builder_append_format(detail_builder,
                                               "unsupported scalar type %u",
                                               (unsigned)actual_value->kind);
  }
  if (*out_matched) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      detail_builder, "actual %s value ",
      loom_testbench_tooling_value_kind_name(actual_value->kind)));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_tooling_value(actual_value, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" does not equal expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_tooling_value(expected_value, detail_builder));
  return iree_ok_status();
}

static bool loom_testbench_f64_close(double actual, double expected,
                                     double absolute_tolerance,
                                     double relative_tolerance,
                                     loom_check_expect_close_nan_t nan_policy) {
  const bool actual_is_nan = isnan(actual);
  const bool expected_is_nan = isnan(expected);
  if (actual_is_nan || expected_is_nan) {
    return nan_policy == LOOM_CHECK_EXPECT_CLOSE_NAN_SAME && actual_is_nan &&
           expected_is_nan;
  }
  return fabs(actual - expected) <=
         absolute_tolerance + relative_tolerance * fabs(expected);
}

static iree_status_t loom_testbench_compare_scalar_close(
    const loom_testbench_close_expectation_plan_t* close_plan,
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  const iree_tooling_value_t* actual_value = &actual->scalar;
  const iree_tooling_value_t* expected_value = &expected->scalar;
  if (actual_value->kind != expected_value->kind) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_tooling_value_kind_name(actual_value->kind),
        loom_testbench_tooling_value_kind_name(expected_value->kind));
  }

  double actual_f64 = 0.0;
  double expected_f64 = 0.0;
  switch (actual_value->kind) {
    case IREE_TOOLING_VALUE_KIND_F32:
      actual_f64 = (double)actual_value->storage.f32;
      expected_f64 = (double)expected_value->storage.f32;
      break;
    case IREE_TOOLING_VALUE_KIND_F64:
      actual_f64 = actual_value->storage.f64;
      expected_f64 = expected_value->storage.f64;
      break;
    default:
      return iree_string_builder_append_format(
          detail_builder,
          "close expectation requires f32/f64 scalar values, "
          "but actual is %s",
          loom_testbench_tooling_value_kind_name(actual_value->kind));
  }

  *out_matched = loom_testbench_f64_close(
      actual_f64, expected_f64, close_plan->absolute_tolerance,
      close_plan->relative_tolerance, close_plan->nan_policy);
  if (*out_matched) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      detail_builder,
      "actual %s value %.17g is not close to expected %.17g "
      "(atol=%.17g, rtol=%.17g)",
      loom_testbench_tooling_value_kind_name(actual_value->kind), actual_f64,
      expected_f64, close_plan->absolute_tolerance,
      close_plan->relative_tolerance);
}

static void loom_testbench_expectation_buffer_view(
    const loom_testbench_value_t* value,
    iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;
  *out_buffer_view = loom_testbench_value_buffer_view(value);
}

static iree_status_t loom_testbench_compare_buffer_exact(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  iree_hal_buffer_view_t* expected_buffer_view = NULL;
  loom_testbench_expectation_buffer_view(actual, &actual_buffer_view);
  loom_testbench_expectation_buffer_view(expected, &expected_buffer_view);
  if (!actual_buffer_view || !expected_buffer_view) {
    return loom_testbench_append_value_kind_mismatch(actual, expected,
                                                     detail_builder);
  }
  iree_hal_buffer_equality_t equality = {
      .mode = IREE_HAL_BUFFER_EQUALITY_EXACT,
  };
  return iree_hal_buffer_view_match_equal(equality, expected_buffer_view,
                                          actual_buffer_view, detail_builder,
                                          out_matched);
}

static double loom_testbench_dense_floating_element_as_f64(
    iree_hal_element_type_t element_type, const uint8_t* element_data) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_f16_to_f32(bits);
    }
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_bf16_to_f32(bits);
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: {
      float value = 0.0f;
      memcpy(&value, element_data, sizeof(value));
      return (double)value;
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64: {
      double value = 0.0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    default:
      return NAN;
  }
}

static bool loom_testbench_hal_element_type_is_close_supported(
    iree_hal_element_type_t element_type) {
  return element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 ||
         element_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 ||
         element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
         element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_64;
}

static iree_status_t loom_testbench_compare_buffer_close(
    const loom_testbench_close_expectation_plan_t* close_plan,
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  iree_hal_buffer_view_t* expected_buffer_view = NULL;
  loom_testbench_expectation_buffer_view(actual, &actual_buffer_view);
  loom_testbench_expectation_buffer_view(expected, &expected_buffer_view);
  if (!actual_buffer_view || !expected_buffer_view) {
    return loom_testbench_append_value_kind_mismatch(actual, expected,
                                                     detail_builder);
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_match_metadata_like(
      expected_buffer_view, actual_buffer_view, detail_builder, out_matched));
  if (!*out_matched) {
    return iree_ok_status();
  }
  *out_matched = false;

  if (iree_hal_buffer_view_encoding_type(actual_buffer_view) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "close expectation requires dense row-major "
                            "buffer views");
  }

  iree_hal_element_type_t element_type =
      iree_hal_buffer_view_element_type(actual_buffer_view);
  if (!loom_testbench_hal_element_type_is_close_supported(element_type)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        detail_builder, IREE_SV("close expectation requires floating-point "
                                "buffer elements, but actual has ")));
    IREE_RETURN_IF_ERROR(
        iree_hal_append_element_type_string(element_type, detail_builder));
    return iree_ok_status();
  }

  iree_hal_buffer_mapping_t actual_mapping = {0};
  iree_hal_buffer_mapping_t expected_mapping = {0};
  bool actual_mapped = false;
  bool expected_mapped = false;
  iree_status_t status = iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(actual_buffer_view),
      IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
      IREE_HAL_WHOLE_BUFFER, &actual_mapping);
  if (iree_status_is_ok(status)) {
    actual_mapped = true;
    status = iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(expected_buffer_view),
        IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
        IREE_HAL_WHOLE_BUFFER, &expected_mapping);
  }
  if (iree_status_is_ok(status)) {
    expected_mapped = true;
    iree_host_size_t element_count =
        iree_hal_buffer_view_element_count(actual_buffer_view);
    iree_host_size_t element_size =
        iree_hal_element_dense_byte_count(element_type);
    const uint8_t* actual_data = actual_mapping.contents.data;
    const uint8_t* expected_data = expected_mapping.contents.data;
    *out_matched = true;
    for (iree_host_size_t i = 0; i < element_count; ++i) {
      double actual_value = loom_testbench_dense_floating_element_as_f64(
          element_type, actual_data + i * element_size);
      double expected_value = loom_testbench_dense_floating_element_as_f64(
          element_type, expected_data + i * element_size);
      if (!loom_testbench_f64_close(
              actual_value, expected_value, close_plan->absolute_tolerance,
              close_plan->relative_tolerance, close_plan->nan_policy)) {
        *out_matched = false;
        status = iree_string_builder_append_format(
            detail_builder,
            "element at index %" PRIhsz
            " (%.17g) is not close to expected (%.17g) "
            "(atol=%.17g, rtol=%.17g)",
            i, actual_value, expected_value, close_plan->absolute_tolerance,
            close_plan->relative_tolerance);
        break;
      }
    }
  }
  if (expected_mapped) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&expected_mapping));
  }
  if (actual_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&actual_mapping));
  }
  return status;
}

static iree_status_t loom_testbench_compare_equal(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (loom_testbench_value_is_scalar(actual) &&
      loom_testbench_value_is_scalar(expected)) {
    return loom_testbench_compare_scalar_equal(actual, expected, detail_builder,
                                               out_matched);
  }

  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  loom_testbench_expectation_buffer_view(actual, &actual_buffer_view);
  if (actual_buffer_view &&
      loom_testbench_hal_element_type_is_close_supported(
          iree_hal_buffer_view_element_type(actual_buffer_view))) {
    loom_testbench_close_expectation_plan_t exact_float = {
        .absolute_tolerance = 0.0,
        .relative_tolerance = 0.0,
        .nan_policy = LOOM_CHECK_EXPECT_CLOSE_NAN_DIFFERENT,
    };
    return loom_testbench_compare_buffer_close(&exact_float, actual, expected,
                                               detail_builder, out_matched);
  }

  return loom_testbench_compare_buffer_exact(actual, expected, detail_builder,
                                             out_matched);
}

static iree_status_t loom_testbench_compare_bitwise(
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (loom_testbench_value_is_scalar(actual) &&
      loom_testbench_value_is_scalar(expected)) {
    return loom_testbench_compare_scalar_exact(actual, expected, detail_builder,
                                               out_matched);
  }
  return loom_testbench_compare_buffer_exact(actual, expected, detail_builder,
                                             out_matched);
}

static iree_status_t loom_testbench_compare_close(
    const loom_testbench_close_expectation_plan_t* close_plan,
    const loom_testbench_value_t* actual,
    const loom_testbench_value_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (loom_testbench_value_is_scalar(actual) &&
      loom_testbench_value_is_scalar(expected)) {
    return loom_testbench_compare_scalar_close(close_plan, actual, expected,
                                               detail_builder, out_matched);
  }
  return loom_testbench_compare_buffer_close(close_plan, actual, expected,
                                             detail_builder, out_matched);
}

static iree_status_t loom_testbench_value_as_nonnegative_dim(
    const loom_testbench_value_t* value, iree_hal_dim_t* out_dim) {
  int64_t signed_value = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &signed_value));
  if (signed_value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dynamic dimension is negative");
  }
  if ((uint64_t)signed_value > (uint64_t)IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dynamic dimension exceeds device size range");
  }
  *out_dim = (iree_hal_dim_t)signed_value;
  return iree_ok_status();
}

static iree_status_t loom_testbench_shape_expectation_dimension(
    const loom_testbench_shape_expectation_plan_t* shape_plan,
    const loom_testbench_value_table_t* table, iree_host_size_t dim_index,
    iree_host_size_t* inout_dynamic_index, iree_hal_dim_t* out_dim) {
  int64_t static_dimension = shape_plan->static_dimensions[dim_index];
  if (static_dimension != INT64_MIN) {
    if (static_dimension < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "static shape dimension is negative");
    }
    if ((uint64_t)static_dimension > (uint64_t)IREE_DEVICE_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "static shape dimension exceeds device size "
                              "range");
    }
    *out_dim = (iree_hal_dim_t)static_dimension;
    return iree_ok_status();
  }

  if (*inout_dynamic_index >= shape_plan->dimension_value_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "shape plan has too few dynamic dimensions");
  }
  loom_value_id_t value_id =
      shape_plan->dimension_value_ids[(*inout_dynamic_index)++];
  const loom_testbench_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_value_table_lookup_borrow(table, value_id, &value));
  return loom_testbench_value_as_nonnegative_dim(value, out_dim);
}

static iree_status_t loom_testbench_compare_shape(
    const loom_testbench_expectation_plan_t* expectation,
    const loom_testbench_value_table_t* table,
    const loom_testbench_value_t* actual, iree_string_builder_t* detail_builder,
    bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  loom_testbench_expectation_buffer_view(actual, &actual_buffer_view);
  if (!actual_buffer_view) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(detail_builder, IREE_SV("actual ")));
    IREE_RETURN_IF_ERROR(
        loom_testbench_append_value_kind(actual, detail_builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        detail_builder, IREE_SV(" has no buffer-view shape")));
    return iree_ok_status();
  }

  const loom_testbench_shape_expectation_plan_t* shape_plan =
      &expectation->shape;
  if (shape_plan->static_dimension_count !=
      iree_hal_buffer_view_shape_rank(actual_buffer_view)) {
    return iree_string_builder_append_format(
        detail_builder,
        "actual rank %" PRIhsz " does not match expected rank %" PRIhsz,
        iree_hal_buffer_view_shape_rank(actual_buffer_view),
        shape_plan->static_dimension_count);
  }

  iree_host_size_t dynamic_index = 0;
  for (iree_host_size_t dim_index = 0;
       dim_index < shape_plan->static_dimension_count; ++dim_index) {
    iree_hal_dim_t expected_dim = 0;
    IREE_RETURN_IF_ERROR(loom_testbench_shape_expectation_dimension(
        shape_plan, table, dim_index, &dynamic_index, &expected_dim));
    iree_hal_dim_t actual_dim =
        iree_hal_buffer_view_shape_dim(actual_buffer_view, dim_index);
    if (actual_dim != expected_dim) {
      return iree_string_builder_append_format(
          detail_builder,
          "actual dimension %" PRIhsz " value %" PRIdim
          " does not match expected %" PRIdim,
          dim_index, actual_dim, expected_dim);
    }
  }
  *out_matched = true;
  return iree_ok_status();
}

static iree_status_t loom_testbench_lookup_expectation_values(
    const loom_testbench_expectation_plan_t* expectation,
    const loom_testbench_value_table_t* table,
    const loom_testbench_value_t** out_actual,
    const loom_testbench_value_t** out_expected) {
  *out_actual = NULL;
  *out_expected = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
      table, expectation->actual_value_id, out_actual));
  if (expectation->expected_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_testbench_value_table_lookup_borrow(
      table, expectation->expected_value_id, out_expected);
}

static iree_string_view_t loom_testbench_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static const loom_named_attr_t* loom_testbench_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_testbench_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

static bool loom_testbench_attr_name_is_in_set(const loom_module_t* module,
                                               const loom_named_attr_t* attr,
                                               const iree_string_view_t* names,
                                               iree_host_size_t name_count) {
  const iree_string_view_t attr_name =
      loom_testbench_module_string(module, attr->name_id);
  for (iree_host_size_t i = 0; i < name_count; ++i) {
    if (iree_string_view_equal(attr_name, names[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_testbench_validate_event_attrs(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    const iree_string_view_t* names, iree_host_size_t name_count,
    iree_string_view_t context) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (!loom_testbench_attr_name_is_in_set(module, &attrs.entries[i], names,
                                            name_count)) {
      const iree_string_view_t attr_name =
          loom_testbench_module_string(module, attrs.entries[i].name_id);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s event expectation has unsupported "
                              "attribute `%.*s`",
                              (int)context.size, context.data,
                              (int)attr_name.size, attr_name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_optional_i64(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, int64_t* out_value) {
  *out_present = false;
  *out_value = 0;
  const loom_named_attr_t* attr =
      loom_testbench_find_named_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation attribute `%.*s` must be i64",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_optional_bool(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, bool* out_value) {
  *out_present = false;
  *out_value = false;
  const loom_named_attr_t* attr =
      loom_testbench_find_named_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_BOOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation attribute `%.*s` must be bool",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value = loom_attr_as_bool(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_optional_string(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, iree_string_view_t* out_value) {
  *out_present = false;
  *out_value = iree_string_view_empty();
  const loom_named_attr_t* attr =
      loom_testbench_find_named_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation attribute `%.*s` must be string",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value =
      loom_testbench_module_string(module, loom_attr_as_string_id(attr->value));
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_optional_dict(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present,
    loom_named_attr_slice_t* out_value) {
  *out_present = false;
  *out_value = loom_named_attr_slice_empty();
  const loom_named_attr_t* attr =
      loom_testbench_find_named_attr(module, attrs, name);
  if (attr == NULL) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation attribute `%.*s` must be dict",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value = loom_attr_as_dict(attr->value);
  return iree_ok_status();
}

static bool loom_testbench_event_type_parse(iree_string_view_t value,
                                            uint32_t* out_type) {
  if (iree_string_view_equal(value, IREE_SV("driver_failure"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("asan_report"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("ubsan_report"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("tsan_report"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("printf"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_PRINTF;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("host_call"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_HOST_CALL;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_type = IREE_HAL_DEVICE_EVENT_TYPE_NONE;
    return true;
  }
  return false;
}

static bool loom_testbench_event_severity_parse(iree_string_view_t value,
                                                uint32_t* out_severity) {
  if (iree_string_view_equal(value, IREE_SV("trace"))) {
    *out_severity = IREE_HAL_DEVICE_EVENT_SEVERITY_TRACE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("info"))) {
    *out_severity = IREE_HAL_DEVICE_EVENT_SEVERITY_INFO;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("warning"))) {
    *out_severity = IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("error"))) {
    *out_severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("fatal"))) {
    *out_severity = IREE_HAL_DEVICE_EVENT_SEVERITY_FATAL;
    return true;
  }
  return false;
}

static bool loom_testbench_asan_access_parse(iree_string_view_t value,
                                             uint32_t* out_kind) {
  if (iree_string_view_equal(value, IREE_SV("read"))) {
    *out_kind = IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("write"))) {
    *out_kind = IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("atomic"))) {
    *out_kind = IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_kind = IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN;
    return true;
  }
  return false;
}

static bool loom_testbench_ubsan_check_parse(iree_string_view_t value,
                                             uint32_t* out_kind) {
  if (iree_string_view_equal(value, IREE_SV("integer_overflow"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_INTEGER_OVERFLOW;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("divide_by_zero"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("alignment"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ALIGNMENT;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("float_nan_contract"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unreachable"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNREACHABLE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_kind = IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN;
    return true;
  }
  return false;
}

static bool loom_testbench_tsan_check_parse(iree_string_view_t value,
                                            uint32_t* out_kind) {
  if (iree_string_view_equal(value, IREE_SV("data_race"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_CHECK_KIND_UNKNOWN;
    return true;
  }
  return false;
}

static bool loom_testbench_tsan_memory_parse(iree_string_view_t value,
                                             uint32_t* out_space) {
  if (iree_string_view_equal(value, IREE_SV("global"))) {
    *out_space = IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_GLOBAL;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("workgroup"))) {
    *out_space = IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("private"))) {
    *out_space = IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_PRIVATE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_space = IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_UNKNOWN;
    return true;
  }
  return false;
}

static bool loom_testbench_tsan_access_parse(iree_string_view_t value,
                                             uint32_t* out_kind) {
  if (iree_string_view_equal(value, IREE_SV("read"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("write"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("read_write"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_READ_WRITE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("atomic"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_kind = IREE_HAL_DEVICE_TSAN_ACCESS_KIND_UNKNOWN;
    return true;
  }
  return false;
}

static iree_status_t loom_testbench_event_optional_enum_string(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name,
    bool (*parse_fn)(iree_string_view_t value, uint32_t* out_value),
    bool* out_present, uint32_t* out_value) {
  iree_string_view_t string_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, name, out_present, &string_value));
  if (!*out_present) {
    return iree_ok_status();
  }
  if (!parse_fn(string_value, out_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation attribute `%.*s` has unknown "
                            "value `%.*s`",
                            (int)name.size, name.data, (int)string_value.size,
                            string_value.data);
  }
  return iree_ok_status();
}

static bool loom_testbench_event_match_optional_u32(uint32_t actual,
                                                    bool expected_present,
                                                    int64_t expected) {
  return !expected_present ||
         (expected >= 0 && (uint64_t)expected == (uint64_t)actual);
}

static bool loom_testbench_event_match_optional_u64(uint64_t actual,
                                                    bool expected_present,
                                                    int64_t expected) {
  return !expected_present || (expected >= 0 && (uint64_t)expected == actual);
}

static bool loom_testbench_event_match_optional_string(
    iree_string_view_t actual, bool expected_present,
    iree_string_view_t expected) {
  return !expected_present || iree_string_view_equal(actual, expected);
}

static bool loom_testbench_event_payload_has_prefix(
    const iree_hal_device_event_t* event, iree_host_size_t prefix_length) {
  return event->payload.data != NULL &&
         event->payload.data_length >= prefix_length;
}

static iree_status_t loom_testbench_event_match_asan(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    const iree_hal_device_event_t* event, bool* out_matches) {
  *out_matches = false;
  const iree_string_view_t keys[] = {
      IREE_SV("access"),
      IREE_SV("access_length"),
      IREE_SV("shadow_value"),
      IREE_SV("site_id"),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_validate_event_attrs(
      module, attrs, keys, IREE_ARRAYSIZE(keys), IREE_SV("asan")));
  if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT ||
      !loom_testbench_event_payload_has_prefix(
          event, sizeof(iree_hal_device_asan_report_t))) {
    return iree_ok_status();
  }
  iree_hal_device_asan_report_t report = {0};
  memcpy(&report, event->payload.data, sizeof(report));

  bool access_present = false;
  uint32_t access = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("access"), loom_testbench_asan_access_parse,
      &access_present, &access));
  bool access_length_present = false;
  int64_t access_length = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("access_length"), &access_length_present,
      &access_length));
  bool shadow_value_present = false;
  int64_t shadow_value = 0;
  IREE_RETURN_IF_ERROR(
      loom_testbench_event_optional_i64(module, attrs, IREE_SV("shadow_value"),
                                        &shadow_value_present, &shadow_value));
  bool site_id_present = false;
  int64_t site_id = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("site_id"), &site_id_present, &site_id));

  *out_matches =
      (!access_present || report.access_kind == access) &&
      loom_testbench_event_match_optional_u64(
          report.access_length, access_length_present, access_length) &&
      loom_testbench_event_match_optional_u64(
          report.shadow_value, shadow_value_present, shadow_value) &&
      loom_testbench_event_match_optional_u64(report.site_id, site_id_present,
                                              site_id);
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_match_ubsan(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    const iree_hal_device_event_t* event, bool* out_matches) {
  *out_matches = false;
  const iree_string_view_t keys[] = {
      IREE_SV("check"),
      IREE_SV("operand0"),
      IREE_SV("operand1"),
      IREE_SV("site_id"),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_validate_event_attrs(
      module, attrs, keys, IREE_ARRAYSIZE(keys), IREE_SV("ubsan")));
  if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT ||
      !loom_testbench_event_payload_has_prefix(
          event, sizeof(iree_hal_device_ubsan_report_t))) {
    return iree_ok_status();
  }
  iree_hal_device_ubsan_report_t report = {0};
  memcpy(&report, event->payload.data, sizeof(report));

  bool check_present = false;
  uint32_t check = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("check"), loom_testbench_ubsan_check_parse,
      &check_present, &check));
  bool operand0_present = false;
  int64_t operand0 = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("operand0"), &operand0_present, &operand0));
  bool operand1_present = false;
  int64_t operand1 = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("operand1"), &operand1_present, &operand1));
  bool site_id_present = false;
  int64_t site_id = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("site_id"), &site_id_present, &site_id));

  *out_matches = (!check_present || report.check_kind == check) &&
                 loom_testbench_event_match_optional_u64(
                     report.operand0, operand0_present, operand0) &&
                 loom_testbench_event_match_optional_u64(
                     report.operand1, operand1_present, operand1) &&
                 loom_testbench_event_match_optional_u64(
                     report.site_id, site_id_present, site_id);
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_match_tsan(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    const iree_hal_device_event_t* event, bool* out_matches) {
  *out_matches = false;
  const iree_string_view_t keys[] = {
      IREE_SV("check"),
      IREE_SV("memory"),
      IREE_SV("current_access"),
      IREE_SV("prior_access"),
      IREE_SV("access_length"),
      IREE_SV("current_site_id"),
      IREE_SV("prior_site_id"),
      IREE_SV("current_atomic"),
      IREE_SV("prior_atomic"),
      IREE_SV("current_workitem_linear"),
      IREE_SV("prior_workitem_linear"),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_validate_event_attrs(
      module, attrs, keys, IREE_ARRAYSIZE(keys), IREE_SV("tsan")));
  if (event->type != IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT ||
      !loom_testbench_event_payload_has_prefix(
          event, sizeof(iree_hal_device_tsan_report_t))) {
    return iree_ok_status();
  }
  iree_hal_device_tsan_report_t report = {0};
  memcpy(&report, event->payload.data, sizeof(report));

  bool check_present = false;
  uint32_t check = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("check"), loom_testbench_tsan_check_parse,
      &check_present, &check));
  bool memory_present = false;
  uint32_t memory = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("memory"), loom_testbench_tsan_memory_parse,
      &memory_present, &memory));
  bool current_access_present = false;
  uint32_t current_access = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("current_access"),
      loom_testbench_tsan_access_parse, &current_access_present,
      &current_access));
  bool prior_access_present = false;
  uint32_t prior_access = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("prior_access"), loom_testbench_tsan_access_parse,
      &prior_access_present, &prior_access));
  bool access_length_present = false;
  int64_t access_length = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("access_length"), &access_length_present,
      &access_length));
  bool current_site_id_present = false;
  int64_t current_site_id = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("current_site_id"), &current_site_id_present,
      &current_site_id));
  bool prior_site_id_present = false;
  int64_t prior_site_id = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("prior_site_id"), &prior_site_id_present,
      &prior_site_id));
  bool current_atomic_present = false;
  bool current_atomic = false;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_bool(
      module, attrs, IREE_SV("current_atomic"), &current_atomic_present,
      &current_atomic));
  bool prior_atomic_present = false;
  bool prior_atomic = false;
  IREE_RETURN_IF_ERROR(
      loom_testbench_event_optional_bool(module, attrs, IREE_SV("prior_atomic"),
                                         &prior_atomic_present, &prior_atomic));
  bool current_linear_present = false;
  bool current_linear = false;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_bool(
      module, attrs, IREE_SV("current_workitem_linear"),
      &current_linear_present, &current_linear));
  bool prior_linear_present = false;
  bool prior_linear = false;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_bool(
      module, attrs, IREE_SV("prior_workitem_linear"), &prior_linear_present,
      &prior_linear));

  const bool report_current_atomic = iree_all_bits_set(
      report.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_ATOMIC);
  const bool report_prior_atomic = iree_all_bits_set(
      report.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_ATOMIC);
  const bool report_current_linear = iree_all_bits_set(
      report.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR);
  const bool report_prior_linear = iree_all_bits_set(
      report.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR);

  *out_matches =
      (!check_present || report.check_kind == check) &&
      (!memory_present || report.memory_space == memory) &&
      (!current_access_present ||
       report.current_access_kind == current_access) &&
      (!prior_access_present || report.prior_access_kind == prior_access) &&
      loom_testbench_event_match_optional_u32(
          report.access_length, access_length_present, access_length) &&
      loom_testbench_event_match_optional_u64(
          report.current_site_id, current_site_id_present, current_site_id) &&
      loom_testbench_event_match_optional_u64(
          report.prior_site_id, prior_site_id_present, prior_site_id) &&
      (!current_atomic_present || report_current_atomic == current_atomic) &&
      (!prior_atomic_present || report_prior_atomic == prior_atomic) &&
      (!current_linear_present || report_current_linear == current_linear) &&
      (!prior_linear_present || report_prior_linear == prior_linear);
  return iree_ok_status();
}

static iree_status_t loom_testbench_event_record_matches(
    const loom_module_t* module,
    const loom_testbench_expectation_plan_t* expectation,
    const loom_testbench_device_event_record_t* record, bool* out_matches) {
  *out_matches = false;
  const loom_named_attr_slice_t attrs = expectation->event.attrs;
  const iree_hal_device_event_t* event = &record->event;

  bool type_present = false;
  uint32_t type = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("type"), loom_testbench_event_type_parse,
      &type_present, &type));
  bool severity_present = false;
  uint32_t severity = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_enum_string(
      module, attrs, IREE_SV("severity"), loom_testbench_event_severity_parse,
      &severity_present, &severity));
  bool driver_present = false;
  iree_string_view_t driver = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, IREE_SV("driver"), &driver_present, &driver));
  bool device_present = false;
  iree_string_view_t device = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, IREE_SV("device"), &device_present, &device));
  bool physical_device_present = false;
  int64_t physical_device = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("physical_device"), &physical_device_present,
      &physical_device));
  bool queue_present = false;
  int64_t queue = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("queue"), &queue_present, &queue));
  bool export_present = false;
  int64_t export_ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("export"), &export_present, &export_ordinal));
  bool site_id_present = false;
  int64_t site_id = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, attrs, IREE_SV("site_id"), &site_id_present, &site_id));
  bool source_file_present = false;
  iree_string_view_t source_file = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, IREE_SV("source_file"), &source_file_present,
      &source_file));
  bool function_present = false;
  iree_string_view_t function = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, IREE_SV("function"), &function_present, &function));
  bool operation_present = false;
  iree_string_view_t operation = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_string(
      module, attrs, IREE_SV("operation"), &operation_present, &operation));

  if (type_present && event->type != type) return iree_ok_status();
  if (severity_present && event->severity != severity) return iree_ok_status();
  if (!loom_testbench_event_match_optional_string(event->source.driver_id,
                                                  driver_present, driver)) {
    return iree_ok_status();
  }
  if (!loom_testbench_event_match_optional_string(event->source.device_id,
                                                  device_present, device)) {
    return iree_ok_status();
  }
  if (!loom_testbench_event_match_optional_u32(
          event->source.physical_device_ordinal, physical_device_present,
          physical_device)) {
    return iree_ok_status();
  }
  if (!loom_testbench_event_match_optional_u32(event->source.queue_ordinal,
                                               queue_present, queue)) {
    return iree_ok_status();
  }
  if (!loom_testbench_event_match_optional_u32(
          event->source.export_ordinal, export_present, export_ordinal)) {
    return iree_ok_status();
  }
  if (site_id_present && event->site == NULL) return iree_ok_status();
  if (event->site != NULL) {
    if (!loom_testbench_event_match_optional_u64(event->site->site_id,
                                                 site_id_present, site_id)) {
      return iree_ok_status();
    }
    if (!loom_testbench_event_match_optional_string(
            event->site->source_file, source_file_present, source_file)) {
      return iree_ok_status();
    }
    if (!loom_testbench_event_match_optional_string(
            event->site->function_name, function_present, function)) {
      return iree_ok_status();
    }
    if (!loom_testbench_event_match_optional_string(
            event->site->operation_name, operation_present, operation)) {
      return iree_ok_status();
    }
  } else if (source_file_present || function_present || operation_present) {
    return iree_ok_status();
  }

  bool asan_present = false;
  loom_named_attr_slice_t asan_attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_dict(
      module, attrs, IREE_SV("asan"), &asan_present, &asan_attrs));
  if (asan_present) {
    bool asan_matches = false;
    IREE_RETURN_IF_ERROR(loom_testbench_event_match_asan(module, asan_attrs,
                                                         event, &asan_matches));
    if (!asan_matches) return iree_ok_status();
  }

  bool ubsan_present = false;
  loom_named_attr_slice_t ubsan_attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_dict(
      module, attrs, IREE_SV("ubsan"), &ubsan_present, &ubsan_attrs));
  if (ubsan_present) {
    bool ubsan_matches = false;
    IREE_RETURN_IF_ERROR(loom_testbench_event_match_ubsan(
        module, ubsan_attrs, event, &ubsan_matches));
    if (!ubsan_matches) return iree_ok_status();
  }

  bool tsan_present = false;
  loom_named_attr_slice_t tsan_attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_dict(
      module, attrs, IREE_SV("tsan"), &tsan_present, &tsan_attrs));
  if (tsan_present) {
    bool tsan_matches = false;
    IREE_RETURN_IF_ERROR(loom_testbench_event_match_tsan(module, tsan_attrs,
                                                         event, &tsan_matches));
    if (!tsan_matches) return iree_ok_status();
  }

  *out_matches = true;
  return iree_ok_status();
}

static iree_status_t loom_testbench_evaluate_event_expectation(
    const loom_testbench_expectation_plan_t* expectation,
    const loom_module_t* module,
    const loom_testbench_case_sample_observations_t* observations,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  if (!iree_string_view_equal(expectation->event.provider, IREE_SV("device"))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "event expectation provider `%.*s` is not configured",
        (int)expectation->event.provider.size,
        expectation->event.provider.data);
  }

  const iree_string_view_t keys[] = {
      IREE_SV("count"),       IREE_SV("type"),     IREE_SV("severity"),
      IREE_SV("driver"),      IREE_SV("device"),   IREE_SV("physical_device"),
      IREE_SV("queue"),       IREE_SV("export"),   IREE_SV("site_id"),
      IREE_SV("source_file"), IREE_SV("function"), IREE_SV("operation"),
      IREE_SV("asan"),        IREE_SV("ubsan"),    IREE_SV("tsan"),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_validate_event_attrs(
      module, expectation->event.attrs, keys, IREE_ARRAYSIZE(keys),
      IREE_SV("device")));

  if (observations == NULL || observations->device_events == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device event expectations require captured device events");
  }
  const loom_testbench_device_event_list_t* events =
      observations->device_events;
  if (events->dropped_count != 0) {
    return iree_string_builder_append_format(
        detail_builder,
        "device event capture dropped %" PRIhsz
        " event(s), so the expectation cannot be evaluated reliably",
        events->dropped_count);
  }

  bool count_present = false;
  int64_t expected_count = 1;
  IREE_RETURN_IF_ERROR(loom_testbench_event_optional_i64(
      module, expectation->event.attrs, IREE_SV("count"), &count_present,
      &expected_count));
  if (!count_present) {
    expected_count = 1;
  }
  if (expected_count < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "event expectation count must be non-negative");
  }

  iree_host_size_t matched_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < events->count;
       ++i) {
    bool event_matches = false;
    status = loom_testbench_event_record_matches(
        module, expectation, &events->records[i], &event_matches);
    if (iree_status_is_ok(status) && event_matches) {
      ++matched_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (matched_count == (iree_host_size_t)expected_count) {
    *out_matched = true;
    return iree_ok_status();
  }

  return iree_string_builder_append_format(
      detail_builder,
      "expected %" PRIi64 " matching device event(s), observed %" PRIhsz
      " among %" PRIhsz " captured event(s)",
      expected_count, matched_count, events->count);
}

static iree_status_t loom_testbench_evaluate_single_expectation(
    const loom_testbench_prepared_expectation_t* prepared,
    const loom_testbench_value_table_t* table,
    const loom_testbench_case_sample_observations_t* observations,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  const loom_testbench_expectation_plan_t* expectation = prepared->plan;
  if (expectation->kind == LOOM_TESTBENCH_EXPECTATION_EVENT) {
    return loom_testbench_evaluate_event_expectation(
        expectation, table->module, observations, detail_builder, out_matched);
  }

  const loom_testbench_value_t* actual = NULL;
  const loom_testbench_value_t* expected = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_lookup_expectation_values(
      expectation, table, &actual, &expected));

  switch (expectation->kind) {
    case LOOM_TESTBENCH_EXPECTATION_EQUAL:
      return loom_testbench_compare_equal(actual, expected, detail_builder,
                                          out_matched);
    case LOOM_TESTBENCH_EXPECTATION_BITWISE:
      return loom_testbench_compare_bitwise(actual, expected, detail_builder,
                                            out_matched);
    case LOOM_TESTBENCH_EXPECTATION_CLOSE:
      return loom_testbench_compare_close(&expectation->close, actual, expected,
                                          detail_builder, out_matched);
    case LOOM_TESTBENCH_EXPECTATION_SHAPE:
      return loom_testbench_compare_shape(expectation, table, actual,
                                          detail_builder, out_matched);
    case LOOM_TESTBENCH_EXPECTATION_CUSTOM:
      return prepared->custom_evaluate.fn(prepared->custom_evaluate.user_data,
                                          expectation, actual, expected,
                                          detail_builder, out_matched);
    case LOOM_TESTBENCH_EXPECTATION_EVENT:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "event expectation should not reach value "
                              "comparison path");
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invalid expectation plan kind %u",
                              (unsigned)expectation->kind);
  }
}

static iree_status_t loom_testbench_append_expectation_failure(
    loom_testbench_expectation_report_t* report,
    iree_host_size_t expectation_index,
    const loom_testbench_expectation_plan_t* expectation,
    iree_string_view_t detail) {
  IREE_ASSERT(report->failure_count < report->failure_capacity);
  loom_testbench_expectation_failure_t* failure =
      &report->failures[report->failure_count++];
  failure->expectation_index = expectation_index;
  failure->expectation = expectation;
  failure->kind = expectation->kind;
  failure->actual_value_id = expectation->actual_value_id;
  failure->expected_value_id = expectation->expected_value_id;
  failure->detail_offset = iree_string_builder_size(&report->detail_builder);
  if (iree_string_view_is_empty(detail)) {
    detail = IREE_SV("expectation did not match");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&report->detail_builder, detail));
  failure->detail_length = iree_string_builder_size(&report->detail_builder) -
                           failure->detail_offset;
  return iree_ok_status();
}

iree_status_t loom_testbench_evaluate_case_expectations(
    const loom_testbench_expectation_schedule_t* schedule,
    const loom_testbench_value_table_t* table,
    const loom_testbench_case_sample_observations_t* observations,
    loom_testbench_expectation_report_t* report) {
  if (report->failure_capacity < schedule->expectation_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expectation report capacity %" PRIhsz
        " is smaller than schedule expectation count %" PRIhsz,
        report->failure_capacity, schedule->expectation_count);
  }

  loom_testbench_expectation_report_reset(report);
  report->expectation_count = schedule->expectation_count;
  iree_string_builder_t detail_builder;
  iree_string_builder_initialize(report->host_allocator, &detail_builder);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t expectation_index = 0;
       iree_status_is_ok(status) &&
       expectation_index < schedule->expectation_count;
       ++expectation_index) {
    const loom_testbench_prepared_expectation_t* prepared =
        &schedule->expectations[expectation_index];
    bool matched = false;
    iree_string_builder_reset(&detail_builder);
    status = loom_testbench_evaluate_single_expectation(
        prepared, table, observations, &detail_builder, &matched);
    if (iree_status_is_ok(status) && matched) {
      ++report->passed_count;
    } else if (iree_status_is_ok(status)) {
      status = loom_testbench_append_expectation_failure(
          report, expectation_index, prepared->plan,
          iree_string_builder_view(&detail_builder));
    }
  }

  iree_string_builder_deinitialize(&detail_builder);
  return status;
}

static iree_status_t loom_testbench_write_expectation_value_id_json(
    loom_value_id_t value_id, loom_output_stream_t* stream) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%u", (unsigned)value_id);
}

static iree_status_t loom_testbench_write_expectation_failure_json(
    const loom_testbench_expectation_report_t* report,
    const loom_testbench_expectation_failure_t* failure,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("index"), failure->expectation_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"),
      iree_make_cstring_view(
          loom_testbench_expectation_kind_name(failure->kind))));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("actual_value_id")));
  IREE_RETURN_IF_ERROR(loom_testbench_write_expectation_value_id_json(
      failure->actual_value_id, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("expected_value_id")));
  IREE_RETURN_IF_ERROR(loom_testbench_write_expectation_value_id_json(
      failure->expected_value_id, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("detail"),
      loom_testbench_expectation_failure_detail(report, failure)));
  return loom_json_object_end(&object);
}

iree_status_t loom_testbench_expectation_report_write_json(
    const loom_testbench_expectation_report_t* report,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("expectation_count"), report->expectation_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("passed_count"), report->passed_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("failure_count"), report->failure_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("failures")));
  loom_json_array_writer_t failures;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &failures));
  for (iree_host_size_t i = 0; i < report->failure_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&failures));
    IREE_RETURN_IF_ERROR(loom_testbench_write_expectation_failure_json(
        report, &report->failures[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&failures));
  return loom_json_object_end(&object);
}
