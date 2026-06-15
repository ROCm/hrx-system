// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/expectation.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/modules/hal/types.h"
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

static const char* loom_testbench_vm_value_type_name(
    iree_vm_value_type_t type) {
  switch (type) {
    case IREE_VM_VALUE_TYPE_I8:
      return "i8";
    case IREE_VM_VALUE_TYPE_I16:
      return "i16";
    case IREE_VM_VALUE_TYPE_I32:
      return "i32";
    case IREE_VM_VALUE_TYPE_I64:
      return "i64";
    case IREE_VM_VALUE_TYPE_F32:
      return "f32";
    case IREE_VM_VALUE_TYPE_F64:
      return "f64";
    default:
      return "unknown";
  }
}

static iree_host_size_t loom_testbench_vm_value_storage_size(
    iree_vm_value_type_t type) {
  switch (type) {
    case IREE_VM_VALUE_TYPE_I8:
      return sizeof(int8_t);
    case IREE_VM_VALUE_TYPE_I16:
      return sizeof(int16_t);
    case IREE_VM_VALUE_TYPE_I32:
    case IREE_VM_VALUE_TYPE_F32:
      return sizeof(int32_t);
    case IREE_VM_VALUE_TYPE_I64:
    case IREE_VM_VALUE_TYPE_F64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

static iree_status_t loom_testbench_append_vm_value(
    iree_vm_value_t value, iree_string_builder_t* detail_builder) {
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      return iree_string_builder_append_format(detail_builder, "%" PRIi8,
                                               value.i8);
    case IREE_VM_VALUE_TYPE_I16:
      return iree_string_builder_append_format(detail_builder, "%" PRIi16,
                                               value.i16);
    case IREE_VM_VALUE_TYPE_I32:
      return iree_string_builder_append_format(detail_builder, "%" PRIi32,
                                               value.i32);
    case IREE_VM_VALUE_TYPE_I64:
      return iree_string_builder_append_format(detail_builder, "%" PRIi64,
                                               value.i64);
    case IREE_VM_VALUE_TYPE_F32:
      return iree_string_builder_append_format(detail_builder, "%g",
                                               (double)value.f32);
    case IREE_VM_VALUE_TYPE_F64:
      return iree_string_builder_append_format(detail_builder, "%g", value.f64);
    default:
      return iree_string_builder_append_cstring(detail_builder, "<unknown>");
  }
}

static iree_status_t loom_testbench_append_variant_kind(
    const iree_vm_variant_t* variant, iree_string_builder_t* detail_builder) {
  if (iree_vm_variant_is_empty(*variant)) {
    return iree_string_builder_append_string(detail_builder, IREE_SV("empty"));
  }
  if (iree_vm_variant_is_value(*variant)) {
    iree_vm_value_t value = iree_vm_variant_value(*variant);
    return iree_string_builder_append_format(
        detail_builder, "scalar %s",
        loom_testbench_vm_value_type_name(value.type));
  }
  if (iree_vm_variant_is_ref(*variant) &&
      iree_hal_buffer_view_isa(variant->ref)) {
    return iree_string_builder_append_string(detail_builder,
                                             IREE_SV("buffer_view"));
  }
  if (iree_vm_variant_is_ref(*variant)) {
    iree_string_view_t type_name = iree_vm_ref_type_name(variant->ref.type);
    return iree_string_builder_append_format(
        detail_builder, "ref %.*s", (int)type_name.size, type_name.data);
  }
  return iree_string_builder_append_string(detail_builder, IREE_SV("unknown"));
}

static iree_status_t loom_testbench_append_variant_kind_mismatch(
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(detail_builder, IREE_SV("actual ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_variant_kind(actual, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" cannot be compared with expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_variant_kind(expected, detail_builder));
  return iree_ok_status();
}

static iree_status_t loom_testbench_compare_scalar_exact(
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_vm_value_t actual_value = iree_vm_variant_value(*actual);
  iree_vm_value_t expected_value = iree_vm_variant_value(*expected);
  if (actual_value.type != expected_value.type) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_vm_value_type_name(actual_value.type),
        loom_testbench_vm_value_type_name(expected_value.type));
  }
  iree_host_size_t value_size =
      loom_testbench_vm_value_storage_size(actual_value.type);
  if (value_size == 0) {
    return iree_string_builder_append_format(detail_builder,
                                             "unsupported scalar type %u",
                                             (unsigned)actual_value.type);
  }
  if (memcmp(actual_value.value_storage, expected_value.value_storage,
             value_size) == 0) {
    *out_matched = true;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      detail_builder, "actual %s value ",
      loom_testbench_vm_value_type_name(actual_value.type)));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_vm_value(actual_value, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" does not match "
                              "expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_vm_value(expected_value, detail_builder));
  return iree_ok_status();
}

static iree_status_t loom_testbench_compare_scalar_equal(
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_vm_value_t actual_value = iree_vm_variant_value(*actual);
  iree_vm_value_t expected_value = iree_vm_variant_value(*expected);
  if (actual_value.type != expected_value.type) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_vm_value_type_name(actual_value.type),
        loom_testbench_vm_value_type_name(expected_value.type));
  }

  switch (actual_value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      *out_matched = actual_value.i8 == expected_value.i8;
      break;
    case IREE_VM_VALUE_TYPE_I16:
      *out_matched = actual_value.i16 == expected_value.i16;
      break;
    case IREE_VM_VALUE_TYPE_I32:
      *out_matched = actual_value.i32 == expected_value.i32;
      break;
    case IREE_VM_VALUE_TYPE_I64:
      *out_matched = actual_value.i64 == expected_value.i64;
      break;
    case IREE_VM_VALUE_TYPE_F32:
      *out_matched = actual_value.f32 == expected_value.f32;
      break;
    case IREE_VM_VALUE_TYPE_F64:
      *out_matched = actual_value.f64 == expected_value.f64;
      break;
    default:
      return iree_string_builder_append_format(detail_builder,
                                               "unsupported scalar type %u",
                                               (unsigned)actual_value.type);
  }
  if (*out_matched) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      detail_builder, "actual %s value ",
      loom_testbench_vm_value_type_name(actual_value.type)));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_vm_value(actual_value, detail_builder));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      detail_builder, IREE_SV(" does not equal expected ")));
  IREE_RETURN_IF_ERROR(
      loom_testbench_append_vm_value(expected_value, detail_builder));
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
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_vm_value_t actual_value = iree_vm_variant_value(*actual);
  iree_vm_value_t expected_value = iree_vm_variant_value(*expected);
  if (actual_value.type != expected_value.type) {
    return iree_string_builder_append_format(
        detail_builder, "actual scalar type %s does not match expected %s",
        loom_testbench_vm_value_type_name(actual_value.type),
        loom_testbench_vm_value_type_name(expected_value.type));
  }

  double actual_f64 = 0.0;
  double expected_f64 = 0.0;
  switch (actual_value.type) {
    case IREE_VM_VALUE_TYPE_F32:
      actual_f64 = (double)actual_value.f32;
      expected_f64 = (double)expected_value.f32;
      break;
    case IREE_VM_VALUE_TYPE_F64:
      actual_f64 = actual_value.f64;
      expected_f64 = expected_value.f64;
      break;
    default:
      return iree_string_builder_append_format(
          detail_builder,
          "close expectation requires f32/f64 scalar values, "
          "but actual is %s",
          loom_testbench_vm_value_type_name(actual_value.type));
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
      loom_testbench_vm_value_type_name(actual_value.type), actual_f64,
      expected_f64, close_plan->absolute_tolerance,
      close_plan->relative_tolerance);
}

static void loom_testbench_variant_buffer_view(
    const iree_vm_variant_t* variant,
    iree_hal_buffer_view_t** out_buffer_view) {
  *out_buffer_view = NULL;
  if (!iree_vm_variant_is_ref(*variant) ||
      !iree_hal_buffer_view_isa(variant->ref)) {
    return;
  }
  *out_buffer_view = iree_hal_buffer_view_deref(variant->ref);
}

static iree_status_t loom_testbench_compare_buffer_exact(
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  iree_hal_buffer_view_t* expected_buffer_view = NULL;
  loom_testbench_variant_buffer_view(actual, &actual_buffer_view);
  loom_testbench_variant_buffer_view(expected, &expected_buffer_view);
  if (!actual_buffer_view || !expected_buffer_view) {
    return loom_testbench_append_variant_kind_mismatch(actual, expected,
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
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  iree_hal_buffer_view_t* expected_buffer_view = NULL;
  loom_testbench_variant_buffer_view(actual, &actual_buffer_view);
  loom_testbench_variant_buffer_view(expected, &expected_buffer_view);
  if (!actual_buffer_view || !expected_buffer_view) {
    return loom_testbench_append_variant_kind_mismatch(actual, expected,
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
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (iree_vm_variant_is_value(*actual) &&
      iree_vm_variant_is_value(*expected)) {
    return loom_testbench_compare_scalar_equal(actual, expected, detail_builder,
                                               out_matched);
  }

  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  loom_testbench_variant_buffer_view(actual, &actual_buffer_view);
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
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (iree_vm_variant_is_value(*actual) &&
      iree_vm_variant_is_value(*expected)) {
    return loom_testbench_compare_scalar_exact(actual, expected, detail_builder,
                                               out_matched);
  }
  return loom_testbench_compare_buffer_exact(actual, expected, detail_builder,
                                             out_matched);
}

static iree_status_t loom_testbench_compare_close(
    const loom_testbench_close_expectation_plan_t* close_plan,
    const iree_vm_variant_t* actual, const iree_vm_variant_t* expected,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  if (iree_vm_variant_is_value(*actual) &&
      iree_vm_variant_is_value(*expected)) {
    return loom_testbench_compare_scalar_close(close_plan, actual, expected,
                                               detail_builder, out_matched);
  }
  return loom_testbench_compare_buffer_close(close_plan, actual, expected,
                                             detail_builder, out_matched);
}

static iree_status_t loom_testbench_variant_as_nonnegative_dim(
    iree_vm_variant_t variant, iree_hal_dim_t* out_dim) {
  if (!iree_vm_variant_is_value(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dynamic dimension is not a scalar value");
  }
  iree_vm_value_t value = iree_vm_variant_value(variant);
  int64_t signed_value = 0;
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      signed_value = value.i8;
      break;
    case IREE_VM_VALUE_TYPE_I16:
      signed_value = value.i16;
      break;
    case IREE_VM_VALUE_TYPE_I32:
      signed_value = value.i32;
      break;
    case IREE_VM_VALUE_TYPE_I64:
      signed_value = value.i64;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dynamic dimension is not an integer value");
  }
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
  const iree_vm_variant_t* variant = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_value_table_lookup_borrow(table, value_id, &variant));
  return loom_testbench_variant_as_nonnegative_dim(*variant, out_dim);
}

static iree_status_t loom_testbench_compare_shape(
    const loom_testbench_expectation_plan_t* expectation,
    const loom_testbench_value_table_t* table, const iree_vm_variant_t* actual,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  *out_matched = false;
  iree_hal_buffer_view_t* actual_buffer_view = NULL;
  loom_testbench_variant_buffer_view(actual, &actual_buffer_view);
  if (!actual_buffer_view) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(detail_builder, IREE_SV("actual ")));
    IREE_RETURN_IF_ERROR(
        loom_testbench_append_variant_kind(actual, detail_builder));
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
    const iree_vm_variant_t** out_actual,
    const iree_vm_variant_t** out_expected) {
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

static iree_status_t loom_testbench_evaluate_single_expectation(
    const loom_testbench_prepared_expectation_t* prepared,
    const loom_testbench_value_table_t* table,
    iree_string_builder_t* detail_builder, bool* out_matched) {
  const loom_testbench_expectation_plan_t* expectation = prepared->plan;
  const iree_vm_variant_t* actual = NULL;
  const iree_vm_variant_t* expected = NULL;
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
        prepared, table, &detail_builder, &matched);
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"index\":%" PRIhsz, failure->expectation_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_testbench_expectation_kind_name(failure->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_value_id\":%u", (unsigned)failure->actual_value_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"expected_value_id\":"));
  IREE_RETURN_IF_ERROR(loom_testbench_write_expectation_value_id_json(
      failure->expected_value_id, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"detail\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_testbench_expectation_failure_detail(report, failure)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

iree_status_t loom_testbench_expectation_report_write_json(
    const loom_testbench_expectation_report_t* report,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"expectation_count\":%" PRIhsz, report->expectation_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"passed_count\":%" PRIhsz, report->passed_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"failure_count\":%" PRIhsz, report->failure_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"failures\":["));
  for (iree_host_size_t i = 0; i < report->failure_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_testbench_write_expectation_failure_json(
        report, &report->failures[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}
