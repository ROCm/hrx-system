// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_metadata_hsaco.h"

#include <stdint.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"

//===----------------------------------------------------------------------===//
// HSACO Metadata Load Planning
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_hsaco_kernel_load_plan_t {
  // Number of reflected HAL-visible parameter records.
  iree_host_size_t parameter_count;
  // Number of HAL binding pointers consumed by the native layout.
  iree_host_size_t binding_count;
  // Number of HAL dispatch-constant spans consumed by the native layout.
  iree_host_size_t constant_span_count;
  // Number of bytes consumed from the HAL dispatch constant stream.
  iree_host_size_t constant_byte_length;
  // Total native kernarg byte length reserved for dispatch.
  iree_host_size_t kernarg_byte_length;
  // Native byte offset of the implicit-argument suffix, or
  // IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE.
  iree_host_size_t implicit_args_byte_offset;
  // Byte length required for the immutable native kernarg layout record.
  iree_host_size_t layout_byte_length;
} iree_hal_amdgpu_hsaco_kernel_load_plan_t;

typedef struct iree_hal_amdgpu_hsaco_loaded_code_object_rebase_t {
  // Source ELF bytes used by the HSACO metadata parser.
  iree_const_byte_span_t source_code_object_data;
  // HSA-loader-owned loaded code-object bytes retained by the executable.
  iree_const_byte_span_t loaded_code_object_data;
} iree_hal_amdgpu_hsaco_loaded_code_object_rebase_t;

static iree_status_t
iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
    const iree_hal_amdgpu_hsaco_loaded_code_object_rebase_t* rebase,
    const char* field_name, iree_string_view_t source_view,
    iree_string_view_t* out_view) {
  IREE_ASSERT_ARGUMENT(rebase);
  IREE_ASSERT_ARGUMENT(field_name);
  IREE_ASSERT_ARGUMENT(out_view);

  *out_view = iree_string_view_empty();
  if (source_view.size == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!source_view.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata %s string has null storage with non-zero length",
        field_name);
  }

  const uintptr_t source_begin =
      (uintptr_t)rebase->source_code_object_data.data;
  const uintptr_t source_pointer = (uintptr_t)source_view.data;
  if (IREE_UNLIKELY(source_pointer < source_begin)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata %s string is outside source ELF",
                            field_name);
  }
  const uintptr_t source_offset_pointer = source_pointer - source_begin;
  if (IREE_UNLIKELY(source_offset_pointer >
                    rebase->source_code_object_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata %s string is outside source ELF",
                            field_name);
  }

  const iree_host_size_t source_offset =
      (iree_host_size_t)source_offset_pointer;
  if (IREE_UNLIKELY(source_view.size >
                    rebase->source_code_object_data.data_length -
                        source_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata %s string extends past source ELF",
                            field_name);
  }
  if (IREE_UNLIKELY(
          source_offset > rebase->loaded_code_object_data.data_length ||
          source_view.size >
              rebase->loaded_code_object_data.data_length - source_offset)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata %s string source offset is outside loaded code object",
        field_name);
  }
  if (IREE_UNLIKELY(memcmp(rebase->loaded_code_object_data.data + source_offset,
                           source_view.data, source_view.size) != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU metadata %s string does not match loaded code object bytes",
        field_name);
  }

  *out_view = iree_make_string_view(
      (const char*)rebase->loaded_code_object_data.data + source_offset,
      source_view.size);
  return iree_ok_status();
}

static bool iree_hal_amdgpu_hsaco_metadata_arg_kind_is_hidden(
    iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN ||
         kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN_NONE;
}

static iree_status_t iree_hal_amdgpu_hsaco_load_plan_check_u16(
    iree_string_view_t symbol_name, iree_string_view_t field_name,
    iree_host_size_t value) {
  if (IREE_UNLIKELY(value > UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel `%.*s` %.*s %" PRIhsz
                            " exceeds uint16_t range",
                            (int)symbol_name.size, symbol_name.data,
                            (int)field_name.size, field_name.data, value);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_load_plan_analyze_kernel(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_kernel_load_plan_t* out_load_plan) {
  IREE_ASSERT_ARGUMENT(kernel);
  IREE_ASSERT_ARGUMENT(out_load_plan);
  memset(out_load_plan, 0, sizeof(*out_load_plan));
  out_load_plan->implicit_args_byte_offset =
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_IMPLICIT_ARGS_NONE;

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
      kernel->symbol_name, IREE_SV("kernarg segment size"),
      kernel->kernarg_segment_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
      kernel->symbol_name, IREE_SV("kernarg segment alignment"),
      kernel->kernarg_segment_alignment));

  iree_host_size_t explicit_kernarg_end = 0;
  iree_host_size_t hidden_args_offset = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < kernel->arg_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &kernel->args[i];
    iree_host_size_t arg_end = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(arg->offset, arg->size, &arg_end))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU kernel `%.*s` argument %" PRIhsz " range overflows",
          (int)kernel->symbol_name.size, kernel->symbol_name.data, i);
    }
    if (IREE_UNLIKELY(arg_end > kernel->kernarg_segment_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` argument %" PRIhsz " ends at %" PRIhsz
          " beyond kernarg segment size %u",
          (int)kernel->symbol_name.size, kernel->symbol_name.data, i, arg_end,
          kernel->kernarg_segment_size);
    }

    if (iree_hal_amdgpu_hsaco_metadata_arg_kind_is_hidden(arg->kind)) {
      hidden_args_offset =
          iree_min(hidden_args_offset, (iree_host_size_t)arg->offset);
      continue;
    }

    explicit_kernarg_end = iree_max(explicit_kernarg_end, arg_end);
    switch (arg->kind) {
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER:
        if (IREE_UNLIKELY(arg->size != sizeof(uint64_t))) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel `%.*s` global_buffer argument %" PRIhsz
              " has unsupported size %u",
              (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
              arg->size);
        }
        if (IREE_UNLIKELY(
                !iree_host_size_has_alignment(arg->offset, sizeof(uint64_t)))) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel `%.*s` global_buffer argument %" PRIhsz
              " offset %u is not 8-byte aligned",
              (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
              arg->offset);
        }
        ++out_load_plan->parameter_count;
        ++out_load_plan->binding_count;
        break;
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE:
        if (IREE_UNLIKELY(arg->size == 0)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel `%.*s` by_value argument %" PRIhsz
              " has zero byte size",
              (int)kernel->symbol_name.size, kernel->symbol_name.data, i);
        }
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
            kernel->symbol_name, IREE_SV("by_value argument size"), arg->size));
        IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
            kernel->symbol_name, IREE_SV("constant source offset"),
            out_load_plan->constant_byte_length));
        if (IREE_UNLIKELY(!iree_host_size_checked_add(
                out_load_plan->constant_byte_length, arg->size,
                &out_load_plan->constant_byte_length))) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AMDGPU kernel `%.*s` constant byte length overflows",
              (int)kernel->symbol_name.size, kernel->symbol_name.data);
        }
        ++out_load_plan->parameter_count;
        ++out_load_plan->constant_span_count;
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel `%.*s` argument %" PRIhsz
            " uses unsupported dispatchable value_kind `%.*s`",
            (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
            (int)arg->value_kind.size, arg->value_kind.data);
    }
  }

  if (IREE_UNLIKELY(out_load_plan->constant_byte_length > UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel `%.*s` constant byte length %" PRIhsz
                            " exceeds uint16_t range",
                            (int)kernel->symbol_name.size,
                            kernel->symbol_name.data,
                            out_load_plan->constant_byte_length);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
      kernel->symbol_name, IREE_SV("parameter count"),
      out_load_plan->parameter_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
      kernel->symbol_name, IREE_SV("binding count"),
      out_load_plan->binding_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
      kernel->symbol_name, IREE_SV("constant span count"),
      out_load_plan->constant_span_count));

  out_load_plan->kernarg_byte_length = kernel->kernarg_segment_size;
  if (hidden_args_offset != IREE_HOST_SIZE_MAX) {
    if (IREE_UNLIKELY(explicit_kernarg_end > hidden_args_offset)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` has visible arguments interleaved with hidden "
          "implicit kernargs",
          (int)kernel->symbol_name.size, kernel->symbol_name.data);
    }
    iree_host_size_t implicit_args_end = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            hidden_args_offset,
            (iree_host_size_t)IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
            &implicit_args_end))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU kernel `%.*s` implicit kernarg suffix overflows",
          (int)kernel->symbol_name.size, kernel->symbol_name.data);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_check_u16(
        kernel->symbol_name, IREE_SV("implicit kernarg suffix end"),
        implicit_args_end));
    out_load_plan->implicit_args_byte_offset = hidden_args_offset;
    out_load_plan->kernarg_byte_length =
        iree_max(out_load_plan->kernarg_byte_length, implicit_args_end);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_kernarg_layout_storage_size(
      out_load_plan->binding_count, out_load_plan->constant_span_count,
      &out_load_plan->layout_byte_length));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_load_plan_add_layout_capacity(
    iree_host_size_t layout_byte_length,
    iree_host_size_t* inout_layout_blob_byte_length) {
  iree_host_size_t aligned_offset = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_align(
          *inout_layout_blob_byte_length,
          iree_alignof(iree_hal_amdgpu_kernarg_layout_t), &aligned_offset))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata layout blob alignment overflow");
  }
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          aligned_offset, layout_byte_length, inout_layout_blob_byte_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata layout blob size overflow");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_hal_amdgpu_executable_metadata_counts_t* out_counts) {
  IREE_ASSERT_ARGUMENT(hsaco_metadata);
  IREE_ASSERT_ARGUMENT(out_counts);
  memset(out_counts, 0, sizeof(*out_counts));

  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          hsaco_metadata->kernel_count, hsaco_metadata->elf_kernel_symbol_count,
          &out_counts->export_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metadata export count overflow");
  }

  for (iree_host_size_t i = 0; i < hsaco_metadata->kernel_count; ++i) {
    iree_hal_amdgpu_hsaco_kernel_load_plan_t load_plan;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_analyze_kernel(
                             &hsaco_metadata->kernels[i], &load_plan),
                         "planning native kernarg layout for kernel `%.*s`",
                         (int)hsaco_metadata->kernels[i].symbol_name.size,
                         hsaco_metadata->kernels[i].symbol_name.data);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            out_counts->parameter_count, load_plan.parameter_count,
            &out_counts->parameter_count))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU metadata parameter count overflow");
    }
    if (IREE_UNLIKELY(out_counts->parameter_count > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU metadata parameter count exceeds reflection offset range");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_load_plan_add_layout_capacity(
        load_plan.layout_byte_length, &out_counts->layout_blob_byte_length));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_load_plan_populate_layout_tables(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    const iree_hal_amdgpu_hsaco_kernel_load_plan_t* load_plan,
    iree_hal_amdgpu_kernarg_layout_t* layout,
    iree_host_size_t layout_storage_capacity) {
  iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots = layout->binding_slots;
  iree_hal_amdgpu_kernarg_constant_span_t* constant_spans =
      (iree_hal_amdgpu_kernarg_constant_span_t*)(binding_slots +
                                                 load_plan->binding_count);

  iree_host_size_t binding_ordinal = 0;
  iree_host_size_t constant_span_ordinal = 0;
  iree_host_size_t constant_source_offset = 0;
  for (iree_host_size_t i = 0; i < kernel->arg_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &kernel->args[i];
    if (iree_hal_amdgpu_hsaco_metadata_arg_kind_is_hidden(arg->kind)) {
      continue;
    }
    switch (arg->kind) {
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER:
        binding_slots[binding_ordinal++] =
            (iree_hal_amdgpu_kernarg_binding_slot_t){
                .target_qword_index =
                    (uint16_t)(arg->offset / sizeof(uint64_t)),
            };
        break;
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE:
        constant_spans[constant_span_ordinal++] =
            (iree_hal_amdgpu_kernarg_constant_span_t){
                .target_byte_offset = (uint16_t)arg->offset,
                .source_byte_offset = (uint16_t)constant_source_offset,
                .byte_length = (uint16_t)arg->size,
            };
        constant_source_offset += arg->size;
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel `%.*s` argument %" PRIhsz
            " uses unsupported dispatchable value_kind `%.*s`",
            (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
            (int)arg->value_kind.size, arg->value_kind.data);
    }
  }

  const iree_hal_amdgpu_kernarg_layout_params_t params = {
      .kernarg_byte_length = load_plan->kernarg_byte_length,
      .kernarg_alignment = kernel->kernarg_segment_alignment,
      .constant_byte_length = load_plan->constant_byte_length,
      .implicit_args_byte_offset = load_plan->implicit_args_byte_offset,
      .binding_count = load_plan->binding_count,
      .binding_slots = binding_slots,
      .constant_span_count = load_plan->constant_span_count,
      .constant_spans = constant_spans,
  };
  return iree_hal_amdgpu_kernarg_layout_initialize(
      &params, layout_storage_capacity, layout);
}

static iree_status_t iree_hal_amdgpu_hsaco_load_plan_populate_parameters(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    const iree_hal_amdgpu_hsaco_loaded_code_object_rebase_t* rebase,
    iree_host_size_t parameter_capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || parameter_capacity == 0);

  iree_host_size_t parameter_ordinal = 0;
  iree_host_size_t binding_ordinal = 0;
  iree_host_size_t constant_source_offset = 0;
  for (iree_host_size_t i = 0; i < kernel->arg_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &kernel->args[i];
    if (iree_hal_amdgpu_hsaco_metadata_arg_kind_is_hidden(arg->kind)) {
      continue;
    }
    if (IREE_UNLIKELY(parameter_ordinal >= parameter_capacity)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AMDGPU metadata parameter output capacity too small");
    }
    iree_hal_executable_function_parameter_t* parameter =
        &out_parameters[parameter_ordinal++];
    memset(parameter, 0, sizeof(*parameter));
    parameter->flags = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE;
    parameter->size = (uint16_t)arg->size;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
            rebase, "parameter name", arg->name, &parameter->name));
    switch (arg->kind) {
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER:
        parameter->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
        parameter->offset = (uint16_t)binding_ordinal++;
        break;
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE:
        parameter->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
        parameter->offset = (uint16_t)constant_source_offset;
        constant_source_offset += arg->size;
        break;
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel `%.*s` argument %" PRIhsz
            " uses unsupported reflected value_kind `%.*s`",
            (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
            (int)arg->value_kind.size, arg->value_kind.data);
    }
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_hsaco_load_plan_populate_workgroup_size(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_executable_export_t* out_export) {
  if (kernel->has_required_workgroup_size) {
    out_export->workgroup_size[0] = kernel->required_workgroup_size[0];
    out_export->workgroup_size[1] = kernel->required_workgroup_size[1];
    out_export->workgroup_size[2] = kernel->required_workgroup_size[2];
  } else {
    out_export->flags |=
        IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE;
  }
}

iree_status_t iree_hal_amdgpu_executable_metadata_populate_from_hsaco(
    const iree_hal_amdgpu_hsaco_metadata_t* hsaco_metadata,
    iree_const_byte_span_t loaded_code_object_data,
    iree_hal_amdgpu_executable_metadata_t* metadata) {
  IREE_ASSERT_ARGUMENT(hsaco_metadata);
  IREE_ASSERT_ARGUMENT(metadata);

  iree_hal_amdgpu_executable_metadata_counts_t counts;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_executable_metadata_calculate_hsaco_counts(hsaco_metadata,
                                                                 &counts));
  if (IREE_UNLIKELY(metadata->export_count != counts.export_count ||
                    metadata->parameter_count < counts.parameter_count ||
                    metadata->layout_blob_capacity <
                        counts.layout_blob_byte_length)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU executable metadata storage does not match HSACO counts");
  }
  if (IREE_UNLIKELY(metadata->layout_blob_used != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU executable metadata layout blob is already populated");
  }

  const iree_hal_amdgpu_hsaco_loaded_code_object_rebase_t rebase = {
      .source_code_object_data = hsaco_metadata->elf_data,
      .loaded_code_object_data = loaded_code_object_data,
  };

  metadata->source =
      hsaco_metadata->kernel_count
          ? IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_HSACO_MESSAGEPACK
          : IREE_HAL_AMDGPU_EXECUTABLE_METADATA_SOURCE_ELF_SYMBOLS;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
          &rebase, "target", hsaco_metadata->target, &metadata->target));
  metadata->code_object_data = loaded_code_object_data;

  iree_host_size_t parameter_offset = 0;
  for (iree_host_size_t i = 0; i < hsaco_metadata->kernel_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel =
        &hsaco_metadata->kernels[i];
    iree_hal_amdgpu_hsaco_kernel_load_plan_t load_plan;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_load_plan_analyze_kernel(kernel, &load_plan),
        "planning native kernarg layout for kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);

    iree_hal_amdgpu_executable_reflection_t* reflection =
        &metadata->reflection[i];
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
            &rebase, "reflection name", kernel->reflection_name,
            &reflection->name));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
            &rebase, "symbol name", kernel->symbol_name,
            &reflection->symbol_name));
    reflection->parameter_offset = (uint32_t)parameter_offset;
    reflection->parameter_count = (uint32_t)load_plan.parameter_count;

    iree_hal_amdgpu_executable_export_t* export_info = &metadata->exports[i];
    export_info->flags = IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_NONE;
    export_info->fixed_group_segment_size = kernel->group_segment_fixed_size;
    export_info->fixed_private_segment_size =
        kernel->private_segment_fixed_size;
    export_info->max_dynamic_workgroup_local_memory =
        UINT32_MAX - kernel->group_segment_fixed_size;
    iree_hal_amdgpu_hsaco_load_plan_populate_workgroup_size(kernel,
                                                            export_info);

    iree_byte_span_t layout_storage = iree_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_metadata_append_layout(
        metadata, load_plan.layout_byte_length, &export_info->kernarg_layout,
        &layout_storage));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_load_plan_populate_layout_tables(
            kernel, &load_plan,
            (iree_hal_amdgpu_kernarg_layout_t*)layout_storage.data,
            layout_storage.data_length),
        "initializing native kernarg layout for kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);

    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_load_plan_populate_parameters(
            kernel, &rebase, load_plan.parameter_count,
            load_plan.parameter_count ? &metadata->parameters[parameter_offset]
                                      : NULL),
        "populating reflected parameters for kernel `%.*s`",
        (int)kernel->symbol_name.size, kernel->symbol_name.data);
    parameter_offset += load_plan.parameter_count;
  }

  for (iree_host_size_t i = 0; i < hsaco_metadata->elf_kernel_symbol_count;
       ++i) {
    const iree_host_size_t export_ordinal = hsaco_metadata->kernel_count + i;
    const iree_hal_amdgpu_hsaco_metadata_elf_kernel_symbol_t* symbol =
        &hsaco_metadata->elf_kernel_symbols[i];
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
            &rebase, "ELF-only reflection name", symbol->name,
            &metadata->reflection[export_ordinal].name));
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_hsaco_loaded_code_object_rebase_string_view(
            &rebase, "ELF-only symbol name", symbol->symbol_name,
            &metadata->reflection[export_ordinal].symbol_name));
    metadata->exports[export_ordinal].flags =
        IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_CUSTOM_DIRECT_ONLY |
        IREE_HAL_AMDGPU_EXECUTABLE_EXPORT_FLAG_REQUIRES_DISPATCH_WORKGROUP_SIZE;
  }

  if (IREE_UNLIKELY(parameter_offset != counts.parameter_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU metadata parameter population count changed");
  }
  return iree_ok_status();
}
