// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/target/entry_selection.h"
#include "loom/target/function_contract.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/compile_report_low.h"

#define LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS 20u

static bool loom_amdgpu_hal_kernel_library_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  if (!loom_low_kernel_def_isa(entry->func.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         loom_amdgpu_target_isa(entry->target_op) &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

typedef struct loom_amdgpu_hal_kernel_library_kernel_plan_t {
  // Target-resolved function entry used to build this kernel.
  loom_target_entry_t* entry;
  // Selected prepared low.kernel.def op for frame.
  loom_op_t* low_function_op;
  // ABI layout derived from prepared target-low IR.
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout;
  // Fixed allocator values derived from the HAL ABI live-ins.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
} loom_amdgpu_hal_kernel_library_kernel_plan_t;

static iree_string_view_t loom_amdgpu_hal_kernel_library_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref) &&
              symbol_ref.module_id == 0 &&
              symbol_ref.symbol_id < module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT(symbol->name_id != LOOM_STRING_ID_INVALID &&
              symbol->name_id < module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit(
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(loom_target_entry_emitter(diagnostic_emitter),
                              &emission);
}

static const loom_op_t* loom_amdgpu_hal_kernel_library_target_record_op(
    const loom_target_entry_t* entry) {
  return entry->target_op ? entry->target_op : entry->func.op;
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit_unknown_processor(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_003, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit_no_descriptor_set(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_004, params, IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_amdgpu_hal_kernel_library_emit_descriptor_set_mismatch(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t target_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->name),
      loom_param_string(processor->descriptor_set.key),
      loom_param_string(target_name),
      loom_param_string(entry->bundle_storage.config.contract_set_key),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_005, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_hal_kernel_library_apply_processor(
    loom_module_t* module, loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  if (iree_string_view_is_empty(processor_name)) {
    return iree_ok_status();
  }
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  if (processor == NULL) {
    return loom_amdgpu_hal_kernel_library_emit_unknown_processor(
        entry, diagnostic_emitter, processor_name);
  }
  if (processor->descriptor_set.ordinal ==
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE ||
      iree_string_view_is_empty(processor->descriptor_set.key)) {
    return loom_amdgpu_hal_kernel_library_emit_no_descriptor_set(
        entry, diagnostic_emitter, processor->name);
  }
  if (!iree_string_view_equal(processor->descriptor_set.key,
                              entry->bundle_storage.config.contract_set_key)) {
    return loom_amdgpu_hal_kernel_library_emit_descriptor_set_mismatch(
        entry, diagnostic_emitter, processor,
        entry->bundle_storage.bundle.name);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_target_record_set_processor(
      module, entry->target_op, processor));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_read_stream_contents(
    iree_io_stream_t* stream, iree_allocator_t allocator,
    iree_const_byte_span_t* out_contents) {
  *out_contents = iree_const_byte_span_empty();
  const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  if (stream_length < 0 || (uint64_t)stream_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO stream length is out of range");
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, (iree_host_size_t)stream_length, (void**)&data));
  iree_status_t status =
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_read(stream, (iree_host_size_t)stream_length, data,
                                 /*out_buffer_length=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_contents =
        iree_make_const_byte_span(data, (iree_host_size_t)stream_length);
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

static void loom_amdgpu_hal_kernel_library_accumulate_wait_action(
    loom_target_compile_report_wait_plan_t* summary,
    const loom_amdgpu_wait_plan_action_t* action) {
  ++summary->action_count;
  switch (action->kind) {
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT:
      ++summary->explicit_action_count;
      break;
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED:
      ++summary->planned_action_count;
      break;
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN:
    default:
      IREE_ASSERT(false, "wait plan action kind must be known");
      break;
  }
  if (action->target_count == 0) {
    ++summary->full_drain_count;
    summary->max_full_drain_outstanding_before =
        iree_max(summary->max_full_drain_outstanding_before,
                 (uint64_t)action->outstanding_before);
  } else {
    ++summary->partial_wait_count;
  }
  const uint64_t drained_count =
      action->outstanding_before > action->target_count
          ? (uint64_t)action->outstanding_before - action->target_count
          : 0;
  summary->drained_count += drained_count;
  summary->max_drained_count =
      iree_max(summary->max_drained_count, drained_count);
  summary->max_outstanding_before = iree_max(
      summary->max_outstanding_before, (uint64_t)action->outstanding_before);
}

static iree_string_view_t loom_amdgpu_hal_kernel_library_wait_action_name(
    loom_amdgpu_wait_plan_action_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT:
      return IREE_SV("explicit");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED:
      return IREE_SV("planned");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t
loom_amdgpu_hal_kernel_library_kernel_descriptor_profile_name(
    loom_amdgpu_kernel_descriptor_profile_t profile) {
  switch (profile) {
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
      return IREE_SV("gfx9");
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
      return IREE_SV("gfx11");
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
      return IREE_SV("gfx12");
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
      return IREE_SV("gfx125");
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_string_view_t
loom_amdgpu_hal_kernel_library_matrix_feature_profile_name(
    loom_amdgpu_matrix_feature_profile_t profile) {
  switch (profile) {
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908:
      return IREE_SV("mfma-gfx908");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A:
      return IREE_SV("mfma-gfx90a");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940:
      return IREE_SV("mfma-gfx940");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950:
      return IREE_SV("mfma-gfx950");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11:
      return IREE_SV("wmma-gfx11");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12:
      return IREE_SV("wmma-gfx12");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250:
      return IREE_SV("wmma-gfx1250");
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_status_t loom_amdgpu_hal_kernel_library_record_target_capability(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t namespace_name, iree_string_view_t key,
    loom_target_compile_report_capability_value_kind_t value_kind,
    uint64_t value_u64, iree_string_view_t value_string) {
  const loom_target_compile_report_target_capability_row_t row = {
      .function_name = function_name,
      .target_family_name = report->target_family_name,
      .namespace_name = namespace_name,
      .key = key,
      .value_kind = value_kind,
      .value_u64 = value_u64,
      .value_string = value_string,
  };
  return loom_target_compile_report_record_target_capability_row(report, &row);
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_target_capability_u64(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t namespace_name, iree_string_view_t key, uint64_t value) {
  return loom_amdgpu_hal_kernel_library_record_target_capability(
      report, function_name, namespace_name, key,
      LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64, value,
      iree_string_view_empty());
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_target_capability_bool(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t namespace_name, iree_string_view_t key, bool value) {
  return loom_amdgpu_hal_kernel_library_record_target_capability(
      report, function_name, namespace_name, key,
      LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL, value ? 1 : 0,
      iree_string_view_empty());
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_target_capability_string(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t namespace_name, iree_string_view_t key,
    iree_string_view_t value) {
  return loom_amdgpu_hal_kernel_library_record_target_capability(
      report, function_name, namespace_name, key,
      LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING, 0, value);
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_target_snapshot_capabilities(
    const loom_target_bundle_t* bundle, iree_string_view_t function_name,
    loom_target_compile_report_t* report) {
  if (bundle == NULL || bundle->snapshot == NULL) {
    return iree_ok_status();
  }
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  if (snapshot->subgroup_size != 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_u64(
            report, function_name, IREE_SV("target"), IREE_SV("subgroup_size"),
            snapshot->subgroup_size));
  }
  if (snapshot->index_bitwidth != 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_u64(
            report, function_name, IREE_SV("target"), IREE_SV("index_bitwidth"),
            snapshot->index_bitwidth));
  }
  if (snapshot->offset_bitwidth != 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_u64(
            report, function_name, IREE_SV("target"),
            IREE_SV("offset_bitwidth"), snapshot->offset_bitwidth));
  }
  if (snapshot->max_flat_workgroup_size != 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_u64(
            report, function_name, IREE_SV("target"),
            IREE_SV("max_flat_workgroup_size"),
            snapshot->max_flat_workgroup_size));
  }
  if (snapshot->max_workgroup_storage_bytes != 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_u64(
            report, function_name, IREE_SV("target"),
            IREE_SV("max_workgroup_storage_bytes"),
            snapshot->max_workgroup_storage_bytes));
  }
  return iree_ok_status();
}

typedef enum loom_amdgpu_hal_kernel_library_narrow_matrix_support_flag_bits_e {
  LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_UNSCALED = 1u << 0,
  LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_SCALED = 1u << 1,
} loom_amdgpu_hal_kernel_library_narrow_matrix_support_flag_bits_t;

typedef uint32_t loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t;

static iree_string_view_t
loom_amdgpu_hal_kernel_library_narrow_matrix_support_kind(
    loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t flags) {
  const loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t
      support_mask =
          LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_UNSCALED |
          LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_SCALED;
  switch (flags & support_mask) {
    case LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_UNSCALED:
      return IREE_SV("unscaled");
    case LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_SCALED:
      return IREE_SV("scaled");
    case LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_UNSCALED |
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_SCALED:
      return IREE_SV("unscaled_scaled");
    default:
      return IREE_SV("none");
  }
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
    loom_target_compile_report_t* report, iree_string_view_t function_name,
    iree_string_view_t namespace_name, iree_string_view_t key,
    loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t flags) {
  return loom_amdgpu_hal_kernel_library_record_target_capability_string(
      report, function_name, namespace_name, key,
      loom_amdgpu_hal_kernel_library_narrow_matrix_support_kind(flags));
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_matrix_feature_capabilities(
    loom_amdgpu_matrix_feature_profile_t profile,
    iree_string_view_t function_name, loom_target_compile_report_t* report) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  if (!loom_amdgpu_matrix_feature_bits_from_profile(profile, &feature_bits)) {
    return iree_ok_status();
  }

  const iree_string_view_t namespace_name = IREE_SV("amdgpu");
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_u64(
          report, function_name, namespace_name, IREE_SV("matrix_feature_bits"),
          feature_bits));

  loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t fp8_bf8_support =
      0;
  if (iree_any_bit_set(feature_bits,
                       LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 |
                           LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12)) {
    fp8_bf8_support |=
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_UNSCALED;
  }
  loom_amdgpu_hal_kernel_library_narrow_matrix_support_flags_t f8f6f4_support =
      0;
  if (iree_any_bit_set(
          feature_bits,
          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4 |
              LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4)) {
    f8f6f4_support |=
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_NARROW_MATRIX_SUPPORT_SCALED;
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
          report, function_name, namespace_name,
          IREE_SV("matrix_fp8_native_kind"), fp8_bf8_support | f8f6f4_support));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
          report, function_name, namespace_name,
          IREE_SV("matrix_bf8_native_kind"), fp8_bf8_support | f8f6f4_support));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
          report, function_name, namespace_name,
          IREE_SV("matrix_fp6_native_kind"), f8f6f4_support));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
          report, function_name, namespace_name,
          IREE_SV("matrix_bf6_native_kind"), f8f6f4_support));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_narrow_matrix_capability(
          report, function_name, namespace_name,
          IREE_SV("matrix_fp4_native_kind"), f8f6f4_support));

  const iree_host_size_t feature_count =
      loom_amdgpu_matrix_feature_info_count();
  for (iree_host_size_t i = 0; i < feature_count; ++i) {
    const loom_amdgpu_matrix_feature_info_t* feature_info =
        loom_amdgpu_matrix_feature_info_at(i);
    if (feature_info == NULL ||
        !iree_all_bits_set(feature_bits, feature_info->feature_bit)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_string(
            report, function_name, namespace_name, IREE_SV("matrix_feature"),
            feature_info->name));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_processor_capabilities(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t function_name, loom_target_compile_report_t* report) {
  if (processor == NULL) {
    return iree_ok_status();
  }
  const iree_string_view_t namespace_name = IREE_SV("amdgpu");
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name, IREE_SV("processor"),
          processor->name));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name, IREE_SV("descriptor_set"),
          processor->descriptor_set.key));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_u64(
          report, function_name, namespace_name,
          IREE_SV("wavefront_default_size"),
          processor->wavefront.default_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_bool(
          report, function_name, namespace_name, IREE_SV("wavefront_32"),
          iree_any_bit_set(processor->wavefront.supported_sizes,
                           LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_bool(
          report, function_name, namespace_name, IREE_SV("wavefront_64"),
          iree_any_bit_set(processor->wavefront.supported_sizes,
                           LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name,
          IREE_SV("kernel_descriptor_profile"),
          loom_amdgpu_hal_kernel_library_kernel_descriptor_profile_name(
              processor->kernel_descriptor.profile)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_u64(
          report, function_name, namespace_name,
          IREE_SV("kernel_descriptor_flags"),
          processor->kernel_descriptor.flags));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name,
          IREE_SV("matrix_feature_profile"),
          loom_amdgpu_hal_kernel_library_matrix_feature_profile_name(
              processor->features.matrix)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_matrix_feature_capabilities(
          processor->features.matrix, function_name, report));
  return loom_amdgpu_hal_kernel_library_record_target_capability_u64(
      report, function_name, namespace_name, IREE_SV("scheduling_bits"),
      processor->features.scheduling);
}

typedef struct loom_amdgpu_hal_kernel_library_wait_endpoint_t {
  // Node index in the schedule table, or UINT32_MAX.
  uint32_t node_index;
  // Scheduled ordinal for |node_index|, or UINT32_MAX.
  uint32_t scheduled_ordinal;
  // Operation mnemonic for |node_index|, or empty.
  iree_string_view_t operation_name;
  // Descriptor key for |node_index|, or empty.
  iree_string_view_t descriptor_key;
  // Descriptor semantic tag for |node_index|, or empty.
  iree_string_view_t semantic_tag;
} loom_amdgpu_hal_kernel_library_wait_endpoint_t;

static loom_amdgpu_hal_kernel_library_wait_endpoint_t
loom_amdgpu_hal_kernel_library_wait_endpoint(
    const loom_amdgpu_wait_plan_t* wait_plan, uint32_t node_index) {
  loom_amdgpu_hal_kernel_library_wait_endpoint_t endpoint = {
      .node_index = UINT32_MAX,
      .scheduled_ordinal = UINT32_MAX,
      .operation_name = iree_string_view_empty(),
      .descriptor_key = iree_string_view_empty(),
      .semantic_tag = iree_string_view_empty(),
  };
  const loom_low_schedule_table_t* schedule = wait_plan->schedule;
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      node_index >= schedule->node_count) {
    return endpoint;
  }

  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  endpoint.node_index = node_index;
  endpoint.scheduled_ordinal = node->scheduled_ordinal;
  if (node->op != NULL) {
    endpoint.operation_name = loom_op_name(schedule->module, node->op);
  }
  if (node->descriptor != NULL) {
    endpoint.descriptor_key = loom_low_descriptor_set_string(
        schedule->target.descriptor_set, node->descriptor->key_string_offset);
    if (node->descriptor->semantic_tag_string_offset !=
        LOOM_LOW_STRING_OFFSET_NONE) {
      endpoint.semantic_tag = loom_low_descriptor_set_string(
          schedule->target.descriptor_set,
          node->descriptor->semantic_tag_string_offset);
    }
  }
  return endpoint;
}

static iree_status_t loom_amdgpu_hal_kernel_library_record_wait_plan(
    loom_target_compile_report_t* report,
    const loom_amdgpu_packet_plan_t* packet_plan) {
  if (report == NULL || packet_plan->wait_plan.action_count == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_plan_t* wait_plan = &packet_plan->wait_plan;
  loom_target_compile_report_wait_plan_t summary = {0};
  loom_target_compile_report_wait_plan_t
      counter_summaries[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT] = {0};
  loom_target_compile_report_wait_plan_t
      reason_summaries[LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT]
                      [LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT] = {0};
  for (iree_host_size_t i = 0; i < wait_plan->action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    IREE_ASSERT(action->counter_id > LOOM_AMDGPU_WAIT_COUNTER_NONE &&
                    action->counter_id <= LOOM_AMDGPU_WAIT_COUNTER_ALU,
                "wait plan action must name a concrete counter");
    IREE_ASSERT(action->reason < LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT,
                "wait plan action must name a concrete reason");
    loom_amdgpu_hal_kernel_library_accumulate_wait_action(&summary, action);
    if (action->counter_id > LOOM_AMDGPU_WAIT_COUNTER_NONE &&
        action->counter_id <= LOOM_AMDGPU_WAIT_COUNTER_ALU) {
      const uint32_t counter_index = action->counter_id - 1;
      loom_amdgpu_hal_kernel_library_accumulate_wait_action(
          &counter_summaries[counter_index], action);
      if (action->reason < LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT) {
        loom_amdgpu_hal_kernel_library_accumulate_wait_action(
            &reason_summaries[counter_index][action->reason], action);
      }
    }
  }
  loom_target_compile_report_record_wait_plan(report, &summary);
  for (uint32_t i = 0; i < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++i) {
    if (counter_summaries[i].action_count == 0) {
      continue;
    }
    const uint32_t counter_id = i + 1;
    const loom_target_compile_report_wait_counter_row_t row = {
        .function_name = report->function_name,
        .counter_name = loom_amdgpu_wait_counter_name(counter_id),
        .counter_id = counter_id,
        .summary = counter_summaries[i],
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_wait_counter_row(report, &row));
  }
  for (uint32_t i = 0; i < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++i) {
    const uint32_t counter_id = i + 1;
    const iree_string_view_t counter_name =
        loom_amdgpu_wait_counter_name(counter_id);
    for (uint32_t reason_id = 0; reason_id < LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT;
         ++reason_id) {
      if (reason_summaries[i][reason_id].action_count == 0) {
        continue;
      }
      const loom_target_compile_report_wait_reason_summary_row_t row = {
          .function_name = report->function_name,
          .counter_name = counter_name,
          .reason_name = loom_amdgpu_wait_plan_reason_name(
              (loom_amdgpu_wait_plan_reason_t)reason_id),
          .counter_id = counter_id,
          .reason_id = reason_id,
          .summary = reason_summaries[i][reason_id],
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_wait_reason_summary_row(report,
                                                                    &row));
    }
  }
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < wait_plan->action_count; ++i) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    const uint32_t outstanding_after = action->target_count;
    const uint32_t drained_count =
        action->outstanding_before > outstanding_after
            ? action->outstanding_before - outstanding_after
            : 0;
    const loom_amdgpu_hal_kernel_library_wait_endpoint_t producer =
        loom_amdgpu_hal_kernel_library_wait_endpoint(wait_plan,
                                                     action->producer_node);
    const loom_amdgpu_hal_kernel_library_wait_endpoint_t consumer =
        loom_amdgpu_hal_kernel_library_wait_endpoint(wait_plan,
                                                     action->consumer_node);
    const loom_target_compile_report_wait_action_row_t row = {
        .function_name = report->function_name,
        .counter_name = loom_amdgpu_wait_counter_name(action->counter_id),
        .action_name =
            loom_amdgpu_hal_kernel_library_wait_action_name(action->kind),
        .reason_name = loom_amdgpu_wait_plan_reason_name(action->reason),
        .counter_id = action->counter_id,
        .action_id = (uint32_t)action->kind,
        .reason_id = (uint32_t)action->reason,
        .block_index = action->block_index,
        .node_index = action->node_index,
        .scheduled_ordinal = action->scheduled_ordinal,
        .producer_node = producer.node_index,
        .producer_scheduled_ordinal = producer.scheduled_ordinal,
        .producer_operation_name = producer.operation_name,
        .producer_descriptor_key = producer.descriptor_key,
        .producer_semantic_tag = producer.semantic_tag,
        .consumer_node = consumer.node_index,
        .consumer_scheduled_ordinal = consumer.scheduled_ordinal,
        .consumer_operation_name = consumer.operation_name,
        .consumer_descriptor_key = consumer.descriptor_key,
        .consumer_semantic_tag = consumer.semantic_tag,
        .target_count = action->target_count,
        .outstanding_before = action->outstanding_before,
        .outstanding_after = outstanding_after,
        .drained_count = drained_count,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_wait_action_row(report, &row));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_build_hsaco_contribution(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_native_preflight_t* preflight,
    iree_string_builder_t* target_listing, loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* table_arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, table_arena, &packet_plan));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_wait_plan(report, &packet_plan));

  if (target_listing != NULL) {
    if (iree_string_builder_size(target_listing) != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(target_listing, "\n\n"));
    }
    const loom_amdgpu_kernel_assembly_options_t assembly_options = {
        .abi_layout = abi_layout,
        .packet_plan = &packet_plan,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_kernel_assembly_with_options(
        &frame->schedule, &frame->allocation, &assembly_options, target_listing,
        table_arena));
  }

  const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
      .abi_layout = abi_layout,
      .preflight = preflight,
      .packet_plan = &packet_plan,
  };
  return loom_amdgpu_build_kernel_hsaco_contribution(
      &frame->schedule, &frame->allocation, &hsaco_options, out_contribution,
      table_arena);
}

static loom_target_compile_report_target_resources_t
loom_amdgpu_hal_kernel_library_target_resources_from_hsaco(
    const loom_amdgpu_kernel_hsaco_summary_t* summary) {
  const loom_amdgpu_kernel_hsaco_target_resources_t* target_resources =
      &summary->target_resources;
  return (loom_target_compile_report_target_resources_t){
      .scalar_register_class = target_resources->scalar_register_class,
      .scalar_register_count = target_resources->scalar_register_count,
      .vector_register_class = target_resources->vector_register_class,
      .vector_register_count = target_resources->vector_register_count,
      .subgroup_size = target_resources->wave_size,
      .max_subgroups_per_simd = target_resources->max_waves_per_simd,
      .resident_subgroups_per_simd = target_resources->resident_waves_per_simd,
      .occupancy_percent = target_resources->occupancy_percent,
      .limiting_resource = target_resources->limiting_resource,
  };
}

static iree_status_t loom_amdgpu_hal_kernel_library_write_hsaco(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count,
    const loom_amdgpu_kernel_hsaco_write_options_t* write_options,
    iree_const_byte_span_t* out_hsaco, iree_arena_allocator_t* table_arena,
    iree_allocator_t allocator) {
  *out_hsaco = iree_const_byte_span_empty();

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  iree_status_t status = loom_amdgpu_write_kernel_hsaco_contributions(
      contributions, contribution_count, write_options, stream, table_arena);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_kernel_library_read_stream_contents(
        stream, allocator, out_hsaco);
  }
  iree_io_stream_release(stream);
  return status;
}

void loom_amdgpu_hal_kernel_library_deinitialize(
    loom_amdgpu_hal_kernel_library_t* library, iree_allocator_t allocator) {
  if (library == NULL) {
    return;
  }
  iree_allocator_free(allocator, (void*)library->target_key.data);
  iree_allocator_free(allocator, library->hsaco_data);
  iree_allocator_free(allocator, library->target_listing_data);
  loom_target_artifact_manifest_json_t artifact_manifest_json = {
      .contents = library->artifact_manifest.contents,
  };
  loom_target_artifact_manifest_json_release(&artifact_manifest_json,
                                             allocator);
  *library = (loom_amdgpu_hal_kernel_library_t){0};
}

static iree_status_t loom_amdgpu_hal_kernel_library_set_contents(
    iree_string_view_t target_key, iree_const_byte_span_t hsaco,
    iree_allocator_t allocator, loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};

  void* target_key_data = NULL;
  iree_status_t status = iree_allocator_clone(
      allocator, iree_make_const_byte_span(target_key.data, target_key.size),
      &target_key_data);
  if (iree_status_is_ok(status)) {
    out_library->target_key =
        iree_make_string_view(target_key_data, target_key.size);
    out_library->hsaco_data = (uint8_t*)hsaco.data;
    out_library->hsaco_data_length = hsaco.data_length;
  }
  if (!iree_status_is_ok(status)) {
    out_library->hsaco_data = NULL;
    out_library->hsaco_data_length = 0;
    loom_amdgpu_hal_kernel_library_deinitialize(out_library, allocator);
  }
  return status;
}

static iree_status_t loom_amdgpu_hal_kernel_library_lookup_func_facts(
    const loom_module_t* module, loom_symbol_ref_t func_ref,
    iree_arena_allocator_t* table_arena,
    const loom_func_symbol_facts_t** out_func_facts) {
  *out_func_facts = NULL;
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, table_arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &fact_table, module, func_ref, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  IREE_ASSERT(*out_func_facts != NULL,
              "selected AMDGPU HAL kernel-library entries have func facts");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_apply_low_kernel_contract(
    const loom_module_t* module, const loom_op_t* low_function_op,
    loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, bool* out_valid) {
  *out_valid = true;
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_low_kernel_def_static_workgroup_size(low_function_op,
                                                 &workgroup_size)) {
    return iree_ok_status();
  }

  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_lookup_func_facts(
      module, entry->func_ref, table_arena, &func_facts));

  return loom_target_function_contract_apply_hal_workgroup_size(
      func_facts, entry->bundle_storage.bundle.name, &workgroup_size,
      loom_target_entry_emitter(diagnostic_emitter), &entry->bundle_storage,
      out_valid);
}

static iree_status_t loom_amdgpu_hal_kernel_library_prepare_kernel_plan(
    loom_module_t* module, loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_hal_kernel_library_kernel_plan_t){
      .entry = entry,
      .low_function_op = entry->func.op,
  };

  bool kernel_contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_apply_low_kernel_contract(
      module, out_plan->low_function_op, entry, diagnostic_emitter, table_arena,
      &kernel_contract_valid));
  if (!kernel_contract_valid) {
    return iree_ok_status();
  }
  if (report != NULL) {
    report->function_name = entry->func_name;
    loom_target_compile_report_record_target_bundle(
        report, &entry->bundle_storage.bundle);
    report->lowered_symbol =
        loom_amdgpu_hal_kernel_library_symbol_name(module, entry->func_ref);
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_snapshot_capabilities(
            &entry->bundle_storage.bundle, entry->func_name, report));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_processor_capabilities(
            loom_amdgpu_target_record_processor(module, entry->target_op),
            entry->func_name, report));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_verify_kernel_abi(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, bool* out_failed,
    iree_arena_allocator_t* table_arena) {
  *out_failed = false;
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      &low_registry->registry, &plan->entry->bundle_storage.bundle,
      &descriptor_set));
  loom_amdgpu_hal_kernel_abi_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
      module, plan->low_function_op, descriptor_set, max_errors,
      loom_target_entry_emitter(diagnostic_emitter), &result, table_arena));
  *out_failed = result.error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_compute_kernel_fixed_values(
    const loom_module_t* module,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    iree_arena_allocator_t* table_arena) {
  if (loom_amdgpu_hal_kernel_abi_has_layout_attr(plan->low_function_op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_attr(
        module, plan->low_function_op, &plan->abi_layout, table_arena));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
        module, plan->low_function_op, &plan->abi_layout, table_arena));
  }
  return loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module, plan->low_function_op, &plan->fixed_values,
      &plan->fixed_value_count, table_arena);
}

typedef struct loom_amdgpu_hal_kernel_library_spill_lowering_context_t {
  // Selected descriptor set used to rewrite structural spill traffic.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_amdgpu_hal_kernel_library_spill_lowering_context_t;

static iree_status_t loom_amdgpu_hal_kernel_library_lower_spill_traffic(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* table_arena,
    loom_low_emission_frame_lower_spill_traffic_result_t* out_result) {
  const loom_amdgpu_hal_kernel_library_spill_lowering_context_t* context =
      (const loom_amdgpu_hal_kernel_library_spill_lowering_context_t*)user_data;
  loom_amdgpu_spill_lowering_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_spill_traffic(
      module, low_function_op, context->descriptor_set, emitter, &result,
      table_arena));
  out_result->error_count = result.error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_materialize_address_state(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* table_arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  (void)user_data;
  return loom_amdgpu_materialize_address_state(module, low_function_op, frame,
                                               table_arena, out_result);
}

static iree_status_t
loom_amdgpu_hal_kernel_library_validate_final_workgroup_storage(
    void* user_data, const loom_low_emission_frame_t* frame,
    iree_arena_allocator_t* table_arena) {
  (void)table_arena;
  const uint64_t limit =
      frame->target.bundle_storage.snapshot.max_workgroup_storage_bytes;
  if (limit == 0) {
    return iree_ok_status();
  }

  loom_low_storage_layout_space_sizes_t sizes = {0};
  IREE_RETURN_IF_ERROR(loom_low_storage_layout_collect_space_sizes(
      frame->module, frame->function_op, &sizes));
  if (sizes.workgroup_bytes <= limit) {
    return iree_ok_status();
  }

  const iree_diagnostic_emitter_t* emitter =
      (const iree_diagnostic_emitter_t*)user_data;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(frame->module, frame->function_op)),
      loom_param_string(loom_low_diagnostic_target_key(&frame->target)),
      loom_param_u64(sizes.workgroup_bytes),
      loom_param_u64(limit),
  };
  const loom_diagnostic_emission_t emission = {
      .op = frame->function_op,
      .error = LOOM_ERR_TARGET_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(*emitter, &emission);
}

static iree_status_t loom_amdgpu_hal_kernel_library_build_kernel_contribution(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_selection_t target_selection,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, iree_string_builder_t* target_listing,
    loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution) {
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      &low_registry->registry, &plan->entry->bundle_storage.bundle,
      &descriptor_set));
  const loom_low_pressure_model_t* pressure_model =
      loom_amdgpu_occupancy_pressure_model(descriptor_set);
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  loom_low_resolved_target_t resolved_target = {
      .bundle_storage = plan->entry->bundle_storage,
      .descriptor_set = descriptor_set,
  };
  loom_target_bundle_storage_rebind(&resolved_target.bundle_storage);
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      &resolved_target, table_arena, &schedule_pair_affinities));
  loom_low_schedule_structural_state_read_list_t schedule_state_reads =
      loom_low_schedule_structural_state_read_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_build_structural_state_reads(
      descriptor_set, table_arena, &schedule_state_reads));

  loom_low_emission_frame_t frame = {0};
  loom_low_planning_statistics_t planning_statistics = {0};
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &low_registry->registry,
      .target_selection =
          {
              .bundle = &plan->entry->bundle_storage.bundle,
              .data = target_selection.data,
          },
      .pressure_model = pressure_model,
      .schedule_pair_affinities = schedule_pair_affinities,
      .schedule_structural_state_reads = schedule_state_reads,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .allocation_fixed_values = plan->fixed_values,
      .allocation_fixed_value_count = plan->fixed_value_count,
      .storage_lease_provider = &storage_lease_provider,
      .emitter = loom_target_entry_emitter(diagnostic_emitter),
      .statistics = report != NULL ? &planning_statistics : NULL,
  };
  if (report != NULL) {
    loom_target_compile_report_record_low_kernel_workload(
        report, plan->low_function_op);
  }
  loom_amdgpu_native_preflight_t preflight = {0};
  loom_amdgpu_hal_kernel_library_spill_lowering_context_t
      spill_lowering_context = {
          .descriptor_set = descriptor_set,
      };
  iree_diagnostic_emitter_t final_validation_emitter = frame_options.emitter;
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_SCRATCH |
                                          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
              .emit_spill_diagnostics = true,
              .record_materialized_spills =
                  loom_target_compile_report_wants_details(
                      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS),
              .emitter = frame_options.emitter,
          },
      .lower_spill_traffic = loom_amdgpu_hal_kernel_library_lower_spill_traffic,
      .lower_spill_traffic_user_data = &spill_lowering_context,
      .materialize_address_state =
          loom_amdgpu_hal_kernel_library_materialize_address_state,
      .materialize_address_state_user_data = NULL,
      .validate_frame =
          loom_amdgpu_hal_kernel_library_validate_final_workgroup_storage,
      .validate_frame_user_data = (void*)&final_validation_emitter,
  };
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      module, plan->low_function_op, &frame_options, &spill_free_options,
      table_arena, &frame));
  if (diagnostic_emitter->error_count != 0) {
    if (report != NULL) {
      loom_target_compile_report_record_low_planning(report,
                                                     &planning_statistics);
      if (frame.allocation.function_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_target_compile_report_record_low_allocation(
            report, &frame.allocation));
      }
    }
    return iree_ok_status();
  }
  const loom_amdgpu_native_preflight_options_t preflight_options = {
      .emitter = frame_options.emitter,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_analyze(
      &frame.schedule, &frame.allocation, &preflight_options, &preflight));
  if (preflight.error_count != 0) {
    return iree_ok_status();
  }
  if (report != NULL) {
    loom_target_compile_report_record_low_planning(report,
                                                   &planning_statistics);
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_low_emission_frame(report, &frame));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_build_hsaco_contribution(
      &frame, &plan->abi_layout, &preflight, target_listing, report,
      out_contribution, table_arena));
  if (report != NULL) {
    loom_target_compile_report_record_emission(
        report, out_contribution->summary.instruction_count,
        out_contribution->summary.text_byte_count,
        out_contribution->summary.text_storage_byte_count);
    loom_target_compile_report_record_memory(
        report, out_contribution->summary.private_segment_fixed_size,
        out_contribution->summary.group_segment_fixed_size);
    const loom_target_compile_report_target_resources_t target_resources =
        loom_amdgpu_hal_kernel_library_target_resources_from_hsaco(
            &out_contribution->summary);
    loom_target_compile_report_record_target_resources(report,
                                                       &target_resources);
  }
  return iree_ok_status();
}

static void loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(
    uint16_t report_count, loom_target_compile_report_t* reports) {
  for (uint16_t i = 0; i < report_count && reports != NULL; ++i) {
    loom_target_compile_report_deinitialize(&reports[i]);
  }
}

static iree_string_view_t loom_amdgpu_hal_kernel_library_rodata_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  IREE_ASSERT(symbol->name_id != LOOM_STRING_ID_INVALID &&
              symbol->name_id < module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_amdgpu_hal_kernel_library_collect_rodata_symbols(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_amdgpu_hsaco_data_symbol_t** out_data_symbols,
    iree_host_size_t* out_data_symbol_count) {
  *out_data_symbols = NULL;
  *out_data_symbol_count = 0;

  iree_host_size_t rodata_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->defining_op && loom_global_rodata_isa(symbol->defining_op)) {
      ++rodata_count;
    }
  }
  if (rodata_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hsaco_data_symbol_t* data_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, rodata_count, sizeof(*data_symbols), (void**)&data_symbols));
  iree_host_size_t data_symbol_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_op_t* op = symbol->defining_op;
    if (!op || !loom_global_rodata_isa(op)) {
      continue;
    }

    const iree_string_view_t name =
        loom_amdgpu_hal_kernel_library_rodata_symbol_name(module, symbol);
    const iree_const_byte_span_t contents = loom_global_rodata_contents(op);
    uint64_t alignment = 0;
    const loom_attribute_t alignment_attr =
        loom_op_const_attrs(op)[loom_global_rodata_alignment_ATTR_INDEX];
    if (!loom_attr_is_absent(alignment_attr)) {
      alignment = (uint64_t)loom_attr_as_i64(alignment_attr);
    }
    data_symbols[data_symbol_count++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = name,
        .initial_contents = contents,
        .byte_length = contents.data_length,
        .alignment = alignment,
    };
  }

  *out_data_symbols = data_symbols;
  *out_data_symbol_count = data_symbol_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_compose_data_symbols(
    loom_amdgpu_runtime_global_flags_t runtime_globals,
    const loom_amdgpu_hsaco_data_symbol_t* data_symbols,
    iree_host_size_t data_symbol_count,
    const loom_amdgpu_hsaco_data_symbol_t* rodata_symbols,
    iree_host_size_t rodata_symbol_count, iree_arena_allocator_t* arena,
    const loom_amdgpu_hsaco_data_symbol_t** out_data_symbols,
    iree_host_size_t* out_data_symbol_count) {
  *out_data_symbols = NULL;
  *out_data_symbol_count = 0;

  const iree_host_size_t runtime_global_symbol_count =
      loom_amdgpu_runtime_global_count(runtime_globals);
  if (data_symbol_count != 0 && data_symbols == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL kernel-library data symbols are "
                            "required when data_symbol_count is non-zero");
  }
  if (rodata_symbol_count != 0 && rodata_symbols == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL kernel-library rodata symbols are "
                            "required when rodata_symbol_count is non-zero");
  }

  iree_host_size_t total_symbol_count = 0;
  if (!iree_host_size_checked_add(runtime_global_symbol_count,
                                  data_symbol_count, &total_symbol_count) ||
      !iree_host_size_checked_add(total_symbol_count, rodata_symbol_count,
                                  &total_symbol_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU HAL kernel-library data symbol count overflow");
  }
  if (total_symbol_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_hsaco_data_symbol_t* composed_symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, total_symbol_count,
                                                 sizeof(*composed_symbols),
                                                 (void**)&composed_symbols));
  iree_host_size_t composed_symbol_count = 0;
  loom_amdgpu_runtime_global_symbols(runtime_globals, composed_symbols,
                                     &composed_symbol_count);
  IREE_ASSERT_EQ(composed_symbol_count, runtime_global_symbol_count);
  if (data_symbol_count != 0) {
    memcpy(composed_symbols + composed_symbol_count, data_symbols,
           data_symbol_count * sizeof(*data_symbols));
    composed_symbol_count += data_symbol_count;
  }
  if (rodata_symbol_count != 0) {
    memcpy(composed_symbols + composed_symbol_count, rodata_symbols,
           rodata_symbol_count * sizeof(*rodata_symbols));
    composed_symbol_count += rodata_symbol_count;
  }
  IREE_ASSERT_EQ(composed_symbol_count, total_symbol_count);

  *out_data_symbols = composed_symbols;
  *out_data_symbol_count = total_symbol_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_entries(
    loom_module_t* module, const loom_target_entry_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_entry_list_t entries, loom_target_selection_t target_selection,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    const loom_amdgpu_hal_kernel_library_options_t* options, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library, iree_allocator_t allocator) {
  *out_emitted = false;
  IREE_ASSERT(entries.count != 0 && entries.values != NULL);

  const uint32_t max_errors = loom_target_entry_max_errors(
      target_options, LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS);
  const bool capture_target_listing =
      options ? options->capture_target_listing : false;
  const loom_amdgpu_runtime_global_flags_t runtime_globals =
      options ? options->runtime_globals : LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  const loom_amdgpu_hsaco_data_symbol_t* data_symbols =
      options ? options->data_symbols : NULL;
  const iree_host_size_t data_symbol_count =
      options ? options->data_symbol_count : 0;
  const loom_amdgpu_hsaco_data_symbol_t* rodata_symbols = NULL;
  iree_host_size_t rodata_symbol_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_collect_rodata_symbols(
      module, table_arena, &rodata_symbols, &rodata_symbol_count));
  loom_target_compile_report_t* entry_reports = NULL;
  if (report != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table_arena, entries.count,
                                                   sizeof(*entry_reports),
                                                   (void**)&entry_reports));
    memset(entry_reports, 0, entries.count * sizeof(*entry_reports));
    for (uint16_t i = 0; i < entries.count; ++i) {
      loom_target_compile_report_initialize(&entry_reports[i],
                                            report->allocator);
      entry_reports[i].requested_detail_flags = report->requested_detail_flags;
      entry_reports[i].artifact_kind =
          LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY;
      entry_reports[i].backend_name = report->backend_name;
      entry_reports[i].target_family_name = report->target_family_name;
      entry_reports[i].target_key = report->target_key;
      entry_reports[i].artifact_format = report->artifact_format;
    }
  }
  loom_amdgpu_hal_kernel_library_kernel_plan_t* plans = NULL;
  iree_status_t status = iree_arena_allocate_array(
      table_arena, entries.count, sizeof(*plans), (void**)&plans);
  bool diagnostics_failed = false;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_prepare_kernel_plan(
        module, &entries.values[i], diagnostic_emitter, table_arena,
        entry_reports != NULL ? &entry_reports[i] : NULL, &plans[i]);
    if (iree_status_is_ok(status) && plans[i].low_function_op == NULL) {
      diagnostics_failed = true;
    }
  }

  bool abi_failed = false;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    bool plan_failed = false;
    status = loom_amdgpu_hal_kernel_library_verify_kernel_abi(
        module, low_registry, &plans[i], diagnostic_emitter, max_errors,
        &plan_failed, table_arena);
    if (iree_status_is_ok(status)) {
      abi_failed |= plan_failed;
    }
  }
  if (abi_failed) {
    diagnostics_failed = true;
  }

  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_compute_kernel_fixed_values(
        module, &plans[i], table_arena);
  }

  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = loom_target_entry_verify_module(
        module, target_options,
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      diagnostics_failed = true;
    }
  }
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = loom_target_entry_verify_low_module(
        module, low_registry, diagnostic_emitter, target_selection, max_errors,
        loom_low_verify_provider_list_empty(), &low_verify_scratch,
        &low_verify_result);
    if (iree_status_is_ok(status) && low_verify_result.error_count != 0) {
      diagnostics_failed = true;
    }
  }

  loom_amdgpu_kernel_hsaco_contribution_t* contributions = NULL;
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = iree_arena_allocate_array(table_arena, entries.count,
                                       sizeof(*contributions),
                                       (void**)&contributions);
  }
  iree_string_builder_t target_listing;
  bool target_listing_initialized = false;
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      capture_target_listing) {
    iree_string_builder_initialize(allocator, &target_listing);
    target_listing_initialized = true;
  }
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_build_kernel_contribution(
        module, low_registry, &plans[i], target_selection, diagnostic_emitter,
        table_arena, target_listing_initialized ? &target_listing : NULL,
        entry_reports != NULL ? &entry_reports[i] : NULL, &contributions[i]);
  }
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    for (uint16_t i = 0; i < entries.count && entry_reports != NULL &&
                         iree_status_is_ok(status);
         ++i) {
      status = loom_target_compile_report_record_entry_report(
          report, &entry_reports[i]);
    }
  }
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      diagnostic_emitter->error_count == 0) {
    iree_const_byte_span_t hsaco = iree_const_byte_span_empty();
    const loom_amdgpu_hsaco_data_symbol_t* code_object_data_symbols = NULL;
    iree_host_size_t code_object_data_symbol_count = 0;
    status = loom_amdgpu_hal_kernel_library_compose_data_symbols(
        runtime_globals, data_symbols, data_symbol_count, rodata_symbols,
        rodata_symbol_count, table_arena, &code_object_data_symbols,
        &code_object_data_symbol_count);
    const loom_amdgpu_kernel_hsaco_write_options_t write_options = {
        .data_symbols = code_object_data_symbols,
        .data_symbol_count = code_object_data_symbol_count,
    };
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_hal_kernel_library_write_hsaco(
          contributions, entries.count,
          code_object_data_symbol_count != 0 ? &write_options : NULL, &hsaco,
          table_arena, allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_hal_kernel_library_set_contents(
          contributions[0].target, hsaco, allocator, out_library);
    }
    if (iree_status_is_ok(status)) {
      hsaco = iree_const_byte_span_empty();
      if (capture_target_listing &&
          iree_string_builder_size(&target_listing) != 0) {
        out_library->target_listing_format = IREE_SV("amdgpu-assembly");
        out_library->target_listing_data_length =
            iree_string_builder_size(&target_listing);
        out_library->target_listing_data =
            iree_string_builder_take_storage(&target_listing);
      }
      if (options != NULL && options->artifact_manifest.mode !=
                                 LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
        loom_target_artifact_manifest_collect_options_t manifest_options =
            options->artifact_manifest;
        manifest_options.artifact_name = options->artifact_name;
        manifest_options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
        manifest_options.flags =
            LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
        manifest_options.artifact_byte_length = out_library->hsaco_data_length;
        loom_target_artifact_manifest_json_t artifact_manifest_json = {0};
        status = loom_target_artifact_manifest_collect_json_from_entries(
            module, entries, &manifest_options, table_arena, allocator,
            &artifact_manifest_json);
        if (iree_status_is_ok(status) &&
            artifact_manifest_json.contents.data != NULL) {
          out_library->artifact_manifest =
              (loom_target_emit_sidecar_artifact_t){
                  .kind =
                      LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
                  .identifier = options->artifact_manifest_identifier,
                  .contents = artifact_manifest_json.contents,
              };
        }
      }
      if (iree_status_is_ok(status)) {
        *out_emitted = true;
      }
    }
    iree_allocator_free(allocator, (void*)hsaco.data);
  }
  if (target_listing_initialized) {
    iree_string_builder_deinitialize(&target_listing);
  }
  loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(entries.count,
                                                            entry_reports);
  return status;
}

iree_status_t loom_amdgpu_emit_hal_kernel_library(
    loom_module_t* module,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_emitted = false;
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};
  const loom_amdgpu_runtime_global_flags_t runtime_globals =
      options ? options->runtime_globals : LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_runtime_global_flags_validate(runtime_globals));
  loom_target_compile_report_t* report = options ? options->report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize_if_empty(report, allocator);
    report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY;
  }
  const loom_target_selection_t target_selection =
      options ? options->target_selection : loom_target_selection_empty();
  const loom_target_entry_options_t target_options = {
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
      .effective_target_bundle = target_selection.bundle,
  };
  loom_target_environment_t target_environment = {0};
  iree_status_t status = loom_target_environment_initialize(
      &loom_amdgpu_target_provider_set, &target_environment);
  loom_target_low_descriptor_registry_t low_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        &target_environment, &low_registry);
  }
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
  const loom_target_entry_predicate_t entry_predicate = {
      .fn = loom_amdgpu_hal_kernel_library_bundle_is_compatible,
      .user_data = NULL,
  };

  // Transient target tables share the module workspace pool for warm reuse.
  iree_arena_allocator_t table_arena;
  iree_arena_initialize(module->arena.block_pool, &table_arena);

  loom_target_entry_list_t entries = {0};
  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_entry_verify_module(
        module, &target_options,
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS, &verify_result);
  }
  bool selected = false;
  if (iree_status_is_ok(status) && verify_result.error_count == 0 &&
      diagnostic_emitter.error_count == 0) {
    status = loom_target_entry_select_all_entries(
        module, &target_options, entry_predicate, &diagnostic_emitter,
        IREE_SV("AMDGPU HAL-native"), &table_arena, &selected, &entries);
  }
  if (iree_status_is_ok(status) && selected && options != NULL) {
    for (uint16_t i = 0; i < entries.count && iree_status_is_ok(status); ++i) {
      status = loom_amdgpu_hal_kernel_library_apply_processor(
          module, &entries.values[i], &diagnostic_emitter, options->processor);
    }
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0 && report != NULL) {
    if (entries.count == 1) {
      loom_target_compile_report_record_target_bundle(
          report, &entries.values[0].bundle_storage.bundle);
    } else if (entries.count > 0) {
      report->target_bundle_name = entries.values[0].bundle_storage.bundle.name;
      if (entries.values[0].bundle_storage.bundle.snapshot != NULL) {
        report->target_snapshot_name =
            entries.values[0].bundle_storage.bundle.snapshot->name;
      }
    }
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0) {
    status = loom_amdgpu_hal_kernel_library_entries(
        module, &target_options, &low_registry, entries, target_selection,
        &diagnostic_emitter, &table_arena, report, options, out_emitted,
        out_library, allocator);
  }
  if (iree_status_is_ok(status) && *out_emitted && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, out_library->hsaco_data_length);
  }

  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_kernel_library_deinitialize(out_library, allocator);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, iree_status_code(status));
  }
  iree_arena_deinitialize(&table_arena);
  loom_target_environment_deinitialize(&target_environment);
  return status;
}
