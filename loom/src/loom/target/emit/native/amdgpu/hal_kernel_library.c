// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/source.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/artifact_symbols.h"
#include "loom/target/emit/native/amdgpu/data_symbol_directory.h"
#include "loom/target/emit/native/amdgpu/function_compile.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/hsaco_build.h"
#include "loom/target/emit/native/amdgpu/kernel_emission.h"
#include "loom/target/emit/native/amdgpu/kernel_plan_build.h"
#include "loom/target/emit/native/amdgpu/kernel_report.h"
#include "loom/target/emit/native/amdgpu/program_artifact.h"
#include "loom/target/emit/native/amdgpu/program_compile.h"
#include "loom/target/emit/native/amdgpu/program_emission.h"
#include "loom/target/emit/native/amdgpu/program_report.h"
#include "loom/target/entry_selection.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/low.h"

#define LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS 20u

static bool loom_amdgpu_hal_kernel_library_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  if (!loom_low_kernel_def_isa(entry->func.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  return bundle && bundle->snapshot && bundle->export_plan &&
         loom_amdgpu_target_facts_cast(entry->target_facts) != NULL &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_hal_kernel_library_emit_target_identity_mismatch(
    const loom_target_entry_t* first_entry,
    const loom_amdgpu_target_identity_t* code_object_identity,
    const loom_target_entry_t* entry,
    const loom_amdgpu_target_identity_t* entry_identity,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* arena) {
  iree_string_view_t code_object_target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
      code_object_identity, arena, &code_object_target_key));
  iree_string_view_t entry_target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_format_arena(
      entry_identity, arena, &entry_target_key));
  const loom_diagnostic_param_t params[] = {
      loom_param_string(entry->func_name),
      loom_param_string(entry_target_key),
      loom_param_string(first_entry->func_name),
      loom_param_string(code_object_target_key),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("first code-object target selected here"),
      .op = first_entry->func.op,
  }};
  const loom_diagnostic_emission_t emission = {
      .op = entry->func.op,
      .export_name = loom_target_facts_export_name(entry->target_facts),
      .error = LOOM_ERR_AMDGPU_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(loom_target_entry_emitter(diagnostic_emitter),
                              &emission);
}

typedef struct loom_amdgpu_hal_kernel_library_kernel_plan_t {
  // Selected prepared low.kernel.def op for frame.
  loom_op_t* low_function_op;
  // Retained source proofs for the prepared packet effects.
  const loom_low_memory_access_map_t* memory_accesses;
  // Retained source execution facts for the prepared Low loops.
  const loom_low_loop_execution_map_t* loop_execution;
  // Resolved representation contract and function target facts.
  loom_low_resolved_target_t target;
  // Verified ABI facts retained for allocation and native emission.
  loom_amdgpu_hal_kernel_abi_verify_result_t abi_verify;
} loom_amdgpu_hal_kernel_library_kernel_plan_t;

static iree_string_view_t loom_amdgpu_hal_kernel_library_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref) &&
              symbol_ref.module_id == 0 &&
              symbol_ref.symbol_id < module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT(symbol->name_id != LOOM_STRING_ID_INVALID &&
              symbol->name_id < module->strings.count);
  return loom_string_table_get(&module->strings, symbol->name_id);
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
                           LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
                           LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250)) {
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
  const iree_string_view_t matrix_feature_namespace_name =
      IREE_SV("amdgpu.matrix_feature");
  for (iree_host_size_t i = 0; i < feature_count; ++i) {
    const loom_amdgpu_matrix_feature_info_t* feature_info =
        loom_amdgpu_matrix_feature_info_at(i);
    if (feature_info == NULL ||
        !iree_all_bits_set(feature_bits, feature_info->feature_bit)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_capability_bool(
            report, function_name, matrix_feature_namespace_name,
            feature_info->name, true));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_library_record_target_profile_capabilities(
    const loom_amdgpu_target_info_t* target,
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t function_name, loom_target_compile_report_t* report) {
  if (target == NULL || processor == NULL) {
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
          target->descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_u64(
          report, function_name, namespace_name,
          IREE_SV("wavefront_default_size"),
          processor->properties.wavefront.default_size));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_bool(
          report, function_name, namespace_name, IREE_SV("wavefront_32"),
          iree_any_bit_set(processor->properties.wavefront.supported_sizes,
                           LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_bool(
          report, function_name, namespace_name, IREE_SV("wavefront_64"),
          iree_any_bit_set(processor->properties.wavefront.supported_sizes,
                           LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name,
          IREE_SV("kernel_descriptor_profile"),
          loom_amdgpu_hal_kernel_library_kernel_descriptor_profile_name(
              processor->properties.kernel_descriptor.profile)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_u64(
          report, function_name, namespace_name,
          IREE_SV("kernel_descriptor_flags"),
          processor->properties.kernel_descriptor.flags));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_target_capability_string(
          report, function_name, namespace_name,
          IREE_SV("matrix_feature_profile"),
          loom_amdgpu_hal_kernel_library_matrix_feature_profile_name(
              processor->properties.features.matrix)));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_library_record_matrix_feature_capabilities(
          processor->properties.features.matrix, function_name, report));
  return loom_amdgpu_hal_kernel_library_record_target_capability_u64(
      report, function_name, namespace_name, IREE_SV("scheduling_bits"),
      processor->properties.features.scheduling);
}

static iree_status_t loom_amdgpu_hal_kernel_library_write_hsaco(
    const loom_amdgpu_hsaco_plan_t* plan, iree_arena_allocator_t* table_arena,
    iree_allocator_t allocator, iree_byte_sequence_t** out_hsaco) {
  return loom_amdgpu_hsaco_serialize_plan(plan, allocator, table_arena,
                                          out_hsaco);
}

void loom_amdgpu_hal_kernel_library_deinitialize(
    loom_amdgpu_hal_kernel_library_t* library, iree_allocator_t allocator) {
  if (library == NULL) {
    return;
  }
  iree_allocator_free(allocator, (void*)library->target_key.data);
  iree_byte_sequence_release(library->hsaco_data);
  iree_byte_sequence_release(library->target_listing_data);
  iree_byte_sequence_release(library->artifact_manifest.contents);
  *library = (loom_amdgpu_hal_kernel_library_t){0};
}

static iree_status_t loom_amdgpu_hal_kernel_library_set_contents(
    iree_string_view_t target_key, iree_byte_sequence_t* hsaco,
    iree_allocator_t allocator, loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};

  void* target_key_data = NULL;
  iree_status_t status = iree_allocator_clone(
      allocator, iree_make_const_byte_span(target_key.data, target_key.size),
      &target_key_data);
  if (iree_status_is_ok(status)) {
    out_library->target_key =
        iree_make_string_view(target_key_data, target_key.size);
    out_library->hsaco_data = hsaco;
  }
  if (!iree_status_is_ok(status)) {
    out_library->hsaco_data = NULL;
    loom_amdgpu_hal_kernel_library_deinitialize(out_library, allocator);
  }
  return status;
}

static void loom_amdgpu_hal_kernel_library_append_manifest_feature(
    loom_amdgpu_target_feature_state_t state, iree_string_view_t off_name,
    iree_string_view_t on_name, iree_string_view_t* feature_names,
    iree_host_size_t* inout_feature_name_count) {
  if (state == LOOM_AMDGPU_TARGET_FEATURE_OFF) {
    feature_names[(*inout_feature_name_count)++] = off_name;
  } else if (state == LOOM_AMDGPU_TARGET_FEATURE_ON) {
    feature_names[(*inout_feature_name_count)++] = on_name;
  }
}

static iree_status_t loom_amdgpu_hal_kernel_library_project_manifest_target(
    const loom_module_t* module, const loom_target_entry_t* entry,
    iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_target_t* inout_target) {
  (void)module;
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(entry->target_facts);
  IREE_ASSERT(target_facts != NULL);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target_facts->identity.target);
  IREE_ASSERT(processor != NULL);

  inout_target->family = IREE_SV("amdgpu");
  inout_target->selector = target_facts->identity.target->name;
  inout_target->processor = processor->name;
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_id_format(
      &target_facts->identity, arena, &inout_target->code_object_target));

  iree_string_view_t feature_names[2] = {0};
  iree_host_size_t feature_name_count = 0;
  loom_amdgpu_hal_kernel_library_append_manifest_feature(
      target_facts->identity.amdhsa_features.sramecc, IREE_SV("sramecc-"),
      IREE_SV("sramecc+"), feature_names, &feature_name_count);
  loom_amdgpu_hal_kernel_library_append_manifest_feature(
      target_facts->identity.amdhsa_features.xnack, IREE_SV("xnack-"),
      IREE_SV("xnack+"), feature_names, &feature_name_count);
  if (feature_name_count != 0) {
    iree_string_view_t* stored_feature_names = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, feature_name_count, sizeof(*stored_feature_names),
        (void**)&stored_feature_names));
    memcpy(stored_feature_names, feature_names,
           feature_name_count * sizeof(*stored_feature_names));
    inout_target->feature_names = stored_feature_names;
    inout_target->feature_name_count = feature_name_count;
  }
  return iree_ok_status();
}

// Reports the kernarg layout that HAL ABI lowering recorded on the prepared
// kernel (loom_amdgpu_map_abi_layout and binding materialization): the
// segment size and, per authored kernel parameter, its kernarg offset,
// payload size and alignment. A by-value payload narrower than its 4-byte
// word occupies the word's low-address bytes.
static iree_status_t loom_amdgpu_hal_kernel_library_project_manifest_interface(
    const loom_module_t* module, const loom_target_entry_t* entry,
    bool include_parameters, iree_arena_allocator_t* arena,
    loom_target_artifact_manifest_interface_t* inout_interface) {
  const loom_op_t* function_op = entry->func.op;
  if (!loom_amdgpu_hal_kernel_abi_has_layout_attr(function_op)) {
    return iree_ok_status();
  }
  // Compilation already admitted this layout, so admission here cannot
  // reject it and needs no diagnostic emitter.
  loom_amdgpu_hal_kernel_abi_layout_t layout = {0};
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_attr(
      module, function_op, (iree_diagnostic_emitter_t){0}, arena, &accepted,
      &layout));
  if (!accepted) {
    return iree_ok_status();
  }
  inout_interface->flags |=
      LOOM_TARGET_ARTIFACT_MANIFEST_INTERFACE_FLAG_ARGUMENT_BYTE_LENGTH;
  inout_interface->argument_byte_length = layout.kernarg_segment_size;
  if (!include_parameters) {
    return iree_ok_status();
  }
  inout_interface->parameters = NULL;
  inout_interface->parameter_detail_count = 0;
  if (layout.parameter_count == 0) {
    return iree_ok_status();
  }
  loom_target_artifact_manifest_parameter_t* parameters = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, layout.parameter_count, sizeof(*parameters), (void**)&parameters));
  for (uint32_t i = 0; i < layout.parameter_count; ++i) {
    parameters[i] = (loom_target_artifact_manifest_parameter_t){
        .kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_UNKNOWN,
        .flags = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX,
        .index = i,
    };
  }
  const loom_target_artifact_manifest_parameter_flags_t layout_flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_INDEX |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_OFFSET |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_LENGTH |
      LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_FLAG_BYTE_ALIGNMENT;
  for (iree_host_size_t i = 0; i < layout.resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource = &layout.resources[i];
    IREE_ASSERT_LT(resource->parameter_index, layout.parameter_count);
    parameters[resource->parameter_index] =
        (loom_target_artifact_manifest_parameter_t){
            .name = resource->name,
            .kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_BINDING,
            .flags = layout_flags,
            .index = resource->parameter_index,
            .byte_offset = resource->kernarg_offset,
            .byte_length = resource->kernarg_size,
            .byte_alignment = resource->kernarg_alignment,
        };
  }
  for (iree_host_size_t i = 0; i < layout.direct_arg_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &layout.direct_args[i];
    IREE_ASSERT_LT(direct_arg->parameter_index, layout.parameter_count);
    parameters[direct_arg->parameter_index] =
        (loom_target_artifact_manifest_parameter_t){
            .name = direct_arg->name,
            .kind = LOOM_TARGET_ARTIFACT_MANIFEST_PARAMETER_KIND_VALUE,
            .flags = layout_flags,
            .index = direct_arg->parameter_index,
            .byte_offset = direct_arg->kernarg_offset,
            .byte_length = direct_arg->payload_size,
            .byte_alignment = direct_arg->kernarg_alignment,
        };
  }
  inout_interface->parameters = parameters;
  inout_interface->parameter_detail_count = layout.parameter_count;
  return iree_ok_status();
}

static const loom_target_artifact_manifest_target_projection_t
    kLoomAmdgpuHalKernelLibraryManifestTargetProjection = {
        .project = loom_amdgpu_hal_kernel_library_project_manifest_target,
        .project_interface =
            loom_amdgpu_hal_kernel_library_project_manifest_interface,
};

static iree_status_t loom_amdgpu_hal_kernel_library_build_kernel_plan(
    loom_module_t* module, loom_target_entry_t* entry,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_hal_kernel_library_kernel_plan_t){0};

  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(entry->target_facts),
      .target = loom_target_entry_emitter(diagnostic_emitter),
  };
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, table_arena);
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, entry->func.op, entry->target_facts,
      &low_registry->registry,
      loom_diagnostic_export_scope_emitter(&diagnostic_scope),
      &out_plan->target));
  if (out_plan->target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  out_plan->low_function_op = entry->func.op;
  const loom_target_function_version_t* version = entry->function_version;
  out_plan->memory_accesses = version != NULL ? version->memory_accesses : NULL;
  out_plan->loop_execution = version != NULL ? version->loop_execution : NULL;
  entry->target_facts = out_plan->target.target_facts;

  if (report != NULL) {
    const loom_amdgpu_target_facts_t* target_facts =
        loom_amdgpu_target_facts_cast(out_plan->target.target_facts);
    IREE_ASSERT(target_facts != NULL);
    report->function_name = entry->func_name;
    const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
    loom_target_compile_report_record_target_bundle(report, bundle);
    report->lowered_symbol =
        loom_amdgpu_hal_kernel_library_symbol_name(module, entry->func_ref);
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_snapshot_capabilities(
            bundle, entry->func_name, report));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_library_record_target_profile_capabilities(
            target_facts->identity.target,
            loom_amdgpu_target_info_target_processor(
                target_facts->identity.target),
            entry->func_name, report));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_verify_kernel_abi(
    const loom_module_t* module,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, iree_arena_allocator_t* table_arena,
    bool* out_failed) {
  *out_failed = false;
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(plan->target.target_facts),
      .target = loom_target_entry_emitter(diagnostic_emitter),
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
      module, plan->low_function_op, plan->target.descriptor_set,
      loom_amdgpu_target_processor_properties_from_resolved_target(
          &plan->target),
      max_errors, loom_diagnostic_export_scope_emitter(&diagnostic_scope),
      table_arena, &plan->abi_verify));
  *out_failed = plan->abi_verify.error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_build_kernel(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    const loom_amdgpu_program_artifact_root_t* artifact_root,
    const loom_amdgpu_program_artifact_emission_t* artifact_emission,
    const loom_amdgpu_data_symbol_directory_t* data_symbols,
    bool include_artifact_functions_in_listing,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, iree_string_builder_t* target_listing,
    loom_target_compile_report_t* report, bool* out_built,
    loom_amdgpu_kernel_emission_result_t* out_emission) {
  *out_built = false;
  *out_emission = (loom_amdgpu_kernel_emission_result_t){0};

  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(plan->target.target_facts),
      .target = loom_target_entry_emitter(diagnostic_emitter),
  };
  const iree_diagnostic_emitter_t emitter =
      loom_diagnostic_export_scope_emitter(&diagnostic_scope);
  const loom_amdgpu_function_compile_options_t compile_options = {
      .descriptor_registry = &low_registry->registry,
      .resolved_target = &plan->target,
      .memory_accesses = plan->memory_accesses,
      .loop_execution = plan->loop_execution,
      .allocation_fixed_values = plan->abi_verify.fixed_values,
      .allocation_fixed_value_count = plan->abi_verify.fixed_value_count,
      .emitter = emitter,
      .report = report,
  };
  if (report != NULL) {
    loom_target_compile_report_record_low_kernel_workload(
        report, plan->low_function_op);
  }
  loom_amdgpu_program_compilation_t compilation = {0};
  bool program_compiled = false;
  if (artifact_root != NULL) {
    const loom_amdgpu_program_compile_options_t program_options = {
        .artifact_root = artifact_root,
        .entry = compile_options,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_program_compile(
        module, plan->low_function_op, &program_options, table_arena,
        &program_compiled, &compilation));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_function_compile(
        module, plan->low_function_op, &compile_options, table_arena,
        &program_compiled, &compilation.entry));
  }
  if (!program_compiled) {
    return iree_ok_status();
  }
  const loom_amdgpu_function_compilation_t* entry_compilation =
      &compilation.entry;
  loom_amdgpu_kernel_plan_t native_plan = {0};
  const loom_amdgpu_kernel_plan_options_t native_plan_options = {
      .abi_verify = &plan->abi_verify,
      .program_resources =
          compilation.has_retained_calls ? &compilation.resources : NULL,
      .emitter = emitter,
      .report = report,
  };
  bool native_plan_accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_plan_build_from_function(
      &entry_compilation->frame, &entry_compilation->plan, &native_plan_options,
      table_arena, &native_plan_accepted, &native_plan));
  if (!native_plan_accepted) {
    if (report != NULL) {
      loom_target_compile_report_record_low_planning(
          report, &entry_compilation->planning_statistics);
      IREE_RETURN_IF_ERROR(loom_target_compile_report_record_low_allocation(
          report, &entry_compilation->frame.allocation));
    }
    return iree_ok_status();
  }
  loom_target_compile_report_low_dynamic_context_t dynamic_context = {0};
  iree_status_t status = iree_ok_status();
  if (report != NULL) {
    loom_target_compile_report_record_low_planning(
        report, &entry_compilation->planning_statistics);
    status = loom_target_compile_report_low_dynamic_context_initialize(
        &entry_compilation->frame, &dynamic_context);
    if (iree_status_is_ok(status)) {
      status =
          loom_target_compile_report_record_low_emission_frame_with_dynamic_context(
              report, &entry_compilation->frame, &dynamic_context);
    }
  }
  loom_amdgpu_kernel_emission_result_t emission_result = {0};
  bool emission_accepted = false;
  const loom_amdgpu_kernel_emission_flags_t emission_flags =
      report != NULL
          ? LOOM_AMDGPU_KERNEL_EMISSION_FLAG_CAPTURE_NATIVE_INSERTIONS
          : LOOM_AMDGPU_KERNEL_EMISSION_FLAG_NONE;
  if (iree_status_is_ok(status)) {
    status =
        compilation.has_retained_calls
            ? loom_amdgpu_program_emission_build(
                  &compilation, artifact_emission, data_symbols, &native_plan,
                  emission_flags, include_artifact_functions_in_listing,
                  target_listing, table_arena, emitter,
                  loom_target_entry_emitter(diagnostic_emitter),
                  &emission_accepted, &emission_result)
            : loom_amdgpu_kernel_emission_build(
                  &entry_compilation->frame, &native_plan, data_symbols,
                  emission_flags, target_listing, table_arena, emitter,
                  &emission_accepted, &emission_result);
  }
  if (iree_status_is_ok(status) && emission_accepted && report != NULL) {
    status = loom_amdgpu_kernel_report_record_emission(
        report, &entry_compilation->frame, &dynamic_context, &native_plan,
        &emission_result);
  }
  if (iree_status_is_ok(status) && emission_accepted && report != NULL) {
    status = loom_amdgpu_program_report_compose_entry(artifact_root, report);
  }
  loom_target_compile_report_low_dynamic_context_deinitialize(&dynamic_context);
  IREE_RETURN_IF_ERROR(status);
  if (!emission_accepted) {
    return iree_ok_status();
  }
  *out_emission = emission_result;
  *out_built = true;
  return iree_ok_status();
}

static void loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(
    uint16_t report_count, loom_target_compile_report_t* reports) {
  for (uint16_t i = 0; i < report_count && reports != NULL; ++i) {
    loom_target_compile_report_deinitialize(&reports[i]);
  }
}

static iree_status_t loom_amdgpu_hal_kernel_library_entries(
    loom_module_t* module, const loom_target_entry_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_entry_list_t entries,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_symbol_reference_table_t* symbol_references,
    const loom_target_function_version_snapshot_t* function_versions,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_emitted = false;
  IREE_ASSERT(entries.count != 0 && entries.values != NULL);

  const uint32_t max_errors = loom_target_entry_max_errors(
      target_options, LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS);
  const bool capture_target_listing =
      options ? options->capture_target_listing : false;
  loom_amdgpu_data_symbol_directory_t data_symbols = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_data_symbol_directory_build(
      module, table_arena, &data_symbols));
  loom_target_compile_report_t* entry_reports = NULL;
  if (report != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table_arena, entries.count,
                                                   sizeof(*entry_reports),
                                                   (void**)&entry_reports));
    memset(entry_reports, 0, entries.count * sizeof(*entry_reports));
    for (uint16_t i = 0; i < entries.count; ++i) {
      loom_target_compile_report_initialize(report->allocator,
                                            &entry_reports[i]);
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
  const loom_target_entry_t* code_object_entry = NULL;
  const loom_amdgpu_target_identity_t* code_object_identity = NULL;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_build_kernel_plan(
        module, &entries.values[i], low_registry, diagnostic_emitter,
        table_arena, entry_reports != NULL ? &entry_reports[i] : NULL,
        &plans[i]);
    if (iree_status_is_ok(status) && plans[i].low_function_op == NULL) {
      diagnostics_failed = true;
    }
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      const loom_amdgpu_target_facts_t* target_facts =
          loom_amdgpu_target_facts_cast(plans[i].target.target_facts);
      IREE_ASSERT(target_facts != NULL);
      if (code_object_identity == NULL) {
        code_object_entry = &entries.values[i];
        code_object_identity = &target_facts->identity;
      } else if (!loom_amdgpu_target_identity_equal(code_object_identity,
                                                    &target_facts->identity)) {
        diagnostics_failed = true;
        status = loom_amdgpu_hal_kernel_library_emit_target_identity_mismatch(
            code_object_entry, code_object_identity, &entries.values[i],
            &target_facts->identity, diagnostic_emitter, table_arena);
      }
    }
  }

  bool abi_failed = false;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    bool plan_failed = false;
    status = loom_amdgpu_hal_kernel_library_verify_kernel_abi(
        module, &plans[i], diagnostic_emitter, max_errors, table_arena,
        &plan_failed);
    if (iree_status_is_ok(status)) {
      abi_failed |= plan_failed;
    }
  }
  if (abi_failed) {
    diagnostics_failed = true;
  }

  loom_amdgpu_program_artifact_t program_artifact = {0};
  loom_amdgpu_program_artifact_emission_t program_artifact_emission = {0};
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      symbol_references != NULL) {
    loom_amdgpu_program_artifact_root_input_t* root_inputs = NULL;
    status = iree_arena_allocate_array(
        table_arena, entries.count, sizeof(*root_inputs), (void**)&root_inputs);
    for (uint16_t i = 0; i < entries.count && iree_status_is_ok(status); ++i) {
      root_inputs[i] = (loom_amdgpu_program_artifact_root_input_t){
          .function_op = plans[i].low_function_op,
          .target = &plans[i].target,
      };
    }
    const loom_amdgpu_program_artifact_options_t artifact_options = {
        .references = symbol_references,
        .function_versions = function_versions,
        .descriptor_registry = &low_registry->registry,
        .report_template = report,
        .emitter = loom_target_entry_emitter(diagnostic_emitter),
    };
    bool artifact_built = false;
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_program_artifact_build(
          module, root_inputs, entries.count, &artifact_options, table_arena,
          &artifact_built, &program_artifact);
    }
    if (iree_status_is_ok(status) && !artifact_built) {
      diagnostics_failed = true;
    }
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      bool artifact_emitted = false;
      status = loom_amdgpu_program_artifact_emission_build(
          &program_artifact, &data_symbols, table_arena,
          loom_target_entry_emitter(diagnostic_emitter), &artifact_emitted,
          &program_artifact_emission);
      diagnostics_failed |= !artifact_emitted;
    }
  }

  loom_amdgpu_hsaco_kernel_t* kernels = NULL;
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = iree_arena_allocate_array(table_arena, entries.count,
                                       sizeof(*kernels), (void**)&kernels);
  }
  iree_string_builder_t target_listing;
  bool target_listing_initialized = false;
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      capture_target_listing) {
    iree_string_builder_initialize(allocator, &target_listing);
    target_listing_initialized = true;
  }
  bool artifact_functions_listed = false;
  uint64_t artifact_body_instruction_count =
      program_artifact_emission.instruction_count;
  uint64_t artifact_entry_instruction_count = 0;
  uint64_t artifact_code_byte_count = program_artifact_emission.code_byte_count;
  uint64_t artifact_coissued_instruction_count =
      program_artifact_emission.coissued_instruction_count;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    const loom_amdgpu_program_artifact_root_t* artifact_root =
        symbol_references != NULL ? &program_artifact.roots[i] : NULL;
    const bool include_artifact_functions_in_listing =
        target_listing_initialized && !artifact_functions_listed &&
        artifact_root != NULL && artifact_root->helper_count != 0;
    bool kernel_built = false;
    loom_amdgpu_kernel_emission_result_t kernel_emission = {0};
    status = loom_amdgpu_hal_kernel_library_build_kernel(
        module, low_registry, &plans[i], artifact_root,
        &program_artifact_emission, &data_symbols,
        include_artifact_functions_in_listing, diagnostic_emitter, table_arena,
        target_listing_initialized ? &target_listing : NULL,
        entry_reports != NULL ? &entry_reports[i] : NULL, &kernel_built,
        &kernel_emission);
    if (iree_status_is_ok(status) && !kernel_built) {
      diagnostics_failed = true;
    } else if (iree_status_is_ok(status)) {
      kernels[i] = kernel_emission.kernel;
      if (report != NULL) {
        if (!iree_checked_add_u64(artifact_body_instruction_count,
                                  kernel_emission.body.instruction_count,
                                  &artifact_body_instruction_count) ||
            !iree_checked_add_u64(artifact_entry_instruction_count,
                                  kernel_emission.entry_instruction_count,
                                  &artifact_entry_instruction_count) ||
            !iree_checked_add_u64(artifact_code_byte_count,
                                  kernel_emission.kernel.text.data_length,
                                  &artifact_code_byte_count) ||
            !iree_checked_add_u64(artifact_coissued_instruction_count,
                                  kernel_emission.coissued_instruction_count,
                                  &artifact_coissued_instruction_count)) {
          status =
              iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                               "AMDGPU artifact emission summary overflowed");
        }
      }
      if (iree_status_is_ok(status) && include_artifact_functions_in_listing) {
        artifact_functions_listed = true;
      }
    }
  }
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    for (uint16_t i = 0; i < entries.count && entry_reports != NULL &&
                         iree_status_is_ok(status);
         ++i) {
      status = loom_target_compile_report_record_entry_report(
          report, &entry_reports[i]);
    }
  }
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    iree_byte_sequence_t* hsaco = NULL;
    iree_string_view_t artifact_target_key = iree_string_view_empty();
    IREE_ASSERT(code_object_identity != NULL);
    status = loom_amdgpu_artifact_key_format_arena(
        code_object_identity, table_arena, &artifact_target_key);
    loom_amdgpu_hsaco_plan_t hsaco_plan = {0};
    if (iree_status_is_ok(status)) {
      const loom_amdgpu_hsaco_input_t hsaco_input = {
          .target_identity = *code_object_identity,
          .kernels = kernels,
          .kernel_count = entries.count,
          .functions = program_artifact_emission.functions,
          .function_count = program_artifact_emission.function_count,
          .data_symbols = data_symbols.symbols,
          .data_symbol_count = data_symbols.symbol_count,
          .data_layout = data_symbols.layout,
      };
      bool symbols_accepted = false;
      status = loom_amdgpu_artifact_symbols_validate(
          module, &hsaco_input, entries, &program_artifact, &data_symbols,
          loom_target_entry_emitter(diagnostic_emitter), table_arena,
          &symbols_accepted);
      diagnostics_failed |= !symbols_accepted;
      if (iree_status_is_ok(status) && !diagnostics_failed) {
        bool plan_accepted = false;
        status = loom_amdgpu_hsaco_plan_build(
            &hsaco_input, table_arena,
            loom_target_entry_emitter(diagnostic_emitter), &plan_accepted,
            &hsaco_plan);
        diagnostics_failed |= !plan_accepted;
      }
    }
    if (iree_status_is_ok(status) && !diagnostics_failed && report != NULL) {
      uint64_t artifact_instruction_count = 0;
      if (!iree_checked_add_u64(artifact_body_instruction_count,
                                artifact_entry_instruction_count,
                                &artifact_instruction_count)) {
        status =
            iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "AMDGPU artifact instruction count overflowed");
      } else {
        loom_target_compile_report_record_emission(
            report, artifact_instruction_count, artifact_code_byte_count,
            hsaco_plan.executable_code_byte_length);
        const loom_target_compile_report_emission_breakdown_t
            emission_breakdown = {
                .body_instruction_count = artifact_body_instruction_count,
                .entry_instruction_count = artifact_entry_instruction_count,
                .coissued_instruction_count =
                    artifact_coissued_instruction_count,
                .coissued_component_count =
                    artifact_coissued_instruction_count * 2u,
            };
        loom_target_compile_report_record_emission_breakdown(
            report, &emission_breakdown);
      }
    }
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      status = loom_amdgpu_hal_kernel_library_write_hsaco(
          &hsaco_plan, table_arena, allocator, &hsaco);
    }
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      status = loom_amdgpu_hal_kernel_library_set_contents(
          artifact_target_key, hsaco, allocator, out_library);
    }
    if (iree_status_is_ok(status) && !diagnostics_failed) {
      hsaco = NULL;
      out_library->target_bundle_storage =
          entries.values[0].target_facts->storage;
      loom_target_bundle_storage_rebind(&out_library->target_bundle_storage);
      if (capture_target_listing &&
          iree_string_builder_size(&target_listing) != 0) {
        const iree_host_size_t listing_length =
            iree_string_builder_size(&target_listing);
        iree_byte_span_t listing_contents = iree_make_byte_span(
            iree_string_builder_take_storage(&target_listing), listing_length);
        status = iree_byte_sequence_create_from_span_move(
            &listing_contents, allocator, &out_library->target_listing_data);
        iree_allocator_free(allocator, listing_contents.data);
        if (iree_status_is_ok(status)) {
          out_library->target_listing_format = IREE_SV("amdgpu-assembly");
        }
      }
      if (iree_status_is_ok(status) && options != NULL &&
          options->artifact_manifest.mode !=
              LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
        loom_target_artifact_manifest_collect_options_t manifest_options =
            options->artifact_manifest;
        manifest_options.artifact_name = options->artifact_name;
        manifest_options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
        manifest_options.target_projection =
            &kLoomAmdgpuHalKernelLibraryManifestTargetProjection;
        manifest_options.flags =
            LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
        manifest_options.artifact_byte_length =
            iree_byte_sequence_length(out_library->hsaco_data);
        loom_target_artifact_manifest_json_t artifact_manifest_json = {0};
        status = loom_target_artifact_manifest_collect_json_from_entries(
            module, entries, &manifest_options, table_arena, allocator,
            &artifact_manifest_json);
        if (iree_status_is_ok(status) &&
            artifact_manifest_json.contents.data != NULL) {
          iree_byte_span_t manifest_contents = iree_make_byte_span(
              (uint8_t*)artifact_manifest_json.contents.data,
              artifact_manifest_json.contents.data_length);
          iree_byte_sequence_t* manifest_sequence = NULL;
          status = iree_byte_sequence_create_from_span_move(
              &manifest_contents, allocator, &manifest_sequence);
          if (iree_status_is_ok(status)) {
            artifact_manifest_json.contents = iree_const_byte_span_empty();
          }
          out_library->artifact_manifest =
              (loom_target_emit_sidecar_artifact_t){
                  .kind =
                      LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
                  .identifier = options->artifact_manifest_identifier,
                  .contents = manifest_sequence,
              };
        }
        loom_target_artifact_manifest_json_release(&artifact_manifest_json,
                                                   allocator);
      }
      if (iree_status_is_ok(status)) {
        *out_emitted = true;
      }
    }
    iree_byte_sequence_release(hsaco);
  }
  if (target_listing_initialized) {
    iree_string_builder_deinitialize(&target_listing);
  }
  loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(entries.count,
                                                            entry_reports);
  return status;
}

iree_status_t loom_amdgpu_compile_hal_kernel_library(
    loom_module_t* module,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_emitted = false;
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};
  loom_target_compile_report_t* report = options ? options->report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize_if_empty(report, allocator);
    report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY;
    report->target_family_name = loom_amdgpu_target_profile_type.name;
  }
  const loom_target_entry_options_t target_options = {
      .function_versions = options ? options->function_versions : NULL,
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
  };
  loom_target_environment_t target_environment = {0};
  iree_status_t status = loom_target_environment_initialize(
      &loom_amdgpu_target_provider_set, &target_environment);
  loom_target_low_descriptor_registry_t low_registry = {0};
  if (iree_status_is_ok(status)) {
    low_registry =
        loom_target_environment_low_descriptor_registry(&target_environment);
  }
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  if (options != NULL && options->diagnostic_emitter.fn != NULL) {
    diagnostic_emitter.forwarding_emitter = options->diagnostic_emitter;
  } else {
    loom_target_entry_diagnostic_emitter_initialize(
        module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
  }
  const loom_target_entry_predicate_t entry_predicate = {
      .fn = loom_amdgpu_hal_kernel_library_bundle_is_compatible,
      .user_data = NULL,
  };

  // Transient target tables share the module workspace pool for warm reuse.
  iree_arena_allocator_t table_arena;
  iree_arena_initialize(module->arena.block_pool, &table_arena);

  loom_target_entry_list_t entries = {0};
  loom_symbol_reference_table_t symbol_references = {0};
  loom_target_function_version_snapshot_t function_versions = {0};
  const bool has_low_internal_calls =
      loom_module_has_low_internal_calls(module);
  bool selected = false;
  if (iree_status_is_ok(status) && has_low_internal_calls) {
    status = loom_symbol_reference_table_build(module, &table_arena,
                                               &symbol_references);
  }
  if (iree_status_is_ok(status) && has_low_internal_calls) {
    status = loom_target_function_version_snapshot_build(
        module, target_options.function_versions, &table_arena,
        &function_versions);
  }
  if (iree_status_is_ok(status) && has_low_internal_calls) {
    status = loom_target_entry_select_all_entries_from_version_snapshot(
        module, &target_options, &function_versions, entry_predicate,
        &diagnostic_emitter, IREE_SV("AMDGPU HAL-native"), &table_arena,
        &selected, &entries);
  } else if (iree_status_is_ok(status)) {
    status = loom_target_entry_select_all_entries(
        module, &target_options, entry_predicate, &diagnostic_emitter,
        IREE_SV("AMDGPU HAL-native"), &table_arena, &selected, &entries);
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0 && report != NULL) {
    if (entries.count == 1) {
      loom_target_compile_report_record_target_bundle(
          report, loom_target_entry_bundle(&entries.values[0]));
    } else if (entries.count > 0) {
      const loom_target_bundle_t* bundle =
          loom_target_entry_bundle(&entries.values[0]);
      report->target_bundle_name = bundle->name;
      if (bundle->snapshot != NULL) {
        report->target_snapshot_name = bundle->snapshot->name;
      }
    }
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0) {
    status = loom_amdgpu_hal_kernel_library_entries(
        module, &target_options, &low_registry, entries, &diagnostic_emitter,
        has_low_internal_calls ? &symbol_references : NULL,
        has_low_internal_calls ? &function_versions : NULL, &table_arena,
        report, options, allocator, out_emitted, out_library);
  }
  if (iree_status_is_ok(status) && *out_emitted && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, iree_byte_sequence_length(out_library->hsaco_data));
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

static iree_status_t loom_amdgpu_hal_kernel_library_emit(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .function_versions = request->function_versions,
      .diagnostic_emitter = request->diagnostic_emitter,
      .max_errors = request->max_errors,
      .report = request->compile_report,
      .capture_target_listing = iree_any_bit_set(
          request->flags, LOOM_TARGET_EMIT_REQUEST_FLAG_TARGET_LISTING),
      .artifact_name = request->identifier,
      .artifact_manifest_identifier = request->artifact_manifest.identifier,
      .artifact_manifest =
          {
              .mode = request->artifact_manifest.mode,
          },
  };
  bool emitted = false;
  loom_amdgpu_hal_kernel_library_t library = {0};
  iree_status_t status = loom_amdgpu_compile_hal_kernel_library(
      request->module, &library_options, request->allocator, &emitted,
      &library);
  if (request->compile_report != NULL) {
    request->compile_report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  }
  if (iree_status_is_ok(status) && emitted && request->compile_report != NULL) {
    // The library owns its exact artifact key. Report serialization happens
    // after library release but before the caller resets invocation scratch.
    char* target_key = NULL;
    status = iree_arena_allocate(request->scratch_arena,
                                 library.target_key.size, (void**)&target_key);
    if (iree_status_is_ok(status)) {
      memcpy(target_key, library.target_key.data, library.target_key.size);
      request->compile_report->target_key =
          iree_make_string_view(target_key, library.target_key.size);
    }
  }
  const bool retain_target_bundle = iree_any_bit_set(
      request->flags, LOOM_TARGET_EMIT_REQUEST_FLAG_RETAIN_TARGET_BUNDLE);
  if (iree_status_is_ok(status) && emitted &&
      (retain_target_bundle || library.artifact_manifest.contents != NULL)) {
    status = loom_target_emit_artifact_retain_metadata(
        retain_target_bundle ? &library.target_bundle_storage.bundle : NULL,
        library.artifact_manifest.contents != NULL ? 1 : 0,
        &library.artifact_manifest, request->allocator, out_artifact);
  }
  if (iree_status_is_ok(status) && emitted) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    out_artifact->contents = library.hsaco_data;
    out_artifact->target_listing_format = library.target_listing_format;
    out_artifact->target_listing_contents = library.target_listing_data;
    library.hsaco_data = NULL;
    library.target_listing_data = NULL;
    *out_emitted = true;
  }
  loom_amdgpu_hal_kernel_library_deinitialize(&library, request->allocator);
  return status;
}

const loom_target_emitter_t loom_amdgpu_hal_kernel_library_emitter = {
    .name = IREE_SVL("amdgpu-hal"),
    .public_artifact_format = IREE_SVL("amdgpu-hsaco"),
    .default_identifier = IREE_SVL("module.hsaco"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .emit = loom_amdgpu_hal_kernel_library_emit,
};

const loom_target_provider_t loom_amdgpu_hal_kernel_library_provider = {
    .emitter_list =
        {
            .values =
                (const loom_target_emitter_t* const[]){
                    &loom_amdgpu_hal_kernel_library_emitter},
            .count = 1,
        },
    .canonical_kernel_emitter = &loom_amdgpu_hal_kernel_library_emitter,
    .canonical_kernel_fact_type = &loom_amdgpu_target_fact_type,
};

static const loom_target_provider_t* const kLoomAmdgpuCompilerProviders[] = {
    &loom_amdgpu_target_provider,
    &loom_amdgpu_hal_kernel_library_provider,
};

const loom_target_provider_set_t loom_amdgpu_compiler_provider_set = {
    .providers = kLoomAmdgpuCompilerProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCompilerProviders),
};
