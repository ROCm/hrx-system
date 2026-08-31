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
#include "iree/io/vec_stream.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/kernel_emission.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/target/entry_selection.h"
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

typedef struct loom_amdgpu_hal_kernel_library_kernel_plan_t {
  // Selected prepared low.kernel.def op for frame.
  loom_op_t* low_function_op;
  // Resolved representation contract and function target facts.
  loom_low_resolved_target_t target;
  // ABI layout derived from prepared target-low IR.
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout;
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
  return module->strings.entries[symbol->name_id];
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
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count,
    const loom_amdgpu_kernel_hsaco_write_options_t* write_options,
    iree_arena_allocator_t* table_arena, iree_allocator_t allocator,
    iree_byte_sequence_t** out_hsaco) {
  *out_hsaco = NULL;

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  iree_status_t status = loom_amdgpu_write_kernel_hsaco_contributions(
      contributions, contribution_count, write_options, stream, table_arena);
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(stream, out_hsaco);
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

static const loom_target_artifact_manifest_target_projection_t
    kLoomAmdgpuHalKernelLibraryManifestTargetProjection = {
        .project = loom_amdgpu_hal_kernel_library_project_manifest_target,
};

static iree_status_t loom_amdgpu_hal_kernel_library_prepare_kernel_plan(
    loom_module_t* module, loom_target_entry_t* entry,
    const loom_low_descriptor_registry_t* low_registry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_hal_kernel_library_kernel_plan_t){0};

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, table_arena);
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, entry->func.op, entry->target_facts, low_registry,
      loom_target_entry_emitter(diagnostic_emitter), &out_plan->target));
  if (out_plan->target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  out_plan->low_function_op = entry->func.op;
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
    uint32_t max_errors, bool* out_failed,
    iree_arena_allocator_t* table_arena) {
  *out_failed = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
      module, plan->low_function_op, plan->target.descriptor_set, max_errors,
      loom_target_entry_emitter(diagnostic_emitter), &plan->abi_verify,
      table_arena));
  *out_failed = plan->abi_verify.error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_prepare_kernel_abi_layout(
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
  return iree_ok_status();
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
  out_result->required_register_value_ids = result.required_register_value_ids;
  out_result->required_register_value_count =
      result.required_register_value_count;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_kernel_library_validate_final_workgroup_storage(
    void* user_data, const loom_low_emission_frame_t* frame,
    iree_arena_allocator_t* table_arena) {
  (void)table_arena;
  const uint64_t limit = loom_low_resolved_target_bundle(&frame->target)
                             ->snapshot->max_workgroup_storage_bytes;
  if (limit == 0) {
    return iree_ok_status();
  }

  const uint64_t workgroup_bytes =
      frame->schedule.storage_layout.space_sizes.workgroup_bytes;
  if (workgroup_bytes <= limit) {
    return iree_ok_status();
  }

  const iree_diagnostic_emitter_t* emitter =
      (const iree_diagnostic_emitter_t*)user_data;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_diagnostic_function_name(frame->module, frame->function_op)),
      loom_param_string(loom_low_diagnostic_target_key(&frame->target)),
      loom_param_u64(workgroup_bytes),
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
    loom_module_t* module, const loom_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, iree_string_builder_t* target_listing,
    loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution) {
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};

  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  const loom_target_residency_model_t* residency_model =
      loom_amdgpu_occupancy_residency_model(&plan->target);
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      &plan->target, table_arena, &schedule_pair_affinities));
  loom_low_schedule_structural_state_read_list_t schedule_state_reads =
      loom_amdgpu_descriptor_structural_state_reads();

  loom_low_emission_frame_t frame = {0};
  loom_low_planning_statistics_t planning_statistics = {0};
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = low_registry,
      .function_target_facts = plan->target.target_facts,
      .residency_model = residency_model,
      .schedule_pair_affinities = schedule_pair_affinities,
      .schedule_structural_state_reads = schedule_state_reads,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .allocation_fixed_values = plan->abi_verify.fixed_values,
      .allocation_fixed_value_count = plan->abi_verify.fixed_value_count,
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
          .descriptor_set = plan->target.descriptor_set,
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

  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_emission_build(
      &frame, &plan->abi_layout, &plan->abi_verify, &preflight, target_listing,
      report, out_contribution, table_arena));
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
    if (symbol->defining_op &&
        loom_global_rodata_def_isa(symbol->defining_op)) {
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
    if (!op || !loom_global_rodata_def_isa(op)) {
      continue;
    }

    const iree_string_view_t name =
        loom_amdgpu_hal_kernel_library_rodata_symbol_name(module, symbol);
    const iree_const_byte_span_t contents = loom_global_rodata_def_contents(op);
    uint64_t alignment = 0;
    const loom_attribute_t alignment_attr =
        loom_op_const_attrs(op)[loom_global_rodata_def_alignment_ATTR_INDEX];
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
    const loom_low_descriptor_registry_t* low_registry,
    loom_low_verify_provider_list_t low_verify_provider_list,
    loom_target_entry_list_t entries,
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
        module, &entries.values[i], low_registry, diagnostic_emitter,
        table_arena, entry_reports != NULL ? &entry_reports[i] : NULL,
        &plans[i]);
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
        module, &plans[i], diagnostic_emitter, max_errors, &plan_failed,
        table_arena);
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
    status = loom_amdgpu_hal_kernel_library_prepare_kernel_abi_layout(
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
        module, low_registry, target_options, diagnostic_emitter,
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS,
        low_verify_provider_list, &low_verify_scratch, &low_verify_result);
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
        module, low_registry, &plans[i], diagnostic_emitter, table_arena,
        target_listing_initialized ? &target_listing : NULL,
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
    iree_byte_sequence_t* hsaco = NULL;
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
          code_object_data_symbol_count != 0 ? &write_options : NULL,
          table_arena, allocator, &hsaco);
    }
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_hal_kernel_library_set_contents(
          contributions[0].artifact_target_key, hsaco, allocator, out_library);
    }
    if (iree_status_is_ok(status)) {
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

static iree_status_t loom_amdgpu_hal_kernel_library_prepare_export_projection(
    loom_target_entry_list_t entries,
    loom_target_emit_export_projection_buffer_t* projection,
    iree_host_size_t* out_projection_count) {
  *out_projection_count = 0;
  if (projection == NULL) return iree_ok_status();
  projection->count = 0;
  if (projection->capacity != 0 && projection->values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU export projection capacity requires caller-owned row storage");
  }

  iree_host_size_t projection_count = 0;
  for (uint16_t i = 0; i < entries.count; ++i) {
    const loom_target_function_version_t* function_version =
        entries.values[i].function_version;
    if (function_version == NULL) continue;
    if (projection_count >= projection->capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU export projection capacity %" PRIhsz
                              " is too small for mapped exports",
                              projection->capacity);
    }
    projection->values[projection_count++] =
        (loom_target_emit_export_projection_t){
            .function_version = &function_version->base,
            .ordinal = i,
        };
  }
  *out_projection_count = projection_count;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_hal_kernel_library(
    loom_module_t* module,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_emitted = false;
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};
  loom_target_emit_export_projection_buffer_t* export_projection =
      options ? options->export_projection : NULL;
  if (export_projection != NULL) export_projection->count = 0;
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
  const loom_target_entry_options_t target_options = {
      .function_versions = options ? options->function_versions : NULL,
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
  };
  const loom_low_descriptor_registry_t* low_registry =
      options ? options->low_environment.descriptor_registry : NULL;
  loom_low_verify_provider_list_t low_verify_provider_list =
      options ? options->low_environment.verify_providers
              : loom_low_verify_provider_list_empty();
  loom_target_environment_t target_environment = {0};
  loom_target_low_descriptor_registry_t standalone_low_registry = {0};
  iree_status_t status = iree_ok_status();
  if (low_registry == NULL) {
    status = loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment);
    if (iree_status_is_ok(status)) {
      status = loom_target_environment_initialize_low_descriptor_registry(
          &target_environment, &standalone_low_registry);
    }
    if (iree_status_is_ok(status)) {
      low_registry = &standalone_low_registry.registry;
      low_verify_provider_list =
          loom_target_environment_low_verify_provider_list(&target_environment);
    }
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
  iree_host_size_t export_projection_count = 0;
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0) {
    status = loom_amdgpu_hal_kernel_library_prepare_export_projection(
        entries, export_projection, &export_projection_count);
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
        module, &target_options, low_registry, low_verify_provider_list,
        entries, &diagnostic_emitter, &table_arena, report, options,
        out_emitted, out_library, allocator);
  }
  if (iree_status_is_ok(status) && *out_emitted && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, iree_byte_sequence_length(out_library->hsaco_data));
  }
  if (iree_status_is_ok(status) && *out_emitted && export_projection != NULL) {
    export_projection->count = export_projection_count;
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
