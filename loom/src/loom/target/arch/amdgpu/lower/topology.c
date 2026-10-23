// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/topology.h"

#include "loom/ops/kernel/launch_config.h"

bool loom_amdgpu_required_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    loom_target_workgroup_size_t* out_size) {
  return loom_amdgpu_required_workgroup_size_from_facts(
      module, function, bundle, /*fact_table=*/NULL, out_size);
}

bool loom_amdgpu_required_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table,
    loom_target_workgroup_size_t* out_size) {
  *out_size = (loom_target_workgroup_size_t){0};

  if (loom_kernel_def_static_workgroup_size_from_facts(module, function.op,
                                                       fact_table, out_size)) {
    return true;
  }

  if (bundle == NULL || bundle->export_plan == NULL ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_size = bundle->export_plan->hal_kernel.required_workgroup_size;
  return out_size->x != 0 || out_size->y != 0 || out_size->z != 0;
}

bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size) {
  return loom_amdgpu_required_flat_workgroup_size_from_facts(
      module, function, bundle, /*fact_table=*/NULL, out_flat_size);
}

bool loom_amdgpu_required_flat_workgroup_size_from_facts(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle,
    const loom_value_fact_table_t* fact_table, uint32_t* out_flat_size) {
  *out_flat_size = 0;
  loom_target_workgroup_size_t size = {0};
  if (!loom_amdgpu_required_workgroup_size_from_facts(module, function, bundle,
                                                      fact_table, &size) ||
      size.x == 0 || size.y == 0 || size.z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size.x * size.y * size.z;
  if (flat_size == 0 || flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

bool loom_amdgpu_resolve_subgroup_wave_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t wavefront_size,
    loom_amdgpu_subgroup_wave_shape_t* out_shape) {
  *out_shape = (loom_amdgpu_subgroup_wave_shape_t){0};
  if (!loom_amdgpu_wavefront_size_is_valid(wavefront_size)) {
    return false;
  }

  uint32_t flat_workgroup_size = 0;
  if (!loom_amdgpu_required_flat_workgroup_size(module, function, bundle,
                                                &flat_workgroup_size) ||
      flat_workgroup_size == 0) {
    return false;
  }

  const uint32_t tail_lane_count = flat_workgroup_size % wavefront_size;
  const bool has_partial_tail =
      flat_workgroup_size > wavefront_size && tail_lane_count != 0;
  *out_shape = (loom_amdgpu_subgroup_wave_shape_t){
      .flat_workgroup_size = flat_workgroup_size,
      .active_lane_count = flat_workgroup_size < wavefront_size
                               ? flat_workgroup_size
                               : wavefront_size,
      .tail_lane_count =
          tail_lane_count != 0 ? tail_lane_count : wavefront_size,
      .has_partial_tail = has_partial_tail,
  };
  return true;
}

uint32_t loom_amdgpu_target_wavefront_size(const loom_target_bundle_t* bundle) {
  if (bundle == NULL || bundle->snapshot == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU preamble target snapshot");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (bundle->snapshot->subgroup_size == 0) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU preamble subgroup size");
    IREE_BUILTIN_UNREACHABLE();
  }
  return bundle->snapshot->subgroup_size;
}

uint32_t loom_amdgpu_target_native_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size) {
  IREE_ASSERT(target_facts != NULL,
              "AMDGPU subgroup communication requires AMDGPU target facts");
  const uint32_t default_wavefront_size =
      target_facts->properties.processor->wavefront.default_size;
  IREE_ASSERT(loom_amdgpu_wavefront_size_is_valid(default_wavefront_size),
              "AMDGPU subgroup communication selected a processor with an "
              "invalid default wavefront size");
  return source_wavefront_size < default_wavefront_size
             ? source_wavefront_size
             : default_wavefront_size;
}

bool loom_amdgpu_target_supports_direct_subgroup_width(
    const loom_amdgpu_target_facts_t* target_facts,
    uint32_t source_wavefront_size, uint32_t required_width) {
  const uint32_t native_subgroup_width =
      loom_amdgpu_target_native_subgroup_width(target_facts,
                                               source_wavefront_size);
  return required_width != 0 && required_width <= native_subgroup_width;
}

bool loom_amdgpu_select_subgroup_wavefront_size(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size) {
  *out_wavefront_size =
      loom_amdgpu_target_wavefront_size(loom_low_lower_context_bundle(context));
  return loom_amdgpu_wavefront_size_is_valid(*out_wavefront_size);
}

bool loom_amdgpu_select_direct_subgroup_width(loom_low_lower_context_t* context,
                                              uint32_t source_wavefront_size,
                                              uint32_t required_width) {
  if (!loom_amdgpu_wavefront_size_is_valid(source_wavefront_size)) {
    return false;
  }
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context));
  return loom_amdgpu_target_supports_direct_subgroup_width(
      target_facts, source_wavefront_size, required_width);
}

bool loom_amdgpu_select_full_wave_direct_subgroup_width(
    loom_low_lower_context_t* context, uint32_t* out_wavefront_size) {
  if (!loom_amdgpu_select_subgroup_wavefront_size(context,
                                                  out_wavefront_size)) {
    return false;
  }
  return loom_amdgpu_select_direct_subgroup_width(context, *out_wavefront_size,
                                                  *out_wavefront_size);
}
