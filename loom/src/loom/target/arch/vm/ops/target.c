// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/ops/target.h"

#include <inttypes.h>

#include "loom/target/arch/vm/ops/ops.h"

iree_status_t loom_vm_target_build_core(loom_builder_t* builder,
                                        loom_symbol_ref_t symbol,
                                        loom_location_id_t location,
                                        loom_op_t** out_target_op) {
  return loom_vm_target_build(
      builder, /*build_flags=*/0, LOOM_VM_TARGET_KIND_CORE, symbol,
      /*codegen_format=*/0, /*artifact_format=*/0,
      /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
      /*offset_bitwidth=*/0, /*max_workgroup_size_x=*/0,
      /*max_workgroup_size_y=*/0, /*max_workgroup_size_z=*/0,
      /*max_flat_workgroup_size=*/0, /*max_workgroup_storage_bytes=*/0,
      /*subgroup_size=*/0, /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
      /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
      /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
      /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
      /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
      /*memory_space_constant=*/0, /*memory_space_private=*/0,
      /*memory_space_host=*/0, /*memory_space_descriptor=*/0,
      /*abi=*/0, /*export_symbol=*/LOOM_STRING_ID_INVALID,
      /*linkage=*/0, /*contract_set_key=*/LOOM_STRING_ID_INVALID,
      /*contract_feature_bits=*/0, location, out_target_op);
}

iree_status_t loom_vm_target_build_core_with_execution_limits(
    loom_builder_t* builder, loom_symbol_ref_t symbol,
    const loom_target_snapshot_t* source_snapshot, loom_location_id_t location,
    loom_op_t** out_target_op) {
  *out_target_op = NULL;
  if (source_snapshot->max_workgroup_storage_bytes > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "maximum workgroup storage size %" PRIu64
                            " exceeds the target attribute domain",
                            source_snapshot->max_workgroup_storage_bytes);
  }
  if (source_snapshot->max_flat_grid_size > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "maximum flat grid size %" PRIu64
                            " exceeds the target attribute domain",
                            source_snapshot->max_flat_grid_size);
  }

  // Every projected field is explicit even when zero. The VM Core row has
  // execution defaults such as subgroup_size=1; omitting an unknown device
  // value would inherit that VM default and change the launch program's
  // observable target facts.
  const loom_vm_target_build_flags_t build_flags =
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_X |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Y |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_SIZE_Z |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_WORKGROUP_SIZE |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_STORAGE_BYTES |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_X |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Y |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_GRID_SIZE_Z |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_FLAT_GRID_SIZE |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_X |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Y |
      LOOM_VM_TARGET_BUILD_FLAG_HAS_MAX_WORKGROUP_COUNT_Z;
  return loom_vm_target_build(
      builder, build_flags, LOOM_VM_TARGET_KIND_CORE, symbol,
      /*codegen_format=*/0, /*artifact_format=*/0,
      /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
      /*offset_bitwidth=*/0, source_snapshot->max_workgroup_size.x,
      source_snapshot->max_workgroup_size.y,
      source_snapshot->max_workgroup_size.z,
      source_snapshot->max_flat_workgroup_size,
      (int64_t)source_snapshot->max_workgroup_storage_bytes,
      source_snapshot->subgroup_size, source_snapshot->max_grid_size.x,
      source_snapshot->max_grid_size.y, source_snapshot->max_grid_size.z,
      (int64_t)source_snapshot->max_flat_grid_size,
      source_snapshot->max_workgroup_count.x,
      source_snapshot->max_workgroup_count.y,
      source_snapshot->max_workgroup_count.z,
      /*memory_space_generic=*/0, /*memory_space_global=*/0,
      /*memory_space_workgroup=*/0, /*memory_space_constant=*/0,
      /*memory_space_private=*/0, /*memory_space_host=*/0,
      /*memory_space_descriptor=*/0, /*abi=*/0,
      /*export_symbol=*/LOOM_STRING_ID_INVALID, /*linkage=*/0,
      /*contract_set_key=*/LOOM_STRING_ID_INVALID,
      /*contract_feature_bits=*/0, location, out_target_op);
}
