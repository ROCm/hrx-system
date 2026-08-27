// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/ops/target.h"

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
