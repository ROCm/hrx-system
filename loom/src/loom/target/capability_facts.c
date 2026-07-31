// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/capability_facts.h"

#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

static bool loom_target_capability_u32(uint32_t value, uint64_t* out_value) {
  if (value == 0) return false;
  *out_value = value;
  return true;
}

static bool loom_target_capability_u64(uint64_t value, uint64_t* out_value) {
  if (value == 0) return false;
  *out_value = value;
  return true;
}

static uint32_t loom_target_workgroup_size_dim(
    const loom_target_workgroup_size_t* size, iree_string_view_t key) {
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_size_x"))) {
    return size->x;
  }
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_size_y"))) {
    return size->y;
  }
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_size_z"))) {
    return size->z;
  }
  return 0;
}

static uint32_t loom_target_grid_size_dim(const loom_target_grid_size_t* size,
                                          iree_string_view_t key) {
  if (iree_string_view_equal(key, IREE_SV("max_grid_size_x"))) {
    return size->x;
  }
  if (iree_string_view_equal(key, IREE_SV("max_grid_size_y"))) {
    return size->y;
  }
  if (iree_string_view_equal(key, IREE_SV("max_grid_size_z"))) {
    return size->z;
  }
  return 0;
}

static uint32_t loom_target_workgroup_count_dim(
    const loom_target_workgroup_count_limit_t* count, iree_string_view_t key) {
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_count_x"))) {
    return count->x;
  }
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_count_y"))) {
    return count->y;
  }
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_count_z"))) {
    return count->z;
  }
  return 0;
}

static bool loom_target_snapshot_query_u64(
    const loom_target_snapshot_t* snapshot, iree_string_view_t key,
    uint64_t* out_value) {
  if (iree_string_view_equal(key, IREE_SV("default_pointer_bitwidth"))) {
    return loom_target_capability_u32(snapshot->default_pointer_bitwidth,
                                      out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("index_bitwidth"))) {
    return loom_target_capability_u32(snapshot->index_bitwidth, out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("offset_bitwidth"))) {
    return loom_target_capability_u32(snapshot->offset_bitwidth, out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("subgroup_size"))) {
    return loom_target_capability_u32(snapshot->subgroup_size, out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("max_flat_workgroup_size"))) {
    return loom_target_capability_u32(snapshot->max_flat_workgroup_size,
                                      out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("max_workgroup_storage_bytes"))) {
    return loom_target_capability_u64(snapshot->max_workgroup_storage_bytes,
                                      out_value);
  }
  if (iree_string_view_equal(key, IREE_SV("max_flat_grid_size"))) {
    return loom_target_capability_u64(snapshot->max_flat_grid_size, out_value);
  }

  uint32_t dim_value =
      loom_target_workgroup_size_dim(&snapshot->max_workgroup_size, key);
  if (dim_value != 0) {
    return loom_target_capability_u32(dim_value, out_value);
  }
  dim_value = loom_target_grid_size_dim(&snapshot->max_grid_size, key);
  if (dim_value != 0) {
    return loom_target_capability_u32(dim_value, out_value);
  }
  dim_value =
      loom_target_workgroup_count_dim(&snapshot->max_workgroup_count, key);
  if (dim_value != 0) {
    return loom_target_capability_u32(dim_value, out_value);
  }
  return false;
}

bool loom_target_fact_context_query_u64(const loom_fact_context_t* context,
                                        iree_string_view_t namespace_name,
                                        iree_string_view_t key,
                                        uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!context || !context->target_facts) {
    return false;
  }
  const loom_target_snapshot_t* snapshot =
      &context->target_facts->storage.snapshot;
  if (iree_string_view_equal(namespace_name, IREE_SV("target"))) {
    return loom_target_snapshot_query_u64(snapshot, key, out_value);
  }
  return false;
}
