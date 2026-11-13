// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_values.h"

#include <string.h>

enum {
  LOOM_SPIRV_VALUE_SEGMENT_SHIFT = 10,
  LOOM_SPIRV_VALUE_SEGMENT_CAPACITY = 1u << LOOM_SPIRV_VALUE_SEGMENT_SHIFT,
  LOOM_SPIRV_VALUE_SEGMENT_MASK = LOOM_SPIRV_VALUE_SEGMENT_CAPACITY - 1u,
};

struct loom_spirv_module_value_segment_t {
  // Complete refs in one directly indexed ordinal range.
  loom_spirv_module_value_ref_t refs[LOOM_SPIRV_VALUE_SEGMENT_CAPACITY];
};

static_assert(sizeof(loom_spirv_module_value_ref_t) == 16,
              "SPIR-V value refs must remain compact");
static_assert(sizeof(loom_spirv_module_value_segment_t) == 16 * 1024,
              "SPIR-V value segments must fit the compiler arena block pool");

iree_status_t loom_spirv_module_value_table_initialize(
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* scratch_arena,
    loom_spirv_module_value_table_t* out_table) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  *out_table = (loom_spirv_module_value_table_t){
      .value_domain = value_domain,
      .ref_count = value_domain->value_count,
      .ref_segment_count =
          (value_domain->value_count + LOOM_SPIRV_VALUE_SEGMENT_MASK) >>
          LOOM_SPIRV_VALUE_SEGMENT_SHIFT,
  };
  if (out_table->ref_segment_count == 0) {
    return iree_ok_status();
  }

  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  iree_status_t status = iree_arena_allocate_array(
      scratch_arena, out_table->ref_segment_count,
      sizeof(*out_table->ref_segments), (void**)&out_table->ref_segments);
  for (iree_host_size_t i = 0;
       i < out_table->ref_segment_count && iree_status_is_ok(status); ++i) {
    status =
        iree_arena_allocate(scratch_arena, sizeof(*out_table->ref_segments[i]),
                            (void**)&out_table->ref_segments[i]);
    if (iree_status_is_ok(status)) {
      memset(out_table->ref_segments[i], 0,
             sizeof(*out_table->ref_segments[i]));
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_checkpoint_restore(&checkpoint);
    *out_table = (loom_spirv_module_value_table_t){0};
  }
  return status;
}

static loom_value_ordinal_t loom_spirv_module_value_table_ordinal(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  IREE_ASSERT_LT(value_ordinal, table->ref_count);
  return value_ordinal;
}

static loom_spirv_module_value_ref_t* loom_spirv_module_value_table_ref(
    loom_spirv_module_value_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  const iree_host_size_t segment_index =
      value_ordinal >> LOOM_SPIRV_VALUE_SEGMENT_SHIFT;
  IREE_ASSERT_LT(segment_index, table->ref_segment_count);
  return &table->ref_segments[segment_index]
              ->refs[value_ordinal & LOOM_SPIRV_VALUE_SEGMENT_MASK];
}

static const loom_spirv_module_value_ref_t*
loom_spirv_module_value_table_const_ref(
    const loom_spirv_module_value_table_t* table,
    loom_value_ordinal_t value_ordinal) {
  const iree_host_size_t segment_index =
      value_ordinal >> LOOM_SPIRV_VALUE_SEGMENT_SHIFT;
  IREE_ASSERT_LT(segment_index, table->ref_segment_count);
  return &table->ref_segments[segment_index]
              ->refs[value_ordinal & LOOM_SPIRV_VALUE_SEGMENT_MASK];
}

void loom_spirv_module_value_table_define(
    loom_spirv_module_value_table_t* table, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref) {
  IREE_ASSERT_ARGUMENT(table);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  loom_spirv_module_value_ref_t* stored_ref =
      loom_spirv_module_value_table_ref(table, value_ordinal);
  if (stored_ref->id != 0) {
    IREE_ASSERT_EQ(stored_ref->id, value_ref.id);
    IREE_ASSERT_EQ(stored_ref->type_id, value_ref.type_id);
    IREE_ASSERT(loom_spirv_value_type_equal(stored_ref->value_type,
                                            value_ref.value_type));
  }
  *stored_ref = value_ref;
}

uint32_t loom_spirv_module_value_table_reserve(
    loom_spirv_module_value_table_t* table,
    loom_spirv_module_builder_t* builder, loom_value_id_t value_id,
    uint32_t type_id, loom_spirv_value_type_t value_type) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(builder);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  loom_spirv_module_value_ref_t* value_ref =
      loom_spirv_module_value_table_ref(table, value_ordinal);
  if (value_ref->id != 0) {
    IREE_ASSERT_EQ(value_ref->type_id, type_id);
    IREE_ASSERT(loom_spirv_value_type_equal(value_ref->value_type, value_type));
    return value_ref->id;
  }

  const uint32_t result_id = loom_spirv_module_builder_allocate_id(builder);
  *value_ref = (loom_spirv_module_value_ref_t){
      .id = result_id,
      .type_id = type_id,
      .value_type = value_type,
  };
  return result_id;
}

loom_spirv_module_value_ref_t loom_spirv_module_value_table_lookup(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(table);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  const loom_spirv_module_value_ref_t value_ref =
      *loom_spirv_module_value_table_const_ref(table, value_ordinal);
  IREE_ASSERT_NE(value_ref.id, 0u);
  IREE_ASSERT_NE(value_ref.type_id, 0u);
  return value_ref;
}

bool loom_spirv_module_value_table_exists(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(table);

  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  return value_ordinal != LOOM_VALUE_ORDINAL_INVALID &&
         value_ordinal < table->ref_count &&
         loom_spirv_module_value_table_const_ref(table, value_ordinal)->id != 0;
}
