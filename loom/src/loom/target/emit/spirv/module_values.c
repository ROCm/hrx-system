// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_values.h"

#include <string.h>

iree_status_t loom_spirv_module_value_table_initialize(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_value_table_t* out_table,
    iree_arena_allocator_t* scratch_arena) {
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT_ARGUMENT(out_table);
  IREE_ASSERT_ARGUMENT(scratch_arena);

  *out_table = (loom_spirv_module_value_table_t){
      .value_domain = value_domain,
      .ref_count = value_domain->value_count,
  };
  if (value_domain->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, value_domain->value_count, sizeof(*out_table->refs),
      (void**)&out_table->refs));
  memset(out_table->refs, 0,
         value_domain->value_count * sizeof(*out_table->refs));
  return iree_ok_status();
}

static loom_value_ordinal_t loom_spirv_module_value_table_ordinal(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  IREE_ASSERT_LT(value_ordinal, table->ref_count);
  return value_ordinal;
}

void loom_spirv_module_value_table_define(
    loom_spirv_module_value_table_t* table, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref) {
  IREE_ASSERT_ARGUMENT(table);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  const loom_spirv_module_value_ref_t existing_ref = table->refs[value_ordinal];
  if (existing_ref.id != 0) {
    IREE_ASSERT_EQ(existing_ref.id, value_ref.id);
    IREE_ASSERT_EQ(existing_ref.type_id, value_ref.type_id);
    IREE_ASSERT(loom_spirv_value_type_equal(existing_ref.value_type,
                                            value_ref.value_type));
  }
  table->refs[value_ordinal] = value_ref;
}

uint32_t loom_spirv_module_value_table_reserve(
    loom_spirv_module_value_table_t* table,
    loom_spirv_module_builder_t* builder, loom_value_id_t value_id,
    uint32_t type_id, loom_spirv_value_type_t value_type) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(builder);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  loom_spirv_module_value_ref_t value_ref = table->refs[value_ordinal];
  if (value_ref.id != 0) {
    IREE_ASSERT_EQ(value_ref.type_id, type_id);
    IREE_ASSERT(loom_spirv_value_type_equal(value_ref.value_type, value_type));
    return value_ref.id;
  }

  const uint32_t result_id = loom_spirv_module_builder_allocate_id(builder);
  value_ref = (loom_spirv_module_value_ref_t){
      .id = result_id,
      .type_id = type_id,
      .value_type = value_type,
  };
  loom_spirv_module_value_table_define(table, value_id, value_ref);
  return result_id;
}

loom_spirv_module_value_ref_t loom_spirv_module_value_table_lookup(
    const loom_spirv_module_value_table_t* table, loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(table);

  const loom_value_ordinal_t value_ordinal =
      loom_spirv_module_value_table_ordinal(table, value_id);
  const loom_spirv_module_value_ref_t value_ref = table->refs[value_ordinal];
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
         value_ordinal < table->ref_count && table->refs[value_ordinal].id != 0;
}
