// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/llvmir/target.h"

#include "loom/ir/module.h"
#include "loom/ops/llvmir/ops.h"

static iree_string_view_t loom_llvmir_target_project_string(
    const loom_module_t* module, const loom_op_t* target_op, uint8_t attr_index,
    loom_target_facts_t* facts) {
  const loom_attribute_t attr = loom_op_const_attrs(target_op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_empty();
  }
  IREE_ASSERT(attr.kind == LOOM_ATTR_STRING);
  const loom_string_id_t string_id = loom_attr_as_string_id(attr);
  IREE_ASSERT(string_id < module->strings.count);
  loom_target_authored_attr_set_insert(&facts->authored_attrs, attr_index);
  return module->strings.entries[string_id];
}

static void loom_llvmir_target_facts_project(const loom_module_t* module,
                                             const loom_op_t* target_op,
                                             loom_target_facts_t* base_facts) {
  loom_llvmir_target_facts_t* facts = (loom_llvmir_target_facts_t*)base_facts;
  facts->target_triple = loom_llvmir_target_project_string(
      module, target_op, loom_llvmir_target_triple_ATTR_INDEX, base_facts);
  facts->data_layout = loom_llvmir_target_project_string(
      module, target_op, loom_llvmir_target_data_layout_ATTR_INDEX, base_facts);
  facts->target_cpu = loom_llvmir_target_project_string(
      module, target_op, loom_llvmir_target_cpu_ATTR_INDEX, base_facts);
  facts->target_features = loom_llvmir_target_project_string(
      module, target_op, loom_llvmir_target_features_ATTR_INDEX, base_facts);
}

const loom_target_fact_type_t loom_llvmir_target_fact_type = {
    .name = IREE_SVL("llvmir"),
    .storage_size = sizeof(loom_llvmir_target_facts_t),
    .project = loom_llvmir_target_facts_project,
};
