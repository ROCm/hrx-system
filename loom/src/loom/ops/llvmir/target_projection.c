// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/target/arch/llvmir/facts.h"

static iree_string_view_t loom_llvmir_target_project_string(
    const loom_target_record_view_t* record, uint8_t attribute_index,
    bool* out_authored) {
  const loom_attribute_t attr =
      loom_target_record_view_attribute(record, attribute_index);
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_empty();
  }
  *out_authored = true;
  return loom_target_record_view_string(record, attr);
}

static void loom_llvmir_target_facts_project(
    const loom_target_record_view_t* record, loom_target_facts_t* base_facts) {
  loom_llvmir_target_facts_t* facts = (loom_llvmir_target_facts_t*)base_facts;
  facts->target_triple = loom_llvmir_target_project_string(
      record, loom_llvmir_target_triple_ATTR_INDEX,
      &facts->authored.target_triple);
  facts->data_layout = loom_llvmir_target_project_string(
      record, loom_llvmir_target_data_layout_ATTR_INDEX,
      &facts->authored.data_layout);
  facts->target_cpu = loom_llvmir_target_project_string(
      record, loom_llvmir_target_cpu_ATTR_INDEX, &facts->authored.target_cpu);
  facts->target_features = loom_llvmir_target_project_string(
      record, loom_llvmir_target_features_ATTR_INDEX,
      &facts->authored.target_features);
}

const loom_target_fact_projector_t loom_llvmir_target_fact_projector = {
    .project = loom_llvmir_target_facts_project,
};
