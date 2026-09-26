// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_uniformity_test_util.h"

#include "loom/analysis/control_uniformity.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom::testing {

iree_status_t CheckControlExecution(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_arena_allocator_t* analysis_arena,
    const std::vector<std::vector<uint16_t>>& successors,
    uint32_t uniform_selectors) {
  const size_t count = successors.size();
  loom_module_t* raw_module = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_module_allocate(context, IREE_SV("control"), block_pool, nullptr,
                           iree_allocator_system(), &raw_module));
  ModulePtr module(raw_module);
  loom_builder_t builder;
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(&builder, IREE_SV("paths"), &name));
  uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module.get(), name, &symbol));
  std::vector<loom_type_t> types(count, loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  loom_op_t* function_op = nullptr;
  IREE_RETURN_IF_ERROR(loom_test_func_build(
      &builder, 0, 0, 0, {0, symbol}, types.data(), types.size(), nullptr, 0,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
  auto function = loom_func_like_cast(module.get(), function_op);
  loom_region_t* region = loom_func_like_body(function);
  region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  std::vector<loom_block_t*> blocks(count);
  blocks[0] = loom_region_entry_block(region);
  for (size_t i = 1; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(module.get(), region, &blocks[i]));
  }
  std::vector<const loom_op_t*> markers;
  for (size_t i = 0; i < count; ++i) {
    loom_builder_set_block(&builder, blocks[i]);
    loom_op_t* op = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_test_use_build(&builder, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
    markers.push_back(op);
    if (successors[i].empty()) {
      IREE_RETURN_IF_ERROR(loom_test_yield_build(&builder, nullptr, 0,
                                                 LOOM_LOCATION_UNKNOWN, &op));
    } else if (successors[i].size() == 1) {
      IREE_RETURN_IF_ERROR(loom_cfg_br_build(&builder, blocks[successors[i][0]],
                                             nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                             &op));
    } else {
      IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(
          &builder, loom_block_arg_id(blocks[0], i), blocks[successors[i][0]],
          blocks[successors[i][1]], LOOM_LOCATION_UNKNOWN, &op));
    }
  }
  loom_verify_result_t verification = {};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module.get(), nullptr, &verification));
  if (verification.error_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "control oracle produced invalid IR");
  }
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&facts, analysis_arena, 8));
  for (size_t i = 0; i < count; ++i) {
    auto selector = loom_value_facts_make(0, 1, 1);
    if (uniform_selectors & (1u << i)) {
      loom_value_facts_mark_workgroup_uniform(&selector);
    } else {
      loom_value_facts_mark_lane_predicate(&selector);
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        &facts, loom_block_arg_id(blocks[0], i), selector));
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&facts, module.get(), function));
  // Each entry argument is constant throughout one lane's execution. Trace
  // every concrete assignment until exit or a repeated block. For a fixed
  // uniform assignment, two lanes may choose any two paths independently, so
  // every pair in their union is a realizable coexecution witness.
  const uint32_t mask = (1u << count) - 1;
  std::vector<uint32_t> visits_by_uniform_choice(mask + 1);
  std::vector<uint32_t> visits_by_choice(mask + 1);
  for (uint32_t choices = 0; choices <= mask; ++choices) {
    uint32_t visited = 0;
    size_t block = 0;
    while (!(visited & (1u << block))) {
      visited |= 1u << block;
      if (successors[block].empty()) {
        break;
      }
      const size_t choice =
          successors[block].size() == 2 ? ((choices >> block) & 1u) : 0;
      block = successors[block][choice];
    }
    visits_by_uniform_choice[choices & uniform_selectors] |= visited;
    visits_by_choice[choices] = visited;
  }
  std::vector<uint32_t> coexecution(count);
  for (uint32_t visited : visits_by_uniform_choice) {
    for (size_t i = 0; i < count; ++i) {
      if (visited & (1u << i)) {
        coexecution[i] |= visited;
      }
    }
  }
  loom_control_uniformity_info_t info;
  loom_control_uniformity_info_initialize(module.get(), &facts, analysis_arena,
                                          &info);
  // Check the returned complete predicate against concrete visits,
  // independently of the graph predecessor proof. Choice bit zero takes a true
  // branch.
  for (size_t block = 0; block < count; ++block) {
    loom_condition_assumption_t condition;
    if (!loom_control_uniformity_prove_single_entry(
            &info, blocks[block], LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP,
            &condition)) {
      continue;
    }
    size_t controller = 0;
    while (controller < count &&
           loom_block_arg_id(blocks[0], controller) != condition.condition) {
      ++controller;
    }
    if (controller == count) {
      return iree_make_status(IREE_STATUS_INTERNAL, "unknown entry selector");
    }
    for (uint32_t choices = 0; choices <= mask; ++choices) {
      const uint32_t visited = visits_by_choice[choices];
      const bool selected =
          ((choices & (1u << controller)) == 0) == condition.assumed_truth;
      const bool expected = (visited & (1u << controller)) && selected;
      if (expected != ((visited & (1u << block)) != 0)) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "block %zu entry predicate disagrees with "
                                "concrete path 0x%x",
                                block, choices);
      }
    }
  }
  for (size_t first = 0; first < count; ++first) {
    for (size_t second = 0; second < count; ++second) {
      if (!(coexecution[first] & (1u << second))) {
        continue;
      }
      bool proven = false;
      IREE_RETURN_IF_ERROR(
          loom_control_uniformity_prove_mutually_exclusive_execution(
              &info, 1, &markers[first], 1, &markers[second],
              LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP, &proven));
      if (proven) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "coexecuting blocks %zu and %zu claimed "
                                "exclusive (uniform selectors 0x%x)",
                                first, second, uniform_selectors);
      }
    }
  }
  return iree_ok_status();
}

}  // namespace loom::testing
