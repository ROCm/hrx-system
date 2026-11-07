// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/cfg_tuple_decomposition.h"

#include "loom/codegen/low/transforms/cfg_tuple_projection.h"
#include "loom/transforms/boundary/projection_driver.h"

#define LOOM_LOW_DECOMPOSE_CFG_TUPLES_STATISTICS(V, statistics_type)       \
  V(statistics_type, block_args_decomposed, "block-args-decomposed",       \
    "Number of tuple-valued low CFG block arguments decomposed.")          \
  V(statistics_type, lane_block_args_inserted, "lane-block-args-inserted", \
    "Number of scalar lane block arguments inserted.")                     \
  V(statistics_type, branch_edges_decomposed, "branch-edges-decomposed",   \
    "Number of low.br edges rewritten to forward scalar lanes.")           \
  V(statistics_type, slices_removed, "slices-removed",                     \
    "Number of low.slice projections replaced by component observations.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_decompose_cfg_tuples_statistics,
                            loom_low_decompose_cfg_tuples_statistics_t,
                            LOOM_LOW_DECOMPOSE_CFG_TUPLES_STATISTICS)

static const loom_pass_info_t kPassInfo = {
    .name = IREE_SVL("low-decompose-cfg-tuples"),
    .description =
        IREE_SVL("Decompose partially sliced low CFG register tuples."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_low_decompose_cfg_tuples_statistics_layout,
};

const loom_pass_info_t* loom_low_decompose_cfg_tuples_pass_info(void) {
  return &kPassInfo;
}

iree_status_t loom_low_decompose_cfg_tuples_run(loom_pass_t* pass,
                                                loom_module_t* module) {
  const loom_boundary_projection_rule_t* rules[] = {
      loom_low_cfg_tuple_boundary_projection_rule(),
  };
  loom_boundary_projection_statistics_t projection_statistics;
  iree_status_t status =
      loom_boundary_projection_run(pass, module, /*version_list=*/NULL,
                                   (loom_boundary_projection_rule_list_t){
                                       .values = rules,
                                       .count = IREE_ARRAYSIZE(rules),
                                   },
                                   /*plan_sink=*/NULL, &projection_statistics);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(projection_statistics.rule_count, IREE_ARRAYSIZE(rules));
    const loom_boundary_projection_rule_statistics_t* rule_statistics =
        &projection_statistics.rules[0];
    loom_low_decompose_cfg_tuples_statistics_t* statistics =
        loom_low_decompose_cfg_tuples_statistics(pass);
    statistics->block_args_decomposed += rule_statistics->projections;
    statistics->lane_block_args_inserted += rule_statistics->components;
    statistics->branch_edges_decomposed +=
        projection_statistics.cfg_edges_rewritten;
    statistics->slices_removed += rule_statistics->destination_uses_rewritten;
  }
  return status;
}
