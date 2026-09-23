// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/boundary_transport.h"

#include "loom/target/pass_environment.h"
#include "loom/transforms/boundary/projection_driver.h"
#include "loom/transforms/view/boundary_projection.h"

#define LOOM_VIEW_BOUNDARY_TRANSPORT_STATISTICS(V, statistics_type) \
  V(statistics_type, functions_rewritten, "functions-rewritten",    \
    "Number of function signatures rewritten for physical views.")  \
  V(statistics_type, calls_rewritten, "calls-rewritten",            \
    "Number of semantic calls rewritten for physical views.")       \
  V(statistics_type, returns_rewritten, "returns-rewritten",        \
    "Number of returns rewritten for physical views.")              \
  V(statistics_type, cfg_edges_rewritten, "cfg-edges-rewritten",    \
    "Number of CFG edges rewritten for physical views.")            \
  V(statistics_type, views_decomposed, "views-decomposed",          \
    "Number of semantic view carriers decomposed into physical values.")

LOOM_PASS_STATISTICS_DEFINE(loom_view_boundary_transport_statistics,
                            loom_view_boundary_transport_statistics_t,
                            LOOM_VIEW_BOUNDARY_TRANSPORT_STATISTICS)

static const loom_pass_info_t kPassInfo = {
    .name = IREE_SVL("decompose-view-boundaries"),
    .description =
        IREE_SVL("Decompose retained views into target-selected carriers."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_view_boundary_transport_statistics_layout,
};

const loom_pass_info_t* loom_decompose_view_boundaries_pass_info(void) {
  return &kPassInfo;
}

iree_status_t loom_decompose_view_boundaries_run(loom_pass_t* pass,
                                                 loom_module_t* module) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  IREE_ASSERT(capability != NULL);
  const loom_function_version_list_t* version_list =
      loom_target_pass_capability_function_versions(capability);
  IREE_ASSERT(version_list != NULL);
  IREE_ASSERT(loom_target_pass_capability_function_version_owner(capability));

  const loom_boundary_projection_rule_t* rules[] = {
      loom_view_boundary_projection_rule(),
  };
  loom_boundary_projection_statistics_t projection_statistics;
  iree_status_t status =
      loom_boundary_projection_run(pass, module, version_list,
                                   (loom_boundary_projection_rule_list_t){
                                       .values = rules,
                                       .count = IREE_ARRAYSIZE(rules),
                                   },
                                   &projection_statistics);

  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(projection_statistics.rule_count, IREE_ARRAYSIZE(rules));
    loom_view_boundary_transport_statistics_t* statistics =
        loom_view_boundary_transport_statistics(pass);
    statistics->functions_rewritten +=
        projection_statistics.functions_rewritten;
    statistics->calls_rewritten += projection_statistics.calls_rewritten;
    statistics->returns_rewritten += projection_statistics.returns_rewritten;
    statistics->cfg_edges_rewritten +=
        projection_statistics.cfg_edges_rewritten;
    statistics->views_decomposed += projection_statistics.rules[0].projections;
  }
  return status;
}
