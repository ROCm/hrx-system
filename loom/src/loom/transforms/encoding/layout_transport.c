// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/encoding/layout_transport.h"

#include "loom/target/pass_environment.h"
#include "loom/transforms/boundary/projection_driver.h"
#include "loom/transforms/encoding/cfg_layout_projection.h"

#define LOOM_CFG_LAYOUT_TRANSPORT_STATISTICS(V, statistics_type)       \
  V(statistics_type, layouts_decomposed, "layouts-decomposed",         \
    "Number of dynamic strided-layout block arguments decomposed.")    \
  V(statistics_type, stride_args_inserted, "stride-args-inserted",     \
    "Number of index-valued stride block arguments inserted.")         \
  V(statistics_type, branch_edges_rewritten, "branch-edges-rewritten", \
    "Number of CFG branch payloads rewritten for scalar layout transport.")

LOOM_PASS_STATISTICS_DEFINE(loom_cfg_layout_transport_statistics,
                            loom_cfg_layout_transport_statistics_t,
                            LOOM_CFG_LAYOUT_TRANSPORT_STATISTICS)

static const loom_pass_info_t loom_decompose_cfg_layout_transports_info = {
    .name = IREE_SVL("decompose-cfg-layout-transports"),
    .description =
        IREE_SVL("Carry dynamic strided layouts as scalar CFG payloads."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_cfg_layout_transport_statistics_layout,
};

const loom_pass_info_t* loom_decompose_cfg_layout_transports_pass_info(void) {
  return &loom_decompose_cfg_layout_transports_info;
}

iree_status_t loom_decompose_cfg_layout_transports_run(loom_pass_t* pass,
                                                       loom_module_t* module) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_function_version_list_t* version_list =
      capability ? loom_target_pass_capability_function_versions(capability)
                 : NULL;
  const loom_boundary_projection_rule_t* rules[] = {
      loom_cfg_layout_boundary_projection_rule(),
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
    loom_cfg_layout_transport_statistics_t* statistics =
        loom_cfg_layout_transport_statistics(pass);
    IREE_ASSERT_EQ(projection_statistics.rule_count, IREE_ARRAYSIZE(rules));
    statistics->layouts_decomposed +=
        projection_statistics.rules[0].projections;
    statistics->stride_args_inserted +=
        projection_statistics.rules[0].components;
    statistics->branch_edges_rewritten +=
        projection_statistics.cfg_edges_rewritten;
  }
  return status;
}
