// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_async_legality.h"

#include "loom/analysis/kernel_async_legality.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"

#define LOOM_KERNEL_ASYNC_LEGALITY_STATISTICS(V, statistics_type) \
  V(statistics_type, blocks_checked, "blocks-checked",            \
    "Number of blocks checked for async stream legality.")        \
  V(statistics_type, groups_checked, "groups-checked",            \
    "Number of kernel.async.group ops checked.")                  \
  V(statistics_type, waits_checked, "waits-checked",              \
    "Number of kernel.async.wait ops checked.")

LOOM_PASS_STATISTICS_DEFINE(loom_kernel_async_legality_statistics,
                            loom_kernel_async_legality_statistics_t,
                            LOOM_KERNEL_ASYNC_LEGALITY_STATISTICS)

static const loom_pass_info_t loom_kernel_async_legality_pass_info_storage = {
    .name = IREE_SVL("kernel-async-legality"),
    .description =
        IREE_SVL("Prove kernel async group/wait streams are lowerable."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_kernel_async_legality_statistics_layout,
};

const loom_pass_info_t* loom_kernel_async_legality_pass_info(void) {
  return &loom_kernel_async_legality_pass_info_storage;
}

iree_status_t loom_kernel_async_legality_run(loom_pass_t* pass,
                                             loom_module_t* module,
                                             loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_local_value_domain_t value_domain = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, body, pass->arena, &value_domain);
  loom_value_fact_table_t* fact_table = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_pass_value_facts_acquire(
        pass, module, loom_pass_value_fact_scope_function(function),
        &fact_table);
  }
  loom_kernel_async_legality_options_t options = {
      .arena = pass->arena,
      .value_domain = &value_domain,
      .fact_table = fact_table,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
  };
  loom_kernel_async_legality_result_t result = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_async_legality_verify_function(module, function,
                                                        &options, &result);
  }
  loom_local_value_domain_release(&value_domain);
  IREE_RETURN_IF_ERROR(status);

  loom_kernel_async_legality_statistics_t* statistics =
      loom_kernel_async_legality_statistics(pass);
  statistics->blocks_checked += (int64_t)result.blocks_checked;
  statistics->groups_checked += (int64_t)result.groups_checked;
  statistics->waits_checked += (int64_t)result.waits_checked;
  return iree_ok_status();
}
