// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/regions.h"

#include "loom/import/cxx/source/error.h"
#include "loom/ir/module.h"
#include "loom/ops/scf/ops.h"

namespace loom::cxx_import {

ConditionalRegions::ConditionalRegions(loom_builder_t& builder)
    : builder_(builder) {
  for (auto& region : regions_) {
    check(loom_module_allocate_region(builder_.module, 1, &region));
  }
}

loom_builder_ip_t ConditionalRegions::enter(size_t arm) {
  return loom_builder_enter_region(&builder_, nullptr, regions_[arm]);
}

loom_op_t* ConditionalRegions::build(loom_value_id_t condition,
                                     std::span<const loom_type_t> result_types,
                                     loom_location_id_t source) {
  check(loom_builder_check_count_range(result_types.size(), UINT16_MAX,
                                       IREE_SV("conditional results")));
  loom_op_t* op;
  check(loom_builder_allocate_op(&builder_, LOOM_OP_SCF_IF, 1,
                                 result_types.size(), 2, 0, 0, source, &op));
  loom_op_operands(op)[0] = condition;
  check(loom_builder_define_results(&builder_, result_types.data(),
                                    result_types.size(), loom_op_results(op)));
  for (size_t arm = 0; arm < regions_.size(); ++arm) {
    loom_op_set_region(op, arm, regions_[arm]);
    loom_op_t* child;
    loom_block_for_each_op(loom_region_entry_block(regions_[arm]), child) {
      child->parent_op = op;
    }
    loom_module_record_region_summaries(op, regions_[arm]);
  }
  check(loom_builder_finalize_op(&builder_, op));
  return op;
}

}  // namespace loom::cxx_import
