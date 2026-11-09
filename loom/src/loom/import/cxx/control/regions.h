// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_REGIONS_H_
#define LOOM_IMPORT_CXX_CONTROL_REGIONS_H_

#include <array>
#include <span>

#include "loom/ops/builder.h"

namespace loom::cxx_import {

// Builds conditional arms before their result signature is known. Source
// evaluation discovers the actual lvalue coordinates; no placeholder values or
// second source traversal are needed. The module owns both detached regions.
class ConditionalRegions {
 public:
  explicit ConditionalRegions(loom_builder_t& builder);

  // Enters one detached arm and returns the previous insertion point. The
  // caller supplies its terminator and restores the returned point.
  loom_builder_ip_t enter(size_t arm);

  // Attaches the completed arms to one conditional at the current insertion
  // point. Result reservations, if any, belong to this operation alone.
  loom_op_t* build(loom_value_id_t condition,
                   std::span<const loom_type_t> result_types,
                   loom_location_id_t source);

 private:
  // Borrowed module construction context, which outlives both regions.
  loom_builder_t& builder_;
  // Then and else regions, each containing one ordinary source block.
  std::array<loom_region_t*, 2> regions_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_REGIONS_H_
