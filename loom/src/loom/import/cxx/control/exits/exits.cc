// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/exits/exits.h"

namespace loom::cxx_import {

StatementExits StatementExits::branch(StatementExits then_exits,
                                      StatementExits else_exits) {
  return {then_exits.paths_ | else_exits.paths_,
          then_exits.requires_callable_exit() ||
              else_exits.requires_callable_exit()};
}

void StatementExits::append(StatementExits child) {
  if (falls_through()) {
    paths_ = (paths_ & ~Fallthrough) | child.paths_;
  }
  callable_count_ += child.requires_callable_exit();
}

StatementExits StatementExits::leave_loop(LoopTest test,
                                          bool always_true) const {
  unsigned outcomes = paths_ & (Return | Diverge);
  if ((paths_ & Break) || (test == LoopTest::BeforeBody && !always_true)) {
    outcomes |= Fallthrough;
  }
  if (reaches_latch()) {
    outcomes |= always_true ? Diverge : Fallthrough;
  }
  return structured(outcomes);
}

void SwitchExits::append_clause(StatementExits clause) {
  paths_ = (paths_ & ~StatementExits::Fallthrough) | clause.paths_;
}

StatementExits SwitchExits::finish(bool has_default) const {
  unsigned outcomes = paths_ & ~StatementExits::Break;
  if (!has_default || (paths_ & StatementExits::Break)) {
    outcomes |= StatementExits::Fallthrough;
  }
  return StatementExits::structured(outcomes);
}

}  // namespace loom::cxx_import
