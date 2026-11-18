// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/exits/exits.h"

#include "iree/testing/gtest.h"

namespace loom::cxx_import {
namespace {

TEST(StatementExitsTest, SequenceRetainsLexicalCountBeyondAnExit) {
  auto returns = StatementExits::terminal(StatementExits::Return);
  auto diverges = StatementExits::terminal(StatementExits::Diverge);
  auto loop_return = returns.leave_loop(StatementExits::LoopTest::AfterBody,
                                        /*always_true=*/false);
  EXPECT_FALSE(returns.requires_callable_exit());
  EXPECT_TRUE(loop_return.requires_callable_exit());

  StatementExits body;
  body.append(returns);
  body.append(diverges);
  body.append(loop_return);
  EXPECT_EQ(body.returns(), ExitFlow::All);
  EXPECT_FALSE(body.falls_through());
  EXPECT_EQ(body.callable_exit_count(), 2u);
  StatementExits outer;
  outer.append(body);
  EXPECT_EQ(outer.callable_exit_count(), 1u);
}

TEST(StatementExitsTest, SwitchAccountsForEverySelectableEntry) {
  using Exits = StatementExits;
  auto returns = Exits::terminal(Exits::Return);
  auto continues = Exits::terminal(Exits::Continue);
  auto breaks = Exits::terminal(Exits::Break);

  SwitchExits selection;
  selection.append_clause({});  // case 0 falls into case 1.
  selection.append_clause(returns);
  EXPECT_EQ(selection.finish(true).returns(), ExitFlow::All);
  EXPECT_TRUE(selection.finish(true).requires_callable_exit());
  EXPECT_TRUE(selection.finish(false).falls_through());  // Unmatched selector.

  selection.append_clause(Exits::branch(continues, breaks));
  auto with_break = selection.finish(true);
  EXPECT_EQ(with_break.returns(), ExitFlow::Some);
  EXPECT_EQ(with_break.continues(), ExitFlow::Some);
  EXPECT_EQ(with_break.breaks(), ExitFlow::None);
  EXPECT_TRUE(with_break.falls_through());

  // Completing another clause does not erase earlier breaks.
  selection.append_clause(returns);
  EXPECT_EQ(selection.finish(true), with_break);
}

TEST(StatementExitsTest, LastClauseControlsFallthroughAtSwitchTail) {
  auto diverges = StatementExits::terminal(StatementExits::Diverge);
  SwitchExits selection;
  selection.append_clause(StatementExits::branch(diverges, {}));
  EXPECT_TRUE(selection.finish(true).falls_through());
  selection.append_clause(diverges);
  auto closed = selection.finish(true);
  EXPECT_FALSE(closed.falls_through());
  EXPECT_TRUE(closed.requires_callable_exit());
  EXPECT_EQ(closed.returns(), ExitFlow::None);
}

}  // namespace
}  // namespace loom::cxx_import
