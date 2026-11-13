// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/construction.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>
#include <cxx/views/symbol_chain.h>

#include "iree/testing/gtest.h"
#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {
namespace {

class ConstructionTest : public ::testing::Test {
 protected:
  static loom_cxx_import_options_t options() {
    loom_cxx_import_options_t result;
    loom_cxx_import_options_initialize(&result);
    return result;
  }
  // Parsed symbols keep the analysis fixture's object identities well formed.
  Source source_{IREE_SV("struct Pair { unsigned first; unsigned second; }; "
                         "void entry(Pair result, unsigned temporary);"),
                 IREE_SV("construction.cxx"), options()};
  // Definition signature supplies two distinct source object identities.
  cxx::FunctionSymbol* function_ =
      cxx::symbol_cast<cxx::OverloadSetSymbol>(
          *source_.unit().globalScope()->find("entry").begin())
          ->functions()
          .front();
  // Two-component record used for partial construction.
  cxx::Symbol* result_ = function_->parameters()[0];
  // Independent scalar used for lexical lifetime and call mapping.
  cxx::Symbol* temporary_ = function_->parameters()[1];
  // Final lexical owner consumed by the translation availability query.
  cxx::AST* owner_ = source_.unit().ast();
};

TEST_F(ConstructionTest,
       BreakPathsConstrainExitWithoutConstrainingFallthrough) {
  Construction construction;
  construction.declare(result_, 2);
  construction.assign(result_, 0, 1);
  auto initial = construction.state();
  construction.enter();
  construction.declare(temporary_, 1);
  construction.assign(temporary_, 0, 1);
  construction.assign(result_, 1, 1);
  auto continuing = construction.state();
  construction.restore(initial);
  construction.break_from_region();
  construction.restore(continuing);
  construction.leave(owner_);
  EXPECT_FALSE(construction.tracked(temporary_));
  // The merged exit includes the break, while the ordinary continuation still
  // reaches a read of the field initialized on that path.
  construction.read(result_, 1, 1);
  construction.finish_loop(construction.state());
  construction.finish_function();
  construction.resolve({{owner_, {result_, temporary_}}});
  EXPECT_TRUE(construction.initialized(owner_, result_, 0, 1));
  EXPECT_FALSE(construction.initialized(owner_, result_, 1, 1));
  EXPECT_FALSE(construction.initialized(owner_, temporary_, 0, 1));
}

TEST_F(ConstructionTest, ReadsBeforeDefinitionsAreInputs) {
  Construction construction;
  construction.declare_parameter(result_, 2);
  construction.read(result_, 1, 1);
  construction.assign(result_, 0, 1);
  construction.read(result_, 0, 1);
  construction.finish_function();
  construction.resolve({});
  const auto& parameter = *construction.parameter(result_);
  EXPECT_FALSE(parameter.inputs.contains({0, 1}));
  EXPECT_TRUE(parameter.inputs.contains({1, 2}));
  EXPECT_TRUE(parameter.outputs.contains({0, 1}));
  EXPECT_FALSE(parameter.outputs.contains({1, 2}));
}

TEST_F(ConstructionTest, ConditionalOutputPreservesTheUntakenPath) {
  Construction construction;
  construction.declare_parameter(result_, 2);
  auto incoming = construction.state();
  construction.assign(result_, 0, 1);
  construction.restore(construction.join(incoming, construction.state()));
  construction.assign(result_, 1, 1);
  construction.finish_function();
  construction.resolve({});
  const auto& parameter = *construction.parameter(result_);
  EXPECT_TRUE(parameter.outputs.contains({0, 2}));
  EXPECT_TRUE(parameter.inputs.contains({0, 1}));
  EXPECT_FALSE(parameter.inputs.contains({1, 2}));
}

TEST_F(ConstructionTest, TransitiveCallsFeedInitializationJoins) {
  Construction leaf;
  leaf.declare_parameter(temporary_, 1);
  leaf.assign(temporary_, 0, 1);
  leaf.finish_function();
  leaf.resolve({});
  Construction forwarding;
  forwarding.declare_parameter(result_, 2);
  forwarding.call(0);
  forwarding.calls()[0].arguments.push_back(
      {result_, 1, {0, 1}, leaf.parameter(temporary_), true, true});
  forwarding.finish_function();
  EXPECT_TRUE(forwarding.propagate_outputs());
  forwarding.resolve({});
  EXPECT_FALSE(forwarding.propagate_inputs());
  EXPECT_TRUE(forwarding.parameter(result_)->inputs.empty());

  Construction caller;
  caller.declare(result_, 2);
  caller.enter();
  auto incoming = caller.state();
  caller.call(0);
  caller.calls()[0].arguments.push_back(
      {result_, 0, {0, 2}, forwarding.parameter(result_), true, true});
  auto then_state = caller.state();
  caller.restore(incoming);
  caller.assign(result_, 1, 1);
  caller.restore(caller.join(then_state, caller.state()));
  caller.leave(owner_);
  caller.finish_function();
  caller.resolve({{owner_, {result_}}});
  EXPECT_FALSE(caller.initialized(owner_, result_, 0, 1));
  EXPECT_TRUE(caller.initialized(owner_, result_, 1, 1));
}

TEST_F(ConstructionTest, TransitiveInputsSubtractEarlierDefinitions) {
  Construction callee;
  callee.declare_parameter(result_, 2);
  callee.read(result_, 0, 2);
  callee.finish_function();
  callee.resolve({});
  Construction caller;
  caller.declare_parameter(result_, 2);
  caller.assign(result_, 0, 1);
  caller.call(0);
  caller.calls()[0].arguments.push_back(
      {result_, 0, {0, 2}, callee.parameter(result_), true, false});
  caller.finish_function();
  caller.resolve({});
  EXPECT_TRUE(caller.propagate_inputs());
  EXPECT_FALSE(caller.propagate_inputs());
  EXPECT_FALSE(caller.parameter(result_)->inputs.contains({0, 1}));
  EXPECT_TRUE(caller.parameter(result_)->inputs.contains({1, 2}));
}

TEST_F(ConstructionTest, ConditionalIdentityDoesNotDefineEveryReferent) {
  Construction callee;
  callee.declare_parameter(temporary_, 1);
  callee.assign(temporary_, 0, 1);
  callee.finish_function();
  callee.resolve({});
  Construction caller;
  caller.declare_parameter(result_, 2);
  caller.call(0);
  caller.calls()[0].arguments.push_back(
      {result_, 1, {0, 1}, callee.parameter(temporary_), false, true});
  caller.finish_function();
  caller.propagate_outputs();
  caller.resolve({});
  EXPECT_TRUE(caller.parameter(result_)->inputs.contains({1, 2}));
  EXPECT_TRUE(caller.parameter(result_)->outputs.contains({1, 2}));
}

TEST_F(ConstructionTest, EveryNormalReturnContributesPreservingInputs) {
  Construction construction;
  construction.declare_parameter(result_, 2);
  auto incoming = construction.state();
  construction.return_from_function();
  construction.end_path();
  EXPECT_FALSE(construction.state().reachable);
  construction.restore(incoming);
  construction.assign(result_, 0, 1);
  construction.finish_function();
  construction.resolve({});
  EXPECT_TRUE(construction.parameter(result_)->inputs.contains({0, 1}));
}

TEST_F(ConstructionTest, StorageWritesDoNotDefineCallerValueComponents) {
  Construction callee;
  callee.declare_parameter(temporary_, 1);
  callee.configure_parameter(temporary_, Construction::ParameterMode::Storage,
                             true);
  Construction caller;
  caller.declare(result_, 2);
  caller.enter();
  caller.call(0);
  caller.calls()[0].arguments.push_back(
      {result_, 1, {0, 1}, callee.parameter(temporary_), true, true});
  caller.leave(owner_);
  caller.finish_function();
  caller.resolve({{owner_, {result_}}});
  EXPECT_FALSE(caller.initialized(owner_, result_, 1, 1));
}

TEST_F(ConstructionTest, ReturningAlternativeDoesNotInventAnUninitializedPath) {
  Construction construction;
  Construction::State returned{false};
  Construction::State continuing{true};
  EXPECT_TRUE(construction.join(returned, continuing).reachable);
  EXPECT_TRUE(construction.join(continuing, returned).reachable);
  EXPECT_FALSE(construction.join(returned, returned).reachable);
}

TEST_F(ConstructionTest, CleanupReadsFollowDefinitionsOnEveryScopeExit) {
  Construction callee;
  callee.declare_parameter(temporary_, 1, /*returned=*/false);
  callee.read(temporary_, 0, 1);
  callee.finish_function();
  callee.resolve({});
  for (unsigned exit = 0; exit < 4; ++exit) {
    Construction caller;
    caller.declare_parameter(result_, 2);
    caller.enter();
    caller.defer_call(0);
    caller.calls()[0].arguments.push_back(
        {result_, 1, {0, 1}, callee.parameter(temporary_), true, false});
    caller.assign(result_, 1, 1);
    switch (exit) {
      case 0:
        break;
      case 1:
        caller.return_from_function();
        caller.end_path();
        break;
      case 2:
        caller.break_from_region();
        break;
      case 3:
        caller.continue_iteration();
        break;
    }
    caller.leave(owner_);
    caller.reach_latch();
    caller.finish_loop(caller.state());
    caller.finish_function();
    caller.resolve({});
    EXPECT_FALSE(caller.propagate_inputs());
    EXPECT_TRUE(caller.parameter(result_)->inputs.empty());
  }
}

TEST_F(ConstructionTest, ConsumedReceiverWritesDoNotBecomeCallerOutputs) {
  Construction callee;
  callee.declare_parameter(result_, 2, /*returned=*/false);
  callee.assign(result_, 0, 1);
  callee.read(result_, 0, 2);
  callee.finish_function();
  callee.resolve({});
  EXPECT_TRUE(callee.parameter(result_)->outputs.empty());
  EXPECT_FALSE(callee.parameter(result_)->inputs.contains({0, 1}));
  EXPECT_TRUE(callee.parameter(result_)->inputs.contains({1, 2}));

  Construction caller;
  caller.declare_parameter(result_, 2);
  caller.defer_call(0);
  caller.calls()[0].arguments.push_back(
      {result_, 0, {0, 2}, callee.parameter(result_), true, false});
  caller.finish_function();
  EXPECT_FALSE(caller.propagate_outputs());
  caller.resolve({});
  EXPECT_TRUE(caller.propagate_inputs());
  EXPECT_TRUE(caller.parameter(result_)->outputs.empty());
  EXPECT_FALSE(caller.parameter(result_)->inputs.contains({0, 1}));
  EXPECT_TRUE(caller.parameter(result_)->inputs.contains({1, 2}));
}

TEST_F(ConstructionTest, ResolvedAssignmentsRetainTheirSourcePosition) {
  Construction construction;
  construction.declare_parameter(result_, 2);
  construction.assign_indirect(0);
  construction.read(result_, 1, 1);
  construction.assignments()[0].destinations.push_back(
      {result_, {1, 2}, 0, true});
  construction.finish_function();
  EXPECT_TRUE(construction.propagate_outputs());
  construction.resolve({});
  EXPECT_TRUE(construction.parameter(result_)->inputs.empty());
  EXPECT_TRUE(construction.parameter(result_)->outputs.contains({1, 2}));
}

TEST_F(ConstructionTest, ArgumentChoicesClassifyOverlappingIdentities) {
  std::vector<Construction::Argument> arguments = {
      {result_, 0, {0, 2}, nullptr, false, false},
      {result_, 0, {0, 2}, nullptr, false, false},
      {temporary_, 0, {2, 3}, nullptr, false, false},
  };
  Construction::classify_definite_arguments(arguments);
  EXPECT_TRUE(arguments[0].definite);
  EXPECT_TRUE(arguments[1].definite);
  EXPECT_TRUE(arguments[2].definite);

  arguments.push_back({temporary_, 0, {1, 2}, nullptr, true, false});
  Construction::classify_definite_arguments(arguments);
  EXPECT_FALSE(arguments[0].definite);
  EXPECT_FALSE(arguments[1].definite);
  EXPECT_TRUE(arguments[2].definite);
  EXPECT_FALSE(arguments[3].definite);
}

TEST_F(ConstructionTest, WideArgumentChoicesUseEndpointSweep) {
  std::vector<Construction::Argument> arguments;
  for (size_t index = 0; index < 9; ++index) {
    arguments.push_back(
        {index == 8 ? temporary_ : result_, 0, {0, 1}, nullptr, false, false});
  }
  arguments.push_back({temporary_, 0, {1, 2}, nullptr, false, false});
  Construction::classify_definite_arguments(arguments);
  for (size_t index = 0; index < 9; ++index) {
    EXPECT_FALSE(arguments[index].definite);
  }
  EXPECT_TRUE(arguments[9].definite);
}

TEST_F(ConstructionTest, DestinationChoicesClassifySourceOverlap) {
  std::vector<Construction::Destination> destinations = {
      {result_, {0, 1}, 0, false},
      {result_, {0, 1}, 0, false},
      {temporary_, {0, 1}, 1, false},
  };
  Construction::classify_definite_destinations(destinations);
  EXPECT_TRUE(destinations[0].definite);
  EXPECT_TRUE(destinations[1].definite);
  EXPECT_TRUE(destinations[2].definite);

  destinations.push_back({temporary_, {0, 1}, 0, true});
  Construction::classify_definite_destinations(destinations);
  EXPECT_FALSE(destinations[0].definite);
  EXPECT_FALSE(destinations[1].definite);
  EXPECT_TRUE(destinations[2].definite);
  EXPECT_FALSE(destinations[3].definite);
}

}  // namespace
}  // namespace loom::cxx_import
