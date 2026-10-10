// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_barrier_lifetime.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

class KernelBarrierLifetimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_kernel_barrier_lifetime_result_t Analyze(const char* source) {
    loom_module_t* raw_module = nullptr;
    const loom_text_parse_options_t parse_options = {};
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("barrier_lifetime_test.loom"),
        &context_, &block_pool_, &parse_options, &raw_module));
    testing::ModulePtr module(raw_module);
    EXPECT_NE(module, nullptr);

    const loom_string_id_t name_id =
        loom_module_lookup_string(module.get(), IREE_SV("test"));
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module.get(), name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    const loom_func_like_t function = loom_func_like_cast(
        module.get(), module->symbols.entries[symbol_id].defining_op);
    EXPECT_TRUE(loom_func_like_isa(function));

    loom_value_fact_table_t facts = {};
    IREE_EXPECT_OK(loom_value_fact_table_initialize(&facts, &analysis_arena_,
                                                    module->values.count));
    loom_type_registry_configure_fact_context(&facts.context);
    IREE_EXPECT_OK(
        loom_value_fact_table_compute(&facts, module.get(), function));
    loom_local_value_domain_t value_domain = {};
    IREE_EXPECT_OK(loom_local_value_domain_acquire_for_region_tree(
        module.get(), loom_func_like_body(function), &analysis_arena_,
        &value_domain));
    const loom_kernel_barrier_lifetime_options_t options = {
        .value_domain = &value_domain,
        .fact_table = &facts,
        .emitter = {},
        .phase_name = IREE_SV("test"),
    };
    loom_kernel_barrier_lifetime_result_t result = {};
    IREE_EXPECT_OK(loom_kernel_barrier_lifetime_verify_function(
        module.get(), function, &options, &result));
    loom_local_value_domain_release(&value_domain);
    module.reset();
    iree_arena_reset(&analysis_arena_);
    return result;
  }

  void ExpectValid(const char* source) {
    const loom_kernel_barrier_lifetime_result_t result = Analyze(source);
    EXPECT_EQ(result.error_count, 0u);
  }

  void ExpectInvalid(const char* source) {
    const loom_kernel_barrier_lifetime_result_t result = Analyze(source);
    EXPECT_EQ(result.error_count, 1u);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
};

TEST_F(KernelBarrierLifetimeTest, AcceptsStraightLinePrivateWork) {
  ExpectValid(R"(
func.def @test() {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  %value = scalar.constant 1 : i32
  %twice = scalar.addi %value, %value : i32
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, AcceptsSequentialPhases) {
  ExpectValid(R"(
func.def @test() {
  %first = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  kernel.barrier.wait %first : kernel.barrier.phase
  %second = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  kernel.barrier.wait %second : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, AcceptsPhaseAcrossCfgDiamond) {
  ExpectValid(R"(
func.def @test(%condition: i1) {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  cfg.cond_br %condition, ^left, ^right
^left:
  %left_value = scalar.constant 1 : i32
  cfg.br ^done
^right:
  %right_value = scalar.constant 2 : i32
  cfg.br ^done
^done:
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, AcceptsNestedPrivateControl) {
  ExpectValid(R"(
func.def @test(%condition: i1, %lo: index, %hi: index, %step: index) {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  scf.if %condition {
    %value = scalar.constant 1 : i32
    scf.for %iv = [%lo to %hi step %step] {
      %next = index.add %iv, %step : index
      scf.yield
    }
    scf.yield
  } else {
    %value = scalar.constant 2 : i32
    scf.yield
  }
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, AcceptsPairDrainedWithinEachLoopIteration) {
  ExpectValid(R"(
func.def @test(%lo: index, %hi: index, %step: index) {
  scf.for %iv = [%lo to %hi step %step] {
    %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
    %next = index.add %iv, %step : index
    kernel.barrier.wait %phase : kernel.barrier.phase
    scf.yield
  }
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, AcceptsPureCallWhilePhaseIsActive) {
  ExpectValid(R"(
func.decl pure @helper()
func.def @test() {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  func.call pure @helper() : ()
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsBranchWithOnePathDrained) {
  ExpectInvalid(R"(
func.def @test(%condition: i1) {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  cfg.cond_br %condition, ^wait, ^skip
^wait:
  kernel.barrier.wait %phase : kernel.barrier.phase
  cfg.br ^done
^skip:
  cfg.br ^done
^done:
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsStructuredRegionThatDrainsPhase) {
  ExpectInvalid(R"(
func.def @test(%condition: i1) {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  scf.if %condition {
    kernel.barrier.wait %phase : kernel.barrier.phase
    scf.yield
  }
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsLoopReentryWithActivePhase) {
  ExpectInvalid(R"(
func.def @test(%condition: i1) {
  cfg.br ^header
^header:
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  cfg.cond_br %condition, ^header, ^done
^done:
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsOverlappingArrivals) {
  ExpectInvalid(R"(
func.def @test() {
  %outer = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  %inner = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  kernel.barrier.wait %inner : kernel.barrier.phase
  kernel.barrier.wait %outer : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsCompleteBarrierWithinPhase) {
  ExpectInvalid(R"(
func.def @test() {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsConvergentCollectiveWithinPhase) {
  ExpectInvalid(R"(
func.def @test(%value: i32) {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  %sum = kernel.workgroup.reduce<addi> %value : i32
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsExitWithActivePhase) {
  ExpectInvalid(R"(
func.def @test() {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  func.return
}
)");
}

TEST_F(KernelBarrierLifetimeTest, RejectsImpureCallWithinPhase) {
  ExpectInvalid(R"(
func.decl @helper()
func.def @test() {
  %phase = kernel.barrier.arrive<workgroup> scope(workgroup) ordering(acq_rel) -> kernel.barrier.phase
  func.call @helper() : ()
  kernel.barrier.wait %phase : kernel.barrier.phase
  func.return
}
)");
}

}  // namespace
}  // namespace loom
