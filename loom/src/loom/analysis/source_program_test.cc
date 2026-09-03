// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_program.h"

#include <algorithm>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SourceProgramTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("source_program_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_func_like_cast(module,
                               module->symbols.entries[symbol_id].defining_op);
  }

  loom_value_id_t FindValue(const loom_module_t* module,
                            const loom_local_value_domain_t& value_domain,
                            iree_string_view_t name) {
    for (loom_value_ordinal_t i = 0; i < value_domain.value_count; ++i) {
      const loom_value_id_t value_id = value_domain.value_ids[i];
      if (iree_string_view_equal(loom_module_value_name(module, value_id),
                                 name)) {
        return value_id;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  bool HasFlow(const loom_source_program_t& program,
               loom_value_id_t source_value_id, loom_value_id_t target_value_id,
               loom_source_program_value_flow_kinds_t kinds) {
    const loom_value_ordinal_t source =
        loom_local_value_domain_ordinal(program.value_domain, source_value_id);
    const loom_value_ordinal_t target =
        loom_local_value_domain_ordinal(program.value_domain, target_value_id);
    for (uint32_t i = 0; i < program.value_flow_count; ++i) {
      if (program.value_flows[i].source == source &&
          program.value_flows[i].target == target &&
          iree_all_bits_set(program.value_flows[i].kinds, kinds)) {
        return true;
      }
    }
    return false;
  }

  void BuildProgram(loom_module_t* module, loom_func_like_t function,
                    loom_local_value_domain_t* out_value_domain,
                    loom_source_program_t* out_program) {
    const loom_region_t* body = loom_func_like_body(function);
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
        module, body, &analysis_arena_, out_value_domain));
    IREE_ASSERT_OK(loom_source_program_build(module, function.op, body,
                                             out_value_domain, &analysis_arena_,
                                             out_program));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
};

TEST_F(SourceProgramTest, RetainsPreorderAndNestedSubtreeLimits) {
  ModulePtr module = ParseModule(R"(
func.def @nested(%condition: i1, %arg: i32) -> (i32) {
  %before = scalar.addi %arg, %arg : i32
  %selected = scf.if %condition -> (i32) {
    %then_value = scalar.addi %before, %arg : i32
    scf.yield %then_value : i32
  } else {
    %else_value = scalar.subi %before, %arg : i32
    scf.yield %else_value : i32
  }
  %after = scalar.addi %selected, %arg : i32
  func.return %after : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("nested"));
  ASSERT_TRUE(loom_func_like_isa(function));
  const loom_region_t* body = loom_func_like_body(function);

  loom_local_value_domain_t value_domain = {};
  IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region_tree(
      module.get(), body, &analysis_arena_, &value_domain));
  loom_source_program_t program = {};
  IREE_ASSERT_OK(loom_source_program_build(module.get(), function.op, body,
                                           &value_domain, &analysis_arena_,
                                           &program));

  ASSERT_EQ(program.region_count, 3u);
  ASSERT_EQ(program.block_count, 3u);
  ASSERT_EQ(program.operation_count, 8u);
  ASSERT_EQ(program.node_count, 11u);

  const loom_source_program_node_kind_t expected_kinds[] = {
      LOOM_SOURCE_PROGRAM_NODE_BLOCK,     LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_BLOCK,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_BLOCK,     LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION, LOOM_SOURCE_PROGRAM_NODE_OPERATION,
      LOOM_SOURCE_PROGRAM_NODE_OPERATION,
  };
  const loom_op_kind_t expected_op_kinds[] = {
      LOOM_OP_SCALAR_ADDI, LOOM_OP_SCF_IF,      LOOM_OP_SCALAR_ADDI,
      LOOM_OP_SCF_YIELD,   LOOM_OP_SCALAR_SUBI, LOOM_OP_SCF_YIELD,
      LOOM_OP_SCALAR_ADDI, LOOM_OP_FUNC_RETURN,
  };
  iree_host_size_t operation_index = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_kinds); ++i) {
    const loom_source_program_node_t* node = &program.nodes[i];
    ASSERT_EQ(node->kind, expected_kinds[i]);
    if (node->kind == LOOM_SOURCE_PROGRAM_NODE_OPERATION) {
      EXPECT_EQ(loom_source_program_node_operation(node)->kind,
                expected_op_kinds[operation_index++]);
    }
  }

  EXPECT_TRUE(iree_any_bit_set(program.nodes[0].flags,
                               LOOM_SOURCE_PROGRAM_NODE_ROOT_ENTRY_BLOCK));
  EXPECT_EQ(program.nodes[0].subtree_limit, program.node_count);
  EXPECT_EQ(program.nodes[1].subtree_limit, 2u);
  EXPECT_EQ(program.nodes[2].subtree_limit, 9u);
  EXPECT_EQ(program.nodes[3].subtree_limit, 6u);
  EXPECT_EQ(program.nodes[6].subtree_limit, 9u);
  EXPECT_EQ(program.nodes[9].subtree_limit, 10u);
  EXPECT_EQ(program.nodes[10].subtree_limit, 11u);
  EXPECT_EQ(program.nodes[0].region_depth, 0u);
  EXPECT_EQ(program.nodes[3].region_depth, 1u);
  EXPECT_EQ(program.nodes[6].region_depth, 1u);
  EXPECT_EQ(program.nodes[0].context_op, function.op);
  EXPECT_EQ(program.nodes[3].context_op,
            loom_source_program_node_operation(&program.nodes[2]));
  EXPECT_EQ(program.nodes[6].context_op,
            loom_source_program_node_operation(&program.nodes[2]));

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceProgramTest, IndexesCfgPayloadRelations) {
  ModulePtr module = ParseModule(R"(
func.def @cfg(%condition: i1, %lhs: i32, %rhs: i32) -> (i32) {
  cfg.cond_br %condition, ^then, ^else
^then:
  cfg.br ^join(%lhs: i32)
^else:
  cfg.br ^join(%rhs: i32)
^join(%joined: i32):
  func.return %joined : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("cfg"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  const loom_value_id_t lhs =
      FindValue(module.get(), value_domain, IREE_SV("lhs"));
  const loom_value_id_t rhs =
      FindValue(module.get(), value_domain, IREE_SV("rhs"));
  const loom_value_id_t joined =
      FindValue(module.get(), value_domain, IREE_SV("joined"));
  const loom_value_id_t condition =
      FindValue(module.get(), value_domain, IREE_SV("condition"));
  ASSERT_NE(lhs, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(rhs, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(joined, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(condition, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(HasFlow(program, lhs, joined,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD));
  EXPECT_TRUE(HasFlow(program, rhs, joined,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD));
  EXPECT_TRUE(HasFlow(program, condition, joined,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
  EXPECT_TRUE(HasFlow(program, condition, lhs,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
  EXPECT_TRUE(HasFlow(program, condition, rhs,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
  const loom_source_program_value_t* joined_record =
      loom_source_program_try_value(&program, joined);
  ASSERT_NE(joined_record, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      joined_record->flags, LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT |
                                LOOM_SOURCE_PROGRAM_VALUE_HAS_CFG_PREDECESSOR));
  const loom_source_program_value_t* condition_record =
      loom_source_program_try_value(&program, condition);
  ASSERT_NE(condition_record, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      condition_record->flags,
      LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT |
          LOOM_SOURCE_PROGRAM_VALUE_ROOT_ENTRY_ARGUMENT |
          LOOM_SOURCE_PROGRAM_VALUE_HAS_CONTROL_CONDITION_USE));
  ASSERT_EQ(condition_record->use_count, 1u);
  const loom_source_program_use_t* condition_uses =
      loom_source_program_value_uses(&program, condition_record);
  ASSERT_NE(condition_uses, nullptr);
  EXPECT_EQ(condition_uses[0].operand_role,
            LOOM_OPERAND_ROLE_CONTROL_CONDITION);

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceProgramTest, IndexesCrossBlockUsesAndConditionOrder) {
  ModulePtr module = ParseModule(R"(
func.def @ordered(%x: i32, %y: i32, %zero: i32, %two: i32,
                  %lhs: i32, %middle: i32, %rhs: i32) -> (i32) {
  %cross_block = scalar.addi %lhs, %middle : i32
  %outer = scalar.cmpi eq, %x, %zero : i32
  %inner = scalar.cmpi eq, %y, %two : i32
  %middle_or_rhs = scf.select %inner, %rhs, %middle : i32
  %selected = scf.select %outer, %cross_block, %middle_or_rhs : i32
  cfg.br ^return
^return:
  func.return %selected : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("ordered"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  const loom_value_id_t outer =
      FindValue(module.get(), value_domain, IREE_SV("outer"));
  const loom_value_id_t inner =
      FindValue(module.get(), value_domain, IREE_SV("inner"));
  const loom_value_id_t selected =
      FindValue(module.get(), value_domain, IREE_SV("selected"));
  ASSERT_NE(outer, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(inner, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(selected, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(HasFlow(program, inner, outer,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONDITION_ORDER));

  const loom_source_program_value_t* selected_record =
      loom_source_program_try_value(&program, selected);
  ASSERT_NE(selected_record, nullptr);
  EXPECT_TRUE(iree_any_bit_set(selected_record->flags,
                               LOOM_SOURCE_PROGRAM_VALUE_HAS_CROSS_BLOCK_USE));
  const loom_source_program_value_t* outer_record =
      loom_source_program_try_value(&program, outer);
  ASSERT_NE(outer_record, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      outer_record->flags, LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_CONDITION_USE));

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceProgramTest, IndexesIdentityAliasAndTiedRelations) {
  ModulePtr module = ParseModule(R"(
func.decl @mutate(%input: i32) -> (%input as i32)
func.def @local(%seed: i32, %data: vector<4xi32>, %rows: index,
                %columns: index) -> (vector<4xi32>) {
  %assumed = scalar.assume %seed [range(%seed, 0, 31)] : i32
  %called = func.call @mutate(%assumed) : (i32) -> (%assumed as i32)
  %fragment = vector.fragment<result> %data shape [%rows, %columns] : vector<4xi32>
  %repacked = vector.fragment.repack<lhs> %fragment shape [%rows, %columns] : vector<4xi32> -> vector<4xi32>
  func.return %repacked : vector<4xi32>
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("local"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  const loom_value_id_t seed =
      FindValue(module.get(), value_domain, IREE_SV("seed"));
  const loom_value_id_t assumed =
      FindValue(module.get(), value_domain, IREE_SV("assumed"));
  const loom_value_id_t called =
      FindValue(module.get(), value_domain, IREE_SV("called"));
  const loom_value_id_t data =
      FindValue(module.get(), value_domain, IREE_SV("data"));
  const loom_value_id_t fragment =
      FindValue(module.get(), value_domain, IREE_SV("fragment"));
  const loom_value_id_t repacked =
      FindValue(module.get(), value_domain, IREE_SV("repacked"));
  ASSERT_NE(seed, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(assumed, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(called, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(data, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(fragment, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(repacked, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(HasFlow(program, seed, assumed,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY));
  EXPECT_TRUE(HasFlow(program, assumed, called,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT));
  EXPECT_TRUE(HasFlow(program, data, fragment,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS));
  EXPECT_FALSE(HasFlow(program, fragment, repacked,
                       LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY |
                           LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS |
                           LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT));

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceProgramTest, IndexesRegionBranchAndLoopStateCycles) {
  ModulePtr module = ParseModule(R"(
func.def @structured(%condition: i1, %seed: i32, %lower: index,
                     %upper: index, %step: index) -> (i32, i32) {
  %selected = scf.if %condition -> (i32) {
    scf.yield %seed : i32
  } else {
    scf.yield %seed : i32
  }
  %counted = scf.for %iv = [%lower to %upper step %step]
      (%carried = %selected : i32) -> (i32) {
    scf.yield %carried : i32
  }
  %conditional = scf.while(%before = %counted : i32) -> (i32) {
    scf.condition %condition, %before : i1, i32
  } do(%body: i32) {
    scf.yield %body : i32
  }
  func.return %counted, %conditional : i32, i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("structured"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  const loom_value_id_t seed =
      FindValue(module.get(), value_domain, IREE_SV("seed"));
  const loom_value_id_t selected =
      FindValue(module.get(), value_domain, IREE_SV("selected"));
  const loom_value_id_t carried =
      FindValue(module.get(), value_domain, IREE_SV("carried"));
  const loom_value_id_t counted =
      FindValue(module.get(), value_domain, IREE_SV("counted"));
  const loom_value_id_t before =
      FindValue(module.get(), value_domain, IREE_SV("before"));
  const loom_value_id_t body =
      FindValue(module.get(), value_domain, IREE_SV("body"));
  const loom_value_id_t conditional =
      FindValue(module.get(), value_domain, IREE_SV("conditional"));
  ASSERT_NE(seed, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(selected, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(carried, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(counted, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(before, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(body, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(conditional, LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(HasFlow(program, seed, selected,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_REGION_YIELD));
  EXPECT_TRUE(HasFlow(program, selected, carried,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY));
  EXPECT_TRUE(HasFlow(program, selected, counted,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY));
  EXPECT_TRUE(HasFlow(program, counted, before,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY));
  EXPECT_TRUE(HasFlow(program, counted, body,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY));
  EXPECT_TRUE(HasFlow(program, counted, conditional,
                      LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY));

  for (uint32_t i = 0; i < program.value_flow_count; ++i) {
    if (i == 0) continue;
    const loom_source_program_value_flow_t previous =
        program.value_flows[i - 1];
    const loom_source_program_value_flow_t current = program.value_flows[i];
    EXPECT_TRUE(previous.source < current.source ||
                (previous.source == current.source &&
                 previous.target < current.target));
  }

  loom_local_value_domain_release(&value_domain);
}

}  // namespace
}  // namespace loom
