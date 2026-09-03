// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_dataflow.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/source_dataflow_verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

enum TestEvidenceBits : loom_source_dataflow_bits_t {
  kReachedByAny = (loom_source_dataflow_bits_t)1u << 0,
  kReachedByAll = (loom_source_dataflow_bits_t)1u << 1,
  kRejectedCandidate = (loom_source_dataflow_bits_t)1u << 2,
  kRequiredByUse = (loom_source_dataflow_bits_t)1u << 3,
  kStructural = (loom_source_dataflow_bits_t)1u << 4,
};

struct SeedConfiguration {
  loom_value_id_t rejected_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t structural_value = LOOM_VALUE_ID_INVALID;
};

iree_status_t SeedValue(void* user_data,
                        const loom_source_dataflow_environment_t* environment,
                        loom_value_id_t value_id,
                        loom_source_dataflow_bits_t* out_bits) {
  (void)user_data;
  const auto* configuration =
      static_cast<const SeedConfiguration*>(environment->configuration);
  *out_bits = 0;
  if (value_id == configuration->rejected_value) {
    *out_bits |= kRejectedCandidate;
  }
  if (value_id == configuration->structural_value) {
    *out_bits |= kStructural;
  }
  return iree_ok_status();
}

iree_status_t ResultIsI32(void* user_data,
                          const loom_source_dataflow_environment_t* environment,
                          const loom_op_t* op, bool* out_matches) {
  (void)user_data;
  const loom_module_t* module = environment->program->module;
  *out_matches =
      op->result_count == 1 &&
      loom_type_is_scalar(
          loom_module_value_type(module, loom_op_const_results(op)[0])) &&
      loom_type_element_type(loom_module_value_type(
          module, loom_op_const_results(op)[0])) == LOOM_SCALAR_TYPE_I32;
  return iree_ok_status();
}

constexpr std::array<uint16_t, LOOM_OP_SCALAR_COUNT_>
MakeScalarOperationIndices() {
  std::array<uint16_t, LOOM_OP_SCALAR_COUNT_> indices = {};
  indices[LOOM_OP_SCALAR_CONSTANT & 0xFFu] = 1;
  indices[LOOM_OP_SCALAR_ADDI & 0xFFu] = 2;
  indices[LOOM_OP_SCALAR_MULI & 0xFFu] = 3;
  return indices;
}

constexpr std::array<uint16_t, LOOM_OP_SCALAR_COUNT_> kScalarOperationIndices =
    MakeScalarOperationIndices();

constexpr loom_source_dataflow_dialect_table_t kDialectTables[] = {
    {
        .operation_count = LOOM_OP_SCALAR_COUNT_,
        .operation_indices = kScalarOperationIndices.data(),
    },
};

constexpr loom_source_dataflow_port_t kPorts[] = {
    // scalar.constant
    {LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD, 0},
    // scalar.addi
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 0},
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 1},
    {LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD, 0},
    // scalar.muli
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 0},
    {LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD, 1},
    {LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD, 0},
};

constexpr loom_source_dataflow_rule_t kRules[] = {
    // scalar.constant seeds both forward domains when it is i32.
    {
        .target_bits = kReachedByAny | kReachedByAll,
        .target_port_mask = 0b1,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_SEED,
        .predicate_index_plus_one = 1,
    },
    // scalar.addi forward existential and conjunctive domains.
    {
        .source_bits = kReachedByAny,
        .target_bits = kReachedByAny,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    {
        .source_bits = kReachedByAll,
        .target_bits = kReachedByAll,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ALL,
    },
    // A rejection bit is the monotone complement of descending feasibility.
    {
        .source_bits = kRejectedCandidate,
        .target_bits = kRejectedCandidate,
        .source_port_mask = 0b011,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    // Use requirements flow backward from an add result to both inputs.
    {
        .source_bits = kRequiredByUse,
        .target_bits = kRequiredByUse,
        .source_port_mask = 0b100,
        .target_port_mask = 0b011,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
    // scalar.muli is a terminal requiring its result and both operands.
    {
        .target_bits = kRequiredByUse,
        .target_port_mask = 0b100,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_SEED,
    },
    {
        .source_bits = kRequiredByUse,
        .target_bits = kRequiredByUse,
        .source_port_mask = 0b100,
        .target_port_mask = 0b011,
        .kind = LOOM_SOURCE_DATAFLOW_RULE_ANY,
    },
};

constexpr loom_source_dataflow_operation_t kOperations[] = {
    {
        .port_start = 0,
        .rule_start = 0,
        .port_count = 1,
        .rule_count = 1,
    },
    {
        .port_start = 1,
        .rule_start = 1,
        .port_count = 3,
        .rule_count = 4,
    },
    {
        .port_start = 4,
        .rule_start = 5,
        .port_count = 3,
        .rule_count = 2,
    },
};

constexpr loom_source_dataflow_predicate_t kPredicates[] = {
    {.fn = ResultIsI32},
};

constexpr loom_source_dataflow_provider_t kProvider = {
    .name = IREE_SVL("source-dataflow-test"),
    .valid_bits = kReachedByAny | kReachedByAll | kRejectedCandidate |
                  kRequiredByUse | kStructural,
    .structural_copy_bits = kStructural,
    .dialect_base_id = LOOM_DIALECT_SCALAR,
    .dialect_count = 1,
    .operation_count = IREE_ARRAYSIZE(kOperations),
    .dialects = kDialectTables,
    .operations = kOperations,
    .port_count = IREE_ARRAYSIZE(kPorts),
    .ports = kPorts,
    .rule_count = IREE_ARRAYSIZE(kRules),
    .rules = kRules,
    .predicate_count = IREE_ARRAYSIZE(kPredicates),
    .predicates = kPredicates,
    .seed_value = {.fn = SeedValue},
};

class SourceDataflowTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
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
                                  IREE_SV("source_dataflow_test.loom"),
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

  const loom_op_t* FindOperation(const loom_source_program_t& program,
                                 loom_op_kind_t kind, uint32_t occurrence) {
    for (loom_source_program_node_ordinal_t i = 0; i < program.node_count;
         ++i) {
      const loom_source_program_node_t* node = &program.nodes[i];
      if (node->kind != LOOM_SOURCE_PROGRAM_NODE_OPERATION) continue;
      const loom_op_t* op = loom_source_program_node_operation(node);
      if (op->kind == kind && occurrence-- == 0) return op;
    }
    return nullptr;
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

TEST_F(SourceDataflowTest, SolvesAnyAllBackwardAndDescendingEvidence) {
  ModulePtr module = ParseModule(R"(
func.def @flow(%a: i32, %b: i32) -> (i32) {
  %c1 = scalar.constant 1 : i32
  %left = scalar.addi %a, %c1 : i32
  %right = scalar.addi %c1, %c1 : i32
  %join = scalar.addi %left, %right : i32
  %terminal = scalar.muli %join, %right : i32
  func.return %terminal : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("flow"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 2);
  SeedConfiguration configuration = {
      .rejected_value = arguments[0],
  };
  const loom_source_dataflow_environment_t environment = {
      .program = &program,
      .configuration = &configuration,
  };
  loom_source_dataflow_result_t result = {};
  IREE_ASSERT_OK(loom_source_dataflow_solve(&kProvider, &environment,
                                            &analysis_arena_, &result));

  const loom_value_id_t constant = loom_scalar_constant_result(
      FindOperation(program, LOOM_OP_SCALAR_CONSTANT, 0));
  const loom_value_id_t left =
      loom_scalar_addi_result(FindOperation(program, LOOM_OP_SCALAR_ADDI, 0));
  const loom_value_id_t right =
      loom_scalar_addi_result(FindOperation(program, LOOM_OP_SCALAR_ADDI, 1));
  const loom_value_id_t join =
      loom_scalar_addi_result(FindOperation(program, LOOM_OP_SCALAR_ADDI, 2));
  const loom_value_id_t terminal =
      loom_scalar_muli_result(FindOperation(program, LOOM_OP_SCALAR_MULI, 0));

  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      &result, constant, kReachedByAny | kReachedByAll | kRequiredByUse));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      &result, left, kReachedByAny | kRejectedCandidate | kRequiredByUse));
  EXPECT_FALSE(
      loom_source_dataflow_result_has_all(&result, left, kReachedByAll));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      &result, right, kReachedByAny | kReachedByAll | kRequiredByUse));
  EXPECT_FALSE(
      loom_source_dataflow_result_has_all(&result, right, kRejectedCandidate));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      &result, join, kReachedByAny | kRejectedCandidate | kRequiredByUse));
  EXPECT_FALSE(
      loom_source_dataflow_result_has_all(&result, join, kReachedByAll));
  EXPECT_TRUE(
      loom_source_dataflow_result_has_all(&result, terminal, kRequiredByUse));
  EXPECT_TRUE(loom_source_dataflow_result_has_all(
      &result, arguments[0], kRejectedCandidate | kRequiredByUse));
  EXPECT_EQ(result.statistics.value_seed_invocation_count,
            value_domain.value_count);
  EXPECT_EQ(result.statistics.predicate_invocation_count, 1u);
  EXPECT_GT(result.statistics.rule_evaluation_count, 0u);
  EXPECT_LT(result.statistics.rule_evaluation_count, 128u);
  EXPECT_EQ(loom_source_dataflow_result_lookup(&result, LOOM_VALUE_ID_INVALID),
            0u);

  loom_source_dataflow_result_t repeated_result = {};
  IREE_ASSERT_OK(loom_source_dataflow_solve(
      &kProvider, &environment, &analysis_arena_, &repeated_result));
  ASSERT_EQ(repeated_result.state_count, result.state_count);
  for (loom_value_ordinal_t i = 0; i < result.state_count; ++i) {
    EXPECT_EQ(repeated_result.states[i], result.states[i]);
  }
  EXPECT_EQ(repeated_result.statistics.rule_evaluation_count,
            result.statistics.rule_evaluation_count);

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceDataflowTest, StructuralRelationsCloseDiamondsAndLoopCycles) {
  ModulePtr module = ParseModule(R"(
func.def @structured(%condition: i1, %seed: i32, %lower: index,
                     %upper: index, %step: index) -> (i32) {
  %selected = scf.if %condition -> (i32) {
    scf.yield %seed : i32
  } else {
    scf.yield %seed : i32
  }
  %loop = scf.for %iv = [%lower to %upper step %step]
      (%carried = %selected : i32) -> (i32) {
    scf.yield %carried : i32
  }
  func.return %loop : i32
}
)");
  loom_func_like_t function = FindFunction(module.get(), IREE_SV("structured"));
  loom_local_value_domain_t value_domain = {};
  loom_source_program_t program = {};
  BuildProgram(module.get(), function, &value_domain, &program);

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  ASSERT_EQ(argument_count, 5);
  SeedConfiguration configuration = {
      .structural_value = arguments[1],
  };
  const loom_source_dataflow_environment_t environment = {
      .program = &program,
      .configuration = &configuration,
  };
  loom_source_dataflow_result_t result = {};
  IREE_ASSERT_OK(loom_source_dataflow_solve(&kProvider, &environment,
                                            &analysis_arena_, &result));

  const loom_op_t* if_op = FindOperation(program, LOOM_OP_SCF_IF, 0);
  const loom_op_t* for_op = FindOperation(program, LOOM_OP_SCF_FOR, 0);
  ASSERT_NE(if_op, nullptr);
  ASSERT_NE(for_op, nullptr);
  const loom_value_id_t selected = loom_op_const_results(if_op)[0];
  const loom_value_id_t loop_result = loom_op_const_results(for_op)[0];
  const loom_region_t* loop_body = loom_loop_like_body(
      loom_loop_like_cast(module.get(), const_cast<loom_op_t*>(for_op)));
  ASSERT_NE(loop_body, nullptr);
  const loom_value_id_t carried =
      loom_block_arg_id(loom_region_const_entry_block(loop_body), 1);

  EXPECT_GT(program.value_relation_count, 0u);
  EXPECT_TRUE(
      loom_source_dataflow_result_has_all(&result, selected, kStructural));
  EXPECT_TRUE(
      loom_source_dataflow_result_has_all(&result, carried, kStructural));
  EXPECT_TRUE(
      loom_source_dataflow_result_has_all(&result, loop_result, kStructural));
  EXPECT_FALSE(
      loom_source_dataflow_result_has_all(&result, arguments[0], kStructural));

  loom_local_value_domain_release(&value_domain);
}

TEST_F(SourceDataflowTest, BuildTimeVerifierRejectsMalformedProvider) {
  IREE_EXPECT_OK(loom_source_dataflow_provider_verify(&kProvider));
  loom_source_dataflow_provider_t malformed = kProvider;
  malformed.structural_copy_bits = (loom_source_dataflow_bits_t)1u << 63;
  iree_status_t status = loom_source_dataflow_provider_verify(&malformed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_source_dataflow_rule_t malformed_rules[IREE_ARRAYSIZE(kRules)];
  std::memcpy(malformed_rules, kRules, sizeof(malformed_rules));
  malformed = kProvider;
  malformed.rules = malformed_rules;
  malformed_rules[0].target_port_mask = (uint32_t)1u << 31;
  status = loom_source_dataflow_provider_verify(&malformed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

}  // namespace
}  // namespace loom
