// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_vector.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ContractVectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* func_vtables =
        loom_func_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC,
                                                 func_vtables, count));
    const loom_op_vtable_t* const* encoding_vtables =
        loom_encoding_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, encoding_vtables, count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    const loom_op_vtable_t* const* index_vtables =
        loom_index_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_INDEX,
                                                 index_vtables, count));
    const loom_op_vtable_t* const* vector_vtables =
        loom_vector_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_VECTOR,
                                                 vector_vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t ParseAndVerify(const char* source, loom_module_t** out_module) {
    *out_module = nullptr;
    loom_text_parse_options_t parse_options = {};
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("contract_vector.loom"),
        &context_, &block_pool_, &parse_options, &module));
    if (module == nullptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parser did not produce a module");
    }

    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    iree_status_t status =
        loom_verify_module(module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "module verification produced %u errors",
                                verify_result.error_count);
    }
    if (!iree_status_is_ok(status)) {
      loom_module_free(module);
      return status;
    }
    *out_module = module;
    return iree_ok_status();
  }

  const loom_op_t* FirstVectorDotOp(const loom_module_t* module) {
    loom_symbol_t* symbol = nullptr;
    loom_module_for_each_symbol(module, symbol) {
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(function)) {
        continue;
      }
      loom_region_t* body = loom_func_like_body(function);
      if (body == nullptr) {
        continue;
      }
      loom_block_t* block = nullptr;
      loom_region_for_each_block(body, block) {
        loom_op_t* op = nullptr;
        loom_block_for_each_op(block, op) {
          if (loom_vector_dot2f_isa(op) || loom_vector_dot4i_isa(op) ||
              loom_vector_dot8i4_isa(op)) {
            return op;
          }
        }
      }
    }
    return nullptr;
  }

  const loom_op_t* FirstVectorMmaOp(const loom_module_t* module) {
    return VectorMmaOpAt(module, 0);
  }

  const loom_op_t* VectorMmaOpAt(const loom_module_t* module,
                                 iree_host_size_t target_index) {
    iree_host_size_t mma_index = 0;
    loom_symbol_t* symbol = nullptr;
    loom_module_for_each_symbol(module, symbol) {
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(function)) {
        continue;
      }
      loom_region_t* body = loom_func_like_body(function);
      if (body == nullptr) {
        continue;
      }
      loom_block_t* block = nullptr;
      loom_region_for_each_block(body, block) {
        loom_op_t* op = nullptr;
        loom_block_for_each_op(block, op) {
          if (loom_vector_mma_isa(op)) {
            if (mma_index == target_index) {
              return op;
            }
            ++mma_index;
          }
        }
      }
    }
    return nullptr;
  }

  loom_func_like_t FirstFunction(const loom_module_t* module) {
    loom_symbol_t* symbol = nullptr;
    loom_module_for_each_symbol(module, symbol) {
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(function)) {
        return function;
      }
    }
    return (loom_func_like_t){0};
  }

  loom_contract_vector_mma_options_t GpuMatrixMmaOptions() {
    return (loom_contract_vector_mma_options_t){
        /*.k_group_size=*/1,
        /*.fragment=*/
        (loom_contract_fragment_t){
            /*.atom_bits=*/LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
            /*.vector_bit_width=*/{},
            /*.source_lane_count=*/{},
            /*.result_lane_count=*/{},
            /*.subgroup_size=*/64,
        },
        /*.capability_class=*/LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
        /*.policy=*/LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ContractVectorTest, BuildsGenericU8S8Dot4Contract) {
  static const char kSource[] = R"(
func.def @u8s8_dot4(%lhs: vector<32xi8>, %rhs: vector<32xi8>, %acc: vector<8xi32>) -> (vector<8xi32>) {
  %r = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<32xi8>, vector<32xi8>, vector<8xi32>
  func.return %r : vector<8xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  ASSERT_TRUE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, nullptr));

  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_VECTOR_DOT);
  EXPECT_EQ(request.arithmetic, LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT);
  EXPECT_EQ(request.shape.m, 8);
  EXPECT_EQ(request.shape.n, 1);
  EXPECT_EQ(request.shape.k, 32);
  EXPECT_EQ(request.k_group_size, 4);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_I32);
  EXPECT_EQ(request.result.numeric_type, LOOM_CONTRACT_NUMERIC_I32);
  EXPECT_EQ(request.fragment.atom_bits, LOOM_CONTRACT_FRAGMENT_VECTOR_LANE);
  EXPECT_EQ(request.fragment.vector_bit_width, 256);
  EXPECT_EQ(request.fragment.source_lane_count, 32);
  EXPECT_EQ(request.fragment.result_lane_count, 8);
  EXPECT_EQ(request.capability_class,
            LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT);
  EXPECT_EQ(request.policy, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
}

TEST_F(ContractVectorTest, DynamicDot4ReportsShapeRejection) {
  static const char kSource[] = R"(
func.def @dynamic_dot4(%n: index, %m: index, %lhs: vector<[%n]xi8>, %rhs: vector<[%n]xi8>, %acc: vector<[%m]xi32>) -> (vector<[%m]xi32>) {
  %r = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<[%n]xi8>, vector<[%n]xi8>, vector<[%m]xi32>
  func.return %r : vector<[%m]xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
}

TEST_F(ContractVectorTest, PackedI4DotDoesNotPretendToBeByteDotContract) {
  static const char kSource[] = R"(
func.def @packed_i4_dot(%lhs: vector<4xi32>, %rhs: vector<4xi32>, %acc: vector<4xi32>) -> (vector<4xi32>) {
  %r = vector.dot8i4<u4s4> %lhs, %rhs, %acc : vector<4xi32>
  func.return %r : vector<4xi32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  const loom_op_t* op = FirstVectorDotOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_dot_op(
      module_ptr.get(), op, LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
      &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST);
}

TEST_F(ContractVectorTest, BuildsDenseMmaContractFromFragmentFacts) {
  static const char kSource[] = R"(
func.def @dense_mma(%lhs_data: vector<8xf16>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>) -> (vector<8xf32>) {
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %k = index.constant 16 : index
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : vector<8xf16>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  ASSERT_TRUE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, nullptr));

  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_MATRIX_MULTIPLY);
  EXPECT_EQ(request.shape.m, 16);
  EXPECT_EQ(request.shape.n, 16);
  EXPECT_EQ(request.shape.k, 16);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_F16);
  EXPECT_EQ(request.lhs.payload_register_count, 4);
  EXPECT_EQ(request.lhs.payload_element_count, 8);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_F16);
  EXPECT_EQ(request.rhs.payload_register_count, 4);
  EXPECT_EQ(request.rhs.payload_element_count, 8);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_EQ(request.accumulator.payload_register_count, 8);
  EXPECT_EQ(request.accumulator.payload_element_count, 8);
  EXPECT_EQ(request.result.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_EQ(request.result.payload_register_count, 8);
  EXPECT_EQ(request.result.payload_element_count, 8);
  EXPECT_EQ(request.arithmetic, LOOM_CONTRACT_ARITHMETIC_MIXED_DOT);
  EXPECT_EQ(request.capability_class,
            LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX);
  EXPECT_EQ(request.fragment.atom_bits, LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE);
  EXPECT_EQ(request.fragment.subgroup_size, 64);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, BuildsDynamicShapeMmaContractFromFragmentFacts) {
  static const char kSource[] = R"(
func.def @dynamic_shape_mma(%m: index, %n: index, %k: index, %lhs_data: vector<8xf16>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>) -> (vector<8xf32>) {
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : vector<8xf16>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(request.shape.m, 0);
  EXPECT_EQ(request.shape.n, 0);
  EXPECT_EQ(request.shape.k, 0);

  loom_vector_fragment_fact_t lhs_fragment = {};
  ASSERT_TRUE(loom_vector_fragment_fact_query_value_facts(
      &fact_table->context,
      loom_value_fact_table_lookup(fact_table, loom_vector_mma_lhs(op)),
      &lhs_fragment));
  loom_vector_fragment_fact_t rhs_fragment = {};
  ASSERT_TRUE(loom_vector_fragment_fact_query_value_facts(
      &fact_table->context,
      loom_value_fact_table_lookup(fact_table, loom_vector_mma_rhs(op)),
      &rhs_fragment));
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.m),
            lhs_fragment.shape_value_ids[0]);
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.n),
            rhs_fragment.shape_value_ids[1]);
  EXPECT_EQ(loom_contract_value_ref_value_id(request.shape_value_refs.k),
            lhs_fragment.shape_value_ids[1]);
  EXPECT_EQ(
      loom_contract_value_ref_value_id(request.shape_value_refs.k_group_size),
      LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(request.k_group_size, 1);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, BuildsEncodedMmaContractFromFragmentFacts) {
  static const char kSource[] = R"(
func.def @encoded_mma(%lhs_data: vector<6xi32>, %rhs_data: vector<6xi32>, %init_data: vector<8xf32>, %scale: vector<1xf16>) -> (vector<8xf32>) {
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %k = index.constant 128 : index
  %schema = encoding.define #matrix_operand<element_format=f6e3m2, payload_elements=32, payload_registers=6, scale_group_elements=32, scale_operands=1, scale_topology=block_1d, zero_scale_fallback=true> : encoding<schema>
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] using {scale = %scale : vector<1xf16>, schema = %schema : encoding<schema>} : vector<6xi32>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] using {scale = %scale : vector<1xf16>, schema = %schema : encoding<schema>} : vector<6xi32>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<6xi32>, vector<6xi32>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  loom_vector_fragment_fact_t lhs_fragment = {};
  ASSERT_TRUE(loom_vector_fragment_fact_query_value_facts(
      &fact_table->context,
      loom_value_fact_table_lookup(fact_table, loom_vector_mma_lhs(op)),
      &lhs_fragment));
  loom_vector_fragment_fact_t rhs_fragment = {};
  ASSERT_TRUE(loom_vector_fragment_fact_query_value_facts(
      &fact_table->context,
      loom_value_fact_table_lookup(fact_table, loom_vector_mma_rhs(op)),
      &rhs_fragment));
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(request.lhs.payload_register_count, 6);
  EXPECT_EQ(request.lhs.payload_element_count, 32);
  EXPECT_TRUE(iree_any_bit_set(request.lhs.encoded.available_auxiliary_operands,
                               LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE));
  EXPECT_EQ(
      loom_contract_value_ref_value_id(
          request.lhs.encoded
              .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE]),
      lhs_fragment.auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE]);
  EXPECT_EQ(loom_contract_value_ref_value_id(
                request.lhs.encoded.auxiliary_value_refs
                    [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_CODEBOOK_TABLE]),
            LOOM_VALUE_ID_INVALID);
  EXPECT_TRUE(iree_any_bit_set(request.lhs.encoded.required_auxiliary_operands,
                               LOOM_CONTRACT_AUXILIARY_OPERAND_SCALE));
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(
      loom_contract_value_ref_value_id(
          request.rhs.encoded
              .auxiliary_value_refs[LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE]),
      rhs_fragment.auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE]);
  EXPECT_EQ(request.shape.k, 128);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, MmaShapeMismatchReportsShapeRejection) {
  static const char kSource[] = R"(
func.def @shape_mismatch_mma(%lhs_data: vector<8xf16>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>) -> (vector<8xf32>) {
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %lhs_k = index.constant 16 : index
  %rhs_k = index.constant 32 : index
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %lhs_k] : vector<8xf16>
  %rhs = vector.fragment<rhs> %rhs_data shape [%rhs_k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, MmaDynamicShapeMismatchReportsShapeRejection) {
  static const char kSource[] = R"(
func.def @dynamic_shape_mismatch_mma(%m: index, %n: index, %lhs_k: index, %rhs_k: index, %lhs_data: vector<8xf16>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>) -> (vector<8xf32>) {
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %lhs_k] : vector<8xf16>
  %rhs = vector.fragment<rhs> %rhs_data shape [%rhs_k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, MmaUnresolvedSchemaReportsSchemaRejection) {
  static const char kSource[] = R"(
func.def @unresolved_schema_mma(%lhs_data: vector<6xi32>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>, %schema: encoding<schema>) -> (vector<8xf32>) {
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %k = index.constant 16 : index
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] using {schema = %schema : encoding<schema>} : vector<6xi32>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %result = vector.mma %lhs, %rhs, %init : vector<6xi32>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_SCHEMA);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, MmaResultFactsCanInitializeNextMma) {
  static const char kSource[] = R"(
func.def @chained_mma(%lhs_data: vector<8xf16>, %rhs_data: vector<8xf16>, %init_data: vector<8xf32>) -> (vector<8xf32>) {
  %m = index.constant 16 : index
  %n = index.constant 16 : index
  %k = index.constant 16 : index
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : vector<8xf16>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : vector<8xf16>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xf32>
  %partial = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  %result = vector.mma %lhs, %rhs, %partial : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = VectorMmaOpAt(module_ptr.get(), 1);
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  ASSERT_TRUE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, nullptr));
  EXPECT_EQ(request.shape.m, 16);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

TEST_F(ContractVectorTest, MmaWithoutFragmentsReportsRoleRejection) {
  static const char kSource[] = R"(
func.def @bare_mma(%lhs: vector<8xf16>, %rhs: vector<8xf16>, %init: vector<8xf32>) -> (vector<8xf32>) {
  %result = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
  func.return %result : vector<8xf32>
}
)";

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseAndVerify(kSource, &module));
  ModulePtr module_ptr(module);

  loom_pass_value_fact_owner_t value_facts = {};
  loom_pass_value_fact_owner_initialize(module->arena.block_pool, &value_facts);
  loom_value_fact_table_t* fact_table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &value_facts, module_ptr.get(),
      loom_pass_value_fact_scope_function(FirstFunction(module_ptr.get())),
      &fact_table));

  const loom_op_t* op = FirstVectorMmaOp(module_ptr.get());
  ASSERT_NE(op, nullptr);
  const loom_contract_vector_mma_options_t options = GpuMatrixMmaOptions();
  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_vector_mma_op(
      module_ptr.get(), fact_table, op, &options, &request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_ROLE);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
}

}  // namespace
