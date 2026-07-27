// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_packet.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class AmdgpuControlPacketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry, IREE_SV("amdgpu.rdna3.core"));
    if (descriptor_set_ == NULL) {
      GTEST_SKIP() << "RDNA3 descriptor set is not linked.";
    }
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    BuildFunctionBody();
  }

  void TearDown() override {
    if (module_ != NULL) {
      loom_module_free(module_);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  void BuildFunctionBody() {
    loom_symbol_ref_t target = AddSymbol(IREE_SV("gfx_target"));
    loom_symbol_ref_t callee = AddSymbol(IREE_SV("test_fn"));
    loom_op_t* function_op = NULL;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*allocation=*/0, /*schedule=*/0, target, /*abi=*/0,
        loom_make_named_attr_slice(NULL, 0),
        loom_make_named_attr_slice(NULL, 0),
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        loom_make_named_attr_slice(NULL, 0), callee, /*arg_types=*/NULL,
        /*arg_types_count=*/0, /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0, /*predicates=*/NULL,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
    body_block_ = loom_region_entry_block(loom_low_func_def_body(function_op));
    loom_builder_initialize(module_, &module_->arena, body_block_, &builder_);
    builder_.ip.parent_op = function_op;
  }

  std::vector<loom_op_t*> Ops() const {
    std::vector<loom_op_t*> ops;
    loom_op_t* op = NULL;
    loom_block_for_each_op(body_block_, op) { ops.push_back(op); }
    return ops;
  }

  iree_string_view_t String(loom_string_id_t string_id) const {
    return module_->strings.entries[string_id];
  }

  const loom_low_descriptor_t* DescriptorForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    const uint32_t descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, descriptor_ref);
    return loom_low_descriptor_set_descriptor_at(descriptor_set_,
                                                 descriptor_ordinal);
  }

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
    EXPECT_EQ(ToString(String(loom_low_op_opcode(op))),
              ToString(loom_low_descriptor_set_string(
                  descriptor_set_, descriptor->key_string_offset)));
  }

  void ExpectLowConstDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_const_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_const_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
    EXPECT_EQ(ToString(String(loom_low_const_opcode(op))),
              ToString(loom_low_descriptor_set_string(
                  descriptor_set_, descriptor->key_string_offset)));
  }

  void ExpectControlOp(const loom_op_t* op,
                       loom_amdgpu_descriptor_ref_t descriptor_ref,
                       iree_string_view_t expected_attr_name,
                       int64_t expected_attr_value,
                       iree_host_size_t expected_operand_count,
                       loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    EXPECT_EQ(loom_low_op_operands(op).count, expected_operand_count);
    loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)),
              ToString(expected_attr_name));
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_attr_value);

    loom_value_slice_t results = loom_low_op_results(op);
    if (expected_result == LOOM_VALUE_ID_INVALID) {
      EXPECT_EQ(results.count, 0u);
    } else {
      ASSERT_EQ(results.count, 1u);
      EXPECT_EQ(loom_value_slice_get(results, 0), expected_result);
    }
  }

  void ExpectPacketOp(const loom_op_t* op,
                      loom_amdgpu_descriptor_ref_t descriptor_ref,
                      iree_host_size_t expected_operand_count,
                      loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    EXPECT_EQ(loom_low_op_operands(op).count, expected_operand_count);
    EXPECT_EQ(loom_low_op_attrs(op).count, 0u);

    loom_value_slice_t results = loom_low_op_results(op);
    if (expected_result == LOOM_VALUE_ID_INVALID) {
      EXPECT_EQ(results.count, 0u);
    } else {
      ASSERT_EQ(results.count, 1u);
      EXPECT_EQ(loom_value_slice_get(results, 0), expected_result);
    }
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuControlPacketTest, EmitsScalarControlPackets) {
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_send_message(
      &builder_, descriptor_set_, /*message=*/1, LOOM_LOCATION_UNKNOWN));
  loom_value_id_t message_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_send_message_rtn_b32(
      &builder_, descriptor_set_, /*message=*/2, LOOM_LOCATION_UNKNOWN,
      &message_result));
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_halt(
      &builder_, descriptor_set_, /*reason=*/5, LOOM_LOCATION_UNKNOWN));
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_trap(
      &builder_, descriptor_set_, /*trap_id=*/3, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 5u);
  ExpectLowConstDescriptorRef(ops[0],
                              LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
  loom_named_attr_slice_t m0_attrs = loom_low_const_attrs(ops[0]);
  ASSERT_EQ(m0_attrs.count, 1u);
  EXPECT_EQ(ToString(String(m0_attrs.entries[0].name_id)), "imm32");
  EXPECT_EQ(loom_attr_as_i64(m0_attrs.entries[0].value), 0);

  ExpectControlOp(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG,
                  IREE_SV("message"), 1, 1, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(loom_low_op_operands(ops[1]).values[0],
            loom_low_const_result(ops[0]));
  ExpectControlOp(ops[2], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG_RTN_B32,
                  IREE_SV("message"), 2, 0, message_result);
  ExpectControlOp(ops[3], LOOM_AMDGPU_DESCRIPTOR_REF_S_SETHALT,
                  IREE_SV("reason"), 5, 0, LOOM_VALUE_ID_INVALID);
  ExpectControlOp(ops[4], LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP, IREE_SV("trapid"),
                  3, 0, LOOM_VALUE_ID_INVALID);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(sgpr_type,
                              loom_module_value_type(module_, message_result)));
}

TEST_F(AmdgpuControlPacketTest, PreservesExplicitM0Dependency) {
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_send_message(
      &builder_, descriptor_set_, /*message=*/1, LOOM_LOCATION_UNKNOWN));
  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 2u);
  const loom_value_id_t m0_payload = loom_low_const_result(ops[0]);

  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_send_message_with_m0(
      &builder_, descriptor_set_, /*message=*/2, m0_payload,
      LOOM_LOCATION_UNKNOWN));

  ops = Ops();
  ASSERT_EQ(ops.size(), 3u);
  ExpectControlOp(ops[2], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG,
                  IREE_SV("message"), 2, 1, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(loom_low_op_operands(ops[2]).values[0], m0_payload);
}

TEST_F(AmdgpuControlPacketTest, EmitsAmdhsaDebugTrap) {
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_debug_trap(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 1u);
  ExpectControlOp(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP, IREE_SV("trapid"),
                  3, 0, LOOM_VALUE_ID_INVALID);
}

TEST_F(AmdgpuControlPacketTest, EmitsAmdhsaFatalTrapForGfx11Plus) {
  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_fatal_trap(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 8u);
  ExpectControlOp(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP, IREE_SV("trapid"),
                  2, 0, LOOM_VALUE_ID_INVALID);
  const loom_value_id_t doorbell = loom_low_op_results(ops[1]).values[0];
  ExpectControlOp(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG_RTN_B32,
                  IREE_SV("message"), 128, 0, doorbell);

  ExpectLowConstDescriptorRef(ops[2], LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32);
  EXPECT_EQ(loom_attr_as_i64(loom_low_const_attrs(ops[2]).entries[0].value),
            0x3FF);
  const loom_value_id_t doorbell_mask = loom_low_const_result(ops[2]);
  const loom_value_id_t doorbell_id = loom_low_op_results(ops[3]).values[0];
  ExpectPacketOp(ops[3], LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, 2, doorbell_id);
  EXPECT_EQ(loom_low_op_operands(ops[3]).values[0], doorbell);
  EXPECT_EQ(loom_low_op_operands(ops[3]).values[1], doorbell_mask);

  ExpectLowConstDescriptorRef(ops[4], LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32);
  EXPECT_EQ(loom_attr_as_i64(loom_low_const_attrs(ops[4]).entries[0].value),
            0x400);
  const loom_value_id_t wave_abort_bit = loom_low_const_result(ops[4]);
  const loom_value_id_t interrupt_payload =
      loom_low_op_results(ops[5]).values[0];
  ExpectPacketOp(ops[5], LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32, 2,
                 interrupt_payload);
  EXPECT_EQ(loom_low_op_operands(ops[5]).values[0], doorbell_id);
  EXPECT_EQ(loom_low_op_operands(ops[5]).values[1], wave_abort_bit);

  const loom_value_id_t m0_payload = loom_low_op_results(ops[6]).values[0];
  ExpectPacketOp(ops[6], LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0, 1,
                 m0_payload);
  EXPECT_EQ(loom_low_op_operands(ops[6]).values[0], interrupt_payload);
  ExpectControlOp(ops[7], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG,
                  IREE_SV("message"), 1, 1, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(loom_low_op_operands(ops[7]).values[0], m0_payload);
}

TEST_F(AmdgpuControlPacketTest, EmitsAmdhsaFatalTrapDirectlyForGfx9) {
  descriptor_set_ = loom_low_descriptor_registry_lookup(
      &low_registry_.registry, IREE_SV("amdgpu.cdna3.core"));
  if (descriptor_set_ == nullptr) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }

  IREE_ASSERT_OK(loom_amdgpu_build_control_packet_fatal_trap(
      &builder_, descriptor_set_, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 1u);
  ExpectControlOp(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP, IREE_SV("trapid"),
                  2, 0, LOOM_VALUE_ID_INVALID);
}

}  // namespace
