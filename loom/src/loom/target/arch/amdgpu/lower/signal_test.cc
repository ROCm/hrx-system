// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/signal.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/abi/signal.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

namespace {

constexpr int64_t kSystemCacheScope = LOOM_CACHE_SCOPE_SYSTEM;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class AmdgpuSignalTest : public ::testing::Test {
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

  void UseDescriptorSet(iree_string_view_t key) {
    descriptor_set_ =
        loom_low_descriptor_registry_lookup(&low_registry_.registry, key);
    if (descriptor_set_ == NULL) {
      GTEST_SKIP() << "AMDGPU descriptor set is not linked: " << ToString(key);
    }
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

  iree_host_size_t PacketOperandCountForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    EXPECT_NE(descriptor, nullptr);
    if (descriptor == nullptr) {
      return 0;
    }
    EXPECT_LT(descriptor->canonical_asm_form_ordinal,
              descriptor_set_->asm_form_count);
    if (descriptor->canonical_asm_form_ordinal >=
        descriptor_set_->asm_form_count) {
      return 0;
    }
    return descriptor_set_->asm_forms[descriptor->canonical_asm_form_ordinal]
        .operand_index_count;
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

  bool LowOpHasDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    if (!loom_low_op_isa(op)) {
      return false;
    }
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    if (descriptor == nullptr) {
      return false;
    }
    return loom_low_op_descriptor_ordinal(op) ==
           loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                      descriptor);
  }

  std::vector<loom_op_t*> OpsForDescriptorRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    std::vector<loom_op_t*> filtered_ops;
    for (loom_op_t* op : Ops()) {
      if (LowOpHasDescriptorRef(op, descriptor_ref)) {
        filtered_ops.push_back(op);
      }
    }
    return filtered_ops;
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

  void ExpectAttrI64(loom_named_attr_slice_t attrs, iree_string_view_t name,
                     int64_t expected_value) const {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (iree_string_view_equal(String(attrs.entries[i].name_id), name)) {
        EXPECT_EQ(loom_attr_as_i64(attrs.entries[i].value), expected_value);
        return;
      }
    }
    FAIL() << "expected attr not found: " << ToString(name);
  }

  void ExpectLowConstU32(loom_value_id_t value,
                         loom_amdgpu_descriptor_ref_t descriptor_ref,
                         uint32_t expected_value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    ASSERT_FALSE(loom_value_is_block_arg(ir_value));
    const loom_op_t* op = loom_value_def_op(ir_value);
    ExpectLowConstDescriptorRef(op, descriptor_ref);
    loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)), "imm32");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_value);
  }

  void ExpectM0ConstOperand(loom_value_id_t value) const {
    ExpectLowConstU32(value, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM, 0);
  }

  void ExpectRegisterType(loom_value_id_t value, uint16_t reg_class_id,
                          uint32_t unit_count) const {
    loom_type_t expected_type = loom_low_register_type(
        descriptor_set_->stable_id, reg_class_id, unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_type, loom_module_value_type(module_, value)));
  }

  const loom_op_t* DefiningOp(loom_value_id_t value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    if (ir_value == nullptr || loom_value_is_block_arg(ir_value)) {
      return nullptr;
    }
    return loom_value_def_op(ir_value);
  }

  void ExpectSgprByteOffsetPart(loom_value_id_t actual_part,
                                loom_value_id_t expected_base,
                                int64_t expected_base_slice_offset,
                                loom_amdgpu_descriptor_ref_t descriptor_ref,
                                uint32_t expected_addend) const {
    const loom_op_t* add_op = DefiningOp(actual_part);
    ASSERT_NE(add_op, nullptr);
    ExpectLowOpDescriptorRef(add_op, descriptor_ref);
    loom_value_slice_t add_operands = loom_low_op_operands(add_op);
    ASSERT_EQ(add_operands.count, 2u);
    ASSERT_EQ(loom_low_op_results(add_op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(add_op), 0),
              actual_part);

    const loom_op_t* slice_op = DefiningOp(add_operands.values[0]);
    ASSERT_NE(slice_op, nullptr);
    ASSERT_TRUE(loom_low_slice_isa(slice_op));
    EXPECT_EQ(loom_low_slice_source(slice_op), expected_base);
    EXPECT_EQ(loom_low_slice_offset(slice_op), expected_base_slice_offset);

    ExpectLowConstU32(add_operands.values[1],
                      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, expected_addend);
    ExpectRegisterType(actual_part, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  }

  void ExpectSgprByteOffsetAddress(loom_value_id_t actual_address,
                                   loom_value_id_t expected_base,
                                   uint32_t expected_byte_offset) const {
    ExpectRegisterType(actual_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
    if (expected_byte_offset == 0) {
      EXPECT_EQ(actual_address, expected_base);
      return;
    }

    const loom_op_t* concat_op = DefiningOp(actual_address);
    ASSERT_NE(concat_op, nullptr);
    ASSERT_TRUE(loom_low_concat_isa(concat_op));
    loom_value_slice_t address_parts = loom_low_concat_sources(concat_op);
    ASSERT_EQ(address_parts.count, 2u);
    ExpectSgprByteOffsetPart(address_parts.values[0], expected_base,
                             /*expected_base_slice_offset=*/0,
                             LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
                             expected_byte_offset);
    ExpectSgprByteOffsetPart(address_parts.values[1], expected_base,
                             /*expected_base_slice_offset=*/1,
                             LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
                             /*expected_addend=*/0);
  }

  const loom_op_t* FindGlobalLoadSaddr(
      loom_amdgpu_descriptor_ref_t descriptor_ref,
      loom_value_id_t expected_base, uint32_t expected_byte_offset) const {
    for (loom_op_t* op : OpsForDescriptorRef(descriptor_ref)) {
      loom_value_slice_t operands = loom_low_op_operands(op);
      if ((operands.count != 2 && operands.count != 3) ||
          operands.values[1] != expected_base) {
        continue;
      }
      loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
      for (iree_host_size_t i = 0; i < attrs.count; ++i) {
        if (iree_string_view_equal(String(attrs.entries[i].name_id),
                                   IREE_SV("offset")) &&
            loom_attr_as_i64(attrs.entries[i].value) == expected_byte_offset) {
          return op;
        }
      }
    }
    return nullptr;
  }

  void ExpectReadfirstlane(const loom_op_t* op, loom_value_id_t expected_source,
                           loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op,
                             LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, 1u);
    EXPECT_EQ(operands.values[0], expected_source);
    ASSERT_EQ(loom_low_op_results(op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
    ExpectRegisterType(expected_source, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    ExpectRegisterType(expected_result, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  }

  void ExpectUniformGlobalLoadB32(loom_value_id_t base,
                                  uint32_t expected_byte_offset,
                                  loom_value_id_t expected_result) const {
    const loom_op_t* load_op =
        FindGlobalLoadSaddr(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
                            base, expected_byte_offset);
    ASSERT_NE(load_op, nullptr);
    const loom_value_id_t vector_value =
        loom_value_slice_get(loom_low_op_results(load_op), 0);
    const loom_op_t* readfirstlane_op = DefiningOp(expected_result);
    ASSERT_NE(readfirstlane_op, nullptr);
    ExpectReadfirstlane(readfirstlane_op, vector_value, expected_result);
  }

  void ExpectUniformGlobalLoadB64(loom_value_id_t base,
                                  uint32_t expected_byte_offset,
                                  loom_value_id_t expected_result) const {
    const loom_op_t* load_op =
        FindGlobalLoadSaddr(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
                            base, expected_byte_offset);
    ASSERT_NE(load_op, nullptr);
    const loom_value_id_t vector_value =
        loom_value_slice_get(loom_low_op_results(load_op), 0);
    const loom_op_t* concat_op = DefiningOp(expected_result);
    ASSERT_NE(concat_op, nullptr);
    ASSERT_TRUE(loom_low_concat_isa(concat_op));
    loom_value_slice_t concat_sources = loom_low_concat_sources(concat_op);
    ASSERT_EQ(concat_sources.count, 2u);
    for (iree_host_size_t i = 0; i < concat_sources.count; ++i) {
      const loom_op_t* readfirstlane_op = DefiningOp(concat_sources.values[i]);
      ASSERT_NE(readfirstlane_op, nullptr);
      const loom_value_id_t vector_lane =
          loom_low_op_operands(readfirstlane_op).values[0];
      const loom_op_t* slice_op = DefiningOp(vector_lane);
      ASSERT_NE(slice_op, nullptr);
      ASSERT_TRUE(loom_low_slice_isa(slice_op));
      EXPECT_EQ(loom_low_slice_source(slice_op), vector_value);
      EXPECT_EQ(loom_low_slice_offset(slice_op), (int64_t)i);
      ExpectReadfirstlane(readfirstlane_op, vector_lane,
                          concat_sources.values[i]);
    }
    ExpectRegisterType(expected_result, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  }

  void ExpectGlobalStoreB64(const loom_op_t* op,
                            loom_value_id_t expected_address) const {
    ExpectLowOpDescriptorRef(op,
                             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR));
    EXPECT_EQ(operands.values[2], expected_address);
    ExpectRegisterType(operands.values[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    ExpectRegisterType(operands.values[1], LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    if (operands.count == 4) {
      ExpectM0ConstOperand(operands.values[3]);
    }
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"), 0);
  }

  void ExpectSignalAddAtomic(const loom_op_t* op,
                             loom_value_id_t expected_signal_address) const {
    ExpectLowOpDescriptorRef(
        op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR));
    ExpectRegisterType(operands.values[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    ExpectRegisterType(operands.values[1], LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    ExpectSgprByteOffsetAddress(operands.values[2], expected_signal_address,
                                LOOM_AMDGPU_SIGNAL_VALUE_OFFSET);
    if (operands.count == 4) {
      ExpectM0ConstOperand(operands.values[3]);
    }
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
  }

  loom_value_id_t BuildSignalAddress() {
    loom_symbol_ref_t signal_symbol = AddSymbol(IREE_SV("test_signal"));
    loom_value_id_t signal_address = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_amdgpu_build_data_symbol_address(
        &builder_, descriptor_set_,
        (loom_amdgpu_data_symbol_address_t){
            /*.symbol=*/signal_symbol,
            /*.byte_offset=*/0,
        },
        LOOM_LOCATION_UNKNOWN, &signal_address));
    return signal_address;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuSignalTest, LoadsSignalNotificationValues) {
  const loom_value_id_t signal_address = BuildSignalAddress();
  loom_amdgpu_signal_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_signal_values(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN,
      &values));

  EXPECT_EQ(values.address, signal_address);
  ExpectUniformGlobalLoadB64(signal_address,
                             LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET,
                             values.event_mailbox_ptr);
  ExpectUniformGlobalLoadB32(signal_address, LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET,
                             values.event_id);

  ExpectRegisterType(values.event_mailbox_ptr, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                     2);
  ExpectRegisterType(values.event_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
}

TEST_F(AmdgpuSignalTest, AddsOneWithRdnaReleaseOrdering) {
  const loom_value_id_t signal_address = BuildSignalAddress();
  IREE_ASSERT_OK(loom_amdgpu_build_signal_add_one_release(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 63);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(vscnt_ops[0]), IREE_SV("vscnt"), 0);

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectSignalAddAtomic(atomic_ops[0], signal_address);
  EXPECT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);
}

TEST_F(AmdgpuSignalTest, AddsOneWithCdnaM0AndWriteback) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  const loom_value_id_t signal_address = BuildSignalAddress();
  IREE_ASSERT_OK(loom_amdgpu_build_signal_add_one_release(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_attrs(writeback_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectSignalAddAtomic(atomic_ops[0], signal_address);
  EXPECT_EQ(loom_low_op_operands(atomic_ops[0]).count, 4u);
  EXPECT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);
}

TEST_F(AmdgpuSignalTest, AddsOneWithGfx12SystemScope) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  const loom_value_id_t signal_address = BuildSignalAddress();
  IREE_ASSERT_OK(loom_amdgpu_build_signal_add_one_release(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectSignalAddAtomic(atomic_ops[0], signal_address);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuSignalTest, MasksMailboxMessageId) {
  const loom_value_id_t signal_address = BuildSignalAddress();
  loom_amdgpu_signal_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_signal_values(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN,
      &values));
  loom_value_id_t message_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_signal_mailbox_message_id(
      &builder_, descriptor_set_, values.event_id, LOOM_LOCATION_UNKNOWN,
      &message_id));

  std::vector<loom_op_t*> and_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  ASSERT_EQ(and_ops.size(), 1u);
  ExpectLowOpDescriptorRef(and_ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  loom_value_slice_t operands = loom_low_op_operands(and_ops[0]);
  ASSERT_EQ(operands.count, 2u);
  EXPECT_EQ(operands.values[0], values.event_id);
  ExpectLowConstU32(operands.values[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
                    LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_GFX9_11_12_MASK);
  EXPECT_EQ(loom_value_slice_get(loom_low_op_results(and_ops[0]), 0),
            message_id);
}

TEST_F(AmdgpuSignalTest, PokesMailboxWithRdnaReleaseStoreAndSendMessage) {
  const loom_value_id_t signal_address = BuildSignalAddress();
  loom_amdgpu_signal_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_signal_values(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN,
      &values));
  IREE_ASSERT_OK(loom_amdgpu_build_signal_poke_mailbox(
      &builder_, descriptor_set_, values.event_mailbox_ptr, values.event_id,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);

  std::vector<loom_op_t*> store_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(store_ops.size(), 1u);
  ExpectGlobalStoreB64(store_ops[0], values.event_mailbox_ptr);
  EXPECT_EQ(loom_low_op_attrs(store_ops[0]).count, 1u);

  std::vector<loom_op_t*> m0_moves =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0);
  ASSERT_EQ(m0_moves.size(), 1u);
  std::vector<loom_op_t*> sendmsg_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG);
  ASSERT_EQ(sendmsg_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(sendmsg_ops[0]), IREE_SV("message"),
                LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG);
  ASSERT_EQ(loom_low_op_operands(sendmsg_ops[0]).count, 1u);
  EXPECT_EQ(loom_low_op_operands(sendmsg_ops[0]).values[0],
            loom_value_slice_get(loom_low_op_results(m0_moves[0]), 0));
}

TEST_F(AmdgpuSignalTest, PokesMailboxWithCdnaM0Store) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  const loom_value_id_t signal_address = BuildSignalAddress();
  loom_amdgpu_signal_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_signal_values(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN,
      &values));
  IREE_ASSERT_OK(loom_amdgpu_build_signal_poke_mailbox(
      &builder_, descriptor_set_, values.event_mailbox_ptr, values.event_id,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_attrs(writeback_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);
  std::vector<loom_op_t*> store_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(store_ops.size(), 1u);
  ExpectGlobalStoreB64(store_ops[0], values.event_mailbox_ptr);
  EXPECT_EQ(loom_low_op_operands(store_ops[0]).count, 4u);
  ExpectAttrI64(loom_low_op_attrs(store_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(store_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuSignalTest, PokesMailboxWithGfx12SystemScope) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  const loom_value_id_t signal_address = BuildSignalAddress();
  loom_amdgpu_signal_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_signal_values(
      &builder_, descriptor_set_, signal_address, LOOM_LOCATION_UNKNOWN,
      &values));
  IREE_ASSERT_OK(loom_amdgpu_build_signal_poke_mailbox(
      &builder_, descriptor_set_, values.event_mailbox_ptr, values.event_id,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);
  std::vector<loom_op_t*> store_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(store_ops.size(), 1u);
  ExpectGlobalStoreB64(store_ops[0], values.event_mailbox_ptr);
  ExpectAttrI64(loom_low_op_attrs(store_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

}  // namespace
