// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include <cstdio>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/renderer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/abi/signal.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"
#include "loom/verify/verify.h"

namespace {

constexpr int64_t kSystemCacheScope = LOOM_CACHE_SCOPE_SYSTEM;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_status_t EmitDiagnosticToStderr(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_type_formatter_t type_formatter = {};
  IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
      emission->error, emission->params, emission->param_count, type_formatter,
      &stream));
  std::fprintf(stderr, "low verifier: %.*s\n",
               (int)iree_string_builder_size(&builder),
               iree_string_builder_buffer(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

class AmdgpuFeedbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna3.core"))) {
      GTEST_SKIP() << "RDNA3 descriptor set is not linked.";
    }
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

  bool ResetModuleForDescriptorSet(iree_string_view_t key) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_lookup(&low_registry_.registry, key);
    if (descriptor_set == NULL) {
      return false;
    }
    if (module_ != NULL) {
      loom_module_free(module_);
    }
    module_ = NULL;
    body_block_ = NULL;
    builder_ = {};
    descriptor_set_ = descriptor_set;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       NULL, iree_allocator_system(),
                                       &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    BuildFunctionBody();
    return true;
  }

  void BuildFunctionBody() {
    loom_symbol_ref_t target = AddSymbol(IREE_SV("gfx_target"));
    loom_string_id_t representation_contract = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder_,
        loom_low_descriptor_set_string(descriptor_set_,
                                       descriptor_set_->key_string_offset),
        &representation_contract));
    loom_op_t* target_op = NULL;
    IREE_CHECK_OK(loom_target_generic_build(
        &builder_,
        LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CODEGEN_FORMAT |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ARTIFACT_FORMAT |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ABI |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CONTRACT_SET_KEY,
        LOOM_TARGET_GENERIC_KIND_REFERENCE, target,
        LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE, LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
        /*offset_bitwidth=*/0, /*max_workgroup_size_x=*/0,
        /*max_workgroup_size_y=*/0, /*max_workgroup_size_z=*/0,
        /*max_flat_workgroup_size=*/0, /*max_workgroup_storage_bytes=*/0,
        /*subgroup_size=*/0,
        /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
        /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
        /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
        /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
        /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
        /*memory_space_constant=*/0, /*memory_space_private=*/0,
        /*memory_space_host=*/0, /*memory_space_descriptor=*/0,
        LOOM_TARGET_ABI_OBJECT_FUNCTION,
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        /*linkage=*/0, representation_contract,
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
    loom_symbol_ref_t callee = AddSymbol(IREE_SV("test_fn"));
    loom_op_t* function_op = NULL;
    IREE_CHECK_OK(loom_low_func_def_build(
        &builder_, LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET,
        /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*allocation=*/0, /*schedule=*/0,
        /*descriptor_set=*/representation_contract, target, /*abi=*/0,
        loom_make_named_attr_slice(NULL, 0),
        loom_make_named_attr_slice(NULL, 0),
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        loom_make_named_attr_slice(NULL, 0),
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
    loom_region_t* body = body_block_->parent_region;
    loom_block_t* block = NULL;
    loom_region_for_each_block(body, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) { ops.push_back(op); }
    }
    return ops;
  }

  std::vector<loom_op_t*> OpsInBlock(loom_block_t* block) const {
    std::vector<loom_op_t*> ops;
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) { ops.push_back(op); }
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
    iree_host_size_t count = 0;
    for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
         ++i) {
      const uint32_t operand_row = descriptor->operand_start + i;
      EXPECT_LT(operand_row, descriptor_set_->operand_count);
      if (operand_row >= descriptor_set_->operand_count) {
        return count;
      }
      const loom_low_operand_t* operand =
          &descriptor_set_->operands[operand_row];
      if (loom_low_operand_role_is_packet_operand(operand->role)) {
        ++count;
      }
    }
    return count;
  }

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
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
    return loom_low_op_descriptor(op) ==
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

  std::vector<loom_op_t*> OpsForDescriptorRefInBlock(
      loom_block_t* block, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    std::vector<loom_op_t*> filtered_ops;
    for (loom_op_t* op : OpsInBlock(block)) {
      if (LowOpHasDescriptorRef(op, descriptor_ref)) {
        filtered_ops.push_back(op);
      }
    }
    return filtered_ops;
  }

  void VerifyModuleOk() {
    loom_verify_options_t options = {
        /*.sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.max_errors=*/20,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  void VerifyLowModuleOk() {
    loom_low_verify_options_t options = {};
    options.descriptor_registry = &low_registry_.registry;
    options.emitter = {EmitDiagnosticToStderr, NULL};
    options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module_);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &options, &scratch, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_status_t BuildFeedbackPacketAddress(
      loom_amdgpu_feedback_packet_address_t* out_packet_address) {
    *out_packet_address = {};
    loom_amdgpu_feedback_config_values_t config_values = {};
    loom_amdgpu_feedback_channel_header_values_t channel_values = {};
    IREE_RETURN_IF_ERROR(
        BuildFeedbackChannelValues(&config_values, &channel_values));
    return loom_amdgpu_build_feedback_uniform_packet_address(
        &builder_, descriptor_set_, channel_values.ring_base,
        LOOM_LOCATION_UNKNOWN, out_packet_address);
  }

  iree_status_t BuildFeedbackChannelValues(
      loom_amdgpu_feedback_config_values_t* out_config_values,
      loom_amdgpu_feedback_channel_header_values_t* out_channel_values) {
    loom_symbol_ref_t config_symbol =
        AddSymbol(IREE_SV("iree_feedback_config"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
        &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
        out_config_values));
    return loom_amdgpu_build_feedback_channel_header_values(
        &builder_, descriptor_set_, out_config_values->channel_base,
        LOOM_LOCATION_UNKNOWN, out_channel_values);
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

  void ExpectAttrAbsent(loom_named_attr_slice_t attrs,
                        iree_string_view_t name) const {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      EXPECT_FALSE(
          iree_string_view_equal(String(attrs.entries[i].name_id), name))
          << "unexpected attr found: " << ToString(name);
    }
  }

  void ExpectLowConstU32(loom_value_id_t value, uint32_t expected_value) const {
    ExpectLowConstU32(value, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                      expected_value);
  }

  void ExpectLowConstU32(loom_value_id_t value,
                         loom_amdgpu_descriptor_ref_t descriptor_ref,
                         uint32_t expected_value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    ASSERT_FALSE(loom_value_is_block_arg(ir_value));
    const loom_op_t* op = loom_value_def_op(ir_value);
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(loom_low_const_isa(op));
    ExpectLowConstDescriptorRef(op, descriptor_ref);
    loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)), "imm32");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_value);
  }

  void ExpectLowConstDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_const_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_const_descriptor(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
  }

  const loom_op_t* DefiningOp(loom_value_id_t value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    if (ir_value == nullptr || loom_value_is_block_arg(ir_value)) {
      return nullptr;
    }
    return loom_value_def_op(ir_value);
  }

  void ExpectRegisterType(loom_value_id_t value, uint16_t register_class,
                          uint32_t unit_count) const {
    const loom_type_t expected_type = loom_low_register_type(
        descriptor_set_->stable_id, register_class, unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_type, loom_module_value_type(module_, value)));
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

  void ExpectRel32Attrs(const loom_op_t* op, loom_symbol_ref_t expected_symbol,
                        int64_t expected_byte_offset) const {
    loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    ASSERT_EQ(attrs.count, 2u);
    ASSERT_EQ(ToString(String(attrs.entries[0].name_id)), "byte_offset");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_byte_offset);
    ASSERT_EQ(ToString(String(attrs.entries[1].name_id)), "symbol");
    loom_symbol_ref_t actual_symbol =
        loom_attr_as_symbol(attrs.entries[1].value);
    EXPECT_EQ(actual_symbol.module_id, expected_symbol.module_id);
    EXPECT_EQ(actual_symbol.symbol_id, expected_symbol.symbol_id);
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
      bool found_offset = false;
      for (iree_host_size_t i = 0; i < attrs.count; ++i) {
        if (iree_string_view_equal(String(attrs.entries[i].name_id),
                                   IREE_SV("offset")) &&
            loom_attr_as_i64(attrs.entries[i].value) == expected_byte_offset) {
          found_offset = true;
          break;
        }
      }
      if (found_offset) {
        return op;
      }
    }
    return nullptr;
  }

  const loom_op_t* FindGlobalLoadSaddrInBlock(
      loom_block_t* block, loom_amdgpu_descriptor_ref_t descriptor_ref,
      loom_value_id_t expected_base, uint32_t expected_byte_offset) const {
    for (loom_op_t* op : OpsForDescriptorRefInBlock(block, descriptor_ref)) {
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

  void ExpectStoreOpWithBase(const loom_op_t* op,
                             loom_amdgpu_descriptor_ref_t descriptor_ref,
                             loom_value_id_t expected_base,
                             uint32_t expected_byte_offset,
                             uint32_t expected_value_unit_count) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, PacketOperandCountForRef(descriptor_ref));
    EXPECT_EQ(operands.values[2], expected_base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        expected_value_unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  void ExpectStoreOp(const loom_op_t* op,
                     loom_amdgpu_descriptor_ref_t descriptor_ref,
                     const loom_amdgpu_feedback_packet_address_t& address,
                     uint32_t expected_byte_offset,
                     uint32_t expected_value_unit_count) const {
    ExpectStoreOpWithBase(op, descriptor_ref, address.base,
                          expected_byte_offset, expected_value_unit_count);
    EXPECT_EQ(loom_low_op_operands(op).values[0], address.byte_offset);
  }

  void ExpectPublishStateStore(
      const loom_op_t* op,
      const loom_amdgpu_feedback_packet_address_t& address) const {
    ExpectStoreOp(op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                  address, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, 1);
    ExpectLowConstU32(loom_low_op_operands(op).values[1],
                      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY);
  }

  void ExpectGlobalAtomicAddU64(const loom_op_t* op,
                                loom_value_id_t expected_base,
                                uint32_t expected_byte_offset) const {
    ExpectLowOpDescriptorRef(
        op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR));
    ExpectSgprByteOffsetAddress(operands.values[2], expected_base,
                                expected_byte_offset);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrAbsent(loom_low_op_attrs(op), IREE_SV("offset"));
  }

  void ExpectGlobalAtomicCompareExchangeU64(
      const loom_op_t* op, loom_value_id_t expected_base,
      uint32_t expected_byte_offset, loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(
        op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(
        operands.count,
        PacketOperandCountForRef(
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR));
    ExpectSgprByteOffsetAddress(operands.values[2], expected_base,
                                expected_byte_offset);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_pair_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4);
    EXPECT_TRUE(
        loom_type_equal(expected_pair_type,
                        loom_module_value_type(module_, operands.values[1])));
    const loom_value_t* pair_value =
        loom_module_value(module_, operands.values[1]);
    ASSERT_FALSE(loom_value_is_block_arg(pair_value));
    ASSERT_TRUE(loom_low_concat_isa(loom_value_def_op(pair_value)));
    ASSERT_EQ(loom_low_op_results(op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
    loom_type_t expected_result_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_result_type,
                        loom_module_value_type(module_, expected_result)));
    ExpectAttrAbsent(loom_low_op_attrs(op), IREE_SV("offset"));
  }

  void ExpectGlobalLoadB64(const loom_op_t* op, loom_value_id_t expected_base,
                           uint32_t expected_byte_offset,
                           loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op,
                             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR));
    EXPECT_EQ(operands.values[1], expected_base);
    if (operands.count == 3) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[2]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    ASSERT_EQ(loom_low_op_results(op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
    loom_type_t expected_result_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_result_type,
                        loom_module_value_type(module_, expected_result)));
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_ = {};
};

TEST_F(AmdgpuFeedbackTest, LoadsCommonFeedbackConfigValues) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &values));

  ExpectUniformGlobalLoadB32(
      values.address, LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET, values.flags);
  ExpectUniformGlobalLoadB64(values.address,
                             LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET,
                             values.channel_base);
  ExpectUniformGlobalLoadB64(values.address,
                             LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET,
                             values.notify_signal);
  ExpectUniformGlobalLoadB64(values.address,
                             LOOM_AMDGPU_FEEDBACK_CONFIG_SOURCE_CONTEXT_OFFSET,
                             values.source_context);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(sgpr_type,
                              loom_module_value_type(module_, values.flags)));
  loom_type_t sgpr_x2_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  EXPECT_TRUE(loom_type_equal(sgpr_x2_type,
                              loom_module_value_type(module_, values.address)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, values.channel_base)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, values.notify_signal)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, values.source_context)));
}

TEST_F(AmdgpuFeedbackTest, TestsFeedbackConfigEnabledFlag) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &values));

  loom_value_id_t enabled_scc = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_enabled_scc(
      &builder_, descriptor_set_, values.flags, LOOM_LOCATION_UNKNOWN,
      &enabled_scc));
  ExpectRegisterType(enabled_scc, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  std::vector<loom_op_t*> and_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  ASSERT_EQ(and_ops.size(), 1u);
  EXPECT_EQ(loom_low_op_operands(and_ops[0]).values[0], values.flags);
  std::vector<loom_op_t*> compare_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_I32);
  ASSERT_EQ(compare_ops.size(), 1u);
  EXPECT_EQ(loom_value_slice_get(loom_low_op_results(compare_ops[0]), 0),
            enabled_scc);

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));
  VerifyModuleOk();
  VerifyLowModuleOk();
}

TEST_F(AmdgpuFeedbackTest, LoadsFeedbackChannelHeaderValues) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  EXPECT_EQ(channel_values.address, config_values.channel_base);
  ExpectUniformGlobalLoadB32(channel_values.address,
                             LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET,
                             channel_values.record_length);
  ExpectUniformGlobalLoadB32(channel_values.address,
                             LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET,
                             channel_values.abi_version);
  ExpectUniformGlobalLoadB32(channel_values.address,
                             LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET,
                             channel_values.flags);
  ExpectUniformGlobalLoadB64(channel_values.address,
                             LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET,
                             channel_values.ring_base);
  ExpectUniformGlobalLoadB64(channel_values.address,
                             LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET,
                             channel_values.ring_capacity);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(
      sgpr_type,
      loom_module_value_type(module_, channel_values.record_length)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_type, loom_module_value_type(module_, channel_values.abi_version)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_type, loom_module_value_type(module_, channel_values.flags)));
  loom_type_t sgpr_x2_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, channel_values.ring_base)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type,
      loom_module_value_type(module_, channel_values.ring_capacity)));
}

TEST_F(AmdgpuFeedbackTest, BuildsUniformPacketAddress) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));

  EXPECT_EQ(packet_address.base, channel_values.ring_base);
  ExpectLowConstU32(packet_address.byte_offset, 0);
  loom_type_t vgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  EXPECT_TRUE(loom_type_equal(
      vgpr_type, loom_module_value_type(module_, packet_address.byte_offset)));
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCount) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCountWithCdnaM0) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ASSERT_EQ(loom_low_op_operands(atomic_ops[0]).count, 4u);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCountWithGfx12SystemScope) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna4.core"))) {
    GTEST_SKIP() << "RDNA4 descriptor set is not linked.";
  }
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHead) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ASSERT_EQ(loom_low_op_attrs(load_op).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("glc"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("dlc"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHeadWithCdnaM0) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ASSERT_EQ(loom_low_op_operands(load_op).count, 3u);
  ASSERT_EQ(loom_low_op_attrs(load_op).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHeadWithGfx12SystemScope) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna4.core"))) {
    GTEST_SKIP() << "RDNA4 descriptor set is not linked.";
  }
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithRdnaAcquireOrdering) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ASSERT_EQ(loom_low_op_attrs(load_op).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("glc"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("dlc"), 1);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 63);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV).size(),
      1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV).size(),
      1u);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithCdnaAcquireOrdering) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ASSERT_EQ(loom_low_op_operands(load_op).count, 3u);
  ASSERT_EQ(loom_low_op_attrs(load_op).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithGfx12AcquireOrdering) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna4.core"))) {
    GTEST_SKIP() << "RDNA4 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  const loom_op_t* load_op = FindGlobalLoadSaddr(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR, channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET);
  ASSERT_NE(load_op, nullptr);
  ExpectGlobalLoadB64(load_op, channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ExpectAttrI64(loom_low_op_attrs(load_op), IREE_SV("scope"),
                kSystemCacheScope);

  std::vector<loom_op_t*> loadcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT);
  ASSERT_EQ(loadcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(loadcnt_ops[0]), IREE_SV("loadcnt"), 0);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithRdnaOrdering) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 2u);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV).size(),
      1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV).size(),
      1u);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithCdnaOrdering) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ASSERT_EQ(loom_low_op_operands(atomic_ops[0]).count, 4u);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 0u);

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);
  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithGfx12Ordering) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna4.core"))) {
    GTEST_SKIP() << "RDNA4 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);
  std::vector<loom_op_t*> loadcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT);
  ASSERT_EQ(loadcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(loadcnt_ops[0]), IREE_SV("loadcnt"), 0);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, BuildsSingleReservationAttempt) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  const uint32_t packet_length =
      (uint32_t)loom_amdgpu_feedback_packet_length(/*payload_length=*/96);
  loom_amdgpu_feedback_reservation_attempt_t attempt = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_attempt(
      &builder_, descriptor_set_, channel_values.address, reservation_head,
      packet_length, LOOM_LOCATION_UNKNOWN, &attempt));

  EXPECT_EQ(attempt.expected_head, reservation_head);
  ExpectRegisterType(attempt.next_head, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(attempt.observed_head, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(attempt.cas_succeeded, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);

  std::vector<loom_op_t*> add_lo_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32);
  ASSERT_EQ(add_lo_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_results(add_lo_ops[0]).count, 2u);
  std::vector<loom_op_t*> add_hi_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32);
  ASSERT_EQ(add_hi_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_results(add_hi_ops[0]).count, 2u);
  ASSERT_EQ(loom_low_op_operands(add_hi_ops[0]).count, 3u);
  EXPECT_EQ(loom_low_op_operands(add_hi_ops[0]).values[2],
            loom_value_slice_get(loom_low_op_results(add_lo_ops[0]), 1));

  const loom_op_t* next_head_concat = DefiningOp(attempt.next_head);
  ASSERT_NE(next_head_concat, nullptr);
  ASSERT_TRUE(loom_low_concat_isa(next_head_concat));
  loom_value_slice_t next_head_parts =
      loom_low_concat_sources(next_head_concat);
  ASSERT_EQ(next_head_parts.count, 2u);
  EXPECT_EQ(next_head_parts.values[0],
            loom_value_slice_get(loom_low_op_results(add_lo_ops[0]), 0));
  EXPECT_EQ(next_head_parts.values[1],
            loom_value_slice_get(loom_low_op_results(add_hi_ops[0]), 0));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
      attempt.observed_head);
  const loom_value_id_t compare_exchange_pair =
      loom_low_op_operands(atomic_ops[0]).values[1];
  const loom_op_t* pair_concat = DefiningOp(compare_exchange_pair);
  ASSERT_NE(pair_concat, nullptr);
  ASSERT_TRUE(loom_low_concat_isa(pair_concat));
  loom_value_slice_t pair_sources = loom_low_concat_sources(pair_concat);
  ASSERT_EQ(pair_sources.count, 2u);
  EXPECT_EQ(pair_sources.values[0], attempt.next_head);
  EXPECT_EQ(pair_sources.values[1], attempt.expected_head);

  std::vector<loom_op_t*> compare_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32);
  ASSERT_EQ(compare_ops.size(), 2u);
  std::vector<loom_op_t*> and_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64);
  ASSERT_EQ(and_ops.size(), 1u);
  loom_value_slice_t and_operands = loom_low_op_operands(and_ops[0]);
  ASSERT_EQ(and_operands.count, 2u);
  EXPECT_EQ(and_operands.values[0],
            loom_value_slice_get(loom_low_op_results(compare_ops[0]), 0));
  EXPECT_EQ(and_operands.values[1],
            loom_value_slice_get(loom_low_op_results(compare_ops[1]), 0));
  EXPECT_EQ(attempt.cas_succeeded,
            loom_value_slice_get(loom_low_op_results(and_ops[0]), 0));
}

TEST_F(AmdgpuFeedbackTest, BuildsReservationCfgWithHotFallthrough) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  const uint32_t packet_length =
      (uint32_t)loom_amdgpu_feedback_packet_length(/*payload_length=*/96);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation(
      &builder_, descriptor_set_, channel_values.address,
      channel_values.ring_base, channel_values.ring_capacity, packet_length,
      LOOM_LOCATION_UNKNOWN, &reservation));

  loom_region_t* body = body_block_->parent_region;
  ASSERT_EQ(body->block_count, 5u);
  loom_block_t* check_block = body_block_;
  loom_block_t* attempt_block = loom_region_block(body, 1);
  loom_block_t* reserved_block = loom_region_block(body, 2);
  loom_block_t* continuation_block = loom_region_block(body, 3);
  loom_block_t* dropped_block = loom_region_block(body, 4);
  EXPECT_EQ(builder_.ip.block, continuation_block);

  EXPECT_EQ(reservation.packet_address.base, channel_values.ring_base);
  EXPECT_EQ(reservation.sequence,
            loom_block_arg_id(continuation_block, /*arg_index=*/0));
  EXPECT_EQ(reservation.packet_address.byte_offset,
            loom_block_arg_id(continuation_block, /*arg_index=*/1));
  EXPECT_EQ(reservation.reserved_mask,
            loom_block_arg_id(continuation_block, /*arg_index=*/2));
  ExpectRegisterType(reservation.sequence, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(reservation.packet_address.byte_offset,
                     LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  ExpectRegisterType(reservation.reserved_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                     2);

  const loom_op_t* check_terminator = loom_block_const_last_op(check_block);
  ASSERT_TRUE(loom_low_cond_br_isa(check_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(check_terminator), attempt_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(check_terminator), dropped_block);
  ExpectRegisterType(loom_low_cond_br_condition(check_terminator),
                     LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  const loom_op_t* attempt_terminator = loom_block_const_last_op(attempt_block);
  ASSERT_TRUE(loom_low_cond_br_isa(attempt_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(attempt_terminator), reserved_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(attempt_terminator), check_block);
  ExpectRegisterType(loom_low_cond_br_condition(attempt_terminator),
                     LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  const loom_op_t* reserved_terminator =
      loom_block_const_last_op(reserved_block);
  ASSERT_TRUE(loom_low_br_isa(reserved_terminator));
  EXPECT_EQ(loom_low_br_dest(reserved_terminator), continuation_block);
  ASSERT_EQ(loom_low_br_args(reserved_terminator).count, 3u);

  const loom_op_t* dropped_terminator = loom_block_const_last_op(dropped_block);
  ASSERT_TRUE(loom_low_br_isa(dropped_terminator));
  EXPECT_EQ(loom_low_br_dest(dropped_terminator), continuation_block);
  ASSERT_EQ(loom_low_br_args(dropped_terminator).count, 3u);

  EXPECT_NE(FindGlobalLoadSaddrInBlock(
                check_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
                channel_values.address,
                LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET),
            nullptr);
  EXPECT_NE(FindGlobalLoadSaddrInBlock(
                check_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
                channel_values.address,
                LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET),
            nullptr);
  ASSERT_EQ(OpsForDescriptorRefInBlock(check_block,
                                       LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULE_U32)
                .size(),
            1u);
  ASSERT_EQ(OpsForDescriptorRefInBlock(check_block,
                                       LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64)
                .size(),
            1u);
  ASSERT_EQ(OpsForDescriptorRefInBlock(
                attempt_block,
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR)
                .size(),
            1u);
  ASSERT_EQ(OpsForDescriptorRefInBlock(attempt_block,
                                       LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64)
                .size(),
            1u);
  ASSERT_EQ(OpsForDescriptorRefInBlock(reserved_block,
                                       LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32)
                .size(),
            1u);
  ASSERT_EQ(
      OpsForDescriptorRefInBlock(
          dropped_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR)
          .size(),
      1u);
  ASSERT_EQ(OpsForDescriptorRef(
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR)
                .size(),
            1u);

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));
  VerifyModuleOk();
  VerifyLowModuleOk();
}

TEST_F(AmdgpuFeedbackTest, TestsReservationSucceededMask) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  const uint32_t packet_length =
      (uint32_t)loom_amdgpu_feedback_packet_length(/*payload_length=*/64);
  loom_amdgpu_feedback_reservation_t reservation = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation(
      &builder_, descriptor_set_, channel_values.address,
      channel_values.ring_base, channel_values.ring_capacity, packet_length,
      LOOM_LOCATION_UNKNOWN, &reservation));

  loom_value_id_t reserved_scc = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_succeeded_scc(
      &builder_, descriptor_set_, reservation.reserved_mask,
      LOOM_LOCATION_UNKNOWN, &reserved_scc));
  ExpectRegisterType(reserved_scc, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  loom_region_t* body = body_block_->parent_region;
  ASSERT_EQ(body->block_count, 5u);
  loom_block_t* continuation_block = loom_region_block(body, 3);
  std::vector<loom_op_t*> compare_ops = OpsForDescriptorRefInBlock(
      continuation_block, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64);
  ASSERT_EQ(compare_ops.size(), 1u);
  EXPECT_EQ(loom_value_slice_get(loom_low_op_results(compare_ops[0]), 0),
            reserved_scc);

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));
  VerifyModuleOk();
  VerifyLowModuleOk();
}

TEST_F(AmdgpuFeedbackTest, EmitsPacketHeaderStores) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));

  loom_amdgpu_feedback_packet_header_t header = {
      /*.record_length=*/(uint32_t)loom_amdgpu_feedback_packet_length(
          /*payload_length=*/64),
      /*.kind=*/LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      /*.flags=*/LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      /*.sequence=*/channel_values.ring_capacity,
      /*.source_dispatch_ptr=*/config_values.notify_signal,
      /*.source_workgroup_id_x=*/config_values.flags,
      /*.source_workitem_id_x=*/channel_values.flags,
      /*.source_context=*/config_values.source_context,
  };
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_packet_header(
      &builder_, descriptor_set_, &packet_address, &header,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 6u);
  ExpectStoreOp(
      b32_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[0]).values[1],
                    header.record_length);
  ExpectStoreOp(
      b32_stores[1], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, 1);
  ExpectLowConstU32(
      loom_low_op_operands(b32_stores[1]).values[1],
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH |
          ((uint32_t)LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN << 16));
  ExpectStoreOp(b32_stores[2],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[2]).values[1],
                    LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC);
  ExpectStoreOp(b32_stores[3],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, 1);
  ExpectStoreOp(b32_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[4]).values[1],
            header.source_workgroup_id_x);
  ExpectStoreOp(b32_stores[5],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[5]).values[1],
            header.source_workitem_id_x);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 5u);
  ExpectStoreOp(b64_stores[0],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[0]).values[1], header.sequence);
  ExpectStoreOp(b64_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[1]).values[1],
            header.source_dispatch_ptr);
  ExpectStoreOp(
      b64_stores[2], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_CONTEXT_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[2]).values[1],
            header.source_context);
  ExpectStoreOp(
      b64_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[4], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, 2);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithRdnaReleaseOrdering) {
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 63);

  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(vscnt_ops[0]), IREE_SV("vscnt"), 0);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithCdnaReleaseOrdering) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "CDNA3 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_attrs(writeback_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithGfx12SystemScope) {
  if (!ResetModuleForDescriptorSet(IREE_SV("amdgpu.rdna4.core"))) {
    GTEST_SKIP() << "RDNA4 descriptor set is not linked.";
  }
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketAndNotifiesHost) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet(
      &builder_, descriptor_set_, &packet_address, config_values.notify_signal,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 3u);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 3u);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(atomic_ops[0], config_values.notify_signal,
                           LOOM_AMDGPU_SIGNAL_VALUE_OFFSET);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 1u);
  const loom_value_id_t mailbox_ptr =
      loom_low_op_operands(b64_stores[0]).values[2];
  ExpectUniformGlobalLoadB64(config_values.notify_signal,
                             LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET,
                             mailbox_ptr);
  ExpectStoreOpWithBase(
      b64_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      mailbox_ptr, /*expected_byte_offset=*/0, /*expected_value_unit_count=*/2);

  std::vector<loom_op_t*> and_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  ASSERT_EQ(and_ops.size(), 1u);
  loom_value_slice_t and_operands = loom_low_op_operands(and_ops[0]);
  ASSERT_EQ(and_operands.count, 2u);
  const loom_value_id_t event_id = and_operands.values[0];
  ExpectUniformGlobalLoadB32(config_values.notify_signal,
                             LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET, event_id);

  std::vector<loom_op_t*> m0_moves =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0);
  ASSERT_EQ(m0_moves.size(), 1u);
  EXPECT_EQ(loom_low_op_operands(m0_moves[0]).values[0],
            loom_value_slice_get(loom_low_op_results(and_ops[0]), 0));
  std::vector<loom_op_t*> sendmsg_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG);
  ASSERT_EQ(sendmsg_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(sendmsg_ops[0]), IREE_SV("message"),
                LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG);
  ASSERT_EQ(loom_low_op_operands(sendmsg_ops[0]).count, 1u);
  EXPECT_EQ(loom_low_op_operands(sendmsg_ops[0]).values[0],
            loom_value_slice_get(loom_low_op_results(m0_moves[0]), 0));
}

}  // namespace
