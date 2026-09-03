// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"

#include <stdint.h>

#include <cstdio>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
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
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"
#include "loom/verify/verify.h"

namespace {

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

class AmdgpuSanitizerRaceReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
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
    loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder_, IREE_SV("amdgpu.rdna3.core"), &contract_set_key));
    loom_op_t* target_op = NULL;
    IREE_ASSERT_OK(loom_target_generic_build(
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
        /*linkage=*/0, contract_set_key,
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
    loom_symbol_ref_t callee = AddSymbol(IREE_SV("test_fn"));
    loom_op_t* function_op = NULL;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET,
        /*visibility=*/0, /*retain=*/0, /*cc=*/0,
        /*purity=*/0, /*inline_policy=*/0, /*allocation=*/0, /*schedule=*/0,
        /*descriptor_set=*/contract_set_key, target, /*abi=*/0,
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
    const uint32_t canonical_asm_form_ordinal =
        loom_low_descriptor_set_descriptor_view(descriptor_set_, descriptor)
            ->canonical_asm_form_ordinal;
    EXPECT_LT(canonical_asm_form_ordinal, descriptor_set_->asm_form_count);
    if (canonical_asm_form_ordinal >= descriptor_set_->asm_form_count) {
      return 0;
    }
    return descriptor_set_->asm_forms[canonical_asm_form_ordinal]
        .operand_index_count;
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

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
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

  void ExpectLowConstU32(loom_value_id_t value, uint32_t expected_value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    ASSERT_FALSE(loom_value_is_block_arg(ir_value));
    const loom_op_t* op = loom_value_def_op(ir_value);
    ASSERT_TRUE(loom_low_const_isa(op));
    ExpectLowConstDescriptorRef(op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
    loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)), "imm32");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_value);
  }

  loom_value_id_t BuildRegisterConstU32(
      loom_amdgpu_descriptor_ref_t descriptor_ref, uint16_t register_class,
      uint32_t value) {
    loom_type_t result_type =
        loom_low_register_type(descriptor_set_->stable_id, register_class, 1);
    const loom_low_descriptor_t* descriptor =
        loom_amdgpu_lookup_descriptor_ref(descriptor_set_, descriptor_ref);
    IREE_ASSERT(descriptor != nullptr);
    loom_string_id_t imm32_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&builder_, IREE_SV("imm32"), &imm32_id));
    loom_named_attr_t imm32_attr = {
        /*.name_id=*/imm32_id,
        /*.reserved=*/0,
        /*.value=*/loom_attr_i64(value),
    };
    loom_op_t* const_op = nullptr;
    IREE_CHECK_OK(loom_low_build_resolved_descriptor_const(
        &builder_, descriptor_set_, descriptor,
        loom_make_named_attr_slice(&imm32_attr, 1), result_type,
        LOOM_LOCATION_UNKNOWN, &const_op));
    return loom_low_const_result(const_op);
  }

  loom_value_id_t BuildVgprU32(uint32_t value) {
    return BuildRegisterConstU32(LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                 LOOM_AMDGPU_REG_CLASS_ID_VGPR, value);
  }

  loom_value_id_t BuildVgprU64(uint64_t value) {
    const loom_value_id_t low_value = BuildVgprU32((uint32_t)value);
    const loom_value_id_t high_value = BuildVgprU32((uint32_t)(value >> 32));
    const loom_value_id_t parts[] = {low_value, high_value};
    loom_type_t result_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    loom_op_t* concat_op = nullptr;
    IREE_CHECK_OK(loom_low_concat_build(&builder_, parts, IREE_ARRAYSIZE(parts),
                                        result_type, LOOM_LOCATION_UNKNOWN,
                                        &concat_op));
    return loom_low_concat_result(concat_op);
  }

  void ExpectRegisterType(loom_value_id_t value, uint16_t register_class,
                          uint32_t unit_count) const {
    const loom_type_t expected_type = loom_low_register_type(
        descriptor_set_->stable_id, register_class, unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_type, loom_module_value_type(module_, value)));
  }

  void ExpectStoreOp(const loom_op_t* op,
                     loom_amdgpu_descriptor_ref_t descriptor_ref,
                     const loom_amdgpu_feedback_packet_address_t& address,
                     uint32_t expected_byte_offset,
                     uint32_t expected_value_unit_count) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, PacketOperandCountForRef(descriptor_ref));
    EXPECT_EQ(operands.values[0], address.byte_offset);
    EXPECT_EQ(operands.values[2], address.base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    const loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        expected_value_unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
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

  iree_status_t BuildFeedbackValues(
      loom_amdgpu_feedback_config_values_t* out_config_values,
      loom_amdgpu_feedback_channel_header_values_t* out_channel_values,
      loom_amdgpu_feedback_packet_address_t* out_packet_address) {
    loom_symbol_ref_t config_symbol =
        AddSymbol(IREE_SV("iree_feedback_config"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
        &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
        out_config_values));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
        &builder_, descriptor_set_, out_config_values->channel_base,
        LOOM_LOCATION_UNKNOWN, out_channel_values));
    return loom_amdgpu_build_feedback_uniform_packet_address(
        &builder_, descriptor_set_, out_channel_values->ring_base,
        LOOM_LOCATION_UNKNOWN, out_packet_address);
  }

  loom_amdgpu_sanitizer_race_report_t MakeReport(
      const loom_amdgpu_feedback_config_values_t& config_values,
      const loom_amdgpu_feedback_channel_header_values_t& channel_values) {
    (void)config_values;
    (void)channel_values;
    return {
        /*.check_kind=*/BuildVgprU32(1),
        /*.flags=*/BuildVgprU32(2),
        /*.memory_space=*/BuildVgprU32(3),
        /*.current_access_kind=*/BuildVgprU32(4),
        /*.prior_access_kind=*/BuildVgprU32(5),
        /*.access_size=*/BuildVgprU32(8),
        /*.current_site_id=*/BuildVgprU64(0x1111222233334444ull),
        /*.prior_site_id=*/BuildVgprU64(0x5555666677778888ull),
        /*.memory_address=*/BuildVgprU64(0x1000200030004000ull),
        /*.shadow_address=*/BuildVgprU64(0x5000600070008000ull),
        /*.shadow_value=*/BuildVgprU64(0x9000A000B000C000ull),
        /*.current_workgroup_id_x=*/BuildVgprU32(11),
        /*.current_workgroup_id_y=*/BuildVgprU32(12),
        /*.current_workgroup_id_z=*/BuildVgprU32(13),
        /*.current_workitem_id_x=*/BuildVgprU32(21),
        /*.current_workitem_id_y=*/BuildVgprU32(22),
        /*.current_workitem_id_z=*/BuildVgprU32(23),
        /*.prior_workgroup_id_x=*/BuildVgprU32(31),
        /*.prior_workgroup_id_y=*/BuildVgprU32(32),
        /*.prior_workgroup_id_z=*/BuildVgprU32(33),
        /*.prior_workitem_id_x=*/BuildVgprU32(41),
        /*.prior_workitem_id_y=*/BuildVgprU32(42),
        /*.prior_workitem_id_z=*/BuildVgprU32(43),
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuSanitizerRaceReportTest, EmitsRaceReportPayloadStores) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(
      BuildFeedbackValues(&config_values, &channel_values, &packet_address));

  const loom_amdgpu_sanitizer_race_report_t report =
      MakeReport(config_values, channel_values);
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_race_report_payload(
      &builder_, descriptor_set_, &packet_address, &report,
      LOOM_LOCATION_UNKNOWN));

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));
  VerifyModuleOk();
  VerifyLowModuleOk();

  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 20u);
  ExpectStoreOp(b32_stores[0],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[0]).values[1],
                    LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  ExpectStoreOp(b32_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[1]).values[1],
                    LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION);
  ExpectStoreOp(b32_stores[2],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, payload_base + LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET, 1);
  ExpectStoreOp(b32_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[5], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[6], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET, 1);
  ExpectStoreOp(b32_stores[7],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[8], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[9], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[10], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[11], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[12], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[13], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[14], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[15], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[16], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[17], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[18], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[19], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET, 1);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 5u);
  ExpectStoreOp(
      b64_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET, 2);
  ExpectStoreOp(b64_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[2], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET, 2);
  ExpectStoreOp(b64_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET, 2);
}

TEST_F(AmdgpuSanitizerRaceReportTest, EmitsFatalRaceReportProducerCfg) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  const loom_amdgpu_feedback_packet_source_t source = {
      /*.dispatch_ptr=*/config_values.notify_signal,
      /*.workgroup_id_x=*/config_values.flags,
      /*.workitem_id_x=*/BuildVgprU32(9),
  };
  const loom_amdgpu_sanitizer_race_report_t report =
      MakeReport(config_values, channel_values);
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_race_report_terminate(
      &builder_, descriptor_set_, config_symbol, &source, &report,
      LOOM_LOCATION_UNKNOWN));

  VerifyModuleOk();
  VerifyLowModuleOk();

  loom_region_t* body = body_block_->parent_region;
  ASSERT_EQ(body->block_count, 8u);
  loom_block_t* config_block = body_block_;
  loom_block_t* feedback_block = loom_region_block(body, 1);
  loom_block_t* attempt_block = loom_region_block(body, 2);
  loom_block_t* reserved_block = loom_region_block(body, 3);
  loom_block_t* continuation_block = loom_region_block(body, 4);
  loom_block_t* report_block = loom_region_block(body, 5);
  loom_block_t* dropped_block = loom_region_block(body, 6);
  loom_block_t* terminal_block = loom_region_block(body, 7);

  const loom_op_t* config_terminator = loom_block_const_last_op(config_block);
  ASSERT_TRUE(loom_low_cond_br_isa(config_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(config_terminator), feedback_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(config_terminator), terminal_block);
  ExpectRegisterType(loom_low_cond_br_condition(config_terminator),
                     LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  const loom_op_t* continuation_terminator =
      loom_block_const_last_op(continuation_block);
  ASSERT_TRUE(loom_low_cond_br_isa(continuation_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(continuation_terminator), report_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(continuation_terminator),
            terminal_block);
  ExpectRegisterType(loom_low_cond_br_condition(continuation_terminator),
                     LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  const loom_op_t* report_terminator = loom_block_const_last_op(report_block);
  ASSERT_TRUE(loom_low_br_isa(report_terminator));
  EXPECT_EQ(loom_low_br_dest(report_terminator), terminal_block);

  const loom_op_t* terminal_terminator =
      loom_block_const_last_op(terminal_block);
  ASSERT_TRUE(loom_low_return_isa(terminal_terminator));
  std::vector<loom_op_t*> trap_ops = OpsForDescriptorRefInBlock(
      terminal_block, LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP);
  EXPECT_TRUE(trap_ops.empty());

  ASSERT_TRUE(loom_low_cond_br_isa(loom_block_const_last_op(feedback_block)));
  ASSERT_TRUE(loom_low_cond_br_isa(loom_block_const_last_op(attempt_block)));
  ASSERT_TRUE(loom_low_br_isa(loom_block_const_last_op(reserved_block)));
  ASSERT_TRUE(loom_low_br_isa(loom_block_const_last_op(dropped_block)));

  std::vector<loom_op_t*> report_b32_stores = OpsForDescriptorRefInBlock(
      report_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(report_b32_stores.size(), 27u);
  std::vector<loom_op_t*> report_b64_stores = OpsForDescriptorRefInBlock(
      report_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(report_b64_stores.size(), 11u);
}

TEST_F(AmdgpuSanitizerRaceReportTest, BranchesColdSitesToSharedReportIsland) {
  loom_region_t* body = body_block_->parent_region;
  loom_block_t* first_site_block = nullptr;
  IREE_ASSERT_OK(loom_region_insert_block(
      module_, body, (uint16_t)(body_block_->region_index + 1),
      &first_site_block));
  loom_block_t* second_site_block = nullptr;
  IREE_ASSERT_OK(loom_region_insert_block(
      module_, body, (uint16_t)(first_site_block->region_index + 1),
      &second_site_block));

  loom_builder_set_block(&builder_, body_block_);
  loom_op_t* entry_branch_op = nullptr;
  IREE_ASSERT_OK(loom_low_br_build(&builder_, first_site_block,
                                   /*args=*/nullptr, /*args_count=*/0,
                                   LOOM_LOCATION_UNKNOWN, &entry_branch_op));

  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_sanitizer_race_report_island_t island = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_race_report_island(
      &builder_, descriptor_set_, second_site_block, config_symbol,
      LOOM_LOCATION_UNKNOWN, &island));

  auto build_site_branch = [&](loom_block_t* site_block) {
    loom_builder_set_block(&builder_, site_block);
    loom_amdgpu_feedback_config_values_t config_values = {};
    IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
        &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
        &config_values));
    loom_amdgpu_feedback_channel_header_values_t channel_values = {};
    IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
        &builder_, descriptor_set_, config_values.channel_base,
        LOOM_LOCATION_UNKNOWN, &channel_values));
    const loom_amdgpu_feedback_packet_source_t source = {
        /*.dispatch_ptr=*/config_values.notify_signal,
        /*.workgroup_id_x=*/config_values.flags,
        /*.workitem_id_x=*/BuildVgprU32(9),
    };
    const loom_amdgpu_sanitizer_race_report_t report =
        MakeReport(config_values, channel_values);
    IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_race_report_branch(
        &builder_, descriptor_set_, &island, &source, &report,
        LOOM_LOCATION_UNKNOWN));
  };
  build_site_branch(first_site_block);
  build_site_branch(second_site_block);

  VerifyModuleOk();
  VerifyLowModuleOk();

  ASSERT_NE(island.entry_block, nullptr);
  ASSERT_NE(island.terminal_block, nullptr);
  EXPECT_EQ(island.entry_block->arg_count, 26u);
  EXPECT_EQ(island.source_args.dispatch_ptr,
            loom_block_arg_id(island.entry_block, 0));
  EXPECT_EQ(island.source_args.workgroup_id_x,
            loom_block_arg_id(island.entry_block, 1));
  EXPECT_EQ(island.source_args.workitem_id_x,
            loom_block_arg_id(island.entry_block, 2));
  EXPECT_EQ(island.report_args.check_kind,
            loom_block_arg_id(island.entry_block, 3));
  EXPECT_EQ(island.report_args.prior_workitem_id_z,
            loom_block_arg_id(island.entry_block, 25));
  ExpectRegisterType(island.source_args.dispatch_ptr,
                     LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  ExpectRegisterType(island.source_args.workgroup_id_x,
                     LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  ExpectRegisterType(island.source_args.workitem_id_x,
                     LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  ExpectRegisterType(island.report_args.current_site_id,
                     LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(island.report_args.prior_workitem_id_z,
                     LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);

  const loom_op_t* first_branch = loom_block_const_last_op(first_site_block);
  const loom_op_t* second_branch = loom_block_const_last_op(second_site_block);
  ASSERT_TRUE(loom_low_br_isa(first_branch));
  ASSERT_TRUE(loom_low_br_isa(second_branch));
  EXPECT_EQ(loom_low_br_dest(first_branch), island.entry_block);
  EXPECT_EQ(loom_low_br_dest(second_branch), island.entry_block);
  ASSERT_EQ(loom_low_br_args(first_branch).count, 26u);
  ASSERT_EQ(loom_low_br_args(second_branch).count, 26u);

  std::vector<loom_op_t*> report_b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  std::vector<loom_op_t*> report_b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  EXPECT_EQ(report_b32_stores.size(), 27u);
  EXPECT_EQ(report_b64_stores.size(), 11u);
  std::vector<loom_op_t*> trap_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP);
  EXPECT_TRUE(trap_ops.empty());
}

}  // namespace
