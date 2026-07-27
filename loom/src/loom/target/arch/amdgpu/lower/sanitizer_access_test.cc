// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_access.h"

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
#include "loom/target/arch/amdgpu/abi/asan.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_report.h"
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

class AmdgpuSanitizerAccessTest : public ::testing::Test {
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
        /*max_flat_workgroup_size=*/0, /*subgroup_size=*/0,
        /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
        /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
        /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
        /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
        /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
        /*memory_space_constant=*/0, /*memory_space_private=*/0,
        /*memory_space_host=*/0, /*memory_space_descriptor=*/0,
        LOOM_TARGET_ABI_OBJECT_FUNCTION,
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        /*linkage=*/0, /*hal_buffer_resource_flags=*/0, contract_set_key,
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
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

  void ExpectRegisterType(loom_value_id_t value, uint16_t register_class,
                          uint32_t unit_count) const {
    const loom_type_t expected_type = loom_low_register_type(
        descriptor_set_->stable_id, register_class, unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_type, loom_module_value_type(module_, value)));
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
    loom_low_verify_options_t options = {
        /*.descriptor_registry=*/&low_registry_.registry,
        /*.target_selection=*/{},
        /*.emitter=*/{EmitDiagnosticToStderr, NULL},
        /*.provider_list=*/{},
        /*.max_errors=*/20,
    };
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module_);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &options, &scratch, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_status_t BuildFaultAddress(loom_symbol_ref_t asan_config_symbol,
                                  loom_value_id_t* out_fault_address) {
    return loom_amdgpu_build_data_symbol_address(
        &builder_, descriptor_set_,
        (loom_amdgpu_data_symbol_address_t){
            /*.symbol=*/asan_config_symbol,
            /*.byte_offset=*/0,
        },
        LOOM_LOCATION_UNKNOWN, out_fault_address);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuSanitizerAccessTest, EmitsSingleShadowByteAccessCheck) {
  loom_symbol_ref_t asan_config_symbol = AddSymbol(IREE_SV("iree_asan_config"));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(BuildFaultAddress(asan_config_symbol, &fault_address));

  loom_amdgpu_sanitizer_access_check_t check = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_check(
      &builder_, descriptor_set_, asan_config_symbol, fault_address,
      /*access_size=*/1, /*wavefront_size=*/32, LOOM_LOCATION_UNKNOWN, &check));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));

  VerifyModuleOk();
  VerifyLowModuleOk();

  ExpectRegisterType(check.failure_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  ExpectRegisterType(check.shadow_address, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(check.shadow_value, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);

  std::vector<loom_op_t*> config_loads = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY);
  ASSERT_EQ(config_loads.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(config_loads[0]), IREE_SV("offset"),
                LOOM_AMDGPU_ASAN_CONFIG_SHADOW_BASE_OFFSET);

  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8).size(),
            1u);
  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 63);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT).size(),
      2u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT).size(),
      1u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32).size(),
            1u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32).size(), 2u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64).size(),
            1u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64).size(),
            1u);
  EXPECT_TRUE(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32).empty());
}

TEST_F(AmdgpuSanitizerAccessTest, EmitsFirstAndLastShadowByteAccessCheck) {
  loom_symbol_ref_t asan_config_symbol = AddSymbol(IREE_SV("iree_asan_config"));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(BuildFaultAddress(asan_config_symbol, &fault_address));

  loom_amdgpu_sanitizer_access_check_t check = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_check(
      &builder_, descriptor_set_, asan_config_symbol, fault_address,
      /*access_size=*/8, /*wavefront_size=*/32, LOOM_LOCATION_UNKNOWN, &check));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));

  VerifyModuleOk();
  VerifyLowModuleOk();

  ExpectRegisterType(check.failure_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  ExpectRegisterType(check.shadow_address, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(check.shadow_value, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8).size(),
            2u);
  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 2u);
  for (loom_op_t* waitcnt_op : waitcnt_ops) {
    ExpectAttrI64(loom_low_op_attrs(waitcnt_op), IREE_SV("vmcnt"), 0);
    ExpectAttrI64(loom_low_op_attrs(waitcnt_op), IREE_SV("lgkmcnt"), 63);
  }
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT).size(),
      4u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT).size(),
      2u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32).size(),
            2u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32).size(), 4u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64).size(),
            2u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64).size(),
            3u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32).size(), 3u);
}

TEST_F(AmdgpuSanitizerAccessTest, FeedsMaskedFailuresToSharedReportIsland) {
  loom_symbol_ref_t asan_config_symbol = AddSymbol(IREE_SV("iree_asan_config"));
  loom_symbol_ref_t feedback_config_symbol =
      AddSymbol(IREE_SV("iree_feedback_config"));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(BuildFaultAddress(asan_config_symbol, &fault_address));

  loom_amdgpu_sanitizer_access_check_t check = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_check(
      &builder_, descriptor_set_, asan_config_symbol, fault_address,
      /*access_size=*/8, /*wavefront_size=*/32, LOOM_LOCATION_UNKNOWN, &check));

  const loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  loom_op_t* workgroup_id_op = NULL;
  IREE_ASSERT_OK(loom_low_slice_build(&builder_, fault_address, /*offset=*/0,
                                      sgpr_type, LOOM_LOCATION_UNKNOWN,
                                      &workgroup_id_op));
  const loom_value_id_t workgroup_id = loom_low_slice_result(workgroup_id_op);
  const loom_amdgpu_feedback_packet_source_t source = {
      /*.dispatch_ptr=*/fault_address,
      /*.workgroup_id_x=*/workgroup_id,
      /*.workitem_id_x=*/workgroup_id,
  };
  const loom_amdgpu_sanitizer_access_report_t report = {
      /*.access_kind=*/LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ,
      /*.flags=*/LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE,
      /*.fault_address=*/fault_address,
      /*.access_size=*/fault_address,
      /*.site_id=*/fault_address,
      /*.shadow_address=*/check.shadow_address,
      /*.shadow_value=*/check.shadow_value,
  };

  loom_amdgpu_sanitizer_access_report_island_t island = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_report_island(
      &builder_, descriptor_set_, body_block_, feedback_config_symbol,
      LOOM_AMDGPU_SANITIZER_ACCESS_KIND_READ,
      LOOM_AMDGPU_SANITIZER_REPORT_FLAG_NONE, LOOM_LOCATION_UNKNOWN, &island));

  loom_builder_set_block(&builder_, body_block_);
  loom_amdgpu_sanitizer_access_report_failure_branch_t branch = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_report_failure_mask_branch(
      &builder_, descriptor_set_, &island, check.failure_mask, &source, &report,
      LOOM_LOCATION_UNKNOWN, &branch));
  loom_op_t* continuation_return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &continuation_return_op));

  VerifyModuleOk();
  VerifyLowModuleOk();

  const loom_op_t* hot_terminator = loom_block_const_last_op(body_block_);
  ASSERT_TRUE(loom_low_cond_br_isa(hot_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(hot_terminator), branch.failure_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(hot_terminator),
            branch.continuation_block);
  ExpectRegisterType(loom_low_cond_br_condition(hot_terminator),
                     LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);

  std::vector<loom_op_t*> saveexec_ops = OpsForDescriptorRefInBlock(
      branch.failure_block, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64);
  ASSERT_EQ(saveexec_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_operands(saveexec_ops[0]).count, 1u);
  EXPECT_EQ(loom_low_op_operands(saveexec_ops[0]).values[0],
            check.failure_mask);
  const loom_op_t* failure_terminator =
      loom_block_const_last_op(branch.failure_block);
  ASSERT_TRUE(loom_low_br_isa(failure_terminator));
  EXPECT_EQ(loom_low_br_dest(failure_terminator), island.entry_block);
  ASSERT_EQ(loom_low_br_args(failure_terminator).count, 8u);
  EXPECT_EQ(loom_low_br_args(failure_terminator).values[6],
            check.shadow_address);
  EXPECT_EQ(loom_low_br_args(failure_terminator).values[7], check.shadow_value);

  EXPECT_TRUE(
      OpsForDescriptorRefInBlock(
          body_block_, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR)
          .empty());
  EXPECT_TRUE(OpsForDescriptorRefInBlock(
                  branch.failure_block,
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR)
                  .empty());
  EXPECT_TRUE(
      OpsForDescriptorRefInBlock(body_block_, LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP)
          .empty());
  EXPECT_TRUE(OpsForDescriptorRefInBlock(branch.failure_block,
                                         LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP)
                  .empty());
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8).size(),
            2u);
  std::vector<loom_op_t*> hot_waitcnt_ops = OpsForDescriptorRefInBlock(
      body_block_, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(hot_waitcnt_ops.size(), 2u);
  for (loom_op_t* waitcnt_op : hot_waitcnt_ops) {
    ExpectAttrI64(loom_low_op_attrs(waitcnt_op), IREE_SV("vmcnt"), 0);
    ExpectAttrI64(loom_low_op_attrs(waitcnt_op), IREE_SV("lgkmcnt"), 63);
  }
  EXPECT_TRUE(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP).empty());
}

TEST_F(AmdgpuSanitizerAccessTest, EmitsPacketizedStaticAccessCheck) {
  loom_symbol_ref_t asan_config_symbol = AddSymbol(IREE_SV("iree_asan_config"));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(BuildFaultAddress(asan_config_symbol, &fault_address));

  loom_amdgpu_sanitizer_access_check_t check = {};
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_check(
      &builder_, descriptor_set_, asan_config_symbol, fault_address,
      /*access_size=*/16, /*wavefront_size=*/32, LOOM_LOCATION_UNKNOWN,
      &check));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));

  VerifyModuleOk();
  VerifyLowModuleOk();

  ExpectRegisterType(check.failure_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  ExpectRegisterType(check.shadow_address, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  ExpectRegisterType(check.shadow_value, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8).size(),
            4u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT).size(),
            4u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT).size(),
      8u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT).size(),
      4u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32).size(),
            4u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U32).size(), 8u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64).size(),
            4u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64).size(),
            7u);
  EXPECT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32).size(), 9u);
}

TEST_F(AmdgpuSanitizerAccessTest, EmitsFailureMaskWithoutReportTuple) {
  loom_symbol_ref_t asan_config_symbol = AddSymbol(IREE_SV("iree_asan_config"));
  loom_value_id_t fault_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(BuildFaultAddress(asan_config_symbol, &fault_address));

  loom_value_id_t failure_mask = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_access_failure_mask(
      &builder_, descriptor_set_, asan_config_symbol, fault_address,
      /*access_size=*/16, /*wavefront_size=*/32, LOOM_LOCATION_UNKNOWN,
      &failure_mask));
  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));

  VerifyModuleOk();
  VerifyLowModuleOk();

  ExpectRegisterType(failure_mask, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_FLAT_LOAD_U8).size(),
            4u);
  EXPECT_EQ(OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT).size(),
            4u);
  EXPECT_TRUE(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32).empty());
}

}  // namespace
