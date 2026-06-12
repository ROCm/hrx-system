// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/builder.h"
#include "loom/pass/registry.h"
#include "loom/pass/testing/registry_verify.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

struct PipelineBuildData {
  const loom_target_environment_t* environment;
};

static iree_status_t BuildMaterializationPipeline(loom_builder_t* builder,
                                                  void* user_data) {
  const PipelineBuildData* data =
      static_cast<const PipelineBuildData*>(user_data);
  return loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION,
      loom_pass_environment_empty(), builder);
}

static iree_string_view_t ModuleString(loom_module_t* module,
                                       loom_string_id_t id) {
  return module->strings.entries[id];
}

static iree_string_view_t AttrStringValue(loom_module_t* module,
                                          loom_attribute_t attr) {
  return ModuleString(module, loom_attr_as_string_id(attr));
}

static const loom_named_attr_t* FindAttr(loom_module_t* module,
                                         loom_named_attr_slice_t attrs,
                                         iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (iree_string_view_equal(ModuleString(module, attrs.entries[i].name_id),
                               name)) {
      return &attrs.entries[i];
    }
  }
  return nullptr;
}

class AmdgpuProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_environment_t target_environment_;
};

TEST_F(AmdgpuProviderTest, ProvidesSortedPassRegistry) {
  const loom_pass_registry_t* registry =
      loom_target_environment_pass_registry(&target_environment_);
  IREE_ASSERT_OK(loom_pass_registry_verify(registry));

  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      registry, IREE_SV("amdgpu-materialize-hal-kernel-abi"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
}

TEST_F(AmdgpuProviderTest, ContributesHalKernelAbiMaterialization) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  PipelineBuildData build_data = {
      /*.environment=*/&target_environment_,
  };
  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("emit"), LOOM_PASS_ANCHOR_MODULE,
      BuildMaterializationPipeline, &build_data, &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 2u);

  loom_op_t* where_op = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_where_isa(where_op));
  EXPECT_TRUE(iree_string_view_equal(
      ModuleString(module.get(), loom_pass_where_predicate(where_op)),
      IREE_SV("target")));

  loom_named_attr_slice_t attrs = loom_pass_where_attrs(where_op);
  ASSERT_EQ(attrs.count, 3u);
  const loom_named_attr_t* target_op_attr =
      FindAttr(module.get(), attrs, IREE_SV("target_op"));
  ASSERT_NE(target_op_attr, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      AttrStringValue(module.get(), target_op_attr->value),
      IREE_SV("amdgpu.target")));
  const loom_named_attr_t* codegen_attr =
      FindAttr(module.get(), attrs, IREE_SV("codegen"));
  ASSERT_NE(codegen_attr, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(AttrStringValue(module.get(), codegen_attr->value),
                             IREE_SV("low_native")));
  const loom_named_attr_t* abi_attr =
      FindAttr(module.get(), attrs, IREE_SV("abi"));
  ASSERT_NE(abi_attr, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      AttrStringValue(module.get(), abi_attr->value), IREE_SV("hal_kernel")));

  loom_block_t* where_body =
      loom_region_entry_block(loom_pass_where_body(where_op));
  ASSERT_NE(where_body, nullptr);
  ASSERT_EQ(where_body->op_count, 2u);
  loom_op_t* run_op = where_body->first_op;
  ASSERT_TRUE(loom_pass_run_isa(run_op));
  EXPECT_TRUE(iree_string_view_equal(
      ModuleString(module.get(), loom_pass_run_key(run_op)),
      IREE_SV("amdgpu-materialize-hal-kernel-abi")));
  EXPECT_TRUE(loom_pass_yield_isa(where_body->last_op));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
}

}  // namespace
}  // namespace loom
