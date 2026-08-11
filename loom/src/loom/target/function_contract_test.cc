// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/function_contract.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/types.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetFunctionContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_target_core_test_low_descriptor_registry_initialize(&low_registry_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_function_contract_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  const loom_func_symbol_facts_t* LookupFunc(const loom_module_t* module,
                                             iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_func_symbol_facts_t* facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  const loom_target_symbol_facts_t* LookupTarget(const loom_module_t* module,
                                                 iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_target_symbol_facts_t* facts =
        loom_target_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  void ResolveContract(const loom_module_t* module,
                       const loom_func_symbol_facts_t* facts,
                       loom_target_bundle_storage_t* out_storage) {
    bool valid = false;
    const loom_target_facts_t* target_facts = nullptr;
    IREE_CHECK_OK(loom_target_function_contract_resolve(
        module, &fact_table_, facts, iree_diagnostic_emitter_t{}, &valid,
        &target_facts, out_storage));
    ASSERT_TRUE(valid);
    ASSERT_NE(target_facts, nullptr);
  }

  const loom_target_facts_t* ResolveFacts(
      const loom_module_t* module, const loom_func_symbol_facts_t* facts) {
    bool valid = false;
    const loom_target_facts_t* target_facts = nullptr;
    IREE_CHECK_OK(loom_target_function_contract_resolve_facts(
        module, &fact_table_, facts, iree_diagnostic_emitter_t{},
        &analysis_arena_, &valid, &target_facts));
    IREE_ASSERT(valid);
    IREE_ASSERT(target_facts != nullptr);
    return target_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with only target-aware dialects needed by these fixtures.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;

  // Descriptor tables required by Low function representation contracts.
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(TargetFunctionContractTest, LowFuncResolvesTargetRecord) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1, default_pointer_bitwidth = 32, index_bitwidth = 32, offset_bitwidth = 32}

low.func.def target<test.low.core>(@test_target) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  const loom_target_bundle_storage_t& storage =
      ResolveFacts(module.get(), facts)->storage;
  EXPECT_TRUE(
      iree_string_view_equal(storage.bundle.name, IREE_SV("test_target")));
  EXPECT_EQ(storage.bundle.snapshot, &storage.snapshot);
  EXPECT_EQ(storage.bundle.export_plan, &storage.export_plan);
  EXPECT_EQ(storage.bundle.config, &storage.config);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_TRUE(
      iree_string_view_equal(storage.export_plan.name, IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_is_empty(storage.export_plan.export_symbol));
}

TEST_F(TargetFunctionContractTest, HalContractOverlaysTargetRecord) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {
  abi = hal_kernel
}

low.kernel.def target<test.low.core>(@test_target) export("dispatch") linkage(dso_local) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  loom_target_bundle_storage_t storage = {};
  ResolveContract(module.get(), facts, &storage);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(storage.export_plan.linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_TRUE(
      iree_string_view_equal(storage.export_plan.name, IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_equal(storage.export_plan.export_symbol,
                                     IREE_SV("dispatch")));
  EXPECT_EQ(storage.export_plan.hal_kernel.flat_workgroup_size_min, 0u);
  EXPECT_EQ(storage.export_plan.hal_kernel.flat_workgroup_size_max, 0u);
}

TEST_F(TargetFunctionContractTest,
       FactRefinementPreservesBaseAndRecordsFunctionContract) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

low.kernel.def target<test.low.core>(@test_target) export("dispatch") linkage(default) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* func_facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  bool valid = false;
  const loom_target_facts_t* base_facts =
      LookupTarget(module.get(), IREE_SV("test_target"))->projection;

  const loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_facts(
      module.get(), func_facts, IREE_SV("test_target"), base_facts,
      iree_diagnostic_emitter_t{}, &analysis_arena_, &valid,
      &function_target_facts));
  ASSERT_TRUE(valid);
  ASSERT_NE(function_target_facts, nullptr);
  EXPECT_NE(function_target_facts, base_facts);
  EXPECT_EQ(function_target_facts->fact_type, base_facts->fact_type);
  EXPECT_EQ(function_target_facts->selector, base_facts->selector);

  EXPECT_EQ(base_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_TRUE(
      iree_string_view_is_empty(base_facts->storage.export_plan.export_symbol));
  EXPECT_EQ(base_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_FALSE(loom_target_facts_field_is_authored(base_facts,
                                                   LOOM_TARGET_FACT_FIELD_ABI));
  EXPECT_FALSE(loom_target_facts_field_is_authored(
      base_facts, LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL));
  EXPECT_FALSE(loom_target_facts_field_is_authored(
      base_facts, LOOM_TARGET_FACT_FIELD_LINKAGE));

  EXPECT_EQ(function_target_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(iree_string_view_equal(
      function_target_facts->storage.export_plan.export_symbol,
      IREE_SV("dispatch")));
  EXPECT_EQ(function_target_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DEFAULT);
  EXPECT_TRUE(loom_target_facts_field_is_authored(function_target_facts,
                                                  LOOM_TARGET_FACT_FIELD_ABI));
  EXPECT_TRUE(loom_target_facts_field_is_authored(
      function_target_facts, LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL));
  EXPECT_TRUE(loom_target_facts_field_is_authored(
      function_target_facts, LOOM_TARGET_FACT_FIELD_LINKAGE));

  const loom_target_workgroup_size_t required_workgroup_size = {
      /*.x=*/64,
      /*.y=*/2,
      /*.z=*/1,
  };
  const loom_target_facts_t* launch_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_hal_workgroup_size(
      func_facts, IREE_SV("test_target"), &required_workgroup_size,
      function_target_facts, iree_diagnostic_emitter_t{}, &analysis_arena_,
      &valid, &launch_facts));
  ASSERT_TRUE(valid);
  ASSERT_NE(launch_facts, nullptr);
  EXPECT_EQ(function_target_facts->storage.export_plan.hal_kernel
                .required_workgroup_size.x,
            0u);
  EXPECT_EQ(function_target_facts->storage.export_plan.hal_kernel
                .required_workgroup_size.y,
            0u);
  EXPECT_EQ(function_target_facts->storage.export_plan.hal_kernel
                .required_workgroup_size.z,
            0u);
  EXPECT_EQ(
      launch_facts->storage.export_plan.hal_kernel.required_workgroup_size.x,
      64u);
  EXPECT_EQ(
      launch_facts->storage.export_plan.hal_kernel.required_workgroup_size.y,
      2u);
  EXPECT_EQ(
      launch_facts->storage.export_plan.hal_kernel.required_workgroup_size.z,
      1u);
}

TEST_F(TargetFunctionContractTest,
       InternalFactRefinementDoesNotInheritArtifactContract) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

func.def @helper() {
  func.return
}
)");

  const loom_func_symbol_facts_t* func_facts =
      LookupFunc(module.get(), IREE_SV("helper"));
  const loom_target_facts_t* base_facts =
      LookupTarget(module.get(), IREE_SV("test_target"))->projection;
  ASSERT_EQ(base_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  ASSERT_EQ(base_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);

  bool valid = false;
  const loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_internal_facts(
      module.get(), func_facts, IREE_SV("test_target"), base_facts,
      iree_diagnostic_emitter_t{}, &analysis_arena_, &valid,
      &function_target_facts));
  ASSERT_TRUE(valid);
  ASSERT_NE(function_target_facts, nullptr);
  EXPECT_EQ(function_target_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_UNKNOWN);
  EXPECT_EQ(function_target_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DEFAULT);
  EXPECT_TRUE(iree_string_view_equal(
      function_target_facts->storage.export_plan.name, IREE_SV("helper")));
  EXPECT_TRUE(iree_string_view_is_empty(
      function_target_facts->storage.export_plan.export_symbol));
  EXPECT_EQ(base_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(base_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);
}

TEST_F(TargetFunctionContractTest, InternalFactRefinementPreservesAuthoredAbi) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target

func.def abi(object_function) @helper() {
  func.return
}
)");

  const loom_func_symbol_facts_t* func_facts =
      LookupFunc(module.get(), IREE_SV("helper"));
  const loom_target_facts_t* base_facts =
      LookupTarget(module.get(), IREE_SV("test_target"))->projection;
  bool valid = false;
  const loom_target_facts_t* function_target_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_internal_facts(
      module.get(), func_facts, IREE_SV("test_target"), base_facts,
      iree_diagnostic_emitter_t{}, &analysis_arena_, &valid,
      &function_target_facts));
  ASSERT_TRUE(valid);
  ASSERT_NE(function_target_facts, nullptr);
  EXPECT_EQ(function_target_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_TRUE(loom_target_facts_field_is_authored(function_target_facts,
                                                  LOOM_TARGET_FACT_FIELD_ABI));
}

TEST_F(TargetFunctionContractTest,
       BundleCompatibilityIgnoresDeviceLimitsAndFunctionAbi) {
  const loom_target_snapshot_t module_snapshot = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
      /*.default_pointer_bitwidth=*/64,
      /*.index_bitwidth=*/32,
      /*.offset_bitwidth=*/64,
      /*.max_workgroup_size=*/{},
      /*.max_flat_workgroup_size=*/64,
      /*.max_workgroup_storage_bytes=*/0,
      /*.subgroup_size=*/64,
  };
  const loom_target_export_plan_t module_export = {
      /*.name=*/IREE_SVL("spirv-shader-entry-point"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
  };
  const loom_target_config_t module_config = {
      /*.name=*/IREE_SVL("spirv.logical.core"),
      /*.contract_set_key=*/IREE_SVL("spirv.logical.core"),
      /*.contract_feature_bits=*/1,
  };
  const loom_target_bundle_t module_bundle = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.snapshot=*/&module_snapshot,
      /*.export_plan=*/&module_export,
      /*.config=*/&module_config,
  };

  loom_target_snapshot_t selected_snapshot = module_snapshot;
  selected_snapshot.name = IREE_SV("spirv-vulkan1.3-bda");
  selected_snapshot.max_flat_workgroup_size = 1024;
  selected_snapshot.subgroup_size = 32;
  loom_target_export_plan_t selected_export = module_export;
  selected_export.name = IREE_SV("spirv-hal-kernel");
  loom_target_config_t selected_config = module_config;
  selected_config.name = IREE_SV("spirv.logical.core.vulkan1.3.bda");
  selected_config.contract_feature_bits = 7;
  const loom_target_bundle_t selected_bundle = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3-bda-hal"),
      /*.snapshot=*/&selected_snapshot,
      /*.export_plan=*/&selected_export,
      /*.config=*/&selected_config,
  };

  EXPECT_TRUE(loom_target_function_contract_bundles_compatible(
      &module_bundle, &selected_bundle));

  selected_export.abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION;
  EXPECT_TRUE(loom_target_function_contract_bundles_compatible(
      &module_bundle, &selected_bundle));
}

TEST_F(TargetFunctionContractTest, BundleCompatibilityRejectsContractShape) {
  loom_target_snapshot_t module_snapshot = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
  };
  loom_target_export_plan_t module_export = {
      /*.name=*/IREE_SVL("spirv-shader-entry-point"),
      /*.export_symbol=*/{},
      /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
  };
  loom_target_config_t module_config = {
      /*.name=*/IREE_SVL("spirv.logical.core"),
      /*.contract_set_key=*/IREE_SVL("spirv.logical.core"),
  };
  const loom_target_bundle_t module_bundle = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.snapshot=*/&module_snapshot,
      /*.export_plan=*/&module_export,
      /*.config=*/&module_config,
  };

  loom_target_snapshot_t selected_snapshot = module_snapshot;
  loom_target_export_plan_t selected_export = module_export;
  loom_target_config_t selected_config = module_config;
  const loom_target_bundle_t selected_bundle = {
      /*.name=*/IREE_SVL("selected"),
      /*.snapshot=*/&selected_snapshot,
      /*.export_plan=*/&selected_export,
      /*.config=*/&selected_config,
  };

  selected_snapshot.codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR;
  EXPECT_FALSE(loom_target_function_contract_bundles_compatible(
      &module_bundle, &selected_bundle));
  selected_snapshot = module_snapshot;

  selected_snapshot.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  EXPECT_FALSE(loom_target_function_contract_bundles_compatible(
      &module_bundle, &selected_bundle));
  selected_snapshot = module_snapshot;

  selected_config.contract_set_key = IREE_SV("other.contract");
  EXPECT_FALSE(loom_target_function_contract_bundles_compatible(
      &module_bundle, &selected_bundle));
}

}  // namespace
}  // namespace loom
