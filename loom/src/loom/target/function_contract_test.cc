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
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
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
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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

  void ResolveContract(const loom_module_t* module,
                       const loom_func_symbol_facts_t* facts,
                       loom_target_bundle_storage_t* out_storage) {
    bool valid = false;
    IREE_CHECK_OK(loom_target_function_contract_resolve(
        module, &fact_table_, facts, iree_diagnostic_emitter_t{}, &valid,
        out_storage));
    ASSERT_TRUE(valid);
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with only target-aware dialects needed by these fixtures.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(TargetFunctionContractTest, LowFuncResolvesTargetRecord) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @test_target {contract_feature_bits = 1, default_pointer_bitwidth = 32, index_bitwidth = 32, offset_bitwidth = 32}

low.func.def target(@test_target) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  loom_target_bundle_storage_t storage = {};
  ResolveContract(module.get(), facts, &storage);
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
  abi = hal_kernel,
  hal_buffer_resource_flags = 7
}

low.kernel.def target(@test_target) export("dispatch") linkage(dso_local) @kernel() {
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
  EXPECT_EQ(storage.export_plan.hal_kernel.buffer_resource_flags, 7u);
}

TEST_F(TargetFunctionContractTest,
       BundleCompatibilityAllowsFeatureAndLimitRefinement) {
  const loom_target_snapshot_t module_snapshot = {
      /*.name=*/IREE_SVL("spirv-vulkan1.3"),
      /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_SPIRV,
      /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY,
      /*.default_pointer_bitwidth=*/64,
      /*.index_bitwidth=*/32,
      /*.offset_bitwidth=*/64,
      /*.max_workgroup_size=*/{},
      /*.max_flat_workgroup_size=*/64,
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

  loom_target_bundle_storage_t storage = {
      /*.snapshot=*/module_snapshot,
      /*.export_plan=*/module_export,
      /*.config=*/module_config,
      /*.bundle=*/module_bundle,
  };
  loom_target_bundle_storage_rebind(&storage);
  loom_target_function_contract_apply_compatible_selection(&selected_bundle,
                                                           &storage);
  EXPECT_TRUE(
      iree_string_view_equal(storage.snapshot.name, selected_snapshot.name));
  EXPECT_EQ(storage.snapshot.max_flat_workgroup_size,
            selected_snapshot.max_flat_workgroup_size);
  EXPECT_TRUE(
      iree_string_view_equal(storage.config.name, selected_config.name));
  EXPECT_EQ(storage.config.contract_feature_bits,
            selected_config.contract_feature_bits);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(
      iree_string_view_equal(storage.export_plan.name, module_export.name));
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
