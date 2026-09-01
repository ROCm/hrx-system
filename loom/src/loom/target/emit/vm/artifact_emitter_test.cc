// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/artifact_emitter.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/ops/registry.h"
#include "loom/target/function_version.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;
using ::loom::testing::ModulePtr;

class VmArtifactEmitterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_vm_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    static const loom_low_descriptor_set_provider_t kDescriptorProviders[] = {
        loom_vm_core_descriptor_set,
    };
    loom_target_low_descriptor_registry_initialize_from_tables(
        &low_registry_, kDescriptorProviders,
        IREE_ARRAYSIZE(kDescriptorProviders));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

// Compiler handle identity deliberately has no IR or bytecode representation,
// so this contract must be observed directly at the artifact emitter boundary.
TEST_F(VmArtifactEmitterTest, ProjectsStableVersionsToSortedExportOrdinals) {
  static const char kSource[] = R"(
vm.target<core> @target

low.func.def target<vm.core>(@target) abi(vm_function) export("z_export") @z_export() asm {
  return
}

low.func.def target<vm.core>(@target) abi(vm_function) export("a_export") @a_export() asm {
  return
}

low.func.def target<vm.core>(@target) abi(vm_function) @private_helper() asm {
  return
}
)";
  DiagnosticCapture parse_capture;
  loom_text_parse_options_t parse_options = {
      /*.diagnostic_sink=*/parse_capture.sink(),
      /*.max_errors=*/20,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry_.registry, &parse_options.low_asm_environment);
  loom_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(loom_text_parse(
      iree_make_cstring_view(kSource), IREE_SV("artifact_emitter_test.loom"),
      &context_, &block_pool_, &parse_options, &raw_module));
  ASSERT_TRUE(parse_capture.diagnostics.empty());
  ModulePtr module(raw_module);
  ASSERT_NE(module, nullptr);

  const loom_symbol_id_t z_symbol_id = loom_module_find_symbol(
      module.get(),
      loom_module_lookup_string(module.get(), IREE_SV("z_export")));
  const loom_symbol_id_t a_symbol_id = loom_module_find_symbol(
      module.get(),
      loom_module_lookup_string(module.get(), IREE_SV("a_export")));
  const loom_symbol_id_t private_symbol_id = loom_module_find_symbol(
      module.get(),
      loom_module_lookup_string(module.get(), IREE_SV("private_helper")));
  ASSERT_NE(z_symbol_id, LOOM_SYMBOL_ID_INVALID);
  ASSERT_NE(a_symbol_id, LOOM_SYMBOL_ID_INVALID);
  ASSERT_NE(private_symbol_id, LOOM_SYMBOL_ID_INVALID);

  loom_target_function_version_t z_version = {};
  z_version.base.type = &loom_target_function_version_type;
  z_version.base.function = loom_func_like_cast(
      module.get(), module->symbols.entries[z_symbol_id].defining_op);
  loom_target_function_version_t a_version = {};
  a_version.base.type = &loom_target_function_version_type;
  a_version.base.function = loom_func_like_cast(
      module.get(), module->symbols.entries[a_symbol_id].defining_op);
  loom_target_function_version_t private_version = {};
  private_version.base.type = &loom_target_function_version_type;
  private_version.base.function = loom_func_like_cast(
      module.get(), module->symbols.entries[private_symbol_id].defining_op);
  ASSERT_TRUE(loom_func_like_isa(z_version.base.function));
  ASSERT_TRUE(loom_func_like_isa(a_version.base.function));
  ASSERT_TRUE(loom_func_like_isa(private_version.base.function));
  loom_function_version_t* version_values[] = {
      &private_version.base,
      &z_version.base,
      &a_version.base,
  };
  const loom_function_version_list_t function_versions = {
      /*.values=*/version_values,
      /*.count=*/IREE_ARRAYSIZE(version_values),
  };

  const loom_target_emitter_list_t emitters =
      loom_vm_artifact_emitter_provider.emitter_list;
  ASSERT_EQ(emitters.count, 1u);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  DiagnosticEmissionCapture emission_capture;
  loom_target_emit_request_t request = {};
  request.low_descriptor_registry = &low_registry_.registry;
  request.module = module.get();
  request.function_versions = &function_versions;
  request.diagnostic_emitter = emission_capture.emitter();
  request.scratch_arena = &scratch_arena;
  request.allocator = iree_allocator_system();
  loom_target_emit_artifact_t artifact = {};
  IREE_ASSERT_OK(emitters.values[0]->emit(&request, &artifact));
  EXPECT_TRUE(emission_capture.emissions.empty());
  ASSERT_NE(artifact.contents, nullptr);
  ASSERT_EQ(artifact.export_projection_count, 2u);
  ASSERT_NE(artifact.export_projections, nullptr);
  EXPECT_EQ(artifact.export_projections[0].function_version_ordinal, 2u);
  EXPECT_EQ(artifact.export_projections[0].ordinal, 0u);
  EXPECT_EQ(artifact.export_projections[1].function_version_ordinal, 1u);
  EXPECT_EQ(artifact.export_projections[1].ordinal, 1u);

  loom_target_emit_artifact_release(&artifact);
  iree_arena_deinitialize(&scratch_arena);
}

}  // namespace
}  // namespace loom
