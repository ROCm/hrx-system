// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnosticEmission;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

class LowVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
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
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("low_verify_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  static loom_target_bundle_storage_t CopyTargetBundle(
      const loom_target_bundle_t* bundle) {
    loom_target_bundle_storage_t storage = {};
    storage.snapshot = *bundle->snapshot;
    storage.export_plan = *bundle->export_plan;
    storage.config = *bundle->config;
    storage.bundle = *bundle;
    loom_target_bundle_storage_rebind(&storage);
    return storage;
  }

  void VerifyModule(loom_module_t* module, loom_target_selection_t selection,
                    DiagnosticEmissionCapture* capture,
                    loom_low_verify_result_t* out_result) {
    loom_low_verify_options_t options = {};
    options.descriptor_registry = &registry_.registry;
    options.target_selection = selection;
    options.emitter = capture->emitter();
    options.provider_list = loom_low_verify_provider_list_empty();
    options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    IREE_EXPECT_OK(
        loom_low_verify_module(module, &options, &scratch, out_result));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(LowVerifyTest, AcceptsWorkgroupStorageWithoutTargetLimit) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
low.func.def target(@target) @uses_workgroup_storage() {
  %storage = low.storage.reserve {byte_alignment = 16, byte_length = 80} : low.storage<workgroup>
  low.return
}
)");
  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), loom_target_selection_empty(), &capture, &result);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.emissions.empty());
}

TEST_F(LowVerifyTest, RejectsWorkgroupStorageAboveSelectedTargetLimit) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
low.func.def target(@target) @uses_workgroup_storage() {
  %storage = low.storage.reserve {byte_alignment = 16, byte_length = 80} : low.storage<workgroup>
  low.return
}
)");
  ASSERT_GT(loom_test_target_bundles.count, 1u);
  loom_target_bundle_storage_t selected_storage =
      CopyTargetBundle(loom_test_target_bundles.values[1]);
  selected_storage.snapshot.max_workgroup_storage_bytes = 64;
  loom_target_bundle_storage_rebind(&selected_storage);
  const loom_target_selection_t selection = {
      /*.bundle=*/&selected_storage.bundle,
      /*.data=*/nullptr,
  };

  DiagnosticEmissionCapture capture;
  loom_low_verify_result_t result = {};
  VerifyModule(module.get(), selection, &capture, &result);
  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_051);
  ASSERT_EQ(emission.params.size(), 4u);
  ASSERT_EQ(emission.string_params.size(), 2u);
  EXPECT_EQ(emission.string_params[0], "uses_workgroup_storage");
  EXPECT_EQ(emission.string_params[1], "test-low");
  ASSERT_EQ(emission.u64_params.size(), 2u);
  EXPECT_EQ(emission.u64_params[0], 80u);
  EXPECT_EQ(emission.u64_params[1], 64u);
}

}  // namespace
}  // namespace loom
