// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

iree_status_t InitializeAmdgpuContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_ops_register_dialect(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class AmdgpuHalArtifactProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeAmdgpuContext(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_text_parse_options_t ParseOptions() const {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    return options;
  }

  iree_status_t ParseModule(iree_string_view_t source, ModulePtr* out_module) {
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(
        source, IREE_SV("amdgpu_hal_artifact_provider_test.loom"), &context_,
        &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParsePreparedArithmeticModule(ModulePtr* out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
        "reg<amdgpu.vgpr>\n"
        "  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
        "reg<amdgpu.vgpr>\n"
        "  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.return\n"
        "}\n";
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_cstring_view(kSource),
                        IREE_SV("amdgpu_hal_artifact_provider_test.loom"),
                        &context_, &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParsePreparedCdnaArithmeticModule(ModulePtr* out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx942> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
        "reg<amdgpu.vgpr>\n"
        "  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
        "reg<amdgpu.vgpr>\n"
        "  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.return\n"
        "}\n";
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_cstring_view(kSource),
                        IREE_SV("amdgpu_hal_artifact_provider_test.loom"),
                        &context_, &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  const loom_op_t* TargetOpFromRef(const loom_module_t* module,
                                   loom_symbol_ref_t target_ref) {
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT(target_ref.module_id == 0);
    IREE_ASSERT(target_ref.symbol_id < module->symbols.count);
    const loom_symbol_t* symbol =
        &module->symbols.entries[target_ref.symbol_id];
    IREE_ASSERT(symbol->defining_op != nullptr);
    return symbol->defining_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(AmdgpuHalArtifactProviderTest,
       ResolveDeviceTargetRefBuildsFamilyTargetRecords) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(R"(
kernel.def @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}
)"),
                             &module));
  ASSERT_NE(module.get(), nullptr);

  struct Case {
    iree_string_view_t processor_name;
    loom_amdgpu_target_kind_t expected_kind;
  };
  static const Case cases[] = {
      {IREE_SV("gfx942"), LOOM_AMDGPU_TARGET_KIND_GFX942},
      {IREE_SV("gfx1150"), LOOM_AMDGPU_TARGET_KIND_GFX1100},
      {IREE_SV("gfx1201"), LOOM_AMDGPU_TARGET_KIND_GFX1200},
      {IREE_SV("gfx1250"), LOOM_AMDGPU_TARGET_KIND_GFX1250},
  };
  for (const Case& c : cases) {
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_target_info_lookup_processor(c.processor_name, &processor));
    ASSERT_NE(processor, nullptr);
    loom_run_hal_device_target_t target = {
        /*.data=*/processor,
        /*.target_storage=*/{},
        /*.target_bundle=*/{},
        /*.target_key=*/processor->processor,
    };

    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.resolve_device_target_ref(
        &loom_amdgpu_hal_artifact_provider, module.get(), &target,
        &target_ref));
    const loom_op_t* target_op = TargetOpFromRef(module.get(), target_ref);
    ASSERT_TRUE(loom_amdgpu_target_isa(target_op));
    EXPECT_EQ(loom_amdgpu_target_kind(target_op), c.expected_kind);
    EXPECT_TRUE(iree_string_view_equal(
        loom_amdgpu_target_record_processor_name(module.get(), target_op),
        processor->processor));

    loom_symbol_ref_t reused_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.resolve_device_target_ref(
        &loom_amdgpu_hal_artifact_provider, module.get(), &target,
        &reused_ref));
    EXPECT_EQ(reused_ref.module_id, target_ref.module_id);
    EXPECT_EQ(reused_ref.symbol_id, target_ref.symbol_id);
  }
}

TEST_F(AmdgpuHalArtifactProviderTest, SelectTargetKeyBuildsOfflineTarget) {
  loom_run_hal_device_target_t target = {};
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.select_target_key(
      &loom_amdgpu_hal_artifact_provider, IREE_SV("gfx1100"),
      iree_allocator_system(), &target));

  ASSERT_NE(target.data, nullptr);
  EXPECT_NE(target.target_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.target_key, IREE_SV("gfx1100")));
  const loom_amdgpu_processor_info_t* processor =
      static_cast<const loom_amdgpu_processor_info_t*>(target.data);
  EXPECT_TRUE(iree_string_view_equal(processor->processor, IREE_SV("gfx1100")));
}

TEST_F(AmdgpuHalArtifactProviderTest, RecordsDetailedReportRows) {
  ModulePtr module;
  IREE_ASSERT_OK(ParsePreparedArithmeticModule(&module));
  ASSERT_NE(module.get(), nullptr);

  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1100"), &processor));
  ASSERT_NE(processor, nullptr);

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;

  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set_ordinal);
  ASSERT_NE(target_bundle, nullptr);

  loom_run_hal_device_target_t target = {
      /*.data=*/processor,
      /*.target_storage=*/{},
      /*.target_bundle=*/target_bundle,
      /*.target_key=*/processor->processor,
  };
  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){0},
      /*source_resolver=*/(loom_source_resolver_t){0}, /*max_errors=*/20,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING,
      /*artifact_manifest=*/nullptr, &report, iree_allocator_system(), &emitted,
      &artifact));
  EXPECT_TRUE(emitted);
  EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(artifact.target_artifact_data.data, artifact.executable_data.data);
  EXPECT_EQ(artifact.target_artifact_data.data_length,
            artifact.executable_data.data_length);
  EXPECT_NE(artifact.target_artifact_data.data, nullptr);
  if (artifact.target_artifact_data.data != nullptr) {
    EXPECT_GT(artifact.target_artifact_data.data_length, 0u);
  }
  EXPECT_TRUE(iree_string_view_equal(artifact.target_listing_format,
                                     IREE_SV("amdgpu-assembly")));
  EXPECT_NE(artifact.target_listing_data.data, nullptr);
  if (artifact.target_listing_data.data != nullptr) {
    const iree_string_view_t target_listing =
        iree_make_string_view((const char*)artifact.target_listing_data.data,
                              artifact.target_listing_data.data_length);
    EXPECT_NE(
        iree_string_view_find(target_listing, IREE_SV(".amdgcn_target"), 0),
        IREE_STRING_VIEW_NPOS);
    EXPECT_NE(iree_string_view_find(target_listing, IREE_SV("v_add_u32"), 0),
              IREE_STRING_VIEW_NPOS);
  }

  EXPECT_EQ(report.source_low_rows.count, 0u);
  EXPECT_GT(report.pressure_rows.count, 0u);
  EXPECT_NE(report.pressure_rows.head, nullptr);

  loom_amdgpu_hal_artifact_provider.deinitialize_artifact(
      &loom_amdgpu_hal_artifact_provider, &artifact, iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
}

TEST_F(AmdgpuHalArtifactProviderTest,
       EmitsModuleTargetWithoutProcessorOverride) {
  ModulePtr module;
  IREE_ASSERT_OK(ParsePreparedCdnaArithmeticModule(&module));
  ASSERT_NE(module.get(), nullptr);

  loom_run_hal_device_target_t target = {};
  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_hal_artifact_provider.emit_artifact(
      &loom_amdgpu_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){0},
      /*source_resolver=*/(loom_source_resolver_t){0}, /*max_errors=*/20,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, &artifact));
  EXPECT_TRUE(emitted);
  EXPECT_NE(
      iree_string_view_find(artifact.executable_format, IREE_SV("gfx942"), 0),
      IREE_STRING_VIEW_NPOS);
  EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(artifact.target_artifact_data.data, artifact.executable_data.data);
  EXPECT_EQ(artifact.target_artifact_data.data_length,
            artifact.executable_data.data_length);
  EXPECT_NE(artifact.target_artifact_data.data, nullptr);
  EXPECT_GT(artifact.target_artifact_data.data_length, 0u);
  EXPECT_EQ(artifact.target_bundle, nullptr);

  loom_amdgpu_hal_artifact_provider.deinitialize_artifact(
      &loom_amdgpu_hal_artifact_provider, &artifact, iree_allocator_system());
}

}  // namespace
}  // namespace loom
