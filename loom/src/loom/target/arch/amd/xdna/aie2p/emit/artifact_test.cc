// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;
using ::loom::testing::ModulePtr;

constexpr char kResidentNeighborSource[] = R"(
aie2p.target<array> @array_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
aie2p.target<core> @core_target {device_profile = "amd.xdna.strix_halo.17f0_11"}

low.func.def retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @resident_neighbor() asm {
  %channel_capacity = constant.u32 1 : reg<aie2p.array.scalar : index>
  %records_per_activation = constant.u32 1 : reg<aie2p.array.scalar : index>
  %first_lane = constant.u32 0 : reg<aie2p.array.scalar : index>
  %second_lane = constant.u32 1 : reg<aie2p.array.scalar : index>
  %worker_count = constant.u32 2 : reg<aie2p.array.scalar : index>
  %column = constant.u32 0 : reg<aie2p.array.scalar : index>
  %producer_row = constant.u32 2 : reg<aie2p.array.scalar : index>
  %consumer_row = constant.u32 3 : reg<aie2p.array.scalar : index>
  %group = group %worker_count
  %producer = worker %group, %first_lane, @produce_i16
  %consumer = worker %group, %second_lane, @consume_i16
  %sender = sender %producer, 0 : reg<aie2p.array.sender : tile<1xi16>>
  %receiver = receiver %consumer, 0 : reg<aie2p.array.receiver : tile<1xi16>>
  %channel = channel %sender, %receiver, %channel_capacity, %records_per_activation : reg<aie2p.array.channel : tile<1xi16>>
  constrain.location %producer, %column, %producer_row
  constrain.location %consumer, %column, %consumer_row
  return
}

low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @produce_i16() asm {
  %output = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}

low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @consume_i16() asm {
  %input = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}
)";

constexpr char kCoreRootSource[] = R"(
aie2p.target<core> @core_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
low.func.def retain target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @core_root() asm {
  return
}
)";

constexpr char kGenericProfileSource[] = R"(
aie2p.target<array> @array_target
low.func.def retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @generic_array() asm {
  return
}
)";

constexpr char kEmptyArraySource[] = R"(
aie2p.target<array> @array_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
low.func.def retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @empty_array() asm {
  return
}
)";

constexpr char kMixedProfileSource[] = R"(
aie2p.target<array> @strix_target {device_profile = "amd.xdna.strix.17f0_10"}
aie2p.target<array> @halo_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
low.func.def retain target<amd.xdna.aie2p.array>(@strix_target) abi(array_program) @strix_array() asm {
  return
}
low.func.def retain target<amd.xdna.aie2p.array>(@halo_target) abi(array_program) @halo_array() asm {
  return
}
)";

constexpr char kAggregateBindingOverflowSource[] = R"(
aie2p.target<configuration> @configuration_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
aie2p.target<array> @array_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
aie2p.target<core> @core_target {device_profile = "amd.xdna.strix_halo.17f0_11"}
low.func.def retain target<amd.xdna.aie2p.configuration>(@configuration_target) abi(array_program) @configuration_entry() asm {
  %columns = constant 1 : reg<aie2p.config.scalar : i64>
  entry %columns, @initialize, @invoke
  return
}
low.func.def target<amd.xdna.aie2p.configuration>(@configuration_target) @initialize() asm {
  %column = constant 0 : reg<aie2p.config.scalar : i64>
  %row = constant 2 : reg<aie2p.config.scalar : i64>
  program.load %column, %row, @configuration_worker
  return
}
low.func.def target<amd.xdna.aie2p.configuration>(@configuration_target) @invoke() asm {
  return
}
low.func.def retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) abi_layout({binding_count = 32768}) @first() asm {
  %channel_capacity = constant.u32 1 : reg<aie2p.array.scalar : index>
  %records_per_activation = constant.u32 1 : reg<aie2p.array.scalar : index>
  %first_lane = constant.u32 0 : reg<aie2p.array.scalar : index>
  %second_lane = constant.u32 1 : reg<aie2p.array.scalar : index>
  %worker_count = constant.u32 2 : reg<aie2p.array.scalar : index>
  %column = constant.u32 0 : reg<aie2p.array.scalar : index>
  %producer_row = constant.u32 2 : reg<aie2p.array.scalar : index>
  %consumer_row = constant.u32 3 : reg<aie2p.array.scalar : index>
  %group = group %worker_count
  %producer = worker %group, %first_lane, @produce_i16
  %consumer = worker %group, %second_lane, @consume_i16
  %sender = sender %producer, 0 : reg<aie2p.array.sender : tile<1xi16>>
  %receiver = receiver %consumer, 0 : reg<aie2p.array.receiver : tile<1xi16>>
  %channel = channel %sender, %receiver, %channel_capacity, %records_per_activation : reg<aie2p.array.channel : tile<1xi16>>
  constrain.location %producer, %column, %producer_row
  constrain.location %consumer, %column, %consumer_row
  return
}
low.func.def retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) abi_layout({binding_count = 32768}) @second() asm {
  %channel_capacity = constant.u32 1 : reg<aie2p.array.scalar : index>
  %records_per_activation = constant.u32 1 : reg<aie2p.array.scalar : index>
  %first_lane = constant.u32 0 : reg<aie2p.array.scalar : index>
  %second_lane = constant.u32 1 : reg<aie2p.array.scalar : index>
  %worker_count = constant.u32 2 : reg<aie2p.array.scalar : index>
  %column = constant.u32 0 : reg<aie2p.array.scalar : index>
  %producer_row = constant.u32 2 : reg<aie2p.array.scalar : index>
  %consumer_row = constant.u32 3 : reg<aie2p.array.scalar : index>
  %group = group %worker_count
  %producer = worker %group, %first_lane, @produce_i16
  %consumer = worker %group, %second_lane, @consume_i16
  %sender = sender %producer, 0 : reg<aie2p.array.sender : tile<1xi16>>
  %receiver = receiver %consumer, 0 : reg<aie2p.array.receiver : tile<1xi16>>
  %channel = channel %sender, %receiver, %channel_capacity, %records_per_activation : reg<aie2p.array.channel : tile<1xi16>>
  constrain.location %producer, %column, %producer_row
  constrain.location %consumer, %column, %consumer_row
  return
}
low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @produce_i16() asm {
  %output = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}
low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @consume_i16() asm {
  %input = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}
low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @configuration_worker() asm {
  return
}
)";

iree_status_t InitializeXdnaContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_target_provider.register_context(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class XdnaArtifactTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &module_block_pool_);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &scratch_block_pool_);
    iree_arena_initialize(&scratch_block_pool_, &scratch_arena_);
    IREE_ASSERT_OK(InitializeXdnaContext(&context_));
    loom_aie2p_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    iree_hal_amd_xdna_image_destroy(image_);
    loom_target_emit_artifact_release(&artifact_);
    iree_byte_sequence_release(contents_);
    loom_target_compile_report_deinitialize(&compile_report_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&scratch_block_pool_);
    iree_arena_block_pool_deinitialize(&module_block_pool_);
  }

  iree_status_t ParseModule(iree_string_view_t source, ModulePtr* out_module) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(source, IREE_SV("xdna_artifact_test.loom"), &context_,
                        &module_block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t module_block_pool_;
  iree_arena_block_pool_t scratch_block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  loom_target_compile_report_t compile_report_ = {};
  loom_target_emit_artifact_t artifact_ = {};
  iree_byte_sequence_t* contents_ = nullptr;
  iree_hal_amd_xdna_image_t* image_ = nullptr;
};

TEST_F(XdnaArtifactTest, EmitsPrivateRetainedControlFreeResidentProduct) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kResidentNeighborSource), &module));

  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/&compile_report_,
      /*.diagnostic_emitter=*/{},
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_system(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  ASSERT_TRUE(emitted);
  ASSERT_NE(contents_, nullptr);
  EXPECT_EQ(compile_report_.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_TRUE(iree_string_view_equal(compile_report_.target_family_name,
                                     IREE_SV("amd.xdna.aie2p")));
  EXPECT_TRUE(iree_string_view_equal(compile_report_.target_key,
                                     IREE_SV("amd.xdna.strix_halo.17f0_11")));

  iree_hal_amd_xdna_aie2p_target_t image_target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &image_target));
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
      contents_, &image_target, iree_allocator_system(), &image_));
  iree_byte_sequence_release(contents_);
  contents_ = nullptr;

  uint32_t entry_ordinal = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image_, IREE_SV("resident_neighbor"), &entry_ordinal));
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image_);
  const iree_xdna_elf_entry_record_t entry =
      iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
  EXPECT_EQ(entry.binding_count, 0u);
  ASSERT_EQ(entry.invocation_count, 2u);

  const iree_xdna_elf_invocation_record_t establishing =
      iree_hal_amd_xdna_image_tables_invocation(tables, entry.first_invocation);
  const iree_xdna_elf_invocation_record_t continuing =
      iree_hal_amd_xdna_image_tables_invocation(tables,
                                                entry.first_invocation + 1);
  EXPECT_GT(establishing.byte_length, 16u);
  EXPECT_EQ(establishing.next_invocation, 1u);
  EXPECT_EQ(continuing.byte_length, 16u);
  EXPECT_EQ(continuing.next_invocation, 1u);
}

TEST_F(XdnaArtifactTest, RejectsNonArtifactEntryRootWithDiagnostic) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kCoreRootSource), &module));
  DiagnosticEmissionCapture capture;
  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/capture.emitter(),
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_null(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_XDNA_053);
  ASSERT_EQ(capture.emissions[0].string_params.size(), 2u);
  EXPECT_EQ(capture.emissions[0].string_params[0], "core_root");
  EXPECT_EQ(capture.emissions[0].string_params[1], "amd.xdna.aie2p.core");
}

TEST_F(XdnaArtifactTest, RejectsGenericProfileWithDiagnostic) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kGenericProfileSource), &module));
  DiagnosticEmissionCapture capture;
  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/capture.emitter(),
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_null(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_XDNA_054);
  ASSERT_EQ(capture.emissions[0].string_params.size(), 1u);
  EXPECT_EQ(capture.emissions[0].string_params[0], "generic_array");
}

TEST_F(XdnaArtifactTest, RejectsMixedProfilesWithDiagnostic) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kMixedProfileSource), &module));
  DiagnosticEmissionCapture capture;
  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/capture.emitter(),
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_null(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_XDNA_055);
  ASSERT_EQ(capture.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(capture.emissions[0].string_params[0], "halo_array");
  EXPECT_EQ(capture.emissions[0].string_params[1],
            "amd.xdna.strix_halo.17f0_11");
  EXPECT_EQ(capture.emissions[0].string_params[2], "amd.xdna.strix.17f0_10");
}

TEST_F(XdnaArtifactTest, RejectsEmptyArrayWithDiagnostic) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kEmptyArraySource), &module));
  DiagnosticEmissionCapture capture;
  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/capture.emitter(),
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_null(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_XDNA_018);
  ASSERT_EQ(capture.emissions[0].u32_params.size(), 2u);
  EXPECT_EQ(capture.emissions[0].u32_params[0], 0u);
  EXPECT_EQ(capture.emissions[0].u32_params[1], 0u);
}

TEST_F(XdnaArtifactTest, RejectsAggregateBindingCountBeforeResidentCompile) {
  ModulePtr module;
  IREE_ASSERT_OK(
      ParseModule(IREE_SV(kAggregateBindingOverflowSource), &module));
  DiagnosticEmissionCapture capture;
  const loom_aie2p_xdna_artifact_request_t request = {
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.compile_report=*/&compile_report_,
      /*.diagnostic_emitter=*/capture.emitter(),
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_system(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_compile_artifact(&request, &emitted, &contents_));
  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.emissions.size(), 1u);
  EXPECT_EQ(capture.emissions[0].error, LOOM_ERR_XDNA_056);
  ASSERT_EQ(capture.emissions[0].string_params.size(), 1u);
  EXPECT_EQ(capture.emissions[0].string_params[0], "binding records");
  ASSERT_EQ(capture.emissions[0].u64_params.size(), 2u);
  EXPECT_EQ(capture.emissions[0].u64_params[0], 65536u);
  EXPECT_EQ(capture.emissions[0].u64_params[1], 65535u);
  EXPECT_EQ(compile_report_.entry_rows.count, 0u);
  EXPECT_EQ(compile_report_.pipeline_plans.count, 0u);
}

TEST_F(XdnaArtifactTest, RetainsArrayTargetBundleForHalExecution) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(IREE_SV(kResidentNeighborSource), &module));

  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_.registry,
      /*.module=*/module.get(),
      /*.function_versions=*/nullptr,
      /*.option_chain=*/nullptr,
      /*.identifier=*/IREE_SV("module.xdna"),
      /*.artifact_manifest=*/{},
      /*.flags=*/LOOM_TARGET_EMIT_REQUEST_FLAG_RETAIN_TARGET_BUNDLE,
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/{},
      /*.max_errors=*/0,
      /*.scratch_arena=*/&scratch_arena_,
      /*.allocator=*/iree_allocator_system(),
  };
  bool emitted = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_artifact_emitter.emit(&request, &emitted, &artifact_));
  ASSERT_TRUE(emitted);
  ASSERT_NE(artifact_.target_bundle, nullptr);
  ASSERT_NE(artifact_.target_bundle->export_plan, nullptr);
  EXPECT_EQ(artifact_.target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_ARRAY_PROGRAM);
  EXPECT_EQ(artifact_.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_NE(artifact_.contents, nullptr);
}

}  // namespace
}  // namespace loom
