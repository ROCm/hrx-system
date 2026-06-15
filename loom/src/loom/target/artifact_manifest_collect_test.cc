// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest_collect.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/artifact_manifest.h"
#include "loom/target/entry_selection.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

bool AcceptEntry(void* user_data, const loom_target_entry_t* entry) {
  (void)user_data;
  (void)entry;
  return true;
}

class ArtifactManifestCollectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("artifact_manifest_collect.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  iree_status_t SelectEntriesAndBuildDependencies(
      const loom_module_t* module, loom_target_entry_list_t* out_entries,
      loom_symbol_dependency_table_t* out_dependencies) {
    loom_target_entry_options_t options = {};
    loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {};
    loom_target_entry_diagnostic_emitter_initialize(
        module, &options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
    const loom_target_entry_predicate_t predicate = {
        /*.fn=*/AcceptEntry,
    };
    bool selected = false;
    IREE_RETURN_IF_ERROR(loom_target_entry_select_all_entries(
        module, &options, predicate, &diagnostic_emitter, IREE_SV("test"),
        &analysis_arena_, &selected, out_entries));
    if (!selected) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "expected exported entries");
    }
    return loom_symbol_dependency_table_build(module, &analysis_arena_,
                                              out_dependencies);
  }

  iree_string_view_t CollectAndFormat(
      const loom_module_t* module,
      const loom_target_artifact_manifest_collect_options_t* collect_options,
      loom_target_artifact_manifest_mode_t format_mode,
      iree_string_builder_t* builder) {
    loom_target_entry_list_t entries;
    loom_symbol_dependency_table_t dependencies;
    IREE_CHECK_OK(
        SelectEntriesAndBuildDependencies(module, &entries, &dependencies));
    loom_target_artifact_manifest_t manifest;
    IREE_CHECK_OK(loom_target_artifact_manifest_collect_from_entries(
        module, entries, &dependencies, collect_options, &analysis_arena_,
        &manifest));

    iree_string_builder_initialize(iree_allocator_system(), builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(builder, &stream);
    const loom_target_artifact_manifest_format_options_t format_options = {
        /*.mode=*/format_mode,
    };
    IREE_CHECK_OK(loom_target_artifact_manifest_format_json(
        &manifest, &format_options, &stream));
    return iree_string_builder_view(builder);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(ArtifactManifestCollectTest, CollectsSummaryFunctionsAndGlobals) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target {
  artifact_format = elf,
  subgroup_size = 32
}

global.constant @weights : i32 = 1
global.variable @scratch : i32

func.def target(@test_target) abi(object_function) export("run") @entry(%input: i32) {
  func.call @forward() : ()
  func.call @helper() : ()
  func.return
}

func.def target(@test_target) abi(object_function) export("aux") @aux() {
  func.return
}

func.def target(@test_target) abi(object_function) @helper() {
  %value = global.load @weights : i32
  global.store %value, @scratch : i32
  func.return
}

func.def target(@test_target) abi(object_function) @unused() {
  func.return
}

func.def target(@test_target) abi(object_function) export("forward") @forward() {
  func.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
  options.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
  options.artifact_byte_length = 0;
  options.artifact_name = IREE_SV("module");
  options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\","
                      "\"name\":\"module\","
                      "\"byte_length\":0},"
                      "\"targets\":[{\"name\":\"test_target\"}],"
                      "\"functions\":[{\"name\":\"run\","
                      "\"source\":\"entry\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":1},"
                      "\"execution\":{\"subgroup_size\":32}},"
                      "{\"name\":\"aux\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":0},"
                      "\"execution\":{\"subgroup_size\":32}},"
                      "{\"name\":\"forward\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":0},"
                      "\"execution\":{\"subgroup_size\":32}}],"
                      "\"globals\":[{\"name\":\"weights\","
                      "\"targets\":[\"test_target\"]},"
                      "{\"name\":\"scratch\","
                      "\"targets\":[\"test_target\"]}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, CollectsGlobalsInModuleOrder) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target {artifact_format = elf}

func.def target(@test_target) abi(object_function) export("run") @entry() {
  %late_value = global.load @late : i32
  %early_value = global.load @early : i32
  func.return
}

global.constant @early : i32 = 1
global.constant @late : i32 = 2
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\"},"
                      "\"targets\":[{\"name\":\"test_target\"}],"
                      "\"functions\":[{\"name\":\"run\","
                      "\"source\":\"entry\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":0}}],"
                      "\"globals\":[{\"name\":\"early\","
                      "\"targets\":[\"test_target\"]},"
                      "{\"name\":\"late\","
                      "\"targets\":[\"test_target\"]}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, CollectsDetailsParameters) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target {artifact_format = spirv_binary}

func.def target(@test_target) abi(object_function) export("entry") @entry(%lhs: i32, %rhs: f32) {
  func.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;
  options.artifact_name = IREE_SV("module");
  options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"details\","
                      "\"artifact\":{\"format\":\"spirv-binary\","
                      "\"name\":\"module\"},"
                      "\"targets\":[{\"name\":\"test_target\","
                      "\"default_pointer_bitwidth\":64,"
                      "\"index_bitwidth\":64,"
                      "\"offset_bitwidth\":64}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":2,"
                      "\"parameters\":[{\"name\":\"lhs\","
                      "\"kind\":\"value\","
                      "\"index\":0},"
                      "{\"name\":\"rhs\","
                      "\"kind\":\"value\","
                      "\"index\":1}]}}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, CollectsTargetDetailsInDetailsMode) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu {
  default_pointer_bitwidth = 64,
  index_bitwidth = 32,
  offset_bitwidth = 64,
  max_workgroup_size_x = 256,
  max_workgroup_size_y = 8,
  max_workgroup_size_z = 4,
  max_flat_workgroup_size = 1024,
  subgroup_size = 32,
  max_grid_size_x = 4096,
  max_grid_size_y = 2048,
  max_grid_size_z = 1024,
  max_flat_grid_size = 8589934592,
  max_workgroup_count_x = 128,
  max_workgroup_count_y = 64,
  max_workgroup_count_z = 32
}

func.def target(@gpu) abi(object_function) export("entry") @entry() {
  func.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"details\","
                      "\"artifact\":{\"format\":\"vm-bytecode\"},"
                      "\"targets\":[{\"name\":\"gpu\","
                      "\"default_pointer_bitwidth\":64,"
                      "\"index_bitwidth\":32,"
                      "\"offset_bitwidth\":64,"
                      "\"max_workgroup_size\":{\"x\":256,\"y\":8,\"z\":4},"
                      "\"max_flat_workgroup_size\":1024,"
                      "\"subgroup_size\":32,"
                      "\"max_grid_size\":{\"x\":4096,"
                      "\"y\":2048,"
                      "\"z\":1024},"
                      "\"max_flat_grid_size\":8589934592,"
                      "\"max_workgroup_count\":{\"x\":128,"
                      "\"y\":64,"
                      "\"z\":32}}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"gpu\"],"
                      "\"interface\":{\"parameter_count\":0},"
                      "\"execution\":{\"subgroup_size\":32}}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest,
       CollectsPreparedLowHalKernelAbiFactsIncludingZero) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu {
  artifact_format = elf,
  abi = hal_kernel,
  subgroup_size = 64
}

low.kernel.def target(@gpu) abi_layout({
  constant_count = 0,
  direct_arg_count = 0,
  direct_arg_names = {},
  direct_arg_sizes = [],
  resource_count = 0,
  uses_kernarg_segment_ptr = false
}) workgroup_size(64, 1, 1) @entry() {
  low.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\"},"
                      "\"targets\":[{\"name\":\"gpu\"}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"gpu\"],"
                      "\"interface\":{\"parameter_count\":0,"
                      "\"binding_count\":0,"
                      "\"constant_byte_length\":0},"
                      "\"execution\":{\"workgroup_size\":[64,1,1],"
                      "\"subgroup_size\":64}}]}")))
      << std::string(output.data, output.size);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest,
       CollectsPreparedLowHalKernelParameterCountFromAbiLayout) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu {
  artifact_format = elf,
  abi = hal_kernel
}

low.kernel.def target(@gpu) abi_layout({
  constant_count = 1,
  direct_arg_count = 1,
  direct_arg_names = {arg0 = "extent"},
  direct_arg_sizes = [4],
  resource_count = 1,
  uses_kernarg_segment_ptr = true
}) workgroup_size(64, 1, 1) @entry() {
  low.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\"},"
                      "\"targets\":[{\"name\":\"gpu\"}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"gpu\"],"
                      "\"interface\":{\"parameter_count\":2,"
                      "\"binding_count\":1,"
                      "\"constant_byte_length\":4},"
                      "\"execution\":{\"workgroup_size\":[64,1,1]}}]}")))
      << std::string(output.data, output.size);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest,
       CollectsLowHalKernelBindingCountBeforeMaterialization) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @gpu {
  artifact_format = elf,
  abi = hal_kernel
}

low.kernel.def target(@gpu) workgroup_size(32, 2, 1) @entry() {
  %binding0 = low.resource<hal_binding> {index = 0, source_type = i64} : i64
  %binding1 = low.resource<hal_binding> {index = 1, source_type = i64} : i64
  low.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\"},"
                      "\"targets\":[{\"name\":\"gpu\"}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"gpu\"],"
                      "\"interface\":{\"parameter_count\":2,"
                      "\"binding_count\":2,"
                      "\"constant_byte_length\":0},"
                      "\"execution\":{\"workgroup_size\":[32,2,1]}}]}")))
      << std::string(output.data, output.size);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, NoneModeLeavesManifestEmpty) {
  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  loom_target_artifact_manifest_t manifest;
  IREE_ASSERT_OK(loom_target_artifact_manifest_collect_from_entries(
      /*module=*/nullptr, loom_target_entry_list_t{},
      /*dependency_table=*/nullptr, &options, /*arena=*/nullptr, &manifest));
  EXPECT_TRUE(iree_string_view_is_empty(manifest.artifact.format));
  EXPECT_EQ(manifest.function_count, 0u);
  EXPECT_EQ(manifest.global_count, 0u);
}

}  // namespace
}  // namespace loom
