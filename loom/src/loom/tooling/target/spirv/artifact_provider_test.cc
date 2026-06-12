// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/artifact_provider.h"

#include <string.h>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/spirv.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

typedef struct fake_query_row_t {
  // HAL query category.
  iree_string_view_t category;
  // HAL query key within |category|.
  iree_string_view_t key;
  // Scalar query value returned to the caller.
  int64_t value;
} fake_query_row_t;

typedef struct fake_hal_device_t {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;
  // Query rows exposed by this fake device.
  const fake_query_row_t* rows;
  // Number of entries in |rows|.
  iree_host_size_t row_count;
} fake_hal_device_t;

typedef struct fake_executable_cache_t {
  // HAL resource header used by executable-cache vtable dispatch.
  iree_hal_resource_t resource;
  // Whether this cache accepts the raw Vulkan BDA executable format.
  bool raw_bda_supported;
} fake_executable_cache_t;

static iree_status_t fake_hal_device_query_i64(iree_hal_device_t* base_device,
                                               iree_string_view_t category,
                                               iree_string_view_t key,
                                               int64_t* out_value) {
  fake_hal_device_t* device = (fake_hal_device_t*)base_device;
  for (iree_host_size_t i = 0; i < device->row_count; ++i) {
    if (iree_string_view_equal(category, device->rows[i].category) &&
        iree_string_view_equal(key, device->rows[i].key)) {
      *out_value = device->rows[i].value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static bool fake_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)caching_mode;
  fake_executable_cache_t* executable_cache =
      (fake_executable_cache_t*)base_executable_cache;
  return executable_cache->raw_bda_supported &&
         iree_string_view_equal(executable_format, IREE_SV("vulkan-spirv-bda"));
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.query_i64 = fake_hal_device_query_i64;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

static iree_hal_executable_cache_vtable_t MakeFakeExecutableCacheVtable() {
  iree_hal_executable_cache_vtable_t vtable = {};
  vtable.can_prepare_format = fake_executable_cache_can_prepare_format;
  return vtable;
}

static const iree_hal_executable_cache_vtable_t kFakeExecutableCacheVtable =
    MakeFakeExecutableCacheVtable();

static fake_hal_device_t FakeDevice(const fake_query_row_t* rows,
                                    iree_host_size_t row_count) {
  fake_hal_device_t device = {};
  device.rows = rows;
  device.row_count = row_count;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device.resource);
  return device;
}

static fake_executable_cache_t FakeExecutableCache(bool raw_bda_supported) {
  fake_executable_cache_t executable_cache = {};
  executable_cache.raw_bda_supported = raw_bda_supported;
  iree_hal_resource_initialize(&kFakeExecutableCacheVtable,
                               &executable_cache.resource);
  return executable_cache;
}

static constexpr fake_query_row_t kBaselineQueryRows[] = {
    {IREE_SVL("vulkan.device"), IREE_SVL("api_version"),
     LOOM_SPIRV_VULKAN_API_VERSION_1_3},
    {IREE_SVL("vulkan.device"), IREE_SVL("subgroup_size"), 32},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_invocations"),
     256},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_x"), 256},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_y"), 128},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_z"), 64},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_x"),
     65535},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_y"),
     65535},
    {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_z"),
     65535},
    {IREE_SVL("vulkan.feature"), IREE_SVL("buffer_device_address"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("subgroup_size_control"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("cooperative_matrix_khr"), 0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("storage_buffer_8bit_access"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("storage_buffer_16bit_access"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_float16"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_float64"), 0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_type"), 0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_dot_product"), 0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_bfloat16_cooperative_matrix"),
     0},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int8"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int16"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int64"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("shader_integer_dot_product"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("vulkan_memory_model"), 1},
    {IREE_SVL("vulkan.feature"), IREE_SVL("vulkan_memory_model_device_scope"),
     0},
};

static iree_status_t InitializeSpirvContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_spirv_ops_register_dialect(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class SpirvVulkanHalArtifactProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeSpirvContext(&context_));
    loom_spirv_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_text_parse_options_t ParseOptions() const {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    return options;
  }

  iree_status_t ParseRawBdaRoundtripModule(ModulePtr* out_module) {
    static const char kSource[] =
        "spirv.target<vulkan1_3> @hal_target {abi = hal_kernel}\n"
        "low.kernel.def target(@hal_target) workgroup_size(1, 1, 1) "
        "@loom_kernel(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm<spirv.logical.core> {\n"
        "  %input = resource<hal_binding> {index = 0, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %output = resource<hal_binding> {index = 1, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %input_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%input, "
        "%byte_offset\n"
        "  %loaded = OpLoad.storage_buffer.i32 %input_view\n"
        "  %incremented = OpIAdd %loaded, %loaded\n"
        "  %output_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%output, %byte_offset\n"
        "  OpStore.storage_buffer.i32 %output_view, %incremented\n"
        "  return\n"
        "}\n";
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_cstring_view(kSource),
                        IREE_SV("spirv_hal_artifact_provider_test.loom"),
                        &context_, &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParseRawBdaMultiEntryModule(ModulePtr* out_module) {
    static const char kSource[] =
        "spirv.target<vulkan1_3> @hal_target {abi = hal_kernel}\n"
        "low.kernel.def target(@hal_target) workgroup_size(1, 1, 1) "
        "@double_i32(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm<spirv.logical.core> {\n"
        "  %input = resource<hal_binding> {index = 0, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %output = resource<hal_binding> {index = 1, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %input_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%input, %byte_offset\n"
        "  %loaded = OpLoad.storage_buffer.i32 %input_view\n"
        "  %result = OpIAdd %loaded, %loaded\n"
        "  %output_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%output, %byte_offset\n"
        "  OpStore.storage_buffer.i32 %output_view, %result\n"
        "  return\n"
        "}\n"
        "low.kernel.def target(@hal_target) workgroup_size(1, 1, 1) "
        "@copy_i32(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm<spirv.logical.core> {\n"
        "  %input = resource<hal_binding> {index = 0, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %output = resource<hal_binding> {index = 1, source_type = "
        "hal.buffer} : reg<spirv.ptr.storage_buffer>\n"
        "  %input_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%input, %byte_offset\n"
        "  %loaded = OpLoad.storage_buffer.i32 %input_view\n"
        "  %output_view = OpPtrAccessChain.storage_buffer.i32.byte_offset "
        "%output, %byte_offset\n"
        "  OpStore.storage_buffer.i32 %output_view, %loaded\n"
        "  return\n"
        "}\n";
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_cstring_view(kSource),
                        IREE_SV("spirv_hal_artifact_provider_test.loom"),
                        &context_, &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t SelectBaselineTarget(loom_run_hal_device_target_t* out_target) {
    fake_hal_device_t device =
        FakeDevice(kBaselineQueryRows, IREE_ARRAYSIZE(kBaselineQueryRows));
    fake_executable_cache_t executable_cache = FakeExecutableCache(true);
    loom_run_hal_runtime_t runtime = {};
    runtime.device = (iree_hal_device_t*)&device;
    runtime.executable_cache = (iree_hal_executable_cache_t*)&executable_cache;
    return loom_spirv_vulkan_hal_artifact_provider.select_device_target(
        &loom_spirv_vulkan_hal_artifact_provider, &runtime,
        iree_allocator_system(), out_target);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(SpirvVulkanHalArtifactProviderTest, SelectsRawBdaDeviceTarget) {
  loom_run_hal_device_target_t target = {};
  IREE_ASSERT_OK(SelectBaselineTarget(&target));

  ASSERT_NE(target.target_bundle, nullptr);
  EXPECT_EQ(target.target_bundle, &target.target_storage.bundle);
  EXPECT_TRUE(iree_string_view_equal(target.target_key,
                                     IREE_SV("spirv-vulkan1.3-bda-hal")));
  EXPECT_EQ(target.target_bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(target.target_bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(target.target_bundle->snapshot->default_pointer_bitwidth, 64u);
  EXPECT_EQ(target.target_bundle->snapshot->offset_bitwidth, 64u);
  EXPECT_EQ(target.target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
}

TEST_F(SpirvVulkanHalArtifactProviderTest, EmitsRawBdaSpirvArtifact) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaRoundtripModule(&module));
  ASSERT_NE(module.get(), nullptr);

  loom_run_hal_device_target_t target = {};
  IREE_ASSERT_OK(SelectBaselineTarget(&target));

  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_artifact_provider.emit_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){0},
      /*source_resolver=*/(loom_source_resolver_t){0}, /*max_errors=*/20,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, &artifact));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(iree_string_view_equal(artifact.executable_format,
                                     IREE_SV("vulkan-spirv-bda")));
  EXPECT_EQ(artifact.target_bundle, target.target_bundle);
  EXPECT_EQ(artifact.target_artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(artifact.target_artifact_data.data, artifact.executable_data.data);
  EXPECT_EQ(artifact.target_artifact_data.data_length,
            artifact.executable_data.data_length);
  ASSERT_NE(artifact.executable_data.data, nullptr);
  ASSERT_GE(artifact.executable_data.data_length, sizeof(uint32_t));

  uint32_t magic = 0;
  memcpy(&magic, artifact.executable_data.data, sizeof(magic));
  EXPECT_EQ(magic, 0x07230203u);

  loom_spirv_vulkan_hal_artifact_provider.deinitialize_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, &artifact,
      iree_allocator_system());
}

TEST_F(SpirvVulkanHalArtifactProviderTest, EmitsAllCompatibleEntries) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaMultiEntryModule(&module));
  ASSERT_NE(module.get(), nullptr);

  loom_run_hal_device_target_t target = {};
  IREE_ASSERT_OK(SelectBaselineTarget(&target));

  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_artifact_provider.emit_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, module.get(), &target,
      /*diagnostic_sink=*/(loom_diagnostic_sink_t){0},
      /*source_resolver=*/(loom_source_resolver_t){0}, /*max_errors=*/20,
      /*artifact_flags=*/LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_NONE,
      /*artifact_manifest=*/nullptr, /*report=*/nullptr,
      iree_allocator_system(), &emitted, &artifact));

  ASSERT_TRUE(emitted);
  ASSERT_NE(artifact.executable_data.data, nullptr);
  ASSERT_EQ(artifact.executable_data.data_length % sizeof(uint32_t), 0u);

  const uint32_t* spirv_words = (const uint32_t*)artifact.executable_data.data;
  const iree_host_size_t spirv_word_count =
      artifact.executable_data.data_length / sizeof(uint32_t);

  iree_hal_vulkan_spirv_module_analysis_t analysis = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_analyze_module(
      spirv_words, spirv_word_count, &analysis));
  EXPECT_EQ(analysis.compute_entry_point_count, 2u);
  EXPECT_EQ(analysis.push_constant_variable_count, 1u);

  iree_hal_vulkan_spirv_bda_dispatch_metadata_t metadata = {};
  IREE_ASSERT_OK(iree_hal_vulkan_spirv_parse_bda_dispatch_metadata(
      spirv_words, spirv_word_count, iree_allocator_system(), &metadata));
  EXPECT_TRUE(metadata.is_present);
  EXPECT_EQ(metadata.root_push_constant_offset, 0u);
  EXPECT_EQ(metadata.root_push_constant_length, 32u);
  EXPECT_EQ(metadata.constant_push_constant_offset, 32u);
  EXPECT_EQ(metadata.constant_byte_length, 8u);
  EXPECT_TRUE(metadata.binding_count_known);
  EXPECT_EQ(metadata.binding_count, 2u);
  iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize(
      &metadata, iree_allocator_system());

  loom_spirv_vulkan_hal_artifact_provider.deinitialize_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, &artifact,
      iree_allocator_system());
}

}  // namespace
}  // namespace loom
