// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/artifact_provider.h"

#include <string.h>

#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/drivers/vulkan/spirv.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/testing/byte_sequence.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

namespace loom {
namespace {

using ::loom::testing::ByteSequenceClone;
using ::loom::testing::ModulePtr;

typedef struct fake_hal_device_t {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;
  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} fake_hal_device_t;

static const iree_hal_device_spec_t* fake_hal_device_spec(
    iree_hal_device_t* base_device) {
  fake_hal_device_t* device = (fake_hal_device_t*)base_device;
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.device_spec = fake_hal_device_spec;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

static void InitializeFakeHalDevice(const iree_hal_device_spec_t* device_spec,
                                    fake_hal_device_t* out_device) {
  out_device->device_spec = device_spec;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &out_device->resource);
}

static iree_hal_vulkan_features_t BaselineVulkanFeatures() {
  return IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL |
         IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64 |
         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT |
         IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL;
}

static iree_status_t CreateBaselineDeviceSpec(
    iree_hal_device_spec_t** out_device_spec) {
  *out_device_spec = NULL;
  iree_hal_device_dispatch_spec_t dispatch = {
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/256,
          /*.maximum_workgroup_size=*/{256, 128, 64},
          /*.maximum_workgroup_count=*/{65535, 65535, 65535},
      },
      /*.subgroup=*/
      {
          /*.default_size=*/32,
          /*.minimum_size=*/32,
          /*.maximum_size=*/32,
          /*.supported_size_mask=*/1ull << 32,
      },
      /*.execution=*/
      {
          /*.unit_count=*/1,
          /*.group_count=*/1,
      },
      /*.addressing=*/
      {
          /*.pointer_size_bits=*/64,
          /*.address_space_bits=*/64,
      },
      /*.flags=*/IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  iree_hal_vulkan_device_spec_t vulkan_spec = {
      /*.api_version=*/LOOM_SPIRV_VULKAN_API_VERSION_1_3,
      /*.driver_version=*/1,
      /*.physical_device_type=*/2,
      /*.enabled_features=*/BaselineVulkanFeatures(),
      /*.flags=*/IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };
  iree_host_size_t vulkan_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      /*property_count=*/0, &vulkan_payload_size));
  std::vector<uint8_t> vulkan_payload_storage(vulkan_payload_size);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_encode(
      &vulkan_spec, /*property_count=*/0, /*properties=*/nullptr,
      iree_make_byte_span(vulkan_payload_storage.data(),
                          vulkan_payload_storage.size())));
  iree_hal_device_spec_facet_t vulkan_facet = {
      /*.schema_id=*/
      iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
      /*.schema_version=*/IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
      /*.payload=*/
      iree_make_const_byte_span(vulkan_payload_storage.data(),
                                vulkan_payload_storage.size()),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/1,
      /*.targets=*/&executable_target,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_add_facet(&builder, &vulkan_facet);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

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
    loom_spirv_vulkan_hal_artifact_provider.deinitialize_device_target(
        &loom_spirv_vulkan_hal_artifact_provider, &target_,
        iree_allocator_system());
    iree_hal_device_spec_release(device_spec_);
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
        "low.kernel.def target<spirv.logical.core>(@hal_target) "
        "workgroup_size(1, 1, 1) "
        "@loom_kernel(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm {\n"
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
        "low.kernel.def target<spirv.logical.core>(@hal_target) "
        "workgroup_size(1, 1, 1) "
        "@double_i32(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm {\n"
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
        "low.kernel.def target<spirv.logical.core>(@hal_target) "
        "workgroup_size(1, 1, 1) "
        "@copy_i32(%byte_offset: "
        "reg<spirv.offset64>) "
        "asm {\n"
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

  iree_status_t SelectBaselineTarget() {
    IREE_RETURN_IF_ERROR(CreateBaselineDeviceSpec(&device_spec_));
    fake_hal_device_t device = {};
    InitializeFakeHalDevice(device_spec_, &device);
    loom_run_hal_runtime_t runtime = {};
    runtime.device = (iree_hal_device_t*)&device;
    return loom_spirv_vulkan_hal_artifact_provider.select_device_target(
        &loom_spirv_vulkan_hal_artifact_provider, &runtime,
        iree_allocator_system(), &target_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  iree_hal_device_spec_t* device_spec_ = nullptr;
  loom_run_hal_device_target_t target_ = {};
};

TEST_F(SpirvVulkanHalArtifactProviderTest, SelectsRawBdaDeviceTarget) {
  IREE_ASSERT_OK(SelectBaselineTarget());
  const loom_run_hal_device_target_t& target = target_;

  const loom_spirv_target_profile_t* target_profile =
      loom_spirv_target_profile_cast(target.target_profile);
  ASSERT_NE(target_profile, nullptr);
  const loom_target_bundle_t* target_bundle =
      loom_run_hal_device_target_bundle(&target);
  ASSERT_NE(target_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.target_key,
                                     IREE_SV("spirv-vulkan1.3-bda-hal")));
  EXPECT_EQ(target_bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(target_bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(target_bundle->snapshot->default_pointer_bitwidth, 64u);
  EXPECT_EQ(target_bundle->snapshot->offset_bitwidth, 64u);
  EXPECT_EQ(target_bundle->export_plan->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
}

TEST_F(SpirvVulkanHalArtifactProviderTest, EmitsRawBdaSpirvArtifact) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaRoundtripModule(&module));
  ASSERT_NE(module.get(), nullptr);

  IREE_ASSERT_OK(SelectBaselineTarget());
  const loom_run_hal_device_target_t& target = target_;

  loom_run_candidate_compile_options_t options = {};
  loom_run_candidate_compile_options_initialize(&options);
  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_artifact_provider.emit_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, module.get(), &target, &options,
      iree_allocator_system(), &emitted, &artifact));

  EXPECT_TRUE(emitted);
  EXPECT_EQ(artifact.hal_target, target.hal_target);
  EXPECT_TRUE(
      iree_string_view_equal(artifact.target_key, IREE_SV("vulkan1.3+bda")));
  ASSERT_NE(artifact.target_bundle, nullptr);
  EXPECT_EQ(artifact.target_bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(artifact.target_bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(artifact.target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(artifact.target_artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(artifact.target_artifact_data, artifact.executable_data);
  ASSERT_NE(artifact.executable_data, nullptr);
  ByteSequenceClone executable_data(iree_allocator_system());
  IREE_ASSERT_OK(executable_data.Clone(artifact.executable_data));
  const iree_const_byte_span_t executable_contents = executable_data.contents();
  ASSERT_GE(executable_contents.data_length, sizeof(uint32_t));

  uint32_t magic = 0;
  memcpy(&magic, executable_contents.data, sizeof(magic));
  EXPECT_EQ(magic, 0x07230203u);

  loom_spirv_vulkan_hal_artifact_provider.deinitialize_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, &artifact,
      iree_allocator_system());
}

TEST_F(SpirvVulkanHalArtifactProviderTest, EmitsAllCompatibleEntries) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaMultiEntryModule(&module));
  ASSERT_NE(module.get(), nullptr);

  IREE_ASSERT_OK(SelectBaselineTarget());
  const loom_run_hal_device_target_t& target = target_;

  loom_run_candidate_compile_options_t options = {};
  loom_run_candidate_compile_options_initialize(&options);
  loom_run_hal_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_spirv_vulkan_hal_artifact_provider.emit_artifact(
      &loom_spirv_vulkan_hal_artifact_provider, module.get(), &target, &options,
      iree_allocator_system(), &emitted, &artifact));

  ASSERT_TRUE(emitted);
  ASSERT_NE(artifact.executable_data, nullptr);
  ByteSequenceClone executable_data(iree_allocator_system());
  IREE_ASSERT_OK(executable_data.Clone(artifact.executable_data));
  const iree_const_byte_span_t executable_contents = executable_data.contents();
  ASSERT_EQ(executable_contents.data_length % sizeof(uint32_t), 0u);

  const uint32_t* spirv_words = (const uint32_t*)executable_contents.data;
  const iree_host_size_t spirv_word_count =
      executable_contents.data_length / sizeof(uint32_t);

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
