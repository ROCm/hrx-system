// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/artifact_provider.h"

#include <string.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/product/kernel.h"
#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/testing/byte_sequence.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/compile/product.h"
#include "loom/tooling/target/spirv/product_provider.h"

namespace loom {
namespace {

using ::loom::testing::ByteSequenceClone;
using ::loom::testing::ModulePtr;

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

class SpirvArtifactProviderTest : public ::testing::Test {
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
        "%input, %byte_offset\n"
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
                        IREE_SV("spirv_artifact_provider_test.loom"), &context_,
                        &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  // Arena blocks used by parsed module storage.
  iree_arena_block_pool_t block_pool_;

  // Dialect context used to parse target-low SPIR-V input.
  loom_context_t context_ = {};

  // Target-low descriptors exposed to the text parser.
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(SpirvArtifactProviderTest, EmitsAuthoredRawBdaArtifact) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaRoundtripModule(&module));
  ASSERT_NE(module.get(), nullptr);

  loom_compile_options_t options = {};
  loom_compile_options_initialize(&options);
  loom_artifact_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_artifact_candidate_emit_module_target(
      &loom_spirv_vulkan_artifact_provider, module.get(), &options,
      iree_allocator_system(), &candidate));

  ASSERT_TRUE(candidate.compiled);
  const loom_artifact_t& artifact = candidate.artifact;
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

  loom_artifact_candidate_deinitialize(&candidate);
}

TEST_F(SpirvArtifactProviderTest, BuildsImmutableKernelProduct) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseRawBdaRoundtripModule(&module));
  ASSERT_NE(module.get(), nullptr);

  loom_compile_options_t options = {};
  loom_compile_options_initialize(&options);
  const loom_product_build_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&low_registry_,
      /*.module=*/module.get(),
      /*.target_profile=*/nullptr,
      /*.target_key=*/{},
      /*.artifact_identifier=*/IREE_SV("roundtrip.spv"),
      /*.export_count=*/1,
      /*.compile_options=*/&options,
      /*.block_pool=*/&block_pool_,
      /*.option_chain=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_product_t* product = nullptr;
  IREE_ASSERT_OK(loom_product_format_provider_build(
      &loom_spirv_binary_product_provider, &request, &product));
  ASSERT_NE(product, nullptr);
  ASSERT_TRUE(loom_product_isa(product, &loom_kernel_product_descriptor));
  EXPECT_EQ(loom_product_export_count(product), 1u);

  const loom_product_artifact_t* artifact =
      loom_kernel_product_loadable_artifact(product);
  ASSERT_NE(artifact, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      artifact->role, IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL)));
  EXPECT_TRUE(iree_string_view_equal(
      artifact->format, IREE_SV(LOOM_SPIRV_PRODUCT_FORMAT_BINARY)));
  EXPECT_TRUE(
      iree_string_view_equal(artifact->identifier, IREE_SV("roundtrip.spv")));
  ASSERT_NE(artifact->contents, nullptr);
  ByteSequenceClone executable_data(iree_allocator_system());
  IREE_ASSERT_OK(executable_data.Clone(artifact->contents));
  const iree_const_byte_span_t executable_contents = executable_data.contents();
  ASSERT_GE(executable_contents.data_length, sizeof(uint32_t));
  uint32_t magic = 0;
  memcpy(&magic, executable_contents.data, sizeof(magic));
  EXPECT_EQ(magic, 0x07230203u);

  loom_product_release(product);
}

}  // namespace
}  // namespace loom
