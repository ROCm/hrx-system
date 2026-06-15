// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

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

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

uint32_t LoadLeU32(const uint8_t* bytes, size_t offset) {
  return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
         ((uint32_t)bytes[offset + 2] << 16) |
         ((uint32_t)bytes[offset + 3] << 24);
}

std::string DiagnosticSummary(const DiagnosticCapture& capture) {
  std::string result;
  for (const CapturedDiagnostic& diagnostic : capture.diagnostics) {
    if (!result.empty()) {
      result += "\n";
    }
    result += diagnostic.error ? diagnostic.error->error_id : "<unknown>";
    result += ": ";
    result += diagnostic.error ? diagnostic.error->summary : "";
    for (const loom_diagnostic_param_t& param : diagnostic.params) {
      result += " [";
      switch (param.kind) {
        case LOOM_PARAM_STRING:
          result.append(param.string.data, param.string.size);
          break;
        case LOOM_PARAM_I64:
          result += std::to_string(param.i64);
          break;
        case LOOM_PARAM_U32:
          result += std::to_string(param.u32);
          break;
        case LOOM_PARAM_U64:
          result += std::to_string(param.u64);
          break;
        default:
          result += "?";
          break;
      }
      result += "]";
    }
  }
  return result;
}

class AmdgpuHalKernelLibraryTest : public ::testing::Test {
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

  void ParseSource(iree_string_view_t source, loom_module_t** out_module) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/parse_capture.sink(),
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &parse_options.low_asm_environment);
    iree_status_t status =
        loom_text_parse(source, IREE_SV("amdgpu_emit_test.loom"), &context_,
                        &block_pool_, &parse_options, out_module);
    if (!parse_capture.diagnostics.empty()) {
      ADD_FAILURE() << DiagnosticSummary(parse_capture);
    }
    IREE_ASSERT_OK(status);
    ASSERT_TRUE(parse_capture.diagnostics.empty());
    ASSERT_NE(*out_module, nullptr);
  }

  void ParseGfx11Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseGfx11KernelWithArguments(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) "
        "abi_layout({constant_count = 1, direct_arg_count = 1, "
        "direct_arg_names = {arg0 = \"extent\"}, direct_arg_sizes = [4], "
        "resource_count = 1, uses_kernarg_segment_ptr = true}) "
        "workgroup_size(64, 1, 1) @loom_kernel() {\n"
        "  %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : "
        "reg<amdgpu.sgpr x2>\n"
        "  %extent = low.op<amdgpu.s_load_dword_offset_only>(%kernarg) "
        "{offset = 8} : (reg<amdgpu.sgpr x2>) -> reg<amdgpu.sgpr>\n"
        "  %view = low.op<amdgpu.s_load_dwordx2_offset_only>(%kernarg) "
        "{offset = 0} : (reg<amdgpu.sgpr x2>) -> reg<amdgpu.sgpr x2>\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseGfx11MultiKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@first_kernel() {\n"
        "  low.return\n"
        "}\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@second_kernel() {\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseGfx942Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx942> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseKernelForProcessor(const loom_amdgpu_processor_info_t* processor,
                               loom_module_t** out_module) {
    const loom_amdgpu_target_record_info_t* record_info =
        loom_amdgpu_target_record_default_info_for_descriptor_set(
            processor->descriptor_set.ordinal);
    ASSERT_NE(record_info, nullptr) << StringViewToString(processor->name);

    std::string source = "amdgpu.target<";
    source.append(record_info->default_processor_name.data,
                  record_info->default_processor_name.size);
    source += "> @gfx_target";
    if (!iree_string_view_equal(processor->name,
                                record_info->default_processor_name)) {
      source += " {processor = \"";
      source.append(processor->name.data, processor->name.size);
      source += "\"}";
    }
    source +=
        "\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";

    ASSERT_NO_FATAL_FAILURE(ParseSource(
        iree_make_string_view(source.data(), source.size()), out_module))
        << StringViewToString(processor->name);
  }

  bool IsDescriptorSetLinked(iree_string_view_t descriptor_set_key) const {
    return loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                               descriptor_set_key) != nullptr;
  }

  bool IsProcessorDescriptorSetLinked(
      const loom_amdgpu_processor_info_t* processor) const {
    return IsDescriptorSetLinked(processor->descriptor_set.key);
  }

  void EmitWithProcessor(iree_string_view_t processor,
                         DiagnosticCapture* capture, bool* out_emitted) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        /*.processor=*/processor,
        /*.target_selection=*/{},
        /*.diagnostic_sink=*/capture->sink(),
        /*.source_resolver=*/{},
        /*.max_errors=*/20,
    };
    iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), out_emitted, &library);
    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  void EmitGfx942Kernel(DiagnosticCapture* capture, bool* out_emitted) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx942Kernel(&module));

    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        /*.processor=*/IREE_SV("gfx942"),
        /*.target_selection=*/{},
        /*.diagnostic_sink=*/capture->sink(),
        /*.source_resolver=*/{},
        /*.max_errors=*/20,
    };
    iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), out_emitted, &library);
    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(AmdgpuHalKernelLibraryTest, UnknownProcessorEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx9999"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_003);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx9999");
}

TEST_F(AmdgpuHalKernelLibraryTest,
       ProcessorWithoutDescriptorSetEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx908"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_004);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx908");
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsGfx942Kernel) {
  if (!IsDescriptorSetLinked(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "amdgpu.cdna3.core is not linked in this build";
  }
  DiagnosticCapture capture;
  bool emitted = false;
  ASSERT_NO_FATAL_FAILURE(EmitGfx942Kernel(&capture, &emitted));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsEveryLinkedSupportedProcessor) {
  iree_host_size_t linked_supported_count = 0;
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0; i < processor_count; ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);
    bool hsaco_supported = false;
    IREE_ASSERT_OK(loom_amdgpu_target_info_processor_supports_hsaco(
        processor, &hsaco_supported));
    if (!hsaco_supported) {
      continue;
    }
    if (!IsProcessorDescriptorSetLinked(processor)) {
      continue;
    }
    ++linked_supported_count;

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForProcessor(processor, &module));

    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        /*.processor=*/{},
        /*.target_selection=*/{},
        /*.diagnostic_sink=*/capture.sink(),
        /*.source_resolver=*/{},
        /*.max_errors=*/20,
    };
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library))
        << StringViewToString(processor->name);

    EXPECT_TRUE(emitted) << StringViewToString(processor->name);
    EXPECT_TRUE(capture.diagnostics.empty())
        << StringViewToString(processor->name);
    EXPECT_NE(library.hsaco_data, nullptr)
        << StringViewToString(processor->name);
    EXPECT_GT(library.hsaco_data_length, 64u)
        << StringViewToString(processor->name);
    if (library.hsaco_data_length > 64u) {
      EXPECT_EQ(LoadLeU32(library.hsaco_data, 48),
                processor->elf.machine_flags | processor->elf.feature_flags)
          << StringViewToString(processor->name);
    }
    EXPECT_NE(
        iree_string_view_find(library.executable_format, processor->name, 0),
        IREE_STRING_VIEW_NPOS)
        << StringViewToString(processor->name);
    std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                      library.hsaco_data_length);
    EXPECT_NE(hsaco.find("loom_kernel.kd"), std::string::npos)
        << StringViewToString(processor->name);

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
  }
  EXPECT_GE(linked_supported_count, 1u);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsArgumentMetadataFromLowKernelAbi) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11KernelWithArguments(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.diagnostic_sink=*/capture.sink(),
      /*.source_resolver=*/{},
      /*.max_errors=*/20,
  };
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
  ASSERT_NE(library.hsaco_data, nullptr);
  std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                    library.hsaco_data_length);
  EXPECT_NE(hsaco.find("binding0"), std::string::npos);
  EXPECT_NE(hsaco.find("global_buffer"), std::string::npos);
  EXPECT_NE(hsaco.find("extent"), std::string::npos);
  EXPECT_NE(hsaco.find("by_value"), std::string::npos);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsAllCompatibleKernels) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11MultiKernel(&module));

  loom_target_artifact_manifest_collect_options_t artifact_manifest_options;
  loom_target_artifact_manifest_collect_options_initialize(
      &artifact_manifest_options);
  artifact_manifest_options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.diagnostic_sink=*/capture.sink(),
      /*.source_resolver=*/{},
      /*.max_errors=*/20,
      /*.report=*/nullptr,
      /*.capture_target_listing=*/false,
      /*.artifact_name=*/{},
      /*.artifact_manifest_identifier=*/{},
      /*.artifact_manifest=*/artifact_manifest_options,
  };
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
  ASSERT_NE(library.hsaco_data, nullptr);
  std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                    library.hsaco_data_length);
  EXPECT_NE(hsaco.find("first_kernel.kd"), std::string::npos);
  EXPECT_NE(hsaco.find("second_kernel.kd"), std::string::npos);

  ASSERT_NE(library.artifact_manifest.contents.data, nullptr);
  std::string manifest(
      reinterpret_cast<const char*>(library.artifact_manifest.contents.data),
      library.artifact_manifest.contents.data_length);
  size_t first_manifest_position = manifest.find("\"name\":\"first_kernel\"");
  size_t second_manifest_position = manifest.find("\"name\":\"second_kernel\"");
  ASSERT_NE(first_manifest_position, std::string::npos) << manifest;
  ASSERT_NE(second_manifest_position, std::string::npos) << manifest;
  EXPECT_LT(first_manifest_position, second_manifest_position) << manifest;

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, DescriptorSetMismatchEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx950"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_005);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx950");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.cdna4.core");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "gfx_target");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "amdgpu.rdna3.core");
}

}  // namespace
}  // namespace loom
