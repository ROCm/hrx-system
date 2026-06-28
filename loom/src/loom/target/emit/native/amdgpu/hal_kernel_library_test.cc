// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/target/emit/native/elf.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/tooling/compile/pipeline.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

iree_status_t InitializeAmdgpuContext(
    const loom_target_environment_t* target_environment,
    loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status =
        loom_target_environment_register_context(target_environment, context);
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

bool SupportsWgpMode(const loom_amdgpu_processor_info_t* processor) {
  switch (processor->kernel_descriptor.profile) {
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12:
      return true;
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9:
    case LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125:
      return false;
  }
  return false;
}

uint32_t LoadLeU32(const uint8_t* bytes, size_t offset) {
  return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
         ((uint32_t)bytes[offset + 2] << 16) |
         ((uint32_t)bytes[offset + 3] << 24);
}

uint16_t LoadLeU16(const std::string& bytes, size_t offset) {
  return (uint16_t)(uint8_t)bytes[offset] |
         ((uint16_t)(uint8_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::string& bytes, size_t offset) {
  return (uint32_t)(uint8_t)bytes[offset] |
         ((uint32_t)(uint8_t)bytes[offset + 1] << 8) |
         ((uint32_t)(uint8_t)bytes[offset + 2] << 16) |
         ((uint32_t)(uint8_t)bytes[offset + 3] << 24);
}

uint64_t LoadLeU64(const std::string& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)(uint8_t)bytes[offset + i] << (8 * i);
  }
  return value;
}

std::string ReadNullTerminatedString(const std::string& bytes, size_t offset) {
  std::string value;
  while (offset < bytes.size() && bytes[offset] != '\0') {
    value.push_back(bytes[offset++]);
  }
  return value;
}

struct Section {
  // Section table ordinal.
  size_t index;
  // Section name from .shstrtab.
  std::string name;
  // ELF section type.
  uint32_t type;
  // ELF section flags.
  uint64_t flags;
  // Runtime virtual address assigned to the section.
  uint64_t address;
  // File offset of section contents.
  uint64_t offset;
  // Byte length of section contents.
  uint64_t size;
  // Linked section index.
  uint32_t link;
  // Section entry size for table-like sections.
  uint64_t entry_size;
};

std::vector<Section> ReadSections(const std::string& bytes) {
  const size_t section_header_offset = (size_t)LoadLeU64(bytes, 40);
  const size_t section_count = LoadLeU16(bytes, 60);
  const size_t section_name_index = LoadLeU16(bytes, 62);
  EXPECT_LT(section_name_index, section_count);

  const size_t section_name_header =
      section_header_offset + section_name_index * 64;
  const size_t section_name_offset =
      (size_t)LoadLeU64(bytes, section_name_header + 24);
  const size_t section_name_size =
      (size_t)LoadLeU64(bytes, section_name_header + 32);
  EXPECT_LE(section_name_offset + section_name_size, bytes.size());

  std::vector<Section> sections;
  sections.reserve(section_count);
  for (size_t i = 0; i < section_count; ++i) {
    const size_t header_offset = section_header_offset + i * 64;
    const uint32_t name_offset = LoadLeU32(bytes, header_offset);
    EXPECT_LT(name_offset, section_name_size);
    sections.push_back({
        /*.index=*/i,
        /*.name=*/
        ReadNullTerminatedString(bytes, section_name_offset + name_offset),
        /*.type=*/LoadLeU32(bytes, header_offset + 4),
        /*.flags=*/LoadLeU64(bytes, header_offset + 8),
        /*.address=*/LoadLeU64(bytes, header_offset + 16),
        /*.offset=*/LoadLeU64(bytes, header_offset + 24),
        /*.size=*/LoadLeU64(bytes, header_offset + 32),
        /*.link=*/LoadLeU32(bytes, header_offset + 40),
        /*.entry_size=*/LoadLeU64(bytes, header_offset + 56),
    });
  }
  return sections;
}

const Section& FindSection(const std::vector<Section>& sections,
                           const char* name) {
  for (const Section& section : sections) {
    if (section.name == name) {
      return section;
    }
  }
  ADD_FAILURE() << "section not found: " << name;
  return sections[0];
}

struct DynamicSymbol {
  // Symbol table ordinal.
  size_t index;
  // Dynamic symbol name.
  std::string name;
  // ELF symbol info byte.
  uint8_t info;
  // Section index containing the symbol.
  uint16_t section_index;
  // Runtime virtual address assigned to the symbol.
  uint64_t value;
  // Byte length of the symbol.
  uint64_t size;
};

DynamicSymbol FindDynamicSymbol(const std::string& bytes,
                                const Section& dynamic_symbol_table,
                                const Section& dynamic_string_table,
                                const char* name) {
  const std::string dynamic_strings = bytes.substr(
      (size_t)dynamic_string_table.offset, (size_t)dynamic_string_table.size);
  const size_t symbol_count =
      (size_t)(dynamic_symbol_table.size / dynamic_symbol_table.entry_size);
  for (size_t i = 0; i < symbol_count; ++i) {
    const size_t offset = (size_t)dynamic_symbol_table.offset +
                          i * dynamic_symbol_table.entry_size;
    DynamicSymbol symbol = {
        /*.index=*/i,
        /*.name=*/
        ReadNullTerminatedString(dynamic_strings, LoadLeU32(bytes, offset)),
        /*.info=*/(uint8_t)bytes[offset + 4],
        /*.section_index=*/LoadLeU16(bytes, offset + 6),
        /*.value=*/LoadLeU64(bytes, offset + 8),
        /*.size=*/LoadLeU64(bytes, offset + 16),
    };
    if (symbol.name == name) {
      return symbol;
    }
  }
  ADD_FAILURE() << "dynamic symbol not found: " << name;
  return {};
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
    IREE_ASSERT_OK(loom_target_environment_initialize(
        &loom_amdgpu_target_provider_set, &target_environment_));
    IREE_ASSERT_OK(InitializeAmdgpuContext(&target_environment_, &context_));
    IREE_ASSERT_OK(loom_target_environment_initialize_low_descriptor_registry(
        &target_environment_, &low_registry_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    loom_target_environment_deinitialize(&target_environment_);
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

  void ParseGfx11DynamicHalKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
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
        "direct_arg_names = {arg0 = \"extent\"}, direct_arg_offsets = [8], "
        "direct_arg_parameter_indices = [1], direct_arg_sizes = [4], "
        "parameter_count = 2, resource_count = 1, resource_offsets = [0], "
        "resource_parameter_indices = [0], uses_kernarg_segment_ptr = true}) "
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

  void ParseGfx11SourceSanitizerKernels(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "kernel.def target(@gfx_target) @read_kernel() {\n"
        "  %one = index.constant 1 : index\n"
        "  %size = index.constant 64 : index\n"
        "  kernel.launch.config workgroups(%one, %one, %one) "
        "workgroup_size(%size, %one, %one) : index\n"
        "} launch(%input: buffer) {\n"
        "  %base = index.constant 0 : offset\n"
        "  %input_global = buffer.assume.memory_space<global> %input : "
        "buffer\n"
        "  %input_view = buffer.view %input_global[%base] : buffer -> "
        "view<1xi32, #dense>\n"
        "  sanitizer.assert.access<read> %input_view[0] : "
        "view<1xi32, #dense>\n"
        "  kernel.return\n"
        "}\n"
        "kernel.def target(@gfx_target) @write_kernel() {\n"
        "  %one = index.constant 1 : index\n"
        "  %size = index.constant 64 : index\n"
        "  kernel.launch.config workgroups(%one, %one, %one) "
        "workgroup_size(%size, %one, %one) : index\n"
        "} launch(%output: buffer) {\n"
        "  %base = index.constant 0 : offset\n"
        "  %output_global = buffer.assume.memory_space<global> %output : "
        "buffer\n"
        "  %output_view = buffer.view %output_global[%base] : buffer -> "
        "view<1xi32, #dense>\n"
        "  sanitizer.assert.access<write> %output_view[0] : "
        "view<1xi32, #dense>\n"
        "  kernel.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void RunPreparedLowPipeline(loom_module_t* module,
                              DiagnosticCapture* capture) {
    loom_compile_pipeline_options_t options = {};
    loom_compile_pipeline_options_initialize(&options);
    options.target_environment = &target_environment_;
    options.low_descriptor_registry = &low_registry_;
    options.diagnostic_sink = capture->sink();
    options.max_errors = 20;
    loom_pass_run_result_t result = {};
    IREE_ASSERT_OK(
        loom_compile_run_pipeline(module, &options, &block_pool_, &result));
    ASSERT_EQ(result.error_count, 0u) << DiagnosticSummary(*capture);
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
        /*.runtime_globals=*/{},
        /*.data_symbols=*/{},
        /*.data_symbol_count=*/{},
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
        /*.runtime_globals=*/{},
        /*.data_symbols=*/{},
        /*.data_symbol_count=*/{},
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
  loom_target_environment_t target_environment_ = {};
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

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsDynamicLocalSizeKernel) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11DynamicHalKernel(&module));

  loom_target_artifact_manifest_collect_options_t artifact_manifest_options;
  loom_target_artifact_manifest_collect_options_initialize(
      &artifact_manifest_options);
  artifact_manifest_options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/{},
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
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

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
  ASSERT_NE(library.hsaco_data, nullptr);
  std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                    library.hsaco_data_length);
  EXPECT_NE(hsaco.find("loom_kernel.kd"), std::string::npos);
  EXPECT_NE(hsaco.find(".max_flat_workgroup_size"), std::string::npos);
  EXPECT_EQ(hsaco.find(".reqd_workgroup_size"), std::string::npos);

  ASSERT_NE(library.artifact_manifest.contents.data, nullptr);
  std::string manifest(
      reinterpret_cast<const char*>(library.artifact_manifest.contents.data),
      library.artifact_manifest.contents.data_length);
  EXPECT_NE(manifest.find("\"name\":\"loom_kernel\""), std::string::npos)
      << manifest;
  EXPECT_NE(manifest.find("\"subgroup_size\":32"), std::string::npos)
      << manifest;
  EXPECT_EQ(manifest.find("\"workgroup_size\""), std::string::npos) << manifest;

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsEveryLinkedSupportedProcessor) {
  iree_host_size_t linked_supported_count = 0;
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0; i < processor_count; ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);
    if (!loom_amdgpu_processor_supports_hsaco(processor)) {
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
        /*.runtime_globals=*/{},
        /*.data_symbols=*/{},
        /*.data_symbol_count=*/{},
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
    EXPECT_NE(iree_string_view_find(library.target_key, processor->name, 0),
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

TEST_F(AmdgpuHalKernelLibraryTest, CapturesCompleteGfx11KernelDirectives) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/{},
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
      /*.diagnostic_sink=*/capture.sink(),
      /*.source_resolver=*/{},
      /*.max_errors=*/20,
      /*.report=*/{},
      /*.capture_target_listing=*/true,
  };
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
  ASSERT_NE(library.target_listing_data, nullptr);
  ASSERT_TRUE(iree_string_view_equal(library.target_listing_format,
                                     IREE_SV("amdgpu-assembly")));
  const std::string listing(library.target_listing_data,
                            library.target_listing_data_length);
  static constexpr const char* kExpectedDirectives[] = {
      "  .amdhsa_user_sgpr_count 0\n",
      "  .amdhsa_user_sgpr_dispatch_ptr 0\n",
      "  .amdhsa_user_sgpr_queue_ptr 0\n",
      "  .amdhsa_user_sgpr_kernarg_segment_ptr 0\n",
      "  .amdhsa_user_sgpr_dispatch_id 0\n",
      "  .amdhsa_user_sgpr_private_segment_size 0\n",
      "  .amdhsa_wavefront_size32 1\n",
      "  .amdhsa_uses_dynamic_stack 0\n",
      "  .amdhsa_enable_private_segment 0\n",
      "  .amdhsa_system_sgpr_workgroup_id_x 0\n",
      "  .amdhsa_system_sgpr_workgroup_id_y 0\n",
      "  .amdhsa_system_sgpr_workgroup_id_z 0\n",
      "  .amdhsa_system_sgpr_workgroup_info 0\n",
      "  .amdhsa_system_vgpr_workitem_id 0\n",
      "  .amdhsa_next_free_vgpr 0\n",
      "  .amdhsa_next_free_sgpr 0\n",
      "  .amdhsa_reserve_vcc 0\n",
      "  .amdhsa_float_round_mode_32 0\n",
      "  .amdhsa_float_round_mode_16_64 0\n",
      "  .amdhsa_float_denorm_mode_32 3\n",
      "  .amdhsa_float_denorm_mode_16_64 3\n",
      "  .amdhsa_dx10_clamp 1\n",
      "  .amdhsa_ieee_mode 1\n",
      "  .amdhsa_fp16_overflow 0\n",
      "  .amdhsa_workgroup_processor_mode 1\n",
      "  .amdhsa_memory_ordered 1\n",
      "  .amdhsa_forward_progress 1\n",
      "  .amdhsa_shared_vgpr_count 0\n",
      "  .amdhsa_inst_pref_size 0\n",
      "  .amdhsa_exception_fp_ieee_invalid_op 0\n",
      "  .amdhsa_exception_fp_denorm_src 0\n",
      "  .amdhsa_exception_fp_ieee_div_zero 0\n",
      "  .amdhsa_exception_fp_ieee_overflow 0\n",
      "  .amdhsa_exception_fp_ieee_underflow 0\n",
      "  .amdhsa_exception_fp_ieee_inexact 0\n",
      "  .amdhsa_exception_int_div_zero 0\n",
  };
  for (const char* directive : kExpectedDirectives) {
    EXPECT_NE(listing.find(directive), std::string::npos) << directive;
  }
  EXPECT_EQ(listing.find(".amdhsa_user_sgpr_private_segment_buffer"),
            std::string::npos);
  EXPECT_EQ(listing.find(".amdhsa_user_sgpr_flat_scratch_init"),
            std::string::npos);
  EXPECT_EQ(listing.find(".amdhsa_reserve_flat_scratch"), std::string::npos);
  EXPECT_EQ(
      listing.find(".amdhsa_system_sgpr_private_segment_wavefront_offset"),
      std::string::npos);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, CapturesTargetSpecificKernelDirectives) {
  static constexpr const char* kProcessors[] = {
      "gfx942", "gfx950", "gfx1100", "gfx1170", "gfx1200", "gfx1250",
  };
  for (const char* processor_name : kProcessors) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_find_processor(
            iree_make_cstring_view(processor_name));
    ASSERT_NE(processor, nullptr) << processor_name;
    if (!loom_amdgpu_processor_supports_hsaco(processor) ||
        !IsProcessorDescriptorSetLinked(processor)) {
      continue;
    }

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForProcessor(processor, &module));

    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        /*.processor=*/{},
        /*.target_selection=*/{},
        /*.runtime_globals=*/{},
        /*.data_symbols=*/{},
        /*.data_symbol_count=*/{},
        /*.diagnostic_sink=*/capture.sink(),
        /*.source_resolver=*/{},
        /*.max_errors=*/20,
        /*.report=*/{},
        /*.capture_target_listing=*/true,
    };
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library))
        << processor_name;

    EXPECT_TRUE(emitted) << processor_name;
    EXPECT_TRUE(capture.diagnostics.empty()) << processor_name;
    ASSERT_NE(library.target_listing_data, nullptr) << processor_name;
    const std::string listing(library.target_listing_data,
                              library.target_listing_data_length);

    const bool has_architected_flat_scratch =
        loom_amdgpu_processor_kernel_descriptor_has_flags(
            processor,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH);
    if (has_architected_flat_scratch) {
      EXPECT_NE(listing.find("  .amdhsa_enable_private_segment 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_user_sgpr_private_segment_buffer"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_user_sgpr_flat_scratch_init"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_reserve_flat_scratch"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_system_sgpr_private_segment_"
                             "wavefront_offset"),
                std::string::npos)
          << processor_name;
    } else {
      EXPECT_NE(listing.find("  .amdhsa_user_sgpr_private_segment_buffer 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_user_sgpr_flat_scratch_init 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_reserve_flat_scratch 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_system_sgpr_private_segment_"
                             "wavefront_offset 0\n"),
                std::string::npos)
          << processor_name;
    }

    const bool has_accum_offset =
        loom_amdgpu_processor_kernel_descriptor_has_flags(
            processor, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET);
    if (has_accum_offset) {
      EXPECT_NE(listing.find("  .amdhsa_accum_offset 4\n"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_tg_split 0\n"), std::string::npos)
          << processor_name;
    } else {
      EXPECT_EQ(listing.find(".amdhsa_accum_offset"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_tg_split"), std::string::npos)
          << processor_name;
    }

    const bool has_dx10_clamp_and_ieee_mode =
        loom_amdgpu_processor_kernel_descriptor_has_flags(
            processor,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE);
    if (has_dx10_clamp_and_ieee_mode) {
      EXPECT_NE(listing.find("  .amdhsa_dx10_clamp 1\n"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_ieee_mode 1\n"), std::string::npos)
          << processor_name;
    } else {
      EXPECT_EQ(listing.find(".amdhsa_dx10_clamp"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_ieee_mode"), std::string::npos)
          << processor_name;
    }

    if (SupportsWgpMode(processor)) {
      EXPECT_NE(listing.find("  .amdhsa_workgroup_processor_mode 1\n"),
                std::string::npos)
          << processor_name;
    } else {
      EXPECT_EQ(listing.find(".amdhsa_workgroup_processor_mode"),
                std::string::npos)
          << processor_name;
    }

    const bool uses_gfx10_sgpr_encoding =
        loom_amdgpu_processor_kernel_descriptor_has_flags(
            processor,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING);
    if (uses_gfx10_sgpr_encoding) {
      EXPECT_NE(listing.find("  .amdhsa_wavefront_size32 1\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_memory_ordered 1\n"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_forward_progress 1\n"),
                std::string::npos)
          << processor_name;
    } else {
      EXPECT_EQ(listing.find(".amdhsa_wavefront_size32"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_memory_ordered"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_forward_progress"), std::string::npos)
          << processor_name;
    }

    if (processor->kernel_descriptor.profile ==
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11) {
      EXPECT_NE(listing.find("  .amdhsa_shared_vgpr_count 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_inst_pref_size 0\n"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_round_robin_scheduling"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_named_barrier_count"), std::string::npos)
          << processor_name;
    } else if (processor->kernel_descriptor.profile ==
               LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12) {
      EXPECT_EQ(listing.find(".amdhsa_shared_vgpr_count"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_inst_pref_size 0\n"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_round_robin_scheduling 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_named_barrier_count"), std::string::npos)
          << processor_name;
    } else if (processor->kernel_descriptor.profile ==
               LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125) {
      EXPECT_EQ(listing.find(".amdhsa_shared_vgpr_count"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_inst_pref_size 0\n"), std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_round_robin_scheduling 0\n"),
                std::string::npos)
          << processor_name;
      EXPECT_NE(listing.find("  .amdhsa_named_barrier_count 0\n"),
                std::string::npos)
          << processor_name;
    } else {
      EXPECT_EQ(listing.find(".amdhsa_shared_vgpr_count"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_inst_pref_size"), std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_round_robin_scheduling"),
                std::string::npos)
          << processor_name;
      EXPECT_EQ(listing.find(".amdhsa_named_barrier_count"), std::string::npos)
          << processor_name;
    }

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
  }
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsArgumentMetadataFromLowKernelAbi) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11KernelWithArguments(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/{},
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
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
      /*.runtime_globals=*/{},
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
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

TEST_F(AmdgpuHalKernelLibraryTest, EmitsRequestedRuntimeGlobals) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
          LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG |
          LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG,
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
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

  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& data = FindSection(sections, ".data");

  EXPECT_EQ(data.type, LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS);
  EXPECT_EQ(data.flags, LOOM_NATIVE_ELF_SECTION_FLAG_WRITE |
                            LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC);
  EXPECT_EQ(data.size,
            LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH +
                LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_BYTE_LENGTH +
                LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);

  const std::string dynstr_contents =
      hsaco.substr((size_t)dynstr.offset, (size_t)dynstr.size);
  ASSERT_EQ(dynsym.entry_size, 24u);
  ASSERT_EQ(dynsym.size, 6u * 24u);

  const size_t asan_symbol = (size_t)dynsym.offset + 3u * 24u;
  const size_t tsan_symbol = asan_symbol + 24u;
  const size_t final_feedback_symbol = tsan_symbol + 24u;
  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(hsaco, asan_symbol + 0)),
            StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME));
  EXPECT_EQ((uint8_t)hsaco[asan_symbol + 4], 0x11u);
  EXPECT_EQ(LoadLeU16(hsaco, asan_symbol + 6), data.index);
  EXPECT_EQ(LoadLeU64(hsaco, asan_symbol + 16),
            LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH);

  EXPECT_EQ(ReadNullTerminatedString(dynstr_contents,
                                     LoadLeU32(hsaco, tsan_symbol + 0)),
            StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_NAME));
  EXPECT_EQ((uint8_t)hsaco[tsan_symbol + 4], 0x11u);
  EXPECT_EQ(LoadLeU16(hsaco, tsan_symbol + 6), data.index);
  EXPECT_EQ(LoadLeU64(hsaco, tsan_symbol + 16),
            LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_BYTE_LENGTH);

  EXPECT_EQ(
      ReadNullTerminatedString(dynstr_contents,
                               LoadLeU32(hsaco, final_feedback_symbol + 0)),
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME));
  EXPECT_EQ((uint8_t)hsaco[final_feedback_symbol + 4], 0x11u);
  EXPECT_EQ(LoadLeU16(hsaco, final_feedback_symbol + 6), data.index);
  EXPECT_EQ(LoadLeU64(hsaco, final_feedback_symbol + 16),
            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest,
       EmitsCallerDataSymbolsAndRel32AddressMaterialization) {
  static constexpr char kSiteSymbolName[] = "loom_sanitizer_sites";
  static const uint8_t kSiteRecords[] = {
      0x00, 0x02, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00,
      0x00, 0x01, 0x01, 0x06, 0x01, 0x01, 0x00, 0x00,
  };
  static const char kSource[] =
      "amdgpu.target<gfx1100> @gfx_target\n"
      "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
      "@loom_kernel() {\n"
      "  %pc = low.op<amdgpu.s_getpc_b64>() : () -> "
      "reg<amdgpu.sgpr x2>\n"
      "  %pc_lo = low.slice %pc[0] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %pc_hi = low.slice %pc[1] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %site_lo = low.op<amdgpu.s_add_u32.rhs_symbol_rel32_lo>(%pc_lo) "
      "{symbol = @loom_sanitizer_sites, byte_offset = 8} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %site_hi = low.op<amdgpu.s_addc_u32.rhs_symbol_rel32_hi>(%pc_hi) "
      "{symbol = @loom_sanitizer_sites, byte_offset = 8} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      ParseSource(iree_make_cstring_view(kSource), &module));

  const loom_amdgpu_hsaco_data_symbol_t site_symbol = {
      /*.name=*/IREE_SV(kSiteSymbolName),
      /*.initial_contents=*/
      iree_make_const_byte_span(kSiteRecords, sizeof(kSiteRecords)),
      /*.byte_length=*/sizeof(kSiteRecords),
      /*.alignment=*/16,
  };
  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG,
      /*.data_symbols=*/&site_symbol,
      /*.data_symbol_count=*/1,
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
  const std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                          library.hsaco_data_length);

  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& text = FindSection(sections, ".text");
  const std::string feedback_name =
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME);
  const DynamicSymbol site =
      FindDynamicSymbol(hsaco, dynsym, dynstr, kSiteSymbolName);
  const DynamicSymbol feedback =
      FindDynamicSymbol(hsaco, dynsym, dynstr, feedback_name.c_str());

  EXPECT_EQ(site.info, 0x11u);
  EXPECT_EQ(site.section_index, rodata.index);
  EXPECT_EQ(site.size, sizeof(kSiteRecords));
  ASSERT_GE(site.value, rodata.address);
  ASSERT_LE(site.value - rodata.address, rodata.size);
  const size_t site_file_offset =
      (size_t)(rodata.offset + (site.value - rodata.address));
  ASSERT_LE(site_file_offset + sizeof(kSiteRecords), hsaco.size());
  EXPECT_EQ(hsaco.substr(site_file_offset, sizeof(kSiteRecords)),
            std::string((const char*)kSiteRecords, sizeof(kSiteRecords)));

  EXPECT_EQ(feedback.info, 0x11u);
  EXPECT_NE(feedback.section_index, site.section_index);
  EXPECT_EQ(feedback.size,
            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);

  const uint64_t base_pc_address = text.address + 4u;
  const uint64_t site_delta = site.value + 8u - base_pc_address;
  ASSERT_LE(text.offset + 20u, hsaco.size());
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 8u), (uint32_t)site_delta);
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 16u),
            (uint32_t)(site_delta >> 32));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest,
       EmitsGlobalRodataSymbolsAndRel32AddressMaterialization) {
  static constexpr char kSiteSymbolName[] = "loom_sanitizer_sites";
  static const uint8_t kSiteRecords[] = {
      0x00, 0x02, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00,
      0x00, 0x01, 0x01, 0x06, 0x01, 0x01, 0x00, 0x00,
  };
  static const char kSource[] =
      "global.rodata @loom_sanitizer_sites = "
      "bytes(\"00020302010100000001010601010000\"), align 16\n"
      "amdgpu.target<gfx1100> @gfx_target\n"
      "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
      "@loom_kernel() {\n"
      "  %pc = low.op<amdgpu.s_getpc_b64>() : () -> "
      "reg<amdgpu.sgpr x2>\n"
      "  %pc_lo = low.slice %pc[0] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %pc_hi = low.slice %pc[1] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %site_lo = low.op<amdgpu.s_add_u32.rhs_symbol_rel32_lo>(%pc_lo) "
      "{symbol = @loom_sanitizer_sites, byte_offset = 8} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %site_hi = low.op<amdgpu.s_addc_u32.rhs_symbol_rel32_hi>(%pc_hi) "
      "{symbol = @loom_sanitizer_sites, byte_offset = 8} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      ParseSource(iree_make_cstring_view(kSource), &module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG,
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
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
  const std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                          library.hsaco_data_length);

  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& text = FindSection(sections, ".text");
  const std::string feedback_name =
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME);
  const DynamicSymbol site =
      FindDynamicSymbol(hsaco, dynsym, dynstr, kSiteSymbolName);
  const DynamicSymbol feedback =
      FindDynamicSymbol(hsaco, dynsym, dynstr, feedback_name.c_str());

  EXPECT_EQ(site.info, 0x11u);
  EXPECT_EQ(site.section_index, rodata.index);
  EXPECT_EQ(site.size, sizeof(kSiteRecords));
  ASSERT_GE(site.value, rodata.address);
  ASSERT_LE(site.value - rodata.address, rodata.size);
  const size_t site_file_offset =
      (size_t)(rodata.offset + (site.value - rodata.address));
  ASSERT_LE(site_file_offset + sizeof(kSiteRecords), hsaco.size());
  EXPECT_EQ(hsaco.substr(site_file_offset, sizeof(kSiteRecords)),
            std::string((const char*)kSiteRecords, sizeof(kSiteRecords)));

  EXPECT_EQ(feedback.info, 0x11u);
  EXPECT_NE(feedback.section_index, site.section_index);
  EXPECT_EQ(feedback.size,
            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);

  const uint64_t base_pc_address = text.address + 4u;
  const uint64_t site_delta = site.value + 8u - base_pc_address;
  ASSERT_LE(text.offset + 20u, hsaco.size());
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 8u), (uint32_t)site_delta);
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 16u),
            (uint32_t)(site_delta >> 32));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsSourceLoweredSanitizerSiteTableRodata) {
  static constexpr char kSiteSymbolName[] = "loom_sanitizer_sites";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11SourceSanitizerKernels(&module));

  DiagnosticCapture capture;
  ASSERT_NO_FATAL_FAILURE(RunPreparedLowPipeline(module, &capture));

  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      /*.processor=*/{},
      /*.target_selection=*/{},
      /*.runtime_globals=*/LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
          LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG,
      /*.data_symbols=*/{},
      /*.data_symbol_count=*/{},
      /*.diagnostic_sink=*/capture.sink(),
      /*.source_resolver=*/{},
      /*.max_errors=*/20,
  };
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
  ASSERT_NE(library.hsaco_data, nullptr);
  const std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                          library.hsaco_data_length);

  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynsym = FindSection(sections, ".dynsym");
  const Section& dynstr = FindSection(sections, ".dynstr");
  const Section& rodata = FindSection(sections, ".rodata");
  const Section& data = FindSection(sections, ".data");
  const DynamicSymbol site =
      FindDynamicSymbol(hsaco, dynsym, dynstr, kSiteSymbolName);
  const std::string asan_name =
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME);
  const std::string feedback_name =
      StringViewToString(LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME);
  const DynamicSymbol asan =
      FindDynamicSymbol(hsaco, dynsym, dynstr, asan_name.c_str());
  const DynamicSymbol feedback =
      FindDynamicSymbol(hsaco, dynsym, dynstr, feedback_name.c_str());

  EXPECT_EQ(site.info, 0x11u);
  EXPECT_EQ(site.section_index, rodata.index);
  EXPECT_EQ(site.value & 7u, 0u);
  ASSERT_GE(site.value, rodata.address);
  ASSERT_LE(site.value - rodata.address, rodata.size);
  ASSERT_LE(site.size, hsaco.size());
  const size_t site_file_offset =
      (size_t)(rodata.offset + (site.value - rodata.address));
  const size_t site_size = (size_t)site.size;
  ASSERT_LE(site_file_offset + site_size, hsaco.size());
  const std::string site_table = hsaco.substr(site_file_offset, site_size);

  ASSERT_GE(site_table.size(),
            LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
                2u * LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(
      LoadLeU32(site_table, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_MAGIC);
  EXPECT_EQ(
      (uint8_t)site_table[LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET],
      LOOM_SANITIZER_SITE_TABLE_VERSION);
  EXPECT_EQ(LoadLeU16(site_table,
                      LOOM_SANITIZER_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(
      LoadLeU32(site_table, LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET),
      2u);

  const size_t record0 = LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH;
  const size_t record1 = record0 + LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH;
  EXPECT_EQ(
      LoadLeU32(site_table,
                record0 + LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
      0u);
  EXPECT_EQ(
      LoadLeU32(site_table,
                record0 + LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
      LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(
      LoadLeU32(site_table,
                record1 + LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
      1u);
  EXPECT_EQ(
      LoadLeU32(site_table,
                record1 + LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
      LOOM_OP_SANITIZER_ASSERT_ACCESS);

  EXPECT_EQ(asan.section_index, data.index);
  EXPECT_EQ(asan.size, LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH);
  EXPECT_EQ(feedback.section_index, data.index);
  EXPECT_EQ(feedback.size,
            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH);

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
