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
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/target/emit/native/elf.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/function_version.h"
#include "loom/target/profile.h"
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

bool HasTargetCapabilityString(const loom_target_compile_report_t& report,
                               const char* namespace_name, const char* key,
                               const char* value) {
  const iree_string_view_t expected_namespace =
      iree_make_cstring_view(namespace_name);
  const iree_string_view_t expected_key = iree_make_cstring_view(key);
  const iree_string_view_t expected_value = iree_make_cstring_view(value);
  for (const loom_target_compile_report_vec_t* vec =
           report.target_capability_rows.head;
       vec != nullptr; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        static_cast<const loom_target_compile_report_target_capability_row_t*>(
            loom_target_compile_report_vec_const_rows(vec));
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_target_capability_row_t& row = rows[i];
      if (row.value_kind ==
              LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_STRING &&
          iree_string_view_equal(row.namespace_name, expected_namespace) &&
          iree_string_view_equal(row.key, expected_key) &&
          iree_string_view_equal(row.value_string, expected_value)) {
        return true;
      }
    }
  }
  return false;
}

bool HasTargetCapabilityU64(const loom_target_compile_report_t& report,
                            const char* namespace_name, const char* key,
                            uint64_t value) {
  const iree_string_view_t expected_namespace =
      iree_make_cstring_view(namespace_name);
  const iree_string_view_t expected_key = iree_make_cstring_view(key);
  for (const loom_target_compile_report_vec_t* vec =
           report.target_capability_rows.head;
       vec != nullptr; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        static_cast<const loom_target_compile_report_target_capability_row_t*>(
            loom_target_compile_report_vec_const_rows(vec));
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_target_capability_row_t& row = rows[i];
      if (row.value_kind == LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_U64 &&
          iree_string_view_equal(row.namespace_name, expected_namespace) &&
          iree_string_view_equal(row.key, expected_key) &&
          row.value_u64 == value) {
        return true;
      }
    }
  }
  return false;
}

bool HasTargetCapabilityBool(const loom_target_compile_report_t& report,
                             const char* namespace_name,
                             iree_string_view_t expected_key, bool value) {
  const iree_string_view_t expected_namespace =
      iree_make_cstring_view(namespace_name);
  for (const loom_target_compile_report_vec_t* vec =
           report.target_capability_rows.head;
       vec != nullptr; vec = vec->next) {
    const loom_target_compile_report_target_capability_row_t* rows =
        static_cast<const loom_target_compile_report_target_capability_row_t*>(
            loom_target_compile_report_vec_const_rows(vec));
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_target_capability_row_t& row = rows[i];
      if (row.value_kind == LOOM_TARGET_COMPILE_REPORT_CAPABILITY_VALUE_BOOL &&
          iree_string_view_equal(row.namespace_name, expected_namespace) &&
          iree_string_view_equal(row.key, expected_key) &&
          (row.value_u64 != 0) == value) {
        return true;
      }
    }
  }
  return false;
}

bool HasWaitCounter(const loom_target_compile_report_t& report,
                    uint32_t counter_id, const char* counter_name) {
  const iree_string_view_t expected_name = iree_make_cstring_view(counter_name);
  for (const loom_target_compile_report_vec_t* vec =
           report.wait_counter_rows.head;
       vec != nullptr; vec = vec->next) {
    const loom_target_compile_report_wait_counter_row_t* rows =
        static_cast<const loom_target_compile_report_wait_counter_row_t*>(
            loom_target_compile_report_vec_const_rows(vec));
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_wait_counter_row_t& row = rows[i];
      if (row.counter_id == counter_id &&
          iree_string_view_equal(row.counter_name, expected_name) &&
          row.summary.action_count > 0) {
        return true;
      }
    }
  }
  return false;
}

bool HasEntry(const loom_target_compile_report_t& report,
              const char* function_name) {
  const iree_string_view_t expected_name =
      iree_make_cstring_view(function_name);
  for (const loom_target_compile_report_vec_t* vec = report.entry_rows.head;
       vec != nullptr; vec = vec->next) {
    const loom_target_compile_report_entry_t* rows =
        static_cast<const loom_target_compile_report_entry_t*>(
            loom_target_compile_report_vec_const_rows(vec));
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      if (iree_string_view_equal(rows[i].function_name, expected_name)) {
        return true;
      }
    }
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
        "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
        "workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseGfx11DynamicHalKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) @loom_kernel() "
        "{\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseWorkgroupStorageKernel(iree_string_view_t processor_name,
                                   uint64_t byte_length,
                                   loom_module_t** out_module) {
    const loom_amdgpu_target_info_t* target = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_target_info_lookup_target(processor_name, &target));

    std::string source = "amdgpu.target<";
    source.append(processor_name.data, processor_name.size);
    source +=
        "> @gfx_target\n"
        "low.kernel.def target<";
    source.append(target->descriptor_set_key.data,
                  target->descriptor_set_key.size);
    source +=
        ">(@gfx_target) workgroup_size(1, 1, 1) "
        "@loom_kernel() {\n"
        "  %storage = low.storage.reserve {byte_alignment = 16, "
        "byte_length = ";
    source += std::to_string(byte_length);
    source +=
        "} : low.storage<workgroup>\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(ParseSource(
        iree_make_string_view(source.data(), source.size()), out_module));
  }

  void EmitWorkgroupStorageKernel(
      iree_string_view_t processor_name, uint64_t byte_length,
      DiagnosticCapture* capture, bool* out_emitted,
      loom_amdgpu_hal_kernel_library_t* out_library) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        ParseWorkgroupStorageKernel(processor_name, byte_length, &module));

    loom_amdgpu_hal_kernel_library_options_t options = {};
    options.diagnostic_sink = capture->sink();
    options.max_errors = 20;
    iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), out_emitted, out_library);
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  void ParseGfx11KernelWithArguments(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
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

  void ParseGfx1250TensorLoadKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1250> @gfx_target\n"
        "kernel.def target(@gfx_target) @tensor_load() {\n"
        "  %one = index.constant 1 : index\n"
        "  %size = index.constant 64 : index\n"
        "  kernel.launch.config workgroups(%one, %one, %one) "
        "workgroup_size(%size, %one, %one) : index\n"
        "} launch(%input: buffer) {\n"
        "  %zero = index.constant 0 : offset\n"
        "  %bytes = index.constant 16384 : offset\n"
        "  %d0 = vector.constant 0 : vector<4xi32>\n"
        "  %d1 = vector.constant 0 : vector<8xi32>\n"
        "  %descriptor = kernel.tensor.lds.descriptor dgroups(%d0, %d1) : "
        "vector<4xi32>, vector<8xi32> -> kernel.tensor.lds.descriptor\n"
        "  %global = buffer.assume.memory_space<global> %input : buffer\n"
        "  %source = buffer.view %global[%zero] : buffer -> "
        "view<64x64xf32>\n"
        "  %scratch = buffer.alloca<workgroup> align(256) %bytes : buffer\n"
        "  %dest = buffer.view %scratch[%zero] : buffer -> "
        "view<64x64xf32>\n"
        "  %copy = kernel.async.tensor.load.to.lds %source to %dest using "
        "%descriptor {cache_scope = cu, cache_temporal = regular} : "
        "view<64x64xf32> to view<64x64xf32>, "
        "kernel.tensor.lds.descriptor -> kernel.async.token\n"
        "  %group = kernel.async.group %copy : kernel.async.token -> "
        "kernel.async.group\n"
        "  kernel.async.wait %group {newer_groups = 0} : "
        "kernel.async.group\n"
        "  kernel.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseGfx11MultiKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
        "workgroup_size(64, 1, 1) "
        "@first_kernel() {\n"
        "  low.return\n"
        "}\n"
        "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
        "workgroup_size(64, 1, 1) "
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
        "view<1xi32>\n"
        "  sanitizer.assert.access<read> %input_view[0] : "
        "view<1xi32>\n"
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
        "view<1xi32>\n"
        "  sanitizer.assert.access<write> %output_view[0] : "
        "view<1xi32>\n"
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
    loom_compile_pipeline_result_t result = {};
    iree_status_t status =
        loom_compile_run_pipeline(module, &options, &block_pool_, &result);
    const uint32_t error_count = result.pass.error_count;
    loom_compile_pipeline_result_deinitialize(&result);
    IREE_ASSERT_OK(status);
    ASSERT_EQ(error_count, 0u) << DiagnosticSummary(*capture);
  }

  void ParseGfx942Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx942> @gfx_target\n"
        "low.kernel.def target<amdgpu.cdna3.core>(@gfx_target) "
        "workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";
    ASSERT_NO_FATAL_FAILURE(
        ParseSource(iree_make_cstring_view(kSource), out_module));
  }

  void ParseKernelForTarget(
      const loom_amdgpu_target_info_t* target, loom_module_t** out_module,
      iree_string_view_t target_attrs = iree_string_view_empty()) {
    ASSERT_NE(target, nullptr);

    std::string source = "amdgpu.target<";
    source.append(target->name.data, target->name.size);
    source += "> @gfx_target";
    if (!iree_string_view_is_empty(target_attrs)) {
      source += " {";
      source.append(target_attrs.data, target_attrs.size);
      source += "}";
    }
    source +=
        "\n"
        "low.kernel.def target<";
    source.append(target->descriptor_set_key.data,
                  target->descriptor_set_key.size);
    source +=
        ">(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";

    ASSERT_NO_FATAL_FAILURE(ParseSource(
        iree_make_string_view(source.data(), source.size()), out_module))
        << StringViewToString(target->name);
  }

  bool IsDescriptorSetLinked(iree_string_view_t descriptor_set_key) const {
    return loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                               descriptor_set_key) != nullptr;
  }

  bool IsTargetDescriptorSetLinked(
      const loom_amdgpu_target_info_t* target) const {
    return IsDescriptorSetLinked(target->descriptor_set_key);
  }

  void EmitGfx942Kernel(DiagnosticCapture* capture, bool* out_emitted) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx942Kernel(&module));

    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {};
    options.diagnostic_sink = capture->sink();
    options.max_errors = 20;
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

TEST_F(AmdgpuHalKernelLibraryTest,
       EmitsGenericGfx11RepresentationForGfx1151Version) {
  const loom_amdgpu_target_info_t* gfx1151 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_target(IREE_SV("gfx1151"), &gfx1151));
  ASSERT_NE(gfx1151, nullptr);
  if (!IsDescriptorSetLinked(IREE_SV("amdgpu.gfx11.generic.core")) ||
      !IsTargetDescriptorSetLinked(gfx1151)) {
    GTEST_SKIP() << "generic gfx11 and gfx1151 descriptor sets are required";
  }

  static const char kSource[] =
      "amdgpu.target<gfx11-generic> @authored_target\n"
      "low.kernel.def target<amdgpu.gfx11.generic.core>(@authored_target) "
      "workgroup_size(64, 1, 1) @loom_kernel() {\n"
      "  %lane = low.live_in<amdgpu.workitem_id.x> : "
      "reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      ParseSource(iree_make_cstring_view(kSource), &module));

  iree_arena_allocator_t version_arena;
  iree_arena_initialize(&block_pool_, &version_arena);
  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &version_arena);
  const loom_string_id_t function_name_id =
      loom_module_lookup_string(module, IREE_SV("loom_kernel"));
  ASSERT_NE(function_name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t function_symbol_id =
      loom_module_find_symbol(module, function_name_id);
  ASSERT_NE(function_symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_symbol_facts_base_t* function_base_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(
      &symbol_facts, module, function_symbol_id, &function_base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(function_base_facts);
  ASSERT_NE(function_facts, nullptr);
  const loom_symbol_facts_base_t* target_base_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, module, function_facts->target_symbol,
      &target_base_facts));
  const loom_target_symbol_facts_t* authored_target =
      loom_target_symbol_facts_cast(target_base_facts);
  ASSERT_NE(authored_target, nullptr);
  const loom_symbol_ref_t authored_target_symbol = authored_target->symbol;

  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(gfx1151, &identity);
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&identity, &target_profile));
  loom_target_facts_t* exact_facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      &target_profile.base, &version_arena, &exact_facts));
  ASSERT_TRUE(loom_target_facts_satisfy_specialization_requirement(
      exact_facts, authored_target->projection));
  loom_target_facts_builder_apply_requirement(authored_target->projection,
                                              exact_facts);

  bool contract_valid = false;
  const loom_target_facts_t* effective_facts = nullptr;
  IREE_ASSERT_OK(loom_target_function_contract_refine_facts(
      module, function_facts, authored_target->name, exact_facts,
      iree_diagnostic_emitter_t{}, &version_arena, &contract_valid,
      &effective_facts));
  ASSERT_TRUE(contract_valid);
  ASSERT_NE(effective_facts, nullptr);
  loom_target_function_version_t function_version = {};
  function_version.base.type = &loom_target_function_version_type;
  function_version.base.function = loom_func_like_cast(
      module, module->symbols.entries[function_symbol_id].defining_op);
  ASSERT_TRUE(loom_func_like_isa(function_version.base.function));
  function_version.authored_target_name = authored_target->name;
  function_version.target_requirement_facts = authored_target->projection;
  function_version.effective_target_facts = effective_facts;
  loom_function_version_t* version_values[] = {
      &function_version.base,
  };
  loom_function_version_list_t function_versions = {};
  function_versions.values = version_values;
  function_versions.count = IREE_ARRAYSIZE(version_values);

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.function_versions = &function_versions;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  options.report = &report;
  bool emitted = false;
  loom_amdgpu_hal_kernel_library_t library = {};
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
  char expected_target_key_storage[128] = {};
  iree_string_view_t expected_target_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_amdgpu_artifact_target_key_format(
      &identity, sizeof(expected_target_key_storage),
      expected_target_key_storage, &expected_target_key));
  EXPECT_TRUE(iree_string_view_equal(library.target_key, expected_target_key))
      << "actual: " << StringViewToString(library.target_key)
      << ", expected: " << StringViewToString(expected_target_key);
  EXPECT_TRUE(
      HasTargetCapabilityString(report, "amdgpu", "processor", "gfx1151"));
  EXPECT_TRUE(HasTargetCapabilityString(report, "amdgpu", "descriptor_set",
                                        "amdgpu.rdna3_5.core"));
  loom_symbol_fact_table_t retained_symbol_facts = {};
  loom_symbol_fact_table_initialize(&retained_symbol_facts, &version_arena);
  const loom_symbol_facts_base_t* retained_target_base_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup_ref(
      &retained_symbol_facts, module, authored_target_symbol,
      &retained_target_base_facts));
  const loom_target_symbol_facts_t* retained_target =
      loom_target_symbol_facts_cast(retained_target_base_facts);
  ASSERT_NE(retained_target, nullptr);
  ASSERT_NE(retained_target->projection, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      loom_target_facts_identity_name(retained_target->projection),
      IREE_SV("gfx11-generic")));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
  iree_arena_deinitialize(&version_arena);
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, RejectsIncompatibleRepresentationContract) {
  const loom_amdgpu_target_info_t* gfx1151 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_target(IREE_SV("gfx1151"), &gfx1151));
  ASSERT_NE(gfx1151, nullptr);
  if (!IsDescriptorSetLinked(IREE_SV("amdgpu.rdna3.core")) ||
      !IsTargetDescriptorSetLinked(gfx1151)) {
    GTEST_SKIP() << "RDNA3 and gfx1151 descriptor sets are required";
  }

  static const char kSource[] =
      "amdgpu.target<gfx1151> @gfx_target\n"
      "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
      "workgroup_size(64, 1, 1) @loom_kernel() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      ParseSource(iree_make_cstring_view(kSource), &module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  bool emitted = true;
  loom_amdgpu_hal_kernel_library_t library = {};
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u) << DiagnosticSummary(capture);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_TARGET_065);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "loom_kernel");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.rdna3.core");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "gfx1151");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "amdgpu.rdna3_5.core");

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsGfx1250HardwareEntryEnvelope) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_target(IREE_SV("gfx1250"), &target));
  ASSERT_NE(target, nullptr);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  ASSERT_NE(processor, nullptr);
  if (!IsTargetDescriptorSetLinked(target)) {
    GTEST_SKIP() << "amdgpu.rdna4.gfx125x.core is not linked in this build";
  }
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseKernelForTarget(target, &module));

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  options.report = &report;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
  ASSERT_NE(library.hsaco_data, nullptr);
  const std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                          library.hsaco_data_length);
  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynamic_symbol_table = FindSection(sections, ".dynsym");
  const Section& dynamic_string_table = FindSection(sections, ".dynstr");
  const Section& text = FindSection(sections, ".text");
  const DynamicSymbol entry = FindDynamicSymbol(
      hsaco, dynamic_symbol_table, dynamic_string_table, "loom_kernel");
  static const uint8_t kExpectedEntryText[] = {
      0x00, 0x40, 0x17, 0xee, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x7e, 0x41, 0x06, 0x80, 0xb9,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xbf,
  };
  ASSERT_EQ(entry.section_index, text.index);
  ASSERT_GE(entry.value, text.address);
  ASSERT_LE(entry.value - text.address, text.size);
  ASSERT_EQ(entry.size, sizeof(kExpectedEntryText));
  const size_t entry_file_offset =
      (size_t)(text.offset + (entry.value - text.address));
  ASSERT_LE(entry_file_offset + sizeof(kExpectedEntryText), hsaco.size());
  EXPECT_EQ(hsaco.substr(entry_file_offset, sizeof(kExpectedEntryText)),
            std::string(reinterpret_cast<const char*>(kExpectedEntryText),
                        sizeof(kExpectedEntryText)));
  EXPECT_EQ(report.entry_rows.count, 1u);
  EXPECT_TRUE(HasEntry(report, "loom_kernel"));
  EXPECT_EQ(report.emitted_instruction_count,
            report.emission_breakdown.body_instruction_count +
                report.emission_breakdown.entry_instruction_count);
  EXPECT_EQ(report.emission_breakdown.entry_instruction_count, 3u);
  EXPECT_EQ(report.target_resources.subgroup_size, 32u);
  EXPECT_EQ(report.target_resources.max_subgroups_per_simd, 16u);
  EXPECT_EQ(report.target_resources.resident_subgroups_per_simd, 8u);
  EXPECT_EQ(report.target_resources.occupancy_percent, 50u);
  EXPECT_TRUE(iree_string_view_equal(report.target_resources.limiting_resource,
                                     IREE_SV("amdgpu.workgroup_slots")));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
  loom_module_free(module);
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
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  options.artifact_manifest = artifact_manifest_options;
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
  EXPECT_EQ(hsaco.find(".cluster_dims"), std::string::npos);

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

TEST_F(AmdgpuHalKernelLibraryTest,
       RejectsGfx11FinalWorkgroupStorageAboveTargetLimit) {
  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(EmitWorkgroupStorageKernel(
      IREE_SV("gfx1100"), 65540, &capture, &emitted, &library));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u) << DiagnosticSummary(capture);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_TARGET_051);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "loom_kernel");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "gfx_target");
  ASSERT_EQ(diagnostic->params.size(), 4u);
  ASSERT_EQ(diagnostic->params[2].kind, LOOM_PARAM_U64);
  EXPECT_EQ(diagnostic->params[2].u64, 65540u);
  ASSERT_EQ(diagnostic->params[3].kind, LOOM_PARAM_U64);
  EXPECT_EQ(diagnostic->params[3].u64, 65536u);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
}

TEST_F(AmdgpuHalKernelLibraryTest, AcceptsGfx125xLargeWorkgroupStorage) {
  static constexpr const char* kProcessors[] = {"gfx1250", "gfx1251"};
  for (const char* processor_name : kProcessors) {
    SCOPED_TRACE(processor_name);
    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    bool emitted = false;
    ASSERT_NO_FATAL_FAILURE(
        EmitWorkgroupStorageKernel(iree_make_cstring_view(processor_name),
                                   114688, &capture, &emitted, &library));

    EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
    EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
    EXPECT_NE(library.hsaco_data, nullptr);
    EXPECT_NE(iree_string_view_find(library.target_key,
                                    iree_make_cstring_view(processor_name), 0),
              IREE_STRING_VIEW_NPOS);

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
  }
}

TEST_F(AmdgpuHalKernelLibraryTest,
       RejectsGfx125xWorkgroupStorageAboveTargetLimit) {
  static constexpr const char* kProcessors[] = {"gfx1250", "gfx1251"};
  for (const char* processor_name : kProcessors) {
    SCOPED_TRACE(processor_name);
    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    bool emitted = true;
    ASSERT_NO_FATAL_FAILURE(
        EmitWorkgroupStorageKernel(iree_make_cstring_view(processor_name),
                                   327681, &capture, &emitted, &library));

    EXPECT_FALSE(emitted);
    ASSERT_EQ(capture.diagnostics.size(), 1u) << DiagnosticSummary(capture);
    const CapturedDiagnostic* diagnostic =
        FindDiagnostic(capture, LOOM_ERR_TARGET_051);
    ASSERT_NE(diagnostic, nullptr);
    EXPECT_EQ(GetStringParam(*diagnostic, 0), "loom_kernel");
    EXPECT_EQ(GetStringParam(*diagnostic, 1), "gfx_target");
    ASSERT_EQ(diagnostic->params.size(), 4u);
    ASSERT_EQ(diagnostic->params[2].kind, LOOM_PARAM_U64);
    EXPECT_EQ(diagnostic->params[2].u64, 327681u);
    ASSERT_EQ(diagnostic->params[3].kind, LOOM_PARAM_U64);
    EXPECT_EQ(diagnostic->params[3].u64, 327680u);

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
  }
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsEveryLinkedCanonicalTarget) {
  iree_host_size_t linked_supported_count = 0;
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr);
    ASSERT_TRUE(
        loom_amdgpu_processor_properties_support_hsaco(&processor->properties));
    if (!IsTargetDescriptorSetLinked(target)) {
      continue;
    }
    ++linked_supported_count;

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForTarget(target, &module));

    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {};
    options.diagnostic_sink = capture.sink();
    options.max_errors = 20;
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library))
        << StringViewToString(target->name);

    EXPECT_TRUE(emitted) << StringViewToString(target->name);
    EXPECT_TRUE(capture.diagnostics.empty())
        << StringViewToString(target->name);
    EXPECT_NE(library.hsaco_data, nullptr) << StringViewToString(target->name);
    EXPECT_GT(library.hsaco_data_length, 64u)
        << StringViewToString(target->name);
    if (library.hsaco_data_length > 64u) {
      EXPECT_EQ(LoadLeU32(library.hsaco_data, 48),
                processor->properties.elf.machine_flags |
                    processor->properties.elf.feature_flags |
                    (processor->properties.elf.generic_version
                     << LOOM_AMDGPU_ELF_GENERIC_VERSION_OFFSET_V6))
          << StringViewToString(target->name);
    }
    EXPECT_NE(iree_string_view_find(library.target_key, target->name, 0),
              IREE_STRING_VIEW_NPOS)
        << StringViewToString(target->name);
    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(target, &identity);
    iree_arena_allocator_t target_id_arena;
    iree_arena_initialize(&block_pool_, &target_id_arena);
    iree_string_view_t expected_target_id = iree_string_view_empty();
    IREE_ASSERT_OK(loom_amdgpu_artifact_target_key_format_arena(
        &identity, &target_id_arena, &expected_target_id));
    EXPECT_TRUE(iree_string_view_equal(library.target_key, expected_target_id))
        << StringViewToString(target->name);
    iree_arena_deinitialize(&target_id_arena);
    std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                      library.hsaco_data_length);
    EXPECT_NE(hsaco.find("loom_kernel.kd"), std::string::npos)
        << StringViewToString(target->name);

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
  }
  EXPECT_GE(linked_supported_count, 1u);
}

TEST_F(AmdgpuHalKernelLibraryTest,
       PreservesStructuredIdentityAcrossArtifactSurfaces) {
  struct TargetCase {
    // Canonical target represented across every artifact surface.
    const loom_amdgpu_target_info_t* target;
    // Target-record attributes selecting AMDHSA feature coordinates.
    std::string target_attrs;
    // Exact canonical feature array expected in the artifact manifest.
    std::string features_json;
  };
  std::vector<TargetCase> cases;
  const loom_amdgpu_target_info_t* gfx942 = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_target(IREE_SV("gfx942"), &gfx942));
  cases.push_back({
      /*.target=*/gfx942,
      /*.target_attrs=*/"sramecc = on, xnack = off",
      /*.features_json=*/"\"features\":[\"sramecc+\",\"xnack-\"]",
  });

  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t target_ordinal = 0; target_ordinal < target_count;
       ++target_ordinal) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_target_at(target_ordinal);
    ASSERT_NE(target, nullptr);
    bool shares_processor = false;
    for (iree_host_size_t other_ordinal = 0; other_ordinal < target_count;
         ++other_ordinal) {
      const loom_amdgpu_target_info_t* other =
          loom_amdgpu_target_info_target_at(other_ordinal);
      ASSERT_NE(other, nullptr);
      if (target != other &&
          target->processor_ordinal == other->processor_ordinal) {
        shares_processor = true;
        break;
      }
    }
    if (shares_processor) {
      cases.push_back({
          /*.target=*/target,
          /*.target_attrs=*/{},
          /*.features_json=*/{},
      });
    }
  }

  iree_host_size_t exercised_count = 0;
  iree_host_size_t exercised_overlay_count = 0;
  for (const TargetCase& test_case : cases) {
    SCOPED_TRACE(StringViewToString(test_case.target->name));
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(test_case.target);
    ASSERT_NE(processor, nullptr);
    if (!IsTargetDescriptorSetLinked(test_case.target)) {
      continue;
    }
    ++exercised_count;

    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(test_case.target, &identity);
    if (test_case.target == gfx942) {
      identity.amdhsa_features.sramecc = LOOM_AMDGPU_TARGET_FEATURE_ON;
      identity.amdhsa_features.xnack = LOOM_AMDGPU_TARGET_FEATURE_OFF;
    }
    char artifact_target_key_storage[128] = {};
    iree_string_view_t artifact_target_key = iree_string_view_empty();
    IREE_ASSERT_OK(loom_amdgpu_artifact_target_key_format(
        &identity, sizeof(artifact_target_key_storage),
        artifact_target_key_storage, &artifact_target_key));
    iree_arena_allocator_t target_id_arena;
    iree_arena_initialize(&block_pool_, &target_id_arena);
    iree_string_view_t code_object_target_view = iree_string_view_empty();
    IREE_ASSERT_OK(loom_amdgpu_amdhsa_code_object_target_id_format(
        &identity, &target_id_arena, &code_object_target_view));
    const std::string code_object_target =
        StringViewToString(code_object_target_view);

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForTarget(
        test_case.target, &module,
        iree_make_string_view(test_case.target_attrs.data(),
                              test_case.target_attrs.size())));
    loom_target_artifact_manifest_collect_options_t manifest_options = {};
    loom_target_artifact_manifest_collect_options_initialize(&manifest_options);
    manifest_options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {};
    options.diagnostic_sink = capture.sink();
    options.max_errors = 20;
    options.artifact_manifest = manifest_options;
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library));
    EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
    EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
    EXPECT_TRUE(
        iree_string_view_equal(library.target_key, artifact_target_key));

    ASSERT_GE(library.hsaco_data_length, 64u);
    loom_amdgpu_amdhsa_target_id_t parsed_target_id = {};
    IREE_ASSERT_OK(loom_amdgpu_target_info_parse_amdhsa_target_id(
        code_object_target_view, &parsed_target_id));
    uint32_t expected_elf_flags = 0;
    IREE_ASSERT_OK(loom_amdgpu_target_info_amdhsa_target_id_elf_flags(
        &parsed_target_id, &expected_elf_flags));
    EXPECT_EQ(LoadLeU32(library.hsaco_data, 48), expected_elf_flags);

    const std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                            library.hsaco_data_length);
    EXPECT_NE(hsaco.find(code_object_target), std::string::npos);

    ASSERT_NE(library.artifact_manifest.contents.data, nullptr);
    const std::string manifest(
        reinterpret_cast<const char*>(library.artifact_manifest.contents.data),
        library.artifact_manifest.contents.data_length);
    EXPECT_NE(manifest.find("\"family\":\"amdgpu\""), std::string::npos)
        << manifest;
    EXPECT_NE(manifest.find(std::string("\"selector\":\"") +
                            StringViewToString(test_case.target->name) + "\""),
              std::string::npos)
        << manifest;
    EXPECT_NE(manifest.find(std::string("\"processor\":\"") +
                            StringViewToString(processor->name) + "\""),
              std::string::npos)
        << manifest;
    EXPECT_NE(manifest.find(std::string("\"code_object_target\":\"") +
                            code_object_target + "\""),
              std::string::npos)
        << manifest;
    if (!test_case.features_json.empty()) {
      EXPECT_NE(manifest.find(test_case.features_json), std::string::npos)
          << manifest;
    }
    if (!iree_string_view_equal(test_case.target->name, processor->name)) {
      ++exercised_overlay_count;
    }

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
    iree_arena_deinitialize(&target_id_arena);
  }
  EXPECT_GE(exercised_count, 1u);
  EXPECT_GE(exercised_overlay_count, 1u);
}

TEST_F(AmdgpuHalKernelLibraryTest, RecordsMatrixFeatureCapabilities) {
  struct MatrixFeatureReportCase {
    // Concrete AMDGPU processor to compile for.
    const char* processor_name;
    // Matrix feature profile expected for |processor_name|.
    const char* profile_name;
    // Native FP8/BF8 support kind expected for |processor_name|.
    const char* fp8_bf8_native_kind;
  };
  static constexpr MatrixFeatureReportCase kCases[] = {
      {"gfx942", "mfma-gfx940", "unscaled"},
      {"gfx950", "mfma-gfx950", "unscaled_scaled"},
      {"gfx1100", "wmma-gfx11", "none"},
      {"gfx1170", "wmma-gfx12", "unscaled"},
      {"gfx1200", "wmma-gfx12", "unscaled"},
      {"gfx1250", "wmma-gfx1250", "unscaled_scaled"},
  };

  iree_host_size_t checked_count = 0;
  for (const MatrixFeatureReportCase& test_case : kCases) {
    const loom_amdgpu_target_info_t* target =
        loom_amdgpu_target_info_find_target(
            iree_make_cstring_view(test_case.processor_name));
    ASSERT_NE(target, nullptr) << test_case.processor_name;
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target);
    ASSERT_NE(processor, nullptr) << test_case.processor_name;
    if (!loom_amdgpu_processor_properties_support_hsaco(
            &processor->properties) ||
        !IsTargetDescriptorSetLinked(target)) {
      continue;
    }
    ++checked_count;

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForTarget(target, &module));

    loom_target_compile_report_t report = {};
    loom_target_compile_report_initialize(&report, iree_allocator_system());
    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {};
    options.diagnostic_sink = capture.sink();
    options.max_errors = 20;
    options.report = &report;
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library))
        << test_case.processor_name;

    EXPECT_TRUE(emitted) << test_case.processor_name;
    EXPECT_TRUE(capture.diagnostics.empty()) << test_case.processor_name;
    EXPECT_TRUE(HasTargetCapabilityString(
        report, "amdgpu", "matrix_feature_profile", test_case.profile_name))
        << test_case.processor_name;
    EXPECT_TRUE(HasTargetCapabilityString(report, "amdgpu",
                                          "matrix_fp8_native_kind",
                                          test_case.fp8_bf8_native_kind))
        << test_case.processor_name;
    EXPECT_TRUE(HasTargetCapabilityString(report, "amdgpu",
                                          "matrix_bf8_native_kind",
                                          test_case.fp8_bf8_native_kind))
        << test_case.processor_name;
    loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
    ASSERT_TRUE(loom_amdgpu_matrix_feature_bits_from_profile(
        processor->properties.features.matrix, &feature_bits))
        << test_case.processor_name;
    EXPECT_TRUE(HasTargetCapabilityU64(report, "amdgpu", "matrix_feature_bits",
                                       feature_bits))
        << test_case.processor_name;
    for (iree_host_size_t i = 0; i < loom_amdgpu_matrix_feature_info_count();
         ++i) {
      const loom_amdgpu_matrix_feature_info_t* feature_info =
          loom_amdgpu_matrix_feature_info_at(i);
      ASSERT_NE(feature_info, nullptr);
      if (!iree_all_bits_set(feature_bits, feature_info->feature_bit)) {
        continue;
      }
      EXPECT_TRUE(HasTargetCapabilityBool(report, "amdgpu.matrix_feature",
                                          feature_info->name, true))
          << test_case.processor_name << ": "
          << StringViewToString(feature_info->name);
    }
    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_target_compile_report_deinitialize(&report);
    loom_module_free(module);
  }
  EXPECT_GE(checked_count, 1u);
}

TEST_F(AmdgpuHalKernelLibraryTest, RecordsTensorWaitCounter) {
  if (!IsDescriptorSetLinked(IREE_SV("amdgpu.rdna4.gfx125x.core"))) {
    GTEST_SKIP() << "amdgpu.rdna4.gfx125x.core is not linked in this build";
  }

  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx1250TensorLoadKernel(&module));
  DiagnosticCapture capture;
  ASSERT_NO_FATAL_FAILURE(RunPreparedLowPipeline(module, &capture));

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  options.report = &report;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
  EXPECT_TRUE(
      HasWaitCounter(report, LOOM_AMDGPU_WAIT_COUNTER_TENSOR, "tensor"));
  EXPECT_GT(report.target_insertion_summary.static_packet_count, 0u);
  EXPECT_GT(report.target_insertion_rows.count, 0u);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsArgumentMetadataFromLowKernelAbi) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11KernelWithArguments(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
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

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  options.report = &report;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
  ASSERT_NE(library.hsaco_data, nullptr);
  std::string hsaco(reinterpret_cast<const char*>(library.hsaco_data),
                    library.hsaco_data_length);
  const std::vector<Section> sections = ReadSections(hsaco);
  const Section& dynamic_symbol_table = FindSection(sections, ".dynsym");
  const Section& dynamic_string_table = FindSection(sections, ".dynstr");
  EXPECT_GT(FindDynamicSymbol(hsaco, dynamic_symbol_table, dynamic_string_table,
                              "first_kernel")
                .size,
            0u);
  EXPECT_GT(FindDynamicSymbol(hsaco, dynamic_symbol_table, dynamic_string_table,
                              "second_kernel")
                .size,
            0u);
  EXPECT_EQ(report.entry_rows.count, 2u);
  EXPECT_TRUE(HasEntry(report, "first_kernel"));
  EXPECT_TRUE(HasEntry(report, "second_kernel"));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsRequestedRuntimeGlobals) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
                            LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG |
                            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
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
       EmitsGfx1250CallerDataSymbolsAndRel32AddressMaterialization) {
  static constexpr char kSiteSymbolName[] = "loom_sanitizer_sites";
  static const uint8_t kSiteRecords[] = {
      0x00, 0x02, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00,
      0x00, 0x01, 0x01, 0x06, 0x01, 0x01, 0x00, 0x00,
  };
  static const char kSource[] =
      "amdgpu.target<gfx1250> @gfx_target\n"
      "global.rodata.decl @loom_sanitizer_sites\n"
      "low.kernel.def target<amdgpu.gfx12_5.generic.core>(@gfx_target) "
      "workgroup_size(64, 1, 1) "
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
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  options.data_symbols = &site_symbol;
  options.data_symbol_count = 1;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted) << DiagnosticSummary(capture);
  EXPECT_TRUE(capture.diagnostics.empty()) << DiagnosticSummary(capture);
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

  const uint64_t base_pc_address = text.address + 28u;
  const uint64_t site_delta = site.value + 8u - base_pc_address;
  ASSERT_LE(text.offset + 44u, hsaco.size());
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 32u), (uint32_t)site_delta);
  EXPECT_EQ(LoadLeU32(hsaco, (size_t)text.offset + 40u),
            (uint32_t)(site_delta >> 32));

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
}

TEST_F(AmdgpuHalKernelLibraryTest, RejectsRel32AddWithoutPcProvenance) {
  static const char kSource[] =
      "amdgpu.target<gfx11-generic> @gfx_target\n"
      "global.rodata.decl @iree_feedback_config\n"
      "low.kernel.def target<amdgpu.gfx11.generic.core>(@gfx_target) "
      "workgroup_size(64, 1, 1) "
      "@loom_kernel() {\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  %address = low.op<amdgpu.s_add_u32.rhs_symbol_rel32_lo>(%zero) "
      "{symbol = @iree_feedback_config, byte_offset = 16} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      ParseSource(iree_make_cstring_view(kSource), &module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
  bool emitted = false;
  iree::Status status(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("does not hold an s_getpc_b64 component"));
  EXPECT_FALSE(emitted);
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
      "global.rodata.def @loom_sanitizer_sites = "
      "align(16) bytes(\"00020302010100000001010601010000\")\n"
      "amdgpu.target<gfx1100> @gfx_target\n"
      "low.kernel.def target<amdgpu.rdna3.core>(@gfx_target) "
      "workgroup_size(64, 1, 1) "
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
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
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
  loom_amdgpu_hal_kernel_library_options_t options = {};
  options.runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
                            LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  options.diagnostic_sink = capture.sink();
  options.max_errors = 20;
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

}  // namespace
}  // namespace loom
