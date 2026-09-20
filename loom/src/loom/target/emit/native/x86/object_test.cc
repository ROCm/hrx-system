// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/object.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/descriptors/low_registry.h"
#include "loom/target/arch/x86/ops/registry.h"
#include "loom/target/function_version.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

iree_status_t RejectDiagnostic(void* user_data,
                               const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  (void)emission;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unexpected x86 object diagnostic");
}

uint16_t LoadLeU16(const uint8_t* data, size_t offset) {
  return (uint16_t)data[offset] | ((uint16_t)data[offset + 1] << 8);
}

uint32_t LoadLeU32(const uint8_t* data, size_t offset) {
  return (uint32_t)data[offset] | ((uint32_t)data[offset + 1] << 8) |
         ((uint32_t)data[offset + 2] << 16) |
         ((uint32_t)data[offset + 3] << 24);
}

uint64_t LoadLeU64(const uint8_t* data, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)data[offset + i] << (i * 8);
  }
  return value;
}

struct ElfSectionView {
  std::string name;
  uint32_t type;
  uint64_t offset;
  uint64_t size;
  uint32_t link;
  uint64_t entry_size;
};

std::vector<ElfSectionView> ParseSections(iree_const_byte_span_t object) {
  const size_t section_table_offset = (size_t)LoadLeU64(object.data, 40);
  const uint16_t section_entry_size = LoadLeU16(object.data, 58);
  const uint16_t section_count = LoadLeU16(object.data, 60);
  const uint16_t string_table_index = LoadLeU16(object.data, 62);
  const size_t string_table_header =
      section_table_offset + (size_t)string_table_index * section_entry_size;
  const size_t string_table_offset =
      (size_t)LoadLeU64(object.data, string_table_header + 24);
  std::vector<ElfSectionView> sections;
  for (uint16_t i = 0; i < section_count; ++i) {
    const size_t header_offset =
        section_table_offset + (size_t)i * section_entry_size;
    const uint32_t name_offset = LoadLeU32(object.data, header_offset);
    const char* name = reinterpret_cast<const char*>(
        object.data + string_table_offset + name_offset);
    sections.push_back(ElfSectionView{
        name,
        LoadLeU32(object.data, header_offset + 4),
        LoadLeU64(object.data, header_offset + 24),
        LoadLeU64(object.data, header_offset + 32),
        LoadLeU32(object.data, header_offset + 40),
        LoadLeU64(object.data, header_offset + 56),
    });
  }
  return sections;
}

size_t FindSection(const std::vector<ElfSectionView>& sections,
                   const char* name) {
  for (size_t i = 0; i < sections.size(); ++i) {
    if (sections[i].name == name) {
      return i;
    }
  }
  ADD_FAILURE() << "missing ELF section " << name;
  return 0;
}

class X86ObjectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_x86_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_x86_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("object_test.loom"), &context_,
                                  &pool_, &options, &module));
    return ModulePtr(module);
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(X86ObjectTest, EmitsDefinitionsImportsAndCallRelocations) {
  ModulePtr module = Parse(R"(
x86.target<scalar> @target

low.func.def target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @helper(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %result = add.gpr64 %value, %value
  return %result
}

low.func.decl import(native, "external_add_one") target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) @external(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>)

low.func.def public target<x86.scalar.core>(@target) abi(object_function) abi_layout({argument_locations = [7], calling_convention = "sysv_x86_64", result_locations = [0], stack_argument_bytes = 0}) export("loom_entry") @caller(%value: reg<x86.gpr64>) -> (reg<x86.gpr64>) asm {
  %doubled = low.func.call @helper(%value) : (reg<x86.gpr64>) -> (reg<x86.gpr64>)
  %result = low.func.call @external(%doubled) : (reg<x86.gpr64>) -> (reg<x86.gpr64>)
  return %result
}
)");

  loom_target_function_version_t function_versions[2] = {};
  loom_function_version_t* version_values[2] = {};
  uint32_t function_index = 0;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(loom_module_block(module.get()), op) {
    if (!loom_low_func_def_isa(op)) {
      continue;
    }
    function_versions[function_index].base.type =
        &loom_target_function_version_type;
    function_versions[function_index].base.function =
        loom_func_like_cast(module.get(), op);
    function_versions[function_index].target_context_ordinal = 0;
    version_values[function_index] = &function_versions[function_index].base;
    ++function_index;
  }
  ASSERT_EQ(function_index, 2u);
  const loom_function_version_list_t versions = {
      version_values,
      IREE_ARRAYSIZE(version_values),
  };
  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&registry_.registry,
      /*.module=*/module.get(),
      /*.function_versions=*/&versions,
      /*.option_chain=*/nullptr,
      /*.identifier=*/{},
      /*.artifact_manifest=*/{},
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/{/*.fn=*/RejectDiagnostic},
      /*.scratch_arena=*/&arena_,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_target_emit_artifact_t artifact = {};
  IREE_ASSERT_OK(loom_x86_elf_object_emitter.emit(&request, &artifact));
  ASSERT_NE(artifact.contents, nullptr);
  EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  iree_const_byte_span_t object = iree_const_byte_span_empty();
  ASSERT_TRUE(
      iree_byte_sequence_try_get_contiguous_span(artifact.contents, &object));
  ASSERT_GT(object.data_length, 64u);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(object.data), 4),
            std::string("\x7F"
                        "ELF",
                        4));
  EXPECT_EQ(LoadLeU16(object.data, 16), 1u);   // ET_REL
  EXPECT_EQ(LoadLeU16(object.data, 18), 62u);  // EM_X86_64

  const std::vector<ElfSectionView> sections = ParseSections(object);
  const ElfSectionView& text = sections[FindSection(sections, ".text")];
  const ElfSectionView& rela_text =
      sections[FindSection(sections, ".rela.text")];
  const ElfSectionView& symbol_table =
      sections[FindSection(sections, ".symtab")];
  ASSERT_EQ(rela_text.entry_size, 24u);
  ASSERT_EQ(rela_text.size, 48u);
  EXPECT_EQ(LoadLeU32(object.data, (size_t)rela_text.offset + 8), 2u);
  EXPECT_EQ(LoadLeU32(object.data, (size_t)rela_text.offset + 24 + 8), 4u);
  EXPECT_EQ((int64_t)LoadLeU64(object.data, (size_t)rela_text.offset + 16), -4);
  EXPECT_EQ((int64_t)LoadLeU64(object.data, (size_t)rela_text.offset + 24 + 16),
            -4);

  const ElfSectionView& symbol_names = sections[symbol_table.link];
  const std::string string_table(
      reinterpret_cast<const char*>(object.data + symbol_names.offset),
      (size_t)symbol_names.size);
  EXPECT_NE(string_table.find("helper"), std::string::npos);
  EXPECT_NE(string_table.find("loom_entry"), std::string::npos);
  EXPECT_NE(string_table.find("external_add_one"), std::string::npos);
  EXPECT_NE(std::find(object.data + text.offset,
                      object.data + text.offset + text.size, 0xE8),
            object.data + text.offset + text.size);
  loom_target_emit_artifact_release(&artifact);
}

}  // namespace
}  // namespace loom
