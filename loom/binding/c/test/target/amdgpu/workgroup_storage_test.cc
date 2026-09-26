// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loom/binding/c/test/testdata/workgroup_storage_testdata.h"
#include "loomc/artifact.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/launch_config.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using PassPtr = HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using TargetPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using LaunchPtr = HandlePtr<loomc_launch_config_program_t,
                            loomc_launch_config_program_release>;

void ExpectSuccess(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    EXPECT_TRUE(loomc_result_succeeded(result))
        << std::string(diagnostic->message.data, diagnostic->message.size);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const auto* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind == kind) {
      return artifact;
    }
  }
  return nullptr;
}

uint64_t ReadLittleEndian(const std::string& bytes, uint64_t offset,
                          size_t width) {
  uint64_t value = 0;
  for (size_t i = 0; i < width; ++i) {
    value |= uint64_t(uint8_t(bytes[size_t(offset) + i])) << (8 * i);
  }
  return value;
}

// Inspect the emitted ELF64LE descriptor through its ordinary symbol/section
// tables. All read ranges are checked here; this needs no runtime or SDK
// loader.
void ExpectFixedStorage(const std::string& bytes, const char* symbol,
                        uint32_t expected_bytes) {
  ASSERT_GE(bytes.size(), 64u);
  ASSERT_EQ(bytes.compare(0, 4, "\177ELF"), 0);
  ASSERT_EQ(uint8_t(bytes[4]), 2u);
  ASSERT_EQ(uint8_t(bytes[5]), 1u);
  const uint64_t sections = ReadLittleEndian(bytes, 40, 8);
  const uint64_t section_count = ReadLittleEndian(bytes, 60, 2);
  ASSERT_EQ(ReadLittleEndian(bytes, 58, 2), 64u);
  ASSERT_LE(sections, bytes.size());
  ASSERT_LE(section_count, (bytes.size() - sections) / 64);
  const size_t symbol_length = std::strlen(symbol) + 1;
  for (uint64_t i = 1; i < section_count; ++i) {
    const uint64_t section = sections + 64 * i;
    const uint64_t type = ReadLittleEndian(bytes, section + 4, 4);
    if (type != 2 && type != 11) {
      continue;  // SHT_SYMTAB or SHT_DYNSYM.
    }
    const uint64_t table = ReadLittleEndian(bytes, section + 24, 8);
    const uint64_t table_bytes = ReadLittleEndian(bytes, section + 32, 8);
    const uint64_t string_section = ReadLittleEndian(bytes, section + 40, 4);
    ASSERT_EQ(ReadLittleEndian(bytes, section + 56, 8), 24u);
    ASSERT_LE(table, bytes.size());
    ASSERT_LE(table_bytes, bytes.size() - table);
    ASSERT_EQ(table_bytes % 24, 0u);
    ASSERT_LT(string_section, section_count);
    const uint64_t strings =
        ReadLittleEndian(bytes, sections + 64 * string_section + 24, 8);
    const uint64_t string_bytes =
        ReadLittleEndian(bytes, sections + 64 * string_section + 32, 8);
    ASSERT_LE(strings, bytes.size());
    ASSERT_LE(string_bytes, bytes.size() - strings);
    for (uint64_t entry = table; entry < table + table_bytes; entry += 24) {
      const uint64_t name = ReadLittleEndian(bytes, entry, 4);
      ASSERT_LT(name, string_bytes);
      if (symbol_length > string_bytes - name ||
          bytes.compare(size_t(strings + name), symbol_length, symbol,
                        symbol_length) != 0) {
        continue;
      }
      const uint64_t definition = ReadLittleEndian(bytes, entry + 6, 2);
      const uint64_t address = ReadLittleEndian(bytes, entry + 8, 8);
      ASSERT_GT(definition, 0u);
      ASSERT_LT(definition, section_count);
      ASSERT_EQ(ReadLittleEndian(bytes, entry + 16, 8), 64u);
      const uint64_t header = sections + 64 * definition;
      const uint64_t base = ReadLittleEndian(bytes, header + 16, 8);
      const uint64_t contents = ReadLittleEndian(bytes, header + 24, 8);
      const uint64_t content_bytes = ReadLittleEndian(bytes, header + 32, 8);
      ASSERT_LE(contents, bytes.size());
      ASSERT_LE(content_bytes, bytes.size() - contents);
      ASSERT_GE(address, base);
      const uint64_t relative = address - base;
      ASSERT_LE(relative, content_bytes);
      ASSERT_GE(content_bytes - relative, 64u);
      EXPECT_EQ(ReadLittleEndian(bytes, contents + relative, 4),
                expected_bytes);
      return;
    }
  }
  FAIL() << "Kernel descriptor symbol not found: " << symbol;
}

class WorkgroupStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_target_environment_t* target = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &target));
    target_.reset(target);
    loomc_context_target_options_t target_options = {};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size = sizeof(target_options);
    target_options.target_environment = target;
    loomc_context_options_t context_options = {};
    context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size = sizeof(context_options);
    context_options.next = &target_options;
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                         loomc_allocator_system(), &context));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);
    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context, nullptr,
                                          loomc_allocator_system(), &compiler));
    compiler_.reset(compiler);
    loomc_target_pipeline_options_t pipeline = {};
    pipeline.type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS;
    pipeline.structure_size = sizeof(pipeline);
    pipeline.kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
    pipeline.control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
    loomc_pass_program_t* passes = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
        context, &pipeline, loomc_allocator_system(), &passes, &result));
    ResultPtr result_owner(result);
    ExpectSuccess(result);
    passes_.reset(passes);
    LoadFixture("launch.loom");
  }

  void LoadFixture(const char* name) {
    const auto* files = loomc_workgroup_storage_testdata_create();
    const auto file_count = loomc_workgroup_storage_testdata_size();
    const auto* file = files;
    for (; file != files + file_count; ++file) {
      if (std::strcmp(file->name, name) == 0) {
        break;
      }
    }
    ASSERT_NE(file, files + file_count) << name;
    loomc_source_options_t source_options = {};
    source_options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    source_options.structure_size = sizeof(source_options);
    source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    source_options.identifier = loomc_make_cstring_view(file->name);
    source_options.contents = loomc_make_byte_span(file->data, file->size);
    source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    SourcePtr source_owner(source);
    module_ = Parse(source);
  }

  ModulePtr Parse(const loomc_source_t* source) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result));
    ResultPtr result_owner(result);
    ExpectSuccess(result);
    return ModulePtr(module);
  }

  ResultPtr Compile(loomc_compile_artifact_flags_t flags) {
    loomc_compile_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS;
    options.structure_size = sizeof(options);
    options.artifact_flags = flags;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_compile_module(compiler_.get(), workspace_.get(),
                                         passes_.get(), module_.get(), &options,
                                         loomc_allocator_system(), &result));
    ResultPtr owner(result);
    ExpectSuccess(result);
    return owner;
  }

  LaunchPtr LoadLaunch(const loomc_result_t* result) {
    const auto* artifact =
        FindArtifact(result, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG);
    EXPECT_NE(artifact, nullptr);
    if (artifact == nullptr) {
      return {};
    }
    loomc_launch_config_program_t* program = nullptr;
    LOOMC_EXPECT_OK(loomc_launch_config_program_load(
        artifact, loomc_allocator_system(), &program));
    return LaunchPtr(program);
  }

  void ExpectTotal(loomc_launch_config_program_t* program, const char* name,
                   uint64_t dynamic_bytes, uint64_t expected_total) {
    auto function = loomc_launch_config_function_invalid();
    LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
        program, loomc_make_cstring_view(name), &function));
    loomc_launch_config_t config = {};
    config.type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG;
    config.structure_size = sizeof(config);
    LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(
        program, function, &dynamic_bytes, 1, &config));
    EXPECT_EQ(config.workgroup_storage_bytes, expected_total);
    EXPECT_EQ(config.workgroup_size.x, 64u);
  }

  std::string Emit(loomc_module_t* module) {
    loomc_emit_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    options.structure_size = sizeof(options);
    options.artifact_format =
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    options.artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_emit_module(target_.get(), workspace_.get(), module,
                                      &options, loomc_allocator_system(),
                                      &result));
    ResultPtr result_owner(result);
    ExpectSuccess(result);
    const auto* artifact = FindArtifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE);
    EXPECT_NE(artifact, nullptr);
    if (artifact == nullptr) {
      return {};
    }
    loomc_byte_span_t bytes = {};
    LOOMC_EXPECT_OK(loomc_byte_sequence_clone(
        artifact->contents, loomc_allocator_system(), &bytes));
    std::string contents(reinterpret_cast<const char*>(bytes.data),
                         bytes.data_length);
    loomc_allocator_free(loomc_allocator_system(), (void*)bytes.data);
    return contents;
  }

  void ExpectSameEmission(loomc_module_t* module, const std::string& expected,
                          const char* context) {
    const std::string actual = Emit(module);
    EXPECT_TRUE(actual == expected)
        << context << ": actual " << actual.size() << " bytes, expected "
        << expected.size() << " bytes";
  }

  // Public owners shared by workflow operations, released in dependency order.
  TargetPtr target_;
  // Context admitting source and target semantics.
  ContextPtr context_;
  // Workspace backing the module and ordinary compilation scratch.
  WorkspacePtr workspace_;
  // Compiler invoked again on the already prepared module.
  CompilerPtr compiler_;
  // Ordinary target pipeline used by both initial and continuing compilation.
  PassPtr passes_;
  // Mutable module carrying durable proofs and the independent layout snapshot.
  ModulePtr module_;
};

TEST_F(WorkgroupStorageTest, CompanionTotalsAreCheckedAndOwnTheirStorage) {
  ResultPtr compiled = Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
  ASSERT_TRUE(loomc_result_succeeded(compiled.get()));
  LaunchPtr launch = LoadLaunch(compiled.get());
  ASSERT_NE(launch, nullptr);
  compiled.reset();
  loomc_workspace_trim(workspace_.get());
  ExpectTotal(launch.get(), "padded_tail", 0, 64);
  ExpectTotal(launch.get(), "padded_tail", 256, 320);
  ExpectTotal(launch.get(), "capacity_only", 0, 0);
  ExpectTotal(launch.get(), "capacity_only", 256, 256);
  auto function = loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      launch.get(), loomc_make_cstring_view("padded_tail"), &function));
  const uint64_t overflowing = INT64_MAX;
  loomc_launch_config_t config = {};
  config.type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG;
  config.structure_size = sizeof(config);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_launch_config_program_invoke(
                             launch.get(), function, &overflowing, 1, &config));
  const uint64_t negative = UINT64_MAX;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE,
                         loomc_launch_config_program_invoke(
                             launch.get(), function, &negative, 1, &config));
}

TEST_F(WorkgroupStorageTest,
       LayoutSurvivesResultReleaseAndNativeReconstruction) {
  ResultPtr compiled = Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
  ASSERT_TRUE(loomc_result_succeeded(compiled.get()));
  LaunchPtr launch = LoadLaunch(compiled.get());
  ASSERT_NE(launch, nullptr);
  compiled.reset();
  launch.reset();
  loomc_workspace_trim(workspace_.get());
  const std::string expected = Emit(module_.get());
  ASSERT_FALSE(expected.empty());
  ExpectFixedStorage(expected, "padded_tail.kd", 64);
  ExpectFixedStorage(expected, "capacity_only.kd", 0);
  ExpectSameEmission(module_.get(), expected, "repeated emission");

  loomc_module_t* cloned = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module_.get(), workspace_.get(),
                                     loomc_allocator_system(), &cloned));
  ModulePtr clone(cloned);
  ExpectSameEmission(clone.get(), expected, "cloned module");

  for (const auto format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format = format;
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
        module_.get(), &options, loomc_allocator_system(), &source));
    SourcePtr serialized(source);
    ModulePtr reparsed = Parse(source);
    ASSERT_NE(reparsed, nullptr);
    ExpectSameEmission(reparsed.get(), expected,
                       format == LOOMC_SOURCE_FORMAT_TEXT ? "text reparse"
                                                          : "bytecode reparse");
  }

  // A continuing compile invalidates the physical snapshot before mutation,
  // while preserving the function versions and memory proofs used by emission.
  ResultPtr continued = Compile(0);
  ASSERT_TRUE(loomc_result_succeeded(continued.get()));
  continued.reset();
  loomc_workspace_trim(workspace_.get());
  ExpectSameEmission(module_.get(), expected, "continued compilation");
}

TEST_F(WorkgroupStorageTest,
       LateCollectiveSharesFinalPrefixWithNativeEmission) {
  LoadFixture("late_collective.loom");
  ResultPtr compiled = Compile(LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG);
  ASSERT_TRUE(loomc_result_succeeded(compiled.get()));
  LaunchPtr launch = LoadLaunch(compiled.get());
  ASSERT_NE(launch, nullptr);
  compiled.reset();
  loomc_workspace_trim(workspace_.get());
  ExpectTotal(launch.get(), "late_collective", 1024, 1152);
  ExpectTotal(launch.get(), "late_collective", 2048, 2176);
  const std::string expected = Emit(module_.get());
  ASSERT_FALSE(expected.empty());
  ExpectFixedStorage(expected, "late_collective.kd", 128);

  // The cloned module has no physical snapshot. Its native frame reconstructs
  // the same fixed prefix through the shared producer.
  loomc_module_t* cloned = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module_.get(), workspace_.get(),
                                     loomc_allocator_system(), &cloned));
  ModulePtr clone(cloned);
  ExpectSameEmission(clone.get(), expected, "cloned late collective");
}

}  // namespace
