// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/wasm.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

constexpr char kSource[] = R"(
config.decl @dead_config : index

func.def public @sum_to(%value: i32) -> (i32) {
  func.return %value : i32
}

func.def public export("artifact_alias") @renamed(%value: i32) -> (i32) {
  %result = func.call @helper(%value) : (i32) -> (i32)
  func.return %result : i32
}

func.def export("private_alias") @helper(%value: i32) -> (i32) {
  func.return %value : i32
}

func.def public @dead_config_user() -> (index) {
  %value = config.get @dead_config : index
  func.return %value : index
}
)";

bool ReadU32Leb(const uint8_t** cursor, const uint8_t* end,
                uint32_t* out_value) {
  uint32_t value = 0;
  for (uint32_t shift = 0; shift <= 28; shift += 7) {
    if (*cursor == end) {
      return false;
    }
    const uint8_t byte = *(*cursor)++;
    if (shift == 28 && (byte & 0xF0u) != 0) {
      return false;
    }
    value |= static_cast<uint32_t>(byte & 0x7Fu) << shift;
    if ((byte & 0x80u) == 0) {
      *out_value = value;
      return true;
    }
  }
  return false;
}

bool ReadString(const uint8_t** cursor, const uint8_t* end,
                std::string* out_value) {
  uint32_t length = 0;
  if (!ReadU32Leb(cursor, end, &length) ||
      static_cast<uint64_t>(length) > static_cast<uint64_t>(end - *cursor)) {
    return false;
  }
  out_value->assign(reinterpret_cast<const char*>(*cursor), length);
  *cursor += length;
  return true;
}

// Reads only the standard export section from compiler-owned artifact bytes.
::testing::AssertionResult ReadFunctionExportNames(
    loomc_byte_span_t contents, std::vector<std::string>* out_names) {
  constexpr uint8_t kWasmHeader[] = {0x00, 0x61, 0x73, 0x6D,
                                     0x01, 0x00, 0x00, 0x00};
  if (contents.data_length < sizeof(kWasmHeader) ||
      memcmp(contents.data, kWasmHeader, sizeof(kWasmHeader)) != 0) {
    return ::testing::AssertionFailure() << "artifact is not a Wasm module";
  }

  const uint8_t* cursor = contents.data + sizeof(kWasmHeader);
  const uint8_t* end = contents.data + contents.data_length;
  while (cursor != end) {
    const uint8_t section_id = *cursor++;
    uint32_t section_size = 0;
    if (!ReadU32Leb(&cursor, end, &section_size) ||
        static_cast<uint64_t>(section_size) >
            static_cast<uint64_t>(end - cursor)) {
      return ::testing::AssertionFailure() << "invalid Wasm section size";
    }
    const uint8_t* section_end = cursor + section_size;
    if (section_id != 7) {
      cursor = section_end;
      continue;
    }

    uint32_t export_count = 0;
    if (!ReadU32Leb(&cursor, section_end, &export_count)) {
      return ::testing::AssertionFailure() << "invalid Wasm export count";
    }
    for (uint32_t i = 0; i < export_count; ++i) {
      std::string name;
      if (!ReadString(&cursor, section_end, &name) || cursor == section_end) {
        return ::testing::AssertionFailure() << "invalid Wasm export name";
      }
      const uint8_t kind = *cursor++;
      uint32_t index = 0;
      if (kind != 0 || !ReadU32Leb(&cursor, section_end, &index)) {
        return ::testing::AssertionFailure()
               << "expected a function export named '" << name << "'";
      }
      out_names->push_back(name);
    }
    if (cursor != section_end) {
      return ::testing::AssertionFailure()
             << "unexpected trailing Wasm export data";
    }
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "Wasm module has no export section";
}

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result = ToString(contents);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
}

loomc_status_t AppendTrace(void* user_data, loomc_string_view_t fragment) {
  static_cast<std::string*>(user_data)->append(fragment.data, fragment.size);
  return loomc_ok_status();
}

struct TraceArtifacts {
  std::string open_reference;
  std::string open_contents;
  std::vector<std::string> references;
  std::vector<std::string> contents;
};

loomc_status_t OpenTraceArtifact(void* user_data,
                                 const loomc_pass_trace_event_t* event,
                                 loomc_pass_trace_artifact_t* out_artifact) {
  auto* artifacts = static_cast<TraceArtifacts*>(user_data);
  const char* point =
      event->point == LOOMC_PASS_TRACE_POINT_BEFORE ? "before" : "after";
  artifacts->open_reference = "ir/" + std::to_string(event->event_ordinal) +
                              "-" + point + "-" + ToString(event->pass_key) +
                              ".loom";
  artifacts->open_contents.clear();
  *out_artifact = (loomc_pass_trace_artifact_t){
      .reference = loomc_make_string_view(artifacts->open_reference.data(),
                                          artifacts->open_reference.size()),
      .sink =
          {
              .write = AppendTrace,
              .user_data = &artifacts->open_contents,
          },
  };
  return loomc_ok_status();
}

loomc_status_t CloseTraceArtifact(void* user_data,
                                  loomc_pass_trace_artifact_t* artifact) {
  (void)artifact;
  auto* artifacts = static_cast<TraceArtifacts*>(user_data);
  artifacts->references.push_back(artifacts->open_reference);
  artifacts->contents.push_back(artifacts->open_contents);
  artifacts->open_reference.clear();
  artifacts->open_contents.clear();
  return loomc_ok_status();
}

::testing::AssertionResult Succeeded(const loomc_result_t* result) {
  if (result != nullptr && loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  auto failure = ::testing::AssertionFailure();
  if (result == nullptr) {
    return failure << "operation did not return a result";
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    failure << ToString(diagnostic->message);
  }
  return failure;
}

TEST(TargetWasmTest, CompilesArtifactWithEmitterDefaultPipeline) {
  loomc_target_environment_t* raw_target_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_wasm(
      loomc_allocator_system(), &raw_target_environment));
  TargetEnvironmentPtr target_environment(raw_target_environment);
  loomc_target_profile_t* raw_target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(), loomc_make_cstring_view("wasm:simd128"),
      loomc_allocator_system(), &raw_target_profile));
  TargetProfilePtr target_profile(raw_target_profile);

  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment.get(),
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &raw_context));
  ContextPtr context(raw_context);

  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  WorkspacePtr workspace(raw_workspace);

  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("exports.loom"),
      .contents = loomc_make_byte_span(kSource, sizeof(kSource) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &raw_source));
  SourcePtr source(raw_source);

  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), nullptr,
      loomc_allocator_system(), &raw_module, &raw_result));
  ModulePtr module(raw_module);
  ResultPtr result(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  CompilerPtr compiler(raw_compiler);
  const loomc_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .next = nullptr,
      .mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY,
      .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
  };
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &report_options,
  };
  const loomc_config_options_t config_options = {
      .bindings = nullptr,
      .binding_count = 0,
      .json_object = loomc_string_view_empty(),
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  const loomc_sanitizer_options_t sanitizer_options = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer_options),
      .next = nullptr,
      .checks = LOOMC_SANITIZER_CHECKS_ASAN_LIKE,
      .flags = LOOMC_SANITIZER_FLAG_NONE,
      .reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP,
  };
  TraceArtifacts trace_artifacts;
  std::string pass_trace;
  const loomc_string_view_t before_filters[] = {
      loomc_make_cstring_view("prepared-low"),
  };
  const loomc_pass_trace_options_t pass_trace_options = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(pass_trace_options),
      .next = &sanitizer_options,
      .format = LOOMC_PASS_TRACE_FORMAT_JSONL,
      .flags = 0,
      .tool_name = loomc_make_cstring_view("loomc-wasm-test"),
      .input_identifier = loomc_make_cstring_view("exports.loom"),
      .before_filters = before_filters,
      .before_filter_count = IREE_ARRAYSIZE(before_filters),
      .after_filters = nullptr,
      .after_filter_count = 0,
      .sink =
          {
              .write = AppendTrace,
              .user_data = &pass_trace,
          },
      .artifact_sink =
          {
              .open = OpenTraceArtifact,
              .close = CloseTraceArtifact,
              .user_data = &trace_artifacts,
          },
  };
  const loomc_string_view_t excluded_roots[] = {
      loomc_make_cstring_view("dead_config_user"),
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &pass_trace_options,
      .roots = nullptr,
      .root_count = 0,
      .excluded_roots = excluded_roots,
      .excluded_root_count = IREE_ARRAYSIZE(excluded_roots),
      .target_profile = target_profile.get(),
      .config = &config_options,
      .emit_options = &emit_options,
  };
  const loomc_artifact_manifest_options_t manifest_options = {
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(manifest_options),
      .next = nullptr,
      .mode = LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
  };
  loomc_emit_options_t manifest_emit_options = emit_options;
  manifest_emit_options.next = &manifest_options;
  loomc_compile_artifact_options_t manifest_compile_options = compile_options;
  manifest_compile_options.emit_options = &manifest_emit_options;
  loomc_result_t* manifest_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &manifest_compile_options, loomc_allocator_system(), &manifest_result));
  ResultPtr manifest_result_owner(manifest_result);
  ASSERT_FALSE(loomc_result_succeeded(manifest_result_owner.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(manifest_result_owner.get()), 1u);
  const loomc_diagnostic_t* manifest_diagnostic =
      loomc_result_diagnostic_at(manifest_result_owner.get(), 0);
  EXPECT_EQ(ToString(manifest_diagnostic->code), "COMPILE/REQUEST");
  EXPECT_EQ(ToString(manifest_diagnostic->message),
            "artifact manifests are only valid for loadable kernel formats");
  EXPECT_EQ(loomc_result_artifact_count(manifest_result_owner.get()), 0u);

  const loomc_string_view_t missing_roots[] = {
      loomc_make_cstring_view("missing"),
  };
  loomc_emit_options_t missing_emit_options = emit_options;
  missing_emit_options.identifier = loomc_make_cstring_view("missing.wasm");
  loomc_compile_artifact_options_t missing_compile_options = compile_options;
  missing_compile_options.roots = missing_roots;
  missing_compile_options.root_count = IREE_ARRAYSIZE(missing_roots);
  missing_compile_options.excluded_roots = nullptr;
  missing_compile_options.excluded_root_count = 0;
  missing_compile_options.emit_options = &missing_emit_options;
  loomc_result_t* missing_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &missing_compile_options, loomc_allocator_system(), &missing_result));
  ResultPtr missing_result_owner(missing_result);
  ASSERT_FALSE(loomc_result_succeeded(missing_result_owner.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(missing_result_owner.get()), 1u);
  EXPECT_EQ(
      ToString(loomc_result_diagnostic_at(missing_result_owner.get(), 0)->code),
      "COMPILE/REQUEST");
  ASSERT_EQ(loomc_result_artifact_count(missing_result_owner.get()), 1u);
  const loomc_artifact_t* missing_report =
      loomc_result_artifact_at(missing_result_owner.get(), 0);
  ASSERT_NE(missing_report, nullptr);
  EXPECT_EQ(missing_report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(missing_report->format),
            LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  EXPECT_EQ(ToString(missing_report->identifier),
            "missing.wasm.compile-report.json");
  const std::string missing_report_contents =
      ToString(missing_report->contents);
  EXPECT_NE(missing_report_contents.find("\"name\":\"FAILED_PRECONDITION\""),
            std::string::npos);
  EXPECT_NE(missing_report_contents.find("\"diagnostic_count\":1"),
            std::string::npos);

  loomc_pass_trace_options_t incomplete_artifact_sink = pass_trace_options;
  incomplete_artifact_sink.artifact_sink.close = nullptr;
  loomc_compile_artifact_options_t incomplete_artifact_options =
      compile_options;
  incomplete_artifact_options.next = &incomplete_artifact_sink;
  loomc_result_t* incomplete_artifact_result = nullptr;
  loomc_status_t incomplete_artifact_status = loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &incomplete_artifact_options, loomc_allocator_system(),
      &incomplete_artifact_result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         incomplete_artifact_status);
  EXPECT_EQ(incomplete_artifact_result, nullptr);
  EXPECT_TRUE(trace_artifacts.open_reference.empty());
  EXPECT_TRUE(trace_artifacts.references.empty());

  loomc_pass_program_t* raw_pass_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &raw_pass_program));
  PassProgramPtr pass_program(raw_pass_program);
  loomc_result_t* rejected_result = nullptr;
  loomc_status_t rejected_status = loomc_compile_artifact(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &rejected_result);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, rejected_status);
  EXPECT_EQ(rejected_result, nullptr);

  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &compile_options, loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);
  EXPECT_NE(pass_trace.find("\"tool\":\"loomc-wasm-test\""), std::string::npos);
  EXPECT_NE(pass_trace.find("\"input\":\"exports.loom\""), std::string::npos);
  EXPECT_NE(pass_trace.find("\"stage\":\"prepared-low\""), std::string::npos);
  EXPECT_NE(pass_trace.find("\"point\":\"before\""), std::string::npos);
  EXPECT_EQ(pass_trace.find("\"point\":\"after\""), std::string::npos);
  ASSERT_FALSE(trace_artifacts.references.empty());
  ASSERT_EQ(trace_artifacts.references.size(), trace_artifacts.contents.size());
  EXPECT_NE(pass_trace.find("\"ir_path\":\"" +
                            trace_artifacts.references.front() + "\""),
            std::string::npos);
  for (const std::string& contents : trace_artifacts.contents) {
    EXPECT_FALSE(contents.empty());
  }

  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_WASM_BINARY);
  EXPECT_EQ(ToString(artifact->identifier), "module.wasm");
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_ASSERT_OK(loomc_byte_sequence_clone(
      artifact->contents, loomc_allocator_system(), &contents));
  std::vector<std::string> export_names;
  ASSERT_TRUE(ReadFunctionExportNames(contents, &export_names));
  std::sort(export_names.begin(), export_names.end());
  const std::vector<std::string> expected_export_names = {"artifact_alias",
                                                          "sum_to"};
  EXPECT_EQ(export_names, expected_export_names);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);

  const loomc_artifact_t* report = loomc_result_artifact_at(result.get(), 1);
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(report->format),
            LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  EXPECT_EQ(ToString(report->identifier), "module.wasm.compile-report.json");
  const std::string report_contents = ToString(report->contents);
  EXPECT_NE(report_contents.find("\"status\":{\"code\":0,\"name\":\"OK\"}"),
            std::string::npos);
  EXPECT_NE(report_contents.find("\"backend\":\"wasm-binary\""),
            std::string::npos);
  EXPECT_NE(report_contents.find("\"artifact_format\":\"wasm_binary\""),
            std::string::npos);

  // The same unresolved declaration must fail when its root survives.
  raw_module = nullptr;
  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), nullptr,
      loomc_allocator_system(), &raw_module, &raw_result));
  module.reset(raw_module);
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));
  loomc_compile_artifact_options_t unresolved_compile_options = compile_options;
  unresolved_compile_options.excluded_roots = nullptr;
  unresolved_compile_options.excluded_root_count = 0;
  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &unresolved_compile_options, loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ASSERT_FALSE(loomc_result_succeeded(result.get()));
  ASSERT_NE(loomc_result_diagnostic_count(result.get()), 0u);
  EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
            "CONFIG/INVALID");
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* failed_report =
      loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(failed_report, nullptr);
  EXPECT_EQ(failed_report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(failed_report->format),
            LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  EXPECT_NE(ToString(failed_report->contents).find("\"diagnostic_count\":1"),
            std::string::npos);
}

}  // namespace
