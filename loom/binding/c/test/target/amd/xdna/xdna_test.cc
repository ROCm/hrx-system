// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amd/xdna.h"

#include <string>

#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;

// Multiple exports share a construction template. Repeated image linking must
// preserve the prepared module and keep returned bytes independently alive.
constexpr char kSource[] = R"(
aie2p.target<array> @device

pipeline.def @record(%memory: pool, %row: index) run() {
  %bytes = index.constant 4 : offset
  %origin = index.constant 0 : offset
  %slots = index.constant 1 : index
  %storage = buffer.alloca<workgroup> pool(%memory) align(4) %bytes : buffer
  %records = buffer.view %storage[%origin] : buffer -> view<1x1xi32>
  %channel = channel.bind<fifo> %records, %slots : view<1x1xi32>, index -> channel<tile<1xi32>>
  pipeline.strand workers([0, %row], [1, 1], [1, 1]) {
    %write, %record = channel.reserve %channel : channel<tile<1xi32>> -> (write<tile<1xi32>>, view<1xi32>)
    %value = scalar.constant 42 : i32
    view.store %value, %record[0] : i32, view<1xi32>
    channel.publish %write : write<tile<1xi32>>
  }
}

pipeline.def<kernel> public retain target(@device) @first() run() {
  %row = index.constant 2 : index
  %memory = pipeline.memory<workgroup>[0, %row] : pool
  pipeline.compose @record[%memory, %row]() : [pool, index]()
}

pipeline.def<kernel> public retain target(@device) @second() run() {
  %south = index.constant 2 : index
  %north = index.constant 3 : index
  %south_memory = pipeline.memory<workgroup>[0, %south] : pool
  %north_memory = pipeline.memory<workgroup>[0, %north] : pool
  pipeline.compose @record[%south_memory, %south]() : [pool, index]()
  pipeline.compose @record[%north_memory, %north]() : [pool, index]()
}
)";

std::string SequenceText(const loomc_byte_sequence_t* sequence) {
  std::string text;
  LOOMC_EXPECT_OK(loomc_byte_sequence_enumerate(
      sequence, {[](void* user_data, loomc_byte_span_t segment) {
                   static_cast<std::string*>(user_data)->append(
                       reinterpret_cast<const char*>(segment.data),
                       segment.data_length);
                   return loomc_ok_status();
                 },
                 &text}));
  return text;
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format,
                                loomc_make_cstring_view(format))) {
      return artifact;
    }
  }
  return nullptr;
}

std::string SerializeModuleToText(const loomc_module_t* module) {
  loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("compiled.loom"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  if (!loomc_status_is_ok(status)) {
    return std::string();
  }
  SourcePtr source_ptr(source);
  const loomc_byte_span_t contents = loomc_source_contents(source_ptr.get());
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

::testing::AssertionResult Succeeded(const loomc_result_t* result) {
  if (loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  auto failure = ::testing::AssertionFailure();
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    failure << std::string(diagnostic->message.data, diagnostic->message.size);
  }
  return failure;
}

class XdnaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_target_environment_t* environment = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_xdna(
        loomc_allocator_system(), &environment));
    environment_.reset(environment);
  }

  // Shared immutable compiler capability package.
  HandlePtr<loomc_target_environment_t, loomc_target_environment_release>
      environment_;
};

TEST_F(XdnaTest, RejectsUnknownDeviceProfile) {
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_xdna(environment_.get(),
                                       loomc_make_cstring_view("not-a-device"),
                                       loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST_F(XdnaTest, RejectsDeviceKeyWithoutStorage) {
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_xdna(environment_.get(), {nullptr, 1},
                                       loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST_F(XdnaTest, RejectsMissingProfileOutput) {
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_xdna(
          environment_.get(),
          loomc_make_cstring_view("amd.xdna.strix_halo.17f0_11"),
          loomc_allocator_system(), nullptr));
}

TEST_F(XdnaTest, RejectsMissingEnvironment) {
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_xdna(
          nullptr, loomc_make_cstring_view("amd.xdna.strix_halo.17f0_11"),
          loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST_F(XdnaTest, PreservesTargetTypesInLowVerificationDiagnostics) {
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = environment_.get()};
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS, .next = &target_options};
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &raw_context));
  HandlePtr<loomc_context_t, loomc_context_release> context(raw_context);
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace);
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  HandlePtr<loomc_compiler_t, loomc_compiler_release> compiler(raw_compiler);
  loomc_pass_program_t* raw_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_pipeline_text(
      context.get(), loomc_make_cstring_view("low-dce"), nullptr,
      loomc_allocator_system(), &raw_program, &raw_result));
  HandlePtr<loomc_pass_program_t, loomc_pass_program_release> program(
      raw_program);
  ResultPtr result(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  // This unused resource would disappear if DCE ran before descriptor
  // verification.
  constexpr char source_text[] = R"(
low.func.def retain target<amd.xdna.aie2p.core> @invalid() asm {
  %unused = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.er>
  return
}
)";
  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .contents = loomc_make_byte_span(source_text, sizeof(source_text) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY};
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &raw_source));
  HandlePtr<loomc_source_t, loomc_source_release> source(raw_source);
  loomc_module_t* raw_module = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), nullptr,
      loomc_allocator_system(), &raw_module, &raw_result));
  ModulePtr module(raw_module);
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  LOOMC_ASSERT_OK(loomc_compile_module(compiler.get(), workspace.get(),
                                       program.get(), module.get(), nullptr,
                                       loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_TRUE(
      loomc_string_view_equal(loomc_result_diagnostic_at(result.get(), 0)->code,
                              loomc_make_cstring_view("TYPE/004")));
  EXPECT_EQ(
      std::string(loomc_result_diagnostic_at(result.get(), 0)->message.data,
                  loomc_result_diagnostic_at(result.get(), 0)->message.size),
      "result 'result' has type reg<aie2p.er>, expected register class "
      "in [aie2p.ep] with 1 unit(s)");

  // Reuse the rejected module: failure must not erase the offending operand or
  // mark it verified for a later direct emission request.
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_XDNA)};
  LOOMC_ASSERT_OK(loomc_emit_module(environment_.get(), workspace.get(),
                                    module.get(), &emit_options,
                                    loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_TRUE(
      loomc_string_view_equal(loomc_result_diagnostic_at(result.get(), 0)->code,
                              loomc_make_cstring_view("TYPE/004")));
  EXPECT_EQ(
      std::string(loomc_result_diagnostic_at(result.get(), 0)->message.data,
                  loomc_result_diagnostic_at(result.get(), 0)->message.size),
      "result 'result' has type reg<aie2p.er>, expected register class "
      "in [aie2p.ep] with 1 unit(s)");
}

TEST_F(XdnaTest, AcceptsNonTerminatedDeviceKeyAndRetainsEnvironment) {
  const std::string expected_key = "amd.xdna.strix_halo.17f0_11";
  std::string key_storage = expected_key + "not-part-of-the-key";
  const loomc_string_view_t key = {key_storage.data(), expected_key.size()};
  loomc_target_profile_t* raw_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_xdna(
      environment_.get(), key, loomc_allocator_system(), &raw_profile));
  HandlePtr<loomc_target_profile_t, loomc_target_profile_release> profile(
      raw_profile);
  key_storage.assign(key_storage.size(), 'x');
  environment_.reset();
}

TEST_F(XdnaTest, PreservesPreparedModuleAcrossRepeatedEmission) {
  loomc_target_profile_t* raw_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_xdna(
      environment_.get(),
      loomc_make_cstring_view("amd.xdna.strix_halo.17f0_11"),
      loomc_allocator_system(), &raw_profile));
  HandlePtr<loomc_target_profile_t, loomc_target_profile_release> profile(
      raw_profile);
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = environment_.get()};
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS, .next = &target_options};
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &raw_context));
  HandlePtr<loomc_context_t, loomc_context_release> context(raw_context);
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace);
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  HandlePtr<loomc_compiler_t, loomc_compiler_release> compiler(raw_compiler);
  loomc_pass_program_t* raw_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), nullptr, loomc_allocator_system(), &raw_program,
      &raw_result));
  HandlePtr<loomc_pass_program_t, loomc_pass_program_release> program(
      raw_program);
  ResultPtr result(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("xdna.loom"),
      .contents = loomc_make_byte_span(kSource, sizeof(kSource) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY};
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &raw_source));
  HandlePtr<loomc_source_t, loomc_source_release> source(raw_source);

  SequencePtr retained;
  std::string first_bytes;
  std::string first_report;
  for (int invocation = 0; invocation < 2; ++invocation) {
    loomc_module_t* raw_module = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), source.get(), nullptr,
        loomc_allocator_system(), &raw_module, &raw_result));
    result.reset(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    ModulePtr module(raw_module);
    const loomc_target_specialization_t specializations[] = {
        {loomc_make_cstring_view("first"), profile.get()},
        {loomc_make_cstring_view("second"), profile.get()}};
    loomc_target_specialization_options_t specialization_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .specializations = specializations,
        .specialization_count = IREE_ARRAYSIZE(specializations)};
    loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .next = &specialization_options};
    LOOMC_ASSERT_OK(loomc_compile_module(
        compiler.get(), workspace.get(), program.get(), module.get(),
        &compile_options, loomc_allocator_system(), &raw_result));
    result.reset(raw_result);
    ASSERT_TRUE(Succeeded(result.get()));
    const std::string prepared_text = SerializeModuleToText(module.get());
    ASSERT_FALSE(prepared_text.empty());

    loomc_emit_options_t emit_options = {};
    emit_options.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit_options.artifact_format =
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_XDNA);
    emit_options.artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
    loomc_compile_report_options_t report_options = {};
    report_options.type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS;
    report_options.structure_size = sizeof(report_options);
    report_options.mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    if (invocation == 0) {
      loomc_artifact_manifest_options_t manifest_options = {
          .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
          .mode = LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY};
      emit_options.next = &manifest_options;
      LOOMC_ASSERT_OK(loomc_emit_module(environment_.get(), workspace.get(),
                                        module.get(), &emit_options,
                                        loomc_allocator_system(), &raw_result));
      result.reset(raw_result);
      EXPECT_FALSE(loomc_result_succeeded(result.get()));
      EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
      ASSERT_GT(loomc_result_diagnostic_count(result.get()), 0u);
      const auto* diagnostic = loomc_result_diagnostic_at(result.get(), 0);
      EXPECT_NE(std::string(diagnostic->message.data, diagnostic->message.size)
                    .find("sidecar artifact manifests are not supported"),
                std::string::npos);
      EXPECT_EQ(SerializeModuleToText(module.get()), prepared_text);
    }
    for (int emission = 0; emission < 2; ++emission) {
      const bool report_requested = emission != 0;
      emit_options.next = report_requested ? &report_options : nullptr;
      LOOMC_ASSERT_OK(loomc_emit_module(environment_.get(), workspace.get(),
                                        module.get(), &emit_options,
                                        loomc_allocator_system(), &raw_result));
      result.reset(raw_result);
      ASSERT_TRUE(Succeeded(result.get()));
      ASSERT_EQ(loomc_result_artifact_count(result.get()),
                report_requested ? 2u : 1u);
      const loomc_artifact_t* executable =
          FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                       LOOMC_ARTIFACT_FORMAT_XDNA);
      ASSERT_NE(executable, nullptr);
      const std::string artifact_bytes = SequenceText(executable->contents);
      ASSERT_FALSE(artifact_bytes.empty());
      if (retained.get() == nullptr) {
        loomc_byte_sequence_retain(executable->contents);
        retained.reset(executable->contents);
        first_bytes = artifact_bytes;
      } else {
        EXPECT_EQ(artifact_bytes, first_bytes);
      }
      if (report_requested) {
        const loomc_artifact_t* report =
            FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                         LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
        ASSERT_NE(report, nullptr);
        const std::string report_bytes = SequenceText(report->contents);
        ASSERT_FALSE(report_bytes.empty());
        if (first_report.empty()) {
          first_report = report_bytes;
        } else {
          EXPECT_EQ(report_bytes, first_report);
        }
      }
      EXPECT_EQ(SerializeModuleToText(module.get()), prepared_text);
    }
  }
  result.reset();
  workspace.reset();
  compiler.reset();
  program.reset();
  source.reset();
  context.reset();
  profile.reset();
  environment_.reset();
  EXPECT_EQ(SequenceText(retained.get()), first_bytes);
}

}  // namespace
