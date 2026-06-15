// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/link.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target/spirv/base.h"
#include "loomc/workspace.h"
#include "target.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

void FakeArtifactRelease(void* storage, iree_allocator_t allocator) {
  iree_allocator_free(allocator, storage);
}

typedef struct FakeArtifactSidecarStorage {
  // Primary fake executable bytes.
  uint8_t contents[4];
  // Fake manifest sidecar descriptor.
  loom_target_emit_sidecar_artifact_t sidecar;
} FakeArtifactSidecarStorage;

void FakeArtifactSidecarRelease(void* storage, iree_allocator_t allocator) {
  iree_allocator_free(allocator, storage);
}

iree_status_t EmitFakeArtifact(const loom_target_emit_request_t* request,
                               loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = {};
  if (request->compile_report != nullptr) {
    loom_target_compile_report_record_emission(
        request->compile_report, /*instruction_count=*/3,
        /*code_byte_count=*/4, /*code_storage_byte_count=*/4);
  }
  static const char kManifestJson[] =
      "{\"kind\":\"loom.artifact_manifest\",\"mode\":\"summary\"}";
  out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  if (request->artifact_manifest.mode ==
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    uint8_t* contents = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(request->allocator, 4, (void**)&contents));
    contents[0] = 0x7F;
    contents[1] = 'L';
    contents[2] = 'O';
    contents[3] = 'M';
    out_artifact->contents = iree_make_const_byte_span(contents, 4);
    out_artifact->storage = contents;
    out_artifact->release = FakeArtifactRelease;
    return iree_ok_status();
  }

  FakeArtifactSidecarStorage* storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      request->allocator, sizeof(*storage), (void**)&storage));
  *storage = {};
  storage->contents[0] = 0x7F;
  storage->contents[1] = 'L';
  storage->contents[2] = 'O';
  storage->contents[3] = 'M';
  storage->sidecar = {
      /*.kind=*/LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
      /*.identifier=*/request->artifact_manifest.identifier,
      /*.contents=*/
      iree_make_const_byte_span((const uint8_t*)kManifestJson,
                                sizeof(kManifestJson) - 1),
  };
  out_artifact->contents = iree_make_const_byte_span(storage->contents, 4);
  out_artifact->sidecars = &storage->sidecar;
  out_artifact->sidecar_count = 1;
  out_artifact->storage = storage;
  out_artifact->release = FakeArtifactSidecarRelease;
  return iree_ok_status();
}

static const loom_target_emitter_t kFakeElfEmitter = {
    /*.name=*/{"fake-elf", 8},
    /*.public_artifact_format=*/{"fake-elf", 8},
    /*.default_identifier=*/{"fake.bin", 8},
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    /*.emit=*/EmitFakeArtifact,
};

static const loom_target_emitter_t kFakeWasmEmitter = {
    /*.name=*/{"fake-wasm", 9},
    /*.public_artifact_format=*/{"fake-wasm", 9},
    /*.default_identifier=*/{"fake.wasm", 9},
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    /*.emit=*/EmitFakeArtifact,
};

static const loom_target_emitter_t* const kFakeElfEmitters[] = {
    &kFakeElfEmitter,
};

static const loom_target_emitter_t* const kFakeWasmEmitters[] = {
    &kFakeWasmEmitter,
};

static const loom_target_provider_t kEmptyProvider = {};

static const loom_target_provider_t kFakeElfProvider = {
    /*.register_context=*/nullptr,
    /*.initialize_low_descriptor_registry=*/nullptr,
    /*.initialize_low_lower_policy_registry=*/nullptr,
    /*.initialize_math_policy_registry=*/nullptr,
    /*.low_legality_provider_list=*/{},
    /*.legalizer_provider_list=*/{},
    /*.low_packet_diagnostic_provider_list=*/{},
    /*.low_asm_diagnostic_provider_list=*/{},
    /*.low_verify_provider_list=*/{},
    /*.emitter_list=*/
    {
        /*.values=*/kFakeElfEmitters,
        /*.count=*/IREE_ARRAYSIZE(kFakeElfEmitters),
    },
    /*.pass_registry=*/nullptr,
    /*.contribute_pipeline=*/nullptr,
};

static const loom_target_provider_t kFakeWasmProvider = {
    /*.register_context=*/nullptr,
    /*.initialize_low_descriptor_registry=*/nullptr,
    /*.initialize_low_lower_policy_registry=*/nullptr,
    /*.initialize_math_policy_registry=*/nullptr,
    /*.low_legality_provider_list=*/{},
    /*.legalizer_provider_list=*/{},
    /*.low_packet_diagnostic_provider_list=*/{},
    /*.low_asm_diagnostic_provider_list=*/{},
    /*.low_verify_provider_list=*/{},
    /*.emitter_list=*/
    {
        /*.values=*/kFakeWasmEmitters,
        /*.count=*/IREE_ARRAYSIZE(kFakeWasmEmitters),
    },
    /*.pass_registry=*/nullptr,
    /*.contribute_pipeline=*/nullptr,
};

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, nullptr, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
}

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* linker = nullptr;
  loomc_status_t status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  LOOMC_EXPECT_OK(status);
  return LinkerPtr(linker);
}

BuilderPtr CreateLinkIndexBuilder(loomc_context_t* context) {
  loomc_link_index_builder_t* builder = nullptr;
  loomc_status_t status = loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &builder);
  LOOMC_EXPECT_OK(status);
  return BuilderPtr(builder);
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

TargetEnvironmentPtr CreateTargetEnvironmentFromProviderSet(
    const loom_target_provider_set_t* provider_set) {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_from_provider_set(
      provider_set, loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateSpirvContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

TargetProfilePtr CreateEmptyProfile(
    loomc_target_environment_t* target_environment) {
  loomc_target_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("test-profile"),
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_empty(
      target_environment, &options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateSelectionFromProfile(loomc_target_profile_t* profile) {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile, loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

loomc_pass_program_options_t PassOptionsWithSelection(
    loomc_target_selection_options_t* target_options) {
  return loomc_pass_program_options_t{
      /*.type=*/LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(loomc_pass_program_options_t),
      /*.next=*/target_options,
      /*.identifier=*/loomc_make_cstring_view("selected-pass-program"),
  };
}

PassProgramPtr CreateEmptyPassProgramWithSelection(
    loomc_context_t* context, loomc_target_selection_t* target_selection) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection,
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context, &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  return PassProgramPtr(pass_program);
}

TargetSelectionPtr CreateEmptySelection() {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status =
      loomc_target_selection_create_empty(loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

ModulePtr CreateIdentityModule(loomc_context_t* context,
                               loomc_workspace_t* workspace,
                               const char* symbol) {
  std::string contents = "func.def public @";
  contents.append(symbol);
  contents.append(R"((%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  SourcePtr source = CreateTextSource("identity.loom", contents.c_str());
  return DeserializeModule(context, workspace, source.get());
}

ResultPtr EmitModule(loomc_target_environment_t* target_environment,
                     loomc_workspace_t* workspace, loomc_module_t* module,
                     const loomc_emit_options_t* options) {
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_emit_module(target_environment, workspace, module, options,
                        loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  return ResultPtr(result);
}

void ExpectFailedEmitTargetResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "EMIT/TARGET");
}

TEST(TargetTest, EmitSelectsOnlyLinkedEmitterWhenFormatOmitted) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(ToString(artifact->format), "fake-elf");
  EXPECT_EQ(ToString(artifact->identifier), "fake.bin");
  ASSERT_EQ(artifact->contents.data_length, 4u);
  EXPECT_EQ(artifact->contents.data[0], 0x7Fu);
}

TEST(TargetTest, EmitReturnsArtifactManifestSidecar) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_artifact_manifest_options_t manifest_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      /*.structure_size=*/sizeof(manifest_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
      /*.identifier=*/loomc_string_view_empty(),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&manifest_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);

  const loomc_artifact_t* primary = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(primary, nullptr);
  EXPECT_EQ(primary->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(primary->format), "fake-elf");
  EXPECT_EQ(ToString(primary->identifier), "fake.bin");
  ASSERT_EQ(primary->contents.data_length, 4u);
  EXPECT_EQ(primary->contents.data[0], 0x7Fu);

  const loomc_artifact_t* manifest = loomc_result_artifact_at(result.get(), 1);
  ASSERT_NE(manifest, nullptr);
  EXPECT_EQ(manifest->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(manifest->format),
            LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON);
  EXPECT_EQ(ToString(manifest->identifier), "fake.bin.manifest.json");
  EXPECT_EQ(ToString(manifest->contents),
            "{\"kind\":\"loom.artifact_manifest\",\"mode\":\"summary\"}");
}

TEST(TargetTest, EmitReturnsCompileReportArtifact) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_compile_report_options_t report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(report_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_COMPILE_REPORT_MODE_SUMMARY,
      /*.identifier=*/loomc_string_view_empty(),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&report_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);

  const loomc_artifact_t* primary = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(primary, nullptr);
  EXPECT_EQ(primary->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(primary->format), "fake-elf");
  EXPECT_EQ(ToString(primary->identifier), "fake.bin");

  const loomc_artifact_t* report = loomc_result_artifact_at(result.get(), 1);
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(report->format),
            LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  EXPECT_EQ(ToString(report->identifier), "fake.bin.compile-report.json");
  const std::string contents = ToString(report->contents);
  EXPECT_NE(contents.find("\"artifact_kind\":\"target-artifact\""),
            std::string::npos);
  EXPECT_NE(contents.find("\"status\":\"OK\""), std::string::npos);
  EXPECT_NE(contents.find("\"backend\":\"fake-elf\""), std::string::npos);
  EXPECT_NE(contents.find("\"executable_format\":\"fake-elf\""),
            std::string::npos);
  EXPECT_NE(contents.find("\"artifact_size\":4"), std::string::npos);
  EXPECT_NE(contents.find("\"instruction_count\":3"), std::string::npos);
}

TEST(TargetTest, EmitArtifactManifestLooseOptionsOverrideTypedDefaults) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  const loomc_option_entry_t entries[] = {
      {
          /*.key=*/loomc_make_cstring_view(
              LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_MODE),
          /*.value=*/loomc_make_cstring_view("summary"),
      },
      {
          /*.key=*/loomc_make_cstring_view(
              LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_IDENTIFIER),
          /*.value=*/loomc_make_cstring_view("sidecar.json"),
      },
  };
  loomc_option_dict_t dict = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      /*.structure_size=*/sizeof(dict),
      /*.next=*/nullptr,
      /*.entries=*/entries,
      /*.entry_count=*/IREE_ARRAYSIZE(entries),
  };
  loomc_artifact_manifest_options_t manifest_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      /*.structure_size=*/sizeof(manifest_options),
      /*.next=*/&dict,
      /*.mode=*/LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
      /*.identifier=*/loomc_make_cstring_view("default.json"),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&manifest_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_make_cstring_view("primary.bin"),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);
  const loomc_artifact_t* primary = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(primary, nullptr);
  EXPECT_EQ(ToString(primary->identifier), "primary.bin");
  const loomc_artifact_t* manifest = loomc_result_artifact_at(result.get(), 1);
  ASSERT_NE(manifest, nullptr);
  EXPECT_EQ(ToString(manifest->identifier), "sidecar.json");
}

TEST(TargetTest, EmitCompileReportLooseOptionsOverrideTypedDefaults) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  const loomc_option_entry_t entries[] = {
      {
          /*.key=*/loomc_make_cstring_view(
              LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_MODE),
          /*.value=*/loomc_make_cstring_view("json-details"),
      },
      {
          /*.key=*/loomc_make_cstring_view(
              LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_IDENTIFIER),
          /*.value=*/loomc_make_cstring_view("report.json"),
      },
  };
  loomc_option_dict_t dict = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      /*.structure_size=*/sizeof(dict),
      /*.next=*/nullptr,
      /*.entries=*/entries,
      /*.entry_count=*/IREE_ARRAYSIZE(entries),
  };
  loomc_compile_report_options_t report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(report_options),
      /*.next=*/&dict,
      /*.mode=*/LOOMC_COMPILE_REPORT_MODE_SUMMARY,
      /*.identifier=*/loomc_make_cstring_view("default.json"),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&report_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_make_cstring_view("primary.bin"),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectSucceededResult(result.get());
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);
  const loomc_artifact_t* primary = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(primary, nullptr);
  EXPECT_EQ(ToString(primary->identifier), "primary.bin");
  const loomc_artifact_t* report = loomc_result_artifact_at(result.get(), 1);
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(ToString(report->identifier), "report.json");
}

TEST(TargetTest, EmitRejectsArtifactManifestIdentifierWithoutMode) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_artifact_manifest_options_t manifest_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      /*.structure_size=*/sizeof(manifest_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_ARTIFACT_MANIFEST_MODE_NONE,
      /*.identifier=*/loomc_make_cstring_view("sidecar.json"),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&manifest_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_emit_module(target_environment.get(), workspace.get(), module.get(),
                        &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
}

TEST(TargetTest, EmitRejectsCompileReportIdentifierWithoutMode) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_compile_report_options_t report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(report_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_COMPILE_REPORT_MODE_NONE,
      /*.identifier=*/loomc_make_cstring_view("report.json"),
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&report_options,
      /*.artifact_format=*/loomc_string_view_empty(),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_emit_module(target_environment.get(), workspace.get(), module.get(),
                        &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
}

TEST(TargetTest, EmitReportsZeroLinkedEmittersThroughResult) {
  const loom_target_provider_t* providers[] = {
      &kEmptyProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), nullptr);
  ExpectFailedEmitTargetResult(result.get());
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
}

TEST(TargetTest, EmitReportsAmbiguousOmittedFormatThroughResult) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
      &kFakeWasmProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), nullptr);
  ExpectFailedEmitTargetResult(result.get());
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
}

TEST(TargetTest, EmitReportsMissingFormatThroughResult) {
  const loom_target_provider_t* providers[] = {
      &kFakeElfProvider,
  };
  loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  TargetEnvironmentPtr target_environment =
      CreateTargetEnvironmentFromProviderSet(&provider_set);
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");

  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.artifact_format=*/loomc_make_cstring_view("missing"),
      /*.identifier=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
  };
  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), &options);
  ExpectFailedEmitTargetResult(result.get());
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
}

LinkIndexPtr CreateSingleSourceLinkIndex(loomc_context_t* context) {
  BuilderPtr builder = CreateLinkIndexBuilder(context);
  SourcePtr source = CreateTextSource("link-input.loom", R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  loomc_link_index_source_options_t source_options = {
      /*.provider_name=*/loomc_make_cstring_view("jit-input"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  loomc_status_t status = loomc_link_index_builder_add_source(
      builder.get(), source.get(), &source_options, nullptr);
  LOOMC_EXPECT_OK(status);

  loomc_link_index_t* link_index = nullptr;
  loomc_result_t* result = nullptr;
  status = loomc_link_index_builder_finish(builder.get(), &link_index, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return LinkIndexPtr(link_index);
}

TEST(TargetTest, RetainReleaseProfileAndSelection) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  loomc_target_profile_retain(profile.get());
  loomc_target_profile_release(profile.get());

  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  loomc_target_selection_retain(selection.get());
  loomc_target_selection_release(selection.get());

  TargetSelectionPtr empty_selection = CreateEmptySelection();
  loomc_target_selection_retain(empty_selection.get());
  loomc_target_selection_release(empty_selection.get());
}

TEST(TargetTest, AcceptsExplicitTargetlessSelectionWithoutTargetEnvironment) {
  ContextPtr context = CreateContext();
  TargetSelectionPtr selection = CreateEmptySelection();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
}

TEST(TargetTest, RejectsProfileSelectionWithoutTargetEnvironment) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateContext();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, AcceptsProfileSelectionWithCompatibleTargetEnvironment) {
  TargetEnvironmentPtr profile_environment = CreateSpirvTargetEnvironment();
  TargetEnvironmentPtr context_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(profile_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(context_environment.get());
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &pass_options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
}

TEST(TargetTest, RejectsDuplicateTargetSelectionOptions) {
  ContextPtr context = CreateContext();
  TargetSelectionPtr selection = CreateEmptySelection();
  loomc_target_selection_options_t second_target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(second_target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_target_selection_options_t first_target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(first_target_options),
      /*.next=*/&second_target_options,
      /*.target_selection=*/selection.get(),
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&first_target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, RejectsTargetSelectionOptionsWithoutSelection) {
  ContextPtr context = CreateContext();
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/nullptr,
  };
  loomc_pass_program_options_t pass_options =
      PassOptionsWithSelection(&target_options);

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, ReusesSelectionAcrossCompileWorkspaces) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program =
      CreateEmptyPassProgramWithSelection(context.get(), selection.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/0,
      /*.config=*/{},
  };

  for (int i = 0; i < 2; ++i) {
    WorkspacePtr workspace = CreateWorkspace();
    ModulePtr module =
        CreateIdentityModule(context.get(), workspace.get(), "entry");
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_compile_module(
        compiler.get(), workspace.get(), pass_program.get(), module.get(),
        &compile_options, loomc_allocator_system(), &result);
    LOOMC_EXPECT_OK(status);
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
  }
}

TEST(TargetTest, ReusesSelectionAcrossLinkWorkspaces) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());
  LinkerPtr linker = CreateLinker(context.get());
  LinkIndexPtr link_index = CreateSingleSourceLinkIndex(context.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      /*.structure_size=*/sizeof(link_options),
      /*.next=*/&target_options,
      /*.link_index=*/link_index.get(),
      /*.module_name=*/loomc_make_cstring_view("linked_jit_module"),
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/LOOMC_LINK_FLAG_INCLUDE_EXPORTED_ROOTS,
  };

  for (int i = 0; i < 2; ++i) {
    WorkspacePtr workspace = CreateWorkspace();
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_link_module(linker.get(), workspace.get(),
                                              &link_options, &module, &result);
    LOOMC_EXPECT_OK(status);
    ModulePtr module_ptr(module);
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
    EXPECT_NE(module_ptr.get(), nullptr);
  }
}

}  // namespace
