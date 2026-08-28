// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/io/byte_sequence.h"
#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/target/function_version.h"
#include "loom/target/profile.h"
#include "loom/target/provider.h"
#include "loom/target/test/target_records.h"
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
#include "loomc/workspace.h"
#include "module.h"
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
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

typedef struct FakeArtifactSidecarStorage {
  // Allocator owning this storage.
  iree_allocator_t allocator;

  // Fake manifest sidecar descriptor.
  loom_target_emit_sidecar_artifact_t sidecar;
} FakeArtifactSidecarStorage;

void FakeArtifactSidecarStorageRelease(void* storage) {
  auto* artifact_storage = static_cast<FakeArtifactSidecarStorage*>(storage);
  iree_allocator_free(artifact_storage->allocator, artifact_storage);
}

iree_status_t CreateFakeArtifactContents(
    iree_const_byte_span_t source, iree_allocator_t allocator,
    iree_io_byte_sequence_t** out_contents) {
  *out_contents = nullptr;
  void* data = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(allocator, source, &data));
  iree_byte_span_t contents = iree_make_byte_span(data, source.data_length);
  iree_status_t status = iree_io_byte_sequence_create_from_span_move(
      &contents, allocator, out_contents);
  iree_allocator_free(allocator, contents.data);
  return status;
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
  static const uint8_t kContents[] = {0x7F, 'L', 'O', 'M'};
  out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  if (request->artifact_manifest.mode ==
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    iree_io_byte_sequence_t* contents = nullptr;
    IREE_RETURN_IF_ERROR(CreateFakeArtifactContents(
        iree_make_const_byte_span(kContents, sizeof(kContents)),
        request->allocator, &contents));
    out_artifact->contents = contents;
    return iree_ok_status();
  }

  FakeArtifactSidecarStorage* storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      request->allocator, sizeof(*storage), (void**)&storage));
  *storage = {};
  storage->allocator = request->allocator;
  iree_io_byte_sequence_t* contents = nullptr;
  iree_status_t status = CreateFakeArtifactContents(
      iree_make_const_byte_span(kContents, sizeof(kContents)),
      request->allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = CreateFakeArtifactContents(
        iree_make_const_byte_span(kManifestJson, sizeof(kManifestJson) - 1),
        request->allocator, &storage->sidecar.contents);
  }
  if (!iree_status_is_ok(status)) {
    iree_io_byte_sequence_release(contents);
    iree_io_byte_sequence_release(storage->sidecar.contents);
    FakeArtifactSidecarStorageRelease(storage);
    return status;
  }
  storage->sidecar.kind =
      LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST;
  storage->sidecar.identifier = request->artifact_manifest.identifier;
  out_artifact->contents = contents;
  out_artifact->sidecars = &storage->sidecar;
  out_artifact->sidecar_count = 1;
  out_artifact->storage = storage;
  out_artifact->release_storage = FakeArtifactSidecarStorageRelease;
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

typedef struct TestTargetProfile {
  // Generic target profile base.
  loom_target_profile_t base;

  // Exact subgroup size projected by this profile.
  uint32_t subgroup_size;
} TestTargetProfile;

static iree_status_t ProjectTestTargetProfileFacts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)arena;
  const auto* profile =
      reinterpret_cast<const TestTargetProfile*>(base_profile);
  out_facts->selector = LOOM_TEST_TARGET_KIND_LOW_CORE;
  if (profile->subgroup_size != 0) {
    out_facts->storage.snapshot.subgroup_size = profile->subgroup_size;
    loom_target_fact_field_set_insert(&out_facts->explicit_fields,
                                      LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  }
  return iree_ok_status();
}

static const loom_target_profile_type_t kTestTargetProfileType = {
    /*.name=*/IREE_SVL("loomc-target-test"),
    /*.fact_type=*/&loom_test_target_fact_type,
    /*.project_facts=*/ProjectTestTargetProfileFacts,
};

static iree_status_t RegisterTestTargetContext(loom_context_t* context) {
  return loom_test_dialect_register(context);
}

// Deliberately lacks a target-definition materializer so generic IR boundaries
// can prove loss prevention without relying on an incomplete production
// target.
static const loom_target_provider_t kTestTargetProvider = {
    /*.profile_type=*/&kTestTargetProfileType,
    /*.materialize_definition=*/nullptr,
    /*.register_context=*/RegisterTestTargetContext,
};

static const loom_target_provider_t* const kTestTargetProviders[] = {
    &kTestTargetProvider,
};

static const loom_target_provider_set_t kTestTargetProviderSet = {
    /*.providers=*/kTestTargetProviders,
    /*.provider_count=*/IREE_ARRAYSIZE(kTestTargetProviders),
};

static void DeinitializeTestTargetProfile(loom_target_profile_t* base_profile,
                                          loomc_allocator_t allocator) {
  loomc_allocator_free(allocator, base_profile);
}

static const loom_target_provider_t kFakeElfProvider = {
    /*.profile_type=*/nullptr,
    /*.materialize_definition=*/nullptr,
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
    /*.profile_type=*/nullptr,
    /*.materialize_definition=*/nullptr,
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

PassProgramPtr CreateEmptyPassProgram(loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context, nullptr, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  return PassProgramPtr(pass_program);
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

TargetEnvironmentPtr CreateTargetEnvironmentFromProviderSet(
    const loom_target_provider_set_t* provider_set) {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_from_provider_set(
      provider_set, loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

TargetEnvironmentPtr CreateTestTargetEnvironment() {
  return CreateTargetEnvironmentFromProviderSet(&kTestTargetProviderSet);
}

ContextPtr CreateTargetContext(loomc_target_environment_t* target_environment) {
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

TargetProfilePtr CreateTestTargetProfile(
    loomc_target_environment_t* target_environment,
    uint32_t subgroup_size = 0) {
  loomc_allocator_t allocator = loomc_allocator_system();
  TestTargetProfile* internal_profile = nullptr;
  LOOMC_EXPECT_OK(loomc_allocator_malloc(allocator, sizeof(*internal_profile),
                                         (void**)&internal_profile));
  IREE_ASSERT(internal_profile != nullptr);
  *internal_profile = TestTargetProfile{
      /*.base=*/
      {
          /*.type=*/&kTestTargetProfileType,
          /*.target_bundle=*/
          loom_target_bundle_table_lookup(&loom_test_target_bundles,
                                          LOOM_TEST_TARGET_KIND_LOW_CORE),
      },
      /*.subgroup_size=*/subgroup_size,
  };
  IREE_ASSERT(internal_profile->base.target_bundle != nullptr);
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create(
      target_environment, loomc_make_cstring_view("test-low-core"),
      &internal_profile->base, DeinitializeTestTargetProfile, allocator,
      &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
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

PassProgramPtr CreatePassProgramFromPipelineText(loomc_context_t* context,
                                                 const char* pipeline_text) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view(pipeline_text), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return PassProgramPtr(pass_program);
}

loom_func_like_t FindFunction(loomc_module_t* module, const char* name) {
  loom_module_t* internal_module = loomc_module_loom_module(module);
  IREE_ASSERT(internal_module != nullptr);
  const loom_string_id_t name_id =
      loom_module_lookup_string(internal_module, iree_make_cstring_view(name));
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(internal_module, name_id);
  IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
  loom_func_like_t function = loom_func_like_cast(
      internal_module, internal_module->symbols.entries[symbol_id].defining_op);
  IREE_ASSERT(loom_func_like_isa(function));
  return function;
}

const loom_target_function_version_t* FindTargetVersion(loomc_module_t* module,
                                                        const char* name) {
  const loom_function_version_list_t* function_versions =
      loomc_module_function_versions(module);
  return function_versions != nullptr
             ? loom_target_function_version_list_find(
                   function_versions, FindFunction(module, name))
             : nullptr;
}

PassProgramPtr CreateTargetPipelinePassProgram(
    loomc_context_t* context, const loomc_target_pipeline_options_t* options) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context, options, loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return PassProgramPtr(pass_program);
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
  EXPECT_NE(contents.find("\"status\":{\"code\":0,\"name\":\"OK\"}"),
            std::string::npos);
  EXPECT_NE(contents.find("\"backend\":\"fake-elf\""), std::string::npos);
  EXPECT_NE(contents.find("\"artifact_format\":\"elf\""), std::string::npos);
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
  loomc_link_index_provider_options_t source_options = {
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

TEST(TargetTest, RetainReleaseProfile) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  loomc_target_profile_retain(profile.get());
  loomc_target_profile_release(profile.get());
}

TEST(TargetTest, AcceptsSanitizerPipelineOptions) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  ContextPtr context = CreateTargetContext(target_environment.get());
  loomc_sanitizer_options_t sanitizer_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      /*.structure_size=*/sizeof(sanitizer_options),
      /*.next=*/nullptr,
      /*.checks=*/LOOMC_SANITIZER_CHECKS_ASAN_LIKE |
          LOOMC_SANITIZER_CHECKS_UBSAN_LIKE | LOOMC_SANITIZER_CHECKS_TSAN_LIKE,
      /*.flags=*/0,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/&sanitizer_options,
      /*.identifier=*/loomc_make_cstring_view("sanitized"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/0,
  };
  PassProgramPtr pass_program =
      CreateTargetPipelinePassProgram(context.get(), &pipeline_options);
  EXPECT_NE(pass_program.get(), nullptr);
}

TEST(TargetTest, RejectsUnknownSanitizerCheckBits) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  ContextPtr context = CreateTargetContext(target_environment.get());
  loomc_sanitizer_options_t sanitizer_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      /*.structure_size=*/sizeof(sanitizer_options),
      /*.next=*/nullptr,
      /*.checks=*/1ull << 63,
      /*.flags=*/0,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/&sanitizer_options,
      /*.identifier=*/loomc_make_cstring_view("bad-sanitizer"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/0,
  };

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_pass_program_create_from_target_pipeline(
                             context.get(), &pipeline_options,
                             loomc_allocator_system(), &pass_program, &result));
  EXPECT_EQ(pass_program, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(TargetTest, RejectsUnknownSanitizerReportingMode) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  ContextPtr context = CreateTargetContext(target_environment.get());
  loomc_sanitizer_options_t sanitizer_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      /*.structure_size=*/sizeof(sanitizer_options),
      /*.next=*/nullptr,
      /*.checks=*/LOOMC_SANITIZER_CHECK_ACCESS,
      /*.flags=*/0,
      /*.reporting_mode=*/(loomc_sanitizer_reporting_mode_t)99,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/&sanitizer_options,
      /*.identifier=*/loomc_make_cstring_view("bad-sanitizer-reporting"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/0,
  };

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_pass_program_create_from_target_pipeline(
                             context.get(), &pipeline_options,
                             loomc_allocator_system(), &pass_program, &result));
  EXPECT_EQ(pass_program, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST(TargetTest, RejectsSanitizerOptionsOnPlainPassProgramOptions) {
  ContextPtr context = CreateContext();
  loomc_sanitizer_options_t sanitizer_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      /*.structure_size=*/sizeof(sanitizer_options),
      /*.next=*/nullptr,
      /*.checks=*/LOOMC_SANITIZER_CHECK_ACCESS,
      /*.flags=*/0,
  };
  loomc_pass_program_options_t pass_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(pass_options),
      /*.next=*/&sanitizer_options,
      /*.identifier=*/loomc_make_cstring_view("plain-pass-program"),
  };

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_pass_program_create_empty(context.get(), &pass_options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, CompileBindsTargetDeclarationsAcrossCommandBoundaries) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr host_profile =
      CreateTestTargetProfile(target_environment.get(), 64);
  TargetProfilePtr shared_device_profile =
      CreateTestTargetProfile(target_environment.get(), 32);
  TargetProfilePtr decode_profile =
      CreateTestTargetProfile(target_environment.get(), 7);
  ContextPtr context = CreateTargetContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePassProgramFromPipelineText(
      context.get(), "specialize-target-callgraph");
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("target_bindings.loom", R"(
target.decl @prefill_device
target.decl @decode_device
target.decl @batch_device

func.def @shared_subgroup_size() -> (index) {
  %size = target.subgroup.size : index
  func.return %size : index
}

kernel.def @prefill_kernel() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  %size = func.call @shared_subgroup_size() : () -> (index)
  kernel.return
}

kernel.def @decode_kernel() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  %size = func.call @shared_subgroup_size() : () -> (index)
  kernel.return
}

command.program.def public target(@prefill_device) @prefill_program() launch() {
  kernel.launch @prefill_kernel() : ()
  command.return
}

command.program.def public target(@decode_device) @decode_program() launch() {
  kernel.launch @decode_kernel() : ()
  command.return
}

command.program.def public target(@batch_device) @batch_program() launch() {
  kernel.launch @prefill_kernel() : ()
  command.return
}

func.def public @host() {
  command.program.launch @prefill_program() : ()
  command.program.launch @decode_program() : ()
  command.program.launch @batch_program() : ()
  func.return
}

func.def public @unbound() {
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("host"),
      /*.target_profile=*/host_profile.get(),
  };
  const loomc_target_binding_t target_bindings[] = {
      {
          /*.target_symbol=*/loomc_make_cstring_view("@prefill_device"),
          /*.target_profile=*/shared_device_profile.get(),
      },
      {
          /*.target_symbol=*/loomc_make_cstring_view("batch_device"),
          /*.target_profile=*/shared_device_profile.get(),
      },
      {
          /*.target_symbol=*/loomc_make_cstring_view("decode_device"),
          /*.target_profile=*/decode_profile.get(),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
      /*.target_bindings=*/target_bindings,
      /*.target_binding_count=*/IREE_ARRAYSIZE(target_bindings),
  };
  const loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
  };

  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_EQ(loomc_result_artifact_count(result_ptr.get()), 1u);
  const loomc_artifact_t* report =
      loomc_result_artifact_at(result_ptr.get(), 0);
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(ToString(report->format), LOOMC_ARTIFACT_FORMAT_JSON);
  const std::string report_contents = ToString(report->contents);
  EXPECT_NE(report_contents.find("\"target_specialization_count\":1"),
            std::string::npos);
  EXPECT_NE(report_contents.find("\"target_binding_count\":3"),
            std::string::npos);

  const loom_function_version_list_t* function_versions =
      loomc_module_function_versions(module.get());
  ASSERT_NE(function_versions, nullptr);
  ASSERT_EQ(function_versions->count, 8u);
  const loom_target_function_version_t* host =
      FindTargetVersion(module.get(), "host");
  const loom_target_function_version_t* prefill_program =
      FindTargetVersion(module.get(), "prefill_program");
  const loom_target_function_version_t* decode_program =
      FindTargetVersion(module.get(), "decode_program");
  const loom_target_function_version_t* batch_program =
      FindTargetVersion(module.get(), "batch_program");
  const loom_target_function_version_t* prefill_kernel =
      FindTargetVersion(module.get(), "prefill_kernel");
  const loom_target_function_version_t* decode_kernel =
      FindTargetVersion(module.get(), "decode_kernel");
  ASSERT_NE(host, nullptr);
  ASSERT_NE(prefill_program, nullptr);
  ASSERT_NE(decode_program, nullptr);
  ASSERT_NE(batch_program, nullptr);
  ASSERT_NE(prefill_kernel, nullptr);
  ASSERT_NE(decode_kernel, nullptr);
  EXPECT_EQ(host->resolved_target.facts->storage.snapshot.subgroup_size, 64u);
  EXPECT_EQ(
      prefill_program->resolved_target.facts->storage.snapshot.subgroup_size,
      32u);
  EXPECT_EQ(
      decode_program->resolved_target.facts->storage.snapshot.subgroup_size,
      7u);
  EXPECT_EQ(
      batch_program->resolved_target.facts->storage.snapshot.subgroup_size,
      32u);
  EXPECT_NE(host->target_context_ordinal,
            prefill_program->target_context_ordinal);
  EXPECT_NE(prefill_program->target_context_ordinal,
            decode_program->target_context_ordinal);
  EXPECT_EQ(prefill_program->target_context_ordinal,
            batch_program->target_context_ordinal);
  EXPECT_EQ(prefill_program->target_context_ordinal,
            prefill_kernel->target_context_ordinal);
  EXPECT_EQ(decode_program->target_context_ordinal,
            decode_kernel->target_context_ordinal);
  EXPECT_EQ(batch_program->target_context_ordinal,
            prefill_kernel->target_context_ordinal);
  EXPECT_TRUE(iree_string_view_equal(prefill_program->authored_target_name,
                                     IREE_SV("prefill_device")));
  EXPECT_TRUE(iree_string_view_equal(decode_program->authored_target_name,
                                     IREE_SV("decode_device")));
  EXPECT_EQ(FindTargetVersion(module.get(), "unbound"), nullptr);

  const loom_module_t* internal_module =
      loomc_module_const_loom_module(module.get());
  iree_host_size_t shared_version_count = 0;
  bool saw_shared_wave32 = false;
  bool saw_shared_wave7 = false;
  for (iree_host_size_t i = 0; i < function_versions->count; ++i) {
    const loom_target_function_version_t* target_version =
        loom_target_function_version_const_cast(function_versions->values[i]);
    ASSERT_NE(target_version, nullptr);
    const loom_symbol_ref_t function_ref =
        loom_func_like_callee(target_version->base.function);
    ASSERT_TRUE(loom_symbol_ref_is_valid(function_ref));
    const loom_symbol_t* function_symbol =
        &internal_module->symbols.entries[function_ref.symbol_id];
    const iree_string_view_t function_name =
        internal_module->strings.entries[function_symbol->name_id];
    const std::string function_name_string(function_name.data,
                                           function_name.size);
    if (function_name_string.rfind("shared_subgroup_size", 0) != 0) {
      continue;
    }
    ++shared_version_count;
    const uint32_t subgroup_size =
        target_version->resolved_target.facts->storage.snapshot.subgroup_size;
    saw_shared_wave32 |= subgroup_size == 32;
    saw_shared_wave7 |= subgroup_size == 7;
  }
  EXPECT_EQ(shared_version_count, 2u);
  EXPECT_TRUE(saw_shared_wave32);
  EXPECT_TRUE(saw_shared_wave7);
}

TEST(TargetTest, CompileRejectsInvalidTargetDeclarationBindings) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  ContextPtr context = CreateTargetContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("invalid_target_bindings.loom", R"(
target.decl @device
target.decl @unused
test.target<low_core> @concrete

func.def public target(@device) @entry() {
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
  };
  const loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
  };

  const auto expect_rejected =
      [&](loomc_status_code_t expected_code,
          const loomc_target_specialization_t* specializations,
          loomc_host_size_t specialization_count,
          const loomc_target_binding_t* target_bindings,
          loomc_host_size_t target_binding_count) {
        target_options.specializations = specializations;
        target_options.specialization_count = specialization_count;
        target_options.target_bindings = target_bindings;
        target_options.target_binding_count = target_binding_count;
        loomc_result_t* result = nullptr;
        LOOMC_EXPECT_STATUS_IS(
            expected_code,
            loomc_compile_module(compiler.get(), workspace.get(),
                                 pass_program.get(), module.get(),
                                 &compile_options, loomc_allocator_system(),
                                 &result));
        EXPECT_EQ(result, nullptr);
        EXPECT_EQ(loomc_module_function_versions(module.get()), nullptr);
      };

  expect_rejected(LOOMC_STATUS_INVALID_ARGUMENT, nullptr, 0, nullptr, 1);

  const loomc_target_binding_t missing_profile_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("device"),
      /*.target_profile=*/nullptr,
  };
  expect_rejected(LOOMC_STATUS_INVALID_ARGUMENT, nullptr, 0,
                  &missing_profile_binding, 1);

  const loomc_target_binding_t missing_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("missing"),
      /*.target_profile=*/profile.get(),
  };
  expect_rejected(LOOMC_STATUS_NOT_FOUND, nullptr, 0, &missing_binding, 1);

  const loomc_target_binding_t concrete_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("concrete"),
      /*.target_profile=*/profile.get(),
  };
  expect_rejected(LOOMC_STATUS_INVALID_ARGUMENT, nullptr, 0, &concrete_binding,
                  1);

  const loomc_target_binding_t duplicate_bindings[] = {
      {
          /*.target_symbol=*/loomc_make_cstring_view("device"),
          /*.target_profile=*/profile.get(),
      },
      {
          /*.target_symbol=*/loomc_make_cstring_view("@device"),
          /*.target_profile=*/profile.get(),
      },
  };
  expect_rejected(LOOMC_STATUS_INVALID_ARGUMENT, nullptr, 0, duplicate_bindings,
                  IREE_ARRAYSIZE(duplicate_bindings));

  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("entry"),
      /*.target_profile=*/profile.get(),
  };
  const loomc_target_binding_t overlapping_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("device"),
      /*.target_profile=*/profile.get(),
  };
  expect_rejected(LOOMC_STATUS_INVALID_ARGUMENT, &specialization, 1,
                  &overlapping_binding, 1);

  const loomc_target_binding_t unused_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("unused"),
      /*.target_profile=*/profile.get(),
  };
  target_options.specializations = nullptr;
  target_options.specialization_count = 0;
  target_options.target_bindings = &unused_binding;
  target_options.target_binding_count = 1;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(loomc_module_function_versions(module.get()), nullptr);
}

TEST(TargetTest, CompileReportsIncompatibleBoundCalleeTarget) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile =
      CreateTestTargetProfile(target_environment.get(), 32);
  ContextPtr context = CreateTargetContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePassProgramFromPipelineText(
      context.get(), "specialize-target-callgraph");
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("incompatible_target_binding.loom", R"(
target.decl @device
test.target<quirky> @quirky

func.def target(@quirky) @helper() {
  func.return
}

func.def public target(@device) @entry() {
  func.call @helper() : ()
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_target_binding_t binding = {
      /*.target_symbol=*/loomc_make_cstring_view("device"),
      /*.target_profile=*/profile.get(),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/nullptr,
      /*.specialization_count=*/0,
      /*.target_bindings=*/&binding,
      /*.target_binding_count=*/1,
  };
  const loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
  };

  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ASSERT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "TARGET/052");
  EXPECT_EQ(loomc_module_function_versions(module.get()), nullptr);
}

TEST(TargetTest, RejectsSerializationWithoutATargetMaterializer) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  ContextPtr context = CreateTargetContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());

  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("entry"),
      /*.target_profile=*/profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
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
    loomc_workspace_trim(workspace.get());
    const loom_function_version_list_t* function_versions =
        loomc_module_function_versions(module.get());
    ASSERT_NE(function_versions, nullptr);
    ASSERT_EQ(function_versions->count, 1u);
    ASSERT_NE(function_versions->values[0], nullptr);
    EXPECT_NE(function_versions->values[0]->type, nullptr);
    EXPECT_NE(function_versions->values[0]->function.op, nullptr);
    loomc_module_serialize_options_t serialize_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
        /*.structure_size=*/sizeof(serialize_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view("module.loom"),
    };
    loomc_source_t* serialized_source = nullptr;
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                           loomc_module_serialize_to_source(
                               module.get(), &serialize_options,
                               loomc_allocator_system(), &serialized_source));
    EXPECT_EQ(serialized_source, nullptr);
    EXPECT_NE(loomc_module_function_versions(module.get()), nullptr);

    WorkspacePtr clone_workspace = CreateWorkspace();
    loomc_module_t* clone = nullptr;
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_FAILED_PRECONDITION,
        loomc_module_clone(module.get(), clone_workspace.get(),
                           loomc_allocator_system(), &clone));
    EXPECT_EQ(clone, nullptr);
    EXPECT_NE(loomc_module_function_versions(module.get()), nullptr);

    FILE* file = tmpfile();
    ASSERT_NE(file, nullptr);
    static constexpr char kSentinel[] = "unchanged";
    ASSERT_EQ(fwrite(kSentinel, 1, sizeof(kSentinel) - 1, file),
              sizeof(kSentinel) - 1);
    const long file_position = ftell(file);
    ASSERT_GE(file_position, 0);
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_FAILED_PRECONDITION,
        loomc_module_serialize_to_file(module.get(), &serialize_options, file));
    EXPECT_EQ(ftell(file), file_position);
    ASSERT_EQ(fseek(file, 0, SEEK_SET), 0);
    char file_contents[sizeof(kSentinel)] = {};
    EXPECT_EQ(fread(file_contents, 1, sizeof(kSentinel) - 1, file),
              sizeof(kSentinel) - 1);
    EXPECT_STREQ(file_contents, kSentinel);
    fclose(file);

    iree::testing::TempFilePath path("loomc-unsealed-target", ".loom");
    FILE* path_file = fopen(path.path().c_str(), "wb");
    ASSERT_NE(path_file, nullptr);
    ASSERT_EQ(fwrite(kSentinel, 1, sizeof(kSentinel) - 1, path_file),
              sizeof(kSentinel) - 1);
    ASSERT_EQ(fclose(path_file), 0);
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_FAILED_PRECONDITION,
        loomc_module_serialize_to_path(
            module.get(), &serialize_options,
            loomc_make_string_view(path.path().data(), path.path().size()),
            loomc_allocator_system()));
    path_file = fopen(path.path().c_str(), "rb");
    ASSERT_NE(path_file, nullptr);
    char path_contents[sizeof(kSentinel)] = {};
    EXPECT_EQ(fread(path_contents, 1, sizeof(kSentinel) - 1, path_file),
              sizeof(kSentinel) - 1);
    EXPECT_STREQ(path_contents, kSentinel);
    ASSERT_EQ(fclose(path_file), 0);
    EXPECT_TRUE(path.Remove());

    const loomc_target_specialization_t missing_specialization = {
        /*.function_symbol=*/loomc_make_cstring_view("missing"),
        /*.target_profile=*/profile.get(),
    };
    target_options.specializations = &missing_specialization;
    result = nullptr;
    LOOMC_EXPECT_STATUS_IS(
        LOOMC_STATUS_NOT_FOUND,
        loomc_compile_module(compiler.get(), workspace.get(),
                             pass_program.get(), module.get(), &compile_options,
                             loomc_allocator_system(), &result));
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(loomc_module_function_versions(module.get()), nullptr);
    target_options.specializations = &specialization;
  }
}

TEST(TargetTest, CompileRejectsAnUnrepresentableModuleArtifact) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  ContextPtr context = CreateTargetContext(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());

  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("entry"),
      /*.target_profile=*/profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("jit_kernel"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
      /*.config=*/{},
  };

  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &result);

  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION, status);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(loomc_module_function_versions(module.get()), nullptr);
}

TEST(TargetTest, RejectsSpecializationOptionsOnPassProgramCreation) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  ContextPtr context = CreateTargetContext(target_environment.get());
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("entry"),
      /*.target_profile=*/profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  loomc_pass_program_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };

  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_pass_program_create_empty(context.get(), &options,
                                      loomc_allocator_system(), &pass_program));
  EXPECT_EQ(pass_program, nullptr);
}

TEST(TargetTest, AcceptsEmptySpecializationOptionsDuringLink) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  ContextPtr context = CreateTargetContext(target_environment.get());
  LinkerPtr linker = CreateLinker(context.get());
  LinkIndexPtr link_index = CreateSingleSourceLinkIndex(context.get());

  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/nullptr,
      /*.specialization_count=*/0,
  };
  loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      /*.structure_size=*/sizeof(link_options),
      /*.next=*/&target_options,
      /*.link_index=*/link_index.get(),
      /*.module_name=*/loomc_make_cstring_view("linked_jit_module"),
      /*.mode=*/LOOMC_LINK_MODE_LINK,
      /*.root_symbols=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS,
  };

  WorkspacePtr workspace = CreateWorkspace();
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_link_module(linker.get(), workspace.get(),
                                    &link_options, &module, &result));
  ModulePtr module_ptr(module);
  ResultPtr result_ptr(result);
  ASSERT_NE(module_ptr, nullptr);
  ExpectSucceededResult(result_ptr.get());
}

TEST(TargetTest, RejectsSpecializationOptionsDuringEmission) {
  TargetEnvironmentPtr target_environment = CreateTestTargetEnvironment();
  TargetProfilePtr profile = CreateTestTargetProfile(target_environment.get());
  ContextPtr context = CreateTargetContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreateIdentityModule(context.get(), workspace.get(), "entry");
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("entry"),
      /*.target_profile=*/profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };

  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNIMPLEMENTED,
      loomc_emit_module(target_environment.get(), workspace.get(), module.get(),
                        &options, loomc_allocator_system(), &result));
  EXPECT_EQ(result, nullptr);
}

}  // namespace
