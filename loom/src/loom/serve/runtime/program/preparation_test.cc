// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/preparation.h"

#include <memory>
#include <string>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/model/card.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, boundary_source, "", "Preparation boundary callers.");

namespace {

class PreparationTest : public ::testing::Test {
 protected:
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation = {};

  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator), &target_environment)));
    IREE_ASSERT_OK(iree_vm_environment_allocate(host_allocator, &environment));
  }

  void TearDown() override {
    loom_serve_preparation_free(preparation);
    iree_vm_environment_free(environment);
    loomc_target_environment_release(target_environment);
  }

  // Allocation policy for the environment and owned declaration copies.
  const iree_allocator_t host_allocator = iree_allocator_system();
  // Explicit local namespace for the authored boundary fixture.
  const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
  // Environment owning the reference types used during preparation.
  iree_vm_environment_t* environment = nullptr;
  // VM-only grammar for the authored capability fixtures.
  loomc_target_environment_t* target_environment = nullptr;
  // Owned declarations independent of the temporary source program.
  loom_serve_preparation_t* preparation = nullptr;
};

TEST_F(PreparationTest, TextConfigurationCopiesValueSpelling) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("text_config"),
      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS, iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &compilation,
      host_allocator, &preparation));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->kind, LOOM_SERVE_PREPARATION_STAGE_COMMAND);
  ASSERT_EQ(stage->config.binding_count, 1);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, DirectDeclarationRetainsProductAndSpecialization) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("direct_config"),
      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS, iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &compilation,
      host_allocator, &preparation));
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 1u);
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->kind, LOOM_SERVE_PREPARATION_STAGE_DISPATCH);
  EXPECT_EQ(stage->tag, 17);
  EXPECT_EQ(std::string(stage->root.data, stage->root.size), "stage_root");
  ASSERT_EQ(stage->config.binding_count, 1u);
  const auto& value = stage->config.bindings[0].value;
  EXPECT_EQ(std::string(value.data, value.size), "true");
}

TEST_F(PreparationTest, WorkspaceLimitConstrainsCompilerReflection) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source),
      IREE_SV("workspace_constrained"), LOOM_SERVE_PREPARATION_FLAG_PARAMETERS,
      iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
      iree_vm_module_span_empty(), &compilation, host_allocator, &preparation));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  EXPECT_EQ(stage->workspace_limit_bytes, 1073741824u);
  IREE_EXPECT_OK(loom_serve_preparation_stage_check_workspace(
      stage, stage->workspace_limit_bytes));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_preparation_stage_check_workspace(
                            stage, stage->workspace_limit_bytes + 1));
}

TEST_F(PreparationTest, CatalogGrowthPreservesEveryDeclaration) {
  // Bound the regression's failure mode: unconditional doubling would ask for
  // gigabytes before reporting a catalog that only needs kilobytes.
  const iree_allocator_t bounded_allocator = {
      nullptr,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** inout_pointer) -> iree_status_t {
        if (command != IREE_ALLOCATOR_COMMAND_FREE &&
            static_cast<const iree_allocator_alloc_params_t*>(params)
                    ->byte_length > 64 * 1024 * 1024) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "catalog allocation exceeded 64 MiB");
        }
        const auto system = iree_allocator_system();
        return system.ctl(system.self, command, params, inout_pointer);
      }};
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("large_catalog"),
      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS, iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &compilation,
      bounded_allocator, &preparation));
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 33);
  for (size_t i = 0; i < 33; ++i) {
    const auto* stage = loom_serve_preparation_stage(preparation, i);
    EXPECT_EQ(std::string(stage->root.data, stage->root.size), "stage_root");
    EXPECT_EQ(stage->tag, i);
    ASSERT_EQ(stage->config.binding_count, 33);
    ASSERT_EQ(stage->parameter_count, 33);
    for (size_t j = 0; j < 33; ++j) {
      const auto& binding = stage->config.bindings[j];
      EXPECT_EQ(std::string(binding.key.data, binding.key.size),
                std::string(j < 10 ? "k0" : "k") + std::to_string(j));
      EXPECT_EQ(std::string(binding.value.data, binding.value.size),
                std::to_string(j));
      const auto& parameter = stage->parameters[j];
      EXPECT_EQ(parameter.binding, j);
      ASSERT_EQ(parameter.source_count, 1);
      ASSERT_EQ(parameter.sources[0].count, 3);
      const auto* paths = parameter.sources[0].values;
      EXPECT_EQ(std::string(paths[0].path.data, paths[0].path.size),
                "/checkpoints/model.safetensors");
      EXPECT_EQ(std::string(paths[1].path.data, paths[1].path.size),
                std::string(j < 10 ? "/checkpoints/k0" : "/checkpoints/k") +
                    std::to_string(j));
      EXPECT_EQ(std::string(paths[2].path.data, paths[2].path.size),
                "/checkpoints/weights.loom");
      EXPECT_EQ(std::string(parameter.policy.data, parameter.policy.size),
                "weights.loom");
    }
  }
}

TEST_F(PreparationTest, CheckpointIdentityIncludesEveryFileAndPolicy) {
  const loom_model_file_t files[] = {{IREE_SV("first.gguf"), {}},
                                     {IREE_SV("second.gguf"), {}}};
  const loom_model_file_list_t sources[] = {{IREE_ARRAYSIZE(files), files}};
  const loom_serve_preparation_parameter_t base = {0, 1, sources,
                                                   IREE_SV("weights.loom")};
  auto peer = base;
  peer.binding = 1;
  EXPECT_TRUE(loom_serve_preparation_parameter_equal(&base, &peer));
  const loom_model_file_t different[] = {files[0], {IREE_SV("other.gguf"), {}}};
  loom_model_file_list_t peer_sources[] = {{2, different}, {1, files}};
  peer.sources = peer_sources;
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer_sources[0] = {1, files};
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  const loom_model_file_t reversed[] = {files[1], files[0]};
  peer_sources[0] = {2, reversed};
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  const loom_model_file_t reinterpreted[] = {
      files[0], {files[1].path, IREE_SV("safetensors")}};
  peer_sources[0] = {2, reinterpreted};
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer_sources[0] = sources[0];
  EXPECT_TRUE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer.source_count = 2;
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
  peer = base;
  peer.policy = IREE_SV("adapter.loom");
  EXPECT_FALSE(loom_serve_preparation_parameter_equal(&base, &peer));
}

TEST_F(PreparationTest, IndependentSourcesPreserveShardOrderAndDuplicateNames) {
  IREE_ASSERT_OK(loom_serve_preparation_allocate(
      target_environment, environment, &sources,
      iree_make_cstring_view(FLAG_boundary_source), IREE_SV("many_sources"),
      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS, iree_vm_variant_span_empty(),
      iree_vm_variant_span_empty(), iree_vm_module_span_empty(), &compilation,
      host_allocator, &preparation));
  const auto* stage = loom_serve_preparation_stage(preparation, 0);
  ASSERT_EQ(stage->parameter_count, 1);
  const auto& parameter = stage->parameters[0];
  ASSERT_EQ(parameter.source_count, 33);
  for (uint32_t s = 0; s < parameter.source_count; ++s) {
    const auto& source = parameter.sources[s];
    ASSERT_EQ(source.count, 2);
    EXPECT_EQ(
        std::string(source.values[0].path.data, source.values[0].path.size),
        "/checkpoints/model.safetensors");
    EXPECT_EQ(
        std::string(source.values[1].path.data, source.values[1].path.size),
        "/checkpoints/weights.loom");
  }
}

TEST_F(PreparationTest, InvalidDeclarationsReleasePartialOwnership) {
  struct loom_serve_test_preparation_case_t {
    // Source entry that reaches a public declaration failure.
    const char* entry;
    // Terminal error from the declaration boundary.
    iree_status_code_t code;
  };
  const loom_serve_test_preparation_case_t cases[] = {
      {"missing_stage", IREE_STATUS_OUT_OF_RANGE},
      {"conflicting_parameter_policy", IREE_STATUS_INVALID_ARGUMENT},
      {"sparse_input", IREE_STATUS_INVALID_ARGUMENT},
      {"negative_input", IREE_STATUS_INVALID_ARGUMENT},
      {"duplicate_input", IREE_STATUS_INVALID_ARGUMENT},
      {"direct_parameter", IREE_STATUS_INVALID_ARGUMENT},
      {"missing_workspace_stage", IREE_STATUS_OUT_OF_RANGE},
      {"duplicate_workspace_limit", IREE_STATUS_ALREADY_EXISTS},
      {"direct_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"zero_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"negative_workspace_limit", IREE_STATUS_INVALID_ARGUMENT},
      {"no_stages", IREE_STATUS_INVALID_ARGUMENT}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.entry);
    IREE_EXPECT_STATUS_IS(
        test.code,
        loom_serve_preparation_allocate(
            target_environment, environment, &sources,
            iree_make_cstring_view(FLAG_boundary_source),
            iree_make_cstring_view(test.entry),
            LOOM_SERVE_PREPARATION_FLAG_PARAMETERS,
            iree_vm_variant_span_empty(), iree_vm_variant_span_empty(),
            iree_vm_module_span_empty(), &compilation, host_allocator,
            &preparation));
    EXPECT_EQ(preparation, nullptr);
  }
}

TEST_F(PreparationTest, EmbeddedNulCannotTruncateDeclarationStrings) {
  for (int32_t field = 0; field < 6; ++field) {
    SCOPED_TRACE(field);
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(field)};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_preparation_allocate(
            target_environment, environment, &sources,
            iree_make_cstring_view(FLAG_boundary_source), IREE_SV("nul_string"),
            LOOM_SERVE_PREPARATION_FLAG_PARAMETERS,
            iree_vm_variant_span_from_array(arguments),
            iree_vm_variant_span_empty(), iree_vm_module_span_empty(),
            &compilation, host_allocator, &preparation));
    EXPECT_EQ(preparation, nullptr);
  }
}

class PreparationCaptureTest : public PreparationTest {
 protected:
  void SetUp() override {
    PreparationTest::SetUp();
    IREE_ASSERT_OK(loom_serve_preparation_module_create(
        environment, card, nullptr, host_allocator, &module));
    const iree_string_view_t roots[] = {IREE_SV("request_stages"),
                                        IREE_SV("configure_cold_stage"),
                                        IREE_SV("conflicting_parameter_policy"),
                                        IREE_SV("no_stages"),
                                        IREE_SV("shared_source"),
                                        IREE_SV("scoped_source"),
                                        IREE_SV("cold_source"),
                                        IREE_SV("bind_source"),
                                        IREE_SV("append_source"),
                                        IREE_SV("declare_forecast"),
                                        IREE_SV("declare_arena"),
                                        IREE_SV("declare_parameter_binding"),
                                        IREE_SV("declare_banks")};
    IREE_ASSERT_OK(loom_serve_program_allocate(
        target_environment, environment, &sources,
        iree_make_cstring_view(FLAG_boundary_source), IREE_ARRAYSIZE(roots),
        roots, (iree_vm_module_span_t){&module, 1}, &compilation,
        host_allocator, &program));
  }

  void TearDown() override {
    loom_serve_preparation_free(second);
    loom_serve_program_free(program);
    iree_vm_module_release(module);
    loom_model_card_release(card);
    PreparationTest::TearDown();
  }

  iree_status_t Invoke(iree_string_view_t entry,
                       iree_vm_variant_span_t arguments = {},
                       iree_vm_variant_span_t results = {}) {
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), entry, &function));
    return iree_vm_invoke(loom_serve_program_invocation(program), function,
                          arguments, results);
  }

  iree_status_t DeclareSource(iree_string_view_t entry, iree_string_view_t path,
                              int32_t* out_ordinal) {
    iree_vm_ref_types_t types = {};
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    iree_vm_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(path.data, path.size), 1,
                             host_allocator, &buffer));
    iree_vm_variant_t argument =
        iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    iree_vm_variant_t result = {};
    auto status = Invoke(entry, {&argument, 1}, {&result, 1});
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(result, out_ordinal);
    }
    iree_vm_variant_reset(&argument);
    iree_vm_variant_reset(&result);
    return status;
  }

  iree_status_t DeclareArena(int64_t capacity, int32_t* out_ordinal) {
    iree_vm_variant_t argument = iree_vm_variant_from_i64(capacity);
    iree_vm_variant_t result = {};
    auto status =
        Invoke(IREE_SV("declare_arena"), {&argument, 1}, {&result, 1});
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(result, out_ordinal);
    }
    iree_vm_variant_reset(&result);
    return status;
  }

  iree_status_t DeclareBanks(int32_t arena, iree_const_byte_span_t bytes,
                             int32_t count, int32_t* out_ordinal) {
    iree_vm_ref_types_t types = {};
    IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    iree_vm_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ, bytes, 4, host_allocator, &buffer));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(arena),
        iree_vm_buffer_variant_from_ptr_move(&types, &buffer),
        iree_vm_variant_from_i32(count)};
    iree_vm_variant_t result = {};
    auto status =
        Invoke(IREE_SV("declare_banks"),
               iree_vm_variant_span_from_array(arguments), {&result, 1});
    if (iree_status_is_ok(status)) {
      status = iree_vm_i32_from_variant(result, out_ordinal);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_reset(&result);
    return status;
  }

  // A single linked source program serves multiple independent captures.
  loom_serve_program_t* program = nullptr;
  // Persistent capability retained independently of the linked program.
  iree_vm_module_t* module = nullptr;
  // Second capture kept live beside the first to prove independent ownership.
  loom_serve_preparation_t* second = nullptr;
  // Optional selected facts retained independently by the declaration module.
  loom_model_card_t* card = nullptr;
};

TEST_F(PreparationCaptureTest, NamedSourcesRequireASelectedCard) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("model"), &ordinal));
  preparation = loom_serve_preparation_module_take(module);
  EXPECT_EQ(loom_serve_preparation_source_count(preparation), 0u);
}

class PreparationCardCaptureTest : public PreparationCaptureTest {
 protected:
  void SetUp() override {
    constexpr char kCard[] = R"json({
      "schema_version": 1,
      "model": {"id": "fixture/text", "name": "Shared source fixture"},
      "programs": {"source": {"directory": "program",
        "inputs": ["program/prepare.loom"], "libraries": [],
        "config": {"width": 8}}},
      "services": {"chat": {"kind": "text", "version": 1}},
      "variants": {"small": {"services": {"chat": {
        "programs": ["source"],
        "parameters": {"model": {"files": ["weight/first.safetensors",
          {"path": "weight/second.data", "format": "safetensors"}]}}
      }}}}
    })json";
    IREE_ASSERT_OK(loom_model_card_parse(iree_make_cstring_view(kCard), nullptr,
                                         host_allocator, &card));
    PreparationCaptureTest::SetUp();
  }
};

TEST_F(PreparationCardCaptureTest, SharedTypedSourcesOutliveTheCardAndModule) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("model"), &ordinal));
  for (int i = 0; i < 2; ++i) {
    iree_vm_variant_t argument = iree_vm_variant_from_i32(ordinal);
    IREE_ASSERT_OK(Invoke(IREE_SV("bind_source"), {&argument, 1}));
  }
  preparation = loom_serve_preparation_module_take(module);
  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  loom_model_card_release(card);
  card = nullptr;

  ASSERT_EQ(loom_serve_preparation_source_count(preparation), 1u);
  const auto files = loom_serve_preparation_source(preparation, ordinal);
  ASSERT_EQ(files.count, 2u);
  EXPECT_EQ(std::string(files.values[0].path.data, files.values[0].path.size),
            "weight/first.safetensors");
  EXPECT_EQ(files.values[0].format.size, 0u);
  EXPECT_EQ(std::string(files.values[1].path.data, files.values[1].path.size),
            "weight/second.data");
  EXPECT_EQ(
      std::string(files.values[1].format.data, files.values[1].format.size),
      "safetensors");
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 2u);
  for (int i = 0; i < 2; ++i) {
    const auto* stage = loom_serve_preparation_stage(preparation, i);
    ASSERT_EQ(stage->parameter_count, 1u);
    const auto& parameter = stage->parameters[0];
    EXPECT_EQ(parameter.source_ordinals[0], ordinal);
    EXPECT_EQ(parameter.sources[0].values, files.values);
    EXPECT_EQ(std::string(parameter.policy.data, parameter.policy.size),
              "program/weights.loom");
    EXPECT_EQ(std::string(stage->config.json_object.data,
                          stage->config.json_object.size),
              "{\"width\":8}");
  }
}

TEST_F(PreparationCardCaptureTest, MissingScopesCannotDeclareASource) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("missing"), &ordinal));
  preparation = loom_serve_preparation_module_take(module);
  EXPECT_EQ(loom_serve_preparation_source_count(preparation), 0u);
}

TEST_F(PreparationCardCaptureTest, CardScopesOwnTheirCompleteFileLists) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("model"), &ordinal));
  iree_vm_variant_t argument = iree_vm_variant_from_i32(ordinal);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("append_source"), {&argument, 1}));
  preparation = loom_serve_preparation_module_take(module);
  EXPECT_EQ(loom_serve_preparation_source_count(preparation), 1u);
  EXPECT_EQ(loom_serve_preparation_source(preparation, ordinal).count, 2u);
}

TEST_F(PreparationCardCaptureTest, DeploymentSourcesExtendTheScopedCatalog) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t scoped = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("model"), &scoped));
  int32_t adapter = -1;
  IREE_ASSERT_OK(DeclareSource(IREE_SV("cold_source"),
                               IREE_SV("adapter.safetensors"), &adapter));
  iree_vm_variant_t argument = iree_vm_variant_from_i32(adapter);
  IREE_ASSERT_OK(Invoke(IREE_SV("append_source"), {&argument, 1}));
  argument = iree_vm_variant_from_i32(adapter);
  IREE_ASSERT_OK(Invoke(IREE_SV("bind_source"), {&argument, 1}));
  preparation = loom_serve_preparation_module_take(module);

  EXPECT_EQ(scoped, 0);
  EXPECT_EQ(adapter, 1);
  ASSERT_EQ(loom_serve_preparation_source_count(preparation), 2u);
  const auto scoped_files = loom_serve_preparation_source(preparation, scoped);
  ASSERT_EQ(scoped_files.count, 2u);
  EXPECT_EQ(std::string(scoped_files.values[1].format.data,
                        scoped_files.values[1].format.size),
            "safetensors");
  const auto adapter_files =
      loom_serve_preparation_source(preparation, adapter);
  ASSERT_EQ(adapter_files.count, 2u);
  EXPECT_EQ(std::string(adapter_files.values[0].path.data,
                        adapter_files.values[0].path.size),
            "/checkpoints/adapter.safetensors");
  EXPECT_EQ(std::string(adapter_files.values[1].path.data,
                        adapter_files.values[1].path.size),
            "/checkpoints/model.safetensors");
  const auto* stage = loom_serve_preparation_stage(preparation, 1);
  ASSERT_EQ(stage->parameter_count, 1u);
  EXPECT_EQ(stage->parameters[0].source_ordinals[0], adapter);
  EXPECT_EQ(stage->parameters[0].sources[0].values, adapter_files.values);
  EXPECT_EQ(std::string(stage->parameters[0].policy.data,
                        stage->parameters[0].policy.size),
            "program/weights.loom");

  loom_serve_preparation_module_begin(module, 2, preparation,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t requested = -1;
  IREE_ASSERT_OK(DeclareSource(IREE_SV("cold_source"),
                               IREE_SV("request.safetensors"), &requested));
  argument = iree_vm_variant_from_i32(requested);
  IREE_ASSERT_OK(Invoke(IREE_SV("append_source"), {&argument, 1}));
  for (int32_t ordinal : {scoped, adapter}) {
    argument = iree_vm_variant_from_i32(ordinal);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          Invoke(IREE_SV("append_source"), {&argument, 1}));
  }
  second = loom_serve_preparation_module_take(module);
  EXPECT_EQ(requested, 2);
  ASSERT_EQ(loom_serve_preparation_source_count(second), 3u);
  EXPECT_EQ(loom_serve_preparation_source(second, scoped).values,
            scoped_files.values);
  EXPECT_EQ(loom_serve_preparation_source(second, adapter).values,
            adapter_files.values);
  EXPECT_EQ(loom_serve_preparation_source(second, requested).count, 2u);
}

TEST_F(PreparationCardCaptureTest,
       CapturedBanksRetainTypedShardsAndPolicyBase) {
  loom_serve_preparation_module_begin(
      module, 0, nullptr,
      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS |
          LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS);
  int32_t source = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("scoped_source"), IREE_SV("model"), &source));
  int32_t arena = -1;
  IREE_ASSERT_OK(DeclareArena(4096, &arena));
  preparation = loom_serve_preparation_module_take(module);
  uint8_t ordinals[4] = {};
  iree_unaligned_store_le_u32(ordinals, source);
  loom_serve_preparation_module_begin(module, 0, preparation,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t last = -1;
  IREE_ASSERT_OK(
      DeclareBanks(arena, iree_make_const_byte_span(ordinals, 4), 2, &last));
  second = loom_serve_preparation_module_take(module);
  EXPECT_EQ(last, 1);
  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  loom_model_card_release(card);
  card = nullptr;
  for (uint32_t index = 0; index < 2; ++index) {
    const auto* bank = loom_serve_preparation_bank(second, index);
    ASSERT_EQ(bank->source_count, 1u);
    EXPECT_EQ(bank->source_ordinals[0], source);
    EXPECT_EQ(bank->arena, 1u);
    EXPECT_EQ(std::string(bank->policy.data, bank->policy.size),
              "program/weights.loom");
    ASSERT_EQ(bank->sources[0].count, 2u);
    const auto& shard = bank->sources[0].values[1];
    EXPECT_EQ(std::string(shard.path.data, shard.path.size),
              "weight/second.data");
    EXPECT_EQ(std::string(shard.format.data, shard.format.size), "safetensors");
  }
}

TEST_F(PreparationCaptureTest, RequestDeclarationsOwnValuesAcrossInvocations) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i64(11)};
  iree_vm_variant_t results[2] = {};
  loom_serve_preparation_module_begin(module, 3, nullptr, 0);
  IREE_ASSERT_OK(Invoke(IREE_SV("request_stages"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  preparation = loom_serve_preparation_module_take(module);
  int32_t command = 0, dispatch = 0;
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &command));
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[1], &dispatch));
  EXPECT_EQ(command, 3);
  EXPECT_EQ(dispatch, 4);

  arguments[0] = iree_vm_variant_from_i64(22);
  loom_serve_preparation_module_begin(module, 7, nullptr, 0);
  IREE_ASSERT_OK(Invoke(IREE_SV("request_stages"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  second = loom_serve_preparation_module_take(module);
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[0], &command));
  IREE_ASSERT_OK(iree_vm_i32_from_variant(results[1], &dispatch));
  EXPECT_EQ(command, 7);
  EXPECT_EQ(dispatch, 8);

  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  const loom_serve_preparation_t* captures[] = {preparation, second};
  for (size_t i = 0; i < IREE_ARRAYSIZE(captures); ++i) {
    ASSERT_EQ(loom_serve_preparation_stage_count(captures[i]), 2u);
    for (size_t j = 0; j < 2; ++j) {
      const auto* stage = loom_serve_preparation_stage(captures[i], j);
      EXPECT_EQ(stage->kind, j ? LOOM_SERVE_PREPARATION_STAGE_DISPATCH
                               : LOOM_SERVE_PREPARATION_STAGE_COMMAND);
      EXPECT_EQ(stage->tag, i ? 22 : 11);
      EXPECT_EQ(stage->workspace_limit_bytes, j ? 0u : 4096u);
      ASSERT_EQ(stage->config.binding_count, 1u);
      const auto& binding = stage->config.bindings[0];
      EXPECT_EQ(std::string(binding.key.data, binding.key.size),
                "feature.enabled");
      EXPECT_EQ(std::string(binding.value.data, binding.value.size),
                i ? "22" : "11");
    }
  }
}

TEST_F(PreparationCaptureTest, EmptyCaptureNeedsNoDeclarationStorage) {
  loom_serve_preparation_module_begin(module, 4, nullptr, 0);
  IREE_ASSERT_OK(Invoke(IREE_SV("no_stages")));
  preparation = loom_serve_preparation_module_take(module);
  EXPECT_EQ(preparation, nullptr);
  EXPECT_EQ(loom_serve_preparation_storage_size(preparation), 0u);
  EXPECT_TRUE(loom_serve_preparation_equal(preparation, nullptr));
  EXPECT_TRUE(loom_serve_preparation_code_equal(preparation, nullptr));
}

TEST_F(PreparationCaptureTest, ForecastBelongsToColdControlNotDeviceIdentity) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(0)};
  loom_serve_preparation_module_begin(module, 0, nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("declare_forecast"),
                               iree_vm_variant_span_from_array(arguments)));
  EXPECT_EQ(loom_serve_preparation_module_take(module), nullptr);
  const iree_status_code_t expected[] = {
      IREE_STATUS_OK, IREE_STATUS_INVALID_ARGUMENT,
      IREE_STATUS_INVALID_ARGUMENT, IREE_STATUS_ALREADY_EXISTS, IREE_STATUS_OK};
  for (int32_t mode = 0; mode < IREE_ARRAYSIZE(expected); ++mode) {
    SCOPED_TRACE(mode);
    arguments[0] = iree_vm_variant_from_i32(mode);
    loom_serve_preparation_module_begin(
        module, 0, nullptr, LOOM_SERVE_PREPARATION_FLAG_MEMORY_FORECAST);
    IREE_EXPECT_STATUS_IS(expected[mode],
                          Invoke(IREE_SV("declare_forecast"),
                                 iree_vm_variant_span_from_array(arguments)));
    auto* captured = loom_serve_preparation_module_take(module);
    if (!mode) {
      preparation = captured;
    } else if (mode == 4) {
      second = captured;
    } else {
      loom_serve_preparation_free(captured);
    }
  }
  ASSERT_NE(preparation, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_FALSE(loom_serve_preparation_equal(preparation, second));
  EXPECT_TRUE(loom_serve_preparation_code_equal(preparation, second));
  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  const auto root = loom_serve_preparation_memory_forecast(preparation);
  EXPECT_EQ(std::string(root.data, root.size), "stage_root");
  const auto other = loom_serve_preparation_memory_forecast(second);
  EXPECT_EQ(std::string(other.data, other.size), "feature.enabled");
}

TEST_F(PreparationCaptureTest, IdentityAndStorageBelongToOwnedDeclarations) {
  for (int64_t tag : {99, 99, 100}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i64(tag)};
    iree_vm_variant_t results[2] = {};
    loom_serve_preparation_module_begin(module, preparation ? 17 : 3, nullptr,
                                        0);
    IREE_ASSERT_OK(Invoke(IREE_SV("request_stages"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    second = loom_serve_preparation_module_take(module);
    EXPECT_FALSE(loom_serve_preparation_equal(second, nullptr));
    EXPECT_FALSE(loom_serve_preparation_code_equal(second, nullptr));
    if (!preparation) {
      preparation = second;
      second = nullptr;
      continue;
    }
    EXPECT_EQ(loom_serve_preparation_equal(preparation, second), tag == 99);
    // This fixture also binds the tag as a config value, so different tags
    // here really are distinct compiler inputs.
    EXPECT_EQ(loom_serve_preparation_code_equal(preparation, second),
              tag == 99);
    EXPECT_EQ(loom_serve_preparation_storage_size(second),
              loom_serve_preparation_storage_size(preparation) +
                  (tag == 100 ? 2 : 0));
    loom_serve_preparation_free(second);
    second = nullptr;
  }
}

TEST_F(PreparationCaptureTest, CapabilityCannotEscapeItsInvocation) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i64(11)};
  iree_vm_variant_t results[2] = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        Invoke(IREE_SV("request_stages"),
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results)));
  arguments[0] = iree_vm_variant_from_i64(11);
  loom_serve_preparation_module_begin(module, 0, nullptr, 0);
  IREE_ASSERT_OK(Invoke(IREE_SV("request_stages"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  preparation = loom_serve_preparation_module_take(module);
  arguments[0] = iree_vm_variant_from_i64(11);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        Invoke(IREE_SV("request_stages"),
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results)));
}

TEST_F(PreparationCaptureTest, RequestCannotReplaceCheckpointDeclarations) {
  loom_serve_preparation_module_begin(module, 4, nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("conflicting_parameter_policy")));
  preparation = loom_serve_preparation_module_take(module);
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 1u);
  EXPECT_EQ(loom_serve_preparation_stage(preparation, 0)->parameter_count, 0u);
}

TEST_F(PreparationCaptureTest, RequestCannotConfigureColdStage) {
  loom_serve_preparation_module_begin(module, 4, nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Invoke(IREE_SV("configure_cold_stage")));
  preparation = loom_serve_preparation_module_take(module);
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 1u);
  EXPECT_EQ(loom_serve_preparation_stage(preparation, 0)->config.binding_count,
            0u);
}

TEST_F(PreparationCaptureTest, RequestOrdinalsCannotOverflow) {
  loom_serve_preparation_module_begin(module, INT32_MAX, nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("configure_cold_stage")));
  preparation = loom_serve_preparation_module_take(module);
  ASSERT_NE(preparation, nullptr);
  EXPECT_EQ(loom_serve_preparation_stage_count(preparation), 0u);
}

TEST_F(PreparationCaptureTest, OneSourceServesMultipleBindingsAndStages) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("shared_source"), IREE_SV("model.irpa"), &ordinal));
  preparation = loom_serve_preparation_module_take(module);
  EXPECT_EQ(ordinal, 0);
  ASSERT_EQ(loom_serve_preparation_source_count(preparation), 1u);
  const auto files = loom_serve_preparation_source(preparation, ordinal);
  ASSERT_EQ(files.count, 2u);
  EXPECT_EQ(std::string(files.values[0].path.data, files.values[0].path.size),
            "/checkpoints/model.irpa");
  EXPECT_EQ(std::string(files.values[1].path.data, files.values[1].path.size),
            "/checkpoints/weights.loom");
  ASSERT_EQ(loom_serve_preparation_stage_count(preparation), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const auto* stage = loom_serve_preparation_stage(preparation, i);
    ASSERT_EQ(stage->parameter_count, i ? 1u : 2u);
    for (size_t j = 0; j < stage->parameter_count; ++j) {
      const auto& parameter = stage->parameters[j];
      ASSERT_EQ(parameter.source_count, 1u);
      EXPECT_EQ(parameter.source_ordinals[0], ordinal);
      EXPECT_EQ(parameter.sources[0].values, files.values);
      EXPECT_EQ(parameter.sources[0].count, files.count);
    }
  }
}

TEST_F(PreparationCaptureTest, ColdOnlySourcesAffectChoiceButNotCodeIdentity) {
  for (const auto path :
       {IREE_SV("model.irpa"), IREE_SV("other.irpa"), IREE_SV("model.irpa")}) {
    loom_serve_preparation_module_begin(module, 0, nullptr,
                                        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
    int32_t ordinal = -1;
    IREE_ASSERT_OK(DeclareSource(IREE_SV("cold_source"), path, &ordinal));
    second = loom_serve_preparation_module_take(module);
    ASSERT_EQ(loom_serve_preparation_source_count(second), 1u);
    EXPECT_EQ(loom_serve_preparation_stage(second, 0)->parameter_count, 0u);
    if (!preparation) {
      preparation = second;
      second = nullptr;
    } else {
      EXPECT_EQ(loom_serve_preparation_equal(preparation, second),
                iree_string_view_equal(path, IREE_SV("model.irpa")));
      EXPECT_TRUE(loom_serve_preparation_code_equal(preparation, second));
      loom_serve_preparation_free(second);
      second = nullptr;
    }
  }
}

TEST_F(PreparationCaptureTest, RequestSourcesExtendButCannotMutateColdPrefix) {
  loom_serve_preparation_module_begin(module, 0, nullptr,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t cold = -1;
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("cold_source"), IREE_SV("cold.irpa"), &cold));
  preparation = loom_serve_preparation_module_take(module);
  const auto cold_files = loom_serve_preparation_source(preparation, cold);
  loom_serve_preparation_module_begin(module, 1, preparation,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t request = -1;
  IREE_ASSERT_OK(DeclareSource(IREE_SV("shared_source"),
                               IREE_SV("request.irpa"), &request));
  iree_vm_variant_t argument = iree_vm_variant_from_i32(cold);
  IREE_ASSERT_OK(Invoke(IREE_SV("bind_source"), {&argument, 1}));
  argument = iree_vm_variant_from_i32(cold);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Invoke(IREE_SV("append_source"), {&argument, 1}));
  second = loom_serve_preparation_module_take(module);
  EXPECT_EQ(request, 1);
  ASSERT_EQ(loom_serve_preparation_source_count(second), 2u);
  EXPECT_EQ(loom_serve_preparation_source(second, cold).values,
            cold_files.values);
  const auto* stage = loom_serve_preparation_stage(second, 2);
  ASSERT_EQ(stage->parameter_count, 1u);
  EXPECT_EQ(stage->parameters[0].source_ordinals[0], cold);
  EXPECT_EQ(stage->parameters[0].sources[0].values, cold_files.values);
  EXPECT_EQ(loom_serve_preparation_source(preparation, cold).count, 1u);
}

TEST_F(PreparationCaptureTest,
       SourceOrdinalsAreCheckedAtTheDeclarationBoundary) {
  for (int32_t ordinal : {-1, 0, INT32_MAX}) {
    for (const auto entry :
         {IREE_SV("bind_source"), IREE_SV("append_source")}) {
      loom_serve_preparation_module_begin(
          module, 0, nullptr, LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
      iree_vm_variant_t argument = iree_vm_variant_from_i32(ordinal);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                            Invoke(entry, {&argument, 1}));
      second = loom_serve_preparation_module_take(module);
      loom_serve_preparation_free(second);
      second = nullptr;
    }
  }
}

TEST_F(PreparationCaptureTest, ParameterArenaRegistrationIsFiniteAndColdOnly) {
  int32_t arena = -1;
  loom_serve_preparation_module_begin(module, 0, nullptr, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        DeclareArena(INT64_C(1) << 34, &arena));
  EXPECT_EQ(loom_serve_preparation_module_take(module), nullptr);
  loom_serve_preparation_module_begin(
      module, 0, nullptr, LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS);
  for (int64_t invalid : {INT64_C(-1), INT64_C(0)}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          DeclareArena(invalid, &arena));
  }
  IREE_ASSERT_OK(DeclareArena(INT64_C(1) << 34, &arena));
  EXPECT_EQ(arena, 0);
  IREE_ASSERT_OK(DeclareArena(4096, &arena));
  EXPECT_EQ(arena, 1);
  IREE_ASSERT_OK(Invoke(IREE_SV("declare_parameter_binding")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Invoke(IREE_SV("declare_parameter_binding")));
  preparation = loom_serve_preparation_module_take(module);
  ASSERT_EQ(loom_serve_preparation_arena_count(preparation), 2u);
  EXPECT_EQ(loom_serve_preparation_arena_capacity(preparation, 0), UINT64_C(1)
                                                                       << 34);
  const auto entry = loom_serve_preparation_bind_parameters(preparation);
  EXPECT_EQ(std::string(entry.data, entry.size), "stage_root");
  loom_serve_preparation_module_begin(
      module, 0, preparation, LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        DeclareArena(8192, &arena));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("declare_parameter_binding")));
  EXPECT_EQ(loom_serve_preparation_module_take(module), nullptr);
}

TEST_F(PreparationCaptureTest,
       BankSelectionsDoNotExpandBindingsOrCodeIdentity) {
  loom_serve_preparation_module_begin(
      module, 0, nullptr,
      LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS |
          LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  int32_t ordinal = -1;
  IREE_ASSERT_OK(DeclareArena(INT64_C(1) << 34, &ordinal));
  IREE_ASSERT_OK(
      DeclareSource(IREE_SV("cold_source"), IREE_SV("bank.irpa"), &ordinal));
  preparation = loom_serve_preparation_module_take(module);
  const uint8_t source_ordinals[] = {0, 0, 0, 0};
  std::unique_ptr<loom_serve_preparation_t,
                  decltype(&loom_serve_preparation_free)>
      previous(nullptr, loom_serve_preparation_free);
  for (int32_t count : {1, 5, 33}) {
    loom_serve_preparation_module_begin(module, 1, preparation,
                                        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
    IREE_ASSERT_OK(DeclareBanks(
        0, iree_make_const_byte_span(source_ordinals, sizeof(source_ordinals)),
        count, &ordinal));
    EXPECT_EQ(ordinal, count - 1);
    second = loom_serve_preparation_module_take(module);
    ASSERT_EQ(loom_serve_preparation_bank_count(second), count);
    ASSERT_EQ(loom_serve_preparation_stage_count(second), 1u);
    const auto* stage = loom_serve_preparation_stage(second, 0);
    ASSERT_EQ(stage->parameter_count, 1u);
    EXPECT_EQ(stage->parameters[0].arena, 1u);
    EXPECT_EQ(stage->parameters[0].source_count, 0u);
    for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
      const auto* bank = loom_serve_preparation_bank(second, i);
      EXPECT_EQ(bank->arena, 1u);
      EXPECT_EQ(bank->binding, 0u);
      ASSERT_EQ(bank->source_count, 1u);
      EXPECT_EQ(bank->source_ordinals[0], 0u);
      EXPECT_EQ(bank->sources[0].values,
                loom_serve_preparation_source(preparation, 0).values);
    }
    if (previous) {
      EXPECT_FALSE(loom_serve_preparation_equal(previous.get(), second));
      EXPECT_TRUE(loom_serve_preparation_code_equal(previous.get(), second));
    }
    previous.reset(second);
    second = nullptr;
  }
}

TEST_F(PreparationCaptureTest,
       BankSourcesAreValidatedBeforeCapturePublication) {
  loom_serve_preparation_module_begin(
      module, 0, nullptr, LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS);
  int32_t ordinal = -1;
  IREE_ASSERT_OK(DeclareArena(4096, &ordinal));
  preparation = loom_serve_preparation_module_take(module);
  const uint8_t source_ordinals[] = {0, 0, 0, 0};
  for (size_t length : {0u, 3u, 4u}) {
    loom_serve_preparation_module_begin(module, 0, preparation,
                                        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
    IREE_EXPECT_STATUS_IS(
        length == 4 ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_INVALID_ARGUMENT,
        DeclareBanks(0, iree_make_const_byte_span(source_ordinals, length), 1,
                     &ordinal));
    second = loom_serve_preparation_module_take(module);
    loom_serve_preparation_free(second);
    second = nullptr;
  }
}

}  // namespace
