// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/sync.h"
#include "loom/model/card.h"
#include "loom/runtime/program/package.h"
#include "loom/runtime/program/program.h"
#include "loom/runtime/program/source_index.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

namespace {

class CxxProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator_), &target_environment_)));
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(host_allocator_, &environment_));
  }

  void TearDown() override {
    iree_vm_environment_free(environment_);
    loomc_target_environment_release(target_environment_);
  }

  const iree_allocator_t host_allocator_ = iree_allocator_system();
  loomc_target_environment_t* target_environment_ = nullptr;
  iree_vm_environment_t* environment_ = nullptr;
};

iree_status_t CreateSourceIdentity(
    loomc_target_environment_t* target_environment,
    const loom_runtime_sources_t* sources, iree_allocator_t host_allocator,
    loom_sha256_t* out_identity) {
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = target_environment,
  };
  const loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_context_create(
      &context_options, loomc_allocator_from_iree(host_allocator), &context)));
  loom_runtime_compilation_report_t report = {};
  loomc_link_index_t* index = nullptr;
  iree_status_t status = loom_runtime_source_index_create(
      context, sources, nullptr, &report, host_allocator, out_identity, &index);
  loomc_link_index_release(index);
  loomc_context_release(context);
  return status;
}

TEST_F(CxxProviderTest, ExplicitAndInferredProvidersExecuteThroughVm) {
  constexpr char kCard[] = R"json({
    "schema_version": 1,
    "model": {"id": "fixture/cxx", "name": "C++ provider fixture"},
    "programs": {"source": {
      "inputs": [
        {"path": "explicit.source", "format": "cxx",
         "options": ["cxx:std=c++20"]},
        "inferred.cxx"
      ],
      "libraries": []
    }},
    "services": {"test": {"kind": "test", "version": 1}},
    "variants": {"default": {"services": {"test": {
      "programs": ["source"]
    }}}}
  })json";
  constexpr char kExplicitSource[] =
      R"cxx(extern "C" int explicit_answer() { return 42; })cxx";
  constexpr char kInferredSource[] =
      R"cxx(extern "C" int inferred_answer() { return 7; })cxx";
  const loom_runtime_package_file_t files[] = {
      {IREE_SV("explicit.source"),
       iree_make_const_byte_span(kExplicitSource, sizeof(kExplicitSource) - 1)},
      {IREE_SV("inferred.cxx"),
       iree_make_const_byte_span(kInferredSource, sizeof(kInferredSource) - 1)},
  };
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_create(IREE_SV("model.json"),
                                             IREE_ARRAYSIZE(files), files,
                                             host_allocator_, &package));
  std::unique_ptr<loom_runtime_package_t,
                  decltype(&loom_runtime_package_release)>
      package_owner(package, loom_runtime_package_release);
  loom_model_card_t* card = nullptr;
  IREE_ASSERT_OK(loom_model_card_parse(iree_make_cstring_view(kCard), nullptr,
                                       host_allocator_, &card));
  std::unique_ptr<loom_model_card_t, decltype(&loom_model_card_release)>
      card_owner(card, loom_model_card_release);

  const loom_runtime_sources_t sources = {.package = package, .card = card};
  const iree_string_view_t roots[] = {IREE_SV("explicit_answer"),
                                      IREE_SV("inferred_answer")};
  loom_runtime_compilation_report_t report = {};
  loom_runtime_program_t* program = nullptr;
  IREE_ASSERT_OK(loom_runtime_program_allocate(
      target_environment_, environment_, &sources, {}, IREE_ARRAYSIZE(roots),
      roots, iree_vm_module_span_empty(), &report, host_allocator_, &program));
  std::unique_ptr<loom_runtime_program_t, decltype(&loom_runtime_program_free)>
      program_owner(program, loom_runtime_program_free);

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
    SCOPED_TRACE(i);
    iree_vm_function_t function = {};
    IREE_ASSERT_OK(loom_runtime_program_lookup(program, roots[i], &function));
    iree_vm_variant_t result = {};
    IREE_ASSERT_OK(iree_vm_invoke(loom_runtime_program_invocation(program),
                                  function, iree_vm_variant_span_empty(),
                                  {&result, 1}));
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &value));
    EXPECT_EQ(value, i == 0 ? 42 : 7);
  }
}

TEST_F(CxxProviderTest, OptionsAndDeclaredDataDefineSourceIdentity) {
  constexpr char kCard[] = R"json({
    "schema_version": 1,
    "model": {"id": "fixture/cxx", "name": "C++ provider fixture"},
    "programs": {"source": {
      "inputs": [{"path": "answer.cxx", "format": "cxx",
                  "options": ["cxx:std=c++20 D=VALUE=%d"],
                  "data": ["metadata.bin"]}],
      "libraries": []
    }},
    "services": {"test": {"kind": "test", "version": 1}},
    "variants": {"default": {"services": {"test": {
      "programs": ["source"]
    }}}}
  })json";
  constexpr char kSource[] = R"cxx(extern "C" int answer() { return 42; })cxx";
  constexpr char kFirstData[] = "first";
  constexpr char kSecondData[] = "second";
  const loom_runtime_package_file_t first_files[] = {
      {IREE_SV("answer.cxx"),
       iree_make_const_byte_span(kSource, sizeof(kSource) - 1)},
      {IREE_SV("metadata.bin"),
       iree_make_const_byte_span(kFirstData, sizeof(kFirstData) - 1)},
  };
  const loom_runtime_package_file_t second_files[] = {
      {IREE_SV("answer.cxx"),
       iree_make_const_byte_span(kSource, sizeof(kSource) - 1)},
      {IREE_SV("metadata.bin"),
       iree_make_const_byte_span(kSecondData, sizeof(kSecondData) - 1)},
  };
  loom_runtime_package_t* first_package = nullptr;
  loom_runtime_package_t* second_package = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_create(
      IREE_SV("model.json"), IREE_ARRAYSIZE(first_files), first_files,
      host_allocator_, &first_package));
  IREE_ASSERT_OK(loom_runtime_package_create(
      IREE_SV("model.json"), IREE_ARRAYSIZE(second_files), second_files,
      host_allocator_, &second_package));
  std::unique_ptr<loom_runtime_package_t,
                  decltype(&loom_runtime_package_release)>
      first_package_owner(first_package, loom_runtime_package_release);
  std::unique_ptr<loom_runtime_package_t,
                  decltype(&loom_runtime_package_release)>
      second_package_owner(second_package, loom_runtime_package_release);

  char first_card_json[sizeof(kCard)];
  char second_card_json[sizeof(kCard)];
  snprintf(first_card_json, sizeof(first_card_json), kCard, 1);
  snprintf(second_card_json, sizeof(second_card_json), kCard, 2);
  loom_model_card_t* first_card = nullptr;
  loom_model_card_t* second_card = nullptr;
  IREE_ASSERT_OK(loom_model_card_parse(iree_make_cstring_view(first_card_json),
                                       nullptr, host_allocator_, &first_card));
  IREE_ASSERT_OK(loom_model_card_parse(iree_make_cstring_view(second_card_json),
                                       nullptr, host_allocator_, &second_card));
  std::unique_ptr<loom_model_card_t, decltype(&loom_model_card_release)>
      first_card_owner(first_card, loom_model_card_release);
  std::unique_ptr<loom_model_card_t, decltype(&loom_model_card_release)>
      second_card_owner(second_card, loom_model_card_release);

  const loom_runtime_sources_t first_sources = {
      .package = first_package,
      .card = first_card,
  };
  const loom_runtime_sources_t changed_data_sources = {
      .package = second_package,
      .card = first_card,
  };
  const loom_runtime_sources_t changed_options_sources = {
      .package = first_package,
      .card = second_card,
  };
  loom_sha256_t first_identity;
  loom_sha256_t changed_data_identity;
  loom_sha256_t changed_options_identity;
  IREE_ASSERT_OK(CreateSourceIdentity(target_environment_, &first_sources,
                                      host_allocator_, &first_identity));
  IREE_ASSERT_OK(CreateSourceIdentity(target_environment_,
                                      &changed_data_sources, host_allocator_,
                                      &changed_data_identity));
  IREE_ASSERT_OK(CreateSourceIdentity(target_environment_,
                                      &changed_options_sources, host_allocator_,
                                      &changed_options_identity));
  EXPECT_FALSE(loom_sha256_equal(&first_identity, &changed_data_identity));
  EXPECT_FALSE(loom_sha256_equal(&first_identity, &changed_options_identity));
}

}  // namespace
