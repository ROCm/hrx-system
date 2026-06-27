// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <cstring>
#include <memory>

#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/streaming_testdata.h"

namespace {

static iree_string_view_t GetEmbeddedTokenizerJson() {
  const iree_file_toc_t* toc = iree_tokenizer_streaming_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_streaming_testdata_size(); ++i) {
    if (std::strcmp(toc[i].name, "bpe_bytelevel_minimal.json") == 0) {
      return iree_make_string_view(toc[i].data, toc[i].size);
    }
  }
  return iree_string_view_empty();
}

struct TokenizerDeleter {
  void operator()(iree_tokenizer_t* tokenizer) const {
    iree_tokenizer_free(tokenizer);
  }
};

using TokenizerPtr = std::unique_ptr<iree_tokenizer_t, TokenizerDeleter>;

struct ScopedRequest {
  ~ScopedRequest() {
    id4_ideogram4_request_deinitialize(&value, iree_allocator_system());
  }

  id4_ideogram4_request_t value = {};
};

struct SessionDeleter {
  void operator()(id4_ideogram4_session_t* session) const {
    id4_ideogram4_session_release(session);
  }
};

using SessionPtr = std::unique_ptr<id4_ideogram4_session_t, SessionDeleter>;

struct GenerationPlanDeleter {
  void operator()(id4_ideogram4_generation_plan_t* plan) const {
    id4_ideogram4_generation_plan_release(plan);
  }
};

using GenerationPlanPtr =
    std::unique_ptr<id4_ideogram4_generation_plan_t, GenerationPlanDeleter>;

static TokenizerPtr LoadTokenizer() {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
      GetEmbeddedTokenizerJson(), iree_allocator_system(), &tokenizer));
  return TokenizerPtr(tokenizer);
}

static id4_ideogram4_generation_plan_policy_t MakeGenerationPolicy() {
  id4_ideogram4_generation_plan_policy_t policy;
  std::memset(&policy, 0, sizeof(policy));
  policy.structure_size = sizeof(policy);
  policy.dit_activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  policy.dit_attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;
  policy.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  return policy;
}

static iree_string_view_t ShortFullRequestJson() {
  return IREE_SV(
      "{\"prompt\":\"a city\",\"generation\":{\"latent_width\":8,"
      "\"latent_height\":8,\"denoise_steps\":2,\"seed\":1,"
      "\"guidance_scale\":3.5}}");
}

static iree_string_view_t LongFullRequestJson() {
  return IREE_SV(
      "{\"prompt\":\"three people walking through a reflective city street "
      "with umbrellas and neon signs\",\"generation\":{\"latent_width\":8,"
      "\"latent_height\":8,\"denoise_steps\":2,\"seed\":1,"
      "\"guidance_scale\":3.5}}");
}

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_group_ = id4::test::CreateLocalSyncDeviceGroup();
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(device_group_, /*index=*/0);
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device, IREE_SV("id4_session_test"), &executable_cache_));

    id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
    std::memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
    kernel_cache_options.structure_size = sizeof(kernel_cache_options);
    kernel_cache_options.target_processor =
        id4_pipeline_kernel_cache_default_target_processor();
    IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
        &kernel_cache_options, iree_allocator_system(), &kernel_cache_));
  }

  void TearDown() override {
    id4_pipeline_kernel_cache_release(kernel_cache_);
    iree_hal_executable_cache_release(executable_cache_);
    iree_hal_device_group_release(device_group_);
  }

  id4_ideogram4_session_create_options_t CreateOptions() {
    id4_pipeline_stage_services_t services;
    std::memset(&services, 0, sizeof(services));
    services.device_group = device_group_;
    services.executable_cache = executable_cache_;
    services.host_allocator = iree_allocator_system();

    id4_ideogram4_session_create_options_t options;
    std::memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.services = services;
    options.kernel_cache = kernel_cache_;
    options.dit_parameter_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16;
    options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
    return options;
  }

  SessionPtr CreateLoadedSession(
      id4_ideogram4_session_create_options_t create_options) {
    id4_ideogram4_session_t* session = nullptr;
    IREE_CHECK_OK(id4_ideogram4_session_create(
        &create_options, iree_allocator_system(), &session));

    id4_ideogram4_session_load_options_t load_options;
    std::memset(&load_options, 0, sizeof(load_options));
    load_options.structure_size = sizeof(load_options);
    id4::test::StageDiagnostics diagnostics = {};
    id4_pipeline_diagnostics_sink_t diagnostics_sink =
        id4::test::DiagnosticsSink(&diagnostics);
    load_options.diagnostics_sink = &diagnostics_sink;
    IREE_CHECK_OK(id4_ideogram4_session_load(session, &load_options));
    return SessionPtr(session);
  }

  SessionPtr CreateLoadedSession() {
    return CreateLoadedSession(CreateOptions());
  }

  GenerationPlanPtr PlanGeneration(id4_ideogram4_session_t* session,
                                   const iree_tokenizer_t* tokenizer,
                                   const id4_ideogram4_request_t* request) {
    id4_ideogram4_generation_plan_options_t plan_options;
    std::memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    plan_options.policy = MakeGenerationPolicy();
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    id4::test::StageDiagnostics diagnostics = {};
    id4_pipeline_diagnostics_sink_t diagnostics_sink =
        id4::test::DiagnosticsSink(&diagnostics);
    plan_options.diagnostics_sink = &diagnostics_sink;

    id4_ideogram4_generation_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        id4_ideogram4_session_plan_generation(session, &plan_options, &plan));
    return GenerationPlanPtr(plan);
  }

  iree_hal_device_group_t* device_group_ = nullptr;
  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  id4_pipeline_kernel_cache_t* kernel_cache_ = nullptr;
};

TEST_F(SessionTest, LoadsOnce) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_session_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_ideogram4_session_load(session, &load_options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_ideogram4_session_load(session, &load_options));

  id4_ideogram4_session_release(session);
}

TEST_F(SessionTest, RequiresVaeActivationFormat) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  create_options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_INVALID;

  id4_ideogram4_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_create(&create_options, iree_allocator_system(),
                                   &session));
  EXPECT_EQ(session, nullptr);
}

TEST_F(SessionTest, RequiresDitParameterFormat) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  create_options.dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;

  id4_ideogram4_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_create(&create_options, iree_allocator_system(),
                                   &session));
  EXPECT_EQ(session, nullptr);
}

TEST_F(SessionTest, RejectsFp8ScopesForBf16DitParameterFormat) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  create_options.parameter_scopes.dit_conditioned_fp8 = IREE_SV("dit_cond_fp8");

  id4_ideogram4_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_create(&create_options, iree_allocator_system(),
                                   &session));
  EXPECT_EQ(session, nullptr);
}

TEST_F(SessionTest, IssueRequiresLoadedSession) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_qwen_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  id4_ideogram4_qwen_execution_t* execution = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_ideogram4_session_issue_qwen(session, &issue_options, &execution));
  EXPECT_EQ(execution, nullptr);

  id4_ideogram4_session_release(session);
}

TEST_F(SessionTest, IssueGenerationRequiresLoadedSession) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_generation_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  id4_ideogram4_generation_execution_t* execution = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        id4_ideogram4_session_issue_generation(
                            session, nullptr, &issue_options, &execution));
  EXPECT_EQ(execution, nullptr);

  id4_ideogram4_session_release(session);
}

TEST_F(SessionTest, IssueGenerationRequiresPreparedBundle) {
  SessionPtr session = CreateLoadedSession();

  id4_ideogram4_generation_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  id4_ideogram4_generation_execution_t* execution = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_issue_generation(session.get(), nullptr,
                                             &issue_options, &execution));
  EXPECT_EQ(execution, nullptr);
}

TEST_F(SessionTest, PlanGenerationRequiresLoadedSession) {
  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  id4_ideogram4_session_t* session = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_generation_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.policy = MakeGenerationPolicy();

  id4_ideogram4_generation_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      id4_ideogram4_session_plan_generation(session, &plan_options, &plan));
  EXPECT_EQ(plan, nullptr);

  id4_ideogram4_session_release(session);
}

TEST_F(SessionTest, PlansGenerationFromDynamicPromptLength) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest short_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      ShortFullRequestJson(), iree_allocator_system(), &short_request.value));
  ScopedRequest long_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      LongFullRequestJson(), iree_allocator_system(), &long_request.value));

  SessionPtr session = CreateLoadedSession();
  id4_ideogram4_generation_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.tokenizer = tokenizer.get();
  plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  plan_options.policy = MakeGenerationPolicy();
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  plan_options.diagnostics_sink = &diagnostics_sink;

  plan_options.request = &short_request.value;
  id4_ideogram4_generation_plan_t* short_plan = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_plan_generation(
      session.get(), &plan_options, &short_plan));
  GenerationPlanPtr short_plan_owner(short_plan);

  plan_options.request = &long_request.value;
  id4_ideogram4_generation_plan_t* long_plan = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_plan_generation(
      session.get(), &plan_options, &long_plan));
  GenerationPlanPtr long_plan_owner(long_plan);

  id4_ideogram4_generation_plan_summary_t short_summary;
  IREE_ASSERT_OK(id4_ideogram4_generation_plan_summary(short_plan_owner.get(),
                                                       &short_summary));
  id4_ideogram4_generation_plan_summary_t long_summary;
  IREE_ASSERT_OK(id4_ideogram4_generation_plan_summary(long_plan_owner.get(),
                                                       &long_summary));
  EXPECT_GT(short_summary.qwen_token_count, 0u);
  EXPECT_GT(long_summary.qwen_token_count, short_summary.qwen_token_count);
  EXPECT_EQ(long_summary.denoise_step_count, 2u);
  EXPECT_EQ(long_summary.diffusion_latent_shape.rank, 4u);
  EXPECT_EQ(long_summary.diffusion_latent_shape.dims[2], 128u);
  EXPECT_EQ(long_summary.decoded_image_shape.rank, 4u);
  EXPECT_EQ(long_summary.decoded_image_shape.dims[0], 128u);
  EXPECT_EQ(long_summary.decoded_image_shape.dims[1], 128u);
  EXPECT_EQ(long_summary.decoded_image_shape.dims[2], 3u);
  EXPECT_EQ(long_summary.decoded_image_shape.dims[3], 1u);
  EXPECT_EQ(long_summary.dit_activation_format,
            ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT);
  EXPECT_EQ(long_summary.dit_attention_implementation,
            ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING);
  EXPECT_EQ(long_summary.vae_tiling.mode, ID4_VAE_TILING_MODE_DISABLED);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(id4_ideogram4_generation_plan_format_json(
      long_plan_owner.get(), &builder));
  iree_string_view_t json = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(json, IREE_SV("\"ideogram4_generation\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(json, IREE_SV("\"residency\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                json, IREE_SV("\"phase_parameter_high_water_mark\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(json, IREE_SV("\"stages\""), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(SessionTest, PlansMixedBf16Fp8DitSources) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      ShortFullRequestJson(), iree_allocator_system(), &request.value));

  id4_ideogram4_session_create_options_t create_options = CreateOptions();
  create_options.parameter_scopes.dit_conditioned = IREE_SV("dit_cond");
  create_options.parameter_scopes.dit_conditioned_fp8 = IREE_SV("dit_cond_fp8");
  create_options.parameter_scopes.dit_unconditioned = IREE_SV("dit_uncond");
  create_options.parameter_scopes.dit_unconditioned_fp8 =
      IREE_SV("dit_uncond_fp8");
  create_options.dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3;
  SessionPtr session = CreateLoadedSession(create_options);

  id4_ideogram4_generation_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.request = &request.value;
  plan_options.tokenizer = tokenizer.get();
  plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  plan_options.policy = MakeGenerationPolicy();
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4_ideogram4_generation_plan_t* plan = nullptr;
  IREE_ASSERT_OK(id4_ideogram4_session_plan_generation(session.get(),
                                                       &plan_options, &plan));
  GenerationPlanPtr plan_owner(plan);

  id4_ideogram4_generation_plan_summary_t summary;
  IREE_ASSERT_OK(
      id4_ideogram4_generation_plan_summary(plan_owner.get(), &summary));
  EXPECT_EQ(summary.dit_activation_format,
            ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT);
  EXPECT_EQ(summary.dit_attention_implementation,
            ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      id4_ideogram4_generation_plan_format_json(plan_owner.get(), &builder));
  iree_string_view_t json = iree_string_builder_view(&builder);
  EXPECT_NE(iree_string_view_find(
                json, IREE_SV("\"source_scope\":\"dit_cond_fp8\""), 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                json, IREE_SV("\"source_scope\":\"dit_uncond_fp8\""), 0),
            IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST_F(SessionTest, RejectsMissingGenerationMetadata) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":\"a city\"}"), iree_allocator_system(),
      &request.value));
  SessionPtr session = CreateLoadedSession();

  id4_ideogram4_generation_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.request = &request.value;
  plan_options.tokenizer = tokenizer.get();
  plan_options.policy = MakeGenerationPolicy();
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;

  id4_ideogram4_generation_plan_t* plan = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_session_plan_generation(
                            session.get(), &plan_options, &plan));
  EXPECT_EQ(plan, nullptr);
}

TEST_F(SessionTest, PrepareGenerationRequiresFinalSignal) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      ShortFullRequestJson(), iree_allocator_system(), &request.value));
  SessionPtr session = CreateLoadedSession();
  GenerationPlanPtr plan =
      PlanGeneration(session.get(), tokenizer.get(), &request.value);

  id4_ideogram4_generation_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);

  id4_ideogram4_generation_bundle_t* bundle = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_prepare_generation(session.get(), plan.get(),
                                               &prepare_options, &bundle));
  EXPECT_EQ(bundle, nullptr);
}

TEST_F(SessionTest, PrepareGenerationRequiresParameterProviders) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      ShortFullRequestJson(), iree_allocator_system(), &request.value));
  SessionPtr session = CreateLoadedSession();
  GenerationPlanPtr plan =
      PlanGeneration(session.get(), tokenizer.get(), &request.value);

  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group_, /*index=*/0);
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           &semaphore));
  iree_hal_semaphore_t* signal_semaphores[] = {semaphore};
  uint64_t signal_payload_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/signal_semaphores,
      /*.payload_values=*/signal_payload_values,
  };

  id4_ideogram4_generation_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.signal_semaphore_list = signal_list;

  id4_ideogram4_generation_bundle_t* bundle = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_session_prepare_generation(session.get(), plan.get(),
                                               &prepare_options, &bundle));
  EXPECT_EQ(bundle, nullptr);

  iree_hal_semaphore_release(semaphore);
}

}  // namespace
