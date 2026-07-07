// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>
#include <vector>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/qwen3_vl_program.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

IREE_FLAG(string, id4_tokenizer, "",
          "Hugging Face tokenizer JSON used to lower the prompt.");
IREE_FLAG(string, id4_request_json, "",
          "Prompt JSON request passed through production request lowering.");
IREE_FLAG(string, id4_request_json_file, "",
          "File containing prompt JSON request passed through production "
          "request lowering.");
IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing the Qwen expected-condition fixture.");
IREE_FLAG_LIST(string, id4_diagnostic_tap,
               "Additional Qwen diagnostic tap names to capture.");

namespace {

using HalSemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;
using QwenExecutionRef =
    id4::test::OwningRef<id4_ideogram4_qwen_execution_t,
                         id4_ideogram4_qwen_execution_release>;
using SessionRef = id4::test::OwningRef<id4_ideogram4_session_t,
                                        id4_ideogram4_session_release>;
using TokenizerRef =
    id4::test::OwningRef<iree_tokenizer_t, iree_tokenizer_free>;

static constexpr char kShortPrompt128[] =
    "{\"prompt\":\"A small red boat on a quiet lake at sunrise.\","
    "\"generation\":{\"latent_width\":8,\"latent_height\":8,"
    "\"denoise_steps\":1,\"seed\":20260626,\"guidance_scale\":3.5}}";

static constexpr char kMediumPrompt128[] =
    "{\"prompt\":\"Three friends walking through a bright city crosswalk with "
    "glass storefronts, natural clothing, normal hands, and soft afternoon "
    "light.\",\"generation\":{\"latent_width\":8,\"latent_height\":8,"
    "\"denoise_steps\":1,\"seed\":20260625,\"guidance_scale\":3.5}}";

static constexpr char kStructuredPrompt128[] =
    "{\"prompt\":{\"high_level_description\":\"A realistic street photograph "
    "of three adults walking together through a modern city crosswalk on a "
    "clear afternoon.\",\"style_description\":{\"medium\":\"photograph\","
    "\"lighting\":\"soft afternoon sunlight\","
    "\"aesthetics\":\"naturalistic candid modern urban lifestyle\"},"
    "\"compositional_deconstruction\":{\"background\":\"clean downtown street "
    "with storefront reflections\",\"elements\":[{\"type\":\"obj\","
    "\"bbox\":[260,170,920,390],\"desc\":\"adult man walking on the left\"},"
    "{\"type\":\"obj\",\"bbox\":[230,390,930,610],\"desc\":\"adult woman "
    "walking in the center\"},{\"type\":\"obj\","
    "\"bbox\":[260,610,920,830],\"desc\":\"adult man walking on the right\"}]}"
    "},\"generation\":{\"latent_width\":8,\"latent_height\":8,"
    "\"denoise_steps\":1,\"seed\":20260625,\"guidance_scale\":3.5}}";

struct ParsedRequest {
  ~ParsedRequest() {
    if (is_initialized) {
      id4_ideogram4_request_deinitialize(&value, iree_allocator_system());
    }
  }

  // Parsed request owned by this wrapper.
  id4_ideogram4_request_t value = {};
  // True when |value| owns allocations.
  bool is_initialized = false;
};

typedef struct PromptBucket {
  // Human-readable prompt bucket label used by assertion output.
  const char* label;
  // Full request JSON passed through the production parser.
  iree_string_view_t json;
} PromptBucket;

static iree_status_t LoadTokenizer(iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = nullptr;

  iree_string_view_t tokenizer_path =
      iree_make_cstring_view(FLAG_id4_tokenizer);
  if (iree_string_view_is_empty(tokenizer_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--id4_tokenizer is required");
  }

  iree_io_file_contents_t* file_contents = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_map(tokenizer_path, IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &file_contents));
  iree_string_view_t tokenizer_json = iree_make_string_view(
      reinterpret_cast<const char*>(file_contents->const_buffer.data),
      file_contents->const_buffer.data_length);
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      tokenizer_json, iree_allocator_system(), out_tokenizer);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t ParseRequestJson(iree_string_view_t request_json,
                                      ParsedRequest* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_json(
      request_json, iree_allocator_system(), &out_request->value));
  out_request->is_initialized = true;
  return iree_ok_status();
}

static iree_status_t LoadRequestJson(std::string* out_request_json) {
  IREE_ASSERT_ARGUMENT(out_request_json);
  out_request_json->clear();

  iree_string_view_t inline_json =
      iree_make_cstring_view(FLAG_id4_request_json);
  iree_string_view_t file_path =
      iree_make_cstring_view(FLAG_id4_request_json_file);
  const bool has_inline_json = !iree_string_view_is_empty(inline_json);
  const bool has_file_path = !iree_string_view_is_empty(file_path);
  if (has_inline_json == has_file_path) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exactly one of --id4_request_json or --id4_request_json_file is "
        "required");
  }
  if (has_inline_json) {
    out_request_json->assign(inline_json.data, inline_json.size);
    return iree_ok_status();
  }

  iree_io_file_contents_t* file_contents = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_io_file_contents_map(file_path, IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &file_contents));
  out_request_json->assign(
      reinterpret_cast<const char*>(file_contents->const_buffer.data),
      file_contents->const_buffer.data_length);
  iree_io_file_contents_free(file_contents);
  return iree_ok_status();
}

static iree_status_t ParseRequestFromFlags(ParsedRequest* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  std::string request_json;
  IREE_RETURN_IF_ERROR(LoadRequestJson(&request_json));
  return ParseRequestJson(
      iree_make_string_view(request_json.data(), request_json.size()),
      out_request);
}

static iree_status_t CreateLoadedSession(
    const id4::test::LiveStageContext& context,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_session_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.qwen_parameter_format = ID4_QWEN3_VL_PARAMETER_FORMAT_BF16;
  create_options.dit_parameter_format = ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16;
  create_options.vae_activation_format =
      ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;

  id4_ideogram4_session_t* session = nullptr;
  IREE_RETURN_IF_ERROR(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_ideogram4_session_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = diagnostics_sink;
  iree_status_t status = id4_ideogram4_session_load(session, &load_options);
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    id4_ideogram4_session_release(session);
  }
  return status;
}

static const id4_pipeline_tensor_layout_t* FindBoundaryLayout(
    const id4_pipeline_plan_t* plan, iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      return &boundary->layout;
    }
  }
  return nullptr;
}

static iree_status_t CountRequestTokens(const id4_ideogram4_request_t* request,
                                        const iree_tokenizer_t* tokenizer,
                                        uint32_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;

  id4_ideogram4_qwen_lowering_options_t lowering_options;
  std::memset(&lowering_options, 0, sizeof(lowering_options));
  lowering_options.structure_size = sizeof(lowering_options);
  lowering_options.request = request;
  lowering_options.tokenizer = tokenizer;
  lowering_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  lowering_options.max_token_count =
      id4_qwen3_vl_program_ideogram4_model_config()->max_token_count;
  lowering_options.vocab_size =
      id4_qwen3_vl_program_ideogram4_model_config()->vocab_size;
  return id4_ideogram4_request_count_qwen_tokens(
      &lowering_options, iree_allocator_system(), out_token_count);
}

static iree_status_t IssueQwenRequest(
    const id4::test::LiveStageContext& context,
    id4_ideogram4_session_t* session, const iree_tokenizer_t* tokenizer,
    iree_io_parameter_provider_t* parameter_provider,
    id4_pipeline_kernel_library_t* kernel_library,
    const id4_ideogram4_request_t* request,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* completion_semaphore, uint64_t completion_value,
    id4_ideogram4_qwen_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(session);
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(parameter_provider);
  IREE_ASSERT_ARGUMENT(kernel_library);
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(completion_semaphore);
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = nullptr;

  id4::test::SemaphoreListStorage completion_signal;
  completion_signal.semaphore = completion_semaphore;
  completion_signal.payload_value = completion_value;

  id4_ideogram4_qwen_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.request = request;
  issue_options.tokenizer = tokenizer;
  issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  issue_options.parameter_provider = parameter_provider;
  issue_options.kernel_library = kernel_library;
  issue_options.device_index = 0;
  issue_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  issue_options.command_buffer_mode = context.command_buffer_mode;
  issue_options.qwen_weight_execution_strategy =
      ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS;
  issue_options.qwen_attention_implementation =
      ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;
  const iree_flag_string_list_t diagnostic_tap_flags =
      FLAG_id4_diagnostic_tap_list();
  std::vector<iree_string_view_t> diagnostic_tap_names;
  diagnostic_tap_names.reserve(diagnostic_tap_flags.count);
  for (iree_host_size_t i = 0; i < diagnostic_tap_flags.count; ++i) {
    diagnostic_tap_names.push_back(diagnostic_tap_flags.values[i]);
  }
  issue_options.diagnostic_tap_names = (iree_string_view_list_t){
      diagnostic_tap_names.size(),
      diagnostic_tap_names.data(),
  };
  issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  issue_options.signal_semaphore_list = completion_signal.list();
  issue_options.diagnostics_sink = diagnostics_sink;

  id4_ideogram4_qwen_execution_t* execution = nullptr;
  iree_status_t status =
      id4_ideogram4_session_issue_qwen(session, &issue_options, &execution);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(completion_semaphore, completion_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_qwen_execution_release(execution);
  }
  return status;
}

static void ExpectConditionLayoutForTokenCount(
    const id4_ideogram4_qwen_execution_t* execution,
    uint32_t expected_token_count) {
  id4_ideogram4_qwen_result_t result;
  std::memset(&result, 0, sizeof(result));
  IREE_ASSERT_OK(id4_ideogram4_qwen_execution_result(execution, &result));
  EXPECT_EQ(result.token_count, expected_token_count);

  const id4_pipeline_plan_t* plan =
      id4_ideogram4_qwen_execution_plan(execution);
  const id4_pipeline_tensor_layout_t* condition_layout =
      FindBoundaryLayout(plan, IREE_SV("condition"));
  ASSERT_NE(condition_layout, nullptr);

  const id4_qwen3_vl_model_config_t* model =
      id4_qwen3_vl_program_ideogram4_model_config();
  const uint64_t expected_condition_row_count =
      (uint64_t)model->selected_layer_count * model->hidden_size;
  EXPECT_EQ(condition_layout->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  EXPECT_EQ(condition_layout->shape.rank, 2u);
  EXPECT_EQ(condition_layout->shape.dims[0], expected_condition_row_count);
  EXPECT_EQ(condition_layout->shape.dims[1], expected_token_count);
  EXPECT_EQ(
      condition_layout->byte_length,
      expected_condition_row_count * expected_token_count * sizeof(float));
  EXPECT_EQ(result.condition_binding.length, condition_layout->byte_length);
}

TEST(Ideogram4QwenSessionIntegration,
     IssuesRequestLoweredQwenForDynamicPromptBuckets) {
  const PromptBucket prompt_buckets[] = {
      {
          // Short prompt bucket.
          "short128",
          iree_make_cstring_view(kShortPrompt128),
      },
      {
          // Medium prompt bucket.
          "medium128",
          iree_make_cstring_view(kMediumPrompt128),
      },
      {
          // Structured prompt bucket.
          "structured128",
          iree_make_cstring_view(kStructuredPrompt128),
      },
  };

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  TokenizerRef tokenizer;
  IREE_ASSERT_OK(LoadTokenizer(tokenizer.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  SessionRef session;
  IREE_ASSERT_OK(
      CreateLoadedSession(context, &diagnostics_sink, session.out()));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  ParameterProviderRef parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), parameter_provider.out()));

  HalSemaphoreRef completion_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, completion_semaphore.out()));

  uint32_t previous_token_count = 0;
  uint64_t completion_value = 0;
  for (const PromptBucket& bucket : prompt_buckets) {
    SCOPED_TRACE(bucket.label);
    ParsedRequest request;
    IREE_ASSERT_OK(ParseRequestJson(bucket.json, &request));

    uint32_t expected_token_count = 0;
    IREE_ASSERT_OK(CountRequestTokens(&request.value, tokenizer.get(),
                                      &expected_token_count));
    EXPECT_GT(expected_token_count, previous_token_count);
    previous_token_count = expected_token_count;

    ++completion_value;
    QwenExecutionRef execution;
    IREE_ASSERT_OK(IssueQwenRequest(
        context, session.get(), tokenizer.get(), parameter_provider.get(),
        kernel_library.get(), &request.value, &diagnostics_sink,
        completion_semaphore.get(), completion_value, execution.out()));
    ExpectConditionLayoutForTokenCount(execution.get(), expected_token_count);
  }
}

TEST(Ideogram4QwenSessionIntegration,
     IssuesRequestLoweredQwenAgainstFixtureCondition) {
  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  TokenizerRef tokenizer;
  IREE_ASSERT_OK(LoadTokenizer(tokenizer.out()));

  ParsedRequest request;
  IREE_ASSERT_OK(ParseRequestFromFlags(&request));

  id4::test::FixtureTensorSet fixture_tensors;
  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required";
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));
  const id4::test::FixtureTensor* expected_condition =
      fixture_tensors.FindTensor(IREE_SV("expected"), IREE_SV("condition"));
  ASSERT_NE(expected_condition, nullptr)
      << "fixture must provide an expected condition tensor";

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  SessionRef session;
  IREE_ASSERT_OK(
      CreateLoadedSession(context, &diagnostics_sink, session.out()));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  ParameterProviderRef parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), parameter_provider.out()));

  HalSemaphoreRef completion_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, completion_semaphore.out()));

  QwenExecutionRef execution;
  IREE_ASSERT_OK(IssueQwenRequest(
      context, session.get(), tokenizer.get(), parameter_provider.get(),
      kernel_library.get(), &request.value, &diagnostics_sink,
      completion_semaphore.get(), 1, execution.out()));

  id4_ideogram4_qwen_result_t result;
  std::memset(&result, 0, sizeof(result));
  IREE_ASSERT_OK(id4_ideogram4_qwen_execution_result(execution.get(), &result));
  uint32_t expected_token_count = 0;
  IREE_ASSERT_OK(id4::test::InferRank1TensorLengthFromFixture(
      fixture_tensors, IREE_SV("token_ids"), ID4_PIPELINE_TENSOR_DTYPE_I32,
      &expected_token_count));
  EXPECT_EQ(result.token_count, expected_token_count);

  const id4_pipeline_plan_t* plan =
      id4_ideogram4_qwen_execution_plan(execution.get());
  const id4_pipeline_tensor_layout_t* condition_layout =
      FindBoundaryLayout(plan, IREE_SV("condition"));
  ASSERT_NE(condition_layout, nullptr);

  id4::test::SemaphoreListStorage completion_wait;
  completion_wait.semaphore = completion_semaphore.get();
  completion_wait.payload_value = 1;
  IREE_ASSERT_OK(id4::test::CompareBindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &result.condition_binding, completion_wait.list(), condition_layout,
      *expected_condition));
}

}  // namespace
