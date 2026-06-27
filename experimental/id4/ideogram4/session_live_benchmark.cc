// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/testing/benchmark.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, id4_tokenizer, "",
          "Hugging Face tokenizer JSON used by live generation benchmarks.");
IREE_FLAG(string, dit_parameter_format, "bf16",
          "DiT parameter format: bf16 or fp8_e4m3.");
IREE_FLAG(string, dit_activation_format, "bf16_linear_input",
          "DiT activation format: bf16_linear_input or f32_canonical.");
IREE_FLAG(string, dit_attention_implementation, "streaming",
          "DiT attention implementation: streaming or materialized_wmma.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 source parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 source parameter scope.");

namespace {

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

struct GenerationPrompt {
  // Full generation request JSON passed through the production parser.
  iree_string_view_t json;
};

static const GenerationPrompt kShortPrompt = {
    // Request JSON for the short prompt bucket.
    iree_make_cstring_view(kShortPrompt128),
};

static const GenerationPrompt kMediumPrompt = {
    // Request JSON for the medium prompt bucket.
    iree_make_cstring_view(kMediumPrompt128),
};

static const GenerationPrompt kStructuredPrompt = {
    // Request JSON for the structured prompt bucket.
    iree_make_cstring_view(kStructuredPrompt128),
};

using GenerationPlanRef =
    id4::test::OwningRef<id4_ideogram4_generation_plan_t,
                         id4_ideogram4_generation_plan_release>;
using GenerationBundleRef =
    id4::test::OwningRef<id4_ideogram4_generation_bundle_t,
                         id4_ideogram4_generation_bundle_release>;
using GenerationExecutionRef =
    id4::test::OwningRef<id4_ideogram4_generation_execution_t,
                         id4_ideogram4_generation_execution_release>;
using HalSemaphoreRef =
    id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>;
using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;
using SessionRef = id4::test::OwningRef<id4_ideogram4_session_t,
                                        id4_ideogram4_session_release>;
using TokenizerRef =
    id4::test::OwningRef<iree_tokenizer_t, iree_tokenizer_free>;

struct ParsedRequest {
  ~ParsedRequest() {
    if (initialized) {
      id4_ideogram4_request_deinitialize(&request, iree_allocator_system());
    }
  }

  // Parsed request owned by this wrapper.
  id4_ideogram4_request_t request = {};
  // True once |request| must be deinitialized.
  bool initialized = false;
};

struct LiveGenerationBenchmarkContext {
  ~LiveGenerationBenchmarkContext() {
    session.reset();
    vae_parameter_provider.reset();
    unconditioned_dit_parameter_provider.reset();
    conditioned_dit_parameter_provider.reset();
    qwen_parameter_provider.reset();
    kernel_library.reset();
    if (runtime_context_initialized) {
      id4_tooling_runtime_context_deinitialize(&runtime_context);
    }
    tokenizer.reset();
  }

  // Runtime context created from standard device and profiling flags.
  id4_tooling_runtime_context_t runtime_context = {};
  // True once |runtime_context| owns initialized resources.
  bool runtime_context_initialized = false;
  // Embedded Loom source library used by generation preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Provider containing Qwen3-VL text encoder weights.
  ParameterProviderRef qwen_parameter_provider;
  // Provider containing conditioned Ideogram 4 DiT weights.
  ParameterProviderRef conditioned_dit_parameter_provider;
  // Provider containing unconditioned Ideogram 4 DiT weights.
  ParameterProviderRef unconditioned_dit_parameter_provider;
  // Provider containing VAE decode weights.
  ParameterProviderRef vae_parameter_provider;
  // Loaded Ideogram 4 session under benchmark.
  SessionRef session;
  // Tokenizer used to lower prompt text into Qwen inputs.
  TokenizerRef tokenizer;
};

static id4_ideogram4_generation_plan_policy_t MakeGenerationPolicy() {
  id4_ideogram4_generation_plan_policy_t policy;
  std::memset(&policy, 0, sizeof(policy));
  policy.structure_size = sizeof(policy);
  policy.dit_attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;
  policy.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  return policy;
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t ParseDitActivationFormat(
    id4_ideogram4_dit_activation_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  iree_string_view_t value = iree_make_cstring_view(FLAG_dit_activation_format);
  if (iree_string_view_equal(value, IREE_SV("bf16_linear_input"))) {
    *out_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("f32_canonical"))) {
    *out_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--dit_activation_format must be bf16_linear_input or f32_canonical");
}

static iree_status_t ParseDitAttentionImplementation(
    id4_ideogram4_dit_attention_implementation_t* out_implementation) {
  IREE_ASSERT_ARGUMENT(out_implementation);
  iree_string_view_t value =
      iree_make_cstring_view(FLAG_dit_attention_implementation);
  if (iree_string_view_equal(value, IREE_SV("streaming"))) {
    *out_implementation = ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("materialized_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--dit_attention_implementation must be streaming or materialized_wmma");
}

static iree_status_t ParseRequest(iree_string_view_t json,
                                  ParsedRequest* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_json(
      json, iree_allocator_system(), &out_request->request));
  out_request->initialized = true;
  return iree_ok_status();
}

static iree_status_t LoadTokenizerFromFlag(iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = nullptr;

  iree_string_view_t path = iree_make_cstring_view(FLAG_id4_tokenizer);
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--id4_tokenizer is required for live generation benchmarks");
  }

  iree_io_file_contents_t* file_contents = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, iree_allocator_system(), &file_contents));
  iree_string_view_t tokenizer_json = iree_make_string_view(
      reinterpret_cast<const char*>(file_contents->const_buffer.data),
      file_contents->const_buffer.data_length);
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      tokenizer_json, iree_allocator_system(), out_tokenizer);
  iree_io_file_contents_free(file_contents);
  return status;
}

static id4_ideogram4_generation_parameter_providers_t
LiveGenerationParameterProviders(
    const LiveGenerationBenchmarkContext& context) {
  return id4_ideogram4_generation_parameter_providers_t{
      // Qwen3-VL text encoder provider.
      /*.qwen=*/context.qwen_parameter_provider.get(),
      // Conditioned DiT provider.
      /*.dit_conditioned=*/context.conditioned_dit_parameter_provider.get(),
      // Unconditioned DiT provider.
      /*.dit_unconditioned=*/
      context.unconditioned_dit_parameter_provider.get(),
      // VAE decode provider.
      /*.vae=*/context.vae_parameter_provider.get(),
  };
}

static iree_status_t CreateLoadedLiveSession(
    const id4_tooling_runtime_context_t* runtime_context,
    id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(runtime_context);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = nullptr;

  id4_ideogram4_session_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services =
      id4_tooling_runtime_context_stage_services(runtime_context);
  create_options.kernel_cache = runtime_context->kernel_cache;
  create_options.parameter_scopes.qwen = IREE_SV("qwen");
  create_options.parameter_scopes.dit_conditioned = IREE_SV("dit_cond");
  create_options.parameter_scopes.dit_unconditioned = IREE_SV("dit_uncond");
  create_options.parameter_scopes.vae = IREE_SV("vae");
  IREE_RETURN_IF_ERROR(
      ParseDitParameterFormat(&create_options.dit_parameter_format));
  switch (create_options.dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      break;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      create_options.parameter_scopes.dit_conditioned_fp8 =
          iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
      create_options.parameter_scopes.dit_unconditioned_fp8 =
          iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid DiT parameter format %" PRIu32,
                              (uint32_t)create_options.dit_parameter_format);
  }
  create_options.vae_activation_format =
      ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;

  id4_ideogram4_session_t* session = nullptr;
  IREE_RETURN_IF_ERROR(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_ideogram4_session_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  iree_status_t status = id4_ideogram4_session_load(session, &load_options);
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    id4_ideogram4_session_release(session);
  }
  return status;
}

static iree_status_t CreateParameterProviders(
    LiveGenerationBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  id4_ideogram4_dit_parameter_format_t dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_RETURN_IF_ERROR(ParseDitParameterFormat(&dit_parameter_format));
  if (dit_parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    id4_tooling_parameter_provider_request_t requests[] = {
        {
            // Qwen3-VL text encoder parameter scope.
            /*.scope=*/IREE_SV("qwen"),
            // Qwen provider output.
            /*.out_provider=*/context->qwen_parameter_provider.out(),
        },
        {
            // Conditioned DiT parameter scope.
            /*.scope=*/IREE_SV("dit_cond"),
            // Conditioned DiT provider output.
            /*.out_provider=*/
            context->conditioned_dit_parameter_provider.out(),
        },
        {
            // Unconditioned DiT parameter scope.
            /*.scope=*/IREE_SV("dit_uncond"),
            // Unconditioned DiT provider output.
            /*.out_provider=*/
            context->unconditioned_dit_parameter_provider.out(),
        },
        {
            // VAE parameter scope.
            /*.scope=*/IREE_SV("vae"),
            // VAE provider output.
            /*.out_provider=*/context->vae_parameter_provider.out(),
        },
    };
    return id4_tooling_create_parameter_providers_from_flags(
        IREE_ARRAYSIZE(requests), requests, iree_allocator_system());
  }

  ParameterProviderRef conditioned_bf16;
  ParameterProviderRef conditioned_fp8;
  ParameterProviderRef unconditioned_bf16;
  ParameterProviderRef unconditioned_fp8;
  const iree_string_view_t conditioned_fp8_scope =
      iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
  const iree_string_view_t unconditioned_fp8_scope =
      iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // Qwen3-VL text encoder parameter scope.
          /*.scope=*/IREE_SV("qwen"),
          // Qwen provider output.
          /*.out_provider=*/context->qwen_parameter_provider.out(),
      },
      {
          // Conditioned DiT BF16 parameter scope.
          /*.scope=*/IREE_SV("dit_cond"),
          // Conditioned DiT BF16 provider output.
          /*.out_provider=*/conditioned_bf16.out(),
      },
      {
          // Conditioned DiT FP8 e4m3 source parameter scope.
          /*.scope=*/conditioned_fp8_scope,
          // Conditioned DiT FP8 e4m3 source provider output.
          /*.out_provider=*/conditioned_fp8.out(),
      },
      {
          // Unconditioned DiT BF16 parameter scope.
          /*.scope=*/IREE_SV("dit_uncond"),
          // Unconditioned DiT BF16 provider output.
          /*.out_provider=*/unconditioned_bf16.out(),
      },
      {
          // Unconditioned DiT FP8 e4m3 source parameter scope.
          /*.scope=*/unconditioned_fp8_scope,
          // Unconditioned DiT FP8 e4m3 source provider output.
          /*.out_provider=*/unconditioned_fp8.out(),
      },
      {
          // VAE parameter scope.
          /*.scope=*/IREE_SV("vae"),
          // VAE provider output.
          /*.out_provider=*/context->vae_parameter_provider.out(),
      },
  };
  IREE_RETURN_IF_ERROR(id4_tooling_create_parameter_providers_from_flags(
      IREE_ARRAYSIZE(requests), requests, iree_allocator_system()));

  const id4_tooling_parameter_provider_set_entry_t conditioned_entries[] = {
      {
          // Conditioned DiT BF16 parameter scope.
          /*.scope=*/IREE_SV("dit_cond"),
          // Conditioned DiT BF16 provider.
          /*.provider=*/conditioned_bf16.get(),
      },
      {
          // Conditioned DiT FP8 e4m3 source parameter scope.
          /*.scope=*/conditioned_fp8_scope,
          // Conditioned DiT FP8 e4m3 source provider.
          /*.provider=*/conditioned_fp8.get(),
      },
  };
  IREE_RETURN_IF_ERROR(id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(conditioned_entries), conditioned_entries,
      iree_allocator_system(),
      context->conditioned_dit_parameter_provider.out()));

  const id4_tooling_parameter_provider_set_entry_t unconditioned_entries[] = {
      {
          // Unconditioned DiT BF16 parameter scope.
          /*.scope=*/IREE_SV("dit_uncond"),
          // Unconditioned DiT BF16 provider.
          /*.provider=*/unconditioned_bf16.get(),
      },
      {
          // Unconditioned DiT FP8 e4m3 source parameter scope.
          /*.scope=*/unconditioned_fp8_scope,
          // Unconditioned DiT FP8 e4m3 source provider.
          /*.provider=*/unconditioned_fp8.get(),
      },
  };
  return id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(unconditioned_entries), unconditioned_entries,
      iree_allocator_system(),
      context->unconditioned_dit_parameter_provider.out());
}

static iree_status_t CreateLiveGenerationBenchmarkContext(
    LiveGenerationBenchmarkContext* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_RETURN_IF_ERROR(LoadTokenizerFromFlag(out_context->tokenizer.out()));

  id4_tooling_runtime_context_options_t runtime_options;
  std::memset(&runtime_options, 0, sizeof(runtime_options));
  runtime_options.structure_size = sizeof(runtime_options);
  runtime_options.executable_cache_identifier =
      IREE_SV("id4_session_live_benchmark");
  IREE_RETURN_IF_ERROR(id4_tooling_runtime_context_initialize_from_flags(
      &runtime_options, iree_allocator_system(),
      &out_context->runtime_context));
  out_context->runtime_context_initialized = true;

  IREE_RETURN_IF_ERROR(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), out_context->kernel_library.out()));
  IREE_RETURN_IF_ERROR(CreateLoadedLiveSession(&out_context->runtime_context,
                                               out_context->session.out()));
  return CreateParameterProviders(out_context);
}

static iree_status_t CreateGenerationPlan(
    const LiveGenerationBenchmarkContext& context, const ParsedRequest& request,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_ideogram4_generation_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.request = &request.request;
  plan_options.tokenizer = context.tokenizer.get();
  plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  plan_options.policy = MakeGenerationPolicy();
  IREE_RETURN_IF_ERROR(
      ParseDitActivationFormat(&plan_options.policy.dit_activation_format));
  IREE_RETURN_IF_ERROR(ParseDitAttentionImplementation(
      &plan_options.policy.dit_attention_implementation));
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_session_plan_generation(context.session.get(),
                                               &plan_options, out_plan);
}

static iree_status_t PrepareGenerationBundle(
    const LiveGenerationBenchmarkContext& context,
    const id4_ideogram4_generation_plan_t* plan,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* prepare_semaphore, uint64_t prepare_value,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  id4::test::SemaphoreListStorage prepare_signal;
  prepare_signal.semaphore = prepare_semaphore;
  prepare_signal.payload_value = prepare_value;

  id4_ideogram4_generation_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_providers =
      LiveGenerationParameterProviders(context);
  prepare_options.kernel_library = context.kernel_library.get();
  prepare_options.command_buffer_mode =
      context.runtime_context.command_buffer_mode;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal.list();
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_session_prepare_generation(context.session.get(), plan,
                                                  &prepare_options, out_bundle);
}

static iree_status_t IssueGenerationBundle(
    const LiveGenerationBenchmarkContext& context,
    id4_ideogram4_generation_bundle_t* bundle, const ParsedRequest& request,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* prepare_semaphore, uint64_t prepare_value,
    iree_hal_semaphore_t* completion_semaphore, uint64_t completion_value,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = nullptr;

  id4::test::SemaphoreListStorage prepare_wait;
  prepare_wait.semaphore = prepare_semaphore;
  prepare_wait.payload_value = prepare_value;
  id4::test::SemaphoreListStorage completion_signal;
  completion_signal.semaphore = completion_semaphore;
  completion_signal.payload_value = completion_value;

  id4_ideogram4_generation_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.request = &request.request;
  issue_options.tokenizer = context.tokenizer.get();
  issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  issue_options.wait_semaphore_list = prepare_wait.list();
  issue_options.signal_semaphore_list = completion_signal.list();
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_session_issue_generation(context.session.get(), bundle,
                                                &issue_options, out_execution);
}

static iree_status_t RunGenerationEndToEndBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const GenerationPrompt* prompt =
      static_cast<const GenerationPrompt*>(benchmark_def->user_data);

  LiveGenerationBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLiveGenerationBenchmarkContext(&context));

  ParsedRequest request;
  IREE_RETURN_IF_ERROR(ParseRequest(prompt->json, &request));

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  iree_hal_device_t* device =
      id4_tooling_runtime_context_primary_device(&context.runtime_context);
  HalSemaphoreRef prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      prepare_semaphore.out()));
  HalSemaphoreRef completion_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
      completion_semaphore.out()));

  uint64_t prepare_value = 0;
  uint64_t completion_value = 0;
  uint64_t iteration_count = 0;
  uint32_t token_count = 0;
  id4_ideogram4_generation_plan_summary_t last_summary;
  std::memset(&last_summary, 0, sizeof(last_summary));
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context.runtime_context.device_group, iree_allocator_system(),
      &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    GenerationPlanRef plan;
    status =
        CreateGenerationPlan(context, request, &diagnostics_sink, plan.out());
    id4_ideogram4_generation_plan_summary_t summary;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_plan_summary(plan.get(), &summary);
    }
    if (iree_status_is_ok(status)) {
      token_count = summary.qwen_token_count;
      last_summary = summary;
    }

    ++prepare_value;
    GenerationBundleRef bundle;
    if (iree_status_is_ok(status)) {
      status = PrepareGenerationBundle(context, plan.get(), &diagnostics_sink,
                                       prepare_semaphore.get(), prepare_value,
                                       bundle.out());
    }

    ++completion_value;
    GenerationExecutionRef execution;
    if (iree_status_is_ok(status)) {
      status = IssueGenerationBundle(context, bundle.get(), request,
                                     &diagnostics_sink, prepare_semaphore.get(),
                                     prepare_value, completion_semaphore.get(),
                                     completion_value, execution.out());
    }
    if (iree_status_is_ok(status)) {
      id4::test::SemaphoreListStorage completion_wait;
      completion_wait.semaphore = completion_semaphore.get();
      completion_wait.payload_value = completion_value;
      status = iree_hal_semaphore_list_wait(completion_wait.list(),
                                            iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      iree_optimization_barrier(execution.get());
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  char label[160];
  std::snprintf(label, sizeof(label),
                "tokens=%" PRIu32 " latent=%" PRIu64 "x%" PRIu64
                " steps=%" PRIu32 " image=%" PRIu64 "x%" PRIu64,
                last_summary.qwen_token_count,
                last_summary.diffusion_latent_shape.dims[0],
                last_summary.diffusion_latent_shape.dims[1],
                last_summary.denoise_step_count,
                last_summary.decoded_image_shape.dims[0],
                last_summary.decoded_image_shape.dims[1]);
  iree_benchmark_set_label(benchmark_state, label);
  iree_benchmark_set_items_processed(
      benchmark_state, static_cast<int64_t>(iteration_count * token_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4SessionGenerationEndToEnd) {
  return RunGenerationEndToEndBenchmark(benchmark_def, benchmark_state);
}

static const iree_benchmark_def_t* RegisterGenerationBenchmark(
    iree_string_view_t name, const GenerationPrompt* prompt) {
  iree_benchmark_def_t* benchmark =
      iree_make_function_benchmark(BM_Ideogram4SessionGenerationEndToEnd);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = IREE_BENCHMARK_UNIT_MILLISECOND;
  benchmark->user_data = prompt;
  return iree_benchmark_register(name, benchmark);
}

static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationEndToEndShort128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationEndToEnd/short128"),
            &kShortPrompt);
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationEndToEndMedium128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationEndToEnd/medium128"),
            &kMediumPrompt);
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationEndToEndStructured128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationEndToEnd/structured128"),
            &kStructuredPrompt);

}  // namespace
