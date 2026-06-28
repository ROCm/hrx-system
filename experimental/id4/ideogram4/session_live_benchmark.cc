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
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3.");
IREE_FLAG(string, dit_activation_format, "bf16_linear_input",
          "DiT activation format: bf16_linear_input or f32_canonical.");
IREE_FLAG(string, dit_attention_implementation, "blocked_wmma",
          "DiT attention implementation: streaming, materialized_wmma, "
          "blocked_wmma, or online_wmma.");
IREE_FLAG(string, dit_feed_forward_implementation, "pytorch_parity",
          "DiT feed-forward implementation: fused_product or "
          "pytorch_parity.");
IREE_FLAG(string, generation_residency, "issue_phases",
          "Generation stage-bundle residency: issue_phases or "
          "selected_phases or all_stage_bundles.");
IREE_FLAG(string, generation_issue_mode, "full",
          "Generation issue mode: full or phases.");
IREE_FLAG(string, generation_resident_phases, "",
          "Comma-separated generation phases retained when "
          "--generation_residency=selected_phases: conditioning, denoise, "
          "decode, or all.");
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

struct GenerationBenchmarkPlanStatistics {
  // Total parameter slab bytes across coarse stage plans.
  iree_device_size_t total_parameter_slab_byte_length;
  // Largest individual parameter slab bytes across coarse stage plans.
  iree_device_size_t largest_parameter_slab_byte_length;
  // Sum of local slab high-water bytes across coarse stage plans.
  iree_device_size_t total_local_slab_high_water_mark;
  // Largest local slab high-water bytes across coarse stage plans.
  iree_device_size_t largest_local_slab_high_water_mark;
  // Total boundary tensor bytes across coarse stage plans.
  iree_device_size_t boundary_tensor_byte_length;
  // Total planned kernel specializations across coarse stage plans.
  iree_host_size_t kernel_count;
  // Total planned dispatches across coarse stage plans.
  iree_host_size_t dispatch_count;
  // Number of populated per-stage statistics entries.
  iree_host_size_t stage_count;
  // Per-stage statistics in generation-plan stage order.
  struct {
    // Stable generation-plan key for this coarse stage.
    iree_string_view_t key;
    // Statistics for this coarse stage plan.
    id4_pipeline_plan_statistics_t statistics;
  } stages[8];
};

struct GenerationPhaseBenchmarkTimingStatistics {
  // Host-observed phase bundle preparation duration.
  iree_duration_t prepare_ns;
  // Host-observed phase issue submission duration.
  iree_duration_t issue_ns;
  // Host-observed wait for phase completion before release.
  iree_duration_t completion_wait_ns;
  // Host-observed phase bundle release duration.
  iree_duration_t release_ns;
};

struct GenerationBenchmarkTimingStatistics {
  // Host-observed generation planning duration.
  iree_duration_t plan_ns;
  // Host-observed generation bundle preparation duration.
  iree_duration_t prepare_ns;
  // Host-observed full generation issue submission duration.
  iree_duration_t issue_ns;
  // Host-observed phase-driven generation begin duration.
  iree_duration_t begin_ns;
  // Host-observed conditioning phase timings.
  GenerationPhaseBenchmarkTimingStatistics conditioning;
  // Host-observed denoise phase timings.
  GenerationPhaseBenchmarkTimingStatistics denoise;
  // Host-observed decode phase timings.
  GenerationPhaseBenchmarkTimingStatistics decode;
  // Host-observed final completion wait duration.
  iree_duration_t final_wait_ns;
  // Host-observed whole benchmark iteration duration.
  iree_duration_t total_ns;
};

enum class GenerationIssueMode {
  kFull,
  kPhases,
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
  // DiT parameter source policy selected by benchmark flags.
  id4_ideogram4_dit_parameter_format_t dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  // Generation stage-bundle residency selected by benchmark flags.
  id4_ideogram4_generation_residency_mode_t generation_residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  // Generation issue mode selected by benchmark flags.
  GenerationIssueMode generation_issue_mode = GenerationIssueMode::kFull;
  // Generation residency phases selected by benchmark flags.
  id4_ideogram4_generation_residency_phase_mask_t
      generation_resident_phase_mask =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_NONE;
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
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;
  policy.dit_feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
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

static iree_string_view_t DitActivationFormatName(
    id4_ideogram4_dit_activation_format_t format) {
  switch (format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
      return IREE_SV("f32_canonical");
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      return IREE_SV("bf16_linear_input");
    default:
      return IREE_SV("invalid");
  }
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
  if (iree_string_view_equal(value, IREE_SV("blocked_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("online_wmma"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--dit_attention_implementation must be streaming, materialized_wmma, "
      "blocked_wmma, or online_wmma");
}

static iree_string_view_t DitAttentionImplementationName(
    id4_ideogram4_dit_attention_implementation_t implementation) {
  switch (implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
      return IREE_SV("streaming");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
      return IREE_SV("materialized_wmma");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
      return IREE_SV("blocked_wmma");
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      return IREE_SV("online_wmma");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t ParseDitFeedForwardImplementation(
    id4_ideogram4_dit_feed_forward_implementation_t* out_implementation) {
  IREE_ASSERT_ARGUMENT(out_implementation);
  iree_string_view_t value =
      iree_make_cstring_view(FLAG_dit_feed_forward_implementation);
  if (iree_string_view_equal(value, IREE_SV("fused_product"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("pytorch_parity"))) {
    *out_implementation =
        ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "--dit_feed_forward_implementation must be "
                          "fused_product or pytorch_parity");
}

static iree_string_view_t DitFeedForwardImplementationName(
    id4_ideogram4_dit_feed_forward_implementation_t implementation) {
  switch (implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
      return IREE_SV("fused_product");
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      return IREE_SV("pytorch_parity");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t ParseGenerationResidencyMode(
    id4_ideogram4_generation_residency_mode_t* out_mode) {
  IREE_ASSERT_ARGUMENT(out_mode);
  iree_string_view_t value = iree_make_cstring_view(FLAG_generation_residency);
  if (iree_string_view_equal(value, IREE_SV("issue_phases"))) {
    *out_mode = ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("selected_phases"))) {
    *out_mode = ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_PHASES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("all_stage_bundles"))) {
    *out_mode = ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--generation_residency must be issue_phases, selected_phases, or "
      "all_stage_bundles");
}

static iree_string_view_t GenerationResidencyModeName(
    id4_ideogram4_generation_residency_mode_t mode) {
  switch (mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
      return IREE_SV("issue_phases");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_PHASES:
      return IREE_SV("selected_phases");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      return IREE_SV("all_stage_bundles");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t ParseGenerationIssueMode(
    GenerationIssueMode* out_issue_mode) {
  IREE_ASSERT_ARGUMENT(out_issue_mode);
  iree_string_view_t value = iree_make_cstring_view(FLAG_generation_issue_mode);
  if (iree_string_view_equal(value, IREE_SV("full"))) {
    *out_issue_mode = GenerationIssueMode::kFull;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("phases"))) {
    *out_issue_mode = GenerationIssueMode::kPhases;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "--generation_issue_mode must be full or phases");
}

static iree_string_view_t GenerationIssueModeName(GenerationIssueMode mode) {
  switch (mode) {
    case GenerationIssueMode::kFull:
      return IREE_SV("full");
    case GenerationIssueMode::kPhases:
      return IREE_SV("phases");
  }
  return IREE_SV("unknown");
}

static iree_status_t ParseGenerationResidentPhaseMask(
    id4_ideogram4_generation_residency_phase_mask_t* out_phase_mask) {
  IREE_ASSERT_ARGUMENT(out_phase_mask);
  *out_phase_mask = ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_NONE;
  iree_string_view_t remaining = iree_string_view_trim(
      iree_make_cstring_view(FLAG_generation_resident_phases));
  if (iree_string_view_is_empty(remaining)) return iree_ok_status();

  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t phase_name = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &phase_name, &remaining);
    phase_name = iree_string_view_trim(phase_name);
    if (iree_string_view_is_empty(phase_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_resident_phases contains an empty phase name");
    }
    if (iree_string_view_equal(phase_name, IREE_SV("all"))) {
      *out_phase_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_ALL;
    } else if (iree_string_view_equal(phase_name, IREE_SV("conditioning"))) {
      *out_phase_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_CONDITIONING;
    } else if (iree_string_view_equal(phase_name, IREE_SV("denoise"))) {
      *out_phase_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_DENOISE;
    } else if (iree_string_view_equal(phase_name, IREE_SV("decode"))) {
      *out_phase_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_DECODE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown --generation_resident_phases value '%.*s'",
          (int)phase_name.size, phase_name.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static uint64_t CeilMiB(iree_device_size_t byte_length) {
  static constexpr iree_device_size_t kMiB = 1024ull * 1024ull;
  return (uint64_t)((byte_length + kMiB - 1) / kMiB);
}

static double AverageMilliseconds(iree_duration_t total_ns,
                                  uint64_t iteration_count) {
  if (iteration_count == 0) return 0.0;
  return static_cast<double>(total_ns) /
         (static_cast<double>(iteration_count) * 1000.0 * 1000.0);
}

static void AddPhaseTiming(
    GenerationPhaseBenchmarkTimingStatistics* target,
    const GenerationPhaseBenchmarkTimingStatistics& add) {
  target->prepare_ns += add.prepare_ns;
  target->issue_ns += add.issue_ns;
  target->completion_wait_ns += add.completion_wait_ns;
  target->release_ns += add.release_ns;
}

static void AddTiming(GenerationBenchmarkTimingStatistics* target,
                      const GenerationBenchmarkTimingStatistics& add) {
  target->plan_ns += add.plan_ns;
  target->prepare_ns += add.prepare_ns;
  target->issue_ns += add.issue_ns;
  target->begin_ns += add.begin_ns;
  AddPhaseTiming(&target->conditioning, add.conditioning);
  AddPhaseTiming(&target->denoise, add.denoise);
  AddPhaseTiming(&target->decode, add.decode);
  target->final_wait_ns += add.final_wait_ns;
  target->total_ns += add.total_ns;
}

static iree_status_t AccumulateGenerationBenchmarkPlanStatistics(
    const id4_ideogram4_generation_plan_t* plan,
    GenerationBenchmarkPlanStatistics* out_statistics) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_statistics);
  std::memset(out_statistics, 0, sizeof(*out_statistics));

  const iree_host_size_t stage_count =
      id4_ideogram4_generation_plan_stage_count(plan);
  if (stage_count > IREE_ARRAYSIZE(out_statistics->stages)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "generation benchmark supports at most %" PRIhsz
                            " coarse stages but plan contains %" PRIhsz,
                            IREE_ARRAYSIZE(out_statistics->stages),
                            stage_count);
  }
  out_statistics->stage_count = stage_count;
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    iree_string_view_t stage_key = iree_string_view_empty();
    const id4_pipeline_plan_t* stage_plan = nullptr;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_stage_at(
        plan, i, &stage_key, &stage_plan));
    id4_pipeline_plan_statistics_t stage_statistics =
        id4_pipeline_plan_statistics(stage_plan);
    out_statistics->stages[i].key = stage_key;
    out_statistics->stages[i].statistics = stage_statistics;
    out_statistics->total_parameter_slab_byte_length +=
        stage_statistics.parameter_slab_byte_length;
    if (stage_statistics.largest_parameter_slab_byte_length >
        out_statistics->largest_parameter_slab_byte_length) {
      out_statistics->largest_parameter_slab_byte_length =
          stage_statistics.largest_parameter_slab_byte_length;
    }
    out_statistics->total_local_slab_high_water_mark +=
        stage_statistics.memory_slab_high_water_mark;
    if (stage_statistics.memory_slab_high_water_mark >
        out_statistics->largest_local_slab_high_water_mark) {
      out_statistics->largest_local_slab_high_water_mark =
          stage_statistics.memory_slab_high_water_mark;
    }
    out_statistics->boundary_tensor_byte_length +=
        stage_statistics.boundary_tensor_byte_length;
    out_statistics->kernel_count += stage_statistics.kernel_count;
    out_statistics->dispatch_count += stage_statistics.dispatch_count;
  }
  return iree_ok_status();
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
    id4_ideogram4_dit_parameter_format_t dit_parameter_format,
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
  create_options.dit_parameter_format = dit_parameter_format;
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
  if (context->dit_parameter_format ==
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
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
  IREE_RETURN_IF_ERROR(
      ParseDitParameterFormat(&out_context->dit_parameter_format));
  IREE_RETURN_IF_ERROR(
      ParseGenerationResidencyMode(&out_context->generation_residency_mode));
  IREE_RETURN_IF_ERROR(
      ParseGenerationIssueMode(&out_context->generation_issue_mode));
  IREE_RETURN_IF_ERROR(ParseGenerationResidentPhaseMask(
      &out_context->generation_resident_phase_mask));
  IREE_RETURN_IF_ERROR(CreateLoadedLiveSession(
      &out_context->runtime_context, out_context->dit_parameter_format,
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
  IREE_RETURN_IF_ERROR(ParseDitFeedForwardImplementation(
      &plan_options.policy.dit_feed_forward_implementation));
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
  prepare_options.residency_mode = context.generation_residency_mode;
  prepare_options.resident_phase_mask = context.generation_resident_phase_mask;
  prepare_options.command_buffer_mode =
      context.runtime_context.command_buffer_mode;
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = prepare_signal.list();
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_session_prepare_generation(context.session.get(), plan,
                                                  &prepare_options, out_bundle);
}

static iree_status_t WaitForSemaphore(iree_hal_semaphore_t* semaphore,
                                      uint64_t payload_value) {
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = semaphore;
  wait.payload_value = payload_value;
  return iree_hal_semaphore_list_wait(wait.list(), iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t PrepareGenerationPhaseBundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_residency_phase_mask_t phase_mask,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value,
    id4_ideogram4_generation_phase_bundle_t** out_phase_bundle) {
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = signal_semaphore;
  signal.payload_value = signal_value;

  id4_ideogram4_generation_phase_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.phase_mask = phase_mask;
  prepare_options.wait_semaphore_list = wait.list();
  prepare_options.signal_semaphore_list = signal.list();
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_generation_bundle_prepare_phase(bundle, &prepare_options,
                                                       out_phase_bundle);
}

static iree_status_t IssueGenerationPhase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_value) {
  id4::test::SemaphoreListStorage wait;
  wait.semaphore = wait_semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = signal_semaphore;
  signal.payload_value = signal_value;

  id4_ideogram4_generation_phase_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = wait.list();
  issue_options.signal_semaphore_list = signal.list();
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_generation_execution_issue_phase(execution, phase_bundle,
                                                        &issue_options);
}

static iree_status_t IssueGenerationBundle(
    const LiveGenerationBenchmarkContext& context,
    id4_ideogram4_generation_bundle_t* bundle, const ParsedRequest& request,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* prepare_semaphore, uint64_t* prepare_value,
    iree_hal_semaphore_t* completion_semaphore, uint64_t* completion_value,
    GenerationBenchmarkTimingStatistics* timing,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(timing);
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = nullptr;

  if (context.generation_issue_mode == GenerationIssueMode::kFull) {
    id4::test::SemaphoreListStorage prepare_wait;
    prepare_wait.semaphore = prepare_semaphore;
    prepare_wait.payload_value = *prepare_value;
    id4::test::SemaphoreListStorage completion_signal;
    completion_signal.semaphore = completion_semaphore;
    completion_signal.payload_value = ++*completion_value;

    id4_ideogram4_generation_issue_options_t issue_options;
    std::memset(&issue_options, 0, sizeof(issue_options));
    issue_options.structure_size = sizeof(issue_options);
    issue_options.request = &request.request;
    issue_options.tokenizer = context.tokenizer.get();
    issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    issue_options.wait_semaphore_list = prepare_wait.list();
    issue_options.signal_semaphore_list = completion_signal.list();
    issue_options.diagnostics_sink = diagnostics_sink;
    const iree_time_t issue_start_time_ns = iree_time_now();
    iree_status_t status = id4_ideogram4_session_issue_generation(
        context.session.get(), bundle, &issue_options, out_execution);
    timing->issue_ns += iree_time_now() - issue_start_time_ns;
    return status;
  }

  id4::test::SemaphoreListStorage begin_wait;
  begin_wait.semaphore = prepare_semaphore;
  begin_wait.payload_value = *prepare_value;
  id4::test::SemaphoreListStorage begin_signal;
  begin_signal.semaphore = prepare_semaphore;
  begin_signal.payload_value = ++*prepare_value;

  id4_ideogram4_generation_begin_options_t begin_options;
  std::memset(&begin_options, 0, sizeof(begin_options));
  begin_options.structure_size = sizeof(begin_options);
  begin_options.request = &request.request;
  begin_options.tokenizer = context.tokenizer.get();
  begin_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  begin_options.wait_semaphore_list = begin_wait.list();
  begin_options.signal_semaphore_list = begin_signal.list();
  begin_options.diagnostics_sink = diagnostics_sink;
  iree_time_t phase_start_time_ns = iree_time_now();
  iree_status_t status = id4_ideogram4_session_begin_generation(
      context.session.get(), bundle, &begin_options, out_execution);
  timing->begin_ns += iree_time_now() - phase_start_time_ns;
  IREE_RETURN_IF_ERROR(status);

  id4_ideogram4_generation_phase_bundle_t* conditioning_phase = nullptr;
  id4_ideogram4_generation_phase_bundle_t* denoise_phase = nullptr;
  id4_ideogram4_generation_phase_bundle_t* decode_phase = nullptr;

  phase_start_time_ns = iree_time_now();
  status = PrepareGenerationPhaseBundle(
      bundle, ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_CONDITIONING,
      diagnostics_sink, prepare_semaphore, *prepare_value, prepare_semaphore,
      ++*prepare_value, &conditioning_phase);
  timing->conditioning.prepare_ns += iree_time_now() - phase_start_time_ns;
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, conditioning_phase, diagnostics_sink, prepare_semaphore,
        *prepare_value, completion_semaphore, ++*completion_value);
    timing->conditioning.issue_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = WaitForSemaphore(completion_semaphore, *completion_value);
    timing->conditioning.completion_wait_ns +=
        iree_time_now() - phase_start_time_ns;
  }
  phase_start_time_ns = iree_time_now();
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_release(conditioning_phase));
  timing->conditioning.release_ns += iree_time_now() - phase_start_time_ns;

  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = PrepareGenerationPhaseBundle(
        bundle, ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_DENOISE,
        diagnostics_sink, completion_semaphore, *completion_value,
        prepare_semaphore, ++*prepare_value, &denoise_phase);
    timing->denoise.prepare_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, denoise_phase, diagnostics_sink, prepare_semaphore,
        *prepare_value, completion_semaphore, ++*completion_value);
    timing->denoise.issue_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = WaitForSemaphore(completion_semaphore, *completion_value);
    timing->denoise.completion_wait_ns += iree_time_now() - phase_start_time_ns;
  }
  phase_start_time_ns = iree_time_now();
  status = iree_status_join(
      status, id4_ideogram4_generation_phase_bundle_release(denoise_phase));
  timing->denoise.release_ns += iree_time_now() - phase_start_time_ns;

  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = PrepareGenerationPhaseBundle(
        bundle, ID4_IDEOGRAM4_GENERATION_RESIDENCY_PHASE_DECODE,
        diagnostics_sink, completion_semaphore, *completion_value,
        prepare_semaphore, ++*prepare_value, &decode_phase);
    timing->decode.prepare_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, decode_phase, diagnostics_sink, prepare_semaphore,
        *prepare_value, completion_semaphore, ++*completion_value);
    timing->decode.issue_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = WaitForSemaphore(completion_semaphore, *completion_value);
    timing->decode.completion_wait_ns += iree_time_now() - phase_start_time_ns;
  }
  phase_start_time_ns = iree_time_now();
  status = iree_status_join(
      status, id4_ideogram4_generation_phase_bundle_release(decode_phase));
  timing->decode.release_ns += iree_time_now() - phase_start_time_ns;
  return status;
}

static iree_status_t AppendGenerationBenchmarkPhaseTimingLabel(
    iree_string_builder_t* builder, iree_string_view_t phase_name,
    const GenerationPhaseBenchmarkTimingStatistics& timing,
    uint64_t iteration_count) {
  return iree_string_builder_append_format(
      builder, " phase.%.*s_ms[prepare=%.3f,issue=%.3f,wait=%.3f,release=%.3f]",
      static_cast<int>(phase_name.size), phase_name.data,
      AverageMilliseconds(timing.prepare_ns, iteration_count),
      AverageMilliseconds(timing.issue_ns, iteration_count),
      AverageMilliseconds(timing.completion_wait_ns, iteration_count),
      AverageMilliseconds(timing.release_ns, iteration_count));
}

static iree_status_t AppendGenerationBenchmarkTimingLabel(
    iree_string_builder_t* builder,
    const LiveGenerationBenchmarkContext& context,
    const GenerationBenchmarkTimingStatistics& timing,
    uint64_t iteration_count) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      " timing_ms[plan=%.3f,prepare=%.3f,issue=%.3f,begin=%.3f,final_wait=%.3f,"
      "total=%.3f]",
      AverageMilliseconds(timing.plan_ns, iteration_count),
      AverageMilliseconds(timing.prepare_ns, iteration_count),
      AverageMilliseconds(timing.issue_ns, iteration_count),
      AverageMilliseconds(timing.begin_ns, iteration_count),
      AverageMilliseconds(timing.final_wait_ns, iteration_count),
      AverageMilliseconds(timing.total_ns, iteration_count)));
  if (context.generation_issue_mode != GenerationIssueMode::kPhases) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(AppendGenerationBenchmarkPhaseTimingLabel(
      builder, IREE_SV("conditioning"), timing.conditioning, iteration_count));
  IREE_RETURN_IF_ERROR(AppendGenerationBenchmarkPhaseTimingLabel(
      builder, IREE_SV("denoise"), timing.denoise, iteration_count));
  return AppendGenerationBenchmarkPhaseTimingLabel(
      builder, IREE_SV("decode"), timing.decode, iteration_count);
}

static iree_status_t AppendGenerationBenchmarkStageLabels(
    iree_string_builder_t* builder,
    const GenerationBenchmarkPlanStatistics& statistics) {
  for (iree_host_size_t i = 0; i < statistics.stage_count; ++i) {
    const iree_string_view_t key = statistics.stages[i].key;
    const id4_pipeline_plan_statistics_t& stage =
        statistics.stages[i].statistics;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        " stage.%.*s[param=%" PRIu64 "MiB,local_hw=%" PRIu64
        "MiB,boundary=%" PRIu64 "MiB,kernels=%" PRIhsz ",dispatches=%" PRIhsz
        "]",
        static_cast<int>(key.size), key.data,
        CeilMiB(stage.parameter_slab_byte_length),
        CeilMiB(stage.memory_slab_high_water_mark),
        CeilMiB(stage.boundary_tensor_byte_length), stage.kernel_count,
        stage.dispatch_count));
  }
  return iree_ok_status();
}

static iree_status_t SetGenerationBenchmarkLabel(
    iree_benchmark_state_t* benchmark_state,
    const LiveGenerationBenchmarkContext& context,
    const id4_ideogram4_generation_plan_summary_t& summary,
    const GenerationBenchmarkPlanStatistics& statistics,
    const GenerationBenchmarkTimingStatistics& timing,
    uint64_t iteration_count) {
  const iree_string_view_t parameter_format =
      id4_ideogram4_dit_parameter_format_name(context.dit_parameter_format);
  const iree_string_view_t activation_format =
      DitActivationFormatName(summary.dit_activation_format);
  const iree_string_view_t attention_implementation =
      DitAttentionImplementationName(summary.dit_attention_implementation);
  const iree_string_view_t feed_forward_implementation =
      DitFeedForwardImplementationName(summary.dit_feed_forward_implementation);
  const iree_string_view_t residency_mode =
      GenerationResidencyModeName(context.generation_residency_mode);
  const iree_string_view_t issue_mode =
      GenerationIssueModeName(context.generation_issue_mode);
  iree_string_builder_t label_builder;
  iree_string_builder_initialize(iree_allocator_system(), &label_builder);
  iree_status_t status = iree_string_builder_append_format(
      &label_builder,
      "tokens=%" PRIu32 " latent=%" PRIu64 "x%" PRIu64 " steps=%" PRIu32
      " image=%" PRIu64 "x%" PRIu64
      " residency=%.*s issue=%.*s resident_phase_mask=0x%08x"
      " params=%.*s activation=%.*s attention=%.*s ff=%.*s"
      " param_total=%" PRIu64 "MiB param_largest=%" PRIu64
      "MiB"
      " local_hw_total=%" PRIu64 "MiB local_hw_largest=%" PRIu64
      "MiB"
      " boundary=%" PRIu64 "MiB kernels=%" PRIhsz " dispatches=%" PRIhsz,
      summary.qwen_token_count, summary.diffusion_latent_shape.dims[0],
      summary.diffusion_latent_shape.dims[1], summary.denoise_step_count,
      summary.decoded_image_shape.dims[0], summary.decoded_image_shape.dims[1],
      static_cast<int>(residency_mode.size), residency_mode.data,
      static_cast<int>(issue_mode.size), issue_mode.data,
      context.generation_resident_phase_mask,
      static_cast<int>(parameter_format.size), parameter_format.data,
      static_cast<int>(activation_format.size), activation_format.data,
      static_cast<int>(attention_implementation.size),
      attention_implementation.data,
      static_cast<int>(feed_forward_implementation.size),
      feed_forward_implementation.data,
      CeilMiB(statistics.total_parameter_slab_byte_length),
      CeilMiB(statistics.largest_parameter_slab_byte_length),
      CeilMiB(statistics.total_local_slab_high_water_mark),
      CeilMiB(statistics.largest_local_slab_high_water_mark),
      CeilMiB(statistics.boundary_tensor_byte_length), statistics.kernel_count,
      statistics.dispatch_count);
  if (iree_status_is_ok(status)) {
    status = AppendGenerationBenchmarkStageLabels(&label_builder, statistics);
  }
  if (iree_status_is_ok(status)) {
    status = AppendGenerationBenchmarkTimingLabel(&label_builder, context,
                                                  timing, iteration_count);
  }
  if (iree_status_is_ok(status)) {
    iree_benchmark_set_label(benchmark_state,
                             iree_string_builder_buffer(&label_builder));
  }
  iree_string_builder_deinitialize(&label_builder);
  return status;
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
  GenerationBenchmarkPlanStatistics last_statistics;
  std::memset(&last_statistics, 0, sizeof(last_statistics));
  GenerationBenchmarkTimingStatistics timing_total;
  std::memset(&timing_total, 0, sizeof(timing_total));
  const iree_hal_command_buffer_mode_t profiled_dispatch_metadata_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  const bool capture_execution_profile =
      iree_all_bits_set(context.runtime_context.command_buffer_mode,
                        profiled_dispatch_metadata_mode);
  bool execution_profile_captured = false;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_ok_status();
  if (!capture_execution_profile) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        context.runtime_context.device_group, iree_allocator_system(),
        &profiling);
  }
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    GenerationBenchmarkTimingStatistics iteration_timing;
    std::memset(&iteration_timing, 0, sizeof(iteration_timing));
    const iree_time_t iteration_start_time_ns = iree_time_now();

    GenerationPlanRef plan;
    iree_time_t phase_start_time_ns = iree_time_now();
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
    GenerationBenchmarkPlanStatistics statistics;
    if (iree_status_is_ok(status)) {
      status =
          AccumulateGenerationBenchmarkPlanStatistics(plan.get(), &statistics);
    }
    if (iree_status_is_ok(status)) {
      last_statistics = statistics;
    }
    iteration_timing.plan_ns += iree_time_now() - phase_start_time_ns;

    ++prepare_value;
    GenerationBundleRef bundle;
    if (iree_status_is_ok(status)) {
      phase_start_time_ns = iree_time_now();
      status = PrepareGenerationBundle(context, plan.get(), &diagnostics_sink,
                                       prepare_semaphore.get(), prepare_value,
                                       bundle.out());
      iteration_timing.prepare_ns += iree_time_now() - phase_start_time_ns;
    }

    const bool profile_this_execution =
        capture_execution_profile && !execution_profile_captured;
    if (iree_status_is_ok(status) && profile_this_execution) {
      id4::test::SemaphoreListStorage prepare_wait;
      prepare_wait.semaphore = prepare_semaphore.get();
      prepare_wait.payload_value = prepare_value;
      status = iree_hal_semaphore_list_wait(prepare_wait.list(),
                                            iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status) && profile_this_execution) {
      status = iree_hal_begin_device_group_profiling_from_flags(
          context.runtime_context.device_group, iree_allocator_system(),
          &profiling);
    }

    GenerationExecutionRef execution;
    if (iree_status_is_ok(status)) {
      status = IssueGenerationBundle(
          context, bundle.get(), request, &diagnostics_sink,
          prepare_semaphore.get(), &prepare_value, completion_semaphore.get(),
          &completion_value, &iteration_timing, execution.out());
    }
    if (iree_status_is_ok(status)) {
      id4::test::SemaphoreListStorage completion_wait;
      completion_wait.semaphore = completion_semaphore.get();
      completion_wait.payload_value = completion_value;
      phase_start_time_ns = iree_time_now();
      status = iree_hal_semaphore_list_wait(completion_wait.list(),
                                            iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
      iteration_timing.final_wait_ns += iree_time_now() - phase_start_time_ns;
    }
    if (iree_status_is_ok(status)) {
      iteration_timing.total_ns += iree_time_now() - iteration_start_time_ns;
      AddTiming(&timing_total, iteration_timing);
      iree_optimization_barrier(execution.get());
      ++iteration_count;
    }
    if (profile_this_execution) {
      status = iree_status_join(status,
                                iree_hal_end_profiling_from_flags(profiling));
      profiling = nullptr;
      execution_profile_captured = true;
    }
  }
  if (!capture_execution_profile) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(SetGenerationBenchmarkLabel(
      benchmark_state, context, last_summary, last_statistics, timing_total,
      iteration_count));
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
