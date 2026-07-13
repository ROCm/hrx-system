// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstring>
#include <string>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/filesystem.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/testing/benchmark.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, id4_tokenizer, "",
          "Hugging Face tokenizer JSON used by live generation benchmarks.");
IREE_FLAG(string, id4_request_json, "",
          "Full JSON prompt/configuration payload for the custom generation "
          "benchmark.");
IREE_FLAG(string, id4_request_json_file, "",
          "File containing the JSON prompt/configuration payload for the "
          "custom generation benchmark.");
IREE_FLAG(string, id4_request_label, "",
          "Stable label for the custom generation benchmark row and plan "
          "JSON file stem. Labels may contain ASCII letters, digits, '_', "
          "'-', and '.'.");
IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving generation plan JSON files.");
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3. fp8_e4m3 uses the FP8 "
          "scopes below as the branch parameter providers.");
IREE_FLAG(string, qwen_parameter_format, "fp8_e4m3_block_scaled",
          "Qwen3-VL parameter format: bf16 or fp8_e4m3_block_scaled.");
IREE_FLAG(string, generation_parameter_source, "checkpoint",
          "Generation parameter source: checkpoint or execution_layout.");
IREE_FLAG(string, dit_activation_format, "bf16_linear_input",
          "DiT activation format: bf16_linear_input or f32_canonical.");
IREE_FLAG(string, dit_weight_execution_format, "fp8_compact_rhs",
          "DiT weight execution format: bf16_resident, fp8_compact_rhs, "
          "fp8_compact_rhs_feed_forward_bf16_resident, or "
          "streaming_compact_rhs.");
IREE_FLAG(string, qwen_weight_execution_strategy, "compact_rhs",
          "Qwen3-VL weight execution strategy: row_major, compact_rhs, "
          "hybrid_compact_rhs, or streaming_compact_rhs.");
IREE_FLAG(string, qwen_attention_implementation, "auto",
          "Qwen3-VL attention implementation: auto, materialized, or wmma.");
IREE_FLAG(string, dit_attention_implementation, "online_wmma",
          "DiT attention implementation: streaming, materialized_wmma, "
          "blocked_wmma, or online_wmma.");
IREE_FLAG(string, dit_feed_forward_implementation, "pytorch_parity",
          "DiT feed-forward implementation: fused_product or "
          "pytorch_parity.");
IREE_FLAG(string, generation_residency, "issue_phases",
          "Generation stage-bundle residency: issue_phases, "
          "phase_stage_bundles, selected_stage_bundles, all_stage_bundles, "
          "or memory_budgeted.");
IREE_FLAG(int64_t, generation_residency_budget, 0,
          "Logical live byte budget for "
          "--generation_residency=memory_budgeted.");
IREE_FLAG(string, generation_issue_mode, "phases",
          "Generation issue mode: full, phases, or stage_serial.");
IREE_FLAG(int64_t, parameter_load_prefetch_segment_distance, 0,
          "Number of future execution segments whose parameter windows may "
          "be loaded before the current segment.");
IREE_FLAG(int64_t, parameter_window_budget, INT64_C(2147483648),
          "Maximum compact parameter bytes retained while issuing a deferred "
          "stage.");
IREE_FLAG(string, generation_resident_stage_bundles, "",
          "Comma-separated generation stage bundles retained or considered by "
          "--generation_residency=selected_stage_bundles or memory_budgeted: "
          "qwen, "
          "sampler_noise, dit_conditioned, dit_unconditioned, "
          "sampler_denoise, decode, or all.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 branch parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 branch parameter scope.");

namespace {

static constexpr char kShortPrompt128[] =
    "{\"prompt\":\"A small red boat on a quiet lake at sunrise.\","
    "\"generation\":{\"latent_width\":8,\"latent_height\":8,"
    "\"sampler\":\"V4_TURBO_12\",\"seed\":20260626}}";

static constexpr char kMediumPrompt128[] =
    "{\"prompt\":\"Three friends walking through a bright city crosswalk with "
    "glass storefronts, natural clothing, normal hands, and soft afternoon "
    "light.\",\"generation\":{\"latent_width\":8,\"latent_height\":8,"
    "\"sampler\":\"V4_TURBO_12\",\"seed\":20260625}}";

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
    "\"sampler\":\"V4_TURBO_12\",\"seed\":20260625}}";

struct GenerationPrompt {
  // Stable benchmark label and plan dump stem for this prompt bucket.
  iree_string_view_t label;
  // Full generation request JSON passed through the production parser.
  iree_string_view_t json;
};

struct GenerationBenchmarkPlanStatistics {
  // Total parameter slab bytes across coarse stage plans.
  iree_device_size_t total_parameter_slab_byte_length;
  // Largest individual parameter slab bytes across coarse stage plans.
  iree_device_size_t largest_parameter_slab_byte_length;
  // Total provider source bytes consumed by parameter loading.
  iree_device_size_t parameter_source_byte_length;
  // Provider source bytes consumed by direct parameter gathers.
  iree_device_size_t parameter_direct_source_byte_length;
  // Provider source bytes consumed by encoded parameter load steps.
  iree_device_size_t parameter_encoded_source_byte_length;
  // Number of direct parameter gather load steps.
  iree_host_size_t parameter_gather_load_step_count;
  // Number of encoded parameter load steps.
  iree_host_size_t parameter_encode_load_step_count;
  // Number of independent parameter readiness groups.
  iree_host_size_t parameter_load_group_count;
  // Number of direct-gather parameter readiness groups.
  iree_host_size_t parameter_gather_load_group_count;
  // Number of encoded parameter readiness groups.
  iree_host_size_t parameter_encode_load_group_count;
  // Parameter loading statistics indexed by load-step kind enum value.
  id4_pipeline_parameter_load_kind_statistics_t parameter_load_kind_statistics
      [ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_CAPACITY];
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
  // Source-program streaming RHS materialization statistics.
  id4::test::ProgramStreamingRhsEncodeStatistics streaming_rhs_encode;
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
  kStageSerial,
};

enum class GenerationResidencyRequestMode {
  kIssuePhases,
  kPhaseStageBundles,
  kSelectedStageBundles,
  kAllStageBundles,
  kMemoryBudgeted,
};

struct GenerationResidencyResolution {
  // Concrete residency mode used by generation preparation and statistics.
  id4_ideogram4_generation_residency_mode_t residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  // Concrete resident stage mask used by generation preparation and statistics.
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  // Concrete phase-local stage masks used by generation preparation and
  // statistics.
  id4_ideogram4_generation_resident_stage_mask_t
      phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT] = {};
};

enum class CustomRequestSource {
  // No custom request source was provided.
  kNone,
  // The request JSON is stored directly in --id4_request_json.
  kInlineJson,
  // The request JSON is stored in the file named by --id4_request_json_file.
  kFile,
};

enum class PromptBenchmarkDisposition {
  // The benchmark row has enough configuration to execute.
  kRun,
  // The benchmark row is intentionally inactive for this invocation.
  kSkip,
};

static const GenerationPrompt kShortPrompt = {
    // Stable short prompt bucket label.
    IREE_SV("short128"),
    // Request JSON for the short prompt bucket.
    iree_make_cstring_view(kShortPrompt128),
};

static const GenerationPrompt kMediumPrompt = {
    // Stable medium prompt bucket label.
    IREE_SV("medium128"),
    // Request JSON for the medium prompt bucket.
    iree_make_cstring_view(kMediumPrompt128),
};

static const GenerationPrompt kStructuredPrompt = {
    // Stable structured prompt bucket label.
    IREE_SV("structured128"),
    // Request JSON for the structured prompt bucket.
    iree_make_cstring_view(kStructuredPrompt128),
};

static const GenerationPrompt kCustomPrompt = {
    // Stable custom prompt bucket label.
    IREE_SV("custom"),
    // Empty JSON selects --id4_request_json or --id4_request_json_file.
    iree_string_view_empty(),
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
using ParameterIndexRef = id4::test::OwningRef<iree_io_parameter_index_t,
                                               iree_io_parameter_index_release>;
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
    vae_parameter_index.reset();
    unconditioned_dit_parameter_index.reset();
    conditioned_dit_parameter_index.reset();
    qwen_parameter_index.reset();
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
  // Qwen3-VL parameter source policy selected by benchmark flags.
  id4_qwen3_vl_parameter_format_t qwen_parameter_format =
      ID4_QWEN3_VL_PARAMETER_FORMAT_INVALID;
  // Common parameter source representation selected by benchmark flags.
  id4_pipeline_parameter_source_kind_t parameter_source_kind =
      ID4_PIPELINE_PARAMETER_SOURCE_KIND_INVALID;
  // Generation stage-bundle residency request selected by benchmark flags.
  GenerationResidencyRequestMode generation_residency_request_mode =
      GenerationResidencyRequestMode::kIssuePhases;
  // Logical live byte budget for memory-budgeted residency selection.
  iree_device_size_t generation_residency_budget_byte_length = 0;
  // Generation issue mode selected by benchmark flags.
  GenerationIssueMode generation_issue_mode = GenerationIssueMode::kPhases;
  // Deferred parameter load lookahead selected by benchmark flags.
  iree_host_size_t parameter_load_prefetch_segment_distance = 0;
  // Maximum compact parameter bytes retained by one deferred stage.
  iree_device_size_t maximum_parameter_window_byte_length = 0;
  // Generation stage bundles selected or considered by benchmark flags.
  id4_ideogram4_generation_resident_stage_mask_t
      generation_resident_stage_mask =
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  // Provider containing Qwen3-VL text encoder weights.
  ParameterProviderRef qwen_parameter_provider;
  // Parsed Qwen3-VL execution-layout archive index when selected.
  ParameterIndexRef qwen_parameter_index;
  // Qwen3-VL provider scope selected by benchmark flags.
  iree_string_view_t qwen_parameter_scope = iree_string_view_empty();
  // Provider containing conditioned Ideogram 4 DiT weights.
  ParameterProviderRef conditioned_dit_parameter_provider;
  // Parsed conditioned DiT execution-layout archive index when selected.
  ParameterIndexRef conditioned_dit_parameter_index;
  // Conditioned DiT provider scope selected by benchmark flags.
  iree_string_view_t conditioned_dit_parameter_scope = iree_string_view_empty();
  // Provider containing unconditioned Ideogram 4 DiT weights.
  ParameterProviderRef unconditioned_dit_parameter_provider;
  // Parsed unconditioned DiT execution-layout archive index when selected.
  ParameterIndexRef unconditioned_dit_parameter_index;
  // Unconditioned DiT provider scope selected by benchmark flags.
  iree_string_view_t unconditioned_dit_parameter_scope =
      iree_string_view_empty();
  // Provider containing VAE decode weights.
  ParameterProviderRef vae_parameter_provider;
  // Parsed VAE execution-layout archive index when selected.
  ParameterIndexRef vae_parameter_index;
  // VAE provider scope selected by benchmark flags.
  iree_string_view_t vae_parameter_scope = iree_string_view_empty();
  // Loaded Ideogram 4 session under benchmark.
  SessionRef session;
  // Tokenizer used to lower prompt text into Qwen inputs.
  TokenizerRef tokenizer;
};

static id4_ideogram4_generation_plan_policy_t MakeGenerationPolicy() {
  id4_ideogram4_generation_plan_policy_t policy;
  std::memset(&policy, 0, sizeof(policy));
  policy.structure_size = sizeof(policy);
  policy.qwen_weight_execution_strategy =
      ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS;
  policy.qwen_attention_implementation =
      ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO;
  policy.dit_attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  policy.dit_feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT;
  policy.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  policy.vae_attention_implementation =
      ID4_VAE_ATTENTION_IMPLEMENTATION_MATERIALIZED;
  return policy;
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t ParseQwenParameterFormat(
    id4_qwen3_vl_parameter_format_t* out_format) {
  iree_status_t status = id4_qwen3_vl_parameter_format_parse(
      iree_make_cstring_view(FLAG_qwen_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--qwen_parameter_format"));
}

static iree_status_t ParseGenerationParameterSourceKind(
    id4_pipeline_parameter_source_kind_t* out_kind) {
  const iree_string_view_t value =
      iree_make_cstring_view(FLAG_generation_parameter_source);
  if (iree_string_view_equal(value, IREE_SV("checkpoint"))) {
    *out_kind = ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("execution_layout"))) {
    *out_kind = ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--generation_parameter_source must be checkpoint or execution_layout");
}

static iree_status_t ParseDitWeightExecutionFormat(
    id4_ideogram4_dit_weight_execution_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_weight_execution_format_parse(
      iree_make_cstring_view(FLAG_dit_weight_execution_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_weight_execution_format"));
}

static iree_status_t ParseQwenWeightExecutionStrategy(
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy) {
  iree_status_t status = id4_qwen3_vl_weight_execution_strategy_parse(
      iree_make_cstring_view(FLAG_qwen_weight_execution_strategy),
      out_strategy);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_weight_execution_strategy"));
}

static iree_status_t ParseQwenAttentionImplementation(
    id4_qwen3_vl_attention_implementation_t* out_implementation) {
  iree_status_t status = id4_qwen3_vl_attention_implementation_parse(
      iree_make_cstring_view(FLAG_qwen_attention_implementation),
      out_implementation);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_attention_implementation"));
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

static iree_status_t ParseGenerationResidencyRequestMode(
    GenerationResidencyRequestMode* out_mode) {
  IREE_ASSERT_ARGUMENT(out_mode);
  iree_string_view_t value = iree_make_cstring_view(FLAG_generation_residency);
  if (iree_string_view_equal(value, IREE_SV("issue_phases"))) {
    *out_mode = GenerationResidencyRequestMode::kIssuePhases;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("phase_stage_bundles"))) {
    *out_mode = GenerationResidencyRequestMode::kPhaseStageBundles;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("selected_stage_bundles"))) {
    *out_mode = GenerationResidencyRequestMode::kSelectedStageBundles;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("all_stage_bundles"))) {
    *out_mode = GenerationResidencyRequestMode::kAllStageBundles;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("memory_budgeted"))) {
    *out_mode = GenerationResidencyRequestMode::kMemoryBudgeted;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--generation_residency must be issue_phases, phase_stage_bundles, "
      "selected_stage_bundles, all_stage_bundles, or memory_budgeted");
}

static iree_string_view_t GenerationResidencyRequestModeName(
    GenerationResidencyRequestMode mode) {
  switch (mode) {
    case GenerationResidencyRequestMode::kIssuePhases:
      return IREE_SV("issue_phases");
    case GenerationResidencyRequestMode::kPhaseStageBundles:
      return IREE_SV("phase_stage_bundles");
    case GenerationResidencyRequestMode::kSelectedStageBundles:
      return IREE_SV("selected_stage_bundles");
    case GenerationResidencyRequestMode::kAllStageBundles:
      return IREE_SV("all_stage_bundles");
    case GenerationResidencyRequestMode::kMemoryBudgeted:
      return IREE_SV("memory_budgeted");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t GenerationResidencyModeName(
    id4_ideogram4_generation_residency_mode_t mode) {
  switch (mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
      return IREE_SV("issue_phases");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
      return IREE_SV("phase_stage_bundles");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      return IREE_SV("selected_stage_bundles");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      return IREE_SV("all_stage_bundles");
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES:
      return IREE_SV("phase_aware_stage_bundles");
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
  if (iree_string_view_equal(value, IREE_SV("stage_serial"))) {
    *out_issue_mode = GenerationIssueMode::kStageSerial;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "--generation_issue_mode must be full, phases, or "
                          "stage_serial");
}

static iree_string_view_t GenerationIssueModeName(GenerationIssueMode mode) {
  switch (mode) {
    case GenerationIssueMode::kFull:
      return IREE_SV("full");
    case GenerationIssueMode::kPhases:
      return IREE_SV("phases");
    case GenerationIssueMode::kStageSerial:
      return IREE_SV("stage_serial");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t ParameterSourceKindName(
    id4_pipeline_parameter_source_kind_t kind) {
  switch (kind) {
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT:
      return IREE_SV("checkpoint");
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT:
      return IREE_SV("execution_layout");
    default:
      return IREE_SV("invalid");
  }
}

static iree_status_t ParseParameterLoadPrefetchRegionDistance(
    iree_host_size_t* out_distance) {
  IREE_ASSERT_ARGUMENT(out_distance);
  if (FLAG_parameter_load_prefetch_segment_distance < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--parameter_load_prefetch_segment_distance must be non-negative");
  }
  if ((uint64_t)FLAG_parameter_load_prefetch_segment_distance >
      IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "--parameter_load_prefetch_segment_distance exceeds host size range");
  }
  *out_distance =
      (iree_host_size_t)FLAG_parameter_load_prefetch_segment_distance;
  return iree_ok_status();
}

static id4_ideogram4_generation_issue_policy_t GenerationIssuePolicy(
    GenerationIssueMode issue_mode) {
  return issue_mode == GenerationIssueMode::kStageSerial
             ? ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL
             : ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT;
}

static iree_status_t ParsePositiveByteBudgetFlag(
    int64_t value, iree_string_view_t flag_name,
    iree_device_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  if (value <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s must be positive",
                            static_cast<int>(flag_name.size), flag_name.data);
  }
  *out_byte_length = static_cast<iree_device_size_t>(value);
  if (static_cast<int64_t>(*out_byte_length) != value) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s value is out of range",
                            static_cast<int>(flag_name.size), flag_name.data);
  }
  return iree_ok_status();
}

static iree_status_t ParseGenerationResidentStageMask(
    id4_ideogram4_generation_resident_stage_mask_t* out_stage_mask) {
  IREE_ASSERT_ARGUMENT(out_stage_mask);
  *out_stage_mask = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  iree_string_view_t remaining = iree_string_view_trim(
      iree_make_cstring_view(FLAG_generation_resident_stage_bundles));
  if (iree_string_view_is_empty(remaining)) return iree_ok_status();

  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t stage_name = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &stage_name, &remaining);
    stage_name = iree_string_view_trim(stage_name);
    if (iree_string_view_is_empty(stage_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_resident_stage_bundles contains an empty stage name");
    }
    if (iree_string_view_equal(stage_name, IREE_SV("all"))) {
      *out_stage_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
    } else if (iree_string_view_equal(stage_name, IREE_SV("qwen"))) {
      *out_stage_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN;
    } else if (iree_string_view_equal(stage_name, IREE_SV("sampler_noise"))) {
      *out_stage_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_NOISE;
    } else if (iree_string_view_equal(stage_name, IREE_SV("dit_conditioned"))) {
      *out_stage_mask |=
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED;
    } else if (iree_string_view_equal(stage_name,
                                      IREE_SV("dit_unconditioned"))) {
      *out_stage_mask |=
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED;
    } else if (iree_string_view_equal(stage_name, IREE_SV("sampler_denoise"))) {
      *out_stage_mask |=
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_SAMPLER_DENOISE;
    } else if (iree_string_view_equal(stage_name, IREE_SV("decode"))) {
      *out_stage_mask |= ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown --generation_resident_stage_bundles value '%.*s'",
          (int)stage_name.size, stage_name.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t ResolveGenerationBenchmarkResidency(
    const LiveGenerationBenchmarkContext& context,
    const id4_ideogram4_generation_plan_t* plan,
    GenerationResidencyResolution* out_resolution) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_resolution);
  out_resolution->residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  out_resolution->resident_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;

  switch (context.generation_residency_request_mode) {
    case GenerationResidencyRequestMode::kIssuePhases:
      out_resolution->residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES;
      out_resolution->resident_stage_mask =
          context.generation_resident_stage_mask;
      return iree_ok_status();
    case GenerationResidencyRequestMode::kPhaseStageBundles:
      out_resolution->residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES;
      out_resolution->resident_stage_mask =
          context.generation_resident_stage_mask;
      return iree_ok_status();
    case GenerationResidencyRequestMode::kSelectedStageBundles:
      out_resolution->residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES;
      out_resolution->resident_stage_mask =
          context.generation_resident_stage_mask;
      return iree_ok_status();
    case GenerationResidencyRequestMode::kAllStageBundles:
      out_resolution->residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES;
      out_resolution->resident_stage_mask =
          context.generation_resident_stage_mask;
      return iree_ok_status();
    case GenerationResidencyRequestMode::kMemoryBudgeted: {
      if (context.generation_resident_stage_mask ==
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--generation_resident_stage_bundles must list candidate stages "
            "for --generation_residency=memory_budgeted");
      }
      id4_ideogram4_generation_residency_select_options_t select_options;
      std::memset(&select_options, 0, sizeof(select_options));
      select_options.structure_size = sizeof(select_options);
      select_options.issue_policy =
          GenerationIssuePolicy(context.generation_issue_mode);
      select_options.candidate_stage_mask =
          context.generation_resident_stage_mask;
      select_options.parameter_source_kind = context.parameter_source_kind;
      select_options.maximum_parameter_window_byte_length =
          context.maximum_parameter_window_byte_length;
      select_options.parameter_load_prefetch_segment_distance =
          context.parameter_load_prefetch_segment_distance;
      select_options.memory_budget_byte_length =
          context.generation_residency_budget_byte_length;
      id4_ideogram4_generation_residency_selection_t selection;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_select_residency(
          plan, &select_options, &selection));
      out_resolution->residency_mode = selection.residency_mode;
      out_resolution->resident_stage_mask = selection.resident_stage_mask;
      std::memcpy(out_resolution->phase_stage_masks,
                  selection.phase_stage_masks,
                  sizeof(out_resolution->phase_stage_masks));
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "generation residency request mode is invalid");
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

static double AverageRegionDistance(iree_host_size_t distance_sum,
                                    iree_host_size_t count) {
  if (count == 0) return 0.0;
  return static_cast<double>(distance_sum) / static_cast<double>(count);
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
    IREE_RETURN_IF_ERROR(
        id4::test::AccumulateProgramStreamingRhsEncodeStatistics(
            stage_plan, &out_statistics->streaming_rhs_encode));
    out_statistics->stages[i].key = stage_key;
    out_statistics->stages[i].statistics = stage_statistics;
    out_statistics->total_parameter_slab_byte_length +=
        stage_statistics.parameter_slab_byte_length;
    out_statistics->parameter_source_byte_length +=
        stage_statistics.parameter_source_byte_length;
    out_statistics->parameter_direct_source_byte_length +=
        stage_statistics.parameter_direct_source_byte_length;
    out_statistics->parameter_encoded_source_byte_length +=
        stage_statistics.parameter_encoded_source_byte_length;
    out_statistics->parameter_gather_load_step_count +=
        stage_statistics.parameter_gather_load_step_count;
    out_statistics->parameter_encode_load_step_count +=
        stage_statistics.parameter_encode_load_step_count;
    out_statistics->parameter_load_group_count +=
        stage_statistics.parameter_load_group_count;
    out_statistics->parameter_gather_load_group_count +=
        stage_statistics.parameter_gather_load_group_count;
    out_statistics->parameter_encode_load_group_count +=
        stage_statistics.parameter_encode_load_group_count;
    for (iree_host_size_t kind = 0;
         kind < ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_CAPACITY; ++kind) {
      out_statistics->parameter_load_kind_statistics[kind].step_count +=
          stage_statistics.parameter_load_kind_statistics[kind].step_count;
      out_statistics->parameter_load_kind_statistics[kind].source_byte_length +=
          stage_statistics.parameter_load_kind_statistics[kind]
              .source_byte_length;
      out_statistics->parameter_load_kind_statistics[kind].target_byte_length +=
          stage_statistics.parameter_load_kind_statistics[kind]
              .target_byte_length;
    }
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

static id4_ideogram4_generation_resource_statistics_options_t
GenerationBenchmarkResourceOptions(
    const LiveGenerationBenchmarkContext& context,
    const GenerationResidencyResolution& residency) {
  id4_ideogram4_generation_resource_statistics_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.residency_mode = residency.residency_mode;
  options.resident_stage_mask = residency.resident_stage_mask;
  std::memcpy(options.phase_stage_masks, residency.phase_stage_masks,
              sizeof(options.phase_stage_masks));
  options.parameter_source_kind = context.parameter_source_kind;
  options.maximum_parameter_window_byte_length =
      context.maximum_parameter_window_byte_length;
  options.parameter_load_prefetch_segment_distance =
      context.parameter_load_prefetch_segment_distance;
  return options;
}

static iree_status_t QueryGenerationBenchmarkResourceStatistics(
    const LiveGenerationBenchmarkContext& context,
    const id4_ideogram4_generation_plan_t* plan,
    const GenerationResidencyResolution& residency,
    id4_ideogram4_generation_resource_statistics_t* out_statistics) {
  const id4_ideogram4_generation_resource_statistics_options_t options =
      GenerationBenchmarkResourceOptions(context, residency);
  return id4_ideogram4_generation_plan_resource_statistics(plan, &options,
                                                           out_statistics);
}

static iree_status_t ParseRequest(iree_string_view_t json,
                                  ParsedRequest* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_json(
      json, iree_allocator_system(), &out_request->request));
  out_request->initialized = true;
  return iree_ok_status();
}

static iree_status_t SelectCustomRequestSource(
    CustomRequestSource* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  *out_source = CustomRequestSource::kNone;
  const iree_string_view_t inline_json =
      iree_make_cstring_view(FLAG_id4_request_json);
  const iree_string_view_t file_path =
      iree_make_cstring_view(FLAG_id4_request_json_file);
  const bool has_inline_json = !iree_string_view_is_empty(inline_json);
  const bool has_file_path = !iree_string_view_is_empty(file_path);
  if (has_inline_json && has_file_path) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "custom generation benchmark requires exactly one of "
        "--id4_request_json or --id4_request_json_file");
  }
  if (has_inline_json) {
    *out_source = CustomRequestSource::kInlineJson;
  } else if (has_file_path) {
    *out_source = CustomRequestSource::kFile;
  }
  return iree_ok_status();
}

static iree_status_t LoadCustomRequestJson(std::string* out_json) {
  IREE_ASSERT_ARGUMENT(out_json);
  out_json->clear();

  CustomRequestSource source = CustomRequestSource::kNone;
  IREE_RETURN_IF_ERROR(SelectCustomRequestSource(&source));
  if (source == CustomRequestSource::kNone) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "custom generation benchmark requires exactly one of "
        "--id4_request_json or --id4_request_json_file");
  }
  if (source == CustomRequestSource::kInlineJson) {
    const iree_string_view_t inline_json =
        iree_make_cstring_view(FLAG_id4_request_json);
    out_json->assign(inline_json.data, inline_json.size);
    return iree_ok_status();
  }

  const iree_string_view_t file_path =
      iree_make_cstring_view(FLAG_id4_request_json_file);
  iree_io_file_contents_t* file_contents = nullptr;
  iree_status_t status =
      iree_io_file_contents_map(file_path, IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &file_contents);
  if (iree_status_is_ok(status)) {
    out_json->assign(
        reinterpret_cast<const char*>(file_contents->const_buffer.data),
        file_contents->const_buffer.data_length);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t ValidatePromptBenchmarkConfiguration(
    iree_benchmark_state_t* benchmark_state, const GenerationPrompt& prompt,
    PromptBenchmarkDisposition* out_disposition) {
  IREE_ASSERT_ARGUMENT(out_disposition);
  *out_disposition = PromptBenchmarkDisposition::kRun;
  if (!iree_string_view_is_empty(prompt.json)) return iree_ok_status();

  CustomRequestSource source = CustomRequestSource::kNone;
  IREE_RETURN_IF_ERROR(SelectCustomRequestSource(&source));
  if (source == CustomRequestSource::kNone) {
    iree_benchmark_skip(benchmark_state,
                        "custom benchmark requires --id4_request_json or "
                        "--id4_request_json_file");
    *out_disposition = PromptBenchmarkDisposition::kSkip;
  }
  return iree_ok_status();
}

static bool IsValidPromptLabelCharacter(char value) {
  return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
         (value >= '0' && value <= '9') || value == '_' || value == '-' ||
         value == '.';
}

static iree_status_t ResolvePromptLabel(const GenerationPrompt& prompt,
                                        iree_string_view_t* out_label) {
  IREE_ASSERT_ARGUMENT(out_label);
  if (!iree_string_view_is_empty(prompt.json)) {
    *out_label = prompt.label;
    return iree_ok_status();
  }

  iree_string_view_t label =
      iree_string_view_trim(iree_make_cstring_view(FLAG_id4_request_label));
  if (iree_string_view_is_empty(label)) {
    *out_label = prompt.label;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < label.size; ++i) {
    if (!IsValidPromptLabelCharacter(label.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--id4_request_label contains invalid character 0x%02x at byte "
          "offset %" PRIhsz,
          (unsigned char)label.data[i], i);
    }
  }
  *out_label = label;
  return iree_ok_status();
}

static iree_status_t ParsePromptRequest(const GenerationPrompt& prompt,
                                        ParsedRequest* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  if (!iree_string_view_is_empty(prompt.json)) {
    return ParseRequest(prompt.json, out_request);
  }

  std::string json;
  IREE_RETURN_IF_ERROR(LoadCustomRequestJson(&json));
  return ParseRequest(iree_make_string_view(json.data(), json.size()),
                      out_request);
}

static iree_status_t WriteGenerationPlanJsonIfRequested(
    iree_string_view_t prompt_label,
    const LiveGenerationBenchmarkContext& context,
    const GenerationResidencyResolution& residency,
    const id4_ideogram4_generation_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  const iree_string_view_t output_directory =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_directory)) return iree_ok_status();

  iree_string_builder_t file_name_builder;
  iree_string_builder_initialize(iree_allocator_system(), &file_name_builder);
  iree_string_builder_t json_builder;
  iree_string_builder_initialize(iree_allocator_system(), &json_builder);
  iree_string_view_t output_path = iree_string_view_empty();

  iree_status_t status =
      id4_tooling_ensure_directory(output_directory, iree_allocator_system());
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &file_name_builder, "%.*s.json", static_cast<int>(prompt_label.size),
        prompt_label.data);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(
        output_directory, iree_string_builder_view(&file_name_builder),
        iree_allocator_system(), &output_path);
  }
  if (iree_status_is_ok(status)) {
    const id4_ideogram4_generation_resource_statistics_options_t options =
        GenerationBenchmarkResourceOptions(context, residency);
    status = id4_ideogram4_generation_plan_format_json(plan, &options,
                                                       &json_builder);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&json_builder);
    status = iree_io_file_contents_write(
        output_path, iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }

  id4_tooling_free_path(&output_path, iree_allocator_system());
  iree_string_builder_deinitialize(&json_builder);
  iree_string_builder_deinitialize(&file_name_builder);
  return status;
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

static id4_ideogram4_generation_parameter_sources_t
LiveGenerationParameterSources(const LiveGenerationBenchmarkContext& context) {
  id4_ideogram4_generation_parameter_sources_t sources = {};
  if (context.parameter_source_kind ==
      ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT) {
    sources.qwen = id4_pipeline_execution_layout_parameter_source(
        context.qwen_parameter_index.get(),
        context.qwen_parameter_provider.get(), context.qwen_parameter_scope);
    sources.dit_conditioned = id4_pipeline_execution_layout_parameter_source(
        context.conditioned_dit_parameter_index.get(),
        context.conditioned_dit_parameter_provider.get(),
        context.conditioned_dit_parameter_scope);
    sources.dit_unconditioned = id4_pipeline_execution_layout_parameter_source(
        context.unconditioned_dit_parameter_index.get(),
        context.unconditioned_dit_parameter_provider.get(),
        context.unconditioned_dit_parameter_scope);
    sources.vae = id4_pipeline_execution_layout_parameter_source(
        context.vae_parameter_index.get(), context.vae_parameter_provider.get(),
        context.vae_parameter_scope);
  } else {
    sources.qwen = id4_pipeline_checkpoint_parameter_source(
        context.qwen_parameter_provider.get());
    sources.dit_conditioned = id4_pipeline_checkpoint_parameter_source(
        context.conditioned_dit_parameter_provider.get());
    sources.dit_unconditioned = id4_pipeline_checkpoint_parameter_source(
        context.unconditioned_dit_parameter_provider.get());
    sources.vae = id4_pipeline_checkpoint_parameter_source(
        context.vae_parameter_provider.get());
  }
  return sources;
}

static iree_status_t CreateLoadedLiveSession(
    const id4_tooling_runtime_context_t* runtime_context,
    id4_ideogram4_dit_parameter_format_t dit_parameter_format,
    id4_qwen3_vl_parameter_format_t qwen_parameter_format,
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
  create_options.parameter_scopes.vae = IREE_SV("vae");
  create_options.qwen_parameter_format = qwen_parameter_format;
  create_options.dit_parameter_format = dit_parameter_format;
  switch (create_options.dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      create_options.parameter_scopes.dit_conditioned = IREE_SV("dit_cond");
      create_options.parameter_scopes.dit_unconditioned = IREE_SV("dit_uncond");
      break;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3: {
      const iree_string_view_t conditioned_fp8_scope =
          iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
      const iree_string_view_t unconditioned_fp8_scope =
          iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
      create_options.parameter_scopes.dit_conditioned = conditioned_fp8_scope;
      create_options.parameter_scopes.dit_conditioned_fp8 =
          conditioned_fp8_scope;
      create_options.parameter_scopes.dit_unconditioned =
          unconditioned_fp8_scope;
      create_options.parameter_scopes.dit_unconditioned_fp8 =
          unconditioned_fp8_scope;
      break;
    }
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

static iree_status_t CreateParameterSources(
    LiveGenerationBenchmarkContext* context) {
  IREE_ASSERT_ARGUMENT(context);
  const bool uses_execution_layout =
      context->parameter_source_kind ==
      ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT;
  context->qwen_parameter_scope = IREE_SV("qwen");
  context->conditioned_dit_parameter_scope =
      uses_execution_layout ? IREE_SV("dit_conditioned")
      : context->dit_parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16
          ? IREE_SV("dit_cond")
          : iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
  context->unconditioned_dit_parameter_scope =
      uses_execution_layout ? IREE_SV("dit_unconditioned")
      : context->dit_parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16
          ? IREE_SV("dit_uncond")
          : iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
  context->vae_parameter_scope = IREE_SV("vae");
  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // Qwen3-VL text encoder parameter scope.
          /*.scope=*/context->qwen_parameter_scope,
          // Qwen provider output.
          /*.out_provider=*/context->qwen_parameter_provider.out(),
          // Qwen archive index output when requested.
          /*.out_index=*/
          uses_execution_layout ? context->qwen_parameter_index.out() : nullptr,
      },
      {
          // Conditioned DiT parameter scope.
          /*.scope=*/context->conditioned_dit_parameter_scope,
          // Conditioned DiT provider output.
          /*.out_provider=*/
          context->conditioned_dit_parameter_provider.out(),
          // Conditioned DiT archive index output when requested.
          /*.out_index=*/
          uses_execution_layout ? context->conditioned_dit_parameter_index.out()
                                : nullptr,
      },
      {
          // Unconditioned DiT parameter scope.
          /*.scope=*/context->unconditioned_dit_parameter_scope,
          // Unconditioned DiT provider output.
          /*.out_provider=*/
          context->unconditioned_dit_parameter_provider.out(),
          // Unconditioned DiT archive index output when requested.
          /*.out_index=*/
          uses_execution_layout
              ? context->unconditioned_dit_parameter_index.out()
              : nullptr,
      },
      {
          // VAE parameter scope.
          /*.scope=*/context->vae_parameter_scope,
          // VAE provider output.
          /*.out_provider=*/context->vae_parameter_provider.out(),
          // VAE archive index output when requested.
          /*.out_index=*/
          uses_execution_layout ? context->vae_parameter_index.out() : nullptr,
      },
  };
  return id4_tooling_create_parameter_providers_from_flags(
      IREE_ARRAYSIZE(requests), requests, iree_allocator_system());
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
      ParseQwenParameterFormat(&out_context->qwen_parameter_format));
  IREE_RETURN_IF_ERROR(
      ParseGenerationParameterSourceKind(&out_context->parameter_source_kind));
  IREE_RETURN_IF_ERROR(ParseGenerationResidencyRequestMode(
      &out_context->generation_residency_request_mode));
  if (out_context->generation_residency_request_mode !=
          GenerationResidencyRequestMode::kMemoryBudgeted &&
      FLAG_generation_residency_budget != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--generation_residency_budget requires "
                            "--generation_residency=memory_budgeted");
  }
  if (out_context->generation_residency_request_mode ==
      GenerationResidencyRequestMode::kMemoryBudgeted) {
    IREE_RETURN_IF_ERROR(ParsePositiveByteBudgetFlag(
        FLAG_generation_residency_budget,
        IREE_SV("--generation_residency_budget"),
        &out_context->generation_residency_budget_byte_length));
  }
  IREE_RETURN_IF_ERROR(
      ParseGenerationIssueMode(&out_context->generation_issue_mode));
  IREE_RETURN_IF_ERROR(ParseParameterLoadPrefetchRegionDistance(
      &out_context->parameter_load_prefetch_segment_distance));
  IREE_RETURN_IF_ERROR(ParsePositiveByteBudgetFlag(
      FLAG_parameter_window_budget, IREE_SV("--parameter_window_budget"),
      &out_context->maximum_parameter_window_byte_length));
  IREE_RETURN_IF_ERROR(ParseGenerationResidentStageMask(
      &out_context->generation_resident_stage_mask));
  IREE_RETURN_IF_ERROR(CreateLoadedLiveSession(
      &out_context->runtime_context, out_context->dit_parameter_format,
      out_context->qwen_parameter_format, out_context->session.out()));
  return CreateParameterSources(out_context);
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
  IREE_RETURN_IF_ERROR(ParseDitWeightExecutionFormat(
      &plan_options.policy.dit_weight_execution_format));
  IREE_RETURN_IF_ERROR(ParseQwenWeightExecutionStrategy(
      &plan_options.policy.qwen_weight_execution_strategy));
  IREE_RETURN_IF_ERROR(ParseQwenAttentionImplementation(
      &plan_options.policy.qwen_attention_implementation));
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
    const GenerationResidencyResolution& residency,
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
  prepare_options.parameter_sources = LiveGenerationParameterSources(context);
  prepare_options.kernel_library = context.kernel_library.get();
  prepare_options.maximum_parameter_window_byte_length =
      context.maximum_parameter_window_byte_length;
  prepare_options.residency_mode = residency.residency_mode;
  prepare_options.resident_stage_mask = residency.resident_stage_mask;
  std::memcpy(prepare_options.phase_stage_masks, residency.phase_stage_masks,
              sizeof(prepare_options.phase_stage_masks));
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
    id4_ideogram4_generation_phase_mask_t phase_mask,
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
    iree_host_size_t parameter_load_prefetch_segment_distance,
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
  issue_options.parameter_load_prefetch_segment_distance =
      parameter_load_prefetch_segment_distance;
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

  if (context.generation_issue_mode == GenerationIssueMode::kFull ||
      context.generation_issue_mode == GenerationIssueMode::kStageSerial) {
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
    issue_options.issue_policy =
        context.generation_issue_mode == GenerationIssueMode::kStageSerial
            ? ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL
            : ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT;
    issue_options.parameter_load_prefetch_segment_distance =
        context.parameter_load_prefetch_segment_distance;
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
      bundle, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING, diagnostics_sink,
      prepare_semaphore, *prepare_value, prepare_semaphore, ++*prepare_value,
      &conditioning_phase);
  timing->conditioning.prepare_ns += iree_time_now() - phase_start_time_ns;
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, conditioning_phase, diagnostics_sink,
        context.parameter_load_prefetch_segment_distance, prepare_semaphore,
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
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE, diagnostics_sink,
        completion_semaphore, *completion_value, prepare_semaphore,
        ++*prepare_value, &denoise_phase);
    timing->denoise.prepare_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, denoise_phase, diagnostics_sink,
        context.parameter_load_prefetch_segment_distance, prepare_semaphore,
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
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE, diagnostics_sink,
        completion_semaphore, *completion_value, prepare_semaphore,
        ++*prepare_value, &decode_phase);
    timing->decode.prepare_ns += iree_time_now() - phase_start_time_ns;
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = IssueGenerationPhase(
        *out_execution, decode_phase, diagnostics_sink,
        context.parameter_load_prefetch_segment_distance, prepare_semaphore,
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

static bool RequiresPreparedGenerationWarmup(
    const GenerationResidencyResolution& residency) {
  return residency.residency_mode ==
             ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES ||
         residency.resident_stage_mask !=
             ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
}

static iree_status_t WarmPreparedGenerationState(
    const LiveGenerationBenchmarkContext& context,
    const GenerationResidencyResolution& residency,
    id4_ideogram4_generation_bundle_t* bundle, const ParsedRequest& request,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_hal_semaphore_t* prepare_semaphore, uint64_t* prepare_value,
    iree_hal_semaphore_t* completion_semaphore, uint64_t* completion_value) {
  IREE_ASSERT_ARGUMENT(bundle);
  IREE_ASSERT_ARGUMENT(diagnostics_sink);
  IREE_ASSERT_ARGUMENT(prepare_value);
  IREE_ASSERT_ARGUMENT(completion_value);
  if (!RequiresPreparedGenerationWarmup(residency)) {
    return iree_ok_status();
  }

  GenerationBenchmarkTimingStatistics warmup_timing;
  std::memset(&warmup_timing, 0, sizeof(warmup_timing));
  GenerationExecutionRef execution;
  const uint64_t initial_completion_value = *completion_value;
  IREE_RETURN_IF_ERROR(IssueGenerationBundle(
      context, bundle, request, diagnostics_sink, prepare_semaphore,
      prepare_value, completion_semaphore, completion_value, &warmup_timing,
      execution.out()));
  if (*completion_value == initial_completion_value) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "prepared-generation warmup did not queue "
                            "generation completion");
  }
  return WaitForSemaphore(completion_semaphore, *completion_value);
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
        " stage.%.*s[param=%" PRIu64 "MiB,src=%" PRIu64
        "MiB,src_direct=%" PRIu64 "MiB,src_encoded=%" PRIu64
        "MiB,load_steps=%" PRIhsz "/%" PRIhsz ",load_groups=%" PRIhsz
        "/%" PRIhsz ",local_hw=%" PRIu64 "MiB,boundary=%" PRIu64
        "MiB,kernels=%" PRIhsz ",dispatches=%" PRIhsz "]",
        static_cast<int>(key.size), key.data,
        CeilMiB(stage.parameter_slab_byte_length),
        CeilMiB(stage.parameter_source_byte_length),
        CeilMiB(stage.parameter_direct_source_byte_length),
        CeilMiB(stage.parameter_encoded_source_byte_length),
        stage.parameter_gather_load_step_count,
        stage.parameter_encode_load_step_count,
        stage.parameter_gather_load_group_count,
        stage.parameter_encode_load_group_count,
        CeilMiB(stage.memory_slab_high_water_mark),
        CeilMiB(stage.boundary_tensor_byte_length), stage.kernel_count,
        stage.dispatch_count));
  }
  return iree_ok_status();
}

static iree_status_t AppendGenerationBenchmarkPhaseStageMasksLabel(
    iree_string_builder_t* builder,
    const GenerationResidencyResolution& residency) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV(" phase_stage_masks=[")));
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_PHASE_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%s0x%08x", i == 0 ? "" : ",",
        residency.phase_stage_masks[i]));
  }
  return iree_string_builder_append_string(builder, IREE_SV("]"));
}

static iree_status_t SetGenerationBenchmarkLabel(
    iree_benchmark_state_t* benchmark_state,
    const LiveGenerationBenchmarkContext& context,
    const GenerationResidencyResolution& residency,
    iree_string_view_t benchmark_scope, iree_string_view_t prompt_label,
    const id4_ideogram4_generation_plan_summary_t& summary,
    const GenerationBenchmarkPlanStatistics& statistics,
    const id4_ideogram4_generation_resource_statistics_t& resource_statistics,
    const id4::test::StageDiagnostics& diagnostics,
    const GenerationBenchmarkTimingStatistics& timing,
    uint64_t iteration_count) {
  const iree_string_view_t parameter_format =
      id4_ideogram4_dit_parameter_format_name(context.dit_parameter_format);
  const iree_string_view_t activation_format =
      DitActivationFormatName(summary.dit_activation_format);
  const iree_string_view_t weight_execution_format =
      id4_ideogram4_dit_weight_execution_format_name(
          summary.dit_weight_execution_format);
  const iree_string_view_t qwen_weight_execution_strategy =
      id4_qwen3_vl_weight_execution_strategy_name(
          summary.qwen_weight_execution_strategy);
  const iree_string_view_t qwen_parameter_format =
      id4_qwen3_vl_parameter_format_name(context.qwen_parameter_format);
  const iree_string_view_t qwen_attention_implementation =
      id4_qwen3_vl_attention_implementation_name(
          summary.qwen_attention_implementation);
  const iree_string_view_t attention_implementation =
      DitAttentionImplementationName(summary.dit_attention_implementation);
  const iree_string_view_t feed_forward_implementation =
      DitFeedForwardImplementationName(summary.dit_feed_forward_implementation);
  const iree_string_view_t residency_request =
      GenerationResidencyRequestModeName(
          context.generation_residency_request_mode);
  const iree_string_view_t residency_mode =
      GenerationResidencyModeName(residency.residency_mode);
  const iree_string_view_t issue_mode =
      GenerationIssueModeName(context.generation_issue_mode);
  const iree_string_view_t parameter_source_kind =
      ParameterSourceKindName(context.parameter_source_kind);
  const bool stage_serial_issue =
      context.generation_issue_mode == GenerationIssueMode::kStageSerial;
  const iree_device_size_t selected_logical_peak_byte_length =
      stage_serial_issue
          ? resource_statistics.stage_serial_total_peak_byte_length
          : resource_statistics.phase_concurrent_total_peak_byte_length;
  iree_string_builder_t label_builder;
  iree_string_builder_initialize(iree_allocator_system(), &label_builder);
  iree_status_t status = iree_string_builder_append_format(
      &label_builder,
      "scope=%.*s prompt=%.*s qwen_tokens=%" PRIu32 " qwen_capacity=%" PRIu32
      " image_tokens=%" PRIu32 " dit_cond_tokens=%" PRIu32
      " dit_cond_capacity=%" PRIu32 " dit_uncond_tokens=%" PRIu32
      " dit_uncond_capacity=%" PRIu32 " latent=%" PRIu64 "x%" PRIu64
      " steps=%" PRIu32 " image=%" PRIu64 "x%" PRIu64
      " residency_request=%.*s residency=%.*s issue=%.*s"
      " prefetch_segments=%" PRIhsz
      " resident_stage_mask=0x%08x residency_budget=%" PRIu64
      "MiB"
      " parameter_source=%.*s params=%.*s activation=%.*s weights=%.*s"
      " qwen_params=%.*s"
      " qwen_weights=%.*s qwen_attention=%.*s attention=%.*s ff=%.*s"
      " param_total=%" PRIu64 "MiB param_largest=%" PRIu64
      "MiB param_source=%" PRIu64 "MiB param_source_direct=%" PRIu64
      "MiB param_source_encoded=%" PRIu64 "MiB param_load_steps[gather=%" PRIhsz
      ",encode=%" PRIhsz
      "]"
      " param_load_groups[total=%" PRIhsz ",gather=%" PRIhsz ",encode=%" PRIhsz
      "]"
      " param_load_submit[count=%" PRIhsz ",gather=%" PRIhsz ",encode=%" PRIhsz
      ",total_ms=%.3f,gather_ms=%.3f,encode_ms=%.3f"
      ",max_ms=%.3f]"
      " local_hw_total=%" PRIu64 "MiB local_hw_largest=%" PRIu64
      "MiB"
      " boundary=%" PRIu64 "MiB kernels=%" PRIhsz " dispatches=%" PRIhsz
      " logical_live[boundary=%" PRIu64 "MiB,taps=%" PRIu64
      "MiB,resident=%" PRIu64 "MiB,phase_peak=%" PRIu64
      "MiB,stage_serial_peak=%" PRIu64 "MiB,selected_peak=%" PRIu64
      "MiB]"
      " prefetch_groups[count=%" PRIhsz
      ",avg_segments=%.2f,max_segments=%" PRIhsz
      "]"
      " direct_gather_groups[count=%" PRIhsz ",requests=%" PRIhsz
      ",source=%" PRIu64 "MiB,target=%" PRIu64 "MiB,max=%" PRIu64
      "MiB]"
      " encode_windows[count=%" PRIhsz ",staging=%" PRIu64 "MiB,max=%" PRIu64
      "MiB,source=%" PRIu64 "MiB,target=%" PRIu64 "MiB,chunks=%" PRIhsz
      ",sources=%" PRIhsz ",batches=%" PRIhsz ",dispatches=%" PRIhsz
      "]"
      " prepare_encode_window[count=%" PRIhsz ",staging=%" PRIu64
      "MiB,max=%" PRIu64 "MiB,source=%" PRIu64 "MiB,target=%" PRIu64
      "MiB,chunks=%" PRIhsz ",sources=%" PRIhsz ",batches=%" PRIhsz
      ",dispatches=%" PRIhsz
      "]"
      " issue_encode_window[count=%" PRIhsz ",staging=%" PRIu64
      "MiB,max=%" PRIu64 "MiB,source=%" PRIu64 "MiB,target=%" PRIu64
      "MiB,chunks=%" PRIhsz ",sources=%" PRIhsz ",batches=%" PRIhsz
      ",dispatches=%" PRIhsz
      "]"
      " program_streaming_rhs_encode[dispatches=%" PRIhsz ",read=%" PRIu64
      "MiB,write=%" PRIu64 "MiB,max_write=%" PRIu64 "MiB]",
      static_cast<int>(benchmark_scope.size), benchmark_scope.data,
      static_cast<int>(prompt_label.size), prompt_label.data,
      summary.qwen_token_count, summary.qwen_token_capacity,
      summary.image_token_count, summary.conditioned_dit_token_count,
      summary.conditioned_dit_token_capacity,
      summary.unconditioned_dit_token_count,
      summary.unconditioned_dit_token_capacity,
      summary.diffusion_latent_shape.dims[0],
      summary.diffusion_latent_shape.dims[1], summary.denoise_step_count,
      summary.decoded_image_shape.dims[0], summary.decoded_image_shape.dims[1],
      static_cast<int>(residency_request.size), residency_request.data,
      static_cast<int>(residency_mode.size), residency_mode.data,
      static_cast<int>(issue_mode.size), issue_mode.data,
      context.parameter_load_prefetch_segment_distance,
      residency.resident_stage_mask,
      CeilMiB(context.generation_residency_budget_byte_length),
      static_cast<int>(parameter_source_kind.size), parameter_source_kind.data,
      static_cast<int>(parameter_format.size), parameter_format.data,
      static_cast<int>(activation_format.size), activation_format.data,
      static_cast<int>(weight_execution_format.size),
      weight_execution_format.data,
      static_cast<int>(qwen_parameter_format.size), qwen_parameter_format.data,
      static_cast<int>(qwen_weight_execution_strategy.size),
      qwen_weight_execution_strategy.data,
      static_cast<int>(qwen_attention_implementation.size),
      qwen_attention_implementation.data,
      static_cast<int>(attention_implementation.size),
      attention_implementation.data,
      static_cast<int>(feed_forward_implementation.size),
      feed_forward_implementation.data,
      CeilMiB(statistics.total_parameter_slab_byte_length),
      CeilMiB(statistics.largest_parameter_slab_byte_length),
      CeilMiB(statistics.parameter_source_byte_length),
      CeilMiB(statistics.parameter_direct_source_byte_length),
      CeilMiB(statistics.parameter_encoded_source_byte_length),
      statistics.parameter_gather_load_step_count,
      statistics.parameter_encode_load_step_count,
      statistics.parameter_load_group_count,
      statistics.parameter_gather_load_group_count,
      statistics.parameter_encode_load_group_count,
      diagnostics.parameter_load_group_submit_count,
      diagnostics.parameter_load_group_submit_gather_count,
      diagnostics.parameter_load_group_submit_encode_count,
      AverageMilliseconds(diagnostics.parameter_load_group_submit_duration_ns,
                          iteration_count),
      AverageMilliseconds(
          diagnostics.parameter_load_group_submit_gather_duration_ns,
          iteration_count),
      AverageMilliseconds(
          diagnostics.parameter_load_group_submit_encode_duration_ns,
          iteration_count),
      AverageMilliseconds(
          diagnostics.parameter_load_group_submit_max_duration_ns, 1),
      CeilMiB(statistics.total_local_slab_high_water_mark),
      CeilMiB(statistics.largest_local_slab_high_water_mark),
      CeilMiB(statistics.boundary_tensor_byte_length), statistics.kernel_count,
      statistics.dispatch_count,
      CeilMiB(resource_statistics.boundary_buffer_byte_length),
      CeilMiB(resource_statistics.diagnostic_tap_buffer_byte_length),
      CeilMiB(resource_statistics.resident_stage_bundle_byte_length),
      CeilMiB(resource_statistics.phase_concurrent_total_peak_byte_length),
      CeilMiB(resource_statistics.stage_serial_total_peak_byte_length),
      CeilMiB(selected_logical_peak_byte_length),
      diagnostics.parameter_load_group_prefetch_submit_count,
      AverageRegionDistance(
          diagnostics.parameter_load_group_prefetch_segment_distance_sum,
          diagnostics.parameter_load_group_prefetch_submit_count),
      diagnostics.parameter_load_group_prefetch_segment_distance_max,
      diagnostics.parameter_direct_gather_group_count,
      diagnostics.parameter_direct_gather_request_count,
      CeilMiB(diagnostics.parameter_direct_gather_source_byte_length),
      CeilMiB(diagnostics.parameter_direct_gather_target_byte_length),
      CeilMiB(diagnostics.parameter_direct_gather_max_source_byte_length),
      diagnostics.parameter_encode_window_count,
      CeilMiB(diagnostics.parameter_encode_window_staging_total_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_encode_window_target_byte_length),
      diagnostics.parameter_encode_window_staging_chunk_count,
      diagnostics.parameter_encode_window_logical_source_count,
      diagnostics.parameter_encode_window_source_gather_batch_count,
      diagnostics.parameter_encode_window_encoder_dispatch_count,
      diagnostics.parameter_prepare_encode_window_count,
      CeilMiB(diagnostics
                  .parameter_prepare_encode_window_staging_total_byte_length),
      CeilMiB(
          diagnostics.parameter_prepare_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_prepare_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_prepare_encode_window_target_byte_length),
      diagnostics.parameter_prepare_encode_window_staging_chunk_count,
      diagnostics.parameter_prepare_encode_window_logical_source_count,
      diagnostics.parameter_prepare_encode_window_source_gather_batch_count,
      diagnostics.parameter_prepare_encode_window_encoder_dispatch_count,
      diagnostics.parameter_issue_encode_window_count,
      CeilMiB(
          diagnostics.parameter_issue_encode_window_staging_total_byte_length),
      CeilMiB(
          diagnostics.parameter_issue_encode_window_staging_max_byte_length),
      CeilMiB(diagnostics.parameter_issue_encode_window_source_byte_length),
      CeilMiB(diagnostics.parameter_issue_encode_window_target_byte_length),
      diagnostics.parameter_issue_encode_window_staging_chunk_count,
      diagnostics.parameter_issue_encode_window_logical_source_count,
      diagnostics.parameter_issue_encode_window_source_gather_batch_count,
      diagnostics.parameter_issue_encode_window_encoder_dispatch_count,
      statistics.streaming_rhs_encode.dispatch_count,
      CeilMiB(statistics.streaming_rhs_encode.read_byte_length),
      CeilMiB(statistics.streaming_rhs_encode.write_byte_length),
      CeilMiB(statistics.streaming_rhs_encode.max_write_byte_length));
  if (iree_status_is_ok(status)) {
    status = AppendGenerationBenchmarkPhaseStageMasksLabel(&label_builder,
                                                           residency);
  }
  if (iree_status_is_ok(status)) {
    status = id4::test::AppendParameterLoadKindStatisticsLabel(
        &label_builder, statistics.parameter_load_kind_statistics);
  }
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
  PromptBenchmarkDisposition benchmark_disposition =
      PromptBenchmarkDisposition::kRun;
  IREE_RETURN_IF_ERROR(ValidatePromptBenchmarkConfiguration(
      benchmark_state, *prompt, &benchmark_disposition));
  if (benchmark_disposition == PromptBenchmarkDisposition::kSkip) {
    return iree_ok_status();
  }

  LiveGenerationBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLiveGenerationBenchmarkContext(&context));

  ParsedRequest request;
  IREE_RETURN_IF_ERROR(ParsePromptRequest(*prompt, &request));
  iree_string_view_t prompt_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(ResolvePromptLabel(*prompt, &prompt_label));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

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
  id4_ideogram4_generation_resource_statistics_t last_resource_statistics;
  std::memset(&last_resource_statistics, 0, sizeof(last_resource_statistics));
  GenerationResidencyResolution last_residency;
  GenerationBenchmarkTimingStatistics timing_total;
  std::memset(&timing_total, 0, sizeof(timing_total));
  bool plan_json_written = false;
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
    GenerationResidencyResolution residency;
    if (iree_status_is_ok(status)) {
      status =
          ResolveGenerationBenchmarkResidency(context, plan.get(), &residency);
    }
    id4_ideogram4_generation_resource_statistics_t resource_statistics;
    if (iree_status_is_ok(status)) {
      status = QueryGenerationBenchmarkResourceStatistics(
          context, plan.get(), residency, &resource_statistics);
    }
    if (iree_status_is_ok(status)) {
      last_residency = residency;
      last_resource_statistics = resource_statistics;
    }
    iteration_timing.plan_ns += iree_time_now() - phase_start_time_ns;
    if (iree_status_is_ok(status) && !plan_json_written) {
      iree_benchmark_pause_timing(benchmark_state);
      status = WriteGenerationPlanJsonIfRequested(prompt_label, context,
                                                  residency, plan.get());
      iree_benchmark_resume_timing(benchmark_state);
      plan_json_written = iree_status_is_ok(status);
    }

    ++prepare_value;
    GenerationBundleRef bundle;
    if (iree_status_is_ok(status)) {
      phase_start_time_ns = iree_time_now();
      status = PrepareGenerationBundle(
          context, plan.get(), residency, &diagnostics_sink,
          prepare_semaphore.get(), prepare_value, bundle.out());
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
      benchmark_state, context, last_residency, IREE_SV("end_to_end"),
      prompt_label, last_summary, last_statistics, last_resource_statistics,
      diagnostics, timing_total, iteration_count));
  iree_benchmark_set_items_processed(
      benchmark_state, static_cast<int64_t>(iteration_count * token_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4SessionGenerationEndToEnd) {
  return RunGenerationEndToEndBenchmark(benchmark_def, benchmark_state);
}

static iree_status_t RunGenerationIssuePreparedBenchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const GenerationPrompt* prompt =
      static_cast<const GenerationPrompt*>(benchmark_def->user_data);
  PromptBenchmarkDisposition benchmark_disposition =
      PromptBenchmarkDisposition::kRun;
  IREE_RETURN_IF_ERROR(ValidatePromptBenchmarkConfiguration(
      benchmark_state, *prompt, &benchmark_disposition));
  if (benchmark_disposition == PromptBenchmarkDisposition::kSkip) {
    return iree_ok_status();
  }

  LiveGenerationBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLiveGenerationBenchmarkContext(&context));

  ParsedRequest request;
  IREE_RETURN_IF_ERROR(ParsePromptRequest(*prompt, &request));
  iree_string_view_t prompt_label = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(ResolvePromptLabel(*prompt, &prompt_label));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

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

  GenerationPlanRef plan;
  IREE_RETURN_IF_ERROR(
      CreateGenerationPlan(context, request, &diagnostics_sink, plan.out()));

  id4_ideogram4_generation_plan_summary_t summary;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_plan_summary(plan.get(), &summary));

  GenerationBenchmarkPlanStatistics statistics;
  IREE_RETURN_IF_ERROR(
      AccumulateGenerationBenchmarkPlanStatistics(plan.get(), &statistics));
  GenerationResidencyResolution residency;
  IREE_RETURN_IF_ERROR(
      ResolveGenerationBenchmarkResidency(context, plan.get(), &residency));
  id4_ideogram4_generation_resource_statistics_t resource_statistics;
  IREE_RETURN_IF_ERROR(QueryGenerationBenchmarkResourceStatistics(
      context, plan.get(), residency, &resource_statistics));

  IREE_RETURN_IF_ERROR(WriteGenerationPlanJsonIfRequested(
      prompt_label, context, residency, plan.get()));

  uint64_t prepare_value = 1;
  uint64_t completion_value = 0;
  GenerationBundleRef bundle;
  IREE_RETURN_IF_ERROR(PrepareGenerationBundle(
      context, plan.get(), residency, &diagnostics_sink,
      prepare_semaphore.get(), prepare_value, bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));

  uint64_t iteration_count = 0;
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
  iree_status_t status = WarmPreparedGenerationState(
      context, residency, bundle.get(), request, &diagnostics_sink,
      prepare_semaphore.get(), &prepare_value, completion_semaphore.get(),
      &completion_value);
  diagnostics = {};
  if (iree_status_is_ok(status) && !capture_execution_profile) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        context.runtime_context.device_group, iree_allocator_system(),
        &profiling);
  }
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    GenerationBenchmarkTimingStatistics iteration_timing;
    std::memset(&iteration_timing, 0, sizeof(iteration_timing));
    const iree_time_t iteration_start_time_ns = iree_time_now();

    const bool profile_this_execution =
        capture_execution_profile && !execution_profile_captured;
    if (profile_this_execution) {
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
      const iree_time_t phase_start_time_ns = iree_time_now();
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
  if (profiling) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(SetGenerationBenchmarkLabel(
      benchmark_state, context, residency, IREE_SV("prepared_issue"),
      prompt_label, summary, statistics, resource_statistics, diagnostics,
      timing_total, iteration_count));
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * summary.qwen_token_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_Ideogram4SessionGenerationIssuePrepared) {
  return RunGenerationIssuePreparedBenchmark(benchmark_def, benchmark_state);
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

static const iree_benchmark_def_t* RegisterPreparedIssueGenerationBenchmark(
    iree_string_view_t name, const GenerationPrompt* prompt) {
  iree_benchmark_def_t* benchmark =
      iree_make_function_benchmark(BM_Ideogram4SessionGenerationIssuePrepared);
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
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationEndToEndCustom_registration
        IREE_ATTRIBUTE_UNUSED = RegisterGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationEndToEnd/custom"),
            &kCustomPrompt);

static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationIssuePreparedShort128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterPreparedIssueGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationIssuePrepared/short128"),
            &kShortPrompt);
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationIssuePreparedMedium128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterPreparedIssueGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationIssuePrepared/medium128"),
            &kMediumPrompt);
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationIssuePreparedStructured128_registration
        IREE_ATTRIBUTE_UNUSED = RegisterPreparedIssueGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationIssuePrepared/structured128"),
            &kStructuredPrompt);
static const iree_benchmark_def_t*
    BM_Ideogram4SessionGenerationIssuePreparedCustom_registration
        IREE_ATTRIBUTE_UNUSED = RegisterPreparedIssueGenerationBenchmark(
            IREE_SV("BM_Ideogram4SessionGenerationIssuePrepared/custom"),
            &kCustomPrompt);

}  // namespace
