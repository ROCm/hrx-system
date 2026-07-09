// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/capture.h"
#include "experimental/id4/tooling/diagnostics.h"
#include "experimental/id4/tooling/filesystem.h"
#include "experimental/id4/tooling/image.h"
#include "experimental/id4/tooling/readback.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/time.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

IREE_FLAG(string, tokenizer, "", "Path to the HuggingFace tokenizer JSON.");
IREE_FLAG(string, prompt, "", "Plain text prompt payload for one generation.");
IREE_FLAG(string, prompt_json, "",
          "Full JSON prompt/configuration payload for one generation.");
IREE_FLAG(string, prompt_json_file, "",
          "Path to a JSON prompt/configuration payload for one generation.");
IREE_FLAG(int32_t, generation_latent_width, 0,
          "Diffusion latent width for --prompt.");
IREE_FLAG(int32_t, generation_latent_height, 0,
          "Diffusion latent height for --prompt.");
IREE_FLAG(int32_t, generation_denoise_steps, 0,
          "Denoise step count for --prompt.");
IREE_FLAG(int64_t, generation_seed, -1,
          "Non-negative deterministic seed for --prompt.");
IREE_FLAG(float, generation_guidance_scale, 0.0f,
          "Positive classifier-free guidance scale for --prompt.");
IREE_FLAG(string, output, "", "Output image path.");
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3. fp8_e4m3 uses the FP8 "
          "scopes below as the branch parameter providers.");
IREE_FLAG(string, qwen_parameter_format, "bf16",
          "Qwen3-VL parameter format: bf16 or fp8_e4m3_block_scaled.");
IREE_FLAG(string, dit_activation_format, "bf16_linear_input",
          "DiT activation format: bf16_linear_input or f32_canonical.");
IREE_FLAG(string, dit_weight_execution_format, "bf16_resident",
          "DiT weight execution format: bf16_resident, fp8_direct, or "
          "fp8_direct_feed_forward_bf16_resident.");
IREE_FLAG(string, qwen_weight_execution_strategy, "hybrid_compact_rhs",
          "Qwen3-VL weight execution strategy: row_major, compact_rhs, or "
          "hybrid_compact_rhs, or streaming_compact_rhs.");
IREE_FLAG(string, qwen_attention_implementation, "auto",
          "Qwen3-VL attention implementation: auto, materialized, or wmma.");
IREE_FLAG(string, dit_attention_implementation, "online_wmma",
          "DiT attention implementation: streaming, materialized_wmma, "
          "blocked_wmma, or online_wmma.");
IREE_FLAG(string, dit_feed_forward_implementation, "fused_product",
          "DiT feed-forward implementation: fused_product or "
          "pytorch_parity.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 branch parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 branch parameter scope.");
IREE_FLAG(string, vae_tiling_mode, "disabled",
          "VAE tiling mode: disabled, explicit_tile_size, relative_tile_size, "
          "or memory_budget.");
IREE_FLAG(int32_t, vae_tile_size_x, 0,
          "VAE latent tile width for --vae_tiling_mode=explicit_tile_size.");
IREE_FLAG(int32_t, vae_tile_size_y, 0,
          "VAE latent tile height for --vae_tiling_mode=explicit_tile_size.");
IREE_FLAG(float, vae_relative_size_x, 0.0f,
          "VAE relative width factor for "
          "--vae_tiling_mode=relative_tile_size.");
IREE_FLAG(float, vae_relative_size_y, 0.0f,
          "VAE relative height factor for "
          "--vae_tiling_mode=relative_tile_size.");
IREE_FLAG(float, vae_overlap, 0.0f,
          "VAE fractional tile overlap for tiling modes.");
IREE_FLAG(int64_t, vae_memory_budget, 0,
          "VAE tile scratch byte budget for --vae_tiling_mode=memory_budget.");
IREE_FLAG(bool, dry_run, false,
          "Plan a full generation request and exit without loading parameters "
          "or issuing device work.");
IREE_FLAG(string, generation_residency, "issue_phases",
          "Generation residency mode: issue_phases, "
          "phase_stage_bundles, selected_stage_bundles, all_stage_bundles, "
          "or memory_budgeted.");
IREE_FLAG(int64_t, generation_residency_budget, 0,
          "Logical live byte budget for "
          "--generation_residency=memory_budgeted.");
IREE_FLAG(string, generation_issue_mode, "phases",
          "Generation issue mode: full, phases, or stage_serial.");
IREE_FLAG(
    int64_t, parameter_load_prefetch_region_distance, 0,
    "Number of future regions whose deferred parameter load groups may be "
    "submitted before the current region is issued.");
IREE_FLAG(string, generation_resident_stage_bundles, "",
          "Comma-separated stage bundles retained or considered by "
          "--generation_residency=selected_stage_bundles or "
          "memory_budgeted: qwen, "
          "sampler_noise, dit_conditioned, dit_unconditioned, "
          "sampler_denoise, decode, or all.");
IREE_FLAG(string, dump_plan, "",
          "Path to write the structured pipeline plan JSON.");
IREE_FLAG(string, dump_diagnostics, "",
          "Directory for loomc, HAL, tensor, and stage diagnostics.");
IREE_FLAG(bool, diagnostic_wait_after_each_region, false,
          "Wait after every scheduler-visible internal region and emit "
          "completion diagnostics. This serializes execution.");
IREE_FLAG(string, diagnostic_region_per_dispatch_stages, "",
          "Comma-separated stage names to plan with one semantic dispatch per "
          "executable region for fault-localization diagnostics.");
IREE_FLAG(string, dump_result_summary, "",
          "Path to write final latent and decoded image F32 summary JSON.");
IREE_FLAG(string, dump_result_tensors, "",
          "Directory to write final latent, decoded image, and diagnostic tap "
          "tensors as id4tensor-v1 files.");
IREE_FLAG_LIST(
    string, diagnostic_tap,
    "Stage-qualified diagnostic tap to capture as <stage>:<tap>. Repeat to "
    "capture multiple taps.");
IREE_FLAG(string, profile_output, "",
          "Path to write queue, dispatch, and memory profiling statistics.");

typedef enum id4_cli_generation_issue_mode_e {
  // Issue the whole generation through the monolithic convenience API.
  ID4_CLI_GENERATION_ISSUE_MODE_FULL = 0,
  // Issue generation through explicit conditioning, denoise, and decode phases.
  ID4_CLI_GENERATION_ISSUE_MODE_PHASES = 1,
  // Issue the whole generation by serializing heavyweight stage submission.
  ID4_CLI_GENERATION_ISSUE_MODE_STAGE_SERIAL = 2,
} id4_cli_generation_issue_mode_t;

typedef enum id4_cli_generation_residency_request_mode_e {
  // Prepare stage bundles at issue-time phase boundaries.
  ID4_CLI_GENERATION_RESIDENCY_REQUEST_ISSUE_PHASES = 0,
  // Prepare materialized stage bundles at phase boundaries.
  ID4_CLI_GENERATION_RESIDENCY_REQUEST_PHASE_STAGE_BUNDLES = 1,
  // Retain exactly the user-selected coarse stage bundles.
  ID4_CLI_GENERATION_RESIDENCY_REQUEST_SELECTED_STAGE_BUNDLES = 2,
  // Retain every coarse stage bundle.
  ID4_CLI_GENERATION_RESIDENCY_REQUEST_ALL_STAGE_BUNDLES = 3,
  // Select retained stage bundles from candidates under a memory budget.
  ID4_CLI_GENERATION_RESIDENCY_REQUEST_MEMORY_BUDGETED = 4,
} id4_cli_generation_residency_request_mode_t;

static uint64_t id4_cli_ceil_mib(iree_device_size_t byte_length) {
  return ((uint64_t)byte_length + 1024u * 1024u - 1u) / (1024u * 1024u);
}

static iree_string_view_t id4_cli_generation_issue_mode_name(
    id4_cli_generation_issue_mode_t issue_mode) {
  switch (issue_mode) {
    case ID4_CLI_GENERATION_ISSUE_MODE_FULL:
      return IREE_SV("full");
    case ID4_CLI_GENERATION_ISSUE_MODE_PHASES:
      return IREE_SV("phases");
    case ID4_CLI_GENERATION_ISSUE_MODE_STAGE_SERIAL:
      return IREE_SV("stage_serial");
    default:
      return IREE_SV("invalid");
  }
}

static iree_string_view_t id4_cli_generation_residency_mode_name(
    id4_ideogram4_generation_residency_mode_t residency_mode) {
  switch (residency_mode) {
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

static iree_status_t id4_cli_load_tokenizer(iree_string_view_t tokenizer_path,
                                            iree_allocator_t host_allocator,
                                            iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  if (iree_string_view_is_empty(tokenizer_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokenizer is required");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      tokenizer_path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json =
        iree_make_string_view((const char*)file_contents->const_buffer.data,
                              file_contents->const_buffer.data_length);
    status = iree_tokenizer_from_huggingface_json(json, host_allocator,
                                                  out_tokenizer);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t id4_cli_parse_request(
    iree_allocator_t host_allocator, id4_ideogram4_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  iree_string_view_t prompt = iree_make_cstring_view(FLAG_prompt);
  iree_string_view_t prompt_json = iree_make_cstring_view(FLAG_prompt_json);
  iree_string_view_t prompt_json_file =
      iree_make_cstring_view(FLAG_prompt_json_file);
  const bool has_prompt = !iree_string_view_is_empty(prompt);
  const bool has_inline_prompt = !iree_string_view_is_empty(prompt_json);
  const bool has_prompt_file = !iree_string_view_is_empty(prompt_json_file);
  const uint32_t prompt_source_count = (has_prompt ? 1u : 0u) +
                                       (has_inline_prompt ? 1u : 0u) +
                                       (has_prompt_file ? 1u : 0u);
  if (prompt_source_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exactly one of --prompt, --prompt_json, or --prompt_json_file is "
        "required");
  }
  if (has_prompt) {
    id4_ideogram4_request_generation_t generation;
    memset(&generation, 0, sizeof(generation));
    if (FLAG_generation_latent_width <= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_latent_width must be greater than zero for --prompt");
    }
    if (FLAG_generation_latent_height <= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_latent_height must be greater than zero for --prompt");
    }
    if (FLAG_generation_denoise_steps <= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_denoise_steps must be greater than zero for --prompt");
    }
    if (FLAG_generation_seed < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--generation_seed must be non-negative for --prompt");
    }
    if (!isfinite(FLAG_generation_guidance_scale) ||
        FLAG_generation_guidance_scale <= 0.0f) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--generation_guidance_scale must be positive "
                              "for --prompt");
    }
    generation.latent_width = (uint32_t)FLAG_generation_latent_width;
    generation.latent_height = (uint32_t)FLAG_generation_latent_height;
    generation.denoise_step_count = (uint32_t)FLAG_generation_denoise_steps;
    generation.seed = (uint64_t)FLAG_generation_seed;
    generation.guidance_scale = FLAG_generation_guidance_scale;
    return id4_ideogram4_request_initialize_text(prompt, &generation,
                                                 host_allocator, out_request);
  }
  if (has_inline_prompt) {
    return id4_ideogram4_request_parse_json(prompt_json, host_allocator,
                                            out_request);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status =
      iree_io_file_contents_map(prompt_json_file, IREE_IO_FILE_ACCESS_READ,
                                host_allocator, &file_contents);
  if (iree_status_is_ok(status)) {
    iree_string_view_t file_json =
        iree_make_string_view((const char*)file_contents->const_buffer.data,
                              file_contents->const_buffer.data_length);
    status = id4_ideogram4_request_parse_json(file_json, host_allocator,
                                              out_request);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t id4_cli_initialize_diagnostics_sink(
    iree_allocator_t host_allocator,
    id4_tooling_diagnostics_file_sink_t* file_sink,
    id4_pipeline_diagnostics_sink_t* out_sink,
    bool* out_file_sink_initialized) {
  IREE_ASSERT_ARGUMENT(file_sink);
  IREE_ASSERT_ARGUMENT(out_sink);
  IREE_ASSERT_ARGUMENT(out_file_sink_initialized);
  memset(file_sink, 0, sizeof(*file_sink));
  *out_file_sink_initialized = false;
  iree_string_view_t dump_directory =
      iree_make_cstring_view(FLAG_dump_diagnostics);
  if (iree_string_view_is_empty(dump_directory)) {
    id4_pipeline_diagnostics_sink_initialize_ignore(out_sink);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(id4_tooling_diagnostics_file_sink_initialize(
      dump_directory, host_allocator, file_sink, out_sink));
  *out_file_sink_initialized = true;
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_dit_parameter_format(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t id4_cli_parse_qwen_parameter_format(
    id4_qwen3_vl_parameter_format_t* out_format) {
  iree_status_t status = id4_qwen3_vl_parameter_format_parse(
      iree_make_cstring_view(FLAG_qwen_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--qwen_parameter_format"));
}

static iree_status_t id4_cli_parse_dit_activation_format(
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

static iree_status_t id4_cli_parse_dit_weight_execution_format(
    id4_ideogram4_dit_weight_execution_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_weight_execution_format_parse(
      iree_make_cstring_view(FLAG_dit_weight_execution_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_weight_execution_format"));
}

static iree_status_t id4_cli_parse_qwen_weight_execution_strategy(
    id4_qwen3_vl_weight_execution_strategy_t* out_strategy) {
  iree_status_t status = id4_qwen3_vl_weight_execution_strategy_parse(
      iree_make_cstring_view(FLAG_qwen_weight_execution_strategy),
      out_strategy);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_weight_execution_strategy"));
}

static iree_status_t id4_cli_parse_qwen_attention_implementation(
    id4_qwen3_vl_attention_implementation_t* out_implementation) {
  iree_status_t status = id4_qwen3_vl_attention_implementation_parse(
      iree_make_cstring_view(FLAG_qwen_attention_implementation),
      out_implementation);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status,
                              IREE_SV("--qwen_attention_implementation"));
}

static iree_status_t id4_cli_parse_dit_attention_implementation(
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

static iree_status_t id4_cli_parse_dit_feed_forward_implementation(
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

static iree_status_t id4_cli_parse_positive_u32_flag(
    int32_t value, iree_string_view_t flag_name, uint32_t* out_value) {
  if (value <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s must be greater than zero",
                            (int)flag_name.size, flag_name.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_positive_i64_flag(
    int64_t value, iree_string_view_t flag_name,
    iree_device_size_t* out_value) {
  if (value <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s must be greater than zero",
                            (int)flag_name.size, flag_name.data);
  }
  *out_value = (iree_device_size_t)value;
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_non_negative_host_size_flag(
    int64_t value, iree_string_view_t flag_name, iree_host_size_t* out_value) {
  if (value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s must be non-negative", (int)flag_name.size,
                            flag_name.data);
  }
  if ((uint64_t)value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s exceeds host size range", (int)flag_name.size,
                            flag_name.data);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static bool id4_cli_vae_tiling_auxiliary_flags_are_default(void) {
  return FLAG_vae_tile_size_x == 0 && FLAG_vae_tile_size_y == 0 &&
         FLAG_vae_relative_size_x == 0.0f && FLAG_vae_relative_size_y == 0.0f &&
         FLAG_vae_overlap == 0.0f && FLAG_vae_memory_budget == 0;
}

static iree_status_t id4_cli_reject_unused_vae_tile_size_flags(
    iree_string_view_t mode) {
  if (FLAG_vae_tile_size_x != 0 || FLAG_vae_tile_size_y != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--vae_tile_size_x/y only apply to "
                            "--vae_tiling_mode=explicit_tile_size, not `%.*s`",
                            (int)mode.size, mode.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_reject_unused_vae_relative_size_flags(
    iree_string_view_t mode) {
  if (FLAG_vae_relative_size_x != 0.0f || FLAG_vae_relative_size_y != 0.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--vae_relative_size_x/y only apply to "
                            "--vae_tiling_mode=relative_tile_size, not `%.*s`",
                            (int)mode.size, mode.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_reject_unused_vae_memory_budget_flag(
    iree_string_view_t mode) {
  if (FLAG_vae_memory_budget != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--vae_memory_budget only applies to "
                            "--vae_tiling_mode=memory_budget, not `%.*s`",
                            (int)mode.size, mode.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_vae_tiling_config(
    id4_vae_tiling_config_t* out_tiling) {
  IREE_ASSERT_ARGUMENT(out_tiling);
  memset(out_tiling, 0, sizeof(*out_tiling));
  iree_string_view_t mode = iree_make_cstring_view(FLAG_vae_tiling_mode);
  if (iree_string_view_equal(mode, IREE_SV("disabled"))) {
    if (!id4_cli_vae_tiling_auxiliary_flags_are_default()) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--vae_tiling_mode=disabled does not accept VAE tiling detail flags");
    }
    out_tiling->mode = ID4_VAE_TILING_MODE_DISABLED;
    return iree_ok_status();
  }
  if (iree_string_view_equal(mode, IREE_SV("explicit_tile_size"))) {
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_relative_size_flags(mode));
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_memory_budget_flag(mode));
    out_tiling->mode = ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE;
    IREE_RETURN_IF_ERROR(id4_cli_parse_positive_u32_flag(
        FLAG_vae_tile_size_x, IREE_SV("--vae_tile_size_x"),
        &out_tiling->tile_size_x));
    IREE_RETURN_IF_ERROR(id4_cli_parse_positive_u32_flag(
        FLAG_vae_tile_size_y, IREE_SV("--vae_tile_size_y"),
        &out_tiling->tile_size_y));
    out_tiling->overlap = FLAG_vae_overlap;
    return iree_ok_status();
  }
  if (iree_string_view_equal(mode, IREE_SV("relative_tile_size"))) {
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_tile_size_flags(mode));
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_memory_budget_flag(mode));
    if (FLAG_vae_relative_size_x <= 0.0f || FLAG_vae_relative_size_y <= 0.0f) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--vae_relative_size_x/y must be greater than zero for "
          "--vae_tiling_mode=relative_tile_size");
    }
    out_tiling->mode = ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE;
    out_tiling->relative_size_x = FLAG_vae_relative_size_x;
    out_tiling->relative_size_y = FLAG_vae_relative_size_y;
    out_tiling->overlap = FLAG_vae_overlap;
    return iree_ok_status();
  }
  if (iree_string_view_equal(mode, IREE_SV("memory_budget"))) {
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_tile_size_flags(mode));
    IREE_RETURN_IF_ERROR(id4_cli_reject_unused_vae_relative_size_flags(mode));
    out_tiling->mode = ID4_VAE_TILING_MODE_MEMORY_BUDGET;
    IREE_RETURN_IF_ERROR(id4_cli_parse_positive_i64_flag(
        FLAG_vae_memory_budget, IREE_SV("--vae_memory_budget"),
        &out_tiling->memory_budget));
    out_tiling->overlap = FLAG_vae_overlap;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--vae_tiling_mode must be disabled, explicit_tile_size, "
      "relative_tile_size, or memory_budget");
}

static iree_status_t id4_cli_parse_generation_residency_request_mode(
    id4_cli_generation_residency_request_mode_t* out_mode) {
  IREE_ASSERT_ARGUMENT(out_mode);
  iree_string_view_t value = iree_make_cstring_view(FLAG_generation_residency);
  if (iree_string_view_equal(value, IREE_SV("issue_phases"))) {
    *out_mode = ID4_CLI_GENERATION_RESIDENCY_REQUEST_ISSUE_PHASES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("phase_stage_bundles"))) {
    *out_mode = ID4_CLI_GENERATION_RESIDENCY_REQUEST_PHASE_STAGE_BUNDLES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("selected_stage_bundles"))) {
    *out_mode = ID4_CLI_GENERATION_RESIDENCY_REQUEST_SELECTED_STAGE_BUNDLES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("all_stage_bundles"))) {
    *out_mode = ID4_CLI_GENERATION_RESIDENCY_REQUEST_ALL_STAGE_BUNDLES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("memory_budgeted"))) {
    *out_mode = ID4_CLI_GENERATION_RESIDENCY_REQUEST_MEMORY_BUDGETED;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--generation_residency must be issue_phases, phase_stage_bundles, "
      "selected_stage_bundles, all_stage_bundles, or memory_budgeted");
}

static iree_status_t id4_cli_parse_generation_issue_mode(
    id4_cli_generation_issue_mode_t* out_mode) {
  IREE_ASSERT_ARGUMENT(out_mode);
  iree_string_view_t value = iree_make_cstring_view(FLAG_generation_issue_mode);
  if (iree_string_view_equal(value, IREE_SV("full"))) {
    *out_mode = ID4_CLI_GENERATION_ISSUE_MODE_FULL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("phases"))) {
    *out_mode = ID4_CLI_GENERATION_ISSUE_MODE_PHASES;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("stage_serial"))) {
    *out_mode = ID4_CLI_GENERATION_ISSUE_MODE_STAGE_SERIAL;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "--generation_issue_mode must be full, phases, or "
                          "stage_serial");
}

static id4_ideogram4_generation_issue_policy_t id4_cli_generation_issue_policy(
    id4_cli_generation_issue_mode_t issue_mode) {
  return issue_mode == ID4_CLI_GENERATION_ISSUE_MODE_STAGE_SERIAL
             ? ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL
             : ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT;
}

static iree_status_t id4_cli_parse_generation_stage_mask(
    iree_string_view_t value, iree_string_view_t flag_name,
    id4_ideogram4_generation_resident_stage_mask_t* out_stage_mask) {
  IREE_ASSERT_ARGUMENT(out_stage_mask);
  *out_stage_mask = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  iree_string_view_t remaining = iree_string_view_trim(value);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t stage_name = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &stage_name, &remaining);
    stage_name = iree_string_view_trim(stage_name);
    if (iree_string_view_is_empty(stage_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s contains an empty stage name",
                              (int)flag_name.size, flag_name.data);
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
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown %.*s value '%.*s'", (int)flag_name.size,
                              flag_name.data, (int)stage_name.size,
                              stage_name.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_generation_resident_stage_mask(
    id4_ideogram4_generation_resident_stage_mask_t* out_stage_mask) {
  return id4_cli_parse_generation_stage_mask(
      iree_make_cstring_view(FLAG_generation_resident_stage_bundles),
      IREE_SV("--generation_resident_stage_bundles"), out_stage_mask);
}

static iree_status_t id4_cli_resolve_generation_residency(
    const id4_ideogram4_generation_plan_t* generation_plan,
    id4_cli_generation_issue_mode_t issue_mode,
    id4_cli_generation_residency_request_mode_t request_mode,
    id4_ideogram4_generation_resident_stage_mask_t requested_stage_mask,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_ideogram4_generation_residency_mode_t* out_residency_mode,
    id4_ideogram4_generation_resident_stage_mask_t* out_resident_stage_mask,
    id4_ideogram4_generation_resident_stage_mask_t* out_phase_stage_masks) {
  IREE_ASSERT_ARGUMENT(generation_plan);
  IREE_ASSERT_ARGUMENT(out_residency_mode);
  IREE_ASSERT_ARGUMENT(out_resident_stage_mask);
  IREE_ASSERT_ARGUMENT(out_phase_stage_masks);
  *out_residency_mode = ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  *out_resident_stage_mask = ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  memset(
      out_phase_stage_masks, 0,
      sizeof(out_phase_stage_masks[0]) * ID4_IDEOGRAM4_GENERATION_PHASE_COUNT);

  if (request_mode != ID4_CLI_GENERATION_RESIDENCY_REQUEST_MEMORY_BUDGETED &&
      FLAG_generation_residency_budget != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--generation_residency_budget requires "
                            "--generation_residency=memory_budgeted");
  }

  switch (request_mode) {
    case ID4_CLI_GENERATION_RESIDENCY_REQUEST_ISSUE_PHASES:
      *out_residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES;
      *out_resident_stage_mask = requested_stage_mask;
      return iree_ok_status();
    case ID4_CLI_GENERATION_RESIDENCY_REQUEST_PHASE_STAGE_BUNDLES:
      *out_residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES;
      *out_resident_stage_mask = requested_stage_mask;
      return iree_ok_status();
    case ID4_CLI_GENERATION_RESIDENCY_REQUEST_SELECTED_STAGE_BUNDLES:
      *out_residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES;
      *out_resident_stage_mask = requested_stage_mask;
      return iree_ok_status();
    case ID4_CLI_GENERATION_RESIDENCY_REQUEST_ALL_STAGE_BUNDLES:
      *out_residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES;
      *out_resident_stage_mask = requested_stage_mask;
      return iree_ok_status();
    case ID4_CLI_GENERATION_RESIDENCY_REQUEST_MEMORY_BUDGETED: {
      if (requested_stage_mask ==
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--generation_resident_stage_bundles must list candidate stages "
            "for --generation_residency=memory_budgeted");
      }
      iree_device_size_t memory_budget = 0;
      IREE_RETURN_IF_ERROR(id4_cli_parse_positive_i64_flag(
          FLAG_generation_residency_budget,
          IREE_SV("--generation_residency_budget"), &memory_budget));
      id4_ideogram4_generation_residency_select_options_t select_options;
      memset(&select_options, 0, sizeof(select_options));
      select_options.structure_size = sizeof(select_options);
      select_options.issue_policy = id4_cli_generation_issue_policy(issue_mode);
      select_options.candidate_stage_mask = requested_stage_mask;
      select_options.parameter_load_prefetch_region_distance =
          parameter_load_prefetch_region_distance;
      select_options.memory_budget_byte_length = memory_budget;
      id4_ideogram4_generation_residency_selection_t selection;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_select_residency(
          generation_plan, &select_options, &selection));
      *out_residency_mode = selection.residency_mode;
      *out_resident_stage_mask = selection.resident_stage_mask;
      memcpy(out_phase_stage_masks, selection.phase_stage_masks,
             sizeof(selection.phase_stage_masks));
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generation residency request mode is invalid");
  }
}

static iree_status_t id4_cli_print_generation_resource_statistics(
    const id4_ideogram4_generation_plan_t* generation_plan,
    id4_cli_generation_issue_mode_t issue_mode,
    id4_cli_generation_residency_request_mode_t request_mode,
    id4_ideogram4_generation_resident_stage_mask_t requested_stage_mask,
    iree_host_size_t parameter_load_prefetch_region_distance) {
  id4_ideogram4_generation_residency_mode_t residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  id4_ideogram4_generation_resident_stage_mask_t
      phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];
  memset(phase_stage_masks, 0, sizeof(phase_stage_masks));
  IREE_RETURN_IF_ERROR(id4_cli_resolve_generation_residency(
      generation_plan, issue_mode, request_mode, requested_stage_mask,
      parameter_load_prefetch_region_distance, &residency_mode,
      &resident_stage_mask, phase_stage_masks));

  id4_ideogram4_generation_resource_statistics_options_t statistics_options;
  memset(&statistics_options, 0, sizeof(statistics_options));
  statistics_options.structure_size = sizeof(statistics_options);
  statistics_options.residency_mode = residency_mode;
  statistics_options.resident_stage_mask = resident_stage_mask;
  memcpy(statistics_options.phase_stage_masks, phase_stage_masks,
         sizeof(statistics_options.phase_stage_masks));
  statistics_options.parameter_load_prefetch_region_distance =
      parameter_load_prefetch_region_distance;
  id4_ideogram4_generation_resource_statistics_t statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_resource_statistics(
      generation_plan, &statistics_options, &statistics));

  const id4_ideogram4_generation_issue_policy_t issue_policy =
      id4_cli_generation_issue_policy(issue_mode);
  const iree_device_size_t selected_peak_byte_length =
      issue_policy == ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL
          ? statistics.stage_serial_total_peak_byte_length
          : statistics.phase_concurrent_total_peak_byte_length;
  const iree_string_view_t issue_name =
      id4_cli_generation_issue_mode_name(issue_mode);
  const iree_string_view_t residency_name =
      id4_cli_generation_residency_mode_name(residency_mode);
  fprintf(stdout,
          "Ideogram 4 generation resources: issue=%.*s residency=%.*s "
          "resident_stage_mask=0x%08x phase_stage_masks=[",
          (int)issue_name.size, issue_name.data, (int)residency_name.size,
          residency_name.data, resident_stage_mask);
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_PHASE_COUNT; ++i) {
    fprintf(stdout, "%s0x%08x", i == 0 ? "" : ",", phase_stage_masks[i]);
  }
  fprintf(stdout,
          "] selected_peak=%" PRIu64 "MiB phase_peak=%" PRIu64
          "MiB stage_serial_peak=%" PRIu64 "MiB resident=%" PRIu64
          "MiB boundary=%" PRIu64 "MiB diagnostic_taps=%" PRIu64 "MiB\n",
          id4_cli_ceil_mib(selected_peak_byte_length),
          id4_cli_ceil_mib(statistics.phase_concurrent_total_peak_byte_length),
          id4_cli_ceil_mib(statistics.stage_serial_total_peak_byte_length),
          id4_cli_ceil_mib(statistics.resident_stage_bundle_byte_length),
          id4_cli_ceil_mib(statistics.boundary_buffer_byte_length),
          id4_cli_ceil_mib(statistics.diagnostic_tap_buffer_byte_length));
  return iree_ok_status();
}

static iree_status_t id4_cli_validate_execution_flags(void) {
  if (strlen(FLAG_output) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--output is required when executing generation");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_validate_dry_run_flags(void) {
  if (strlen(FLAG_profile_output) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--profile_output requires generation execution; omit --dry_run");
  }
  if (strlen(FLAG_dump_result_summary) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--dump_result_summary requires generation execution; omit --dry_run");
  }
  if (strlen(FLAG_dump_result_tensors) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--dump_result_tensors requires generation execution; omit --dry_run");
  }
  if (strlen(FLAG_output) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output requires generation execution; omit --dry_run");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_write_profile_statistics(
    iree_hal_profile_statistics_sink_t* statistics_sink) {
  IREE_ASSERT_ARGUMENT(statistics_sink);
  if (strlen(FLAG_profile_output) == 0) return iree_ok_status();

  FILE* file = fopen(FLAG_profile_output, "w");
  if (!file) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to open --profile_output=%s: %s",
                            FLAG_profile_output, strerror(errno));
  }
  iree_status_t status =
      iree_hal_profile_statistics_sink_fprint(file, statistics_sink);
  if (fclose(file) != 0) {
    status = iree_status_join(
        status, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "failed to close --profile_output=%s: %s",
                                 FLAG_profile_output, strerror(errno)));
  }
  return status;
}

static iree_status_t id4_cli_begin_profile_statistics(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_profile_statistics_sink_t** out_statistics_sink,
    bool* out_profile_started) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_statistics_sink);
  IREE_ASSERT_ARGUMENT(out_profile_started);
  *out_statistics_sink = NULL;
  *out_profile_started = false;
  if (strlen(FLAG_profile_output) == 0) return iree_ok_status();

  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  iree_status_t status =
      iree_hal_profile_statistics_sink_create(host_allocator, &statistics_sink);
  if (iree_status_is_ok(status)) {
    iree_hal_device_profiling_options_t profiling_options = {0};
    profiling_options.data_families =
        IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
        IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
        IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
        IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
    profiling_options.sink =
        iree_hal_profile_statistics_sink_base(statistics_sink);
    status = iree_hal_device_profiling_begin(device, &profiling_options);
  }
  if (iree_status_is_ok(status)) {
    *out_statistics_sink = statistics_sink;
    *out_profile_started = true;
  } else {
    iree_hal_profile_statistics_sink_release(statistics_sink);
  }
  return status;
}

static iree_status_t id4_cli_end_profile_statistics(
    iree_hal_device_t* device,
    iree_hal_profile_statistics_sink_t* statistics_sink) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(statistics_sink);

  iree_status_t status = iree_hal_device_profiling_flush(device);
  status = iree_status_join(status, iree_hal_device_profiling_end(device));
  if (iree_status_is_ok(status)) {
    status = id4_cli_write_profile_statistics(statistics_sink);
  }
  return status;
}

static iree_hal_command_buffer_mode_t id4_cli_generation_command_buffer_mode(
    iree_hal_command_buffer_mode_t base_mode) {
  if (strlen(FLAG_profile_output) == 0) return base_mode;
  return base_mode | IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
         IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
}

static iree_status_t id4_cli_write_generation_plan(
    iree_string_view_t output_path, const id4_ideogram4_generation_plan_t* plan,
    iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(output_path)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      id4_ideogram4_generation_plan_format_json(plan, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        output_path, iree_make_const_byte_span(json.data, json.size),
        host_allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static id4_pipeline_tensor_shape_t id4_cli_convert_program_shape(
    id4_pipeline_program_shape_t shape) {
  id4_pipeline_tensor_shape_t tensor_shape;
  memset(&tensor_shape, 0, sizeof(tensor_shape));
  tensor_shape.rank = shape.rank;
  memcpy(tensor_shape.dims, shape.dims, sizeof(tensor_shape.dims));
  return tensor_shape;
}

static iree_status_t id4_cli_append_json_string(iree_string_builder_t* builder,
                                                iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    switch (value.data[i]) {
      case '\\': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\\"));
        break;
      }
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\""));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      default: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
            builder, iree_make_string_view(value.data + i, 1)));
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t id4_cli_append_shape_json(
    iree_string_builder_t* builder, id4_pipeline_tensor_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"rank\":%" PRIu32 ",\"dims\":[", shape.rank));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static uint16_t id4_cli_load_u16(const uint8_t* bytes) {
  uint16_t value = 0;
  memcpy(&value, bytes, sizeof(value));
  return value;
}

static float id4_cli_load_f32(const uint8_t* bytes) {
  float value = 0.0f;
  memcpy(&value, bytes, sizeof(value));
  return value;
}

static iree_status_t id4_cli_load_tensor_element_as_f32(
    id4_pipeline_tensor_dtype_t dtype, const uint8_t* bytes,
    iree_host_size_t index, float* out_value) {
  const iree_device_size_t dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor summary dtype is invalid");
  }
  if (index > IREE_HOST_SIZE_MAX / dtype_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor summary byte offset overflow");
  }
  const iree_host_size_t byte_offset =
      index * (iree_host_size_t)dtype_byte_length;
  switch (dtype) {
    case ID4_PIPELINE_TENSOR_DTYPE_F32:
      *out_value = id4_cli_load_f32(bytes + byte_offset);
      return iree_ok_status();
    case ID4_PIPELINE_TENSOR_DTYPE_BF16:
      *out_value = iree_math_bf16_to_f32(id4_cli_load_u16(bytes + byte_offset));
      return iree_ok_status();
    default: {
      iree_string_view_t dtype_name = id4_pipeline_tensor_dtype_format(dtype);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "tensor summary dtype `%.*s` cannot be summarized as F32",
          (int)dtype_name.size, dtype_name.data);
    }
  }
}

static iree_status_t id4_cli_append_tensor_summary_json(
    iree_string_builder_t* builder, iree_string_view_t key,
    const id4_pipeline_tensor_layout_t* layout, iree_const_byte_span_t bytes) {
  const iree_device_size_t dtype_byte_length =
      id4_pipeline_tensor_dtype_byte_length(layout->dtype);
  if (dtype_byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor summary `%.*s` has invalid dtype",
                            (int)key.size, key.data);
  }
  if (bytes.data_length % dtype_byte_length != 0) {
    iree_string_view_t dtype_name =
        id4_pipeline_tensor_dtype_format(layout->dtype);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tensor summary `%.*s` byte length %" PRIhsz
                            " is not a multiple of dtype `%.*s` size %" PRIu64,
                            (int)key.size, key.data, bytes.data_length,
                            (int)dtype_name.size, dtype_name.data,
                            (uint64_t)dtype_byte_length);
  }
  const iree_host_size_t element_count =
      bytes.data_length / (iree_host_size_t)dtype_byte_length;
  iree_host_size_t finite_count = 0;
  iree_host_size_t nan_count = 0;
  iree_host_size_t infinity_count = 0;
  iree_host_size_t first_nonfinite_index = IREE_HOST_SIZE_MAX;
  float finite_min = 0.0f;
  float finite_max = 0.0f;
  double finite_sum = 0.0;
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    float value = 0.0f;
    IREE_RETURN_IF_ERROR(id4_cli_load_tensor_element_as_f32(
        layout->dtype, bytes.data, i, &value));
    if (isfinite(value)) {
      if (finite_count == 0) {
        finite_min = value;
        finite_max = value;
      } else {
        if (value < finite_min) finite_min = value;
        if (value > finite_max) finite_max = value;
      }
      finite_sum += (double)value;
      ++finite_count;
    } else {
      if (first_nonfinite_index == IREE_HOST_SIZE_MAX) {
        first_nonfinite_index = i;
      }
      if (isnan(value)) {
        ++nan_count;
      } else {
        ++infinity_count;
      }
    }
  }

  IREE_RETURN_IF_ERROR(id4_cli_append_json_string(builder, key));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":{"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "\"dtype\":"));
  IREE_RETURN_IF_ERROR(id4_cli_append_json_string(
      builder, id4_pipeline_tensor_dtype_format(layout->dtype)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"shape\":"));
  IREE_RETURN_IF_ERROR(id4_cli_append_shape_json(builder, layout->shape));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"byte_length\":%" PRIhsz ",\"element_count\":%" PRIhsz
      ",\"finite_count\":%" PRIhsz ",\"nan_count\":%" PRIhsz
      ",\"infinity_count\":%" PRIhsz,
      bytes.data_length, element_count, finite_count, nan_count,
      infinity_count));
  if (first_nonfinite_index == IREE_HOST_SIZE_MAX) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, ",\"first_nonfinite_index\":null"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"first_nonfinite_index\":%" PRIhsz, first_nonfinite_index));
  }
  if (finite_count == 0) {
    return iree_string_builder_append_cstring(
        builder,
        ",\"finite_min\":null,\"finite_max\":null,\"finite_mean\":null}");
  }
  return iree_string_builder_append_format(
      builder,
      ",\"finite_min\":%.9g,\"finite_max\":%.9g,\"finite_mean\":%.17g}",
      finite_min, finite_max, finite_sum / (double)finite_count);
}

typedef struct id4_cli_result_tensor_summary_t {
  // Stable JSON key used for this tensor summary.
  iree_string_view_t key;
  // Logical tensor layout represented by |bytes|.
  id4_pipeline_tensor_layout_t layout;
  // Host readback bytes for this tensor.
  iree_const_byte_span_t bytes;
} id4_cli_result_tensor_summary_t;

static iree_status_t id4_cli_write_result_summary(
    iree_string_view_t output_path, iree_host_size_t tensor_count,
    const id4_cli_result_tensor_summary_t* tensors,
    iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(output_path)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status = iree_string_builder_append_cstring(&builder, "{");
  for (iree_host_size_t i = 0; i < tensor_count && iree_status_is_ok(status);
       ++i) {
    if (i != 0) {
      status = iree_string_builder_append_cstring(&builder, ",");
    }
    if (iree_status_is_ok(status)) {
      status = id4_cli_append_tensor_summary_json(
          &builder, tensors[i].key, &tensors[i].layout, tensors[i].bytes);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "}\n");
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        output_path, iree_make_const_byte_span(json.data, json.size),
        host_allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t id4_cli_append_result_tensor_record_json(
    iree_string_builder_t* builder, iree_host_size_t ordinal,
    iree_string_view_t file_name,
    const id4_cli_result_tensor_summary_t* tensor) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"ordinal\":%" PRIhsz ",\"key\":", ordinal));
  IREE_RETURN_IF_ERROR(id4_cli_append_json_string(builder, tensor->key));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"file\":"));
  IREE_RETURN_IF_ERROR(id4_cli_append_json_string(builder, file_name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"dtype\":"));
  IREE_RETURN_IF_ERROR(id4_cli_append_json_string(
      builder, id4_pipeline_tensor_dtype_format(tensor->layout.dtype)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"shape\":"));
  IREE_RETURN_IF_ERROR(
      id4_cli_append_shape_json(builder, tensor->layout.shape));
  return iree_string_builder_append_format(
      builder, ",\"byte_length\":%" PRIhsz "}", tensor->bytes.data_length);
}

static iree_status_t id4_cli_write_result_tensors(
    iree_string_view_t output_directory, iree_host_size_t tensor_count,
    const id4_cli_result_tensor_summary_t* tensors,
    iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(output_directory)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(output_directory, host_allocator));

  iree_string_builder_t manifest_builder;
  iree_string_builder_initialize(host_allocator, &manifest_builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &manifest_builder,
      "{\n  \"schema_version\": 1,\n  \"format\": \"id4tensor-v1\",\n"
      "  \"records\": [\n");
  for (iree_host_size_t i = 0; i < tensor_count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t file_name = iree_string_view_empty();
    iree_string_view_t file_path = iree_string_view_empty();
    char file_name_storage[64];
    const int file_name_length =
        snprintf(file_name_storage, sizeof(file_name_storage),
                 "result_%04" PRIhsz ".id4tensor", i);
    if (file_name_length < 0 ||
        (iree_host_size_t)file_name_length >= sizeof(file_name_storage)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "result tensor file name is too large");
    }
    if (iree_status_is_ok(status)) {
      file_name = iree_make_string_view(file_name_storage,
                                        (iree_host_size_t)file_name_length);
      status = id4_tooling_format_child_path(output_directory, file_name,
                                             host_allocator, &file_path);
    }
    if (iree_status_is_ok(status)) {
      status = id4_tooling_capture_write_tensor_file(
          file_path, &tensors[i].layout, tensors[i].bytes, host_allocator);
    }
    if (iree_status_is_ok(status)) {
      if (i != 0) {
        status = iree_string_builder_append_cstring(&manifest_builder, ",\n");
      }
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(&manifest_builder, "    ");
      }
      if (iree_status_is_ok(status)) {
        status = id4_cli_append_result_tensor_record_json(
            &manifest_builder, i, file_name, &tensors[i]);
      }
    }
    id4_tooling_free_path(&file_path, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_cstring(&manifest_builder, "\n  ]\n}\n");
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t manifest_path = iree_string_view_empty();
    status = id4_tooling_format_child_path(output_directory,
                                           IREE_SV("manifest.json"),
                                           host_allocator, &manifest_path);
    if (iree_status_is_ok(status)) {
      iree_string_view_t manifest = iree_string_builder_view(&manifest_builder);
      status = iree_io_file_contents_write(
          manifest_path,
          iree_make_const_byte_span(manifest.data, manifest.size),
          host_allocator);
    }
    id4_tooling_free_path(&manifest_path, host_allocator);
  }
  iree_string_builder_deinitialize(&manifest_builder);
  return status;
}

typedef struct id4_cli_diagnostic_tap_request_t {
  // Stable summary key preserving the caller's stage-qualified tap spelling.
  iree_string_view_t summary_key;
  // Stable generation stage key owning the requested tap.
  iree_string_view_t stage_key;
  // Diagnostic tap name inside the owning stage.
  iree_string_view_t tap_name;
} id4_cli_diagnostic_tap_request_t;

typedef struct id4_cli_generation_diagnostic_taps_t {
  // Number of parsed stage-qualified tap requests.
  iree_host_size_t request_count;
  // Parsed tap requests in flag order.
  id4_cli_diagnostic_tap_request_t* requests;
  // Number of stage groups supplied to generation planning.
  iree_host_size_t list_count;
  // Stage-grouped tap selections supplied to generation planning.
  id4_ideogram4_generation_stage_diagnostic_tap_list_t* lists;
  // Contiguous tap-name storage referenced by |lists|.
  iree_string_view_t* grouped_tap_names;
} id4_cli_generation_diagnostic_taps_t;

static void id4_cli_generation_diagnostic_taps_deinitialize(
    id4_cli_generation_diagnostic_taps_t* taps,
    iree_allocator_t host_allocator) {
  if (!taps) return;
  iree_allocator_free(host_allocator, taps->grouped_tap_names);
  iree_allocator_free(host_allocator, taps->lists);
  iree_allocator_free(host_allocator, taps->requests);
  memset(taps, 0, sizeof(*taps));
}

static iree_status_t id4_cli_parse_diagnostic_tap_value(
    iree_string_view_t value, id4_cli_diagnostic_tap_request_t* out_request) {
  memset(out_request, 0, sizeof(*out_request));
  value = iree_string_view_trim(value);
  const iree_host_size_t delimiter =
      iree_string_view_find(value, IREE_SV(":"), 0);
  if (delimiter == IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--diagnostic_tap value `%.*s` must be stage-qualified as "
        "<stage>:<tap>",
        (int)value.size, value.data);
  }
  iree_string_view_t stage_key =
      iree_string_view_trim(iree_string_view_substr(value, 0, delimiter));
  iree_string_view_t tap_name = iree_string_view_trim(
      iree_string_view_substr(value, delimiter + 1, IREE_STRING_VIEW_NPOS));
  if (iree_string_view_is_empty(stage_key) ||
      iree_string_view_is_empty(tap_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--diagnostic_tap value `%.*s` must contain non-empty stage and tap "
        "names",
        (int)value.size, value.data);
  }
  out_request->summary_key = value;
  out_request->stage_key = stage_key;
  out_request->tap_name = tap_name;
  return iree_ok_status();
}

static bool id4_cli_diagnostic_tap_request_matches_stage(
    const id4_cli_diagnostic_tap_request_t* request,
    iree_string_view_t stage_key) {
  return iree_string_view_equal(request->stage_key, stage_key);
}

static iree_status_t id4_cli_parse_diagnostic_taps(
    iree_allocator_t host_allocator,
    id4_cli_generation_diagnostic_taps_t* out_taps) {
  memset(out_taps, 0, sizeof(*out_taps));
  const iree_flag_string_list_t flags = FLAG_diagnostic_tap_list();
  if (flags.count == 0) return iree_ok_status();

  id4_cli_generation_diagnostic_taps_t taps;
  memset(&taps, 0, sizeof(taps));
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, flags.count, sizeof(taps.requests[0]),
      (void**)&taps.requests);
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, flags.count,
                                    sizeof(taps.lists[0]), (void**)&taps.lists);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, flags.count,
                                         sizeof(taps.grouped_tap_names[0]),
                                         (void**)&taps.grouped_tap_names);
  }
  for (iree_host_size_t i = 0; i < flags.count && iree_status_is_ok(status);
       ++i) {
    status =
        id4_cli_parse_diagnostic_tap_value(flags.values[i], &taps.requests[i]);
    if (!iree_status_is_ok(status)) break;
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(taps.requests[i].stage_key,
                                 taps.requests[j].stage_key) &&
          iree_string_view_equal(taps.requests[i].tap_name,
                                 taps.requests[j].tap_name)) {
        status = iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "--diagnostic_tap contains duplicate request `%.*s`",
            (int)taps.requests[i].summary_key.size,
            taps.requests[i].summary_key.data);
        break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    taps.request_count = flags.count;
    iree_host_size_t grouped_tap_offset = 0;
    for (iree_host_size_t i = 0; i < taps.request_count; ++i) {
      bool first_stage_request = true;
      for (iree_host_size_t j = 0; j < i; ++j) {
        if (id4_cli_diagnostic_tap_request_matches_stage(
                &taps.requests[j], taps.requests[i].stage_key)) {
          first_stage_request = false;
          break;
        }
      }
      if (!first_stage_request) continue;

      iree_host_size_t stage_tap_count = 0;
      for (iree_host_size_t j = i; j < taps.request_count; ++j) {
        if (id4_cli_diagnostic_tap_request_matches_stage(
                &taps.requests[j], taps.requests[i].stage_key)) {
          taps.grouped_tap_names[grouped_tap_offset + stage_tap_count] =
              taps.requests[j].tap_name;
          ++stage_tap_count;
        }
      }
      taps.lists[taps.list_count].stage_key = taps.requests[i].stage_key;
      taps.lists[taps.list_count].tap_names = (iree_string_view_list_t){
          stage_tap_count, &taps.grouped_tap_names[grouped_tap_offset]};
      ++taps.list_count;
      grouped_tap_offset += stage_tap_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_taps = taps;
  } else {
    id4_cli_generation_diagnostic_taps_deinitialize(&taps, host_allocator);
  }
  return status;
}

static iree_status_t id4_cli_emit_timing(
    id4_pipeline_diagnostics_sink_t* diagnostics_sink, iree_string_view_t key,
    iree_string_view_t message, iree_time_t start_time_ns,
    iree_time_t end_time_ns) {
  id4_pipeline_timing_diagnostic_t timing = {
      // Monotonic start timestamp for this CLI phase.
      .start_time_ns = start_time_ns,
      // Monotonic end timestamp for this CLI phase.
      .end_time_ns = end_time_ns,
      // Elapsed duration for this CLI phase.
      .duration_ns = end_time_ns - start_time_ns,
  };
  id4_pipeline_diagnostic_event_t event = {
      // This event describes a host-observed CLI phase duration.
      .kind = ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_TIMING,
      // CLI phase timings are emitted outside any one model stage.
      .stage_name = IREE_SV("id4.cli"),
      // Stable timing key identifying the CLI phase.
      .key = key,
      // Short phase summary.
      .message = message,
      // No parameter slab payload is attached to timing events.
      .parameter_slab = NULL,
      // No kernel payload is attached to timing events.
      .kernel = NULL,
      // Timing payload valid for this emit call.
      .timing = &timing,
  };
  return id4_pipeline_diagnostics_emit(diagnostics_sink, &event);
}

static iree_hal_semaphore_list_t id4_cli_single_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = semaphore_storage,
      // Stack-backed payload value.
      .payload_values = payload_storage,
  };
  return list;
}

static iree_status_t id4_cli_wait_for_semaphore(iree_hal_semaphore_t* semaphore,
                                                uint64_t payload_value) {
  iree_hal_semaphore_t* semaphore_storage = semaphore;
  uint64_t payload_storage = payload_value;
  iree_hal_semaphore_list_t wait_list = {
      // One semaphore edge to observe before continuing.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = &semaphore_storage,
      // Stack-backed payload value.
      .payload_values = &payload_storage,
  };
  return iree_hal_semaphore_list_wait(wait_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_cli_prepare_generation_phase_bundle(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_payload_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_phase_bundle_t** out_phase_bundle) {
  iree_hal_semaphore_t* wait_semaphore_storage = NULL;
  uint64_t wait_payload_storage = 0;
  iree_hal_semaphore_t* signal_semaphore_storage = NULL;
  uint64_t signal_payload_storage = 0;
  iree_hal_semaphore_list_t wait_list = id4_cli_single_semaphore_list(
      &wait_semaphore_storage, &wait_payload_storage, wait_semaphore,
      wait_payload_value);
  iree_hal_semaphore_list_t signal_list = id4_cli_single_semaphore_list(
      &signal_semaphore_storage, &signal_payload_storage, signal_semaphore,
      signal_payload_value);

  id4_ideogram4_generation_phase_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.phase_mask = phase_mask;
  prepare_options.wait_semaphore_list = wait_list;
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_generation_bundle_prepare_phase(bundle, &prepare_options,
                                                       out_phase_bundle);
}

static iree_status_t id4_cli_issue_generation_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_t* wait_semaphore, uint64_t wait_payload_value,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value,
    id4_pipeline_stage_issue_flags_t stage_issue_flags,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphore_storage = NULL;
  uint64_t wait_payload_storage = 0;
  iree_hal_semaphore_t* signal_semaphore_storage = NULL;
  uint64_t signal_payload_storage = 0;
  iree_hal_semaphore_list_t wait_list = id4_cli_single_semaphore_list(
      &wait_semaphore_storage, &wait_payload_storage, wait_semaphore,
      wait_payload_value);
  iree_hal_semaphore_list_t signal_list = id4_cli_single_semaphore_list(
      &signal_semaphore_storage, &signal_payload_storage, signal_semaphore,
      signal_payload_value);

  id4_ideogram4_generation_phase_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.wait_semaphore_list = wait_list;
  issue_options.stage_issue_flags = stage_issue_flags;
  issue_options.parameter_load_prefetch_region_distance =
      parameter_load_prefetch_region_distance;
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_generation_execution_issue_phase(execution, phase_bundle,
                                                        &issue_options);
}

static iree_status_t id4_cli_prepare_issue_release_generation_phase(
    id4_ideogram4_generation_bundle_t* bundle,
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_string_view_t prepare_event_name, iree_string_view_t issue_event_name,
    iree_string_view_t wait_event_name, iree_string_view_t release_event_name,
    iree_hal_semaphore_t* phase_wait_semaphore,
    uint64_t phase_wait_payload_value, iree_hal_semaphore_t* prepare_semaphore,
    uint64_t* inout_prepare_payload_value,
    iree_hal_semaphore_t* completion_semaphore,
    uint64_t* inout_completion_payload_value,
    id4_pipeline_stage_issue_flags_t stage_issue_flags,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_phase_bundle_t* phase_bundle = NULL;
  iree_time_t phase_start_time_ns = iree_time_now();
  iree_status_t status = id4_cli_prepare_generation_phase_bundle(
      bundle, phase_mask, phase_wait_semaphore, phase_wait_payload_value,
      prepare_semaphore, ++*inout_prepare_payload_value, diagnostics_sink,
      &phase_bundle);
  if (iree_status_is_ok(status)) {
    status = id4_cli_emit_timing(diagnostics_sink, prepare_event_name,
                                 IREE_SV("prepared generation phase"),
                                 phase_start_time_ns, iree_time_now());
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = id4_cli_issue_generation_phase(
        execution, phase_bundle, prepare_semaphore,
        *inout_prepare_payload_value, completion_semaphore,
        ++*inout_completion_payload_value, stage_issue_flags,
        parameter_load_prefetch_region_distance, diagnostics_sink);
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(diagnostics_sink, issue_event_name,
                                   IREE_SV("issued generation phase"),
                                   phase_start_time_ns, iree_time_now());
    }
  }
  if (iree_status_is_ok(status)) {
    phase_start_time_ns = iree_time_now();
    status = id4_cli_wait_for_semaphore(completion_semaphore,
                                        *inout_completion_payload_value);
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(diagnostics_sink, wait_event_name,
                                   IREE_SV("waited for generation phase"),
                                   phase_start_time_ns, iree_time_now());
    }
  }
  phase_start_time_ns = iree_time_now();
  status = iree_status_join(
      status, id4_ideogram4_generation_phase_bundle_release(phase_bundle));
  if (iree_status_is_ok(status)) {
    status = id4_cli_emit_timing(diagnostics_sink, release_event_name,
                                 IREE_SV("released generation phase"),
                                 phase_start_time_ns, iree_time_now());
  }
  return status;
}

static iree_status_t id4_cli_issue_generation_full(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_request_t* request, iree_tokenizer_t* tokenizer,
    id4_ideogram4_generation_issue_policy_t issue_policy,
    iree_hal_semaphore_list_t prepare_wait_list,
    iree_hal_semaphore_list_t completion_signal_list,
    id4_pipeline_stage_issue_flags_t stage_issue_flags,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_execution_t** out_execution) {
  id4_ideogram4_generation_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.request = request;
  issue_options.tokenizer = tokenizer;
  issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  issue_options.issue_policy = issue_policy;
  issue_options.stage_issue_flags = stage_issue_flags;
  issue_options.parameter_load_prefetch_region_distance =
      parameter_load_prefetch_region_distance;
  issue_options.wait_semaphore_list = prepare_wait_list;
  issue_options.signal_semaphore_list = completion_signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  return id4_ideogram4_session_issue_generation(session, bundle, &issue_options,
                                                out_execution);
}

static iree_status_t id4_cli_issue_generation_phases(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_request_t* request, iree_tokenizer_t* tokenizer,
    iree_hal_semaphore_t* prepare_semaphore,
    uint64_t* inout_prepare_payload_value,
    iree_hal_semaphore_t* completion_semaphore,
    uint64_t* inout_completion_payload_value,
    id4_pipeline_stage_issue_flags_t stage_issue_flags,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_ideogram4_generation_execution_t** out_execution) {
  *out_execution = NULL;
  iree_hal_semaphore_t* begin_wait_semaphore_storage = NULL;
  uint64_t begin_wait_payload_storage = 0;
  iree_hal_semaphore_t* begin_signal_semaphore_storage = NULL;
  uint64_t begin_signal_payload_storage = 0;
  iree_hal_semaphore_list_t begin_wait_list = id4_cli_single_semaphore_list(
      &begin_wait_semaphore_storage, &begin_wait_payload_storage,
      prepare_semaphore, *inout_prepare_payload_value);
  iree_hal_semaphore_list_t begin_signal_list = id4_cli_single_semaphore_list(
      &begin_signal_semaphore_storage, &begin_signal_payload_storage,
      prepare_semaphore, ++*inout_prepare_payload_value);

  id4_ideogram4_generation_begin_options_t begin_options;
  memset(&begin_options, 0, sizeof(begin_options));
  begin_options.structure_size = sizeof(begin_options);
  begin_options.request = request;
  begin_options.tokenizer = tokenizer;
  begin_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  begin_options.wait_semaphore_list = begin_wait_list;
  begin_options.signal_semaphore_list = begin_signal_list;
  begin_options.diagnostics_sink = diagnostics_sink;

  id4_ideogram4_generation_execution_t* execution = NULL;
  iree_time_t phase_start_time_ns = iree_time_now();
  iree_status_t status = id4_ideogram4_session_begin_generation(
      session, bundle, &begin_options, &execution);
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_emit_timing(diagnostics_sink, IREE_SV("cli.begin_generation"),
                            IREE_SV("began phase-driven generation"),
                            phase_start_time_ns, iree_time_now());
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_prepare_issue_release_generation_phase(
        bundle, execution, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
        IREE_SV("cli.prepare_conditioning"), IREE_SV("cli.issue_conditioning"),
        IREE_SV("cli.wait_conditioning"), IREE_SV("cli.release_conditioning"),
        prepare_semaphore, *inout_prepare_payload_value, prepare_semaphore,
        inout_prepare_payload_value, completion_semaphore,
        inout_completion_payload_value, stage_issue_flags,
        parameter_load_prefetch_region_distance, diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_prepare_issue_release_generation_phase(
        bundle, execution, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
        IREE_SV("cli.prepare_denoise"), IREE_SV("cli.issue_denoise"),
        IREE_SV("cli.wait_denoise"), IREE_SV("cli.release_denoise"),
        completion_semaphore, *inout_completion_payload_value,
        prepare_semaphore, inout_prepare_payload_value, completion_semaphore,
        inout_completion_payload_value, stage_issue_flags,
        parameter_load_prefetch_region_distance, diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_prepare_issue_release_generation_phase(
        bundle, execution, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
        IREE_SV("cli.prepare_decode"), IREE_SV("cli.issue_decode"),
        IREE_SV("cli.wait_decode"), IREE_SV("cli.release_decode"),
        completion_semaphore, *inout_completion_payload_value,
        prepare_semaphore, inout_prepare_payload_value, completion_semaphore,
        inout_completion_payload_value, stage_issue_flags,
        parameter_load_prefetch_region_distance, diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

static iree_status_t id4_cli_create_loaded_session(
    id4_tooling_runtime_context_t* runtime_context,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator, id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  id4_ideogram4_session_create_options_t session_options;
  memset(&session_options, 0, sizeof(session_options));
  session_options.structure_size = sizeof(session_options);
  session_options.services =
      id4_tooling_runtime_context_stage_services(runtime_context);
  session_options.kernel_cache = runtime_context->kernel_cache;
  session_options.parameter_scopes.qwen = IREE_SV("qwen");
  session_options.parameter_scopes.vae = IREE_SV("vae");
  IREE_RETURN_IF_ERROR(id4_cli_parse_qwen_parameter_format(
      &session_options.qwen_parameter_format));
  IREE_RETURN_IF_ERROR(id4_cli_parse_dit_parameter_format(
      &session_options.dit_parameter_format));
  switch (session_options.dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      session_options.parameter_scopes.dit_conditioned = IREE_SV("dit_cond");
      session_options.parameter_scopes.dit_unconditioned =
          IREE_SV("dit_uncond");
      break;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3: {
      const iree_string_view_t dit_conditioned_fp8_scope =
          iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
      const iree_string_view_t dit_unconditioned_fp8_scope =
          iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
      session_options.parameter_scopes.dit_conditioned =
          dit_conditioned_fp8_scope;
      session_options.parameter_scopes.dit_conditioned_fp8 =
          dit_conditioned_fp8_scope;
      session_options.parameter_scopes.dit_unconditioned =
          dit_unconditioned_fp8_scope;
      session_options.parameter_scopes.dit_unconditioned_fp8 =
          dit_unconditioned_fp8_scope;
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid DiT parameter format %" PRIu32,
                              (uint32_t)session_options.dit_parameter_format);
  }
  session_options.vae_activation_format =
      ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
  IREE_RETURN_IF_ERROR(id4_ideogram4_session_create(
      &session_options, host_allocator, out_session));

  id4_ideogram4_session_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = diagnostics_sink;
  iree_status_t status =
      id4_ideogram4_session_load(*out_session, &load_options);
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_session_release(*out_session);
    *out_session = NULL;
  }
  return status;
}

static iree_status_t id4_cli_make_generation_plan_policy(
    id4_ideogram4_generation_plan_policy_t* out_policy) {
  IREE_ASSERT_ARGUMENT(out_policy);
  id4_ideogram4_generation_plan_policy_t policy;
  memset(&policy, 0, sizeof(policy));
  policy.structure_size = sizeof(policy);
  IREE_RETURN_IF_ERROR(
      id4_cli_parse_dit_activation_format(&policy.dit_activation_format));
  IREE_RETURN_IF_ERROR(id4_cli_parse_dit_weight_execution_format(
      &policy.dit_weight_execution_format));
  IREE_RETURN_IF_ERROR(id4_cli_parse_qwen_weight_execution_strategy(
      &policy.qwen_weight_execution_strategy));
  IREE_RETURN_IF_ERROR(id4_cli_parse_qwen_attention_implementation(
      &policy.qwen_attention_implementation));
  IREE_RETURN_IF_ERROR(id4_cli_parse_dit_attention_implementation(
      &policy.dit_attention_implementation));
  IREE_RETURN_IF_ERROR(id4_cli_parse_dit_feed_forward_implementation(
      &policy.dit_feed_forward_implementation));
  IREE_RETURN_IF_ERROR(id4_cli_parse_vae_tiling_config(&policy.vae_tiling));
  *out_policy = policy;
  return iree_ok_status();
}

static void id4_cli_release_parameter_providers(
    id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_release(providers->vae);
  iree_io_parameter_provider_release(providers->dit_unconditioned);
  iree_io_parameter_provider_release(providers->dit_conditioned);
  iree_io_parameter_provider_release(providers->qwen);
  memset(providers, 0, sizeof(*providers));
}

static iree_status_t id4_cli_create_parameter_providers(
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_parameter_providers_t* out_providers) {
  memset(out_providers, 0, sizeof(*out_providers));
  id4_ideogram4_dit_parameter_format_t dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_RETURN_IF_ERROR(
      id4_cli_parse_dit_parameter_format(&dit_parameter_format));

  iree_status_t status = iree_ok_status();
  if (dit_parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    id4_tooling_parameter_provider_request_t requests[] = {
        {
            // Qwen3-VL text encoder parameter scope.
            .scope = IREE_SV("qwen"),
            // Qwen provider output.
            .out_provider = &out_providers->qwen,
        },
        {
            // Conditioned DiT parameter scope.
            .scope = IREE_SV("dit_cond"),
            // Conditioned DiT provider output.
            .out_provider = &out_providers->dit_conditioned,
        },
        {
            // Unconditioned DiT parameter scope.
            .scope = IREE_SV("dit_uncond"),
            // Unconditioned DiT provider output.
            .out_provider = &out_providers->dit_unconditioned,
        },
        {
            // VAE parameter scope.
            .scope = IREE_SV("vae"),
            // VAE provider output.
            .out_provider = &out_providers->vae,
        },
    };
    status = id4_tooling_create_parameter_providers_from_flags(
        IREE_ARRAYSIZE(requests), requests, host_allocator);
  } else {
    const iree_string_view_t dit_conditioned_fp8_scope =
        iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    const iree_string_view_t dit_unconditioned_fp8_scope =
        iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
    id4_tooling_parameter_provider_request_t requests[] = {
        {
            // Qwen3-VL text encoder parameter scope.
            .scope = IREE_SV("qwen"),
            // Qwen provider output.
            .out_provider = &out_providers->qwen,
        },
        {
            // Conditioned DiT FP8 e4m3 parameter scope.
            .scope = dit_conditioned_fp8_scope,
            // Conditioned DiT FP8 e4m3 provider output.
            .out_provider = &out_providers->dit_conditioned,
        },
        {
            // Unconditioned DiT FP8 e4m3 parameter scope.
            .scope = dit_unconditioned_fp8_scope,
            // Unconditioned DiT FP8 e4m3 provider output.
            .out_provider = &out_providers->dit_unconditioned,
        },
        {
            // VAE parameter scope.
            .scope = IREE_SV("vae"),
            // VAE provider output.
            .out_provider = &out_providers->vae,
        },
    };
    status = id4_tooling_create_parameter_providers_from_flags(
        IREE_ARRAYSIZE(requests), requests, host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    id4_cli_release_parameter_providers(out_providers);
  }
  return status;
}

static iree_status_t id4_cli_run_generation_dry_run(
    iree_allocator_t host_allocator) {
  id4_ideogram4_request_t request;
  memset(&request, 0, sizeof(request));
  iree_tokenizer_t* tokenizer = NULL;
  id4_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  bool runtime_context_initialized = false;
  id4_ideogram4_session_t* session = NULL;
  id4_ideogram4_generation_plan_t* generation_plan = NULL;
  id4_tooling_diagnostics_file_sink_t diagnostics_file_sink;
  memset(&diagnostics_file_sink, 0, sizeof(diagnostics_file_sink));
  bool diagnostics_file_sink_initialized = false;
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  id4_cli_generation_diagnostic_taps_t diagnostic_taps;
  memset(&diagnostic_taps, 0, sizeof(diagnostic_taps));
  id4_cli_generation_issue_mode_t generation_issue_mode =
      ID4_CLI_GENERATION_ISSUE_MODE_PHASES;
  id4_cli_generation_residency_request_mode_t generation_residency_mode =
      ID4_CLI_GENERATION_RESIDENCY_REQUEST_ISSUE_PHASES;
  iree_host_size_t parameter_load_prefetch_region_distance = 0;
  id4_ideogram4_generation_resident_stage_mask_t
      region_per_dispatch_stage_mask =
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  id4_ideogram4_generation_resident_stage_mask_t requested_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;

  iree_status_t status = id4_cli_validate_dry_run_flags();
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_issue_mode(&generation_issue_mode);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_residency_request_mode(
        &generation_residency_mode);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_non_negative_host_size_flag(
        FLAG_parameter_load_prefetch_region_distance,
        IREE_SV("--parameter_load_prefetch_region_distance"),
        &parameter_load_prefetch_region_distance);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_parse_generation_resident_stage_mask(&requested_stage_mask);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_request(host_allocator, &request);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_stage_mask(
        iree_make_cstring_view(FLAG_diagnostic_region_per_dispatch_stages),
        IREE_SV("--diagnostic_region_per_dispatch_stages"),
        &region_per_dispatch_stage_mask);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_diagnostic_taps(host_allocator, &diagnostic_taps);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_initialize_diagnostics_sink(
        host_allocator, &diagnostics_file_sink, &diagnostics_sink,
        &diagnostics_file_sink_initialized);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_load_tokenizer(iree_make_cstring_view(FLAG_tokenizer),
                                    host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    id4_tooling_runtime_context_options_t runtime_options;
    memset(&runtime_options, 0, sizeof(runtime_options));
    runtime_options.structure_size = sizeof(runtime_options);
    runtime_options.executable_cache_identifier = IREE_SV("id4");
    status = id4_tooling_runtime_context_initialize_from_flags(
        &runtime_options, host_allocator, &runtime_context);
    runtime_context_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_loaded_session(&runtime_context, &diagnostics_sink,
                                           host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    status = id4_cli_make_generation_plan_policy(&plan_options.policy);
    plan_options.region_per_dispatch_stage_mask =
        region_per_dispatch_stage_mask;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.stage_diagnostic_tap_list_count = diagnostic_taps.list_count;
    plan_options.stage_diagnostic_tap_lists = diagnostic_taps.lists;
    plan_options.diagnostics_sink = &diagnostics_sink;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_session_plan_generation(session, &plan_options,
                                                     &generation_plan);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_write_generation_plan(iree_make_cstring_view(FLAG_dump_plan),
                                      generation_plan, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    status = id4_ideogram4_generation_plan_summary(generation_plan, &summary);
    if (iree_status_is_ok(status)) {
      fprintf(stdout,
              "Ideogram 4 generation plan: qwen_tokens=%" PRIu32
              " qwen_capacity=%" PRIu32 " image_tokens=%" PRIu32
              " dit_cond_tokens=%" PRIu32 " dit_cond_capacity=%" PRIu32
              " dit_uncond_tokens=%" PRIu32 " dit_uncond_capacity=%" PRIu32
              " latent=%" PRIu64 "x%" PRIu64 "x%" PRIu64 " image=%" PRIu64
              "x%" PRIu64 " steps=%" PRIu32 "\n",
              summary.qwen_token_count, summary.qwen_token_capacity,
              summary.image_token_count, summary.conditioned_dit_token_count,
              summary.conditioned_dit_token_capacity,
              summary.unconditioned_dit_token_count,
              summary.unconditioned_dit_token_capacity,
              summary.diffusion_latent_shape.dims[0],
              summary.diffusion_latent_shape.dims[1],
              summary.diffusion_latent_shape.dims[2],
              summary.decoded_image_shape.dims[0],
              summary.decoded_image_shape.dims[1], summary.denoise_step_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_print_generation_resource_statistics(
        generation_plan, generation_issue_mode, generation_residency_mode,
        requested_stage_mask, parameter_load_prefetch_region_distance);
  }

  id4_ideogram4_generation_plan_release(generation_plan);
  id4_ideogram4_session_release(session);
  if (runtime_context_initialized) {
    id4_tooling_runtime_context_deinitialize(&runtime_context);
  }
  if (diagnostics_file_sink_initialized) {
    status = iree_status_join(
        status,
        id4_tooling_diagnostics_file_sink_deinitialize(&diagnostics_file_sink));
  }
  id4_cli_generation_diagnostic_taps_deinitialize(&diagnostic_taps,
                                                  host_allocator);
  iree_tokenizer_free(tokenizer);
  id4_ideogram4_request_deinitialize(&request, host_allocator);
  return status;
}

static iree_status_t id4_cli_write_decoded_image(
    iree_string_view_t output_path,
    id4_ideogram4_generation_plan_summary_t summary,
    iree_const_byte_span_t decoded_image, iree_allocator_t host_allocator) {
  id4_tooling_validate_f32_rgb_image_contents_options_t validation_options;
  memset(&validation_options, 0, sizeof(validation_options));
  validation_options.structure_size = sizeof(validation_options);
  validation_options.shape =
      id4_cli_convert_program_shape(summary.decoded_image_shape);
  validation_options.pixels = decoded_image;
  validation_options.normalization =
      ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
  IREE_RETURN_IF_ERROR(
      id4_tooling_validate_f32_rgb_image_contents(&validation_options));

  id4_tooling_write_f32_rgb_ppm_options_t image_options;
  memset(&image_options, 0, sizeof(image_options));
  image_options.structure_size = sizeof(image_options);
  image_options.path = output_path;
  image_options.shape =
      id4_cli_convert_program_shape(summary.decoded_image_shape);
  image_options.pixels = decoded_image;
  image_options.normalization =
      ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
  image_options.host_allocator = host_allocator;
  return id4_tooling_write_f32_rgb_ppm(&image_options);
}

static iree_status_t id4_cli_run_generation(iree_allocator_t host_allocator) {
  id4_ideogram4_request_t request;
  memset(&request, 0, sizeof(request));
  iree_tokenizer_t* tokenizer = NULL;
  id4_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  bool runtime_context_initialized = false;
  id4_pipeline_kernel_library_t* kernel_library = NULL;
  id4_ideogram4_generation_parameter_providers_t parameter_providers;
  memset(&parameter_providers, 0, sizeof(parameter_providers));
  id4_ideogram4_session_t* session = NULL;
  id4_ideogram4_generation_plan_t* generation_plan = NULL;
  id4_ideogram4_generation_bundle_t* generation_bundle = NULL;
  iree_hal_semaphore_t* prepare_semaphore = NULL;
  iree_hal_semaphore_t* completion_semaphore = NULL;
  id4_ideogram4_generation_execution_t* execution = NULL;
  bool generation_was_prepared = false;
  bool generation_was_issued = false;
  id4_tooling_host_bytes_t decoded_image_bytes;
  memset(&decoded_image_bytes, 0, sizeof(decoded_image_bytes));
  id4_tooling_host_bytes_t conditioned_velocity_bytes;
  memset(&conditioned_velocity_bytes, 0, sizeof(conditioned_velocity_bytes));
  id4_tooling_host_bytes_t unconditioned_velocity_bytes;
  memset(&unconditioned_velocity_bytes, 0,
         sizeof(unconditioned_velocity_bytes));
  id4_tooling_host_bytes_t denoised_latent_bytes;
  memset(&denoised_latent_bytes, 0, sizeof(denoised_latent_bytes));
  id4_tooling_host_bytes_t final_latent_bytes;
  memset(&final_latent_bytes, 0, sizeof(final_latent_bytes));
  id4_cli_generation_diagnostic_taps_t diagnostic_taps;
  memset(&diagnostic_taps, 0, sizeof(diagnostic_taps));
  id4_tooling_host_bytes_t* diagnostic_tap_bytes = NULL;
  id4_cli_result_tensor_summary_t* result_summary_tensors = NULL;
  id4_tooling_diagnostics_file_sink_t diagnostics_file_sink;
  memset(&diagnostics_file_sink, 0, sizeof(diagnostics_file_sink));
  bool diagnostics_file_sink_initialized = false;
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  iree_hal_profile_statistics_sink_t* profile_statistics_sink = NULL;
  bool profile_started = false;
  id4_cli_generation_issue_mode_t generation_issue_mode =
      ID4_CLI_GENERATION_ISSUE_MODE_PHASES;
  id4_cli_generation_residency_request_mode_t generation_residency_mode =
      ID4_CLI_GENERATION_RESIDENCY_REQUEST_ISSUE_PHASES;
  iree_host_size_t parameter_load_prefetch_region_distance = 0;
  id4_ideogram4_generation_resident_stage_mask_t
      region_per_dispatch_stage_mask =
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  id4_pipeline_stage_issue_flags_t stage_issue_flags = 0;
  const iree_time_t generation_start_time_ns = iree_time_now();

  iree_status_t status = id4_cli_validate_execution_flags();
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_issue_mode(&generation_issue_mode);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_residency_request_mode(
        &generation_residency_mode);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_non_negative_host_size_flag(
        FLAG_parameter_load_prefetch_region_distance,
        IREE_SV("--parameter_load_prefetch_region_distance"),
        &parameter_load_prefetch_region_distance);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_request(host_allocator, &request);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_generation_stage_mask(
        iree_make_cstring_view(FLAG_diagnostic_region_per_dispatch_stages),
        IREE_SV("--diagnostic_region_per_dispatch_stages"),
        &region_per_dispatch_stage_mask);
  }
  if (FLAG_diagnostic_wait_after_each_region) {
    stage_issue_flags |= ID4_PIPELINE_STAGE_ISSUE_FLAG_WAIT_AFTER_EACH_REGION;
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_diagnostic_taps(host_allocator, &diagnostic_taps);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_initialize_diagnostics_sink(
        host_allocator, &diagnostics_file_sink, &diagnostics_sink,
        &diagnostics_file_sink_initialized);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_load_tokenizer(iree_make_cstring_view(FLAG_tokenizer),
                                    host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    id4_tooling_runtime_context_options_t runtime_options;
    memset(&runtime_options, 0, sizeof(runtime_options));
    runtime_options.structure_size = sizeof(runtime_options);
    runtime_options.executable_cache_identifier = IREE_SV("id4");
    status = id4_tooling_runtime_context_initialize_from_flags(
        &runtime_options, host_allocator, &runtime_context);
    runtime_context_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_create_embedded_kernel_library(host_allocator,
                                                        &kernel_library);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_loaded_session(&runtime_context, &diagnostics_sink,
                                           host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_parameter_providers(host_allocator,
                                                &parameter_providers);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &prepare_semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &completion_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_emit_timing(&diagnostics_sink, IREE_SV("cli.setup"),
                            IREE_SV("initialized CLI generation resources"),
                            generation_start_time_ns, iree_time_now());
  }
  if (iree_status_is_ok(status)) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    id4_ideogram4_generation_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    status = id4_cli_make_generation_plan_policy(&plan_options.policy);
    plan_options.region_per_dispatch_stage_mask =
        region_per_dispatch_stage_mask;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.stage_diagnostic_tap_list_count = diagnostic_taps.list_count;
    plan_options.stage_diagnostic_tap_lists = diagnostic_taps.lists;
    plan_options.diagnostics_sink = &diagnostics_sink;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_session_plan_generation(session, &plan_options,
                                                     &generation_plan);
    }
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(
          &diagnostics_sink, IREE_SV("cli.plan_generation"),
          IREE_SV("planned generation"), phase_start_time_ns, iree_time_now());
    }
  }
  id4_ideogram4_generation_plan_summary_t summary;
  memset(&summary, 0, sizeof(summary));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_plan_summary(generation_plan, &summary);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_write_generation_plan(iree_make_cstring_view(FLAG_dump_plan),
                                      generation_plan, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = id4_cli_begin_profile_statistics(
        device, host_allocator, &profile_statistics_sink, &profile_started);
  }
  iree_hal_semaphore_t* prepare_semaphore_storage = NULL;
  uint64_t prepare_signal_payload_storage = 1;
  uint64_t prepare_payload_storage = prepare_signal_payload_storage;
  iree_hal_semaphore_list_t prepare_signal_list =
      iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    prepare_signal_list = id4_cli_single_semaphore_list(
        &prepare_semaphore_storage, &prepare_signal_payload_storage,
        prepare_semaphore, prepare_signal_payload_storage);
    id4_ideogram4_generation_prepare_options_t prepare_options;
    memset(&prepare_options, 0, sizeof(prepare_options));
    prepare_options.structure_size = sizeof(prepare_options);
    prepare_options.parameter_providers = parameter_providers;
    prepare_options.kernel_library = kernel_library;
    id4_ideogram4_generation_resident_stage_mask_t requested_stage_mask =
        ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
    if (iree_status_is_ok(status)) {
      status =
          id4_cli_parse_generation_resident_stage_mask(&requested_stage_mask);
    }
    if (iree_status_is_ok(status)) {
      status = id4_cli_resolve_generation_residency(
          generation_plan, generation_issue_mode, generation_residency_mode,
          requested_stage_mask, parameter_load_prefetch_region_distance,
          &prepare_options.residency_mode, &prepare_options.resident_stage_mask,
          prepare_options.phase_stage_masks);
    }
    prepare_options.command_buffer_mode =
        id4_cli_generation_command_buffer_mode(
            runtime_context.command_buffer_mode);
    prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    prepare_options.signal_semaphore_list = prepare_signal_list;
    prepare_options.diagnostics_sink = &diagnostics_sink;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_session_prepare_generation(
          session, generation_plan, &prepare_options, &generation_bundle);
    }
    generation_was_prepared = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(&diagnostics_sink,
                                   IREE_SV("cli.prepare_generation"),
                                   IREE_SV("prepared generation bundle"),
                                   phase_start_time_ns, iree_time_now());
    }
  }
  iree_hal_semaphore_t* completion_semaphore_storage = NULL;
  uint64_t completion_payload_storage = 0;
  iree_hal_semaphore_list_t completion_list = iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    completion_list = id4_cli_single_semaphore_list(
        &completion_semaphore_storage, &completion_payload_storage,
        completion_semaphore, completion_payload_storage);
    switch (generation_issue_mode) {
      case ID4_CLI_GENERATION_ISSUE_MODE_FULL: {
        ++completion_payload_storage;
        status = id4_cli_issue_generation_full(
            session, generation_bundle, &request, tokenizer,
            id4_cli_generation_issue_policy(generation_issue_mode),
            prepare_signal_list, completion_list, stage_issue_flags,
            parameter_load_prefetch_region_distance, &diagnostics_sink,
            &execution);
        break;
      }
      case ID4_CLI_GENERATION_ISSUE_MODE_STAGE_SERIAL: {
        ++completion_payload_storage;
        status = id4_cli_issue_generation_full(
            session, generation_bundle, &request, tokenizer,
            id4_cli_generation_issue_policy(generation_issue_mode),
            prepare_signal_list, completion_list, stage_issue_flags,
            parameter_load_prefetch_region_distance, &diagnostics_sink,
            &execution);
        break;
      }
      case ID4_CLI_GENERATION_ISSUE_MODE_PHASES: {
        status = id4_cli_issue_generation_phases(
            session, generation_bundle, &request, tokenizer, prepare_semaphore,
            &prepare_payload_storage, completion_semaphore,
            &completion_payload_storage, stage_issue_flags,
            parameter_load_prefetch_region_distance, &diagnostics_sink,
            &execution);
        break;
      }
      default: {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "generation issue mode is invalid");
        break;
      }
    }
    generation_was_issued = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(&diagnostics_sink,
                                   IREE_SV("cli.issue_generation"),
                                   IREE_SV("issued generation work"),
                                   phase_start_time_ns, iree_time_now());
    }
  }
  if (generation_was_issued) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    iree_status_t wait_status = iree_hal_semaphore_list_wait(
        completion_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    if (!iree_status_is_ok(wait_status) && execution) {
      wait_status = iree_status_join(
          wait_status, id4_ideogram4_generation_execution_check_failures(
                           execution, &diagnostics_sink));
    }
    status = iree_status_join(status, wait_status);
    if (iree_status_is_ok(status)) {
      status =
          id4_cli_emit_timing(&diagnostics_sink, IREE_SV("cli.wait_completion"),
                              IREE_SV("waited for generation completion"),
                              phase_start_time_ns, iree_time_now());
    }
  } else if (generation_was_prepared) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(prepare_signal_list,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    if (iree_status_is_ok(status)) {
      status =
          id4_cli_emit_timing(&diagnostics_sink, IREE_SV("cli.wait_prepare"),
                              IREE_SV("waited for prepare completion"),
                              phase_start_time_ns, iree_time_now());
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    id4_ideogram4_generation_result_t result;
    memset(&result, 0, sizeof(result));
    status = id4_ideogram4_generation_execution_result(execution, &result);
    id4_tooling_readback_buffer_binding_options_t readback_options;
    memset(&readback_options, 0, sizeof(readback_options));
    readback_options.structure_size = sizeof(readback_options);
    readback_options.device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    readback_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    readback_options.binding = result.decoded_image_binding;
    readback_options.wait_semaphore_list = completion_list;
    readback_options.host_allocator = host_allocator;
    const bool capture_result_tensors = strlen(FLAG_dump_result_summary) != 0 ||
                                        strlen(FLAG_dump_result_tensors) != 0;
    if (iree_status_is_ok(status)) {
      status = id4_tooling_readback_buffer_binding(&readback_options,
                                                   &decoded_image_bytes);
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      readback_options.binding = result.conditioned_velocity_binding;
      status = id4_tooling_readback_buffer_binding(&readback_options,
                                                   &conditioned_velocity_bytes);
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      readback_options.binding = result.unconditioned_velocity_binding;
      status = id4_tooling_readback_buffer_binding(
          &readback_options, &unconditioned_velocity_bytes);
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      readback_options.binding = result.denoised_latent_binding;
      status = id4_tooling_readback_buffer_binding(&readback_options,
                                                   &denoised_latent_bytes);
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      readback_options.binding = result.final_latent_binding;
      status = id4_tooling_readback_buffer_binding(&readback_options,
                                                   &final_latent_bytes);
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      if (diagnostic_taps.request_count != 0) {
        status = iree_allocator_malloc_array(
            host_allocator, diagnostic_taps.request_count,
            sizeof(diagnostic_tap_bytes[0]), (void**)&diagnostic_tap_bytes);
        if (iree_status_is_ok(status)) {
          memset(
              diagnostic_tap_bytes, 0,
              diagnostic_taps.request_count * sizeof(diagnostic_tap_bytes[0]));
        }
      }
    }
    for (iree_host_size_t i = 0;
         capture_result_tensors && i < diagnostic_taps.request_count &&
         iree_status_is_ok(status);
         ++i) {
      const id4_pipeline_tensor_layout_t* tap_layout = NULL;
      iree_hal_buffer_binding_t tap_binding;
      status = id4_ideogram4_generation_execution_find_diagnostic_tap(
          execution, diagnostic_taps.requests[i].stage_key,
          diagnostic_taps.requests[i].tap_name, &tap_layout, &tap_binding);
      if (iree_status_is_ok(status)) {
        readback_options.binding = tap_binding;
        status = id4_tooling_readback_buffer_binding(&readback_options,
                                                     &diagnostic_tap_bytes[i]);
      }
    }
    if (iree_status_is_ok(status) && capture_result_tensors) {
      id4_pipeline_tensor_shape_t latent_shape =
          id4_cli_convert_program_shape(summary.diffusion_latent_shape);
      id4_pipeline_tensor_shape_t image_shape =
          id4_cli_convert_program_shape(summary.decoded_image_shape);
      iree_host_size_t result_summary_tensor_count = 0;
      if (!iree_host_size_checked_add(5, diagnostic_taps.request_count,
                                      &result_summary_tensor_count)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "result summary tensor count overflow");
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc_array(
            host_allocator, result_summary_tensor_count,
            sizeof(result_summary_tensors[0]), (void**)&result_summary_tensors);
      }
      if (iree_status_is_ok(status)) {
        result_summary_tensors[0] = (id4_cli_result_tensor_summary_t){
            .key = IREE_SV("conditioned_velocity"),
            .layout =
                {
                    .name = IREE_SV("conditioned_velocity"),
                    .dtype = ID4_PIPELINE_TENSOR_DTYPE_F32,
                    .shape = latent_shape,
                    .byte_length = conditioned_velocity_bytes.length,
                    .alignment = 0,
                },
            .bytes =
                iree_make_const_byte_span(conditioned_velocity_bytes.data,
                                          conditioned_velocity_bytes.length),
        };
        result_summary_tensors[1] = (id4_cli_result_tensor_summary_t){
            .key = IREE_SV("unconditioned_velocity"),
            .layout =
                {
                    .name = IREE_SV("unconditioned_velocity"),
                    .dtype = ID4_PIPELINE_TENSOR_DTYPE_F32,
                    .shape = latent_shape,
                    .byte_length = unconditioned_velocity_bytes.length,
                    .alignment = 0,
                },
            .bytes =
                iree_make_const_byte_span(unconditioned_velocity_bytes.data,
                                          unconditioned_velocity_bytes.length),
        };
        result_summary_tensors[2] = (id4_cli_result_tensor_summary_t){
            .key = IREE_SV("denoised_latent"),
            .layout =
                {
                    .name = IREE_SV("denoised_latent"),
                    .dtype = ID4_PIPELINE_TENSOR_DTYPE_F32,
                    .shape = latent_shape,
                    .byte_length = denoised_latent_bytes.length,
                    .alignment = 0,
                },
            .bytes = iree_make_const_byte_span(denoised_latent_bytes.data,
                                               denoised_latent_bytes.length),
        };
        result_summary_tensors[3] = (id4_cli_result_tensor_summary_t){
            .key = IREE_SV("final_latent"),
            .layout =
                {
                    .name = IREE_SV("final_latent"),
                    .dtype = ID4_PIPELINE_TENSOR_DTYPE_F32,
                    .shape = latent_shape,
                    .byte_length = final_latent_bytes.length,
                    .alignment = 0,
                },
            .bytes = iree_make_const_byte_span(final_latent_bytes.data,
                                               final_latent_bytes.length),
        };
        result_summary_tensors[4] = (id4_cli_result_tensor_summary_t){
            .key = IREE_SV("decoded_image"),
            .layout =
                {
                    .name = IREE_SV("decoded_image"),
                    .dtype = ID4_PIPELINE_TENSOR_DTYPE_F32,
                    .shape = image_shape,
                    .byte_length = decoded_image_bytes.length,
                    .alignment = 0,
                },
            .bytes = iree_make_const_byte_span(decoded_image_bytes.data,
                                               decoded_image_bytes.length),
        };
      }
      for (iree_host_size_t i = 0;
           i < diagnostic_taps.request_count && iree_status_is_ok(status);
           ++i) {
        const id4_pipeline_tensor_layout_t* tap_layout = NULL;
        iree_hal_buffer_binding_t tap_binding;
        status = id4_ideogram4_generation_execution_find_diagnostic_tap(
            execution, diagnostic_taps.requests[i].stage_key,
            diagnostic_taps.requests[i].tap_name, &tap_layout, &tap_binding);
        if (iree_status_is_ok(status)) {
          result_summary_tensors[5 + i] = (id4_cli_result_tensor_summary_t){
              .key = diagnostic_taps.requests[i].summary_key,
              .layout = *tap_layout,
              .bytes = iree_make_const_byte_span(
                  diagnostic_tap_bytes[i].data, diagnostic_tap_bytes[i].length),
          };
        }
      }
      if (iree_status_is_ok(status)) {
        status = id4_cli_write_result_summary(
            iree_make_cstring_view(FLAG_dump_result_summary),
            result_summary_tensor_count, result_summary_tensors,
            host_allocator);
      }
      if (iree_status_is_ok(status)) {
        status = id4_cli_write_result_tensors(
            iree_make_cstring_view(FLAG_dump_result_tensors),
            result_summary_tensor_count, result_summary_tensors,
            host_allocator);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(&diagnostics_sink, IREE_SV("cli.readback"),
                                   IREE_SV("read back decoded image"),
                                   phase_start_time_ns, iree_time_now());
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_time_t phase_start_time_ns = iree_time_now();
    status = id4_cli_write_decoded_image(
        iree_make_cstring_view(FLAG_output), summary,
        iree_make_const_byte_span(decoded_image_bytes.data,
                                  decoded_image_bytes.length),
        host_allocator);
    if (iree_status_is_ok(status)) {
      status = id4_cli_emit_timing(
          &diagnostics_sink, IREE_SV("cli.write_output"),
          IREE_SV("wrote decoded image"), phase_start_time_ns, iree_time_now());
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_emit_timing(&diagnostics_sink, IREE_SV("cli.total"),
                                 IREE_SV("completed generation command"),
                                 generation_start_time_ns, iree_time_now());
  }
  if (profile_started) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_status_join(status, id4_cli_end_profile_statistics(
                                          device, profile_statistics_sink));
    profile_started = false;
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "Ideogram 4 generation complete: qwen_tokens=%" PRIu32
            " image_tokens=%" PRIu32 " image=%" PRIu64 "x%" PRIu64
            " output=%s\n",
            summary.qwen_token_count, summary.image_token_count,
            summary.decoded_image_shape.dims[0],
            summary.decoded_image_shape.dims[1], FLAG_output);
  }

  for (iree_host_size_t i = 0; i < diagnostic_taps.request_count; ++i) {
    if (diagnostic_tap_bytes) {
      id4_tooling_host_bytes_deinitialize(&diagnostic_tap_bytes[i],
                                          host_allocator);
    }
  }
  iree_allocator_free(host_allocator, result_summary_tensors);
  iree_allocator_free(host_allocator, diagnostic_tap_bytes);
  id4_tooling_host_bytes_deinitialize(&final_latent_bytes, host_allocator);
  id4_tooling_host_bytes_deinitialize(&denoised_latent_bytes, host_allocator);
  id4_tooling_host_bytes_deinitialize(&unconditioned_velocity_bytes,
                                      host_allocator);
  id4_tooling_host_bytes_deinitialize(&conditioned_velocity_bytes,
                                      host_allocator);
  id4_tooling_host_bytes_deinitialize(&decoded_image_bytes, host_allocator);
  id4_ideogram4_generation_execution_release(execution);
  id4_ideogram4_generation_bundle_release(generation_bundle);
  id4_ideogram4_generation_plan_release(generation_plan);
  iree_hal_semaphore_release(prepare_semaphore);
  iree_hal_semaphore_release(completion_semaphore);
  iree_hal_profile_statistics_sink_release(profile_statistics_sink);
  id4_ideogram4_session_release(session);
  id4_cli_release_parameter_providers(&parameter_providers);
  id4_pipeline_kernel_library_release(kernel_library);
  if (runtime_context_initialized) {
    id4_tooling_runtime_context_deinitialize(&runtime_context);
  }
  if (diagnostics_file_sink_initialized) {
    status = iree_status_join(
        status,
        id4_tooling_diagnostics_file_sink_deinitialize(&diagnostics_file_sink));
  }
  id4_cli_generation_diagnostic_taps_deinitialize(&diagnostic_taps,
                                                  host_allocator);
  iree_tokenizer_free(tokenizer);
  id4_ideogram4_request_deinitialize(&request, host_allocator);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "id4",
      "Experimental Ideogram 4 HAL pipeline runner.\n"
      "\n"
      "Loads a prompt request, tokenizes it, and either plans or runs a "
      "full generation.\n"
      "Pass exactly one of --prompt=..., --prompt_json=..., or "
      "--prompt_json_file=....\n"
      "Plain --prompt requests also require --generation_* flags.\n"
      "Model parameters are loaded with standard --parameters= flags.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  int exit_code = EXIT_SUCCESS;
  if (argc > 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "no positional arguments expected");
  } else if (FLAG_dry_run) {
    status = id4_cli_run_generation_dry_run(host_allocator);
  } else {
    status = id4_cli_run_generation(host_allocator);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
