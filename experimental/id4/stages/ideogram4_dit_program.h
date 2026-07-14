// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_
#define EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_

#include <stdint.h>

#include "experimental/id4/pipeline/program.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current image-token prelude flattened token-count limit.
#define ID4_IDEOGRAM4_DIT_PRELUDE_IMAGE_MAX_TOKEN_COUNT 65536u

// Current combined text+image prelude flattened token-count limit.
#define ID4_IDEOGRAM4_DIT_PRELUDE_MAX_TOKEN_COUNT 131072u

// BF16 packed activation token-capacity alignment used by transformer blocks.
#define ID4_IDEOGRAM4_DIT_BF16_TOKEN_CAPACITY_BLOCK 128u

// Maximum formatted DiT parameter key or tensor diagnostic name byte length.
#define ID4_IDEOGRAM4_DIT_PROGRAM_FORMAT_BUFFER_CAPACITY 192

// Conditioning path selected for an Ideogram4 DiT forward request.
typedef enum id4_ideogram4_dit_conditioning_mode_e {
  // Invalid conditioning mode.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_INVALID = 0,
  // Request uses only image tokens and the unconditioned DiT weights.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED = 1,
  // Request imports Qwen condition tokens and runs the conditioned prelude.
  ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED = 2,
} id4_ideogram4_dit_conditioning_mode_t;

// Activation storage format selected while authoring DiT intermediates.
typedef enum id4_ideogram4_dit_activation_format_e {
  // Invalid activation storage format.
  ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_INVALID = 0,
  // Reference format matching tensor taps: F32 channel-major tensors.
  ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL = 1,
  // Production format for linear inputs: BF16 token-major tensors.
  ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT = 2,
} id4_ideogram4_dit_activation_format_t;

// Attention implementation selected while authoring DiT transformer blocks.
typedef enum id4_ideogram4_dit_attention_implementation_e {
  // Invalid attention implementation.
  ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_INVALID = 0,
  // Streaming scalar attention that does not materialize scores.
  ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING = 1,
  // BF16 WMMA attention that materializes score and probability tensors.
  ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA = 2,
  // BF16 WMMA attention using reusable query-block score/probability scratch.
  ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA = 3,
  // BF16 WMMA attention selecting bounded normalized scratch for small token
  // counts and an online recurrence for larger token counts.
  ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA = 4,
} id4_ideogram4_dit_attention_implementation_t;

// Feed-forward implementation selected while authoring DiT transformer blocks.
typedef enum id4_ideogram4_dit_feed_forward_implementation_e {
  // Invalid feed-forward implementation.
  ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_INVALID = 0,
  // Implementation may fuse projection, activation, and product work.
  ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT = 1,
  // Implementation preserves PyTorch W1, W3, product, and W2 boundaries.
  ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY = 2,
} id4_ideogram4_dit_feed_forward_implementation_t;

// Physical storage format selected for a DiT parameter source.
typedef enum id4_ideogram4_dit_parameter_storage_e {
  // Invalid parameter storage format.
  ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_INVALID = 0,
  // BF16 tensor stored directly in the selected parameter source.
  ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_BF16 = 1,
  // FP8 e4m3 weight tensor with a sibling F32 row-scale tensor.
  ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED = 2,
} id4_ideogram4_dit_parameter_storage_t;

// Execution storage strategy selected for DiT linear weights.
typedef enum id4_ideogram4_dit_weight_execution_format_e {
  // Invalid weight execution format.
  ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_INVALID = 0,
  // Prepare FP8 sources into persistent BF16 execution tensors.
  ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT = 1,
  // Prepare FP8 sources into persistent compact-RHS FP8 execution tensors and
  // bind row scales directly to compute kernels.
  ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS = 2,
  // Prepare most FP8 sources into persistent compact-RHS FP8 execution tensors,
  // but prepare transformer feed-forward weights into persistent BF16 tensors.
  ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS_FEED_FORWARD_BF16_RESIDENT =
      3,
  // Keep FP8 source weights resident and stream BF16 compact RHS tiles through
  // transient program storage before each linear consumer.
  ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_STREAMING_COMPACT_RHS = 4,
} id4_ideogram4_dit_weight_execution_format_t;

// Parses a DiT linear weight execution format name.
iree_status_t id4_ideogram4_dit_weight_execution_format_parse(
    iree_string_view_t value,
    id4_ideogram4_dit_weight_execution_format_t* out_format);

// Returns the stable DiT linear weight execution format name.
iree_string_view_t id4_ideogram4_dit_weight_execution_format_name(
    id4_ideogram4_dit_weight_execution_format_t format);

// Ideogram4 DiT model dimensions used when authoring the forward program.
typedef struct id4_ideogram4_dit_model_config_t {
  // Number of transformer blocks in the DiT.
  uint32_t layer_count;
  // Channel count of each VAE latent image token.
  uint32_t input_channel_count;
  // Transformer hidden-state channel count.
  uint32_t hidden_size;
  // Feed-forward intermediate channel count.
  uint32_t intermediate_size;
  // Transformer attention head count.
  uint32_t attention_head_count;
  // AdaLN conditioning vector channel count.
  uint32_t adaln_size;
  // Qwen condition feature channel count consumed by llm_cond_norm.
  uint32_t llm_feature_count;
  // Number of image-indicator embedding rows.
  uint32_t image_indicator_count;
} id4_ideogram4_dit_model_config_t;

// One adapter's contiguous rank segment within a composed DiT LoRA target.
typedef struct id4_ideogram4_dit_lora_segment_t {
  // Provider scope containing this segment's down and up parameters.
  iree_string_view_t source_scope;
  // Adapter ordinal whose issue-time strength controls this segment.
  iree_host_size_t adapter_ordinal;
  // First rank position assigned to this segment in the composed projection.
  uint32_t rank_offset;
  // Number of rank positions assigned to this segment.
  uint32_t rank;
  // Provider parameter key for the BF16 down-projection matrix [rank, input].
  iree_string_view_t down_parameter_key;
  // Provider parameter key for the BF16 up-projection matrix [output, rank].
  iree_string_view_t up_parameter_key;
} id4_ideogram4_dit_lora_segment_t;

// Composed low-rank update for one conditioned-DiT linear parameter.
typedef struct id4_ideogram4_dit_lora_target_t {
  // Canonical base parameter key patched by this update.
  iree_string_view_t base_parameter_key;
  // Input feature count consumed by every segment's down projection.
  uint32_t input_size;
  // Output feature count produced by every segment's up projection.
  uint32_t output_size;
  // Sum of segment ranks in adapter order.
  uint32_t total_rank;
  // Number of entries in |segments|.
  iree_host_size_t segment_count;
  // Contiguous segments ordered by adapter ordinal.
  const id4_ideogram4_dit_lora_segment_t* segments;
} id4_ideogram4_dit_lora_target_t;

// Borrowed static LoRA topology used while authoring conditioned DiT.
typedef struct id4_ideogram4_dit_lora_topology_t {
  // Number of ordered adapters represented by issue-time strengths.
  iree_host_size_t adapter_count;
  // Number of unique patched base parameters in |targets|.
  iree_host_size_t target_count;
  // Composed targets with storage owned by the topology provider.
  const id4_ideogram4_dit_lora_target_t* targets;
} id4_ideogram4_dit_lora_topology_t;

// Dynamic request dimensions used when authoring the DiT forward program.
typedef struct id4_ideogram4_dit_request_config_t {
  // Latent tensor shape supplied by the sampler.
  id4_pipeline_program_shape_t latent_shape;
  // Conditioning path for this DiT request.
  id4_ideogram4_dit_conditioning_mode_t conditioning_mode;
  // Number of imported Qwen condition token positions.
  uint32_t text_token_count;
} id4_ideogram4_dit_request_config_t;

// Calculates the flattened image-token count for an Ideogram 4 latent tensor.
iree_status_t id4_ideogram4_dit_program_image_token_count(
    id4_ideogram4_dit_model_config_t model,
    id4_pipeline_program_shape_t latent_shape, uint32_t* out_token_count);

// Calculates the BF16 packed token capacity used by DiT transformer blocks.
iree_status_t id4_ideogram4_dit_program_calculate_bf16_token_capacity(
    uint32_t total_token_count, uint32_t* out_token_capacity);

// Exact-source rule for one logical DiT parameter key.
typedef struct id4_ideogram4_dit_parameter_source_rule_t {
  // Logical parameter key matched exactly, such as
  // layers.0.attention.qkv.weight.
  iree_string_view_t key;
  // Provider source scope containing this parameter and any sibling metadata.
  iree_string_view_t source_scope;
  // Physical storage format expected from the selected source scope.
  id4_ideogram4_dit_parameter_storage_t storage;
} id4_ideogram4_dit_parameter_source_rule_t;

// Resolved parameter source policy used while authoring a DiT program.
typedef struct id4_ideogram4_dit_parameter_sources_t {
  // Provider source scope used for parameters without an exact rule.
  iree_string_view_t default_scope;
  // Number of exact parameter source rules.
  iree_host_size_t rule_count;
  // Exact parameter source rules borrowed for the authoring call.
  const id4_ideogram4_dit_parameter_source_rule_t* rules;
} id4_ideogram4_dit_parameter_sources_t;

// Options for authoring an Ideogram4 DiT forward semantic program.
typedef struct id4_ideogram4_dit_program_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parameter source policy used when loading DiT parameters.
  id4_ideogram4_dit_parameter_sources_t parameter_sources;
  // Static model dimensions.
  id4_ideogram4_dit_model_config_t model;
  // Dynamic request dimensions.
  id4_ideogram4_dit_request_config_t request;
  // Activation storage format for internal linear-input producers.
  id4_ideogram4_dit_activation_format_t activation_format;
  // Execution storage strategy selected for linear weights.
  id4_ideogram4_dit_weight_execution_format_t weight_execution_format;
  // Attention implementation selected for transformer blocks.
  id4_ideogram4_dit_attention_implementation_t attention_implementation;
  // Feed-forward implementation selected for transformer blocks.
  id4_ideogram4_dit_feed_forward_implementation_t feed_forward_implementation;
  // Diagnostic tap names requested by the stage plan.
  iree_string_view_list_t diagnostic_tap_names;
  // Static conditioned-DiT LoRA topology; empty selects exact base execution.
  id4_ideogram4_dit_lora_topology_t lora_topology;
} id4_ideogram4_dit_program_options_t;

// Authors the Ideogram4 DiT forward program into |builder|.
iree_status_t id4_ideogram4_dit_program_author_forward(
    const id4_ideogram4_dit_program_options_t* options,
    id4_pipeline_program_builder_t* builder);

// Formats a layer-ordinal parameter key using the canonical DiT key scheme.
iree_status_t id4_ideogram4_dit_program_format_layer_parameter(
    uint32_t layer_ordinal, iree_string_view_t suffix, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string);

// Formats the canonical output-row scale key for an FP8 weight parameter.
iree_status_t id4_ideogram4_dit_program_format_parameter_scale_key(
    iree_string_view_t weight_key, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_string);

// Returns the Ideogram 4 DiT model configuration.
const id4_ideogram4_dit_model_config_t*
id4_ideogram4_dit_program_ideogram4_model_config(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_STAGES_IDEOGRAM4_DIT_PROGRAM_H_
