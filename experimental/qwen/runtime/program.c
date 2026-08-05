// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/program.h"

#include <inttypes.h>
#include <string.h>

#include "experimental/qwen/runtime/loom_jit.h"
#include "experimental/qwen/runtime/loom_source.h"
#include "experimental/qwen/runtime/model_shape.h"
#include "experimental/qwen/runtime/parameters.h"
#include "experimental/qwen/runtime/program_layout.h"
#include "experimental/qwen/runtime/request_state.h"
#include "iree/base/internal/atomics.h"

#define QWEN_PROGRAM_BINDING_COUNT 6
#define QWEN_PROGRAM_CONFIG_BINDING_CAPACITY 16
#define QWEN_PROGRAM_CONFIG_VALUE_CAPACITY 32
#define QWEN_PROGRAM_INITIAL_SEMAPHORE_CAPACITY 8
typedef enum qwen_program_binding_slot_e {
  // Complete resident model parameter allocation.
  QWEN_PROGRAM_BINDING_MODEL = 0,
  // Per-issue transient program allocation.
  QWEN_PROGRAM_BINDING_TRANSIENT = 1,
  // Selected layer's key cache.
  QWEN_PROGRAM_BINDING_KEY_CACHE = 2,
  // Selected layer's value cache.
  QWEN_PROGRAM_BINDING_VALUE_CACHE = 3,
  // Request hidden state, compact control, and derived attention metadata.
  QWEN_PROGRAM_BINDING_REQUEST_STATE = 4,
  // Host-visible completed result staging.
  QWEN_PROGRAM_BINDING_OUTPUT_STAGING = 5,
} qwen_program_binding_slot_t;

typedef enum qwen_program_qkv_schedule_e {
  // Fused F32 RMSNorm-to-Q8 plus row-parallel Q/K/V contractions.
  QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS = 0,
  // Materialized F32 RMSNorm plus tiled Q/K/V WMMA projections.
  QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA = 1,
} qwen_program_qkv_schedule_t;

typedef enum qwen_program_attention_schedule_e {
  // General grouped-query attention used by prefill and layer programs.
  QWEN_PROGRAM_ATTENTION_SCHEDULE_GENERAL = 0,
  // One-token split-K attention with fused last-arrival reduction.
  QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT = 1,
} qwen_program_attention_schedule_t;

typedef enum qwen_program_attention_postprocess_schedule_e {
  // Standalone head postprocessing after complete Q/K/V projections.
  QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE = 0,
  // Last-arrival head postprocessing inside the decode Q/K/V projection.
  QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_FUSED_QKV = 1,
} qwen_program_attention_postprocess_schedule_t;

typedef enum qwen_program_attention_output_schedule_e {
  // Packed Q8_1 input plus the row-parallel direct Q4_K contraction.
  QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 = 0,
  // Materialized F32 input plus the tiled Q4_K WMMA contraction.
  QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_F32_WMMA = 1,
} qwen_program_attention_output_schedule_t;

typedef enum qwen_program_attention_prepare_schedule_e {
  // Each attention stage prepares its projection input from hidden state.
  QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_PER_LAYER = 0,
  // Each preceding feed-forward stage publishes the next projection input.
  QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_INTERLAYER = 1,
} qwen_program_attention_prepare_schedule_t;

typedef enum qwen_program_feed_forward_schedule_e {
  // Materialized F32 RMSNorm plus grouped F16 routed projections.
  QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16 = 0,
  // Fused RMSNorm/Q8 production plus direct quantized routed projections.
  QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8 = 1,
} qwen_program_feed_forward_schedule_t;

typedef enum qwen_program_expert_table_schedule_e {
  // Independent expert assignment and partition-table dispatches.
  QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_SEPARATE = 0,
  // One Prefill-512 dispatch publishes both table representations.
  QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_FUSED_PREFILL_512 = 1,
} qwen_program_expert_table_schedule_t;

typedef enum qwen_program_executable_ordinal_e {
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_METADATA = 0,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_PREPARE_Q8 = 1,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q4 = 2,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q6 = 3,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_POSTPROCESS = 4,
  QWEN_PROGRAM_EXECUTABLE_FLASH_ATTENTION = 5,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_F32_WMMA = 6,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_QUANTIZE_Q8 = 7,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_DIRECT = 8,
  QWEN_PROGRAM_EXECUTABLE_RMSNORM_F32 = 9,
  QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION = 10,
  QWEN_PROGRAM_EXECUTABLE_ROUTER_TOP8 = 11,
  QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE = 12,
  QWEN_PROGRAM_EXECUTABLE_PARTITION_TABLE = 13,
  QWEN_PROGRAM_EXECUTABLE_GATE_UP = 14,
  QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4 = 15,
  QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6 = 16,
  QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE = 17,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_QUERY_Q4_WMMA = 18,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_KEY_VALUE_Q4_WMMA = 19,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_VALUE_Q6_WMMA = 20,
  QWEN_PROGRAM_EXECUTABLE_TOKEN_EMBEDDING = 21,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_RMSNORM_F32 = 22,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_PROJECTION = 23,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_TOP8 = 24,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_EXPERT_TABLE = 25,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_PARTITION_TABLE = 26,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_GATE_UP = 27,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTED_DOWN = 28,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_WEIGHTED_REDUCE = 29,
  QWEN_PROGRAM_EXECUTABLE_FINAL_RMSNORM_QUANTIZE_Q8 = 30,
  QWEN_PROGRAM_EXECUTABLE_VOCABULARY_PARTIAL_ARGMAX_Q6 = 31,
  QWEN_PROGRAM_EXECUTABLE_GREEDY_ARGMAX_PARTIALS = 32,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_NEXT_Q8 = 33,
  QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32 = 34,
  QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32_NEXT_Q8 = 35,
  QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4_NEXT_Q8 = 36,
  QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6_F32_NEXT_Q8 = 37,
  QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION_TOP8_FUSED = 38,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q4 = 39,
  QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q6 = 40,
  QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE_NEXT_RMSNORM = 41,
  QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE_PARTITION_FUSED = 42,
  QWEN_PROGRAM_EXECUTABLE_ROUTE_TRACE = 43,
  QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTE_TRACE = 44,
  QWEN_PROGRAM_EXECUTABLE_COUNT = 45,
} qwen_program_executable_ordinal_t;

// Complete output publication owned by a nonterminal grouped feed-forward.
typedef struct qwen_program_next_attention_publication_t {
  // Learned RMSNorm weight for the next transformer layer.
  qwen_parameter_span_t norm_weight;
  // F32 normalized projection input consumed by the next attention stage.
  qwen_program_span_t projection_input;
} qwen_program_next_attention_publication_t;

// Executable specializations consumed by one grouped feed-forward recording.
typedef struct qwen_program_grouped_feed_forward_executables_t {
  // F32 RMSNorm specialization.
  qwen_program_executable_ordinal_t rmsnorm;
  // Router projection specialization.
  qwen_program_executable_ordinal_t router_projection;
  // Normalized top-8 selection specialization.
  qwen_program_executable_ordinal_t router_top8;
  // Expert assignment-table specialization.
  qwen_program_executable_ordinal_t expert_table;
  // Grouped-dispatch partition-table specialization.
  qwen_program_executable_ordinal_t partition_table;
  // Routed gate/up SwiGLU specialization.
  qwen_program_executable_ordinal_t gate_up;
  // Storage-specific routed down-projection specialization.
  qwen_program_executable_ordinal_t routed_down;
  // Routed residual reduction specialization.
  qwen_program_executable_ordinal_t weighted_reduce;
} qwen_program_grouped_feed_forward_executables_t;

struct qwen_program_t {
  // Reference count for shared program ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for program-owned host allocations.
  iree_allocator_t host_allocator;
  // Model retained for executable and parameter ownership.
  qwen_model_t* model;
  // Mathematical scope recorded by the program.
  qwen_program_kind_t kind;
  // Selected transformer layer.
  iree_host_size_t layer_index;
  // First model layer covered by the issue-time cache bindings.
  iree_host_size_t first_cache_layer;
  // Number of consecutive model layers covered by the cache bindings.
  iree_host_size_t cache_layer_count;
  // Exact active token rows consumed by each issue.
  iree_host_size_t token_count;
  // Exact visible K/V rows for fixed programs, or decode class upper bound.
  iree_host_size_t context_count;
  // Physical K/V rows consumed by attention after masked tile padding.
  iree_host_size_t attention_context_count;
  // Compatible request token-storage capacity.
  iree_host_size_t token_capacity;
  // Compatible request K/V storage capacity.
  iree_host_size_t context_capacity;
  // Optional request storage behavior addressed by this command buffer.
  qwen_request_flags_t request_flags;
  // Q/K/V contraction family selected for this token shape.
  qwen_program_qkv_schedule_t qkv_schedule;
  // FlashAttention family selected for this program role.
  qwen_program_attention_schedule_t attention_schedule;
  // Q/K/V-to-head-postprocessing boundary selected for this program role.
  qwen_program_attention_postprocess_schedule_t attention_postprocess_schedule;
  // Attention output contraction family selected for this token shape.
  qwen_program_attention_output_schedule_t attention_output_schedule;
  // Ownership of the normalized input consumed by each attention stage.
  qwen_program_attention_prepare_schedule_t attention_prepare_schedule;
  // Routed feed-forward family selected for this program role.
  qwen_program_feed_forward_schedule_t feed_forward_schedule;
  // Expert and partition table publication selected for nonterminal layers.
  qwen_program_expert_table_schedule_t expert_table_schedule;
  // Prepared executable for each unique recorded kernel specialization.
  qwen_loom_executable_t* executables[QWEN_PROGRAM_EXECUTABLE_COUNT];
  // Number of dispatches recorded in the command buffer.
  iree_host_size_t dispatch_count;
  // Reusable indirect command buffer.
  iree_hal_command_buffer_t* command_buffer;
  // Internal scratch-allocation and issue-completion timeline.
  iree_hal_semaphore_t* timeline_semaphore;
  // Latest terminal issue value reserved on |timeline_semaphore|.
  uint64_t timeline_value;
  // Reusable layer scratch used by either program kind.
  qwen_layer_program_layout_t layer_layout;
  // Full-model-only terminal and endpoint scratch.
  qwen_full_program_layout_t full_layout;
  // Complete per-issue transient allocation size.
  iree_device_size_t transient_byte_length;
  // Host-visible result bytes written by the recorded command buffer.
  iree_device_size_t output_staging_byte_length;
  // Reusable merged wait-semaphore pointer storage.
  iree_hal_semaphore_t** wait_semaphores;
  // Reusable merged wait payload storage.
  uint64_t* wait_values;
  // Allocated entries in the merged wait arrays.
  iree_host_size_t wait_capacity;
  // Reusable merged signal-semaphore pointer storage.
  iree_hal_semaphore_t** signal_semaphores;
  // Reusable merged signal payload storage.
  uint64_t* signal_values;
  // Allocated entries in the merged signal arrays.
  iree_host_size_t signal_capacity;
};

// Bounded cold-path config set assembled before a prepare request copies it.
typedef struct qwen_program_config_binding_list_t {
  // Ordered static and dynamically formatted config bindings.
  qwen_loom_config_binding_t bindings[QWEN_PROGRAM_CONFIG_BINDING_CAPACITY];
  // Stable storage for dynamically appended integer binding values.
  char value_storage[QWEN_PROGRAM_CONFIG_BINDING_CAPACITY]
                    [QWEN_PROGRAM_CONFIG_VALUE_CAPACITY];
  // Number of initialized entries in |bindings|.
  iree_host_size_t count;
} qwen_program_config_binding_list_t;

static void qwen_program_config_binding_list_initialize(
    iree_host_size_t static_binding_count,
    const qwen_loom_config_binding_t* static_bindings,
    qwen_program_config_binding_list_t* out_list) {
  IREE_ASSERT(static_binding_count <= IREE_ARRAYSIZE(out_list->bindings));
  memset(out_list, 0, sizeof(*out_list));
  if (static_binding_count != 0) {
    memcpy(out_list->bindings, static_bindings,
           static_binding_count * sizeof(out_list->bindings[0]));
  }
  out_list->count = static_binding_count;
}

static void qwen_program_config_binding_list_append_index(
    qwen_program_config_binding_list_t* list, iree_string_view_t key,
    int64_t value) {
  IREE_ASSERT(list->count < IREE_ARRAYSIZE(list->bindings));
  char* value_storage = list->value_storage[list->count];
  const int value_length = iree_snprintf(
      value_storage, QWEN_PROGRAM_CONFIG_VALUE_CAPACITY, "%" PRId64, value);
  IREE_ASSERT(value_length >= 0 && (iree_host_size_t)value_length <
                                       QWEN_PROGRAM_CONFIG_VALUE_CAPACITY);
  list->bindings[list->count++] = (qwen_loom_config_binding_t){
      .key = key,
      .value =
          iree_make_string_view(value_storage, (iree_host_size_t)value_length),
  };
}

static void qwen_program_config_binding_list_initialize_with_token_capacity(
    iree_host_size_t static_binding_count,
    const qwen_loom_config_binding_t* static_bindings, int64_t token_capacity,
    qwen_program_config_binding_list_t* out_list) {
  qwen_program_config_binding_list_initialize(static_binding_count,
                                              static_bindings, out_list);
  qwen_program_config_binding_list_append_index(
      out_list, IREE_SV("qwen3_moe.workload.token_capacity"), token_capacity);
}

static const qwen_loom_config_binding_t
    qwen_attention_prepare_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_qkv_q6_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.query_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.value_uses_q6"),
            .value = IREE_SVL("1"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_qkv_q4_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.query_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.value_uses_q6"),
            .value = IREE_SVL("0"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_qkv_postprocess_q6_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.query_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.value_uses_q6"),
            .value = IREE_SVL("1"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.head_size"),
            .value = IREE_SVL("128"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_qkv_postprocess_q4_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.query_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.value_uses_q6"),
            .value = IREE_SVL("0"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.head_size"),
            .value = IREE_SVL("128"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_query_wmma_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.input_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_accumulation"),
            .value = IREE_SVL("0"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_key_value_wmma_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.input_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_accumulation"),
            .value = IREE_SVL("0"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_postprocess_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.attention.head_size"),
            .value = IREE_SVL("128"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.query_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_size"),
            .value = IREE_SVL("512"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t qwen_flash_attention_config_bindings[] =
    {
        {
            .key = IREE_SVL("qwen3_moe.attention.query_head_count"),
            .value = IREE_SVL("32"),
        },
        {
            .key = IREE_SVL("qwen3_moe.attention.key_value_head_count"),
            .value = IREE_SVL("4"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_output_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.input_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_accumulation"),
            .value = IREE_SVL("1"),
        },
};

static const qwen_loom_config_binding_t
    qwen_attention_output_next_q8_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.input_size"),
            .value = IREE_SVL("4096"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.dense_quantized.output_accumulation"),
            .value = IREE_SVL("1"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t
    qwen_router_projection_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.router.expert_count"),
            .value = IREE_SVL("128"),
        },
};

static const qwen_loom_config_binding_t qwen_router_top8_config_bindings[] = {
    {
        .key = IREE_SVL("qwen3_moe.router.expert_count"),
        .value = IREE_SVL("128"),
    },
    {
        .key = IREE_SVL("qwen3_moe.router.route_count"),
        .value = IREE_SVL("8"),
    },
};

static const qwen_loom_config_binding_t
    qwen_router_projection_top8_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.router.expert_count"),
            .value = IREE_SVL("128"),
        },
        {
            .key = IREE_SVL("qwen3_moe.router.route_count"),
            .value = IREE_SVL("8"),
        },
};

static const qwen_loom_config_binding_t qwen_gate_up_config_bindings[] = {
    {
        .key = IREE_SVL("qwen3_moe.routed_gate_up.input_size"),
        .value = IREE_SVL("2048"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_gate_up.output_size"),
        .value = IREE_SVL("768"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_gate_up.route_count"),
        .value = IREE_SVL("8"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_gate_up.expert_count"),
        .value = IREE_SVL("128"),
    },
};

static const qwen_loom_config_binding_t qwen_routed_down_config_bindings[] = {
    {
        .key = IREE_SVL("qwen3_moe.routed_down.input_size"),
        .value = IREE_SVL("768"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_down.output_size"),
        .value = IREE_SVL("2048"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_down.route_count"),
        .value = IREE_SVL("8"),
    },
    {
        .key = IREE_SVL("qwen3_moe.routed_down.expert_count"),
        .value = IREE_SVL("128"),
    },
};

static const qwen_loom_config_binding_t
    qwen_direct_routed_down_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.routed_down.input_size"),
            .value = IREE_SVL("768"),
        },
        {
            .key = IREE_SVL("qwen3_moe.routed_down.output_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.routed_down.route_count"),
            .value = IREE_SVL("8"),
        },
        {
            .key = IREE_SVL("qwen3_moe.routed_down.expert_count"),
            .value = IREE_SVL("128"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static const qwen_loom_config_binding_t
    qwen_weighted_reduce_next_rmsnorm_config_bindings[] = {
        {
            .key = IREE_SVL("qwen3_moe.routed_down.output_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.routed_down.route_count"),
            .value = IREE_SVL("8"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.hidden_size"),
            .value = IREE_SVL("2048"),
        },
        {
            .key = IREE_SVL("qwen3_moe.model.rms_epsilon"),
            .value = IREE_SVL("0.000001"),
        },
};

static iree_status_t qwen_program_reserve_semaphore_storage(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t*** out_semaphores, uint64_t** out_values) {
  *out_semaphores = NULL;
  *out_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(**out_semaphores),
                                                     (void**)out_semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, capacity, sizeof(**out_values), (void**)out_values);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_values);
    iree_allocator_free(host_allocator, *out_semaphores);
    *out_values = NULL;
    *out_semaphores = NULL;
  }
  return status;
}

static iree_status_t qwen_program_ensure_semaphore_storage(
    iree_host_size_t required_capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t*** semaphores, uint64_t** values,
    iree_host_size_t* capacity) {
  if (required_capacity <= *capacity) return iree_ok_status();
  iree_host_size_t new_capacity = *capacity;
  while (new_capacity < required_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen semaphore storage capacity overflows");
    }
    new_capacity *= 2;
  }
  iree_hal_semaphore_t** new_semaphores = NULL;
  uint64_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(qwen_program_reserve_semaphore_storage(
      new_capacity, host_allocator, &new_semaphores, &new_values));
  iree_allocator_free(host_allocator, *values);
  iree_allocator_free(host_allocator, *semaphores);
  *semaphores = new_semaphores;
  *values = new_values;
  *capacity = new_capacity;
  return iree_ok_status();
}

static void qwen_program_destroy(qwen_program_t* program) {
  iree_allocator_free(program->host_allocator, program->signal_values);
  iree_allocator_free(program->host_allocator, program->signal_semaphores);
  iree_allocator_free(program->host_allocator, program->wait_values);
  iree_allocator_free(program->host_allocator, program->wait_semaphores);
  iree_hal_semaphore_release(program->timeline_semaphore);
  iree_hal_command_buffer_release(program->command_buffer);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(program->executables); ++i) {
    qwen_loom_executable_release(program->executables[i]);
  }
  qwen_model_release(program->model);
  iree_allocator_t host_allocator = program->host_allocator;
  iree_allocator_free(host_allocator, program);
}

typedef struct qwen_program_prepare_batch_t {
  // Model whose JIT consumes every request in this batch.
  qwen_model_t* model;
  // Allocator used for copied request descriptors.
  iree_allocator_t host_allocator;
  // Number of initialized prepare requests.
  iree_host_size_t request_count;
  // Embedded source descriptors kept alive through batch execution.
  qwen_loom_source_module_t source_modules[QWEN_PROGRAM_EXECUTABLE_COUNT];
  // Owned exact JIT request descriptors.
  qwen_loom_jit_prepare_options_t requests[QWEN_PROGRAM_EXECUTABLE_COUNT];
  // Program executable slots that receive successful batch outputs.
  qwen_loom_executable_t** destinations[QWEN_PROGRAM_EXECUTABLE_COUNT];
} qwen_program_prepare_batch_t;

static iree_status_t qwen_program_prepare_copy_string(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void qwen_program_prepare_request_deinitialize(
    qwen_loom_jit_prepare_options_t* request, iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, (void*)request->workload_arguments);
  qwen_loom_config_binding_t* config_bindings =
      (qwen_loom_config_binding_t*)request->config_bindings;
  if (config_bindings) {
    for (iree_host_size_t i = 0; i < request->config_binding_count; ++i) {
      iree_allocator_free(host_allocator, (void*)config_bindings[i].value.data);
      iree_allocator_free(host_allocator, (void*)config_bindings[i].key.data);
    }
  }
  iree_allocator_free(host_allocator, config_bindings);
  iree_allocator_free(host_allocator, (void*)request->function_name.data);
  memset(request, 0, sizeof(*request));
}

static void qwen_program_prepare_batch_deinitialize(
    qwen_program_prepare_batch_t* batch) {
  for (iree_host_size_t i = 0; i < batch->request_count; ++i) {
    qwen_program_prepare_request_deinitialize(&batch->requests[i],
                                              batch->host_allocator);
  }
  memset(batch, 0, sizeof(*batch));
}

static iree_status_t qwen_program_prepare_batch_append(
    qwen_program_prepare_batch_t* batch, iree_string_view_t module_path,
    iree_string_view_t function_name, iree_host_size_t config_binding_count,
    const qwen_loom_config_binding_t* config_bindings,
    iree_host_size_t workload_argument_count, const int64_t* workload_arguments,
    qwen_loom_executable_t** destination) {
  if (batch->request_count == QWEN_PROGRAM_EXECUTABLE_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen program executable batch is full");
  }

  const iree_host_size_t request_ordinal = batch->request_count;
  qwen_loom_jit_prepare_options_t* request = &batch->requests[request_ordinal];
  memset(request, 0, sizeof(*request));
  request->structure_size = sizeof(*request);
  request->source_module = &batch->source_modules[request_ordinal];
  request->config_binding_count = config_binding_count;
  request->workload_argument_count = workload_argument_count;

  iree_status_t status = qwen_loom_source_lookup(
      module_path, &batch->source_modules[request_ordinal]);
  if (iree_status_is_ok(status)) {
    status = qwen_program_prepare_copy_string(
        function_name, batch->host_allocator, &request->function_name);
  }

  qwen_loom_config_binding_t* owned_config_bindings = NULL;
  if (iree_status_is_ok(status) && config_binding_count != 0) {
    status = iree_allocator_malloc_array(
        batch->host_allocator, config_binding_count,
        sizeof(owned_config_bindings[0]), (void**)&owned_config_bindings);
  }
  if (iree_status_is_ok(status) && config_binding_count != 0) {
    memset(owned_config_bindings, 0,
           config_binding_count * sizeof(owned_config_bindings[0]));
    request->config_bindings = owned_config_bindings;
    for (iree_host_size_t i = 0;
         i < config_binding_count && iree_status_is_ok(status); ++i) {
      status = qwen_program_prepare_copy_string(config_bindings[i].key,
                                                batch->host_allocator,
                                                &owned_config_bindings[i].key);
      if (iree_status_is_ok(status)) {
        status = qwen_program_prepare_copy_string(
            config_bindings[i].value, batch->host_allocator,
            &owned_config_bindings[i].value);
      }
    }
  }

  int64_t* owned_workload_arguments = NULL;
  if (iree_status_is_ok(status) && workload_argument_count != 0) {
    status = iree_allocator_malloc_array(
        batch->host_allocator, workload_argument_count,
        sizeof(owned_workload_arguments[0]), (void**)&owned_workload_arguments);
  }
  if (iree_status_is_ok(status) && workload_argument_count != 0) {
    memcpy(owned_workload_arguments, workload_arguments,
           workload_argument_count * sizeof(owned_workload_arguments[0]));
    request->workload_arguments = owned_workload_arguments;
  }

  if (iree_status_is_ok(status)) {
    batch->destinations[request_ordinal] = destination;
    ++batch->request_count;
  } else {
    qwen_program_prepare_request_deinitialize(request, batch->host_allocator);
  }
  return status;
}

static iree_status_t qwen_program_prepare_batches_execute(
    iree_host_size_t batch_count, qwen_program_prepare_batch_t* batches) {
  iree_host_size_t request_count = 0;
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    if (!iree_host_size_checked_add(request_count, batches[i].request_count,
                                    &request_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen program prepare request count overflows");
    }
  }
  if (request_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen program prepare union is empty");
  }

  iree_allocator_t host_allocator = batches[0].host_allocator;
  qwen_loom_jit_prepare_options_t* requests = NULL;
  qwen_loom_executable_t** outputs = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, request_count, sizeof(requests[0]), (void**)&requests);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, request_count,
                                         sizeof(outputs[0]), (void**)&outputs);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t request_ordinal = 0;
    for (iree_host_size_t i = 0; i < batch_count; ++i) {
      memcpy(&requests[request_ordinal], batches[i].requests,
             batches[i].request_count * sizeof(requests[0]));
      request_ordinal += batches[i].request_count;
    }
    status = qwen_loom_jit_prepare_batch(qwen_model_loom_jit(batches[0].model),
                                         request_count, requests, outputs);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t request_ordinal = 0;
    for (iree_host_size_t i = 0; i < batch_count; ++i) {
      for (iree_host_size_t j = 0; j < batches[i].request_count; ++j) {
        *batches[i].destinations[j] = outputs[request_ordinal++];
      }
    }
  }
  iree_allocator_free(host_allocator, outputs);
  iree_allocator_free(host_allocator, requests);
  return status;
}

// Prepares decode attention for one reusable 64-row context class. Active
// context is represented by the device-produced mask and therefore does not
// participate in JIT identity.
static iree_status_t qwen_program_prepare_decode_flash_attention(
    qwen_program_t* program, qwen_program_prepare_batch_t* batch) {
  qwen_program_config_binding_list_t config_bindings;
  qwen_program_config_binding_list_initialize(
      IREE_ARRAYSIZE(qwen_flash_attention_config_bindings),
      qwen_flash_attention_config_bindings, &config_bindings);
  qwen_program_config_binding_list_append_index(
      &config_bindings, IREE_SV("qwen3_moe.attention.key_value_token_capacity"),
      (int64_t)program->context_count);
  const int64_t workload[] = {(int64_t)program->context_count};
  return qwen_program_prepare_batch_append(
      batch, IREE_SV(QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16),
      IREE_SV("qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8"),
      config_bindings.count, config_bindings.bindings, IREE_ARRAYSIZE(workload),
      workload, &program->executables[QWEN_PROGRAM_EXECUTABLE_FLASH_ATTENTION]);
}

static qwen_program_attention_schedule_t qwen_program_select_attention_schedule(
    qwen_program_kind_t kind) {
  return kind == QWEN_PROGRAM_KIND_DECODE
             ? QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT
             : QWEN_PROGRAM_ATTENTION_SCHEDULE_GENERAL;
}

static qwen_program_attention_postprocess_schedule_t
qwen_program_select_attention_postprocess_schedule(qwen_program_kind_t kind) {
  return kind == QWEN_PROGRAM_KIND_DECODE
             ? QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_FUSED_QKV
             : QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE;
}

static qwen_program_attention_output_schedule_t
qwen_program_select_attention_output_schedule(iree_host_size_t token_count) {
  // The direct-Q8 result is established only for one-token decode. Keep every
  // larger shape on the measured WMMA schedule until its crossover is captured.
  return token_count == 1 ? QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8
                          : QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_F32_WMMA;
}

static qwen_program_attention_prepare_schedule_t
qwen_program_select_attention_prepare_schedule(qwen_program_kind_t kind) {
  return kind == QWEN_PROGRAM_KIND_LAYER
             ? QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_PER_LAYER
             : QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_INTERLAYER;
}

static qwen_program_feed_forward_schedule_t
qwen_program_select_feed_forward_schedule(qwen_program_kind_t kind) {
  // The direct route is established for full-model decode. Layer programs and
  // every prefill row retain the grouped correctness and performance baseline.
  return kind == QWEN_PROGRAM_KIND_DECODE
             ? QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8
             : QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16;
}

static qwen_program_expert_table_schedule_t
qwen_program_select_expert_table_schedule(qwen_program_kind_t kind,
                                          iree_host_size_t token_count) {
  return kind == QWEN_PROGRAM_KIND_PREFILL && token_count == 512
             ? QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_FUSED_PREFILL_512
             : QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_SEPARATE;
}

static bool qwen_program_kind_is_full_model(qwen_program_kind_t kind) {
  return kind == QWEN_PROGRAM_KIND_PREFILL || kind == QWEN_PROGRAM_KIND_DECODE;
}

static qwen_full_program_layout_flags_t qwen_program_full_layout_flags(
    qwen_program_kind_t kind, iree_host_size_t token_count) {
  if (kind == QWEN_PROGRAM_KIND_DECODE) {
    return QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION |
           QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_STAGE_COMPLETION;
  }
  return qwen_program_select_expert_table_schedule(kind, token_count) ==
                 QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_FUSED_PREFILL_512
             ? QWEN_FULL_PROGRAM_LAYOUT_FLAG_PREFILL_FUSED_EXPERT_TABLE
             : QWEN_FULL_PROGRAM_LAYOUT_FLAG_NONE;
}

static iree_hal_command_category_t qwen_program_command_categories(
    qwen_program_kind_t kind) {
  return qwen_program_kind_is_full_model(kind)
             ? IREE_HAL_COMMAND_CATEGORY_DISPATCH
             : IREE_HAL_COMMAND_CATEGORY_DISPATCH |
                   IREE_HAL_COMMAND_CATEGORY_TRANSFER;
}

static iree_status_t qwen_program_prepare_layer_executables(
    qwen_program_t* program, qwen_program_prepare_batch_t* batch) {
  const int64_t token_count = (int64_t)program->token_count;
  const int64_t route_stride = QWEN_MODEL_ROUTE_COUNT;
  const int64_t route_count = QWEN_MODEL_ROUTE_COUNT;
  const int64_t expert_count = QWEN_MODEL_EXPERT_COUNT;
  const int64_t attention_context_count =
      (int64_t)program->attention_context_count;
  const int64_t token_workload[] = {token_count};
  const int64_t attention_metadata_workload[] = {
      token_count,
      attention_context_count,
  };
  const int64_t attention_postprocess_workload[] = {
      token_count,
      attention_context_count,
  };
  const int64_t flash_attention_workload[] = {
      token_count,
      attention_context_count,
  };
  const int64_t attention_output_quantize_workload[] = {
      token_count,
      QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE,
  };
  const int64_t attention_output_quantize_group_capacity =
      token_count * (QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE /
                     QWEN_PROGRAM_Q8_1_X4_GROUP_ELEMENT_COUNT);
  const int64_t router_top8_workload[] = {
      token_count,
      route_stride,
  };
  const int64_t expert_table_workload[] = {
      token_count,
      route_count,
      route_stride,
      expert_count,
  };
  const int64_t partition_table_workload[] = {
      token_count,
      route_count,
      expert_count,
  };
  const int64_t direct_gate_up_workload[] = {
      token_count,
      route_count,
      route_stride,
      expert_count,
      QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE,
  };
  const int64_t direct_routed_down_workload[] = {
      token_count,  QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE,
      route_count,  route_stride,
      expert_count, QWEN_MODEL_HIDDEN_SIZE,
  };

  qwen_program_config_binding_list_t attention_metadata_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      /*static_binding_count=*/0, /*static_bindings=*/NULL, token_count,
      &attention_metadata_config_binding_list);
  qwen_program_config_binding_list_append_index(
      &attention_metadata_config_binding_list,
      IREE_SV("qwen.attention.metadata_context_capacity"),
      attention_context_count);

  qwen_program_config_binding_list_t attention_prepare_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_prepare_config_bindings),
      qwen_attention_prepare_config_bindings, token_count,
      &attention_prepare_config_binding_list);

  qwen_program_config_binding_list_t attention_qkv_q6_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_qkv_q6_config_bindings),
      qwen_attention_qkv_q6_config_bindings, token_count,
      &attention_qkv_q6_config_binding_list);

  qwen_program_config_binding_list_t attention_qkv_q4_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_qkv_q4_config_bindings),
      qwen_attention_qkv_q4_config_bindings, token_count,
      &attention_qkv_q4_config_binding_list);

  qwen_program_config_binding_list_t
      attention_qkv_postprocess_q6_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_qkv_postprocess_q6_config_bindings),
      qwen_attention_qkv_postprocess_q6_config_bindings, token_count,
      &attention_qkv_postprocess_q6_config_binding_list);

  qwen_program_config_binding_list_t
      attention_qkv_postprocess_q4_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_qkv_postprocess_q4_config_bindings),
      qwen_attention_qkv_postprocess_q4_config_bindings, token_count,
      &attention_qkv_postprocess_q4_config_binding_list);

  qwen_program_config_binding_list_t attention_postprocess_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_postprocess_config_bindings),
      qwen_attention_postprocess_config_bindings, token_count,
      &attention_postprocess_config_binding_list);

  qwen_program_config_binding_list_t flash_attention_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_flash_attention_config_bindings),
      qwen_flash_attention_config_bindings, token_count,
      &flash_attention_config_binding_list);

  qwen_program_config_binding_list_t attention_query_wmma_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_query_wmma_config_bindings),
      qwen_attention_query_wmma_config_bindings, token_count,
      &attention_query_wmma_config_binding_list);

  qwen_program_config_binding_list_t
      attention_key_value_wmma_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_key_value_wmma_config_bindings),
      qwen_attention_key_value_wmma_config_bindings, token_count,
      &attention_key_value_wmma_config_binding_list);

  qwen_program_config_binding_list_t attention_output_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_output_config_bindings),
      qwen_attention_output_config_bindings, token_count,
      &attention_output_config_binding_list);

  qwen_program_config_binding_list_t
      attention_output_next_q8_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_output_next_q8_config_bindings),
      qwen_attention_output_next_q8_config_bindings, token_count,
      &attention_output_next_q8_config_binding_list);

  qwen_program_config_binding_list_t
      attention_output_quantize_config_binding_list;
  qwen_program_config_binding_list_initialize(
      /*static_binding_count=*/0, /*static_bindings=*/NULL,
      &attention_output_quantize_config_binding_list);
  qwen_program_config_binding_list_append_index(
      &attention_output_quantize_config_binding_list,
      IREE_SV("ggml.quantize_q8_1_x4.group_capacity"),
      attention_output_quantize_group_capacity);

  qwen_program_config_binding_list_t router_projection_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_router_projection_config_bindings),
      qwen_router_projection_config_bindings, token_count,
      &router_projection_config_binding_list);

  qwen_program_config_binding_list_t router_top8_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_router_top8_config_bindings),
      qwen_router_top8_config_bindings, token_count,
      &router_top8_config_binding_list);

  qwen_program_config_binding_list_t router_projection_top8_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_router_projection_top8_config_bindings),
      qwen_router_projection_top8_config_bindings, token_count,
      &router_projection_top8_config_binding_list);

  qwen_program_config_binding_list_t gate_up_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_gate_up_config_bindings),
      qwen_gate_up_config_bindings, token_count, &gate_up_config_binding_list);

  qwen_program_config_binding_list_t routed_down_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_routed_down_config_bindings),
      qwen_routed_down_config_bindings, token_count,
      &routed_down_config_binding_list);

  qwen_program_config_binding_list_t direct_routed_down_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_direct_routed_down_config_bindings),
      qwen_direct_routed_down_config_bindings, token_count,
      &direct_routed_down_config_binding_list);

  iree_status_t status = qwen_program_prepare_batch_append(
      batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_METADATA),
      IREE_SV("qwen_attention_metadata"),
      attention_metadata_config_binding_list.count,
      attention_metadata_config_binding_list.bindings,
      IREE_ARRAYSIZE(attention_metadata_workload), attention_metadata_workload,
      &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_METADATA]);
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED),
        IREE_SV("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4"),
        attention_prepare_config_binding_list.count,
        attention_prepare_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_PREPARE_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED),
        IREE_SV("qwen3_moe_attention_qkv_quantized"),
        attention_qkv_q4_config_binding_list.count,
        attention_qkv_q4_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q4]);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED),
        IREE_SV("qwen3_moe_attention_qkv_quantized"),
        attention_qkv_q6_config_binding_list.count,
        attention_qkv_q6_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q6]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_FUSED_QKV) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_QKV_POSTPROCESS_FUSED),
        IREE_SV("qwen3_moe_attention_qkv_postprocess_fused_decode"),
        attention_qkv_postprocess_q4_config_binding_list.count,
        attention_qkv_postprocess_q4_config_binding_list.bindings,
        IREE_ARRAYSIZE(attention_postprocess_workload),
        attention_postprocess_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q4]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_FUSED_QKV) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_QKV_POSTPROCESS_FUSED),
        IREE_SV("qwen3_moe_attention_qkv_postprocess_fused_decode"),
        attention_qkv_postprocess_q6_config_binding_list.count,
        attention_qkv_postprocess_q6_config_binding_list.bindings,
        IREE_ARRAYSIZE(attention_postprocess_workload),
        attention_postprocess_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q6]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16),
        IREE_SV("qwen3_moe_attention_postprocess_f32_f16"),
        attention_postprocess_config_binding_list.count,
        attention_postprocess_config_binding_list.bindings,
        IREE_ARRAYSIZE(attention_postprocess_workload),
        attention_postprocess_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_POSTPROCESS]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_schedule == QWEN_PROGRAM_ATTENTION_SCHEDULE_GENERAL) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16),
        IREE_SV("qwen3_moe_flash_attention_f32_f16_wmma"),
        flash_attention_config_binding_list.count,
        flash_attention_config_binding_list.bindings,
        IREE_ARRAYSIZE(flash_attention_workload), flash_attention_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_FLASH_ATTENTION]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_schedule ==
          QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT) {
    status = qwen_program_prepare_decode_flash_attention(program, batch);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_F32_WMMA) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q4k_f16_wmma"),
        attention_output_config_binding_list.count,
        attention_output_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_F32_WMMA]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->attention_schedule !=
          QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4),
        IREE_SV("ggml_quantize_q8_1_x4_f32"),
        attention_output_quantize_config_binding_list.count,
        attention_output_quantize_config_binding_list.bindings,
        IREE_ARRAYSIZE(attention_output_quantize_workload),
        attention_output_quantize_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_QUANTIZE_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q4k_q8_1_x4"),
        attention_output_config_binding_list.count,
        attention_output_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_DIRECT]);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8"),
        attention_output_next_q8_config_binding_list.count,
        attention_output_next_q8_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_NEXT_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED),
        IREE_SV("qwen3_moe_rmsnorm_f32"),
        attention_prepare_config_binding_list.count,
        attention_prepare_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_RMSNORM_F32]);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q4k_f16_wmma"),
        attention_query_wmma_config_binding_list.count,
        attention_query_wmma_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_QUERY_Q4_WMMA]);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q4k_f16_wmma"),
        attention_key_value_wmma_config_binding_list.count,
        attention_key_value_wmma_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ATTENTION_KEY_VALUE_Q4_WMMA]);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16),
        IREE_SV("qwen3_moe_dense_linear_q6k_f16_wmma"),
        attention_key_value_wmma_config_binding_list.count,
        attention_key_value_wmma_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ATTENTION_VALUE_Q6_WMMA]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32),
        IREE_SV("qwen3_moe_router_projection_f32_four_row_wave32"),
        router_projection_config_binding_list.count,
        router_projection_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTER_TOP8_F32),
        IREE_SV("qwen3_moe_router_top8_f32"),
        router_top8_config_binding_list.count,
        router_top8_config_binding_list.bindings,
        IREE_ARRAYSIZE(router_top8_workload), router_top8_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTER_TOP8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32),
        IREE_SV("qwen3_moe_router_projection_top8_fused_decode_f32"),
        router_projection_top8_config_binding_list.count,
        router_projection_top8_config_binding_list.bindings,
        IREE_ARRAYSIZE(router_top8_workload), router_top8_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION_TOP8_FUSED]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_build_expert_table"),
        gate_up_config_binding_list.count, gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(expert_table_workload), expert_table_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_build_expert_partition_table"),
        gate_up_config_binding_list.count, gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(partition_table_workload), partition_table_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_PARTITION_TABLE]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma"),
        gate_up_config_binding_list.count, gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_GATE_UP]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8),
        IREE_SV("qwen3_moe_routed_gate_up_swiglu_q4k_q8"),
        gate_up_config_binding_list.count, gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(direct_gate_up_workload), direct_gate_up_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8),
        IREE_SV("qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8"),
        gate_up_config_binding_list.count, gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(direct_gate_up_workload), direct_gate_up_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32_NEXT_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        IREE_SV("qwen3_moe_routed_down_q4k_f16_wmma_grouped"),
        routed_down_config_binding_list.count,
        routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        IREE_SV("qwen3_moe_routed_down_q6k_f16_wmma_grouped"),
        routed_down_config_binding_list.count,
        routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8),
        IREE_SV("qwen3_moe_routed_down_q4k_q8_1_x4_next_q8"),
        direct_routed_down_config_binding_list.count,
        direct_routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(direct_routed_down_workload),
        direct_routed_down_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4_NEXT_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_F32),
        IREE_SV("qwen3_moe_routed_down_q6k_f32_wave64_next_q8"),
        direct_routed_down_config_binding_list.count,
        direct_routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(direct_routed_down_workload),
        direct_routed_down_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6_F32_NEXT_Q8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        IREE_SV("qwen3_moe_routed_down_weighted_reduce_f16_f32"),
        routed_down_config_binding_list.count,
        routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(token_workload), token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE]);
  }
  return status;
}

static iree_status_t qwen_program_prepare_full_model_executables(
    qwen_program_t* program, qwen_program_prepare_batch_t* batch) {
  const qwen_quantized_storage_t terminal_storage =
      qwen_model_parameter_layout(program->model)
          ->layers[QWEN_MODEL_LAYER_COUNT - 1]
          .value_and_down_storage;
  const iree_string_view_t terminal_routed_down_function =
      terminal_storage == QWEN_QUANTIZED_STORAGE_Q6_K
          ? IREE_SV("qwen3_moe_routed_down_q6k_f16_wmma_grouped")
          : IREE_SV("qwen3_moe_routed_down_q4k_f16_wmma_grouped");
  const int64_t terminal_token_count = 1;
  const int64_t route_stride = QWEN_MODEL_ROUTE_COUNT;
  const int64_t route_count = QWEN_MODEL_ROUTE_COUNT;
  const int64_t expert_count = QWEN_MODEL_EXPERT_COUNT;
  const int64_t embedding_workload[] = {
      (int64_t)program->token_count,
      QWEN_MODEL_VOCABULARY_SIZE,
  };
  const int64_t interlayer_token_workload[] = {
      (int64_t)program->token_count,
  };
  const int64_t expert_table_partition_workload[] = {
      (int64_t)program->token_count,
      route_count,
      route_stride,
      expert_count,
  };
  const int64_t terminal_token_workload[] = {terminal_token_count};
  const int64_t terminal_router_top8_workload[] = {
      terminal_token_count,
      route_stride,
  };
  const int64_t terminal_expert_table_workload[] = {
      terminal_token_count,
      route_count,
      route_stride,
      expert_count,
  };
  const int64_t terminal_partition_table_workload[] = {
      terminal_token_count,
      route_count,
      expert_count,
  };
  const int64_t vocabulary_projection_workload[] = {
      terminal_token_count,
      QWEN_MODEL_HIDDEN_SIZE,
      QWEN_MODEL_VOCABULARY_SIZE,
  };
  const int64_t vocabulary_argmax_workload[] = {
      QWEN_MODEL_VOCABULARY_PARTIAL_COUNT,
      (int64_t)program->token_count,
  };
  const int64_t route_trace_workload[] = {
      (int64_t)program->token_count,
      0,
      0,
      (int64_t)program->context_capacity,
  };
  const int64_t terminal_route_trace_workload[] = {
      terminal_token_count,
      (int64_t)program->token_count - 1,
      QWEN_MODEL_LAYER_COUNT - 1,
      (int64_t)program->context_capacity,
  };

  qwen_program_config_binding_list_t token_embedding_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      /*static_binding_count=*/0, /*static_bindings=*/NULL,
      (int64_t)program->token_count, &token_embedding_config_binding_list);

  qwen_program_config_binding_list_t vocabulary_projection_config_binding_list;
  qwen_program_config_binding_list_initialize(
      /*static_binding_count=*/0, /*static_bindings=*/NULL,
      &vocabulary_projection_config_binding_list);
  qwen_program_config_binding_list_append_index(
      &vocabulary_projection_config_binding_list,
      IREE_SV("ggml.linear_q6k_q8_1_x4.output_capacity"),
      QWEN_MODEL_VOCABULARY_SIZE);

  qwen_program_config_binding_list_t
      weighted_reduce_next_rmsnorm_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_weighted_reduce_next_rmsnorm_config_bindings),
      qwen_weighted_reduce_next_rmsnorm_config_bindings,
      (int64_t)program->token_count,
      &weighted_reduce_next_rmsnorm_config_binding_list);

  qwen_program_config_binding_list_t
      terminal_attention_prepare_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_attention_prepare_config_bindings),
      qwen_attention_prepare_config_bindings, terminal_token_count,
      &terminal_attention_prepare_config_binding_list);

  qwen_program_config_binding_list_t
      terminal_router_projection_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_router_projection_config_bindings),
      qwen_router_projection_config_bindings, terminal_token_count,
      &terminal_router_projection_config_binding_list);

  qwen_program_config_binding_list_t terminal_router_top8_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_router_top8_config_bindings),
      qwen_router_top8_config_bindings, terminal_token_count,
      &terminal_router_top8_config_binding_list);

  qwen_program_config_binding_list_t terminal_gate_up_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_gate_up_config_bindings),
      qwen_gate_up_config_bindings, terminal_token_count,
      &terminal_gate_up_config_binding_list);

  qwen_program_config_binding_list_t terminal_routed_down_config_binding_list;
  qwen_program_config_binding_list_initialize_with_token_capacity(
      IREE_ARRAYSIZE(qwen_routed_down_config_bindings),
      qwen_routed_down_config_bindings, terminal_token_count,
      &terminal_routed_down_config_binding_list);

  iree_status_t status = qwen_program_prepare_batch_append(
      batch, IREE_SV(QWEN_LOOM_SOURCE_TOKEN_EMBEDDING_Q4K),
      IREE_SV("qwen_token_embedding_q4k"),
      token_embedding_config_binding_list.count,
      token_embedding_config_binding_list.bindings,
      IREE_ARRAYSIZE(embedding_workload), embedding_workload,
      &program->executables[QWEN_PROGRAM_EXECUTABLE_TOKEN_EMBEDDING]);
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_NEXT_RMSNORM_F32),
        IREE_SV("qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32"),
        weighted_reduce_next_rmsnorm_config_binding_list.count,
        weighted_reduce_next_rmsnorm_config_binding_list.bindings,
        IREE_ARRAYSIZE(interlayer_token_workload), interlayer_token_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE_NEXT_RMSNORM]);
  }
  if (iree_status_is_ok(status) &&
      program->expert_table_schedule ==
          QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_FUSED_PREFILL_512) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_EXPERT_TABLE_PARTITION_FUSED),
        IREE_SV("qwen3_moe_build_expert_table_partition_prefill_512"),
        /*config_binding_count=*/0, /*config_bindings=*/NULL,
        IREE_ARRAYSIZE(expert_table_partition_workload),
        expert_table_partition_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE_PARTITION_FUSED]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED),
        IREE_SV("qwen3_moe_rmsnorm_f32"),
        terminal_attention_prepare_config_binding_list.count,
        terminal_attention_prepare_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_RMSNORM_F32]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32),
        IREE_SV("qwen3_moe_router_projection_f32_four_row_wave32"),
        terminal_router_projection_config_binding_list.count,
        terminal_router_projection_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_PROJECTION]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTER_TOP8_F32),
        IREE_SV("qwen3_moe_router_top8_f32"),
        terminal_router_top8_config_binding_list.count,
        terminal_router_top8_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_router_top8_workload),
        terminal_router_top8_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_TOP8]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_build_expert_table"),
        terminal_gate_up_config_binding_list.count,
        terminal_gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_expert_table_workload),
        terminal_expert_table_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_EXPERT_TABLE]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_build_expert_partition_table"),
        terminal_gate_up_config_binding_list.count,
        terminal_gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_partition_table_workload),
        terminal_partition_table_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_PARTITION_TABLE]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16),
        IREE_SV("qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma"),
        terminal_gate_up_config_binding_list.count,
        terminal_gate_up_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_GATE_UP]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        terminal_routed_down_function,
        terminal_routed_down_config_binding_list.count,
        terminal_routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTED_DOWN]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_F16),
        IREE_SV("qwen3_moe_routed_down_weighted_reduce_f16_f32"),
        terminal_routed_down_config_binding_list.count,
        terminal_routed_down_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_WEIGHTED_REDUCE]);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED),
        IREE_SV("qwen3_moe_attention_rmsnorm_quantize_q8_1_x4"),
        terminal_attention_prepare_config_binding_list.count,
        terminal_attention_prepare_config_binding_list.bindings,
        IREE_ARRAYSIZE(terminal_token_workload), terminal_token_workload,
        &program
             ->executables[QWEN_PROGRAM_EXECUTABLE_FINAL_RMSNORM_QUANTIZE_Q8]);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6),
        IREE_SV("ggml_linear_q6k_q8_1_x4_partial_argmax"),
        vocabulary_projection_config_binding_list.count,
        vocabulary_projection_config_binding_list.bindings,
        IREE_ARRAYSIZE(vocabulary_projection_workload),
        vocabulary_projection_workload,
        &program->executables
             [QWEN_PROGRAM_EXECUTABLE_VOCABULARY_PARTIAL_ARGMAX_Q6]);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_prepare_batch_append(
        batch,
        IREE_SV(QWEN_LOOM_SOURCE_GREEDY_ARGMAX_PARTIALS_BRINGUP_WORKAROUND),
        IREE_SV("qwen_greedy_argmax_partials_bringup_workaround"),
        /*config_binding_count=*/0, /*config_bindings=*/NULL,
        IREE_ARRAYSIZE(vocabulary_argmax_workload), vocabulary_argmax_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_GREEDY_ARGMAX_PARTIALS]);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(program->request_flags, QWEN_REQUEST_FLAG_ROUTE_TRACE)) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTE_TRACE),
        IREE_SV("qwen3_moe_route_trace_capture"),
        /*config_binding_count=*/0, /*config_bindings=*/NULL,
        IREE_ARRAYSIZE(route_trace_workload), route_trace_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_ROUTE_TRACE]);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(program->request_flags, QWEN_REQUEST_FLAG_ROUTE_TRACE)) {
    status = qwen_program_prepare_batch_append(
        batch, IREE_SV(QWEN_LOOM_SOURCE_ROUTE_TRACE),
        IREE_SV("qwen3_moe_route_trace_capture"),
        /*config_binding_count=*/0, /*config_bindings=*/NULL,
        IREE_ARRAYSIZE(terminal_route_trace_workload),
        terminal_route_trace_workload,
        &program->executables[QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTE_TRACE]);
  }
  return status;
}

static iree_hal_buffer_ref_t qwen_program_model_ref(
    qwen_parameter_span_t span) {
  return iree_hal_make_indirect_buffer_ref(QWEN_PROGRAM_BINDING_MODEL,
                                           span.offset, span.length);
}

static iree_hal_buffer_ref_t qwen_program_transient_ref(
    qwen_program_span_t span) {
  return iree_hal_make_indirect_buffer_ref(QWEN_PROGRAM_BINDING_TRANSIENT,
                                           span.offset, span.length);
}

static iree_hal_buffer_ref_t qwen_program_request_ref(
    qwen_request_span_t span) {
  return iree_hal_make_indirect_buffer_ref(QWEN_PROGRAM_BINDING_REQUEST_STATE,
                                           span.offset, span.length);
}

static iree_hal_buffer_ref_t qwen_program_output_staging_ref(
    iree_device_size_t length) {
  return iree_hal_make_indirect_buffer_ref(QWEN_PROGRAM_BINDING_OUTPUT_STAGING,
                                           /*offset=*/0, length);
}

static iree_status_t qwen_program_record_dispatch(
    qwen_program_t* program, qwen_program_executable_ordinal_t ordinal,
    iree_host_size_t constant_count, const uint32_t* constants,
    iree_host_size_t binding_count, const iree_hal_buffer_ref_t* bindings) {
  qwen_loom_executable_t* executable = program->executables[ordinal];
  const iree_const_byte_span_t constant_data = iree_make_const_byte_span(
      constants, constant_count * sizeof(constants[0]));
  const iree_hal_buffer_ref_list_t binding_list = {
      .count = binding_count,
      .values = bindings,
  };
  iree_status_t status = iree_hal_command_buffer_dispatch(
      program->command_buffer, qwen_loom_executable_hal_executable(executable),
      qwen_loom_executable_function(executable),
      qwen_loom_executable_dispatch_config(executable), constant_data,
      binding_list, IREE_HAL_DISPATCH_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    ++program->dispatch_count;
  }
  return status;
}

static iree_status_t qwen_program_record_dispatch_barrier(
    qwen_program_t* program) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      program->command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL);
}

static iree_status_t qwen_program_record_route_trace(
    qwen_program_t* program,
    qwen_program_executable_ordinal_t executable_ordinal,
    iree_host_size_t layer_index, uint32_t token_count,
    uint32_t source_token_offset,
    const qwen_request_storage_layout_t* request_layout,
    const qwen_layer_program_layout_t* transient) {
  if (!iree_any_bit_set(program->request_flags,
                        QWEN_REQUEST_FLAG_ROUTE_TRACE)) {
    return iree_ok_status();
  }
  const uint32_t constants[] = {
      token_count,
      source_token_offset,
      (uint32_t)layer_index,
      (uint32_t)program->context_capacity,
  };
  const iree_hal_buffer_ref_t bindings[] = {
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_transient_ref(transient->route_weights),
      qwen_program_request_ref(request_layout->control),
      qwen_program_request_ref(request_layout->route_trace.ids),
      qwen_program_request_ref(request_layout->route_trace.weights),
  };
  return qwen_program_record_dispatch(program, executable_ordinal,
                                      IREE_ARRAYSIZE(constants), constants,
                                      IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t qwen_program_record_attention_metadata(
    qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const uint32_t constants[] = {
      (uint32_t)program->token_count,
      (uint32_t)program->attention_context_count,
  };
  const iree_hal_buffer_ref_t bindings[] = {
      qwen_program_request_ref(request_layout->control),
      qwen_program_request_ref(request_layout->positions),
      qwen_program_request_ref(request_layout->key_cache_indices),
      qwen_program_request_ref(request_layout->value_cache_indices),
      qwen_program_request_ref(request_layout->attention_mask),
  };
  IREE_RETURN_IF_ERROR(qwen_program_record_dispatch(
      program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_METADATA,
      IREE_ARRAYSIZE(constants), constants, IREE_ARRAYSIZE(bindings),
      bindings));
  return qwen_program_record_dispatch_barrier(program);
}

static iree_status_t qwen_program_record_token_embedding(
    qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const uint32_t constants[] = {
      (uint32_t)program->token_count,
      QWEN_MODEL_VOCABULARY_SIZE,
  };
  const iree_hal_buffer_ref_t bindings[] = {
      qwen_program_request_ref(request_layout->token_ids),
      qwen_program_model_ref(
          qwen_model_parameter_layout(program->model)->token_embedding),
      qwen_program_request_ref(request_layout->hidden_state),
  };
  IREE_RETURN_IF_ERROR(qwen_program_record_dispatch(
      program, QWEN_PROGRAM_EXECUTABLE_TOKEN_EMBEDDING,
      IREE_ARRAYSIZE(constants), constants, IREE_ARRAYSIZE(bindings),
      bindings));
  return qwen_program_record_dispatch_barrier(program);
}

static iree_status_t qwen_program_record_attention(
    qwen_program_t* program, iree_host_size_t layer_index,
    iree_host_size_t cache_window_layer_index,
    const qwen_request_storage_layout_t* request_layout) {
  const qwen_parameter_layout_t* parameter_layout =
      qwen_model_parameter_layout(program->model);
  const qwen_layer_parameters_t* parameters =
      &parameter_layout->layers[layer_index];
  const qwen_layer_program_layout_t* transient = &program->layer_layout;
  const qwen_layer_program_layout_t* feed_forward_transient =
      qwen_program_kind_is_full_model(program->kind) &&
              layer_index == QWEN_MODEL_LAYER_COUNT - 1
          ? &program->full_layout.terminal_layer
          : transient;
  const bool uses_q6 =
      parameters->value_and_down_storage == QWEN_QUANTIZED_STORAGE_Q6_K;
  const qwen_program_executable_ordinal_t quantized_attention_qkv_ordinal =
      uses_q6 ? QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q6
              : QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_Q4;
  const qwen_program_executable_ordinal_t fused_attention_qkv_ordinal =
      uses_q6 ? QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q6
              : QWEN_PROGRAM_EXECUTABLE_ATTENTION_QKV_POSTPROCESS_FUSED_Q4;
  const qwen_program_executable_ordinal_t attention_prepare_ordinal =
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA
          ? QWEN_PROGRAM_EXECUTABLE_RMSNORM_F32
          : QWEN_PROGRAM_EXECUTABLE_ATTENTION_PREPARE_Q8;
  const bool input_was_published_by_previous_layer =
      program->attention_prepare_schedule ==
          QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_INTERLAYER &&
      layer_index > 0;
  const bool records_attention_prepare = !input_was_published_by_previous_layer;

  const uint32_t token_count = (uint32_t)program->token_count;
  const uint32_t attention_context_count =
      (uint32_t)program->attention_context_count;
  const iree_device_size_t layer_cache_offset =
      cache_window_layer_index * request_layout->layer_cache_byte_length;
  iree_status_t status = iree_ok_status();

  const uint32_t attention_prepare_constants[] = {token_count};
  const iree_hal_buffer_ref_t attention_prepare_bindings[] = {
      qwen_program_request_ref(request_layout->hidden_state),
      qwen_program_model_ref(parameters->attention_norm),
      qwen_program_transient_ref(transient->projection_input_scratch),
  };
  if (iree_status_is_ok(status) && records_attention_prepare) {
    status = qwen_program_record_dispatch(
        program, attention_prepare_ordinal,
        IREE_ARRAYSIZE(attention_prepare_constants),
        attention_prepare_constants, IREE_ARRAYSIZE(attention_prepare_bindings),
        attention_prepare_bindings);
  }
  if (iree_status_is_ok(status) && records_attention_prepare) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t attention_projection_constants[] = {token_count};
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    const iree_hal_buffer_ref_t attention_qkv_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->query),
        qwen_program_model_ref(parameters->key),
        qwen_program_model_ref(parameters->value),
        qwen_program_transient_ref(transient->raw_query),
        qwen_program_transient_ref(transient->raw_key),
        qwen_program_transient_ref(transient->raw_value),
    };
    status = qwen_program_record_dispatch(
        program, quantized_attention_qkv_ordinal,
        IREE_ARRAYSIZE(attention_projection_constants),
        attention_projection_constants, IREE_ARRAYSIZE(attention_qkv_bindings),
        attention_qkv_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_FUSED_QKV) {
    const uint32_t attention_qkv_constants[] = {
        token_count,
        attention_context_count,
    };
    const iree_hal_buffer_ref_t attention_qkv_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->query),
        qwen_program_model_ref(parameters->key),
        qwen_program_model_ref(parameters->value),
        qwen_program_request_ref(request_layout->positions),
        qwen_program_request_ref(request_layout->key_cache_indices),
        qwen_program_request_ref(request_layout->value_cache_indices),
        qwen_program_transient_ref(transient->raw_query),
        qwen_program_transient_ref(transient->raw_key),
        qwen_program_transient_ref(transient->raw_value),
        qwen_program_model_ref(parameters->query_norm),
        qwen_program_model_ref(parameters->key_norm),
        qwen_program_model_ref(parameter_layout->rope_inverse_frequencies),
        qwen_program_transient_ref(transient->rotated_query),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_KEY_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_VALUE_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        qwen_program_transient_ref(
            program->full_layout.decode_completion.grouped_stage),
    };
    status = qwen_program_record_dispatch(
        program, fused_attention_qkv_ordinal,
        IREE_ARRAYSIZE(attention_qkv_constants), attention_qkv_constants,
        IREE_ARRAYSIZE(attention_qkv_bindings), attention_qkv_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    const iree_hal_buffer_ref_t query_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->query),
        qwen_program_transient_ref(transient->raw_query),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_QUERY_Q4_WMMA,
        IREE_ARRAYSIZE(attention_projection_constants),
        attention_projection_constants, IREE_ARRAYSIZE(query_bindings),
        query_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    const iree_hal_buffer_ref_t key_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->key),
        qwen_program_transient_ref(transient->raw_key),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_KEY_VALUE_Q4_WMMA,
        IREE_ARRAYSIZE(attention_projection_constants),
        attention_projection_constants, IREE_ARRAYSIZE(key_bindings),
        key_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->qkv_schedule == QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA) {
    const qwen_program_executable_ordinal_t value_ordinal =
        uses_q6 ? QWEN_PROGRAM_EXECUTABLE_ATTENTION_VALUE_Q6_WMMA
                : QWEN_PROGRAM_EXECUTABLE_ATTENTION_KEY_VALUE_Q4_WMMA;
    const iree_hal_buffer_ref_t value_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->value),
        qwen_program_transient_ref(transient->raw_value),
    };
    status = qwen_program_record_dispatch(
        program, value_ordinal, IREE_ARRAYSIZE(attention_projection_constants),
        attention_projection_constants, IREE_ARRAYSIZE(value_bindings),
        value_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t attention_postprocess_constants[] = {
      token_count,
      attention_context_count,
  };
  const iree_hal_buffer_ref_t attention_postprocess_bindings[] = {
      qwen_program_request_ref(request_layout->positions),
      qwen_program_request_ref(request_layout->key_cache_indices),
      qwen_program_request_ref(request_layout->value_cache_indices),
      qwen_program_transient_ref(transient->raw_query),
      qwen_program_transient_ref(transient->raw_key),
      qwen_program_transient_ref(transient->raw_value),
      qwen_program_model_ref(parameters->query_norm),
      qwen_program_model_ref(parameters->key_norm),
      qwen_program_model_ref(parameter_layout->rope_inverse_frequencies),
      qwen_program_transient_ref(transient->rotated_query),
      iree_hal_make_indirect_buffer_ref(
          QWEN_PROGRAM_BINDING_KEY_CACHE, layer_cache_offset,
          request_layout->layer_cache_byte_length),
      iree_hal_make_indirect_buffer_ref(
          QWEN_PROGRAM_BINDING_VALUE_CACHE, layer_cache_offset,
          request_layout->layer_cache_byte_length),
  };
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_POSTPROCESS,
        IREE_ARRAYSIZE(attention_postprocess_constants),
        attention_postprocess_constants,
        IREE_ARRAYSIZE(attention_postprocess_bindings),
        attention_postprocess_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_postprocess_schedule ==
          QWEN_PROGRAM_ATTENTION_POSTPROCESS_SCHEDULE_SEPARATE) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  if (iree_status_is_ok(status) &&
      program->attention_schedule == QWEN_PROGRAM_ATTENTION_SCHEDULE_GENERAL) {
    const uint32_t flash_attention_constants[] = {
        token_count,
        attention_context_count,
    };
    const iree_hal_buffer_ref_t flash_attention_bindings[] = {
        qwen_program_transient_ref(transient->rotated_query),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_KEY_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_VALUE_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        qwen_program_request_ref(request_layout->attention_mask),
        qwen_program_transient_ref(transient->attention_output),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_FLASH_ATTENTION,
        IREE_ARRAYSIZE(flash_attention_constants), flash_attention_constants,
        IREE_ARRAYSIZE(flash_attention_bindings), flash_attention_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_schedule ==
          QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT) {
    const uint32_t flash_attention_constants[] = {
        attention_context_count,
    };
    const iree_hal_buffer_ref_t flash_attention_bindings[] = {
        qwen_program_transient_ref(transient->rotated_query),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_KEY_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        iree_hal_make_indirect_buffer_ref(
            QWEN_PROGRAM_BINDING_VALUE_CACHE, layer_cache_offset,
            request_layout->layer_cache_byte_length),
        qwen_program_request_ref(request_layout->attention_mask),
        qwen_program_transient_ref(
            program->full_layout.attention_partial_maximums),
        qwen_program_transient_ref(program->full_layout.attention_partial_sums),
        qwen_program_transient_ref(
            program->full_layout.attention_partial_outputs),
        qwen_program_transient_ref(
            program->full_layout.decode_completion.attention),
        qwen_program_transient_ref(transient->attention_output),
        qwen_program_transient_ref(transient->projection_input_scratch),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_FLASH_ATTENTION,
        IREE_ARRAYSIZE(flash_attention_constants), flash_attention_constants,
        IREE_ARRAYSIZE(flash_attention_bindings), flash_attention_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->attention_schedule !=
          QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT) {
    const uint32_t quantize_constants[] = {
        token_count,
        QWEN_MODEL_QUERY_HEAD_COUNT * QWEN_MODEL_HEAD_SIZE,
    };
    const iree_hal_buffer_ref_t quantize_bindings[] = {
        qwen_program_transient_ref(transient->attention_output),
        qwen_program_transient_ref(transient->projection_input_scratch),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_QUANTIZE_Q8,
        IREE_ARRAYSIZE(quantize_constants), quantize_constants,
        IREE_ARRAYSIZE(quantize_bindings), quantize_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->attention_schedule !=
          QWEN_PROGRAM_ATTENTION_SCHEDULE_DECODE_SPLIT) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t attention_output_constants[] = {token_count};
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    const iree_hal_buffer_ref_t attention_output_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->attention_output),
        qwen_program_request_ref(request_layout->hidden_state),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_DIRECT,
        IREE_ARRAYSIZE(attention_output_constants), attention_output_constants,
        IREE_ARRAYSIZE(attention_output_bindings), attention_output_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_DIRECT_Q8 &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    const iree_hal_buffer_ref_t attention_output_bindings[] = {
        qwen_program_transient_ref(transient->projection_input_scratch),
        qwen_program_model_ref(parameters->attention_output),
        qwen_program_request_ref(request_layout->hidden_state),
        qwen_program_model_ref(parameters->feed_forward_norm),
        qwen_program_transient_ref(feed_forward_transient->feed_forward_norm),
        qwen_program_transient_ref(
            program->full_layout.decode_completion.shared),
        qwen_program_transient_ref(
            feed_forward_transient->projection_input_scratch),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_Q4_NEXT_Q8,
        IREE_ARRAYSIZE(attention_output_constants), attention_output_constants,
        IREE_ARRAYSIZE(attention_output_bindings), attention_output_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->attention_output_schedule ==
          QWEN_PROGRAM_ATTENTION_OUTPUT_SCHEDULE_F32_WMMA) {
    const iree_hal_buffer_ref_t attention_output_bindings[] = {
        qwen_program_transient_ref(transient->attention_output),
        qwen_program_model_ref(parameters->attention_output),
        qwen_program_request_ref(request_layout->hidden_state),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ATTENTION_OUTPUT_F32_WMMA,
        IREE_ARRAYSIZE(attention_output_constants), attention_output_constants,
        IREE_ARRAYSIZE(attention_output_bindings), attention_output_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  return status;
}

static iree_status_t qwen_program_record_grouped_feed_forward(
    qwen_program_t* program, iree_host_size_t layer_index, uint32_t token_count,
    uint32_t source_token_offset,
    const qwen_request_storage_layout_t* request_layout,
    iree_hal_buffer_ref_t hidden_state,
    const qwen_layer_program_layout_t* transient,
    const qwen_program_grouped_feed_forward_executables_t* executables,
    qwen_program_expert_table_schedule_t expert_table_schedule,
    qwen_program_span_t expert_table_completion_counter,
    const qwen_program_next_attention_publication_t*
        next_attention_publication) {
  const qwen_layer_parameters_t* parameters =
      &qwen_model_parameter_layout(program->model)->layers[layer_index];
  const uint32_t route_count = QWEN_MODEL_ROUTE_COUNT;
  const uint32_t route_stride = QWEN_MODEL_ROUTE_COUNT;
  const uint32_t expert_count = QWEN_MODEL_EXPERT_COUNT;
  iree_status_t status = iree_ok_status();

  const uint32_t feed_forward_norm_constants[] = {token_count};
  const iree_hal_buffer_ref_t feed_forward_norm_bindings[] = {
      hidden_state,
      qwen_program_model_ref(parameters->feed_forward_norm),
      qwen_program_transient_ref(transient->feed_forward_norm),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, executables->rmsnorm,
        IREE_ARRAYSIZE(feed_forward_norm_constants),
        feed_forward_norm_constants, IREE_ARRAYSIZE(feed_forward_norm_bindings),
        feed_forward_norm_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t router_projection_constants[] = {token_count};
  const iree_hal_buffer_ref_t router_projection_bindings[] = {
      qwen_program_transient_ref(transient->feed_forward_norm),
      qwen_program_model_ref(parameters->router),
      qwen_program_transient_ref(transient->router_logits),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, executables->router_projection,
        IREE_ARRAYSIZE(router_projection_constants),
        router_projection_constants, IREE_ARRAYSIZE(router_projection_bindings),
        router_projection_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t router_top8_constants[] = {
      token_count,
      route_stride,
  };
  const iree_hal_buffer_ref_t router_top8_bindings[] = {
      qwen_program_transient_ref(transient->router_logits),
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_transient_ref(transient->route_weights),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, executables->router_top8,
        IREE_ARRAYSIZE(router_top8_constants), router_top8_constants,
        IREE_ARRAYSIZE(router_top8_bindings), router_top8_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }
  if (iree_status_is_ok(status)) {
    const qwen_program_executable_ordinal_t route_trace_ordinal =
        token_count == program->token_count
            ? QWEN_PROGRAM_EXECUTABLE_ROUTE_TRACE
            : QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTE_TRACE;
    status = qwen_program_record_route_trace(
        program, route_trace_ordinal, layer_index, token_count,
        source_token_offset, request_layout, transient);
  }

  const uint32_t expert_table_constants[] = {
      token_count,
      route_count,
      route_stride,
      expert_count,
  };
  if (expert_table_schedule ==
      QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_FUSED_PREFILL_512) {
    const iree_hal_buffer_ref_t expert_table_bindings[] = {
        qwen_program_transient_ref(transient->route_ids),
        qwen_program_transient_ref(transient->expert_table),
        qwen_program_transient_ref(transient->partition_table),
        qwen_program_transient_ref(expert_table_completion_counter),
    };
    if (iree_status_is_ok(status)) {
      status = qwen_program_record_dispatch(
          program, QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE_PARTITION_FUSED,
          IREE_ARRAYSIZE(expert_table_constants), expert_table_constants,
          IREE_ARRAYSIZE(expert_table_bindings), expert_table_bindings);
    }
  } else {
    const iree_hal_buffer_ref_t expert_table_bindings[] = {
        qwen_program_transient_ref(transient->route_ids),
        qwen_program_transient_ref(transient->expert_table),
    };
    if (iree_status_is_ok(status)) {
      status = qwen_program_record_dispatch(
          program, executables->expert_table,
          IREE_ARRAYSIZE(expert_table_constants), expert_table_constants,
          IREE_ARRAYSIZE(expert_table_bindings), expert_table_bindings);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_program_record_dispatch_barrier(program);
    }

    const uint32_t partition_table_constants[] = {
        token_count,
        route_count,
        expert_count,
    };
    const iree_hal_buffer_ref_t partition_table_bindings[] = {
        qwen_program_transient_ref(transient->expert_table),
        qwen_program_transient_ref(transient->partition_table),
    };
    if (iree_status_is_ok(status)) {
      status = qwen_program_record_dispatch(
          program, executables->partition_table,
          IREE_ARRAYSIZE(partition_table_constants), partition_table_constants,
          IREE_ARRAYSIZE(partition_table_bindings), partition_table_bindings);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t gate_up_constants[] = {token_count};
  const iree_hal_buffer_ref_t gate_up_bindings[] = {
      qwen_program_transient_ref(transient->feed_forward_norm),
      qwen_program_transient_ref(transient->expert_table),
      qwen_program_transient_ref(transient->partition_table),
      qwen_program_model_ref(parameters->expert_gate),
      qwen_program_model_ref(parameters->expert_up),
      qwen_program_transient_ref(transient->swiglu),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, executables->gate_up, IREE_ARRAYSIZE(gate_up_constants),
        gate_up_constants, IREE_ARRAYSIZE(gate_up_bindings), gate_up_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t routed_down_constants[] = {token_count};
  const iree_hal_buffer_ref_t routed_down_bindings[] = {
      qwen_program_transient_ref(transient->swiglu),
      qwen_program_transient_ref(transient->expert_table),
      qwen_program_model_ref(parameters->expert_down),
      qwen_program_transient_ref(transient->routed_projection_scratch),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, executables->routed_down,
        IREE_ARRAYSIZE(routed_down_constants), routed_down_constants,
        IREE_ARRAYSIZE(routed_down_bindings), routed_down_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t residual_publication_constants[] = {token_count};
  if (iree_status_is_ok(status) && next_attention_publication) {
    const iree_hal_buffer_ref_t residual_publication_bindings[] = {
        qwen_program_transient_ref(transient->route_weights),
        qwen_program_transient_ref(transient->routed_projection_scratch),
        hidden_state,
        qwen_program_model_ref(next_attention_publication->norm_weight),
        qwen_program_transient_ref(
            next_attention_publication->projection_input),
    };
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE_NEXT_RMSNORM,
        IREE_ARRAYSIZE(residual_publication_constants),
        residual_publication_constants,
        IREE_ARRAYSIZE(residual_publication_bindings),
        residual_publication_bindings);
  }
  if (iree_status_is_ok(status) && !next_attention_publication) {
    const iree_hal_buffer_ref_t residual_publication_bindings[] = {
        qwen_program_transient_ref(transient->route_weights),
        qwen_program_transient_ref(transient->routed_projection_scratch),
        hidden_state,
    };
    status = qwen_program_record_dispatch(
        program, executables->weighted_reduce,
        IREE_ARRAYSIZE(residual_publication_constants),
        residual_publication_constants,
        IREE_ARRAYSIZE(residual_publication_bindings),
        residual_publication_bindings);
  }
  return status;
}

static iree_status_t qwen_program_record_direct_feed_forward(
    qwen_program_t* program, iree_host_size_t layer_index, uint32_t token_count,
    uint32_t source_token_offset,
    const qwen_request_storage_layout_t* request_layout,
    iree_hal_buffer_ref_t hidden_state,
    const qwen_layer_program_layout_t* transient,
    qwen_parameter_span_t next_norm_weight,
    qwen_program_span_t gate_up_completion_counters,
    qwen_program_span_t shared_completion_counter,
    qwen_program_span_t next_q8_output) {
  const qwen_layer_parameters_t* parameters =
      &qwen_model_parameter_layout(program->model)->layers[layer_index];
  const bool uses_q6 =
      parameters->value_and_down_storage == QWEN_QUANTIZED_STORAGE_Q6_K;
  const qwen_program_executable_ordinal_t routed_down_ordinal =
      uses_q6 ? QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6_F32_NEXT_Q8
              : QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4_NEXT_Q8;
  const uint32_t route_count = QWEN_MODEL_ROUTE_COUNT;
  const uint32_t route_stride = QWEN_MODEL_ROUTE_COUNT;
  const uint32_t expert_count = QWEN_MODEL_EXPERT_COUNT;
  iree_status_t status = iree_ok_status();

  const uint32_t router_constants[] = {
      token_count,
      route_stride,
  };
  const iree_hal_buffer_ref_t router_bindings[] = {
      qwen_program_transient_ref(transient->feed_forward_norm),
      qwen_program_model_ref(parameters->router),
      qwen_program_transient_ref(transient->router_logits),
      qwen_program_transient_ref(shared_completion_counter),
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_transient_ref(transient->route_weights),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION_TOP8_FUSED,
        IREE_ARRAYSIZE(router_constants), router_constants,
        IREE_ARRAYSIZE(router_bindings), router_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }
  if (iree_status_is_ok(status)) {
    const qwen_program_executable_ordinal_t route_trace_ordinal =
        token_count == program->token_count
            ? QWEN_PROGRAM_EXECUTABLE_ROUTE_TRACE
            : QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTE_TRACE;
    status = qwen_program_record_route_trace(
        program, route_trace_ordinal, layer_index, token_count,
        source_token_offset, request_layout, transient);
  }

  const uint32_t gate_up_constants[] = {
      token_count,
      route_count,
      route_stride,
      expert_count,
      QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE,
  };
  const iree_hal_buffer_ref_t gate_up_f32_bindings[] = {
      qwen_program_transient_ref(transient->projection_input_scratch),
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_model_ref(parameters->expert_gate),
      qwen_program_model_ref(parameters->expert_up),
      qwen_program_transient_ref(transient->swiglu),
  };
  const iree_hal_buffer_ref_t gate_up_next_q8_bindings[] = {
      qwen_program_transient_ref(transient->projection_input_scratch),
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_model_ref(parameters->expert_gate),
      qwen_program_model_ref(parameters->expert_up),
      qwen_program_transient_ref(transient->swiglu),
      qwen_program_transient_ref(gate_up_completion_counters),
      qwen_program_transient_ref(transient->routed_projection_scratch),
  };
  if (iree_status_is_ok(status) && uses_q6) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32,
        IREE_ARRAYSIZE(gate_up_constants), gate_up_constants,
        IREE_ARRAYSIZE(gate_up_f32_bindings), gate_up_f32_bindings);
  }
  if (iree_status_is_ok(status) && !uses_q6) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_GATE_UP_Q8_TO_F32_NEXT_Q8,
        IREE_ARRAYSIZE(gate_up_constants), gate_up_constants,
        IREE_ARRAYSIZE(gate_up_next_q8_bindings), gate_up_next_q8_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t routed_down_constants[] = {
      token_count,  QWEN_MODEL_EXPERT_INTERMEDIATE_SIZE,
      route_count,  route_stride,
      expert_count, QWEN_MODEL_HIDDEN_SIZE,
  };
  const iree_hal_buffer_ref_t routed_down_input =
      uses_q6
          ? qwen_program_transient_ref(transient->swiglu)
          : qwen_program_transient_ref(transient->routed_projection_scratch);
  const iree_hal_buffer_ref_t routed_down_bindings[] = {
      routed_down_input,
      qwen_program_transient_ref(transient->route_ids),
      qwen_program_transient_ref(transient->route_weights),
      qwen_program_model_ref(parameters->expert_down),
      hidden_state,
      qwen_program_model_ref(next_norm_weight),
      qwen_program_transient_ref(shared_completion_counter),
      qwen_program_transient_ref(next_q8_output),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, routed_down_ordinal, IREE_ARRAYSIZE(routed_down_constants),
        routed_down_constants, IREE_ARRAYSIZE(routed_down_bindings),
        routed_down_bindings);
  }
  return status;
}

static iree_status_t qwen_program_record_layer_body(
    qwen_program_t* program, iree_host_size_t layer_index,
    iree_host_size_t cache_window_layer_index,
    const qwen_request_storage_layout_t* request_layout) {
  IREE_RETURN_IF_ERROR(qwen_program_record_attention(
      program, layer_index, cache_window_layer_index, request_layout));
  if (program->feed_forward_schedule ==
      QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    return qwen_program_record_direct_feed_forward(
        program, layer_index, (uint32_t)program->token_count,
        /*source_token_offset=*/0, request_layout,
        qwen_program_request_ref(request_layout->hidden_state),
        &program->layer_layout,
        qwen_model_parameter_layout(program->model)
            ->layers[layer_index + 1]
            .attention_norm,
        program->full_layout.decode_completion.grouped_stage,
        program->full_layout.decode_completion.shared,
        program->layer_layout.projection_input_scratch);
  }

  const bool uses_q6 =
      qwen_model_parameter_layout(program->model)
          ->layers[layer_index]
          .value_and_down_storage == QWEN_QUANTIZED_STORAGE_Q6_K;
  const qwen_program_grouped_feed_forward_executables_t executables = {
      .rmsnorm = QWEN_PROGRAM_EXECUTABLE_RMSNORM_F32,
      .router_projection = QWEN_PROGRAM_EXECUTABLE_ROUTER_PROJECTION,
      .router_top8 = QWEN_PROGRAM_EXECUTABLE_ROUTER_TOP8,
      .expert_table = QWEN_PROGRAM_EXECUTABLE_EXPERT_TABLE,
      .partition_table = QWEN_PROGRAM_EXECUTABLE_PARTITION_TABLE,
      .gate_up = QWEN_PROGRAM_EXECUTABLE_GATE_UP,
      .routed_down = uses_q6 ? QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q6
                             : QWEN_PROGRAM_EXECUTABLE_ROUTED_DOWN_Q4,
      .weighted_reduce = QWEN_PROGRAM_EXECUTABLE_WEIGHTED_REDUCE,
  };
  qwen_program_next_attention_publication_t next_attention_publication;
  const qwen_program_next_attention_publication_t*
      next_attention_publication_ptr = NULL;
  if (program->attention_prepare_schedule ==
      QWEN_PROGRAM_ATTENTION_PREPARE_SCHEDULE_INTERLAYER) {
    next_attention_publication = (qwen_program_next_attention_publication_t){
        .norm_weight = qwen_model_parameter_layout(program->model)
                           ->layers[layer_index + 1]
                           .attention_norm,
        .projection_input = program->layer_layout.projection_input_scratch,
    };
    next_attention_publication_ptr = &next_attention_publication;
  }
  return qwen_program_record_grouped_feed_forward(
      program, layer_index, (uint32_t)program->token_count,
      /*source_token_offset=*/0, request_layout,
      qwen_program_request_ref(request_layout->hidden_state),
      &program->layer_layout, &executables, program->expert_table_schedule,
      program->full_layout.expert_table_completion,
      next_attention_publication_ptr);
}

static qwen_request_span_t qwen_program_last_hidden_state_row(
    const qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const iree_device_size_t row_byte_length =
      QWEN_MODEL_HIDDEN_SIZE * sizeof(float);
  return (qwen_request_span_t){
      .offset = request_layout->hidden_state.offset +
                (program->token_count - 1) * row_byte_length,
      .length = row_byte_length,
  };
}

static iree_status_t qwen_program_record_terminal_feed_forward(
    qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const iree_host_size_t layer_index = QWEN_MODEL_LAYER_COUNT - 1;
  const iree_hal_buffer_ref_t hidden_state = qwen_program_request_ref(
      qwen_program_last_hidden_state_row(program, request_layout));
  if (program->feed_forward_schedule ==
      QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_DIRECT_Q8) {
    return qwen_program_record_direct_feed_forward(
        program, layer_index, /*token_count=*/1,
        /*source_token_offset=*/(uint32_t)(program->token_count - 1),
        request_layout, hidden_state, &program->full_layout.terminal_layer,
        qwen_model_parameter_layout(program->model)->output_norm,
        program->full_layout.decode_completion.grouped_stage,
        program->full_layout.decode_completion.shared,
        program->full_layout.final_quantized_hidden_state);
  }

  const qwen_program_grouped_feed_forward_executables_t executables = {
      .rmsnorm = QWEN_PROGRAM_EXECUTABLE_TERMINAL_RMSNORM_F32,
      .router_projection = QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_PROJECTION,
      .router_top8 = QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTER_TOP8,
      .expert_table = QWEN_PROGRAM_EXECUTABLE_TERMINAL_EXPERT_TABLE,
      .partition_table = QWEN_PROGRAM_EXECUTABLE_TERMINAL_PARTITION_TABLE,
      .gate_up = QWEN_PROGRAM_EXECUTABLE_TERMINAL_GATE_UP,
      .routed_down = QWEN_PROGRAM_EXECUTABLE_TERMINAL_ROUTED_DOWN,
      .weighted_reduce = QWEN_PROGRAM_EXECUTABLE_TERMINAL_WEIGHTED_REDUCE,
  };
  return qwen_program_record_grouped_feed_forward(
      program, layer_index, /*token_count=*/1,
      /*source_token_offset=*/(uint32_t)(program->token_count - 1),
      request_layout, hidden_state, &program->full_layout.terminal_layer,
      &executables, QWEN_PROGRAM_EXPERT_TABLE_SCHEDULE_SEPARATE,
      /*expert_table_completion_counter=*/(qwen_program_span_t){0},
      /*next_attention_publication=*/NULL);
}

static iree_status_t qwen_program_record_full_model_endpoint(
    qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const qwen_parameter_layout_t* parameters =
      qwen_model_parameter_layout(program->model);
  const qwen_full_program_layout_t* transient = &program->full_layout;
  const iree_hal_buffer_ref_t final_hidden_state = qwen_program_request_ref(
      qwen_program_last_hidden_state_row(program, request_layout));
  const uint32_t terminal_token_count = 1;
  iree_status_t status = iree_ok_status();

  const uint32_t final_prepare_constants[] = {terminal_token_count};
  const iree_hal_buffer_ref_t final_prepare_bindings[] = {
      final_hidden_state,
      qwen_program_model_ref(parameters->output_norm),
      qwen_program_transient_ref(transient->final_quantized_hidden_state),
  };
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_FINAL_RMSNORM_QUANTIZE_Q8,
        IREE_ARRAYSIZE(final_prepare_constants), final_prepare_constants,
        IREE_ARRAYSIZE(final_prepare_bindings), final_prepare_bindings);
  }
  if (iree_status_is_ok(status) &&
      program->feed_forward_schedule ==
          QWEN_PROGRAM_FEED_FORWARD_SCHEDULE_GROUPED_F16) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t vocabulary_projection_constants[] = {
      terminal_token_count,
      QWEN_MODEL_HIDDEN_SIZE,
      QWEN_MODEL_VOCABULARY_SIZE,
  };
  const iree_hal_buffer_ref_t vocabulary_projection_bindings[] = {
      qwen_program_transient_ref(transient->final_quantized_hidden_state),
      qwen_program_model_ref(parameters->output),
      qwen_program_transient_ref(transient->vocabulary_argmax.partial_logits),
      qwen_program_transient_ref(transient->vocabulary_argmax.partial_ids),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_VOCABULARY_PARTIAL_ARGMAX_Q6,
        IREE_ARRAYSIZE(vocabulary_projection_constants),
        vocabulary_projection_constants,
        IREE_ARRAYSIZE(vocabulary_projection_bindings),
        vocabulary_projection_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }

  const uint32_t argmax_constants[] = {
      QWEN_MODEL_VOCABULARY_PARTIAL_COUNT,
      (uint32_t)program->token_count,
  };
  const iree_hal_buffer_ref_t argmax_bindings[] = {
      qwen_program_transient_ref(transient->vocabulary_argmax.partial_logits),
      qwen_program_transient_ref(transient->vocabulary_argmax.partial_ids),
      qwen_program_request_ref(request_layout->token_ids),
      qwen_program_request_ref(request_layout->control),
      qwen_program_output_staging_ref(sizeof(int32_t)),
  };
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch(
        program, QWEN_PROGRAM_EXECUTABLE_GREEDY_ARGMAX_PARTIALS,
        IREE_ARRAYSIZE(argmax_constants), argmax_constants,
        IREE_ARRAYSIZE(argmax_bindings), argmax_bindings);
  }
  return status;
}

static iree_status_t qwen_program_record_hidden_state_copy(
    qwen_program_t* program,
    const qwen_request_storage_layout_t* request_layout) {
  const iree_hal_memory_barrier_t output_memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
      program->command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &output_memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL));
  qwen_request_span_t active_hidden_state = request_layout->hidden_state;
  active_hidden_state.length = program->output_staging_byte_length;
  return iree_hal_command_buffer_copy_buffer(
      program->command_buffer, qwen_program_request_ref(active_hidden_state),
      iree_hal_make_indirect_buffer_ref(QWEN_PROGRAM_BINDING_OUTPUT_STAGING,
                                        /*offset=*/0,
                                        active_hidden_state.length),
      IREE_HAL_COPY_FLAG_NONE);
}

static iree_status_t qwen_program_record_layer(qwen_program_t* program) {
  qwen_request_storage_layout_t request_layout;
  IREE_RETURN_IF_ERROR(qwen_request_storage_layout_calculate(
      program->token_capacity, program->context_capacity,
      program->request_flags, &request_layout));

  iree_status_t status = iree_hal_command_buffer_begin(program->command_buffer);
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_attention_metadata(program, &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_layer_body(program, program->layer_index,
                                            /*cache_window_layer_index=*/0,
                                            &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_hidden_state_copy(program, &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(program->command_buffer);
  }
  return status;
}

static iree_status_t qwen_program_record_full_model(qwen_program_t* program) {
  qwen_request_storage_layout_t request_layout;
  IREE_RETURN_IF_ERROR(qwen_request_storage_layout_calculate(
      program->token_capacity, program->context_capacity,
      program->request_flags, &request_layout));

  iree_status_t status = iree_hal_command_buffer_begin(program->command_buffer);
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_token_embedding(program, &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_attention_metadata(program, &request_layout);
  }
  for (iree_host_size_t layer_index = 0;
       iree_status_is_ok(status) && layer_index < QWEN_MODEL_LAYER_COUNT - 1;
       ++layer_index) {
    status = qwen_program_record_layer_body(
        program, layer_index, /*cache_window_layer_index=*/layer_index,
        &request_layout);
    if (iree_status_is_ok(status)) {
      status = qwen_program_record_dispatch_barrier(program);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_attention(
        program, QWEN_MODEL_LAYER_COUNT - 1,
        /*cache_window_layer_index=*/QWEN_MODEL_LAYER_COUNT - 1,
        &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_program_record_terminal_feed_forward(program, &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_dispatch_barrier(program);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_record_full_model_endpoint(program, &request_layout);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(program->command_buffer);
  }
  return status;
}

static iree_status_t qwen_program_validate_options(
    qwen_model_t* model, const qwen_program_options_t* options,
    qwen_program_attention_schedule_t* out_attention_schedule,
    iree_host_size_t* out_attention_context_count) {
  if (!model || !options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen model and program options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen program options structure is too small");
  }
  if (options->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen program option extensions are unsupported");
  }
  if (options->kind != QWEN_PROGRAM_KIND_LAYER &&
      options->kind != QWEN_PROGRAM_KIND_PREFILL &&
      options->kind != QWEN_PROGRAM_KIND_DECODE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen program kind %d is not implemented",
                            (int)options->kind);
  }
  if (options->kind == QWEN_PROGRAM_KIND_LAYER &&
      options->layer_index >= QWEN_MODEL_LAYER_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen layer index %" PRIhsz " is outside [0, %d)",
                            options->layer_index, QWEN_MODEL_LAYER_COUNT);
  }
  if (qwen_program_kind_is_full_model(options->kind) &&
      options->layer_index != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen full-model program layer index must remain zero");
  }
  if (!qwen_program_kind_is_full_model(options->kind) &&
      iree_any_bit_set(options->request_flags, QWEN_REQUEST_FLAG_ROUTE_TRACE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen route tracing requires a full-model "
                            "program");
  }
  qwen_request_storage_layout_t request_layout;
  IREE_RETURN_IF_ERROR(qwen_request_storage_layout_calculate(
      options->token_capacity, options->context_capacity,
      options->request_flags, &request_layout));
  if (options->context_count < options->token_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen program context count %" PRIhsz
                            " is less than token count %" PRIhsz,
                            options->context_count, options->token_count);
  }
  IREE_RETURN_IF_ERROR(qwen_request_active_shape_validate(
      options->token_capacity, options->context_capacity, options->token_count,
      options->context_count - options->token_count));
  if (options->kind == QWEN_PROGRAM_KIND_DECODE && options->token_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode programs require exactly one active token");
  }
  if (options->kind == QWEN_PROGRAM_KIND_DECODE &&
      (options->context_count < QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE ||
       options->context_count > QWEN_PROGRAM_DECODE_CONTEXT_LIMIT ||
       options->context_count % QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode context class must be a multiple of %d in [%d, %d]",
        QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE,
        QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE,
        QWEN_PROGRAM_DECODE_CONTEXT_LIMIT);
  }
  const qwen_program_attention_schedule_t attention_schedule =
      qwen_program_select_attention_schedule(options->kind);
  iree_host_size_t attention_context_count = options->context_count;
  // General attention consumes native 64-row K/V blocks. Metadata masks the
  // backed suffix while request shape and every non-attention stage retain the
  // exact logical context count.
  if (attention_schedule == QWEN_PROGRAM_ATTENTION_SCHEDULE_GENERAL &&
      !iree_host_size_checked_align(attention_context_count,
                                    QWEN_PROGRAM_ATTENTION_CONTEXT_ALIGNMENT,
                                    &attention_context_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen attention context alignment overflows");
  }
  if (attention_context_count > options->context_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen attention context count %" PRIhsz
                            " exceeds request context capacity %" PRIhsz,
                            attention_context_count, options->context_capacity);
  }
  const iree_hal_command_buffer_mode_t allowed_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  if (iree_any_bit_set(options->command_buffer_mode, ~allowed_mode)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen reusable programs accept only profiling metadata command-buffer "
        "mode bits");
  }
  if (qwen_program_kind_is_full_model(options->kind)) {
    qwen_full_program_layout_t layout;
    IREE_RETURN_IF_ERROR(qwen_full_program_layout_calculate(
        options->token_count, options->context_count,
        qwen_program_full_layout_flags(options->kind, options->token_count),
        &layout));
  } else {
    qwen_layer_program_layout_t layout;
    IREE_RETURN_IF_ERROR(
        qwen_layer_program_layout_calculate(options->token_count, &layout));
  }
  *out_attention_schedule = attention_schedule;
  *out_attention_context_count = attention_context_count;
  return iree_ok_status();
}

void qwen_program_options_initialize(qwen_program_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_program_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .kind = QWEN_PROGRAM_KIND_LAYER,
      .layer_index = 0,
      .token_count = 1,
      .context_count = 1,
      .token_capacity = 1,
      .context_capacity = 1,
      .request_flags = QWEN_REQUEST_FLAG_NONE,
      .command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
}

iree_host_size_t qwen_program_decode_context_class(
    iree_host_size_t context_base) {
  return (context_base / QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE + 1) *
         QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE;
}

static iree_status_t qwen_program_prepare_describe(
    qwen_model_t* model, const qwen_program_options_t* options,
    iree_allocator_t host_allocator,
    qwen_program_prepare_batch_t* prepare_batch, qwen_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;
  qwen_program_attention_schedule_t attention_schedule;
  iree_host_size_t attention_context_count;
  IREE_RETURN_IF_ERROR(qwen_program_validate_options(
      model, options, &attention_schedule, &attention_context_count));

  qwen_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*program),
                                             (void**)&program));
  memset(program, 0, sizeof(*program));
  iree_atomic_ref_count_init(&program->ref_count);
  program->host_allocator = host_allocator;
  program->model = model;
  qwen_model_retain(model);
  program->kind = options->kind;
  program->layer_index = options->layer_index;
  program->first_cache_layer =
      qwen_program_kind_is_full_model(options->kind) ? 0 : options->layer_index;
  program->cache_layer_count = qwen_program_kind_is_full_model(options->kind)
                                   ? QWEN_MODEL_LAYER_COUNT
                                   : 1;
  program->token_count = options->token_count;
  program->context_count = options->context_count;
  program->attention_context_count = attention_context_count;
  program->token_capacity = options->token_capacity;
  program->context_capacity = options->context_capacity;
  program->request_flags = options->request_flags;
  // Decode is validated to contain exactly one active token and uses the
  // row-quantized QKV path. Every multirow layer and prefill program uses the
  // F32/WMMA path.
  program->qkv_schedule = options->kind == QWEN_PROGRAM_KIND_DECODE
                              ? QWEN_PROGRAM_QKV_SCHEDULE_QUANTIZED_ROWS
                              : QWEN_PROGRAM_QKV_SCHEDULE_F32_WMMA;
  program->attention_schedule = attention_schedule;
  program->attention_postprocess_schedule =
      qwen_program_select_attention_postprocess_schedule(options->kind);
  program->attention_output_schedule =
      qwen_program_select_attention_output_schedule(options->token_count);
  program->attention_prepare_schedule =
      qwen_program_select_attention_prepare_schedule(options->kind);
  program->feed_forward_schedule =
      qwen_program_select_feed_forward_schedule(options->kind);
  program->expert_table_schedule = qwen_program_select_expert_table_schedule(
      options->kind, options->token_count);

  iree_status_t status = iree_ok_status();
  if (qwen_program_kind_is_full_model(options->kind)) {
    status = qwen_full_program_layout_calculate(
        options->token_count, options->context_count,
        qwen_program_full_layout_flags(options->kind, options->token_count),
        &program->full_layout);
    if (iree_status_is_ok(status)) {
      program->layer_layout = program->full_layout.layer;
      program->transient_byte_length =
          program->full_layout.transient_byte_length;
      program->output_staging_byte_length = sizeof(int32_t);
    }
  } else {
    status = qwen_layer_program_layout_calculate(options->token_count,
                                                 &program->layer_layout);
    if (iree_status_is_ok(status)) {
      program->transient_byte_length =
          program->layer_layout.transient_byte_length;
      status = qwen_model_hidden_state_byte_length(
          options->token_count, &program->output_staging_byte_length);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_reserve_semaphore_storage(
        QWEN_PROGRAM_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
        &program->wait_semaphores, &program->wait_values);
  }
  if (iree_status_is_ok(status)) {
    program->wait_capacity = QWEN_PROGRAM_INITIAL_SEMAPHORE_CAPACITY;
    status = qwen_program_reserve_semaphore_storage(
        QWEN_PROGRAM_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
        &program->signal_semaphores, &program->signal_values);
  }
  if (iree_status_is_ok(status)) {
    program->signal_capacity = QWEN_PROGRAM_INITIAL_SEMAPHORE_CAPACITY;
    status = iree_hal_semaphore_create(
        qwen_model_device(model), qwen_model_queue_affinity(model),
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &program->timeline_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_program_prepare_layer_executables(program, prepare_batch);
  }
  if (iree_status_is_ok(status) &&
      qwen_program_kind_is_full_model(options->kind)) {
    status =
        qwen_program_prepare_full_model_executables(program, prepare_batch);
  }

  if (iree_status_is_ok(status)) {
    *out_program = program;
  } else {
    qwen_program_destroy(program);
  }
  return status;
}

static iree_status_t qwen_program_prepare_finalize(
    qwen_program_t* program, const qwen_program_options_t* options) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      qwen_model_device(program->model), options->command_buffer_mode,
      qwen_program_command_categories(options->kind),
      qwen_model_queue_affinity(program->model), QWEN_PROGRAM_BINDING_COUNT,
      &program->command_buffer));
  return qwen_program_kind_is_full_model(options->kind)
             ? qwen_program_record_full_model(program)
             : qwen_program_record_layer(program);
}

iree_status_t qwen_program_prepare_batch(qwen_model_t* model,
                                         iree_host_size_t program_count,
                                         const qwen_program_options_t* options,
                                         iree_allocator_t host_allocator,
                                         qwen_program_t** out_programs) {
  IREE_ASSERT_ARGUMENT(out_programs);
  if (program_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen program prepare batch must not be empty");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen program prepare options are required");
  }
  for (iree_host_size_t i = 0; i < program_count; ++i) {
    out_programs[i] = NULL;
  }

  qwen_program_t** programs = NULL;
  qwen_program_prepare_batch_t* prepare_batches = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, program_count, sizeof(programs[0]), (void**)&programs);
  if (iree_status_is_ok(status)) {
    memset(programs, 0, program_count * sizeof(programs[0]));
    status = iree_allocator_malloc_array(host_allocator, program_count,
                                         sizeof(prepare_batches[0]),
                                         (void**)&prepare_batches);
  }
  if (iree_status_is_ok(status)) {
    memset(prepare_batches, 0, program_count * sizeof(prepare_batches[0]));
  }

  for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
       ++i) {
    prepare_batches[i].model = model;
    prepare_batches[i].host_allocator = host_allocator;
    status = qwen_program_prepare_describe(model, &options[i], host_allocator,
                                           &prepare_batches[i], &programs[i]);
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_program_prepare_batches_execute(program_count, prepare_batches);
  }
  for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
       ++i) {
    status = qwen_program_prepare_finalize(programs[i], &options[i]);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      out_programs[i] = programs[i];
      programs[i] = NULL;
    }
  }

  if (prepare_batches) {
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      qwen_program_prepare_batch_deinitialize(&prepare_batches[i]);
    }
  }
  if (programs) {
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      if (programs[i]) qwen_program_destroy(programs[i]);
    }
  }
  iree_allocator_free(host_allocator, prepare_batches);
  iree_allocator_free(host_allocator, programs);
  return status;
}

iree_status_t qwen_program_prepare(qwen_model_t* model,
                                   const qwen_program_options_t* options,
                                   iree_allocator_t host_allocator,
                                   qwen_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  return qwen_program_prepare_batch(model, 1, options, host_allocator,
                                    out_program);
}

void qwen_program_retain(qwen_program_t* program) {
  if (program) {
    iree_atomic_ref_count_inc(&program->ref_count);
  }
}

void qwen_program_release(qwen_program_t* program) {
  if (program && iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    qwen_program_destroy(program);
  }
}

static iree_status_t qwen_program_build_issue_wait_list(
    qwen_program_t* program, qwen_request_t* request,
    iree_hal_semaphore_list_t caller_waits,
    iree_hal_semaphore_list_t* out_waits) {
  const iree_hal_semaphore_list_t model_ready =
      qwen_model_ready_semaphore_list(program->model);
  if (caller_waits.count > IREE_HOST_SIZE_MAX - model_ready.count - 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue wait count overflows");
  }
  const iree_host_size_t wait_count =
      caller_waits.count + model_ready.count + 1;
  IREE_RETURN_IF_ERROR(qwen_program_ensure_semaphore_storage(
      wait_count, program->host_allocator, &program->wait_semaphores,
      &program->wait_values, &program->wait_capacity));

  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < model_ready.count; ++i) {
    program->wait_semaphores[cursor] = model_ready.semaphores[i];
    program->wait_values[cursor] = model_ready.payload_values[i];
    ++cursor;
  }
  program->wait_semaphores[cursor] = qwen_request_timeline_semaphore(request);
  program->wait_values[cursor] = qwen_request_timeline_value(request);
  ++cursor;
  for (iree_host_size_t i = 0; i < caller_waits.count; ++i) {
    program->wait_semaphores[cursor] = caller_waits.semaphores[i];
    program->wait_values[cursor] = caller_waits.payload_values[i];
    ++cursor;
  }
  *out_waits = (iree_hal_semaphore_list_t){
      .count = cursor,
      .semaphores = program->wait_semaphores,
      .payload_values = program->wait_values,
  };
  return iree_ok_status();
}

static iree_status_t qwen_program_build_issue_signal_list(
    qwen_program_t* program, qwen_request_t* request,
    uint64_t program_completion_value, uint64_t request_completion_value,
    iree_hal_semaphore_list_t caller_signals,
    iree_hal_semaphore_list_t* out_signals) {
  if (caller_signals.count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue signal count overflows");
  }
  const iree_host_size_t signal_count = caller_signals.count + 2;
  IREE_RETURN_IF_ERROR(qwen_program_ensure_semaphore_storage(
      signal_count, program->host_allocator, &program->signal_semaphores,
      &program->signal_values, &program->signal_capacity));

  program->signal_semaphores[0] = program->timeline_semaphore;
  program->signal_values[0] = program_completion_value;
  program->signal_semaphores[1] = qwen_request_timeline_semaphore(request);
  program->signal_values[1] = request_completion_value;
  for (iree_host_size_t i = 0; i < caller_signals.count; ++i) {
    program->signal_semaphores[i + 2] = caller_signals.semaphores[i];
    program->signal_values[i + 2] = caller_signals.payload_values[i];
  }
  *out_signals = (iree_hal_semaphore_list_t){
      .count = signal_count,
      .semaphores = program->signal_semaphores,
      .payload_values = program->signal_values,
  };
  return iree_ok_status();
}

static void qwen_program_fail_after_partial_submission(qwen_program_t* program,
                                                       qwen_request_t* request,
                                                       iree_status_t status) {
  qwen_request_fail(request, iree_status_clone(status));
  iree_hal_semaphore_fail(program->timeline_semaphore,
                          iree_status_clone(status));
}

iree_status_t qwen_program_issue(
    qwen_program_t* program, qwen_request_t* request,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!program || !request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen program and request are required");
  }
  if (qwen_request_model(request) != program->model ||
      qwen_request_token_capacity(request) != program->token_capacity ||
      qwen_request_context_capacity(request) != program->context_capacity ||
      qwen_request_flags(request) != program->request_flags) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen request model, storage capacities, and flags must match the "
        "prepared program");
  }
  const iree_host_size_t request_context_base =
      qwen_request_context_base(request);
  const iree_host_size_t decode_class_base =
      program->kind == QWEN_PROGRAM_KIND_DECODE
          ? program->context_count - QWEN_PROGRAM_DECODE_CONTEXT_CLASS_SIZE
          : 0;
  const bool active_shape_matches =
      program->kind == QWEN_PROGRAM_KIND_DECODE
          ? qwen_request_active_token_count(request) == 1 &&
                request_context_base >= decode_class_base &&
                request_context_base < program->context_count
          : qwen_request_active_token_count(request) == program->token_count &&
                request_context_base ==
                    program->context_count - program->token_count;
  if (!active_shape_matches) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen request active shape must fit the prepared program");
  }
  const qwen_request_input_kind_t expected_input_kind =
      qwen_program_kind_is_full_model(program->kind)
          ? QWEN_REQUEST_INPUT_KIND_TOKEN_IDS
          : QWEN_REQUEST_INPUT_KIND_HIDDEN_STATE;
  if (qwen_request_input_kind(request) != expected_input_kind) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen %s program requires a %s request reset",
        qwen_program_kind_is_full_model(program->kind) ? "full-model" : "layer",
        qwen_program_kind_is_full_model(program->kind) ? "token-ID"
                                                       : "hidden-state");
  }
  const qwen_program_span_t completion_counter_initialization =
      program->full_layout.completion_initialization;
  const bool initializes_completion_counters =
      completion_counter_initialization.length != 0;
  const uint64_t issue_timeline_step_count =
      initializes_completion_counters ? 4 : 3;
  if (program->timeline_value >
          IREE_HAL_SEMAPHORE_MAX_VALUE - issue_timeline_step_count ||
      qwen_request_timeline_value(request) == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen issue timeline is exhausted");
  }

  uint64_t completed_program_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(program->timeline_semaphore,
                                                &completed_program_value));
  if (completed_program_value < program->timeline_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen program already has an in-flight issue at value %" PRIu64,
        program->timeline_value);
  }

  iree_hal_semaphore_list_t alloca_waits;
  IREE_RETURN_IF_ERROR(qwen_program_build_issue_wait_list(
      program, request, wait_semaphore_list, &alloca_waits));

  uint64_t scratch_ready_value = program->timeline_value + 1;
  uint64_t execute_ready_value =
      program->timeline_value + (initializes_completion_counters ? 2 : 1);
  uint64_t execute_complete_value = execute_ready_value + 1;
  const uint64_t program_complete_value = execute_complete_value + 1;
  const uint64_t request_complete_value =
      qwen_request_timeline_value(request) + 1;

  iree_hal_semaphore_t* program_timeline = program->timeline_semaphore;
  iree_hal_semaphore_list_t scratch_ready_signals = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &scratch_ready_value,
  };
  iree_hal_semaphore_list_t scratch_ready_waits = scratch_ready_signals;
  iree_hal_semaphore_list_t execute_ready_signals = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &execute_ready_value,
  };
  iree_hal_semaphore_list_t execute_ready_waits = execute_ready_signals;
  iree_hal_semaphore_list_t execute_complete_signals = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &execute_complete_value,
  };
  iree_hal_semaphore_list_t execute_complete_waits = execute_complete_signals;

  iree_hal_semaphore_list_t completion_signals;
  IREE_RETURN_IF_ERROR(qwen_program_build_issue_signal_list(
      program, request, program_complete_value, request_complete_value,
      signal_semaphore_list, &completion_signals));

  iree_hal_device_t* device = qwen_model_device(program->model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(program->model);
  const iree_hal_buffer_params_t scratch_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               (initializes_completion_counters
                    ? IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET
                    : IREE_HAL_BUFFER_USAGE_NONE),
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = queue_affinity,
      .min_alignment = 64,
  };
  iree_hal_buffer_t* scratch_buffer = NULL;
  iree_status_t status = iree_hal_device_queue_alloca(
      device, queue_affinity, alloca_waits, scratch_ready_signals,
      /*pool=*/NULL, scratch_params, program->transient_byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &scratch_buffer);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (initializes_completion_counters) {
    const uint32_t zero_pattern = 0;
    status = iree_hal_device_queue_fill(
        device, queue_affinity, scratch_ready_waits, execute_ready_signals,
        scratch_buffer, completion_counter_initialization.offset,
        completion_counter_initialization.length, &zero_pattern,
        sizeof(zero_pattern), IREE_HAL_FILL_FLAG_NONE);
    if (!iree_status_is_ok(status)) {
      iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
          device, queue_affinity, scratch_ready_waits,
          iree_hal_semaphore_list_empty(), scratch_buffer,
          IREE_HAL_DEALLOCA_FLAG_NONE);
      iree_hal_buffer_release(scratch_buffer);
      qwen_program_fail_after_partial_submission(program, request, status);
      return iree_status_join(status, cleanup_status);
    }
  }

  const qwen_request_storage_layout_t* request_layout =
      qwen_request_storage_layout(request);
  const iree_device_size_t cache_binding_offset =
      program->first_cache_layer * request_layout->layer_cache_byte_length;
  const iree_device_size_t cache_binding_length =
      program->cache_layer_count * request_layout->layer_cache_byte_length;
  const qwen_model_statistics_t model_statistics =
      qwen_model_statistics(program->model);
  const iree_hal_buffer_binding_t bindings[QWEN_PROGRAM_BINDING_COUNT] = {
      [QWEN_PROGRAM_BINDING_MODEL] =
          {
              .buffer = qwen_model_parameter_buffer(program->model),
              .offset = 0,
              .length = model_statistics.allocation_bytes,
          },
      [QWEN_PROGRAM_BINDING_TRANSIENT] =
          {
              .buffer = scratch_buffer,
              .offset = 0,
              .length = program->transient_byte_length,
          },
      [QWEN_PROGRAM_BINDING_KEY_CACHE] =
          {
              .buffer = qwen_request_storage_buffer(request),
              .offset = request_layout->key_cache.offset + cache_binding_offset,
              .length = cache_binding_length,
          },
      [QWEN_PROGRAM_BINDING_VALUE_CACHE] =
          {
              .buffer = qwen_request_storage_buffer(request),
              .offset =
                  request_layout->value_cache.offset + cache_binding_offset,
              .length = cache_binding_length,
          },
      [QWEN_PROGRAM_BINDING_REQUEST_STATE] =
          {
              .buffer = qwen_request_storage_buffer(request),
              .offset = 0,
              .length = request_layout->dispatch_state_byte_length,
          },
      [QWEN_PROGRAM_BINDING_OUTPUT_STAGING] =
          {
              .buffer = qwen_request_output_staging_buffer(request),
              .offset = 0,
              .length = program->output_staging_byte_length,
          },
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  status = iree_hal_device_queue_execute(
      device, queue_affinity, execute_ready_waits, execute_complete_signals,
      program->command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, execute_ready_waits,
        iree_hal_semaphore_list_empty(), scratch_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_buffer_release(scratch_buffer);
    qwen_program_fail_after_partial_submission(program, request, status);
    return iree_status_join(status, cleanup_status);
  }

  status = iree_hal_device_queue_dealloca(
      device, queue_affinity, execute_complete_waits, completion_signals,
      scratch_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  iree_hal_buffer_release(scratch_buffer);
  if (!iree_status_is_ok(status)) {
    qwen_program_fail_after_partial_submission(program, request, status);
    return status;
  }

  program->timeline_value = program_complete_value;
  if (qwen_program_kind_is_full_model(program->kind)) {
    const iree_host_size_t next_context_base =
        request_context_base + program->token_count;
    qwen_request_commit_selected_token_signal(request, request_complete_value,
                                              next_context_base);
  } else {
    qwen_request_commit_hidden_state_signal(request, request_complete_value);
  }
  return iree_ok_status();
}

iree_host_size_t qwen_program_token_count(const qwen_program_t* program) {
  return program ? program->token_count : 0;
}

iree_host_size_t qwen_program_dispatch_count(const qwen_program_t* program) {
  return program ? program->dispatch_count : 0;
}

iree_device_size_t qwen_program_transient_byte_length(
    const qwen_program_t* program) {
  return program ? program->transient_byte_length : 0;
}
