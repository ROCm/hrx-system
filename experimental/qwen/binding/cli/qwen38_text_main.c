// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/programs/qwen38_text_model_source.h"
#include "experimental/qwen/tooling/command_program.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "loomc/target/cmd/hal.h"

#define QWEN38_PREFILL_CHUNK_CAPACITY 512
#define QWEN38_GENERATION_CAPACITY 128
#define QWEN38_MAX_CONTEXT_CAPACITY 262144
#define QWEN38_HIDDEN_BYTE_LENGTH 20480
#define QWEN38_HIDDEN_CAPACITY_BYTE_LENGTH 10485760
#define QWEN38_GDN_STATE_BYTE_LENGTH 156893184
#define QWEN38_ATTENTION_CACHE_BYTES_PER_TOKEN 65536
#define QWEN38_MTP_ATTENTION_CACHE_BYTES_PER_TOKEN 4096
#define QWEN38_TARGET_CONTROL_WORD_COUNT 3
#define QWEN38_TARGET_CONTROL_BYTE_LENGTH \
  (QWEN38_TARGET_CONTROL_WORD_COUNT * sizeof(int32_t))
#define QWEN38_PAIR_CONTROL_BYTE_LENGTH (2 * sizeof(int32_t))
#define QWEN38_MTP_DRAFT_DEPTH 4
#define QWEN38_MTP_PROGRESS_WORD_COUNT 8
#define QWEN38_MTP_PROGRESS_BYTE_LENGTH \
  (QWEN38_MTP_PROGRESS_WORD_COUNT * sizeof(int32_t))
#define QWEN38_DECODE_TEXT_CAPACITY 65536
#define QWEN38_PREFILL_SCHEDULE_COUNT 4

enum qwen38_progress_word_e {
  QWEN38_PROGRESS_GENERATED_COUNT = 0,
  QWEN38_PROGRESS_ACCEPTED_DRAFT_COUNT = 1,
  QWEN38_PROGRESS_DEPTH_HISTOGRAM = 2,
  QWEN38_PROGRESS_DONE = 7,
};

typedef enum qwen38_prefill_schedule_e {
  QWEN38_PREFILL_SCHEDULE_C64T32 = 0,
  QWEN38_PREFILL_SCHEDULE_X64 = 1,
  QWEN38_PREFILL_SCHEDULE_X128 = 2,
  QWEN38_PREFILL_SCHEDULE_C128 = 3,
} qwen38_prefill_schedule_t;

IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, prompt,
          "Explain why reproducible benchmarks need controlled clocks in "
          "three short paragraphs.",
          "User prompt to tokenize and run.");
IREE_FLAG(string, prompt_file, "",
          "Optional prompt file path, overriding --prompt.");
IREE_FLAG(string, second_prompt, "",
          "Optional second user turn executed in the same model session.");
IREE_FLAG(bool, raw_prompt, false,
          "Treat --prompt as an already formatted model prompt.");
IREE_FLAG(int32_t, max_tokens, 16,
          "Maximum number of greedy output tokens, from 1 through 128.");
IREE_FLAG(int32_t, context_capacity, 4096,
          "Token and KV-cache capacity, from 1 through 262144.");
IREE_FLAG(bool, mtp, true,
          "Enable the model's depth-4 MTP proposal and verification path.");
IREE_FLAG(bool, verbose, false,
          "Print prompt token IDs and generation progress diagnostics.");
IREE_FLAG(string, residual_output, "",
          "Optional raw F32 final residual-row output path.");
IREE_FLAG(string, hidden_output, "",
          "Optional raw F32 final normalized-hidden output path.");

static const char* const qwen38_text_usage =
    "Runs Qwen3.8 text prefill and greedy MTP generation through reusable "
    "Loom command programs sharing one model parameter pack.\n"
    "\n"
    "Required flags:\n"
    "  --device=<AMDGPU device URI>\n"
    "  --parameters=<Qwen3.8-27B UD-Q5_K_XL GGUF path>\n"
    "  --tokenizer=<Qwen3.8 tokenizer.json path>\n"
    "\n"
    "Input and generation:\n"
    "  --prompt=<user text>\n"
    "  --prompt_file=<path to prompt text>\n"
    "  --second_prompt=<second user turn>\n"
    "  --raw_prompt=true to bypass the no-thinking chat wrapper\n"
    "  --max_tokens=<greedy output count from 1 through 128>\n"
    "  --context_capacity=<prompt and generation capacity>\n"
    "\n"
    "Optional diagnostic output:\n"
    "  --residual_output=<raw 5120-element F32 final residual row>\n"
    "  --hidden_output=<raw 5120-element F32 final normalized hidden row>\n";

static iree_status_t qwen38_allocate_buffer(
    iree_hal_device_t* device, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t usage, iree_device_size_t minimum_alignment,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = minimum_alignment,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t qwen38_fill_and_wait(iree_hal_device_t* device,
                                          iree_hal_buffer_t* buffer) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  const uint32_t zero = 0;
  iree_status_t status = iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signals, buffer, 0, iree_hal_buffer_byte_length(buffer), &zero,
      sizeof(zero), IREE_HAL_FILL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static iree_status_t qwen38_queue_program(
    iree_hal_device_t* device, iree_hal_semaphore_t* execution_semaphore,
    uint64_t* inout_execution_value, qwen_tooling_command_program_t* program,
    iree_hal_buffer_binding_table_t binding_table) {
  uint64_t wait_value = *inout_execution_value;
  uint64_t signal_value = wait_value + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, waits, signals,
      qwen_tooling_command_program_command_buffer(program), binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) *inout_execution_value = signal_value;
  return status;
}

static iree_status_t qwen38_queue_update(
    iree_hal_device_t* device, iree_hal_semaphore_t* execution_semaphore,
    uint64_t* inout_execution_value, const void* source_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  uint64_t wait_value = *inout_execution_value;
  uint64_t signal_value = wait_value + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_update(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, waits, signals, source_buffer,
      /*source_offset=*/0, target_buffer, target_offset, length,
      IREE_HAL_UPDATE_FLAG_NONE);
  if (iree_status_is_ok(status)) *inout_execution_value = signal_value;
  return status;
}

static qwen38_prefill_schedule_t qwen38_prefill_schedule_for_token_count(
    iree_host_size_t token_count) {
  if (token_count == 32) return QWEN38_PREFILL_SCHEDULE_C64T32;
  if (token_count <= 64) return QWEN38_PREFILL_SCHEDULE_X64;
  if (token_count <= 480) return QWEN38_PREFILL_SCHEDULE_X128;
  return QWEN38_PREFILL_SCHEDULE_C128;
}

static iree_string_view_t qwen38_prefill_root_name_for_schedule(
    bool mtp, qwen38_prefill_schedule_t schedule) {
  if (mtp) {
    switch (schedule) {
      case QWEN38_PREFILL_SCHEDULE_C64T32:
        return IREE_SV("qwen38_text_prefill_mtp_greedy_schedule_c64t32");
      case QWEN38_PREFILL_SCHEDULE_X64:
        return IREE_SV("qwen38_text_prefill_mtp_greedy_schedule_x64");
      case QWEN38_PREFILL_SCHEDULE_X128:
        return IREE_SV("qwen38_text_prefill_mtp_greedy_schedule_x128");
      case QWEN38_PREFILL_SCHEDULE_C128:
        return IREE_SV("qwen38_text_prefill_mtp_greedy_schedule_c128");
    }
  }
  switch (schedule) {
    case QWEN38_PREFILL_SCHEDULE_C64T32:
      return IREE_SV(
          "qwen38_text_prefill_greedy_with_hidden_schedule_c64t32");
    case QWEN38_PREFILL_SCHEDULE_X64:
      return IREE_SV("qwen38_text_prefill_greedy_with_hidden_schedule_x64");
    case QWEN38_PREFILL_SCHEDULE_X128:
      return IREE_SV("qwen38_text_prefill_greedy_with_hidden_schedule_x128");
    case QWEN38_PREFILL_SCHEDULE_C128:
      return IREE_SV("qwen38_text_prefill_greedy_with_hidden_schedule_c128");
  }
  return iree_string_view_empty();
}

static iree_string_view_t qwen38_chunk_root_name(bool mtp) {
  return mtp ? IREE_SV("qwen38_text_prefill_mtp_chunk")
             : IREE_SV("qwen38_text_prefill_chunk");
}

static iree_status_t qwen38_prepare_text_model(
    qwen_tooling_runtime_context_t* runtime_context, bool mtp,
    uint32_t prefill_schedule_mask,
    iree_string_view_t context_capacity, iree_allocator_t host_allocator,
    qwen_tooling_command_program_set_t** out_program_set) {
  if (qwen38_text_model_source_size() != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen3.8 text model source must contain exactly one file");
  }
  const iree_file_toc_t* files = qwen38_text_model_source_create();
  iree_string_view_t root_names[8];
  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < QWEN38_PREFILL_SCHEDULE_COUNT; ++i) {
    if ((prefill_schedule_mask & (1u << i)) == 0) continue;
    root_names[root_count++] = qwen38_prefill_root_name_for_schedule(
        mtp, (qwen38_prefill_schedule_t)i);
  }
  root_names[root_count++] = qwen38_chunk_root_name(mtp);
  root_names[root_count++] = IREE_SV("qwen38_text_decode_greedy_with_hidden");
  if (mtp) {
    root_names[root_count++] = IREE_SV("qwen38_mtp_propose_depth4");
    root_names[root_count++] = IREE_SV("qwen38_mtp_target_verify_depth4");
  }
  const loomc_config_binding_t config_bindings[] = {{
      .key = loomc_make_cstring_view("qwen38.attention.cache_capacity"),
      .value = loomc_make_string_view(context_capacity.data,
                                      context_capacity.size),
  }};
  const qwen_tooling_command_program_set_options_t options = {
      .source_identifier = iree_make_cstring_view(files[0].name),
      .source_contents =
          iree_make_const_byte_span(files[0].data, files[0].size),
      .root_names = root_names,
      .root_count = root_count,
      .config =
          {
              .bindings = config_bindings,
              .binding_count = IREE_ARRAYSIZE(config_bindings),
              .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
          },
  };
  return qwen_tooling_command_program_set_create(
      runtime_context, &options, host_allocator, out_program_set);
}

static iree_status_t qwen38_build_prompt(iree_string_view_t prompt,
                                         bool raw_prompt,
                                         iree_string_builder_t* out_builder) {
  if (raw_prompt) {
    return iree_string_builder_append_string(out_builder, prompt);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(out_builder, "<|im_start|>user\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(out_builder, prompt));
  return iree_string_builder_append_cstring(
      out_builder,
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

// Builds the next span from the pending assistant EOS through the next
// assistant-generation boundary. The target state intentionally has not
// consumed the EOS token yet, so it is the first input of every appended turn.
static iree_status_t qwen38_build_followup_span(
    iree_string_view_t prompt, iree_string_builder_t* out_builder) {
  iree_string_builder_reset(out_builder);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(out_builder, "<|im_end|>\n"));
  return qwen38_build_prompt(prompt, /*raw_prompt=*/false, out_builder);
}

static double qwen38_elapsed_ms(iree_time_t start, iree_time_t end) {
  return (double)(end - start) / 1000000.0;
}

static double qwen38_rate(iree_host_size_t item_count, iree_time_t start,
                          iree_time_t end) {
  return end != start
             ? (double)item_count * 1000000000.0 / (double)(end - start)
             : 0.0;
}

typedef struct qwen38_generation_stats_t {
  // Total output tokens committed during the turn, including the first token.
  int32_t generated_count;
  // Number of completed depth-four MTP verification cycles.
  int32_t mtp_cycle_count;
  // Number of outputs after the first token committed by MTP cycles.
  int32_t mtp_committed_count;
  // Number of matching draft tokens committed by MTP cycles.
  int32_t mtp_accepted_draft_count;
  // MTP cycle counts indexed by the number of matching committed drafts.
  int32_t mtp_depth_counts[QWEN38_MTP_DRAFT_DEPTH + 1];
  // True when the final committed output is the configured EOS token.
  bool done;
} qwen38_generation_stats_t;

// Advances one turn after final prefill has published its first output.
//
// The reusable command buffers retain all executable and parameter resources.
// The host observes only the small coherent progress record after each MTP
// cycle or ordinary transition so no work can be issued beyond EOS.
static iree_status_t qwen38_generate_output(
    iree_hal_device_t* device, iree_hal_semaphore_t* execution_semaphore,
    uint64_t* inout_execution_value, bool mtp, int32_t max_tokens,
    qwen_tooling_command_program_t* decode_program,
    iree_hal_buffer_binding_table_t decode_binding_table,
    qwen_tooling_command_program_t* propose_program,
    iree_hal_buffer_binding_table_t propose_binding_table,
    qwen_tooling_command_program_t* verify_program,
    iree_hal_buffer_binding_table_t verify_binding_table,
    const iree_hal_buffer_mapping_t* progress_mapping,
    qwen38_generation_stats_t* out_stats) {
  qwen38_generation_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  const uint8_t* progress_data = progress_mapping->contents.data;
  stats.generated_count = (int32_t)iree_unaligned_load_le_u32(
      progress_data + QWEN38_PROGRESS_GENERATED_COUNT * sizeof(uint32_t));
  stats.done =
      iree_unaligned_load_le_u32(
          progress_data + QWEN38_PROGRESS_DONE * sizeof(uint32_t)) != 0;
  if (stats.generated_count != 1) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "prefill published %" PRId32 " generated tokens",
        stats.generated_count);
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && mtp && !stats.done &&
         max_tokens - stats.generated_count >= QWEN38_MTP_DRAFT_DEPTH) {
    const int32_t prior_generated_count = stats.generated_count;
    status = qwen38_queue_program(device, execution_semaphore,
                                  inout_execution_value, propose_program,
                                  propose_binding_table);
    if (iree_status_is_ok(status)) {
      status = qwen38_queue_program(device, execution_semaphore,
                                    inout_execution_value, verify_program,
                                    verify_binding_table);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(execution_semaphore,
                                       *inout_execution_value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      const int32_t next_generated_count =
          (int32_t)iree_unaligned_load_le_u32(
              progress_data +
              QWEN38_PROGRESS_GENERATED_COUNT * sizeof(uint32_t));
      const int32_t next_accepted_draft_count =
          (int32_t)iree_unaligned_load_le_u32(
              progress_data +
              QWEN38_PROGRESS_ACCEPTED_DRAFT_COUNT * sizeof(uint32_t));
      stats.done =
          iree_unaligned_load_le_u32(
              progress_data + QWEN38_PROGRESS_DONE * sizeof(uint32_t)) != 0;
      const int32_t maximum_accepted_draft_count =
          (stats.mtp_cycle_count + 1) * QWEN38_MTP_DRAFT_DEPTH;
      if (next_generated_count < prior_generated_count + 1 ||
          next_generated_count >
              prior_generated_count + QWEN38_MTP_DRAFT_DEPTH ||
          next_generated_count > max_tokens ||
          next_accepted_draft_count < stats.mtp_accepted_draft_count ||
          next_accepted_draft_count > maximum_accepted_draft_count) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "MTP cycle advanced from %" PRId32 " to %" PRId32
            " outputs with %" PRId32 " cumulative accepted drafts",
            prior_generated_count, next_generated_count,
            next_accepted_draft_count);
      } else {
        stats.generated_count = next_generated_count;
        ++stats.mtp_cycle_count;
        stats.mtp_committed_count = stats.generated_count - 1;
        stats.mtp_accepted_draft_count = next_accepted_draft_count;
        int32_t observed_cycle_count = 0;
        for (int32_t depth = 0; depth <= QWEN38_MTP_DRAFT_DEPTH; ++depth) {
          stats.mtp_depth_counts[depth] =
              (int32_t)iree_unaligned_load_le_u32(
                  progress_data +
                  (QWEN38_PROGRESS_DEPTH_HISTOGRAM + depth) *
                      sizeof(uint32_t));
          observed_cycle_count += stats.mtp_depth_counts[depth];
        }
        if (observed_cycle_count != stats.mtp_cycle_count) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "MTP depth histogram accounts for %" PRId32 " of %" PRId32
              " completed cycles",
              observed_cycle_count, stats.mtp_cycle_count);
        }
      }
    }
  }

  while (iree_status_is_ok(status) && !stats.done &&
         stats.generated_count < max_tokens) {
    const int32_t prior_generated_count = stats.generated_count;
    status = qwen38_queue_program(device, execution_semaphore,
                                  inout_execution_value, decode_program,
                                  decode_binding_table);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(execution_semaphore,
                                       *inout_execution_value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      stats.generated_count = (int32_t)iree_unaligned_load_le_u32(
          progress_data +
          QWEN38_PROGRESS_GENERATED_COUNT * sizeof(uint32_t));
      stats.done =
          iree_unaligned_load_le_u32(
              progress_data + QWEN38_PROGRESS_DONE * sizeof(uint32_t)) != 0;
      if (stats.generated_count != prior_generated_count + 1) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "decode transition advanced from %" PRId32 " to %" PRId32
            " outputs",
            prior_generated_count, stats.generated_count);
      }
    }
  }

  if (iree_status_is_ok(status)) *out_stats = stats;
  return status;
}

typedef struct qwen38_turn_result_t {
  // Target position of the pending final output token.
  int32_t pending_position;
  // Number of token IDs available in the decoded output span.
  iree_host_size_t visible_token_count;
  // Number of bytes written to the caller's decoded-text storage.
  iree_host_size_t decoded_text_length;
} qwen38_turn_result_t;

// Reads the bounded user-visible result after a turn-completion wait.
//
// This transfer is the turn output boundary; generation never performs a
// token transfer while deciding MTP acceptance or EOS.
static iree_status_t qwen38_read_turn_result(
    iree_hal_device_t* device, iree_tokenizer_t* tokenizer, int32_t eos_token,
    iree_hal_buffer_t* token_buffer, iree_hal_buffer_t* config_buffer,
    iree_device_size_t control_data_offset, int32_t expected_pending_position,
    const qwen38_generation_stats_t* generation_stats, int32_t* token_ids,
    char* decoded_text, iree_allocator_t host_allocator,
    qwen38_turn_result_t* out_result) {
  iree_device_size_t output_token_byte_length = 0;
  if (!iree_device_size_checked_mul(
          (iree_device_size_t)generation_stats->generated_count + 1,
          sizeof(*token_ids), &output_token_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "turn token history size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, token_buffer, 0, token_ids, output_token_byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  int32_t control_values[QWEN38_TARGET_CONTROL_WORD_COUNT] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      device, config_buffer, control_data_offset, control_values,
      sizeof(control_values), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  if (control_values[0] != expected_pending_position ||
      control_values[1] != generation_stats->generated_count ||
      control_values[2] != eos_token) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "generation control is [position=%" PRId32 ", generated=%" PRId32
        ", eos=%" PRId32 "], expected [%" PRId32 ", %" PRId32 ", %" PRId32
        "]",
        control_values[0], control_values[1], control_values[2],
        expected_pending_position, generation_stats->generated_count,
        eos_token);
  }

  iree_host_size_t visible_token_count =
      (iree_host_size_t)generation_stats->generated_count;
  for (iree_host_size_t i = 0; i < visible_token_count; ++i) {
    if (token_ids[i + 1] == eos_token) {
      visible_token_count = i + 1;
      break;
    }
  }
  if (generation_stats->done &&
      (visible_token_count !=
           (iree_host_size_t)generation_stats->generated_count ||
       token_ids[generation_stats->generated_count] != eos_token)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "device reported EOS after %" PRId32
        " outputs but token history does not end in EOS",
        generation_stats->generated_count);
  }

  iree_host_size_t decoded_text_length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode(
      tokenizer,
      iree_tokenizer_make_token_id_list(token_ids + 1, visible_token_count),
      IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      iree_make_mutable_string_view(decoded_text,
                                    QWEN38_DECODE_TEXT_CAPACITY - 1),
      host_allocator, &decoded_text_length));
  decoded_text[decoded_text_length] = '\0';
  *out_result = (qwen38_turn_result_t){
      .pending_position = control_values[0],
      .visible_token_count = visible_token_count,
      .decoded_text_length = decoded_text_length,
  };
  return iree_ok_status();
}

static bool qwen38_place_aligned_range(iree_host_size_t byte_length,
                                       iree_host_size_t alignment,
                                       iree_host_size_t* cursor,
                                       iree_host_size_t* out_offset) {
  iree_host_size_t byte_offset = 0;
  if (!iree_host_size_checked_align(*cursor, alignment, &byte_offset) ||
      !iree_host_size_checked_add(byte_offset, byte_length, cursor)) {
    return false;
  }
  *out_offset = byte_offset;
  return true;
}

static iree_status_t qwen38_text_run(void) {
  const iree_allocator_t host_allocator = iree_allocator_system();
  const iree_time_t run_start = iree_time_now();
  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  iree_io_file_contents_t* tokenizer_contents = NULL;
  iree_io_file_contents_t* prompt_contents = NULL;
  iree_tokenizer_t* tokenizer = NULL;
  iree_string_builder_t formatted_prompt;
  iree_string_builder_initialize(host_allocator, &formatted_prompt);
  qwen_tooling_command_program_set_t* program_set = NULL;
  qwen_tooling_command_program_t* chunk_program = NULL;
  qwen_tooling_command_program_t*
      prefill_programs[QWEN38_PREFILL_SCHEDULE_COUNT] = {0};
  qwen_tooling_command_program_t* prefill_program = NULL;
  qwen_tooling_command_program_t* decode_program = NULL;
  qwen_tooling_command_program_t* propose_program = NULL;
  qwen_tooling_command_program_t* verify_program = NULL;
  const loomc_cmd_program_info_t* chunk_info = NULL;
  const loomc_cmd_program_info_t*
      prefill_infos[QWEN38_PREFILL_SCHEDULE_COUNT] = {0};
  const loomc_cmd_program_info_t* prefill_info = NULL;
  const loomc_cmd_program_info_t* decode_info = NULL;
  const loomc_cmd_program_info_t* propose_info = NULL;
  const loomc_cmd_program_info_t* verify_info = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_buffer_t* residual_buffer = NULL;
  iree_hal_buffer_t* state_buffer = NULL;
  iree_hal_buffer_t* cache_buffer = NULL;
  iree_hal_buffer_t* mtp_cache_buffer = NULL;
  iree_hal_buffer_t* normalized_hidden_buffer = NULL;
  iree_hal_buffer_t* token_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  iree_hal_buffer_t* config_buffer = NULL;
  iree_hal_buffer_t* progress_buffer = NULL;
  iree_hal_semaphore_t* execution_semaphore = NULL;
  iree_hal_profiling_from_flags_t* profiling = NULL;
  iree_hal_buffer_mapping_t progress_mapping = {0};
  bool progress_mapping_active = false;
  uint8_t* config_data = NULL;
  char* decoded_text = NULL;
  char context_capacity_value[32] = {0};
  int32_t* token_ids = NULL;
  float residual_values[QWEN38_HIDDEN_BYTE_LENGTH / sizeof(float)];
  float hidden_values[QWEN38_HIDDEN_BYTE_LENGTH / sizeof(float)];
  iree_host_size_t prompt_token_count = 0;
  int32_t eos_token = -1;
  iree_host_size_t intermediate_chunk_count = 0;
  iree_host_size_t final_chunk_token_count = 0;
  uint32_t prefill_schedule_mask = 0;
  iree_host_size_t chunk_config_data_offset = 0;
  iree_host_size_t prefill_config_data_offset = 0;
  iree_host_size_t prefill_config_data_length = 0;
  iree_host_size_t prefill_config_data_alignment = 1;
  iree_host_size_t control_data_offset = 0;
  iree_host_size_t mtp_control_data_offset = 0;
  iree_host_size_t current_token_data_offset = 0;
  iree_host_size_t draft_tokens_data_offset = 0;
  iree_host_size_t verify_tokens_data_offset = 0;
  iree_host_size_t replay_control_data_offset = 0;
  iree_host_size_t acceptance_data_offset = 0;
  iree_host_size_t mtp_state_data_length = 0;
  iree_device_size_t token_byte_length = 0;
  iree_device_size_t prefill_chunk_token_byte_length = 0;
  iree_device_size_t final_chunk_token_byte_length = 0;
  iree_device_size_t attention_cache_byte_length = 0;
  iree_device_size_t mtp_attention_cache_byte_length = 0;
  iree_time_t ready_time = run_start;
  iree_time_t prefill_start = run_start;
  iree_time_t first_token_time = run_start;
  iree_time_t decode_start = run_start;
  iree_time_t decode_end = run_start;

  iree_status_t status = iree_ok_status();
  if (FLAG_tokenizer[0] == '\0') {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--tokenizer must name tokenizer.json");
  } else if (FLAG_max_tokens < 1 ||
             FLAG_max_tokens > QWEN38_GENERATION_CAPACITY) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max_tokens must be between 1 and %d",
                              QWEN38_GENERATION_CAPACITY);
  } else if (FLAG_context_capacity < 1 ||
             FLAG_context_capacity > QWEN38_MAX_CONTEXT_CAPACITY) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--context_capacity must be between 1 and %d",
                              QWEN38_MAX_CONTEXT_CAPACITY);
  }
  if (iree_status_is_ok(status)) {
    const int value_length =
        snprintf(context_capacity_value, sizeof(context_capacity_value),
                 "%" PRId32, FLAG_context_capacity);
    if (value_length < 0 ||
        (iree_host_size_t)value_length >= sizeof(context_capacity_value)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "context capacity spelling overflow");
    }
  }

  if (iree_status_is_ok(status) &&
      (!iree_device_size_checked_mul(
           (iree_device_size_t)FLAG_context_capacity, sizeof(*token_ids),
           &token_byte_length) ||
       !iree_device_size_checked_mul(
           (iree_device_size_t)FLAG_context_capacity,
           QWEN38_ATTENTION_CACHE_BYTES_PER_TOKEN,
           &attention_cache_byte_length) ||
       !iree_device_size_checked_mul(
           (iree_device_size_t)FLAG_context_capacity,
           QWEN38_MTP_ATTENTION_CACHE_BYTES_PER_TOKEN,
           &mtp_attention_cache_byte_length) ||
       !iree_device_size_checked_mul(QWEN38_PREFILL_CHUNK_CAPACITY,
                                     sizeof(*token_ids),
                                     &prefill_chunk_token_byte_length))) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "context allocation size overflow");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, (iree_host_size_t)FLAG_context_capacity,
        sizeof(*token_ids), (void**)&token_ids);
    if (iree_status_is_ok(status)) {
      memset(token_ids, 0, (iree_host_size_t)token_byte_length);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(FLAG_tokenizer),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &tokenizer_contents);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t tokenizer_json = iree_make_string_view(
        (const char*)tokenizer_contents->const_buffer.data,
        tokenizer_contents->const_buffer.data_length);
    status = iree_tokenizer_from_huggingface_json(tokenizer_json,
                                                  host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    eos_token = iree_tokenizer_vocab_lookup(iree_tokenizer_vocab(tokenizer),
                                            IREE_SV("<|im_end|>"));
    if (eos_token < 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "Qwen tokenizer does not define the <|im_end|> turn terminator");
    }
  }
  if (iree_status_is_ok(status)) {
    if (FLAG_prompt_file[0] != '\0') {
      status = iree_io_file_contents_map(
          iree_make_cstring_view(FLAG_prompt_file), IREE_IO_FILE_ACCESS_READ,
          host_allocator, &prompt_contents);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t prompt =
        prompt_contents
            ? iree_make_string_view(
                  (const char*)prompt_contents->const_buffer.data,
                  prompt_contents->const_buffer.data_length)
            : iree_make_cstring_view(FLAG_prompt);
    status =
        qwen38_build_prompt(prompt, FLAG_raw_prompt, &formatted_prompt);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        tokenizer, iree_string_builder_view(&formatted_prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(token_ids, /*token_offsets=*/NULL,
                                         /*type_ids=*/NULL,
                                         (iree_host_size_t)
                                             FLAG_context_capacity),
        host_allocator, &prompt_token_count);
  }
  if (iree_status_is_ok(status) && FLAG_verbose) {
    fprintf(stderr, "qwen38: prompt token IDs:");
    for (iree_host_size_t i = 0; i < prompt_token_count; ++i) {
      fprintf(stderr, " %" PRId32, token_ids[i]);
    }
    fprintf(stderr, "\n");
  }
  if (iree_status_is_ok(status) &&
      (prompt_token_count == 0 ||
       prompt_token_count + (iree_host_size_t)FLAG_max_tokens - 1 >
           (iree_host_size_t)FLAG_context_capacity)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prompt (%" PRIhsz ") plus generation (%d) exceeds context %d",
        prompt_token_count, FLAG_max_tokens, FLAG_context_capacity);
  }
  if (iree_status_is_ok(status)) {
    intermediate_chunk_count =
        (prompt_token_count - 1) / QWEN38_PREFILL_CHUNK_CAPACITY;
    final_chunk_token_count =
        prompt_token_count -
        intermediate_chunk_count * QWEN38_PREFILL_CHUNK_CAPACITY;
    if (!iree_device_size_checked_mul(final_chunk_token_count,
                                      sizeof(*token_ids),
                                      &final_chunk_token_byte_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "final prompt token span overflow");
    } else {
      const qwen38_prefill_schedule_t initial_prefill_schedule =
          qwen38_prefill_schedule_for_token_count(final_chunk_token_count);
      prefill_schedule_mask = 1u << initial_prefill_schedule;
    }
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    status = qwen38_build_followup_span(
        iree_make_cstring_view(FLAG_second_prompt), &formatted_prompt);
  }
  iree_host_size_t prepared_second_prompt_token_count = 0;
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    status = iree_tokenizer_encode(
        tokenizer, iree_string_builder_view(&formatted_prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(
            token_ids + prompt_token_count, /*token_offsets=*/NULL,
            /*type_ids=*/NULL,
            (iree_host_size_t)FLAG_context_capacity - prompt_token_count),
        host_allocator, &prepared_second_prompt_token_count);
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    if (prepared_second_prompt_token_count == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "the second prompt tokenized to no tokens");
    } else {
      const iree_host_size_t prepared_final_chunk_token_count =
          (prepared_second_prompt_token_count - 1) %
              QWEN38_PREFILL_CHUNK_CAPACITY +
          1;
      const qwen38_prefill_schedule_t second_prefill_schedule =
          qwen38_prefill_schedule_for_token_count(
              prepared_final_chunk_token_count);
      prefill_schedule_mask |= 1u << second_prefill_schedule;
    }
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }
  if (iree_status_is_ok(status)) {
    device = qwen_tooling_runtime_context_device(&runtime_context);
    status =
        iree_hal_begin_profiling_from_flags(device, host_allocator, &profiling);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: tokenized %" PRIhsz
            " prompt tokens; preparing prefill, decode, and MTP together...\n",
            prompt_token_count);
    status = qwen38_prepare_text_model(
        &runtime_context, FLAG_mtp, prefill_schedule_mask,
        iree_make_cstring_view(context_capacity_value), host_allocator,
        &program_set);
  }
  if (iree_status_is_ok(status)) {
    chunk_program = qwen_tooling_command_program_set_lookup(
        program_set, qwen38_chunk_root_name(FLAG_mtp));
    const qwen38_prefill_schedule_t initial_prefill_schedule =
        qwen38_prefill_schedule_for_token_count(final_chunk_token_count);
    for (iree_host_size_t i = 0; i < QWEN38_PREFILL_SCHEDULE_COUNT; ++i) {
      if ((prefill_schedule_mask & (1u << i)) == 0) continue;
      prefill_programs[i] = qwen_tooling_command_program_set_lookup(
          program_set, qwen38_prefill_root_name_for_schedule(
                           FLAG_mtp, (qwen38_prefill_schedule_t)i));
      prefill_infos[i] =
          qwen_tooling_command_program_info(prefill_programs[i]);
    }
    prefill_program = prefill_programs[initial_prefill_schedule];
    prefill_info = prefill_infos[initial_prefill_schedule];
    if (FLAG_verbose) {
      const iree_string_view_t root_name =
          qwen38_prefill_root_name_for_schedule(FLAG_mtp,
                                                initial_prefill_schedule);
      fprintf(stderr, "qwen38: selected %.*s for %" PRIhsz
                      " final prompt tokens\n",
              (int)root_name.size, root_name.data, final_chunk_token_count);
    }
    decode_program = qwen_tooling_command_program_set_lookup(
        program_set, IREE_SV("qwen38_text_decode_greedy_with_hidden"));
    if (FLAG_mtp) {
      propose_program = qwen_tooling_command_program_set_lookup(
          program_set, IREE_SV("qwen38_mtp_propose_depth4"));
      verify_program = qwen_tooling_command_program_set_lookup(
          program_set, IREE_SV("qwen38_mtp_target_verify_depth4"));
    }
    chunk_info = qwen_tooling_command_program_info(chunk_program);
    decode_info = qwen_tooling_command_program_info(decode_program);
    if (FLAG_mtp) {
      propose_info = qwen_tooling_command_program_info(propose_program);
      verify_info = qwen_tooling_command_program_info(verify_program);
    }
    const iree_host_size_t expected_chunk_binding_count = FLAG_mtp ? 9 : 7;
    const iree_host_size_t expected_chunk_transient_binding = FLAG_mtp ? 7 : 5;
    const iree_host_size_t expected_chunk_config_binding = FLAG_mtp ? 8 : 6;
    if (!chunk_info || !decode_info ||
        (FLAG_mtp && (!propose_info || !verify_info)) ||
        chunk_info->rebindable_binding_count !=
                            expected_chunk_binding_count ||
        chunk_info->transient.binding_index !=
            expected_chunk_transient_binding ||
        chunk_info->config.binding_index != expected_chunk_config_binding ||
        chunk_info->config.required_byte_length == 0 ||
        decode_info->rebindable_binding_count != 8 ||
        decode_info->transient.binding_index != 7 ||
        decode_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID ||
        (FLAG_mtp &&
         (propose_info->rebindable_binding_count != 10 ||
          propose_info->transient.binding_index != 9 ||
          propose_info->config.binding_index !=
              LOOMC_CMD_PROGRAM_BINDING_INVALID ||
          verify_info->rebindable_binding_count != 13 ||
          verify_info->transient.binding_index != 12 ||
          verify_info->config.binding_index !=
              LOOMC_CMD_PROGRAM_BINDING_INVALID))) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the text command programs published an incompatible ABI");
    }
  }
  for (iree_host_size_t i = 0;
       i < QWEN38_PREFILL_SCHEDULE_COUNT && iree_status_is_ok(status); ++i) {
    if ((prefill_schedule_mask & (1u << i)) == 0) continue;
    const loomc_cmd_program_info_t* info = prefill_infos[i];
    const iree_host_size_t expected_prefill_binding_count = FLAG_mtp ? 10 : 9;
    const iree_host_size_t expected_prefill_transient_binding = FLAG_mtp ? 8 : 7;
    const iree_host_size_t expected_prefill_config_binding = FLAG_mtp ? 9 : 8;
    if (!info ||
        info->rebindable_binding_count != expected_prefill_binding_count ||
        info->transient.binding_index != expected_prefill_transient_binding ||
        info->config.binding_index != expected_prefill_config_binding ||
        info->config.required_byte_length == 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "prefill schedule %" PRIhsz " published an incompatible ABI", i);
    } else {
      prefill_config_data_length = iree_max(
          prefill_config_data_length,
          (iree_host_size_t)info->config.required_byte_length);
      prefill_config_data_alignment = iree_max(
          prefill_config_data_alignment,
          (iree_host_size_t)info->config.minimum_alignment);
    }
  }
  if (iree_status_is_ok(status)) {
    mtp_state_data_length = 0;
    if (!qwen38_place_aligned_range(
            (iree_host_size_t)chunk_info->config.required_byte_length,
            (iree_host_size_t)chunk_info->config.minimum_alignment,
            &mtp_state_data_length, &chunk_config_data_offset) ||
        !qwen38_place_aligned_range(
            prefill_config_data_length, prefill_config_data_alignment,
            &mtp_state_data_length, &prefill_config_data_offset) ||
        !qwen38_place_aligned_range(QWEN38_TARGET_CONTROL_BYTE_LENGTH,
                                    /*alignment=*/64,
                                    &mtp_state_data_length,
                                    &control_data_offset) ||
        !qwen38_place_aligned_range(QWEN38_PAIR_CONTROL_BYTE_LENGTH,
                                    /*alignment=*/64,
                                    &mtp_state_data_length,
                                    &mtp_control_data_offset) ||
        !qwen38_place_aligned_range(sizeof(int32_t), /*alignment=*/64,
                                    &mtp_state_data_length,
                                    &current_token_data_offset) ||
        !qwen38_place_aligned_range(
            QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t), /*alignment=*/64,
            &mtp_state_data_length, &draft_tokens_data_offset) ||
        !qwen38_place_aligned_range(
            QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t), /*alignment=*/64,
            &mtp_state_data_length, &verify_tokens_data_offset) ||
        !qwen38_place_aligned_range(QWEN38_PAIR_CONTROL_BYTE_LENGTH,
                                    /*alignment=*/64,
                                    &mtp_state_data_length,
                                    &replay_control_data_offset) ||
        !qwen38_place_aligned_range(QWEN38_PAIR_CONTROL_BYTE_LENGTH,
                                    /*alignment=*/64, &mtp_state_data_length,
                                    &acceptance_data_offset)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command control allocation is too large");
    }
  }

  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_CAPACITY_BYTE_LENGTH,
        &residual_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_BYTE_LENGTH,
        &normalized_hidden_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_GDN_STATE_BYTE_LENGTH, &state_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, attention_cache_byte_length,
        &cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, mtp_attention_cache_byte_length,
        &mtp_cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/64, token_byte_length, &token_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_device_size_t transient_byte_length =
        chunk_info->transient.required_byte_length;
    for (iree_host_size_t i = 0; i < QWEN38_PREFILL_SCHEDULE_COUNT; ++i) {
      if ((prefill_schedule_mask & (1u << i)) == 0) continue;
      transient_byte_length =
          iree_max(transient_byte_length,
                   prefill_infos[i]->transient.required_byte_length);
    }
    transient_byte_length =
        iree_max(transient_byte_length,
                 decode_info->transient.required_byte_length);
    if (FLAG_mtp) {
      transient_byte_length =
          iree_max(transient_byte_length,
                   propose_info->transient.required_byte_length);
      transient_byte_length =
          iree_max(transient_byte_length,
                   verify_info->transient.required_byte_length);
    }
    iree_device_size_t transient_alignment =
        chunk_info->transient.minimum_alignment;
    for (iree_host_size_t i = 0; i < QWEN38_PREFILL_SCHEDULE_COUNT; ++i) {
      if ((prefill_schedule_mask & (1u << i)) == 0) continue;
      transient_alignment =
          iree_max(transient_alignment,
                   prefill_infos[i]->transient.minimum_alignment);
    }
    transient_alignment =
        iree_max(transient_alignment, decode_info->transient.minimum_alignment);
    if (FLAG_mtp) {
      transient_alignment = iree_max(
          transient_alignment, propose_info->transient.minimum_alignment);
      transient_alignment = iree_max(
          transient_alignment, verify_info->transient.minimum_alignment);
    }
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        transient_alignment, transient_byte_length, &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
            IREE_HAL_BUFFER_USAGE_TRANSFER,
        LOOMC_CMD_HAL_CONFIG_ALIGNMENT, mtp_state_data_length, &config_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
        /*minimum_alignment=*/64, QWEN38_MTP_PROGRESS_BYTE_LENGTH,
        &progress_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        progress_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        /*byte_offset=*/0, QWEN38_MTP_PROGRESS_BYTE_LENGTH, &progress_mapping);
    if (iree_status_is_ok(status)) {
      progress_mapping_active = true;
      memset(progress_mapping.contents.data, 0,
             QWEN38_MTP_PROGRESS_BYTE_LENGTH);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t config_alignment =
        (iree_host_size_t)iree_max((uint64_t)LOOMC_CMD_HAL_CONFIG_ALIGNMENT,
                                   iree_max(
                                       chunk_info->config.minimum_alignment,
                                       (uint64_t)prefill_config_data_alignment));
    status = iree_allocator_malloc_aligned(
        host_allocator, mtp_state_data_length, config_alignment, /*offset=*/0,
        (void**)&config_data);
    if (iree_status_is_ok(status)) {
      memset(config_data, 0, mtp_state_data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, QWEN38_DECODE_TEXT_CAPACITY,
                                   (void**)&decoded_text);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution_semaphore);
  }

  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, state_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, mtp_cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, normalized_hidden_buffer);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t chunk_argument_bits[] = {QWEN38_PREFILL_CHUNK_CAPACITY};
    status = qwen_tooling_command_program_populate_config(
        chunk_program, chunk_argument_bits, IREE_ARRAYSIZE(chunk_argument_bits),
        iree_make_byte_span(
            config_data + chunk_config_data_offset,
            (iree_host_size_t)chunk_info->config.required_byte_length));
  }
  if (iree_status_is_ok(status)) {
    const uint64_t argument_bits[] = {(uint64_t)final_chunk_token_count};
    status = qwen_tooling_command_program_populate_config(
        prefill_program, argument_bits, IREE_ARRAYSIZE(argument_bits),
        iree_make_byte_span(
            config_data + prefill_config_data_offset,
            (iree_host_size_t)prefill_info->config.required_byte_length));
  }
  if (iree_status_is_ok(status)) {
    iree_unaligned_store_le_u32(config_data + control_data_offset,
                                (uint32_t)final_chunk_token_count);
    iree_unaligned_store_le_u32(
        config_data + control_data_offset + sizeof(uint32_t),
        (uint32_t)(intermediate_chunk_count *
                   QWEN38_PREFILL_CHUNK_CAPACITY));
    iree_unaligned_store_le_u32(
        config_data + control_data_offset + 2 * sizeof(uint32_t),
        (uint32_t)eos_token);
    status = iree_hal_device_transfer_h2d(
        device, config_data, config_buffer, 0, mtp_state_data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  iree_hal_buffer_binding_t prefill_bindings[10] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = progress_buffer,
             .offset = 0,
             .length = QWEN38_MTP_PROGRESS_BYTE_LENGTH},
      [8] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [9] = {.buffer = config_buffer,
             .offset = prefill_config_data_offset,
             .length =
                 prefill_info ? prefill_info->config.required_byte_length : 0},
  };
  iree_hal_buffer_binding_t ordinary_prefill_bindings[9] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = progress_buffer,
             .offset = 0,
             .length = QWEN38_MTP_PROGRESS_BYTE_LENGTH},
      [7] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [8] = {.buffer = config_buffer,
             .offset = prefill_config_data_offset,
             .length =
                 prefill_info ? prefill_info->config.required_byte_length : 0},
  };
  const iree_hal_buffer_binding_t chunk_bindings[7] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = config_buffer,
             .offset = chunk_config_data_offset,
             .length = chunk_info ? chunk_info->config.required_byte_length
                                  : 0},
  };
  const iree_hal_buffer_binding_t mtp_chunk_bindings[9] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [8] = {.buffer = config_buffer,
             .offset = chunk_config_data_offset,
             .length = chunk_info ? chunk_info->config.required_byte_length
                                  : 0},
  };
  const iree_hal_buffer_binding_t decode_bindings[8] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = progress_buffer,
             .offset = 0,
             .length = QWEN38_MTP_PROGRESS_BYTE_LENGTH},
      [7] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_t propose_bindings[10] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = config_buffer,
             .offset = mtp_control_data_offset,
             .length = QWEN38_PAIR_CONTROL_BYTE_LENGTH},
      [3] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = config_buffer,
             .offset = current_token_data_offset,
             .length = sizeof(int32_t)},
      [7] = {.buffer = config_buffer,
             .offset = draft_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [8] = {.buffer = config_buffer,
             .offset = verify_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [9] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_t verify_bindings[13] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_TARGET_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = config_buffer,
             .offset = draft_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [8] = {.buffer = config_buffer,
             .offset = verify_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [9] = {.buffer = config_buffer,
             .offset = replay_control_data_offset,
             .length = QWEN38_PAIR_CONTROL_BYTE_LENGTH},
      [10] = {.buffer = config_buffer,
              .offset = acceptance_data_offset,
              .length = QWEN38_PAIR_CONTROL_BYTE_LENGTH},
      [11] = {.buffer = progress_buffer,
              .offset = 0,
              .length = QWEN38_MTP_PROGRESS_BYTE_LENGTH},
      [12] = {.buffer = transient_buffer,
              .offset = 0,
              .length = IREE_HAL_WHOLE_BUFFER},
  };
  uint64_t execution_value = 0;
  if (iree_status_is_ok(status)) {
    ready_time = iree_time_now();
    fprintf(stderr, "qwen38: materialized command programs:\n");
    for (iree_host_size_t i = 0; i < QWEN38_PREFILL_SCHEDULE_COUNT; ++i) {
      if ((prefill_schedule_mask & (1u << i)) == 0) continue;
      const iree_string_view_t root_name =
          qwen38_prefill_root_name_for_schedule(
              FLAG_mtp, (qwen38_prefill_schedule_t)i);
      fprintf(stderr,
              "  %.*s: %" PRIhsz " commands, %" PRIu64
              " transient bytes\n",
              (int)root_name.size, root_name.data,
              prefill_infos[i]->command_count,
              (uint64_t)prefill_infos[i]->transient.required_byte_length);
    }
    fprintf(stderr,
            "  decode: %" PRIhsz " commands, %" PRIu64
            " transient bytes\n",
            decode_info->command_count,
            (uint64_t)decode_info->transient.required_byte_length);
    if (FLAG_mtp) {
      fprintf(stderr,
              "  MTP propose: %" PRIhsz " commands, %" PRIu64
              " transient bytes\n"
              "  MTP verify: %" PRIhsz " commands, %" PRIu64
              " transient bytes\n",
              propose_info->command_count,
              (uint64_t)propose_info->transient.required_byte_length,
              verify_info->command_count,
              (uint64_t)verify_info->transient.required_byte_length);
    }
    fprintf(stderr, "qwen38: executing...\n");
  }
  if (iree_status_is_ok(status)) {
    prefill_start = iree_time_now();
    for (iree_host_size_t chunk = 0;
         chunk < intermediate_chunk_count && iree_status_is_ok(status);
         ++chunk) {
      const iree_host_size_t token_offset =
          chunk * QWEN38_PREFILL_CHUNK_CAPACITY;
      const int32_t chunk_control[QWEN38_TARGET_CONTROL_WORD_COUNT] = {
          QWEN38_PREFILL_CHUNK_CAPACITY, (int32_t)token_offset, eos_token};
      status = qwen38_queue_update(
          device, execution_semaphore, &execution_value,
          token_ids + token_offset, token_buffer, /*target_offset=*/0,
          prefill_chunk_token_byte_length);
      if (iree_status_is_ok(status)) {
        status = qwen38_queue_update(
            device, execution_semaphore, &execution_value, chunk_control,
            config_buffer, control_data_offset, sizeof(chunk_control));
      }
      if (iree_status_is_ok(status)) {
        const iree_hal_buffer_binding_table_t chunk_binding_table =
            FLAG_mtp
                ? (iree_hal_buffer_binding_table_t){
                      .count = IREE_ARRAYSIZE(mtp_chunk_bindings),
                      .bindings = mtp_chunk_bindings,
                  }
                : (iree_hal_buffer_binding_table_t){
                      .count = IREE_ARRAYSIZE(chunk_bindings),
                      .bindings = chunk_bindings,
                  };
        status = qwen38_queue_program(
            device, execution_semaphore, &execution_value, chunk_program,
            chunk_binding_table);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t final_token_offset =
        intermediate_chunk_count * QWEN38_PREFILL_CHUNK_CAPACITY;
    const int32_t final_control[QWEN38_TARGET_CONTROL_WORD_COUNT] = {
        (int32_t)final_chunk_token_count, (int32_t)final_token_offset,
        eos_token};
    status = qwen38_queue_update(
        device, execution_semaphore, &execution_value,
        token_ids + final_token_offset, token_buffer, /*target_offset=*/0,
        final_chunk_token_byte_length);
    if (iree_status_is_ok(status)) {
      status = qwen38_queue_update(
          device, execution_semaphore, &execution_value, final_control,
          config_buffer, control_data_offset, sizeof(final_control));
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_buffer_binding_table_t prefill_binding_table =
          FLAG_mtp
              ? (iree_hal_buffer_binding_table_t){
                    .count = IREE_ARRAYSIZE(prefill_bindings),
                    .bindings = prefill_bindings,
                }
              : (iree_hal_buffer_binding_table_t){
                    .count = IREE_ARRAYSIZE(ordinary_prefill_bindings),
                    .bindings = ordinary_prefill_bindings,
                };
      status = qwen38_queue_program(device, execution_semaphore,
                                    &execution_value, prefill_program,
                                    prefill_binding_table);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(execution_semaphore, execution_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
    first_token_time = iree_time_now();
    decode_start = first_token_time;
  }

  qwen38_generation_stats_t generation_stats;
  memset(&generation_stats, 0, sizeof(generation_stats));
  if (iree_status_is_ok(status)) {
    status = qwen38_generate_output(
        device, execution_semaphore, &execution_value, FLAG_mtp,
        FLAG_max_tokens, decode_program,
        (iree_hal_buffer_binding_table_t){
            .count = IREE_ARRAYSIZE(decode_bindings),
            .bindings = decode_bindings,
        },
        propose_program,
        (iree_hal_buffer_binding_table_t){
            .count = IREE_ARRAYSIZE(propose_bindings),
            .bindings = propose_bindings,
        },
        verify_program,
        (iree_hal_buffer_binding_table_t){
            .count = IREE_ARRAYSIZE(verify_bindings),
            .bindings = verify_bindings,
        },
        &progress_mapping, &generation_stats);
  }
  const int32_t generated_count = generation_stats.generated_count;
  if (iree_status_is_ok(status)) decode_end = iree_time_now();
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  profiling = NULL;

  qwen38_turn_result_t turn_result;
  memset(&turn_result, 0, sizeof(turn_result));
  if (iree_status_is_ok(status)) {
    status = qwen38_read_turn_result(
        device, tokenizer, eos_token, token_buffer, config_buffer,
        control_data_offset,
        (int32_t)prompt_token_count + generated_count - 1,
        &generation_stats, token_ids, decoded_text, host_allocator,
        &turn_result);
  }
  if (iree_status_is_ok(status) && FLAG_residual_output[0] != '\0') {
    status = iree_hal_device_transfer_d2h(
        device, residual_buffer, 0, residual_values, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_hidden_output[0] != '\0') {
    status = iree_hal_device_transfer_d2h(
        device, normalized_hidden_buffer, 0, hidden_values,
        QWEN38_HIDDEN_BYTE_LENGTH, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_residual_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_residual_output),
        iree_make_const_byte_span(residual_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status) && FLAG_hidden_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_hidden_output),
        iree_make_const_byte_span(hidden_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t decode_transition_count =
        (iree_host_size_t)generated_count - 1;
    fprintf(stdout,
            "Qwen3.8 command-program generation\n"
            "  prompt tokens:        %" PRIhsz
            "\n"
            "  generated tokens:     %d\n"
            "  parameter bytes:      %" PRIu64
            "\n"
            "  MTP cycles:           %" PRId32 " (%" PRId32 " tokens, %" PRId32
            " accepted drafts)\n"
            "  MTP accepted depths:  0:%" PRId32 " 1:%" PRId32 " 2:%" PRId32
            " 3:%" PRId32 " 4:%" PRId32
            "\n"
            "  setup to ready:       %.3f ms\n"
            "  prefill:              %.3f ms (%.2f tok/s)\n"
            "  cold time to token:   %.3f ms\n"
            "  decode:               %.3f ms (%.2f tok/s)\n"
            "  total through output: %.3f ms\n",
            prompt_token_count, generated_count,
            (uint64_t)qwen_tooling_command_program_set_parameter_byte_length(
                program_set),
            generation_stats.mtp_cycle_count,
            generation_stats.mtp_committed_count,
            generation_stats.mtp_accepted_draft_count,
            generation_stats.mtp_depth_counts[0],
            generation_stats.mtp_depth_counts[1],
            generation_stats.mtp_depth_counts[2],
            generation_stats.mtp_depth_counts[3],
            generation_stats.mtp_depth_counts[4],
            qwen38_elapsed_ms(run_start, ready_time),
            qwen38_elapsed_ms(prefill_start, first_token_time),
            qwen38_rate(prompt_token_count, prefill_start, first_token_time),
            qwen38_elapsed_ms(run_start, first_token_time),
            qwen38_elapsed_ms(decode_start, decode_end),
            qwen38_rate(decode_transition_count, decode_start, decode_end),
            qwen38_elapsed_ms(run_start, decode_end));
    fprintf(stdout, "  output token IDs:");
    for (int32_t i = 0; i < generated_count; ++i) {
      fprintf(stdout, " %" PRId32, token_ids[i + 1]);
    }
    fprintf(stdout, "\n\n%s\n", decoded_text);
  }

  // An optional second turn is a bounded proof that the materialized program
  // set and all persistent model state survive a real chat boundary.
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    if (!generation_stats.done) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the first turn reached --max_tokens before EOS");
    }
  }

  iree_host_size_t second_prompt_token_count = 0;
  iree_host_size_t second_intermediate_chunk_count = 0;
  iree_host_size_t second_final_chunk_token_count = 0;
  qwen38_generation_stats_t second_generation_stats;
  memset(&second_generation_stats, 0, sizeof(second_generation_stats));
  qwen38_turn_result_t second_turn_result;
  memset(&second_turn_result, 0, sizeof(second_turn_result));
  iree_time_t second_prefill_start = run_start;
  iree_time_t second_first_token_time = run_start;
  iree_time_t second_decode_end = run_start;
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    status = qwen38_build_followup_span(
        iree_make_cstring_view(FLAG_second_prompt), &formatted_prompt);
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    status = iree_tokenizer_encode(
        tokenizer, iree_string_builder_view(&formatted_prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(
            token_ids, /*token_offsets=*/NULL, /*type_ids=*/NULL,
            (iree_host_size_t)FLAG_context_capacity),
        host_allocator, &second_prompt_token_count);
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0' &&
      FLAG_verbose) {
    fprintf(stderr, "qwen38: second-turn token IDs:");
    for (iree_host_size_t i = 0; i < second_prompt_token_count; ++i) {
      fprintf(stderr, " %" PRId32, token_ids[i]);
    }
    fprintf(stderr, "\n");
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    iree_host_size_t remaining_context = 0;
    if (turn_result.pending_position < 0 ||
        turn_result.pending_position >= FLAG_context_capacity) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "first turn left pending position %" PRId32
          " outside context capacity %" PRId32,
          turn_result.pending_position, FLAG_context_capacity);
    } else {
      remaining_context = (iree_host_size_t)FLAG_context_capacity -
                          (iree_host_size_t)turn_result.pending_position;
    }
    if (iree_status_is_ok(status) &&
        (second_prompt_token_count == 0 ||
        second_prompt_token_count + (iree_host_size_t)FLAG_max_tokens - 1 >
            remaining_context)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "second turn span (%" PRIhsz
          ") plus generation (%" PRId32
          ") exceeds the remaining context (%" PRIhsz ")",
          second_prompt_token_count, FLAG_max_tokens, remaining_context);
    } else if (iree_status_is_ok(status)) {
      second_intermediate_chunk_count =
          (second_prompt_token_count - 1) / QWEN38_PREFILL_CHUNK_CAPACITY;
      second_final_chunk_token_count =
          second_prompt_token_count -
          second_intermediate_chunk_count * QWEN38_PREFILL_CHUNK_CAPACITY;
      const qwen38_prefill_schedule_t second_prefill_schedule =
          qwen38_prefill_schedule_for_token_count(
              second_final_chunk_token_count);
      prefill_program = prefill_programs[second_prefill_schedule];
      prefill_info = prefill_infos[second_prefill_schedule];
      if (FLAG_verbose) {
        const iree_string_view_t root_name =
            qwen38_prefill_root_name_for_schedule(FLAG_mtp,
                                                  second_prefill_schedule);
        fprintf(stderr, "qwen38: selected %.*s for %" PRIhsz
                        " final second-turn tokens\n",
                (int)root_name.size, root_name.data,
                second_final_chunk_token_count);
      }
      prefill_bindings[9].length =
          prefill_info->config.required_byte_length;
      ordinary_prefill_bindings[8].length =
          prefill_info->config.required_byte_length;
    }
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    memset(progress_mapping.contents.data, 0,
           QWEN38_MTP_PROGRESS_BYTE_LENGTH);
    const uint64_t argument_bits[] = {
        (uint64_t)second_final_chunk_token_count};
    status = qwen_tooling_command_program_populate_config(
        prefill_program, argument_bits, IREE_ARRAYSIZE(argument_bits),
        iree_make_byte_span(
            config_data + prefill_config_data_offset,
            (iree_host_size_t)prefill_info->config.required_byte_length));
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    second_prefill_start = iree_time_now();
    status = iree_hal_begin_profiling_from_flags(device, host_allocator,
                                                 &profiling);
    if (iree_status_is_ok(status)) {
      status = qwen38_queue_update(
          device, execution_semaphore, &execution_value,
          config_data + prefill_config_data_offset, config_buffer,
          prefill_config_data_offset,
          (iree_device_size_t)prefill_info->config.required_byte_length);
    }
    for (iree_host_size_t chunk = 0;
         chunk < second_intermediate_chunk_count &&
         iree_status_is_ok(status);
         ++chunk) {
      const iree_host_size_t token_offset =
          chunk * QWEN38_PREFILL_CHUNK_CAPACITY;
      const int32_t chunk_control[QWEN38_TARGET_CONTROL_WORD_COUNT] = {
          QWEN38_PREFILL_CHUNK_CAPACITY,
          turn_result.pending_position + (int32_t)token_offset, eos_token};
      status = qwen38_queue_update(
          device, execution_semaphore, &execution_value,
          token_ids + token_offset, token_buffer, /*target_offset=*/0,
          prefill_chunk_token_byte_length);
      if (iree_status_is_ok(status)) {
        status = qwen38_queue_update(
            device, execution_semaphore, &execution_value, chunk_control,
            config_buffer, control_data_offset, sizeof(chunk_control));
      }
      if (iree_status_is_ok(status)) {
        const iree_hal_buffer_binding_table_t chunk_binding_table =
            FLAG_mtp
                ? (iree_hal_buffer_binding_table_t){
                      .count = IREE_ARRAYSIZE(mtp_chunk_bindings),
                      .bindings = mtp_chunk_bindings,
                  }
                : (iree_hal_buffer_binding_table_t){
                      .count = IREE_ARRAYSIZE(chunk_bindings),
                      .bindings = chunk_bindings,
                  };
        status = qwen38_queue_program(device, execution_semaphore,
                                      &execution_value, chunk_program,
                                      chunk_binding_table);
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_host_size_t final_token_offset =
          second_intermediate_chunk_count * QWEN38_PREFILL_CHUNK_CAPACITY;
      iree_device_size_t final_token_byte_length = 0;
      if (!iree_device_size_checked_mul(second_final_chunk_token_count,
                                        sizeof(*token_ids),
                                        &final_token_byte_length)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "second-turn final token span overflow");
      }
      if (iree_status_is_ok(status)) {
        status = qwen38_queue_update(
            device, execution_semaphore, &execution_value,
            token_ids + final_token_offset, token_buffer, /*target_offset=*/0,
            final_token_byte_length);
      }
      const int32_t final_control[QWEN38_TARGET_CONTROL_WORD_COUNT] = {
          (int32_t)second_final_chunk_token_count,
          turn_result.pending_position + (int32_t)final_token_offset,
          eos_token};
      if (iree_status_is_ok(status)) {
        status = qwen38_queue_update(
            device, execution_semaphore, &execution_value, final_control,
            config_buffer, control_data_offset, sizeof(final_control));
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_buffer_binding_table_t prefill_binding_table =
          FLAG_mtp
              ? (iree_hal_buffer_binding_table_t){
                    .count = IREE_ARRAYSIZE(prefill_bindings),
                    .bindings = prefill_bindings,
                }
              : (iree_hal_buffer_binding_table_t){
                    .count = IREE_ARRAYSIZE(ordinary_prefill_bindings),
                    .bindings = ordinary_prefill_bindings,
                };
      status = qwen38_queue_program(device, execution_semaphore,
                                    &execution_value, prefill_program,
                                    prefill_binding_table);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(execution_semaphore, execution_value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      second_first_token_time = iree_time_now();
      status = qwen38_generate_output(
          device, execution_semaphore, &execution_value, FLAG_mtp,
          FLAG_max_tokens, decode_program,
          (iree_hal_buffer_binding_table_t){
              .count = IREE_ARRAYSIZE(decode_bindings),
              .bindings = decode_bindings,
          },
          propose_program,
          (iree_hal_buffer_binding_table_t){
              .count = IREE_ARRAYSIZE(propose_bindings),
              .bindings = propose_bindings,
          },
          verify_program,
          (iree_hal_buffer_binding_table_t){
              .count = IREE_ARRAYSIZE(verify_bindings),
              .bindings = verify_bindings,
          },
          &progress_mapping, &second_generation_stats);
    }
    if (iree_status_is_ok(status)) second_decode_end = iree_time_now();
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
    profiling = NULL;
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    const int32_t expected_pending_position =
        turn_result.pending_position + (int32_t)second_prompt_token_count +
        second_generation_stats.generated_count - 1;
    status = qwen38_read_turn_result(
        device, tokenizer, eos_token, token_buffer, config_buffer,
        control_data_offset, expected_pending_position,
        &second_generation_stats, token_ids, decoded_text, host_allocator,
        &second_turn_result);
  }
  if (iree_status_is_ok(status) && FLAG_second_prompt[0] != '\0') {
    const iree_host_size_t decode_transition_count =
        (iree_host_size_t)second_generation_stats.generated_count - 1;
    fprintf(stdout,
            "\nQwen3.8 command-program turn 2\n"
            "  appended tokens:       %" PRIhsz
            "\n"
            "  generated tokens:      %" PRId32
            "\n"
            "  MTP cycles:            %" PRId32
            "\n"
            "  prefill:               %.3f ms (%.2f tok/s)\n"
            "  decode:                %.3f ms (%.2f tok/s)\n",
            second_prompt_token_count,
            second_generation_stats.generated_count,
            second_generation_stats.mtp_cycle_count,
            qwen38_elapsed_ms(second_prefill_start, second_first_token_time),
            qwen38_rate(second_prompt_token_count, second_prefill_start,
                        second_first_token_time),
            qwen38_elapsed_ms(second_first_token_time, second_decode_end),
            qwen38_rate(decode_transition_count, second_first_token_time,
                        second_decode_end));
    fprintf(stdout, "  output token IDs:");
    for (int32_t i = 0; i < second_generation_stats.generated_count; ++i) {
      fprintf(stdout, " %" PRId32, token_ids[i + 1]);
    }
    fprintf(stdout, "\n\n%s\n", decoded_text);
  }

  if (progress_mapping_active) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&progress_mapping));
  }
  iree_hal_semaphore_release(execution_semaphore);
  iree_allocator_free(host_allocator, token_ids);
  iree_allocator_free(host_allocator, decoded_text);
  iree_allocator_free_aligned(host_allocator, config_data);
  iree_hal_buffer_release(progress_buffer);
  iree_hal_buffer_release(config_buffer);
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(token_buffer);
  iree_hal_buffer_release(normalized_hidden_buffer);
  iree_hal_buffer_release(mtp_cache_buffer);
  iree_hal_buffer_release(cache_buffer);
  iree_hal_buffer_release(state_buffer);
  iree_hal_buffer_release(residual_buffer);
  qwen_tooling_command_program_set_release(program_set);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  iree_string_builder_deinitialize(&formatted_prompt);
  iree_tokenizer_free(tokenizer);
  if (prompt_contents) iree_io_file_contents_free(prompt_contents);
  if (tokenizer_contents) iree_io_file_contents_free(tokenizer_contents);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen38-cli", qwen38_text_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) status = qwen38_text_run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
