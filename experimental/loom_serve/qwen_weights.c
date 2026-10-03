// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/qwen_weights.h"

#include <stdio.h>

#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"

enum { QWEN_WEIGHT_LANES = 4 };

typedef struct qwen_weight_span_t {
  // Borrowed key and final byte placement from the compiled command.
  loom_cmd_program_parameter_t parameter;
  // Borrowed final allocation or root view receiving the file bytes.
  iree_hal_buffer_t* buffer;
  // FFN gate/up bytes require the eight-channel in-place permutation.
  bool prepare;
} qwen_weight_span_t;

typedef struct qwen_weight_group_t {
  // Borrowed consecutive spans in one final allocation. Prepared groups own
  // exactly one tensor; consecutive canonical spans share one readiness edge.
  qwen_weight_span_t* spans;
  // Number of spans exposed to the parameter provider.
  iree_host_size_t count;
  // Previous preparation frontier on this group's lane.
  uint64_t wait_value;
  // Read and preparation frontier established by this group.
  uint64_t signal_value;
} qwen_weight_group_t;

static iree_status_t qwen_weight_allocate(iree_hal_device_t* device,
                                          iree_device_size_t length,
                                          iree_device_size_t alignment,
                                          iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = alignment;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, length, out_buffer);
}

static bool qwen_weight_requires_preparation(iree_string_view_t key) {
  return iree_string_view_starts_with(key, IREE_SV("blk.")) &&
         (iree_string_view_ends_with(key, IREE_SV(".ffn_gate.weight")) ||
          iree_string_view_ends_with(key, IREE_SV(".ffn_up.weight")));
}

static const qwen_weight_span_t* qwen_weight_find(
    iree_host_size_t count, const qwen_weight_span_t* spans,
    iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so shared target/MTP tensors are read and prepared exactly
// once.
static iree_status_t qwen_weight_resolve_root(
    iree_hal_device_t* device, const loom_cmd_program_t* program,
    loom_cmd_program_parameter_root_t root, iree_host_size_t* span_count,
    qwen_weight_span_t* spans, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  const iree_host_size_t prior_count = *span_count;
  const qwen_weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const qwen_weight_span_t* existing =
        qwen_weight_find(prior_count, spans, parameter.key);
    if (!count++) {
      shared = existing;
      if (shared) {
        if (shared->parameter.byte_offset < parameter.byte_offset) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "shared parameter root begins before storage");
        }
        origin = shared->parameter.byte_offset - parameter.byte_offset;
      }
    }
    if ((existing != NULL) != (shared != NULL) ||
        (existing &&
         (existing->buffer != shared->buffer ||
          existing->parameter.byte_offset != origin + parameter.byte_offset ||
          existing->parameter.byte_length != parameter.byte_length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter root cannot share existing placement");
    }
    if (qwen_weight_requires_preparation(parameter.key) &&
        parameter.byte_length != 61276160) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "FFN gate/up requires K5120/N17408 Q5 weights");
    }
  }
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty model parameter root");
  }
  if (shared) {
    return iree_hal_buffer_subspan(shared->buffer, origin,
                                   root.required_byte_length, allocator,
                                   out_buffer);
  }
  IREE_RETURN_IF_ERROR(qwen_weight_allocate(
      device, root.required_byte_length, root.minimum_alignment, out_buffer));
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (qwen_weight_span_t){
        .parameter = parameter,
        .buffer = *out_buffer,
        .prepare = qwen_weight_requires_preparation(parameter.key),
    };
  }
  return iree_ok_status();
}

static iree_status_t qwen_weight_enumerate(void* user_data, iree_host_size_t i,
                                           iree_string_view_t* out_key,
                                           iree_io_parameter_span_t* out_span) {
  const qwen_weight_group_t* group = user_data;
  const loom_cmd_program_parameter_t parameter = group->spans[i].parameter;
  *out_key = parameter.key;
  *out_span = (iree_io_parameter_span_t){
      .buffer_offset = parameter.byte_offset,
      .length = parameter.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t qwen_weight_stream(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_io_parameter_provider_t* provider,
    iree_hal_command_buffer_t* prepare, iree_host_size_t span_count,
    qwen_weight_span_t* spans, iree_allocator_t allocator) {
  qwen_weight_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, span_count, sizeof(*groups), (void**)&groups));
  iree_io_parameter_gather_t* gathers = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, span_count, sizeof(*gathers), (void**)&gathers);
  iree_hal_semaphore_t* reads[QWEN_WEIGHT_LANES] = {0};
  iree_hal_semaphore_t* ready[QWEN_WEIGHT_LANES] = {0};
  uint64_t frontiers[QWEN_WEIGHT_LANES] = {0};
  for (iree_host_size_t i = 0;
       i < QWEN_WEIGHT_LANES && iree_status_is_ok(status); ++i) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &reads[i]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready[i]);
    }
  }
  iree_host_size_t group_count = 0;
  uint64_t loaded_bytes = 0;
  uint64_t prepared_bytes = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      const qwen_weight_span_t* span = &spans[i];
      if (!i || span->prepare || spans[i - 1].prepare ||
          span->buffer != spans[i - 1].buffer) {
        groups[group_count++].spans = &spans[i];
      }
      ++groups[group_count - 1].count;
      loaded_bytes += span->parameter.byte_length;
      if (span->prepare) {
        prepared_bytes += span->parameter.byte_length;
      }
    }
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      qwen_weight_group_t* group = &groups[i];
      const iree_host_size_t lane = i % QWEN_WEIGHT_LANES;
      group->wait_value = frontiers[lane];
      group->signal_value = ++frontiers[lane];
      gathers[i] = (iree_io_parameter_gather_t){
          .target_buffer = group->spans[0].buffer,
          .count = group->count,
          .enumerator = {qwen_weight_enumerate, group},
          .wait_semaphore_list = {1, &ready[lane], &group->wait_value},
          .signal_semaphore_list = {1,
                                    group->spans[0].prepare ? &reads[lane]
                                                            : &ready[lane],
                                    &group->signal_value},
      };
    }
    fprintf(stderr,
            "Streaming %.3f GiB of unique weights; preparing %.3f GiB in "
            "place across %zu groups and %u lanes...\n",
            loaded_bytes / 1073741824.0, prepared_bytes / 1073741824.0,
            group_count, QWEN_WEIGHT_LANES);
    status = iree_io_parameter_provider_gather_batch(provider, device, transfer,
                                                     group_count, gathers);
  }
  for (iree_host_size_t i = 0; i < group_count && iree_status_is_ok(status);
       ++i) {
    qwen_weight_group_t* group = &groups[i];
    const qwen_weight_span_t* span = group->spans;
    if (!span->prepare) {
      continue;
    }
    const iree_host_size_t lane = i % QWEN_WEIGHT_LANES;
    const iree_hal_buffer_binding_t binding = {
        span->buffer, span->parameter.byte_offset, span->parameter.byte_length};
    status = iree_hal_queue_execute(
        dispatch,
        (iree_hal_semaphore_list_t){1, &reads[lane], &group->signal_value},
        (iree_hal_semaphore_list_t){1, &ready[lane], &group->signal_value},
        prepare, (iree_hal_buffer_binding_table_t){1, &binding},
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(
        (iree_hal_semaphore_list_t){QWEN_WEIGHT_LANES, ready, frontiers},
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    // Release waits for preparations that were never submitted. Queue-owned
    // file, command and buffer references survive until those operations
    // retire.
    for (iree_host_size_t i = 0; i < QWEN_WEIGHT_LANES; ++i) {
      if (reads[i]) {
        iree_hal_semaphore_fail(reads[i], iree_status_clone(status));
      }
      if (ready[i]) {
        iree_hal_semaphore_fail(ready[i], iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0; i < QWEN_WEIGHT_LANES; ++i) {
    iree_hal_semaphore_release(reads[i]);
    iree_hal_semaphore_release(ready[i]);
  }
  iree_allocator_free(allocator, gathers);
  iree_allocator_free(allocator, groups);
  return status;
}

iree_status_t loom_serve_qwen_weights_load(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_jit_t* jit,
    iree_hal_command_buffer_mode_t command_mode,
    iree_host_size_t target_stage_count, iree_host_size_t stage_count,
    const loom_serve_qwen_weight_stage_t* stages,
    iree_string_view_t weights_path, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  loom_serve_jit_stage_t* compiled = NULL;
  const loomc_config_options_t config = {
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  iree_status_t status = loom_serve_jit_compile(
      jit, IREE_SV("qwen38_prepare_ffn"), &config, &compiled);
  iree_hal_command_buffer_t* prepare = NULL;
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_requirements_t* requirements =
        &loom_serve_jit_stage_program(compiled)->requirements;
    if (requirements->fixed_buffer_count ||
        requirements->rebindable_binding_count != 1 ||
        requirements->transient.required_byte_length ||
        requirements->launch_counts.required_byte_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight preparer requires one mutable binding "
                                "and no global scratch");
    } else {
      status =
          loom_serve_jit_stage_record(compiled, iree_hal_queue_family(dispatch),
                                      command_mode, NULL, &prepare);
    }
  }
  iree_host_size_t capacity = stages[0].program->parameters.count;
  for (iree_host_size_t i = target_stage_count; i < stage_count; ++i) {
    capacity += stages[i].program->parameters.count;
  }
  qwen_weight_span_t* spans = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*spans), (void**)&spans);
  }
  iree_host_size_t span_count = 0;
  if (iree_status_is_ok(status)) {
    loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(stages[0].program, 0);
    for (iree_host_size_t i = 1; i < target_stage_count; ++i) {
      const loom_cmd_program_parameter_root_t other =
          loom_cmd_program_parameter_root_at(stages[i].program, 0);
      root.required_byte_length =
          iree_max(root.required_byte_length, other.required_byte_length);
      root.minimum_alignment =
          iree_max(root.minimum_alignment, other.minimum_alignment);
    }
    status =
        qwen_weight_resolve_root(device, stages[0].program, root, &span_count,
                                 spans, host_allocator, &stages[0].buffers[0]);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 1; i < target_stage_count; ++i) {
      stages[i].buffers[0] = stages[0].buffers[0];
      iree_hal_buffer_retain(stages[i].buffers[0]);
    }
  }
  for (iree_host_size_t i = target_stage_count;
       i < stage_count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_t* program = stages[i].program;
    for (uint32_t r = 0;
         r < program->parameter_roots.count && iree_status_is_ok(status); ++r) {
      const loom_cmd_program_parameter_root_t root =
          loom_cmd_program_parameter_root_at(program, r);
      status = qwen_weight_resolve_root(
          device, program, root, &span_count, spans, host_allocator,
          &stages[i].buffers[root.fixed_buffer_index]);
    }
  }
  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_parameter_file_to_index(weights_path, index,
                                                         host_allocator);
  }
  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(iree_string_view_empty(),
                                                     index, QWEN_WEIGHT_LANES,
                                                     host_allocator, &provider);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_weight_stream(device, transfer, dispatch, provider, prepare,
                                span_count, spans, host_allocator);
  }
  iree_io_parameter_provider_release(provider);
  iree_io_parameter_index_release(index);
  iree_allocator_free(host_allocator, spans);
  iree_hal_command_buffer_release(prepare);
  loom_serve_jit_stage_destroy(compiled);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
