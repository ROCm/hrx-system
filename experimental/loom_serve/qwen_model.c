// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared Qwen3.8-27B residency and retained row state.

#include "experimental/loom_serve/qwen_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/block_pool.h"
#include "experimental/loom_serve/command.h"
#include "experimental/loom_serve/execution.h"
#include "experimental/loom_serve/jit.h"
#include "experimental/loom_serve/module.h"
#include "experimental/loom_serve/program.h"
#include "experimental/loom_serve/weights.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/path.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/init.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/sync.h"

enum {
  QWEN_RESIDUAL = 0,
  QWEN_CONTROL = 1,
  QWEN_GDN = 2,
  QWEN_ATTENTION = 3,
  QWEN_TOKENS = 4,
  QWEN_PROGRESS = 5,
  QWEN_WORKSPACE = 6,
  QWEN_BINDING_COUNT = 7,
  QWEN_TOKEN_CAPACITY = 512,
  QWEN_ROW_CAPACITY = LOOM_SERVE_QWEN_ROW_CAPACITY,
  QWEN_BLOCK_TOKENS = 64,
  QWEN_MAP_ORIGIN = 1024,
  QWEN_SPAN_CAPACITY = 64,
  QWEN_EPOCH_SELECTION = 3 + 5 * QWEN_SPAN_CAPACITY,
  QWEN_EPOCH_WORDS = QWEN_EPOCH_SELECTION + QWEN_SPAN_CAPACITY,
  QWEN_VERIFY_OPTIONS = QWEN_EPOCH_WORDS,
  QWEN_VERIFY_EOS = QWEN_VERIFY_OPTIONS + 2 * QWEN_SPAN_CAPACITY,
  QWEN_VERIFY_WORDS = QWEN_VERIFY_EOS + 1,
};

typedef struct qwen_stage_t {
  // Unique native export name, established during cold loading.
  char name[32];
  // Owned JIT command image and loaded executable entries.
  loom_serve_jit_stage_t* compiled;
  // Parsed view borrowing compiled.
  loom_cmd_program_t program;
  // Reusable command retaining its executable and fixed weight resources.
  iree_hal_command_buffer_t* command;
  // Owned views of immutable parameter groups, in fixed-root order.
  iree_hal_buffer_t* fixed_buffers[7];
} qwen_stage_t;

struct loom_serve_qwen_model_t {
  // Host allocation policy used for all owned resources.
  iree_allocator_t allocator;
  // Async I/O service outliving device teardown.
  iree_async_proactor_pool_t* proactor_pool;
  // Completion registry outliving the device group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Owned GPU device and its allocation domain.
  iree_hal_device_t* device;
  // Device group owning the semaphore namespace.
  iree_hal_device_group_t* group;
  // Borrowed exact dispatch queue, retained by execution.
  iree_hal_queue_t* dispatch;
  // Borrowed exact transfer queue, retained by execution.
  iree_hal_queue_t* transfer;
  // Ordered stage/input submissions with independently retired feedback.
  loom_serve_execution_t* execution;
  // Optional flag-selected profiling session, ended after accepted work drains.
  iree_hal_profiling_from_flags_t* profiling;
  // Cold recording policy for every reusable stage in this residency.
  iree_hal_command_buffer_mode_t command_mode;
  // Shared source index and compiler for this model residency.
  loom_serve_jit_t* jit;
  // Prefill, decode, then packed-epoch programs, owned in option order.
  qwen_stage_t* stages;
  // Number of allocated stages, including partially initialized cold entries.
  iree_host_size_t stage_count;
  // Number of target packed shapes, independent of auxiliary MTP stages.
  iree_host_size_t shape_count;
  // Immutable prepared shape capacities, owned in epoch option order.
  loom_serve_qwen_shape_t* shapes;
  // Cold-resolved native exports in the one shared VM process.
  iree_vm_function_t* functions;
  // Context specialization shared by every compiled stage.
  iree_host_size_t context_capacity;
  // Maximum active input chunk accepted by the selected row-prefill path.
  iree_host_size_t prefill_capacity;
  // Number of preallocated rows.
  iree_host_size_t row_count;
  // Fixed row records, owned by this model.
  loom_serve_qwen_row_t* rows;
  // Private recurrent/control spans followed by pooled KV, or dense row KV.
  iree_hal_buffer_t* row_arena;
  // Private-page ownership shared by target and draft cache planes.
  struct {
    // Physical token capacity shared by rows; zero selects dense comparison.
    iree_host_size_t capacity;
    // Maximum logical block count per row, independent of physical capacity.
    iree_host_size_t blocks_per_row;
    // Fixed free-ID metadata; owned IDs are returned only after retirement.
    loom_serve_block_pool_t pool;
    // Fixed row-major host maps, retained through queued uploads.
    uint32_t* maps;
  } cache;
  // Model-wide residual storage, serialized by the execution timeline.
  iree_hal_buffer_t* residual;
  // Model-wide packed transient storage for the larger stage.
  iree_hal_buffer_t* workspace;
  // Environment owning canonical VM reference types.
  iree_vm_environment_t* environment;
  // HAL reference type handles borrowed from environment.
  iree_hal_module_types_t types;
  // Native stage submission module retained by program.
  iree_vm_module_t* native_module;
  // Shared source-JIT model program; request state is not VM process state.
  loom_serve_program_t* program;
  // Cold-resolved model step function.
  iree_vm_function_t step;
  // Reusable packed-epoch bindings and transfer payloads. Device buffers at
  // slots 1, 2, 4 and 5 are owned; other slots borrow model-wide storage.
  struct {
    // Residual, metadata, origins, state arena, input IDs, output IDs,
    // workspace.
    iree_hal_buffer_t* buffers[QWEN_BINDING_COUNT];
    // Immutable cold GDN/KV byte origins for each resident row.
    int64_t origins[QWEN_ROW_CAPACITY][2];
    // Header, spans, selected-row map, and optional verification controls.
    int32_t metadata[QWEN_VERIFY_WORDS];
    // Padded input payload, retained until completion or destruction drains it.
    int32_t tokens[QWEN_TOKEN_CAPACITY];
    // Compact predictions downloaded before committing the row records.
    int32_t outputs[QWEN_ROW_CAPACITY];
  } epoch;
  // Optional block-64 residency and private proposal work. Warm stages consume
  // committed target residuals before the next target invocation can reuse
  // them.
  struct {
    // First auxiliary stage: propose, warm shapes, then verify shapes.
    iree_host_size_t first_stage;
    // Normalized committed target hidden, indexed by retained row.
    iree_hal_buffer_t* carry;
    // Device-produced accepted span metadata consumed by MTP catch-up.
    iree_hal_buffer_t* committed;
    // Device-produced {consumed, output_count, token[4]} records.
    iree_hal_buffer_t* results;
    // Stable readback backing, retained through completion or terminal drain.
    int32_t records[QWEN_ROW_CAPACITY][6];
    // Private one-layer attention cache shared by proposal and catch-up.
    iree_hal_buffer_t* cache;
    // Immutable resident-row origins consumed by proposal and catch-up.
    iree_hal_buffer_t* row_table;
    // Immutable row origins in the private one-layer attention cache.
    int64_t origins[QWEN_ROW_CAPACITY][2];
  } mtp;
  // Tokenizer owning its parsed vocabulary and encoding data.
  iree_tokenizer_t* tokenizer;
  // Token identifying the end of a generated assistant turn.
  int32_t eos_token;
};

struct loom_serve_qwen_row_t {
  // Borrowed owning model; all row operations use its single invocation.
  loom_serve_qwen_model_t* model;
  // Private spans in the row arena; residual/workspace entries borrow model.
  iree_hal_buffer_t* buffers[QWEN_BINDING_COUNT];
  // Number of input tokens actually consumed into KV/GDN state.
  iree_host_size_t position;
  // Owned physical pages, including any in-flight speculative suffix.
  uint32_t block_count;
  // Whether the latest completed work selected a token at the current position.
  bool has_prediction;
  // Completed stage counts and end-to-end host durations.
  loom_serve_qwen_metrics_t metrics;
  // Host payloads remain alive until the model drains accepted transfers.
  struct {
    // Active prefill count, absolute base, and EOS ID.
    int32_t control[3];
    // Fixed-capacity padded upload and current-token readback at slot zero.
    int32_t tokens[QWEN_TOKEN_CAPACITY];
    // Device-generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } transfer;
};

static iree_string_view_t qwen_file_text(iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_hal_queue_t* qwen_select_queue(
    iree_hal_device_t* device, iree_hal_queue_family_role_flags_t role) {
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    if (queues->families[i].provisioned_queue_count &&
        iree_all_bits_set(queues->families[i].role_flags, role)) {
      return iree_hal_device_queue(device, i, 0);
    }
  }
  return NULL;
}

static iree_status_t qwen_create_device(loom_serve_qwen_model_t* runner) {
  iree_hal_driver_registry_t* registry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_driver_registry_allocate(runner->allocator, &registry));
  iree_status_t status = iree_hal_register_all_available_drivers(registry);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), NULL,
        iree_async_proactor_pool_options_default(), runner->allocator,
        &runner->proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = runner->proactor_pool;
    params.event_sink = iree_hal_device_event_sink_stderr();
    status = iree_hal_create_device(registry, IREE_SV("amdgpu"), &params,
                                    runner->allocator, &runner->device);
  }
  iree_hal_driver_registry_free(registry);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), runner->allocator,
      &runner->frontier_tracker));
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      runner->device, runner->frontier_tracker, runner->allocator,
      &runner->group));
  runner->dispatch = qwen_select_queue(
      runner->device, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
  runner->transfer = qwen_select_queue(
      runner->device, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER);
  if (!runner->dispatch || !runner->transfer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device needs dispatch and transfer queues");
  }
  return loom_serve_execution_create(runner->dispatch, runner->transfer,
                                     runner->allocator, &runner->execution);
}

static iree_status_t qwen_check_layouts(loom_serve_qwen_model_t* runner) {
  const loom_cmd_program_t* prefill = &runner->stages[0].program;
  for (iree_host_size_t i = 0; i < runner->shape_count + 2; ++i) {
    const loom_cmd_program_t* program = &runner->stages[i].program;
    if (program->requirements.fixed_buffer_count != 1 ||
        program->parameter_roots.count != 1 ||
        program->requirements.rebindable_binding_count != QWEN_BINDING_COUNT ||
        program->requirements.transient.binding_index != QWEN_WORKSPACE ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the Qwen row layout");
    }
    if (prefill->parameters.count != program->parameters.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter counts differ");
    }
    for (uint32_t i = 0; i < prefill->parameters.count; ++i) {
      const loom_cmd_program_parameter_t lhs =
          loom_cmd_program_parameter_at(prefill, i);
      const loom_cmd_program_parameter_t rhs =
          loom_cmd_program_parameter_at(program, i);
      if (!iree_string_view_equal(lhs.key, rhs.key) ||
          lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.byte_offset != rhs.byte_offset ||
          lhs.byte_length != rhs.byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stage parameter placement %u differs", i);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t qwen_allocate_buffer(loom_serve_qwen_model_t* runner,
                                          iree_device_size_t length,
                                          iree_device_size_t minimum_alignment,
                                          iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = minimum_alignment;
  return iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(runner->device), params, length, out_buffer);
}

// Weight placement and preparation are cold model-wide work. Session rows
// never own weights and every compiled shape retains views of this residency.
static iree_status_t qwen_load_weights(loom_serve_qwen_model_t* model,
                                       iree_string_view_t weights_path,
                                       iree_string_view_t source_directory) {
  loom_serve_weight_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, model->stage_count, sizeof(*stages), (void**)&stages));
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    stages[i] = (loom_serve_weight_stage_t){
        .program = &model->stages[i].program,
        .buffers = model->stages[i].fixed_buffers,
    };
  }
  char* policy_path = NULL;
  iree_status_t status =
      iree_file_path_join(source_directory, IREE_SV("weights.loom"),
                          model->allocator, &policy_path);
  if (iree_status_is_ok(status)) {
    status = loom_serve_weights_load(
        model->device, model->transfer, model->dispatch, model->jit,
        model->command_mode, model->shape_count + 2, model->stage_count, stages,
        weights_path, iree_make_cstring_view(policy_path), model->allocator);
  }
  iree_allocator_free(model->allocator, policy_path);
  iree_allocator_free(model->allocator, stages);
  return status;
}

static iree_status_t qwen_create_program(loom_serve_qwen_model_t* runner,
                                         iree_string_view_t source_directory) {
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(runner->allocator, &runner->environment));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(runner->environment, &table));
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &runner->types));
  loom_serve_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(runner->allocator, runner->stage_count,
                                  sizeof(*stages), (void**)&stages));
  for (iree_host_size_t i = 0; i < runner->stage_count; ++i) {
    stages[i] = (loom_serve_stage_t){
        iree_make_cstring_view(runner->stages[i].name),
        runner->stages[i].command,
        runner->stages[i].program.requirements.rebindable_binding_count};
  }
  iree_status_t status = loom_serve_module_create(
      &runner->types, runner->execution, runner->stage_count, stages,
      runner->allocator, &runner->native_module);
  iree_allocator_free(runner->allocator, stages);
  IREE_RETURN_IF_ERROR(status);
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      source_directory, IREE_SV("control.loom"), runner->allocator, &path));
  iree_vm_module_t* libraries[] = {runner->native_module};
  status = loom_serve_program_create(
      runner->environment, iree_make_cstring_view(path), IREE_SV("step"),
      iree_vm_module_span_from_array(libraries), runner->allocator,
      &runner->program);
  iree_allocator_free(runner->allocator, path);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_process_t* process = loom_serve_program_process(runner->program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("step"), &runner->step));
  for (iree_host_size_t i = 2;
       i < runner->stage_count && iree_status_is_ok(status); ++i) {
    status = iree_vm_process_lookup_function(
        process, IREE_SV("runner"),
        iree_make_cstring_view(runner->stages[i].name), &runner->functions[i]);
  }
  return status;
}

static iree_status_t qwen_load_tokenizer(loom_serve_qwen_model_t* model,
                                         iree_string_view_t path) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 model->allocator, &contents));
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      qwen_file_text(contents), model->allocator, &model->tokenizer);
  iree_io_file_contents_free(contents);
  IREE_RETURN_IF_ERROR(status);
  model->eos_token = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(model->tokenizer), IREE_SV("<|im_end|>"));
  if (model->eos_token < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no <|im_end|>");
  }
  return iree_ok_status();
}

static uint64_t qwen_row_table_length(const loom_serve_qwen_model_t* model) {
  return model->cache.capacity
             ? QWEN_MAP_ORIGIN + model->row_count *
                                     model->cache.blocks_per_row *
                                     sizeof(uint32_t)
             : sizeof(model->epoch.origins);
}

static iree_status_t qwen_allocate_rows(loom_serve_qwen_model_t* model) {
  if (model->cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_block_pool_initialize(
        (uint32_t)(model->cache.capacity / QWEN_BLOCK_TOKENS), model->allocator,
        &model->cache.pool));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->allocator, model->row_count * model->cache.blocks_per_row,
        sizeof(*model->cache.maps), (void**)&model->cache.maps));
  }
  uint64_t workspace_length = 0;
  uint64_t workspace_alignment = 0;
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    workspace_length = iree_max(
        workspace_length,
        model->stages[i].program.requirements.transient.required_byte_length);
    workspace_alignment = iree_max(
        workspace_alignment,
        model->stages[i].program.requirements.transient.minimum_alignment);
  }
  IREE_RETURN_IF_ERROR(
      qwen_allocate_buffer(model, 10485760, 256, &model->residual));
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(
      model, workspace_length, workspace_alignment, &model->workspace));
  const uint64_t lengths[QWEN_BINDING_COUNT] = {
      0,
      12,
      156893184,
      model->cache.capacity ? 0 : model->context_capacity * 65536ull,
      QWEN_TOKEN_CAPACITY * sizeof(int32_t),
      8 * sizeof(int32_t),
      0,
  };
  uint64_t offsets[QWEN_BINDING_COUNT] = {0};
  uint64_t stride = 0;
  for (iree_host_size_t i = QWEN_CONTROL; i <= QWEN_PROGRESS; ++i) {
    offsets[i] = stride;
    stride += (lengths[i] + 255) & ~255ull;
  }
  const uint64_t row_bytes = stride * model->row_count;
  const uint64_t pool_bytes = model->cache.capacity * 65536ull;
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, row_bytes + pool_bytes, 256,
                                            &model->row_arena));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->allocator, model->row_count,
                                  sizeof(*model->rows), (void**)&model->rows));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_qwen_row_t* row = &model->rows[i];
    row->model = model;
    row->buffers[QWEN_RESIDUAL] = model->residual;
    row->buffers[QWEN_WORKSPACE] = model->workspace;
    model->epoch.origins[i][0] = (int64_t)(i * stride + offsets[QWEN_GDN]);
    model->epoch.origins[i][1] =
        (int64_t)(model->cache.capacity ? row_bytes
                                        : i * stride + offsets[QWEN_ATTENTION]);
    for (iree_host_size_t binding = QWEN_CONTROL;
         binding <= QWEN_PROGRESS && iree_status_is_ok(status); ++binding) {
      if (lengths[binding]) {
        status = iree_hal_buffer_subspan(
            model->row_arena, i * stride + offsets[binding], lengths[binding],
            model->allocator, &row->buffers[binding]);
      }
    }
  }
  if (iree_status_is_ok(status) && model->stage_count > 2) {
    model->epoch.buffers[0] = model->residual;
    model->epoch.buffers[3] = model->row_arena;
    model->epoch.buffers[6] = model->workspace;
    const uint64_t epoch_lengths[] = {
        0, sizeof(model->epoch.metadata), qwen_row_table_length(model),
        0, sizeof(model->epoch.tokens),   sizeof(model->epoch.outputs),
        0,
    };
    for (iree_host_size_t i = 0;
         i < QWEN_BINDING_COUNT && iree_status_is_ok(status); ++i) {
      if (epoch_lengths[i]) {
        status = qwen_allocate_buffer(model, epoch_lengths[i], 256,
                                      &model->epoch.buffers[i]);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const uint32_t zero = 0;
    iree_hal_buffer_t* buffers[] = {model->row_arena, model->residual,
                                    model->workspace};
    iree_hal_transfer_operation_t fills[IREE_ARRAYSIZE(buffers)] = {0};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
      fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
      fills[i].fill.target_buffer = buffers[i];
      fills[i].fill.length =
          i == 0 ? row_bytes : iree_hal_buffer_byte_length(buffers[i]);
      fills[i].fill.pattern = &zero;
      fills[i].fill.pattern_length = sizeof(zero);
    }
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(fills), fills, &completion);
    if (iree_status_is_ok(status) && model->stage_count > 2) {
      const iree_hal_transfer_operation_t upload = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = model->epoch.origins,
                     .target_buffer = model->epoch.buffers[2],
                     .length = sizeof(model->epoch.origins)},
      };
      status = loom_serve_execution_transfer(model->execution, 1, &upload,
                                             &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "Residency: %zu rows, %.3f GiB retained arena, %.3f GiB shared "
            "workspace, %zu-token context, %zu-token prefill capacity.\n",
            model->row_count, (row_bytes + pool_bytes) / 1073741824.0,
            workspace_length / 1073741824.0, model->context_capacity,
            model->prefill_capacity);
  }
  return status;
}

static iree_status_t qwen_allocate_mtp(loom_serve_qwen_model_t* model) {
  if (!model->mtp.first_stage) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, model->row_count * 20480,
                                            256, &model->mtp.carry));
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, QWEN_EPOCH_WORDS * 4, 256,
                                            &model->mtp.committed));
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, sizeof(model->mtp.records),
                                            256, &model->mtp.results));
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, qwen_row_table_length(model),
                                            256, &model->mtp.row_table));
  iree_status_t status = qwen_allocate_buffer(
      model,
      (model->cache.capacity ? model->cache.capacity
                             : model->row_count * model->context_capacity) *
          4096ull,
      256, &model->mtp.cache);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      model->mtp.origins[i][1] =
          model->cache.capacity
              ? 0
              : (int64_t)(i * model->context_capacity * 4096ull);
    }
    const uint32_t zero = 0;
    const iree_hal_transfer_operation_t transfers[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
         .fill = {.target_buffer = model->mtp.carry,
                  .length = iree_hal_buffer_byte_length(model->mtp.carry),
                  .pattern = &zero,
                  .pattern_length = sizeof(zero)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->mtp.origins,
                    .target_buffer = model->mtp.row_table,
                    .length = sizeof(model->mtp.origins)}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(transfers), transfers, &completion);
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
  }
  return status;
}

// The adapter owns model dimensions; the compiler owns placement, launch
// counts, and native entry mapping. Shape values are applied once during
// command materialization and travel with its kernel source requests.
static iree_status_t qwen_compile_stage(loom_serve_qwen_model_t* model,
                                        iree_string_view_t root,
                                        loom_serve_qwen_shape_t shape,
                                        iree_host_size_t token_capacity,
                                        iree_host_size_t q8_capacity,
                                        qwen_stage_t* stage) {
  const struct {
    // Model-authored configuration symbol.
    const char* key;
    // Integer specialization supplied to command and kernel materialization.
    iree_host_size_t value;
  } values[] = {
      {"runner.qwen38.prefill_token_count", shape.token_capacity},
      {"runner.qwen38.span_capacity", shape.span_capacity},
      {"ggml.linear_q4k_q8_1_x4.token_capacity", token_capacity},
      {"ggml.linear_q4k_q8_1_x4.output_capacity", 48},
      {"ggml.linear_q5k_q8_1_x4.token_capacity", token_capacity},
      {"ggml.linear_q5k_q8_1_x4.output_capacity", 17408},
      {"ggml.linear_q6k_f32_decode.output_capacity", 5120},
      {"ggml.linear_q6k_q8_1_x4.token_capacity", token_capacity},
      {"ggml.linear_q6k_q8_1_x4.output_capacity", 248320},
      {"ggml.linear_q8_0_q8_1_x4.token_capacity", q8_capacity},
      {"ggml.linear_q8_0_q8_1_x4.output_capacity",
       q8_capacity == 1 ? 1024 : 5120},
      {"ggml.quantize_q8_1_x4.group_capacity", 136 * token_capacity},
      {"qwen38.attention.cache_capacity", model->context_capacity},
      {"qwen38.attention.pool_capacity", model->cache.capacity},
      {"qwen38.attention.decode_split_count", 10},
      {"qwen38.ffn.input_size", 5120},
      {"qwen38.ffn.output_size", 17408},
      {"qwen38.greedy_argmax.output_capacity", 248320},
  };
  char text[IREE_ARRAYSIZE(values)][32];
  loomc_config_binding_t bindings[IREE_ARRAYSIZE(values)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    snprintf(text[i], sizeof(text[i]), "%zu", values[i].value);
    bindings[i] =
        (loomc_config_binding_t){loomc_make_cstring_view(values[i].key),
                                 loomc_make_cstring_view(text[i])};
  }
  const loomc_config_options_t config = {
      .bindings = bindings,
      .binding_count = IREE_ARRAYSIZE(bindings),
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  IREE_RETURN_IF_ERROR(
      loom_serve_jit_compile(model->jit, root, &config, &stage->compiled));
  stage->program = *loom_serve_jit_stage_program(stage->compiled);
  return iree_ok_status();
}

static iree_status_t qwen_initialize(loom_serve_qwen_model_t* model,
                                     const loom_serve_qwen_options_t* options) {
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  model->command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  model->shape_count = options->epoch_count;
  model->prefill_capacity = options->prefill_capacity;
  if (options->enable_mtp || options->pool_capacity) {
    iree_host_size_t packed_capacity = 0;
    for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
      packed_capacity =
          iree_max(packed_capacity, options->epoch_shapes[i].token_capacity);
    }
    // Pooled/MTP single-row prefill uses the same packed addressing path.
    // Publish that path's usable capacity to every frontend before it chunks.
    model->prefill_capacity =
        iree_min(model->prefill_capacity, packed_capacity);
  }
  model->context_capacity = options->context_capacity;
  model->cache.capacity = options->pool_capacity;
  model->cache.blocks_per_row =
      (options->context_capacity + QWEN_BLOCK_TOKENS - 1) / QWEN_BLOCK_TOKENS;
  iree_host_size_t stage_count = model->shape_count + 2;
  if (options->enable_mtp) {
    model->mtp.first_stage = stage_count;
    stage_count += 2 * model->shape_count + 1;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, stage_count, sizeof(*model->stages),
      (void**)&model->stages));
  model->stage_count = stage_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, stage_count, sizeof(*model->functions),
      (void**)&model->functions));
  if (model->shape_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        model->allocator,
        iree_make_const_byte_span(options->epoch_shapes,
                                  model->shape_count * sizeof(*model->shapes)),
        (void**)&model->shapes));
  }
  IREE_RETURN_IF_ERROR(qwen_create_device(model));
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(
      model->device, model->dispatch, options->source_directory,
      &options->kernel_sanitizer, model->allocator, &model->jit));
  IREE_RETURN_IF_ERROR(qwen_load_tokenizer(model, options->tokenizer_path));
  snprintf(model->stages[0].name, sizeof(model->stages[0].name), "prefill");
  snprintf(model->stages[1].name, sizeof(model->stages[1].name), "decode");
  const loom_serve_qwen_shape_t isolated = {options->prefill_capacity,
                                            model->row_count};
  IREE_RETURN_IF_ERROR(qwen_compile_stage(model, IREE_SV("qwen38_prefill"),
                                          isolated, QWEN_TOKEN_CAPACITY, 1,
                                          &model->stages[0]));
  IREE_RETURN_IF_ERROR(qwen_compile_stage(model,
                                          IREE_SV("qwen38_text_decode_greedy"),
                                          isolated, 1, 1, &model->stages[1]));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->shape_count && iree_status_is_ok(status); ++i) {
    qwen_stage_t* stage = &model->stages[i + 2];
    snprintf(stage->name, sizeof(stage->name), "epoch_%zu", i);
    status =
        qwen_compile_stage(model, IREE_SV("qwen38_epoch"), model->shapes[i],
                           QWEN_TOKEN_CAPACITY, 1, stage);
  }
  for (iree_host_size_t i = model->shape_count + 2;
       i < stage_count && iree_status_is_ok(status); ++i) {
    qwen_stage_t* stage = &model->stages[i];
    const iree_host_size_t ordinal = i - model->mtp.first_stage;
    const bool verifies = ordinal >= 1 + model->shape_count;
    const iree_host_size_t shape_index =
        ordinal == 0 ? 0 : (ordinal - 1) % model->shape_count;
    const loom_serve_qwen_shape_t shape =
        ordinal == 0 ? (loom_serve_qwen_shape_t){32, model->row_count}
                     : model->shapes[shape_index];
    const iree_string_view_t root = ordinal == 0 ? IREE_SV("qwen38_mtp_propose")
                                    : verifies   ? IREE_SV("qwen38_mtp_verify")
                                                 : IREE_SV("qwen38_mtp_warm");
    snprintf(stage->name, sizeof(stage->name), "mtp_%zu", ordinal);
    status = qwen_compile_stage(model, root, shape, QWEN_TOKEN_CAPACITY,
                                QWEN_TOKEN_CAPACITY, stage);
    if (iree_status_is_ok(status)) {
      const uint32_t fixed_count = ordinal == 0 ? 5 : verifies ? 1 : 7;
      const uint32_t binding_count = ordinal == 0 ? 6 : verifies ? 8 : 7;
      const loom_cmd_program_requirements_t requirements =
          stage->program.requirements;
      if (requirements.fixed_buffer_count != fixed_count ||
          stage->program.parameter_roots.count != fixed_count ||
          requirements.rebindable_binding_count != binding_count ||
          requirements.transient.binding_index != binding_count - 1 ||
          requirements.launch_counts.required_byte_length != 0) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "MTP command has an incompatible model layout");
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(qwen_check_layouts(model));
  IREE_RETURN_IF_ERROR(iree_hal_begin_device_group_profiling_from_flags(
      model->group, model->allocator, &model->profiling));
  IREE_RETURN_IF_ERROR(qwen_load_weights(model, options->weights_path,
                                         options->source_directory));
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    qwen_stage_t* stage = &model->stages[i];
    status = loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(model->dispatch),
        model->command_mode, stage->fixed_buffers, &stage->command);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(qwen_create_program(model, options->source_directory));
  IREE_RETURN_IF_ERROR(qwen_allocate_rows(model));
  IREE_RETURN_IF_ERROR(qwen_allocate_mtp(model));
  return iree_ok_status();
}

iree_status_t loom_serve_qwen_model_create(
    const loom_serve_qwen_options_t* options, iree_allocator_t host_allocator,
    loom_serve_qwen_model_t** out_model) {
  *out_model = NULL;
  if (options->row_count < 1 || options->row_count > QWEN_ROW_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen row count must be in [1, 16]");
  }
  if (options->prefill_capacity < 1 ||
      options->prefill_capacity > QWEN_TOKEN_CAPACITY ||
      options->context_capacity < options->prefill_capacity ||
      options->context_capacity > 262144 ||
      (options->enable_mtp && !options->epoch_count) ||
      options->pool_capacity > 4194304 ||
      options->pool_capacity % QWEN_BLOCK_TOKENS ||
      (options->pool_capacity && !options->epoch_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid model capacities; pooled KV and MTP require epoch shapes");
  }
  for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
    const loom_serve_qwen_shape_t shape = options->epoch_shapes[i];
    if (shape.token_capacity < 1 ||
        shape.token_capacity > QWEN_TOKEN_CAPACITY ||
        shape.token_capacity > options->context_capacity ||
        shape.span_capacity < 1 || shape.span_capacity > QWEN_ROW_CAPACITY) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch %zu has invalid token/span capacities", i);
    }
  }
  loom_serve_qwen_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  model->row_count = options->row_count;
  iree_status_t status = qwen_initialize(model, options);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  }
  return status;
}

iree_status_t loom_serve_qwen_model_destroy(loom_serve_qwen_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = model->execution
                             ? loom_serve_execution_drain(model->execution)
                             : iree_ok_status();
  status = iree_status_join(
      status, iree_hal_end_profiling_from_flags(model->profiling));
  loom_serve_program_destroy(model->program);
  iree_vm_module_release(model->native_module);
  iree_vm_environment_free(model->environment);
  if (model->rows) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      for (iree_host_size_t binding = QWEN_CONTROL; binding <= QWEN_PROGRESS;
           ++binding) {
        iree_hal_buffer_release(model->rows[i].buffers[binding]);
      }
    }
  }
  iree_allocator_free(model->allocator, model->rows);
  iree_allocator_free(model->allocator, model->cache.maps);
  loom_serve_block_pool_deinitialize(&model->cache.pool);
  iree_hal_buffer_release(model->epoch.buffers[1]);
  iree_hal_buffer_release(model->epoch.buffers[2]);
  iree_hal_buffer_release(model->epoch.buffers[4]);
  iree_hal_buffer_release(model->epoch.buffers[5]);
  iree_hal_buffer_release(model->mtp.carry);
  iree_hal_buffer_release(model->mtp.committed);
  iree_hal_buffer_release(model->mtp.results);
  iree_hal_buffer_release(model->mtp.cache);
  iree_hal_buffer_release(model->mtp.row_table);
  iree_hal_buffer_release(model->row_arena);
  iree_hal_buffer_release(model->residual);
  iree_hal_buffer_release(model->workspace);
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    iree_hal_command_buffer_release(model->stages[i].command);
    loom_serve_jit_stage_destroy(model->stages[i].compiled);
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(model->stages[i].fixed_buffers); ++j) {
      iree_hal_buffer_release(model->stages[i].fixed_buffers[j]);
    }
  }
  iree_allocator_free(model->allocator, model->functions);
  iree_allocator_free(model->allocator, model->shapes);
  iree_allocator_free(model->allocator, model->stages);
  loom_serve_execution_release(model->execution);
  iree_hal_device_group_release(model->group);
  loom_serve_jit_destroy(model->jit);
  iree_hal_device_release(model->device);
  iree_async_frontier_tracker_release(model->frontier_tracker);
  iree_async_proactor_pool_release(model->proactor_pool);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model);
  return status;
}

loom_serve_qwen_row_t* loom_serve_qwen_model_row(loom_serve_qwen_model_t* model,
                                                 iree_host_size_t index) {
  return &model->rows[index];
}

iree_tokenizer_t* loom_serve_qwen_model_tokenizer(
    loom_serve_qwen_model_t* model) {
  return model->tokenizer;
}

iree_host_size_t loom_serve_qwen_model_context_capacity(
    const loom_serve_qwen_model_t* model) {
  return model->context_capacity;
}

loom_serve_qwen_pool_usage_t loom_serve_qwen_model_pool_usage(
    const loom_serve_qwen_model_t* model) {
  return (loom_serve_qwen_pool_usage_t){
      .block_size = QWEN_BLOCK_TOKENS,
      .capacity = model->cache.capacity,
      .available =
          (iree_host_size_t)model->cache.pool.available * QWEN_BLOCK_TOKENS,
  };
}

iree_host_size_t loom_serve_qwen_row_pool_usage(
    const loom_serve_qwen_row_t* row) {
  return (iree_host_size_t)row->block_count * QWEN_BLOCK_TOKENS;
}

iree_host_size_t loom_serve_qwen_model_prefill_capacity(
    const loom_serve_qwen_model_t* model) {
  return model->prefill_capacity;
}

iree_host_size_t loom_serve_qwen_model_shape_count(
    const loom_serve_qwen_model_t* model) {
  return model->shape_count;
}

const loom_serve_qwen_shape_t* loom_serve_qwen_model_shapes(
    const loom_serve_qwen_model_t* model) {
  return model->shapes;
}

// Only a retired frontier may release pages. A rejected suffix within the
// retained last page stays private; subsequent appends overwrite it before use.
static void qwen_trim_blocks(loom_serve_qwen_row_t* row) {
  loom_serve_qwen_model_t* model = row->model;
  if (!model->cache.capacity) {
    return;
  }
  const uint32_t keep =
      (uint32_t)((row->position + QWEN_BLOCK_TOKENS - 1) / QWEN_BLOCK_TOKENS);
  const iree_host_size_t row_index = (iree_host_size_t)(row - model->rows);
  loom_serve_block_pool_release(
      &model->cache.pool, row->block_count - keep,
      model->cache.maps + row_index * model->cache.blocks_per_row + keep);
  row->block_count = keep;
}

iree_status_t loom_serve_qwen_row_reset(loom_serve_qwen_row_t* row) {
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t fills[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[QWEN_GDN],
                .length = iree_hal_buffer_byte_length(row->buffers[QWEN_GDN]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[QWEN_PROGRESS],
                .length = sizeof(row->transfer.progress),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(fills), fills, &completion);
  if (iree_status_is_ok(status) && row->model->mtp.first_stage) {
    const iree_hal_transfer_operation_t clear_carry = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = row->model->mtp.carry,
                 .target_offset = (uint64_t)(row - row->model->rows) * 20480,
                 .length = 20480,
                 .pattern = &zero,
                 .pattern_length = sizeof(zero)},
    };
    status = loom_serve_execution_transfer(row->model->execution, 1,
                                           &clear_carry, &completion);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_wait(row->model->execution, completion);
  }
  if (iree_status_is_ok(status)) {
    row->position = 0;
    qwen_trim_blocks(row);
    row->has_prediction = false;
    memset(&row->transfer, 0, sizeof(row->transfer));
    memset(&row->metrics, 0, sizeof(row->metrics));
  }
  return status;
}

static iree_status_t qwen_invoke(loom_serve_qwen_model_t* model,
                                 iree_vm_function_t function,
                                 iree_vm_variant_span_t arguments) {
  iree_vm_variant_t results[1] = {0};
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->program), function,
                     arguments, iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(arguments);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

static iree_status_t qwen_step(loom_serve_qwen_row_t* row, int32_t initialize) {
  loom_serve_qwen_model_t* model = row->model;
  iree_vm_variant_t arguments[QWEN_BINDING_COUNT + 1] = {0};
  arguments[0] = iree_vm_variant_from_i32(initialize);
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    arguments[i + 1] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->types, row->buffers[i]);
  }
  IREE_RETURN_IF_ERROR(qwen_invoke(model, model->step,
                                   iree_vm_variant_span_from_array(arguments)));
  const iree_hal_transfer_operation_t downloads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[QWEN_TOKENS],
                    .target = row->transfer.tokens,
                    .length = sizeof(int32_t)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[QWEN_PROGRESS],
                    .target = row->transfer.progress,
                    .length = sizeof(row->transfer.progress)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_feedback(
      model->execution, IREE_ARRAYSIZE(downloads), downloads, &completion));
  return loom_serve_execution_feedback_wait(model->execution, completion);
}

static iree_status_t qwen_invoke_stage(loom_serve_qwen_model_t* model,
                                       iree_host_size_t stage_index,
                                       iree_hal_buffer_t* const* buffers) {
  const iree_host_size_t count =
      model->stages[stage_index].program.requirements.rebindable_binding_count;
  iree_vm_variant_t arguments[8] = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    arguments[i] =
        iree_hal_buffer_variant_from_ptr_borrowed(&model->types, buffers[i]);
  }
  return qwen_invoke(model, model->functions[stage_index],
                     iree_vm_variant_span_from_ptr(arguments, count));
}

// The whole epoch has passed its capacity check. Publish only newly assigned
// map entries on the existing ordered transfer path. The fixed host maps stay
// alive until completion, including partial submission failure and destruction.
static iree_status_t qwen_grow_blocks(loom_serve_qwen_model_t* model,
                                      iree_host_size_t span_count,
                                      const loom_serve_qwen_span_t* spans) {
  if (!model->cache.capacity) {
    return iree_ok_status();
  }
  iree_hal_transfer_operation_t uploads[2 * QWEN_ROW_CAPACITY] = {0};
  iree_host_size_t upload_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    loom_serve_qwen_row_t* row = &model->rows[span->row_index];
    const uint32_t needed =
        (uint32_t)((row->position + span->token_count + QWEN_BLOCK_TOKENS - 1) /
                   QWEN_BLOCK_TOKENS);
    const uint32_t count = needed - row->block_count;
    if (!count) {
      continue;
    }
    const iree_host_size_t map_index =
        span->row_index * model->cache.blocks_per_row + row->block_count;
    uint32_t* blocks = model->cache.maps + map_index;
    loom_serve_block_pool_acquire(&model->cache.pool, count, blocks);
    row->block_count = needed;
    uploads[upload_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = blocks,
                   .target_buffer = model->epoch.buffers[2],
                   .target_offset =
                       QWEN_MAP_ORIGIN + map_index * sizeof(*blocks),
                   .length = count * sizeof(*blocks)},
    };
    if (model->mtp.first_stage) {
      uploads[upload_count] = uploads[upload_count - 1];
      uploads[upload_count++].upload.target_buffer = model->mtp.row_table;
    }
  }
  if (!upload_count) {
    return iree_ok_status();
  }
  uint64_t completion = 0;
  return loom_serve_execution_transfer(model->execution, upload_count, uploads,
                                       &completion);
}

// One allocation-free epoch path owns input validation, immutable uploads,
// target/catch-up ordering, and the final host publication frontier.
static iree_status_t qwen_epoch(loom_serve_qwen_model_t* model,
                                iree_host_size_t shape_index,
                                iree_host_size_t span_count,
                                const loom_serve_qwen_span_t* spans,
                                const uint32_t* output_limits,
                                loom_serve_qwen_result_t* out_results) {
  if (shape_index >= model->shape_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packed Qwen shape index is not loaded");
  }
  const loom_serve_qwen_shape_t shape = model->shapes[shape_index];
  if (!span_count || span_count > shape.span_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "epoch span count exceeds stage capacity");
  }
  // Validate the caller's plan before changing reusable payloads or submitting
  // any work. The device consumes this one established partition invariant.
  uint32_t resident_mask = 0;
  iree_host_size_t token_count = 0;
  uint32_t required_blocks = 0;
  bool proposes = false;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    if (span->row_index >= model->row_count || !span->token_count ||
        span->token_count > shape.token_capacity - token_count ||
        (span->flags & ~(LOOM_SERVE_QWEN_SPAN_FLAG_SELECT |
                         LOOM_SERVE_QWEN_SPAN_FLAG_PROPOSE))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid Qwen epoch span %zu", i);
    }
    const uint32_t resident_bit = 1u << span->row_index;
    if (resident_mask & resident_bit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch repeats resident row %zu",
                              span->row_index);
    }
    if (span->token_count >
        model->context_capacity - model->rows[span->row_index].position) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "epoch span %zu exceeds remaining context", i);
    }
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_PROPOSE);
    if (generates && (!output_limits || !output_limits[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "proposal requires a speculative Qwen span");
    }
    if (output_limits && output_limits[i]) {
      const loom_serve_qwen_row_t* row = &model->rows[span->row_index];
      if (output_limits[i] > 4 || span->token_count != 4 ||
          !iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT) ||
          !row->has_prediction || row->transfer.progress[7] ||
          span->token_ids[0] != row->transfer.tokens[0]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid speculative Qwen span %zu", i);
      }
    }
    if (model->cache.capacity) {
      const loom_serve_qwen_row_t* row = &model->rows[span->row_index];
      required_blocks += (uint32_t)((row->position + span->token_count +
                                     QWEN_BLOCK_TOKENS - 1) /
                                    QWEN_BLOCK_TOKENS) -
                         row->block_count;
    }
    resident_mask |= resident_bit;
    token_count += span->token_count;
    proposes |= generates;
  }

  if (required_blocks > model->cache.pool.available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch needs %u KV blocks; %u are available",
                            required_blocks, model->cache.pool.available);
  }
  IREE_RETURN_IF_ERROR(qwen_grow_blocks(model, span_count, spans));

  memset(model->epoch.tokens, 0, sizeof(model->epoch.tokens));
  iree_host_size_t token_begin = 0;
  iree_host_size_t output_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    const loom_serve_qwen_row_t* row = &model->rows[span->row_index];
    int32_t* descriptor = &model->epoch.metadata[3 + 5 * i];
    descriptor[0] = (int32_t)span->token_count;
    descriptor[1] = (int32_t)row->position;
    descriptor[2] = (int32_t)token_begin;
    descriptor[3] = (int32_t)span->row_index;
    descriptor[4] = -1;
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_PROPOSE);
    memcpy(model->epoch.tokens + token_begin, span->token_ids,
           (generates ? 1 : span->token_count) * sizeof(int32_t));
    token_begin += span->token_count;
    if (iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT)) {
      descriptor[4] = (int32_t)output_count;
      const iree_host_size_t selections =
          output_limits && output_limits[i] ? 4 : 1;
      for (iree_host_size_t j = 0; j < selections; ++j) {
        model->epoch.metadata[QWEN_EPOCH_SELECTION + output_count++] =
            (int32_t)(token_begin - selections + j);
      }
    }
    if (output_limits) {
      model->epoch.metadata[QWEN_VERIFY_OPTIONS + 2 * i] =
          generates ? 2 : output_limits[i] != 0;
      model->epoch.metadata[QWEN_VERIFY_OPTIONS + 2 * i + 1] =
          (int32_t)output_limits[i];
    }
  }
  model->epoch.metadata[0] = (int32_t)token_count;
  model->epoch.metadata[1] = (int32_t)span_count;
  model->epoch.metadata[2] = (int32_t)output_count;
  model->epoch.metadata[QWEN_VERIFY_EOS] = model->eos_token;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->epoch.metadata,
                  .target_buffer = model->epoch.buffers[1],
                  .length = sizeof(model->epoch.metadata)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->epoch.tokens,
                  .target_buffer = model->epoch.buffers[4],
                  .length = sizeof(model->epoch.tokens)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion));
  if (proposes) {
    iree_hal_buffer_t* proposal_buffers[] = {
        model->epoch.buffers[1], model->epoch.buffers[4], model->mtp.row_table,
        model->mtp.cache,        model->mtp.carry,        model->workspace,
    };
    IREE_RETURN_IF_ERROR(
        qwen_invoke_stage(model, model->mtp.first_stage, proposal_buffers));
  }
  if (output_limits) {
    iree_hal_buffer_t* verify_buffers[] = {
        model->residual,    model->epoch.buffers[1], model->epoch.buffers[2],
        model->row_arena,   model->epoch.buffers[4], model->mtp.committed,
        model->mtp.results, model->workspace,
    };
    IREE_RETURN_IF_ERROR(qwen_invoke_stage(
        model, model->mtp.first_stage + 1 + model->shape_count + shape_index,
        verify_buffers));
  } else {
    IREE_RETURN_IF_ERROR(
        qwen_invoke_stage(model, shape_index + 2, model->epoch.buffers));
  }
  // Results are immutable until this epoch retires. Download can fork from
  // target completion while cache-only catch-up consumes other target outputs.
  if (output_limits) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = model->mtp.results,
                     .target = model->mtp.records,
                     .length = span_count * sizeof(model->mtp.records[0])},
    };
    IREE_RETURN_IF_ERROR(loom_serve_execution_feedback(model->execution, 1,
                                                       &download, &completion));
  } else if (output_count) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = model->epoch.buffers[5],
                     .target = model->epoch.outputs,
                     .length = output_count * sizeof(int32_t)},
    };
    IREE_RETURN_IF_ERROR(loom_serve_execution_feedback(model->execution, 1,
                                                       &download, &completion));
  }
  if (model->mtp.first_stage) {
    iree_hal_buffer_t* warm_buffers[] = {
        model->residual,
        output_limits ? model->mtp.committed : model->epoch.buffers[1],
        model->mtp.row_table,
        model->mtp.cache,
        model->mtp.carry,
        model->epoch.buffers[4],
        model->workspace,
    };
    IREE_RETURN_IF_ERROR(qwen_invoke_stage(
        model, model->mtp.first_stage + 1 + shape_index, warm_buffers));
  }
  // The synchronous model boundary joins catch-up and feedback independently
  // before publishing host positions or reusing either branch's payloads.
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    loom_serve_qwen_row_t* row = &model->rows[span->row_index];
    if (output_limits) {
      const int32_t* record = model->mtp.records[i];
      out_results[i].consumed_count = (iree_host_size_t)record[0];
      out_results[i].output_count = (iree_host_size_t)record[1];
      memcpy(out_results[i].tokens, record + 2, sizeof(out_results[i].tokens));
      row->position += out_results[i].consumed_count;
      qwen_trim_blocks(row);
      row->has_prediction = out_results[i].output_count != 0;
      if (row->has_prediction) {
        row->transfer.tokens[0] =
            out_results[i].tokens[out_results[i].output_count - 1];
        row->transfer.progress[0] = record[1];
        row->transfer.progress[7] = row->transfer.tokens[0] == model->eos_token;
      }
      continue;
    }
    row->position += span->token_count;
    qwen_trim_blocks(row);
    const int32_t output = model->epoch.metadata[3 + 5 * i + 4];
    row->has_prediction = output >= 0;
    if (row->has_prediction) {
      row->transfer.tokens[0] = model->epoch.outputs[output];
      row->transfer.progress[0] = 1;
      row->transfer.progress[7] = row->transfer.tokens[0] == model->eos_token;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_qwen_model_epoch(loom_serve_qwen_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_qwen_span_t* spans) {
  return qwen_epoch(model, shape_index, span_count, spans, NULL, NULL);
}

iree_status_t loom_serve_qwen_model_verify(
    loom_serve_qwen_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_qwen_span_t* spans,
    const uint32_t* output_limits, loom_serve_qwen_result_t* out_results) {
  if (!model->mtp.first_stage) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "MTP artifacts were not loaded");
  }
  return qwen_epoch(model, shape_index, span_count, spans, output_limits,
                    out_results);
}

iree_status_t loom_serve_qwen_row_prefill(loom_serve_qwen_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids) {
  loom_serve_qwen_model_t* model = row->model;
  if (!count || count > model->prefill_capacity ||
      count > model->context_capacity - row->position) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prefill chunk exceeds stage or context capacity");
  }
  const iree_time_t start = iree_time_now();
  if (model->mtp.first_stage || model->cache.capacity) {
    iree_host_size_t shape_index = 0;
    // The public prefill capacity is bounded by the largest prepared shape.
    while (model->shapes[shape_index].token_capacity < count) {
      ++shape_index;
    }
    const loom_serve_qwen_span_t span = {
        (iree_host_size_t)(row - model->rows),
        count,
        token_ids,
        LOOM_SERVE_QWEN_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(
        loom_serve_qwen_model_epoch(model, shape_index, 1, &span));
    row->metrics.prefill_tokens += count;
    ++row->metrics.prefill_steps;
    row->metrics.prefill_duration += iree_time_now() - start;
    return iree_ok_status();
  }
  memset(row->transfer.tokens, 0, sizeof(row->transfer.tokens));
  memcpy(row->transfer.tokens, token_ids, count * sizeof(int32_t));
  row->transfer.control[0] = (int32_t)count;
  row->transfer.control[1] = (int32_t)row->position;
  row->transfer.control[2] = model->eos_token;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.control,
                  .target_buffer = row->buffers[QWEN_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[QWEN_TOKENS],
                  .length = sizeof(row->transfer.tokens)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = qwen_step(row, 1);
  }
  if (iree_status_is_ok(status)) {
    row->position += count;
    row->has_prediction = true;
    row->metrics.prefill_tokens += count;
    ++row->metrics.prefill_steps;
    row->metrics.prefill_duration += iree_time_now() - start;
  }
  return status;
}

iree_status_t loom_serve_qwen_row_decode(loom_serve_qwen_row_t* row) {
  if (!row->has_prediction) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode requires a selected prediction");
  }
  if (row->position == row->model->context_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen context is full");
  }
  const iree_time_t start = iree_time_now();
  if (row->model->mtp.first_stage || row->model->cache.capacity) {
    const int32_t token = row->transfer.tokens[0];
    const loom_serve_qwen_span_t span = {
        (iree_host_size_t)(row - row->model->rows),
        1,
        &token,
        LOOM_SERVE_QWEN_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(loom_serve_qwen_model_epoch(row->model, 0, 1, &span));
    ++row->metrics.decode_steps;
    row->metrics.decode_duration += iree_time_now() - start;
    return iree_ok_status();
  }
  // The host row owns the input position and prediction regardless of which
  // schedule produced it. The isolated decode command consumes that snapshot.
  row->transfer.control[0] = (int32_t)row->position;
  row->transfer.control[1] = row->transfer.progress[0];
  row->transfer.control[2] = row->model->eos_token;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.control,
                  .target_buffer = row->buffers[QWEN_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[QWEN_TOKENS],
                  .length = sizeof(int32_t)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = qwen_step(row, 0);
  }
  if (iree_status_is_ok(status)) {
    ++row->position;
    ++row->metrics.decode_steps;
    row->metrics.decode_duration += iree_time_now() - start;
  }
  return status;
}

int32_t loom_serve_qwen_row_token(const loom_serve_qwen_row_t* row) {
  return row->transfer.tokens[0];
}

bool loom_serve_qwen_row_is_eos(const loom_serve_qwen_row_t* row) {
  return row->transfer.progress[7] != 0;
}

iree_host_size_t loom_serve_qwen_row_position(
    const loom_serve_qwen_row_t* row) {
  return row->position;
}

loom_serve_qwen_metrics_t loom_serve_qwen_row_metrics(
    const loom_serve_qwen_row_t* row) {
  return row->metrics;
}
