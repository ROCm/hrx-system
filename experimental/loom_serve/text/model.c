// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared source-defined text model residency and retained row state.

#include "experimental/loom_serve/text/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/runtime/command.h"
#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/jit.h"
#include "experimental/loom_serve/runtime/json.h"
#include "experimental/loom_serve/runtime/module.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/program.h"
#include "experimental/loom_serve/runtime/weights.h"
#include "experimental/loom_serve/storage/block_pool.h"
#include "experimental/loom_serve/storage/relocation.h"
#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

enum {
  TEXT_RESIDUAL = 0,
  TEXT_CONTROL = 1,
  TEXT_RECURRENT = 2,
  TEXT_ATTENTION = 3,
  TEXT_TOKENS = 4,
  TEXT_PROGRESS = 5,
  TEXT_WORKSPACE = 6,
  TEXT_BINDING_COUNT = 7,
  TEXT_TOKEN_CAPACITY = 512,
  TEXT_ROW_CAPACITY = LOOM_SERVE_TEXT_ROW_CAPACITY,
  TEXT_STORAGE_ALLOCATIONS = 0,
  TEXT_STORAGE_VIEWS = 1,
  TEXT_STORAGE_TARGET_ORIGINS = 2,
  TEXT_STORAGE_DRAFT_ORIGINS = 3,
  TEXT_STORAGE_GEOMETRY = 4,
  TEXT_STORAGE_RESULT_COUNT = 5,
  TEXT_ALLOCATION_COUNT = 11,
  TEXT_HOST_PLAN = 0,
  TEXT_HOST_INPUT = 1,
  TEXT_HOST_METADATA = 2,
  TEXT_HOST_TOKENS = 3,
  TEXT_HOST_OUTPUTS = 4,
  TEXT_HOST_RESULTS = 5,
  TEXT_HOST_NEXT_RESULTS = 6,
  TEXT_HOST_PROGRESS = 7,
  TEXT_HOST_BUFFER_COUNT = 8,
};

typedef struct text_stage_t {
  // Owned JIT command image and loaded executable entries.
  loom_serve_jit_stage_t* compiled;
  // Parsed view borrowing compiled.
  loom_cmd_program_t program;
  // Reusable command retaining its executable and fixed weight resources.
  iree_hal_command_buffer_t* command;
  // Number of owned fixed-root slots, including partial initialization.
  uint32_t fixed_buffer_count;
  // Owned views of immutable parameter groups, in reflected fixed-root order.
  iree_hal_buffer_t** fixed_buffers;
} text_stage_t;

typedef struct text_cache_region_t {
  // Source allocation slot containing these repeated cache planes.
  iree_host_size_t allocation;
  // Source-owned disjoint cache planes inside that allocation.
  loom_serve_block_region_t blocks;
} text_cache_region_t;

struct loom_serve_text_model_t {
  // Host allocation policy used for all owned resources.
  iree_allocator_t allocator;
  // Borrowed shared owner outliving every model resource and host I/O payload.
  loom_serve_device_t* device_owner;
  // GPU device and allocation domain borrowed from device_owner.
  iree_hal_device_t* device;
  // Semaphore namespace borrowed from device_owner.
  iree_hal_device_group_t* group;
  // Borrowed exact dispatch queue, retained by execution.
  iree_hal_queue_t* dispatch;
  // Borrowed exact transfer queue, retained by execution.
  iree_hal_queue_t* transfer;
  // Borrowed stage/input timelines with independently retired feedback.
  loom_serve_execution_t* execution;
  // Cold recording policy for every reusable stage in this residency.
  iree_hal_command_buffer_mode_t command_mode;
  // Shared source index and compiler for this model residency.
  loom_serve_jit_t* jit;
  // Retained checkpoint/preparation plan owning stable parameter reservations.
  loom_serve_weights_t* weights;
  // Source declarations retained independently of the temporary bootstrap VM.
  loom_serve_preparation_t* preparation;
  // Leading stages with identical shared parameter placement.
  iree_host_size_t shared_stage_count;
  // Source-declared command programs, owned in declaration order.
  text_stage_t* stages;
  // Number of allocated stages, including partially initialized cold entries.
  iree_host_size_t stage_count;
  // Number of target packed shapes, independent of auxiliary MTP stages.
  iree_host_size_t shape_count;
  // Immutable prepared shape capacities, owned in epoch option order.
  loom_serve_packing_shape_t* shapes;
  // Context specialization shared by every compiled stage.
  iree_host_size_t context_capacity;
  // Maximum active input chunk accepted by the selected row-prefill path.
  iree_host_size_t prefill_capacity;
  // Number of preallocated rows.
  iree_host_size_t row_count;
  // Fixed row records, owned by this model.
  loom_serve_text_row_t* rows;
  // Private recurrent/control spans followed by pooled KV, or dense row KV.
  iree_hal_buffer_t* row_arena;
  // Cold source results survive their VM and all accepted initialization work.
  struct {
    // Owned source descriptors and upload payloads, released after retirement.
    iree_vm_buffer_t* buffers[TEXT_STORAGE_RESULT_COUNT];
    // Borrowed immutable mappings of those source-owned buffers.
    iree_const_byte_span_t bytes[TEXT_STORAGE_RESULT_COUNT];
  } initialization;
  // Private-page ownership shared by target and draft cache planes.
  struct {
    // Addressable token capacity shared by rows; zero selects dense comparison.
    iree_host_size_t capacity;
    // Positions in one page, declared by the model source.
    iree_host_size_t block_size;
    // Byte origin of the row-major page maps in each device origin table.
    iree_device_size_t map_origin;
    // Maximum logical block count per row, independent of physical capacity.
    iree_host_size_t blocks_per_row;
    // Fixed free-ID metadata; owned IDs are returned only after retirement.
    loom_serve_block_pool_t pool;
    // Fixed row-major host maps, retained through queued uploads.
    uint32_t* maps;
    // Cold relocation plan indexed by physical ID, reused at maintenance cuts.
    uint32_t* destinations;
  } cache;
  // Elastic physical backing, separate from logical row/block ownership.
  struct {
    // Physical allocation domain borrowed from the shared device owner.
    loom_serve_memory_pool_t* pool;
    // Mutable state accounting, excluding parameters sharing the same pool.
    loom_serve_memory_statistics_t statistics;
    // Optional virtual reservations in source allocation order.
    loom_serve_virtual_buffer_t* buffers[TEXT_ALLOCATION_COUNT];
    // Always-live initialized prefixes outside row-owned private storage.
    iree_device_size_t private_lengths[TEXT_ALLOCATION_COUNT];
    // Owned source-defined cache plane groups.
    text_cache_region_t* regions;
    // Number of validated cache plane groups.
    iree_host_size_t region_count;
  } memory;
  // Model-wide residual storage, serialized by the execution timeline.
  iree_hal_buffer_t* residual;
  // Model-wide packed transient storage for the larger stage.
  iree_hal_buffer_t* workspace;
  // Environment owning canonical VM reference types.
  iree_vm_environment_t* environment;
  // Canonical byte-buffer reference types borrowed from the environment.
  iree_vm_ref_types_t vm_types;
  // Source-owned stage layout interpreted only by the warm source program.
  iree_vm_buffer_t* control_state;
  // HAL reference type handles borrowed from environment.
  iree_hal_module_types_t types;
  // Native stage submission module retained by program.
  iree_vm_module_t* native_module;
  // Shared source-JIT model program; request state is not VM process state.
  loom_serve_program_t* program;
  // Cold-resolved model step function.
  iree_vm_function_t step;
  // Source-owned packed target, feedback and MTP submission sequence.
  iree_vm_function_t epoch_step;
  // Source-owned conversion of semantic spans to opaque device uploads.
  iree_vm_function_t encode_epoch;
  // Source-owned conversion of retired feedback to semantic row progress.
  iree_vm_function_t publish_epoch;
  // Chat formatting functions borrowing the same model-wide VM process.
  loom_serve_text_chat_policy_t chat_policy;
  // Cold retained packet and feedback storage, reused by the single owner.
  struct {
    // Owned VM buffers retained through all accepted work and feedback.
    iree_vm_buffer_t* buffers[TEXT_HOST_BUFFER_COUNT];
    // Stable writable mappings, valid for the corresponding buffer lifetime.
    iree_byte_span_t bytes[TEXT_HOST_BUFFER_COUNT];
  } host;
  // Reusable packed-epoch bindings and transfer payloads. Device buffers at
  // slots 1, 2, 4 and 5 are owned; other slots borrow model-wide storage.
  struct {
    // Residual, metadata, origins, state arena, input IDs, output IDs,
    // workspace.
    iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  } epoch;
  // Optional draft residency and private proposal work. Warm stages consume
  // committed target residuals before the next target invocation can reuse
  // them.
  struct {
    // Whether the requested residency includes speculative state.
    bool enabled;
    // Normalized committed target hidden, indexed by retained row.
    iree_hal_buffer_t* carry;
    // Source-declared byte extent of one row's retained hidden state.
    iree_device_size_t carry_stride;
    // Device-produced accepted span metadata consumed by MTP catch-up.
    iree_hal_buffer_t* committed;
    // Device-produced feedback, whose record layout belongs to source.
    iree_hal_buffer_t* results;
    // Owned view of the second result bank and its original-span tags.
    iree_hal_buffer_t* next_results;
    // Source-declared split between first and second feedback banks in bytes.
    iree_device_size_t feedback_split;
    // Private one-layer attention cache shared by proposal and catch-up.
    iree_hal_buffer_t* cache;
    // Immutable resident-row origins consumed by proposal and catch-up.
    iree_hal_buffer_t* row_table;
  } mtp;
  // Tokenizer owning its parsed vocabulary and encoding data.
  iree_tokenizer_t* tokenizer;
  // Token identifying the end of a generated assistant turn.
  int32_t eos_token;
};

struct loom_serve_text_row_t {
  // Borrowed owning model; all row operations use its single invocation.
  loom_serve_text_model_t* model;
  // Private spans in the row arena; residual/workspace entries borrow model.
  iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  // Number of input tokens actually consumed into KV/recurrent state.
  iree_host_size_t position;
  // Owned physical pages, including any in-flight speculative suffix.
  uint32_t block_count;
  // Whether the latest completed work selected a token at the current position.
  bool has_prediction;
  // Completed stage counts and end-to-end host durations.
  loom_serve_text_metrics_t metrics;
  // Host payloads remain alive until the model drains accepted transfers.
  struct {
    // Active prefill count, absolute base, and EOS ID.
    int32_t control[3];
    // Fixed-capacity padded upload and current-token readback at slot zero.
    int32_t tokens[TEXT_TOKEN_CAPACITY];
    // Device-generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } transfer;
};

static iree_string_view_t text_file_text(iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t text_check_layouts(loom_serve_text_model_t* runner) {
  const loom_cmd_program_t* prefill = &runner->stages[0].program;
  const loom_serve_preparation_stage_t* first =
      loom_serve_preparation_stage(runner->preparation, 0);
  if (!first->parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source must declare a checkpoint domain");
  }
  const loom_serve_preparation_parameter_t* checkpoint = &first->parameters[0];
  for (iree_host_size_t i = 0; i < runner->stage_count; ++i) {
    const loom_cmd_program_t* program = &runner->stages[i].program;
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(runner->preparation, i);
    const uint32_t binding_count =
        program->requirements.rebindable_binding_count;
    const uint32_t roots = program->requirements.fixed_buffer_count;
    if (program->parameter_roots.count != roots ||
        declaration->parameter_count != roots || binding_count < 1 ||
        binding_count > 8 ||
        (program->requirements.transient.binding_index != UINT32_MAX &&
         program->requirements.transient.binding_index != binding_count - 1) ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the token runner bindings");
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_serve_preparation_parameter_t* parameter =
          &declaration->parameters[root];
      if (parameter->binding != root ||
          loom_cmd_program_parameter_root_at(program, root)
                  .fixed_buffer_index != root ||
          !iree_string_view_equal(parameter->path, checkpoint->path) ||
          !iree_string_view_equal(parameter->policy, checkpoint->policy)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stages must declare the same checkpoint domain");
      }
    }
    if (i >= runner->shared_stage_count) {
      continue;
    }
    if (roots != 1 || binding_count != TEXT_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared stages must implement the target bindings");
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

static iree_status_t text_allocate_buffer(loom_serve_text_model_t* runner,
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
static iree_status_t text_prepare_weights(loom_serve_text_model_t* model,
                                          iree_string_view_t source_directory) {
  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    root_count += model->stages[i].program.parameter_roots.count;
  }
  loom_serve_weight_root_t* roots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, root_count, sizeof(*roots), (void**)&roots));
  iree_host_size_t root_index = 0;
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    const loom_cmd_program_t* program = &model->stages[i].program;
    for (uint32_t r = 0; r < program->parameter_roots.count; ++r) {
      const loom_cmd_program_parameter_root_t root =
          loom_cmd_program_parameter_root_at(program, r);
      roots[root_index++] = (loom_serve_weight_root_t){
          .program = program,
          .root = root,
          .buffer = &model->stages[i].fixed_buffers[root.fixed_buffer_index],
      };
    }
  }
  char* policy_path = NULL;
  const loom_serve_preparation_parameter_t* checkpoint =
      &loom_serve_preparation_stage(model->preparation, 0)->parameters[0];
  iree_status_t status = iree_file_path_join(
      source_directory, checkpoint->policy, model->allocator, &policy_path);
  if (iree_status_is_ok(status)) {
    status = loom_serve_weights_create(
        model->device, model->transfer, model->dispatch, model->memory.pool,
        model->jit, model->command_mode, model->shared_stage_count, root_count,
        roots, checkpoint->path, iree_make_cstring_view(policy_path),
        &model->weights, model->allocator);
  }
  iree_allocator_free(model->allocator, policy_path);
  iree_allocator_free(model->allocator, roots);
  return status;
}

static iree_status_t text_create_program(loom_serve_text_model_t* runner,
                                         iree_string_view_t source_directory) {
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(runner->environment, &table));
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &runner->types));
  const iree_host_size_t lengths[TEXT_HOST_BUFFER_COUNT] = {
      2 * TEXT_ROW_CAPACITY * 8 * sizeof(int32_t),
      2 * TEXT_TOKEN_CAPACITY * sizeof(int32_t),
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->epoch.buffers[1])
          : 0,
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->epoch.buffers[4])
          : 0,
      runner->shape_count
          ? iree_hal_buffer_byte_length(runner->epoch.buffers[5])
          : 0,
      runner->mtp.enabled ? runner->mtp.feedback_split : 0,
      runner->mtp.enabled
          ? iree_hal_buffer_byte_length(runner->mtp.next_results)
          : 0,
      TEXT_ROW_CAPACITY * 12 * sizeof(int32_t),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < TEXT_HOST_BUFFER_COUNT && iree_status_is_ok(status); ++i) {
    status = iree_vm_buffer_create(lengths[i], iree_alignof(int64_t),
                                   runner->allocator, &runner->host.buffers[i]);
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_write(runner->host.buffers[i], 0, lengths[i],
                                        &runner->host.bytes[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  loom_serve_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(runner->allocator, runner->stage_count,
                                  sizeof(*stages), (void**)&stages));
  for (iree_host_size_t i = 0; i < runner->stage_count; ++i) {
    stages[i] = (loom_serve_stage_t){
        runner->stages[i].command,
        runner->stages[i].program.requirements.rebindable_binding_count};
  }
  const iree_byte_span_t feedback[] = {
      runner->host.bytes[TEXT_HOST_OUTPUTS],
      runner->host.bytes[TEXT_HOST_RESULTS],
      runner->host.bytes[TEXT_HOST_NEXT_RESULTS],
  };
  const loom_serve_module_options_t options = {
      .binding_capacity = 8,
      .stages = {runner->stage_count, stages},
      .feedback = {IREE_ARRAYSIZE(feedback), feedback},
  };
  status = loom_serve_module_create(&runner->types, runner->execution, options,
                                    runner->allocator, &runner->native_module);
  iree_allocator_free(runner->allocator, stages);
  IREE_RETURN_IF_ERROR(status);
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      source_directory, IREE_SV("control.loom"), runner->allocator, &path));
  iree_vm_module_t* libraries[] = {runner->native_module, NULL, NULL, NULL};
  const iree_string_view_t roots[] = {
      IREE_SVL("step"),         IREE_SVL("epoch"),
      IREE_SVL("encode_epoch"), IREE_SVL("publish_epoch"),
      IREE_SVL("render_tool"),  IREE_SVL("prepare_input"),
      IREE_SVL("text_end"),     IREE_SVL("complete_text"),
      IREE_SVL("model_name"),   IREE_SVL("chat_begin"),
      IREE_SVL("chat_message"), IREE_SVL("chat_end"),
      IREE_SVL("parse_tools")};
  status = loom_serve_input_module_create(
      runner->environment, runner->tokenizer, &libraries[1], runner->allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(runner->environment, &libraries[2],
                                           runner->allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(
        runner->environment, &libraries[3], runner->allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_create(
        runner->environment, iree_make_cstring_view(path),
        IREE_ARRAYSIZE(roots), roots, iree_vm_module_span_from_array(libraries),
        runner->allocator, &runner->program);
  }
  iree_vm_module_release(libraries[3]);
  iree_vm_module_release(libraries[2]);
  iree_vm_module_release(libraries[1]);
  iree_allocator_free(runner->allocator, path);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_policy_initialize(
      runner->environment, runner->program, &runner->chat_policy));
  iree_vm_process_t* process = loom_serve_program_process(runner->program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("step"), &runner->step));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("encode_epoch"),
      &runner->encode_epoch));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("publish_epoch"),
      &runner->publish_epoch));
  return iree_vm_process_lookup_function(process, IREE_SV("model"),
                                         IREE_SV("epoch"), &runner->epoch_step);
}

static iree_status_t text_load_tokenizer(loom_serve_text_model_t* model,
                                         iree_string_view_t path,
                                         iree_string_view_t terminal_marker) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 model->allocator, &contents));
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      text_file_text(contents), model->allocator, &model->tokenizer);
  iree_io_file_contents_free(contents);
  IREE_RETURN_IF_ERROR(status);
  model->eos_token = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(model->tokenizer), terminal_marker);
  if (model->eos_token < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no source-declared terminal marker");
  }
  return iree_ok_status();
}

static void text_release_initialization(loom_serve_text_model_t* model) {
  for (iree_host_size_t i = 0; i < TEXT_STORAGE_RESULT_COUNT; ++i) {
    iree_vm_buffer_release(model->initialization.buffers[i]);
  }
  memset(&model->initialization, 0, sizeof(model->initialization));
}

// Source records cross the external model boundary once. Native consumers use
// the established sizes directly; HAL owns allocation and subspan validation.
static iree_status_t text_prepare_storage(loom_serve_text_model_t* model,
                                          iree_vm_variant_t* results) {
  const iree_host_size_t lengths[] = {
      TEXT_ALLOCATION_COUNT * 3 * sizeof(int64_t),
      model->row_count * 5 * 2 * sizeof(int64_t),
      TEXT_ROW_CAPACITY * 2 * sizeof(int64_t),
      TEXT_ROW_CAPACITY * 2 * sizeof(int64_t),
      4 * sizeof(int64_t),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < TEXT_STORAGE_RESULT_COUNT && iree_status_is_ok(status); ++i) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->vm_types, &results[i], &model->initialization.buffers[i]);
    const iree_host_size_t length =
        model->initialization.buffers[i]
            ? iree_vm_buffer_length(model->initialization.buffers[i])
            : 0;
    const bool geometry = i == TEXT_STORAGE_GEOMETRY;
    if (iree_status_is_ok(status) &&
        (geometry ? (length < lengths[i] || (length - lengths[i]) % 40)
                  : length != lengths[i])) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "source storage result %zu has invalid size", i);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_read(model->initialization.buffers[i], 0,
                                       length, &model->initialization.bytes[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const uint8_t* geometry =
      model->initialization.bytes[TEXT_STORAGE_GEOMETRY].data;
  const uint64_t block_size = iree_unaligned_load_le_u64(geometry);
  const uint64_t map_origin = iree_unaligned_load_le_u64(geometry + 8);
  const uint64_t carry_stride = iree_unaligned_load_le_u64(geometry + 16);
  const uint64_t feedback_split = iree_unaligned_load_le_u64(geometry + 24);
  if (!block_size || block_size > UINT32_MAX ||
      model->cache.capacity % block_size || map_origin > INT64_MAX ||
      !carry_stride || carry_stride > INT64_MAX / model->row_count ||
      !feedback_split || feedback_split > INT64_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source cache geometry cannot represent this pool");
  }
  model->cache.block_size = (iree_host_size_t)block_size;
  model->cache.map_origin = map_origin;
  model->cache.blocks_per_row =
      (model->context_capacity + block_size - 1) / block_size;
  model->mtp.carry_stride = carry_stride;
  model->mtp.feedback_split = feedback_split;
  model->memory.region_count =
      (model->initialization.bytes[TEXT_STORAGE_GEOMETRY].data_length - 32) /
      40;
  if ((model->memory.region_count != 0) != (model->cache.capacity != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pooled state requires source cache regions");
  }
  if (!model->memory.region_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, model->memory.region_count,
      sizeof(*model->memory.regions), (void**)&model->memory.regions));
  for (iree_host_size_t i = 0; i < model->memory.region_count; ++i) {
    const uint8_t* record = geometry + 32 + i * 40;
    const uint64_t allocation = iree_unaligned_load_le_u64(record);
    const uint64_t origin = iree_unaligned_load_le_u64(record + 8);
    const uint64_t count = iree_unaligned_load_le_u64(record + 16);
    const uint64_t stride = iree_unaligned_load_le_u64(record + 24);
    const uint64_t block_bytes = iree_unaligned_load_le_u64(record + 32);
    if (allocation >= TEXT_ALLOCATION_COUNT || !count || !stride ||
        !block_bytes) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid cache region %zu", i);
    }
    const uint64_t length = iree_unaligned_load_le_u64(
        model->initialization.bytes[TEXT_STORAGE_ALLOCATIONS].data +
        allocation * 24);
    const uint64_t private_length = iree_unaligned_load_le_u64(
        model->initialization.bytes[TEXT_STORAGE_ALLOCATIONS].data +
        allocation * 24 + 16);
    const uint64_t blocks = model->cache.capacity / block_size;
    if (origin < private_length || origin > length ||
        count - 1 > (length - origin) / stride ||
        block_bytes > (length - origin - (count - 1) * stride) / blocks ||
        block_bytes > stride / blocks) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cache region %zu exceeds its allocation", i);
    }
    model->memory.regions[i] =
        (text_cache_region_t){.allocation = allocation,
                              .blocks = {origin, count, stride, block_bytes}};
  }
  return iree_ok_status();
}

// Bootstrap returns ordinary source-owned references. Its temporary program
// is gone before device setup; the shared environment outlives warm control.
static iree_status_t text_prepare(loom_serve_text_model_t* model,
                                  const loom_serve_text_options_t* options) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(allocator, &model->environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    model->environment, IREE_SV("vm")),
                                &model->vm_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("prepare.loom"), allocator, &path));
  iree_vm_buffer_t* shapes = NULL;
  iree_status_t status =
      iree_vm_buffer_create(options->epoch_count * 2 * sizeof(int64_t),
                            iree_alignof(int64_t), allocator, &shapes);
  iree_byte_span_t shape_bytes = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(shapes, 0, iree_vm_buffer_length(shapes),
                                      &shape_bytes);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
      iree_unaligned_store_le_u64(shape_bytes.data + i * 16,
                                  options->epoch_shapes[i].token_capacity);
      iree_unaligned_store_le_u64(shape_bytes.data + i * 16 + 8,
                                  options->epoch_shapes[i].span_capacity);
    }
  }
  iree_vm_buffer_t* weights = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)options->weights_path.data,
                            options->weights_path.size),
        iree_vm_buffer_release_callback_null(), allocator, &weights);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64((int64_t)options->prefill_capacity),
      iree_vm_variant_from_i64((int64_t)options->context_capacity),
      iree_vm_variant_from_i64((int64_t)options->pool_capacity),
      iree_vm_variant_from_i64((int64_t)options->row_count),
      iree_vm_variant_from_i32(options->enable_mtp),
      iree_vm_buffer_variant_from_ptr_move(&model->vm_types, &shapes),
      iree_vm_variant_from_i64((int64_t)options->epoch_count),
      iree_vm_buffer_variant_from_ptr_move(&model->vm_types, &weights),
  };
  iree_vm_variant_t results[4 + TEXT_STORAGE_RESULT_COUNT] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_create(
        model->environment, iree_make_cstring_view(path), IREE_SV("prepare"),
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results), iree_vm_module_span_empty(),
        &model->preparation, allocator);
  }
  int64_t prefill_capacity = 0;
  int32_t shared_stage_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &prefill_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[1], &shared_stage_count);
  }
  if (iree_status_is_ok(status) &&
      (prefill_capacity < 1 || prefill_capacity > TEXT_TOKEN_CAPACITY ||
       shared_stage_count < 1 ||
       (iree_host_size_t)shared_stage_count >
           loom_serve_preparation_stage_count(model->preparation))) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "source bootstrap exceeds token runner capacities");
  }
  if (iree_status_is_ok(status)) {
    model->prefill_capacity = (iree_host_size_t)prefill_capacity;
    model->shared_stage_count = (iree_host_size_t)shared_stage_count;
    status = iree_vm_buffer_ptr_from_variant_move(&model->vm_types, &results[2],
                                                  &model->control_state);
  }
  if (iree_status_is_ok(status) && !model->control_state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source control state must not be null");
  }
  iree_vm_buffer_t* terminal = NULL;
  iree_const_byte_span_t terminal_bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->vm_types,
                                                      results[3], &terminal);
  }
  if (iree_status_is_ok(status) && !terminal) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source terminal marker must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        terminal, 0, iree_vm_buffer_length(terminal), &terminal_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = text_prepare_storage(model, results + 4);
  }
  if (iree_status_is_ok(status)) {
    status = text_load_tokenizer(
        model, options->tokenizer_path,
        iree_make_string_view((const char*)terminal_bytes.data,
                              terminal_bytes.data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_allocator_free(allocator, path);
  return status;
}

static iree_status_t text_allocate_state(loom_serve_text_model_t* model) {
  if (model->cache.capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_block_pool_initialize(
        (uint32_t)(model->cache.capacity / model->cache.block_size),
        model->allocator, &model->cache.pool));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->allocator, model->row_count * model->cache.blocks_per_row,
        sizeof(*model->cache.maps), (void**)&model->cache.maps));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->allocator, model->cache.pool.capacity,
        sizeof(*model->cache.destinations),
        (void**)&model->cache.destinations));
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
  IREE_RETURN_IF_ERROR(text_allocate_buffer(
      model, workspace_length, workspace_alignment, &model->workspace));
  // Binding roles are the private adapter contract. Sizes, views, initial
  // contents and zero extents are produced by the model's source bootstrap.
  iree_hal_buffer_t** allocations[TEXT_ALLOCATION_COUNT] = {
      &model->residual,         &model->row_arena,
      &model->epoch.buffers[1], &model->epoch.buffers[2],
      &model->epoch.buffers[4], &model->epoch.buffers[5],
      &model->mtp.carry,        &model->mtp.committed,
      &model->mtp.results,      &model->mtp.cache,
      &model->mtp.row_table,
  };
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t transfers[TEXT_ALLOCATION_COUNT + 3] = {0};
  iree_host_size_t transfer_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < TEXT_ALLOCATION_COUNT && iree_status_is_ok(status); ++i) {
    const uint8_t* record =
        model->initialization.bytes[TEXT_STORAGE_ALLOCATIONS].data + i * 24;
    const uint64_t length = iree_unaligned_load_le_u64(record);
    const uint64_t alignment = iree_unaligned_load_le_u64(record + 8);
    const uint64_t clear_length = iree_unaligned_load_le_u64(record + 16);
    const bool required =
        i < 2 || (i < 6 ? model->shape_count != 0 : model->mtp.enabled);
    if ((length != 0) != required || clear_length > length ||
        length > INT64_MAX || !alignment ||
        !iree_is_power_of_two_uint64(alignment)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid source allocation record %zu", i);
    } else if (length) {
      bool elastic = false;
      for (iree_host_size_t j = 0; j < model->memory.region_count; ++j) {
        elastic |= model->memory.regions[j].allocation == i;
      }
      if (model->memory.pool && model->cache.capacity && elastic) {
        status = loom_serve_virtual_buffer_create(
            model->memory.pool, length, alignment, &model->memory.statistics,
            &model->memory.buffers[i]);
        if (iree_status_is_ok(status)) {
          *allocations[i] =
              loom_serve_virtual_buffer_handle(model->memory.buffers[i]);
          iree_hal_buffer_retain(*allocations[i]);
          if (clear_length && i != 1) {
            model->memory.private_lengths[i] = clear_length;
            status = loom_serve_virtual_buffer_commit(model->memory.buffers[i],
                                                      0, clear_length);
          }
        }
      } else {
        status = text_allocate_buffer(model, length, alignment, allocations[i]);
      }
    }
    if (iree_status_is_ok(status) && clear_length &&
        !(i == 1 && model->memory.buffers[i])) {
      transfers[transfer_count++] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {.target_buffer = *allocations[i],
                   .length = clear_length,
                   .pattern = &zero,
                   .pattern_length = sizeof(zero)},
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(model->allocator, model->row_count,
                                    sizeof(*model->rows), (void**)&model->rows);
  }
  for (iree_host_size_t i = 0;
       i < model->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_text_row_t* row = &model->rows[i];
    row->model = model;
    row->buffers[TEXT_RESIDUAL] = model->residual;
    row->buffers[TEXT_WORKSPACE] = model->workspace;
    for (iree_host_size_t binding = TEXT_CONTROL;
         binding <= TEXT_PROGRESS && iree_status_is_ok(status); ++binding) {
      const uint8_t* view =
          model->initialization.bytes[TEXT_STORAGE_VIEWS].data +
          (i * 5 + binding - TEXT_CONTROL) * 16;
      const uint64_t offset = iree_unaligned_load_le_u64(view);
      const uint64_t length = iree_unaligned_load_le_u64(view + 8);
      const bool required = binding != TEXT_ATTENTION || !model->cache.capacity;
      if ((length != 0) != required) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid source view for row %zu binding %zu",
                                  i, binding);
      } else if (length) {
        status =
            iree_hal_buffer_subspan(model->row_arena, offset, length,
                                    model->allocator, &row->buffers[binding]);
      }
    }
  }
  if (iree_status_is_ok(status) && model->shape_count) {
    model->epoch.buffers[0] = model->residual;
    model->epoch.buffers[3] = model->row_arena;
    model->epoch.buffers[6] = model->workspace;
    const iree_const_byte_span_t origins =
        model->initialization.bytes[TEXT_STORAGE_TARGET_ORIGINS];
    transfers[transfer_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = origins.data,
                   .target_buffer = model->epoch.buffers[2],
                   .length = origins.data_length},
    };
  }
  if (iree_status_is_ok(status) && model->mtp.enabled) {
    status = iree_hal_buffer_subspan(
        model->mtp.results, model->mtp.feedback_split, IREE_HAL_WHOLE_BUFFER,
        model->allocator, &model->mtp.next_results);
  }
  if (iree_status_is_ok(status) && model->mtp.enabled) {
    const iree_const_byte_span_t origins =
        model->initialization.bytes[TEXT_STORAGE_DRAFT_ORIGINS];
    transfers[transfer_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = origins.data,
                   .target_buffer = model->mtp.row_table,
                   .length = origins.data_length},
    };
  }
  if (iree_status_is_ok(status)) {
    transfers[transfer_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = model->workspace,
                 .length = workspace_length,
                 .pattern = &zero,
                 .pattern_length = sizeof(zero)},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(model->execution, transfer_count,
                                           transfers, &completion);
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "Residency: %zu rows, %.3f GiB retained arena, %.3f GiB shared "
            "workspace, %zu-token context, %zu-token prefill capacity.\n",
            model->row_count,
            iree_hal_buffer_byte_length(model->row_arena) / 1073741824.0,
            workspace_length / 1073741824.0, model->context_capacity,
            model->prefill_capacity);
  }
  return status;
}

static iree_status_t text_initialize(loom_serve_text_model_t* model,
                                     const loom_serve_text_options_t* options) {
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  model->command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  model->shape_count = options->epoch_count;
  model->context_capacity = options->context_capacity;
  model->cache.capacity = options->pool_capacity;
  model->mtp.enabled = options->enable_mtp;
  if (model->shape_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        model->allocator,
        iree_make_const_byte_span(options->epoch_shapes,
                                  model->shape_count * sizeof(*model->shapes)),
        (void**)&model->shapes));
  }
  IREE_RETURN_IF_ERROR(text_prepare(model, options));
  const iree_host_size_t stage_count =
      loom_serve_preparation_stage_count(model->preparation);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, stage_count, sizeof(*model->stages),
      (void**)&model->stages));
  model->stage_count = stage_count;
  model->device = loom_serve_device_handle(model->device_owner);
  model->group = loom_serve_device_group(model->device_owner);
  model->dispatch = loom_serve_device_dispatch_queue(model->device_owner);
  model->transfer = loom_serve_device_transfer_queue(model->device_owner);
  model->execution = loom_serve_device_execution(model->device_owner);
  model->memory.pool = loom_serve_device_memory_pool(model->device_owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(
      model->device, model->dispatch, options->source_directory,
      &options->kernel_sanitizer, model->allocator, &model->jit));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    text_stage_t* stage = &model->stages[i];
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    status = loom_serve_jit_compile(model->jit, declaration->root,
                                    &declaration->config, &stage->compiled);
    if (iree_status_is_ok(status)) {
      stage->program = *loom_serve_jit_stage_program(stage->compiled);
    }
    if (iree_status_is_ok(status) &&
        stage->program.requirements.fixed_buffer_count) {
      status = iree_allocator_malloc_array(
          model->allocator, stage->program.requirements.fixed_buffer_count,
          sizeof(*stage->fixed_buffers), (void**)&stage->fixed_buffers);
      if (iree_status_is_ok(status)) {
        stage->fixed_buffer_count =
            stage->program.requirements.fixed_buffer_count;
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(text_check_layouts(model));
  IREE_RETURN_IF_ERROR(text_prepare_weights(model, options->source_directory));
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    text_stage_t* stage = &model->stages[i];
    status = loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(model->dispatch),
        model->command_mode, stage->fixed_buffers, &stage->command);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(text_allocate_state(model));
  IREE_RETURN_IF_ERROR(text_create_program(model, options->source_directory));
  text_release_initialization(model);
  return iree_ok_status();
}

iree_status_t loom_serve_text_model_create(
    loom_serve_device_t* device, const loom_serve_text_options_t* options,
    loom_serve_text_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  if (options->row_count < 1 || options->row_count > TEXT_ROW_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text row count must be in [1, 16]");
  }
  if (options->prefill_capacity < 1 ||
      options->prefill_capacity > TEXT_TOKEN_CAPACITY ||
      options->context_capacity < options->prefill_capacity ||
      options->context_capacity > 262144 || options->epoch_count > INT32_MAX ||
      options->epoch_count > SIZE_MAX / (2 * sizeof(int64_t)) ||
      (options->enable_mtp && !options->epoch_count) ||
      options->pool_capacity > 4194304 ||
      (options->pool_capacity && !options->epoch_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid model capacities; pooled KV and MTP require epoch shapes");
  }
  for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
    const loom_serve_packing_shape_t shape = options->epoch_shapes[i];
    if (shape.token_capacity < 1 ||
        shape.token_capacity > TEXT_TOKEN_CAPACITY ||
        shape.token_capacity > options->context_capacity ||
        shape.span_capacity < 1 || shape.span_capacity > TEXT_ROW_CAPACITY) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "epoch %zu has invalid token/span capacities", i);
    }
  }
  loom_serve_text_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  model->device_owner = device;
  model->row_count = options->row_count;
  iree_status_t status = text_initialize(model, options);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_text_model_destroy(model));
  }
  return status;
}

iree_status_t loom_serve_text_model_destroy(loom_serve_text_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = model->execution
                             ? loom_serve_execution_drain(model->execution)
                             : iree_ok_status();
  loom_serve_text_chat_policy_deinitialize(&model->chat_policy);
  loom_serve_program_destroy(model->program);
  iree_vm_module_release(model->native_module);
  text_release_initialization(model);
  iree_vm_buffer_release(model->control_state);
  for (iree_host_size_t i = 0; i < TEXT_HOST_BUFFER_COUNT; ++i) {
    iree_vm_buffer_release(model->host.buffers[i]);
  }
  iree_vm_environment_free(model->environment);
  loom_serve_preparation_destroy(model->preparation);
  if (model->rows) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        iree_hal_buffer_release(model->rows[i].buffers[binding]);
      }
    }
  }
  iree_allocator_free(model->allocator, model->rows);
  iree_allocator_free(model->allocator, model->cache.maps);
  iree_allocator_free(model->allocator, model->cache.destinations);
  loom_serve_block_pool_deinitialize(&model->cache.pool);
  iree_hal_buffer_release(model->epoch.buffers[1]);
  iree_hal_buffer_release(model->epoch.buffers[2]);
  iree_hal_buffer_release(model->epoch.buffers[4]);
  iree_hal_buffer_release(model->epoch.buffers[5]);
  iree_hal_buffer_release(model->mtp.carry);
  iree_hal_buffer_release(model->mtp.committed);
  iree_hal_buffer_release(model->mtp.results);
  iree_hal_buffer_release(model->mtp.next_results);
  iree_hal_buffer_release(model->mtp.cache);
  iree_hal_buffer_release(model->mtp.row_table);
  iree_hal_buffer_release(model->row_arena);
  iree_hal_buffer_release(model->residual);
  iree_hal_buffer_release(model->workspace);
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    iree_hal_command_buffer_release(model->stages[i].command);
    loom_serve_jit_stage_destroy(model->stages[i].compiled);
    for (iree_host_size_t j = 0; j < model->stages[i].fixed_buffer_count; ++j) {
      iree_hal_buffer_release(model->stages[i].fixed_buffers[j]);
    }
    iree_allocator_free(model->allocator, model->stages[i].fixed_buffers);
  }
  iree_allocator_free(model->allocator, model->shapes);
  iree_allocator_free(model->allocator, model->stages);
  loom_serve_jit_destroy(model->jit);
  iree_allocator_free(model->allocator, model->memory.regions);
  iree_status_t memory_status = loom_serve_weights_destroy(model->weights);
  for (iree_host_size_t i = 0; i < TEXT_ALLOCATION_COUNT; ++i) {
    memory_status = iree_status_join(
        memory_status,
        loom_serve_virtual_buffer_destroy(model->memory.buffers[i]));
  }
  // A failed platform unmap/free must not destroy the allocation domain still
  // owning those handles. The terminal error stops the process's serving run.
  if (!iree_status_is_ok(memory_status)) {
    return iree_status_join(status, memory_status);
  }
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model);
  return status;
}

loom_serve_text_row_t* loom_serve_text_model_row(loom_serve_text_model_t* model,
                                                 iree_host_size_t index) {
  return &model->rows[index];
}

loom_serve_memory_statistics_t loom_serve_text_model_memory_statistics(
    const loom_serve_text_model_t* model) {
  return model->memory.statistics;
}

loom_serve_memory_statistics_t loom_serve_text_model_weight_statistics(
    const loom_serve_text_model_t* model) {
  return loom_serve_weights_statistics(model->weights);
}

iree_status_t loom_serve_text_model_activate(loom_serve_text_model_t* model) {
  return loom_serve_weights_activate(model->weights);
}

iree_status_t loom_serve_text_model_deactivate(loom_serve_text_model_t* model) {
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  return loom_serve_weights_deactivate(model->weights);
}

iree_status_t loom_serve_text_model_trim(
    loom_serve_text_model_t* model, loom_serve_text_trim_result_t* out_result) {
  *out_result = (loom_serve_text_trim_result_t){0};
  if (!model->memory.pool) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  const uint64_t released_before =
      loom_serve_memory_pool_statistics(model->memory.pool).released_bytes;
  out_result->moved_blocks = loom_serve_block_pool_plan_compaction(
      &model->cache.pool, model->cache.destinations);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       out_result->moved_blocks && i < model->memory.region_count &&
       iree_status_is_ok(status);
       ++i) {
    const text_cache_region_t* region = &model->memory.regions[i];
    uint64_t copied_bytes = 0;
    status = loom_serve_block_region_relocate(
        model->execution, model->memory.buffers[region->allocation],
        &region->blocks, model->cache.pool.capacity, model->cache.destinations,
        &copied_bytes);
    out_result->copied_bytes += copied_bytes;
  }
  // Even a rejected later copy batch leaves earlier submissions owning their
  // sources/destinations. Observe retirement before publishing or returning.
  status =
      iree_status_join(status, loom_serve_execution_drain(model->execution));
  if (iree_status_is_ok(status) && out_result->moved_blocks) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      uint32_t* blocks = model->cache.maps + i * model->cache.blocks_per_row;
      for (uint32_t j = 0; j < model->rows[i].block_count; ++j) {
        blocks[j] = model->cache.destinations[blocks[j]];
      }
    }
    const iree_device_size_t length = model->row_count *
                                      model->cache.blocks_per_row *
                                      sizeof(*model->cache.maps);
    const iree_hal_transfer_operation_t uploads[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->cache.maps,
                    .target_buffer = model->epoch.buffers[2],
                    .target_offset = model->cache.map_origin,
                    .length = length}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->cache.maps,
                    .target_buffer = model->mtp.row_table,
                    .target_offset = model->cache.map_origin,
                    .length = length}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, model->mtp.enabled ? 2 : 1, uploads, &completion);
    status =
        iree_status_join(status, loom_serve_execution_drain(model->execution));
  }
  if (iree_status_is_ok(status)) {
    loom_serve_block_pool_commit_compaction(&model->cache.pool);
    for (iree_host_size_t i = 0; i < TEXT_ALLOCATION_COUNT; ++i) {
      if (!model->memory.buffers[i]) {
        continue;
      }
      loom_serve_virtual_buffer_begin_trim(model->memory.buffers[i]);
      if (model->memory.private_lengths[i]) {
        loom_serve_virtual_buffer_keep(model->memory.buffers[i], 0,
                                       model->memory.private_lengths[i]);
      }
    }
    for (iree_host_size_t i = 0;
         model->memory.buffers[1] && i < model->row_count; ++i) {
      if (!model->rows[i].position) {
        continue;
      }
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        iree_hal_buffer_t* view = model->rows[i].buffers[binding];
        if (view) {
          loom_serve_virtual_buffer_keep(model->memory.buffers[1],
                                         iree_hal_buffer_byte_offset(view),
                                         iree_hal_buffer_byte_length(view));
        }
      }
    }
    const uint32_t live =
        model->cache.pool.capacity - model->cache.pool.available;
    for (iree_host_size_t i = 0; live && i < model->memory.region_count; ++i) {
      const text_cache_region_t* region = &model->memory.regions[i];
      for (iree_host_size_t plane = 0; plane < region->blocks.count; ++plane) {
        loom_serve_virtual_buffer_keep(
            model->memory.buffers[region->allocation],
            region->blocks.origin + plane * region->blocks.stride,
            live * region->blocks.block_bytes);
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < TEXT_ALLOCATION_COUNT && iree_status_is_ok(status); ++i) {
    if (model->memory.buffers[i]) {
      status = loom_serve_virtual_buffer_trim(model->memory.buffers[i]);
    }
  }
  out_result->released_bytes =
      loom_serve_memory_pool_statistics(model->memory.pool).released_bytes -
      released_before;
  return status;
}

iree_tokenizer_t* loom_serve_text_model_tokenizer(
    loom_serve_text_model_t* model) {
  return model->tokenizer;
}

const loom_serve_text_chat_policy_t* loom_serve_text_model_chat_policy(
    const loom_serve_text_model_t* model) {
  return &model->chat_policy;
}

iree_host_size_t loom_serve_text_model_context_capacity(
    const loom_serve_text_model_t* model) {
  return model->context_capacity;
}

loom_serve_text_pool_usage_t loom_serve_text_model_pool_usage(
    const loom_serve_text_model_t* model) {
  return (loom_serve_text_pool_usage_t){
      .block_size = model->cache.block_size,
      .capacity = model->cache.capacity,
      .available = (iree_host_size_t)model->cache.pool.available *
                   model->cache.block_size,
  };
}

iree_host_size_t loom_serve_text_row_pool_usage(
    const loom_serve_text_row_t* row) {
  return (iree_host_size_t)row->block_count * row->model->cache.block_size;
}

iree_host_size_t loom_serve_text_model_prefill_capacity(
    const loom_serve_text_model_t* model) {
  return model->prefill_capacity;
}

iree_host_size_t loom_serve_text_model_shape_count(
    const loom_serve_text_model_t* model) {
  return model->shape_count;
}

const loom_serve_packing_shape_t* loom_serve_text_model_shapes(
    const loom_serve_text_model_t* model) {
  return model->shapes;
}

// Only a retired frontier may release pages. A rejected suffix within the
// retained last page stays private; subsequent appends overwrite it before use.
static void text_trim_blocks(loom_serve_text_row_t* row) {
  loom_serve_text_model_t* model = row->model;
  if (!model->cache.capacity) {
    return;
  }
  const uint32_t keep =
      (uint32_t)((row->position + model->cache.block_size - 1) /
                 model->cache.block_size);
  const iree_host_size_t row_index = (iree_host_size_t)(row - model->rows);
  loom_serve_block_pool_release(
      &model->cache.pool, row->block_count - keep,
      model->cache.maps + row_index * model->cache.blocks_per_row + keep);
  row->block_count = keep;
}

iree_status_t loom_serve_text_row_reset(loom_serve_text_row_t* row) {
  // An unused elastic row has no initialized device state to clear.
  if (row->model->memory.pool && !row->position) {
    return iree_ok_status();
  }
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t fills[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_RECURRENT],
                .length =
                    iree_hal_buffer_byte_length(row->buffers[TEXT_RECURRENT]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[TEXT_PROGRESS],
                .length = sizeof(row->transfer.progress),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(fills), fills, &completion);
  if (iree_status_is_ok(status) && row->model->mtp.enabled) {
    const iree_hal_transfer_operation_t clear_carry = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = row->model->mtp.carry,
                 .target_offset = (uint64_t)(row - row->model->rows) *
                                  row->model->mtp.carry_stride,
                 .length = row->model->mtp.carry_stride,
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
    text_trim_blocks(row);
    row->has_prediction = false;
    memset(&row->transfer, 0, sizeof(row->transfer));
    memset(&row->metrics, 0, sizeof(row->metrics));
  }
  return status;
}

static iree_status_t text_invoke(loom_serve_text_model_t* model,
                                 iree_vm_function_t function,
                                 iree_vm_variant_span_t arguments,
                                 iree_vm_variant_span_t results) {
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->program), function,
                     arguments, results);
  iree_vm_variant_span_reset(arguments);
  return status;
}

static iree_status_t text_step(loom_serve_text_row_t* row, int32_t initialize) {
  loom_serve_text_model_t* model = row->model;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_activate(model));
  iree_vm_variant_t arguments[TEXT_BINDING_COUNT + 2] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->vm_types,
                                               model->control_state),
      iree_vm_variant_from_i32(initialize),
  };
  for (iree_host_size_t i = 0; i < TEXT_BINDING_COUNT; ++i) {
    arguments[i + 2] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->types, row->buffers[i]);
  }
  iree_vm_variant_t results[1] = {0};
  iree_status_t status = text_invoke(model, model->step,
                                     iree_vm_variant_span_from_array(arguments),
                                     iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_RETURN_IF_ERROR(status);
  const iree_hal_transfer_operation_t downloads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[TEXT_TOKENS],
                    .target = row->transfer.tokens,
                    .length = sizeof(int32_t)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[TEXT_PROGRESS],
                    .target = row->transfer.progress,
                    .length = sizeof(row->transfer.progress)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_feedback(
      model->execution, IREE_ARRAYSIZE(downloads), downloads, &completion));
  return loom_serve_execution_feedback_wait(model->execution, completion);
}

// The whole epoch has passed its capacity check. Publish only newly assigned
// map entries on the existing ordered transfer path. The fixed host maps stay
// alive until completion, including partial submission failure and destruction.
static iree_status_t text_grow_blocks(loom_serve_text_model_t* model,
                                      iree_host_size_t span_count,
                                      const loom_serve_text_span_t* spans,
                                      const uint32_t* extents) {
  if (!model->cache.capacity) {
    return iree_ok_status();
  }
  iree_hal_transfer_operation_t uploads[2 * TEXT_ROW_CAPACITY] = {0};
  iree_host_size_t upload_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    loom_serve_text_row_t* row = &model->rows[span->row_index];
    const uint32_t needed =
        (uint32_t)((row->position + extents[i] + model->cache.block_size - 1) /
                   model->cache.block_size);
    const uint32_t count = needed - row->block_count;
    if (!count) {
      continue;
    }
    if (model->memory.buffers[1] && !row->position) {
      // Cold row activation initializes only that row's private views. The
      // logical row address stays fixed even when its physical slabs trim.
      iree_hal_transfer_operation_t fills[5] = {0};
      iree_host_size_t fill_count = 0;
      const uint32_t zero = 0;
      for (iree_host_size_t binding = TEXT_CONTROL; binding <= TEXT_PROGRESS;
           ++binding) {
        iree_hal_buffer_t* view = row->buffers[binding];
        if (!view) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_commit(
            model->memory.buffers[1], iree_hal_buffer_byte_offset(view),
            iree_hal_buffer_byte_length(view)));
        fills[fill_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
            .fill = {.target_buffer = view,
                     .length = iree_hal_buffer_byte_length(view),
                     .pattern = &zero,
                     .pattern_length = sizeof(zero)},
        };
      }
      uint64_t completion = 0;
      IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
          model->execution, fill_count, fills, &completion));
    }
    const iree_host_size_t map_index =
        span->row_index * model->cache.blocks_per_row + row->block_count;
    uint32_t* blocks = model->cache.maps + map_index;
    loom_serve_block_pool_acquire(&model->cache.pool, count, blocks);
    row->block_count = needed;
    for (iree_host_size_t j = 0;
         model->memory.pool && j < model->memory.region_count; ++j) {
      const text_cache_region_t* cache_region = &model->memory.regions[j];
      const loom_serve_block_region_t* region = &cache_region->blocks;
      for (iree_host_size_t plane = 0; plane < region->count; ++plane) {
        for (iree_host_size_t block = 0; block < count; ++block) {
          IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_commit(
              model->memory.buffers[cache_region->allocation],
              region->origin + plane * region->stride +
                  blocks[block] * region->block_bytes,
              region->block_bytes));
        }
      }
    }
    uploads[upload_count++] = (iree_hal_transfer_operation_t){
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = blocks,
                   .target_buffer = model->epoch.buffers[2],
                   .target_offset =
                       model->cache.map_origin + map_index * sizeof(*blocks),
                   .length = count * sizeof(*blocks)},
    };
    if (model->mtp.enabled) {
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

// The semantic host record is {length, position, row, input_begin, input_count,
// flags, output_credit, first_span}. Known IDs are copied once into a
// contiguous host stream. Source owns all device packet offsets and selection
// policy.
static iree_host_size_t text_prepare_plan(loom_serve_text_model_t* model,
                                          iree_host_size_t span_count,
                                          const loom_serve_text_span_t* spans,
                                          const uint32_t* output_limits,
                                          const uint32_t* first_indices,
                                          iree_host_size_t record_begin,
                                          iree_host_size_t input_begin) {
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    const iree_host_size_t input_count =
        span->token_ids && !(first_indices && generates)
            ? (generates ? 1 : span->token_count)
            : 0;
    const uint32_t fields[] = {
        (uint32_t)span->token_count,
        (uint32_t)model->rows[span->row_index].position,
        (uint32_t)span->row_index,
        (uint32_t)input_begin,
        (uint32_t)input_count,
        span->flags,
        output_limits ? output_limits[i] : 0,
        first_indices ? first_indices[i] : (uint32_t)i,
    };
    uint8_t* record = model->host.bytes[TEXT_HOST_PLAN].data +
                      (record_begin + i) * sizeof(fields);
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fields); ++j) {
      iree_unaligned_store_le_u32(record + j * sizeof(uint32_t), fields[j]);
    }
    if (input_count) {
      memcpy(model->host.bytes[TEXT_HOST_INPUT].data + input_begin * 4,
             span->token_ids, input_count * sizeof(int32_t));
    }
    input_begin += input_count;
  }
  return input_begin;
}

// One allocation-free cohort validates inputs, publishes immutable uploads,
// invokes source-owned submissions, and joins the host publication frontier.
static iree_status_t text_epoch(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_result_t* out_results) {
  if (shape_index >= model->shape_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packed text shape index is not loaded");
  }
  const loom_serve_packing_shape_t shape = model->shapes[shape_index];
  if (!span_count || span_count > shape.span_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "epoch span count exceeds stage capacity");
  }
  // Validate the caller's plan before changing reusable payloads or submitting
  // any work. The device consumes this one established partition invariant.
  uint32_t resident_mask = 0;
  iree_host_size_t token_count = 0;
  uint32_t extents[TEXT_ROW_CAPACITY];
  uint32_t first_indices[TEXT_ROW_CAPACITY];
  uint32_t next_indices[TEXT_ROW_CAPACITY];
  uint32_t continued_speculative_mask = 0;
  const iree_host_size_t epoch_count = continuation ? 2 : 1;
  bool proposes = false;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    if (span->row_index >= model->row_count || !span->token_count ||
        !span->token_ids ||
        span->token_count > shape.token_capacity - token_count ||
        (span->flags & ~(LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                         LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid text epoch span %zu", i);
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
        iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    if (generates && (!output_limits || !output_limits[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "proposal requires a speculative text span");
    }
    if (output_limits && output_limits[i]) {
      const loom_serve_text_row_t* row = &model->rows[span->row_index];
      if (output_limits[i] > 4 * epoch_count || span->token_count != 4 ||
          !iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT) ||
          !row->has_prediction || row->transfer.progress[7] ||
          span->token_ids[0] != row->transfer.tokens[0]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid speculative text span %zu", i);
      }
    }
    extents[i] = (uint32_t)span->token_count;
    first_indices[span->row_index] = (uint32_t)i;
    resident_mask |= resident_bit;
    token_count += span->token_count;
    proposes |= generates;
  }

  if (continuation) {
    if (continuation->shape_index >= model->shape_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "continuation shape index is not loaded");
    }
    const loom_serve_packing_shape_t next_shape =
        model->shapes[continuation->shape_index];
    if (!continuation->span_count ||
        continuation->span_count > next_shape.span_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "continuation span count exceeds capacity");
    }
    uint32_t next_mask = 0;
    iree_host_size_t next_tokens = 0;
    for (iree_host_size_t i = 0; i < continuation->span_count; ++i) {
      const loom_serve_text_span_t* span = &continuation->spans[i];
      if (span->row_index >= model->row_count || !span->token_count ||
          span->token_count > next_shape.token_capacity - next_tokens ||
          (span->flags & ~(LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                           LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid continuation span %zu", i);
      }
      const uint32_t bit = 1u << span->row_index;
      if (!(resident_mask & bit) || (next_mask & bit)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "continuation must use distinct first-plan rows");
      }
      const uint32_t first = first_indices[span->row_index];
      const loom_serve_text_span_t* prior = &spans[first];
      const uint32_t limit = continuation->output_limits[i];
      const iree_host_size_t remaining =
          model->context_capacity - model->rows[span->row_index].position;
      if (limit) {
        if (limit > 8 || span->token_count != 4 ||
            span->flags != (LOOM_SERVE_TEXT_SPAN_FLAG_SELECT |
                            LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE) ||
            !iree_any_bit_set(prior->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT) ||
            (output_limits[first] && limit > output_limits[first])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid speculative continuation span %zu",
                                  i);
        }
        const iree_host_size_t extent = output_limits[first]
                                            ? iree_min(8u, limit + 3)
                                            : prior->token_count + 4;
        extents[first] = (uint32_t)iree_min(remaining, extent);
        continued_speculative_mask |= 1u << first;
      } else {
        if (!span->token_ids || output_limits[first] || prior->flags ||
            iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "known continuation requires unselected known input");
        }
        if (span->token_count > remaining - prior->token_count) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "continuation span %zu exceeds context", i);
        }
        extents[first] += (uint32_t)span->token_count;
      }
      next_indices[i] = first;
      next_mask |= bit;
      next_tokens += span->token_count;
    }
  }

  uint32_t required_blocks = 0;
  for (iree_host_size_t i = 0; model->cache.capacity && i < span_count; ++i) {
    const loom_serve_text_row_t* row = &model->rows[spans[i].row_index];
    required_blocks +=
        (uint32_t)((row->position + extents[i] + model->cache.block_size - 1) /
                   model->cache.block_size) -
        row->block_count;
  }
  if (required_blocks > model->cache.pool.available) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch needs %u KV blocks; %u are available",
                            required_blocks, model->cache.pool.available);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_activate(model));
  IREE_RETURN_IF_ERROR(text_grow_blocks(model, span_count, spans, extents));

  const iree_host_size_t input_count =
      text_prepare_plan(model, span_count, spans, output_limits, NULL, 0, 0);
  if (continuation) {
    text_prepare_plan(model, continuation->span_count, continuation->spans,
                      continuation->output_limits, next_indices,
                      TEXT_ROW_CAPACITY, input_count);
  }
  iree_vm_variant_t encode_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_PLAN]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_INPUT]),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(continuation ? (int32_t)continuation->span_count
                                            : 0),
      iree_vm_variant_from_i32(model->eos_token),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_METADATA]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_TOKENS]),
  };
  iree_vm_variant_t encode_results[1] = {0};
  iree_status_t status =
      text_invoke(model, model->encode_epoch,
                  iree_vm_variant_span_from_array(encode_arguments),
                  iree_vm_variant_span_from_array(encode_results));
  int32_t output_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(encode_results[0], &output_count);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(encode_results));
  IREE_RETURN_IF_ERROR(status);
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->host.bytes[TEXT_HOST_METADATA].data,
                  .target_buffer = model->epoch.buffers[1],
                  .length = model->host.bytes[TEXT_HOST_METADATA].data_length}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->host.bytes[TEXT_HOST_TOKENS].data,
                  .target_buffer = model->epoch.buffers[4],
                  .length = model->host.bytes[TEXT_HOST_TOKENS].data_length}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion));
  iree_hal_buffer_t* buffers[] = {
      model->residual,         model->epoch.buffers[1], model->epoch.buffers[2],
      model->row_arena,        model->epoch.buffers[4], model->epoch.buffers[5],
      model->workspace,        model->mtp.carry,        model->mtp.committed,
      model->mtp.results,      model->mtp.cache,        model->mtp.row_table,
      model->mtp.next_results,
  };
  iree_vm_variant_t arguments[9 + IREE_ARRAYSIZE(buffers)] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->vm_types,
                                               model->control_state),
      iree_vm_variant_from_i32((int32_t)shape_index),
      iree_vm_variant_from_i32(proposes),
      iree_vm_variant_from_i32(output_limits != NULL),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(output_count),
      iree_vm_variant_from_i32((int32_t)epoch_count),
      iree_vm_variant_from_i32(continuation ? (int32_t)continuation->shape_index
                                            : 0),
      iree_vm_variant_from_i32(continued_speculative_mask != 0),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    arguments[9 + i] =
        iree_hal_buffer_variant_from_ptr_borrowed(&model->types, buffers[i]);
  }
  iree_vm_variant_t results[1] = {0};
  status = text_invoke(model, model->epoch_step,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  // The synchronous model boundary joins catch-up and feedback independently
  // before publishing host positions or reusing either branch's payloads,
  // including when a later native call rejects after earlier submissions.
  status =
      iree_status_join(status, loom_serve_execution_drain(model->execution));
  IREE_RETURN_IF_ERROR(status);
  iree_vm_variant_t publish_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_PLAN]),
      iree_vm_variant_from_i32((int32_t)span_count),
      iree_vm_variant_from_i32(output_limits != NULL),
      iree_vm_variant_from_i32(continuation != NULL),
      iree_vm_variant_from_i32((int32_t)continued_speculative_mask),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_OUTPUTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_NEXT_RESULTS]),
      iree_vm_buffer_variant_from_ptr_borrowed(
          &model->vm_types, model->host.buffers[TEXT_HOST_PROGRESS]),
  };
  IREE_RETURN_IF_ERROR(
      text_invoke(model, model->publish_epoch,
                  iree_vm_variant_span_from_array(publish_arguments),
                  iree_vm_variant_span_empty()));
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const uint8_t* record =
        model->host.bytes[TEXT_HOST_PROGRESS].data + i * 12 * sizeof(int32_t);
    loom_serve_text_result_t result = {
        .consumed_count = iree_unaligned_load_le_u32(record),
        .known_count = iree_unaligned_load_le_u32(record + 4),
        .output_count = iree_unaligned_load_le_u32(record + 8),
        .verification_count = iree_unaligned_load_le_u32(record + 12),
    };
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(result.tokens); ++j) {
      result.tokens[j] =
          (int32_t)iree_unaligned_load_le_u32(record + 16 + j * 4);
    }
    if (out_results) {
      out_results[i] = result;
    }
    loom_serve_text_row_t* row = &model->rows[spans[i].row_index];
    row->position += result.consumed_count;
    text_trim_blocks(row);
    row->has_prediction = result.output_count != 0;
    if (row->has_prediction) {
      row->transfer.tokens[0] = result.tokens[result.output_count - 1];
      row->transfer.progress[0] = (int32_t)result.output_count;
      row->transfer.progress[7] = row->transfer.tokens[0] == model->eos_token;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_text_model_epoch(loom_serve_text_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_text_span_t* spans) {
  return text_epoch(model, shape_index, span_count, spans, NULL, NULL, NULL);
}

iree_status_t loom_serve_text_model_verify(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t span_count, const loom_serve_text_span_t* spans,
    const uint32_t* output_limits,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_result_t* out_results) {
  if (!model->mtp.enabled) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "MTP artifacts were not loaded");
  }
  return text_epoch(model, shape_index, span_count, spans, output_limits,
                    continuation, out_results);
}

iree_status_t loom_serve_text_row_prefill(loom_serve_text_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids) {
  loom_serve_text_model_t* model = row->model;
  if (!count || count > model->prefill_capacity ||
      count > model->context_capacity - row->position) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prefill chunk exceeds stage or context capacity");
  }
  const iree_time_t start = iree_time_now();
  if (model->mtp.enabled || model->cache.capacity) {
    iree_host_size_t shape_index = 0;
    // The public prefill capacity is bounded by the largest prepared shape.
    while (model->shapes[shape_index].token_capacity < count) {
      ++shape_index;
    }
    const loom_serve_text_span_t span = {
        (iree_host_size_t)(row - model->rows),
        count,
        token_ids,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(
        loom_serve_text_model_epoch(model, shape_index, 1, &span));
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
                  .target_buffer = row->buffers[TEXT_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[TEXT_TOKENS],
                  .length = sizeof(row->transfer.tokens)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = text_step(row, 1);
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

iree_status_t loom_serve_text_row_decode(loom_serve_text_row_t* row) {
  if (!row->has_prediction) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode requires a selected prediction");
  }
  if (row->position == row->model->context_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "text context is full");
  }
  const iree_time_t start = iree_time_now();
  if (row->model->mtp.enabled || row->model->cache.capacity) {
    const int32_t token = row->transfer.tokens[0];
    const loom_serve_text_span_t span = {
        (iree_host_size_t)(row - row->model->rows),
        1,
        &token,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(row->model, 0, 1, &span));
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
                  .target_buffer = row->buffers[TEXT_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[TEXT_TOKENS],
                  .length = sizeof(int32_t)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = text_step(row, 0);
  }
  if (iree_status_is_ok(status)) {
    ++row->position;
    ++row->metrics.decode_steps;
    row->metrics.decode_duration += iree_time_now() - start;
  }
  return status;
}

int32_t loom_serve_text_row_token(const loom_serve_text_row_t* row) {
  return row->transfer.tokens[0];
}

bool loom_serve_text_row_is_eos(const loom_serve_text_row_t* row) {
  return row->transfer.progress[7] != 0;
}

iree_host_size_t loom_serve_text_row_position(
    const loom_serve_text_row_t* row) {
  return row->position;
}

loom_serve_text_metrics_t loom_serve_text_row_metrics(
    const loom_serve_text_row_t* row) {
  return row->metrics;
}
