// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/model.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_profiling.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/service/speech/state.h"
#include "loomc/iree.h"

typedef struct speech_model_stage_t {
  // Compiled source root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Reusable command retaining the model's immutable parameter domains.
  iree_hal_command_buffer_t* command;
  // Number of fixed parameter roots in this stage's reflection.
  uint32_t parameter_count;
  // Owned fixed buffer references, indexed by this stage's fixed binding.
  iree_hal_buffer_t** parameters;
} speech_model_stage_t;

typedef struct speech_invocation_t {
  // Intrusive model ownership, including requests left by terminal failures.
  struct speech_invocation_t* next;
  // Link owning this node, permitting constant-time removal.
  struct speech_invocation_t** previous_next;
  // Source-owned mutable CPU state, distinct from shared model conditioning.
  iree_vm_buffer_t* state;
  // Immutable request upload, retained through every accepted device consumer.
  iree_vm_buffer_t* payload;
  // Stable resident row, independent of its ordinal in any numerical batch.
  iree_host_size_t row_index;
  // Successful state commitment retained through request release.
  bool reserved;
  // Source next-work bytes, refreshed at each retired model selection cut.
  iree_vm_buffer_t* work;
  // Playback position after the most recently collected block.
  uint64_t frame_position;
} speech_invocation_t;

struct loom_serve_speech_model_t {
  // Allocator owning model, request and feedback storage.
  iree_allocator_t host_allocator;
  // Copied preparation evidence independent of compiler result lifetime.
  loom_serve_compilation_report_t compilation;
  // Owned source model identity, independent of the deployment route.
  iree_string_view_t name;
  // Immutable interleaved waveform geometry.
  loom_serve_speech_format_t format;
  // Maximum frames returned by one source completion.
  iree_host_size_t block_capacity;
  // Number of stable resident request rows declared by source.
  iree_host_size_t request_capacity;
  // Maximum additive work in one compatible source batch.
  uint64_t batch_units;
  // Maximum immutable request upload bytes.
  iree_host_size_t input_capacity;
  // Per-request mutable device state extent.
  iree_host_size_t stream_capacity;
  // Whole-cohort device output extent, independent of feedback size.
  iree_host_size_t output_capacity;
  // Model-owned reusable conditioning bank extent.
  iree_host_size_t conditioning_capacity;
  // Maximum compact work packet and source-produced batch upload bytes.
  iree_host_size_t batch_input_capacity;
  // Immutable tokenizer; input.encode uses independent encode states.
  iree_tokenizer_t* tokenizer;
  // Borrowed shared device, outliving this model.
  loom_serve_device_t* owner;
  // Independent model work and feedback timelines on shared queues.
  loom_serve_execution_t* execution;
  // Stable source-addressed input/history and reusable conditioning ownership.
  loom_serve_speech_state_t state;
  // Shared-device admission group pinning all checkpoint domains together.
  loom_serve_residency_t* residency;
  // Explicit owned requests, independent of the serialized VM process.
  speech_invocation_t* invocations;
  // Initial conditioning upload was accepted on this model's work timeline.
  bool conditioning_initialized;
  // Live-source compiler borrowing the shared host compiler queue.
  loom_serve_jit_t* jit;
  // Owned source stage/configuration/checkpoint declarations.
  loom_serve_preparation_t* preparation;
  // Prepared numerical stages in source declaration order.
  struct {
    // Owned array extent, including partially initialized slots.
    uint32_t count;
    // Compiled stage and retained command ownership.
    speech_model_stage_t* values;
  } stages;
  // Source control program and its retained VM reference environment.
  struct {
    // Environment outliving all escaped buffers and linked capabilities.
    iree_vm_environment_t* environment;
    // Canonical HAL reference types borrowed from the environment.
    iree_hal_module_types_t hal_types;
    // Canonical VM reference types borrowed from the environment.
    iree_vm_ref_types_t vm_types;
    // Immutable source stage tags in little-endian i64 order.
    iree_vm_buffer_t* tags;
    // One source process used by serialized owner/CPU calls.
    loom_serve_program_t* program;
    // Opaque model state; source publishes conditioning only after retirement.
    iree_vm_buffer_t* state;
    // Initial conditioning bytes, retained through model queue retirement.
    iree_vm_buffer_t* initial_conditioning;
    // Read-only view of retired feedback, borrowed only during complete.
    iree_vm_buffer_t* feedback;
    // CPU request preparation entry.
    iree_vm_function_t prepare_request;
    // CPU next-work facts, evaluated before cohort selection.
    iree_vm_function_t prepare_work;
    // CPU numerical cohort packing and phase/shape selection.
    iree_vm_function_t prepare_batch;
    // Owner-side bounded submission entry.
    iree_vm_function_t generate;
    // CPU interpretation of successfully retired output.
    iree_vm_function_t complete;
  } control;
  // Checkpoint domains shared by compatible roots across numerical stages.
  struct {
    // Number of domains, including partially initialized plans.
    iree_host_size_t count;
    // First declaration of each domain, borrowed from preparation.
    const loom_serve_preparation_parameter_t** sources;
    // Owned checkpoint and preparation plans in domain order.
    loom_serve_weights_t** plans;
  } weights;
  // One retained numerical cohort, independent of resident request lifetime.
  struct {
    // Selected handles in original packed order, owned by invocations above.
    speech_invocation_t** invocations;
    // Source upload retained through accepted transfer ownership.
    iree_vm_buffer_t* payload;
    // Source execution/publication plan retained through all collection calls.
    iree_vm_buffer_t* plan;
    // Accepted compact input and output roots, relinquished after submission.
    iree_hal_buffer_t* buffers[2];
  } batch;
  // Bounded host feedback, exclusive through source completion and encoding.
  iree_byte_span_t output;
};

// Releases model-owned consumers, not borrowed upload/feedback payloads or the
// storage protection those consumers still require. Partial loads use this same
// sequence, with no submitted queue work to retire.
static void loom_serve_speech_model_release_consumers(
    loom_serve_speech_model_t* model) {
  loom_serve_program_free(model->control.program);
  model->control.program = NULL;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    speech_model_stage_t* stage = &model->stages.values[i];
    iree_hal_command_buffer_release(stage->command);
    stage->command = NULL;
    for (uint32_t j = 0; j < stage->parameter_count; ++j) {
      iree_hal_buffer_release(stage->parameters[j]);
      stage->parameters[j] = NULL;
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->batch.buffers); ++i) {
    iree_hal_buffer_release(model->batch.buffers[i]);
    model->batch.buffers[i] = NULL;
  }
}

void loom_serve_speech_model_retire(loom_serve_speech_model_t* model,
                                    void (*wake)(void* user_data),
                                    void* user_data) {
  loom_serve_execution_set_retirement_observer(model->execution, wake,
                                               user_data);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->weights.plans[i], wake,
                                               user_data);
  }
  loom_serve_speech_state_retire(&model->state, wake, user_data);
  loom_serve_speech_model_release_consumers(model);
}

bool loom_serve_speech_model_is_retired(loom_serve_speech_model_t* model) {
  if (!model) {
    return true;
  }
  bool retired =
      !model->execution || loom_serve_execution_is_retired(model->execution);
  for (iree_host_size_t i = 0; model->weights.plans && i < model->weights.count;
       ++i) {
    retired &= loom_serve_weights_is_retired(model->weights.plans[i]);
  }
  return retired && loom_serve_speech_state_is_retired(&model->state);
}

// Consumes a retired model or a load that never submitted device work. A failed
// platform release preserves the model and remaining reservation handles.
static iree_status_t loom_serve_speech_model_destroy_resources(
    loom_serve_speech_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_speech_model_release_consumers(model);
  loom_serve_speech_state_retire(&model->state, NULL, NULL);
  IREE_ASSERT(loom_serve_speech_model_is_retired(model));
  loom_serve_residency_free(model->residency);
  model->residency = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       model->weights.plans && i < model->weights.count &&
       iree_status_is_ok(status);
       ++i) {
    status = loom_serve_weights_free(model->weights.plans[i]);
    if (iree_status_is_ok(status)) {
      model->weights.plans[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_speech_state_deinitialize(&model->state);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_release(model->control.tags);
    iree_vm_buffer_release(model->control.state);
    for (uint32_t i = 0; i < model->stages.count; ++i) {
      loom_serve_jit_stage_free(model->stages.values[i].compiled);
      iree_allocator_free(model->host_allocator,
                          model->stages.values[i].parameters);
    }
    iree_allocator_free(model->host_allocator, model->stages.values);
    loom_serve_preparation_free(model->preparation);
    while (model->invocations) {
      speech_invocation_t* invocation = model->invocations;
      model->invocations = invocation->next;
      iree_vm_buffer_release(invocation->state);
      iree_vm_buffer_release(invocation->payload);
      iree_vm_buffer_release(invocation->work);
      iree_allocator_free(model->host_allocator, invocation);
    }
    iree_vm_buffer_release(model->control.initial_conditioning);
    iree_vm_buffer_release(model->control.feedback);
    iree_vm_buffer_release(model->batch.payload);
    iree_vm_buffer_release(model->batch.plan);
    iree_allocator_free(model->host_allocator, model->batch.invocations);
    iree_vm_environment_free(model->control.environment);
    loom_serve_execution_release(model->execution);
    iree_allocator_free(model->host_allocator, model->weights.sources);
    iree_allocator_free(model->host_allocator, model->weights.plans);
    iree_allocator_free(model->host_allocator, (void*)model->name.data);
    loom_serve_jit_free(model->jit);
    iree_tokenizer_free(model->tokenizer);
    iree_allocator_free(model->host_allocator, model->output.data);
    iree_allocator_free(model->host_allocator, model);
  }
  return status;
}

iree_status_t loom_serve_speech_model_free_retired(
    loom_serve_speech_model_t* model) {
  return loom_serve_speech_model_destroy_resources(model);
}

iree_status_t loom_serve_speech_model_free(loom_serve_speech_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_speech_model_retire(model, NULL, NULL);
  iree_status_t status = loom_serve_execution_drain(model->execution);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_await_retirement(model->weights.plans[i]);
  }
  loom_serve_retirement_await(&model->state.retirement);
  return iree_status_join(status,
                          loom_serve_speech_model_destroy_resources(model));
}

static iree_status_t loom_serve_speech_load_tokenizer(
    loom_serve_speech_model_t* model, const loom_serve_sources_t* sources,
    iree_string_view_t directory, iree_string_view_t asset) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_path_resolve(directory, asset, model->host_allocator, &path));
  loomc_source_t* contents = NULL;
  iree_status_t status = loom_serve_sources_open(
      sources, iree_make_cstring_view(path), model->host_allocator, &contents);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t bytes = loomc_source_contents(contents);
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)bytes.data, bytes.data_length),
        model->host_allocator, &model->tokenizer);
  }
  loomc_source_release(contents);
  iree_allocator_free(model->host_allocator, path);
  return status;
}

// Cold source results are consumed while their environment is live. Source
// state survives destruction of the cold program and is opaque to this owner.
static iree_status_t loom_serve_speech_prepare(
    loom_serve_speech_model_t* model,
    const loom_serve_speech_model_options_t* options) {
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(
      host_allocator, &model->control.environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    model->control.environment, IREE_SV("vm")),
                                &model->control.vm_types));
  iree_vm_module_t* modules[2] = {0};
  iree_status_t status = loom_serve_input_module_create(
      model->control.environment, NULL, host_allocator, &modules[0]);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(model->control.environment,
                                           host_allocator, &modules[1]);
  }
  iree_vm_variant_t arguments[2] = {0};
  const iree_string_view_t inputs[] = {
      options->asset_directory,
      options->options.size ? options->options : IREE_SV("{}")};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_create(inputs[i].size, 8, host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      iree_byte_span_t bytes;
      status = iree_vm_buffer_map_write(buffer, 0, inputs[i].size, &bytes);
      if (iree_status_is_ok(status)) {
        memcpy(bytes.data, inputs[i].data, inputs[i].size);
      }
      arguments[i] = iree_vm_buffer_variant_from_ptr_move(
          &model->control.vm_types, &buffer);
    }
  }
  iree_vm_variant_t results[15] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(model->owner)),
        model->control.environment, &options->sources, options->sources.prepare,
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){modules, 2}, &model->compilation,
        host_allocator, &model->preparation);
  }
  int64_t capacities[11] = {0};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(capacities) && iree_status_is_ok(status); ++i) {
    status = iree_vm_i64_from_variant(results[i], &capacities[i]);
    if (iree_status_is_ok(status) &&
        (capacities[i] <= 0 || (uint64_t)capacities[i] > SIZE_MAX)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "source capacities must be positive host sizes");
    }
  }
  if (iree_status_is_ok(status) &&
      (capacities[2] % LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT ||
       capacities[4] % LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT ||
       capacities[0] > INT64_MAX / capacities[2] ||
       capacities[0] > INT64_MAX / capacities[4] ||
       (uint64_t)capacities[0] > SIZE_MAX / (uint64_t)capacities[2] ||
       (uint64_t)capacities[0] > SIZE_MAX / (uint64_t)capacities[4] ||
       (uint64_t)capacities[0] > SIZE_MAX / 24)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source speech row geometry is invalid");
  }
  iree_const_byte_span_t metadata[2] = {0};
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[11 + i], &buffer);
    if (iree_status_is_ok(status) && !buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source metadata must not be null");
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_read(buffer, 0, iree_vm_buffer_length(buffer),
                                       &metadata[i]);
    }
  }
  if (iree_status_is_ok(status) && !metadata[1].data_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source model name must not be empty");
  }
  if (iree_status_is_ok(status)) {
    model->request_capacity = (iree_host_size_t)capacities[0];
    model->batch_units = (uint64_t)capacities[1];
    model->input_capacity = (iree_host_size_t)capacities[2];
    model->conditioning_capacity = (iree_host_size_t)capacities[3];
    model->stream_capacity = (iree_host_size_t)capacities[4];
    model->batch_input_capacity = (iree_host_size_t)capacities[5];
    model->output_capacity = (iree_host_size_t)capacities[6];
    model->output.data_length = (iree_host_size_t)capacities[7];
    model->block_capacity = (iree_host_size_t)capacities[8];
    if ((uint64_t)capacities[9] > UINT32_MAX ||
        (uint64_t)capacities[10] > UINT32_MAX ||
        (uint64_t)capacities[10] > SIZE_MAX / sizeof(float) ||
        (uint64_t)capacities[8] >
            (uint64_t)capacities[7] /
                ((uint64_t)capacities[10] * sizeof(float))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "source waveform geometry exceeds its feedback capacity");
    } else {
      model->format.sample_rate = (uint32_t)capacities[9];
      model->format.channels = (uint32_t)capacities[10];
    }
  }
  if (iree_status_is_ok(status)) {
    char* name = NULL;
    status = iree_allocator_clone(host_allocator, metadata[1], (void**)&name);
    model->name = iree_make_string_view(name, metadata[1].data_length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[13], &model->control.state);
  }
  if (iree_status_is_ok(status) && !model->control.state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source state must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[14],
        &model->control.initial_conditioning);
  }
  if (iree_status_is_ok(status) &&
      (!model->control.initial_conditioning ||
       iree_vm_buffer_length(model->control.initial_conditioning) >
           model->conditioning_capacity)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "source initial conditioning exceeds declared storage");
  }
  if (iree_status_is_ok(status) && metadata[0].data_length) {
    status = loom_serve_speech_load_tokenizer(
        model, &options->sources, options->asset_directory,
        iree_make_string_view((const char*)metadata[0].data,
                              metadata[0].data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(modules[0]);
  iree_vm_module_release(modules[1]);
  if (iree_status_is_ok(status)) {
    const iree_host_size_t count =
        loom_serve_preparation_stage_count(model->preparation);
    status = iree_allocator_malloc_array(host_allocator, count,
                                         sizeof(*model->stages.values),
                                         (void**)&model->stages.values);
    if (iree_status_is_ok(status)) {
      model->stages.count = (uint32_t)count;
    }
  }
  return status;
}

// Source declarations name each stage's fixed bindings independently. The
// weight owner validates compatible placement when roots share a checkpoint.
static iree_status_t loom_serve_speech_check_layouts(
    loom_serve_speech_model_t* model) {
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const speech_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    const uint32_t roots =
        program ? program->requirements.fixed_buffer_count : 0;
    if (declaration->parameter_count != roots) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "speech stage %u requires %u checkpoint bindings",
                              i, roots);
    }
    for (iree_host_size_t j = 0; j < declaration->parameter_count; ++j) {
      if (declaration->parameters[j].binding >= roots) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "speech stage %u checkpoint binding %u is outside [0, %u)", i,
            declaration->parameters[j].binding, roots);
      }
    }
    if (!program) {
      if (loom_serve_dispatch_binding_count(
              loom_serve_jit_stage_dispatch(stage->compiled)) != 5) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "direct speech stage requires work, request, "
                                "conditioning, stream and output buffers");
      }
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_serve_preparation_stage_check_workspace(
        declaration, program->requirements.transient.required_byte_length));
    const bool has_workspace =
        program->requirements.transient.required_byte_length != 0;
    if (program->requirements.rebindable_binding_count != 5 + has_workspace ||
        (has_workspace && program->requirements.transient.binding_index != 5) ||
        program->requirements.launch_counts.binding_index != UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "stage %u does not implement the speech stream ABI", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_create_control(
    loom_serve_speech_model_t* model, iree_vm_module_t* module) {
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      model->stages.count * sizeof(int64_t), iree_alignof(int64_t),
      host_allocator, &model->control.tags));
  iree_byte_span_t tags = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(
      model->control.tags, 0, iree_vm_buffer_length(model->control.tags),
      &tags));
  loom_serve_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, model->stages.count, sizeof(*stages), (void**)&stages));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(model->stages.values[i].compiled);
    stages[i] = (loom_serve_stage_t){
        .command_buffer = model->stages.values[i].command,
        .binding_count = 5,
        .transient = program ? program->requirements.transient
                             : (loom_cmd_program_transient_requirement_t){0},
        .dispatch =
            loom_serve_jit_stage_dispatch(model->stages.values[i].compiled),
    };
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = 5,
      .stages = {model->stages.count, stages},
      .feedback = {1, &model->output},
  };
  iree_vm_module_t* native_modules[3] = {0};
  iree_vm_buffer_t* read_only_tags = NULL;
  iree_status_t status = iree_vm_buffer_subspan(
      model->control.tags, 0, tags.data_length, IREE_VM_BUFFER_ACCESS_FLAG_READ,
      host_allocator, &read_only_tags);
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_release(model->control.tags);
    model->control.tags = read_only_tags;
    status =
        loom_serve_module_create(&model->control.hal_types, model->execution,
                                 options, host_allocator, &native_modules[0]);
  }
  iree_allocator_free(host_allocator, stages);
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(model->control.environment,
                                            model->tokenizer, host_allocator,
                                            &native_modules[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(model->control.environment,
                                           host_allocator, &native_modules[2]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link(module,
                                     (iree_vm_module_span_t){native_modules, 3},
                                     host_allocator, &model->control.program);
  }
  iree_vm_module_release(native_modules[0]);
  iree_vm_module_release(native_modules[1]);
  iree_vm_module_release(native_modules[2]);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_process_t* process =
      loom_serve_program_process(model->control.program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("prepare_request"),
      &model->control.prepare_request));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("prepare_work"),
      &model->control.prepare_work));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("prepare_batch"),
      &model->control.prepare_batch));
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("generate"),
      &model->control.generate));
  return iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("complete"), &model->control.complete);
}

// Cold binding records retain the source's checkpoint identity until all
// compatible stage roots have joined one existing parameter-storage owner.
typedef struct speech_parameter_binding_t {
  // Index into the model's checkpoint domain arrays.
  iree_host_size_t domain;
  // Compiled placement and stage-owned output buffer slot.
  loom_serve_weight_root_t root;
} speech_parameter_binding_t;

static iree_status_t loom_serve_speech_bind_parameters(
    loom_serve_speech_model_t* model) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_host_size_t binding_count = 0;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    speech_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    const uint32_t count =
        program ? program->requirements.fixed_buffer_count : 0;
    if (count > SIZE_MAX - binding_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "speech parameter binding count overflow");
    }
    if (count) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          host_allocator, count, sizeof(*stage->parameters),
          (void**)&stage->parameters));
      stage->parameter_count = count;
      binding_count += count;
    }
  }
  if (!binding_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, binding_count, sizeof(*model->weights.sources),
      (void**)&model->weights.sources));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, binding_count, sizeof(*model->weights.plans),
      (void**)&model->weights.plans));
  speech_parameter_binding_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, binding_count, sizeof(*bindings), (void**)&bindings));
  iree_host_size_t cursor = 0;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    speech_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    for (iree_host_size_t j = 0; j < declaration->parameter_count; ++j) {
      const loom_serve_preparation_parameter_t* parameter =
          &declaration->parameters[j];
      iree_host_size_t domain = 0;
      while (domain < model->weights.count &&
             !loom_serve_preparation_parameter_equal(
                 parameter, model->weights.sources[domain])) {
        ++domain;
      }
      if (domain == model->weights.count) {
        model->weights.sources[model->weights.count++] = parameter;
      }
      bindings[cursor++] = (speech_parameter_binding_t){
          .domain = domain,
          .root = {program,
                   loom_cmd_program_parameter_root_at(program,
                                                      parameter->binding),
                   &stage->parameters[parameter->binding]},
      };
    }
  }
  loom_serve_weight_root_t* roots = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, binding_count, sizeof(*roots), (void**)&roots);
  for (iree_host_size_t domain = 0;
       domain < model->weights.count && iree_status_is_ok(status); ++domain) {
    iree_host_size_t count = 0;
    for (iree_host_size_t i = 0; i < binding_count; ++i) {
      if (bindings[i].domain == domain) {
        roots[count++] = bindings[i].root;
      }
    }
    status = loom_serve_weights_allocate(
        loom_serve_device_handle(model->owner),
        loom_serve_device_transfer_queue(model->owner),
        loom_serve_device_dispatch_queue(model->owner),
        loom_serve_device_storage_pool(model->owner),
        loom_serve_device_memory_pool(model->owner), 0, count, roots,
        host_allocator, &model->weights.plans[domain]);
  }
  iree_allocator_free(host_allocator, roots);
  iree_allocator_free(host_allocator, bindings);
  return status;
}

static iree_status_t loom_serve_speech_model_bind(
    loom_serve_speech_model_t* model) {
  IREE_RETURN_IF_ERROR(loom_serve_speech_check_layouts(model));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create_execution(model->owner, &model->execution));
  return loom_serve_speech_bind_parameters(model);
}

static iree_status_t loom_serve_speech_model_link(
    loom_serve_speech_model_t* model, iree_vm_module_t* module,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(model->owner), model->execution,
      model->weights.count, model->weights.plans, host_allocator,
      &model->residency));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    speech_model_stage_t* stage = &model->stages.values[i];
    if (loom_serve_jit_stage_program(stage->compiled)) {
      IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
          stage->compiled, iree_hal_queue_family(dispatch), command_mode, NULL,
          stage->parameters, &stage->command));
    }
  }
  // Packed work/output are ephemeral; histories retain stable arena addresses.
  // Compiler-private workspace is registered independently by the runner.
  IREE_RETURN_IF_ERROR(loom_serve_execution_reserve_workspace(
      model->execution,
      iree_max(model->batch_input_capacity, model->output_capacity),
      LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT));
  IREE_RETURN_IF_ERROR(loom_serve_speech_state_allocate(
      &model->state, model->owner, model->request_capacity,
      model->input_capacity, model->stream_capacity,
      model->conditioning_capacity, LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, model->request_capacity,
      sizeof(*model->batch.invocations), (void**)&model->batch.invocations));
  return loom_serve_speech_create_control(model, module);
}

typedef enum speech_load_phase_e {
  SPEECH_LOAD_PREPARE,
  SPEECH_LOAD_STAGES,
  SPEECH_LOAD_WEIGHTS,
  SPEECH_LOAD_LINK,
  SPEECH_LOAD_COMPLETE,
} speech_load_phase_t;

struct loom_serve_speech_load_t {
  // One private bootstrap task, retired before any native compilation starts.
  loomc_task_t task;
  // Allocator owning this operation, independent of the transferred model.
  iree_allocator_t host_allocator;
  // Owned partial model, transferred only after successful terminal advance.
  loom_serve_speech_model_t* model;
  // Borrowed immutable package inputs, retained by the registration owner.
  loom_serve_speech_model_options_t options;
  // Owner-only next phase; callbacks never make shared-state transitions.
  speech_load_phase_t phase;
  // Next stage or weight domain within the current phase.
  iree_host_size_t index;
  // Recording policy shared by consumer and weight preparation commands.
  iree_hal_command_buffer_mode_t command_mode;
  // Prepared control bytecode, not yet linked to device capabilities.
  iree_vm_module_t* module;
  // Cancellation is irreversible, but never skips accepted work retirement.
  bool cancelled;
  // Private CPU task outcome, handed off by its final release callback.
  iree_status_t task_status;
  // Completion publication shared with workers; owner consumes under mutex.
  struct {
    // Joins a callback's final access with the owner's next turn/destruction.
    iree_slim_mutex_t mutex;
    // An accepted CPU/JIT operation still owns its borrowed storage.
    bool pending;
    // Owned completion status, transferred to the next owner turn.
    iree_status_t status;
    // Nonblocking notification of the serialized owner.
    loom_serve_speech_load_wake_fn_t wake;
    // Borrowed notification payload, retained through terminal advance.
    void* user_data;
  } event;
};

static void loom_serve_speech_load_complete(void* user_data,
                                            iree_status_t status) {
  loom_serve_speech_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t loom_serve_speech_load_prepare(
    loom_serve_speech_load_t* load) {
  loom_serve_speech_model_t* model = load->model;
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(loom_serve_speech_prepare(model, &load->options));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, model->output.data_length, (void**)&model->output.data));
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ, model->output,
                          iree_vm_buffer_release_callback_null(),
                          host_allocator, &model->control.feedback));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  const iree_string_view_t roots[] = {
      IREE_SVL("prepare_request"), IREE_SVL("prepare_work"),
      IREE_SVL("prepare_batch"), IREE_SVL("generate"), IREE_SVL("complete")};
  iree_status_t status = loom_serve_program_compile(
      loom_serve_host_target_environment(loom_serve_device_host(model->owner)),
      model->control.environment, &load->options.sources,
      load->options.sources.control, IREE_ARRAYSIZE(roots), roots,
      &model->compilation, host_allocator, &load->module);
  IREE_RETURN_IF_ERROR(status);
  return loom_serve_jit_allocate(
      loom_serve_device_handle(model->owner),
      loom_serve_device_dispatch_queue(model->owner),
      loom_serve_device_host(model->owner), &load->options.sources, NULL,
      &model->compilation, host_allocator, &model->jit);
}

static void loom_serve_speech_load_execute(loomc_task_t* task,
                                           loomc_host_size_t worker_ordinal) {
  loom_serve_speech_load_t* load = (loom_serve_speech_load_t*)task;
  load->task_status = loom_serve_speech_load_prepare(load);
}

static void loom_serve_speech_load_release(loomc_task_t* task) {
  loom_serve_speech_load_t* load = (loom_serve_speech_load_t*)task;
  loom_serve_speech_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t speech_load_vtable = {
    .execute = loom_serve_speech_load_execute,
    .destroy = loom_serve_speech_load_release,
};

static void loom_serve_speech_load_compiled(void* user_data,
                                            iree_status_t status,
                                            loom_serve_jit_stage_t* compiled) {
  loom_serve_speech_load_t* load = user_data;
  load->model->stages.values[load->index - 1].compiled = compiled;
  loom_serve_speech_load_complete(load, status);
}

iree_status_t loom_serve_speech_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_speech_model_options_t* options,
    loom_serve_speech_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_speech_load_t** out_load) {
  *out_load = NULL;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  loom_serve_speech_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*load), (void**)&load));
  load->host_allocator = host_allocator;
  load->options = *options;
  load->command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  load->event.wake = wake;
  load->event.user_data = user_data;
  iree_slim_mutex_initialize(&load->event.mutex);
  loomc_task_initialize(&speech_load_vtable, &load->task);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*load->model), (void**)&load->model);
  if (iree_status_is_ok(status)) {
    load->model->host_allocator = host_allocator;
    loom_serve_speech_state_initialize(host_allocator, &load->model->state);
    load->model->owner = device;
    *out_load = load;
  } else {
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(host_allocator, load);
  }
  return status;
}

iree_status_t loom_serve_speech_load_advance(
    loom_serve_speech_load_t* load, loom_serve_speech_load_flags_t flags,
    bool* out_complete) {
  *out_complete = false;
  load->cancelled |=
      iree_any_bit_set(flags, LOOM_SERVE_SPEECH_LOAD_FLAG_CANCEL);
  iree_slim_mutex_lock(&load->event.mutex);
  if (load->event.pending) {
    iree_slim_mutex_unlock(&load->event.mutex);
    return iree_ok_status();
  }
  iree_status_t status = load->event.status;
  load->event.status = iree_ok_status();
  iree_slim_mutex_unlock(&load->event.mutex);
  if (iree_status_is_ok(status) && load->cancelled) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "model load cancelled");
  }
  loom_serve_speech_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->owner);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != SPEECH_LOAD_COMPLETE) {
    switch (load->phase) {
      case SPEECH_LOAD_PREPARE:
        load->phase = SPEECH_LOAD_STAGES;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case SPEECH_LOAD_STAGES:
        if (load->index < model->stages.count) {
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, load->index++);
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status =
              declaration->kind == LOOM_SERVE_PREPARATION_STAGE_DISPATCH
                  ? loom_serve_jit_compile_dispatch_async(
                        model->jit, declaration->root, &declaration->config,
                        loom_serve_speech_load_compiled, load)
                  : loom_serve_jit_compile_async(
                        model->jit, declaration->root, &declaration->config,
                        loom_serve_speech_load_compiled, load);
          submitted = iree_status_is_ok(status);
        } else {
          status = loom_serve_speech_model_bind(model);
          load->index = 0;
          load->phase = SPEECH_LOAD_WEIGHTS;
        }
        break;
      case SPEECH_LOAD_WEIGHTS:
        if (load->index < model->weights.count) {
          const iree_host_size_t i = load->index++;
          const loom_serve_preparation_parameter_t* parameter =
              model->weights.sources[i];
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_weights_prepare_async(
              model->weights.plans[i], host, model->jit, load->command_mode,
              &load->options.sources, parameter->paths, parameter->policy,
              loom_serve_speech_load_complete, load);
          submitted = iree_status_is_ok(status);
        } else {
          load->phase = SPEECH_LOAD_LINK;
        }
        break;
      case SPEECH_LOAD_LINK:
        status = loom_serve_speech_model_link(model, load->module,
                                              load->command_mode);
        load->phase = SPEECH_LOAD_COMPLETE;
        break;
      case SPEECH_LOAD_COMPLETE:
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Rejection owes no callback; a consumed completion has already retired.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = false;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = SPEECH_LOAD_COMPLETE;
  }
  *out_complete = load->phase == SPEECH_LOAD_COMPLETE;
  return status;
}

const loom_serve_compilation_report_t* loom_serve_speech_load_report(
    const loom_serve_speech_load_t* load) {
  return &load->model->compilation;
}

loom_serve_speech_model_t* loom_serve_speech_load_take_model(
    loom_serve_speech_load_t* load) {
  loom_serve_speech_model_t* model = load->model;
  load->model = NULL;
  return model;
}

iree_status_t loom_serve_speech_load_free(loom_serve_speech_load_t* load) {
  if (!load) {
    return iree_ok_status();
  }
  iree_status_t status = loom_serve_speech_model_destroy_resources(load->model);
  if (iree_status_is_ok(status)) {
    iree_vm_module_release(load->module);
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(load->host_allocator, load);
  }
  return status;
}

static void loom_serve_speech_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_speech_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_speech_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_speech_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_speech_load_t* load = NULL;
  iree_status_t status = loom_serve_speech_load_allocate(
      device, options, loom_serve_speech_load_notify, &notification,
      host_allocator, &load);
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_speech_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  if (load) {
    if (iree_status_is_ok(status)) {
      *out_model = loom_serve_speech_load_take_model(load);
    }
    status = iree_status_join(status, loom_serve_speech_load_free(load));
  }
  iree_notification_deinitialize(&notification);
  return status;
}

loom_serve_residency_t* loom_serve_speech_model_residency(
    const loom_serve_speech_model_t* model) {
  return model->residency;
}

static void loom_serve_speech_invocation_free(loom_serve_speech_model_t* model,
                                              speech_invocation_t* invocation) {
  *invocation->previous_next = invocation->next;
  if (invocation->next) {
    invocation->next->previous_next = invocation->previous_next;
  }
  iree_vm_buffer_release(invocation->state);
  iree_vm_buffer_release(invocation->payload);
  iree_vm_buffer_release(invocation->work);
  iree_allocator_free(model->host_allocator, invocation);
}

static iree_status_t loom_serve_speech_generator_prepare(
    void* self, iree_host_size_t row_index,
    const loom_serve_speech_request_t* request, void** out_invocation) {
  *out_invocation = NULL;
  loom_serve_speech_model_t* model = self;
  iree_vm_variant_t arguments[6] = {iree_vm_buffer_variant_from_ptr_borrowed(
      &model->control.vm_types, model->control.state)};
  const iree_string_view_t strings[] = {
      iree_string_builder_view(&request->input),
      iree_string_builder_view(&request->voice),
      iree_string_builder_view(&request->instructions)};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(strings) && iree_status_is_ok(status); ++i) {
    // Source request state may retain a string beyond the invocation itself.
    // Its VM reference must own its bytes rather than borrowing the HTTP claim.
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(strings[i].data, strings[i].size), 1,
        model->host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[1 + i] = iree_vm_buffer_variant_from_ptr_move(
          &model->control.vm_types, &buffer);
    }
  }
  arguments[4] = iree_vm_variant_from_i64((int64_t)request->seed);
  arguments[5] = iree_vm_variant_from_f32(request->speed);
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                       model->control.prepare_request,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
  }
  iree_vm_buffer_t* state = NULL;
  iree_vm_buffer_t* payload = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[0], &state);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[1], &payload);
  }
  if (iree_status_is_ok(status) &&
      (!state || !payload || !iree_vm_buffer_length(payload) ||
       iree_vm_buffer_length(payload) > model->input_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source request exceeds declared speech storage");
  }
  speech_invocation_t* invocation = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(model->host_allocator, sizeof(*invocation),
                                   (void**)&invocation);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_retain(state);
    iree_vm_buffer_retain(payload);
    invocation->state = state;
    invocation->payload = payload;
    invocation->row_index = row_index;
    invocation->next = model->invocations;
    invocation->previous_next = &model->invocations;
    if (invocation->next) {
      invocation->next->previous_next = &invocation->next;
    }
    model->invocations = invocation;
    *out_invocation = invocation;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

static void loom_serve_speech_generator_discard(void* self, void* invocation) {
  loom_serve_speech_model_t* model = self;
  loom_serve_residency_cancel_admission(model->residency);
  loom_serve_speech_invocation_free(model, invocation);
}

static iree_status_t loom_serve_speech_generator_admit(
    void* self, void* opaque_invocation,
    loom_serve_residency_admission_t* out_admission,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  loom_serve_speech_model_t* model = self;
  speech_invocation_t* invocation = opaque_invocation;
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_try_admit(model->residency, out_admission));
  if (*out_admission != LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    return iree_ok_status();
  }
  // Activation already owns a pin and may have accepted device work. Even a
  // state-capacity denial retains that pin until complete_admit.
  IREE_RETURN_IF_ERROR(loom_serve_speech_state_try_reserve(
      &model->state, invocation->row_index, &invocation->reserved));
  if (invocation->reserved) {
    iree_const_byte_span_t payload;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
        invocation->payload, 0, iree_vm_buffer_length(invocation->payload),
        &payload));
    iree_hal_transfer_operation_t uploads[2] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {
             .source = payload.data,
             .target_buffer = loom_serve_speech_state_buffer(
                 &model->state, LOOM_SERVE_SPEECH_STATE_INPUT),
             .target_offset = invocation->row_index * model->input_capacity,
             .length = payload.data_length}}};
    iree_host_size_t upload_count = 1;
    if (!model->conditioning_initialized) {
      iree_const_byte_span_t initial;
      IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
          model->control.initial_conditioning, 0,
          iree_vm_buffer_length(model->control.initial_conditioning),
          &initial));
      if (initial.data_length) {
        uploads[upload_count++] = (iree_hal_transfer_operation_t){
            .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
            .upload = {.source = initial.data,
                       .target_buffer = loom_serve_speech_state_buffer(
                           &model->state, LOOM_SERVE_SPEECH_STATE_CONDITIONING),
                       .length = initial.data_length}};
      }
    }
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
        model->execution, upload_count, uploads, &value));
    // A failed timeline terminates the model; partial initialization is never
    // published as a reusable cache entry by source completion.
    model->conditioning_initialized = true;
  }
  *out_frontier = loom_serve_execution_frontier(model->execution);
  return iree_ok_status();
}

static bool loom_serve_speech_generator_complete_admit(
    void* self, void* opaque_invocation) {
  loom_serve_speech_model_t* model = self;
  speech_invocation_t* invocation = opaque_invocation;
  if (!invocation->reserved) {
    loom_serve_residency_release(model->residency);
  }
  return invocation->reserved;
}

static iree_status_t loom_serve_speech_generator_prepare_work(
    void* self, void* opaque_invocation, loom_serve_speech_work_t* out_work) {
  *out_work = (loom_serve_speech_work_t){0};
  loom_serve_speech_model_t* model = self;
  speech_invocation_t* invocation = opaque_invocation;
  iree_vm_buffer_release(invocation->work);
  invocation->work = NULL;
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               invocation->state)};
  iree_vm_variant_t results[3] = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.prepare_work, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  iree_vm_buffer_t* work = NULL;
  int64_t units = 0;
  int64_t tag = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[0], &work);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &units);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[2], &tag);
  }
  if (iree_status_is_ok(status) &&
      (!work || iree_vm_buffer_length(work) > model->batch_input_capacity ||
       units <= 0 || (uint64_t)units > model->batch_units)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source speech work exceeds declared capacity");
  }
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_retain(work);
    invocation->work = work;
    *out_work = (loom_serve_speech_work_t){.units = (uint64_t)units,
                                           .tag = (uint64_t)tag};
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

static void loom_serve_speech_generator_release_batch(void* self) {
  loom_serve_speech_model_t* model = self;
  iree_vm_buffer_release(model->batch.payload);
  model->batch.payload = NULL;
  iree_vm_buffer_release(model->batch.plan);
  model->batch.plan = NULL;
}

static iree_status_t loom_serve_speech_generator_prepare_batch(
    void* self, iree_host_size_t count, void* const* invocations) {
  loom_serve_speech_model_t* model = self;
  iree_host_size_t length = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    speech_invocation_t* invocation = invocations[i];
    const iree_host_size_t size = iree_vm_buffer_length(invocation->work);
    if (size > model->batch_input_capacity - length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "source speech work exceeds batch packet capacity");
    }
    length += size;
  }
  iree_vm_buffer_t* ranges = NULL;
  iree_vm_buffer_t* packed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(count * 24, 8, model->host_allocator, &ranges));
  iree_status_t status =
      iree_vm_buffer_create(length, 8, model->host_allocator, &packed);
  iree_byte_span_t records = iree_byte_span_empty();
  iree_byte_span_t bytes = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(ranges, 0, count * 24, &records);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(packed, 0, length, &bytes);
  }
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    speech_invocation_t* invocation = invocations[i];
    iree_const_byte_span_t work;
    status = iree_vm_buffer_map_read(
        invocation->work, 0, iree_vm_buffer_length(invocation->work), &work);
    if (iree_status_is_ok(status)) {
      iree_unaligned_store_le_u64(records.data + i * 24, invocation->row_index);
      iree_unaligned_store_le_u64(records.data + i * 24 + 8, offset);
      iree_unaligned_store_le_u64(records.data + i * 24 + 16, work.data_length);
      memcpy(bytes.data + offset, work.data, work.data_length);
      offset += work.data_length;
    }
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &ranges),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &packed)};
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                       model->control.prepare_batch,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[0], &model->batch.payload);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[1], &model->batch.plan);
  }
  if (iree_status_is_ok(status) &&
      (!model->batch.payload || !model->batch.plan ||
       !iree_vm_buffer_length(model->batch.payload) ||
       iree_vm_buffer_length(model->batch.payload) >
           model->batch_input_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source speech batch exceeds declared storage");
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      model->batch.invocations[i] = invocations[i];
    }
  } else {
    loom_serve_speech_generator_release_batch(model);
  }
  return status;
}

static iree_status_t loom_serve_speech_generator_submit(
    void* self, loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  loom_serve_speech_model_t* model = self;
  iree_const_byte_span_t payload;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      model->batch.payload, 0, iree_vm_buffer_length(model->batch.payload),
      &payload));
  const iree_host_size_t lengths[] = {payload.data_length,
                                      model->output_capacity};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(lengths) && iree_status_is_ok(status); ++i) {
    const iree_hal_pool_reservation_request_t reservation =
        iree_hal_make_pool_reservation_request(
            lengths[i], LOOM_SERVE_SPEECH_BUFFER_ALIGNMENT);
    status = loom_serve_execution_alloca(model->execution, &reservation,
                                         &model->batch.buffers[i]);
  }
  uint64_t value = 0;
  if (iree_status_is_ok(status)) {
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = payload.data,
                   .target_buffer = model->batch.buffers[0],
                   .length = payload.data_length}};
    status =
        loom_serve_execution_transfer(model->execution, 1, &upload, &value);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                                 model->batch.plan),
        iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                  model->batch.buffers[0]),
        iree_hal_buffer_variant_from_ptr_borrowed(
            &model->control.hal_types,
            loom_serve_speech_state_buffer(&model->state,
                                           LOOM_SERVE_SPEECH_STATE_INPUT)),
        iree_hal_buffer_variant_from_ptr_borrowed(
            &model->control.hal_types,
            loom_serve_speech_state_buffer(
                &model->state, LOOM_SERVE_SPEECH_STATE_CONDITIONING)),
        iree_hal_buffer_variant_from_ptr_borrowed(
            &model->control.hal_types,
            loom_serve_speech_state_buffer(&model->state,
                                           LOOM_SERVE_SPEECH_STATE_STREAM)),
        iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                  model->batch.buffers[1])};
    status = iree_vm_invoke(
        loom_serve_program_invocation(model->control.program),
        model->control.generate, iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_empty());
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_dealloca(model->execution,
                                           model->batch.buffers[0], &value);
  }
  if (iree_status_is_ok(status)) {
    // Output can retire only after the independent feedback branch consumes it.
    loom_serve_execution_frontier_t frontier =
        loom_serve_execution_frontier(model->execution);
    status = loom_serve_execution_barrier(
        model->execution, (iree_hal_semaphore_list_t){
                              1, &frontier.semaphores[1], &frontier.values[1]});
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_dealloca(model->execution,
                                           model->batch.buffers[1], &value);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->batch.buffers); ++i) {
    iree_hal_buffer_release(model->batch.buffers[i]);
    model->batch.buffers[i] = NULL;
  }
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_serve_execution_frontier(model->execution);
  }
  return status;
}

static iree_status_t loom_serve_speech_generator_collect(
    void* self, iree_host_size_t ordinal,
    loom_serve_speech_block_t* out_block) {
  *out_block = (loom_serve_speech_block_t){0};
  loom_serve_speech_model_t* model = self;
  speech_invocation_t* invocation = model->batch.invocations[ordinal];
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               invocation->state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->batch.plan),
      iree_vm_variant_from_i64((int64_t)ordinal),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.feedback)};
  iree_vm_variant_t results[4] = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.complete, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  int64_t values[3] = {0};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(values) && iree_status_is_ok(status); ++i) {
    status = iree_vm_i64_from_variant(results[i], &values[i]);
  }
  int32_t complete = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[3], &complete);
  }
  const uint64_t frame_bytes = model->format.channels * sizeof(float);
  if (iree_status_is_ok(status) &&
      (values[0] < 0 || (uint64_t)values[0] != invocation->frame_position ||
       values[1] < 0 || (uint64_t)values[1] > model->output.data_length ||
       values[2] < 0 || (uint64_t)values[2] > model->block_capacity ||
       (uint64_t)values[2] >
           (model->output.data_length - (uint64_t)values[1]) / frame_bytes ||
       values[2] > INT64_MAX - values[0] || (complete != 0 && complete != 1))) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source waveform range or position is invalid");
  }
  if (iree_status_is_ok(status)) {
    *out_block = (loom_serve_speech_block_t){
        .position = invocation->frame_position,
        .frame_count = (iree_host_size_t)values[2],
        .samples = iree_make_const_byte_span(
            model->output.data + values[1],
            (iree_host_size_t)values[2] * frame_bytes),
        .complete = complete != 0};
    invocation->frame_position += (uint64_t)values[2];
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

static iree_status_t loom_serve_speech_generator_release(
    void* self, void* opaque_invocation) {
  loom_serve_speech_model_t* model = self;
  speech_invocation_t* invocation = opaque_invocation;
  IREE_RETURN_IF_ERROR(loom_serve_speech_state_release_row(
      &model->state, invocation->row_index));
  loom_serve_residency_release(model->residency);
  loom_serve_speech_invocation_free(model, invocation);
  return iree_ok_status();
}

static loom_serve_weight_activity_t loom_serve_speech_generator_weight_activity(
    void* self) {
  loom_serve_speech_model_t* model = self;
  return loom_serve_residency_activity(model->residency);
}

loom_serve_speech_generator_t loom_serve_speech_model_generator(
    loom_serve_speech_model_t* model) {
  return (loom_serve_speech_generator_t){
      .self = model,
      .format = model->format,
      .block_capacity = model->block_capacity,
      .request_capacity = model->request_capacity,
      .batch_units = model->batch_units,
      .prepare = loom_serve_speech_generator_prepare,
      .discard = loom_serve_speech_generator_discard,
      .try_admit = loom_serve_speech_generator_admit,
      .complete_admit = loom_serve_speech_generator_complete_admit,
      .prepare_work = loom_serve_speech_generator_prepare_work,
      .prepare_batch = loom_serve_speech_generator_prepare_batch,
      .submit = loom_serve_speech_generator_submit,
      .collect = loom_serve_speech_generator_collect,
      .release_batch = loom_serve_speech_generator_release_batch,
      .release = loom_serve_speech_generator_release,
      .tokenizer = model->tokenizer,
      .capacity_domain = loom_serve_residency_cache(model->residency),
      .weight_activity = loom_serve_speech_generator_weight_activity};
}
