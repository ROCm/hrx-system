// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/model.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/tooling/device_profiling.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/runtime/program/module.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/runtime/program/stage_set.h"
#include "loom/serve/runtime/program/tokenizer_load.h"
#include "loom/serve/service/executables.h"
#include "loomc/iree.h"

struct loom_serve_finite_model_t {
  // Allocator owning this model and its invocation metadata.
  iree_allocator_t host_allocator;
  // Copied startup evidence, independent of every compiler result lifetime.
  loom_serve_compilation_report_t compilation;
  // Owned public identifier returned by the source bootstrap.
  iree_string_view_t name;
  // Positive additive cohort capacity returned by the cold source.
  uint64_t batch_limit;
  // Maximum uploaded bytes, including source-defined padding.
  iree_host_size_t input_capacity;
  // Maximum device output geometry, independent of the live result prefix.
  iree_host_size_t output_capacity;
  // Maximum live result bytes within the physical device output.
  iree_host_size_t result_capacity;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Borrowed shared device/backing owner, outliving the model.
  loom_serve_device_t* owner;
  // Owned model work and feedback timelines, independent of other models.
  loom_serve_execution_t* execution;
  // Shared-device admission group protecting every checkpoint domain together.
  loom_serve_residency_t* residency;
  // Cohort input remains live through actual producer retirement.
  struct {
    // Prepared upload, retained even after partial enqueue failure.
    iree_vm_buffer_t* payload;
    // Owned checked result metadata until successful submission transfers it.
    loom_serve_finite_result_t* result;
    // Source-selected retained command ordinal.
    uint32_t stage;
  } invocation;
  // Cold live-source compiler borrowing the shared host task pool.
  loom_serve_jit_t* jit;
  // Owned source declarations, independent of the bootstrap VM and arguments.
  loom_serve_preparation_t* preparation;
  // Code generations and batch captures, independent of checkpoint residency.
  loom_serve_executables_t* executables;
  // Idle request-code policy captured independently of source-owned options.
  loom_serve_variant_cache_options_t specialization_cache;
  // Cold retained shapes sharing one parameter and issue-time buffer bank.
  struct {
    // Number of owned slots, including partially initialized slots on failure.
    uint32_t count;
    // Owned stage array in source declaration order.
    loom_serve_jit_stage_t** values;
    // Original recordings, independent of the compiler product lifetime.
    loom_serve_stage_table_t* table;
  } stages;
  // Shared source control and its native submission capabilities.
  struct {
    // Environment owning the canonical buffer reference types.
    iree_vm_environment_t* environment;
    // HAL reference types borrowed from the environment.
    iree_hal_module_types_t hal_types;
    // Core byte-buffer reference types borrowed from the environment.
    iree_vm_ref_types_t vm_types;
    // Owned stage tags in little-endian i64 declaration order.
    iree_vm_buffer_t* tags;
    // One source-JIT process for every serialized request in this residency.
    loom_serve_program_t* program;
    // Retained runner module whose execution table can change independently.
    iree_vm_module_t* runner;
    // Scoped request declaration capability, retained beside the runner.
    iree_vm_module_t* preparation;
    // Opaque source-owned state, retained across serialized requests.
    iree_vm_buffer_t* state;
    // Cold-resolved request preparation function.
    iree_vm_function_t prepare_request;
    // Cold-resolved cohort construction function.
    iree_vm_function_t prepare_batch;
    // Cold-resolved device command submission function.
    iree_vm_function_t run;
  } control;
  // Shared immutable parameter domains in command reflection order.
  struct {
    // Number of owned slots, including partial initialization.
    uint32_t count;
    // Owned HAL buffer array; uninitialized slots are NULL.
    iree_hal_buffer_t** values;
    // Owned checkpoint/preparation plans in the same domain order.
    loom_serve_weights_t** plans;
  } weights;
  // Queue-allocated cohort input/output roots, released before execution drain.
  iree_hal_buffer_t* buffers[2];
};

// Releases model-owned consumers, not borrowed upload payloads or the
// storage protection those consumers still require. Partial loads use this same
// sequence, with no submitted queue work to retire.
static void loom_serve_finite_model_release_consumers(
    loom_serve_finite_model_t* model) {
  loom_serve_executables_free(model->executables);
  model->executables = NULL;
  loom_serve_program_free(model->control.program);
  model->control.program = NULL;
  iree_vm_module_release(model->control.runner);
  iree_vm_module_release(model->control.preparation);
  model->control.preparation = NULL;
  model->control.runner = NULL;
  loom_serve_stage_table_free(model->stages.table);
  model->stages.table = NULL;
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    iree_hal_buffer_release(model->weights.values[i]);
    model->weights.values[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->buffers); ++i) {
    iree_hal_buffer_release(model->buffers[i]);
    model->buffers[i] = NULL;
  }
}

void loom_serve_finite_model_retire(loom_serve_finite_model_t* model,
                                    void (*wake)(void* user_data),
                                    void* user_data) {
  loom_serve_execution_set_retirement_observer(model->execution, wake,
                                               user_data);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->weights.plans[i], wake,
                                               user_data);
  }
  loom_serve_finite_model_release_consumers(model);
}

bool loom_serve_finite_model_is_retired(loom_serve_finite_model_t* model) {
  if (!model) {
    return true;
  }
  bool retired =
      !model->execution || loom_serve_execution_is_retired(model->execution);
  for (iree_host_size_t i = 0; model->weights.plans && i < model->weights.count;
       ++i) {
    retired &= loom_serve_weights_is_retired(model->weights.plans[i]);
  }
  return retired;
}

// Consumes a retired model or a load that never submitted device work. A failed
// platform release preserves the model and remaining reservation handles.
static iree_status_t loom_serve_finite_model_destroy_resources(
    loom_serve_finite_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_finite_model_release_consumers(model);
  IREE_ASSERT(loom_serve_finite_model_is_retired(model));
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
    iree_vm_buffer_release(model->control.tags);
    iree_vm_buffer_release(model->control.state);
    for (uint32_t i = 0; i < model->stages.count; ++i) {
      loom_serve_jit_stage_free(model->stages.values[i]);
    }
    iree_allocator_free(model->host_allocator, model->stages.values);
    loom_serve_preparation_free(model->preparation);
    loom_serve_finite_result_release(model->invocation.result);
    iree_vm_buffer_release(model->invocation.payload);
    iree_vm_environment_free(model->control.environment);
    if (model->execution) {
      // Results can retain the timeline after this model and its wake payload
      // are gone. No producer views remain, so detach the completed observer.
      loom_serve_execution_set_retirement_observer(model->execution, NULL,
                                                   NULL);
    }
    loom_serve_execution_release(model->execution);
    iree_allocator_free(model->host_allocator, model->weights.values);
    iree_allocator_free(model->host_allocator, model->weights.plans);
    iree_allocator_free(model->host_allocator, (void*)model->name.data);
    loom_serve_jit_release(model->jit);
    iree_tokenizer_free(model->tokenizer);
    iree_allocator_free(model->host_allocator, model);
  }
  return status;
}

iree_status_t loom_serve_finite_model_free_retired(
    loom_serve_finite_model_t* model) {
  return loom_serve_finite_model_destroy_resources(model);
}

iree_status_t loom_serve_finite_model_free(loom_serve_finite_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_finite_model_retire(model, NULL, NULL);
  iree_status_t status = loom_serve_execution_drain(model->execution);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_await_retirement(model->weights.plans[i]);
  }
  return iree_status_join(status,
                          loom_serve_finite_model_destroy_resources(model));
}

// Cold source results are consumed while their environment is live. Source
// state survives destruction of the cold program and is opaque to this owner.
static iree_status_t loom_serve_finite_prepare(
    loom_serve_finite_model_t* model,
    const loom_serve_finite_model_options_t* options,
    void (*wake)(void* user_data), void* user_data,
    loom_serve_tokenizer_load_t** out_tokenizer_load) {
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
  const iree_string_view_t inputs[] = {options->asset_directory,
                                       options->options};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
        iree_make_const_byte_span(inputs[i].data, inputs[i].size), 8,
        host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      arguments[i] = iree_vm_buffer_variant_from_ptr_move(
          &model->control.vm_types, &buffer);
    }
  }
  iree_vm_variant_t results[7] = {0};
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
  int64_t capacities[4] = {0};
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    status = iree_vm_i64_from_variant(results[i], &capacities[i]);
    if (iree_status_is_ok(status) &&
        (capacities[i] <= 0 || (uint64_t)capacities[i] > SIZE_MAX)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "source capacities must be positive host sizes");
    }
  }
  iree_const_byte_span_t metadata[2] = {0};
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[4 + i], &buffer);
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
    char* name = NULL;
    status = iree_allocator_clone(host_allocator, metadata[1], (void**)&name);
    model->name = iree_make_string_view(name, metadata[1].data_length);
    model->batch_limit = (uint64_t)capacities[0];
    model->input_capacity = (iree_host_size_t)capacities[1];
    model->output_capacity = (iree_host_size_t)capacities[2];
    model->result_capacity = (iree_host_size_t)capacities[3];
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[6], &model->control.state);
  }
  if (iree_status_is_ok(status) && !model->control.state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source state must not be null");
  }
  if (iree_status_is_ok(status) && metadata[0].data_length) {
    char* path = NULL;
    status = loom_serve_path_resolve(
        options->asset_directory,
        iree_make_string_view((const char*)metadata[0].data,
                              metadata[0].data_length),
        host_allocator, &path);
    if (iree_status_is_ok(status)) {
      status = loom_serve_tokenizer_load_start(
          loom_serve_host_compiler_sink(loom_serve_device_host(model->owner)),
          &options->sources, iree_make_cstring_view(path), wake, user_data,
          host_allocator, out_tokenizer_load);
    }
    iree_allocator_free(host_allocator, path);
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

// The source catalog is external. Every retained shape must describe the same
// immutable placement before a single checkpoint byte is loaded or shared.
static iree_status_t loom_serve_finite_check_layouts(
    loom_serve_finite_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0]);
  const loom_serve_stage_record_options_t recording = {.binding_count = 2};
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const loom_serve_jit_stage_t* stage = model->stages.values[i];
    const loom_cmd_program_t* program = loom_serve_jit_stage_program(stage);
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    IREE_RETURN_IF_ERROR(loom_serve_preparation_stage_check_parameters(
        declaration, program,
        loom_serve_preparation_stage(model->preparation, 0), canonical));
    IREE_RETURN_IF_ERROR(loom_serve_stage_record_check(&recording, stage));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_finite_create_control(
    loom_serve_finite_model_t* model, iree_vm_module_t* module) {
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      model->stages.count * sizeof(int64_t), iree_alignof(int64_t),
      host_allocator, &model->control.tags));
  iree_byte_span_t tags = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(
      model->control.tags, 0, iree_vm_buffer_length(model->control.tags),
      &tags));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = 2,
      .stages = {model->stages.table->part_count, model->stages.table->parts},
  };
  iree_vm_module_t* native_modules[4] = {0};
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
  if (iree_status_is_ok(status)) {
    model->control.runner = native_modules[0];
    iree_vm_module_retain(model->control.runner);
  }
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
    status = loom_serve_preparation_module_create(
        model->control.environment, host_allocator, &native_modules[3]);
    model->control.preparation = native_modules[3];
    iree_vm_module_retain(model->control.preparation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link(module,
                                     (iree_vm_module_span_t){native_modules, 4},
                                     host_allocator, &model->control.program);
  }
  iree_vm_module_release(native_modules[0]);
  iree_vm_module_release(native_modules[1]);
  iree_vm_module_release(native_modules[2]);
  iree_vm_module_release(native_modules[3]);
  IREE_RETURN_IF_ERROR(status);
  const iree_vm_signature_type_t buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = model->control.vm_types.buffer};
  const iree_vm_signature_type_t device_buffer = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_REF,
      .value.ref = model->control.hal_types.buffer};
  const iree_vm_signature_type_t i32 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I32};
  const iree_vm_signature_type_t i64 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_I64};
  const iree_vm_signature_type_t request_arguments[] = {buffer, buffer};
  const iree_vm_signature_type_t request_results[] = {buffer, i64, i64};
  const iree_vm_signature_type_t batch_arguments[] = {buffer, buffer, buffer,
                                                      buffer};
  const iree_vm_signature_type_t batch_results[] = {i32, buffer, buffer};
  const iree_vm_signature_type_t run_arguments[] = {i32, device_buffer,
                                                    device_buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("prepare_request"),
       {request_arguments, IREE_ARRAYSIZE(request_arguments)},
       {request_results, IREE_ARRAYSIZE(request_results)},
       &model->control.prepare_request},
      {IREE_SVL("prepare_batch"),
       {batch_arguments, IREE_ARRAYSIZE(batch_arguments)},
       {batch_results, IREE_ARRAYSIZE(batch_results)},
       &model->control.prepare_batch},
      {IREE_SVL("run"),
       {run_arguments, IREE_ARRAYSIZE(run_arguments)},
       {NULL, 0},
       &model->control.run},
  };
  return loom_serve_program_bind(model->control.program,
                                 IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t loom_serve_finite_model_bind(
    loom_serve_finite_model_t* model) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_finite_check_layouts(model));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create_execution(model->owner, &model->execution));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0]);
  const uint32_t roots = program ? program->requirements.fixed_buffer_count : 0;
  if (roots) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, roots, sizeof(*model->weights.values),
        (void**)&model->weights.values));
    model->weights.count = roots;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, roots, sizeof(*model->weights.plans),
        (void**)&model->weights.plans));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < roots && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_root_t reflected =
        loom_cmd_program_parameter_root_at(program, i);
    const loom_serve_weight_root_t root = {
        program, reflected,
        &model->weights.values[reflected.fixed_buffer_index]};
    status = loom_serve_weights_allocate(
        device, loom_serve_device_transfer_queue(model->owner), dispatch,
        loom_serve_device_storage_pool(model->owner),
        loom_serve_device_memory_pool(model->owner), 0, 1, &root,
        host_allocator, &model->weights.plans[i]);
  }
  return status;
}

static iree_status_t loom_serve_finite_model_link(
    loom_serve_finite_model_t* model, iree_vm_module_t* module,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(model->owner), model->execution,
      model->weights.count, model->weights.plans, host_allocator,
      &model->residency));
  const loom_serve_executable_options_t options = {
      .device = model->owner,
      .jit = model->jit,
      .preparation = model->preparation,
      .layout = loom_serve_jit_stage_program(model->stages.values[0]),
      .recording = {.queue_family = iree_hal_queue_family(dispatch),
                    .parameters = model->weights.values,
                    .command_mode = command_mode,
                    .binding_count = 2},
      .cache = model->specialization_cache};
  IREE_RETURN_IF_ERROR(loom_serve_stage_table_allocate(
      &options.recording, NULL, model->stages.count, model->stages.values,
      host_allocator, &model->stages.table));
  IREE_RETURN_IF_ERROR(loom_serve_finite_create_control(model, module));
  loom_serve_executable_options_t bound_options = options;
  bound_options.stages = model->stages.table;
  bound_options.runner = model->control.runner;
  return loom_serve_executables_allocate(&bound_options, host_allocator,
                                         &model->executables);
}

loom_serve_execution_update_t loom_serve_finite_model_execution_update(
    loom_serve_finite_model_t* model) {
  return loom_serve_executables_update(model->executables);
}

typedef enum loom_serve_finite_load_phase_e {
  FINITE_LOAD_PREPARE,
  FINITE_LOAD_STAGES,
  FINITE_LOAD_BIND,
  FINITE_LOAD_WEIGHTS,
  FINITE_LOAD_LINK,
  FINITE_LOAD_COMPLETE,
} loom_serve_finite_load_phase_t;

struct loom_serve_finite_load_t {
  // One private bootstrap task, retired before any native compilation starts.
  loomc_task_t task;
  // Allocator owning this operation, independent of the transferred model.
  iree_allocator_t host_allocator;
  // Owned partial model, transferred only after successful terminal advance.
  loom_serve_finite_model_t* model;
  // Borrowed immutable package inputs, retained by the registration owner.
  loom_serve_finite_model_options_t options;
  // Owner-only next phase; callbacks never make shared-state transitions.
  loom_serve_finite_load_phase_t phase;
  // Next parameter domain during checkpoint preparation.
  uint32_t index;
  // Recording policy shared by consumer and weight preparation commands.
  iree_hal_command_buffer_mode_t command_mode;
  // Prepared control bytecode, not yet linked to device capabilities.
  iree_vm_module_t* module;
  // Cancellation is irreversible, but never skips accepted work retirement.
  bool cancelled;
  // Private CPU task outcome, handed off by its final release callback.
  iree_status_t task_status;
  // Tokenizer construction overlaps control/native compilation and indexing.
  loom_serve_tokenizer_load_t* tokenizer_load;
  // Terminal model outcome retained until tokenizer work also retires.
  iree_status_t terminal_status;
  // Completion publication shared with workers; owner consumes under mutex.
  struct {
    // Joins a callback's final access with the owner's next turn/destruction.
    iree_slim_mutex_t mutex;
    // An accepted CPU/JIT operation still owns its borrowed storage.
    bool pending;
    // Owned completion status, transferred to the next owner turn.
    iree_status_t status;
    // Nonblocking notification of the serialized owner.
    loom_serve_finite_load_wake_fn_t wake;
    // Borrowed notification payload, retained through terminal advance.
    void* user_data;
  } event;
};

static void loom_serve_finite_load_complete(void* user_data,
                                            iree_status_t status) {
  loom_serve_finite_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t loom_serve_finite_load_prepare(
    loom_serve_finite_load_t* load) {
  loom_serve_finite_model_t* model = load->model;
  model->specialization_cache = load->options.specialization_cache;
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(
      loom_serve_finite_prepare(model, &load->options, load->event.wake,
                                load->event.user_data, &load->tokenizer_load));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  const iree_string_view_t roots[] = {
      IREE_SVL("prepare_request"), IREE_SVL("prepare_batch"), IREE_SVL("run")};
  iree_status_t status = loom_serve_program_compile(
      loom_serve_host_target_environment(loom_serve_device_host(model->owner)),
      model->control.environment, &load->options.sources,
      load->options.sources.control, IREE_ARRAYSIZE(roots), roots,
      &model->compilation, host_allocator, &load->module);
  IREE_RETURN_IF_ERROR(status);
  return loom_serve_jit_create(
      loom_serve_device_handle(model->owner),
      loom_serve_device_dispatch_queue(model->owner),
      loom_serve_device_host(model->owner), &load->options.sources, NULL,
      &model->compilation, host_allocator, &model->jit);
}

static void loom_serve_finite_load_execute(loomc_task_t* task,
                                           loomc_host_size_t worker_ordinal) {
  loom_serve_finite_load_t* load = (loom_serve_finite_load_t*)task;
  load->task_status = loom_serve_finite_load_prepare(load);
}

static void loom_serve_finite_load_release(loomc_task_t* task) {
  loom_serve_finite_load_t* load = (loom_serve_finite_load_t*)task;
  loom_serve_finite_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t finite_load_vtable = {
    .execute = loom_serve_finite_load_execute,
    .destroy = loom_serve_finite_load_release,
};

static void loom_serve_finite_load_compiled(void* user_data,
                                            iree_status_t status,
                                            loom_serve_stage_set_t* compiled) {
  loom_serve_finite_load_t* load = user_data;
  for (iree_host_size_t i = 0; i < compiled->count; ++i) {
    load->model->stages.values[i] = compiled->values[i];
    compiled->values[i] = NULL;
  }
  loom_serve_stage_set_free(compiled);
  loom_serve_finite_load_complete(load, status);
}

iree_status_t loom_serve_finite_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_finite_load_t** out_load) {
  *out_load = NULL;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  loom_serve_finite_load_t* load = NULL;
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
  loomc_task_initialize(&finite_load_vtable, &load->task);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*load->model), (void**)&load->model);
  if (iree_status_is_ok(status)) {
    load->model->host_allocator = host_allocator;
    load->model->owner = device;
    *out_load = load;
  } else {
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(host_allocator, load);
  }
  return status;
}

iree_status_t loom_serve_finite_load_advance(
    loom_serve_finite_load_t* load, loom_serve_finite_load_flags_t flags,
    bool* out_complete) {
  *out_complete = false;
  load->cancelled |=
      iree_any_bit_set(flags, LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL);
  iree_slim_mutex_lock(&load->event.mutex);
  if (load->event.pending) {
    iree_slim_mutex_unlock(&load->event.mutex);
    return iree_ok_status();
  }
  iree_status_t status =
      iree_status_join(load->terminal_status, load->event.status);
  load->terminal_status = iree_ok_status();
  load->event.status = iree_ok_status();
  iree_slim_mutex_unlock(&load->event.mutex);
  if (iree_status_is_ok(status) && load->cancelled) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "model load cancelled");
  }
  loom_serve_finite_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->owner);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != FINITE_LOAD_COMPLETE) {
    switch (load->phase) {
      case FINITE_LOAD_PREPARE:
        load->phase = FINITE_LOAD_STAGES;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case FINITE_LOAD_STAGES:
        load->phase = FINITE_LOAD_BIND;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = loom_serve_stage_set_compile_catalog_async(
            model->jit, model->preparation, loom_serve_finite_load_compiled,
            load, load->host_allocator);
        submitted = iree_status_is_ok(status);
        break;
      case FINITE_LOAD_BIND:
        status = loom_serve_finite_model_bind(model);
        load->phase = FINITE_LOAD_WEIGHTS;
        break;
      case FINITE_LOAD_WEIGHTS:
        if (load->index < model->weights.count) {
          const uint32_t i = load->index++;
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, 0);
          const loom_serve_preparation_parameter_t* parameter =
              &declaration->parameters[i];
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_weights_prepare_async(
              model->weights.plans[i], host, model->jit, load->command_mode,
              &load->options.sources, parameter->paths, parameter->policy,
              loom_serve_finite_load_complete, load);
          submitted = iree_status_is_ok(status);
        } else {
          load->phase = FINITE_LOAD_LINK;
        }
        break;
      case FINITE_LOAD_LINK:
        if (!loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                            &model->tokenizer)) {
          return iree_ok_status();
        }
        if (iree_status_is_ok(status)) {
          status = loom_serve_finite_model_link(model, load->module,
                                                load->command_mode);
        }
        load->phase = FINITE_LOAD_COMPLETE;
        break;
      case FINITE_LOAD_COMPLETE:
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Rejection owes no callback; a consumed completion has already retired.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = false;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = FINITE_LOAD_COMPLETE;
  }
  if (load->phase == FINITE_LOAD_COMPLETE &&
      !loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                      &model->tokenizer)) {
    load->terminal_status = status;
    return iree_ok_status();
  }
  *out_complete = load->phase == FINITE_LOAD_COMPLETE;
  return status;
}

const loom_serve_compilation_report_t* loom_serve_finite_load_report(
    const loom_serve_finite_load_t* load) {
  return &load->model->compilation;
}

loom_serve_finite_model_t* loom_serve_finite_load_take_model(
    loom_serve_finite_load_t* load) {
  loom_serve_finite_model_t* model = load->model;
  load->model = NULL;
  return model;
}

iree_status_t loom_serve_finite_load_free(loom_serve_finite_load_t* load) {
  if (!load) {
    return iree_ok_status();
  }
  iree_status_t status = loom_serve_finite_model_destroy_resources(load->model);
  if (iree_status_is_ok(status)) {
    iree_vm_module_release(load->module);
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(load->host_allocator, load);
  }
  return status;
}

static void loom_serve_finite_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_finite_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_finite_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_finite_load_t* load = NULL;
  iree_status_t status = loom_serve_finite_load_allocate(
      device, options, loom_serve_finite_load_notify, &notification,
      host_allocator, &load);
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_finite_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  if (load) {
    if (iree_status_is_ok(status)) {
      *out_model = loom_serve_finite_load_take_model(load);
    }
    status = iree_status_join(status, loom_serve_finite_load_free(load));
  }
  iree_notification_deinitialize(&notification);
  return status;
}

uint64_t loom_serve_finite_model_batch_limit(
    const loom_serve_finite_model_t* model) {
  return model->batch_limit;
}

iree_host_size_t loom_serve_finite_model_input_capacity(
    const loom_serve_finite_model_t* model) {
  return model->input_capacity;
}

const iree_tokenizer_t* loom_serve_finite_model_tokenizer(
    const loom_serve_finite_model_t* model) {
  return model->tokenizer;
}

loom_serve_residency_t* loom_serve_finite_model_residency(
    const loom_serve_finite_model_t* model) {
  return model->residency;
}

iree_status_t loom_serve_finite_model_prepare_request(
    loom_serve_finite_model_t* model, iree_vm_buffer_t* input,
    loom_serve_finite_prepared_t* out_prepared) {
  memset(out_prepared, 0, sizeof(*out_prepared));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               input)};
  iree_vm_variant_t results[3] = {0};
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                     model->control.prepare_request,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  int64_t units = 0, tag = 0;
  iree_vm_buffer_t* bytes = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[0], &bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &units);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[2], &tag);
  }
  if (iree_status_is_ok(status) &&
      (!bytes || units <= 0 || (uint64_t)units > model->batch_limit)) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "source request must fit declared batch capacity");
  }
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_retain(bytes);
    *out_prepared =
        (loom_serve_finite_prepared_t){bytes, (uint64_t)units, (uint64_t)tag};
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

void loom_serve_finite_model_release_batch(loom_serve_finite_model_t* model) {
  loom_serve_executables_release(model->executables);
  iree_vm_buffer_release(model->invocation.payload);
  loom_serve_finite_result_release(model->invocation.result);
  memset(&model->invocation, 0, sizeof(model->invocation));
}

iree_status_t loom_serve_finite_model_prepare_batch(
    loom_serve_finite_model_t* model, iree_host_size_t count,
    const loom_serve_finite_prepared_t* requests,
    const iree_host_size_t* expected_rows) {
  iree_host_size_t length = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t size = iree_vm_buffer_length(requests[i].bytes);
    if (size > SIZE_MAX - length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "prepared cohort exceeds host address space");
    }
    length += size;
  }
  iree_vm_buffer_t* ranges = NULL;
  iree_vm_buffer_t* packed = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(count * 16, 8, model->host_allocator, &ranges));
  loom_serve_executables_capture(model->executables);
  loom_serve_preparation_module_begin(model->control.preparation,
                                      model->stages.count, 0);
  iree_status_t status =
      iree_vm_buffer_create(length, 8, model->host_allocator, &packed);
  iree_byte_span_t records = iree_byte_span_empty();
  iree_byte_span_t target = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(ranges, 0, count * 16, &records);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(packed, 0, length, &target);
  }
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_const_byte_span_t input;
    status = iree_vm_buffer_map_read(
        requests[i].bytes, 0, iree_vm_buffer_length(requests[i].bytes), &input);
    if (iree_status_is_ok(status)) {
      iree_unaligned_store_le_u64(records.data + i * 16, offset);
      iree_unaligned_store_le_u64(records.data + i * 16 + 8, input.data_length);
      memcpy(target.data + offset, input.data, input.data_length);
      offset += input.data_length;
    }
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &ranges),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &packed)};
  iree_vm_variant_t results[3] = {0};
  iree_vm_buffer_t* plan = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                       model->control.prepare_batch,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
  }
  int32_t stage = 0;
  loom_serve_preparation_t* declarations =
      loom_serve_preparation_module_take(model->control.preparation);
  const iree_host_size_t stage_count =
      model->stages.count +
      (declarations ? loom_serve_preparation_stage_count(declarations) : 0);
  loom_serve_executables_declare(model->executables, declarations);
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &stage);
  }
  if (iree_status_is_ok(status) &&
      (stage < 0 || (uint32_t)stage >= stage_count)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source selected an undeclared command");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[1], &model->invocation.payload);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[2], &plan);
  }
  if (iree_status_is_ok(status) &&
      (!model->invocation.payload || !plan ||
       !iree_vm_buffer_length(model->invocation.payload) ||
       iree_vm_buffer_length(model->invocation.payload) >
           model->input_capacity)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source batch exceeds declared input storage");
  }
  if (iree_status_is_ok(status)) {
    model->invocation.stage = (uint32_t)stage;
    status = loom_serve_finite_result_create(
        plan, model->output_capacity, model->result_capacity, count,
        expected_rows, model->host_allocator, &model->invocation.result);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (!iree_status_is_ok(status)) {
    loom_serve_finite_model_release_batch(model);
  }
  return status;
}

void loom_serve_finite_model_set_observer(loom_serve_finite_model_t* model,
                                          loom_serve_wake_callback_t observer) {
  loom_serve_executables_set_observer(model->executables, observer);
}

iree_status_t loom_serve_finite_model_advance_batch(
    loom_serve_finite_model_t* model, bool cancel, bool* out_complete) {
  return loom_serve_executables_advance(model->executables, cancel,
                                        out_complete);
}

iree_status_t loom_serve_finite_model_submit(
    loom_serve_finite_model_t* model, loom_serve_finite_result_t** out_result,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_result = NULL;
  *out_frontier = (loom_serve_execution_frontier_t){0};
  iree_const_byte_span_t input;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      model->invocation.payload, 0,
      iree_vm_buffer_length(model->invocation.payload), &input));
  loom_serve_execution_t* execution = model->execution;
  const iree_host_size_t output_length =
      loom_serve_finite_result_output_length(model->invocation.result);
  // Admission ceilings bound source plans, not physical slab geometry. Register
  // only this cohort's extents before allocating either buffer; command-private
  // workspace is registered separately from its reflected requirements.
  IREE_RETURN_IF_ERROR(loom_serve_execution_reserve_workspace(
      execution, iree_max(input.data_length, output_length),
      LOOM_SERVE_FINITE_BUFFER_ALIGNMENT));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    const iree_hal_pool_reservation_request_t request = {
        .allocation_size = i ? output_length : input.data_length,
        .min_alignment = LOOM_SERVE_FINITE_BUFFER_ALIGNMENT,
    };
    status =
        loom_serve_execution_alloca(execution, &request, &model->buffers[i]);
  }
  uint64_t value = 0;
  if (iree_status_is_ok(status)) {
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload = {.source = input.data,
                   .target_buffer = model->buffers[0],
                   .length = input.data_length}};
    status = loom_serve_execution_transfer(execution, 1, &upload, &value);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32((int32_t)model->invocation.stage),
        iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                  model->buffers[0]),
        iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                  model->buffers[1])};
    status = iree_vm_invoke(
        loom_serve_program_invocation(model->control.program),
        model->control.run, iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_empty());
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_execution_dealloca(execution, model->buffers[0], &value);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_finite_result_bind(model->invocation.result, execution,
                                  model->buffers[1]);
    *out_result = model->invocation.result;
    model->invocation.result = NULL;
    *out_frontier = loom_serve_finite_result_ready(*out_result);
  }
  // Producer captures retain tracked views through actual queue retirement.
  // The result owns the raw output root independently, so its later consumers
  // cannot keep source parameters or command programs alive.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_hal_buffer_release(model->buffers[i]);
    model->buffers[i] = NULL;
  }
  return status;
}

void loom_serve_finite_model_complete(loom_serve_finite_model_t* model) {
  loom_serve_residency_release(model->residency);
  loom_serve_finite_model_release_batch(model);
}
