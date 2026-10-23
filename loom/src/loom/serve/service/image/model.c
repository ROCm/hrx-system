// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/model.h"

#include "iree/tooling/device_profiling.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/bank.h"
#include "loom/serve/runtime/parameter/module.h"
#include "loom/serve/runtime/parameter/sources.h"
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

struct loom_serve_image_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t host_allocator;
  // Copied startup evidence, independent of every compiler result lifetime.
  loom_serve_compilation_report_t compilation;
  // Owned public identifier returned by the source bootstrap.
  iree_string_view_t name;
  // Maximum opaque request byte length returned by the source bootstrap.
  iree_host_size_t input_capacity;
  // Positive additive source cost admitted in one image cohort.
  uint64_t batch_limit;
  // Soft per-submission command target, rounded to full barrier waves.
  uint32_t command_quantum;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Borrowed shared device/backing owner, outliving the model.
  loom_serve_device_t* owner;
  // Owned model work and feedback timelines, independent of other models.
  loom_serve_execution_t* execution;
  // Owned checkpoint roots and plans, outliving borrowing invocation
  // residencies.
  loom_serve_parameter_bank_t* parameters;
  // Atomic invocation working set and independent admission interest.
  loom_serve_residency_t* residency;
  // Idle request-code policy, independent of source-owned geometry options.
  loom_serve_variant_cache_options_t specialization_cache;
  // One invocation spanning CPU preparation, device submission and completion.
  struct {
    // Upload bytes retained through success or terminal queue retirement.
    iree_vm_buffer_t* payload;
    // Retained command selected by source preparation.
    uint32_t stage;
    // Source-owned next-part cursor; negative denotes terminal output.
    int32_t cursor;
    // Atomically acquired input, output, and independently releasable
    // workspace.
    iree_hal_pool_reservation_request_t storage[3];
    // Exact opaque input retained across every part, released after final work.
    iree_hal_buffer_t* input;
    // Compiler live values preserved across parts, released after final work.
    iree_hal_buffer_t* workspace;
    // Independent result retained through the final feedback submission.
    iree_hal_buffer_t* output;
    // Upload and allocation have been accepted for this invocation.
    bool started;
  } invocation;
  // Source compiler borrowing the host's shared task queue.
  loom_serve_jit_t* jit;
  // Owned source declarations, independent of the bootstrap VM and arguments.
  loom_serve_preparation_t* preparation;
  // Owned immutable indexes shared by resident and demand-read consumers.
  loom_serve_parameter_sources_t* parameter_sources;
  // Shared source-generation and request-code ownership.
  loom_serve_executables_t* executables;
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
    // Retained native runner whose command table changes without VM relinking.
    iree_vm_module_t* runner;
    // Scoped request declaration capture, including checkpoint bindings.
    iree_vm_module_t* preparation;
    // Retained demand-read capability, rebound with each request's sources.
    iree_vm_module_t* parameters;
    // Opaque source-owned state, retained across serialized requests.
    iree_vm_buffer_t* state;
    // Cold-resolved request preparation function.
    iree_vm_function_t prepare_request;
    // Cold-resolved cohort packing and executable declaration function.
    iree_vm_function_t prepare_batch;
    // Cold-resolved image-submission and feedback function.
    iree_vm_function_t generate;
  } control;
  // Exact-size serialized invocation feedback, reused until the next prepare.
  iree_byte_span_t output;
};

static void loom_serve_image_model_release_buffers(
    loom_serve_image_model_t* model) {
  iree_hal_buffer_release(model->invocation.input);
  model->invocation.input = NULL;
  iree_hal_buffer_release(model->invocation.workspace);
  model->invocation.workspace = NULL;
  iree_hal_buffer_release(model->invocation.output);
  model->invocation.output = NULL;
}

static void loom_serve_image_model_release_consumers(
    loom_serve_image_model_t* model) {
  loom_serve_executables_retire(model->executables);
  loom_serve_program_free(model->control.program);
  model->control.program = NULL;
  iree_vm_module_release(model->control.runner);
  model->control.runner = NULL;
  iree_vm_module_release(model->control.preparation);
  model->control.preparation = NULL;
  iree_vm_module_release(model->control.parameters);
  model->control.parameters = NULL;
  loom_serve_stage_table_free(model->stages.table);
  model->stages.table = NULL;
  loom_serve_image_model_release_buffers(model);
}

void loom_serve_image_model_retire(loom_serve_image_model_t* model,
                                   void (*wake)(void* user_data),
                                   void* user_data) {
  if (model->executables) {
    loom_serve_executables_set_observer(
        model->executables, (loom_serve_wake_callback_t){wake, user_data});
  }
  if (model->execution) {
    loom_serve_execution_set_retirement_observer(model->execution, wake,
                                                 user_data);
  }
  loom_serve_image_model_release_consumers(model);
  loom_serve_parameter_bank_retire(model->parameters, wake, user_data);
}

bool loom_serve_image_model_is_retired(loom_serve_image_model_t* model) {
  if (!model) {
    return true;
  }
  bool retired =
      !model->execution || loom_serve_execution_is_retired(model->execution);
  retired &= loom_serve_parameter_bank_is_retired(model->parameters);
  retired &= loom_serve_executables_is_retired(model->executables);
  return retired;
}

// Consumes a retired model or a load that never submitted device work. A failed
// platform release preserves the model and remaining reservation handles.
static iree_status_t loom_serve_image_model_destroy_resources(
    loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_image_model_release_consumers(model);
  loom_serve_parameter_bank_retire(model->parameters, NULL, NULL);
  IREE_ASSERT(loom_serve_image_model_is_retired(model));
  IREE_RETURN_IF_ERROR(loom_serve_executables_free(model->executables));
  model->executables = NULL;
  loom_serve_residency_free(model->residency);
  model->residency = NULL;
  iree_status_t status = loom_serve_parameter_bank_free(model->parameters);
  if (iree_status_is_ok(status)) {
    model->parameters = NULL;
    iree_vm_buffer_release(model->control.tags);
    iree_vm_buffer_release(model->control.state);
    for (uint32_t i = 0; i < model->stages.count; ++i) {
      loom_serve_jit_stage_free(model->stages.values[i]);
    }
    iree_allocator_free(model->host_allocator, model->stages.values);
    loom_serve_preparation_free(model->preparation);
    loom_serve_parameter_sources_release(model->parameter_sources);
    iree_vm_buffer_release(model->invocation.payload);
    iree_vm_environment_free(model->control.environment);
    loom_serve_execution_release(model->execution);
    iree_allocator_free(model->host_allocator, (void*)model->name.data);
    loom_serve_jit_release(model->jit);
    iree_tokenizer_free(model->tokenizer);
    iree_allocator_free(model->host_allocator, model->output.data);
    iree_allocator_free(model->host_allocator, model);
  }
  return status;
}

iree_status_t loom_serve_image_model_free_retired(
    loom_serve_image_model_t* model) {
  return loom_serve_image_model_destroy_resources(model);
}

iree_status_t loom_serve_image_model_free(loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_image_model_retire(model, NULL, NULL);
  iree_status_t status = loom_serve_execution_drain(model->execution);
  status = iree_status_join(
      status, loom_serve_parameter_bank_await_retirement(model->parameters));
  status = iree_status_join(
      status, loom_serve_executables_await_retirement(model->executables));
  return iree_status_join(status,
                          loom_serve_image_model_destroy_resources(model));
}

// Cold source results are consumed while their environment is live. Source
// state survives destruction of the cold program and is opaque to this owner.
static iree_status_t loom_serve_image_prepare(
    loom_serve_image_model_t* model,
    const loom_serve_image_model_options_t* options,
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
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64(options->height),
      iree_vm_variant_from_i64(options->width),
      iree_vm_variant_from_i64(options->text_tokens),
      {0},
      {0},
      {0}};
  const iree_string_view_t inputs[] = {
      options->checkpoint_directory, options->adapter_path,
      iree_string_view_is_empty(options->options) ? IREE_SV("{}")
                                                  : options->options};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_create(inputs[i].size, 8, host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      iree_byte_span_t bytes;
      status = iree_vm_buffer_map_write(buffer, 0, inputs[i].size, &bytes);
      if (iree_status_is_ok(status) && inputs[i].size) {
        memcpy(bytes.data, inputs[i].data, inputs[i].size);
      }
      arguments[3 + i] = iree_vm_buffer_variant_from_ptr_move(
          &model->control.vm_types, &buffer);
    }
  }
  iree_vm_variant_t results[5] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(model->owner)),
        model->control.environment, &options->sources, options->sources.prepare,
        IREE_SV("prepare"), LOOM_SERVE_PREPARATION_FLAG_PARAMETERS,
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){modules, IREE_ARRAYSIZE(modules)},
        &model->compilation, host_allocator, &model->preparation);
  }
  int64_t capacity = 0;
  int64_t batch_limit = 0;
  iree_vm_buffer_t* metadata[2] = {0};
  iree_const_byte_span_t bytes[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &batch_limit);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &capacity);
  }
  if (iree_status_is_ok(status) &&
      (batch_limit <= 0 || capacity <= 0 || (uint64_t)capacity > SIZE_MAX)) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "source requires positive batch and input capacities");
  }
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(
        &model->control.vm_types, results[i + 2], &metadata[i]);
    if (iree_status_is_ok(status) && !metadata[i]) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source image metadata must not be null");
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_buffer_map_read(
          metadata[i], 0, iree_vm_buffer_length(metadata[i]), &bytes[i]);
    }
  }
  if (iree_status_is_ok(status) &&
      (!bytes[0].data_length || !bytes[1].data_length)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source requires tokenizer asset and model name");
  }
  if (iree_status_is_ok(status)) {
    char* name = NULL;
    status = iree_allocator_clone(host_allocator, bytes[1], (void**)&name);
    model->name = iree_make_string_view(name, bytes[1].data_length);
    model->input_capacity = (iree_host_size_t)capacity;
    model->batch_limit = (uint64_t)batch_limit;
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[4], &model->control.state);
  }
  if (iree_status_is_ok(status) && !model->control.state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source image state must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_tokenizer_load_start(
        loom_serve_host_compiler_sink(loom_serve_device_host(model->owner)),
        &options->sources, options->checkpoint_directory,
        iree_make_string_view((const char*)bytes[0].data, bytes[0].data_length),
        wake, user_data, host_allocator, out_tokenizer_load);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(modules); ++i) {
    iree_vm_module_release(modules[i]);
  }
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
static iree_status_t loom_serve_image_check_layouts(
    loom_serve_image_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0]);
  const loom_serve_stage_record_options_t recording = {
      .binding_count = 2, .workspace = LOOM_SERVE_STAGE_WORKSPACE_INVOCATION};
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

static iree_status_t loom_serve_image_create_control(
    loom_serve_image_model_t* model, iree_vm_module_t* module) {
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
      .binding_capacity = 3,
      .stages = {model->stages.table->part_count, model->stages.table->parts},
      .feedback = {1, &model->output},
  };
  iree_vm_module_t* native_modules[5] = {0};
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
        model->control.environment,
        loom_serve_preparation_card(model->preparation),
        loom_serve_preparation_component(model->preparation), host_allocator,
        &native_modules[3]);
    model->control.preparation = native_modules[3];
    iree_vm_module_retain(model->control.preparation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_parameters_module_create(
        model->control.environment, &model->control.hal_types,
        loom_serve_device_transfer_queue(model->owner),
        model->parameter_sources, host_allocator, &native_modules[4]);
    model->control.parameters = native_modules[4];
    iree_vm_module_retain(model->control.parameters);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link_service(
        module, loom_serve_preparation_card(model->preparation),
        loom_serve_preparation_component(model->preparation),
        (iree_vm_module_span_t){native_modules, 5}, host_allocator,
        &model->control.program);
  }
  iree_vm_module_release(native_modules[0]);
  iree_vm_module_release(native_modules[1]);
  iree_vm_module_release(native_modules[2]);
  iree_vm_module_release(native_modules[3]);
  iree_vm_module_release(native_modules[4]);
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
  const iree_vm_signature_type_t request_arguments[] = {buffer, i64, i64,
                                                        buffer, i64, buffer};
  const iree_vm_signature_type_t request_results[] = {buffer, i64, i64};
  const iree_vm_signature_type_t batch_arguments[] = {buffer, buffer, buffer,
                                                      buffer};
  const iree_vm_signature_type_t batch_results[] = {i32, buffer};
  const iree_vm_signature_type_t generate_arguments[] = {
      i32, i32, i32, i64, device_buffer, device_buffer, device_buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("prepare_request"),
       {request_arguments, IREE_ARRAYSIZE(request_arguments)},
       {request_results, IREE_ARRAYSIZE(request_results)},
       &model->control.prepare_request},
      {IREE_SVL("prepare_batch"),
       {batch_arguments, IREE_ARRAYSIZE(batch_arguments)},
       {batch_results, IREE_ARRAYSIZE(batch_results)},
       &model->control.prepare_batch},
      {IREE_SVL("generate"),
       {generate_arguments, IREE_ARRAYSIZE(generate_arguments)},
       {&i32, 1},
       &model->control.generate},
  };
  return loom_serve_program_bind(model->control.program,
                                 IREE_ARRAYSIZE(bindings), bindings);
}

static iree_status_t loom_serve_image_model_bind(
    loom_serve_image_model_t* model) {
  IREE_RETURN_IF_ERROR(loom_serve_image_check_layouts(model));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create_execution(model->owner, &model->execution));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0]);
  return loom_serve_parameter_bank_allocate(
      model->owner, program, model->host_allocator, &model->parameters);
}

static iree_status_t loom_serve_image_model_link(
    loom_serve_image_model_t* model, iree_vm_module_t* module,
    const loom_serve_sources_t* sources,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t host_allocator = model->host_allocator;
  loom_serve_residency_group_t* group =
      loom_serve_parameter_bank_group(model->parameters);
  IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(model->owner), model->execution,
      group ? 1 : 0, &group, host_allocator, &model->residency));
  const loom_serve_executable_options_t options = {
      .device = model->owner,
      .jit = model->jit,
      .sources = sources,
      .execution = model->execution,
      .residency = model->residency,
      .preparation = model->preparation,
      .parameter_sources = model->parameter_sources,
      .parameters = model->parameters,
      .recording = {.queue_family = iree_hal_queue_family(
                        loom_serve_device_dispatch_queue(model->owner)),
                    .parameters =
                        loom_serve_parameter_bank_buffers(model->parameters),
                    .command_mode = command_mode,
                    .binding_count = 2,
                    .workspace = LOOM_SERVE_STAGE_WORKSPACE_INVOCATION,
                    .command_quantum = model->command_quantum},
      .cache = model->specialization_cache,
      .parameter_cache = {16, 1024 * 1024 * 1024}};
  IREE_RETURN_IF_ERROR(loom_serve_stage_table_allocate(
      &options.recording, NULL, model->stages.count, model->stages.values,
      host_allocator, &model->stages.table));
  IREE_RETURN_IF_ERROR(loom_serve_image_create_control(model, module));
  loom_serve_executable_options_t bound_options = options;
  bound_options.stages = model->stages.table;
  bound_options.runner = model->control.runner;
  bound_options.parameter_module = model->control.parameters;
  return loom_serve_executables_allocate(&bound_options, host_allocator,
                                         &model->executables);
}

typedef enum loom_serve_image_load_phase_e {
  IMAGE_LOAD_PREPARE,
  IMAGE_LOAD_STAGES,
  IMAGE_LOAD_BIND,
  IMAGE_LOAD_WEIGHTS,
  IMAGE_LOAD_LINK,
  IMAGE_LOAD_COMPLETE,
} loom_serve_image_load_phase_t;

struct loom_serve_image_load_t {
  // One private bootstrap task, retired before any native compilation starts.
  loomc_task_t task;
  // Allocator owning this operation, independent of the transferred model.
  iree_allocator_t host_allocator;
  // Owned partial model, transferred only after successful terminal advance.
  loom_serve_image_model_t* model;
  // Borrowed immutable package inputs, retained by the registration owner.
  loom_serve_image_model_options_t options;
  // Owner-only next phase; callbacks never make shared-state transitions.
  loom_serve_image_load_phase_t phase;
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
  // Terminal outcome retained until tokenizer and backing ownership retire.
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
    loom_serve_image_load_wake_fn_t wake;
    // Borrowed notification payload, retained through terminal advance.
    void* user_data;
  } event;
};

static void loom_serve_image_load_complete(void* user_data,
                                           iree_status_t status) {
  loom_serve_image_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t loom_serve_image_load_prepare(
    loom_serve_image_load_t* load) {
  loom_serve_image_model_t* model = load->model;
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(
      loom_serve_image_prepare(model, &load->options, load->event.wake,
                               load->event.user_data, &load->tokenizer_load));
  for (iree_host_size_t i = 0; i < model->stages.count; ++i) {
    if (loom_serve_preparation_stage(model->preparation, i)->kind !=
        LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "image stages require command parameter reflection");
    }
  }
  const loom_model_file_list_t* paths = NULL;
  const uint32_t source_count =
      loom_serve_preparation_local_sources(model->preparation, &paths);
  IREE_RETURN_IF_ERROR(loom_serve_parameter_sources_create(
      NULL, &load->options.sources, source_count, paths, host_allocator,
      &model->parameter_sources));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  const iree_string_view_t roots[] = {IREE_SVL("prepare_request"),
                                      IREE_SVL("prepare_batch"),
                                      IREE_SVL("generate")};
  iree_status_t status = loom_serve_program_compile_service(
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

static void loom_serve_image_load_execute(loomc_task_t* task,
                                          loomc_host_size_t worker_ordinal) {
  loom_serve_image_load_t* load = (loom_serve_image_load_t*)task;
  load->task_status = loom_serve_image_load_prepare(load);
}

static void loom_serve_image_load_release(loomc_task_t* task) {
  loom_serve_image_load_t* load = (loom_serve_image_load_t*)task;
  loom_serve_image_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t image_load_vtable = {
    .execute = loom_serve_image_load_execute,
    .destroy = loom_serve_image_load_release,
};

static void loom_serve_image_load_compiled(void* user_data,
                                           iree_status_t status,
                                           loom_serve_stage_set_t* compiled) {
  loom_serve_image_load_t* load = user_data;
  for (iree_host_size_t i = 0; i < compiled->count; ++i) {
    load->model->stages.values[i] = compiled->values[i];
    compiled->values[i] = NULL;
  }
  loom_serve_stage_set_free(compiled);
  loom_serve_image_load_complete(load, status);
}

iree_status_t loom_serve_image_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    loom_serve_image_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_image_load_t** out_load) {
  *out_load = NULL;
  if (!options->height || !options->width ||
      (uint64_t)options->height * options->width > SIZE_MAX / 12) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "output RGB geometry must fit host storage");
  }
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  loom_serve_image_load_t* load = NULL;
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
  loomc_task_initialize(&image_load_vtable, &load->task);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*load->model), (void**)&load->model);
  if (iree_status_is_ok(status)) {
    load->model->host_allocator = host_allocator;
    load->model->owner = device;
    load->model->command_quantum = options->command_quantum;
    load->model->specialization_cache = options->specialization_cache;
    *out_load = load;
  } else {
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(host_allocator, load);
  }
  return status;
}

iree_status_t loom_serve_image_load_advance(loom_serve_image_load_t* load,
                                            loom_serve_image_load_flags_t flags,
                                            bool* out_complete) {
  *out_complete = false;
  load->cancelled |= iree_any_bit_set(flags, LOOM_SERVE_IMAGE_LOAD_FLAG_CANCEL);
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
  loom_serve_image_model_t* model = load->model;
  loom_serve_host_t* host = loom_serve_device_host(model->owner);
  bool submitted = false;
  while (iree_status_is_ok(status) && !submitted &&
         load->phase != IMAGE_LOAD_COMPLETE) {
    switch (load->phase) {
      case IMAGE_LOAD_PREPARE:
        load->phase = IMAGE_LOAD_STAGES;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = iree_status_from_loomc(loomc_task_sink_submit(
            loom_serve_host_compiler_sink(host), &load->task));
        submitted = iree_status_is_ok(status);
        break;
      case IMAGE_LOAD_STAGES:
        load->phase = IMAGE_LOAD_BIND;
        iree_slim_mutex_lock(&load->event.mutex);
        load->event.pending = true;
        iree_slim_mutex_unlock(&load->event.mutex);
        status = loom_serve_stage_set_compile_catalog_async(
            model->jit, model->preparation, NULL,
            loom_serve_image_load_compiled, load, load->host_allocator);
        submitted = iree_status_is_ok(status);
        break;
      case IMAGE_LOAD_BIND:
        status = loom_serve_image_model_bind(model);
        load->phase = IMAGE_LOAD_WEIGHTS;
        break;
      case IMAGE_LOAD_WEIGHTS:
        if (load->index < loom_serve_parameter_bank_count(model->parameters)) {
          const uint32_t i = load->index++;
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, 0);
          const loom_serve_preparation_parameter_t* parameter =
              &declaration->parameters[i];
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_weights_prepare_async(
              loom_serve_parameter_bank_weights(model->parameters, i), host,
              model->jit, load->command_mode, &load->options.sources, parameter,
              model->parameter_sources, loom_serve_image_load_complete, load);
          submitted = iree_status_is_ok(status);
        } else {
          load->phase = IMAGE_LOAD_LINK;
        }
        break;
      case IMAGE_LOAD_LINK:
        if (!loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                            &model->tokenizer)) {
          return iree_ok_status();
        }
        if (iree_status_is_ok(status)) {
          status = loom_serve_image_model_link(
              model, load->module, &load->options.sources, load->command_mode);
        }
        load->phase = IMAGE_LOAD_COMPLETE;
        break;
      case IMAGE_LOAD_COMPLETE:
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Rejection owes no callback; a consumed completion has already retired.
    iree_slim_mutex_lock(&load->event.mutex);
    load->event.pending = false;
    iree_slim_mutex_unlock(&load->event.mutex);
    load->phase = IMAGE_LOAD_COMPLETE;
  }
  if (load->phase == IMAGE_LOAD_COMPLETE &&
      !loom_serve_tokenizer_load_take(&load->tokenizer_load, &status,
                                      &model->tokenizer)) {
    load->terminal_status = status;
    return iree_ok_status();
  }
  if (!iree_status_is_ok(status)) {
    // Binding may already have queued backing allocation. Failure of a later
    // preparation step does not retire that accepted ownership.
    loom_serve_image_model_retire(model, load->event.wake,
                                  load->event.user_data);
    if (!loom_serve_image_model_is_retired(model)) {
      load->terminal_status = status;
      return iree_ok_status();
    }
  }
  *out_complete = load->phase == IMAGE_LOAD_COMPLETE;
  return status;
}

const loom_serve_compilation_report_t* loom_serve_image_load_report(
    const loom_serve_image_load_t* load) {
  return &load->model->compilation;
}

loom_serve_image_model_t* loom_serve_image_load_take_model(
    loom_serve_image_load_t* load) {
  loom_serve_image_model_t* model = load->model;
  load->model = NULL;
  return model;
}

iree_status_t loom_serve_image_load_free(loom_serve_image_load_t* load) {
  if (!load) {
    return iree_ok_status();
  }
  iree_status_t status = loom_serve_image_model_destroy_resources(load->model);
  if (iree_status_is_ok(status)) {
    iree_vm_module_release(load->module);
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(load->host_allocator, load);
  }
  return status;
}

static void loom_serve_image_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_image_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_image_model_t** out_model) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_image_load_t* load = NULL;
  iree_status_t status = loom_serve_image_load_allocate(
      device, options, loom_serve_image_load_notify, &notification,
      host_allocator, &load);
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_image_load_advance(load, 0, &complete);
    if (complete) {
      iree_notification_cancel_wait(&notification);
    } else {
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  if (load) {
    if (iree_status_is_ok(status)) {
      *out_model = loom_serve_image_load_take_model(load);
    }
    status = iree_status_join(status, loom_serve_image_load_free(load));
  }
  iree_notification_deinitialize(&notification);
  return status;
}

iree_string_view_t loom_serve_image_model_name(
    const loom_serve_image_model_t* model) {
  return model->name;
}

iree_status_t loom_serve_image_model_prepare_request(
    loom_serve_image_model_t* model, const loom_serve_image_request_t* request,
    loom_serve_prepared_request_t* out_prepared) {
  *out_prepared = (loom_serve_prepared_request_t){0};
  const iree_string_view_t prompt = iree_string_builder_view(&request->prompt);
  const iree_string_view_t selections =
      iree_string_builder_view(&request->loras);
  iree_vm_buffer_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)prompt.data, prompt.size),
      iree_vm_buffer_release_callback_null(), model->host_allocator, &text));
  iree_vm_buffer_t* loras = NULL;
  iree_status_t status = iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)selections.data, selections.size),
      iree_vm_buffer_release_callback_null(), model->host_allocator, &loras);
  if (!iree_status_is_ok(status)) {
    iree_vm_buffer_release(text);
    return status;
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_variant_from_i64(request->height),
      iree_vm_variant_from_i64(request->width),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &text),
      iree_vm_variant_from_i64((int64_t)request->seed),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &loras),
  };
  iree_vm_variant_t results[3] = {0};
  status = iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                          model->control.prepare_request,
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results));
  iree_vm_buffer_t* bytes = NULL;
  int64_t units = 0;
  int64_t tag = 0;
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
        (loom_serve_prepared_request_t){bytes, (uint64_t)units, (uint64_t)tag};
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

static iree_status_t loom_serve_image_prepare_batch(
    loom_serve_image_model_t* model, iree_host_size_t count,
    const loom_serve_prepared_request_t* requests, int64_t* out_tag) {
  iree_vm_buffer_t* ranges = NULL;
  iree_vm_buffer_t* packed = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_prepared_requests_pack(
      count, requests, model->host_allocator, &ranges, &packed));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &ranges),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &packed)};
  iree_vm_variant_t results[2] = {0};
  loom_serve_preparation_module_begin(model->control.preparation,
                                      model->stages.count, model->preparation,
                                      LOOM_SERVE_PREPARATION_FLAG_PARAMETERS);
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.prepare_batch, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  loom_serve_preparation_t* declarations =
      loom_serve_preparation_module_take(model->control.preparation);
  const iree_host_size_t stage_count =
      model->stages.count +
      (declarations ? loom_serve_preparation_stage_count(declarations) : 0);
  loom_serve_executables_declare(model->executables, declarations);
  int32_t selected = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &selected);
  }
  if (iree_status_is_ok(status) &&
      (selected < 0 || (uint32_t)selected >= stage_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "model selected undeclared stage %d", selected);
  }
  iree_vm_buffer_t* payload = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&model->control.vm_types,
                                                      results[1], &payload);
  }
  if (iree_status_is_ok(status) &&
      (!payload || !iree_vm_buffer_length(payload) ||
       iree_vm_buffer_length(payload) > model->input_capacity)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source request exceeds declared storage");
  }
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_retain(payload);
    model->invocation.payload = payload;
    model->invocation.stage = (uint32_t)selected;
    *out_tag =
        (uint32_t)selected < model->stages.count
            ? loom_serve_preparation_stage(model->preparation, selected)->tag
            : loom_serve_preparation_stage(
                  declarations, (uint32_t)selected - model->stages.count)
                  ->tag;
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

static iree_status_t loom_serve_image_model_submit_stage(
    loom_serve_image_model_t* model) {
  const loom_serve_stage_launch_t* stage =
      loom_serve_executables_stage(model->executables, model->invocation.stage);
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32((int32_t)stage->first_part),
      iree_vm_variant_from_i32((int32_t)stage->part_count),
      iree_vm_variant_from_i32(model->invocation.cursor),
      iree_vm_variant_from_i64((int64_t)model->output.data_length),
      iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                model->invocation.input),
      iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                model->invocation.output),
      iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                model->invocation.workspace),
  };
  iree_vm_variant_t result = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.generate, iree_vm_variant_span_from_array(arguments),
      (iree_vm_variant_span_t){&result, 1});
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(result, &model->invocation.cursor);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset((iree_vm_variant_span_t){&result, 1});
  return status;
}

iree_status_t loom_serve_image_model_activate(loom_serve_image_model_t* model) {
  return loom_serve_residency_activate(model->residency);
}

iree_status_t loom_serve_image_model_deactivate(loom_serve_image_model_t* model,
                                                bool* out_complete) {
  return loom_serve_executables_deactivate(model->executables, out_complete);
}

loom_serve_residency_t* loom_serve_image_model_residency(
    const loom_serve_image_model_t* model) {
  return loom_serve_executables_residency(model->executables);
}

loom_serve_residency_snapshot_t loom_serve_image_model_parameters(
    const loom_serve_image_model_t* model) {
  return loom_serve_executables_parameters(model->executables);
}

void loom_serve_image_model_set_pinned(loom_serve_image_model_t* model,
                                       bool pinned) {
  loom_serve_executables_set_pinned(model->executables, pinned);
}

iree_status_t loom_serve_image_model_query_status(
    loom_serve_image_model_t* model) {
  return model ? loom_serve_executables_query_status(model->executables)
               : iree_ok_status();
}

bool loom_serve_image_model_is_pending(const loom_serve_image_model_t* model) {
  return model && loom_serve_executables_is_pending(model->executables);
}

loom_serve_memory_usage_t loom_serve_image_model_weight_statistics(
    const loom_serve_image_model_t* model) {
  return loom_serve_executables_parameters(model->executables).memory;
}

loom_serve_weight_activity_t loom_serve_image_model_weight_activity(
    const loom_serve_image_model_t* model) {
  return loom_serve_executables_parameters(model->executables).activity;
}

iree_status_t loom_serve_image_model_prepare_batch(
    loom_serve_image_model_t* model, iree_host_size_t count,
    const loom_serve_prepared_request_t* requests,
    const iree_host_size_t* output_lengths,
    loom_serve_image_prepared_t* out_prepared) {
  *out_prepared = (loom_serve_image_prepared_t){0};
  iree_host_size_t output_length = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (output_lengths[i] >
        iree_min(IREE_HOST_SIZE_MAX, INT64_MAX) - output_length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image cohort output exceeds address space");
    }
    output_length += output_lengths[i];
  }
  loom_serve_executables_capture(model->executables);
  int64_t tag = 0;
  iree_status_t status =
      loom_serve_image_prepare_batch(model, count, requests, &tag);
  if (iree_status_is_ok(status) && output_length != model->output.data_length) {
    status = iree_allocator_realloc(model->host_allocator, output_length,
                                    (void**)&model->output.data);
    if (iree_status_is_ok(status)) {
      model->output.data_length = output_length;
      loom_serve_module_bind_feedback(model->control.runner, 0, model->output);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_image_model_discard_batch(model);
    return status;
  }
  model->invocation.cursor = 0;
  model->invocation.started = false;
  *out_prepared = (loom_serve_image_prepared_t){
      .stage = model->invocation.stage,
      .tag = tag,
      .input_bytes = iree_vm_buffer_length(model->invocation.payload)};
  return iree_ok_status();
}

void loom_serve_image_model_discard_batch(loom_serve_image_model_t* model) {
  loom_serve_executables_release(model->executables);
  iree_vm_buffer_release(model->invocation.payload);
  model->invocation.payload = NULL;
}

void loom_serve_image_model_set_observer(loom_serve_image_model_t* model,
                                         loom_serve_wake_callback_t observer) {
  loom_serve_executables_set_observer(model->executables, observer);
}

iree_status_t loom_serve_image_model_advance_batch(
    loom_serve_image_model_t* model, bool cancel, bool* out_complete) {
  IREE_RETURN_IF_ERROR(
      loom_serve_executables_advance(model->executables, cancel, out_complete));
  if (!cancel && *out_complete) {
    const loom_serve_stage_launch_t* stage = loom_serve_executables_stage(
        model->executables, model->invocation.stage);
    const iree_device_size_t alignment =
        iree_max(stage->transient.minimum_alignment, 1);
    if (stage->transient.required_byte_length >
            IREE_DEVICE_SIZE_MAX - (alignment - 1) ||
        model->output.data_length > IREE_DEVICE_SIZE_MAX - (alignment - 1) -
                                        stage->transient.required_byte_length) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image frame exceeds device address space");
    }
    model->invocation.storage[1] = iree_hal_make_pool_reservation_request(
        model->output.data_length, alignment);
    model->invocation.storage[2] = iree_hal_make_pool_reservation_request(
        stage->transient.required_byte_length, alignment);
    const iree_device_size_t workspace_bytes =
        iree_device_align(model->output.data_length, alignment) +
        stage->transient.required_byte_length;
    const iree_device_size_t input_bytes =
        iree_vm_buffer_length(model->invocation.payload);
    if (input_bytes > IREE_DEVICE_SIZE_MAX - (alignment - 1) ||
        iree_device_align(input_bytes, alignment) >
            IREE_DEVICE_SIZE_MAX - workspace_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "image invocation exceeds device address space");
    }
    model->invocation.storage[0] =
        iree_hal_make_pool_reservation_request(input_bytes, alignment);
    const iree_device_size_t invocation_bytes =
        iree_device_align(input_bytes, alignment) + workspace_bytes;
    // The output never escapes the invocation: collection follows both work
    // and feedback retirement. Its storage and all split command parts share
    // one private high-water until that collection releases the admission pin.
    IREE_RETURN_IF_ERROR(loom_serve_residency_prepare_working_set(
        loom_serve_executables_residency(model->executables), invocation_bytes,
        0));
    // Register geometry for every member of this allocation transaction so
    // they can occupy one backing range without sharing their lifetimes.
    return loom_serve_execution_reserve_workspace(model->execution,
                                                  invocation_bytes, alignment);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_image_begin(loom_serve_image_model_t* model) {
  iree_vm_buffer_t* payload = model->invocation.payload;
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      payload, 0, iree_vm_buffer_length(payload), &input));
  const bool has_workspace = model->invocation.storage[2].allocation_size != 0;
  iree_hal_buffer_t* buffers[3] = {NULL, NULL, NULL};
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_alloca(model->execution, has_workspace ? 3 : 2,
                                  model->invocation.storage, buffers));
  model->invocation.input = buffers[0];
  model->invocation.output = buffers[1];
  model->invocation.workspace = buffers[2];
  if (!has_workspace) {
    // The invocation ABI still has a workspace binding when no live values
    // require storage. Preserve an empty view without allocating backing.
    IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(model->invocation.output, 0, 0,
                                                 model->host_allocator,
                                                 &model->invocation.workspace));
  }
  iree_hal_transfer_operation_t upload = {0};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input.data;
  upload.upload.target_buffer = model->invocation.input;
  upload.upload.length = input.data_length;
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_transfer(model->execution, 1, &upload, &completion));
  model->invocation.started = true;
  return iree_ok_status();
}

iree_status_t loom_serve_image_model_submit(
    loom_serve_image_model_t* model, loom_serve_image_submit_flags_t flags,
    loom_serve_execution_frontier_t* out_frontier, bool* out_complete) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  *out_complete = false;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, LOOM_SERVE_IMAGE_SUBMIT_FLAG_CANCEL)) {
    model->invocation.cursor = -1;
  } else {
    if (!model->invocation.started) {
      status = loom_serve_image_begin(model);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_image_model_submit_stage(model);
    }
  }
  if (iree_status_is_ok(status) && model->invocation.cursor < 0) {
    // Input and compiler live values follow the work timeline. Feedback owns
    // the independent output, so it cannot delay workspace reuse. Cancellation
    // between parts has the same last-work boundary and no output consumer.
    if (model->invocation.input) {
      uint64_t completion = 0;
      status = loom_serve_execution_dealloca(
          model->execution, model->invocation.input, &completion);
    }
    if (iree_status_is_ok(status) && model->invocation.workspace &&
        model->invocation.storage[2].allocation_size) {
      uint64_t completion = 0;
      status = loom_serve_execution_dealloca(
          model->execution, model->invocation.workspace, &completion);
    }
    loom_serve_image_model_release_buffers(model);
  }
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_serve_execution_frontier(model->execution);
    *out_complete = model->invocation.cursor < 0;
  }
  return status;
}

iree_const_byte_span_t loom_serve_image_model_complete(
    loom_serve_image_model_t* model) {
  iree_vm_buffer_release(model->invocation.payload);
  model->invocation.payload = NULL;
  loom_serve_residency_release(
      loom_serve_executables_residency(model->executables));
  loom_serve_executables_release(model->executables);
  return iree_make_const_byte_span(model->output.data,
                                   model->output.data_length);
}

iree_status_t loom_serve_image_model_generate(
    loom_serve_image_model_t* model, const loom_serve_image_request_t* request,
    loom_serve_image_prepared_t* out_prepared,
    iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  *out_prepared = (loom_serve_image_prepared_t){0};
  loom_serve_prepared_request_t prepared_request = {0};
  IREE_RETURN_IF_ERROR(loom_serve_image_model_prepare_request(
      model, request, &prepared_request));
  const iree_host_size_t output_length =
      (iree_host_size_t)request->width * request->height * 12;
  iree_status_t status = loom_serve_image_model_prepare_batch(
      model, 1, &prepared_request, &output_length, out_prepared);
  iree_vm_buffer_release(prepared_request.bytes);
  IREE_RETURN_IF_ERROR(status);
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_image_model_set_observer(
      model, (loom_serve_wake_callback_t){loom_serve_image_load_notify,
                                          &notification});
  bool resolved = false;
  while (iree_status_is_ok(status) && !resolved) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_image_model_advance_batch(model, false, &resolved);
    if (resolved || !iree_status_is_ok(status)) {
      iree_notification_cancel_wait(&notification);
    } else {
      iree_notification_commit_wait(&notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  loom_serve_image_model_set_observer(model, (loom_serve_wake_callback_t){0});
  iree_notification_deinitialize(&notification);
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_acquire_wait(
        loom_serve_executables_residency(model->executables));
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_image_model_discard_batch(model);
    return status;
  }
  bool complete = false;
  while (iree_status_is_ok(status) && !complete) {
    loom_serve_execution_frontier_t frontier;
    status = loom_serve_image_model_submit(model, 0, &frontier, &complete);
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, frontier.values[0]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback_wait(model->execution,
                                                  frontier.values[1]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_drain(model->execution);
  }
  if (iree_status_is_ok(status)) {
    *out_rgb = loom_serve_image_model_complete(model);
  }
  return status;
}

static iree_status_t loom_serve_image_generator_admit(
    void* self, loom_serve_residency_admission_t* out_admission) {
  loom_serve_residency_t* residency = loom_serve_image_model_residency(self);
  IREE_RETURN_IF_ERROR(
      loom_serve_residency_try_admit(residency, out_admission));
  if (*out_admission == LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED) {
    loom_serve_residency_cancel_admission(residency);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_image_generator_prepare(
    void* self, const loom_serve_image_request_t* request,
    loom_serve_prepared_request_t* out_prepared) {
  return loom_serve_image_model_prepare_request(self, request, out_prepared);
}

static iree_status_t loom_serve_image_generator_prepare_batch(
    void* self, iree_host_size_t count,
    const loom_serve_prepared_request_t* requests,
    const iree_host_size_t* output_lengths,
    loom_serve_image_prepared_t* out_prepared) {
  return loom_serve_image_model_prepare_batch(self, count, requests,
                                              output_lengths, out_prepared);
}

static void loom_serve_image_generator_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_image_model_set_observer(self, observer);
}

static iree_status_t loom_serve_image_generator_advance(void* self, bool cancel,
                                                        bool* out_complete) {
  return loom_serve_image_model_advance_batch(self, cancel, out_complete);
}

static iree_status_t loom_serve_image_generator_submit(
    void* self, loom_serve_image_submit_flags_t flags,
    loom_serve_execution_frontier_t* out_frontier, bool* out_complete) {
  return loom_serve_image_model_submit(self, flags, out_frontier, out_complete);
}

static void loom_serve_image_generator_discard(void* self) {
  loom_serve_residency_cancel_admission(loom_serve_image_model_residency(self));
  loom_serve_image_model_discard_batch(self);
}

static iree_const_byte_span_t loom_serve_image_generator_complete(void* self) {
  return loom_serve_image_model_complete(self);
}

static loom_serve_weight_activity_t loom_serve_image_generator_weight_activity(
    void* self) {
  return loom_serve_image_model_weight_activity(self);
}

static iree_status_t loom_serve_image_generator_query_status(void* self) {
  return loom_serve_image_model_query_status(self);
}

static bool loom_serve_image_generator_is_pending(void* self) {
  return loom_serve_image_model_is_pending(self);
}

loom_serve_image_generator_t loom_serve_image_model_generator(
    loom_serve_image_model_t* model) {
  return (loom_serve_image_generator_t){
      .self = model,
      .batch_limit = model->batch_limit,
      .prepare = loom_serve_image_generator_prepare,
      .prepare_batch = loom_serve_image_generator_prepare_batch,
      .set_observer = loom_serve_image_generator_set_observer,
      .advance_batch = loom_serve_image_generator_advance,
      .discard_batch = loom_serve_image_generator_discard,
      .try_admit = loom_serve_image_generator_admit,
      .submit = loom_serve_image_generator_submit,
      .complete = loom_serve_image_generator_complete,
      .weight_activity = loom_serve_image_generator_weight_activity,
      .query_status = loom_serve_image_generator_query_status,
      .is_pending = loom_serve_image_generator_is_pending,
      .execution_update = loom_serve_executables_update(model->executables),
      .tokenizer = model->tokenizer,
      .capacity_domain = loom_serve_residency_cache(model->residency)};
}
