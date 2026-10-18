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
#include "loom/serve/runtime/execution/retirement.h"
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
  // Soft per-submission command target, rounded to full barrier waves.
  uint32_t command_quantum;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Borrowed shared device/backing owner, outliving the model.
  loom_serve_device_t* owner;
  // Owned model work and feedback timelines, independent of other models.
  loom_serve_execution_t* execution;
  // Final input/output view ownership before host payload release.
  loom_serve_retirement_t retirement;
  // Shared-device admission group protecting every checkpoint domain together.
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
    // Exact request frame containing RGB and aligned command-private storage.
    iree_hal_pool_reservation_request_t storage;
    // Compiler-aligned workspace offset following the frame's RGB prefix.
    iree_device_size_t workspace_offset;
    // Frame view preserving compiler live values across all command parts.
    iree_hal_buffer_t* workspace;
    // Frame view retained through the final feedback submission.
    iree_hal_buffer_t* output;
    // Upload and allocation have been accepted for this invocation.
    bool started;
  } invocation;
  // Source compiler borrowing the host's shared task queue.
  loom_serve_jit_t* jit;
  // Owned source declarations, independent of the bootstrap VM and arguments.
  loom_serve_preparation_t* preparation;
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
    // Invocation-only declaration capture, with no parameter authority.
    iree_vm_module_t* preparation;
    // Opaque source-owned state, retained across serialized requests.
    iree_vm_buffer_t* state;
    // Cold-resolved request preparation function.
    iree_vm_function_t prepare_request;
    // Cold-resolved image-submission and feedback function.
    iree_vm_function_t generate;
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
  // Fixed-capacity opaque input; request output and workspace are queue-owned.
  iree_hal_buffer_t* input;
  // Exact-size serialized invocation feedback, reused until the next prepare.
  iree_byte_span_t output;
};

static void loom_serve_image_model_release_consumers(
    loom_serve_image_model_t* model) {
  loom_serve_executables_free(model->executables);
  model->executables = NULL;
  loom_serve_program_free(model->control.program);
  model->control.program = NULL;
  iree_vm_module_release(model->control.runner);
  model->control.runner = NULL;
  iree_vm_module_release(model->control.preparation);
  model->control.preparation = NULL;
  loom_serve_stage_table_free(model->stages.table);
  model->stages.table = NULL;
  iree_hal_buffer_release(model->invocation.workspace);
  model->invocation.workspace = NULL;
  iree_hal_buffer_release(model->invocation.output);
  model->invocation.output = NULL;
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    iree_hal_buffer_release(model->weights.values[i]);
    model->weights.values[i] = NULL;
  }
  iree_hal_buffer_release(model->input);
  model->input = NULL;
}

void loom_serve_image_model_retire(loom_serve_image_model_t* model,
                                   void (*wake)(void* user_data),
                                   void* user_data) {
  if (model->execution) {
    loom_serve_execution_set_retirement_observer(model->execution, wake,
                                                 user_data);
  }
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->weights.plans[i], wake,
                                               user_data);
  }
  loom_serve_retirement_set_observer(&model->retirement, wake, user_data);
  loom_serve_image_model_release_consumers(model);
}

bool loom_serve_image_model_is_retired(loom_serve_image_model_t* model) {
  if (!model) {
    return true;
  }
  bool retired =
      !model->execution || loom_serve_execution_is_retired(model->execution);
  for (iree_host_size_t i = 0; model->weights.plans && i < model->weights.count;
       ++i) {
    retired &= loom_serve_weights_is_retired(model->weights.plans[i]);
  }
  return retired && loom_serve_retirement_is_idle(&model->retirement);
}

// Consumes a retired model or a load that never submitted device work. A failed
// platform release preserves the model and remaining reservation handles.
static iree_status_t loom_serve_image_model_destroy_resources(
    loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_image_model_release_consumers(model);
  IREE_ASSERT(loom_serve_image_model_is_retired(model));
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
    loom_serve_retirement_deinitialize(&model->retirement);
    iree_vm_buffer_release(model->control.tags);
    iree_vm_buffer_release(model->control.state);
    for (uint32_t i = 0; i < model->stages.count; ++i) {
      loom_serve_jit_stage_free(model->stages.values[i]);
    }
    iree_allocator_free(model->host_allocator, model->stages.values);
    loom_serve_preparation_free(model->preparation);
    iree_vm_buffer_release(model->invocation.payload);
    iree_vm_environment_free(model->control.environment);
    loom_serve_execution_release(model->execution);
    iree_allocator_free(model->host_allocator, model->weights.values);
    iree_allocator_free(model->host_allocator, model->weights.plans);
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
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_await_retirement(model->weights.plans[i]);
  }
  loom_serve_retirement_await(&model->retirement);
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
  iree_vm_variant_t results[4] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_allocate(
        loom_serve_host_target_environment(
            loom_serve_device_host(model->owner)),
        model->control.environment, &options->sources, options->sources.prepare,
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){modules, IREE_ARRAYSIZE(modules)},
        &model->compilation, host_allocator, &model->preparation);
  }
  int64_t capacity = 0;
  iree_vm_buffer_t* metadata[2] = {0};
  iree_const_byte_span_t bytes[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &capacity);
  }
  if (iree_status_is_ok(status) &&
      (capacity <= 0 || (uint64_t)capacity > SIZE_MAX)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source request capacity must fit host storage");
  }
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(
        &model->control.vm_types, results[i + 1], &metadata[i]);
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
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[3], &model->control.state);
  }
  if (iree_status_is_ok(status) && !model->control.state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source image state must not be null");
  }
  if (iree_status_is_ok(status)) {
    char* path = NULL;
    status = loom_serve_path_resolve(
        options->checkpoint_directory,
        iree_make_string_view((const char*)bytes[0].data, bytes[0].data_length),
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
  const iree_vm_signature_type_t f32 = {
      .kind = IREE_VM_SIGNATURE_TYPE_KIND_SCALAR,
      .value.scalar = IREE_VM_SCALAR_TYPE_F32};
  const iree_vm_signature_type_t request_arguments[] = {
      buffer, buffer, i32, i64, i64, buffer, i64, f32};
  const iree_vm_signature_type_t request_results[] = {i32, buffer};
  const iree_vm_signature_type_t generate_arguments[] = {
      i32, i32, i32, i64, device_buffer, device_buffer, device_buffer};
  const loom_serve_program_binding_t bindings[] = {
      {IREE_SVL("prepare_request"),
       {request_arguments, IREE_ARRAYSIZE(request_arguments)},
       {request_results, IREE_ARRAYSIZE(request_results)},
       &model->control.prepare_request},
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
  const iree_allocator_t host_allocator = model->host_allocator;
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_image_check_layouts(model));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create_execution(model->owner, &model->execution));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0]);
  const uint32_t roots = program->requirements.fixed_buffer_count;
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

static iree_status_t loom_serve_image_model_link(
    loom_serve_image_model_t* model, iree_vm_module_t* module,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t host_allocator = model->host_allocator;
  IREE_RETURN_IF_ERROR(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(model->owner), model->execution,
      model->weights.count, model->weights.plans, host_allocator,
      &model->residency));
  const loom_serve_executable_options_t options = {
      .device = model->owner,
      .jit = model->jit,
      .preparation = model->preparation,
      .layout = loom_serve_jit_stage_program(model->stages.values[0]),
      .recording = {.queue_family = iree_hal_queue_family(
                        loom_serve_device_dispatch_queue(model->owner)),
                    .parameters = model->weights.values,
                    .command_mode = command_mode,
                    .binding_count = 2,
                    .workspace = LOOM_SERVE_STAGE_WORKSPACE_INVOCATION,
                    .command_quantum = model->command_quantum},
      .cache = model->specialization_cache};
  iree_status_t status = loom_serve_stage_table_allocate(
      &options.recording, NULL, model->stages.count, model->stages.values,
      host_allocator, &model->stages.table);
  if (iree_status_is_ok(status)) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(model->input_capacity, 1);
    status = loom_serve_execution_alloca_from_pool(
        model->execution, loom_serve_device_storage_pool(model->owner), 1,
        &request, &model->input);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_retirement_track(&model->retirement, &model->input,
                                         host_allocator);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_image_create_control(model, module));
  loom_serve_executable_options_t bound_options = options;
  bound_options.stages = model->stages.table;
  bound_options.runner = model->control.runner;
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
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  const iree_string_view_t roots[] = {IREE_SVL("prepare_request"),
                                      IREE_SVL("generate")};
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
    loom_serve_retirement_initialize(&load->model->retirement);
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
            model->jit, model->preparation, loom_serve_image_load_compiled,
            load, load->host_allocator);
        submitted = iree_status_is_ok(status);
        break;
      case IMAGE_LOAD_BIND:
        status = loom_serve_image_model_bind(model);
        load->phase = IMAGE_LOAD_WEIGHTS;
        break;
      case IMAGE_LOAD_WEIGHTS:
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
              loom_serve_image_load_complete, load);
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
          status = loom_serve_image_model_link(model, load->module,
                                               load->command_mode);
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

static iree_status_t loom_serve_image_prepare_request(
    loom_serve_image_model_t* model, iree_string_view_t prompt, uint64_t seed,
    float strength, uint32_t width, uint32_t height, uint32_t* out_stage,
    int64_t* out_tag, iree_vm_buffer_t** out_payload) {
  *out_payload = NULL;
  iree_vm_buffer_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)prompt.data, prompt.size),
      iree_vm_buffer_release_callback_null(), model->host_allocator, &text));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_variant_from_i32((int32_t)model->stages.count),
      iree_vm_variant_from_i64(height),
      iree_vm_variant_from_i64(width),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &text),
      iree_vm_variant_from_i64((int64_t)seed),
      iree_vm_variant_from_f32(strength),
  };
  iree_vm_variant_t results[2] = {0};
  loom_serve_preparation_module_begin(model->control.preparation,
                                      model->stages.count, 0);
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                     model->control.prepare_request,
                     iree_vm_variant_span_from_array(arguments),
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
    *out_payload = payload;
    *out_stage = (uint32_t)selected;
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
                                                model->input),
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
  return loom_serve_residency_deactivate(model->residency, out_complete);
}

loom_serve_residency_t* loom_serve_image_model_residency(
    const loom_serve_image_model_t* model) {
  return model->residency;
}

loom_serve_memory_statistics_t loom_serve_image_model_weight_statistics(
    const loom_serve_image_model_t* model) {
  return loom_serve_residency_statistics(model->residency);
}

loom_serve_weight_activity_t loom_serve_image_model_weight_activity(
    const loom_serve_image_model_t* model) {
  return loom_serve_residency_activity(model->residency);
}

iree_status_t loom_serve_image_model_prepare_request(
    loom_serve_image_model_t* model, iree_string_view_t prompt, uint64_t seed,
    float strength, uint32_t width, uint32_t height,
    loom_serve_image_prepared_t* out_prepared) {
  *out_prepared = (loom_serve_image_prepared_t){0};
  loom_serve_executables_capture(model->executables);
  int64_t tag = 0;
  iree_status_t status = loom_serve_image_prepare_request(
      model, prompt, seed, strength, width, height, &model->invocation.stage,
      &tag, &model->invocation.payload);
  const iree_host_size_t output_length = (iree_host_size_t)width * height * 12;
  if (iree_status_is_ok(status) && output_length != model->output.data_length) {
    status = iree_allocator_realloc(model->host_allocator, output_length,
                                    (void**)&model->output.data);
    if (iree_status_is_ok(status)) {
      model->output.data_length = output_length;
      loom_serve_module_bind_feedback(model->control.runner, 0, model->output);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_image_model_discard_prepared_request(model);
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

void loom_serve_image_model_discard_prepared_request(
    loom_serve_image_model_t* model) {
  loom_serve_executables_release(model->executables);
  iree_vm_buffer_release(model->invocation.payload);
  model->invocation.payload = NULL;
}

void loom_serve_image_model_set_observer(loom_serve_image_model_t* model,
                                         loom_serve_wake_callback_t observer) {
  loom_serve_executables_set_observer(model->executables, observer);
}

iree_status_t loom_serve_image_model_advance_request(
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
    model->invocation.workspace_offset =
        iree_device_align(model->output.data_length, alignment);
    model->invocation.storage = iree_hal_make_pool_reservation_request(
        model->invocation.workspace_offset +
            stage->transient.required_byte_length,
        alignment);
    return loom_serve_execution_reserve_workspace(
        model->execution, model->invocation.storage.allocation_size, alignment);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_image_begin(loom_serve_image_model_t* model) {
  iree_vm_buffer_t* payload = model->invocation.payload;
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      payload, 0, iree_vm_buffer_length(payload), &input));
  iree_hal_buffer_t* frame = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_execution_alloca(
      model->execution, &model->invocation.storage, &frame));
  iree_status_t status =
      iree_hal_buffer_subspan(frame, 0, model->output.data_length,
                              model->host_allocator, &model->invocation.output);
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_subspan(frame, model->invocation.workspace_offset,
                                     model->invocation.storage.allocation_size -
                                         model->invocation.workspace_offset,
                                     model->host_allocator,
                                     &model->invocation.workspace);
  }
  iree_hal_buffer_release(frame);
  IREE_RETURN_IF_ERROR(status);
  iree_hal_transfer_operation_t upload = {0};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input.data;
  upload.upload.target_buffer = model->input;
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
    // Feedback forks from work. Its captured output view retains the frame
    // through actual download retirement; early work-timeline dealloca would
    // invalidate that source. The shared scope returns the frame to its pool
    // after the last queue/caller reference retires, including cancellation.
    iree_hal_buffer_release(model->invocation.workspace);
    model->invocation.workspace = NULL;
    iree_hal_buffer_release(model->invocation.output);
    model->invocation.output = NULL;
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
  loom_serve_residency_release(model->residency);
  loom_serve_executables_release(model->executables);
  return iree_make_const_byte_span(model->output.data,
                                   model->output.data_length);
}

iree_status_t loom_serve_image_model_generate(
    loom_serve_image_model_t* model, iree_string_view_t prompt, uint64_t seed,
    float strength, uint32_t width, uint32_t height,
    loom_serve_image_prepared_t* out_prepared,
    iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_image_model_prepare_request(
      model, prompt, seed, strength, width, height, out_prepared));
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_image_model_set_observer(
      model, (loom_serve_wake_callback_t){loom_serve_image_load_notify,
                                          &notification});
  bool resolved = false;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !resolved) {
    const uint32_t ticket = iree_notification_prepare_wait(&notification);
    status = loom_serve_image_model_advance_request(model, false, &resolved);
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
    status = loom_serve_residency_acquire_wait(model->residency);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_image_model_discard_prepared_request(model);
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
  return loom_serve_residency_try_admit(loom_serve_image_model_residency(self),
                                        out_admission);
}

static iree_status_t loom_serve_image_generator_prepare(
    void* self, const loom_serve_image_request_t* request,
    loom_serve_image_prepared_t* out_prepared) {
  return loom_serve_image_model_prepare_request(
      self, iree_string_builder_view(&request->prompt), request->seed,
      request->strength, request->width, request->height, out_prepared);
}

static void loom_serve_image_generator_set_observer(
    void* self, loom_serve_wake_callback_t observer) {
  loom_serve_image_model_set_observer(self, observer);
}

static iree_status_t loom_serve_image_generator_advance(void* self, bool cancel,
                                                        bool* out_complete) {
  return loom_serve_image_model_advance_request(self, cancel, out_complete);
}

static iree_status_t loom_serve_image_generator_submit(
    void* self, loom_serve_image_submit_flags_t flags,
    loom_serve_execution_frontier_t* out_frontier, bool* out_complete) {
  return loom_serve_image_model_submit(self, flags, out_frontier, out_complete);
}

static void loom_serve_image_generator_discard(void* self) {
  loom_serve_residency_cancel_admission(loom_serve_image_model_residency(self));
  loom_serve_image_model_discard_prepared_request(self);
}

static iree_const_byte_span_t loom_serve_image_generator_complete(void* self) {
  return loom_serve_image_model_complete(self);
}

static loom_serve_weight_activity_t loom_serve_image_generator_weight_activity(
    void* self) {
  return loom_serve_image_model_weight_activity(self);
}

loom_serve_image_generator_t loom_serve_image_model_generator(
    loom_serve_image_model_t* model) {
  return (loom_serve_image_generator_t){
      .self = model,
      .prepare = loom_serve_image_generator_prepare,
      .set_observer = loom_serve_image_generator_set_observer,
      .advance_prepared_request = loom_serve_image_generator_advance,
      .discard_prepared_request = loom_serve_image_generator_discard,
      .try_admit = loom_serve_image_generator_admit,
      .submit = loom_serve_image_generator_submit,
      .complete = loom_serve_image_generator_complete,
      .weight_activity = loom_serve_image_generator_weight_activity,
      .execution_update = loom_serve_executables_update(model->executables),
      .tokenizer = model->tokenizer,
      .capacity_domain = loom_serve_residency_cache(model->residency)};
}
