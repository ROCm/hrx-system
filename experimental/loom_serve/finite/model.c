// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/finite/model.h"

#include <stdio.h>
#include <string.h>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/json.h"
#include "experimental/loom_serve/runtime/module.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/program.h"
#include "experimental/loom_serve/runtime/weights.h"
#include "iree/base/internal/path.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loomc/iree.h"

typedef struct finite_model_stage_t {
  // Compiled source root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Reusable command retaining the model's immutable parameter domains.
  iree_hal_command_buffer_t* command;
} finite_model_stage_t;

struct loom_serve_finite_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
  // Copied startup evidence, independent of every compiler result lifetime.
  loom_serve_compilation_report_t compilation;
  // Owned public identifier returned by the source bootstrap.
  iree_string_view_t name;
  // Positive additive cohort capacity returned by the cold source.
  uint64_t batch_limit;
  // Maximum uploaded bytes, including source-defined padding.
  iree_host_size_t input_capacity;
  // Maximum device output geometry, independent of compact feedback.
  iree_host_size_t output_capacity;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Borrowed shared device/backing owner, outliving the model.
  loom_serve_device_t* owner;
  // Owned model work and feedback timelines, independent of other models.
  loom_serve_execution_t* execution;
  // Shared-device admission group protecting every checkpoint domain together.
  loom_serve_residency_t* residency;
  // Cohort storage remains live through device retirement and CPU encoding.
  struct {
    // Prepared upload, retained even after partial enqueue failure.
    iree_vm_buffer_t* payload;
    // Owned source metadata describing compact result rows.
    iree_vm_buffer_t* plan;
    // Borrowed checked plan records, whose owner is plan.
    iree_const_byte_span_t records;
    // Source-selected retained command ordinal.
    uint32_t stage;
    // Physical output allocation extent required by the selected command.
    iree_host_size_t output_length;
    // Live compact feedback bytes written by source.run.
    iree_host_size_t feedback_length;
    // Original HTTP request count in stable cohort order.
    iree_host_size_t request_count;
  } invocation;
  // Cold live-source compiler borrowing the shared host task pool.
  loom_serve_jit_t* jit;
  // Owned source declarations, independent of the bootstrap VM and arguments.
  loom_serve_preparation_t* preparation;
  // Cold retained shapes sharing one parameter and issue-time buffer bank.
  struct {
    // Number of owned slots, including partially initialized slots on failure.
    uint32_t count;
    // Owned stage array in source declaration order.
    finite_model_stage_t* values;
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
    // Opaque source-owned state, retained across serialized requests.
    iree_vm_buffer_t* state;
    // Cold-resolved request preparation function.
    iree_vm_function_t prepare_request;
    // Cold-resolved cohort construction function.
    iree_vm_function_t prepare_batch;
    // Cold-resolved command submission and compact feedback function.
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
  // Compact vector feedback, exclusive through the encoder process release.
  iree_byte_span_t output;
};

// Consumes a model after execution retirement, or a load that never submitted
// device work. Weight views lose every cached consumer before plan destruction.
static iree_status_t finite_model_destroy_resources(
    loom_serve_finite_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  loom_serve_program_destroy(model->control.program);
  iree_vm_buffer_release(model->control.tags);
  iree_vm_buffer_release(model->control.state);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    iree_hal_command_buffer_release(model->stages.values[i].command);
    loom_serve_jit_stage_destroy(model->stages.values[i].compiled);
  }
  iree_allocator_free(model->allocator, model->stages.values);
  loom_serve_preparation_destroy(model->preparation);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    iree_hal_buffer_release(model->weights.values[i]);
  }
  iree_vm_buffer_release(model->invocation.plan);
  iree_vm_buffer_release(model->invocation.payload);
  iree_vm_environment_free(model->control.environment);
  loom_serve_residency_destroy(model->residency);
  loom_serve_execution_release(model->execution);
  for (iree_host_size_t i = 0; model->weights.plans && i < model->weights.count;
       ++i) {
    status = iree_status_join(
        status, loom_serve_weights_destroy(model->weights.plans[i]));
  }
  iree_allocator_free(model->allocator, model->weights.values);
  iree_allocator_free(model->allocator, model->weights.plans);
  iree_allocator_free(model->allocator, (void*)model->name.data);
  loom_serve_jit_destroy(model->jit);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model->output.data);
  iree_allocator_free(model->allocator, model);
  return status;
}

iree_status_t loom_serve_finite_model_destroy(
    loom_serve_finite_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->buffers); ++i) {
    iree_hal_buffer_release(model->buffers[i]);
  }
  iree_status_t status = loom_serve_execution_drain(model->execution);
  return iree_status_join(status, finite_model_destroy_resources(model));
}

static iree_status_t finite_load_tokenizer(loom_serve_finite_model_t* model,
                                           iree_string_view_t directory,
                                           iree_string_view_t asset) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_file_path_join(directory, asset, model->allocator, &path));
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(path), model->allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        model->allocator, &model->tokenizer);
  }
  iree_io_file_contents_free(contents);
  iree_allocator_free(model->allocator, path);
  return status;
}

// Cold source results are consumed while their environment is live. Source
// state survives destruction of the cold program and is opaque to this owner.
static iree_status_t finite_prepare(
    loom_serve_finite_model_t* model,
    const loom_serve_finite_model_options_t* options) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(allocator, &model->control.environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    model->control.environment, IREE_SV("vm")),
                                &model->control.vm_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("prepare.loom"), allocator, &path));
  iree_vm_module_t* modules[2] = {0};
  iree_status_t status = loom_serve_input_module_create(
      model->control.environment, NULL, &modules[0], allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(model->control.environment,
                                           &modules[1], allocator);
  }
  iree_vm_variant_t arguments[2] = {0};
  const iree_string_view_t inputs[] = {options->asset_directory,
                                       options->options};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(inputs) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_create(inputs[i].size, 8, allocator, &buffer);
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
  iree_vm_variant_t results[7] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_create(
        model->control.environment, iree_make_cstring_view(path),
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){modules, 2}, &model->compilation,
        &model->preparation, allocator);
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
    status = iree_allocator_clone(allocator, metadata[1], (void**)&name);
    model->name = iree_make_string_view(name, metadata[1].data_length);
    model->batch_limit = (uint64_t)capacities[0];
    model->input_capacity = (iree_host_size_t)capacities[1];
    model->output_capacity = (iree_host_size_t)capacities[2];
    model->output.data_length = (iree_host_size_t)capacities[3];
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
    status = finite_load_tokenizer(
        model, options->asset_directory,
        iree_make_string_view((const char*)metadata[0].data,
                              metadata[0].data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(modules[0]);
  iree_vm_module_release(modules[1]);
  iree_allocator_free(allocator, path);
  if (iree_status_is_ok(status)) {
    const iree_host_size_t count =
        loom_serve_preparation_stage_count(model->preparation);
    status = iree_allocator_malloc_array(allocator, count,
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
static iree_status_t finite_check_layouts(loom_serve_finite_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const iree_host_size_t binding_count = 3;
  const uint32_t roots = canonical->requirements.fixed_buffer_count;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const finite_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    if (program->requirements.rebindable_binding_count != binding_count ||
        program->requirements.transient.binding_index != 2 ||
        program->requirements.fixed_buffer_count != roots ||
        program->parameter_roots.count != roots ||
        declaration->parameter_count != roots ||
        program->requirements.launch_counts.binding_index != UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "stage %u does not implement the finite vector ABI", i);
    }
    const loom_serve_preparation_stage_t* maximum =
        loom_serve_preparation_stage(model->preparation, 0);
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_serve_preparation_parameter_t* parameter =
          &declaration->parameters[root];
      const loom_serve_preparation_parameter_t* shared =
          &maximum->parameters[root];
      if (parameter->binding != root ||
          loom_cmd_program_parameter_root_at(program, root)
                  .fixed_buffer_index != parameter->binding ||
          !iree_string_view_equal(parameter->path, shared->path) ||
          !iree_string_view_equal(parameter->policy, shared->policy)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage %u must declare the same ordered checkpoint domains", i);
      }
    }
    if (!i) {
      continue;
    }
    if (canonical->parameters.count != program->parameters.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "stage %u parameter count differs from shared layout", i);
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_cmd_program_parameter_root_t lhs =
          loom_cmd_program_parameter_root_at(canonical, root);
      const loom_cmd_program_parameter_root_t rhs =
          loom_cmd_program_parameter_root_at(program, root);
      if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.required_byte_length != rhs.required_byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage %u parameter root %u differs from shared layout", i, root);
      }
    }
    for (uint32_t parameter = 0; parameter < canonical->parameters.count;
         ++parameter) {
      const loom_cmd_program_parameter_t lhs =
          loom_cmd_program_parameter_at(canonical, parameter);
      const loom_cmd_program_parameter_t rhs =
          loom_cmd_program_parameter_at(program, parameter);
      if (!iree_string_view_equal(lhs.key, rhs.key) ||
          lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.byte_offset != rhs.byte_offset ||
          lhs.byte_length != rhs.byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage %u parameter placement %u differs from shared layout", i,
            parameter);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t finite_create_control(loom_serve_finite_model_t* model,
                                           iree_vm_module_t* module) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
      model->stages.count * sizeof(int64_t), iree_alignof(int64_t), allocator,
      &model->control.tags));
  iree_byte_span_t tags = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(
      model->control.tags, 0, iree_vm_buffer_length(model->control.tags),
      &tags));
  loom_serve_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, model->stages.count, sizeof(*stages), (void**)&stages));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    stages[i] = (loom_serve_stage_t){
        .command_buffer = model->stages.values[i].command,
        .binding_count = 2,
        .transient =
            loom_serve_jit_stage_program(model->stages.values[i].compiled)
                ->requirements.transient,
    };
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = 2,
      .stages = {model->stages.count, stages},
      .feedback = {1, &model->output},
  };
  iree_vm_module_t* native_modules[3] = {0};
  iree_vm_buffer_t* read_only_tags = NULL;
  iree_status_t status = iree_vm_buffer_subspan(
      model->control.tags, 0, tags.data_length, IREE_VM_BUFFER_ACCESS_FLAG_READ,
      allocator, &read_only_tags);
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_release(model->control.tags);
    model->control.tags = read_only_tags;
    status =
        loom_serve_module_create(&model->control.hal_types, model->execution,
                                 options, allocator, &native_modules[0]);
  }
  iree_allocator_free(allocator, stages);
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(model->control.environment,
                                            model->tokenizer,
                                            &native_modules[1], allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(model->control.environment,
                                           &native_modules[2], allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_link(module,
                                     (iree_vm_module_span_t){native_modules, 3},
                                     allocator, &model->control.program);
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
      process, IREE_SV("model"), IREE_SV("prepare_batch"),
      &model->control.prepare_batch));
  return iree_vm_process_lookup_function(process, IREE_SV("model"),
                                         IREE_SV("run"), &model->control.run);
}

static iree_status_t finite_model_bind(loom_serve_finite_model_t* model) {
  const iree_allocator_t allocator = model->allocator;
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(finite_check_layouts(model));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create_execution(model->owner, &model->execution));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const uint32_t roots = program->requirements.fixed_buffer_count;
  if (roots) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, roots, sizeof(*model->weights.values),
        (void**)&model->weights.values));
    model->weights.count = roots;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        allocator, roots, sizeof(*model->weights.plans),
        (void**)&model->weights.plans));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < roots && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_root_t reflected =
        loom_cmd_program_parameter_root_at(program, i);
    const loom_serve_weight_root_t root = {
        program, reflected,
        &model->weights.values[reflected.fixed_buffer_index]};
    status = loom_serve_weights_create(
        device, loom_serve_device_transfer_queue(model->owner), dispatch,
        loom_serve_device_memory_pool(model->owner), 0, 1, &root,
        &model->weights.plans[i], allocator);
  }
  return status;
}

static iree_status_t finite_model_link(
    loom_serve_finite_model_t* model, iree_vm_module_t* module,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t allocator = model->allocator;
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_residency_create(
      loom_serve_device_residency_cache(model->owner), model->execution,
      model->weights.count, model->weights.plans, &model->residency,
      allocator));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    finite_model_stage_t* stage = &model->stages.values[i];
    IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(dispatch), command_mode,
        model->weights.values, &stage->command));
  }
  // Register maximum cohort geometry without committing any backing. Roots
  // and reflected private workspace borrow the shared pool only when submitted.
  IREE_RETURN_IF_ERROR(loom_serve_execution_reserve_workspace(
      model->execution, iree_max(model->input_capacity, model->output_capacity),
      256));
  return finite_create_control(model, module);
}

typedef enum finite_load_phase_e {
  FINITE_LOAD_PREPARE,
  FINITE_LOAD_STAGES,
  FINITE_LOAD_WEIGHTS,
  FINITE_LOAD_LINK,
  FINITE_LOAD_COMPLETE,
} finite_load_phase_t;

struct loom_serve_finite_load_t {
  // One private bootstrap task, retired before any native compilation starts.
  loomc_task_t task;
  // Allocator owning this operation, independent of the transferred model.
  iree_allocator_t allocator;
  // Owned partial model, transferred only after successful terminal advance.
  loom_serve_finite_model_t* model;
  // Borrowed immutable package inputs, retained by the registration owner.
  loom_serve_finite_model_options_t options;
  // Owner-only next phase; callbacks never make shared-state transitions.
  finite_load_phase_t phase;
  // Next stage or weight domain within the current phase.
  uint32_t index;
  // Recording policy shared by consumer and weight preparation commands.
  iree_hal_command_buffer_mode_t command_mode;
  // Prepared control bytecode, not yet linked to device capabilities.
  iree_vm_module_t* module;
  // Owned current weight policy path, retained through its async completion.
  char* policy_path;
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
    loom_serve_finite_load_wake_fn_t wake;
    // Borrowed notification payload, retained through terminal advance.
    void* user_data;
  } event;
};

static void finite_load_complete(void* user_data, iree_status_t status) {
  loom_serve_finite_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t finite_load_prepare(loom_serve_finite_load_t* load) {
  loom_serve_finite_model_t* model = load->model;
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(finite_prepare(model, &load->options));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, model->output.data_length, (void**)&model->output.data));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(load->options.source_directory,
                                           IREE_SV("control.loom"), allocator,
                                           &path));
  const iree_string_view_t roots[] = {
      IREE_SVL("prepare_request"), IREE_SVL("prepare_batch"), IREE_SVL("run")};
  iree_status_t status = loom_serve_program_compile(
      model->control.environment, iree_make_cstring_view(path),
      IREE_ARRAYSIZE(roots), roots, &model->compilation, allocator,
      &load->module);
  iree_allocator_free(allocator, path);
  IREE_RETURN_IF_ERROR(status);
  return loom_serve_jit_create(loom_serve_device_handle(model->owner),
                               loom_serve_device_dispatch_queue(model->owner),
                               loom_serve_device_host(model->owner),
                               load->options.source_directory, NULL,
                               &model->compilation, allocator, &model->jit);
}

static void finite_load_execute(loomc_task_t* task,
                                loomc_host_size_t worker_ordinal) {
  (void)worker_ordinal;
  loom_serve_finite_load_t* load = (loom_serve_finite_load_t*)task;
  load->task_status = finite_load_prepare(load);
}

static void finite_load_release(loomc_task_t* task) {
  loom_serve_finite_load_t* load = (loom_serve_finite_load_t*)task;
  finite_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t finite_load_vtable = {
    .execute = finite_load_execute,
    .destroy = finite_load_release,
};

static void finite_load_compiled(void* user_data, iree_status_t status,
                                 loom_serve_jit_stage_t* compiled) {
  loom_serve_finite_load_t* load = user_data;
  load->model->stages.values[load->index - 1].compiled = compiled;
  finite_load_complete(load, status);
}

iree_status_t loom_serve_finite_load_create(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_load_wake_fn_t wake, void* user_data,
    loom_serve_finite_load_t** out_load, iree_allocator_t host_allocator) {
  *out_load = NULL;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  loom_serve_finite_load_t* load = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*load), (void**)&load));
  load->allocator = host_allocator;
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
    load->model->allocator = host_allocator;
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
  iree_status_t status = load->event.status;
  load->event.status = iree_ok_status();
  iree_slim_mutex_unlock(&load->event.mutex);
  iree_allocator_free(load->allocator, load->policy_path);
  load->policy_path = NULL;
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
        if (load->index < model->stages.count) {
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, load->index++);
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_jit_compile_async(model->jit, declaration->root,
                                                &declaration->config,
                                                finite_load_compiled, load);
          submitted = iree_status_is_ok(status);
        } else {
          status = finite_model_bind(model);
          load->index = 0;
          load->phase = FINITE_LOAD_WEIGHTS;
        }
        break;
      case FINITE_LOAD_WEIGHTS:
        if (load->index < model->weights.count) {
          const uint32_t i = load->index++;
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, 0);
          const loom_serve_preparation_parameter_t* parameter =
              &declaration->parameters[i];
          status = iree_file_path_join(load->options.source_directory,
                                       parameter->policy, load->allocator,
                                       &load->policy_path);
          if (iree_status_is_ok(status)) {
            iree_slim_mutex_lock(&load->event.mutex);
            load->event.pending = true;
            iree_slim_mutex_unlock(&load->event.mutex);
            status = loom_serve_weights_prepare_async(
                model->weights.plans[i], host, model->jit, load->command_mode,
                parameter->path, iree_make_cstring_view(load->policy_path),
                finite_load_complete, load);
            submitted = iree_status_is_ok(status);
          }
        } else {
          load->phase = FINITE_LOAD_LINK;
        }
        break;
      case FINITE_LOAD_LINK:
        status = finite_model_link(model, load->module, load->command_mode);
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

iree_status_t loom_serve_finite_load_destroy(loom_serve_finite_load_t* load) {
  if (!load) {
    return iree_ok_status();
  }
  iree_vm_module_release(load->module);
  iree_status_t status = finite_model_destroy_resources(load->model);
  iree_allocator_free(load->allocator, load->policy_path);
  iree_slim_mutex_deinitialize(&load->event.mutex);
  iree_allocator_free(load->allocator, load);
  return status;
}

static void finite_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_finite_model_create(
    loom_serve_device_t* device,
    const loom_serve_finite_model_options_t* options,
    loom_serve_finite_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_finite_load_t* load = NULL;
  iree_status_t status =
      loom_serve_finite_load_create(device, options, finite_load_notify,
                                    &notification, &load, host_allocator);
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
    loom_output_stream_t output;
    loom_output_stream_for_file(stderr, &output);
    status = iree_status_join(
        status, loom_serve_compilation_write(
                    loom_serve_finite_load_report(load), &output));
    if (iree_status_is_ok(status)) {
      *out_model = loom_serve_finite_load_take_model(load);
    }
    status = iree_status_join(status, loom_serve_finite_load_destroy(load));
  }
  iree_notification_deinitialize(&notification);
  return status;
}

uint64_t loom_serve_finite_model_batch_limit(
    const loom_serve_finite_model_t* model) {
  return model->batch_limit;
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

// This is the external source-plan boundary. Native queries below directly
// consume the checked offsets; they neither rescan nor revalidate the plan.
static iree_status_t finite_check_plan(loom_serve_finite_model_t* model,
                                       iree_host_size_t request_count,
                                       const iree_host_size_t* expected_rows) {
  iree_const_byte_span_t bytes;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      model->invocation.plan, 0, iree_vm_buffer_length(model->invocation.plan),
      &bytes));
  if (bytes.data_length < 32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan requires a complete header");
  }
  const uint64_t output_length = iree_unaligned_load_le_u64(bytes.data);
  const uint64_t feedback_length = iree_unaligned_load_le_u64(bytes.data + 8);
  const uint64_t count = iree_unaligned_load_le_u64(bytes.data + 16);
  const uint64_t rows = iree_unaligned_load_le_u64(bytes.data + 24);
  if (!output_length || output_length > model->output_capacity ||
      !feedback_length || feedback_length > model->output.data_length ||
      feedback_length > output_length || count != request_count ||
      count > (bytes.data_length - 32) / 24 ||
      rows != (bytes.data_length - 32 - count * 24) / 16 ||
      bytes.data_length != 32 + count * 24 + rows * 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan has invalid extents");
  }
  uint64_t first_row = 0;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    const uint8_t* record = bytes.data + 32 + i * 24;
    const uint64_t row_count = iree_unaligned_load_le_u64(record + 8);
    if (iree_unaligned_load_le_u64(record) != first_row ||
        row_count != expected_rows[i] || row_count > rows - first_row) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source result rows differ from request %zu", i);
    }
    first_row += row_count;
  }
  if (first_row != rows) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source result plan contains unclaimed rows");
  }
  for (uint64_t i = 0; i < rows; ++i) {
    const uint8_t* record = bytes.data + 32 + count * 24 + i * 16;
    const uint64_t offset = iree_unaligned_load_le_u64(record);
    const uint64_t width = iree_unaligned_load_le_u64(record + 8);
    if (offset % 4 || offset > feedback_length || !width ||
        width > (feedback_length - offset) / 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source vector lies outside compact feedback");
    }
  }
  model->invocation.records = bytes;
  model->invocation.output_length = (iree_host_size_t)output_length;
  model->invocation.feedback_length = (iree_host_size_t)feedback_length;
  model->invocation.request_count = request_count;
  return iree_ok_status();
}

void loom_serve_finite_model_release_batch(loom_serve_finite_model_t* model) {
  iree_vm_buffer_release(model->invocation.payload);
  iree_vm_buffer_release(model->invocation.plan);
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
      iree_vm_buffer_create(count * 16, 8, model->allocator, &ranges));
  iree_status_t status =
      iree_vm_buffer_create(length, 8, model->allocator, &packed);
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
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                       model->control.prepare_batch,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_array(results));
  }
  int32_t stage = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &stage);
  }
  if (iree_status_is_ok(status) &&
      (stage < 0 || (uint32_t)stage >= model->stages.count)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source selected an undeclared command");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[1], &model->invocation.payload);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &model->control.vm_types, &results[2], &model->invocation.plan);
  }
  if (iree_status_is_ok(status) &&
      (!model->invocation.payload || !model->invocation.plan ||
       !iree_vm_buffer_length(model->invocation.payload) ||
       iree_vm_buffer_length(model->invocation.payload) >
           model->input_capacity)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source batch exceeds declared input storage");
  }
  if (iree_status_is_ok(status)) {
    model->invocation.stage = (uint32_t)stage;
    status = finite_check_plan(model, count, expected_rows);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (!iree_status_is_ok(status)) {
    loom_serve_finite_model_release_batch(model);
  }
  return status;
}

iree_status_t loom_serve_finite_model_submit(
    loom_serve_finite_model_t* model,
    loom_serve_execution_frontier_t* out_frontier) {
  *out_frontier = (loom_serve_execution_frontier_t){0};
  iree_const_byte_span_t input;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      model->invocation.payload, 0,
      iree_vm_buffer_length(model->invocation.payload), &input));
  loom_serve_execution_t* execution = model->execution;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    const iree_host_size_t length =
        i ? model->invocation.output_length : input.data_length;
    const iree_hal_pool_reservation_request_t request = {
        .params = {.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
                   .access = IREE_HAL_MEMORY_ACCESS_ALL,
                   .usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                            IREE_HAL_BUFFER_USAGE_TRANSFER,
                   .min_alignment = 256},
        .allocation_size = length,
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
        iree_vm_variant_from_i64((int64_t)model->invocation.feedback_length),
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
    // The final device reader is on the feedback branch, not the work branch.
    // Neither shared queue affinity nor submission order implies this edge.
    loom_serve_execution_frontier_t frontier =
        loom_serve_execution_frontier(execution);
    status = loom_serve_execution_barrier(
        execution, (iree_hal_semaphore_list_t){1, &frontier.semaphores[1],
                                               &frontier.values[1]});
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_execution_dealloca(execution, model->buffers[1], &value);
  }
  // Execution tracks final queue ownership on both success and failure. Caller
  // roots must be released before outer drain can join that ownership.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_hal_buffer_release(model->buffers[i]);
    model->buffers[i] = NULL;
  }
  if (iree_status_is_ok(status)) {
    *out_frontier = loom_serve_execution_frontier(execution);
  }
  return status;
}

void loom_serve_finite_model_complete(loom_serve_finite_model_t* model) {
  loom_serve_residency_release(model->residency);
}

uint64_t loom_serve_finite_model_input_tokens(
    const loom_serve_finite_model_t* model, iree_host_size_t request) {
  return iree_unaligned_load_le_u64(model->invocation.records.data + 32 +
                                    request * 24 + 16);
}

iree_const_byte_span_t loom_serve_finite_model_row(
    const loom_serve_finite_model_t* model, iree_host_size_t request,
    iree_host_size_t row) {
  const uint8_t* records = model->invocation.records.data;
  const uint64_t first =
      iree_unaligned_load_le_u64(records + 32 + request * 24);
  const uint8_t* vector =
      records + 32 + model->invocation.request_count * 24 + (first + row) * 16;
  return iree_make_const_byte_span(
      model->output.data + iree_unaligned_load_le_u64(vector),
      iree_unaligned_load_le_u64(vector + 8) * 4);
}
