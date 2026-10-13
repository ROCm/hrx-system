// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/model.h"

#include <inttypes.h>
#include <stdio.h>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/json.h"
#include "experimental/loom_serve/runtime/module.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/program.h"
#include "experimental/loom_serve/runtime/retirement.h"
#include "experimental/loom_serve/runtime/weights.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loomc/iree.h"

typedef struct image_model_stage_t {
  // Compiled image root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Contiguous command parts sharing one compiler layout and workspace.
  struct {
    // Owned reusable commands, independent of the compiler product lifetime.
    iree_hal_command_buffer_t** values;
    // Successfully recorded parts, including partial construction for cleanup.
    uint32_t count;
    // First part in the warm VM's flat runner stage table.
    uint32_t first_stage;
  } parts;
} image_model_stage_t;

struct loom_serve_image_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
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
  // One invocation spanning CPU preparation, device submission and completion.
  struct {
    // Upload bytes retained through success or terminal queue retirement.
    iree_vm_buffer_t* payload;
    // Retained command selected by source preparation.
    uint32_t stage;
    // Source-owned next-part cursor; negative denotes terminal output.
    int32_t cursor;
    // One queue allocation preserving compiler live values across all parts.
    iree_hal_buffer_t* workspace;
    // Upload and allocation have been accepted for this invocation.
    bool started;
  } invocation;
  // Source compiler borrowing the host's shared task queue.
  loom_serve_jit_t* jit;
  // Owned source declarations, independent of the bootstrap VM and arguments.
  loom_serve_preparation_t* preparation;
  // Cold retained shapes sharing one parameter and issue-time buffer bank.
  struct {
    // Number of owned slots, including partially initialized slots on failure.
    uint32_t count;
    // Owned stage array in source declaration order.
    image_model_stage_t* values;
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
  // Inputs and final RGB; private command workspace uses shared device backing.
  iree_hal_buffer_binding_t bindings[2];
  // Completed NCHW F32 RGB feedback, overwritten by the next generation.
  iree_byte_span_t output;
};

static void image_model_release_consumers(loom_serve_image_model_t* model) {
  loom_serve_program_destroy(model->control.program);
  model->control.program = NULL;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    image_model_stage_t* stage = &model->stages.values[i];
    for (uint32_t j = 0; j < stage->parts.count; ++j) {
      iree_hal_command_buffer_release(stage->parts.values[j]);
      stage->parts.values[j] = NULL;
    }
    stage->parts.count = 0;
  }
  iree_hal_buffer_release(model->invocation.workspace);
  model->invocation.workspace = NULL;
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    iree_hal_buffer_release(model->weights.values[i]);
    model->weights.values[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->bindings); ++i) {
    iree_hal_buffer_release(model->bindings[i].buffer);
    model->bindings[i].buffer = NULL;
  }
}

void loom_serve_image_model_retire(loom_serve_image_model_t* model,
                                   void (*wake)(void* user_data),
                                   void* user_data) {
  loom_serve_execution_set_retirement_observer(model->execution, wake,
                                               user_data);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_set_retirement_observer(model->weights.plans[i], wake,
                                               user_data);
  }
  loom_serve_retirement_set_observer(&model->retirement, wake, user_data);
  image_model_release_consumers(model);
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
static iree_status_t image_model_destroy_resources(
    loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  image_model_release_consumers(model);
  IREE_ASSERT(loom_serve_image_model_is_retired(model));
  loom_serve_residency_destroy(model->residency);
  model->residency = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       model->weights.plans && i < model->weights.count &&
       iree_status_is_ok(status);
       ++i) {
    status = loom_serve_weights_destroy(model->weights.plans[i]);
    if (iree_status_is_ok(status)) {
      model->weights.plans[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_serve_retirement_deinitialize(&model->retirement);
    iree_vm_buffer_release(model->control.tags);
    iree_vm_buffer_release(model->control.state);
    for (uint32_t i = 0; i < model->stages.count; ++i) {
      loom_serve_jit_stage_destroy(model->stages.values[i].compiled);
      iree_allocator_free(model->allocator,
                          model->stages.values[i].parts.values);
    }
    iree_allocator_free(model->allocator, model->stages.values);
    loom_serve_preparation_destroy(model->preparation);
    iree_vm_buffer_release(model->invocation.payload);
    iree_vm_environment_free(model->control.environment);
    loom_serve_execution_release(model->execution);
    iree_allocator_free(model->allocator, model->weights.values);
    iree_allocator_free(model->allocator, model->weights.plans);
    iree_allocator_free(model->allocator, (void*)model->name.data);
    loom_serve_jit_destroy(model->jit);
    iree_tokenizer_free(model->tokenizer);
    iree_allocator_free(model->allocator, model->output.data);
    iree_allocator_free(model->allocator, model);
  }
  return status;
}

iree_status_t loom_serve_image_model_destroy_retired(
    loom_serve_image_model_t* model) {
  return image_model_destroy_resources(model);
}

iree_status_t loom_serve_image_model_destroy(loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  loom_serve_image_model_retire(model, NULL, NULL);
  iree_status_t status = loom_serve_execution_drain(model->execution);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    loom_serve_weights_await_retirement(model->weights.plans[i]);
  }
  loom_serve_retirement_await(&model->retirement);
  return iree_status_join(status, image_model_destroy_resources(model));
}

static iree_status_t image_load_tokenizer(loom_serve_image_model_t* model,
                                          iree_string_view_t base_directory,
                                          iree_string_view_t directory,
                                          iree_string_view_t asset) {
  char* resolved_directory = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_path_resolve(
      base_directory, directory, model->allocator, &resolved_directory));
  char* path = NULL;
  iree_status_t status =
      loom_serve_path_resolve(iree_make_cstring_view(resolved_directory), asset,
                              model->allocator, &path);
  iree_allocator_free(model->allocator, resolved_directory);
  IREE_RETURN_IF_ERROR(status);
  iree_io_file_contents_t* contents = NULL;
  status = iree_io_file_contents_read(iree_make_cstring_view(path),
                                      model->allocator, &contents);
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
static iree_status_t image_prepare(
    loom_serve_image_model_t* model,
    const loom_serve_image_model_options_t* options) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(allocator, &model->control.environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    model->control.environment, IREE_SV("vm")),
                                &model->control.vm_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_path_resolve(options->sources.base_directory,
                                               options->sources.prepare,
                                               allocator, &path));
  iree_vm_module_t* modules[2] = {0};
  iree_status_t status = loom_serve_input_module_create(
      model->control.environment, NULL, &modules[0], allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(model->control.environment,
                                           &modules[1], allocator);
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
    status = iree_vm_buffer_create(inputs[i].size, 8, allocator, &buffer);
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
    status = loom_serve_preparation_create(
        model->control.environment, iree_make_cstring_view(path),
        IREE_SV("prepare"), iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){modules, IREE_ARRAYSIZE(modules)},
        &model->compilation, &model->preparation, allocator);
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
    status = iree_allocator_clone(allocator, bytes[1], (void**)&name);
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
    status = image_load_tokenizer(
        model, options->sources.base_directory, options->checkpoint_directory,
        iree_make_string_view((const char*)bytes[0].data,
                              bytes[0].data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(modules); ++i) {
    iree_vm_module_release(modules[i]);
  }
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
static iree_status_t image_check_layouts(loom_serve_image_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const iree_host_size_t binding_count = 3;
  const uint32_t roots = canonical->requirements.fixed_buffer_count;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const image_model_stage_t* stage = &model->stages.values[i];
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
          "stage %u does not implement the diffusion image ABI", i);
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

static iree_status_t image_create_control(loom_serve_image_model_t* model,
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
  const image_model_stage_t* last =
      &model->stages.values[model->stages.count - 1];
  const uint32_t command_count = last->parts.first_stage + last->parts.count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, command_count, sizeof(*stages), (void**)&stages));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const image_model_stage_t* stage = &model->stages.values[i];
    for (uint32_t j = 0; j < stage->parts.count; ++j) {
      stages[stage->parts.first_stage + j] = (loom_serve_stage_t){
          .command_buffer = stage->parts.values[j], .binding_count = 3};
    }
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = 3,
      .stages = {command_count, stages},
      .feedback = {1, &model->output},
  };
  iree_vm_module_t* native_modules[2] = {0};
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
    status = loom_serve_program_link(module,
                                     (iree_vm_module_span_t){native_modules, 2},
                                     allocator, &model->control.program);
  }
  iree_vm_module_release(native_modules[0]);
  iree_vm_module_release(native_modules[1]);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_process_t* process =
      loom_serve_program_process(model->control.program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("prepare_request"),
      &model->control.prepare_request));
  return iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("generate"), &model->control.generate);
}

static iree_status_t image_model_bind(loom_serve_image_model_t* model) {
  const iree_allocator_t allocator = model->allocator;
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(image_check_layouts(model));
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
        loom_serve_device_storage_pool(model->owner),
        loom_serve_device_memory_pool(model->owner), 0, 1, &root,
        &model->weights.plans[i], allocator);
  }
  return status;
}

// Cold partitioning preserves the compiler's full-barrier cuts and layout.
// All part commands are owned immediately, including during partial failure.
static iree_status_t image_record_stage(
    loom_serve_image_model_t* model, image_model_stage_t* stage,
    iree_hal_command_buffer_mode_t command_mode) {
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(stage->compiled);
  IREE_RETURN_IF_ERROR(loom_serve_execution_reserve_workspace(
      model->execution, program->requirements.transient.required_byte_length,
      program->requirements.transient.minimum_alignment));
  const uint32_t capacity =
      model->command_quantum ? iree_max(program->commands.count, 1) : 1;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, capacity, sizeof(*stage->parts.values),
      (void**)&stage->parts.values));
  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(program, &iterator);
  loom_cmd_program_barrier_wave_t wave;
  loom_cmd_program_command_range_t range = {0};
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave)) {
    if (model->command_quantum && range.command_count &&
        (uint64_t)range.command_count + wave.commands.command_count >
            model->command_quantum) {
      status = loom_serve_jit_stage_record(
          stage->compiled, iree_hal_queue_family(dispatch), command_mode,
          &range, model->weights.values,
          &stage->parts.values[stage->parts.count]);
      if (iree_status_is_ok(status)) {
        ++stage->parts.count;
      }
      range.first_command = wave.commands.first_command;
      range.command_count = 0;
    }
    range.command_count += wave.commands.command_count;
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(dispatch), command_mode, &range,
        model->weights.values, &stage->parts.values[stage->parts.count]);
    if (iree_status_is_ok(status)) {
      ++stage->parts.count;
    }
  }
  return status;
}

static iree_status_t image_model_link(
    loom_serve_image_model_t* model, iree_vm_module_t* module,
    iree_hal_command_buffer_mode_t command_mode) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(loom_serve_residency_create(
      loom_serve_device_residency_cache(model->owner), model->execution,
      model->weights.count, model->weights.plans, &model->residency,
      allocator));
  iree_status_t status = iree_ok_status();
  uint64_t part_count = 0;
  for (uint32_t i = 0; i < model->stages.count && iree_status_is_ok(status);
       ++i) {
    image_model_stage_t* stage = &model->stages.values[i];
    stage->parts.first_stage = (uint32_t)part_count;
    status = image_record_stage(model, stage, command_mode);
    part_count += stage->parts.count;
    if (iree_status_is_ok(status) && part_count > INT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "image command parts exceed VM stage capacity");
    }
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(model->bindings) && iree_status_is_ok(status); ++i) {
    const iree_device_size_t length =
        i == 0 ? model->input_capacity : model->output.data_length;
    status = iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(model->owner),
        iree_hal_make_pool_reservation_request(length, 1),
        iree_infinite_timeout(), &model->bindings[i].buffer);
    if (iree_status_is_ok(status)) {
      status = loom_serve_retirement_track(
          &model->retirement, &model->bindings[i].buffer, allocator);
    }
    model->bindings[i].length = length;
  }
  IREE_RETURN_IF_ERROR(status);
  return image_create_control(model, module);
}

static iree_status_t image_model_write_residency(
    const loom_serve_image_model_t* model, loom_output_stream_t* output) {
  uint64_t workspace_length = 0;
  uint64_t workspace_alignment = 0;
  uint64_t executable_count = 0;
  uint64_t entry_count = 0;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(model->stages.values[i].compiled);
    workspace_length = iree_max(
        workspace_length, program->requirements.transient.required_byte_length);
    workspace_alignment = iree_max(
        workspace_alignment, program->requirements.transient.minimum_alignment);
    executable_count += program->requirements.executable_count;
    entry_count += program->requirements.entry_count;
  }
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  uint64_t parameter_bytes = 0;
  for (uint32_t i = 0; i < program->requirements.fixed_buffer_count; ++i) {
    parameter_bytes +=
        loom_cmd_program_parameter_root_at(program, i).required_byte_length;
  }
  const uint64_t input_bytes = model->input_capacity;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      "{\"event\":\"image_residency\",\"parameter_bytes\":%" PRIu64
      ",\"input_bytes\":%" PRIu64
      ",\"output_bytes\":%zu"
      ",\"workspace_bytes\":%" PRIu64 ",\"workspace_alignment\":%" PRIu64
      ",\"kernels\":%" PRIu64 ",\"entries\":%" PRIu64 ",\"stages\":[",
      parameter_bytes, input_bytes, model->output.data_length, workspace_length,
      workspace_alignment, executable_count, entry_count));
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const image_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* reflected =
        loom_serve_jit_stage_program(stage->compiled);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output,
        "%s{\"stage\":%u,\"tag\":%" PRId64 ",\"workspace_bytes\":%" PRIu64
        ",\"workspace_alignment\":%" PRIu64 ",\"kernels\":%u,\"entries\":%u}",
        i ? "," : "", i,
        loom_serve_preparation_stage(model->preparation, i)->tag,
        reflected->requirements.transient.required_byte_length,
        reflected->requirements.transient.minimum_alignment,
        reflected->requirements.executable_count,
        reflected->requirements.entry_count));
  }
  return loom_output_stream_write_cstring(output, "]}\n");
}

typedef enum image_load_phase_e {
  IMAGE_LOAD_PREPARE,
  IMAGE_LOAD_STAGES,
  IMAGE_LOAD_WEIGHTS,
  IMAGE_LOAD_LINK,
  IMAGE_LOAD_COMPLETE,
} image_load_phase_t;

struct loom_serve_image_load_t {
  // One private bootstrap task, retired before any native compilation starts.
  loomc_task_t task;
  // Allocator owning this operation, independent of the transferred model.
  iree_allocator_t allocator;
  // Owned partial model, transferred only after successful terminal advance.
  loom_serve_image_model_t* model;
  // Borrowed immutable package inputs, retained by the registration owner.
  loom_serve_image_model_options_t options;
  // Owner-only next phase; callbacks never make shared-state transitions.
  image_load_phase_t phase;
  // Next stage or weight domain within the current phase.
  uint32_t index;
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
    loom_serve_image_load_wake_fn_t wake;
    // Borrowed notification payload, retained through terminal advance.
    void* user_data;
  } event;
};

static void image_load_complete(void* user_data, iree_status_t status) {
  loom_serve_image_load_t* load = user_data;
  iree_slim_mutex_lock(&load->event.mutex);
  load->event.status = status;
  load->event.pending = false;
  load->event.wake(load->event.user_data);
  iree_slim_mutex_unlock(&load->event.mutex);
}

static iree_status_t image_load_prepare(loom_serve_image_load_t* load) {
  loom_serve_image_model_t* model = load->model;
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(image_prepare(model, &load->options));
  model->output.data_length =
      (iree_host_size_t)load->options.height * load->options.width * 12;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, model->output.data_length, (void**)&model->output.data));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_path_resolve(load->options.sources.base_directory,
                              load->options.sources.control, allocator, &path));
  const iree_string_view_t roots[] = {IREE_SVL("prepare_request"),
                                      IREE_SVL("generate")};
  iree_status_t status = loom_serve_program_compile(
      model->control.environment, iree_make_cstring_view(path),
      IREE_ARRAYSIZE(roots), roots, &model->compilation, allocator,
      &load->module);
  iree_allocator_free(allocator, path);
  IREE_RETURN_IF_ERROR(status);
  return loom_serve_jit_create(loom_serve_device_handle(model->owner),
                               loom_serve_device_dispatch_queue(model->owner),
                               loom_serve_device_host(model->owner),
                               &load->options.sources, NULL,
                               &model->compilation, allocator, &model->jit);
}

static void image_load_execute(loomc_task_t* task,
                               loomc_host_size_t worker_ordinal) {
  (void)worker_ordinal;
  loom_serve_image_load_t* load = (loom_serve_image_load_t*)task;
  load->task_status = image_load_prepare(load);
}

static void image_load_release(loomc_task_t* task) {
  loom_serve_image_load_t* load = (loom_serve_image_load_t*)task;
  image_load_complete(load, load->task_status);
}

static const loomc_task_vtable_t image_load_vtable = {
    .execute = image_load_execute,
    .destroy = image_load_release,
};

static void image_load_compiled(void* user_data, iree_status_t status,
                                loom_serve_jit_stage_t* compiled) {
  loom_serve_image_load_t* load = user_data;
  load->model->stages.values[load->index - 1].compiled = compiled;
  image_load_complete(load, status);
}

iree_status_t loom_serve_image_load_create(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    loom_serve_image_load_wake_fn_t wake, void* user_data,
    loom_serve_image_load_t** out_load, iree_allocator_t host_allocator) {
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
  load->allocator = host_allocator;
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
    load->model->allocator = host_allocator;
    loom_serve_retirement_initialize(&load->model->retirement);
    load->model->owner = device;
    load->model->command_quantum = options->command_quantum;
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
  iree_status_t status = load->event.status;
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
        if (load->index < model->stages.count) {
          const loom_serve_preparation_stage_t* declaration =
              loom_serve_preparation_stage(model->preparation, load->index++);
          if (declaration->kind != LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
            status = iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "image stages require command parameter reflection");
            break;
          }
          iree_slim_mutex_lock(&load->event.mutex);
          load->event.pending = true;
          iree_slim_mutex_unlock(&load->event.mutex);
          status = loom_serve_jit_compile_async(model->jit, declaration->root,
                                                &declaration->config,
                                                image_load_compiled, load);
          submitted = iree_status_is_ok(status);
        } else {
          status = image_model_bind(model);
          load->index = 0;
          load->phase = IMAGE_LOAD_WEIGHTS;
        }
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
              load->options.sources.base_directory, parameter->path,
              parameter->policy, image_load_complete, load);
          submitted = iree_status_is_ok(status);
        } else {
          load->phase = IMAGE_LOAD_LINK;
        }
        break;
      case IMAGE_LOAD_LINK:
        status = image_model_link(model, load->module, load->command_mode);
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

iree_status_t loom_serve_image_load_destroy(loom_serve_image_load_t* load) {
  if (!load) {
    return iree_ok_status();
  }
  iree_status_t status = image_model_destroy_resources(load->model);
  if (iree_status_is_ok(status)) {
    iree_vm_module_release(load->module);
    iree_slim_mutex_deinitialize(&load->event.mutex);
    iree_allocator_free(load->allocator, load);
  }
  return status;
}

static void image_load_notify(void* user_data) {
  iree_notification_post((iree_notification_t*)user_data, IREE_ALL_WAITERS);
}

iree_status_t loom_serve_image_model_create(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    loom_serve_image_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_image_load_t* load = NULL;
  iree_status_t status = loom_serve_image_load_create(
      device, options, image_load_notify, &notification, &load, host_allocator);
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
    loom_output_stream_t output;
    loom_output_stream_for_file(stderr, &output);
    status = iree_status_join(
        status, loom_serve_compilation_write(loom_serve_image_load_report(load),
                                             &output));
    if (iree_status_is_ok(status)) {
      loom_output_stream_for_file(stdout, &output);
      status = image_model_write_residency(load->model, &output);
    }
    if (iree_status_is_ok(status)) {
      *out_model = loom_serve_image_load_take_model(load);
    }
    status = iree_status_join(status, loom_serve_image_load_destroy(load));
  }
  iree_notification_deinitialize(&notification);
  return status;
}

iree_string_view_t loom_serve_image_model_name(
    const loom_serve_image_model_t* model) {
  return model->name;
}

static iree_status_t image_prepare_request(loom_serve_image_model_t* model,
                                           iree_string_view_t prompt,
                                           uint64_t seed, float strength,
                                           uint32_t* out_stage,
                                           iree_vm_buffer_t** out_payload) {
  *out_payload = NULL;
  iree_vm_buffer_t* text = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)prompt.data, prompt.size),
      iree_vm_buffer_release_callback_null(), model->allocator, &text));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.state),
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_variant_from_i32((int32_t)model->stages.count),
      iree_vm_buffer_variant_from_ptr_move(&model->control.vm_types, &text),
      iree_vm_variant_from_i64((int64_t)seed),
      iree_vm_variant_from_f32(strength),
  };
  iree_vm_variant_t results[2] = {0};
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(model->control.program),
                     model->control.prepare_request,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  int32_t selected = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &selected);
  }
  if (iree_status_is_ok(status) &&
      (selected < 0 || (uint32_t)selected >= model->stages.count)) {
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
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

static iree_status_t image_submit(loom_serve_image_model_t* model) {
  const image_model_stage_t* stage =
      &model->stages.values[model->invocation.stage];
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32((int32_t)stage->parts.first_stage),
      iree_vm_variant_from_i32((int32_t)stage->parts.count),
      iree_vm_variant_from_i32(model->invocation.cursor),
      iree_vm_variant_from_i64((int64_t)model->output.data_length),
      iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                model->bindings[0].buffer),
      iree_hal_buffer_variant_from_ptr_borrowed(&model->control.hal_types,
                                                model->bindings[1].buffer),
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

iree_status_t loom_serve_image_model_deactivate(
    loom_serve_image_model_t* model) {
  return loom_serve_residency_deactivate(model->residency);
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
    float strength, loom_serve_image_prepared_t* out_prepared) {
  *out_prepared = (loom_serve_image_prepared_t){0};
  IREE_RETURN_IF_ERROR(image_prepare_request(model, prompt, seed, strength,
                                             &model->invocation.stage,
                                             &model->invocation.payload));
  model->invocation.cursor = 0;
  model->invocation.started = false;
  *out_prepared = (loom_serve_image_prepared_t){
      .stage = model->invocation.stage,
      .tag = loom_serve_preparation_stage(model->preparation,
                                          model->invocation.stage)
                 ->tag,
      .input_bytes = iree_vm_buffer_length(model->invocation.payload)};
  return iree_ok_status();
}

void loom_serve_image_model_discard_prepared_request(
    loom_serve_image_model_t* model) {
  iree_vm_buffer_release(model->invocation.payload);
  model->invocation.payload = NULL;
}

static iree_status_t image_begin(loom_serve_image_model_t* model) {
  iree_vm_buffer_t* payload = model->invocation.payload;
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      payload, 0, iree_vm_buffer_length(payload), &input));
  const loom_cmd_program_t* program = loom_serve_jit_stage_program(
      model->stages.values[model->invocation.stage].compiled);
  const iree_hal_pool_reservation_request_t workspace =
      iree_hal_make_pool_reservation_request(
          program->requirements.transient.required_byte_length,
          program->requirements.transient.minimum_alignment);
  IREE_RETURN_IF_ERROR(loom_serve_execution_alloca(
      model->execution, &workspace, &model->invocation.workspace));
  iree_hal_transfer_operation_t upload = {0};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input.data;
  upload.upload.target_buffer = model->bindings[0].buffer;
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
      status = image_begin(model);
    }
    if (iree_status_is_ok(status)) {
      status = image_submit(model);
    }
  }
  if (iree_status_is_ok(status) && model->invocation.cursor < 0) {
    uint64_t completion = 0;
    status = loom_serve_execution_dealloca(
        model->execution, model->invocation.workspace, &completion);
    iree_hal_buffer_release(model->invocation.workspace);
    model->invocation.workspace = NULL;
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
  return iree_make_const_byte_span(model->output.data,
                                   model->output.data_length);
}

iree_status_t loom_serve_image_model_generate(
    loom_serve_image_model_t* model, iree_string_view_t prompt, uint64_t seed,
    float strength, loom_serve_image_prepared_t* out_prepared,
    iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_serve_image_model_prepare_request(
      model, prompt, seed, strength, out_prepared));
  iree_status_t status = loom_serve_residency_acquire(model->residency);
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

static iree_status_t image_generator_admit(
    void* self, loom_serve_residency_admission_t* out_admission) {
  return loom_serve_residency_try_admit(loom_serve_image_model_residency(self),
                                        out_admission);
}

static iree_status_t image_generator_prepare(
    void* self, const loom_serve_image_request_t* request,
    loom_serve_image_prepared_t* out_prepared) {
  return loom_serve_image_model_prepare_request(
      self, iree_string_builder_view(&request->prompt), request->seed,
      request->strength, out_prepared);
}

static iree_status_t image_generator_submit(
    void* self, loom_serve_image_submit_flags_t flags,
    loom_serve_execution_frontier_t* out_frontier, bool* out_complete) {
  return loom_serve_image_model_submit(self, flags, out_frontier, out_complete);
}

static void image_generator_discard(void* self) {
  loom_serve_residency_cancel_admission(loom_serve_image_model_residency(self));
  loom_serve_image_model_discard_prepared_request(self);
}

static iree_const_byte_span_t image_generator_complete(void* self) {
  return loom_serve_image_model_complete(self);
}

static loom_serve_weight_activity_t image_generator_weight_activity(
    void* self) {
  return loom_serve_image_model_weight_activity(self);
}

loom_serve_image_generator_t loom_serve_image_model_generator(
    loom_serve_image_model_t* model) {
  return (loom_serve_image_generator_t){
      .self = model,
      .prepare = image_generator_prepare,
      .discard_prepared_request = image_generator_discard,
      .try_admit = image_generator_admit,
      .submit = image_generator_submit,
      .complete = image_generator_complete,
      .weight_activity = image_generator_weight_activity,
      .tokenizer = model->tokenizer,
      .capacity_domain = loom_serve_residency_cache(model->residency)};
}
