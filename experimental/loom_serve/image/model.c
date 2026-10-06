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
#include "experimental/loom_serve/runtime/module.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/program.h"
#include "experimental/loom_serve/runtime/weights.h"
#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

typedef struct image_model_stage_t {
  // Compiled image root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Reusable command retaining the model's immutable parameter domains.
  iree_hal_command_buffer_t* command;
} image_model_stage_t;

struct loom_serve_image_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
  // Owned public identifier returned by the source bootstrap.
  iree_string_view_t name;
  // Maximum opaque request byte length returned by the source bootstrap.
  iree_host_size_t input_capacity;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Shared device/timeline ownership, outliving all accepted work.
  loom_serve_device_t* owner;
  // Optional profiling session, ended after accepted work drains.
  iree_hal_profiling_from_flags_t* profiling;
  // Cold live-source compiler and its task pool.
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
  // Inputs, final RGB and reflected workspace; partial slots are NULL.
  iree_hal_buffer_binding_t bindings[3];
  // Completed NCHW F32 RGB feedback, overwritten by the next generation.
  iree_byte_span_t output;
};

iree_status_t loom_serve_image_model_destroy(loom_serve_image_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (model->owner) {
    status =
        loom_serve_execution_drain(loom_serve_device_execution(model->owner));
  }
  status = iree_status_join(
      status, iree_hal_end_profiling_from_flags(model->profiling));
  loom_serve_program_destroy(model->control.program);
  iree_vm_buffer_release(model->control.tags);
  iree_vm_buffer_release(model->control.state);
  iree_vm_environment_free(model->control.environment);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    iree_hal_command_buffer_release(model->stages.values[i].command);
    loom_serve_jit_stage_destroy(model->stages.values[i].compiled);
  }
  iree_allocator_free(model->allocator, model->stages.values);
  loom_serve_preparation_destroy(model->preparation);
  for (iree_host_size_t i = 0; i < model->weights.count; ++i) {
    iree_hal_buffer_release(model->weights.values[i]);
    if (model->weights.plans) {
      status = iree_status_join(
          status, loom_serve_weights_destroy(model->weights.plans[i]));
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->bindings); ++i) {
    iree_hal_buffer_release(model->bindings[i].buffer);
  }
  iree_allocator_free(model->allocator, model->weights.values);
  iree_allocator_free(model->allocator, model->weights.plans);
  iree_allocator_free(model->allocator, (void*)model->name.data);
  loom_serve_jit_destroy(model->jit);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model->output.data);
  loom_serve_device_destroy(model->owner);
  iree_allocator_free(model->allocator, model);
  return status;
}

static iree_status_t image_load_tokenizer(loom_serve_image_model_t* model,
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
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("prepare.loom"), allocator, &path));
  iree_vm_module_t* input = NULL;
  iree_status_t status = loom_serve_input_module_create(
      model->control.environment, NULL, &input, allocator);
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64(options->height),
      iree_vm_variant_from_i64(options->width),
      iree_vm_variant_from_i64(options->text_tokens),
      {0},
      {0}};
  const iree_string_view_t paths[] = {options->checkpoint_directory,
                                      options->adapter_path};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(paths) && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t* buffer = NULL;
    status = iree_vm_buffer_wrap(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_byte_span((void*)paths[i].data, paths[i].size),
        iree_vm_buffer_release_callback_null(), allocator, &buffer);
    if (iree_status_is_ok(status)) {
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
        (iree_vm_module_span_t){&input, 1}, &model->preparation, allocator);
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
    status =
        image_load_tokenizer(model, options->checkpoint_directory,
                             iree_make_string_view((const char*)bytes[0].data,
                                                   bytes[0].data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(input);
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
                                          iree_string_view_t directory) {
  const iree_allocator_t allocator = model->allocator;
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
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
    stages[i] = (loom_serve_stage_t){model->stages.values[i].command, 3};
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = 3,
      .stages = {model->stages.count, stages},
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
    status = loom_serve_module_create(&model->control.hal_types,
                                      loom_serve_device_execution(model->owner),
                                      options, allocator, &native_modules[0]);
  }
  iree_allocator_free(allocator, stages);
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(model->control.environment,
                                            model->tokenizer,
                                            &native_modules[1], allocator);
  }
  char* path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_file_path_join(directory, IREE_SV("control.loom"), allocator,
                                 &path);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t roots[] = {IREE_SVL("prepare_request"),
                                        IREE_SVL("generate")};
    status = loom_serve_program_create(
        model->control.environment, iree_make_cstring_view(path),
        IREE_ARRAYSIZE(roots), roots,
        (iree_vm_module_span_t){native_modules, 2}, allocator,
        &model->control.program);
  }
  iree_allocator_free(allocator, path);
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

static iree_status_t image_model_initialize(
    loom_serve_image_model_t* model,
    const loom_serve_image_model_options_t* options) {
  const iree_allocator_t allocator = model->allocator;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  const iree_hal_command_buffer_mode_t command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  IREE_RETURN_IF_ERROR(image_prepare(model, options));
  model->output.data_length =
      (iree_host_size_t)options->height * options->width * 12;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, model->output.data_length, (void**)&model->output.data));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create(IREE_SV("amdgpu"), allocator, &model->owner));
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(device, dispatch,
                                             options->source_directory, NULL,
                                             allocator, &model->jit));
  uint64_t workspace_length = 0;
  uint64_t workspace_alignment = 0;
  uint64_t executable_count = 0;
  uint64_t entry_count = 0;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    image_model_stage_t* stage = &model->stages.values[i];
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    IREE_RETURN_IF_ERROR(loom_serve_jit_compile(
        model->jit, declaration->root, &declaration->config, &stage->compiled));
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    workspace_length = iree_max(
        workspace_length, program->requirements.transient.required_byte_length);
    workspace_alignment = iree_max(
        workspace_alignment, program->requirements.transient.minimum_alignment);
    executable_count += program->requirements.executable_count;
    entry_count += program->requirements.entry_count;
  }
  IREE_RETURN_IF_ERROR(image_check_layouts(model));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const uint32_t roots = program->requirements.fixed_buffer_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, roots, sizeof(*model->weights.values),
      (void**)&model->weights.values));
  model->weights.count = roots;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, roots, sizeof(*model->weights.plans),
      (void**)&model->weights.plans));
  const iree_host_size_t binding_count = 3;
  uint64_t parameter_bytes = 0;
  for (uint32_t i = 0; i < roots; ++i) {
    parameter_bytes +=
        loom_cmd_program_parameter_root_at(program, i).required_byte_length;
  }
  const uint64_t input_bytes = model->input_capacity;
  printf("{\"event\":\"image_residency\",\"parameter_bytes\":%" PRIu64
         ",\"input_bytes\":%" PRIu64
         ",\"output_bytes\":%zu"
         ",\"workspace_bytes\":%" PRIu64 ",\"workspace_alignment\":%" PRIu64
         ",\"kernels\":%" PRIu64 ",\"entries\":%" PRIu64 ",\"stages\":[",
         parameter_bytes, input_bytes, model->output.data_length,
         workspace_length, workspace_alignment, executable_count, entry_count);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const image_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* reflected =
        loom_serve_jit_stage_program(stage->compiled);
    printf("%s{\"stage\":%u,\"tag\":%" PRId64 ",\"workspace_bytes\":%" PRIu64
           ",\"workspace_alignment\":%" PRIu64
           ",\"kernels\":%u,\"entries\":%u}",
           i ? "," : "", i,
           loom_serve_preparation_stage(model->preparation, i)->tag,
           reflected->requirements.transient.required_byte_length,
           reflected->requirements.transient.minimum_alignment,
           reflected->requirements.executable_count,
           reflected->requirements.entry_count);
  }
  printf("]}\n");
  fflush(stdout);
  IREE_RETURN_IF_ERROR(iree_hal_begin_device_group_profiling_from_flags(
      loom_serve_device_group(model->owner), allocator, &model->profiling));
  const loom_serve_preparation_stage_t* declaration =
      loom_serve_preparation_stage(model->preparation, 0);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < roots && iree_status_is_ok(status); ++i) {
    const loom_serve_preparation_parameter_t* parameter =
        &declaration->parameters[i];
    char* policy = NULL;
    status = iree_file_path_join(options->source_directory, parameter->policy,
                                 allocator, &policy);
    if (iree_status_is_ok(status)) {
      const loom_cmd_program_parameter_root_t reflected =
          loom_cmd_program_parameter_root_at(program, i);
      const loom_serve_weight_root_t root = {
          program, reflected,
          &model->weights.values[reflected.fixed_buffer_index]};
      status = loom_serve_weights_create(
          device, loom_serve_device_transfer_queue(model->owner), dispatch,
          NULL, model->jit, command_mode, 0, 1, &root, parameter->path,
          iree_make_cstring_view(policy), &model->weights.plans[i], allocator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_weights_activate(model->weights.plans[i]);
    }
    iree_allocator_free(allocator, policy);
  }
  IREE_RETURN_IF_ERROR(status);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    image_model_stage_t* stage = &model->stages.values[i];
    IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(dispatch), command_mode,
        model->weights.values, &stage->command));
  }
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const bool workspace = i == 2;
    const iree_device_size_t length = i == 0      ? model->input_capacity
                                      : workspace ? workspace_length
                                                  : model->output.data_length;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment = workspace ? workspace_alignment : 256;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, length,
        &model->bindings[i].buffer);
    model->bindings[i].length = length;
  }
  IREE_RETURN_IF_ERROR(status);
  return image_create_control(model, options->source_directory);
}

iree_status_t loom_serve_image_model_create(
    const loom_serve_image_model_options_t* options,
    loom_serve_image_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  if (!options->height || !options->width ||
      (uint64_t)options->height * options->width > SIZE_MAX / 12) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "output RGB geometry must fit host storage");
  }
  loom_serve_image_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  iree_status_t status = image_model_initialize(model, options);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_image_model_destroy(model));
  }
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

static iree_status_t image_submit(loom_serve_image_model_t* model,
                                  uint32_t stage) {
  iree_vm_variant_t arguments[2 + 3] = {0};
  arguments[0] = iree_vm_variant_from_i32((int32_t)stage);
  arguments[1] = iree_vm_variant_from_i64((int64_t)model->output.data_length);
  for (iree_host_size_t i = 0; i < 3; ++i) {
    arguments[2 + i] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->control.hal_types, model->bindings[i].buffer);
  }
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.generate, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_empty());
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_image_model_generate(loom_serve_image_model_t* model,
                                              iree_string_view_t prompt,
                                              uint64_t seed, float strength,
                                              iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  const iree_time_t prepare_begin = iree_time_now();
  fprintf(stderr, "{\"event\":\"image_preparing\"}\n");
  iree_vm_buffer_t* payload = NULL;
  uint32_t stage_index = 0;
  iree_status_t status = image_prepare_request(model, prompt, seed, strength,
                                               &stage_index, &payload);
  IREE_RETURN_IF_ERROR(status);
  iree_const_byte_span_t input = iree_const_byte_span_empty();
  status = iree_vm_buffer_map_read(payload, 0, iree_vm_buffer_length(payload),
                                   &input);
  const iree_time_t prepare_end = iree_time_now();
  fprintf(stderr,
          "{\"event\":\"image_prepared\",\"prepare_ns\":%" PRId64
          ",\"stage\":%u,\"input_bytes\":%zu}\n",
          prepare_end - prepare_begin, stage_index, input.data_length);
  iree_hal_transfer_operation_t upload = {0};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input.data;
  upload.upload.target_buffer = model->bindings[0].buffer;
  upload.upload.length = input.data_length;
  loom_serve_execution_t* execution = loom_serve_device_execution(model->owner);
  uint64_t completion = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_transfer(execution, 1, &upload, &completion);
  }
  if (iree_status_is_ok(status)) {
    status = image_submit(model, stage_index);
  }
  const iree_time_t submit_end = iree_time_now();
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "{\"event\":\"image_submitted\",\"submit_ns\":%" PRId64 "}\n",
            submit_end - prepare_end);
  }
  // Native ownership follows accepted work, not a source-returned integer.
  // This also retires a successful upload preceding a later VM rejection.
  status = iree_status_join(status, loom_serve_execution_drain(execution));
  const iree_time_t completion_end = iree_time_now();
  if (iree_status_is_ok(status)) {
    *out_rgb = iree_make_const_byte_span(model->output.data,
                                         model->output.data_length);
  }
  iree_vm_buffer_release(payload);
  if (iree_status_is_ok(status)) {
    // Submission overlaps device work. The remaining wait includes queued
    // transfers and final readback, not an isolated GPU execution interval.
    fprintf(stderr,
            "{\"event\":\"image_execution\",\"prepare_ns\":%" PRId64
            ",\"submit_ns\":%" PRId64 ",\"completion_wait_ns\":%" PRId64 "}\n",
            prepare_end - prepare_begin, submit_end - prepare_end,
            completion_end - submit_end);
  }
  return status;
}
