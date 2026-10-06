// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/krea/model.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "experimental/loom_serve/models/krea/request.h"
#include "experimental/loom_serve/runtime/device.h"
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

typedef struct krea2_model_stage_t {
  // Retained text extent specialized into this command and its kernels.
  uint32_t text_tokens;
  // Compiled image root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Reusable command retaining the model's immutable parameter domains.
  iree_hal_command_buffer_t* command;
} krea2_model_stage_t;

struct loom_serve_krea2_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
  // Fixed pixel height for the prepared command.
  uint32_t height;
  // Fixed pixel width for the prepared command.
  uint32_t width;
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
    // Owned stage array; slot zero is the configured maximum prompt extent.
    krea2_model_stage_t* values;
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
    // Cold-resolved stage-selection function.
    iree_vm_function_t select;
    // Cold-resolved image-submission and feedback function.
    iree_vm_function_t generate;
  } control;
  // Encoder, Turbo, optional adapter and VAE fixed buffers.
  iree_hal_buffer_t* weights[4];
  // Inputs, final RGB and reflected workspace; partial slots are NULL.
  iree_hal_buffer_binding_t bindings[LOOM_SERVE_KREA2_INPUT_COUNT + 2];
  // Number of input slots; the adapted command has the extra strength slot.
  iree_host_size_t input_count;
  // Completed NCHW F32 RGB feedback, overwritten by the next generation.
  iree_byte_span_t output;
};

iree_status_t loom_serve_krea2_model_destroy(loom_serve_krea2_model_t* model) {
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
  iree_vm_environment_free(model->control.environment);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    iree_hal_command_buffer_release(model->stages.values[i].command);
    loom_serve_jit_stage_destroy(model->stages.values[i].compiled);
  }
  iree_allocator_free(model->allocator, model->stages.values);
  loom_serve_preparation_destroy(model->preparation);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->weights); ++i) {
    iree_hal_buffer_release(model->weights[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->bindings); ++i) {
    iree_hal_buffer_release(model->bindings[i].buffer);
  }
  loom_serve_jit_destroy(model->jit);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model->output.data);
  loom_serve_device_destroy(model->owner);
  iree_allocator_free(model->allocator, model);
  return status;
}

static iree_status_t krea2_load_tokenizer(loom_serve_krea2_model_t* model,
                                          iree_string_view_t directory) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      directory, IREE_SV("tokenizer/tokenizer.json"), model->allocator, &path));
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

static iree_status_t krea2_prepare(
    loom_serve_krea2_model_t* model,
    const loom_serve_krea2_model_options_t* options) {
  const iree_allocator_t allocator = model->allocator;
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("prepare.loom"), allocator, &path));
  iree_vm_environment_t* environment = NULL;
  iree_status_t status = iree_vm_environment_allocate(allocator, &environment);
  iree_vm_ref_types_t types = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64(options->height),
      iree_vm_variant_from_i64(options->width),
      iree_vm_variant_from_i64(options->text_tokens),
      {0},
      {0},
  };
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
      arguments[3 + i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_create(
        environment, iree_make_cstring_view(path), IREE_SV("prepare"),
        iree_vm_variant_span_from_array(arguments), &model->preparation,
        allocator);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_environment_free(environment);
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
  for (uint32_t i = 0; i < model->stages.count && iree_status_is_ok(status);
       ++i) {
    const int64_t tag =
        loom_serve_preparation_stage(model->preparation, i)->tag;
    if (tag <= 0 || tag > UINT32_MAX ||
        (i == 0 && tag != options->text_tokens)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "stage %u must name a valid text extent; "
                                "stage zero must use the requested maximum",
                                i);
    } else {
      model->stages.values[i].text_tokens = (uint32_t)tag;
    }
  }
  return status;
}

// The source catalog is external. Every retained shape must describe the same
// immutable placement before a single checkpoint byte is loaded or shared.
static iree_status_t krea2_check_layouts(loom_serve_krea2_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const bool adapted = model->input_count == LOOM_SERVE_KREA2_INPUT_COUNT;
  const iree_host_size_t binding_count = model->input_count + 2;
  const uint32_t roots = adapted ? 4 : 3;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const krea2_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(model->preparation, i);
    if (program->requirements.rebindable_binding_count != binding_count ||
        program->requirements.transient.binding_index !=
            model->input_count + 1 ||
        program->requirements.fixed_buffer_count != roots ||
        program->parameter_roots.count != roots ||
        declaration->parameter_count != roots ||
        program->requirements.launch_counts.binding_index != UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "text%u source root does not implement the Krea image ABI",
          stage->text_tokens);
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
            "text%u must declare the same ordered checkpoint domains",
            stage->text_tokens);
      }
    }
    if (!i) {
      continue;
    }
    if (canonical->parameters.count != program->parameters.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "text%u parameter count differs from maximum",
                              stage->text_tokens);
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_cmd_program_parameter_root_t lhs =
          loom_cmd_program_parameter_root_at(canonical, root);
      const loom_cmd_program_parameter_root_t rhs =
          loom_cmd_program_parameter_root_at(program, root);
      if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.required_byte_length != rhs.required_byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "text%u parameter root %u differs from maximum",
                                stage->text_tokens, root);
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
            "text%u parameter placement %u differs from maximum",
            stage->text_tokens, parameter);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t krea2_create_control(loom_serve_krea2_model_t* model,
                                          iree_string_view_t directory) {
  const iree_allocator_t allocator = model->allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(allocator, &model->control.environment));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(model->control.environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &model->control.hal_types));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    model->control.environment, IREE_SV("vm")),
                                &model->control.vm_types));
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
    stages[i] = (loom_serve_stage_t){model->stages.values[i].command,
                                     (uint16_t)(model->input_count + 2)};
    iree_unaligned_store_le_u64(
        tags.data + i * sizeof(int64_t),
        (uint64_t)loom_serve_preparation_stage(model->preparation, i)->tag);
  }
  const loom_serve_module_options_t options = {
      .binding_capacity = LOOM_SERVE_KREA2_INPUT_COUNT + 2,
      .stages = {model->stages.count, stages},
      .feedback = {1, &model->output},
  };
  iree_vm_module_t* native_module = NULL;
  iree_vm_buffer_t* read_only_tags = NULL;
  iree_status_t status = iree_vm_buffer_subspan(
      model->control.tags, 0, tags.data_length, IREE_VM_BUFFER_ACCESS_FLAG_READ,
      allocator, &read_only_tags);
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_release(model->control.tags);
    model->control.tags = read_only_tags;
    status = loom_serve_module_create(&model->control.hal_types,
                                      loom_serve_device_execution(model->owner),
                                      options, allocator, &native_module);
  }
  iree_allocator_free(allocator, stages);
  char* path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_file_path_join(directory, IREE_SV("control.loom"), allocator,
                                 &path);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t roots[] = {IREE_SVL("select_stage"),
                                        IREE_SVL("generate")};
    status = loom_serve_program_create(
        model->control.environment, iree_make_cstring_view(path),
        IREE_ARRAYSIZE(roots), roots,
        (iree_vm_module_span_t){&native_module, 1}, allocator,
        &model->control.program);
  }
  iree_allocator_free(allocator, path);
  iree_vm_module_release(native_module);
  IREE_RETURN_IF_ERROR(status);
  iree_vm_process_t* process =
      loom_serve_program_process(model->control.program);
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("select_stage"),
      &model->control.select));
  return iree_vm_process_lookup_function(
      process, IREE_SV("model"), IREE_SV("generate"), &model->control.generate);
}

static iree_status_t krea2_model_initialize(
    loom_serve_krea2_model_t* model,
    const loom_serve_krea2_model_options_t* options,
    iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]) {
  const iree_allocator_t allocator = model->allocator;
  const bool adapted = options->adapter_path.size != 0;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  const iree_hal_command_buffer_mode_t command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  IREE_RETURN_IF_ERROR(
      krea2_load_tokenizer(model, options->checkpoint_directory));
  IREE_RETURN_IF_ERROR(krea2_prepare(model, options));
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
    krea2_model_stage_t* stage = &model->stages.values[i];
    iree_host_size_t stage_sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
    IREE_RETURN_IF_ERROR(loom_serve_krea2_request_measure(
        options->height, options->width, stage->text_tokens, stage_sizes));
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
    for (iree_host_size_t input = 0; input < IREE_ARRAYSIZE(stage_sizes);
         ++input) {
      sizes[input] = iree_max(sizes[input], stage_sizes[input]);
    }
  }
  IREE_RETURN_IF_ERROR(krea2_check_layouts(model));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const uint32_t roots = adapted ? 4 : 3;
  const iree_host_size_t binding_count = model->input_count + 2;
  uint64_t parameter_bytes = 0;
  for (uint32_t i = 0; i < roots; ++i) {
    parameter_bytes +=
        loom_cmd_program_parameter_root_at(program, i).required_byte_length;
  }
  uint64_t input_bytes = 0;
  for (iree_host_size_t i = 0; i < model->input_count; ++i) {
    input_bytes +=
        sizes[i + (!adapted && i >= LOOM_SERVE_KREA2_INPUT_STRENGTH)];
  }
  printf("{\"event\":\"image_residency\",\"parameter_bytes\":%" PRIu64
         ",\"input_bytes\":%" PRIu64
         ",\"output_bytes\":%zu"
         ",\"workspace_bytes\":%" PRIu64 ",\"workspace_alignment\":%" PRIu64
         ",\"kernels\":%" PRIu64 ",\"entries\":%" PRIu64 ",\"stages\":[",
         parameter_bytes, input_bytes, model->output.data_length,
         workspace_length, workspace_alignment, executable_count, entry_count);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const krea2_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* reflected =
        loom_serve_jit_stage_program(stage->compiled);
    printf("%s{\"text_tokens\":%u,\"workspace_bytes\":%" PRIu64
           ",\"workspace_alignment\":%" PRIu64
           ",\"kernels\":%u,\"entries\":%u}",
           i ? "," : "", stage->text_tokens,
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
          program, reflected, &model->weights[reflected.fixed_buffer_index]};
      status = loom_serve_weights_load(
          device, loom_serve_device_transfer_queue(model->owner), dispatch,
          model->jit, command_mode, 0, 1, &root, parameter->path,
          iree_make_cstring_view(policy), allocator);
    }
    iree_allocator_free(allocator, policy);
  }
  IREE_RETURN_IF_ERROR(status);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    krea2_model_stage_t* stage = &model->stages.values[i];
    IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(dispatch), command_mode,
        model->weights, &stage->command));
  }
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const bool workspace = i == model->input_count + 1;
    const iree_device_size_t length =
        i < model->input_count
            ? sizes[i + (!adapted && i >= LOOM_SERVE_KREA2_INPUT_STRENGTH)]
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
  return krea2_create_control(model, options->source_directory);
}

iree_status_t loom_serve_krea2_model_create(
    const loom_serve_krea2_model_options_t* options,
    loom_serve_krea2_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
  IREE_RETURN_IF_ERROR(loom_serve_krea2_request_measure(
      options->height, options->width, options->text_tokens, sizes));
  loom_serve_krea2_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  model->height = options->height;
  model->width = options->width;
  model->input_count =
      LOOM_SERVE_KREA2_INPUT_COUNT - !options->adapter_path.size;
  iree_status_t status = krea2_model_initialize(model, options, sizes);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_krea2_model_destroy(model));
  }
  return status;
}

static iree_status_t krea2_select_stage(loom_serve_krea2_model_t* model,
                                        uint32_t token_count,
                                        uint32_t* out_stage) {
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&model->control.vm_types,
                                               model->control.tags),
      iree_vm_variant_from_i32((int32_t)model->stages.count),
      iree_vm_variant_from_i32((int32_t)token_count),
  };
  iree_vm_variant_t results[1] = {0};
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.select, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_from_array(results));
  int32_t selected = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &selected);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status) &&
      (selected < 0 || (uint32_t)selected >= model->stages.count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "model selected undeclared stage %d", selected);
  }
  if (iree_status_is_ok(status)) {
    *out_stage = (uint32_t)selected;
  }
  return status;
}

static iree_status_t krea2_submit(loom_serve_krea2_model_t* model,
                                  uint32_t stage) {
  const bool adapted = model->input_count == LOOM_SERVE_KREA2_INPUT_COUNT;
  iree_vm_variant_t arguments[3 + LOOM_SERVE_KREA2_INPUT_COUNT + 2] = {0};
  arguments[0] = iree_vm_variant_from_i32((int32_t)stage);
  arguments[1] = iree_vm_variant_from_i32(adapted);
  arguments[2] = iree_vm_variant_from_i64((int64_t)model->output.data_length);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_KREA2_INPUT_COUNT + 2; ++i) {
    iree_hal_buffer_t* buffer = NULL;
    if (adapted || i != LOOM_SERVE_KREA2_INPUT_STRENGTH) {
      buffer =
          model->bindings[i - (!adapted && i > LOOM_SERVE_KREA2_INPUT_STRENGTH)]
              .buffer;
    }
    arguments[3 + i] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->control.hal_types, buffer);
  }
  iree_status_t status = iree_vm_invoke(
      loom_serve_program_invocation(model->control.program),
      model->control.generate, iree_vm_variant_span_from_array(arguments),
      iree_vm_variant_span_empty());
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  return status;
}

iree_status_t loom_serve_krea2_model_generate(loom_serve_krea2_model_t* model,
                                              iree_string_view_t prompt,
                                              uint64_t seed, float strength,
                                              iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  const bool adapted = model->input_count == LOOM_SERVE_KREA2_INPUT_COUNT;
  if (!isfinite(strength)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "adapter strength must be finite");
  }
  if (!adapted && strength != 1.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "strength requires an adapter");
  }
  const iree_time_t prepare_begin = iree_time_now();
  fprintf(stderr, "{\"event\":\"image_preparing\"}\n");
  loom_serve_krea2_prompt_t* prepared = NULL;
  loom_serve_krea2_request_t* request = NULL;
  const krea2_model_stage_t* stage = &model->stages.values[0];
  uint32_t stage_index = 0;
  uint32_t token_count = 0;
  iree_status_t status =
      loom_serve_krea2_prompt_create(model->tokenizer, stage->text_tokens,
                                     prompt, &prepared, model->allocator);
  if (iree_status_is_ok(status)) {
    token_count = loom_serve_krea2_prompt_token_count(prepared);
    status = krea2_select_stage(model, token_count, &stage_index);
  }
  if (iree_status_is_ok(status)) {
    stage = &model->stages.values[stage_index];
    const loom_serve_krea2_request_options_t options = {
        model->height, model->width, stage->text_tokens, seed, strength};
    status = loom_serve_krea2_request_create(prepared, options, &request,
                                             model->allocator);
  }
  loom_serve_krea2_prompt_destroy(prepared);
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t prepare_end = iree_time_now();
  fprintf(stderr,
          "{\"event\":\"image_prepared\",\"prepare_ns\":%" PRId64
          ",\"text_tokens\":%u,\"prefix_prompt_tokens\":%u}\n",
          prepare_end - prepare_begin, stage->text_tokens, token_count);
  iree_hal_transfer_operation_t uploads[LOOM_SERVE_KREA2_INPUT_COUNT] = {0};
  for (iree_host_size_t i = 0; i < model->input_count; ++i) {
    const loom_serve_krea2_input_t kind =
        (loom_serve_krea2_input_t)(i + (!adapted &&
                                        i >= LOOM_SERVE_KREA2_INPUT_STRENGTH));
    const iree_const_byte_span_t input =
        loom_serve_krea2_request_input(request, kind);
    uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i].upload.source = input.data;
    uploads[i].upload.target_buffer = model->bindings[i].buffer;
    uploads[i].upload.length = input.data_length;
  }
  loom_serve_execution_t* execution = loom_serve_device_execution(model->owner);
  uint64_t completion = 0;
  status = loom_serve_execution_transfer(execution, model->input_count, uploads,
                                         &completion);
  if (iree_status_is_ok(status)) {
    status = krea2_submit(model, stage_index);
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
  loom_serve_krea2_request_destroy(request);
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
