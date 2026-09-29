// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single-row Qwen3.8-27B bring-up through shared VM/native command execution.
// Model math and tensor placements come from independently compiled stages.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/command.h"
#include "experimental/loom_serve/execution.h"
#include "experimental/loom_serve/module.h"
#include "experimental/loom_serve/qwen_model_data.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/init.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, prefill, "",
          "Directory containing compiled prefill artifacts.");
IREE_FLAG(string, decode, "",
          "Directory containing compiled decode artifacts.");
IREE_FLAG(string, weights, "", "Canonical Qwen3.8-27B UD-Q5_K_XL GGUF path.");
IREE_FLAG(string, tokenizer, "", "HuggingFace tokenizer.json path.");
IREE_FLAG(string, prompt, "What is 2+2? Answer with one number.",
          "User text wrapped as a no-thinking Qwen turn.");
IREE_FLAG(
    int32_t, prefill_tokens, 24,
    "Exact prompt length specialized into the supplied prefill artifact.");
IREE_FLAG(int32_t, max_tokens, 16,
          "Maximum greedy continuation length (1-128).");

enum {
  QWEN_RESIDUAL = 0,
  QWEN_CONTROL = 1,
  QWEN_GDN = 2,
  QWEN_ATTENTION = 3,
  QWEN_TOKENS = 4,
  QWEN_PROGRESS = 5,
  QWEN_WORKSPACE = 6,
  QWEN_BINDING_COUNT = 7,
  QWEN_CONTEXT_CAPACITY = 512,
  QWEN_GENERATION_CAPACITY = 128,
};

typedef struct qwen_stage_t {
  // Owned immutable artifact manifest bytes.
  iree_io_file_contents_t* manifest;
  // Owned portable command bytes backing program.
  iree_io_file_contents_t* contents;
  // Parsed view borrowing contents.
  loom_cmd_program_t program;
  // Reusable command retaining its executable and fixed weight resources.
  iree_hal_command_buffer_t* command;
} qwen_stage_t;

typedef struct qwen_runner_t {
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
  // Ordered stage and host-I/O submission domain.
  loom_serve_execution_t* execution;
  // Prefill and decode artifacts, in that order.
  qwen_stage_t stages[2];
  // Single resident parameter slab shared by both stages.
  iree_hal_buffer_t* weights;
  // One retained row and shared scratch, in program binding order.
  iree_hal_buffer_t* buffers[QWEN_BINDING_COUNT];
  // Environment owning canonical VM reference types.
  iree_vm_environment_t* environment;
  // HAL reference type handles borrowed from environment.
  iree_hal_module_types_t types;
  // Native stage submission module retained by program.
  iree_vm_module_t* native_module;
  // Compiled model-control module retained by program.
  iree_vm_module_t* bytecode_module;
  // Shared linked model program.
  iree_vm_program_t* program;
  // Fixed invocation backing, independent of token count.
  iree_alignas(iree_alignof(iree_max_align_t)) uint8_t
      invocation_storage[16384];
  // Invocation borrowing invocation_storage.
  iree_vm_invocation_t* invocation;
  // One model process; request state is not VM process state.
  iree_vm_process_t* process;
  // Cold-resolved model step function.
  iree_vm_function_t step;
  // Tokenizer owning its parsed vocabulary and encoding data.
  iree_tokenizer_t* tokenizer;
  // Host transfer storage retained until execution is drained.
  struct {
    // Prefill count/base/EOS; overwritten by decode position/count/EOS.
    int32_t control[3];
    // Prompt input, then selected token and generated-token history.
    int32_t tokens[QWEN_CONTEXT_CAPACITY];
    // Generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } row;
} qwen_runner_t;

static iree_string_view_t qwen_file_text(iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

// Reads the fixed artifact-directory layout; metadata contains names, not code.
static iree_status_t qwen_read_artifact(
    iree_string_view_t directory, iree_string_view_t relative_path,
    iree_allocator_t allocator, iree_io_file_contents_t** out_contents) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_file_path_join(directory, relative_path, allocator, &path));
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(path), allocator, out_contents);
  iree_allocator_free(allocator, path);
  return status;
}

// Manifest names are bounded compiler-generated filenames/symbols. JSON
// unescaping remains at this external artifact boundary.
static iree_status_t qwen_json_name(iree_string_view_t object,
                                    iree_string_view_t key, char storage[1024],
                                    iree_string_view_t* out_name) {
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_lookup_string(
      object, key, iree_make_mutable_string_view(storage, 1024), &length));
  *out_name = iree_make_string_view(storage, length);
  return iree_ok_status();
}

static iree_status_t qwen_stage_manifest_program(
    const qwen_stage_t* stage, iree_string_view_t* out_program) {
  iree_string_view_t programs = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      qwen_file_text(stage->manifest), IREE_SV("programs"), &programs));
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(programs, &count));
  if (count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "each stage must contain exactly one program");
  }
  return iree_json_array_get(programs, 0, out_program);
}

static iree_status_t qwen_stage_read(iree_string_view_t directory,
                                     iree_allocator_t allocator,
                                     qwen_stage_t* stage) {
  IREE_RETURN_IF_ERROR(qwen_read_artifact(directory, IREE_SV("manifest.json"),
                                          allocator, &stage->manifest));
  iree_string_view_t manifest = qwen_file_text(stage->manifest);
  iree_string_view_t version = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      manifest, IREE_SV("schema_version"), &version));
  uint64_t schema_version = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(version, &schema_version));
  char name_storage[1024];
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      qwen_json_name(manifest, IREE_SV("format"), name_storage, &name));
  if (schema_version != 2 ||
      !iree_string_view_equal(name, IREE_SV("loom-command-set"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a version 2 loom-command-set manifest");
  }
  iree_string_view_t program = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(qwen_stage_manifest_program(stage, &program));
  IREE_RETURN_IF_ERROR(
      qwen_json_name(program, IREE_SV("artifact"), name_storage, &name));
  char* commands_directory = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(directory, IREE_SV("commands"),
                                           allocator, &commands_directory));
  iree_status_t status =
      qwen_read_artifact(iree_make_cstring_view(commands_directory), name,
                         allocator, &stage->contents);
  iree_allocator_free(allocator, commands_directory);
  if (iree_status_is_ok(status)) {
    status =
        loom_cmd_program_parse(stage->contents->const_buffer, &stage->program);
  }
  return status;
}

static iree_status_t qwen_stage_load_entry(
    qwen_runner_t* runner, iree_string_view_t directory,
    iree_string_view_t entry, const iree_hal_executable_target_t* target,
    loom_serve_command_entry_t* out_entry) {
  char name_storage[1024];
  iree_string_view_t request = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      qwen_json_name(entry, IREE_SV("source_request"), name_storage, &request));
  iree_string_builder_t path;
  iree_string_builder_initialize(runner->allocator, &path);
  const iree_string_view_t stem = iree_file_path_stem(request);
  iree_status_t status = iree_string_builder_append_format(
      &path, "kernels/%.*s.hsaco", (int)stem.size, stem.data);
  iree_io_file_contents_t* image = NULL;
  if (iree_status_is_ok(status)) {
    status = qwen_read_artifact(directory, iree_string_builder_view(&path),
                                runner->allocator, &image);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data = image->const_buffer;
    status = iree_hal_executable_load(iree_hal_queue_family(runner->dispatch),
                                      target, &params, &out_entry->executable);
  }
  iree_io_file_contents_free(image);
  iree_string_builder_deinitialize(&path);
  if (iree_status_is_ok(status)) {
    iree_string_view_t symbol = iree_string_view_empty();
    status = qwen_json_name(entry, IREE_SV("symbol"), name_storage, &symbol);
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_lookup_function_by_name(
          out_entry->executable, symbol, &out_entry->function);
    }
  }
  return status;
}

static iree_status_t qwen_stage_prepare(qwen_runner_t* runner,
                                        iree_string_view_t directory,
                                        qwen_stage_t* stage) {
  iree_string_view_t program = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(qwen_stage_manifest_program(stage, &program));
  iree_string_view_t requirements = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      program, IREE_SV("entry_requirements"), &requirements));
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(requirements, &count));
  if (count != stage->program.requirements.entry_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "manifest and command entry counts differ");
  }
  iree_string_view_t manifest_entries = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      qwen_file_text(stage->manifest), IREE_SV("entries"), &manifest_entries));
  iree_hal_executable_target_selection_t selection = {0};
  selection.family = IREE_SV("amdgpu");
  selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT;
  selection.physical_device_affinity =
      iree_hal_queue_family_spec(iree_hal_queue_family(runner->dispatch))
          ->physical_device_affinity;
  const iree_hal_executable_target_selection_result_t target =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(runner->device), &selection);
  if (target.outcome != IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no unambiguous AMDGPU executable target");
  }
  loom_serve_command_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      runner->allocator, count, sizeof(*entries), (void**)&entries));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_string_view_t ordinal_text = iree_string_view_empty();
    uint64_t ordinal = 0;
    status = iree_json_array_get(requirements, i, &ordinal_text);
    if (iree_status_is_ok(status)) {
      status = iree_json_parse_uint64(ordinal_text, &ordinal);
    }
    iree_string_view_t entry = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      status = iree_json_array_get(manifest_entries, ordinal, &entry);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_stage_load_entry(runner, directory, entry, target.target,
                                     &entries[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_command_create(
        iree_hal_queue_family(runner->dispatch), &stage->program, 1,
        &runner->weights, count, entries, runner->allocator, &stage->command);
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_executable_release(entries[i].executable);
  }
  iree_allocator_free(runner->allocator, entries);
  return status;
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

static iree_status_t qwen_create_device(qwen_runner_t* runner) {
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

static iree_status_t qwen_check_layouts(qwen_runner_t* runner) {
  const loom_cmd_program_t* prefill = &runner->stages[0].program;
  const loom_cmd_program_t* decode = &runner->stages[1].program;
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const loom_cmd_program_t* program = &runner->stages[i].program;
    if (program->requirements.fixed_buffer_count != 1 ||
        program->parameter_roots.count != 1 ||
        program->requirements.rebindable_binding_count != QWEN_BINDING_COUNT ||
        program->requirements.transient.binding_index != QWEN_WORKSPACE ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the Qwen row layout");
    }
  }
  if (prefill->parameters.count != decode->parameters.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage parameter counts differ");
  }
  for (uint32_t i = 0; i < prefill->parameters.count; ++i) {
    const loom_cmd_program_parameter_t lhs =
        loom_cmd_program_parameter_at(prefill, i);
    const loom_cmd_program_parameter_t rhs =
        loom_cmd_program_parameter_at(decode, i);
    if (!iree_string_view_equal(lhs.key, rhs.key) ||
        lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
        lhs.byte_offset != rhs.byte_offset ||
        lhs.byte_length != rhs.byte_length ||
        lhs.minimum_alignment != rhs.minimum_alignment) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter placement %u differs", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t qwen_allocate_buffer(qwen_runner_t* runner,
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

static iree_status_t qwen_parameter_span(void* user_data, iree_host_size_t i,
                                         iree_string_view_t* out_key,
                                         iree_io_parameter_span_t* out_span) {
  const loom_cmd_program_t* program = (const loom_cmd_program_t*)user_data;
  const loom_cmd_program_parameter_t parameter =
      loom_cmd_program_parameter_at(program, i);
  *out_key = parameter.key;
  *out_span = (iree_io_parameter_span_t){
      .buffer_offset = parameter.byte_offset,
      .length = parameter.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t qwen_load_weights(qwen_runner_t* runner) {
  const loom_cmd_program_t* program = &runner->stages[0].program;
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(program, 0);
  const loom_cmd_program_parameter_root_t decode_root =
      loom_cmd_program_parameter_root_at(&runner->stages[1].program, 0);
  const uint64_t length =
      iree_max(root.required_byte_length, decode_root.required_byte_length);
  const uint64_t alignment =
      iree_max(root.minimum_alignment, decode_root.minimum_alignment);
  IREE_RETURN_IF_ERROR(
      qwen_allocate_buffer(runner, length, alignment, &runner->weights));
  fprintf(stderr, "Loading %u parameters into one %.3f GiB weight slab...\n",
          program->parameters.count, length / 1073741824.0);
  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(runner->allocator, &index));
  iree_status_t status = iree_tooling_append_parameter_file_to_index(
      iree_make_cstring_view(FLAG_weights), index, runner->allocator);
  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_string_view_empty(), index, 4, runner->allocator, &provider);
  }
  iree_hal_semaphore_t* ready = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(runner->device,
                                       IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready);
  }
  uint64_t ready_value = 1;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_provider_gather(
        provider, runner->device, runner->transfer,
        iree_hal_semaphore_list_empty(),
        (iree_hal_semaphore_list_t){1, &ready, &ready_value},
        iree_string_view_empty(), runner->weights, program->parameters.count,
        (iree_io_parameter_enumerator_t){qwen_parameter_span, (void*)program});
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_semaphore_wait(ready, ready_value, iree_infinite_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE);
    }
  }
  iree_hal_semaphore_release(ready);
  iree_io_parameter_provider_release(provider);
  iree_io_parameter_index_release(index);
  return status;
}

static iree_status_t qwen_create_model(qwen_runner_t* runner) {
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(runner->allocator, &runner->environment));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(runner->environment, &table));
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &runner->types));
  const loom_serve_stage_t stages[] = {
      {IREE_SVL("prefill"), runner->stages[0].command, QWEN_BINDING_COUNT},
      {IREE_SVL("decode"), runner->stages[1].command, QWEN_BINDING_COUNT},
  };
  IREE_RETURN_IF_ERROR(loom_serve_module_create(
      &runner->types, runner->execution, IREE_ARRAYSIZE(stages), stages,
      runner->allocator, &runner->native_module));
  const struct iree_file_toc_t* image = loom_serve_qwen_model_data_create();
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      runner->environment, IREE_SV("model"),
      (iree_vm_bytecode_module_storage_t){
          iree_make_const_byte_span(image[0].data, image[0].size),
          iree_allocator_null()},
      runner->allocator, &runner->bytecode_module));
  iree_vm_module_t* libraries[] = {runner->native_module};
  IREE_RETURN_IF_ERROR(iree_vm_program_create(
      (iree_vm_program_modules_t){runner->bytecode_module,
                                  iree_vm_module_span_from_array(libraries)},
      runner->allocator, &runner->program));
  IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
      iree_make_byte_span(runner->invocation_storage,
                          sizeof(runner->invocation_storage)),
      &runner->invocation));
  IREE_RETURN_IF_ERROR(iree_vm_process_create(
      runner->program, runner->invocation, iree_vm_variant_span_empty(),
      runner->allocator, &runner->process));
  return iree_vm_process_lookup_function(runner->process, IREE_SV("model"),
                                         IREE_SV("step"), &runner->step);
}

static iree_status_t qwen_tokenize_prompt(qwen_runner_t* runner) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(
      iree_make_cstring_view(FLAG_tokenizer), IREE_IO_FILE_ACCESS_READ,
      runner->allocator, &contents));
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      qwen_file_text(contents), runner->allocator, &runner->tokenizer);
  iree_io_file_contents_free(contents);
  IREE_RETURN_IF_ERROR(status);
  runner->row.control[2] = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(runner->tokenizer), IREE_SV("<|im_end|>"));
  if (runner->row.control[2] < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no <|im_end|>");
  }
  iree_string_builder_t prompt;
  iree_string_builder_initialize(runner->allocator, &prompt);
  status = iree_string_builder_append_format(
      &prompt,
      "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
      "<think>\n\n</think>\n\n",
      FLAG_prompt);
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        runner->tokenizer, iree_string_builder_view(&prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(runner->row.tokens, NULL, NULL,
                                         QWEN_CONTEXT_CAPACITY),
        runner->allocator, &count);
  }
  iree_string_builder_deinitialize(&prompt);
  IREE_RETURN_IF_ERROR(status);
  if (count != (iree_host_size_t)FLAG_prefill_tokens) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prompt has %zu tokens; artifact specializes %d",
                            count, FLAG_prefill_tokens);
  }
  runner->row.control[0] = (int32_t)count;
  fprintf(stderr, "Prompt: %zu tokens; EOS: %d\n", count,
          runner->row.control[2]);
  return iree_ok_status();
}

static iree_status_t qwen_initialize_row(qwen_runner_t* runner) {
  const uint64_t workspace_length = iree_max(
      runner->stages[0].program.requirements.transient.required_byte_length,
      runner->stages[1].program.requirements.transient.required_byte_length);
  const uint64_t workspace_alignment = iree_max(
      runner->stages[0].program.requirements.transient.minimum_alignment,
      runner->stages[1].program.requirements.transient.minimum_alignment);
  const uint64_t lengths[QWEN_BINDING_COUNT] = {
      10485760,
      12,
      156893184,
      QWEN_CONTEXT_CAPACITY * 65536ull,
      sizeof(runner->row.tokens),
      sizeof(runner->row.progress),
      workspace_length,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < QWEN_BINDING_COUNT && iree_status_is_ok(status); ++i) {
    status = qwen_allocate_buffer(
        runner, lengths[i], i == QWEN_WORKSPACE ? workspace_alignment : 256,
        &runner->buffers[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t fills[QWEN_BINDING_COUNT] = {0};
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill.target_buffer = runner->buffers[i];
    fills[i].fill.length = lengths[i];
    fills[i].fill.pattern = &zero;
    fills[i].fill.pattern_length = sizeof(zero);
  }
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      runner->execution, IREE_ARRAYSIZE(fills), fills, &completion));
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = runner->row.control,
                  .target_buffer = runner->buffers[QWEN_CONTROL],
                  .length = sizeof(runner->row.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = runner->row.tokens,
                  .target_buffer = runner->buffers[QWEN_TOKENS],
                  .length = sizeof(runner->row.tokens)}},
  };
  return loom_serve_execution_transfer(
      runner->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
}

static iree_status_t qwen_step(qwen_runner_t* runner, int32_t initialize) {
  iree_vm_variant_t arguments[QWEN_BINDING_COUNT + 1] = {0};
  arguments[0] = iree_vm_variant_from_i32(initialize);
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    arguments[i + 1] = iree_hal_buffer_variant_from_ptr_borrowed(
        &runner->types, runner->buffers[i]);
  }
  iree_vm_variant_t results[1] = {0};
  iree_status_t status =
      iree_vm_invoke(runner->invocation, runner->step,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_RETURN_IF_ERROR(status);
  // Downloads wait on the accepted VM submission in the same execution domain.
  const iree_hal_transfer_operation_t downloads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = runner->buffers[QWEN_TOKENS],
                    .target = runner->row.tokens,
                    .length = sizeof(runner->row.tokens)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = runner->buffers[QWEN_PROGRESS],
                    .target = runner->row.progress,
                    .length = sizeof(runner->row.progress)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      runner->execution, IREE_ARRAYSIZE(downloads), downloads, &completion));
  return loom_serve_execution_wait(runner->execution, completion);
}

static iree_status_t qwen_run(qwen_runner_t* runner) {
  if (!FLAG_weights[0] || !FLAG_tokenizer[0] || !FLAG_prefill[0] ||
      !FLAG_decode[0] || FLAG_max_tokens < 1 ||
      FLAG_max_tokens > QWEN_GENERATION_CAPACITY || FLAG_prefill_tokens < 1 ||
      FLAG_prefill_tokens > QWEN_CONTEXT_CAPACITY - FLAG_max_tokens) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provide weights/tokenizer/stage paths and lengths "
                            "fitting the 512-token context / 128-token output");
  }
  IREE_RETURN_IF_ERROR(qwen_tokenize_prompt(runner));
  IREE_RETURN_IF_ERROR(qwen_stage_read(iree_make_cstring_view(FLAG_prefill),
                                       runner->allocator, &runner->stages[0]));
  IREE_RETURN_IF_ERROR(qwen_stage_read(iree_make_cstring_view(FLAG_decode),
                                       runner->allocator, &runner->stages[1]));
  IREE_RETURN_IF_ERROR(qwen_check_layouts(runner));
  IREE_RETURN_IF_ERROR(qwen_create_device(runner));
  IREE_RETURN_IF_ERROR(qwen_load_weights(runner));
  fprintf(stderr, "Weights ready; preparing reusable commands...\n");
  IREE_RETURN_IF_ERROR(qwen_stage_prepare(
      runner, iree_make_cstring_view(FLAG_prefill), &runner->stages[0]));
  IREE_RETURN_IF_ERROR(qwen_stage_prepare(
      runner, iree_make_cstring_view(FLAG_decode), &runner->stages[1]));
  IREE_RETURN_IF_ERROR(qwen_create_model(runner));
  IREE_RETURN_IF_ERROR(qwen_initialize_row(runner));
  iree_status_t status = iree_ok_status();
  for (int32_t i = 0; i < FLAG_max_tokens && iree_status_is_ok(status) &&
                      !runner->row.progress[7];
       ++i) {
    fprintf(stderr, "%s %d...\n", i == 0 ? "Prefill" : "Decode", i);
    status = qwen_step(runner, i == 0);
    if (iree_status_is_ok(status) && runner->row.progress[0] != i + 1) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "generation count %d after step %d",
                                runner->row.progress[0], i);
    }
    if (iree_status_is_ok(status)) {
      fprintf(stderr, "Token %d: %d%s\n", i, runner->row.tokens[0],
              runner->row.progress[7] ? " (EOS)" : "");
    }
  }
  if (iree_status_is_ok(status)) {
    char text[8192];
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode(
        runner->tokenizer,
        iree_tokenizer_make_token_id_list(&runner->row.tokens[1],
                                          runner->row.progress[0]),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(text, sizeof(text)), runner->allocator,
        &length);
    if (iree_status_is_ok(status) &&
        (fwrite(text, 1, length, stdout) != length || fflush(stdout) != 0)) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS, "writing generated text");
    }
  }
  return status;
}

static void qwen_deinitialize(qwen_runner_t* runner) {
  iree_vm_process_release(runner->process);
  if (runner->invocation) {
    iree_vm_invocation_deinitialize(runner->invocation);
  }
  iree_vm_program_release(runner->program);
  iree_vm_module_release(runner->native_module);
  iree_vm_module_release(runner->bytecode_module);
  iree_vm_environment_free(runner->environment);
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    iree_hal_buffer_release(runner->buffers[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(runner->stages); ++i) {
    iree_hal_command_buffer_release(runner->stages[i].command);
    iree_io_file_contents_free(runner->stages[i].contents);
    iree_io_file_contents_free(runner->stages[i].manifest);
  }
  iree_hal_buffer_release(runner->weights);
  loom_serve_execution_release(runner->execution);
  iree_hal_device_group_release(runner->group);
  iree_hal_device_release(runner->device);
  iree_async_frontier_tracker_release(runner->frontier_tracker);
  iree_async_proactor_pool_release(runner->proactor_pool);
  iree_tokenizer_free(runner->tokenizer);
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  qwen_runner_t runner = {.allocator = iree_allocator_system()};
  iree_status_t status = qwen_run(&runner);
  // Host upload/download storage lives in runner until all accepted work ends.
  if (runner.execution) {
    status =
        iree_status_join(status, loom_serve_execution_drain(runner.execution));
  }
  qwen_deinitialize(&runner);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
