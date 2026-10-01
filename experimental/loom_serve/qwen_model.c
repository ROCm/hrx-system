// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared Qwen3.8-27B residency and fixed retained rows.

#include "experimental/loom_serve/qwen_model.h"

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
#include "iree/hal/drivers/init.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

enum {
  QWEN_RESIDUAL = 0,
  QWEN_CONTROL = 1,
  QWEN_GDN = 2,
  QWEN_ATTENTION = 3,
  QWEN_TOKENS = 4,
  QWEN_PROGRESS = 5,
  QWEN_WORKSPACE = 6,
  QWEN_BINDING_COUNT = 7,
  QWEN_TOKEN_CAPACITY = 512,
  QWEN_ROW_CAPACITY = 8,
  QWEN_SPAN_CAPACITY = 64,
  QWEN_EPOCH_SELECTION = 3 + 5 * QWEN_SPAN_CAPACITY,
  QWEN_EPOCH_WORDS = QWEN_EPOCH_SELECTION + QWEN_SPAN_CAPACITY,
};

typedef struct qwen_stage_t {
  // Unique native export name, established during cold loading.
  char name[32];
  // Owned auxiliary artifact path; null for directories borrowed at creation.
  char* directory;
  // Owned immutable artifact manifest bytes.
  iree_io_file_contents_t* manifest;
  // Owned portable command bytes backing program.
  iree_io_file_contents_t* contents;
  // Parsed view borrowing contents.
  loom_cmd_program_t program;
  // Reusable command retaining its executable and fixed weight resources.
  iree_hal_command_buffer_t* command;
  // Owned views of immutable parameter groups, in fixed-root order.
  iree_hal_buffer_t* fixed_buffers[5];
} qwen_stage_t;

struct loom_serve_qwen_model_t {
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
  // Prefill, decode, then packed-epoch artifacts, owned in option order.
  qwen_stage_t* stages;
  // Number of allocated stages, including partially initialized cold entries.
  iree_host_size_t stage_count;
  // Number of target packed shapes, independent of auxiliary MTP stages.
  iree_host_size_t shape_count;
  // Immutable prepared shape capacities, owned in epoch option order.
  loom_serve_qwen_shape_t* shapes;
  // Cold-resolved native exports in the one shared VM process.
  iree_vm_function_t* functions;
  // Target parameter slab; auxiliary stages retain views of shared tensors.
  iree_hal_buffer_t* weights;
  // Compile-time capacity checked against every loaded stage configuration.
  iree_host_size_t context_capacity;
  // Maximum active input count accepted by the prepared prefill stage.
  iree_host_size_t prefill_capacity;
  // Number of preallocated rows.
  iree_host_size_t row_count;
  // Fixed row records, owned by this model.
  loom_serve_qwen_row_t* rows;
  // One device allocation partitioned into private retained row spans.
  iree_hal_buffer_t* row_arena;
  // Model-wide residual storage, serialized by the execution timeline.
  iree_hal_buffer_t* residual;
  // Model-wide packed transient storage for the larger stage.
  iree_hal_buffer_t* workspace;
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
  // Reusable packed-epoch bindings and transfer payloads. Device buffers at
  // slots 1, 2, 4 and 5 are owned; other slots borrow model-wide storage.
  struct {
    // Residual, metadata, origins, state arena, input IDs, output IDs,
    // workspace.
    iree_hal_buffer_t* buffers[QWEN_BINDING_COUNT];
    // Immutable cold GDN/KV byte origins for each resident row.
    int64_t origins[QWEN_ROW_CAPACITY][2];
    // Header, fixed descriptor slots and selected packed-row mapping.
    int32_t metadata[QWEN_EPOCH_WORDS];
    // Padded input payload, retained until completion or destruction drains it.
    int32_t tokens[QWEN_TOKEN_CAPACITY];
    // Compact predictions downloaded before committing the row records.
    int32_t outputs[QWEN_ROW_CAPACITY];
  } epoch;
  // Optional block-64 residency and private proposal work. Warm stages consume
  // committed target residuals before the next target invocation can reuse
  // them.
  struct {
    // First auxiliary stage: draft, begin, then one warm stage per target
    // shape.
    iree_host_size_t first_stage;
    // Normalized committed target hidden, indexed by retained row.
    iree_hal_buffer_t* carry;
    // One-token proposal bindings followed by its compiler transient slab.
    iree_hal_buffer_t* buffers[8];
    // Immutable row origins in the private one-layer attention cache.
    int64_t origins[QWEN_ROW_CAPACITY][2];
    // Reusable proposal control payload, separate from immutable target work.
    int32_t metadata[QWEN_EPOCH_WORDS];
    // Padded proposal input IDs, retained until transfer completion.
    int32_t tokens[QWEN_TOKEN_CAPACITY];
    // Compact output IDs downloaded after each proposal round.
    int32_t outputs[QWEN_ROW_CAPACITY];
  } mtp;
  // Tokenizer owning its parsed vocabulary and encoding data.
  iree_tokenizer_t* tokenizer;
  // Token identifying the end of a generated assistant turn.
  int32_t eos_token;
};

struct loom_serve_qwen_row_t {
  // Borrowed owning model; all row operations use its single invocation.
  loom_serve_qwen_model_t* model;
  // Private spans in the row arena; residual/workspace entries borrow model.
  iree_hal_buffer_t* buffers[QWEN_BINDING_COUNT];
  // Number of input tokens actually consumed into KV/GDN state.
  iree_host_size_t position;
  // Whether the latest completed work selected a token at the current position.
  bool has_prediction;
  // Completed stage counts and end-to-end host durations.
  loom_serve_qwen_metrics_t metrics;
  // Host payloads remain alive until the model drains accepted transfers.
  struct {
    // Active prefill count, absolute base, and EOS ID.
    int32_t control[3];
    // Fixed-capacity padded upload and current-token readback at slot zero.
    int32_t tokens[QWEN_TOKEN_CAPACITY];
    // Device-generated count at zero and EOS flag at seven.
    int32_t progress[8];
  } transfer;
};

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
    loom_serve_qwen_model_t* runner, iree_string_view_t directory,
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

static iree_status_t qwen_stage_prepare(loom_serve_qwen_model_t* runner,
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
        iree_hal_queue_family(runner->dispatch), &stage->program,
        stage->program.requirements.fixed_buffer_count, stage->fixed_buffers,
        count, entries, runner->allocator, &stage->command);
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

static iree_status_t qwen_create_device(loom_serve_qwen_model_t* runner) {
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
    params.event_sink = iree_hal_device_event_sink_stderr();
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

static iree_status_t qwen_check_layouts(loom_serve_qwen_model_t* runner) {
  const loom_cmd_program_t* prefill = &runner->stages[0].program;
  for (iree_host_size_t i = 0; i < runner->shape_count + 2; ++i) {
    const loom_cmd_program_t* program = &runner->stages[i].program;
    if (program->requirements.fixed_buffer_count != 1 ||
        program->parameter_roots.count != 1 ||
        program->requirements.rebindable_binding_count != QWEN_BINDING_COUNT ||
        program->requirements.transient.binding_index != QWEN_WORKSPACE ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the Qwen row layout");
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

static iree_status_t qwen_allocate_buffer(loom_serve_qwen_model_t* runner,
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

typedef struct qwen_root_parameters_t {
  // Borrowed parsed program whose parameter records back the enumeration.
  const loom_cmd_program_t* program;
  // Parameter ordinals selected once for this fixed root during cold loading.
  const uint32_t* indices;
} qwen_root_parameters_t;

static iree_status_t qwen_root_parameter_span(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const qwen_root_parameters_t* parameters = user_data;
  return qwen_parameter_span((void*)parameters->program, parameters->indices[i],
                             out_key, out_span);
}

// Cold artifact placement resolves shared tensors to existing storage. MTP
// command roots group their new parameters independently; shared embedding,
// normalization and output roots are views, not second copies of target data.
static bool qwen_find_parameter(loom_serve_qwen_model_t* model,
                                iree_host_size_t before_stage,
                                iree_string_view_t key,
                                loom_cmd_program_parameter_t* out_parameter,
                                iree_hal_buffer_t** out_buffer) {
  for (iree_host_size_t i = 0; i < before_stage; ++i) {
    const qwen_stage_t* stage = &model->stages[i];
    for (uint32_t j = 0; j < stage->program.parameters.count; ++j) {
      const loom_cmd_program_parameter_t parameter =
          loom_cmd_program_parameter_at(&stage->program, j);
      if (iree_string_view_equal(parameter.key, key)) {
        *out_parameter = parameter;
        *out_buffer = stage->fixed_buffers[parameter.fixed_buffer_index];
        return true;
      }
    }
  }
  return false;
}

static iree_status_t qwen_load_auxiliary_weights(
    loom_serve_qwen_model_t* model, iree_host_size_t stage_index,
    iree_io_parameter_provider_t* provider, iree_hal_semaphore_t* ready,
    uint64_t* ready_value) {
  qwen_stage_t* stage = &model->stages[stage_index];
  const loom_cmd_program_t* program = &stage->program;
  if (!program->parameter_roots.count) {
    return iree_ok_status();
  }
  uint32_t* indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->allocator, program->parameters.count,
                                  sizeof(*indices), (void**)&indices));
  iree_status_t status = iree_ok_status();
  for (uint32_t root_index = 0;
       root_index < program->parameter_roots.count && iree_status_is_ok(status);
       ++root_index) {
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, root_index);
    iree_host_size_t count = 0;
    for (uint32_t i = 0; i < program->parameters.count; ++i) {
      if (loom_cmd_program_parameter_at(program, i).fixed_buffer_index ==
          root.fixed_buffer_index) {
        indices[count++] = i;
      }
    }
    if (!count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "MTP stage %s has an empty parameter root",
                                stage->name);
      continue;
    }
    const loom_cmd_program_parameter_t first =
        loom_cmd_program_parameter_at(program, indices[0]);
    loom_cmd_program_parameter_t existing = {0};
    iree_hal_buffer_t* source = NULL;
    const bool shared =
        qwen_find_parameter(model, stage_index, first.key, &existing, &source);
    uint64_t origin = 0;
    if (shared) {
      if (existing.byte_offset < first.byte_offset) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "shared parameter root begins before storage");
      } else {
        origin = existing.byte_offset - first.byte_offset;
      }
    }
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      const loom_cmd_program_parameter_t parameter =
          loom_cmd_program_parameter_at(program, indices[i]);
      iree_hal_buffer_t* prior_buffer = NULL;
      const bool found = qwen_find_parameter(model, stage_index, parameter.key,
                                             &existing, &prior_buffer);
      if (found != shared ||
          (found && (prior_buffer != source ||
                     existing.byte_offset != origin + parameter.byte_offset ||
                     existing.byte_length != parameter.byte_length))) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage %s parameter root cannot share the existing placement",
            stage->name);
      }
    }
    if (iree_status_is_ok(status) && shared) {
      status = iree_hal_buffer_subspan(
          source, origin, root.required_byte_length, model->allocator,
          &stage->fixed_buffers[root.fixed_buffer_index]);
    } else if (iree_status_is_ok(status)) {
      status = qwen_allocate_buffer(
          model, root.required_byte_length, root.minimum_alignment,
          &stage->fixed_buffers[root.fixed_buffer_index]);
      if (iree_status_is_ok(status)) {
        ++*ready_value;
        qwen_root_parameters_t parameters = {program, indices};
        status = iree_io_parameter_provider_gather(
            provider, model->device, model->transfer,
            iree_hal_semaphore_list_empty(),
            (iree_hal_semaphore_list_t){1, &ready, ready_value},
            iree_string_view_empty(),
            stage->fixed_buffers[root.fixed_buffer_index], count,
            (iree_io_parameter_enumerator_t){qwen_root_parameter_span,
                                             &parameters});
        if (iree_status_is_ok(status)) {
          status = iree_hal_semaphore_wait(ready, *ready_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE);
        }
      }
    }
  }
  iree_allocator_free(model->allocator, indices);
  return status;
}

static iree_status_t qwen_load_weights(loom_serve_qwen_model_t* runner,
                                       iree_string_view_t weights_path) {
  const loom_cmd_program_t* program = &runner->stages[0].program;
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(program, 0);
  uint64_t length = root.required_byte_length;
  uint64_t alignment = root.minimum_alignment;
  for (iree_host_size_t i = 1; i < runner->shape_count + 2; ++i) {
    const loom_cmd_program_parameter_root_t stage_root =
        loom_cmd_program_parameter_root_at(&runner->stages[i].program, 0);
    length = iree_max(length, stage_root.required_byte_length);
    alignment = iree_max(alignment, stage_root.minimum_alignment);
  }
  IREE_RETURN_IF_ERROR(
      qwen_allocate_buffer(runner, length, alignment, &runner->weights));
  fprintf(stderr, "Loading %u parameters into one %.3f GiB weight slab...\n",
          program->parameters.count, length / 1073741824.0);
  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_create(runner->allocator, &index));
  iree_status_t status = iree_tooling_append_parameter_file_to_index(
      weights_path, index, runner->allocator);
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
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < runner->shape_count + 2; ++i) {
      runner->stages[i].fixed_buffers[0] = runner->weights;
      iree_hal_buffer_retain(runner->weights);
    }
  }
  for (iree_host_size_t i = runner->shape_count + 2;
       i < runner->stage_count && iree_status_is_ok(status); ++i) {
    status =
        qwen_load_auxiliary_weights(runner, i, provider, ready, &ready_value);
  }
  iree_hal_semaphore_release(ready);
  iree_io_parameter_provider_release(provider);
  iree_io_parameter_index_release(index);
  return status;
}

static iree_status_t qwen_create_program(loom_serve_qwen_model_t* runner) {
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(runner->allocator, &runner->environment));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(runner->environment, &table));
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &runner->types));
  loom_serve_stage_t* stages = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(runner->allocator, runner->stage_count,
                                  sizeof(*stages), (void**)&stages));
  for (iree_host_size_t i = 0; i < runner->stage_count; ++i) {
    stages[i] = (loom_serve_stage_t){
        iree_make_cstring_view(runner->stages[i].name),
        runner->stages[i].command,
        runner->stages[i].program.requirements.rebindable_binding_count};
  }
  iree_status_t status = loom_serve_module_create(
      &runner->types, runner->execution, runner->stage_count, stages,
      runner->allocator, &runner->native_module);
  iree_allocator_free(runner->allocator, stages);
  IREE_RETURN_IF_ERROR(status);
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
  IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
      runner->process, IREE_SV("model"), IREE_SV("step"), &runner->step));
  for (iree_host_size_t i = 2;
       i < runner->stage_count && iree_status_is_ok(status); ++i) {
    status = iree_vm_process_lookup_function(
        runner->process, IREE_SV("runner"),
        iree_make_cstring_view(runner->stages[i].name), &runner->functions[i]);
  }
  return status;
}

static iree_status_t qwen_read_capacities(loom_serve_qwen_model_t* model,
                                          iree_string_view_t directory,
                                          iree_host_size_t* out_prefill,
                                          iree_host_size_t* out_context,
                                          iree_host_size_t* out_spans) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(qwen_read_artifact(directory, IREE_SV("config.json"),
                                          model->allocator, &contents));
  iree_string_view_t text = qwen_file_text(contents);
  iree_string_view_t value = iree_string_view_empty();
  uint64_t prefill = 0;
  uint64_t context = 0;
  uint64_t spans = 0;
  iree_status_t status = iree_json_lookup_object_value(
      text, IREE_SV("runner.qwen38.prefill_token_count"), &value);
  if (iree_status_is_ok(status)) {
    status = iree_json_parse_uint64(value, &prefill);
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_lookup_object_value(
        text, IREE_SV("qwen38.attention.cache_capacity"), &value);
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_parse_uint64(value, &context);
  }
  if (iree_status_is_ok(status) && out_spans) {
    status = iree_json_lookup_object_value(
        text, IREE_SV("runner.qwen38.span_capacity"), &value);
    if (iree_status_is_ok(status)) {
      status = iree_json_parse_uint64(value, &spans);
    }
    if (iree_status_is_ok(status) && (spans < 1 || spans > QWEN_ROW_CAPACITY)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid Qwen epoch span capacity");
    }
  }
  if (iree_status_is_ok(status) &&
      (prefill < 1 || prefill > QWEN_TOKEN_CAPACITY || context < prefill ||
       context > 262144)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid Qwen prefill/context capacities");
  }
  if (iree_status_is_ok(status)) {
    *out_prefill = (iree_host_size_t)prefill;
    *out_context = (iree_host_size_t)context;
    if (out_spans) {
      *out_spans = (iree_host_size_t)spans;
    }
  }
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t qwen_load_tokenizer(loom_serve_qwen_model_t* model,
                                         iree_string_view_t path) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 model->allocator, &contents));
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      qwen_file_text(contents), model->allocator, &model->tokenizer);
  iree_io_file_contents_free(contents);
  IREE_RETURN_IF_ERROR(status);
  model->eos_token = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(model->tokenizer), IREE_SV("<|im_end|>"));
  if (model->eos_token < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no <|im_end|>");
  }
  return iree_ok_status();
}

static iree_status_t qwen_allocate_rows(loom_serve_qwen_model_t* model) {
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
  IREE_RETURN_IF_ERROR(
      qwen_allocate_buffer(model, 10485760, 256, &model->residual));
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(
      model, workspace_length, workspace_alignment, &model->workspace));
  const uint64_t lengths[QWEN_BINDING_COUNT] = {
      0,
      12,
      156893184,
      model->context_capacity * 65536ull,
      QWEN_TOKEN_CAPACITY * sizeof(int32_t),
      8 * sizeof(int32_t),
      0,
  };
  uint64_t offsets[QWEN_BINDING_COUNT] = {0};
  uint64_t stride = 0;
  for (iree_host_size_t i = QWEN_CONTROL; i <= QWEN_PROGRESS; ++i) {
    offsets[i] = stride;
    stride += (lengths[i] + 255) & ~255ull;
  }
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, stride * model->row_count,
                                            256, &model->row_arena));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(model->allocator, model->row_count,
                                  sizeof(*model->rows), (void**)&model->rows));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < model->row_count && iree_status_is_ok(status); ++i) {
    loom_serve_qwen_row_t* row = &model->rows[i];
    row->model = model;
    row->buffers[QWEN_RESIDUAL] = model->residual;
    row->buffers[QWEN_WORKSPACE] = model->workspace;
    model->epoch.origins[i][0] = (int64_t)(i * stride + offsets[QWEN_GDN]);
    model->epoch.origins[i][1] =
        (int64_t)(i * stride + offsets[QWEN_ATTENTION]);
    for (iree_host_size_t binding = QWEN_CONTROL;
         binding <= QWEN_PROGRESS && iree_status_is_ok(status); ++binding) {
      status = iree_hal_buffer_subspan(
          model->row_arena, i * stride + offsets[binding], lengths[binding],
          model->allocator, &row->buffers[binding]);
    }
  }
  if (iree_status_is_ok(status) && model->stage_count > 2) {
    model->epoch.buffers[0] = model->residual;
    model->epoch.buffers[3] = model->row_arena;
    model->epoch.buffers[6] = model->workspace;
    const uint64_t epoch_lengths[] = {
        0, sizeof(model->epoch.metadata), sizeof(model->epoch.origins),
        0, sizeof(model->epoch.tokens),   sizeof(model->epoch.outputs),
        0,
    };
    for (iree_host_size_t i = 0;
         i < QWEN_BINDING_COUNT && iree_status_is_ok(status); ++i) {
      if (epoch_lengths[i]) {
        status = qwen_allocate_buffer(model, epoch_lengths[i], 256,
                                      &model->epoch.buffers[i]);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const uint32_t zero = 0;
    iree_hal_buffer_t* buffers[] = {model->row_arena, model->residual,
                                    model->workspace};
    iree_hal_transfer_operation_t fills[IREE_ARRAYSIZE(buffers)] = {0};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
      fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
      fills[i].fill.target_buffer = buffers[i];
      fills[i].fill.length = iree_hal_buffer_byte_length(buffers[i]);
      fills[i].fill.pattern = &zero;
      fills[i].fill.pattern_length = sizeof(zero);
    }
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(fills), fills, &completion);
    if (iree_status_is_ok(status) && model->stage_count > 2) {
      const iree_hal_transfer_operation_t upload = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {.source = model->epoch.origins,
                     .target_buffer = model->epoch.buffers[2],
                     .length = sizeof(model->epoch.origins)},
      };
      status = loom_serve_execution_transfer(model->execution, 1, &upload,
                                             &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "Residency: %zu rows, %.3f GiB retained arena, %.3f GiB shared "
            "workspace, %zu-token context, %zu-token prefill capacity.\n",
            model->row_count, stride * model->row_count / 1073741824.0,
            workspace_length / 1073741824.0, model->context_capacity,
            model->prefill_capacity);
  }
  return status;
}

static iree_status_t qwen_allocate_mtp(loom_serve_qwen_model_t* model) {
  if (!model->mtp.first_stage) {
    return iree_ok_status();
  }
  model->mtp.buffers[0] = model->residual;
  model->mtp.buffers[7] = model->workspace;
  IREE_RETURN_IF_ERROR(qwen_allocate_buffer(model, model->row_count * 20480,
                                            256, &model->mtp.carry));
  const uint64_t lengths[8] = {
      0,
      sizeof(model->mtp.metadata),
      sizeof(model->mtp.origins),
      model->row_count * model->context_capacity * 4096ull,
      32 * 20480,
      sizeof(model->mtp.tokens),
      sizeof(model->mtp.outputs),
      0,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 1; i < 7 && iree_status_is_ok(status); ++i) {
    status =
        qwen_allocate_buffer(model, lengths[i], 256, &model->mtp.buffers[i]);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      model->mtp.origins[i][1] =
          (int64_t)(i * model->context_capacity * 4096ull);
    }
    const uint32_t zero = 0;
    const iree_hal_transfer_operation_t transfers[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
         .fill = {.target_buffer = model->mtp.carry,
                  .length = iree_hal_buffer_byte_length(model->mtp.carry),
                  .pattern = &zero,
                  .pattern_length = sizeof(zero)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
         .fill = {.target_buffer = model->mtp.buffers[4],
                  .length = lengths[4],
                  .pattern = &zero,
                  .pattern_length = sizeof(zero)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->mtp.origins,
                    .target_buffer = model->mtp.buffers[2],
                    .length = sizeof(model->mtp.origins)}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(transfers), transfers, &completion);
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
  }
  return status;
}

static iree_status_t qwen_initialize(loom_serve_qwen_model_t* model,
                                     const loom_serve_qwen_options_t* options) {
  model->shape_count = options->epoch_count;
  const bool uses_mtp = !iree_string_view_is_empty(options->mtp_directory);
  if (uses_mtp && !model->shape_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MTP requires packed target shapes");
  }
  iree_host_size_t stage_count = options->epoch_count + 2;
  if (uses_mtp) {
    model->mtp.first_stage = stage_count;
    stage_count += options->epoch_count + 2;
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, stage_count, sizeof(*model->stages),
      (void**)&model->stages));
  model->stage_count = stage_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      model->allocator, model->stage_count, sizeof(*model->functions),
      (void**)&model->functions));
  if (options->epoch_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        model->allocator, options->epoch_count, sizeof(*model->shapes),
        (void**)&model->shapes));
  }
  snprintf(model->stages[0].name, sizeof(model->stages[0].name), "prefill");
  snprintf(model->stages[1].name, sizeof(model->stages[1].name), "decode");
  IREE_RETURN_IF_ERROR(qwen_read_capacities(model, options->prefill_directory,
                                            &model->prefill_capacity,
                                            &model->context_capacity, NULL));
  iree_host_size_t decode_prefill_capacity = 0;
  iree_host_size_t decode_context_capacity = 0;
  IREE_RETURN_IF_ERROR(qwen_read_capacities(model, options->decode_directory,
                                            &decode_prefill_capacity,
                                            &decode_context_capacity, NULL));
  if (model->context_capacity != decode_context_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prefill/decode context capacities differ");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < options->epoch_count && iree_status_is_ok(status); ++i) {
    snprintf(model->stages[i + 2].name, sizeof(model->stages[i + 2].name),
             "epoch_%zu", i);
    iree_host_size_t epoch_context = 0;
    status = qwen_read_capacities(
        model, options->epoch_directories[i], &model->shapes[i].token_capacity,
        &epoch_context, &model->shapes[i].span_capacity);
    if (iree_status_is_ok(status) && epoch_context != model->context_capacity) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "epoch %zu/isolated context capacities differ", i);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(qwen_load_tokenizer(model, options->tokenizer_path));
  IREE_RETURN_IF_ERROR(qwen_stage_read(options->prefill_directory,
                                       model->allocator, &model->stages[0]));
  IREE_RETURN_IF_ERROR(qwen_stage_read(options->decode_directory,
                                       model->allocator, &model->stages[1]));
  for (iree_host_size_t i = 0;
       i < options->epoch_count && iree_status_is_ok(status); ++i) {
    status = qwen_stage_read(options->epoch_directories[i], model->allocator,
                             &model->stages[i + 2]);
  }
  IREE_RETURN_IF_ERROR(status);
  for (iree_host_size_t i = model->shape_count + 2;
       i < model->stage_count && iree_status_is_ok(status); ++i) {
    qwen_stage_t* stage = &model->stages[i];
    const iree_host_size_t ordinal = i - model->mtp.first_stage;
    char relative[32];
    if (ordinal < 2) {
      snprintf(relative, sizeof(relative), "%s",
               ordinal == 0 ? "draft" : "begin");
    } else {
      snprintf(relative, sizeof(relative), "warm%zu",
               model->shapes[ordinal - 2].token_capacity);
    }
    snprintf(stage->name, sizeof(stage->name), "mtp_%zu", ordinal);
    status = iree_file_path_join(options->mtp_directory,
                                 iree_make_cstring_view(relative),
                                 model->allocator, &stage->directory);
    if (iree_status_is_ok(status)) {
      status = qwen_stage_read(iree_make_cstring_view(stage->directory),
                               model->allocator, stage);
    }
    if (iree_status_is_ok(status)) {
      const uint32_t fixed_count = ordinal == 0 ? 5 : ordinal == 1 ? 0 : 4;
      const uint32_t binding_count = ordinal == 0 ? 8 : ordinal == 1 ? 3 : 7;
      const uint32_t transient_index =
          ordinal == 1 ? UINT32_MAX : binding_count - 1;
      const loom_cmd_program_requirements_t requirements =
          stage->program.requirements;
      if (requirements.fixed_buffer_count != fixed_count ||
          stage->program.parameter_roots.count != fixed_count ||
          requirements.rebindable_binding_count != binding_count ||
          requirements.transient.binding_index != transient_index ||
          requirements.launch_counts.required_byte_length != 0) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "MTP %s command has an incompatible layout",
                                  relative);
      }
    }
    iree_host_size_t tokens = 0, context = 0, spans = 0;
    if (iree_status_is_ok(status)) {
      status =
          qwen_read_capacities(model, iree_make_cstring_view(stage->directory),
                               &tokens, &context, &spans);
    }
    if (iree_status_is_ok(status) &&
        (context != model->context_capacity ||
         tokens !=
             (ordinal < 2 ? 32 : model->shapes[ordinal - 2].token_capacity) ||
         spans !=
             (ordinal < 2 ? 8 : model->shapes[ordinal - 2].span_capacity))) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "MTP %s shape/context does not match target residency", relative);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(qwen_check_layouts(model));
  IREE_RETURN_IF_ERROR(qwen_create_device(model));
  IREE_RETURN_IF_ERROR(qwen_load_weights(model, options->weights_path));
  IREE_RETURN_IF_ERROR(
      qwen_stage_prepare(model, options->prefill_directory, &model->stages[0]));
  IREE_RETURN_IF_ERROR(
      qwen_stage_prepare(model, options->decode_directory, &model->stages[1]));
  for (iree_host_size_t i = 0;
       i < options->epoch_count && iree_status_is_ok(status); ++i) {
    status = qwen_stage_prepare(model, options->epoch_directories[i],
                                &model->stages[i + 2]);
  }
  for (iree_host_size_t i = model->shape_count + 2;
       i < model->stage_count && iree_status_is_ok(status); ++i) {
    status = qwen_stage_prepare(
        model, iree_make_cstring_view(model->stages[i].directory),
        &model->stages[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(qwen_create_program(model));
  IREE_RETURN_IF_ERROR(qwen_allocate_rows(model));
  return qwen_allocate_mtp(model);
}

iree_status_t loom_serve_qwen_model_create(
    const loom_serve_qwen_options_t* options, iree_allocator_t host_allocator,
    loom_serve_qwen_model_t** out_model) {
  *out_model = NULL;
  if (options->row_count < 1 || options->row_count > QWEN_ROW_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen row count must be in [1, 8]");
  }
  loom_serve_qwen_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  model->row_count = options->row_count;
  iree_status_t status = qwen_initialize(model, options);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  }
  return status;
}

iree_status_t loom_serve_qwen_model_destroy(loom_serve_qwen_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = model->execution
                             ? loom_serve_execution_drain(model->execution)
                             : iree_ok_status();
  iree_vm_process_release(model->process);
  if (model->invocation) {
    iree_vm_invocation_deinitialize(model->invocation);
  }
  iree_vm_program_release(model->program);
  iree_vm_module_release(model->native_module);
  iree_vm_module_release(model->bytecode_module);
  iree_vm_environment_free(model->environment);
  if (model->rows) {
    for (iree_host_size_t i = 0; i < model->row_count; ++i) {
      for (iree_host_size_t binding = QWEN_CONTROL; binding <= QWEN_PROGRESS;
           ++binding) {
        iree_hal_buffer_release(model->rows[i].buffers[binding]);
      }
    }
  }
  iree_allocator_free(model->allocator, model->rows);
  iree_hal_buffer_release(model->epoch.buffers[1]);
  iree_hal_buffer_release(model->epoch.buffers[2]);
  iree_hal_buffer_release(model->epoch.buffers[4]);
  iree_hal_buffer_release(model->epoch.buffers[5]);
  iree_hal_buffer_release(model->mtp.carry);
  for (iree_host_size_t i = 1; i < 7; ++i) {
    iree_hal_buffer_release(model->mtp.buffers[i]);
  }
  iree_hal_buffer_release(model->row_arena);
  iree_hal_buffer_release(model->residual);
  iree_hal_buffer_release(model->workspace);
  for (iree_host_size_t i = 0; i < model->stage_count; ++i) {
    iree_hal_command_buffer_release(model->stages[i].command);
    iree_io_file_contents_free(model->stages[i].contents);
    iree_io_file_contents_free(model->stages[i].manifest);
    iree_allocator_free(model->allocator, model->stages[i].directory);
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(model->stages[i].fixed_buffers); ++j) {
      iree_hal_buffer_release(model->stages[i].fixed_buffers[j]);
    }
  }
  iree_allocator_free(model->allocator, model->functions);
  iree_allocator_free(model->allocator, model->shapes);
  iree_allocator_free(model->allocator, model->stages);
  iree_hal_buffer_release(model->weights);
  loom_serve_execution_release(model->execution);
  iree_hal_device_group_release(model->group);
  iree_hal_device_release(model->device);
  iree_async_frontier_tracker_release(model->frontier_tracker);
  iree_async_proactor_pool_release(model->proactor_pool);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model);
  return status;
}

loom_serve_qwen_row_t* loom_serve_qwen_model_row(loom_serve_qwen_model_t* model,
                                                 iree_host_size_t index) {
  return &model->rows[index];
}

iree_tokenizer_t* loom_serve_qwen_model_tokenizer(
    loom_serve_qwen_model_t* model) {
  return model->tokenizer;
}

iree_host_size_t loom_serve_qwen_model_context_capacity(
    const loom_serve_qwen_model_t* model) {
  return model->context_capacity;
}

iree_host_size_t loom_serve_qwen_model_prefill_capacity(
    const loom_serve_qwen_model_t* model) {
  return model->prefill_capacity;
}

iree_host_size_t loom_serve_qwen_model_shape_count(
    const loom_serve_qwen_model_t* model) {
  return model->shape_count;
}

const loom_serve_qwen_shape_t* loom_serve_qwen_model_shapes(
    const loom_serve_qwen_model_t* model) {
  return model->shapes;
}

iree_status_t loom_serve_qwen_row_reset(loom_serve_qwen_row_t* row) {
  const uint32_t zero = 0;
  iree_hal_transfer_operation_t fills[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[QWEN_GDN],
                .length = iree_hal_buffer_byte_length(row->buffers[QWEN_GDN]),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
       .fill = {.target_buffer = row->buffers[QWEN_PROGRESS],
                .length = sizeof(row->transfer.progress),
                .pattern = &zero,
                .pattern_length = sizeof(zero)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(fills), fills, &completion);
  if (iree_status_is_ok(status) && row->model->mtp.first_stage) {
    const iree_hal_transfer_operation_t clear_carry = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill = {.target_buffer = row->model->mtp.carry,
                 .target_offset = (uint64_t)(row - row->model->rows) * 20480,
                 .length = 20480,
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
    row->has_prediction = false;
    memset(&row->transfer, 0, sizeof(row->transfer));
    memset(&row->metrics, 0, sizeof(row->metrics));
  }
  return status;
}

static iree_status_t qwen_invoke(loom_serve_qwen_model_t* model,
                                 iree_vm_function_t function,
                                 iree_vm_variant_span_t arguments) {
  iree_vm_variant_t results[1] = {0};
  iree_status_t status =
      iree_vm_invoke(model->invocation, function, arguments,
                     iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(arguments);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  return status;
}

static iree_status_t qwen_step(loom_serve_qwen_row_t* row, int32_t initialize) {
  loom_serve_qwen_model_t* model = row->model;
  iree_vm_variant_t arguments[QWEN_BINDING_COUNT + 1] = {0};
  arguments[0] = iree_vm_variant_from_i32(initialize);
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    arguments[i + 1] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->types, row->buffers[i]);
  }
  IREE_RETURN_IF_ERROR(qwen_invoke(model, model->step,
                                   iree_vm_variant_span_from_array(arguments)));
  const iree_hal_transfer_operation_t downloads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[QWEN_TOKENS],
                    .target = row->transfer.tokens,
                    .length = sizeof(int32_t)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
       .download = {.source_buffer = row->buffers[QWEN_PROGRESS],
                    .target = row->transfer.progress,
                    .length = sizeof(row->transfer.progress)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(downloads), downloads, &completion));
  return loom_serve_execution_wait(model->execution, completion);
}

static iree_status_t qwen_invoke_stage(loom_serve_qwen_model_t* model,
                                       iree_host_size_t stage_index,
                                       iree_hal_buffer_t* const* buffers) {
  const iree_host_size_t count =
      model->stages[stage_index].program.requirements.rebindable_binding_count;
  iree_vm_variant_t arguments[8] = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    arguments[i] =
        iree_hal_buffer_variant_from_ptr_borrowed(&model->types, buffers[i]);
  }
  return qwen_invoke(model, model->functions[stage_index],
                     iree_vm_variant_span_from_ptr(arguments, count));
}

iree_status_t loom_serve_qwen_model_epoch(loom_serve_qwen_model_t* model,
                                          iree_host_size_t shape_index,
                                          iree_host_size_t span_count,
                                          const loom_serve_qwen_span_t* spans) {
  if (shape_index >= model->shape_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packed Qwen shape index is not loaded");
  }
  const loom_serve_qwen_shape_t shape = model->shapes[shape_index];
  if (!span_count || span_count > shape.span_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "epoch span count exceeds stage capacity");
  }
  // Validate the caller's plan before changing reusable payloads or submitting
  // any work. The device consumes this one established partition invariant.
  uint32_t resident_mask = 0;
  iree_host_size_t token_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    if (span->row_index >= model->row_count || !span->token_count ||
        span->token_count > shape.token_capacity - token_count ||
        (span->flags & ~LOOM_SERVE_QWEN_SPAN_FLAG_SELECT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid Qwen epoch span %zu", i);
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
    resident_mask |= resident_bit;
    token_count += span->token_count;
  }

  memset(model->epoch.tokens, 0, sizeof(model->epoch.tokens));
  iree_host_size_t token_begin = 0;
  iree_host_size_t output_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    const loom_serve_qwen_row_t* row = &model->rows[span->row_index];
    int32_t* descriptor = &model->epoch.metadata[3 + 5 * i];
    descriptor[0] = (int32_t)span->token_count;
    descriptor[1] = (int32_t)row->position;
    descriptor[2] = (int32_t)token_begin;
    descriptor[3] = (int32_t)span->row_index;
    descriptor[4] = -1;
    memcpy(model->epoch.tokens + token_begin, span->token_ids,
           span->token_count * sizeof(int32_t));
    token_begin += span->token_count;
    if (iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT)) {
      descriptor[4] = (int32_t)output_count;
      model->epoch.metadata[QWEN_EPOCH_SELECTION + output_count++] =
          (int32_t)(token_begin - 1);
    }
  }
  model->epoch.metadata[0] = (int32_t)token_count;
  model->epoch.metadata[1] = (int32_t)span_count;
  model->epoch.metadata[2] = (int32_t)output_count;
  const iree_hal_transfer_operation_t uploads[] = {
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->epoch.metadata,
                  .target_buffer = model->epoch.buffers[1],
                  .length = sizeof(model->epoch.metadata)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = model->epoch.tokens,
                  .target_buffer = model->epoch.buffers[4],
                  .length = sizeof(model->epoch.tokens)}},
  };
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion));
  iree_vm_variant_t arguments[QWEN_BINDING_COUNT] = {0};
  for (iree_host_size_t i = 0; i < QWEN_BINDING_COUNT; ++i) {
    arguments[i] = iree_hal_buffer_variant_from_ptr_borrowed(
        &model->types, model->epoch.buffers[i]);
  }
  IREE_RETURN_IF_ERROR(qwen_invoke(model, model->functions[shape_index + 2],
                                   iree_vm_variant_span_from_array(arguments)));
  if (model->mtp.first_stage) {
    iree_hal_buffer_t* warm_buffers[] = {
        model->residual,       model->epoch.buffers[1], model->mtp.buffers[2],
        model->mtp.buffers[3], model->mtp.carry,        model->epoch.buffers[4],
        model->workspace,
    };
    IREE_RETURN_IF_ERROR(qwen_invoke_stage(
        model, model->mtp.first_stage + 2 + shape_index, warm_buffers));
  }
  if (output_count) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer = model->epoch.buffers[5],
                     .target = model->epoch.outputs,
                     .length = output_count * sizeof(int32_t)},
    };
    IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(model->execution, 1,
                                                       &download, &completion));
  }
  // This synchronous owner has submitted no later work. The execution frontier
  // is the download when outputs exist, otherwise the model invocation itself.
  IREE_RETURN_IF_ERROR(loom_serve_execution_drain(model->execution));
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_qwen_span_t* span = &spans[i];
    loom_serve_qwen_row_t* row = &model->rows[span->row_index];
    row->position += span->token_count;
    const int32_t output = model->epoch.metadata[3 + 5 * i + 4];
    row->has_prediction = output >= 0;
    if (row->has_prediction) {
      row->transfer.tokens[0] = model->epoch.outputs[output];
      row->transfer.progress[0] = 1;
      row->transfer.progress[7] = row->transfer.tokens[0] == model->eos_token;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_qwen_model_propose(loom_serve_qwen_model_t* model,
                                            iree_host_size_t row_count,
                                            const iree_host_size_t* row_indices,
                                            int32_t (*out_tokens)[3]) {
  if (!model->mtp.first_stage) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "MTP artifacts were not loaded");
  }
  if (!row_count || row_count > model->row_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid MTP proposal row count");
  }
  uint32_t resident_mask = 0;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    const iree_host_size_t index = row_indices[i];
    if (index >= model->row_count || (resident_mask & (1u << index))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid or repeated MTP proposal row");
    }
    const loom_serve_qwen_row_t* row = &model->rows[index];
    if (!row->has_prediction || row->transfer.progress[7] ||
        model->context_capacity - row->position < 4) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "MTP needs a pending non-EOS token and four context slots");
    }
    resident_mask |= 1u << index;
  }
  memset(model->mtp.tokens, 0, sizeof(model->mtp.tokens));
  model->mtp.metadata[0] = (int32_t)row_count;
  model->mtp.metadata[1] = (int32_t)row_count;
  model->mtp.metadata[2] = (int32_t)row_count;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    const loom_serve_qwen_row_t* row = &model->rows[row_indices[i]];
    int32_t* descriptor = &model->mtp.metadata[3 + 5 * i];
    descriptor[0] = 1;
    descriptor[1] = (int32_t)row->position;
    descriptor[2] = (int32_t)i;
    descriptor[3] = (int32_t)row_indices[i];
    descriptor[4] = (int32_t)i;
    model->mtp.tokens[i] = row->transfer.tokens[0];
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t round = 0; round < 3 && iree_status_is_ok(status);
       ++round) {
    const iree_hal_transfer_operation_t uploads[] = {
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->mtp.metadata,
                    .target_buffer = model->mtp.buffers[1],
                    .length = sizeof(model->mtp.metadata)}},
        {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
         .upload = {.source = model->mtp.tokens,
                    .target_buffer = model->mtp.buffers[5],
                    .length = sizeof(model->mtp.tokens)}},
    };
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(
        model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
    if (iree_status_is_ok(status) && round == 0) {
      iree_hal_buffer_t* begin_buffers[] = {
          model->mtp.buffers[1],
          model->mtp.carry,
          model->mtp.buffers[4],
      };
      status =
          qwen_invoke_stage(model, model->mtp.first_stage + 1, begin_buffers);
    }
    if (iree_status_is_ok(status)) {
      status =
          qwen_invoke_stage(model, model->mtp.first_stage, model->mtp.buffers);
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {.source_buffer = model->mtp.buffers[6],
                       .target = model->mtp.outputs,
                       .length = row_count * sizeof(int32_t)},
      };
      status = loom_serve_execution_transfer(model->execution, 1, &download,
                                             &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_wait(model->execution, completion);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < row_count; ++i) {
        out_tokens[i][round] = model->mtp.outputs[i];
        model->mtp.tokens[i] = model->mtp.outputs[i];
        ++model->mtp.metadata[3 + 5 * i + 1];
      }
    }
  }
  return status;
}

iree_status_t loom_serve_qwen_row_prefill(loom_serve_qwen_row_t* row,
                                          iree_host_size_t count,
                                          const int32_t* token_ids) {
  loom_serve_qwen_model_t* model = row->model;
  if (!count || count > model->prefill_capacity ||
      count > model->context_capacity - row->position) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prefill chunk exceeds stage or context capacity");
  }
  const iree_time_t start = iree_time_now();
  if (model->mtp.first_stage) {
    iree_host_size_t shape_index = 0;
    for (; shape_index < model->shape_count; ++shape_index) {
      if (model->shapes[shape_index].token_capacity >= count) {
        break;
      }
    }
    if (shape_index == model->shape_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "MTP prefill chunk exceeds all loaded packed shapes");
    }
    const loom_serve_qwen_span_t span = {
        (iree_host_size_t)(row - model->rows),
        count,
        token_ids,
        LOOM_SERVE_QWEN_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(
        loom_serve_qwen_model_epoch(model, shape_index, 1, &span));
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
                  .target_buffer = row->buffers[QWEN_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[QWEN_TOKENS],
                  .length = sizeof(row->transfer.tokens)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = qwen_step(row, 1);
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

iree_status_t loom_serve_qwen_row_decode(loom_serve_qwen_row_t* row) {
  if (!row->has_prediction) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decode requires a selected prediction");
  }
  if (row->position == row->model->context_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen context is full");
  }
  const iree_time_t start = iree_time_now();
  if (row->model->mtp.first_stage) {
    const int32_t token = row->transfer.tokens[0];
    const loom_serve_qwen_span_t span = {
        (iree_host_size_t)(row - row->model->rows),
        1,
        &token,
        LOOM_SERVE_QWEN_SPAN_FLAG_SELECT,
    };
    IREE_RETURN_IF_ERROR(loom_serve_qwen_model_epoch(row->model, 0, 1, &span));
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
                  .target_buffer = row->buffers[QWEN_CONTROL],
                  .length = sizeof(row->transfer.control)}},
      {.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
       .upload = {.source = row->transfer.tokens,
                  .target_buffer = row->buffers[QWEN_TOKENS],
                  .length = sizeof(int32_t)}},
  };
  uint64_t completion = 0;
  iree_status_t status = loom_serve_execution_transfer(
      row->model->execution, IREE_ARRAYSIZE(uploads), uploads, &completion);
  if (iree_status_is_ok(status)) {
    status = qwen_step(row, 0);
  }
  if (iree_status_is_ok(status)) {
    ++row->position;
    ++row->metrics.decode_steps;
    row->metrics.decode_duration += iree_time_now() - start;
  }
  return status;
}

int32_t loom_serve_qwen_row_token(const loom_serve_qwen_row_t* row) {
  return row->transfer.tokens[0];
}

bool loom_serve_qwen_row_is_eos(const loom_serve_qwen_row_t* row) {
  return row->transfer.progress[7] != 0;
}

iree_host_size_t loom_serve_qwen_row_position(
    const loom_serve_qwen_row_t* row) {
  return row->position;
}

loom_serve_qwen_metrics_t loom_serve_qwen_row_metrics(
    const loom_serve_qwen_row_t* row) {
  return row->metrics;
}
