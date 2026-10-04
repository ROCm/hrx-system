// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/weights.h"

#include <stdio.h>

#include "experimental/loom_serve/program.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

enum { WEIGHT_LANES = 4 };

typedef struct weight_span_t {
  // Borrowed key and final byte placement from the compiled command.
  loom_cmd_program_parameter_t parameter;
  // Borrowed final allocation or root view receiving the file bytes.
  iree_hal_buffer_t* buffer;
  // Owned cached in-place preparation command, or null for unchanged bytes.
  iree_hal_command_buffer_t* prepare;
} weight_span_t;

typedef struct weight_group_t {
  // Borrowed consecutive spans in one final allocation. Prepared groups own
  // exactly one tensor; consecutive canonical spans share one readiness edge.
  weight_span_t* spans;
  // Number of spans exposed to the parameter provider.
  iree_host_size_t count;
  // Previous preparation frontier on this group's lane.
  uint64_t wait_value;
  // Read and preparation frontier established by this group.
  uint64_t signal_value;
} weight_group_t;

static iree_status_t weight_allocate(iree_hal_device_t* device,
                                     iree_device_size_t length,
                                     iree_device_size_t alignment,
                                     iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = alignment;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, length, out_buffer);
}

static const weight_span_t* weight_find(iree_host_size_t count,
                                        const weight_span_t* spans,
                                        iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so tensors shared by multiple stages are read and prepared
// exactly once.
static iree_status_t weight_resolve_root(iree_hal_device_t* device,
                                         const loom_cmd_program_t* program,
                                         loom_cmd_program_parameter_root_t root,
                                         iree_host_size_t* span_count,
                                         weight_span_t* spans,
                                         iree_allocator_t allocator,
                                         iree_hal_buffer_t** out_buffer) {
  const iree_host_size_t prior_count = *span_count;
  const weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const weight_span_t* existing =
        weight_find(prior_count, spans, parameter.key);
    if (!count++) {
      shared = existing;
      if (shared) {
        if (shared->parameter.byte_offset < parameter.byte_offset) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "shared parameter root begins before storage");
        }
        origin = shared->parameter.byte_offset - parameter.byte_offset;
      }
    }
    if ((existing != NULL) != (shared != NULL) ||
        (existing &&
         (existing->buffer != shared->buffer ||
          existing->parameter.byte_offset != origin + parameter.byte_offset ||
          existing->parameter.byte_length != parameter.byte_length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter root cannot share existing placement");
    }
  }
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty model parameter root");
  }
  if (shared) {
    return iree_hal_buffer_subspan(shared->buffer, origin,
                                   root.required_byte_length, allocator,
                                   out_buffer);
  }
  IREE_RETURN_IF_ERROR(weight_allocate(device, root.required_byte_length,
                                       root.minimum_alignment, out_buffer));
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (weight_span_t){
        .parameter = parameter,
        .buffer = *out_buffer,
    };
  }
  return iree_ok_status();
}

// A cached preparer owns the returned VM string until cold planning completes.
typedef struct weight_preparer_t {
  // Owned VM result backing the root name, independent of the invocation.
  iree_vm_buffer_t* root;
  // Owned recorded command retaining its executables independently of the JIT.
  iree_hal_command_buffer_t* command;
} weight_preparer_t;

static iree_status_t weight_query_preparer(
    loom_serve_program_t* program, const iree_vm_ref_types_t* types,
    iree_vm_function_t function, loom_cmd_program_parameter_t parameter,
    iree_allocator_t allocator, iree_vm_buffer_t** out_root) {
  *out_root = NULL;
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)parameter.key.data, parameter.key.size),
      iree_vm_buffer_release_callback_null(), allocator, &key));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(types, &key)};
  iree_vm_variant_t results[2] = {0};
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(program), function,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  iree_vm_buffer_t* root = NULL;
  int64_t byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(types, &results[0], &root);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &byte_length);
  }
  if (iree_status_is_ok(status)) {
    if (!root || !iree_all_bits_set(iree_vm_buffer_access(root),
                                    IREE_VM_BUFFER_ACCESS_FLAG_READ)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy must return a readable root");
    } else if (!iree_vm_buffer_length(root) && byte_length != 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unchanged weight policy must return zero byte length");
    } else if (iree_vm_buffer_length(root) &&
               (byte_length <= 0 ||
                (uint64_t)byte_length != parameter.byte_length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy for '%.*s' requires %" PRId64
                                " bytes, reflected parameter has %" PRIu64,
                                (int)parameter.key.size, parameter.key.data,
                                byte_length, (uint64_t)parameter.byte_length);
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_root = root;
  } else {
    iree_vm_buffer_release(root);
  }
  return status;
}

static iree_string_view_t weight_preparer_name(iree_vm_buffer_t* root) {
  return iree_make_string_view(iree_vm_buffer_const_data(root),
                               iree_vm_buffer_length(root));
}

static iree_status_t weight_prepare_spans(
    loom_serve_jit_t* jit, iree_hal_queue_t* dispatch,
    iree_hal_command_buffer_mode_t command_mode, iree_string_view_t policy_path,
    iree_host_size_t span_count, weight_span_t* spans,
    iree_allocator_t allocator) {
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
  loom_serve_program_t* program = NULL;
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  iree_status_t status = loom_serve_program_create(
      environment, policy_path, IREE_ARRAYSIZE(roots), roots,
      iree_vm_module_span_empty(), allocator, &program);
  iree_vm_ref_types_t types = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"),
        IREE_SV("prepare_weight"), &function);
  }
  weight_preparer_t* preparers = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        allocator, span_count, sizeof(*preparers), (void**)&preparers);
  }
  iree_host_size_t preparer_count = 0;
  for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
       ++i) {
    iree_vm_buffer_t* root = NULL;
    status = weight_query_preparer(program, &types, function,
                                   spans[i].parameter, allocator, &root);
    if (iree_status_is_ok(status) && iree_vm_buffer_length(root)) {
      const iree_string_view_t name = weight_preparer_name(root);
      iree_host_size_t p = 0;
      while (p < preparer_count &&
             !iree_string_view_equal(name,
                                     weight_preparer_name(preparers[p].root))) {
        ++p;
      }
      if (p == preparer_count) {
        weight_preparer_t* preparer = &preparers[preparer_count++];
        preparer->root = root;
        root = NULL;
        const loomc_config_options_t config = {
            .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        };
        loom_serve_jit_stage_t* compiled = NULL;
        status = loom_serve_jit_compile(jit, name, &config, &compiled);
        if (iree_status_is_ok(status)) {
          const loom_cmd_program_requirements_t* requirements =
              &loom_serve_jit_stage_program(compiled)->requirements;
          if (requirements->fixed_buffer_count ||
              requirements->rebindable_binding_count != 1 ||
              requirements->transient.required_byte_length ||
              requirements->launch_counts.required_byte_length) {
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "weight preparer '%.*s' requires one mutable binding "
                "and no global scratch",
                (int)name.size, name.data);
          } else {
            status = loom_serve_jit_stage_record(
                compiled, iree_hal_queue_family(dispatch), command_mode, NULL,
                &preparer->command);
          }
        }
        loom_serve_jit_stage_destroy(compiled);
      }
      if (iree_status_is_ok(status)) {
        spans[i].prepare = preparers[p].command;
        iree_hal_command_buffer_retain(spans[i].prepare);
      }
    }
    iree_vm_buffer_release(root);
  }
  for (iree_host_size_t p = 0; p < preparer_count; ++p) {
    iree_vm_buffer_release(preparers[p].root);
    iree_hal_command_buffer_release(preparers[p].command);
  }
  iree_allocator_free(allocator, preparers);
  loom_serve_program_destroy(program);
  iree_vm_environment_free(environment);
  return status;
}

static iree_status_t weight_enumerate(void* user_data, iree_host_size_t i,
                                      iree_string_view_t* out_key,
                                      iree_io_parameter_span_t* out_span) {
  const weight_group_t* group = user_data;
  const loom_cmd_program_parameter_t parameter = group->spans[i].parameter;
  *out_key = parameter.key;
  *out_span = (iree_io_parameter_span_t){
      .buffer_offset = parameter.byte_offset,
      .length = parameter.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t weight_stream(iree_hal_device_t* device,
                                   iree_hal_queue_t* transfer,
                                   iree_hal_queue_t* dispatch,
                                   iree_io_parameter_provider_t* provider,
                                   iree_host_size_t span_count,
                                   weight_span_t* spans,
                                   iree_allocator_t allocator) {
  weight_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, span_count, sizeof(*groups), (void**)&groups));
  iree_io_parameter_gather_t* gathers = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, span_count, sizeof(*gathers), (void**)&gathers);
  iree_hal_semaphore_t* reads[WEIGHT_LANES] = {0};
  iree_hal_semaphore_t* ready[WEIGHT_LANES] = {0};
  uint64_t frontiers[WEIGHT_LANES] = {0};
  for (iree_host_size_t i = 0; i < WEIGHT_LANES && iree_status_is_ok(status);
       ++i) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &reads[i]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready[i]);
    }
  }
  iree_host_size_t group_count = 0;
  uint64_t loaded_bytes = 0;
  uint64_t prepared_bytes = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      const weight_span_t* span = &spans[i];
      if (!i || span->prepare || spans[i - 1].prepare ||
          span->buffer != spans[i - 1].buffer) {
        groups[group_count++].spans = &spans[i];
      }
      ++groups[group_count - 1].count;
      loaded_bytes += span->parameter.byte_length;
      if (span->prepare) {
        prepared_bytes += span->parameter.byte_length;
      }
    }
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      weight_group_t* group = &groups[i];
      const iree_host_size_t lane = i % WEIGHT_LANES;
      group->wait_value = frontiers[lane];
      group->signal_value = ++frontiers[lane];
      gathers[i] = (iree_io_parameter_gather_t){
          .target_buffer = group->spans[0].buffer,
          .count = group->count,
          .enumerator = {weight_enumerate, group},
          .wait_semaphore_list = {1, &ready[lane], &group->wait_value},
          .signal_semaphore_list = {1,
                                    group->spans[0].prepare ? &reads[lane]
                                                            : &ready[lane],
                                    &group->signal_value},
      };
    }
    fprintf(stderr,
            "Streaming %.3f GiB of unique weights; preparing %.3f GiB in "
            "place across %zu groups and %u lanes...\n",
            loaded_bytes / 1073741824.0, prepared_bytes / 1073741824.0,
            group_count, WEIGHT_LANES);
    status = iree_io_parameter_provider_gather_batch(provider, device, transfer,
                                                     group_count, gathers);
  }
  for (iree_host_size_t i = 0; i < group_count && iree_status_is_ok(status);
       ++i) {
    weight_group_t* group = &groups[i];
    const weight_span_t* span = group->spans;
    if (!span->prepare) {
      continue;
    }
    const iree_host_size_t lane = i % WEIGHT_LANES;
    const iree_hal_buffer_binding_t binding = {
        span->buffer, span->parameter.byte_offset, span->parameter.byte_length};
    status = iree_hal_queue_execute(
        dispatch,
        (iree_hal_semaphore_list_t){1, &reads[lane], &group->signal_value},
        (iree_hal_semaphore_list_t){1, &ready[lane], &group->signal_value},
        span->prepare, (iree_hal_buffer_binding_table_t){1, &binding},
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(
        (iree_hal_semaphore_list_t){WEIGHT_LANES, ready, frontiers},
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    // Release waits for preparations that were never submitted. Queue-owned
    // file, command and buffer references survive until those operations
    // retire.
    for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
      if (reads[i]) {
        iree_hal_semaphore_fail(reads[i], iree_status_clone(status));
      }
      if (ready[i]) {
        iree_hal_semaphore_fail(ready[i], iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    iree_hal_semaphore_release(reads[i]);
    iree_hal_semaphore_release(ready[i]);
  }
  iree_allocator_free(allocator, gathers);
  iree_allocator_free(allocator, groups);
  return status;
}

iree_status_t loom_serve_weights_load(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, loom_serve_jit_t* jit,
    iree_hal_command_buffer_mode_t command_mode,
    iree_host_size_t shared_stage_count, iree_host_size_t stage_count,
    const loom_serve_weight_stage_t* stages, iree_string_view_t weights_path,
    iree_string_view_t policy_path, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t capacity =
      shared_stage_count ? stages[0].program->parameters.count : 0;
  for (iree_host_size_t i = shared_stage_count; i < stage_count; ++i) {
    capacity += stages[i].program->parameters.count;
  }
  weight_span_t* spans = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, capacity, sizeof(*spans), (void**)&spans);
  iree_host_size_t span_count = 0;
  if (shared_stage_count && iree_status_is_ok(status)) {
    loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(stages[0].program, 0);
    for (iree_host_size_t i = 1; i < shared_stage_count; ++i) {
      const loom_cmd_program_parameter_root_t other =
          loom_cmd_program_parameter_root_at(stages[i].program, 0);
      root.required_byte_length =
          iree_max(root.required_byte_length, other.required_byte_length);
      root.minimum_alignment =
          iree_max(root.minimum_alignment, other.minimum_alignment);
    }
    status = weight_resolve_root(device, stages[0].program, root, &span_count,
                                 spans, host_allocator, &stages[0].buffers[0]);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 1; i < shared_stage_count; ++i) {
      stages[i].buffers[0] = stages[0].buffers[0];
      iree_hal_buffer_retain(stages[i].buffers[0]);
    }
  }
  for (iree_host_size_t i = shared_stage_count;
       i < stage_count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_t* program = stages[i].program;
    for (uint32_t r = 0;
         r < program->parameter_roots.count && iree_status_is_ok(status); ++r) {
      const loom_cmd_program_parameter_root_t root =
          loom_cmd_program_parameter_root_at(program, r);
      status = weight_resolve_root(device, program, root, &span_count, spans,
                                   host_allocator,
                                   &stages[i].buffers[root.fixed_buffer_index]);
    }
  }
  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    status = weight_prepare_spans(jit, dispatch, command_mode, policy_path,
                                  span_count, spans, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(host_allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_parameter_file_to_index(weights_path, index,
                                                         host_allocator);
  }
  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(iree_string_view_empty(),
                                                     index, WEIGHT_LANES,
                                                     host_allocator, &provider);
  }
  if (iree_status_is_ok(status)) {
    status = weight_stream(device, transfer, dispatch, provider, span_count,
                           spans, host_allocator);
  }
  iree_io_parameter_provider_release(provider);
  iree_io_parameter_index_release(index);
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    iree_hal_command_buffer_release(spans[i].prepare);
  }
  iree_allocator_free(host_allocator, spans);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
