// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/stages.h"

#include "experimental/loom_serve/text/state.h"

static iree_status_t text_check_layouts(
    const loom_serve_preparation_t* preparation, iree_host_size_t shared_count,
    iree_host_size_t stage_count, const loom_serve_text_stage_t* stages) {
  const loom_cmd_program_t* prefill = &stages[0].program;
  const loom_serve_preparation_stage_t* first =
      loom_serve_preparation_stage(preparation, 0);
  if (!first->parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source must declare a checkpoint domain");
  }
  const loom_serve_preparation_parameter_t* checkpoint = &first->parameters[0];
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    const loom_cmd_program_t* program = &stages[i].program;
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(preparation, i);
    const uint32_t binding_count =
        program->requirements.rebindable_binding_count;
    const uint32_t roots = program->requirements.fixed_buffer_count;
    if (program->parameter_roots.count != roots ||
        declaration->parameter_count != roots || binding_count < 1 ||
        binding_count > 8 ||
        (program->requirements.transient.binding_index != UINT32_MAX &&
         program->requirements.transient.binding_index != binding_count - 1) ||
        program->requirements.launch_counts.required_byte_length != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage does not match the token runner bindings");
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_serve_preparation_parameter_t* parameter =
          &declaration->parameters[root];
      if (parameter->binding != root ||
          loom_cmd_program_parameter_root_at(program, root)
                  .fixed_buffer_index != root) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "stage parameter bindings must match fixed-root order");
      }
    }
    if (i >= shared_count) {
      continue;
    }
    const bool has_workspace =
        program->requirements.transient.required_byte_length != 0;
    if (roots != 1 || binding_count - has_workspace != TEXT_BINDING_COUNT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared stages must implement the target bindings");
    }
    if (!iree_string_view_equal(declaration->parameters[0].path,
                                checkpoint->path) ||
        !iree_string_view_equal(declaration->parameters[0].policy,
                                checkpoint->policy)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "shared target stages must use one checkpoint domain");
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

// Group roots once during binding. The original stage order preserves the
// identical-layout target prefix in the first domain.
iree_status_t loom_serve_text_stages_bind(
    loom_serve_device_t* device, const loom_serve_preparation_t* preparation,
    iree_host_size_t shared_count, iree_host_size_t stage_count,
    loom_serve_text_stage_t* stages, loom_serve_text_parameters_t* parameters,
    iree_allocator_t allocator) {
  iree_status_t status = iree_ok_status();
  iree_host_size_t root_count = 0;
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_text_stage_t* stage = &stages[i];
    stage->program = *loom_serve_jit_stage_program(stage->compiled);
    const uint32_t count = stage->program.requirements.fixed_buffer_count;
    if (count) {
      status = iree_allocator_malloc_array(allocator, count,
                                           sizeof(*stage->fixed_buffers),
                                           (void**)&stage->fixed_buffers);
      if (iree_status_is_ok(status)) {
        stage->fixed_buffer_count = count;
        root_count += count;
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      text_check_layouts(preparation, shared_count, stage_count, stages));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, root_count,
                                                   sizeof(*parameters->plans),
                                                   (void**)&parameters->plans));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, root_count, sizeof(*parameters->sources),
      (void**)&parameters->sources));
  loom_serve_weight_root_t* roots = NULL;
  iree_host_size_t* domains = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, root_count, sizeof(*roots), (void**)&roots));
  status = iree_allocator_malloc_array(allocator, root_count, sizeof(*domains),
                                       (void**)&domains);
  if (iree_status_is_ok(status)) {
    iree_host_size_t root_index = 0;
    for (iree_host_size_t i = 0; i < stage_count; ++i) {
      const loom_serve_preparation_stage_t* declaration =
          loom_serve_preparation_stage(preparation, i);
      for (uint32_t r = 0; r < declaration->parameter_count; ++r) {
        const loom_serve_preparation_parameter_t* source =
            &declaration->parameters[r];
        iree_host_size_t domain = 0;
        for (; domain < parameters->count; ++domain) {
          const loom_serve_preparation_parameter_t* prior =
              parameters->sources[domain];
          if (iree_string_view_equal(source->path, prior->path) &&
              iree_string_view_equal(source->policy, prior->policy)) {
            break;
          }
        }
        if (domain == parameters->count) {
          parameters->sources[parameters->count++] = source;
        }
        domains[root_index++] = domain;
      }
    }
    for (iree_host_size_t domain = 0;
         domain < parameters->count && iree_status_is_ok(status); ++domain) {
      iree_host_size_t root_index = 0;
      iree_host_size_t count = 0;
      for (iree_host_size_t i = 0; i < stage_count; ++i) {
        const loom_cmd_program_t* program = &stages[i].program;
        for (uint32_t r = 0; r < program->parameter_roots.count; ++r) {
          if (domains[root_index++] != domain) {
            continue;
          }
          const loom_cmd_program_parameter_root_t root =
              loom_cmd_program_parameter_root_at(program, r);
          roots[count++] = (loom_serve_weight_root_t){
              .program = program,
              .root = root,
              .buffer = &stages[i].fixed_buffers[root.fixed_buffer_index],
          };
        }
      }
      status = loom_serve_weights_create(
          loom_serve_device_handle(device),
          loom_serve_device_transfer_queue(device),
          loom_serve_device_dispatch_queue(device),
          loom_serve_device_storage_pool(device),
          loom_serve_device_memory_pool(device), domain == 0 ? shared_count : 0,
          count, roots, &parameters->plans[domain], allocator);
    }
  }
  iree_allocator_free(allocator, domains);
  iree_allocator_free(allocator, roots);
  return status;
}
