// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"
#include "iree/hal/drivers/amdxdna/dispatch.h"
#include "iree/hal/utils/resource_set.h"

typedef enum iree_hal_amdxdna_cmd_type_e {
  IREE_HAL_AMDXDNA_CMD_EXECUTION_BARRIER = 0,
  IREE_HAL_AMDXDNA_CMD_ADVISE_BUFFER,
  IREE_HAL_AMDXDNA_CMD_COLLECTIVE,
  IREE_HAL_AMDXDNA_CMD_DISPATCH,
} iree_hal_amdxdna_cmd_type_t;

typedef struct iree_hal_amdxdna_cmd_header_t {
  struct iree_hal_amdxdna_cmd_header_t* next;
  iree_hal_amdxdna_cmd_type_t type;
} iree_hal_amdxdna_cmd_header_t;

typedef struct iree_hal_amdxdna_cmd_list_t {
  iree_arena_allocator_t arena;
  iree_hal_amdxdna_cmd_header_t* head;
  iree_hal_amdxdna_cmd_header_t* tail;
} iree_hal_amdxdna_cmd_list_t;

static void iree_hal_amdxdna_cmd_list_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdxdna_cmd_list_t* out_cmd_list) {
  iree_arena_initialize(block_pool, &out_cmd_list->arena);
  out_cmd_list->head = NULL;
  out_cmd_list->tail = NULL;
}

static bool iree_hal_amdxdna_cmd_list_is_empty(
    const iree_hal_amdxdna_cmd_list_t* cmd_list) {
  return cmd_list->head == NULL;
}

static void iree_hal_amdxdna_cmd_list_reset(
    iree_hal_amdxdna_cmd_list_t* cmd_list) {
  iree_arena_reset(&cmd_list->arena);
  cmd_list->head = NULL;
  cmd_list->tail = NULL;
}

static void iree_hal_amdxdna_cmd_list_deinitialize(
    iree_hal_amdxdna_cmd_list_t* cmd_list) {
  iree_hal_amdxdna_cmd_list_reset(cmd_list);
}

static iree_status_t iree_hal_amdxdna_cmd_list_append(
    iree_hal_amdxdna_cmd_list_t* cmd_list,
    iree_hal_amdxdna_cmd_type_t command_type, iree_host_size_t command_size,
    void** out_cmd) {
  iree_hal_amdxdna_cmd_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&cmd_list->arena, command_size, (void**)&header));
  header->next = NULL;
  header->type = command_type;
  if (!cmd_list->head) {
    cmd_list->head = header;
  } else {
    cmd_list->tail->next = header;
  }
  cmd_list->tail = header;
  *out_cmd = header;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_cmd_list_clone_data(
    iree_hal_amdxdna_cmd_list_t* cmd_list, const void* source_data,
    iree_host_size_t data_length, void** out_target_data) {
  void* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&cmd_list->arena, data_length, &target_data));
  memcpy(target_data, source_data, data_length);
  *out_target_data = target_data;
  return iree_ok_status();
}

typedef struct iree_hal_amdxdna_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_c_device_caps_t native_caps;
  iree_hal_resource_set_t* resource_set;
  iree_hal_amdxdna_cmd_list_t cmd_list;
} iree_hal_amdxdna_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_command_buffer_vtable;

static iree_hal_amdxdna_command_buffer_t* iree_hal_amdxdna_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdxdna_command_buffer_vtable);
  return (iree_hal_amdxdna_command_buffer_t*)base_value;
}

static iree_status_t iree_hal_amdxdna_command_buffer_retain_resources(
    iree_hal_amdxdna_command_buffer_t* command_buffer, iree_host_size_t count,
    const void* resources) {
  if (!command_buffer->resource_set || count == 0) return iree_ok_status();
  return iree_hal_resource_set_insert(command_buffer->resource_set, count,
                                      resources);
}

static iree_status_t iree_hal_amdxdna_command_buffer_retain_resources_strided(
    iree_hal_amdxdna_command_buffer_t* command_buffer, iree_host_size_t count,
    const void* data, iree_host_size_t offset, iree_host_size_t stride) {
  if (!command_buffer->resource_set || count == 0) return iree_ok_status();
  return iree_hal_resource_set_insert_strided(command_buffer->resource_set,
                                              count, data, offset, stride);
}

static iree_status_t iree_hal_amdxdna_command_buffer_unsupported_transfer(
    const char* operation) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna %s requires native blit support; host-emulated map/sync/memcpy "
      "transfers are not available on command buffers",
      operation);
}

iree_status_t iree_hal_amdxdna_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_amdxdna_native_c_device_caps_t* native_caps,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(native_caps);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*command_buffer) +
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
      (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
        &iree_hal_amdxdna_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->native_caps = *native_caps;
    command_buffer->resource_set = NULL;
    iree_hal_amdxdna_cmd_list_initialize(block_pool, &command_buffer->cmd_list);
    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
      status = iree_hal_resource_set_allocate(block_pool,
                                              &command_buffer->resource_set);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else if (command_buffer) {
    iree_hal_command_buffer_destroy(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdxdna_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_cmd_list_deinitialize(&command_buffer->cmd_list);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdxdna_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return command_buffer &&
         iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_amdxdna_command_buffer_vtable);
}

static iree_status_t iree_hal_amdxdna_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  if (!iree_hal_amdxdna_cmd_list_is_empty(&command_buffer->cmd_list)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_hal_resource_set_freeze(command_buffer->resource_set);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  (void)base_command_buffer;
  (void)label;
  (void)label_color;
  (void)location;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  return iree_ok_status();
}

typedef struct iree_hal_amdxdna_cmd_execution_barrier_t {
  iree_hal_amdxdna_cmd_header_t header;
  iree_hal_execution_stage_t source_stage_mask;
  iree_hal_execution_stage_t target_stage_mask;
  iree_hal_execution_barrier_flags_t flags;
  iree_host_size_t memory_barrier_count;
  const iree_hal_memory_barrier_t* memory_barriers;
  iree_host_size_t buffer_barrier_count;
  const iree_hal_buffer_barrier_t* buffer_barriers;
} iree_hal_amdxdna_cmd_execution_barrier_t;

static iree_status_t iree_hal_amdxdna_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (flags == IREE_HAL_EXECUTION_BARRIER_FLAG_NONE &&
      buffer_barrier_count == 0) {
    // The amdxdna direct command buffer only performs explicit host/device
    // synchronization for buffer barriers. Memory-only ordering barriers have
    // no native work to replay, and command order is already preserved by the
    // command-buffer stream.
    return iree_ok_status();
  }
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_hal_amdxdna_cmd_execution_barrier_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_append(
      &command_buffer->cmd_list, IREE_HAL_AMDXDNA_CMD_EXECUTION_BARRIER,
      sizeof(*cmd), (void**)&cmd));
  cmd->source_stage_mask = source_stage_mask;
  cmd->target_stage_mask = target_stage_mask;
  cmd->flags = flags;
  cmd->memory_barrier_count = memory_barrier_count;
  cmd->memory_barriers = NULL;
  cmd->buffer_barrier_count = buffer_barrier_count;
  cmd->buffer_barriers = NULL;
  if (memory_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_clone_data(
        &command_buffer->cmd_list, memory_barriers,
        memory_barrier_count * sizeof(memory_barriers[0]),
        (void**)&cmd->memory_barriers));
  }
  if (buffer_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_clone_data(
        &command_buffer->cmd_list, buffer_barriers,
        buffer_barrier_count * sizeof(buffer_barriers[0]),
        (void**)&cmd->buffer_barriers));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_apply_execution_barrier(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdxdna_cmd_execution_barrier_t* cmd) {
  const iree_hal_buffer_barrier_t* buffer_barriers = cmd->buffer_barriers;
  iree_hal_buffer_barrier_t* resolved_buffer_barriers = NULL;
  if (cmd->buffer_barrier_count > 0) {
    resolved_buffer_barriers = (iree_hal_buffer_barrier_t*)iree_alloca(
        cmd->buffer_barrier_count * sizeof(*resolved_buffer_barriers));
    memcpy(resolved_buffer_barriers, cmd->buffer_barriers,
           cmd->buffer_barrier_count * sizeof(*resolved_buffer_barriers));
    for (iree_host_size_t i = 0; i < cmd->buffer_barrier_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
          binding_table, cmd->buffer_barriers[i].buffer_ref,
          &resolved_buffer_barriers[i].buffer_ref));
    }
    buffer_barriers = resolved_buffer_barriers;
  }
  return iree_hal_command_buffer_execution_barrier(
      target_command_buffer, cmd->source_stage_mask, cmd->target_stage_mask,
      cmd->flags, cmd->memory_barrier_count, cmd->memory_barriers,
      cmd->buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_amdxdna_command_buffer_atomic_wait(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "amdxdna command buffers do not support atomic waits");
}

static iree_status_t iree_hal_amdxdna_command_buffer_atomic_store(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna command buffers do not support atomic stores");
}

static iree_status_t iree_hal_amdxdna_command_buffer_atomic_rmw(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  (void)base_command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "amdxdna command buffers do not support atomic read-modify-write");
}

typedef struct iree_hal_amdxdna_cmd_advise_buffer_t {
  iree_hal_amdxdna_cmd_header_t header;
  iree_hal_buffer_ref_t buffer_ref;
  iree_hal_memory_advise_flags_t flags;
  uint64_t arg0;
  uint64_t arg1;
} iree_hal_amdxdna_cmd_advise_buffer_t;

static iree_status_t iree_hal_amdxdna_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  if (buffer_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_command_buffer_retain_resources(
        command_buffer, 1, &buffer_ref.buffer));
  }
  iree_hal_amdxdna_cmd_advise_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_append(
      &command_buffer->cmd_list, IREE_HAL_AMDXDNA_CMD_ADVISE_BUFFER,
      sizeof(*cmd), (void**)&cmd));
  cmd->buffer_ref = buffer_ref;
  cmd->flags = flags;
  cmd->arg0 = arg0;
  cmd->arg1 = arg1;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_apply_advise_buffer(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdxdna_cmd_advise_buffer_t* cmd) {
  iree_hal_buffer_ref_t buffer_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->buffer_ref, &buffer_ref));
  return iree_hal_command_buffer_advise_buffer(
      target_command_buffer, buffer_ref, cmd->flags, cmd->arg0, cmd->arg1);
}

static iree_status_t iree_hal_amdxdna_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)target_ref;
  (void)pattern;
  (void)pattern_length;
  (void)flags;
  return iree_hal_amdxdna_command_buffer_unsupported_transfer(
      "command_buffer_fill_buffer");
}

static iree_status_t iree_hal_amdxdna_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  if (target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)source_buffer;
  (void)source_offset;
  (void)target_ref;
  (void)flags;
  return iree_hal_amdxdna_command_buffer_unsupported_transfer(
      "command_buffer_update_buffer");
}

static iree_status_t iree_hal_amdxdna_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  if (source_ref.length == 0 && target_ref.length == 0) return iree_ok_status();
  (void)base_command_buffer;
  (void)source_ref;
  (void)target_ref;
  (void)flags;
  return iree_hal_amdxdna_command_buffer_unsupported_transfer(
      "command_buffer_copy_buffer");
}

typedef struct iree_hal_amdxdna_cmd_collective_t {
  iree_hal_amdxdna_cmd_header_t header;
  iree_hal_channel_t* channel;
  iree_hal_collective_op_t op;
  uint32_t param;
  iree_hal_buffer_ref_t send_ref;
  iree_hal_buffer_ref_t recv_ref;
  iree_device_size_t element_count;
} iree_hal_amdxdna_cmd_collective_t;

static iree_status_t iree_hal_amdxdna_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_host_size_t resource_count = 0;
  const void* resources[3] = {NULL, NULL, NULL};
  resources[resource_count++] = channel;
  if (send_ref.buffer) resources[resource_count++] = send_ref.buffer;
  if (recv_ref.buffer) resources[resource_count++] = recv_ref.buffer;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_command_buffer_retain_resources(
      command_buffer, resource_count, resources));
  iree_hal_amdxdna_cmd_collective_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_append(
      &command_buffer->cmd_list, IREE_HAL_AMDXDNA_CMD_COLLECTIVE, sizeof(*cmd),
      (void**)&cmd));
  cmd->channel = channel;
  cmd->op = op;
  cmd->param = param;
  cmd->send_ref = send_ref;
  cmd->recv_ref = recv_ref;
  cmd->element_count = element_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_apply_collective(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdxdna_cmd_collective_t* cmd) {
  iree_hal_buffer_ref_t send_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->send_ref, &send_ref));
  iree_hal_buffer_ref_t recv_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->recv_ref, &recv_ref));
  return iree_hal_command_buffer_collective(target_command_buffer, cmd->channel,
                                            cmd->op, cmd->param, send_ref,
                                            recv_ref, cmd->element_count);
}

typedef struct iree_hal_amdxdna_cmd_dispatch_t {
  iree_hal_amdxdna_cmd_header_t header;
  iree_hal_amdxdna_dispatch_plan_t plan;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} iree_hal_amdxdna_cmd_dispatch_t;

static iree_status_t iree_hal_amdxdna_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_validate_dispatch(&config, flags));
  if (iree_hal_amdxdna_dispatch_is_zero_workgroups(&config, flags)) {
    return iree_ok_status();
  }
  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_hal_amdxdna_dispatch_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_dispatch_plan_initialize(
      &command_buffer->native_caps, executable, export_ordinal, &plan));
  iree_host_size_t resource_count = 0;
  const void* resources[2] = {NULL, NULL};
  resources[resource_count++] = executable;
  if (iree_hal_dispatch_uses_indirect_parameters(flags) &&
      config.workgroup_count_ref.buffer) {
    resources[resource_count++] = config.workgroup_count_ref.buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_command_buffer_retain_resources(
      command_buffer, resource_count, resources));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_command_buffer_retain_resources_strided(
      command_buffer, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_host_size_t total_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdxdna_cmd_dispatch_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(constants.data_length, uint8_t,
                                iree_max_align_t, &constants_offset),
      IREE_STRUCT_FIELD_ALIGNED(bindings.count, iree_hal_buffer_ref_t, 1,
                                &bindings_offset)));

  iree_hal_amdxdna_cmd_dispatch_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_cmd_list_append(
      &command_buffer->cmd_list, IREE_HAL_AMDXDNA_CMD_DISPATCH, total_size,
      (void**)&cmd));
  cmd->plan = plan;
  cmd->config = config;
  cmd->flags = flags;

  uint8_t* cmd_base = (uint8_t*)cmd;
  memcpy(cmd_base + constants_offset, constants.data, constants.data_length);
  cmd->constants = iree_make_const_byte_span(cmd_base + constants_offset,
                                             constants.data_length);
  cmd->bindings.count = bindings.count;
  memcpy(cmd_base + bindings_offset, bindings.values,
         bindings.count * sizeof(bindings.values[0]));
  cmd->bindings.values = (iree_hal_buffer_ref_t*)(cmd_base + bindings_offset);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_apply_dispatch(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdxdna_cmd_dispatch_t* cmd) {
  iree_hal_dispatch_config_t config = cmd->config;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->config.workgroup_count_ref,
      &config.workgroup_count_ref));
  iree_hal_buffer_ref_t* binding_refs = NULL;
  if (cmd->bindings.count > 0) {
    binding_refs = (iree_hal_buffer_ref_t*)iree_alloca(cmd->bindings.count *
                                                       sizeof(*binding_refs));
  }
  for (iree_host_size_t i = 0; i < cmd->bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, cmd->bindings.values[i], &binding_refs[i]));
  }
  const iree_hal_buffer_ref_list_t binding_ref_list = {
      .count = cmd->bindings.count,
      .values = binding_refs,
  };
  return iree_hal_amdxdna_direct_command_buffer_dispatch_plan(
      target_command_buffer, &cmd->plan, cmd->constants, binding_ref_list,
      cmd->flags);
}

typedef iree_status_t (*iree_hal_amdxdna_cmd_apply_fn_t)(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdxdna_cmd_header_t* cmd_header);

static const iree_hal_amdxdna_cmd_apply_fn_t
    iree_hal_amdxdna_cmd_apply_table[] = {
        [IREE_HAL_AMDXDNA_CMD_EXECUTION_BARRIER] =
            (iree_hal_amdxdna_cmd_apply_fn_t)
                iree_hal_amdxdna_apply_execution_barrier,
        [IREE_HAL_AMDXDNA_CMD_ADVISE_BUFFER] = (iree_hal_amdxdna_cmd_apply_fn_t)
            iree_hal_amdxdna_apply_advise_buffer,
        [IREE_HAL_AMDXDNA_CMD_COLLECTIVE] =
            (iree_hal_amdxdna_cmd_apply_fn_t)iree_hal_amdxdna_apply_collective,
        [IREE_HAL_AMDXDNA_CMD_DISPATCH] =
            (iree_hal_amdxdna_cmd_apply_fn_t)iree_hal_amdxdna_apply_dispatch,
};

iree_status_t iree_hal_amdxdna_command_buffer_apply(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_command_buffer_t* command_buffer =
      iree_hal_amdxdna_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_hal_command_buffer_begin(target_command_buffer);
  if (iree_status_is_ok(status)) {
    for (iree_hal_amdxdna_cmd_header_t* cmd = command_buffer->cmd_list.head;
         cmd != NULL; cmd = cmd->next) {
      status = iree_hal_amdxdna_cmd_apply_table[cmd->type](
          target_command_buffer, binding_table, cmd);
      if (!iree_status_is_ok(status)) break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(target_command_buffer);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(command_buffer->base.mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    iree_hal_amdxdna_cmd_list_reset(&command_buffer->cmd_list);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_command_buffer_vtable = {
        .destroy = iree_hal_amdxdna_command_buffer_destroy,
        .begin = iree_hal_amdxdna_command_buffer_begin,
        .end = iree_hal_amdxdna_command_buffer_end,
        .begin_debug_group = iree_hal_amdxdna_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_amdxdna_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_amdxdna_command_buffer_execution_barrier,
        .atomic_wait = iree_hal_amdxdna_command_buffer_atomic_wait,
        .atomic_store = iree_hal_amdxdna_command_buffer_atomic_store,
        .atomic_rmw = iree_hal_amdxdna_command_buffer_atomic_rmw,
        .advise_buffer = iree_hal_amdxdna_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_amdxdna_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdxdna_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdxdna_command_buffer_copy_buffer,
        .collective = iree_hal_amdxdna_command_buffer_collective,
        .dispatch = iree_hal_amdxdna_command_buffer_dispatch,
};
