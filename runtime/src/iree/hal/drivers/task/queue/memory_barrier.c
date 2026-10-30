// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/queue/memory_barrier.h"

#include "iree/hal/drivers/task/atomic.h"
#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/hal/utils/resource_set.h"

// One native action whose descriptor is owned by the operation arena.
typedef struct iree_hal_task_queue_memory_action_t {
  // Direct logical operand retained by the operation resource set.
  iree_hal_buffer_ref_t ref;
  // Exact selected operation, including native instruction and fence choices.
  iree_hal_memory_transition_recipe_info_t recipe;
} iree_hal_task_queue_memory_action_t;

struct iree_hal_task_queue_memory_barriers_t {
  // Number of actions executed before the operation body.
  iree_host_size_t before_count;
  // Number of actions executed after the operation body.
  iree_host_size_t after_count;
  // Before actions followed by after actions, in descriptor order.
  iree_hal_task_queue_memory_action_t actions[];
};

static iree_hal_atomic_flags_t iree_hal_task_queue_barrier_flags(
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE;
  }
  iree_hal_atomic_flags_t flags = IREE_HAL_ATOMIC_FLAG_NONE;
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    flags |= iree_hal_task_barrier_resolve_atomic_flags(&barriers->values[i]);
  }
  return flags;
}

static iree_status_t iree_hal_task_queue_barrier_action_count(
    const iree_hal_barrier_list_t* barriers, iree_host_size_t* out_count) {
  *out_count = 0;
  if (!barriers) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    for (iree_host_size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
      const iree_hal_memory_transition_recipe_t* recipe =
          barrier->buffer_barriers[j].recipe;
      if (!recipe) {
        continue;
      }
      for (uint32_t k = 0; k < recipe->operation_count; ++k) {
        if (recipe->operations[k].executor ==
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE) {
          continue;
        }
        if (!iree_host_size_checked_add(*out_count, 1, out_count)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "cache action count overflows");
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_barrier_copy_actions(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_barrier_list_t* barriers,
    iree_hal_task_queue_memory_action_t* actions) {
  if (!barriers) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < barriers->count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    for (iree_host_size_t j = 0;
         j < barrier->buffer_barrier_count && iree_status_is_ok(status); ++j) {
      const iree_hal_buffer_barrier_t* range = &barrier->buffer_barriers[j];
      if (!range->recipe) {
        continue;
      }
      iree_hal_task_queue_memory_action_t* first_action = actions;
      for (uint32_t k = 0; k < range->recipe->operation_count; ++k) {
        if (range->recipe->operations[k].executor ==
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE) {
          continue;
        }
        actions->ref = range->buffer_ref;
        actions->recipe = range->recipe->operations[k];
        ++actions;
      }
      if (actions != first_action &&
          !(operation->type == IREE_HAL_TASK_QUEUE_OP_DISPATCH &&
            iree_any_bit_set(
                operation->dispatch.flags,
                IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES))) {
        status = iree_hal_resource_set_insert(operation->resource_set, 1,
                                              &range->buffer_ref.buffer);
      }
    }
  }
  return status;
}

iree_status_t iree_hal_task_queue_memory_barriers_capture(
    iree_hal_task_queue_op_t* operation,
    const iree_hal_queue_barriers_t* barriers) {
  const iree_hal_barrier_list_t* before = barriers ? barriers->before : NULL;
  const iree_hal_barrier_list_t* after = barriers ? barriers->after : NULL;
  const iree_hal_atomic_flags_t before_flags =
      iree_hal_task_queue_barrier_flags(before);
  const iree_hal_atomic_flags_t after_flags =
      iree_hal_task_queue_barrier_flags(after);
  if (before_flags & IREE_HAL_ATOMIC_FLAG_ACQUIRE) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_ACQUIRE;
  }
  if (before_flags & IREE_HAL_ATOMIC_FLAG_RELEASE) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_RELEASE;
  }
  if (after_flags & IREE_HAL_ATOMIC_FLAG_ACQUIRE) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_ACQUIRE;
  }
  if (after_flags & IREE_HAL_ATOMIC_FLAG_RELEASE) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_RELEASE;
  }
  if (!before) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_DEFAULT;
  }
  if (!after) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_DEFAULT;
  }
  iree_host_size_t before_count = 0, after_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_task_queue_barrier_action_count(before, &before_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_task_queue_barrier_action_count(after, &after_count));
  if (!before_count && !after_count) {
    return iree_ok_status();
  }
  iree_host_size_t count = 0, bytes = 0;
  if (!iree_host_size_checked_add(before_count, after_count, &count) ||
      !iree_host_size_checked_mul(
          count, sizeof(iree_hal_task_queue_memory_action_t), &bytes) ||
      !iree_host_size_checked_add(bytes, sizeof(*operation->memory_barriers),
                                  &bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "cache action storage overflows");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &operation->arena, bytes, (void**)&operation->memory_barriers));
  operation->memory_barriers->before_count = before_count;
  operation->memory_barriers->after_count = after_count;
  IREE_RETURN_IF_ERROR(iree_hal_task_queue_barrier_copy_actions(
      operation, before, operation->memory_barriers->actions));
  return iree_hal_task_queue_barrier_copy_actions(
      operation, after, operation->memory_barriers->actions + before_count);
}

static iree_status_t iree_hal_task_queue_barrier_execute_actions(
    iree_host_size_t count,
    const iree_hal_task_queue_memory_action_t* actions) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_task_queue_memory_action_t* action = &actions[i];
    const iree_hal_buffer_ref_t ref = action->ref;
    if (action->recipe.executor ==
        IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT) {
      iree_byte_span_t span;
      status = iree_hal_buffer_native_host_span(ref.buffer, ref.offset,
                                                ref.length, &span);
      if (iree_status_is_ok(status)) {
        iree_hal_memory_transition_execute_host_direct(
            &action->recipe, span.data, span.data_length);
      }
    } else if (action->recipe.operation ==
               IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH) {
      status =
          iree_hal_buffer_flush_host_range(ref.buffer, ref.offset, ref.length);
    } else {
      status = iree_hal_buffer_invalidate_host_range(ref.buffer, ref.offset,
                                                     ref.length);
    }
  }
  return status;
}

// Default one-shot boundaries use already-qualified native buffer operations.
// Explicit recipes above always execute, including on CPU-coherent storage.
static iree_status_t iree_hal_task_queue_barrier_default_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_memory_access_t access,
    iree_hal_memory_access_t boundary_access) {
  if (!(access & boundary_access) || !length ||
      iree_any_bit_set(iree_hal_buffer_memory_type(buffer),
                       IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return iree_ok_status();
  }
  return boundary_access == IREE_HAL_MEMORY_ACCESS_READ
             ? iree_hal_buffer_invalidate_host_range(buffer, offset, length)
             : iree_hal_buffer_flush_host_range(buffer, offset, length);
}

static iree_status_t iree_hal_task_queue_barrier_default_transfer(
    const iree_hal_transfer_operation_t* transfer,
    iree_hal_memory_access_t access) {
  switch (transfer->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      return iree_hal_task_queue_barrier_default_range(
          transfer->fill.target_buffer, transfer->fill.target_offset,
          transfer->fill.length, IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return iree_hal_task_queue_barrier_default_range(
          transfer->update.target_buffer, transfer->update.target_offset,
          transfer->update.length, IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY: {
      IREE_RETURN_IF_ERROR(iree_hal_task_queue_barrier_default_range(
          transfer->copy.source_buffer, transfer->copy.source_offset,
          transfer->copy.length, IREE_HAL_MEMORY_ACCESS_READ, access));
      return iree_hal_task_queue_barrier_default_range(
          transfer->copy.target_buffer, transfer->copy.target_offset,
          transfer->copy.length, IREE_HAL_MEMORY_ACCESS_WRITE, access);
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      return iree_hal_task_queue_barrier_default_range(
          transfer->upload.target_buffer, transfer->upload.target_offset,
          transfer->upload.length, IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      return iree_hal_task_queue_barrier_default_range(
          transfer->download.source_buffer, transfer->download.source_offset,
          transfer->download.length, IREE_HAL_MEMORY_ACCESS_READ, access);
  }
  IREE_ASSERT_UNREACHABLE("unhandled transfer operation");
  return iree_ok_status();
}

static iree_status_t iree_hal_task_queue_barrier_default_operands(
    iree_hal_task_queue_op_t* operation, iree_hal_memory_access_t access) {
  iree_status_t status = iree_ok_status();
  switch (operation->type) {
    case IREE_HAL_TASK_QUEUE_OP_READ:
      return iree_hal_task_queue_barrier_default_range(
          operation->read.buffer, operation->read.buffer_offset,
          operation->read.length, IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TASK_QUEUE_OP_WRITE:
      return iree_hal_task_queue_barrier_default_range(
          operation->write.buffer, operation->write.buffer_offset,
          operation->write.length, IREE_HAL_MEMORY_ACCESS_READ, access);
    case IREE_HAL_TASK_QUEUE_OP_TRANSFER:
      for (iree_host_size_t i = 0;
           i < operation->transfer.operation_count && iree_status_is_ok(status);
           ++i) {
        status = iree_hal_task_queue_barrier_default_transfer(
            &operation->transfer.operations[i], access);
      }
      break;
    case IREE_HAL_TASK_QUEUE_OP_ATOMIC_WAIT:
      return iree_hal_task_queue_barrier_default_range(
          operation->atomic_wait.target_buffer,
          operation->atomic_wait.target_offset,
          iree_hal_atomic_width_byte_count(operation->atomic_wait.params.width),
          IREE_HAL_MEMORY_ACCESS_READ, access);
    case IREE_HAL_TASK_QUEUE_OP_TIMESTAMP:
      return iree_hal_task_queue_barrier_default_range(
          operation->timestamp.target_buffer,
          operation->timestamp.target_offset, sizeof(uint64_t),
          IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TASK_QUEUE_OP_ATOMIC_STORE:
      return iree_hal_task_queue_barrier_default_range(
          operation->atomic_store.target_buffer,
          operation->atomic_store.target_offset,
          iree_hal_atomic_width_byte_count(
              operation->atomic_store.params.width),
          IREE_HAL_MEMORY_ACCESS_WRITE, access);
    case IREE_HAL_TASK_QUEUE_OP_ATOMIC_RMW:
      return iree_hal_task_queue_barrier_default_range(
          operation->atomic_rmw.target_buffer,
          operation->atomic_rmw.target_offset,
          iree_hal_atomic_width_byte_count(operation->atomic_rmw.params.width),
          IREE_HAL_MEMORY_ACCESS_ALL, access);
    case IREE_HAL_TASK_QUEUE_OP_DISPATCH:
      if (iree_hal_dispatch_uses_indirect_parameters(
              operation->dispatch.flags)) {
        const iree_hal_buffer_ref_t ref =
            operation->dispatch.config.workgroup_count_ref;
        status = iree_hal_task_queue_barrier_default_range(
            ref.buffer, ref.offset, ref.length, IREE_HAL_MEMORY_ACCESS_READ,
            access);
      }
      for (iree_host_size_t i = 0;
           i < operation->dispatch.binding_count && iree_status_is_ok(status);
           ++i) {
        const iree_hal_buffer_ref_t ref = operation->dispatch.bindings[i];
        status = iree_hal_task_queue_barrier_default_range(
            ref.buffer, ref.offset, ref.length,
            iree_hal_buffer_allowed_access(ref.buffer), access);
      }
      break;
    default:
      // Recorded command buffers carry their own range recipes. Other operation
      // kinds do not expose payload operands to this execution boundary.
      break;
  }
  return status;
}

iree_status_t iree_hal_task_queue_memory_barriers_before(
    iree_hal_task_queue_op_t* operation) {
  iree_hal_task_atomic_fence(
      ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_ACQUIRE)
           ? IREE_HAL_ATOMIC_FLAG_ACQUIRE
           : 0) |
      ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_RELEASE)
           ? IREE_HAL_ATOMIC_FLAG_RELEASE
           : 0));
  if (operation->memory_barriers) {
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_barrier_execute_actions(
        operation->memory_barriers->before_count,
        operation->memory_barriers->actions));
  }
  return (operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_DEFAULT)
             ? iree_hal_task_queue_barrier_default_operands(
                   operation, IREE_HAL_MEMORY_ACCESS_READ)
             : iree_ok_status();
}

iree_status_t iree_hal_task_queue_memory_barriers_after(
    iree_hal_task_queue_op_t* operation) {
  if (operation->memory_barriers) {
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_barrier_execute_actions(
        operation->memory_barriers->after_count,
        operation->memory_barriers->actions +
            operation->memory_barriers->before_count));
  }
  if (operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_DEFAULT) {
    IREE_RETURN_IF_ERROR(iree_hal_task_queue_barrier_default_operands(
        operation, IREE_HAL_MEMORY_ACCESS_WRITE));
  }
  iree_hal_task_atomic_fence(
      ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_ACQUIRE)
           ? IREE_HAL_ATOMIC_FLAG_ACQUIRE
           : 0) |
      ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_RELEASE)
           ? IREE_HAL_ATOMIC_FLAG_RELEASE
           : 0));
  return iree_ok_status();
}

bool iree_hal_task_queue_memory_barriers_file_needs_worker(
    const iree_hal_task_queue_op_t* operation) {
  return (operation->memory_barriers &&
          operation->memory_barriers->after_count) ||
         ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_AFTER_DEFAULT) &&
          operation->type == IREE_HAL_TASK_QUEUE_OP_READ &&
          operation->read.length &&
          !iree_any_bit_set(iree_hal_buffer_memory_type(operation->read.buffer),
                            IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}
