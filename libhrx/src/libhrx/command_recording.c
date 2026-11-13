// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "command_recording.h"

iree_hal_dispatch_config_t hrx_command_dispatch_config(
    const hrx_command_dispatch_t* dispatch) {
  iree_hal_dispatch_config_t config = {
      .workgroup_count = {dispatch->workgroup_count[0],
                          dispatch->workgroup_count[1],
                          dispatch->workgroup_count[2]},
      .workgroup_size = {dispatch->workgroup_size[0],
                         dispatch->workgroup_size[1],
                         dispatch->workgroup_size[2]},
  };
  if (dispatch->extended_geometry) {
    const hrx_command_dispatch_extended_t* extended = (const void*)dispatch;
    memcpy(config.workitem_count, extended->workitem_count,
           sizeof(config.workitem_count));
    config.dynamic_workgroup_local_memory =
        extended->dynamic_workgroup_local_memory;
  }
  return config;
}

static iree_status_t hrx_command_record_atomic(
    iree_hal_command_buffer_t* command_buffer,
    const hrx_atomic_operation_t* operation,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage) {
  switch (operation->kind) {
    case HRX_ATOMIC_OPERATION_WAIT:
      return iree_hal_command_buffer_atomic_wait(
          command_buffer, source_stage, target_stage,
          iree_hal_make_buffer_ref(operation->target_buffer,
                                   operation->target_offset,
                                   operation->params.wait.width / 8),
          operation->params.wait);
    case HRX_ATOMIC_OPERATION_STORE:
      return iree_hal_command_buffer_atomic_store(
          command_buffer, source_stage, target_stage,
          iree_hal_make_buffer_ref(operation->target_buffer,
                                   operation->target_offset,
                                   operation->params.store.width / 8),
          operation->params.store);
    case HRX_ATOMIC_OPERATION_UPDATE:
      return iree_hal_command_buffer_atomic_rmw(
          command_buffer, source_stage, target_stage,
          iree_hal_make_buffer_ref(operation->target_buffer,
                                   operation->target_offset,
                                   operation->params.update.width / 8),
          operation->params.update);
    default:
      IREE_CHECK_UNREACHABLE("invalid resolved atomic operation");
  }
}

iree_status_t hrx_command_record_atomics(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t operation_count,
    const hrx_atomic_operation_t* operations,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < operation_count;
       ++i) {
    status = hrx_command_record_atomic(
        command_buffer, &operations[i],
        i == 0 ? source_stage : IREE_HAL_EXECUTION_STAGE_ATOMIC, target_stage);
  }
  return status;
}

static iree_status_t hrx_command_record_copy(
    iree_hal_command_buffer_t* command_buffer, const hrx_command_copy_t* copy) {
  if (!copy->length) {
    return iree_ok_status();
  }
  if (!copy->strided) {
    if (copy->source_buffer == copy->target_buffer &&
        copy->source_offset == copy->target_offset) {
      return iree_ok_status();
    }
    return iree_hal_command_buffer_copy_buffer(
        command_buffer,
        iree_hal_make_buffer_ref(copy->source_buffer, copy->source_offset,
                                 copy->length),
        iree_hal_make_buffer_ref(copy->target_buffer, copy->target_offset,
                                 copy->length),
        copy->flags);
  }
  const hrx_command_copy_strided_t* strided = (const void*)copy;
  iree_status_t status = iree_ok_status();
  for (iree_device_size_t z = 0;
       iree_status_is_ok(status) && z < strided->slice_count; ++z) {
    iree_device_size_t source_offset =
        copy->source_offset + z * strided->source_slice_stride;
    iree_device_size_t target_offset =
        copy->target_offset + z * strided->target_slice_stride;
    for (iree_device_size_t y = 0;
         iree_status_is_ok(status) && y < strided->row_count; ++y) {
      if (copy->source_buffer != copy->target_buffer ||
          source_offset != target_offset) {
        status = iree_hal_command_buffer_copy_buffer(
            command_buffer,
            iree_hal_make_buffer_ref(copy->source_buffer, source_offset,
                                     copy->length),
            iree_hal_make_buffer_ref(copy->target_buffer, target_offset,
                                     copy->length),
            copy->flags);
      }
      source_offset += strided->source_row_stride;
      target_offset += strided->target_row_stride;
    }
  }
  return status;
}

iree_status_t hrx_graph_record_commands(
    const hrx_graph_schedule_t* schedule, uint32_t node_start_index,
    uint32_t node_count, uint8_t stream_index,
    const iree_hal_barrier_t* dependency_barrier,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_barrier_flags_t* out_release_flags) {
  *out_release_flags = IREE_HAL_BARRIER_FLAG_NONE;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < node_count; ++i) {
    const uint32_t index = schedule->order[node_start_index + i];
    const uint8_t flags = schedule->flags[index];
    if ((flags >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT) != stream_index) {
      continue;
    }
    if (flags & HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE) {
      status =
          iree_hal_command_buffer_barrier(command_buffer, dependency_barrier);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    const hrx_command_t* command = schedule->sources[index];
    switch (command->kind) {
      case HRX_COMMAND_DISPATCH: {
        const hrx_command_dispatch_t* dispatch = (const void*)command;
        status = iree_hal_command_buffer_dispatch(
            command_buffer, dispatch->executable,
            iree_hal_executable_function_from_index(dispatch->export_ordinal),
            hrx_command_dispatch_config(dispatch), dispatch->constants,
            dispatch->bindings, dispatch->flags);
        break;
      }
      case HRX_COMMAND_COPY: {
        const hrx_command_copy_t* copy = (const void*)command;
        status = hrx_command_record_copy(command_buffer, copy);
        if (copy->release_to_host) {
          *out_release_flags |= IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
        }
        break;
      }
      case HRX_COMMAND_FILL: {
        const hrx_command_fill_t* fill = (const void*)command;
        status = iree_hal_command_buffer_fill_buffer(
            command_buffer, fill->target, &fill->pattern, fill->pattern_size,
            fill->flags);
        break;
      }
      case HRX_COMMAND_ATOMIC_BATCH: {
        const hrx_command_atomic_batch_t* batch = (const void*)command;
        status = hrx_command_record_atomics(
            command_buffer, batch->operation_count, batch->operations,
            batch->source_stage, batch->target_stage);
        break;
      }
      default:
        IREE_CHECK_UNREACHABLE("invalid resolved command");
    }
  }
  return status;
}
