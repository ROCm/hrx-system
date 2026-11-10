// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/group.h"

struct loom_serve_execution_group_t {
  iree_allocator_t allocator;
  loom_serve_execution_t* execution;
  iree_host_size_t capacity;
  iree_host_size_t count;
  bool submitted;
  bool retired;
  loom_serve_group_node_t nodes[];
};

static iree_hal_buffer_t* loom_serve_group_transfer_buffer(
    const iree_hal_transfer_operation_t* op) {
  switch (op->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      return op->upload.target_buffer;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return op->update.target_buffer;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      return op->download.source_buffer;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      return op->fill.target_buffer;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      return op->copy.source_buffer;
    default:
      return NULL;
  }
}

static void loom_serve_group_node_retire(loom_serve_group_node_t* node) {
  iree_hal_command_buffer_release(node->command_buffer);
  node->command_buffer = NULL;
  iree_hal_executable_release(node->executable);
  node->executable = NULL;
  for (iree_host_size_t i = 0; i < node->bindings.count; ++i) {
    iree_hal_buffer_release(node->bindings.bindings[i].buffer);
  }
  for (iree_host_size_t i = 0; i < node->refs.count; ++i) {
    iree_hal_buffer_release(node->refs.values[i].buffer);
  }
  node->bindings.count = node->refs.count = 0;
  if (node->kind == LOOM_SERVE_GROUP_TRANSFER) {
    iree_hal_buffer_release(loom_serve_group_transfer_buffer(&node->transfer));
    if (node->transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
      iree_hal_buffer_release(node->transfer.copy.target_buffer);
    }
    // Payload pointers stay owned until actual retirement.
    switch (node->transfer.type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        node->transfer.upload.target_buffer = NULL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        node->transfer.update.target_buffer = NULL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        node->transfer.download.source_buffer = NULL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        node->transfer.fill.target_buffer = NULL;
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        node->transfer.copy.source_buffer = node->transfer.copy.target_buffer =
            NULL;
        break;
      default:
        break;
    }
  }
}

void loom_serve_execution_group_retire(loom_serve_execution_group_t* group) {
  if (!group || group->retired) {
    return;
  }
  group->retired = true;
  for (iree_host_size_t i = 0; i < group->count; ++i) {
    loom_serve_group_node_retire(&group->nodes[i]);
  }
}

void loom_serve_execution_group_free(loom_serve_execution_group_t* group) {
  if (!group) {
    return;
  }
  loom_serve_execution_group_retire(group);
  for (iree_host_size_t i = 0; i < group->count; ++i) {
    loom_serve_group_node_t* node = &group->nodes[i];
    iree_allocator_free(group->allocator, (void*)node->bindings.bindings);
    iree_allocator_free(group->allocator, (void*)node->refs.values);
    if (node->kind == LOOM_SERVE_GROUP_TRANSFER) {
      if (node->transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
        iree_allocator_free(group->allocator,
                            (void*)node->transfer.upload.source);
      } else if (node->transfer.type ==
                 IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) {
        iree_allocator_free(group->allocator,
                            (void*)node->transfer.update.source_buffer);
      } else if (node->transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL) {
        iree_allocator_free(group->allocator,
                            (void*)node->transfer.fill.pattern);
      }
    }
  }
  loom_serve_execution_release(group->execution);
  iree_allocator_free(group->allocator, group);
}

iree_status_t loom_serve_execution_group_create(
    loom_serve_execution_t* execution, iree_host_size_t capacity,
    iree_allocator_t allocator, loom_serve_execution_group_t** out_group) {
  *out_group = NULL;
  if (!execution || !capacity || capacity > 65536) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "group capacity must be in [1, 65536]");
  }
  loom_serve_execution_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      allocator, sizeof(*group), capacity, sizeof(*group->nodes),
      (void**)&group));
  group->allocator = allocator;
  group->execution = execution;
  loom_serve_execution_retain(execution);
  group->capacity = capacity;
  *out_group = group;
  return iree_ok_status();
}

static bool loom_serve_group_range(iree_hal_buffer_t* buffer,
                                   iree_device_size_t offset,
                                   iree_device_size_t length) {
  return buffer && offset <= iree_hal_buffer_byte_length(buffer) &&
         (length == IREE_HAL_WHOLE_BUFFER ||
          length <= iree_hal_buffer_byte_length(buffer) - offset);
}

static iree_status_t loom_serve_group_validate_node(
    const loom_serve_group_node_t* node) {
  if (node->kind == LOOM_SERVE_GROUP_EXECUTE && node->command_buffer) {
    if (node->bindings.count > 65536 ||
        (node->bindings.count && !node->bindings.bindings)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid group binding table");
    }
    for (iree_host_size_t i = 0; i < node->bindings.count; ++i) {
      const iree_hal_buffer_binding_t* binding = &node->bindings.bindings[i];
      if (binding->buffer &&
          !loom_serve_group_range(binding->buffer, binding->offset,
                                  binding->length)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "group command binding exceeds its buffer");
      }
    }
    return iree_hal_command_buffer_validate_submission(node->command_buffer,
                                                       node->bindings);
  }
  if (node->kind == LOOM_SERVE_GROUP_DISPATCH && node->executable) {
    if (node->refs.count > 65536 || (node->refs.count && !node->refs.values)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid group dispatch bindings");
    }
    for (iree_host_size_t i = 0; i < node->refs.count; ++i) {
      const iree_hal_buffer_ref_t* binding = &node->refs.values[i];
      if (!loom_serve_group_range(binding->buffer, binding->offset,
                                  binding->length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "group dispatch requires direct in-range buffers");
      }
    }
    return iree_ok_status();
  }
  if (node->kind == LOOM_SERVE_GROUP_TRANSFER) {
    const iree_hal_transfer_operation_t* op = &node->transfer;
    bool valid = false;
    switch (op->type) {
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
        valid =
            (!op->upload.length || op->upload.source) &&
            op->upload.length <= IREE_HOST_SIZE_MAX &&
            loom_serve_group_range(op->upload.target_buffer,
                                   op->upload.target_offset, op->upload.length);
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
        valid =
            (!op->update.length || op->update.source_buffer) &&
            op->update.length <= IREE_HOST_SIZE_MAX &&
            op->update.source_offset <=
                IREE_HOST_SIZE_MAX - op->update.length &&
            loom_serve_group_range(op->update.target_buffer,
                                   op->update.target_offset, op->update.length);
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
        valid = (!op->download.length || op->download.target) &&
                op->download.length <= IREE_HOST_SIZE_MAX &&
                loom_serve_group_range(op->download.source_buffer,
                                       op->download.source_offset,
                                       op->download.length);
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
        valid = op->fill.pattern &&
                (op->fill.pattern_length == 1 || op->fill.pattern_length == 2 ||
                 op->fill.pattern_length == 4) &&
                loom_serve_group_range(op->fill.target_buffer,
                                       op->fill.target_offset, op->fill.length);
        break;
      case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
        valid =
            loom_serve_group_range(op->copy.source_buffer,
                                   op->copy.source_offset, op->copy.length) &&
            loom_serve_group_range(op->copy.target_buffer,
                                   op->copy.target_offset, op->copy.length);
        break;
      default:
        break;
    }
    if (valid) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid group transfer range or payload");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid group node");
}

iree_status_t loom_serve_execution_group_append(
    loom_serve_execution_group_t* group, const loom_serve_group_node_t* input) {
  if (group->submitted || group->retired) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "group is closed");
  }
  if (group->count == group->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "group capacity exhausted");
  }
  IREE_RETURN_IF_ERROR(loom_serve_group_validate_node(input));
  loom_serve_group_node_t node = *input;
  node.command_buffer = NULL;
  node.executable = NULL;
  node.bindings = (iree_hal_buffer_binding_table_t){0};
  node.refs = (iree_hal_buffer_ref_list_t){0};
  iree_status_t status = iree_ok_status();
  if (input->kind == LOOM_SERVE_GROUP_EXECUTE && input->command_buffer) {
    iree_hal_buffer_binding_t* bindings = NULL;
    if (input->bindings.count) {
      IREE_RETURN_IF_ERROR(iree_allocator_clone(
          group->allocator,
          iree_make_const_byte_span(input->bindings.bindings,
                                    input->bindings.count * sizeof(*bindings)),
          (void**)&bindings));
    }
    node.bindings =
        (iree_hal_buffer_binding_table_t){input->bindings.count, bindings};
    node.command_buffer = input->command_buffer;
    iree_hal_command_buffer_retain(node.command_buffer);
    for (iree_host_size_t i = 0; i < node.bindings.count; ++i) {
      iree_hal_buffer_retain(bindings[i].buffer);
    }
  } else if (input->kind == LOOM_SERVE_GROUP_DISPATCH && input->executable) {
    iree_hal_buffer_ref_t* refs = NULL;
    if (input->refs.count) {
      IREE_RETURN_IF_ERROR(iree_allocator_clone(
          group->allocator,
          iree_make_const_byte_span(input->refs.values,
                                    input->refs.count * sizeof(*refs)),
          (void**)&refs));
    }
    node.refs = (iree_hal_buffer_ref_list_t){input->refs.count, refs};
    node.executable = input->executable;
    iree_hal_executable_retain(node.executable);
    for (iree_host_size_t i = 0; i < node.refs.count; ++i) {
      iree_hal_buffer_retain(refs[i].buffer);
    }
  } else if (input->kind == LOOM_SERVE_GROUP_TRANSFER &&
             loom_serve_group_transfer_buffer(&input->transfer)) {
    void* payload = NULL;
    if (input->transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD) {
      status = iree_allocator_clone(
          group->allocator,
          iree_make_const_byte_span(input->transfer.upload.source,
                                    input->transfer.upload.length),
          &payload);
      node.transfer.upload.source = payload;
    } else if (input->transfer.type ==
               IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) {
      status = iree_allocator_clone(
          group->allocator,
          iree_make_const_byte_span(
              (const uint8_t*)input->transfer.update.source_buffer +
                  input->transfer.update.source_offset,
              input->transfer.update.length),
          &payload);
      node.transfer.update.source_buffer = payload;
      node.transfer.update.source_offset = 0;
    } else if (input->transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL) {
      status = iree_allocator_clone(
          group->allocator,
          iree_make_const_byte_span(input->transfer.fill.pattern,
                                    input->transfer.fill.pattern_length),
          &payload);
      node.transfer.fill.pattern = payload;
    }
    IREE_RETURN_IF_ERROR(status);
    iree_hal_buffer_retain(loom_serve_group_transfer_buffer(&node.transfer));
    if (node.transfer.type == IREE_HAL_TRANSFER_OPERATION_TYPE_COPY) {
      iree_hal_buffer_retain(node.transfer.copy.target_buffer);
    }
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid group node");
  }
  group->nodes[group->count++] = node;
  return iree_ok_status();
}

iree_status_t loom_serve_execution_group_submit(
    loom_serve_execution_group_t* group,
    loom_serve_group_submission_t* out_submission) {
  *out_submission = (loom_serve_group_submission_t){0};
  if (group->submitted || group->retired || !group->count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "group must be nonempty and unsubmitted");
  }
  group->submitted = true;
  return loom_serve_execution_submit_group(group->execution, group->count,
                                           group->nodes, out_submission);
}
