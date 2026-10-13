// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_call.h"

IREE_API_EXPORT iree_status_t iree_vm_native_call_schema_validate(
    const iree_vm_native_call_schema_t* schema, uint32_t ref_type_count) {
  if (!schema || !schema->layout_count || schema->layout_count > 65536 ||
      !schema->layouts || (schema->field_count && !schema->fields) ||
      schema->root_ordinal >= schema->layout_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native layout table is invalid");
  }
  uint32_t field_base = 0;
  for (uint32_t i = 0; i < schema->layout_count; ++i) {
    const iree_vm_native_call_layout_t* layout = &schema->layouts[i];
    if (layout->field_base != field_base ||
        layout->field_count > schema->field_count - field_base) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native layout field slices are invalid");
    }
    uint32_t word_count = 0;
    uint32_t depth = 0;
    for (uint32_t j = 0; j < layout->field_count; ++j) {
      const iree_vm_native_call_field_t field = schema->fields[field_base + j];
      const uint16_t kind = field.kind & ~IREE_VM_NATIVE_CALL_FIELD_RESULT;
      if (kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
        if (field.kind != IREE_VM_NATIVE_CALL_FIELD_RANGE ||
            field.type_ordinal >= i) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native range must name an earlier layout");
        }
        depth = iree_max(depth, 1u + schema->layouts[field.type_ordinal].depth);
        word_count += 2;
      } else {
        if (kind == IREE_VM_NATIVE_CALL_FIELD_REF) {
          if (field.type_ordinal >= ref_type_count) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "native ref type is out of range");
          }
        } else if (kind > UINT8_MAX ||
                   !iree_vm_scalar_type_is_valid((iree_vm_scalar_type_t)kind) ||
                   field.type_ordinal != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native scalar type is invalid");
        }
        ++word_count;
      }
    }
    if (layout->word_count != word_count || layout->depth != depth) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native layout stride or depth is invalid");
    }
    field_base += layout->field_count;
  }
  if (field_base != schema->field_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native layouts do not cover their field table");
  }
  return iree_ok_status();
}

typedef struct iree_vm_native_call_verifier_t {
  // External declaration being checked.
  const iree_vm_native_callsite_t* callsite;
  // Next unclaimed word in the canonical depth-first tail.
  uint32_t cursor;
  // Checked sum of native scratch contributions.
  iree_host_size_t scratch_size;
  // One byte per destination scalar, ref, or local byte.
  uint8_t* destinations;
} iree_vm_native_call_verifier_t;

typedef struct iree_vm_native_call_record_frame_t {
  // Fixed record layout for this range.
  const iree_vm_native_call_layout_t* layout;
  // Word offset of the current fixed record.
  uint32_t offset;
  // Number of records still to visit, including the current record.
  uint32_t remaining;
  // Index of the next field within the current record.
  uint32_t field_index;
  // Word offset of the next field within the current fixed record.
  uint32_t field_offset;
} iree_vm_native_call_record_frame_t;

static iree_status_t iree_vm_native_call_verify_selector(
    iree_vm_native_call_verifier_t* verifier,
    iree_vm_native_call_field_kind_t kind, uint32_t selector) {
  const iree_vm_native_call_storage_t storage = verifier->callsite->storage;
  iree_host_size_t destination = selector;
  uint32_t width = 1;
  if ((kind & ~IREE_VM_NATIVE_CALL_FIELD_RESULT) ==
      IREE_VM_NATIVE_CALL_FIELD_REF) {
    if (selector >= storage.ref_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native call ref selector is out of range");
    }
    destination += storage.value_count;
  } else if (selector & IREE_VM_NATIVE_CALL_LOCAL_VALUE) {
    const uint32_t offset = selector & ~IREE_VM_NATIVE_CALL_LOCAL_VALUE;
    if (offset > storage.local_byte_length ||
        storage.local_byte_length - offset < sizeof(uint64_t)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native call local scalar is out of range");
    }
    destination =
        (iree_host_size_t)storage.value_count + storage.ref_count + offset;
    width = sizeof(uint64_t);
  } else if (selector >= storage.value_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "native call value selector is out of range");
  }
  if (kind & IREE_VM_NATIVE_CALL_FIELD_RESULT) {
    for (uint32_t i = 0; i < width; ++i) {
      if (verifier->destinations[destination + i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "native call result destinations overlap");
      }
      verifier->destinations[destination + i] = 1;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_vm_native_call_verify_range(
    iree_vm_native_call_verifier_t* verifier,
    const iree_vm_native_call_layout_t* layout, uint32_t scratch_per_record,
    uint32_t offset, uint32_t count) {
  const uint64_t word_count = (uint64_t)count * layout->word_count;
  if (offset != verifier->cursor ||
      word_count > verifier->callsite->word_count - verifier->cursor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native call range has noncanonical extent");
  }
  verifier->cursor += (uint32_t)word_count;
  const uint64_t scratch_size = (uint64_t)count * scratch_per_record;
  if (scratch_size > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add(verifier->scratch_size,
                                  (iree_host_size_t)scratch_size,
                                  &verifier->scratch_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native call scratch size overflows host size");
  }
  return iree_ok_status();
}

// The stack follows only materialized records, not paths through the
// declaration DAG. Empty ranges are checked without expanding their element
// layouts.
static iree_status_t iree_vm_native_call_verify_records(
    iree_vm_native_call_verifier_t* verifier,
    const iree_vm_native_export_t* target,
    iree_vm_native_call_record_frame_t* stack) {
  const iree_vm_native_call_schema_t* schema = target->schema;
  const iree_vm_native_call_layout_t* root =
      &schema->layouts[schema->root_ordinal];
  IREE_RETURN_IF_ERROR(iree_vm_native_call_verify_range(
      verifier, root,
      target->scratch_sizes ? target->scratch_sizes[schema->root_ordinal] : 0,
      0, 1));
  stack[0] =
      (iree_vm_native_call_record_frame_t){.layout = root, .remaining = 1};
  uint32_t depth = 1;
  iree_status_t status = iree_ok_status();
  while (depth && iree_status_is_ok(status)) {
    iree_vm_native_call_record_frame_t* frame = &stack[depth - 1];
    if (frame->field_index == frame->layout->field_count) {
      if (--frame->remaining == 0 || frame->layout->field_count == 0) {
        --depth;
      } else {
        frame->offset += frame->layout->word_count;
        frame->field_index = 0;
        frame->field_offset = 0;
      }
      continue;
    }
    const iree_vm_native_call_field_t field =
        schema->fields[frame->layout->field_base + frame->field_index++];
    const uint32_t* words =
        verifier->callsite->record + frame->offset + frame->field_offset;
    if (field.kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      frame->field_offset += 2;
      const iree_vm_native_call_layout_t* child =
          &schema->layouts[field.type_ordinal];
      status = iree_vm_native_call_verify_range(
          verifier, child,
          target->scratch_sizes ? target->scratch_sizes[field.type_ordinal] : 0,
          words[1], words[0]);
      if (iree_status_is_ok(status) && words[0] && child->field_count) {
        stack[depth++] = (iree_vm_native_call_record_frame_t){
            .layout = child, .offset = words[1], .remaining = words[0]};
      }
    } else {
      ++frame->field_offset;
      status =
          iree_vm_native_call_verify_selector(verifier, field.kind, *words);
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_native_call_verify(
    const iree_vm_native_export_t* target,
    const iree_vm_native_callsite_t* callsite, iree_allocator_t host_allocator,
    iree_host_size_t* out_scratch_size) {
  if (!target || !callsite || !out_scratch_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "native target, callsite and scratch output are required");
  }
  if (target->version != callsite->version) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "native call schema version mismatch");
  }
  if ((callsite->word_count && !callsite->record) ||
      !iree_host_ptr_has_alignment(callsite->record, iree_alignof(uint32_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native call record is missing or misaligned");
  }
  iree_host_size_t destination_count = callsite->storage.value_count;
  if (!iree_host_size_checked_add(
          destination_count, callsite->storage.ref_count, &destination_count) ||
      !iree_host_size_checked_add(destination_count,
                                  callsite->storage.local_byte_length,
                                  &destination_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native call storage extent overflows host size");
  }
  const uint32_t stack_count =
      1u + target->schema->layouts[target->schema->root_ordinal].depth;
  iree_host_size_t storage_size = 0, stack_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      destination_count, &storage_size,
      IREE_STRUCT_FIELD(stack_count, iree_vm_native_call_record_frame_t,
                        &stack_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, (void**)&storage));
  iree_vm_native_call_verifier_t verifier = {.callsite = callsite,
                                             .destinations = storage};
  iree_status_t status = iree_vm_native_call_verify_records(
      &verifier, target,
      (iree_vm_native_call_record_frame_t*)(storage + stack_offset));
  if (iree_status_is_ok(status) && verifier.cursor != callsite->word_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native call record has trailing words");
  }
  if (iree_status_is_ok(status) && target->verify) {
    status = target->verify(callsite->record);
  }
  iree_allocator_free(host_allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_scratch_size = verifier.scratch_size;
  }
  return status;
}
