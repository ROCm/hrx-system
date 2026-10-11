// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_call.h"

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

static iree_status_t iree_vm_native_call_verify_selector(
    iree_vm_native_call_verifier_t* verifier,
    iree_vm_native_call_field_kind_t kind, uint32_t selector) {
  const iree_vm_native_call_storage_t storage = verifier->callsite->storage;
  iree_host_size_t destination = selector;
  uint32_t width = 1;
  if (kind == IREE_VM_NATIVE_CALL_FIELD_REF ||
      kind == IREE_VM_NATIVE_CALL_FIELD_RESULT_REF) {
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
  if (kind == IREE_VM_NATIVE_CALL_FIELD_RESULT_VALUE ||
      kind == IREE_VM_NATIVE_CALL_FIELD_RESULT_REF) {
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

static iree_status_t iree_vm_native_call_verify_records(
    iree_vm_native_call_verifier_t* verifier,
    const iree_vm_native_call_schema_t* schema, uint32_t offset,
    uint32_t count) {
  const uint64_t word_count = (uint64_t)count * schema->word_count;
  if (offset != verifier->cursor ||
      word_count > verifier->callsite->word_count - verifier->cursor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native call range has noncanonical extent");
  }
  verifier->cursor += (uint32_t)word_count;
  const uint64_t scratch_size = (uint64_t)count * schema->scratch_size;
  if (scratch_size > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add(verifier->scratch_size,
                                  (iree_host_size_t)scratch_size,
                                  &verifier->scratch_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "native call scratch size overflows host size");
  }
  if (schema->field_count == 0) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t* record =
        verifier->callsite->record + offset + i * schema->word_count;
    for (uint32_t j = 0; j < schema->field_count; ++j) {
      const iree_vm_native_call_field_t* field = &schema->fields[j];
      if (field->kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
        IREE_RETURN_IF_ERROR(iree_vm_native_call_verify_records(
            verifier, field->element, record[field->offset + 1],
            record[field->offset]));
      } else {
        IREE_RETURN_IF_ERROR(iree_vm_native_call_verify_selector(
            verifier, field->kind, record[field->offset]));
      }
    }
  }
  return iree_ok_status();
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
  iree_vm_native_call_verifier_t verifier = {.callsite = callsite};
  if (destination_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, destination_count, (void**)&verifier.destinations));
    memset(verifier.destinations, 0, destination_count);
  }
  iree_status_t status =
      iree_vm_native_call_verify_records(&verifier, target->schema, 0, 1);
  if (iree_status_is_ok(status) && verifier.cursor != callsite->word_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native call record has trailing words");
  }
  if (iree_status_is_ok(status) && target->verify) {
    status = target->verify(callsite->record);
  }
  iree_allocator_free(host_allocator, verifier.destinations);
  if (iree_status_is_ok(status)) {
    *out_scratch_size = verifier.scratch_size;
  }
  return status;
}
