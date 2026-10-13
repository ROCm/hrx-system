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
    const iree_vm_native_call_layout_t* layout, uint32_t offset,
    uint32_t count) {
  const uint64_t word_count = (uint64_t)count * layout->word_count;
  if (offset != verifier->cursor ||
      word_count > verifier->callsite->word_count - verifier->cursor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native call range has noncanonical extent");
  }
  verifier->cursor += (uint32_t)word_count;
  return iree_ok_status();
}

// The stack follows only materialized records, not paths through the
// declaration DAG. Empty ranges are checked without expanding their element
// layouts.
static iree_status_t iree_vm_native_call_verify_records(
    iree_vm_native_call_verifier_t* verifier,
    iree_vm_native_call_record_frame_t* stack) {
  const iree_vm_native_call_schema_t* schema = &verifier->callsite->schema;
  const iree_vm_native_call_layout_t* root =
      &schema->layouts[schema->root_ordinal];
  IREE_RETURN_IF_ERROR(iree_vm_native_call_verify_range(verifier, root, 0, 1));
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
      status =
          iree_vm_native_call_verify_range(verifier, child, words[1], words[0]);
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

IREE_API_EXPORT iree_status_t
iree_vm_native_call_verify(const iree_vm_native_callsite_t* callsite,
                           iree_allocator_t host_allocator) {
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
      1u + callsite->schema.layouts[callsite->schema.root_ordinal].depth;
  iree_host_size_t storage_size = 0, stack_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      destination_count, &storage_size,
      IREE_STRUCT_FIELD_ALIGNED(
          stack_count, iree_vm_native_call_record_frame_t,
          iree_alignof(iree_vm_native_call_record_frame_t), &stack_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, (void**)&storage));
  iree_vm_native_call_verifier_t verifier = {.callsite = callsite,
                                             .destinations = storage};
  iree_status_t status = iree_vm_native_call_verify_records(
      &verifier, (iree_vm_native_call_record_frame_t*)(storage + stack_offset));
  if (iree_status_is_ok(status) && verifier.cursor != callsite->word_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native call record has trailing words");
  }
  iree_allocator_free(host_allocator, storage);
  return status;
}

//===----------------------------------------------------------------------===//
// Exact structural linking
//===----------------------------------------------------------------------===//

typedef struct iree_vm_native_call_equivalence_t {
  // Parent in the union-by-rank forest, initially this node's own index.
  uint32_t parent;
  // Upper bound on the forest height.
  uint8_t rank;
} iree_vm_native_call_equivalence_t;

typedef struct iree_vm_native_call_match_frame_t {
  // Layout ordinal in the caller table.
  uint16_t caller_ordinal;
  // Layout ordinal in the target table.
  uint16_t target_ordinal;
  // Next field whose child layout needs comparison.
  uint32_t field_index;
} iree_vm_native_call_match_frame_t;

static uint32_t iree_vm_native_call_equivalence_find(
    iree_vm_native_call_equivalence_t* nodes, uint32_t index) {
  while (nodes[index].parent != index) {
    nodes[index].parent = nodes[nodes[index].parent].parent;
    index = nodes[index].parent;
  }
  return index;
}

// Comparing a pair unifies its equivalence classes after checking the local
// fields, then visits its child pairs. A repeated pair (including transitive
// equivalence through duplicated layouts) requires no second descent. There
// are at most caller_count + target_count - 1 successful unions, regardless of
// how many paths the DAG has. The explicit traversal stack follows decreasing
// layout depths; C stack use is independent of external nesting.
static bool iree_vm_native_call_layouts_match(
    const iree_vm_native_call_schema_t* caller,
    const iree_vm_ref_type_t* caller_ref_types,
    const iree_vm_native_call_schema_t* target,
    const iree_vm_ref_type_t* target_ref_types,
    iree_vm_native_call_equivalence_t* nodes,
    iree_vm_native_call_match_frame_t* stack) {
  const uint32_t node_count = caller->layout_count + target->layout_count;
  for (uint32_t i = 0; i < node_count; ++i) {
    nodes[i].parent = i;
  }
  stack[0] = (iree_vm_native_call_match_frame_t){
      .caller_ordinal = caller->root_ordinal,
      .target_ordinal = target->root_ordinal,
  };
  uint32_t depth = 1;
  while (depth) {
    iree_vm_native_call_match_frame_t* frame = &stack[depth - 1];
    const iree_vm_native_call_layout_t* left =
        &caller->layouts[frame->caller_ordinal];
    const iree_vm_native_call_layout_t* right =
        &target->layouts[frame->target_ordinal];
    if (frame->field_index == 0) {
      uint32_t a =
          iree_vm_native_call_equivalence_find(nodes, frame->caller_ordinal);
      uint32_t b = iree_vm_native_call_equivalence_find(
          nodes, caller->layout_count + frame->target_ordinal);
      if (a == b) {
        --depth;
        continue;
      }
      if (left->field_count != right->field_count ||
          left->depth != right->depth) {
        return false;
      }
      for (uint32_t i = 0; i < left->field_count; ++i) {
        const iree_vm_native_call_field_t x =
            caller->fields[left->field_base + i];
        const iree_vm_native_call_field_t y =
            target->fields[right->field_base + i];
        if (x.kind != y.kind || ((x.kind & ~IREE_VM_NATIVE_CALL_FIELD_RESULT) ==
                                     IREE_VM_NATIVE_CALL_FIELD_REF &&
                                 caller_ref_types[x.type_ordinal] !=
                                     target_ref_types[y.type_ordinal])) {
          return false;
        }
      }
      if (nodes[a].rank < nodes[b].rank) {
        nodes[a].parent = b;
      } else {
        nodes[b].parent = a;
        if (nodes[a].rank == nodes[b].rank) {
          ++nodes[a].rank;
        }
      }
    }
    bool descended = false;
    while (frame->field_index < left->field_count) {
      const uint32_t index = frame->field_index++;
      const iree_vm_native_call_field_t x =
          caller->fields[left->field_base + index];
      if (x.kind == IREE_VM_NATIVE_CALL_FIELD_RANGE) {
        stack[depth++] = (iree_vm_native_call_match_frame_t){
            .caller_ordinal = x.type_ordinal,
            .target_ordinal =
                target->fields[right->field_base + index].type_ordinal,
        };
        descended = true;
        break;
      }
    }
    if (!descended) {
      --depth;
    }
  }
  return true;
}

// Consumes only an admitted record and its matched target layout. The only
// possible failure is overflow of the native implementation's scratch extent.
static iree_status_t iree_vm_native_call_measure_scratch(
    const iree_vm_native_export_t* target, const uint32_t* record,
    iree_vm_native_call_record_frame_t* stack,
    iree_host_size_t* out_scratch_size) {
  const iree_vm_native_call_schema_t* schema = target->schema;
  iree_host_size_t scratch_size = target->scratch_sizes[schema->root_ordinal];
  stack[0] = (iree_vm_native_call_record_frame_t){
      .layout = &schema->layouts[schema->root_ordinal], .remaining = 1};
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
    if (field.kind != IREE_VM_NATIVE_CALL_FIELD_RANGE) {
      ++frame->field_offset;
      continue;
    }
    const uint32_t* words = record + frame->offset + frame->field_offset;
    frame->field_offset += 2;
    const uint64_t contribution =
        (uint64_t)words[0] * target->scratch_sizes[field.type_ordinal];
    if (contribution > IREE_HOST_SIZE_MAX ||
        !iree_host_size_checked_add(
            scratch_size, (iree_host_size_t)contribution, &scratch_size)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "native call scratch size overflows host size");
    } else if (words[0] && schema->layouts[field.type_ordinal].field_count) {
      stack[depth++] = (iree_vm_native_call_record_frame_t){
          .layout = &schema->layouts[field.type_ordinal],
          .offset = words[1],
          .remaining = words[0],
      };
    }
  }
  if (iree_status_is_ok(status)) {
    *out_scratch_size = scratch_size;
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_vm_native_call_link(
    const iree_vm_native_export_t* target,
    const iree_vm_ref_type_t* target_ref_types,
    const iree_vm_native_callsite_t* callsite,
    const iree_vm_ref_type_t* caller_ref_types, iree_allocator_t host_allocator,
    iree_host_size_t* out_scratch_size) {
  if (target->version != callsite->version) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "native call schema version mismatch");
  }
  const uint32_t stack_count =
      1u + callsite->schema.layouts[callsite->schema.root_ordinal].depth;
  iree_host_size_t storage_size = 0, nodes_offset = 0, match_offset = 0,
                   records_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(
          callsite->schema.layout_count + target->schema->layout_count,
          iree_vm_native_call_equivalence_t, &nodes_offset),
      IREE_STRUCT_FIELD(stack_count, iree_vm_native_call_match_frame_t,
                        &match_offset),
      IREE_STRUCT_FIELD(target->scratch_sizes ? stack_count : 0,
                        iree_vm_native_call_record_frame_t, &records_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, storage_size, (void**)&storage));
  iree_status_t status = iree_ok_status();
  if (!iree_vm_native_call_layouts_match(
          &callsite->schema, caller_ref_types, target->schema, target_ref_types,
          (iree_vm_native_call_equivalence_t*)(storage + nodes_offset),
          (iree_vm_native_call_match_frame_t*)(storage + match_offset))) {
    status = iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "native call layout mismatch");
  }
  iree_host_size_t scratch_size = 0;
  if (iree_status_is_ok(status) && target->scratch_sizes) {
    status = iree_vm_native_call_measure_scratch(
        target, callsite->record,
        (iree_vm_native_call_record_frame_t*)(storage + records_offset),
        &scratch_size);
  }
  if (iree_status_is_ok(status) && target->verify) {
    status = target->verify(callsite->record);
  }
  iree_allocator_free(host_allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_scratch_size = scratch_size;
  }
  return status;
}
