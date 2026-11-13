// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_layout.h"

#include <string.h>

#include "loom/analysis/storage_layout.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/type_builders.h"
#include "loom/rewrite/rewriter.h"

iree_host_size_t loom_low_storage_space_set_names(
    loom_low_storage_space_set_t set, iree_host_size_t capacity,
    iree_string_view_t* out_names) {
  static const loom_storage_space_t kStorageSpaceOrder[] = {
      LOOM_STORAGE_SPACE_STACK,
      LOOM_STORAGE_SPACE_SCRATCH,
      LOOM_STORAGE_SPACE_PRIVATE,
      LOOM_STORAGE_SPACE_WORKGROUP,
  };
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    const loom_storage_space_t storage_space = kStorageSpaceOrder[i];
    if (!loom_low_storage_space_set_contains(set, storage_space)) {
      continue;
    }
    if (count < capacity) {
      out_names[count] = loom_low_storage_type_space_name(storage_space);
    }
    ++count;
  }
  return count;
}

iree_status_t loom_low_storage_layout_hoist_reservations(
    loom_module_t* module, loom_region_t* body, iree_arena_allocator_t* arena) {
  loom_op_t* insertion_op = loom_region_entry_block(body)->first_op;
  loom_rewriter_t rewriter = {0};
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < body->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_op_t* op = body->blocks[block_index]->first_op;
    while (op != NULL && iree_status_is_ok(status)) {
      loom_op_t* next_op = op->next_op;
      if (loom_low_storage_reserve_isa(op)) {
        if (op == insertion_op) {
          insertion_op = next_op;
        } else {
          if (rewriter.module == NULL) {
            loom_rewriter_initialize(&rewriter, module, arena);
          }
          status = loom_rewriter_move_before(&rewriter, op, insertion_op);
        }
      } else if (op == insertion_op &&
                 loom_low_function_entry_preamble_op(op)) {
        insertion_op = next_op;
      }
      op = next_op;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static uint64_t* loom_low_storage_layout_space_size(
    loom_low_storage_layout_space_sizes_t* sizes, loom_storage_space_t space) {
  switch (space) {
    case LOOM_STORAGE_SPACE_STACK:
      return &sizes->stack_bytes;
    case LOOM_STORAGE_SPACE_SCRATCH:
      return &sizes->scratch_bytes;
    case LOOM_STORAGE_SPACE_PRIVATE:
      return &sizes->private_bytes;
    case LOOM_STORAGE_SPACE_WORKGROUP:
      return &sizes->workgroup_bytes;
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified storage reservation must have a valid storage space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

void loom_low_storage_layout_pack_reservation(
    loom_storage_space_t space, uint64_t byte_size, uint64_t byte_alignment,
    loom_low_storage_layout_space_sizes_t* sizes,
    loom_low_storage_layout_reservation_t* out_reservation) {
  uint64_t* space_size = loom_low_storage_layout_space_size(sizes, space);
  const uint64_t byte_offset = iree_align_uint64(*space_size, byte_alignment);
  *space_size = byte_offset + byte_size;
  *out_reservation = (loom_low_storage_layout_reservation_t){
      .space = space,
      .byte_offset = byte_offset,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
  };
}

void loom_low_storage_layout_builder_initialize(
    loom_low_storage_layout_builder_t* out_builder) {
  *out_builder = (loom_low_storage_layout_builder_t){0};
}

loom_low_storage_layout_requirement_t loom_low_storage_layout_requirement(
    const loom_low_storage_layout_t* layout, loom_storage_space_t space) {
  loom_low_storage_layout_space_sizes_t sizes = layout->space_sizes;
  const uint64_t byte_length =
      *loom_low_storage_layout_space_size(&sizes, space);
  return (loom_low_storage_layout_requirement_t){
      .byte_length = byte_length,
      .minimum_alignment = byte_length ? layout->minimum_alignments[space] : 0,
  };
}

iree_status_t loom_low_storage_layout_builder_append(
    const loom_module_t* module, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_builder_t* builder) {
  if (loom_low_storage_view_isa(op)) {
    if (builder->view_count == builder->view_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, builder->view_count, iree_max(builder->view_count + 1, 4u),
          sizeof(*builder->views), &builder->view_capacity,
          (void**)&builder->views));
    }
    builder->views[builder->view_count++] = (loom_low_storage_layout_view_t){
        .value_id = loom_low_storage_view_result(op),
        .source_value_id = loom_low_storage_view_source(op),
        .byte_offset = (uint64_t)loom_low_storage_view_offset(op),
        .byte_length = (uint64_t)loom_low_storage_view_byte_length(op),
    };
    return iree_ok_status();
  }

  loom_low_storage_layout_reservation_t reservation;
  const loom_value_id_t storage_value_id = loom_low_storage_reserve_storage(op);
  const loom_storage_space_t storage_space =
      loom_type_storage_space(loom_module_value_type(module, storage_value_id));
  loom_low_storage_layout_pack_reservation(
      storage_space, (uint64_t)loom_low_storage_reserve_byte_length(op),
      (uint64_t)loom_low_storage_reserve_byte_alignment(op),
      &builder->space_sizes, &reservation);
  const iree_host_size_t minimum_capacity = builder->record_count + 1;
  if (minimum_capacity > builder->record_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, builder->record_count, iree_max(minimum_capacity, 4u),
        sizeof(*builder->records), &builder->record_capacity,
        (void**)&builder->records));
  }
  builder->records[builder->record_count++] =
      (loom_low_storage_layout_record_t){
          .storage_value_id = storage_value_id,
          .reservation = reservation,
      };
  uint64_t* alignment = &builder->minimum_alignments[reservation.space];
  *alignment = iree_max(*alignment, reservation.byte_alignment);
  return iree_ok_status();
}

static iree_host_size_t loom_low_storage_layout_bucket(loom_value_id_t value_id,
                                                       iree_host_size_t mask) {
  // Mix both halves of the product so sparse SSA domains do not cluster when
  // their storage values share low bits.
  const uint64_t product = (uint64_t)value_id * UINT64_C(11400714819323198485);
  return (iree_host_size_t)(product ^ (product >> 32)) & mask;
}

static uint32_t loom_low_storage_layout_handle_ordinal(
    const loom_low_storage_layout_index_t* index, loom_value_id_t value_id) {
  const iree_host_size_t mask = index->bucket_count - 1;
  iree_host_size_t bucket = loom_low_storage_layout_bucket(value_id, mask);
  for (;;) {
    const uint32_t ordinal = index->buckets[bucket];
    IREE_ASSERT_NE(ordinal, UINT32_MAX,
                   "storage reference must belong to the retained layout");
    if (index->handles[ordinal].value_id == value_id) {
      return ordinal;
    }
    bucket = (bucket + 1) & mask;
  }
}

iree_status_t loom_low_storage_layout_builder_finish(
    const loom_low_storage_layout_builder_t* builder,
    iree_arena_allocator_t* arena, loom_low_storage_layout_t* out_layout) {
  *out_layout = (loom_low_storage_layout_t){
      .space_sizes = builder->space_sizes,
      .records = builder->records,
      .record_count = builder->record_count,
  };
  memcpy(out_layout->minimum_alignments, builder->minimum_alignments,
         sizeof(out_layout->minimum_alignments));
  const iree_host_size_t handle_count =
      builder->record_count + builder->view_count;
  if (handle_count == 0) {
    return iree_ok_status();
  }

  loom_low_storage_layout_handle_t* handles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, handle_count, sizeof(*handles), (void**)&handles));
  iree_host_size_t bucket_count = 1;
  while (bucket_count < handle_count * 2) {
    bucket_count *= 2;
  }
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  const loom_low_storage_layout_index_t index = {
      .handles = handles,
      .buckets = buckets,
      .bucket_count = bucket_count,
  };
  for (iree_host_size_t i = 0; i < handle_count; ++i) {
    if (i < builder->record_count) {
      handles[i] = (loom_low_storage_layout_handle_t){
          .value_id = builder->records[i].storage_value_id,
          .reservation_ordinal = (uint32_t)i,
          .byte_length = builder->records[i].reservation.byte_size,
      };
    } else {
      const loom_low_storage_layout_view_t* view =
          &builder->views[i - builder->record_count];
      handles[i] = (loom_low_storage_layout_handle_t){
          .value_id = view->value_id,
          .reservation_ordinal = UINT32_MAX,
          .byte_offset = view->byte_offset,
          .byte_length = view->byte_length,
      };
    }
    iree_host_size_t bucket =
        loom_low_storage_layout_bucket(handles[i].value_id, bucket_count - 1);
    while (buckets[bucket] != UINT32_MAX) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    buckets[bucket] = (uint32_t)i;
  }

  // Resolve each chain to its nearest resolved ancestor, then flatten every
  // unresolved member on the path. Each view is visited at most twice, even
  // when block declaration order puts a child before its source. No C recursion
  // or per-query source traversal is required.
  for (iree_host_size_t i = builder->record_count; i < handle_count; ++i) {
    uint32_t ordinal = (uint32_t)i;
    uint64_t byte_offset = 0;
    while (handles[ordinal].reservation_ordinal == UINT32_MAX) {
      byte_offset += handles[ordinal].byte_offset;
      ordinal = loom_low_storage_layout_handle_ordinal(
          &index,
          builder->views[ordinal - builder->record_count].source_value_id);
    }
    byte_offset += handles[ordinal].byte_offset;
    const uint32_t reservation_ordinal = handles[ordinal].reservation_ordinal;
    ordinal = (uint32_t)i;
    while (handles[ordinal].reservation_ordinal == UINT32_MAX) {
      const uint32_t source_ordinal = loom_low_storage_layout_handle_ordinal(
          &index,
          builder->views[ordinal - builder->record_count].source_value_id);
      const uint64_t relative_offset = handles[ordinal].byte_offset;
      handles[ordinal].byte_offset = byte_offset;
      handles[ordinal].reservation_ordinal = reservation_ordinal;
      byte_offset -= relative_offset;
      ordinal = source_ordinal;
    }
  }
  out_layout->index = index;
  return iree_ok_status();
}

bool loom_low_storage_layout_accumulate_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes) {
  const loom_value_id_t storage_value_id =
      loom_low_storage_reserve_storage(reserve_op);
  uint64_t* space_size = loom_low_storage_layout_space_size(
      sizes, loom_type_storage_space(
                 loom_module_value_type(module, storage_value_id)));
  return loom_storage_layout_append(
      (uint64_t)loom_low_storage_reserve_byte_length(reserve_op),
      (uint64_t)loom_low_storage_reserve_byte_alignment(reserve_op), space_size,
      NULL);
}

void loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_index_t* index,
    const loom_low_storage_layout_record_t* records,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference) {
  const loom_low_storage_layout_handle_t* handle =
      &index->handles[loom_low_storage_layout_handle_ordinal(index,
                                                             storage_value_id)];
  *out_reference = (loom_low_storage_layout_reference_t){
      .reservation = records[handle->reservation_ordinal].reservation,
      .byte_offset = handle->byte_offset,
      .byte_length = handle->byte_length,
  };
}
