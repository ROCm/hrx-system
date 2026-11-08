// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"

#include "iree/base/checked_math.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/parameter.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

iree_status_t loom_aie2p_stack_layout_append(loom_storage_space_t space,
                                             uint64_t byte_size,
                                             uint64_t byte_alignment,
                                             loom_aie2p_stack_layout_t* layout,
                                             uint64_t* out_depth) {
  uint64_t depth = 0;
  if (space == LOOM_STORAGE_SPACE_STACK) {
    if (!iree_checked_add_u64(layout->byte_length, byte_size, &depth) ||
        !iree_checked_align_u64(depth, byte_alignment, &depth)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P stack layout exceeds the byte domain");
    }
    layout->byte_length = depth;
    layout->minimum_alignment =
        iree_max(layout->minimum_alignment, byte_alignment);
  }
  if (layout->count == layout->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        layout->arena, layout->count, iree_max(layout->count + 1, 4u),
        sizeof(*layout->depths), &layout->capacity, (void**)&layout->depths));
  }
  layout->depths[layout->count++] = depth;
  *out_depth = depth;
  return iree_ok_status();
}

iree_status_t loom_aie2p_stack_layout_extend(
    const loom_low_storage_layout_t* source,
    loom_aie2p_stack_layout_t* layout) {
  IREE_ASSERT_LE(layout->count, source->record_count);
  iree_status_t status = iree_ok_status();
  while (layout->count < source->record_count && iree_status_is_ok(status)) {
    const loom_low_storage_layout_reservation_t* reservation =
        &source->records[layout->count].reservation;
    uint64_t depth = 0;
    status = loom_aie2p_stack_layout_append(
        reservation->space, reservation->byte_size, reservation->byte_alignment,
        layout, &depth);
  }
  return status;
}

iree_status_t loom_aie2p_stack_frame_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_storage_layout_t* source, loom_aie2p_stack_layout_t* layout,
    uint64_t callee_byte_length, bool is_helper,
    iree_diagnostic_emitter_t emitter, bool* out_accepted,
    loom_aie2p_stack_frame_t* out_frame) {
  *out_accepted = false;
  *out_frame = (loom_aie2p_stack_frame_t){.storage_layout = *source};
  if (source->space_sizes.stack_bytes == 0 && layout->byte_length == 0 &&
      callee_byte_length == 0) {
    *out_accepted = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_stack_layout_extend(source, layout));
  const uint64_t alignment = iree_max(UINT64_C(64), layout->minimum_alignment);
  // PADDXM's signed 19-bit displacement is a multiple of 64. Its positive
  // range exceeds every supported AIE2P tile's complete local SRAM capacity.
  const uint64_t maximum_byte_length = (UINT64_C(1) << 18) - 64;
  uint64_t byte_length = 0;
  if (!iree_checked_align_u64(layout->byte_length, alignment, &byte_length) ||
      byte_length > maximum_byte_length) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(
            loom_low_diagnostic_function_name(module, function_op)),
        loom_param_u64(iree_max(byte_length, layout->byte_length)),
        loom_param_u64(maximum_byte_length),
    };
    const loom_diagnostic_emission_t emission = {
        .op = function_op,
        .error = LOOM_ERR_XDNA_064,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  out_frame->byte_length = byte_length;
  if (!is_helper) {
    out_frame->storage = (loom_aie2p_leaf_storage_requirement_t){
        .byte_length = byte_length + callee_byte_length,
        .minimum_alignment = alignment,
    };
  }
  if (byte_length != 0) {
    loom_low_storage_layout_record_t* records = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(layout->arena, source->record_count,
                                  sizeof(*records), (void**)&records));
    for (iree_host_size_t i = 0; i < source->record_count; ++i) {
      records[i] = source->records[i];
      if (records[i].reservation.space == LOOM_STORAGE_SPACE_STACK) {
        records[i].reservation.byte_offset = byte_length - layout->depths[i];
      }
    }
    out_frame->storage_layout.records = records;
    out_frame->storage_layout.space_sizes.stack_bytes = byte_length;
    out_frame->storage_layout.minimum_alignments[LOOM_STORAGE_SPACE_STACK] =
        alignment;
  }
  *out_accepted = true;
  return iree_ok_status();
}
