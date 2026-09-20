// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_frame.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/move_sequence.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/registers.h"

#define LOOM_X86_SYSV_FRAME_SLOT_NONE UINT32_MAX
#define LOOM_X86_SYSV_GPR_COUNT 16
#define LOOM_X86_SYSV_VECTOR_REGISTER_COUNT 32
#define LOOM_X86_SYSV_MASK_REGISTER_COUNT 8

loom_low_allocation_reserved_range_t
loom_x86_sysv_frame_stack_pointer_reservation(void) {
  return (loom_low_allocation_reserved_range_t){
      .register_class = IREE_SV("x86.gpr64"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = LOOM_X86_SYSV_GPR_RSP,
      .location_count = 1,
  };
}

typedef struct loom_x86_sysv_call_build_t {
  loom_x86_sysv_abi_layout_t abi_layout;
  uint64_t vector_widths;
  uint16_t gpr_mask;
  uint8_t mask_register_mask;
} loom_x86_sysv_call_build_t;

typedef struct loom_x86_sysv_cycle_slot_t {
  uint16_t descriptor_reg_class_id;
  uint32_t slot_index;
} loom_x86_sysv_cycle_slot_t;

typedef enum loom_x86_sysv_preservation_direction_e {
  LOOM_X86_SYSV_PRESERVATION_SAVE = 0,
  LOOM_X86_SYSV_PRESERVATION_RESTORE = 1,
} loom_x86_sysv_preservation_direction_t;

typedef struct loom_x86_sysv_frame_build_t {
  const loom_low_emission_frame_t* frame;
  iree_arena_allocator_t* arena;
  iree_arena_allocator_t* scratch_arena;
  loom_x86_sysv_frame_plan_t* plan;
  loom_x86_sysv_call_build_t* calls;
  loom_x86_sysv_call_plan_t* call_plans;
  loom_x86_sysv_return_plan_t* return_plans;
  loom_x86_sysv_frame_slot_t* slots;
  uint32_t slot_capacity;
  uint32_t caller_gpr_slots[LOOM_X86_SYSV_GPR_COUNT];
  uint32_t caller_vector_slots[LOOM_X86_SYSV_VECTOR_REGISTER_COUNT];
  uint32_t caller_mask_slots[LOOM_X86_SYSV_MASK_REGISTER_COUNT];
  uint32_t callee_gpr_slots[LOOM_X86_SYSV_GPR_COUNT];
  uint64_t caller_vector_widths;
  uint16_t caller_gpr_mask;
  uint8_t caller_mask_register_mask;
  loom_x86_sysv_cycle_slot_t* cycle_slots;
  uint16_t cycle_slot_count;
  loom_low_move_sequence_scratch_t move_scratch;
  loom_low_move_t* moves;
  uint32_t move_capacity;
  uint32_t move_count;
  uint16_t gpr64_class_id;
  uint16_t callee_gpr_mask;
} loom_x86_sysv_frame_build_t;

static uint32_t loom_x86_sysv_abi_register_argument_count(
    const loom_x86_sysv_abi_layout_t* layout) {
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < layout->argument_count; ++i) {
    count +=
        loom_x86_sysv_abi_location_is_register(layout->argument_locations[i]);
  }
  return count;
}

static iree_status_t loom_x86_sysv_frame_call_layouts(
    loom_x86_sysv_frame_build_t* build, uint32_t* out_max_outgoing_bytes) {
  *out_max_outgoing_bytes = 0;
  const loom_low_allocation_table_t* allocation = &build->frame->allocation;
  if (allocation->call_point_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->scratch_arena, allocation->call_point_count, sizeof(*build->calls),
      (void**)&build->calls));
  memset(build->calls, 0, allocation->call_point_count * sizeof(*build->calls));
  for (iree_host_size_t i = 0; i < allocation->call_point_count; ++i) {
    const uint32_t node_index = allocation->call_points[i].node_index;
    const loom_op_t* call_op = build->frame->schedule.nodes[node_index].op;
    const loom_symbol_ref_t callee_ref = loom_low_func_call_callee(call_op);
    const loom_op_t* callee_op =
        build->frame->module->symbols.entries[callee_ref.symbol_id].defining_op;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_function_layout_parse(
        build->frame->module, build->frame->target.descriptor_set, callee_op,
        build->scratch_arena, &build->calls[i].abi_layout));
    *out_max_outgoing_bytes =
        iree_max(*out_max_outgoing_bytes,
                 build->calls[i].abi_layout.stack_argument_bytes);
  }
  return iree_ok_status();
}

static uint8_t loom_x86_sysv_frame_vector_width_code(
    loom_x86_register_class_t register_class) {
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_XMM:
      return 1;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return 2;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return 3;
    default:
      return 0;
  }
}

static uint8_t loom_x86_sysv_frame_vector_width(uint64_t widths,
                                                uint32_t physical_register) {
  return (uint8_t)((widths >> (physical_register * 2)) & 3u);
}

static void loom_x86_sysv_frame_set_vector_width(uint64_t* widths,
                                                 uint32_t physical_register,
                                                 uint8_t width) {
  const uint32_t shift = physical_register * 2;
  const uint64_t mask = UINT64_C(3) << shift;
  const uint8_t old_width = (uint8_t)((*widths & mask) >> shift);
  if (width > old_width) {
    *widths = (*widths & ~mask) | ((uint64_t)width << shift);
  }
}

static uint8_t loom_x86_sysv_frame_call_vector_width(
    const loom_x86_sysv_call_build_t* call, uint32_t physical_register) {
  return loom_x86_sysv_frame_vector_width(call->vector_widths,
                                          physical_register);
}

static iree_status_t loom_x86_sysv_frame_mark_call_storage(
    loom_x86_sysv_frame_build_t* build, loom_x86_sysv_call_build_t* call,
    const loom_low_allocation_assignment_t* assignment) {
  loom_x86_register_class_t register_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_logical_register_class(
      build->frame->target.descriptor_set, assignment->descriptor_reg_class_id,
      &register_class));
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const uint32_t physical_register = assignment->location_base + i;
    switch (register_class) {
      case LOOM_X86_REGISTER_CLASS_GPR32:
      case LOOM_X86_REGISTER_CLASS_GPR64:
        if (physical_register >= LOOM_X86_SYSV_GPR_COUNT) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "x86 GPR assignment is out of range");
        }
        if (loom_x86_sysv_gpr_is_caller_saved(physical_register)) {
          const uint16_t bit = (uint16_t)(1u << physical_register);
          call->gpr_mask |= bit;
          build->caller_gpr_mask |= bit;
        }
        break;
      case LOOM_X86_REGISTER_CLASS_XMM:
      case LOOM_X86_REGISTER_CLASS_YMM:
      case LOOM_X86_REGISTER_CLASS_ZMM:
        if (physical_register >= LOOM_X86_SYSV_VECTOR_REGISTER_COUNT) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "x86 vector assignment is out of range");
        }
        const uint8_t width =
            loom_x86_sysv_frame_vector_width_code(register_class);
        loom_x86_sysv_frame_set_vector_width(&call->vector_widths,
                                             physical_register, width);
        loom_x86_sysv_frame_set_vector_width(&build->caller_vector_widths,
                                             physical_register, width);
        break;
      case LOOM_X86_REGISTER_CLASS_K:
        if (physical_register >= LOOM_X86_SYSV_MASK_REGISTER_COUNT) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "x86 mask assignment is out of range");
        }
        const uint8_t bit = (uint8_t)(1u << physical_register);
        call->mask_register_mask |= bit;
        build->caller_mask_register_mask |= bit;
        break;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_x86_sysv_frame_first_call_at_or_after(
    const loom_low_allocation_table_t* allocation, uint32_t point) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = allocation->call_point_count;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    const loom_low_allocation_call_point_t* call =
        &allocation->call_points[middle];
    const loom_liveness_operation_point_t* operation =
        &allocation->liveness.operation_points[call->operation_index];
    if (operation->start_point < point) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower;
}

static iree_status_t loom_x86_sysv_frame_mark_segment_calls(
    loom_x86_sysv_frame_build_t* build,
    const loom_low_allocation_assignment_t* assignment, uint32_t start_point,
    uint32_t end_point) {
  const loom_low_allocation_table_t* allocation = &build->frame->allocation;
  iree_host_size_t call_index =
      loom_x86_sysv_frame_first_call_at_or_after(allocation, start_point);
  while (call_index < allocation->call_point_count) {
    const loom_low_allocation_call_point_t* call_point =
        &allocation->call_points[call_index];
    const loom_liveness_operation_point_t* operation =
        &allocation->liveness.operation_points[call_point->operation_index];
    if (operation->start_point >= end_point) {
      break;
    }
    if (operation->end_point < end_point) {
      IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_mark_call_storage(
          build, &build->calls[call_index], assignment));
    }
    ++call_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_find_preservation(
    loom_x86_sysv_frame_build_t* build) {
  const loom_low_allocation_table_t* allocation = &build->frame->allocation;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "x86 SysV frame planning requires spill-free physical allocation");
    }
    loom_x86_register_class_t register_class = 0;
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_logical_register_class(
        build->frame->target.descriptor_set,
        assignment->descriptor_reg_class_id, &register_class));
    if (register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
        register_class == LOOM_X86_REGISTER_CLASS_GPR64) {
      for (uint32_t unit = 0; unit < assignment->location_count; ++unit) {
        const uint32_t physical_register = assignment->location_base + unit;
        if (physical_register == LOOM_X86_SYSV_GPR_RSP) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "x86 allocation assigned rsp; the SysV reservation is missing");
        }
        if (physical_register >= LOOM_X86_SYSV_GPR_COUNT) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "x86 GPR assignment is out of range");
        }
        if (loom_x86_sysv_gpr_is_callee_saved(physical_register)) {
          build->callee_gpr_mask |= (uint16_t)(1u << physical_register);
        }
      }
    }

    const loom_liveness_segment_range_t segments =
        assignment->liveness_segments;
    if (segments.count == 0) {
      IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_mark_segment_calls(
          build, assignment, assignment->start_point, assignment->end_point));
      continue;
    }
    for (uint32_t j = 0; j < segments.count; ++j) {
      const loom_liveness_segment_t segment =
          allocation->storage_segments[segments.start + j];
      IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_mark_segment_calls(
          build, assignment, segment.start_point, segment.end_point));
    }
  }
  return iree_ok_status();
}

static uint32_t loom_x86_sysv_frame_call_preserved_count(
    const loom_x86_sysv_call_build_t* call) {
  uint32_t count = (uint32_t)iree_math_count_ones_u32(call->gpr_mask) +
                   (uint32_t)iree_math_count_ones_u32(call->mask_register_mask);
  for (uint32_t i = 0; i < LOOM_X86_SYSV_VECTOR_REGISTER_COUNT; ++i) {
    count += loom_x86_sysv_frame_call_vector_width(call, i) != 0;
  }
  return count;
}

static uint32_t loom_x86_sysv_frame_caller_slot_count(
    const loom_x86_sysv_frame_build_t* build) {
  uint32_t vector_count = 0;
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_VECTOR_REGISTER_COUNT; ++reg) {
    vector_count +=
        loom_x86_sysv_frame_vector_width(build->caller_vector_widths, reg) != 0;
  }
  return (uint32_t)iree_math_count_ones_u32(build->caller_gpr_mask) +
         (uint32_t)iree_math_count_ones_u32(build->caller_mask_register_mask) +
         vector_count;
}

static bool loom_x86_sysv_frame_may_need_cycle_slot(
    const loom_x86_sysv_frame_build_t* build) {
  const uint32_t entry_register_count =
      loom_x86_sysv_abi_register_argument_count(&build->plan->abi_layout);
  if (entry_register_count >= 2) {
    return true;
  }
  for (uint32_t i = 0; i < build->plan->call_count; ++i) {
    const loom_x86_sysv_call_build_t* call = &build->calls[i];
    const uint32_t call_register_count =
        loom_x86_sysv_abi_register_argument_count(&call->abi_layout);
    if (call_register_count >= 2 ||
        call->abi_layout.result_count +
                loom_x86_sysv_frame_call_preserved_count(call) >=
            2) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_x86_sysv_frame_append_slot(
    loom_x86_sysv_frame_build_t* build, loom_x86_sysv_frame_slot_kind_t kind,
    uint64_t byte_offset, uint32_t byte_size, uint32_t byte_alignment,
    uint32_t* out_slot_index) {
  if (byte_alignment > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV frame slot alignment exceeds u16");
  }
  if (build->plan->slot_count >= build->slot_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV frame slot capacity exhausted");
  }
  const uint32_t slot_index = build->plan->slot_count++;
  build->slots[slot_index] = (loom_x86_sysv_frame_slot_t){
      .kind = kind,
      .byte_offset = byte_offset,
      .byte_size = byte_size,
      .byte_alignment = (uint16_t)byte_alignment,
  };
  *out_slot_index = slot_index;
  return iree_ok_status();
}

static uint32_t loom_x86_sysv_frame_vector_byte_size(uint8_t width) {
  return UINT32_C(8) << width;
}

static iree_status_t loom_x86_sysv_frame_allocate_caller_slots(
    loom_x86_sysv_frame_build_t* build) {
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_GPR_COUNT; ++reg) {
    if ((build->caller_gpr_mask & (1u << reg)) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_slot(
        build, LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE, 0, 8, 8,
        &build->caller_gpr_slots[reg]));
  }
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_VECTOR_REGISTER_COUNT; ++reg) {
    const uint8_t width =
        loom_x86_sysv_frame_vector_width(build->caller_vector_widths, reg);
    if (width == 0) {
      continue;
    }
    const uint32_t byte_size = loom_x86_sysv_frame_vector_byte_size(width);
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_slot(
        build, LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE, 0, byte_size,
        iree_min(byte_size, 16u), &build->caller_vector_slots[reg]));
  }
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_MASK_REGISTER_COUNT; ++reg) {
    if ((build->caller_mask_register_mask & (1u << reg)) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_slot(
        build, LOOM_X86_SYSV_FRAME_SLOT_CALLER_SAVE, 0, 8, 8,
        &build->caller_mask_slots[reg]));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_reg_class_for_vector_width(
    loom_x86_sysv_frame_build_t* build, uint8_t width,
    uint16_t* out_reg_class_id) {
  const loom_x86_register_class_t register_class =
      width == 1   ? LOOM_X86_REGISTER_CLASS_XMM
      : width == 2 ? LOOM_X86_REGISTER_CLASS_YMM
                   : LOOM_X86_REGISTER_CLASS_ZMM;
  return loom_x86_descriptor_set_register_class_id(
      build->frame->target.descriptor_set, register_class, out_reg_class_id);
}

static iree_status_t loom_x86_sysv_frame_reg_class_byte_size(
    loom_x86_sysv_frame_build_t* build, uint16_t descriptor_reg_class_id,
    uint32_t* out_byte_size) {
  const loom_low_reg_class_t* register_class =
      &build->frame->target.descriptor_set
           ->reg_classes[descriptor_reg_class_id];
  if (register_class->alloc_unit_bits == 0 ||
      (register_class->alloc_unit_bits & 7u) != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 move scratch requires a byte-addressable register class");
  }
  *out_byte_size = register_class->alloc_unit_bits / 8;
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_resolve_temporary(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_t* moves, iree_host_size_t move_count,
    loom_low_move_location_t* out_temporary, bool* out_resolved) {
  (void)moves;
  (void)move_count;
  loom_x86_sysv_frame_build_t* build = (loom_x86_sysv_frame_build_t*)user_data;
  uint32_t byte_size = 0;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_reg_class_byte_size(
      build, storage_class->descriptor_reg_class_id, &byte_size));
  const loom_low_descriptor_set_t* descriptor_set =
      build->frame->target.descriptor_set;
  for (uint16_t i = 0; i < build->cycle_slot_count; ++i) {
    loom_x86_sysv_cycle_slot_t* cycle_slot = &build->cycle_slots[i];
    if (!loom_low_allocation_storage_reg_classes_share(
            descriptor_set, cycle_slot->descriptor_reg_class_id,
            storage_class->descriptor_reg_class_id)) {
      continue;
    }
    loom_x86_sysv_frame_slot_t* slot = &build->slots[cycle_slot->slot_index];
    slot->byte_size = iree_max(slot->byte_size, byte_size);
    slot->byte_alignment =
        iree_max(slot->byte_alignment, iree_min(byte_size, 16u));
    *out_temporary = (loom_low_move_location_t){
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = cycle_slot->slot_index,
    };
    *out_resolved = true;
    return iree_ok_status();
  }

  IREE_ASSERT_LT(build->cycle_slot_count,
                 build->frame->target.descriptor_set->reg_class_count);
  uint32_t slot_index = 0;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_slot(
      build, LOOM_X86_SYSV_FRAME_SLOT_MOVE_SCRATCH, 0, byte_size,
      iree_min(byte_size, 16u), &slot_index));
  build->cycle_slots[build->cycle_slot_count++] = (loom_x86_sysv_cycle_slot_t){
      .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
      .slot_index = slot_index,
  };
  *out_temporary = (loom_low_move_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
      .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
      .location = slot_index,
  };
  *out_resolved = true;
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_resolve_moves(
    loom_x86_sysv_frame_build_t* build, uint32_t raw_move_count,
    loom_x86_sysv_frame_move_range_t* out_range) {
  out_range->start = build->move_count;
  out_range->count = 0;
  iree_host_size_t resolved_count = 0;
  bool complete = false;
  const loom_low_move_sequence_options_t options = {
      .descriptor_set = build->frame->target.descriptor_set,
      .resolve_temporary =
          {
              .fn = loom_x86_sysv_frame_resolve_temporary,
              .user_data = build,
          },
  };
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_resolve(
      &build->move_scratch, raw_move_count, &options,
      build->move_capacity - build->move_count,
      build->moves ? build->moves + build->move_count : NULL, &resolved_count,
      &complete));
  IREE_ASSERT(complete, "x86 frame move scratch must always resolve");
  if (resolved_count > UINT32_MAX - build->move_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV move table exceeds u32 range");
  }
  out_range->count = (uint32_t)resolved_count;
  build->move_count += (uint32_t)resolved_count;
  return iree_ok_status();
}

static loom_low_move_location_t loom_x86_sysv_frame_register_location(
    uint16_t descriptor_reg_class_id, uint32_t physical_register) {
  return (loom_low_move_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .location = physical_register,
  };
}

static loom_low_move_location_t loom_x86_sysv_frame_slot_location(
    uint16_t descriptor_reg_class_id, uint32_t slot_index) {
  return (loom_low_move_location_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .location = slot_index,
  };
}

static loom_low_move_location_t loom_x86_sysv_frame_assignment_location(
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT_EQ(assignment->location_kind,
                 LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  IREE_ASSERT_EQ(assignment->location_count, 1);
  return loom_x86_sysv_frame_register_location(
      assignment->descriptor_reg_class_id, assignment->location_base);
}

static loom_low_move_location_t loom_x86_sysv_frame_abi_location(
    loom_type_t type, int64_t abi_location) {
  IREE_ASSERT(loom_x86_sysv_abi_location_is_register(abi_location));
  const uint16_t descriptor_reg_class_id =
      loom_low_register_type_class_id(type);
  return loom_x86_sysv_frame_register_location(descriptor_reg_class_id,
                                               (uint32_t)abi_location);
}

static iree_status_t loom_x86_sysv_frame_append_entry_moves(
    loom_x86_sysv_frame_build_t* build) {
  const loom_func_like_t function = loom_func_like_const_cast(
      build->frame->module, build->frame->function_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  uint32_t move_count = 0;
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    const int64_t abi_location = build->plan->abi_layout.argument_locations[i];
    if (!loom_x86_sysv_abi_location_is_register(abi_location)) {
      continue;
    }
    const loom_value_id_t value_id = argument_ids[i];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_try_map_active_value_assignment(
            &build->frame->allocation, value_id, NULL);
    if (assignment == NULL) {
      continue;
    }
    const loom_type_t type =
        loom_module_value_type(build->frame->module, value_id);
    build->move_scratch.moves[move_count++] = (loom_low_move_t){
        .destination = loom_x86_sysv_frame_assignment_location(assignment),
        .source = loom_x86_sysv_frame_abi_location(type, abi_location),
    };
  }
  return loom_x86_sysv_frame_resolve_moves(build, move_count,
                                           &build->plan->entry_moves);
}

static void loom_x86_sysv_frame_append_preservation_move(
    loom_x86_sysv_frame_build_t* build,
    loom_x86_sysv_preservation_direction_t direction,
    loom_low_move_location_t register_location,
    loom_low_move_location_t slot_location, uint32_t* move_count) {
  loom_low_move_t* move = &build->move_scratch.moves[(*move_count)++];
  if (direction == LOOM_X86_SYSV_PRESERVATION_SAVE) {
    move->destination = slot_location;
    move->source = register_location;
  } else {
    move->destination = register_location;
    move->source = slot_location;
  }
}

static iree_status_t loom_x86_sysv_frame_append_preservation_moves(
    loom_x86_sysv_frame_build_t* build, uint16_t gpr_mask,
    uint64_t vector_widths, uint8_t mask_register_mask,
    const uint32_t* gpr_slots, const uint32_t* vector_slots,
    const uint32_t* mask_slots,
    loom_x86_sysv_preservation_direction_t direction, uint32_t* move_count) {
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_GPR_COUNT; ++reg) {
    if ((gpr_mask & (1u << reg)) == 0) {
      continue;
    }
    loom_x86_sysv_frame_append_preservation_move(
        build, direction,
        loom_x86_sysv_frame_register_location(build->gpr64_class_id, reg),
        loom_x86_sysv_frame_slot_location(build->gpr64_class_id,
                                          gpr_slots[reg]),
        move_count);
  }
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_VECTOR_REGISTER_COUNT; ++reg) {
    const uint8_t width = loom_x86_sysv_frame_vector_width(vector_widths, reg);
    if (width == 0) {
      continue;
    }
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_reg_class_for_vector_width(
        build, width, &reg_class_id));
    loom_x86_sysv_frame_append_preservation_move(
        build, direction,
        loom_x86_sysv_frame_register_location(reg_class_id, reg),
        loom_x86_sysv_frame_slot_location(reg_class_id, vector_slots[reg]),
        move_count);
  }
  uint16_t mask_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (mask_register_mask != 0) {
    IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_register_class_id(
        build->frame->target.descriptor_set, LOOM_X86_REGISTER_CLASS_K,
        &mask_class_id));
  }
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_MASK_REGISTER_COUNT; ++reg) {
    if ((mask_register_mask & (1u << reg)) == 0) {
      continue;
    }
    loom_x86_sysv_frame_append_preservation_move(
        build, direction,
        loom_x86_sysv_frame_register_location(mask_class_id, reg),
        loom_x86_sysv_frame_slot_location(mask_class_id, mask_slots[reg]),
        move_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_append_call_saves(
    loom_x86_sysv_frame_build_t* build, const loom_x86_sysv_call_build_t* call,
    loom_x86_sysv_frame_move_range_t* out_range) {
  uint32_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_preservation_moves(
      build, call->gpr_mask, call->vector_widths, call->mask_register_mask,
      build->caller_gpr_slots, build->caller_vector_slots,
      build->caller_mask_slots, LOOM_X86_SYSV_PRESERVATION_SAVE, &move_count));
  return loom_x86_sysv_frame_resolve_moves(build, move_count, out_range);
}

static iree_status_t loom_x86_sysv_frame_append_call_arguments(
    loom_x86_sysv_frame_build_t* build, const loom_op_t* call_op,
    const loom_x86_sysv_abi_layout_t* layout,
    loom_x86_sysv_frame_move_range_t* out_range) {
  const loom_value_slice_t operands = loom_low_func_call_operands(call_op);
  uint32_t move_count = 0;
  iree_host_size_t operand_index = 0;
  for (iree_host_size_t i = 0; i < layout->argument_count; ++i) {
    const int64_t abi_location = layout->argument_locations[i];
    if (!loom_x86_sysv_abi_location_is_register(abi_location)) {
      continue;
    }
    IREE_ASSERT_LT(operand_index, operands.count);
    const loom_value_id_t value_id = operands.values[operand_index++];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_map_active_value_assignment(
            &build->frame->allocation, value_id, NULL);
    const loom_type_t type =
        loom_module_value_type(build->frame->module, value_id);
    build->move_scratch.moves[move_count++] = (loom_low_move_t){
        .destination = loom_x86_sysv_frame_abi_location(type, abi_location),
        .source = loom_x86_sysv_frame_assignment_location(assignment),
    };
  }
  IREE_ASSERT_EQ(operand_index, operands.count);
  return loom_x86_sysv_frame_resolve_moves(build, move_count, out_range);
}

static iree_status_t loom_x86_sysv_frame_append_call_results_and_restores(
    loom_x86_sysv_frame_build_t* build, const loom_op_t* call_op,
    const loom_x86_sysv_call_build_t* call,
    loom_x86_sysv_frame_move_range_t* out_range) {
  uint32_t move_count = 0;
  const loom_value_id_t* results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < call_op->result_count; ++i) {
    const loom_value_id_t value_id = results[i];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_map_active_value_assignment(
            &build->frame->allocation, value_id, NULL);
    const loom_type_t type =
        loom_module_value_type(build->frame->module, value_id);
    build->move_scratch.moves[move_count++] = (loom_low_move_t){
        .destination = loom_x86_sysv_frame_assignment_location(assignment),
        .source = loom_x86_sysv_frame_abi_location(
            type, call->abi_layout.result_locations[i]),
    };
  }
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_preservation_moves(
      build, call->gpr_mask, call->vector_widths, call->mask_register_mask,
      build->caller_gpr_slots, build->caller_vector_slots,
      build->caller_mask_slots, LOOM_X86_SYSV_PRESERVATION_RESTORE,
      &move_count));
  return loom_x86_sysv_frame_resolve_moves(build, move_count, out_range);
}

static iree_status_t loom_x86_sysv_frame_append_calls(
    loom_x86_sysv_frame_build_t* build) {
  const loom_low_allocation_table_t* allocation = &build->frame->allocation;
  for (uint32_t i = 0; i < build->plan->call_count; ++i) {
    const loom_low_allocation_call_point_t* call_point =
        &allocation->call_points[i];
    const loom_op_t* call_op =
        build->frame->schedule.nodes[call_point->node_index].op;
    loom_x86_sysv_call_plan_t* call_plan = &build->call_plans[i];
    call_plan->node_index = call_point->node_index;
    call_plan->move_start = build->move_count;
    loom_x86_sysv_frame_move_range_t range = {0};
    IREE_RETURN_IF_ERROR(
        loom_x86_sysv_frame_append_call_saves(build, &build->calls[i], &range));
    call_plan->save_move_count = range.count;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_call_arguments(
        build, call_op, &build->calls[i].abi_layout, &range));
    call_plan->argument_move_count = range.count;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_call_results_and_restores(
        build, call_op, &build->calls[i], &range));
    call_plan->result_and_restore_move_count = range.count;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_append_returns(
    loom_x86_sysv_frame_build_t* build) {
  const loom_low_schedule_table_t* schedule = &build->frame->schedule;
  for (uint32_t i = 0; i < build->plan->return_count; ++i) {
    const uint32_t node_index = schedule->return_node_indices[i];
    const loom_op_t* return_op = schedule->nodes[node_index].op;
    const loom_value_id_t* operands = loom_op_const_operands(return_op);
    for (uint16_t j = 0; j < return_op->operand_count; ++j) {
      const loom_value_id_t value_id = operands[j];
      const loom_low_allocation_assignment_t* assignment =
          loom_low_allocation_map_active_value_assignment(
              &build->frame->allocation, value_id, NULL);
      const loom_type_t type =
          loom_module_value_type(build->frame->module, value_id);
      build->move_scratch.moves[j] = (loom_low_move_t){
          .destination = loom_x86_sysv_frame_abi_location(
              type, build->plan->abi_layout.result_locations[j]),
          .source = loom_x86_sysv_frame_assignment_location(assignment),
      };
    }
    loom_x86_sysv_return_plan_t* return_plan = &build->return_plans[i];
    return_plan->node_index = node_index;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_resolve_moves(
        build, return_op->operand_count, &return_plan->result_moves));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_allocate_callee_slots(
    loom_x86_sysv_frame_build_t* build) {
  for (uint32_t reg = 0; reg < LOOM_X86_SYSV_GPR_COUNT; ++reg) {
    if ((build->callee_gpr_mask & (1u << reg)) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_slot(
        build, LOOM_X86_SYSV_FRAME_SLOT_CALLEE_SAVE, 0, 8, 8,
        &build->callee_gpr_slots[reg]));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_append_callee_moves(
    loom_x86_sysv_frame_build_t* build) {
  uint32_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_preservation_moves(
      build, build->callee_gpr_mask, /*vector_widths=*/0,
      /*mask_register_mask=*/0, build->callee_gpr_slots,
      /*vector_slots=*/NULL, /*mask_slots=*/NULL,
      LOOM_X86_SYSV_PRESERVATION_SAVE, &move_count));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_resolve_moves(
      build, move_count, &build->plan->callee_save_moves));
  move_count = 0;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_append_preservation_moves(
      build, build->callee_gpr_mask, /*vector_widths=*/0,
      /*mask_register_mask=*/0, build->callee_gpr_slots,
      /*vector_slots=*/NULL, /*mask_slots=*/NULL,
      LOOM_X86_SYSV_PRESERVATION_RESTORE, &move_count));
  return loom_x86_sysv_frame_resolve_moves(build, move_count,
                                           &build->plan->callee_restore_moves);
}

static bool loom_x86_sysv_frame_accumulate_group_bound(uint32_t raw_count,
                                                       uint64_t* total_bound,
                                                       uint32_t* max_group) {
  *max_group = iree_max(*max_group, raw_count);
  const uint64_t cycle_bound = raw_count / 2;
  return iree_checked_add_u64(*total_bound, raw_count + cycle_bound,
                              total_bound);
}

static iree_status_t loom_x86_sysv_frame_allocate_moves(
    loom_x86_sysv_frame_build_t* build) {
  uint64_t move_bound = 0;
  uint32_t max_group = 0;
  const uint32_t argument_count =
      loom_x86_sysv_abi_register_argument_count(&build->plan->abi_layout);
  if (!loom_x86_sysv_frame_accumulate_group_bound(argument_count, &move_bound,
                                                  &max_group)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV move bound overflows");
  }
  const uint32_t callee_save_count =
      (uint32_t)iree_math_count_ones_u32(build->callee_gpr_mask);
  if (!loom_x86_sysv_frame_accumulate_group_bound(callee_save_count,
                                                  &move_bound, &max_group) ||
      !loom_x86_sysv_frame_accumulate_group_bound(callee_save_count,
                                                  &move_bound, &max_group)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV move bound overflows");
  }
  for (uint32_t i = 0; i < build->plan->call_count; ++i) {
    const uint32_t preserve_count =
        loom_x86_sysv_frame_call_preserved_count(&build->calls[i]);
    const uint32_t counts[] = {
        preserve_count,
        loom_x86_sysv_abi_register_argument_count(&build->calls[i].abi_layout),
        preserve_count + (uint32_t)build->calls[i].abi_layout.result_count,
    };
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(counts); ++j) {
      if (!loom_x86_sysv_frame_accumulate_group_bound(counts[j], &move_bound,
                                                      &max_group)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "x86 SysV move bound overflows");
      }
    }
  }
  for (uint32_t i = 0; i < build->plan->return_count; ++i) {
    if (!loom_x86_sysv_frame_accumulate_group_bound(
            (uint32_t)build->plan->abi_layout.result_count, &move_bound,
            &max_group)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV move bound overflows");
    }
  }
  if (move_bound > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV move table exceeds u32 range");
  }
  build->move_capacity = (uint32_t)move_bound;
  if (build->move_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->arena, build->move_capacity, sizeof(*build->moves),
        (void**)&build->moves));
  }
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_initialize(
      build->scratch_arena, max_group, &build->move_scratch));
  build->plan->moves = build->moves;
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_layout_stack(
    loom_x86_sysv_frame_build_t* build, uint32_t max_outgoing_bytes) {
  const loom_low_storage_layout_t* storage_layout =
      &build->frame->schedule.requirements.storage_layout;
  if (storage_layout->space_sizes.scratch_bytes != 0 ||
      storage_layout->space_sizes.private_bytes != 0 ||
      storage_layout->space_sizes.workgroup_bytes != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 SysV frames only support function-local stack storage");
  }
  const loom_low_storage_layout_requirement_t stack_requirement =
      loom_low_storage_layout_requirement(storage_layout,
                                          LOOM_STORAGE_SPACE_STACK);
  if (stack_requirement.minimum_alignment > 16) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 SysV stack storage alignment above 16 bytes requires dynamic "
        "realignment");
  }

  uint64_t cursor = max_outgoing_bytes;
  if (stack_requirement.byte_length != 0) {
    if (!iree_checked_align_u64(cursor, stack_requirement.minimum_alignment,
                                &cursor)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV stack storage alignment overflows");
    }
    build->plan->stack_storage_offset = cursor;
    build->plan->stack_storage_size = stack_requirement.byte_length;
    if (!iree_checked_add_u64(cursor, stack_requirement.byte_length, &cursor)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV stack storage size overflows");
    }
  }
  for (uint32_t i = 0; i < build->plan->slot_count; ++i) {
    loom_x86_sysv_frame_slot_t* slot = &build->slots[i];
    if (!iree_checked_align_u64(cursor, slot->byte_alignment, &cursor)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV frame slot alignment overflows");
    }
    slot->byte_offset = cursor;
    if (!iree_checked_add_u64(cursor, slot->byte_size, &cursor)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV frame slot size overflows");
    }
  }
  if (cursor != 0 || build->plan->call_count != 0) {
    uint64_t biased_size = 0;
    if (!iree_checked_add_u64(cursor, 8, &biased_size) ||
        !iree_checked_align_u64(biased_size, 16, &biased_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 SysV frame size overflows");
    }
    build->plan->frame_size = biased_size - 8;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_sysv_frame_initialize_storage(
    loom_x86_sysv_frame_build_t* build) {
  const uint32_t caller_slot_count =
      loom_x86_sysv_frame_caller_slot_count(build);
  const uint32_t callee_slot_count =
      (uint32_t)iree_math_count_ones_u32(build->callee_gpr_mask);
  const uint32_t cycle_slot_limit =
      loom_x86_sysv_frame_may_need_cycle_slot(build)
          ? build->frame->target.descriptor_set->reg_class_count
          : 0;
  uint64_t slot_capacity =
      (uint64_t)caller_slot_count + callee_slot_count + cycle_slot_limit;
  if (slot_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV frame slot table exceeds u32 range");
  }
  build->slot_capacity = (uint32_t)slot_capacity;
  if (build->slot_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->arena, build->slot_capacity, sizeof(*build->slots),
        (void**)&build->slots));
  }
  if (cycle_slot_limit != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, cycle_slot_limit, sizeof(*build->cycle_slots),
        (void**)&build->cycle_slots));
  }
  build->plan->slots = build->slots;
  return loom_x86_sysv_frame_allocate_caller_slots(build);
}

iree_status_t loom_x86_sysv_frame_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_x86_sysv_frame_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(frame);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_x86_sysv_frame_plan_t){0};
  if (!loom_low_func_def_isa(frame->function_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 SysV frame requires low.func.def");
  }
  IREE_ASSERT_EQ(frame->schedule.error_count, 0,
                 "x86 frame planning requires a completed schedule");
  IREE_ASSERT_EQ(frame->allocation.error_count, 0,
                 "x86 frame planning requires a completed allocation");
  IREE_ASSERT_EQ(frame->schedule.call_node_count,
                 frame->allocation.call_point_count,
                 "allocation must retain every scheduled call");
  IREE_ASSERT_EQ(frame->schedule.return_node_count,
                 frame->schedule.requirements.return_count,
                 "schedule must retain every function return");
  if (frame->schedule.call_node_count > UINT32_MAX ||
      frame->schedule.return_node_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 SysV control boundary count exceeds u32");
  }

  loom_x86_sysv_frame_build_t build = {
      .frame = frame,
      .arena = arena,
      .scratch_arena = scratch_arena,
      .plan = out_plan,
  };
  memset(build.caller_gpr_slots, 0xFF, sizeof(build.caller_gpr_slots));
  memset(build.caller_vector_slots, 0xFF, sizeof(build.caller_vector_slots));
  memset(build.caller_mask_slots, 0xFF, sizeof(build.caller_mask_slots));
  memset(build.callee_gpr_slots, 0xFF, sizeof(build.callee_gpr_slots));
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_register_class_id(
      frame->target.descriptor_set, LOOM_X86_REGISTER_CLASS_GPR64,
      &build.gpr64_class_id));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_function_layout_parse(
      frame->module, frame->target.descriptor_set, frame->function_op,
      scratch_arena, &out_plan->abi_layout));
  uint32_t max_outgoing_bytes = 0;
  IREE_RETURN_IF_ERROR(
      loom_x86_sysv_frame_call_layouts(&build, &max_outgoing_bytes));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_find_preservation(&build));

  out_plan->call_count = (uint32_t)frame->allocation.call_point_count;
  out_plan->return_count = (uint32_t)frame->schedule.return_node_count;
  loom_x86_sysv_call_plan_t* call_plans = NULL;
  loom_x86_sysv_return_plan_t* return_plans = NULL;
  if (out_plan->call_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_plan->call_count, sizeof(*call_plans), (void**)&call_plans));
    memset(call_plans, 0, out_plan->call_count * sizeof(*call_plans));
  }
  if (out_plan->return_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_plan->return_count, sizeof(*return_plans),
        (void**)&return_plans));
    memset(return_plans, 0, out_plan->return_count * sizeof(*return_plans));
  }
  out_plan->calls = call_plans;
  out_plan->returns = return_plans;
  build.call_plans = call_plans;
  build.return_plans = return_plans;

  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_initialize_storage(&build));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_frame_allocate_moves(&build));
  loom_low_allocation_value_scratch_t value_scratch = {0};
  iree_status_t status = loom_low_allocation_acquire_value_scratch(
      &frame->allocation, &value_scratch);
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_append_entry_moves(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_append_calls(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_append_returns(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_allocate_callee_slots(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_append_callee_moves(&build);
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  if (iree_status_is_ok(status)) {
    status = loom_x86_sysv_frame_layout_stack(&build, max_outgoing_bytes);
  }
  if (iree_status_is_ok(status)) {
    out_plan->move_count = build.move_count;
  }
  return status;
}
