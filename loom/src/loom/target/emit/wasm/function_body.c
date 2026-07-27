// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/function_body.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/emit/wasm/binary_writer.h"
#include "loom/target/emit/wasm/types.h"

enum {
  LOOM_WASM_OPCODE_BLOCK = 0x02,
  LOOM_WASM_OPCODE_LOOP = 0x03,
  LOOM_WASM_OPCODE_END = 0x0B,
  LOOM_WASM_OPCODE_ELSE = 0x05,
  LOOM_WASM_OPCODE_IF = 0x04,
  LOOM_WASM_OPCODE_BR = 0x0C,
  LOOM_WASM_OPCODE_BR_IF = 0x0D,
  LOOM_WASM_OPCODE_CALL = 0x10,
  LOOM_WASM_OPCODE_RETURN = 0x0F,
  LOOM_WASM_OPCODE_LOCAL_GET = 0x20,
  LOOM_WASM_OPCODE_LOCAL_SET = 0x21,
  LOOM_WASM_OPCODE_I32_CONST = 0x41,
  LOOM_WASM_OPCODE_I32_EQZ = 0x45,
  LOOM_WASM_OPCODE_I32_ADD = 0x6A,
  LOOM_WASM_OPCODE_I32_SUB = 0x6B,
  LOOM_WASM_OPCODE_I32_MUL = 0x6C,
  LOOM_WASM_OPCODE_I32_LT_S = 0x48,
  LOOM_WASM_OPCODE_I32_LT_U = 0x49,
  LOOM_WASM_OPCODE_F32_ADD = 0x92,
  LOOM_WASM_OPCODE_SIMD_PREFIX = 0xFD,
};

enum {
  LOOM_WASM_BLOCK_TYPE_EMPTY = 0x40,
};

enum {
  LOOM_WASM_SIMD_SUBOPCODE_V128_LOAD = 0x00,
  LOOM_WASM_SIMD_SUBOPCODE_V128_STORE = 0x0B,
  LOOM_WASM_SIMD_SUBOPCODE_V128_CONST = 0x0C,
  LOOM_WASM_SIMD_SUBOPCODE_I8X16_SHUFFLE = 0x0D,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_SPLAT = 0x11,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_EXTRACT_LANE = 0x1B,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_REPLACE_LANE = 0x1C,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_EXTRACT_LANE = 0x1F,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_REPLACE_LANE = 0x20,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_EQ = 0x37,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_NE = 0x38,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_S = 0x39,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_U = 0x3A,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_S = 0x3B,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_U = 0x3C,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_S = 0x3D,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_U = 0x3E,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_S = 0x3F,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_U = 0x40,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_EQ = 0x41,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_LT = 0x43,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_GT = 0x44,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_LE = 0x45,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_GE = 0x46,
  LOOM_WASM_SIMD_SUBOPCODE_V128_BITSELECT = 0x52,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_ADD = 0xAE,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_SUB = 0xB1,
  LOOM_WASM_SIMD_SUBOPCODE_I32X4_MUL = 0xB5,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_ADD = 0xE4,
  LOOM_WASM_SIMD_SUBOPCODE_F32X4_MUL = 0xE6,
};

enum {
  LOOM_WASM_ENCODING_V128_LOAD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_LOAD,
  LOOM_WASM_ENCODING_V128_STORE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_STORE,
  LOOM_WASM_ENCODING_V128_CONST =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_V128_CONST,
  LOOM_WASM_ENCODING_I8X16_SHUFFLE = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                     LOOM_WASM_SIMD_SUBOPCODE_I8X16_SHUFFLE,
  LOOM_WASM_ENCODING_I32X4_SPLAT = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                   LOOM_WASM_SIMD_SUBOPCODE_I32X4_SPLAT,
  LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_I32X4_EXTRACT_LANE,
  LOOM_WASM_ENCODING_I32X4_REPLACE_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_I32X4_REPLACE_LANE,
  LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_F32X4_EXTRACT_LANE,
  LOOM_WASM_ENCODING_F32X4_REPLACE_LANE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
      LOOM_WASM_SIMD_SUBOPCODE_F32X4_REPLACE_LANE,
  LOOM_WASM_ENCODING_I32X4_EQ =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_EQ,
  LOOM_WASM_ENCODING_I32X4_NE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_NE,
  LOOM_WASM_ENCODING_I32X4_LT_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_S,
  LOOM_WASM_ENCODING_I32X4_LT_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LT_U,
  LOOM_WASM_ENCODING_I32X4_GT_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_S,
  LOOM_WASM_ENCODING_I32X4_GT_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GT_U,
  LOOM_WASM_ENCODING_I32X4_LE_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_S,
  LOOM_WASM_ENCODING_I32X4_LE_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_LE_U,
  LOOM_WASM_ENCODING_I32X4_GE_S =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_S,
  LOOM_WASM_ENCODING_I32X4_GE_U =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_GE_U,
  LOOM_WASM_ENCODING_F32X4_EQ =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_EQ,
  LOOM_WASM_ENCODING_F32X4_LT =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_LT,
  LOOM_WASM_ENCODING_F32X4_GT =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_GT,
  LOOM_WASM_ENCODING_F32X4_LE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_LE,
  LOOM_WASM_ENCODING_F32X4_GE =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_GE,
  LOOM_WASM_ENCODING_V128_BITSELECT = (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) |
                                      LOOM_WASM_SIMD_SUBOPCODE_V128_BITSELECT,
  LOOM_WASM_ENCODING_I32X4_ADD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_ADD,
  LOOM_WASM_ENCODING_I32X4_SUB =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_SUB,
  LOOM_WASM_ENCODING_I32X4_MUL =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_I32X4_MUL,
  LOOM_WASM_ENCODING_F32X4_ADD =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_ADD,
  LOOM_WASM_ENCODING_F32X4_MUL =
      (LOOM_WASM_OPCODE_SIMD_PREFIX << 8) | LOOM_WASM_SIMD_SUBOPCODE_F32X4_MUL,
};

typedef struct loom_wasm_local_entry_t {
  // Descriptor-set-local register class ID from allocation.
  uint16_t descriptor_reg_class_id;
  // Class-local target-id assignment base from allocation.
  uint32_t location_base;
  // Wasm value type stored in this local.
  loom_wasm_value_type_t value_type;
  // Function-local Wasm index assigned by this emitter.
  uint32_t local_index;
} loom_wasm_local_entry_t;

typedef struct loom_wasm_local_value_entry_t {
  // Function-local Wasm index assigned to this SSA value.
  uint32_t local_index;
  // Wasm value type stored in this local.
  loom_wasm_value_type_t value_type;
} loom_wasm_local_value_entry_t;

typedef struct loom_wasm_local_layout_t {
  // Allocator used for local metadata arrays.
  iree_allocator_t allocator;
  // Map from class-local allocation target ids to Wasm local indices.
  loom_wasm_local_entry_t* entries;
  // Number of initialized map entries.
  iree_host_size_t entry_count;
  // Allocated map entry capacity.
  iree_host_size_t entry_capacity;
  // Wasm value types indexed by final Wasm local index.
  loom_wasm_value_type_t* local_types;
  // Number of Wasm locals including parameters.
  iree_host_size_t local_type_count;
  // Allocated local type capacity.
  iree_host_size_t local_type_capacity;
  // Number of ABI parameter locals at the start of |local_types|.
  uint32_t parameter_count;
  // Value-ordinal to local lookup table.
  loom_wasm_local_value_entry_t* values;
  // Number of records in |values|.
  iree_host_size_t value_count;
} loom_wasm_local_layout_t;

typedef struct loom_wasm_attr_name_ids_t {
  // Module string ID for wasm.i32.const's immediate payload.
  loom_string_id_t i32_value;
  // Module string ID for wasm.v128.const's low 64-bit immediate payload.
  loom_string_id_t lo64;
  // Module string ID for wasm.v128.const's high 64-bit immediate payload.
  loom_string_id_t hi64;
  // Module string ID for i32x4 lane-immediate payloads.
  loom_string_id_t lane;
  // Module string IDs for i8x16.shuffle byte-lane immediate payloads.
  loom_string_id_t shuffle_lanes[16];
} loom_wasm_attr_name_ids_t;

typedef struct loom_wasm_emit_state_t {
  // Allocation table supplying class-local target ids.
  const loom_low_allocation_table_t* allocation;
  // Module-owned options used by structural instructions such as calls.
  const loom_wasm_function_body_options_t* options;
  // Cached module attr-name IDs used to decode low packet immediates.
  loom_wasm_attr_name_ids_t attr_names;
  // Derived Wasm local namespace layout.
  loom_wasm_local_layout_t locals;
  // Mutable body payload writer.
  loom_wasm_binary_writer_t writer;
  // Structural facts observed while emitting the function body.
  loom_wasm_function_body_flags_t flags;
} loom_wasm_emit_state_t;

static const iree_string_view_t kWasmAttrI32ValueName = IREE_SVL("i32_value");
static const iree_string_view_t kWasmAttrLo64Name = IREE_SVL("lo64");
static const iree_string_view_t kWasmAttrHi64Name = IREE_SVL("hi64");
static const iree_string_view_t kWasmAttrLaneName = IREE_SVL("lane");
static const iree_string_view_t kWasmShuffleLaneAttrNames[16] = {
    IREE_SVL("lane0"),  IREE_SVL("lane1"),  IREE_SVL("lane2"),
    IREE_SVL("lane3"),  IREE_SVL("lane4"),  IREE_SVL("lane5"),
    IREE_SVL("lane6"),  IREE_SVL("lane7"),  IREE_SVL("lane8"),
    IREE_SVL("lane9"),  IREE_SVL("lane10"), IREE_SVL("lane11"),
    IREE_SVL("lane12"), IREE_SVL("lane13"), IREE_SVL("lane14"),
    IREE_SVL("lane15"),
};

static void loom_wasm_attr_name_ids_initialize(
    const loom_module_t* module, loom_wasm_attr_name_ids_t* out_attr_names) {
  *out_attr_names = (loom_wasm_attr_name_ids_t){
      .i32_value = loom_module_lookup_string(module, kWasmAttrI32ValueName),
      .lo64 = loom_module_lookup_string(module, kWasmAttrLo64Name),
      .hi64 = loom_module_lookup_string(module, kWasmAttrHi64Name),
      .lane = loom_module_lookup_string(module, kWasmAttrLaneName),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kWasmShuffleLaneAttrNames);
       ++i) {
    out_attr_names->shuffle_lanes[i] =
        loom_module_lookup_string(module, kWasmShuffleLaneAttrNames[i]);
  }
}

static iree_status_t loom_wasm_write_opcode(loom_wasm_binary_writer_t* writer,
                                            uint32_t encoding_id) {
  if (encoding_id <= UINT8_MAX) {
    return loom_wasm_binary_write_u8(writer, (uint8_t)encoding_id);
  }
  const uint32_t prefix = encoding_id >> 8;
  const uint32_t subopcode = encoding_id & 0xFFu;
  if (prefix != LOOM_WASM_OPCODE_SIMD_PREFIX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unsupported Wasm opcode prefix 0x%02" PRIx32,
                            prefix);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(writer, (uint8_t)prefix));
  return loom_wasm_binary_write_u32_leb(writer, subopcode);
}

static void loom_wasm_local_layout_deinitialize(
    loom_wasm_local_layout_t* layout) {
  iree_allocator_free(layout->allocator, layout->entries);
  iree_allocator_free(layout->allocator, layout->local_types);
  iree_allocator_free(layout->allocator, layout->values);
  *layout = (loom_wasm_local_layout_t){0};
}

static iree_status_t loom_wasm_local_layout_initialize_values(
    loom_wasm_local_layout_t* layout, iree_host_size_t value_count) {
  layout->value_count = value_count;
  if (value_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(value_count, sizeof(*layout->values),
                                  &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local value map size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(layout->allocator, byte_length,
                                             (void**)&layout->values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    layout->values[i] = (loom_wasm_local_value_entry_t){
        .local_index = UINT32_MAX,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_reserve_entries(
    loom_wasm_local_layout_t* layout, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= layout->entry_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      layout->entry_capacity ? layout->entry_capacity * 2 : 16;
  if (new_capacity < layout->entry_capacity ||
      new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(*layout->entries),
                                  &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local entry table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(layout->allocator, byte_length,
                                              (void**)&layout->entries));
  layout->entry_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_reserve_types(
    loom_wasm_local_layout_t* layout, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= layout->local_type_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      layout->local_type_capacity ? layout->local_type_capacity * 2 : 16;
  if (new_capacity < layout->local_type_capacity ||
      new_capacity < minimum_capacity) {
    new_capacity = minimum_capacity;
  }
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(*layout->local_types),
                                  &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local type table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_realloc(layout->allocator, byte_length,
                                              (void**)&layout->local_types));
  layout->local_type_capacity = new_capacity;
  return iree_ok_status();
}

static loom_wasm_local_entry_t* loom_wasm_local_layout_find_entry(
    loom_wasm_local_layout_t* layout, uint16_t descriptor_reg_class_id,
    uint32_t location_base) {
  for (iree_host_size_t i = 0; i < layout->entry_count; ++i) {
    loom_wasm_local_entry_t* entry = &layout->entries[i];
    if (entry->descriptor_reg_class_id == descriptor_reg_class_id &&
        entry->location_base == location_base) {
      return entry;
    }
  }
  return NULL;
}

static iree_status_t loom_wasm_local_layout_append_type(
    loom_wasm_local_layout_t* layout, loom_wasm_value_type_t value_type,
    uint32_t* out_local_index) {
  if (layout->local_type_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm local index exceeds u32");
  }
  IREE_RETURN_IF_ERROR(loom_wasm_local_layout_reserve_types(
      layout, layout->local_type_count + 1));
  *out_local_index = (uint32_t)layout->local_type_count;
  layout->local_types[layout->local_type_count++] = value_type;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_add_entry(
    loom_wasm_local_layout_t* layout, uint16_t descriptor_reg_class_id,
    uint32_t location_base, loom_wasm_value_type_t value_type,
    uint32_t local_index) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_reserve_entries(layout, layout->entry_count + 1));
  layout->entries[layout->entry_count++] = (loom_wasm_local_entry_t){
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .location_base = location_base,
      .value_type = value_type,
      .local_index = local_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_wasm_validate_target_id_assignment(
    const loom_low_allocation_assignment_t* assignment,
    loom_wasm_value_type_t* out_value_type) {
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u is not allocated to a target-local id",
        (unsigned)assignment->value_id);
  }
  if (assignment->location_count != 1 || assignment->unit_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u uses a multi-unit target-id assignment",
        (unsigned)assignment->value_id);
  }
  if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm value %u is not allocated as a register value",
        (unsigned)assignment->value_id);
  }
  return loom_wasm_value_type_from_descriptor_register_class(
      assignment->descriptor_reg_class_id, out_value_type);
}

static iree_status_t loom_wasm_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal,
    const loom_low_allocation_assignment_t** out_assignment,
    loom_wasm_value_type_t* out_value_type) {
  if (out_value_ordinal) {
    *out_value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  }
  *out_assignment = NULL;

  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(allocation->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no allocation ordinal",
                            (unsigned)value_id);
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(allocation,
                                                       value_ordinal, NULL);
  if (assignment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no allocation assignment",
                            (unsigned)value_id);
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_validate_target_id_assignment(assignment, out_value_type));
  if (out_value_ordinal) {
    *out_value_ordinal = value_ordinal;
  }
  *out_assignment = assignment;
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_set_value(
    loom_wasm_local_layout_t* layout, loom_value_ordinal_t value_ordinal,
    uint32_t local_index, loom_wasm_value_type_t value_type) {
  if (value_ordinal >= layout->value_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm allocation ordinal %u is out of local range",
                            (unsigned)value_ordinal);
  }
  loom_wasm_local_value_entry_t* value = &layout->values[value_ordinal];
  if (value->local_index != UINT32_MAX) {
    if (value->local_index != local_index || value->value_type != value_type) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Wasm allocation ordinal %u changed local "
                              "mapping",
                              (unsigned)value_ordinal);
    }
    return iree_ok_status();
  }
  *value = (loom_wasm_local_value_entry_t){
      .local_index = local_index,
      .value_type = value_type,
  };
  return iree_ok_status();
}

static iree_status_t loom_wasm_local_layout_add_parameter(
    const loom_low_allocation_table_t* allocation,
    loom_wasm_local_layout_t* layout, loom_value_id_t value_id,
    uint32_t parameter_index) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  const loom_low_allocation_assignment_t* assignment = NULL;
  loom_wasm_value_type_t value_type = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_map_assignment(
      allocation, value_id, &value_ordinal, &assignment, &value_type));
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_append_type(layout, value_type, &local_index));
  if (local_index != parameter_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm parameter index overflow");
  }

  loom_wasm_local_entry_t* existing = loom_wasm_local_layout_find_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base);
  if (existing) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm parameters %u and %" PRIu32 " share allocator target id %" PRIu32
        " for one register class",
        existing->local_index, parameter_index, assignment->location_base);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_local_layout_add_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base,
      value_type, parameter_index));
  return loom_wasm_local_layout_set_value(layout, value_ordinal,
                                          parameter_index, value_type);
}

static iree_status_t loom_wasm_local_layout_add_assignment(
    const loom_low_allocation_table_t* allocation,
    loom_wasm_local_layout_t* layout,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(allocation->module,
                                               assignment->value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no allocation ordinal",
                            (unsigned)assignment->value_id);
  }
  loom_wasm_value_type_t value_type = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_validate_target_id_assignment(assignment, &value_type));
  loom_wasm_local_entry_t* existing = loom_wasm_local_layout_find_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base);
  if (existing) {
    if (existing->value_type != value_type) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Wasm target id %" PRIu32 " changes value type",
                              assignment->location_base);
    }
    return loom_wasm_local_layout_set_value(layout, value_ordinal,
                                            existing->local_index, value_type);
  }

  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_local_layout_append_type(layout, value_type, &local_index));
  IREE_RETURN_IF_ERROR(loom_wasm_local_layout_add_entry(
      layout, assignment->descriptor_reg_class_id, assignment->location_base,
      value_type, local_index));
  return loom_wasm_local_layout_set_value(layout, value_ordinal, local_index,
                                          value_type);
}

static iree_status_t loom_wasm_build_local_layout(
    const loom_low_allocation_table_t* allocation, iree_allocator_t allocator,
    loom_wasm_local_layout_t* out_layout) {
  *out_layout = (loom_wasm_local_layout_t){
      .allocator = allocator,
  };

  loom_func_like_t function = loom_func_like_cast(
      allocation->module, (loom_op_t*)allocation->function_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm emission requires a func-like low function");
  }

  IREE_RETURN_IF_ERROR(loom_wasm_local_layout_initialize_values(
      out_layout, allocation->liveness.value_count));
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function, &parameter_count);
  out_layout->parameter_count = parameter_count;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < parameter_count && iree_status_is_ok(status); ++i) {
    status = loom_wasm_local_layout_add_parameter(allocation, out_layout,
                                                  parameter_ids[i], i);
  }
  for (iree_host_size_t i = 0;
       i < allocation->assignment_count && iree_status_is_ok(status); ++i) {
    status = loom_wasm_local_layout_add_assignment(allocation, out_layout,
                                                   &allocation->assignments[i]);
  }
  if (!iree_status_is_ok(status)) {
    loom_wasm_local_layout_deinitialize(out_layout);
  }
  return status;
}

static iree_status_t loom_wasm_lookup_local(
    loom_wasm_emit_state_t* state, loom_value_id_t value_id,
    uint32_t* out_local_index, loom_wasm_value_type_t* out_value_type) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(state->allocation->module,
                                               value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= state->locals.value_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no local ordinal",
                            (unsigned)value_id);
  }
  const loom_wasm_local_value_entry_t* value =
      &state->locals.values[value_ordinal];
  if (value->local_index == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm value %u has no local mapping",
                            (unsigned)value_id);
  }
  *out_local_index = value->local_index;
  if (out_value_type) {
    *out_value_type = value->value_type;
  }
  return iree_ok_status();
}

static const loom_named_attr_t* loom_wasm_find_named_attr_by_id(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_wasm_read_i64_attr(loom_named_attr_slice_t attrs,
                                             iree_string_view_t name,
                                             loom_string_id_t name_id,
                                             int64_t* out_value) {
  const loom_named_attr_t* attr =
      loom_wasm_find_named_attr_by_id(attrs, name_id);
  if (!attr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm emission missing required '%.*s' attribute",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm attribute '%.*s' must be i64", (int)name.size,
                            name.data);
  }
  *out_value = attr->value.i64;
  return iree_ok_status();
}

static iree_status_t loom_wasm_read_u8_attr(loom_named_attr_slice_t attrs,
                                            iree_string_view_t name,
                                            loom_string_id_t name_id,
                                            uint8_t maximum_value,
                                            uint8_t* out_value) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_read_i64_attr(attrs, name, name_id, &value));
  if (value < 0 || value > maximum_value) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm attribute '%.*s' is outside [0, %" PRIu8 "]",
                            (int)name.size, name.data, maximum_value);
  }
  *out_value = (uint8_t)value;
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_local_get(loom_wasm_emit_state_t* state,
                                              loom_value_id_t value_id) {
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_lookup_local(state, value_id, &local_index, NULL));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_GET));
  return loom_wasm_binary_write_u32_leb(&state->writer, local_index);
}

static iree_status_t loom_wasm_emit_local_set(loom_wasm_emit_state_t* state,
                                              loom_value_id_t value_id) {
  uint32_t local_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_lookup_local(state, value_id, &local_index, NULL));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_SET));
  return loom_wasm_binary_write_u32_leb(&state->writer, local_index);
}

static iree_status_t loom_wasm_emit_memarg(loom_wasm_emit_state_t* state,
                                           uint32_t alignment_exponent,
                                           uint32_t offset) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, alignment_exponent));
  return loom_wasm_binary_write_u32_leb(&state->writer, offset);
}

static iree_status_t loom_wasm_emit_i32_const(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "wasm.i32.const must be a unary low.const");
  }
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrI32ValueName,
                              state->attr_names.i32_value, &value));
  if (value < INT32_MIN || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "wasm.i32.const value is outside the verified i32 bit-pattern range");
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_i32_leb(&state->writer, (int32_t)(uint32_t)value));
  return loom_wasm_emit_local_set(state, loom_low_const_result(op));
}

static iree_status_t loom_wasm_emit_v128_const(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_const_isa(op) || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "wasm.v128.const must be a unary low.const");
  }
  int64_t low_bits = 0;
  int64_t high_bits = 0;
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrLo64Name,
                              state->attr_names.lo64, &low_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_read_i64_attr(loom_low_const_attrs(op), kWasmAttrHi64Name,
                              state->attr_names.hi64, &high_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u64_le(&state->writer, (uint64_t)low_bits));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u64_le(&state->writer, (uint64_t)high_bits));
  return loom_wasm_emit_local_set(state, loom_low_const_result(op));
}

static iree_status_t loom_wasm_emit_unary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 1 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm unary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_binary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm binary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_ternary_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 3 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm ternary packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[2]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_lane_stack_op(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  const bool extracts_lane =
      descriptor->encoding_id == LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE ||
      descriptor->encoding_id == LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE;
  const uint16_t expected_operand_count = extracts_lane ? 1 : 2;
  if (!loom_low_op_isa(op) || op->operand_count != expected_operand_count ||
      op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm lane packet shape is invalid");
  }
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  uint8_t lane = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_read_u8_attr(attrs, kWasmAttrLaneName,
                                              state->attr_names.lane,
                                              /*maximum_value=*/3, &lane));
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, lane));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_i8x16_shuffle(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "wasm.i8x16.shuffle packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kWasmShuffleLaneAttrNames);
       ++i) {
    uint8_t lane = 0;
    IREE_RETURN_IF_ERROR(loom_wasm_read_u8_attr(
        attrs, kWasmShuffleLaneAttrNames[i], state->attr_names.shuffle_lanes[i],
        /*maximum_value=*/31, &lane));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, lane));
  }
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_v128_load(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 1 || op->result_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "wasm.v128.load packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  loom_value_slice_t results = loom_low_op_results(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memarg(state, /*alignment_exponent=*/4,
                                             /*offset=*/0));
  return loom_wasm_emit_local_set(state, results.values[0]);
}

static iree_status_t loom_wasm_emit_v128_store(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  if (!loom_low_op_isa(op) || op->operand_count != 2 || op->result_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "wasm.v128.store packet shape is invalid");
  }
  loom_value_slice_t operands = loom_low_op_operands(op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[0]));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[1]));
  IREE_RETURN_IF_ERROR(
      loom_wasm_write_opcode(&state->writer, descriptor->encoding_id));
  return loom_wasm_emit_memarg(state, /*alignment_exponent=*/4, /*offset=*/0);
}

static iree_status_t loom_wasm_emit_descriptor_packet(
    loom_wasm_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_id) {
    case LOOM_WASM_OPCODE_I32_CONST:
      return loom_wasm_emit_i32_const(state, op, descriptor);
    case LOOM_WASM_ENCODING_V128_CONST:
      return loom_wasm_emit_v128_const(state, op, descriptor);
    case LOOM_WASM_OPCODE_I32_ADD:
    case LOOM_WASM_OPCODE_I32_SUB:
    case LOOM_WASM_OPCODE_I32_MUL:
    case LOOM_WASM_OPCODE_I32_LT_U:
    case LOOM_WASM_OPCODE_F32_ADD:
    case LOOM_WASM_ENCODING_I32X4_EQ:
    case LOOM_WASM_ENCODING_I32X4_NE:
    case LOOM_WASM_ENCODING_I32X4_LT_S:
    case LOOM_WASM_ENCODING_I32X4_LT_U:
    case LOOM_WASM_ENCODING_I32X4_GT_S:
    case LOOM_WASM_ENCODING_I32X4_GT_U:
    case LOOM_WASM_ENCODING_I32X4_LE_S:
    case LOOM_WASM_ENCODING_I32X4_LE_U:
    case LOOM_WASM_ENCODING_I32X4_GE_S:
    case LOOM_WASM_ENCODING_I32X4_GE_U:
    case LOOM_WASM_ENCODING_I32X4_ADD:
    case LOOM_WASM_ENCODING_I32X4_SUB:
    case LOOM_WASM_ENCODING_I32X4_MUL:
    case LOOM_WASM_ENCODING_F32X4_EQ:
    case LOOM_WASM_ENCODING_F32X4_LT:
    case LOOM_WASM_ENCODING_F32X4_GT:
    case LOOM_WASM_ENCODING_F32X4_LE:
    case LOOM_WASM_ENCODING_F32X4_GE:
    case LOOM_WASM_ENCODING_F32X4_ADD:
    case LOOM_WASM_ENCODING_F32X4_MUL:
      return loom_wasm_emit_binary_stack_op(state, op, descriptor);
    case LOOM_WASM_ENCODING_V128_BITSELECT:
      return loom_wasm_emit_ternary_stack_op(state, op, descriptor);
    case LOOM_WASM_ENCODING_I8X16_SHUFFLE:
      return loom_wasm_emit_i8x16_shuffle(state, op, descriptor);
    case LOOM_WASM_ENCODING_I32X4_SPLAT:
      return loom_wasm_emit_unary_stack_op(state, op, descriptor);
    case LOOM_WASM_ENCODING_I32X4_EXTRACT_LANE:
    case LOOM_WASM_ENCODING_F32X4_EXTRACT_LANE:
    case LOOM_WASM_ENCODING_I32X4_REPLACE_LANE:
    case LOOM_WASM_ENCODING_F32X4_REPLACE_LANE:
      return loom_wasm_emit_lane_stack_op(state, op, descriptor);
    case LOOM_WASM_ENCODING_V128_LOAD:
      return loom_wasm_emit_v128_load(state, op, descriptor);
    case LOOM_WASM_ENCODING_V128_STORE:
      return loom_wasm_emit_v128_store(state, op, descriptor);
    default: {
      iree_string_view_t key = loom_low_descriptor_set_string(
          state->allocation->target.descriptor_set,
          descriptor->key_string_offset);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Wasm descriptor '%.*s' is unsupported",
                              (int)key.size, key.data);
    }
  }
}

static iree_status_t loom_wasm_record_descriptor_flags(
    loom_wasm_emit_state_t* state, const loom_low_descriptor_t* descriptor) {
  if (descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->allocation->target.descriptor_set;
  const uint16_t schedule_class_id = descriptor->schedule_class_id;
  if (schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Wasm descriptor references invalid schedule class %" PRIu16,
        schedule_class_id);
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[schedule_class_id];
  const loom_low_schedule_class_flags_t memory_flags =
      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD |
      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE;
  if (iree_any_bit_set(schedule_class->flags, memory_flags)) {
    state->flags |= LOOM_WASM_FUNCTION_BODY_FLAG_USES_MEMORY;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_low_copy(loom_wasm_emit_state_t* state,
                                             const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_local_get(state, loom_low_copy_source(op)));
  return loom_wasm_emit_local_set(state, loom_low_copy_result(op));
}

static iree_status_t loom_wasm_emit_low_func_call(loom_wasm_emit_state_t* state,
                                                  const loom_op_t* op) {
  if (!loom_low_func_call_isa(op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm call emission requires low.func.call");
  }
  if (state->options == NULL ||
      state->options->resolve_function_index == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm function-body emission requires a module function-index resolver "
        "for low.func.call");
  }

  loom_value_slice_t operands = loom_low_func_call_operands(op);
  for (iree_host_size_t i = 0; i < operands.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, operands.values[i]));
  }

  uint32_t function_index = 0;
  IREE_RETURN_IF_ERROR(state->options->resolve_function_index(
      state->options->resolve_function_index_user_data,
      loom_low_func_call_callee(op), &function_index));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_CALL));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, function_index));

  loom_value_slice_t results = loom_low_func_call_results(op);
  for (iree_host_size_t i = results.count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(
        loom_wasm_emit_local_set(state, results.values[i - 1]));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_low_return(loom_wasm_emit_state_t* state,
                                               const loom_op_t* op) {
  loom_value_slice_t values = loom_low_return_values(op);
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, values.values[i]));
  }
  return loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_RETURN);
}

static iree_status_t loom_wasm_emit_op(loom_wasm_emit_state_t* state,
                                       const loom_op_t* op);

static iree_status_t loom_wasm_emit_structured_region_before_terminator(
    loom_wasm_emit_state_t* state, const loom_region_t* region,
    const loom_op_t* terminator) {
  if (region == NULL || region->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm structured control requires a non-empty low "
                            "region");
  }
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm structured control does not support CFG low regions");
  }
  if (region->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm structured control requires single-block low regions");
  }
  if (terminator == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm structured control requires verified low.scf.yield terminators");
  }

  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* op = block->first_op;
  for (; op && op != terminator; op = op->next_op) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_op(state, op));
  }
  if (op != terminator) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm structured control terminator is not in its low region");
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_value_move(loom_wasm_emit_state_t* state,
                                               loom_value_id_t source_value,
                                               loom_value_id_t target_value) {
  uint32_t source_local_index = 0;
  uint32_t target_local_index = 0;
  loom_wasm_value_type_t source_type = 0;
  loom_wasm_value_type_t target_type = 0;
  IREE_RETURN_IF_ERROR(loom_wasm_lookup_local(
      state, source_value, &source_local_index, &source_type));
  IREE_RETURN_IF_ERROR(loom_wasm_lookup_local(
      state, target_value, &target_local_index, &target_type));
  if (source_type != target_type) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm low value move type mismatch");
  }
  if (source_local_index == target_local_index) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_GET));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, source_local_index));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOCAL_SET));
  return loom_wasm_binary_write_u32_leb(&state->writer, target_local_index);
}

static iree_status_t loom_wasm_emit_value_moves(
    loom_wasm_emit_state_t* state, const loom_value_id_t* source_values,
    const loom_value_id_t* target_values, iree_host_size_t value_count) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_wasm_emit_value_move(state, source_values[i], target_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_low_scf_yield_to_results(
    loom_wasm_emit_state_t* state, const loom_op_t* yield,
    const loom_value_id_t* result_values, iree_host_size_t result_count) {
  if (result_count == 0) {
    return iree_ok_status();
  }
  if (yield == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "resultful low.scf region is missing a verified low.scf.yield");
  }
  loom_value_slice_t yielded_values = loom_low_scf_yield_values(yield);
  if (yielded_values.count != result_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "resultful low.scf.yield operand count does not match parent results");
  }
  return loom_wasm_emit_value_moves(state, yielded_values.values, result_values,
                                    result_count);
}

static iree_status_t loom_wasm_emit_low_scf_if(loom_wasm_emit_state_t* state,
                                               const loom_op_t* op) {
  loom_value_slice_t results = loom_low_scf_if_results(op);
  loom_region_t* then_region = loom_low_scf_if_then_region(op);
  loom_region_t* else_region = loom_low_scf_if_else_region(op);

  if (results.count != 0 && else_region == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "resultful low.scf.if is missing a verified else_region");
  }
  loom_region_branch_t branch =
      loom_region_branch_cast(state->allocation->module, (loom_op_t*)op);
  loom_op_t* then_yield = loom_region_branch_region_terminator(
      state->allocation->module, branch, 0);

  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_local_get(state, loom_low_scf_if_condition(op)));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_IF));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_BLOCK_TYPE_EMPTY));

  IREE_RETURN_IF_ERROR(loom_wasm_emit_structured_region_before_terminator(
      state, then_region, then_yield));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_low_scf_yield_to_results(
      state, then_yield, results.values, results.count));

  if (else_region != NULL) {
    loom_op_t* else_yield = loom_region_branch_region_terminator(
        state->allocation->module, branch, 1);
    IREE_RETURN_IF_ERROR(
        loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_ELSE));
    IREE_RETURN_IF_ERROR(loom_wasm_emit_structured_region_before_terminator(
        state, else_region, else_yield));
    IREE_RETURN_IF_ERROR(loom_wasm_emit_low_scf_yield_to_results(
        state, else_yield, results.values, results.count));
  }
  return loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_END);
}

static iree_status_t loom_wasm_emit_branch_depth(loom_wasm_emit_state_t* state,
                                                 uint8_t opcode,
                                                 uint32_t depth) {
  IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, opcode));
  return loom_wasm_binary_write_u32_leb(&state->writer, depth);
}

static iree_status_t loom_wasm_emit_low_scf_for(loom_wasm_emit_state_t* state,
                                                const loom_op_t* op) {
  const loom_region_t* body_region = loom_low_scf_for_body(op);
  const loom_block_t* body_block = loom_region_const_entry_block(body_region);
  if (body_block == NULL || body_block->arg_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm low.scf.for requires a verified body entry block");
  }
  const loom_op_t* yield = body_block->last_op;
  if (yield == NULL || !loom_low_scf_yield_isa(yield)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm low.scf.for requires a verified low.scf.yield terminator");
  }

  const loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  const loom_value_slice_t results = loom_low_scf_for_results(op);
  const loom_value_slice_t yielded_values = loom_low_scf_yield_values(yield);
  if (body_block->arg_count != iter_args.count + 1 ||
      results.count != iter_args.count ||
      yielded_values.count != iter_args.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm low.scf.for verified shape mismatch");
  }

  const loom_value_id_t iv_value = loom_block_arg_id(body_block, 0);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_value_move(
      state, loom_low_scf_for_lower_bound(op), iv_value));
  for (iree_host_size_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_value_move(
        state, iter_args.values[i], loom_block_arg_id(body_block, i + 1)));
  }

  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_BLOCK));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_BLOCK_TYPE_EMPTY));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_LOOP));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_BLOCK_TYPE_EMPTY));

  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, iv_value));
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_local_get(state, loom_low_scf_for_upper_bound(op)));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_I32_LT_S));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_I32_EQZ));
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_branch_depth(state, LOOM_WASM_OPCODE_BR_IF, /*depth=*/1));

  IREE_RETURN_IF_ERROR(loom_wasm_emit_structured_region_before_terminator(
      state, body_region, yield));
  for (iree_host_size_t i = 0; i < yielded_values.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_value_move(
        state, yielded_values.values[i], loom_block_arg_id(body_block, i + 1)));
  }
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_get(state, iv_value));
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_local_get(state, loom_low_scf_for_step(op)));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_I32_ADD));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_set(state, iv_value));
  IREE_RETURN_IF_ERROR(
      loom_wasm_emit_branch_depth(state, LOOM_WASM_OPCODE_BR, /*depth=*/0));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_END));
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_END));

  for (iree_host_size_t i = 0; i < results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_value_move(
        state, loom_block_arg_id(body_block, i + 1), results.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_structural_op(loom_wasm_emit_state_t* state,
                                                  const loom_op_t* op) {
  if (loom_low_copy_isa(op)) {
    return loom_wasm_emit_low_copy(state, op);
  }
  if (loom_low_func_call_isa(op)) {
    return loom_wasm_emit_low_func_call(state, op);
  }
  if (loom_low_return_isa(op)) {
    return loom_wasm_emit_low_return(state, op);
  }
  if (loom_low_scf_if_isa(op)) {
    return loom_wasm_emit_low_scf_if(state, op);
  }
  if (loom_low_scf_for_isa(op)) {
    return loom_wasm_emit_low_scf_for(state, op);
  }
  iree_string_view_t op_name = loom_op_name(state->allocation->module, op);
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "verified Wasm low function contains unsupported "
                          "structural op '%.*s'",
                          (int)op_name.size, op_name.data);
}

static iree_status_t loom_wasm_emit_op(loom_wasm_emit_state_t* state,
                                       const loom_op_t* op) {
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->allocation->module, &state->allocation->target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return loom_wasm_emit_structural_op(state, op);
  }
  if (packet.descriptor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm descriptor packet '%.*s' did not resolve",
                            (int)packet.key.size, packet.key.data);
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_record_descriptor_flags(state, packet.descriptor));
  return loom_wasm_emit_descriptor_packet(state, op, packet.descriptor);
}

static iree_status_t loom_wasm_emit_local_declarations(
    loom_wasm_emit_state_t* state) {
  uint32_t declaration_count = 0;
  for (iree_host_size_t i = state->locals.parameter_count;
       i < state->locals.local_type_count;) {
    ++declaration_count;
    loom_wasm_value_type_t value_type = state->locals.local_types[i++];
    while (i < state->locals.local_type_count &&
           state->locals.local_types[i] == value_type) {
      ++i;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_wasm_binary_write_u32_leb(&state->writer, declaration_count));
  for (iree_host_size_t i = state->locals.parameter_count;
       i < state->locals.local_type_count;) {
    loom_wasm_value_type_t value_type = state->locals.local_types[i++];
    uint32_t run_length = 1;
    while (i < state->locals.local_type_count &&
           state->locals.local_types[i] == value_type) {
      ++run_length;
      ++i;
    }
    IREE_RETURN_IF_ERROR(
        loom_wasm_binary_write_u32_leb(&state->writer, run_length));
    IREE_RETURN_IF_ERROR(loom_wasm_binary_write_u8(&state->writer, value_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_region(loom_wasm_emit_state_t* state,
                                           const loom_region_t* region) {
  if (region == NULL || region->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm function-body emission requires a non-empty "
                            "low region");
  }
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm function-body emission does not support CFG low regions");
  }
  if (region->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm function-body emission requires structured single-block low "
        "regions");
  }

  const loom_block_t* block = loom_region_const_entry_block(region);
  for (const loom_op_t* op = block->first_op; op; op = op->next_op) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_op(state, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_function_body_payload(
    loom_wasm_emit_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_wasm_emit_local_declarations(state));
  const loom_region_t* body =
      loom_low_function_const_body(state->allocation->function_op);
  IREE_RETURN_IF_ERROR(loom_wasm_emit_region(state, body));
  return loom_wasm_binary_write_u8(&state->writer, LOOM_WASM_OPCODE_END);
}

static iree_status_t loom_wasm_validate_allocation(
    const loom_low_allocation_table_t* allocation) {
  if (allocation->module == NULL || allocation->function_op == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm emission requires an allocation table naming a low function");
  }
  if (allocation->target.descriptor_set !=
      loom_wasm_core_simd128_descriptor_set()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Wasm emission requires wasm.core.simd128");
  }
  if (allocation->spill_count != 0 || allocation->spill_plan_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm emission requires unspilled allocation tables");
  }
  if (allocation->edge_copy_count != 0 ||
      allocation->edge_copy_group_count != 0 ||
      allocation->edge_copy_temporary_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm emission requires structured allocation tables without CFG edge "
        "copies");
  }
  if (allocation->packet_move_temporary_count != 0 ||
      allocation->packet_move_temporary_group_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm emission requires allocation tables without packet-local move "
        "temporaries");
  }
  return iree_ok_status();
}

void loom_wasm_function_body_deinitialize(loom_wasm_function_body_t* body,
                                          iree_allocator_t allocator) {
  if (!body) {
    return;
  }
  iree_allocator_free(allocator, body->data);
  *body = (loom_wasm_function_body_t){0};
}

iree_status_t loom_wasm_emit_function_body(
    const loom_low_allocation_table_t* allocation,
    const loom_wasm_function_body_options_t* options,
    iree_allocator_t allocator, loom_wasm_function_body_t* out_body) {
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(out_body);
  *out_body = (loom_wasm_function_body_t){0};
  IREE_RETURN_IF_ERROR(loom_wasm_validate_allocation(allocation));

  loom_wasm_emit_state_t state = {
      .allocation = allocation,
      .options = options,
  };
  loom_wasm_attr_name_ids_initialize(allocation->module, &state.attr_names);
  loom_wasm_binary_writer_initialize(allocator, &state.writer);
  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  if (iree_status_is_ok(status)) {
    status = loom_wasm_build_local_layout(allocation, allocator, &state.locals);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_emit_function_body_payload(&state);
  }

  loom_wasm_binary_writer_t output_writer;
  loom_wasm_binary_writer_initialize(allocator, &output_writer);
  if (iree_status_is_ok(status) && state.writer.length > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm function body exceeds u32 size");
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_u32_leb(&output_writer,
                                            (uint32_t)state.writer.length);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_binary_write_bytes(&output_writer, state.writer.data,
                                          state.writer.length);
  }
  if (iree_status_is_ok(status) && state.locals.local_type_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Wasm local count exceeds u32");
  }
  if (iree_status_is_ok(status)) {
    *out_body = (loom_wasm_function_body_t){
        .data = output_writer.data,
        .data_length = output_writer.length,
        .body_length = state.writer.length,
        .parameter_count = state.locals.parameter_count,
        .local_count = (uint32_t)state.locals.local_type_count,
        .flags = state.flags,
    };
    output_writer.data = NULL;
  }

  loom_wasm_binary_writer_deinitialize(&output_writer);
  loom_low_allocation_release_value_scratch(&scratch);
  loom_wasm_binary_writer_deinitialize(&state.writer);
  loom_wasm_local_layout_deinitialize(&state.locals);
  return status;
}
