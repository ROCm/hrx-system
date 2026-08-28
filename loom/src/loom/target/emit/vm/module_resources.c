// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_resources.h"

#include <string.h>

#include "loom/ops/global/ops.h"
#include "loom/target/arch/vm/abi/layout.h"
#include "loom/target/arch/vm/ops/ops.h"

typedef uint8_t loom_vm_module_global_bank_t;
enum loom_vm_module_global_bank_e {
  LOOM_VM_MODULE_GLOBAL_BANK_NONE = 0,
  LOOM_VM_MODULE_GLOBAL_BANK_VALUE = 1,
  LOOM_VM_MODULE_GLOBAL_BANK_REF = 2,
  LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION = 3,
  LOOM_VM_MODULE_GLOBAL_BANK_COUNT_ = 4,
};

typedef struct loom_vm_module_global_record_t {
  // Physical global bank selected by the logical source type.
  loom_vm_module_global_bank_t bank;
  // True when the source is an immutable global definition.
  bool is_immutable;
  // Physical ordinal in |bank|.
  uint16_t ordinal;
  // Logical source global type.
  loom_type_t type;
} loom_vm_module_global_record_t;

typedef struct loom_vm_module_resource_counts_t {
  // Number of direct constant-pool records.
  uint32_t constants;
  // Total global count in each physical bank.
  uint32_t globals[LOOM_VM_MODULE_GLOBAL_BANK_COUNT_];
  // Immutable global count in each physical bank.
  uint32_t immutable_globals[LOOM_VM_MODULE_GLOBAL_BANK_COUNT_];
  // Number of direct rodata records.
  uint32_t rodata;
} loom_vm_module_resource_counts_t;

typedef struct loom_vm_module_resource_presence_t {
  // Occupancy bytes indexed by constant-pool ordinal.
  uint8_t* constants;
  // Occupancy bytes indexed by value-global ordinal.
  uint8_t* value_globals;
} loom_vm_module_resource_presence_t;

static iree_status_t loom_vm_module_resource_resolve_source(
    const loom_module_t* module, loom_symbol_ref_t source_ref,
    const loom_op_t** out_source_op) {
  *out_source_op = NULL;
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
      source_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module resource has an invalid source symbol");
  }
  const loom_op_t* source_op =
      module->symbols.entries[source_ref.symbol_id].defining_op;
  if (source_op == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module resource source is undefined");
  }
  *out_source_op = source_op;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_global_record_resolve(
    const loom_module_t* module, const loom_op_t* resource_op,
    loom_vm_module_global_record_t* out_record) {
  *out_record = (loom_vm_module_global_record_t){0};
  const loom_op_t* source_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_resolve_source(
      module, loom_vm_global_source(resource_op), &source_op));

  loom_value_id_t type_value = LOOM_VALUE_ID_INVALID;
  if (loom_global_constant_isa(source_op)) {
    out_record->is_immutable = true;
    type_value = loom_global_constant_type(source_op);
  } else if (loom_global_variable_isa(source_op)) {
    type_value = loom_global_variable_type(source_op);
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vm.global source must be a global constant or variable");
  }

  const int64_t ordinal = loom_vm_global_ordinal(resource_op);
  if (ordinal < 0 || ordinal > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM global ordinal is outside the u16 domain");
  }
  out_record->ordinal = (uint16_t)ordinal;
  out_record->type = loom_module_value_type(module, type_value);
  loom_vm_call_abi_bank_t abi_bank = LOOM_VM_CALL_ABI_BANK_NONE;
  if (!loom_vm_call_abi_try_classify_logical_type(module, out_record->type,
                                                  &abi_bank)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM global has an unsupported logical type");
  }
  switch (abi_bank) {
    case LOOM_VM_CALL_ABI_BANK_VALUE:
      out_record->bank = LOOM_VM_MODULE_GLOBAL_BANK_VALUE;
      break;
    case LOOM_VM_CALL_ABI_BANK_REF:
      out_record->bank = LOOM_VM_MODULE_GLOBAL_BANK_REF;
      break;
    case LOOM_VM_CALL_ABI_BANK_FUNCTION:
      out_record->bank = LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM global has no physical storage bank");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_rodata_layout_resolve(
    const loom_module_t* module, const loom_op_t* resource_op,
    uint16_t* out_ordinal, loom_vm_module_rodata_layout_t* out_layout) {
  *out_ordinal = 0;
  *out_layout = (loom_vm_module_rodata_layout_t){0};
  const loom_op_t* source_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_resolve_source(
      module, loom_vm_rodata_source(resource_op), &source_op));
  if (!loom_global_rodata_def_isa(source_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vm.rodata source must be a defined read-only data symbol");
  }

  const int64_t ordinal = loom_vm_rodata_ordinal(resource_op);
  if (ordinal < 0 || ordinal > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM rodata ordinal is outside the u16 domain");
  }
  uint64_t minimum_alignment = 1;
  const loom_attribute_t alignment_attr = loom_op_const_attrs(
      source_op)[loom_global_rodata_def_alignment_ATTR_INDEX];
  if (!loom_attr_is_absent(alignment_attr)) {
    minimum_alignment = (uint64_t)loom_attr_as_i64(alignment_attr);
  }
  if (!iree_is_power_of_two_uint64(minimum_alignment) ||
      minimum_alignment > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM rodata alignment must be a positive power-of-two u32");
  }

  *out_ordinal = (uint16_t)ordinal;
  *out_layout = (loom_vm_module_rodata_layout_t){
      .contents = loom_global_rodata_def_contents(source_op),
      .minimum_alignment = (uint32_t)minimum_alignment,
  };
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_layout_count(
    const loom_module_t* module, loom_vm_module_resource_counts_t* out_counts) {
  *out_counts = (loom_vm_module_resource_counts_t){0};
  const loom_op_t* op = NULL;
  loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
    if (loom_vm_constant_isa(op)) {
      const int64_t ordinal = loom_vm_constant_ordinal(op);
      if (ordinal < 0 || ordinal > UINT16_MAX) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM constant-pool ordinal is outside the u16 domain");
      }
      if (out_counts->constants == 65536u) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM constant-pool count exceeds the u16 ordinal domain");
      }
      ++out_counts->constants;
    } else if (loom_vm_global_isa(op)) {
      loom_vm_module_global_record_t record;
      IREE_RETURN_IF_ERROR(
          loom_vm_module_global_record_resolve(module, op, &record));
      if (out_counts->globals[record.bank] == 65536u) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM global count exceeds the u16 ordinal domain");
      }
      ++out_counts->globals[record.bank];
      if (record.is_immutable) {
        ++out_counts->immutable_globals[record.bank];
      }
    } else if (loom_vm_rodata_isa(op)) {
      uint16_t ordinal = 0;
      loom_vm_module_rodata_layout_t rodata;
      IREE_RETURN_IF_ERROR(
          loom_vm_module_rodata_layout_resolve(module, op, &ordinal, &rodata));
      if (out_counts->rodata == 65536u) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM rodata count exceeds the u16 ordinal domain");
      }
      ++out_counts->rodata;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_layout_allocate(
    const loom_vm_module_resource_counts_t* counts,
    iree_arena_allocator_t* arena,
    loom_vm_module_resource_presence_t* out_presence,
    loom_vm_module_resource_layout_t* out_layout) {
  *out_presence = (loom_vm_module_resource_presence_t){0};
  *out_layout = (loom_vm_module_resource_layout_t){
      .constant_count = counts->constants,
      .value_global_count = counts->globals[LOOM_VM_MODULE_GLOBAL_BANK_VALUE],
      .immutable_value_global_count =
          counts->immutable_globals[LOOM_VM_MODULE_GLOBAL_BANK_VALUE],
      .ref_global_count = counts->globals[LOOM_VM_MODULE_GLOBAL_BANK_REF],
      .immutable_ref_global_count =
          counts->immutable_globals[LOOM_VM_MODULE_GLOBAL_BANK_REF],
      .function_global_count =
          counts->globals[LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION],
      .immutable_function_global_count =
          counts->immutable_globals[LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION],
      .rodata_count = counts->rodata,
      .rodata_section_alignment = IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT,
  };
  if (out_layout->constant_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_layout->constant_count, sizeof(*out_layout->constant_cells),
        (void**)&out_layout->constant_cells));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_layout->constant_count, sizeof(*out_presence->constants),
        (void**)&out_presence->constants));
    memset(out_presence->constants, 0,
           out_layout->constant_count * sizeof(*out_presence->constants));
  }
  if (out_layout->value_global_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_layout->value_global_count,
                                  sizeof(*out_presence->value_globals),
                                  (void**)&out_presence->value_globals));
    memset(
        out_presence->value_globals, 0,
        out_layout->value_global_count * sizeof(*out_presence->value_globals));
  }
  if (out_layout->ref_global_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_layout->ref_global_count,
                                  sizeof(*out_layout->ref_global_types),
                                  (void**)&out_layout->ref_global_types));
    memset(
        out_layout->ref_global_types, 0,
        out_layout->ref_global_count * sizeof(*out_layout->ref_global_types));
  }
  if (out_layout->function_global_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_layout->function_global_count,
                                  sizeof(*out_layout->function_global_types),
                                  (void**)&out_layout->function_global_types));
    memset(out_layout->function_global_types, 0,
           out_layout->function_global_count *
               sizeof(*out_layout->function_global_types));
  }
  if (out_layout->rodata_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_layout->rodata_count, sizeof(*out_layout->rodata),
        (void**)&out_layout->rodata));
    memset(out_layout->rodata, 0,
           out_layout->rodata_count * sizeof(*out_layout->rodata));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_resource_layout_insert_global(
    const loom_vm_module_global_record_t* record, uint8_t* value_present,
    loom_vm_module_resource_layout_t* layout) {
  uint32_t global_count = 0;
  uint32_t immutable_count = 0;
  switch (record->bank) {
    case LOOM_VM_MODULE_GLOBAL_BANK_VALUE:
      global_count = layout->value_global_count;
      immutable_count = layout->immutable_value_global_count;
      break;
    case LOOM_VM_MODULE_GLOBAL_BANK_REF:
      global_count = layout->ref_global_count;
      immutable_count = layout->immutable_ref_global_count;
      break;
    case LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION:
      global_count = layout->function_global_count;
      immutable_count = layout->immutable_function_global_count;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("valid VM global bank");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (record->ordinal >= global_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM global ordinals must densely cover their physical bank");
  }
  if (record->is_immutable != (record->ordinal < immutable_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VM immutable globals must form a dense physical prefix");
  }

  bool occupied = false;
  switch (record->bank) {
    case LOOM_VM_MODULE_GLOBAL_BANK_VALUE:
      occupied = value_present[record->ordinal] != 0;
      if (!occupied) value_present[record->ordinal] = 1;
      break;
    case LOOM_VM_MODULE_GLOBAL_BANK_REF:
      occupied = loom_type_kind(layout->ref_global_types[record->ordinal]) !=
                 LOOM_TYPE_NONE;
      if (!occupied) layout->ref_global_types[record->ordinal] = record->type;
      break;
    case LOOM_VM_MODULE_GLOBAL_BANK_FUNCTION:
      occupied =
          loom_type_kind(layout->function_global_types[record->ordinal]) !=
          LOOM_TYPE_NONE;
      if (!occupied) {
        layout->function_global_types[record->ordinal] = record->type;
      }
      break;
    default:
      IREE_ASSERT_UNREACHABLE("valid VM global bank");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (occupied) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM global ordinal is duplicated");
  }
  return iree_ok_status();
}

iree_status_t loom_vm_module_resource_layout_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_vm_module_resource_layout_t* out_layout) {
  loom_vm_module_resource_counts_t counts;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_layout_count(module, &counts));
  loom_vm_module_resource_presence_t presence;
  IREE_RETURN_IF_ERROR(loom_vm_module_resource_layout_allocate(
      &counts, arena, &presence, out_layout));

  const loom_op_t* op = NULL;
  loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
    if (loom_vm_constant_isa(op)) {
      const uint16_t ordinal = (uint16_t)loom_vm_constant_ordinal(op);
      if (ordinal >= out_layout->constant_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM constant-pool ordinals must densely cover the physical table");
      }
      if (presence.constants[ordinal] != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM constant-pool ordinal is duplicated");
      }
      presence.constants[ordinal] = 1;
      out_layout->constant_cells[ordinal] = (uint64_t)loom_vm_constant_bits(op);
    } else if (loom_vm_global_isa(op)) {
      loom_vm_module_global_record_t record;
      IREE_RETURN_IF_ERROR(
          loom_vm_module_global_record_resolve(module, op, &record));
      IREE_RETURN_IF_ERROR(loom_vm_module_resource_layout_insert_global(
          &record, presence.value_globals, out_layout));
    } else if (loom_vm_rodata_isa(op)) {
      uint16_t ordinal = 0;
      loom_vm_module_rodata_layout_t rodata;
      IREE_RETURN_IF_ERROR(
          loom_vm_module_rodata_layout_resolve(module, op, &ordinal, &rodata));
      if (ordinal >= out_layout->rodata_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM rodata ordinals must densely cover the physical table");
      }
      if (out_layout->rodata[ordinal].minimum_alignment != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "VM rodata ordinal is duplicated");
      }
      out_layout->rodata[ordinal] = rodata;
      out_layout->rodata_section_alignment = iree_max(
          out_layout->rodata_section_alignment, rodata.minimum_alignment);
    }
  }
  return iree_ok_status();
}
