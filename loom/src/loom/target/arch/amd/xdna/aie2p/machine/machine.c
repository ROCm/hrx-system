// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

#include <inttypes.h>

#include "iree/base/internal/math.h"

// One physical register and its slices in the atomic-unit and subregister
// tables.
typedef struct loom_aie2p_physical_register_t {
  // Byte offset of the stable target name in the machine string table.
  uint16_t name_offset;
  // Byte offset of the native assembly spelling in the string table.
  uint16_t assembly_name_offset;
  // First occupied atomic-unit row.
  uint16_t atomic_unit_start;
  // First named subregister row.
  uint16_t subregister_start;
  // Direct hardware encoding used without an operand adapter.
  uint16_t hardware_encoding;
  // Number of occupied atomic-unit rows.
  uint8_t atomic_unit_count;
  // Number of named subregister rows.
  uint8_t subregister_count;
} loom_aie2p_physical_register_t;

// One physical subregister and its named projection index.
typedef struct loom_aie2p_subregister_t {
  // Descriptor-set physical-register ID of the projected register.
  uint16_t register_id;
  // Byte offset of the stable subregister-index name.
  uint16_t index_name_offset;
} loom_aie2p_subregister_t;

// One deduplicated register and spill layout.
typedef struct loom_aie2p_register_layout_t {
  // In-register value width in bits.
  uint16_t register_size_bits;
  // Required register alignment in bits.
  uint16_t alignment_bits;
  // Spill-slot width in bits.
  uint16_t spill_size_bits;
  // Required spill-slot alignment in bits.
  uint16_t spill_alignment_bits;
} loom_aie2p_register_layout_t;

// One deduplicated machine value-type sequence.
typedef struct loom_aie2p_value_type_group_t {
  // First stable value-type name offset.
  uint16_t value_type_start;
  // Number of accepted value-type names.
  uint8_t value_type_count;
  // Reserved for future group properties.
  uint8_t reserved;
} loom_aie2p_value_type_group_t;

// One ordered physical-register color domain.
typedef struct loom_aie2p_register_class_t {
  // Byte offset of the stable class name.
  uint16_t name_offset;
  // First physical-register candidate row.
  uint16_t candidate_start;
  // Deduplicated register and spill layout ID.
  uint8_t layout_id;
  // Deduplicated accepted value-type group ID.
  uint8_t value_type_group_id;
  // Number of ordered physical-register candidates.
  uint8_t candidate_count;
  // Allocation, scheduling, and pressure policy bits.
  uint8_t flags;
} loom_aie2p_register_class_t;

// One deduplicated operand-local register encoding map.
typedef struct loom_aie2p_register_encoding_map_t {
  // First sorted physical-register/value pair.
  uint16_t entry_start;
  // Number of physical-register/value pairs.
  uint8_t entry_count;
  // Reserved for future map properties.
  uint8_t reserved;
} loom_aie2p_register_encoding_map_t;

// One named register adapter selecting a complete encoding map.
typedef struct loom_aie2p_register_adapter_t {
  // Byte offset of the stable adapter name.
  uint16_t name_offset;
  // Register class defining the mapped physical-register domain.
  uint16_t register_class_id;
  // Deduplicated effective architectural encoding-map ID.
  uint8_t encoding_map_id;
  // Reserved for future adapter properties.
  uint8_t reserved;
} loom_aie2p_register_adapter_t;

// One scaled immediate domain.
typedef struct loom_aie2p_immediate_t {
  // Required semantic value granularity.
  uint32_t step;
  // Byte offset of the stable immediate-domain name.
  uint16_t name_offset;
  // Semantic width including fixed and implicit bits.
  uint8_t semantic_width_bits;
  // Width physically carried by the instruction field.
  uint8_t encoded_width_bits;
  // Signed, negative-only, and symbol-reference behavior.
  uint8_t flags;
  // Reserved for future immediate-domain properties.
  uint8_t reserved[3];
} loom_aie2p_immediate_t;

// One deduplicated explicit operand sequence, with definitions first.
typedef struct loom_aie2p_operand_list_t {
  // First explicit machine-operand row.
  uint16_t operand_start;
  // Number of explicit definition operands.
  uint8_t output_count;
  // Number of explicit use operands.
  uint8_t input_count;
} loom_aie2p_operand_list_t;

// One explicit operand. The upper two type bits carry the operand kind.
typedef struct loom_aie2p_machine_operand_t {
  // Byte offset of the stable operand field name.
  uint16_t name_offset;
  // Low 14 bits select a type; high two bits select its domain.
  uint16_t type_and_kind;
} loom_aie2p_machine_operand_t;

// One deduplicated list of physical-register IDs.
typedef struct loom_aie2p_register_list_t {
  // First physical-register ID row.
  uint16_t register_start;
  // Number of physical-register IDs.
  uint8_t register_count;
  // Reserved for future list properties.
  uint8_t reserved;
} loom_aie2p_register_list_t;

// One definition/use equality constraint encoded as operand ordinals.
typedef struct loom_aie2p_machine_tie_t {
  // Explicit definition operand ordinal.
  uint8_t definition_ordinal;
  // Explicit use operand ordinal.
  uint8_t use_ordinal;
} loom_aie2p_machine_tie_t;

// One deduplicated tie sequence.
typedef struct loom_aie2p_tie_list_t {
  // First definition/use tie row.
  uint16_t tie_start;
  // Number of definition/use ties.
  uint8_t tie_count;
  // Reserved for future list properties.
  uint8_t reserved;
} loom_aie2p_tie_list_t;

// One concrete physical instruction form.
typedef struct loom_aie2p_machine_form_t {
  // Byte offset of the stable instruction name.
  uint16_t name_offset;
  // Byte offset of the native assembly template.
  uint16_t assembly_offset;
  // Deduplicated explicit operand-list ID.
  uint16_t operand_list_id;
  // Machine selection and side-effect properties.
  uint16_t property_flags;
  // Deduplicated implicit-definition register-list ID.
  uint8_t implicit_def_list_id;
  // Deduplicated implicit-use register-list ID.
  uint8_t implicit_use_list_id;
  // Deduplicated explicit definition/use tie-list ID.
  uint8_t tie_list_id;
  // Normalized physical control-flow behavior.
  uint8_t control_flow_kind;
} loom_aie2p_machine_form_t;

#include "loom/target/arch/amd/xdna/aie2p/machine/machine_tables.inl"

static_assert(sizeof(loom_aie2p_physical_register_t) == 12,
              "AIE2P physical registers must remain compact");
static_assert(sizeof(loom_aie2p_subregister_t) == 4,
              "AIE2P subregisters must remain compact");
static_assert(sizeof(loom_aie2p_register_layout_t) == 8,
              "AIE2P register layouts must remain compact");
static_assert(sizeof(loom_aie2p_register_class_t) == 8,
              "AIE2P register classes must remain compact");
static_assert(sizeof(loom_aie2p_register_adapter_t) == 6,
              "AIE2P register adapters must remain compact");
static_assert(sizeof(loom_aie2p_immediate_t) == 12,
              "AIE2P immediate domains must remain compact");
static_assert(sizeof(loom_aie2p_machine_operand_t) == 4,
              "AIE2P machine operands must remain compact");
static_assert(sizeof(loom_aie2p_machine_form_t) == 12,
              "AIE2P machine forms must remain compact");
static_assert(
    sizeof(kLoomAie2pMachineStrings) + sizeof(kLoomAie2pAtomicUnitNameOffsets) +
            sizeof(kLoomAie2pPhysicalRegisters) +
            sizeof(kLoomAie2pPhysicalRegisterAtomicUnits) +
            sizeof(kLoomAie2pSubregisters) + sizeof(kLoomAie2pRegisterLayouts) +
            sizeof(kLoomAie2pValueTypeGroups) +
            sizeof(kLoomAie2pValueTypeNameOffsets) +
            sizeof(kLoomAie2pRegisterClasses) +
            sizeof(kLoomAie2pRegisterClassCandidates) +
            sizeof(kLoomAie2pRegisterEncodingMaps) +
            sizeof(kLoomAie2pRegisterEncodingMapRegisterIds) +
            sizeof(kLoomAie2pRegisterEncodingMapValues) +
            sizeof(kLoomAie2pRegisterAdapters) + sizeof(kLoomAie2pImmediates) +
            sizeof(kLoomAie2pOperandLists) + sizeof(kLoomAie2pMachineOperands) +
            sizeof(kLoomAie2pRegisterLists) +
            sizeof(kLoomAie2pRegisterListValues) + sizeof(kLoomAie2pTieLists) +
            sizeof(kLoomAie2pMachineTies) + sizeof(kLoomAie2pMachineForms) <=
        96 * 1024,
    "complete AIE2P machine tables must remain within 96 KiB");

static iree_string_view_t loom_aie2p_machine_string(uint16_t offset) {
  return iree_make_cstring_view(kLoomAie2pMachineStrings + offset);
}

static const loom_aie2p_physical_register_t*
loom_aie2p_machine_physical_register(
    loom_aie2p_physical_register_id_t register_id) {
  if (register_id >= IREE_ARRAYSIZE(kLoomAie2pPhysicalRegisters)) {
    return NULL;
  }
  return &kLoomAie2pPhysicalRegisters[register_id];
}

static const loom_aie2p_register_class_t* loom_aie2p_machine_register_class(
    loom_aie2p_register_class_id_t register_class_id) {
  if (register_class_id == LOOM_AIE2P_REGISTER_CLASS_ID_INVALID ||
      register_class_id >= IREE_ARRAYSIZE(kLoomAie2pRegisterClasses)) {
    return NULL;
  }
  return &kLoomAie2pRegisterClasses[register_class_id];
}

static const loom_aie2p_register_adapter_t* loom_aie2p_machine_register_adapter(
    loom_aie2p_register_adapter_id_t adapter_id) {
  if (adapter_id == LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT ||
      adapter_id >= IREE_ARRAYSIZE(kLoomAie2pRegisterAdapters)) {
    return NULL;
  }
  return &kLoomAie2pRegisterAdapters[adapter_id];
}

static const loom_aie2p_immediate_t* loom_aie2p_machine_immediate(
    loom_aie2p_immediate_id_t immediate_id) {
  if (immediate_id == LOOM_AIE2P_IMMEDIATE_ID_INVALID ||
      immediate_id >= IREE_ARRAYSIZE(kLoomAie2pImmediates)) {
    return NULL;
  }
  return &kLoomAie2pImmediates[immediate_id];
}

static const loom_aie2p_machine_form_t* loom_aie2p_machine_form(
    loom_aie2p_machine_form_id_t form_id) {
  if (form_id == LOOM_AIE2P_MACHINE_FORM_ID_INVALID ||
      form_id >= IREE_ARRAYSIZE(kLoomAie2pMachineForms)) {
    return NULL;
  }
  return &kLoomAie2pMachineForms[form_id];
}

static uint64_t loom_aie2p_machine_low_bit_mask(uint8_t bit_count) {
  return UINT64_MAX >> (64u - bit_count);
}

static int64_t loom_aie2p_machine_sign_extend(uint64_t value,
                                              uint8_t bit_count) {
  const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1u);
  return (int64_t)((value ^ sign_bit) - sign_bit);
}

iree_host_size_t loom_aie2p_machine_atomic_unit_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pAtomicUnitNameOffsets);
}

iree_host_size_t loom_aie2p_machine_physical_register_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pPhysicalRegisters);
}

iree_host_size_t loom_aie2p_machine_register_class_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pRegisterClasses) - 1;
}

iree_host_size_t loom_aie2p_machine_register_adapter_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pRegisterAdapters) - 1;
}

iree_host_size_t loom_aie2p_machine_immediate_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pImmediates) - 1;
}

iree_host_size_t loom_aie2p_machine_form_count(void) {
  return IREE_ARRAYSIZE(kLoomAie2pMachineForms) - 1;
}

iree_string_view_t loom_aie2p_machine_atomic_unit_name(
    loom_aie2p_atomic_unit_id_t atomic_unit) {
  if (atomic_unit >= IREE_ARRAYSIZE(kLoomAie2pAtomicUnitNameOffsets)) {
    return iree_string_view_empty();
  }
  return loom_aie2p_machine_string(
      kLoomAie2pAtomicUnitNameOffsets[atomic_unit]);
}

loom_aie2p_physical_register_id_t loom_aie2p_machine_find_physical_register(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomAie2pPhysicalRegisters);
       ++i) {
    if (iree_string_view_equal(
            name, loom_aie2p_machine_string(
                      kLoomAie2pPhysicalRegisters[i].name_offset))) {
      return (loom_aie2p_physical_register_id_t)i;
    }
  }
  return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
}

loom_aie2p_register_class_id_t loom_aie2p_machine_find_register_class(
    iree_string_view_t name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kLoomAie2pRegisterClasses);
       ++i) {
    if (iree_string_view_equal(name,
                               loom_aie2p_machine_string(
                                   kLoomAie2pRegisterClasses[i].name_offset))) {
      return (loom_aie2p_register_class_id_t)i;
    }
  }
  return LOOM_AIE2P_REGISTER_CLASS_ID_INVALID;
}

loom_aie2p_register_adapter_id_t loom_aie2p_machine_find_register_adapter(
    iree_string_view_t name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kLoomAie2pRegisterAdapters);
       ++i) {
    if (iree_string_view_equal(
            name, loom_aie2p_machine_string(
                      kLoomAie2pRegisterAdapters[i].name_offset))) {
      return (loom_aie2p_register_adapter_id_t)i;
    }
  }
  return LOOM_AIE2P_REGISTER_ADAPTER_ID_INVALID;
}

loom_aie2p_immediate_id_t loom_aie2p_machine_find_immediate(
    iree_string_view_t name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kLoomAie2pImmediates); ++i) {
    if (iree_string_view_equal(
            name,
            loom_aie2p_machine_string(kLoomAie2pImmediates[i].name_offset))) {
      return (loom_aie2p_immediate_id_t)i;
    }
  }
  return LOOM_AIE2P_IMMEDIATE_ID_INVALID;
}

loom_aie2p_machine_form_id_t loom_aie2p_machine_find_form(
    iree_string_view_t name) {
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(kLoomAie2pMachineForms);
       ++i) {
    if (iree_string_view_equal(
            name,
            loom_aie2p_machine_string(kLoomAie2pMachineForms[i].name_offset))) {
      return (loom_aie2p_machine_form_id_t)i;
    }
  }
  return LOOM_AIE2P_MACHINE_FORM_ID_INVALID;
}

bool loom_aie2p_machine_query_physical_register(
    loom_aie2p_physical_register_id_t register_id,
    loom_aie2p_physical_register_info_t* out_info) {
  const loom_aie2p_physical_register_t* register_row =
      loom_aie2p_machine_physical_register(register_id);
  if (!register_row || !out_info) return false;
  *out_info = (loom_aie2p_physical_register_info_t){
      .name = loom_aie2p_machine_string(register_row->name_offset),
      .assembly_name =
          loom_aie2p_machine_string(register_row->assembly_name_offset),
      .hardware_encoding = register_row->hardware_encoding,
      .atomic_unit_count = register_row->atomic_unit_count,
      .subregister_count = register_row->subregister_count,
  };
  return true;
}

loom_aie2p_atomic_unit_id_t loom_aie2p_machine_physical_register_atomic_unit(
    loom_aie2p_physical_register_id_t register_id, uint8_t ordinal) {
  const loom_aie2p_physical_register_t* register_row =
      loom_aie2p_machine_physical_register(register_id);
  if (!register_row || ordinal >= register_row->atomic_unit_count) {
    return LOOM_AIE2P_ATOMIC_UNIT_ID_INVALID;
  }
  return kLoomAie2pPhysicalRegisterAtomicUnits[register_row->atomic_unit_start +
                                               ordinal];
}

bool loom_aie2p_machine_query_subregister(
    loom_aie2p_physical_register_id_t register_id, uint8_t ordinal,
    loom_aie2p_subregister_info_t* out_info) {
  const loom_aie2p_physical_register_t* register_row =
      loom_aie2p_machine_physical_register(register_id);
  if (!register_row || ordinal >= register_row->subregister_count ||
      !out_info) {
    return false;
  }
  const loom_aie2p_subregister_t* subregister =
      &kLoomAie2pSubregisters[register_row->subregister_start + ordinal];
  *out_info = (loom_aie2p_subregister_info_t){
      .register_id = subregister->register_id,
      .index_name = loom_aie2p_machine_string(subregister->index_name_offset),
  };
  return true;
}

bool loom_aie2p_machine_query_register_class(
    loom_aie2p_register_class_id_t register_class_id,
    loom_aie2p_register_class_info_t* out_info) {
  const loom_aie2p_register_class_t* register_class =
      loom_aie2p_machine_register_class(register_class_id);
  if (!register_class || !out_info) return false;
  const loom_aie2p_register_layout_t* layout =
      &kLoomAie2pRegisterLayouts[register_class->layout_id];
  const loom_aie2p_value_type_group_t* value_types =
      &kLoomAie2pValueTypeGroups[register_class->value_type_group_id];
  *out_info = (loom_aie2p_register_class_info_t){
      .name = loom_aie2p_machine_string(register_class->name_offset),
      .register_size_bits = layout->register_size_bits,
      .alignment_bits = layout->alignment_bits,
      .spill_size_bits = layout->spill_size_bits,
      .spill_alignment_bits = layout->spill_alignment_bits,
      .flags = register_class->flags,
      .candidate_count = register_class->candidate_count,
      .value_type_count = value_types->value_type_count,
  };
  return true;
}

loom_aie2p_physical_register_id_t loom_aie2p_machine_register_class_candidate(
    loom_aie2p_register_class_id_t register_class_id, uint8_t ordinal) {
  const loom_aie2p_register_class_t* register_class =
      loom_aie2p_machine_register_class(register_class_id);
  if (!register_class || ordinal >= register_class->candidate_count) {
    return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  }
  return kLoomAie2pRegisterClassCandidates[register_class->candidate_start +
                                           ordinal];
}

iree_string_view_t loom_aie2p_machine_register_class_value_type(
    loom_aie2p_register_class_id_t register_class_id, uint8_t ordinal) {
  const loom_aie2p_register_class_t* register_class =
      loom_aie2p_machine_register_class(register_class_id);
  if (!register_class) return iree_string_view_empty();
  const loom_aie2p_value_type_group_t* value_types =
      &kLoomAie2pValueTypeGroups[register_class->value_type_group_id];
  if (ordinal >= value_types->value_type_count) return iree_string_view_empty();
  return loom_aie2p_machine_string(
      kLoomAie2pValueTypeNameOffsets[value_types->value_type_start + ordinal]);
}

bool loom_aie2p_machine_query_register_adapter(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_register_adapter_info_t* out_info) {
  const loom_aie2p_register_adapter_t* adapter =
      loom_aie2p_machine_register_adapter(adapter_id);
  if (!adapter || !out_info) return false;
  *out_info = (loom_aie2p_register_adapter_info_t){
      .name = loom_aie2p_machine_string(adapter->name_offset),
      .register_class_id = adapter->register_class_id,
  };
  return true;
}

static bool loom_aie2p_machine_try_adapt_register(
    const loom_aie2p_register_adapter_t* adapter,
    loom_aie2p_physical_register_id_t register_id, uint8_t* out_value) {
  const loom_aie2p_register_encoding_map_t* encoding_map =
      &kLoomAie2pRegisterEncodingMaps[adapter->encoding_map_id];
  iree_host_size_t low = encoding_map->entry_start;
  iree_host_size_t high = low + encoding_map->entry_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const loom_aie2p_physical_register_id_t candidate =
        kLoomAie2pRegisterEncodingMapRegisterIds[middle];
    if (candidate < register_id) {
      low = middle + 1;
    } else if (candidate > register_id) {
      high = middle;
    } else {
      *out_value = kLoomAie2pRegisterEncodingMapValues[middle];
      return true;
    }
  }
  return false;
}

bool loom_aie2p_machine_encode_register(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_physical_register_id_t register_id, uint8_t* out_value) {
  const loom_aie2p_physical_register_t* register_row =
      loom_aie2p_machine_physical_register(register_id);
  if (!register_row || !out_value) return false;
  if (adapter_id == LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT) {
    *out_value = (uint8_t)register_row->hardware_encoding;
    return true;
  }
  const loom_aie2p_register_adapter_t* adapter =
      loom_aie2p_machine_register_adapter(adapter_id);
  if (!adapter) return false;
  return loom_aie2p_machine_try_adapt_register(adapter, register_id, out_value);
}

uint8_t loom_aie2p_machine_adapt_allocated_register(
    loom_aie2p_register_adapter_id_t adapter_id,
    loom_aie2p_physical_register_id_t register_id) {
  const loom_aie2p_physical_register_t* register_row =
      &kLoomAie2pPhysicalRegisters[register_id];
  if (adapter_id == LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT) {
    return (uint8_t)register_row->hardware_encoding;
  }
  const loom_aie2p_register_adapter_t* adapter =
      &kLoomAie2pRegisterAdapters[adapter_id];
  uint8_t encoded_value = 0;
  const bool found = loom_aie2p_machine_try_adapt_register(adapter, register_id,
                                                           &encoded_value);
  IREE_ASSERT(found &&
              "allocated AIE2P register is absent from its generated adapter");
  return encoded_value;
}

bool loom_aie2p_machine_query_immediate(loom_aie2p_immediate_id_t immediate_id,
                                        loom_aie2p_immediate_info_t* out_info) {
  const loom_aie2p_immediate_t* immediate =
      loom_aie2p_machine_immediate(immediate_id);
  if (!immediate || !out_info) return false;
  *out_info = (loom_aie2p_immediate_info_t){
      .name = loom_aie2p_machine_string(immediate->name_offset),
      .semantic_width_bits = immediate->semantic_width_bits,
      .encoded_width_bits = immediate->encoded_width_bits,
      .step = immediate->step,
      .flags = immediate->flags,
  };
  return true;
}

iree_status_t loom_aie2p_machine_encode_immediate(
    loom_aie2p_immediate_id_t immediate_id, int64_t value,
    uint64_t* out_encoded_value) {
  const loom_aie2p_immediate_t* immediate =
      loom_aie2p_machine_immediate(immediate_id);
  if (!immediate || !out_encoded_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE2P immediate encoding request");
  }
  if (value % immediate->step != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P immediate is not aligned to its step");
  }
  const uint8_t fixed_zero_bits =
      (uint8_t)iree_math_count_trailing_zeros_u32(immediate->step);
  int64_t minimum = 0;
  int64_t maximum = 0;
  if (iree_any_bit_set(immediate->flags, LOOM_AIE2P_IMMEDIATE_FLAG_NEGATIVE)) {
    minimum =
        -(INT64_C(1) << (immediate->encoded_width_bits + fixed_zero_bits));
    maximum = -(int64_t)immediate->step;
  } else if (iree_any_bit_set(immediate->flags,
                              LOOM_AIE2P_IMMEDIATE_FLAG_SIGNED)) {
    const uint8_t value_bits = immediate->encoded_width_bits + fixed_zero_bits;
    minimum = -(INT64_C(1) << (value_bits - 1u));
    maximum = (INT64_C(1) << (value_bits - 1u)) - immediate->step;
  } else {
    minimum = 0;
    maximum =
        (INT64_C(1) << (immediate->encoded_width_bits + fixed_zero_bits)) -
        immediate->step;
  }
  if (value < minimum || value > maximum) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P immediate is outside its encoded domain");
  }
  *out_encoded_value =
      loom_aie2p_machine_encode_verified_immediate(immediate_id, value);
  return iree_ok_status();
}

uint64_t loom_aie2p_machine_encode_verified_immediate(
    loom_aie2p_immediate_id_t immediate_id, int64_t value) {
  const loom_aie2p_immediate_t* immediate = &kLoomAie2pImmediates[immediate_id];
  const int64_t scaled_value = value / immediate->step;
  return (uint64_t)scaled_value &
         loom_aie2p_machine_low_bit_mask(immediate->encoded_width_bits);
}

iree_status_t loom_aie2p_machine_decode_immediate(
    loom_aie2p_immediate_id_t immediate_id, uint64_t encoded_value,
    int64_t* out_value) {
  const loom_aie2p_immediate_t* immediate =
      loom_aie2p_machine_immediate(immediate_id);
  if (!immediate || !out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AIE2P immediate decoding request");
  }
  const uint64_t encoded_mask =
      loom_aie2p_machine_low_bit_mask(immediate->encoded_width_bits);
  if (encoded_value > encoded_mask) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "encoded AIE2P immediate exceeds its field width");
  }
  int64_t scaled_value = 0;
  if (iree_any_bit_set(immediate->flags, LOOM_AIE2P_IMMEDIATE_FLAG_NEGATIVE)) {
    scaled_value = loom_aie2p_machine_sign_extend(
        encoded_value | (UINT64_C(1) << immediate->encoded_width_bits),
        immediate->encoded_width_bits + 1u);
  } else if (iree_any_bit_set(immediate->flags,
                              LOOM_AIE2P_IMMEDIATE_FLAG_SIGNED)) {
    scaled_value = loom_aie2p_machine_sign_extend(
        encoded_value, immediate->encoded_width_bits);
  } else {
    scaled_value = (int64_t)encoded_value;
  }
  *out_value = scaled_value * immediate->step;
  return iree_ok_status();
}

bool loom_aie2p_machine_query_form(loom_aie2p_machine_form_id_t form_id,
                                   loom_aie2p_machine_form_info_t* out_info) {
  const loom_aie2p_machine_form_t* form = loom_aie2p_machine_form(form_id);
  if (!form || !out_info) return false;
  const loom_aie2p_operand_list_t* operands =
      &kLoomAie2pOperandLists[form->operand_list_id];
  const loom_aie2p_register_list_t* implicit_defs =
      &kLoomAie2pRegisterLists[form->implicit_def_list_id];
  const loom_aie2p_register_list_t* implicit_uses =
      &kLoomAie2pRegisterLists[form->implicit_use_list_id];
  const loom_aie2p_tie_list_t* ties = &kLoomAie2pTieLists[form->tie_list_id];
  *out_info = (loom_aie2p_machine_form_info_t){
      .name = loom_aie2p_machine_string(form->name_offset),
      .assembly = loom_aie2p_machine_string(form->assembly_offset),
      .property_flags = form->property_flags,
      .control_flow_kind =
          (loom_aie2p_control_flow_kind_t)form->control_flow_kind,
      .output_count = operands->output_count,
      .input_count = operands->input_count,
      .implicit_def_count = implicit_defs->register_count,
      .implicit_use_count = implicit_uses->register_count,
      .tie_count = ties->tie_count,
  };
  return true;
}

bool loom_aie2p_machine_query_form_operand(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal,
    loom_aie2p_machine_operand_info_t* out_info) {
  const loom_aie2p_machine_form_t* form = loom_aie2p_machine_form(form_id);
  if (!form || !out_info) return false;
  const loom_aie2p_operand_list_t* operand_list =
      &kLoomAie2pOperandLists[form->operand_list_id];
  if (ordinal >= operand_list->output_count + operand_list->input_count) {
    return false;
  }
  const loom_aie2p_machine_operand_t* operand =
      &kLoomAie2pMachineOperands[operand_list->operand_start + ordinal];
  *out_info = (loom_aie2p_machine_operand_info_t){
      .name = loom_aie2p_machine_string(operand->name_offset),
      .type_id = operand->type_and_kind & 0x3FFFu,
      .kind = (loom_aie2p_machine_operand_kind_t)(operand->type_and_kind >> 14),
  };
  return true;
}

static loom_aie2p_physical_register_id_t loom_aie2p_machine_form_register(
    loom_aie2p_machine_form_id_t form_id, uint8_t list_id, uint8_t ordinal) {
  if (!loom_aie2p_machine_form(form_id)) {
    return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  }
  const loom_aie2p_register_list_t* list = &kLoomAie2pRegisterLists[list_id];
  if (ordinal >= list->register_count) {
    return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  }
  return kLoomAie2pRegisterListValues[list->register_start + ordinal];
}

loom_aie2p_physical_register_id_t loom_aie2p_machine_form_implicit_def(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal) {
  const loom_aie2p_machine_form_t* form = loom_aie2p_machine_form(form_id);
  if (!form) return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  return loom_aie2p_machine_form_register(form_id, form->implicit_def_list_id,
                                          ordinal);
}

loom_aie2p_physical_register_id_t loom_aie2p_machine_form_implicit_use(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal) {
  const loom_aie2p_machine_form_t* form = loom_aie2p_machine_form(form_id);
  if (!form) return LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID;
  return loom_aie2p_machine_form_register(form_id, form->implicit_use_list_id,
                                          ordinal);
}

bool loom_aie2p_machine_query_form_tie(
    loom_aie2p_machine_form_id_t form_id, uint8_t ordinal,
    loom_aie2p_machine_tie_info_t* out_info) {
  const loom_aie2p_machine_form_t* form = loom_aie2p_machine_form(form_id);
  if (!form || !out_info) return false;
  const loom_aie2p_tie_list_t* tie_list =
      &kLoomAie2pTieLists[form->tie_list_id];
  if (ordinal >= tie_list->tie_count) return false;
  const loom_aie2p_machine_tie_t* tie =
      &kLoomAie2pMachineTies[tie_list->tie_start + ordinal];
  *out_info = (loom_aie2p_machine_tie_info_t){
      .definition_ordinal = tie->definition_ordinal,
      .use_ordinal = tie->use_ordinal,
  };
  return true;
}
