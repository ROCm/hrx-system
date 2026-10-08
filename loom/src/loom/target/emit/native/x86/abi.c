// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/abi.h"

#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/registers.h"

// SysV AMD64 integer-class arguments use RDI, RSI, RDX, RCX, R8, and R9.
static const uint8_t kSysvIntegerArgumentRegisters[] = {7, 6, 2, 1, 8, 9};

// Module-internal multiple results use the platform's ordinary primitive
// result prefixes before overflowing to caller-owned storage.
static const uint8_t kInternalIntegerResultRegisters[] = {0, 2};

static const loom_low_call_clobber_t kSysvGprClobbers[] = {
    {LOOM_X86_REGISTER_CLASS_GPR64, 0, 3},
    {LOOM_X86_REGISTER_CLASS_GPR64, 6, 6},
};

iree_status_t loom_x86_module_abi_initialize(
    iree_host_size_t symbol_count, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, loom_x86_module_abi_t* out_module_abi) {
  *out_module_abi = (loom_x86_module_abi_t){
      .function_count = function_count,
      .symbol_count = symbol_count,
  };
  if (function_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, function_count, sizeof(*out_module_abi->functions),
        (void**)&out_module_abi->functions));
    memset(out_module_abi->functions, 0,
           function_count * sizeof(*out_module_abi->functions));
  }
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, symbol_count, sizeof(*out_module_abi->functions_by_symbol),
        (void**)&out_module_abi->functions_by_symbol));
    memset(out_module_abi->functions_by_symbol, 0,
           symbol_count * sizeof(*out_module_abi->functions_by_symbol));
  }
  return iree_ok_status();
}

void loom_x86_module_abi_bind(loom_x86_module_abi_t* module_abi,
                              iree_host_size_t function_index,
                              loom_symbol_id_t symbol_id) {
  IREE_ASSERT_LT(function_index, module_abi->function_count);
  IREE_ASSERT_LT(symbol_id, module_abi->symbol_count);
  IREE_ASSERT_EQ(module_abi->functions_by_symbol[symbol_id], NULL);
  module_abi->functions_by_symbol[symbol_id] =
      &module_abi->functions[function_index];
}

const loom_x86_function_abi_t* loom_x86_module_abi_lookup(
    const loom_x86_module_abi_t* module_abi, loom_symbol_ref_t function) {
  if (function.module_id != 0 ||
      function.symbol_id >= module_abi->symbol_count) {
    return NULL;
  }
  return module_abi->functions_by_symbol[function.symbol_id];
}

iree_status_t loom_x86_function_call_contract_validate(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_x86_module_abi_t* module_abi = user_data;
  if (loom_x86_module_abi_lookup(module_abi, callee) == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 callee has no native SysV ABI plan");
  }
  return iree_ok_status();
}

const loom_low_call_contract_t* loom_x86_function_call_contract(
    void* user_data, loom_symbol_ref_t callee) {
  const loom_x86_function_abi_t* function_abi =
      loom_x86_module_abi_lookup(user_data, callee);
  IREE_ASSERT_NE(function_abi, NULL);
  return &function_abi->call_contract;
}

loom_low_call_clobber_list_t loom_x86_function_common_call_clobbers(
    void* user_data, const loom_low_descriptor_set_t* descriptor_set) {
  (void)user_data;
  static const loom_low_call_clobber_t clobbers[] = {
      {LOOM_X86_REGISTER_CLASS_XMM, 0, 16},
      {LOOM_X86_REGISTER_CLASS_YMM, 0, 16},
      {LOOM_X86_REGISTER_CLASS_ZMM, 0, 32},
      {LOOM_X86_REGISTER_CLASS_K, 0, 8},
  };
  uint16_t register_class = LOOM_X86_REGISTER_CLASS_ZMM;
  if (register_class >= descriptor_set->reg_class_count ||
      descriptor_set->reg_classes[register_class].allocatable_count == 0) {
    register_class = LOOM_X86_REGISTER_CLASS_YMM;
  }
  if (register_class >= descriptor_set->reg_class_count ||
      descriptor_set->reg_classes[register_class].allocatable_count == 0) {
    register_class = LOOM_X86_REGISTER_CLASS_XMM;
  }
  if (register_class >= descriptor_set->reg_class_count ||
      descriptor_set->reg_classes[register_class].allocatable_count == 0) {
    return (loom_low_call_clobber_list_t){0};
  }
  const iree_host_size_t clobber_index =
      register_class - LOOM_X86_REGISTER_CLASS_XMM;
  IREE_ASSERT_EQ(descriptor_set->reg_classes[register_class].allocatable_count,
                 clobbers[clobber_index].count);
  const bool has_mask_registers =
      LOOM_X86_REGISTER_CLASS_K < descriptor_set->reg_class_count &&
      descriptor_set->reg_classes[LOOM_X86_REGISTER_CLASS_K]
              .allocatable_count != 0;
  if (has_mask_registers) {
    IREE_ASSERT_EQ(register_class, LOOM_X86_REGISTER_CLASS_ZMM);
    IREE_ASSERT_EQ(descriptor_set->reg_classes[LOOM_X86_REGISTER_CLASS_K]
                       .allocatable_count,
                   clobbers[3].count);
  }
  return (loom_low_call_clobber_list_t){
      .values = &clobbers[clobber_index],
      .count = has_mask_registers ? 2 : 1,
  };
}

static bool loom_x86_abi_carrier_class(loom_type_t carrier,
                                       uint16_t* out_register_class) {
  if (!loom_low_type_is_register(carrier) ||
      loom_low_register_type_unit_count(carrier) != 1) {
    return false;
  }
  const uint16_t register_class = loom_low_register_type_class_id(carrier);
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_GPR32:
    case LOOM_X86_REGISTER_CLASS_GPR64:
    case LOOM_X86_REGISTER_CLASS_XMM:
    case LOOM_X86_REGISTER_CLASS_YMM:
    case LOOM_X86_REGISTER_CLASS_ZMM:
      *out_register_class = register_class;
      return true;
    default:
      return false;
  }
}

static bool loom_x86_abi_classify_value(
    loom_type_t logical_type, loom_type_t carrier,
    loom_x86_call_abi_classification_t* out_classification) {
  uint16_t carrier_class = 0;
  if (!loom_x86_abi_carrier_class(carrier, &carrier_class)) {
    return false;
  }
  if (loom_type_equal(logical_type, loom_type_none())) {
    const uint16_t byte_length =
        loom_x86_call_abi_register_byte_length(carrier_class);
    *out_classification = (loom_x86_call_abi_classification_t){
        .abi_class = carrier_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                             carrier_class == LOOM_X86_REGISTER_CLASS_GPR64
                         ? LOOM_X86_CALL_ABI_CLASS_INTEGER
                         : LOOM_X86_CALL_ABI_CLASS_SSE,
        .carrier_register_class = carrier_class,
        .boundary_register_class = carrier_class,
        .byte_length = byte_length,
        .byte_alignment = (uint8_t)byte_length,
    };
    return byte_length != 0;
  }
  if (!loom_x86_call_abi_classify_source_type(logical_type,
                                              out_classification)) {
    return false;
  }
  return out_classification->carrier_register_class == carrier_class;
}

static bool loom_x86_abi_signature(
    const loom_module_t* module, loom_func_like_t function,
    const loom_func_type_data_t** out_signature) {
  *out_signature = NULL;
  // The HAL task adapter has a fixed physical platform signature. Its ABI
  // layout describes logical dispatch parameters and remains owned by task
  // metadata construction; it is not the native function boundary.
  if (loom_func_like_abi(function) == LOOM_TARGET_ABI_HAL_KERNEL) {
    return true;
  }
  const loom_named_attr_slice_t layout =
      loom_low_func_decl_isa(function.op)
          ? loom_low_func_decl_abi_layout(function.op)
          : loom_low_func_def_abi_layout(function.op);
  if (layout.count == 0) {
    return true;
  }
  if (layout.count != 1 || layout.entries[0].value.kind != LOOM_ATTR_TYPE ||
      !iree_string_view_equal(
          loom_string_table_get(&module->strings, layout.entries[0].name_id),
          IREE_SV("signature"))) {
    return false;
  }
  const loom_type_t signature =
      loom_type_table_get(&module->types, layout.entries[0].value.type_id);
  if (!loom_type_is_function(signature)) {
    return false;
  }
  const loom_func_type_data_t* data = loom_type_func_data(signature);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(function, &argument_count);
  if (data->arg_count != argument_count ||
      data->result_count != function.op->result_count) {
    return false;
  }
  *out_signature = data;
  return true;
}

static bool loom_x86_abi_register_available(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t register_class,
    uint32_t location) {
  return register_class < descriptor_set->reg_class_count &&
         location <
             descriptor_set->reg_classes[register_class].allocatable_count;
}

static bool loom_x86_abi_has_upper_vector_state(uint16_t register_class,
                                                uint32_t location) {
  return (register_class == LOOM_X86_REGISTER_CLASS_YMM ||
          register_class == LOOM_X86_REGISTER_CLASS_ZMM) &&
         location < 16;
}

static iree_status_t loom_x86_abi_allocate_values(
    iree_host_size_t count, iree_arena_allocator_t* arena,
    loom_low_allocation_abi_location_t** out_locations,
    loom_x86_abi_value_t** out_values) {
  *out_locations = NULL;
  *out_values = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(**out_locations), (void**)out_locations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(**out_values), (void**)out_values));
  memset(*out_locations, 0, count * sizeof(**out_locations));
  memset(*out_values, 0, count * sizeof(**out_values));
  return iree_ok_status();
}

iree_status_t loom_x86_function_abi_prepare(
    const loom_module_t* module, loom_func_like_t function,
    const loom_low_resolved_target_t* resolved_target,
    iree_arena_allocator_t* arena, bool* out_supported,
    iree_string_view_t* out_constraint, loom_x86_function_abi_t* out_abi) {
  *out_supported = false;
  *out_constraint = iree_string_view_empty();
  *out_abi = (loom_x86_function_abi_t){.target = *resolved_target};
  const loom_low_descriptor_set_t* descriptor_set =
      resolved_target->descriptor_set;

  const loom_func_type_data_t* signature = NULL;
  if (!loom_x86_abi_signature(module, function, &signature)) {
    *out_constraint = IREE_SV("native x86 requires one valid ABI signature");
    return iree_ok_status();
  }
  if (function.op->result_count > 1 &&
      (!loom_func_like_is_module_internal(function) ||
       function.op->region_count == 0)) {
    *out_constraint = IREE_SV(
        "native x86 public and imported functions require one platform result");
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_id_t* result_ids = loom_op_const_results(function.op);
  loom_low_allocation_abi_location_t* argument_locations = NULL;
  loom_low_allocation_abi_location_t* result_locations = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_abi_allocate_values(
      argument_count, arena, &argument_locations, &out_abi->arguments));
  IREE_RETURN_IF_ERROR(loom_x86_abi_allocate_values(
      function.op->result_count, arena, &result_locations, &out_abi->results));

  uint16_t register_counts[2] = {0, 0};
  uint64_t stack_bytes = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t logical_type =
        signature ? signature->types[i] : loom_type_none();
    const loom_type_t carrier = loom_module_value_type(module, argument_ids[i]);
    loom_x86_call_abi_classification_t classification;
    if (!loom_x86_abi_classify_value(logical_type, carrier, &classification)) {
      *out_constraint =
          IREE_SV("native x86 argument logical type and Low carrier disagree");
      return iree_ok_status();
    }
    loom_x86_abi_value_t* value = &out_abi->arguments[i];
    *value = (loom_x86_abi_value_t){
        .stack_offset = UINT32_MAX,
        .byte_length = classification.byte_length,
        .byte_alignment = classification.byte_alignment,
        .action = classification.action,
    };
    const uint16_t register_index = register_counts[classification.abi_class]++;
    const uint16_t register_limit =
        classification.abi_class == LOOM_X86_CALL_ABI_CLASS_INTEGER
            ? IREE_ARRAYSIZE(kSysvIntegerArgumentRegisters)
            : 8;
    if (register_index < register_limit) {
      const uint32_t location =
          classification.abi_class == LOOM_X86_CALL_ABI_CLASS_INTEGER
              ? kSysvIntegerArgumentRegisters[register_index]
              : register_index;
      if (!loom_x86_abi_register_available(
              descriptor_set, classification.boundary_register_class,
              location)) {
        *out_constraint = IREE_SV(
            "native x86 ABI register class is unavailable in this profile");
        return iree_ok_status();
      }
      argument_locations[i] = (loom_low_allocation_abi_location_t){
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .descriptor_reg_class_id = classification.boundary_register_class,
          .location_base = location,
      };
      out_abi->has_upper_vector_register_argument |=
          loom_x86_abi_has_upper_vector_state(
              classification.boundary_register_class, location);
    } else {
      const uint32_t alignment = iree_max(8u, value->byte_alignment);
      stack_bytes = iree_host_align(stack_bytes, alignment);
      if (stack_bytes > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "x86 ABI argument area exceeds 32 bits");
      }
      value->stack_offset = (uint32_t)stack_bytes;
      stack_bytes += iree_host_align(value->byte_length, 8u);
      out_abi->stack_argument_alignment = (uint8_t)iree_max(
          out_abi->stack_argument_alignment, iree_max(16u, alignment));
      out_abi->has_simd_stack_argument |=
          classification.abi_class == LOOM_X86_CALL_ABI_CLASS_SSE;
    }
  }
  if (stack_bytes > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 ABI argument area exceeds 32 bits");
  }
  out_abi->stack_argument_bytes = (uint32_t)stack_bytes;
  out_abi->call_storage_bytes = (uint32_t)stack_bytes;
  out_abi->call_storage_alignment = out_abi->stack_argument_alignment;

  uint16_t result_register_counts[2] = {0, 0};
  for (uint16_t i = 0; i < function.op->result_count; ++i) {
    const loom_type_t logical_type =
        signature ? signature->types[argument_count + i] : loom_type_none();
    const loom_type_t carrier = loom_module_value_type(module, result_ids[i]);
    loom_x86_call_abi_classification_t classification;
    if (!loom_x86_abi_classify_value(logical_type, carrier, &classification)) {
      *out_constraint =
          IREE_SV("native x86 result logical type and Low carrier disagree");
      return iree_ok_status();
    }
    loom_x86_abi_value_t* value = &out_abi->results[i];
    *value = (loom_x86_abi_value_t){
        .stack_offset = UINT32_MAX,
        .byte_length = classification.byte_length,
        .byte_alignment = classification.byte_alignment,
        .action = classification.action,
    };
    const uint16_t register_index =
        result_register_counts[classification.abi_class]++;
    const uint16_t register_limit = 2;
    if (register_index < register_limit) {
      const uint32_t location =
          classification.abi_class == LOOM_X86_CALL_ABI_CLASS_INTEGER
              ? kInternalIntegerResultRegisters[register_index]
              : register_index;
      if (!loom_x86_abi_register_available(
              descriptor_set, classification.boundary_register_class,
              location)) {
        *out_constraint = IREE_SV(
            "native x86 result register class is unavailable in this profile");
        return iree_ok_status();
      }
      result_locations[i] = (loom_low_allocation_abi_location_t){
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .descriptor_reg_class_id = classification.boundary_register_class,
          .location_base = location,
      };
      out_abi->has_upper_vector_result |= loom_x86_abi_has_upper_vector_state(
          classification.boundary_register_class, location);
      continue;
    }
    const uint32_t alignment = iree_max(8u, value->byte_alignment);
    stack_bytes = iree_host_align(stack_bytes, alignment);
    if (stack_bytes > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "x86 ABI call storage exceeds 32 bits");
    }
    value->stack_offset = (uint32_t)stack_bytes;
    stack_bytes += iree_host_align(value->byte_length, 8u);
    out_abi->call_storage_alignment = (uint8_t)iree_max(
        out_abi->call_storage_alignment, iree_max(16u, alignment));
    out_abi->has_indirect_results = true;
  }
  if (stack_bytes > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "x86 ABI call storage exceeds 32 bits");
  }
  out_abi->call_storage_bytes = (uint32_t)stack_bytes;

  out_abi->call_contract = (loom_low_call_contract_t){
      .arguments = argument_locations,
      .argument_count = argument_count,
      .results = result_locations,
      .result_count = function.op->result_count,
      .clobbers = {kSysvGprClobbers, IREE_ARRAYSIZE(kSysvGprClobbers)},
  };
  *out_supported = true;
  return iree_ok_status();
}
