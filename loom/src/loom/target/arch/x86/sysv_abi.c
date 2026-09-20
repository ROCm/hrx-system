// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/sysv_abi.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/registers.h"

static const uint32_t kLoomX86SysvIntegerArgumentRegisters[] = {
    LOOM_X86_SYSV_GPR_RDI, LOOM_X86_SYSV_GPR_RSI, LOOM_X86_SYSV_GPR_RDX,
    LOOM_X86_SYSV_GPR_RCX, LOOM_X86_SYSV_GPR_R8,  LOOM_X86_SYSV_GPR_R9,
};

static iree_status_t loom_x86_sysv_abi_type_is_scalar_gpr(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    bool* out_is_scalar_gpr) {
  *out_is_scalar_gpr = false;
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != 1 ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          descriptor_set->stable_id) {
    return iree_ok_status();
  }
  loom_x86_register_class_t register_class = 0;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_logical_register_class(
      descriptor_set, loom_low_register_type_class_id(type), &register_class));
  *out_is_scalar_gpr = register_class == LOOM_X86_REGISTER_CLASS_GPR32 ||
                       register_class == LOOM_X86_REGISTER_CLASS_GPR64;
  return iree_ok_status();
}

iree_status_t loom_x86_sysv_abi_layout_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    iree_arena_allocator_t* arena, loom_x86_sysv_abi_layout_t* out_layout,
    bool* out_supported) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_layout);
  IREE_ASSERT_ARGUMENT(out_supported);
  *out_layout = (loom_x86_sysv_abi_layout_t){0};
  *out_supported = false;

  if (argument_count > UINT16_MAX || result_count > 1 ||
      (argument_count != 0 && argument_types == NULL) ||
      (result_count != 0 && result_types == NULL)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    bool is_scalar_gpr = false;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_type_is_scalar_gpr(
        descriptor_set, argument_types[i], &is_scalar_gpr));
    if (!is_scalar_gpr) {
      return iree_ok_status();
    }
  }
  if (result_count != 0) {
    bool is_scalar_gpr = false;
    IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_type_is_scalar_gpr(
        descriptor_set, result_types[0], &is_scalar_gpr));
    if (!is_scalar_gpr) {
      return iree_ok_status();
    }
  }

  int64_t* argument_locations = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, argument_count, sizeof(*argument_locations),
        (void**)&argument_locations));
  }
  uint32_t stack_argument_bytes = 0;
  for (iree_host_size_t i = 0; i < argument_count; ++i) {
    if (i < IREE_ARRAYSIZE(kLoomX86SysvIntegerArgumentRegisters)) {
      argument_locations[i] = kLoomX86SysvIntegerArgumentRegisters[i];
    } else {
      argument_locations[i] =
          loom_x86_sysv_abi_stack_location(stack_argument_bytes);
      stack_argument_bytes += 8;
    }
  }

  int64_t* result_locations = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, result_count,
                                                   sizeof(*result_locations),
                                                   (void**)&result_locations));
    result_locations[0] = LOOM_X86_SYSV_GPR_RAX;
  }

  *out_layout = (loom_x86_sysv_abi_layout_t){
      .argument_locations = argument_locations,
      .argument_count = argument_count,
      .result_locations = result_locations,
      .result_count = result_count,
      .stack_argument_bytes = stack_argument_bytes,
  };
  *out_supported = true;
  return iree_ok_status();
}

typedef struct loom_x86_sysv_abi_attr_keys_t {
  loom_string_id_t argument_locations;
  loom_string_id_t calling_convention;
  loom_string_id_t result_locations;
  loom_string_id_t stack_argument_bytes;
} loom_x86_sysv_abi_attr_keys_t;

static iree_status_t loom_x86_sysv_abi_intern_attr_keys(
    loom_module_t* module, loom_x86_sysv_abi_attr_keys_t* out_keys) {
  *out_keys = (loom_x86_sysv_abi_attr_keys_t){0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("argument_locations"), &out_keys->argument_locations));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("calling_convention"), &out_keys->calling_convention));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("result_locations"), &out_keys->result_locations));
  return loom_module_intern_string(module, IREE_SV("stack_argument_bytes"),
                                   &out_keys->stack_argument_bytes);
}

iree_status_t loom_x86_sysv_abi_layout_make_attr(
    loom_module_t* module, const loom_x86_sysv_abi_layout_t* layout,
    loom_attribute_t* out_attr) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(out_attr);
  *out_attr = loom_attr_absent();
  if (layout->argument_count > UINT16_MAX ||
      layout->result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "x86 SysV ABI signature exceeds attribute limits");
  }

  loom_x86_sysv_abi_attr_keys_t keys = {0};
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_intern_attr_keys(module, &keys));
  loom_string_id_t convention = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("sysv_x86_64"), &convention));
  loom_named_attr_t entries[] = {
      {
          .name_id = keys.argument_locations,
          .value = loom_attr_i64_array((int64_t*)layout->argument_locations,
                                       (uint16_t)layout->argument_count),
      },
      {
          .name_id = keys.calling_convention,
          .value = loom_attr_string(convention),
      },
      {
          .name_id = keys.result_locations,
          .value = loom_attr_i64_array((int64_t*)layout->result_locations,
                                       (uint16_t)layout->result_count),
      },
      {
          .name_id = keys.stack_argument_bytes,
          .value = loom_attr_i64(layout->stack_argument_bytes),
      },
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      out_attr);
}

static loom_x86_sysv_abi_attr_keys_t loom_x86_sysv_abi_lookup_attr_keys(
    const loom_module_t* module) {
  return (loom_x86_sysv_abi_attr_keys_t){
      .argument_locations =
          loom_module_lookup_string(module, IREE_SV("argument_locations")),
      .calling_convention =
          loom_module_lookup_string(module, IREE_SV("calling_convention")),
      .result_locations =
          loom_module_lookup_string(module, IREE_SV("result_locations")),
      .stack_argument_bytes =
          loom_module_lookup_string(module, IREE_SV("stack_argument_bytes")),
  };
}

static const loom_attribute_t* loom_x86_sysv_abi_find_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key) {
  if (key == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == key) {
      return &attrs.entries[i].value;
    }
  }
  return NULL;
}

static iree_status_t loom_x86_sysv_abi_require_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key,
    loom_attr_kind_t expected_kind, const loom_attribute_t** out_attr) {
  *out_attr = loom_x86_sysv_abi_find_attr(attrs, key);
  if (*out_attr == NULL || (*out_attr)->kind != expected_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86 SysV ABI layout has a missing or malformed field");
  }
  return iree_ok_status();
}

iree_status_t loom_x86_sysv_abi_layout_parse(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_slice_t attrs, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, iree_arena_allocator_t* scratch_arena,
    loom_x86_sysv_abi_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_layout);
  *out_layout = (loom_x86_sysv_abi_layout_t){0};
  if (attrs.count != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 SysV ABI layout requires exactly four fields");
  }

  const loom_x86_sysv_abi_attr_keys_t keys =
      loom_x86_sysv_abi_lookup_attr_keys(module);
  const loom_attribute_t* argument_locations_attr = NULL;
  const loom_attribute_t* convention_attr = NULL;
  const loom_attribute_t* result_locations_attr = NULL;
  const loom_attribute_t* stack_argument_bytes_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_require_attr(
      attrs, keys.argument_locations, LOOM_ATTR_I64_ARRAY,
      &argument_locations_attr));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_require_attr(
      attrs, keys.calling_convention, LOOM_ATTR_STRING, &convention_attr));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_require_attr(
      attrs, keys.result_locations, LOOM_ATTR_I64_ARRAY,
      &result_locations_attr));
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_require_attr(
      attrs, keys.stack_argument_bytes, LOOM_ATTR_I64,
      &stack_argument_bytes_attr));

  const loom_string_id_t convention_id =
      loom_attr_as_string_id(*convention_attr);
  if (convention_id >= module->strings.count ||
      !iree_string_view_equal(module->strings.entries[convention_id],
                              IREE_SV("sysv_x86_64"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 ABI layout has an unsupported convention");
  }
  if (argument_locations_attr->count != argument_count ||
      result_locations_attr->count != result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86 SysV ABI layout location counts do not match its signature");
  }
  const int64_t stack_argument_bytes =
      loom_attr_as_i64(*stack_argument_bytes_attr);
  if (stack_argument_bytes < 0 || stack_argument_bytes > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86 SysV ABI stack argument extent is outside the u32 range");
  }

  loom_x86_sysv_abi_layout_t expected = {0};
  bool supported = false;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_layout_build(
      descriptor_set, argument_types, argument_count, result_types,
      result_count, scratch_arena, &expected, &supported));
  if (!supported) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 SysV ABI does not yet support this target-low signature");
  }
  if ((argument_count != 0 &&
       memcmp(argument_locations_attr->i64_array, expected.argument_locations,
              argument_count * sizeof(*expected.argument_locations)) != 0) ||
      (result_count != 0 &&
       memcmp(result_locations_attr->i64_array, expected.result_locations,
              result_count * sizeof(*expected.result_locations)) != 0) ||
      (uint32_t)stack_argument_bytes != expected.stack_argument_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 SysV ABI layout does not match canonical "
                            "signature classification");
  }

  *out_layout = (loom_x86_sysv_abi_layout_t){
      .argument_locations = argument_locations_attr->i64_array,
      .argument_count = argument_count,
      .result_locations = result_locations_attr->i64_array,
      .result_count = result_count,
      .stack_argument_bytes = (uint32_t)stack_argument_bytes,
  };
  return iree_ok_status();
}

bool loom_x86_sysv_gpr_is_caller_saved(uint32_t physical_register) {
  switch (physical_register) {
    case LOOM_X86_SYSV_GPR_RAX:
    case LOOM_X86_SYSV_GPR_RCX:
    case LOOM_X86_SYSV_GPR_RDX:
    case LOOM_X86_SYSV_GPR_RSI:
    case LOOM_X86_SYSV_GPR_RDI:
    case LOOM_X86_SYSV_GPR_R8:
    case LOOM_X86_SYSV_GPR_R9:
    case LOOM_X86_SYSV_GPR_R10:
    case LOOM_X86_SYSV_GPR_R11:
      return true;
    default:
      return false;
  }
}

bool loom_x86_sysv_gpr_is_callee_saved(uint32_t physical_register) {
  switch (physical_register) {
    case LOOM_X86_SYSV_GPR_RBX:
    case LOOM_X86_SYSV_GPR_RSP:
    case LOOM_X86_SYSV_GPR_RBP:
    case LOOM_X86_SYSV_GPR_R12:
    case LOOM_X86_SYSV_GPR_R13:
    case LOOM_X86_SYSV_GPR_R14:
    case LOOM_X86_SYSV_GPR_R15:
      return true;
    default:
      return false;
  }
}
