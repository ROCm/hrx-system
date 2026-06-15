// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/arch/x86/contracts/avx2.h"
#include "loom/target/arch/x86/contracts/avx2_lower_rules.h"
#include "loom/target/arch/x86/contracts/avx512.h"
#include "loom/target/arch/x86/contracts/avx512_lower_rules.h"
#include "loom/target/arch/x86/contracts/packed_dot.h"
#include "loom/target/arch/x86/contracts/packed_dot_lower_rules.h"
#include "loom/target/arch/x86/contracts/scalar.h"
#include "loom/target/arch/x86/contracts/scalar_lower_rules.h"
#include "loom/target/arch/x86/lower/internal.h"

static bool loom_x86_type_is_vector_16xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_16xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_16xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_x86_type_is_scalar_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_x86_type_is_scalar_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_x86_type_is_scalar_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

static bool loom_x86_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_x86_type_is_address_gpr64(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      return true;
    default:
      return false;
  }
}

static bool loom_x86_scalar_register_class_for_source_type(
    loom_type_t source_type, loom_x86_register_class_t* out_register_class) {
  if (loom_x86_type_is_address_gpr64(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_GPR64;
    return true;
  }
  if (loom_x86_type_is_scalar_i1(source_type) ||
      loom_x86_type_is_scalar_i32(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_GPR32;
    return true;
  }
  if (loom_x86_type_is_scalar_i64(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_GPR64;
    return true;
  }
  return false;
}

static bool loom_x86_avx2_register_class_for_source_type(
    loom_type_t source_type, loom_x86_register_class_t* out_register_class) {
  if (loom_x86_scalar_register_class_for_source_type(source_type,
                                                     out_register_class)) {
    return true;
  }
  if (loom_x86_type_is_scalar_f32(source_type) ||
      loom_x86_type_is_vector_4xi32(source_type) ||
      loom_x86_type_is_vector_4xf32(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_XMM;
    return true;
  }
  return false;
}

static bool loom_x86_avx512_register_class_for_source_type(
    loom_type_t source_type, loom_x86_register_class_t* out_register_class) {
  if (loom_x86_avx2_register_class_for_source_type(source_type,
                                                   out_register_class)) {
    return true;
  }
  if (loom_x86_type_is_vector_4xi1(source_type) ||
      loom_x86_type_is_vector_16xi1(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_K;
    return true;
  }
  if (loom_x86_type_is_vector_16xi32(source_type) ||
      loom_x86_type_is_vector_16xf32(source_type)) {
    *out_register_class = LOOM_X86_REGISTER_CLASS_ZMM;
    return true;
  }
  return false;
}

iree_status_t loom_x86_make_register_type(
    loom_low_lower_context_t* context, loom_x86_register_class_t register_class,
    loom_type_t* out_type) {
  uint16_t descriptor_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_x86_descriptor_set_register_class_id(
      loom_low_lower_context_descriptor_set(context), register_class,
      &descriptor_reg_class_id));
  return loom_low_lower_make_register_type(context, descriptor_reg_class_id, 1,
                                           out_type);
}

static iree_status_t loom_x86_map_avx512_type(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_type_t source_type,
                                              loom_type_t* out_low_type) {
  loom_x86_register_class_t register_class = 0;
  if (loom_x86_avx512_register_class_for_source_type(source_type,
                                                     &register_class)) {
    return loom_x86_make_register_type(context, register_class, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_x86_map_avx2_type(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_type_t source_type,
                                            loom_type_t* out_low_type) {
  loom_x86_register_class_t register_class = 0;
  if (loom_x86_avx2_register_class_for_source_type(source_type,
                                                   &register_class)) {
    return loom_x86_make_register_type(context, register_class, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_x86_map_scalar_type(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_type_t source_type,
                                              loom_type_t* out_low_type) {
  loom_x86_register_class_t register_class = 0;
  if (loom_x86_scalar_register_class_for_source_type(source_type,
                                                     &register_class)) {
    return loom_x86_make_register_type(context, register_class, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_x86_map_scalar_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_x86_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_GPR64, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_source_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_x86_map_scalar_type(user_data, context, source_function_op,
                                  source_type, &out_argument->abi_type);
}

static iree_status_t loom_x86_map_avx512_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_x86_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_GPR64, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_source_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_x86_map_avx512_type(user_data, context, source_function_op,
                                  source_type, &out_argument->abi_type);
}

static iree_status_t loom_x86_map_avx2_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_x86_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_GPR64, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_source_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_x86_map_avx2_type(user_data, context, source_function_op,
                                source_type, &out_argument->abi_type);
}

static iree_status_t loom_x86_map_avx512_packed_dot_type(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
    loom_type_t* out_low_type) {
  uint32_t vector_bit_width = 0;
  if (loom_x86_packed_dot_type_static_vector_bit_width(source_type,
                                                       &vector_bit_width)) {
    loom_x86_register_class_t register_class = 0;
    if (loom_x86_register_class_for_vector_bit_width(vector_bit_width,
                                                     &register_class)) {
      return loom_x86_make_register_type(context, register_class, out_low_type);
    }
  }
  return loom_x86_map_avx512_type(user_data, context, source_op, source_type,
                                  out_low_type);
}

static iree_status_t loom_x86_map_avx512_packed_dot_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_x86_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_GPR64, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_source_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_x86_map_avx512_packed_dot_type(user_data, context,
                                             source_function_op, source_type,
                                             &out_argument->abi_type);
}

static const loom_low_lower_rule_set_t* const kX86Avx512RuleSets[] = {
    &loom_x86_avx512_lower_rule_set,
};

static const loom_low_lower_rule_set_t* const kX86Avx2RuleSets[] = {
    &loom_x86_avx2_lower_rule_set,
};

static const loom_low_lower_rule_set_t* const kX86ScalarRuleSets[] = {
    &loom_x86_scalar_lower_rule_set,
};

static const loom_low_lower_rule_set_t* const kX86PackedDotRuleSets[] = {
    &loom_x86_packed_dot_lower_rule_set,
};

static const loom_low_lower_rule_set_t* const kX86Avx512PackedDotRuleSets[] = {
    &loom_x86_avx512_lower_rule_set,
    &loom_x86_packed_dot_lower_rule_set,
};

static const loom_target_contract_binding_t kX86Avx512ContractBindings[] = {
    {&loom_x86_avx512_contract_fragment, 0},
};

static const loom_target_contract_binding_t kX86Avx2ContractBindings[] = {
    {&loom_x86_avx2_contract_fragment, 0},
};

static const loom_target_contract_binding_t kX86ScalarContractBindings[] = {
    {&loom_x86_scalar_contract_fragment, 0},
};

static const loom_target_contract_binding_t kX86PackedDotContractBindings[] = {
    {&loom_x86_packed_dot_contract_fragment, 0},
};

static const loom_target_contract_binding_t
    kX86Avx512PackedDotContractBindings[] = {
        {&loom_x86_avx512_contract_fragment, 0},
        {&loom_x86_packed_dot_contract_fragment, 1},
};

static const loom_low_lower_policy_t kX86Avx512LowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_x86_map_avx512_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx512_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86Avx512RuleSets),
            .values = kX86Avx512RuleSets,
        },
    .contract_bindings = kX86Avx512ContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kX86Avx512ContractBindings),
};

static const loom_low_lower_policy_t kX86Avx2LowLowerPolicy = {
    .name = IREE_SVL("x86-avx2-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_x86_map_avx2_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx2_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86Avx2RuleSets),
            .values = kX86Avx2RuleSets,
        },
    .contract_bindings = kX86Avx2ContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kX86Avx2ContractBindings),
};

static const loom_low_lower_policy_t kX86ScalarLowLowerPolicy = {
    .name = IREE_SVL("x86-scalar-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_x86_map_scalar_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_scalar_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86ScalarRuleSets),
            .values = kX86ScalarRuleSets,
        },
    .contract_bindings = kX86ScalarContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kX86ScalarContractBindings),
};

static const loom_low_lower_policy_t kX86PackedDotLowLowerPolicy = {
    .name = IREE_SVL("x86-packed-dot-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_x86_map_packed_dot_type, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86PackedDotRuleSets),
            .values = kX86PackedDotRuleSets,
        },
    .contract_bindings = kX86PackedDotContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kX86PackedDotContractBindings),
};

static const loom_low_lower_policy_t kX86Avx512PackedDotLowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-packed-dot-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_x86_map_avx512_packed_dot_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx512_packed_dot_argument,
                     .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kX86Avx512PackedDotRuleSets),
            .values = kX86Avx512PackedDotRuleSets,
        },
    .contract_bindings = kX86Avx512PackedDotContractBindings,
    .contract_binding_count =
        IREE_ARRAYSIZE(kX86Avx512PackedDotContractBindings),
};

const loom_low_lower_policy_t* loom_x86_avx512_low_lower_policy(void) {
  return &kX86Avx512LowLowerPolicy;
}

const loom_low_lower_policy_t* loom_x86_avx2_low_lower_policy(void) {
  return &kX86Avx2LowLowerPolicy;
}

const loom_low_lower_policy_t* loom_x86_scalar_low_lower_policy(void) {
  return &kX86ScalarLowLowerPolicy;
}

const loom_low_lower_policy_t* loom_x86_packed_dot_low_lower_policy(void) {
  return &kX86PackedDotLowLowerPolicy;
}

void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("x86.scalar.core"),
          .policy = &kX86ScalarLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx2.core"),
          .policy = &kX86Avx2LowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx512.core"),
          .policy = &kX86Avx512LowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.packed_dot.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx512_vnni.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx512_bf16.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx_vnni.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx_vnni_int8.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx_vnni_int16.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx10_2.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.avx512_packed_dot.core"),
          .policy = &kX86Avx512PackedDotLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
