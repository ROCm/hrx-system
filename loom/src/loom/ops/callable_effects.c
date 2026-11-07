// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/callable_effects.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

bool loom_callable_effects_may_access_memory(loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return true;
  }
  const loom_callable_effect_contract_t explicit_contract =
      loom_func_like_effect_contract(function);
  loom_region_t* body = loom_func_like_body(function);
  if (body && loom_region_has_memory_accesses(body)) {
    return true;
  }
  return explicit_contract != LOOM_CALLABLE_EFFECT_CONTRACT_PURE &&
         (explicit_contract != LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED ||
          !body);
}

loom_memory_space_mask_t loom_callable_effects_written_memory_spaces(
    loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return (loom_memory_space_mask_t)(1u
                                      << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  }
  const loom_region_t* body = loom_func_like_body(function);
  if (body) {
    return loom_region_written_memory_spaces(body);
  }
  switch (loom_func_like_effect_contract(function)) {
    case LOOM_CALLABLE_EFFECT_CONTRACT_PURE:
    case LOOM_CALLABLE_EFFECT_CONTRACT_READONLY:
      return 0;
    default:
      return (
          loom_memory_space_mask_t)(1u << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  }
}

bool loom_callable_effects_is_pure(loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return false;
  }
  const loom_callable_effect_contract_t explicit_contract =
      loom_func_like_effect_contract(function);
  if (explicit_contract != LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return explicit_contract == LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
  }
  loom_region_t* body = loom_func_like_body(function);
  if (body) {
    return !loom_region_has_read_effects(body) &&
           !loom_region_has_write_effects(body) &&
           !loom_region_has_convergent_effects(body) &&
           !loom_region_has_observable_effects(body) &&
           !loom_region_has_resource_identities(body);
  }
  return loom_func_like_effect_contract(function) ==
         LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
}

bool loom_callable_effects_has_readonly_contract(loom_func_like_t function) {
  return loom_func_like_isa(function) &&
         loom_func_like_effect_contract(function) ==
             LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
}

loom_callable_effect_contract_t loom_callable_effects_callee_contract(
    const loom_module_t* module, loom_symbol_ref_t callee) {
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= module->symbols.count) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  if (!symbol->defining_op) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  const loom_func_like_t function =
      loom_func_like_cast(module, symbol->defining_op);
  const loom_callable_effect_contract_t explicit_contract =
      loom_func_like_effect_contract(function);
  if (explicit_contract != LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return explicit_contract;
  }
  if (loom_callable_effects_is_pure(function)) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
  }
  if (loom_callable_effects_has_readonly_contract(function)) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
  }
  return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
}

bool loom_callable_effects_contract_satisfies(
    loom_callable_effect_contract_t provided,
    loom_callable_effect_contract_t required) {
  if (required == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return true;
  }
  switch (required) {
    case LOOM_CALLABLE_EFFECT_CONTRACT_PURE:
      return provided == LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
    case LOOM_CALLABLE_EFFECT_CONTRACT_READONLY:
      return provided == LOOM_CALLABLE_EFFECT_CONTRACT_PURE ||
             provided == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
    default:
      return false;
  }
}

loom_callable_effect_contract_t loom_callable_effects_classify_traits(
    loom_trait_flags_t traits) {
  const loom_trait_flags_t forbidden_traits =
      LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_NON_DETERMINISTIC |
      LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_CONVERGENT |
      LOOM_TRAIT_MEMORY_FENCE | LOOM_TRAIT_OBSERVABLE_EFFECT;
  if (iree_any_bit_set(traits, forbidden_traits) ||
      loom_traits_has_resource_identity(traits)) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  return iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY)
             ? LOOM_CALLABLE_EFFECT_CONTRACT_READONLY
             : LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
}

static loom_callable_effect_contract_t loom_callable_effects_classify_region(
    const loom_module_t* module, const loom_region_t* region) {
  if (!region) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
  }
  loom_callable_effect_contract_t contract = LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_callable_effect_contract_t op_contract =
          loom_callable_effects_classify_traits(
              loom_op_effective_traits(module, op));
      const loom_op_semantics_t semantics = loom_op_semantics(module, op);
      const loom_callable_effect_contract_t application_contract =
          loom_callable_effects_application_contract(op, semantics);
      if (semantics.lifetime_effects_only) {
        op_contract = LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
      } else if (application_contract !=
                 LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
        // The declared contract describes payload effects. Handle creation
        // and consumed arguments separately constrain ownership rewrites.
        op_contract = application_contract;
      }
      if (op_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
        return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
      }
      if (op_contract == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY) {
        contract = LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        const loom_callable_effect_contract_t nested_contract =
            loom_callable_effects_classify_region(module, regions[i]);
        if (nested_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
          return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
        }
        if (nested_contract == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY) {
          contract = LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
        }
      }
    }
  }
  return contract;
}

loom_callable_effect_contract_t loom_callable_effects_classify_body(
    const loom_module_t* module, const loom_region_t* body) {
  return loom_callable_effects_classify_region(module, body);
}

loom_callable_effect_contract_t loom_callable_effects_classify_function(
    const loom_module_t* module, loom_func_like_t function) {
  if (!loom_func_like_isa(function)) {
    return LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
  }
  const loom_callable_effect_contract_t explicit_contract =
      loom_func_like_effect_contract(function);
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return explicit_contract;
  }
  const loom_callable_effect_contract_t body_contract =
      loom_callable_effects_classify_body(module, body);
  if (explicit_contract == LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED) {
    return body_contract;
  }
  return loom_callable_effects_contract_satisfies(body_contract,
                                                  explicit_contract)
             ? explicit_contract
             : LOOM_CALLABLE_EFFECT_CONTRACT_UNSPECIFIED;
}

bool loom_callable_effects_body_is_readonly(const loom_module_t* module,
                                            const loom_region_t* body) {
  return loom_callable_effects_contract_satisfies(
      loom_callable_effects_classify_body(module, body),
      LOOM_CALLABLE_EFFECT_CONTRACT_READONLY);
}

loom_trait_flags_t loom_callable_effects_traits(
    const loom_module_t* module, const loom_op_t* op,
    loom_callable_effect_contract_t contract) {
  loom_trait_flags_t traits = LOOM_TRAIT_CALLABLE_BOUNDARY;
  switch (contract) {
    case LOOM_CALLABLE_EFFECT_CONTRACT_PURE:
      traits |= LOOM_TRAIT_PURE;
      break;
    case LOOM_CALLABLE_EFFECT_CONTRACT_READONLY:
      traits |= LOOM_TRAIT_READS_MEMORY;
      break;
    default:
      traits |= LOOM_TRAIT_UNKNOWN_EFFECTS;
      break;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t type =
        loom_module_value_type(module, loom_op_const_results(op)[i]);
    const loom_type_t* element = loom_type_register_value_type(type);
    if (loom_type_is_reference(element ? *element : type)) {
      traits = (traits & ~LOOM_TRAIT_PURE) | LOOM_TRAIT_UNIQUE_IDENTITY;
      break;
    }
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t value = loom_op_const_operands(op)[i];
    // Effective traits are also constructed for unresolved parser operands.
    if (value == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_type_t type = loom_module_value_type(module, value);
    const loom_type_t* element = loom_type_register_value_type(type);
    type = element ? *element : type;
    if (loom_type_is_reference(type) &&
        loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_OWNED) {
      traits = (traits & ~LOOM_TRAIT_PURE) | LOOM_TRAIT_OBSERVABLE_EFFECT;
      break;
    }
  }
  return traits;
}
