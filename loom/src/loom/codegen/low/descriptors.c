// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <stdint.h>

#include "loom/util/stable_id.h"

static iree_string_view_t loom_low_descriptor_set_string_view(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset) {
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_bstring_view(
      loom_bstring_table_get(&descriptor_set->string_table, string_offset));
}

uint64_t loom_low_descriptor_stable_id_from_key(iree_string_view_t key) {
  return loom_stable_id_from_string(key);
}

static bool loom_low_descriptor_constraint_ties_operands(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_operand_index,
    uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED &&
        constraint->lhs_operand_index == result_operand_index &&
        constraint->rhs_operand_index == packet_operand_index) {
      return true;
    }
  }
  return false;
}

bool loom_low_descriptor_operands_are_tied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t lhs_operand_index,
    uint16_t rhs_operand_index) {
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  const loom_low_operand_t* lhs = &operands[lhs_operand_index];
  const loom_low_operand_t* rhs = &operands[rhs_operand_index];
  if (lhs->role == LOOM_LOW_OPERAND_ROLE_RESULT &&
      loom_low_operand_role_is_packet_operand(rhs->role)) {
    return loom_low_descriptor_constraint_ties_operands(
        descriptor_set, descriptor, lhs_operand_index, rhs_operand_index);
  }
  if (rhs->role == LOOM_LOW_OPERAND_ROLE_RESULT &&
      loom_low_operand_role_is_packet_operand(lhs->role)) {
    return loom_low_descriptor_constraint_ties_operands(
        descriptor_set, descriptor, rhs_operand_index, lhs_operand_index);
  }
  return false;
}

static iree_string_view_t loom_low_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_low_descriptor_set_string_view(descriptor_set,
                                             descriptor_set->key_string_offset);
}

iree_host_size_t loom_low_descriptor_registry_descriptor_set_count(
    const loom_low_descriptor_registry_t* registry) {
  if (registry == NULL) {
    return 0;
  }
  return registry->descriptor_set_count +
         registry->descriptor_set_provider_count;
}

const loom_low_descriptor_set_t* loom_low_descriptor_registry_descriptor_set_at(
    const loom_low_descriptor_registry_t* registry, iree_host_size_t index) {
  if (registry == NULL) {
    return NULL;
  }
  if (index < registry->descriptor_set_count) {
    if (registry->descriptor_sets == NULL) {
      return NULL;
    }
    return registry->descriptor_sets[index];
  }
  index -= registry->descriptor_set_count;
  if (index < registry->descriptor_set_provider_count) {
    if (registry->descriptor_set_providers == NULL) {
      return NULL;
    }
    loom_low_descriptor_set_provider_t provider =
        registry->descriptor_set_providers[index];
    return provider ? provider() : NULL;
  }
  return NULL;
}

static const loom_low_descriptor_set_t*
loom_low_descriptor_registry_descriptor_set_at_trusted(
    const loom_low_descriptor_registry_t* registry, iree_host_size_t index) {
  if (index < registry->descriptor_set_count) {
    return registry->descriptor_sets[index];
  }
  index -= registry->descriptor_set_count;
  loom_low_descriptor_set_provider_t provider =
      registry->descriptor_set_providers[index];
  return provider();
}

const loom_low_descriptor_set_t* loom_low_descriptor_registry_lookup(
    const loom_low_descriptor_registry_t* registry, iree_string_view_t key) {
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at_trusted(registry, i);
    iree_string_view_t candidate_key =
        loom_low_descriptor_set_key(descriptor_set);
    if (!iree_string_view_equal(candidate_key, key)) {
      continue;
    }
    return descriptor_set;
  }
  return NULL;
}

const loom_low_descriptor_set_t* loom_low_descriptor_registry_lookup_by_id(
    const loom_low_descriptor_registry_t* registry, uint64_t stable_id) {
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at_trusted(registry, i);
    if (descriptor_set->stable_id != stable_id) {
      continue;
    }
    return descriptor_set;
  }
  return NULL;
}

iree_string_view_t loom_low_descriptor_set_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset) {
  return loom_low_descriptor_set_string_view(descriptor_set, string_offset);
}

bool loom_low_descriptor_set_lookup_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class) {
  *out_descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class = NULL;
  }
  for (uint32_t i = 0;
       i < descriptor_set->reg_class_count && i < LOOM_LOW_REG_CLASS_NONE;
       ++i) {
    const loom_low_reg_class_t* register_class =
        &descriptor_set->reg_classes[i];
    iree_string_view_t descriptor_register_class_name =
        loom_low_descriptor_set_string_view(descriptor_set,
                                            register_class->name_string_offset);
    if (!iree_string_view_equal(register_class_name,
                                descriptor_register_class_name)) {
      continue;
    }
    *out_descriptor_register_class_id = (uint16_t)i;
    if (out_descriptor_register_class) {
      *out_descriptor_register_class = register_class;
    }
    return true;
  }
  return false;
}

const loom_low_descriptor_t* loom_low_descriptor_set_descriptor_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  if (descriptor_set == NULL ||
      descriptor_ordinal >= descriptor_set->descriptor_count) {
    return NULL;
  }
  return &descriptor_set->descriptors[descriptor_ordinal];
}

uint32_t loom_low_descriptor_set_descriptor_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor_set->descriptors == NULL ||
      descriptor == NULL) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  const uintptr_t base = (uintptr_t)descriptor_set->descriptors;
  const uintptr_t address = (uintptr_t)descriptor;
  if (address < base) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  const uintptr_t offset = address - base;
  if (offset % sizeof(*descriptor_set->descriptors) != 0) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  const uintptr_t ordinal = offset / sizeof(*descriptor_set->descriptors);
  if (ordinal >= descriptor_set->descriptor_count || ordinal > UINT32_MAX) {
    return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  }
  return (uint32_t)ordinal;
}

const loom_low_asm_form_t* loom_low_descriptor_set_asm_form_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t asm_form_ordinal) {
  if (descriptor_set == NULL ||
      asm_form_ordinal >= descriptor_set->asm_form_count) {
    return NULL;
  }
  return &descriptor_set->asm_forms[asm_form_ordinal];
}

uint32_t loom_low_descriptor_set_lookup_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  }
  return descriptor->canonical_asm_form_ordinal;
}

uint32_t loom_low_descriptor_set_lookup_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t low = 0;
  uint32_t high = descriptor_set->descriptor_ref_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_descriptor_ref_t* descriptor_ref =
        &descriptor_set->descriptor_refs[mid];
    iree_string_view_t descriptor_ref_key = loom_low_descriptor_set_string_view(
        descriptor_set, descriptor_ref->key_string_offset);
    const int comparison = iree_string_view_compare(descriptor_ref_key, key);
    if (comparison == 0) {
      return descriptor_ref->descriptor_ordinal;
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

uint32_t loom_low_descriptor_set_lookup_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic) {
  uint32_t low = 0;
  uint32_t high = descriptor_set->asm_form_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_asm_form_t* asm_form = &descriptor_set->asm_forms[mid];
    iree_string_view_t asm_mnemonic = loom_low_descriptor_set_string_view(
        descriptor_set, asm_form->mnemonic_string_offset);
    const int comparison = iree_string_view_compare(asm_mnemonic, mnemonic);
    if (comparison == 0) {
      if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
        return LOOM_LOW_ASM_FORM_ORDINAL_NONE;
      }
      return mid;
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return LOOM_LOW_ASM_FORM_ORDINAL_NONE;
}
