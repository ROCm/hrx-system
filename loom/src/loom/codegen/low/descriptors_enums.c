// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

bool loom_low_operand_role_is_packet_operand(loom_low_operand_role_t role) {
  return role == LOOM_LOW_OPERAND_ROLE_OPERAND ||
         role == LOOM_LOW_OPERAND_ROLE_PREDICATE ||
         role == LOOM_LOW_OPERAND_ROLE_RESOURCE;
}

bool loom_low_spill_slot_space_is_valid(loom_low_spill_slot_space_t space) {
  switch (space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      return true;
    default:
      return false;
  }
}

iree_string_view_t loom_low_operand_role_name(loom_low_operand_role_t role) {
  switch (role) {
    case LOOM_LOW_OPERAND_ROLE_RESULT:
      return IREE_SV("result");
    case LOOM_LOW_OPERAND_ROLE_OPERAND:
      return IREE_SV("operand");
    case LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT:
      return IREE_SV("operand_result");
    case LOOM_LOW_OPERAND_ROLE_PREDICATE:
      return IREE_SV("predicate");
    case LOOM_LOW_OPERAND_ROLE_RESOURCE:
      return IREE_SV("resource");
    case LOOM_LOW_OPERAND_ROLE_IMPLICIT:
      return IREE_SV("implicit");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_operand_address_map_kind_name(
    loom_low_operand_address_map_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT:
      return IREE_SV("direct");
    case LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET:
      return IREE_SV("low_subset");
    case LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE:
      return IREE_SV("target_state");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_immediate_kind_name(
    loom_low_immediate_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
      return IREE_SV("signed");
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
      return IREE_SV("unsigned");
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return IREE_SV("ordinal");
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      return IREE_SV("enum");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_effect_kind_name(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return IREE_SV("read");
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return IREE_SV("write");
    case LOOM_LOW_EFFECT_KIND_CALL:
      return IREE_SV("call");
    case LOOM_LOW_EFFECT_KIND_BARRIER:
      return IREE_SV("barrier");
    case LOOM_LOW_EFFECT_KIND_COUNTER:
      return IREE_SV("counter");
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
      return IREE_SV("convergent");
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return IREE_SV("control");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_memory_space_name(
    loom_low_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_NONE:
      return IREE_SV("none");
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
      return IREE_SV("generic");
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
      return IREE_SV("global");
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return IREE_SV("workgroup");
    case LOOM_LOW_MEMORY_SPACE_STACK:
      return IREE_SV("stack");
    case LOOM_LOW_MEMORY_SPACE_VM_REF:
      return IREE_SV("vm_ref");
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return IREE_SV("wasm_memory");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_spill_slot_space_name(
    loom_low_spill_slot_space_t space) {
  switch (space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      return IREE_SV("stack");
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      return IREE_SV("scratch");
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      return IREE_SV("private");
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      return IREE_SV("lds");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_constraint_kind_name(
    loom_low_constraint_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_CONSTRAINT_KIND_TIED:
      return IREE_SV("tied");
    case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE:
      return IREE_SV("commutable");
    case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE:
      return IREE_SV("destructive");
    case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
      return IREE_SV("early_clobber");
    case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
      return IREE_SV("rematerializable");
    case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
      return IREE_SV("foldable");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_latency_kind_name(loom_low_latency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LATENCY_KIND_EXACT:
      return IREE_SV("exact");
    case LOOM_LOW_LATENCY_KIND_ESTIMATE:
      return IREE_SV("estimate");
    case LOOM_LOW_LATENCY_KIND_VARIABLE:
      return IREE_SV("variable");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_model_quality_name(
    loom_low_model_quality_t quality) {
  switch (quality) {
    case LOOM_LOW_MODEL_QUALITY_EXACT:
      return IREE_SV("exact");
    case LOOM_LOW_MODEL_QUALITY_CALIBRATED:
      return IREE_SV("calibrated");
    case LOOM_LOW_MODEL_QUALITY_ESTIMATED:
      return IREE_SV("estimated");
    case LOOM_LOW_MODEL_QUALITY_FALLBACK:
      return IREE_SV("fallback");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_resource_kind_name(loom_low_resource_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_KIND_SCALAR_ALU:
      return IREE_SV("scalar_alu");
    case LOOM_LOW_RESOURCE_KIND_VECTOR_ALU:
      return IREE_SV("vector_alu");
    case LOOM_LOW_RESOURCE_KIND_MATRIX:
      return IREE_SV("matrix");
    case LOOM_LOW_RESOURCE_KIND_LOAD:
      return IREE_SV("load");
    case LOOM_LOW_RESOURCE_KIND_STORE:
      return IREE_SV("store");
    case LOOM_LOW_RESOURCE_KIND_CONTROL:
      return IREE_SV("control");
    case LOOM_LOW_RESOURCE_KIND_ADDRESS:
      return IREE_SV("address");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_hazard_kind_name(loom_low_hazard_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_KIND_MIN_DISTANCE:
      return IREE_SV("min_distance");
    case LOOM_LOW_HAZARD_KIND_WAIT_COUNTER:
      return IREE_SV("wait_counter");
    case LOOM_LOW_HAZARD_KIND_BYPASS:
      return IREE_SV("bypass");
    case LOOM_LOW_HAZARD_KIND_FUSION:
      return IREE_SV("fusion");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_hazard_reference_kind_name(
    loom_low_hazard_reference_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE:
      return IREE_SV("resource");
    case LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER:
      return IREE_SV("counter");
    case LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET:
      return IREE_SV("target");
    default:
      return IREE_SV("unknown");
  }
}
