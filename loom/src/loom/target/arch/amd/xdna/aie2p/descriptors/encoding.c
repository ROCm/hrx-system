// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

// Component projection shared by every physical register in one aggregate
// move family.
typedef struct loom_aie2p_move_projection_t {
  // First physical component in the generated component table.
  uint16_t component_start;
  // Number of physical components in this register.
  uint8_t component_count;
  // Generated route realizing the aggregate move.
  uint8_t route_id;
} loom_aie2p_move_projection_t;

// Exact raw-encoding metadata for one aggregate move-step mechanism.
typedef struct loom_aie2p_move_step_encoding_info_t {
  // Native instruction form selected for the move step.
  loom_aie2p_instruction_id_t instruction_id;
  // Instruction field receiving the destination register encoding.
  loom_aie2p_encoding_field_id_t destination_field_id;
  // Instruction field receiving the source register encoding.
  loom_aie2p_encoding_field_id_t source_field_id;
  // Machine adapter mapping the destination physical register.
  loom_aie2p_register_adapter_id_t destination_adapter_id;
  // Machine adapter mapping the source physical register.
  loom_aie2p_register_adapter_id_t source_adapter_id;
} loom_aie2p_move_step_encoding_info_t;

// Allocation move membership and register-part projections are resolved from
// the owned machine tables at generation time.
#include "loom/target/arch/amd/xdna/aie2p/descriptors/register_tables.inl"

#define LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT 16u

static_assert(sizeof(loom_aie2p_move_projection_t) == 4,
              "AIE2P move projections must remain compact");
static_assert(sizeof(loom_aie2p_move_step_encoding_info_t) == 10,
              "AIE2P move encoding rows must remain compact");

uint32_t loom_aie2p_descriptor_register_part_units(uint16_t register_part_id) {
  return register_part_id == LOOM_LOW_REGISTER_PART_NONE
             ? UINT32_MAX
             : kRegisterPartAtomicUnitMasks[register_part_id];
}

loom_aie2p_move_route_id_t loom_aie2p_descriptor_resolve_move_route(
    loom_aie2p_physical_register_id_t source,
    loom_aie2p_physical_register_id_t destination) {
  const uint32_t direct_candidates =
      kMoveSourceMasks[source] & kMoveDestinationMasks[destination];
  if (direct_candidates != 0) {
    return (loom_aie2p_move_route_id_t)(kMoveRouteDirectFirst +
                                        iree_math_count_trailing_zeros_u32(
                                            direct_candidates));
  }

  const uint8_t source_category = kMoveScalarSourceCategories[source];
  const uint8_t destination_category =
      kMoveScalarDestinationCategories[destination];
  if (source_category != UINT8_MAX && destination_category != UINT8_MAX) {
    return (loom_aie2p_move_route_id_t)(kMoveRouteScalarFirst +
                                        source_category *
                                            kMoveRouteScalarDestinationCount +
                                        destination_category);
  }

  const loom_aie2p_move_projection_t* source_projection =
      &kMoveProjections[kMoveProjectionIndices[source]];
  const loom_aie2p_move_projection_t* destination_projection =
      &kMoveProjections[kMoveProjectionIndices[destination]];
  return source_projection->route_id != 0 &&
                 source_projection->route_id == destination_projection->route_id
             ? source_projection->route_id
             : LOOM_AIE2P_MOVE_ROUTE_ID_INVALID;
}

uint8_t loom_aie2p_descriptor_move_route_step_count(
    loom_aie2p_move_route_id_t route_id) {
  IREE_ASSERT(route_id != LOOM_AIE2P_MOVE_ROUTE_ID_INVALID &&
              route_id < kMoveRouteLimit);
  return route_id < kMoveRouteAggregateFirst
             ? 1
             : kMoveAggregateStepCounts[route_id - kMoveRouteAggregateFirst];
}

static void loom_aie2p_descriptor_append_move_resource_class(
    uint16_t schedule_class_id, loom_aie2p_move_step_profile_t* profile) {
  if (schedule_class_id == UINT16_MAX) {
    return;
  }
  IREE_ASSERT_LT(profile->resource_schedule_class_count,
                 LOOM_AIE2P_MOVE_STEP_MAX_RESOURCE_CLASS_COUNT);
  profile
      ->resource_schedule_class_ids[profile->resource_schedule_class_count++] =
      schedule_class_id;
}

static void loom_aie2p_descriptor_set_scalar_move_resources(
    uint8_t source_category, uint8_t destination_category,
    loom_aie2p_move_step_profile_t* profile) {
  if (source_category + 1u == kMoveRouteScalarSourceCount) {
    loom_aie2p_descriptor_append_move_resource_class(
        kMoveScalarCoreIdScheduleClassId, profile);
    return;
  }
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarCommonScheduleClassId, profile);
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarDestinationScheduleClassIds[destination_category], profile);
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveScalarSourceScheduleClassIds[source_category], profile);
}

loom_aie2p_move_step_profile_t loom_aie2p_descriptor_move_step_profile(
    loom_aie2p_move_step_profile_id_t profile_id) {
  IREE_ASSERT(profile_id != LOOM_AIE2P_MOVE_STEP_PROFILE_ID_INVALID &&
              profile_id < kMoveRouteLimit);
  loom_aie2p_move_step_profile_t profile = {
      .encoding = LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR,
  };
  if (profile_id < kMoveRouteScalarFirst) {
    profile.descriptor_ordinal =
        kMoveDescriptorOrdinals[profile_id - kMoveRouteDirectFirst];
    return profile;
  }
  if (profile_id < kMoveRouteAggregateFirst) {
    const uint8_t scalar_route = profile_id - kMoveRouteScalarFirst;
    profile.descriptor_ordinal = kMoveScalarDescriptorOrdinal;
    loom_aie2p_descriptor_set_scalar_move_resources(
        scalar_route / kMoveRouteScalarDestinationCount,
        scalar_route % kMoveRouteScalarDestinationCount, &profile);
    return profile;
  }

  const uint8_t aggregate_route = profile_id - kMoveRouteAggregateFirst;
  profile.descriptor_ordinal = kMoveEventDescriptorOrdinal;
  profile.encoding = kMoveAggregateEncodingKinds[aggregate_route];
  IREE_ASSERT(profile.encoding != LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR &&
              "scalar aggregate steps retain their exact scalar profile");
  loom_aie2p_descriptor_append_move_resource_class(
      kMoveAggregateScheduleClassIds[aggregate_route], &profile);
  return profile;
}

loom_aie2p_move_route_step_t loom_aie2p_descriptor_move_route_step(
    loom_aie2p_move_route_id_t route_id, loom_aie2p_register_move_t move,
    uint8_t step_index) {
  IREE_ASSERT(route_id != LOOM_AIE2P_MOVE_ROUTE_ID_INVALID &&
              route_id < kMoveRouteLimit);
  IREE_ASSERT_LT(step_index,
                 loom_aie2p_descriptor_move_route_step_count(route_id));
  loom_aie2p_move_route_step_t step = {
      .move = move,
  };
  if (route_id < kMoveRouteScalarFirst) {
    step.profile_id = route_id;
    return step;
  }
  if (route_id < kMoveRouteAggregateFirst) {
    step.profile_id = route_id;
    return step;
  }

  const uint8_t aggregate_route = route_id - kMoveRouteAggregateFirst;
  const loom_aie2p_move_projection_t* source_projection =
      &kMoveProjections[kMoveProjectionIndices[move.source]];
  const loom_aie2p_move_projection_t* destination_projection =
      &kMoveProjections[kMoveProjectionIndices[move.destination]];
  IREE_ASSERT_EQ(source_projection->route_id, route_id);
  IREE_ASSERT_EQ(destination_projection->route_id, route_id);
  step.move = (loom_aie2p_register_move_t){
      .source = kMoveProjectionComponents[source_projection->component_start +
                                          step_index],
      .destination =
          kMoveProjectionComponents[destination_projection->component_start +
                                    step_index],
  };
  if (kMoveAggregateEncodingKinds[aggregate_route] ==
      LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR) {
    const uint8_t source_category =
        kMoveScalarSourceCategories[step.move.source];
    const uint8_t destination_category =
        kMoveScalarDestinationCategories[step.move.destination];
    IREE_ASSERT(source_category != UINT8_MAX &&
                destination_category != UINT8_MAX);
    step.profile_id =
        (loom_aie2p_move_step_profile_id_t)(kMoveRouteScalarFirst +
                                            source_category *
                                                kMoveRouteScalarDestinationCount +
                                            destination_category);
  } else {
    step.profile_id = route_id;
  }
  return step;
}

uint32_t loom_aie2p_descriptor_select_constant_encoding(
    uint32_t descriptor_ordinal, const int64_t* immediate_values) {
  uint32_t short_ordinal;
  switch (descriptor_ordinal) {
    case AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32:
    case AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_BYTE_OFFSET_I32:
      short_ordinal = AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT;
      break;
    case AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_ADDRESS_INDEX_I20:
      short_ordinal =
          AIE2P_CORE_DESCRIPTOR_REF_MATERIALIZE_STATIC_ADDRESS_INDEX_I20_SHORT;
      break;
    default:
      return descriptor_ordinal;
  }
  return immediate_values[0] >= -1024 && immediate_values[0] <= 1023
             ? short_ordinal
             : descriptor_ordinal;
}

static void loom_aie2p_descriptor_append_field(
    loom_aie2p_encoding_field_id_t field_id, uint64_t value,
    iree_host_size_t* field_count,
    loom_aie2p_encoding_field_value_t* field_values) {
  IREE_ASSERT(*field_count < LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT &&
              "verified AIE2P descriptor exceeds field storage");
  field_values[*field_count] = (loom_aie2p_encoding_field_value_t){
      .field_id = field_id,
      .value = value,
  };
  ++*field_count;
}

loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal,
    const loom_low_allocation_assignment_t* const* operand_assignments,
    const int64_t* immediate_values) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];

  loom_aie2p_encoding_field_value_t
      field_values[LOOM_AIE2P_DESCRIPTOR_MAX_ENCODING_FIELD_COUNT];
  iree_host_size_t field_count = 0;
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor_layout->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field =
        &descriptor_set->encoding_field_values
             [descriptor_layout->encoding_field_value_start + i];
    loom_aie2p_descriptor_append_field(field->encoding_field_id, field->value,
                                       &field_count, field_values);
  }

  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (operand->encoding_field_id == 0) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment = operand_assignments[i];
    const uint8_t encoded_register =
        loom_aie2p_machine_adapt_allocated_register(
            operand->encoding_adapter_id,
            (loom_aie2p_physical_register_id_t)assignment->location_base);
    loom_aie2p_descriptor_append_field(operand->encoding_field_id,
                                       encoded_register, &field_count,
                                       field_values);
  }

  for (uint16_t i = 0; i < descriptor_layout->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor_layout->immediate_start + i];
    if (immediate->encoding_field_id == 0) {
      continue;
    }
    const uint64_t encoded_immediate =
        loom_aie2p_machine_encode_verified_immediate(immediate->encoding_id,
                                                     immediate_values[i]);
    loom_aie2p_descriptor_append_field(immediate->encoding_field_id,
                                       encoded_immediate, &field_count,
                                       field_values);
  }

  return loom_aie2p_encoding_pack_verified_instruction(
      descriptor->encoding_id, field_values, field_count);
}

loom_aie2p_encoded_slot_t loom_aie2p_descriptor_encode_move_step(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_aie2p_move_route_step_t step) {
  const loom_aie2p_move_step_profile_t profile =
      loom_aie2p_descriptor_move_step_profile(step.profile_id);
  if (profile.encoding == LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR) {
    const loom_low_allocation_assignment_t destination_assignment = {
        .unit_count = 1,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = step.move.destination,
        .location_count = 1,
    };
    const loom_low_allocation_assignment_t source_assignment = {
        .unit_count = 1,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = step.move.source,
        .location_count = 1,
    };
    const loom_low_allocation_assignment_t* operand_assignments[] = {
        &destination_assignment,
        &source_assignment,
    };
    return loom_aie2p_descriptor_encode(
        descriptor_set, profile.descriptor_ordinal, operand_assignments, NULL);
  }

  const iree_host_size_t aggregate_route =
      step.profile_id - kMoveRouteAggregateFirst;
  const loom_aie2p_move_step_encoding_info_t* encoding =
      &kMoveAggregateEncodingInfos[aggregate_route];
  IREE_ASSERT(encoding->instruction_id != 0);
  const loom_aie2p_encoding_field_value_t fields[] = {
      {
          .field_id = encoding->destination_field_id,
          .value = loom_aie2p_machine_adapt_allocated_register(
              encoding->destination_adapter_id, step.move.destination),
      },
      {
          .field_id = encoding->source_field_id,
          .value = loom_aie2p_machine_adapt_allocated_register(
              encoding->source_adapter_id, step.move.source),
      },
  };
  return loom_aie2p_encoding_pack_verified_instruction(
      encoding->instruction_id, fields, IREE_ARRAYSIZE(fields));
}
