// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/lower/workgroup_reduce.h"

#include "loom/analysis/storage_layout.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/facts.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/lower/collective_arithmetic.h"

// Immutable scalar-carrier rows share storage and conversion metadata across
// combining kinds. Boolean values use integer scratch; unsigned arithmetic
// uses explicit bitcasts at the signless source boundary.
typedef struct loom_spirv_workgroup_carrier_t {
  // Scalar semantic type of the arithmetic register.
  loom_scalar_type_t element_type;
  // Scalar type of the identity constant, before an unsigned-view bitcast.
  loom_scalar_type_t constant_element_type;
  // Bytes and alignment of one scratch element.
  uint8_t byte_count;
  // Register class for an addressed scratch element.
  uint16_t pointer_class;
  // Register class for the typed scratch array root.
  uint16_t array_class;
  // Immediate spelling for the identity's integer or floating-point bits.
  iree_string_view_t constant_key;
  // Union of this carrier's instruction requirements.
  uint64_t required_features;
  // Scalar constant instruction, unused for Boolean identities.
  uint32_t constant;
  // Optional unsigned view of a signless source or integer identity constant.
  uint32_t input_cast;
  // Optional cast returning an unsigned result to its signless source view.
  uint32_t output_cast;
  // Typed array-element address instruction.
  uint32_t address;
  // Typed scratch read instruction.
  uint32_t load;
  // Typed scratch write instruction.
  uint32_t store;
} loom_spirv_workgroup_carrier_t;

typedef struct loom_spirv_workgroup_arithmetic_t {
  // Union of native collective and local-combine feature requirements.
  uint64_t required_features;
  // Exact integer encoding of the empty stripe's arithmetic identity.
  int64_t identity_bits;
  // Native subgroup reduction instruction.
  uint32_t subgroup;
  // Binary arithmetic instruction, or comparison when |select| is present.
  uint32_t combine;
  // Selection instruction for integer extrema, or no descriptor.
  uint32_t select;
  // Index of the shared scalar arithmetic/storage carrier.
  uint8_t carrier;
} loom_spirv_workgroup_arithmetic_t;

typedef struct loom_spirv_workgroup_source_t {
  // Feature requirements for native source packet transport.
  uint64_t required_features;
  // Scalar extraction instructions for two-, three-, and four-lane packets.
  uint32_t extract[3];
  // Packet construction instructions for two-, three-, and four-lane packets.
  uint32_t construct[3];
  // One-based arithmetic row by kind; zero denotes an unsupported family cell.
  uint8_t arithmetic[LOOM_COMBINING_KIND_COUNT_];
} loom_spirv_workgroup_source_t;

#include "loom/target/arch/spirv/lower/workgroup_reduce_rows.inl"

typedef struct loom_spirv_workgroup_payload_t {
  // Immutable native packet transport metadata.
  const loom_spirv_workgroup_source_t* source;
  // Immutable selected combining instructions and empty-stripe identity.
  const loom_spirv_workgroup_arithmetic_t* arithmetic;
  // Immutable arithmetic and scratch representation.
  const loom_spirv_workgroup_carrier_t* carrier;
  // Optional function-arena ordered-key arithmetic for floating extrema.
  loom_spirv_collective_extrema_t* extrema;
  // Optional function-arena magnitude/sign arithmetic for uniform products.
  loom_spirv_collective_product_t* product;
  // Native source/result packet type; vector1 uses a scalar register.
  loom_type_t packet_type;
  // Scalar source/result component type before arithmetic conversion.
  loom_type_t source_type;
  // Scalar native reduction and stripe-accumulator type.
  loom_type_t arithmetic_type;
  // Scalar scratch element type; Boolean payloads use integer storage.
  loom_type_t storage_type;
  // Scalar type used to construct the arithmetic identity.
  loom_type_t constant_type;
  // Addressed scratch-element type.
  loom_type_t pointer_type;
  // Typed scratch-array root type.
  loom_type_t array_type;
  // Interned immediate name for identity bits.
  loom_string_id_t constant_key;
  // Number of independent scalar components in this source packet.
  uint8_t component_count;
} loom_spirv_workgroup_payload_t;

typedef struct loom_spirv_workgroup_reduce_plan_t {
  // Function-arena positional payload plans.
  loom_spirv_workgroup_payload_t* payloads;
  // Number of source/result payload pairs.
  uint16_t payload_count;
  // Total scalar components sharing topology and synchronization.
  uint16_t component_count;
  // Conservative maximum number of subgroup partials in this workgroup.
  uint32_t capacity;
  // Native rank, subgroup count, and loop-coordinate carrier.
  loom_type_t integer_type;
  // Native topology condition carrier.
  loom_type_t predicate_type;
  // Interned integer literal field used for topology constants.
  loom_string_id_t value_key;
  // Interned immediate field used for packet extraction.
  loom_string_id_t component_key;
} loom_spirv_workgroup_reduce_plan_t;

static iree_status_t loom_spirv_workgroup_type(
    loom_low_lower_context_t* context, loom_scalar_type_t element,
    loom_type_t* out_type) {
  return loom_low_lower_make_typed_register_type(
      context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1, loom_type_scalar(element),
      out_type);
}

static iree_status_t loom_spirv_workgroup_reject(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t role, iree_string_view_t subject,
    iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(IREE_SV("kernel.workgroup.reduce")),
      loom_param_string(role),
      loom_param_string(subject),
      loom_param_string(constraint),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_TARGET_003_REF, params,
                                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_spirv_select_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  const loom_target_workgroup_size_t* size =
      loom_low_lower_context_workgroup_size(context);
  if (!size) {
    return loom_spirv_workgroup_reject(context, source_op, IREE_SV("topology"),
                                       IREE_SV("workgroup"),
                                       IREE_SV("fixed_workgroup_size"));
  }
  uint64_t capacity = (uint64_t)size->x * size->y * size->z;
  const loom_spirv_target_facts_t* facts = loom_spirv_target_facts_cast(
      loom_low_lower_context_target_facts(context));
  if (facts->subgroup.maximum_workgroup_subgroup_count != 0) {
    capacity =
        iree_min(capacity, facts->subgroup.maximum_workgroup_subgroup_count);
  }
  if (capacity == 0 || capacity >= UINT32_MAX) {
    return loom_spirv_workgroup_reject(context, source_op, IREE_SV("topology"),
                                       IREE_SV("workgroup"),
                                       IREE_SV("subgroup_capacity"));
  }
  const loom_combining_kind_t kind =
      loom_kernel_workgroup_reduce_kind(source_op);
  loom_spirv_workgroup_reduce_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_spirv_workgroup_reduce_plan_t){
      .payload_count = source_op->operand_count,
      .capacity = (uint32_t)capacity};
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, plan->payload_count, sizeof(*plan->payloads),
      (void**)&plan->payloads));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_type(context, LOOM_SCALAR_TYPE_I32,
                                                 &plan->integer_type));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_type(context, LOOM_SCALAR_TYPE_I1,
                                                 &plan->predicate_type));
  loom_module_t* module = loom_low_lower_context_module(context);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("i32_value"),
                                                 &plan->value_key));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("component_index"), &plan->component_key));
  const uint64_t available_features =
      loom_low_lower_context_bundle(context)->config->contract_feature_bits;
  uint32_t component_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_type_t type =
        loom_module_value_type(module, loom_op_const_operands(source_op)[i]);
    const loom_scalar_type_t element = loom_type_element_type(type);
    const uint8_t source_index =
        element < IREE_ARRAYSIZE(kWorkgroupSourceIndices)
            ? kWorkgroupSourceIndices[element]
            : 0;
    const uint64_t lanes = loom_type_is_scalar(type) ? 1
                           : loom_type_is_all_static(type)
                               ? loom_type_dim_static_size_at(type, 0)
                               : 0;
    if (source_index == 0 || lanes == 0 || lanes > 4) {
      status = loom_low_lower_emit_source_type_unsupported(
          context, source_op, IREE_SV("values"), type);
      break;
    }
    loom_spirv_workgroup_payload_t* payload = &plan->payloads[i];
    *payload = (loom_spirv_workgroup_payload_t){
        .source = &kWorkgroupSources[source_index - 1],
        .component_count = (uint8_t)lanes};
    const uint8_t arithmetic_index = payload->source->arithmetic[kind];
    // Verification establishes the type/kind pair; rows cover the entire native
    // family. Math-contract eligibility belongs to the selected recipe guards.
    payload->arithmetic = &kWorkgroupArithmetic[arithmetic_index - 1];
    payload->carrier = &kWorkgroupCarriers[payload->arithmetic->carrier];
    const uint64_t required_features = payload->source->required_features |
                                       payload->arithmetic->required_features |
                                       payload->carrier->required_features;
    const uint64_t missing_features = required_features & ~available_features;
    if (missing_features != 0) {
      const loom_spirv_feature_atom_t missing_feature =
          (loom_spirv_feature_atom_t)iree_math_count_trailing_zeros_u64(
              missing_features);
      status = loom_spirv_workgroup_reject(
          context, source_op, IREE_SV("feature"),
          loom_spirv_feature_atom_name(missing_feature),
          IREE_SV("native_collective_features"));
      break;
    }
    component_count += payload->component_count;
    // The structured stripe loop carries every component plus its coordinates.
    if (component_count > UINT16_MAX - 4) {
      status = loom_spirv_workgroup_reject(
          context, source_op, IREE_SV("payload"), IREE_SV("values"),
          IREE_SV("structured_loop_arity"));
      break;
    }
    status = loom_low_lower_make_typed_register_type(
        context, SPIRV_LOGICAL_CORE_REG_CLASS_ID_ID, 1,
        lanes == 1 ? loom_type_scalar(element) : type, &payload->packet_type);
    if (iree_status_is_ok(status)) {
      status =
          loom_spirv_workgroup_type(context, element, &payload->source_type);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_type(
          context, payload->carrier->element_type, &payload->arithmetic_type);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_type(
          context, payload->carrier->constant_element_type,
          &payload->constant_type);
    }
    payload->storage_type = element == LOOM_SCALAR_TYPE_I1
                                ? plan->integer_type
                                : payload->arithmetic_type;
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_make_register_type(
          context, payload->carrier->pointer_class, 1, &payload->pointer_type);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_make_register_type(
          context, payload->carrier->array_class, 1, &payload->array_type);
    }
    if (iree_status_is_ok(status)) {
      status = loom_module_intern_string(module, payload->carrier->constant_key,
                                         &payload->constant_key);
    }
    if (iree_status_is_ok(status) &&
        element != payload->carrier->element_type) {
      status = loom_low_lower_allocate_function_array(
          context, 1, sizeof(*payload->extrema), (void**)&payload->extrema);
      if (iree_status_is_ok(status)) {
        status = loom_spirv_collective_extrema_initialize(
            context,
            loom_spirv_collective_extrema_row(loom_type_scalar(element)), kind,
            payload->extrema);
      }
    }
    if (iree_status_is_ok(status) &&
        loom_spirv_collective_product_needed(
            context, kind, loom_op_const_operands(source_op)[i])) {
      status = loom_low_lower_allocate_function_array(
          context, 1, sizeof(*payload->product), (void**)&payload->product);
      if (iree_status_is_ok(status)) {
        status = loom_spirv_collective_product_initialize(context, element,
                                                          payload->product);
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_plan_value_type(
          context, loom_op_const_results(source_op)[i], payload->packet_type);
    }
  }
  if (iree_status_is_ok(status)) {
    plan->component_count = (uint16_t)component_count;
    *out_plan = loom_low_lower_plan_make(LOOM_OP_KERNEL_WORKGROUP_REDUCE, plan);
  }
  return status;
}

iree_status_t loom_spirv_workgroup_reduce_measure_storage(
    loom_low_lower_plan_t selected_plan, uint64_t* inout_byte_extent) {
  const loom_spirv_workgroup_reduce_plan_t* plan = selected_plan.target_data;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = &plan->payloads[i];
    status = loom_storage_layout_append(
        ((uint64_t)plan->capacity + 1) * payload->carrier->byte_count *
            payload->component_count,
        payload->carrier->byte_count, inout_byte_extent, NULL);
  }
  return status;
}

static iree_status_t loom_spirv_workgroup_emit_attributes(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint32_t descriptor_ref, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attributes,
    loom_type_t result_type, loom_value_id_t* out_value) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ref)};
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count, attributes,
      out_value ? &result_type : NULL, out_value ? 1 : 0, NULL, 0, location,
      &emitted));
  if (out_value) {
    *out_value = loom_op_const_results(emitted)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_workgroup_emit(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint32_t descriptor_ref, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_type_t result_type,
    loom_value_id_t* out_value) {
  return loom_spirv_workgroup_emit_attributes(
      context, location, descriptor_ref, operands, operand_count,
      loom_named_attr_slice_empty(), result_type, out_value);
}

static iree_status_t loom_spirv_workgroup_constant(
    loom_low_lower_context_t* context, loom_location_id_t location,
    uint32_t descriptor_ref, loom_string_id_t key, int64_t literal,
    loom_type_t type, loom_value_id_t* out_value) {
  const loom_named_attr_t attribute = {.name_id = key,
                                       .value = loom_attr_i64(literal)};
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = loom_low_descriptor_set_descriptor_at(
          loom_low_lower_context_descriptor_set(context), descriptor_ref)};
  loom_op_t* emitted = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor,
      key != LOOM_STRING_ID_INVALID ? loom_make_named_attr_slice(&attribute, 1)
                                    : loom_named_attr_slice_empty(),
      type, location, &emitted));
  *out_value = loom_op_const_results(emitted)[0];
  return iree_ok_status();
}

static iree_status_t loom_spirv_workgroup_identity(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_workgroup_payload_t* payload, loom_value_id_t* out_value) {
  if (payload->carrier->element_type == LOOM_SCALAR_TYPE_I1) {
    return loom_spirv_workgroup_constant(
        context, location,
        payload->arithmetic->identity_bits
            ? SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_TRUE_BOOL
            : SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_FALSE_BOOL,
        LOOM_STRING_ID_INVALID, 0, payload->arithmetic_type, out_value);
  }
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_constant(
      context, location, payload->carrier->constant, payload->constant_key,
      payload->arithmetic->identity_bits, payload->constant_type, out_value));
  if (payload->carrier->input_cast != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
        context, location, payload->carrier->input_cast, out_value, 1,
        payload->arithmetic_type, out_value));
  }
  return iree_ok_status();
}

// SPIR-V has logical subgroup arithmetic but no Boolean Workgroup arrays.
// Integer storage is local to this boundary; both reduction levels stay
// logical.
static iree_status_t loom_spirv_workgroup_storage_value(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_workgroup_payload_t* payload, loom_value_id_t value,
    bool storing, loom_value_id_t zero, loom_value_id_t one,
    loom_value_id_t* out_value) {
  if (payload->carrier->element_type != LOOM_SCALAR_TYPE_I1) {
    *out_value = value;
    return iree_ok_status();
  }
  if (storing) {
    return loom_spirv_workgroup_emit(
        context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_SELECT_I32,
        (loom_value_id_t[]){value, one, zero}, 3, payload->storage_type,
        out_value);
  }
  return loom_spirv_workgroup_emit(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_I_NOT_EQUAL_I32,
      (loom_value_id_t[]){value, zero}, 2, payload->arithmetic_type, out_value);
}

// Both subgroup stages use the retained source arithmetic policy. In
// particular, a singleton workgroup's final fold may remain uniform too.
static iree_status_t loom_spirv_workgroup_reduce_subgroup(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_workgroup_payload_t* payload, loom_value_id_t value,
    loom_value_id_t* out_value) {
  if (payload->product) {
    return loom_spirv_collective_product_emit(
        context, location, payload->product, LOOM_SPIRV_COLLECTIVE_REDUCE,
        loom_named_attr_slice_empty(), value, out_value);
  }
  return loom_spirv_workgroup_emit(context, location,
                                   payload->arithmetic->subgroup, &value, 1,
                                   payload->arithmetic_type, out_value);
}

static iree_status_t loom_spirv_workgroup_combine(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_workgroup_payload_t* payload, loom_type_t predicate_type,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  const bool selects =
      payload->arithmetic->select != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  loom_value_id_t combined;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location, payload->arithmetic->combine,
      (loom_value_id_t[]){lhs, rhs}, 2,
      selects ? predicate_type : payload->arithmetic_type, &combined));
  if (selects) {
    return loom_spirv_workgroup_emit(context, location,
                                     payload->arithmetic->select,
                                     (loom_value_id_t[]){combined, lhs, rhs}, 3,
                                     payload->arithmetic_type, out_value);
  }
  *out_value = combined;
  return iree_ok_status();
}

typedef struct loom_spirv_workgroup_component_t {
  // Retained payload representation shared by this packet's scalar components.
  const loom_spirv_workgroup_payload_t* payload;
  // Typed Workgroup array for subgroup partials and the separate final slot.
  loom_value_id_t array;
  // Final-result address, whose reads are retired by the next first barrier.
  loom_value_id_t result_pointer;
  // Subgroup partial converted to the typed scratch representation.
  loom_value_id_t partial;
  // Ordered-key constants reused when decoding floating extrema.
  loom_spirv_extrema_constants_t extrema_constants;
} loom_spirv_workgroup_component_t;

static iree_status_t loom_spirv_workgroup_prepare_components(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_reduce_plan_t* plan, loom_value_id_t zero,
    loom_value_id_t one, loom_spirv_workgroup_component_t* components,
    loom_value_id_t* values) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_location_id_t location = source_op->location;
  uint16_t position = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = &plan->payloads[i];
    const loom_value_id_t packet = loom_low_lower_lookup_value(
        context, loom_op_const_operands(source_op)[i]);
    loom_value_id_t identity;
    status =
        loom_spirv_workgroup_identity(context, location, payload, &identity);
    for (uint8_t j = 0;
         j < payload->component_count && iree_status_is_ok(status);
         ++j, ++position) {
      loom_spirv_workgroup_component_t* component = &components[position];
      *component = (loom_spirv_workgroup_component_t){.payload = payload};
      values[position] = identity;
      loom_value_id_t value = packet;
      if (payload->component_count > 1) {
        const loom_named_attr_t attribute = {.name_id = plan->component_key,
                                             .value = loom_attr_i64(j)};
        status = loom_spirv_workgroup_emit_attributes(
            context, location,
            payload->source->extract[payload->component_count - 2], &packet, 1,
            loom_make_named_attr_slice(&attribute, 1), payload->source_type,
            &value);
      }
      if (iree_status_is_ok(status) && payload->extrema) {
        status = loom_spirv_collective_extrema_encode(
            context, location, payload->extrema, value, false,
            &component->extrema_constants, &value);
      }
      if (iree_status_is_ok(status) &&
          payload->carrier->input_cast != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        status = loom_spirv_workgroup_emit(context, location,
                                           payload->carrier->input_cast, &value,
                                           1, payload->arithmetic_type, &value);
      }
      if (iree_status_is_ok(status)) {
        status = loom_spirv_workgroup_reduce_subgroup(context, location,
                                                      payload, value, &value);
      }
      if (iree_status_is_ok(status)) {
        status = loom_spirv_workgroup_storage_value(context, location, payload,
                                                    value, true, zero, one,
                                                    &component->partial);
      }
      loom_op_t* storage = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_low_storage_reserve_build(
            builder,
            ((uint64_t)plan->capacity + 1) * payload->carrier->byte_count,
            payload->carrier->byte_count,
            loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), location,
            &storage);
      }
      loom_op_t* address = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_low_storage_address_build(
            builder, loom_low_storage_reserve_storage(storage), 0,
            payload->array_type, location, &address);
      }
      if (iree_status_is_ok(status)) {
        component->array = loom_low_storage_address_result(address);
      }
    }
  }
  return status;
}

static iree_status_t loom_spirv_workgroup_fold_partials(
    loom_low_lower_context_t* context, loom_location_id_t location,
    const loom_spirv_workgroup_reduce_plan_t* plan,
    const loom_spirv_workgroup_component_t* components, loom_value_id_t rank,
    loom_value_id_t subgroup_count, loom_value_id_t zero, loom_value_id_t one,
    loom_value_id_t* values) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_value_id_t active_count;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_IADD_REDUCE_I32,
      &one, 1, plan->integer_type, &active_count));
  loom_op_t* fold = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_for_build(
      builder, 0, LOOM_LOW_SCF_FOR_SIGNEDNESS_UNSIGNED, rank, subgroup_count,
      active_count, values, plan->component_count, NULL, NULL, 0,
      LOOM_VALUE_ID_INVALID, 0, location, &fold));
  const loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, fold, loom_low_scf_for_body(fold));
  const loom_value_id_t position =
      loom_region_entry_arg_id(loom_low_scf_for_body(fold), 0);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->component_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = components[i].payload;
    const loom_value_id_t accumulator =
        loom_region_entry_arg_id(loom_low_scf_for_body(fold), i + 1);
    loom_value_id_t pointer, loaded;
    status = loom_spirv_workgroup_emit(
        context, location, payload->carrier->address,
        (loom_value_id_t[]){components[i].array, position}, 2,
        payload->pointer_type, &pointer);
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_emit(context, location,
                                         payload->carrier->load, &pointer, 1,
                                         payload->storage_type, &loaded);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_storage_value(
          context, location, payload, loaded, false, zero, one, &loaded);
    }
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_combine(context, location, payload,
                                            plan->predicate_type, accumulator,
                                            loaded, &values[i]);
    }
  }
  loom_op_t* terminator = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_scf_yield_build(builder, values, plan->component_count,
                                      location, &terminator);
  }
  loom_builder_restore(builder, saved_ip);
  for (uint16_t i = 0; i < plan->component_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = components[i].payload;
    const loom_value_id_t local_total =
        loom_low_scf_for_results(fold).values[i];
    status = loom_spirv_workgroup_reduce_subgroup(context, location, payload,
                                                  local_total, &values[i]);
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_storage_value(
          context, location, payload, values[i], true, zero, one, &values[i]);
    }
  }
  return status;
}

static iree_status_t loom_spirv_workgroup_load_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_spirv_workgroup_reduce_plan_t* plan,
    const loom_spirv_workgroup_component_t* components, loom_value_id_t zero,
    loom_value_id_t one) {
  const loom_location_id_t location = source_op->location;
  uint16_t position = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->payload_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = &plan->payloads[i];
    loom_value_id_t values[4];
    for (uint8_t j = 0;
         j < payload->component_count && iree_status_is_ok(status);
         ++j, ++position) {
      const loom_spirv_workgroup_component_t* component = &components[position];
      status = loom_spirv_workgroup_emit(
          context, location, payload->carrier->load, &component->result_pointer,
          1, payload->storage_type, &values[j]);
      if (iree_status_is_ok(status)) {
        status = loom_spirv_workgroup_storage_value(context, location, payload,
                                                    values[j], false, zero, one,
                                                    &values[j]);
      }
      if (iree_status_is_ok(status) && payload->extrema) {
        status = loom_spirv_collective_extrema_decode(
            context, location, payload->extrema, component->extrema_constants,
            values[j], false, &values[j]);
      }
      if (iree_status_is_ok(status) &&
          payload->carrier->output_cast != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        status = loom_spirv_workgroup_emit(
            context, location, payload->carrier->output_cast, &values[j], 1,
            payload->source_type, &values[j]);
      }
    }
    if (iree_status_is_ok(status) && payload->component_count > 1) {
      status = loom_spirv_workgroup_emit(
          context, location,
          payload->source->construct[payload->component_count - 2], values,
          payload->component_count, payload->packet_type, &values[0]);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_bind_value(
          context, loom_op_const_results(source_op)[i], values[0]);
    }
  }
  return status;
}

iree_status_t loom_spirv_emit_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t selected_plan) {
  const loom_spirv_workgroup_reduce_plan_t* plan = selected_plan.target_data;
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_location_id_t location = source_op->location;
  loom_spirv_workgroup_component_t* components = NULL;
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, plan->component_count, sizeof(*components),
      (void**)&components));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, plan->component_count, sizeof(*values), (void**)&values));
  loom_value_id_t zero, one, subgroup, subgroup_count, rank;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_constant(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32,
      plan->value_key, 0, plan->integer_type, &zero));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_constant(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32,
      plan->value_key, 1, plan->integer_type, &one));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_BUILTIN_SUBGROUP_ID, NULL, 0,
      plan->integer_type, &subgroup));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_LOAD_BUILTIN_SUBGROUP_COUNT, NULL, 0,
      plan->integer_type, &subgroup_count));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_GROUP_NON_UNIFORM_IADD_EXCLUSIVE_SCAN_I32,
      &one, 1, plan->integer_type, &rank));
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_prepare_components(
      context, source_op, plan, zero, one, components, values));
  loom_value_id_t leader;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_I_EQUAL_I32,
      (loom_value_id_t[]){rank, zero}, 2, plan->predicate_type, &leader));
  loom_op_t* publish = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_if_build(builder, 0, leader, NULL, 0, NULL,
                                             0, location, &publish));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      builder, publish, loom_low_scf_if_then_region(publish));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < plan->component_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = components[i].payload;
    loom_value_id_t pointer;
    status = loom_spirv_workgroup_emit(
        context, location, payload->carrier->address,
        (loom_value_id_t[]){components[i].array, subgroup}, 2,
        payload->pointer_type, &pointer);
    if (iree_status_is_ok(status)) {
      status = loom_spirv_workgroup_emit(
          context, location, payload->carrier->store,
          (loom_value_id_t[]){pointer, components[i].partial}, 2,
          loom_type_none(), NULL);
    }
  }
  loom_op_t* terminator = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_scf_yield_build(builder, NULL, 0, location, &terminator);
  }
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONTROL_BARRIER_WORKGROUP_WORKGROUP_ACQ_REL,
      NULL, 0, loom_type_none(), NULL));
  loom_value_id_t result_position;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_constant(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONSTANT_I32,
      plan->value_key, plan->capacity, plan->integer_type, &result_position));
  for (uint16_t i = 0; i < plan->component_count && iree_status_is_ok(status);
       ++i) {
    const loom_spirv_workgroup_payload_t* payload = components[i].payload;
    status = loom_spirv_workgroup_emit(
        context, location, payload->carrier->address,
        (loom_value_id_t[]){components[i].array, result_position}, 2,
        payload->pointer_type, &components[i].result_pointer);
  }
  IREE_RETURN_IF_ERROR(status);
  loom_value_id_t first_subgroup;
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location, SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_I_EQUAL_I32,
      (loom_value_id_t[]){subgroup, zero}, 2, plan->predicate_type,
      &first_subgroup));
  loom_op_t* combine = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_if_build(builder, 0, first_subgroup, NULL,
                                             0, NULL, 0, location, &combine));
  saved_ip = loom_builder_enter_region(builder, combine,
                                       loom_low_scf_if_then_region(combine));
  status = loom_spirv_workgroup_fold_partials(context, location, plan,
                                              components, rank, subgroup_count,
                                              zero, one, values);
  loom_op_t* publish_total = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_scf_if_build(builder, 0, leader, NULL, 0, NULL, 0,
                                   location, &publish_total);
  }
  if (iree_status_is_ok(status)) {
    const loom_builder_ip_t publish_ip = loom_builder_enter_region(
        builder, publish_total, loom_low_scf_if_then_region(publish_total));
    for (uint16_t i = 0; i < plan->component_count && iree_status_is_ok(status);
         ++i) {
      status = loom_spirv_workgroup_emit(
          context, location, components[i].payload->carrier->store,
          (loom_value_id_t[]){components[i].result_pointer, values[i]}, 2,
          loom_type_none(), NULL);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_low_scf_yield_build(builder, NULL, 0, location, &terminator);
    }
    loom_builder_restore(builder, publish_ip);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_scf_yield_build(builder, NULL, 0, location, &terminator);
  }
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);
  // Distinct partial and final slots make repeated calls safe. The first
  // barrier retires prior result reads before overwriting the final slot; this
  // barrier retires partial reads before the next call's partial writes.
  IREE_RETURN_IF_ERROR(loom_spirv_workgroup_emit(
      context, location,
      SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_OP_CONTROL_BARRIER_WORKGROUP_WORKGROUP_ACQ_REL,
      NULL, 0, loom_type_none(), NULL));
  return loom_spirv_workgroup_load_results(context, source_op, plan, components,
                                           zero, one);
}
