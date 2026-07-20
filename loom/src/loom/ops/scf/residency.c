// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scf/residency.h"

#include "loom/ir/context.h"
#include "loom/ops/scf/ops.h"

#define LOOM_SCF_RESIDENCY_SOURCE_WITNESS_VERSION 1

static const iree_string_view_t kWitnessVersion = IREE_SVL("version");
static const iree_string_view_t kWitnessOpName = IREE_SVL("op_name");
static const iree_string_view_t kWitnessResultIndex = IREE_SVL("result_index");
static const iree_string_view_t kWitnessInstanceFlags =
    IREE_SVL("instance_flags");
static const iree_string_view_t kWitnessOperandCount =
    IREE_SVL("operand_count");
static const iree_string_view_t kWitnessResultCount = IREE_SVL("result_count");
static const iree_string_view_t kWitnessTiedResultCount =
    IREE_SVL("tied_result_count");
static const iree_string_view_t kWitnessRegionCount = IREE_SVL("region_count");
static const iree_string_view_t kWitnessSuccessorCount =
    IREE_SVL("successor_count");
static const iree_string_view_t kWitnessAttributes = IREE_SVL("attributes");

static iree_status_t loom_scf_residency_make_named_attr(
    loom_module_t* module, iree_string_view_t name, loom_attribute_t value,
    loom_named_attr_t* out_entry) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, name, &name_id));
  *out_entry = (loom_named_attr_t){
      .name_id = name_id,
      .value = value,
  };
  return iree_ok_status();
}

static iree_status_t loom_scf_residency_capture_source_attributes(
    loom_builder_t* builder, const loom_op_t* source_op,
    loom_attribute_t* out_attributes) {
  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, source_op);
  if (vtable == NULL || vtable->attribute_count != source_op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency source producer has inconsistent attribute metadata");
  }
  loom_named_attr_t* entries = NULL;
  if (source_op->attribute_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, source_op->attribute_count,
                                  sizeof(*entries), (void**)&entries));
  }
  iree_host_size_t entry_count = 0;
  const loom_attribute_t* attributes = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    if (loom_attr_is_absent(attributes[i])) continue;
    IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
        builder->module,
        loom_attr_descriptor_name(&vtable->attr_descriptors[i]), attributes[i],
        &entries[entry_count++]));
  }
  return loom_module_make_canonical_attr_dict(
      builder->module, loom_make_named_attr_slice(entries, entry_count),
      out_attributes);
}

iree_status_t loom_scf_residency_candidate_build_for_proven_source(
    loom_builder_t* builder, int64_t candidate_id, int64_t recompute_cost,
    const loom_op_t* source_op, uint16_t source_result_index,
    bool preserves_baseline, loom_location_id_t location,
    loom_op_t** out_candidate_op) {
  if (builder == NULL || builder->module == NULL || source_op == NULL ||
      out_candidate_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency candidate source builder requires non-NULL inputs");
  }
  *out_candidate_op = NULL;
  if (iree_any_bit_set(source_op->flags, LOOM_OP_FLAG_DEAD) ||
      source_result_index >= source_op->result_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "residency candidate source result is unavailable");
  }
  if (source_op->region_count != 0 || source_op->successor_count != 0 ||
      source_op->tied_result_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate source is not a finite expression recipe");
  }

  loom_attribute_t source_attributes = {0};
  IREE_RETURN_IF_ERROR(loom_scf_residency_capture_source_attributes(
      builder, source_op, &source_attributes));
  const loom_value_id_t source =
      loom_op_const_results(source_op)[source_result_index];
  const loom_type_t source_type =
      loom_module_value_type(builder->module, source);
  loom_string_id_t op_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, loom_op_name(builder->module, source_op), &op_name_id));

  loom_named_attr_t witness_entries[10];
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessVersion,
      loom_attr_i64(LOOM_SCF_RESIDENCY_SOURCE_WITNESS_VERSION),
      &witness_entries[0]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessOpName, loom_attr_string(op_name_id),
      &witness_entries[1]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessResultIndex, loom_attr_i64(source_result_index),
      &witness_entries[2]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessInstanceFlags,
      loom_attr_i64(source_op->instance_flags), &witness_entries[3]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessOperandCount,
      loom_attr_i64(source_op->operand_count), &witness_entries[4]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessResultCount,
      loom_attr_i64(source_op->result_count), &witness_entries[5]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessTiedResultCount,
      loom_attr_i64(source_op->tied_result_count), &witness_entries[6]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessRegionCount,
      loom_attr_i64(source_op->region_count), &witness_entries[7]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessSuccessorCount,
      loom_attr_i64(source_op->successor_count), &witness_entries[8]));
  IREE_RETURN_IF_ERROR(loom_scf_residency_make_named_attr(
      builder->module, kWitnessAttributes, source_attributes,
      &witness_entries[9]));
  loom_attribute_t source_witness = {0};
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      builder->module,
      loom_make_named_attr_slice(witness_entries,
                                 IREE_ARRAYSIZE(witness_entries)),
      &source_witness));

  return loom_scf_residency_candidate_build(
      builder, candidate_id, recompute_cost, source,
      loom_op_const_operands(source_op), source_op->operand_count, source_type,
      source_witness, preserves_baseline, location, out_candidate_op);
}

static const loom_named_attr_t* loom_scf_residency_find_witness_entry(
    const loom_module_t* module, loom_named_attr_slice_t witness,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < witness.count; ++i) {
    if (witness.entries[i].name_id >= module->strings.count) continue;
    if (iree_string_view_equal(
            module->strings.entries[witness.entries[i].name_id], name)) {
      return &witness.entries[i];
    }
  }
  return NULL;
}

static iree_status_t loom_scf_residency_witness_i64(
    const loom_module_t* module, loom_named_attr_slice_t witness,
    iree_string_view_t name, int64_t* out_value) {
  const loom_named_attr_t* entry =
      loom_scf_residency_find_witness_entry(module, witness, name);
  if (entry == NULL || entry->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate source witness field '%.*s' is missing or "
        "malformed",
        (int)name.size, name.data);
  }
  *out_value = entry->value.i64;
  return iree_ok_status();
}

static bool loom_scf_residency_source_attributes_equal(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_attribute_t witness_attributes) {
  if (witness_attributes.kind != LOOM_ATTR_DICT) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, source_op);
  if (vtable == NULL || vtable->attribute_count != source_op->attribute_count) {
    return false;
  }
  iree_host_size_t present_count = 0;
  const loom_attribute_t* attributes = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    if (loom_attr_is_absent(attributes[i])) continue;
    ++present_count;
    const iree_string_view_t name =
        loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
    const loom_named_attr_t* witness_entry =
        loom_scf_residency_find_witness_entry(
            module,
            loom_make_named_attr_slice(witness_attributes.dict_entries,
                                       witness_attributes.count),
            name);
    if (witness_entry == NULL ||
        !loom_attribute_equal(&attributes[i], &witness_entry->value)) {
      return false;
    }
  }
  return present_count == witness_attributes.count;
}

iree_status_t loom_scf_residency_candidate_validate_proven_source(
    const loom_module_t* module, const loom_op_t* candidate_op) {
  if (module == NULL || candidate_op == NULL ||
      !loom_scf_residency_candidate_isa(candidate_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "residency candidate validation requires a candidate operation");
  }
  loom_value_id_t source = loom_scf_residency_candidate_source(candidate_op);
  for (iree_host_size_t depth = 0; depth <= module->values.count; ++depth) {
    if (source == LOOM_VALUE_ID_INVALID || source >= module->values.count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "residency candidate source is invalid");
    }
    const loom_value_t* value = loom_module_value(module, source);
    if (loom_value_is_block_arg(value)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "residency candidate source producer was replaced by a block "
          "argument");
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!loom_scf_residency_candidate_isa(defining_op)) break;
    source = loom_scf_residency_candidate_source(defining_op);
    if (depth == module->values.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "residency candidate source marker graph contains a cycle");
    }
  }

  const loom_value_t* source_value = loom_module_value(module, source);
  const loom_op_t* source_op = loom_value_def_op(source_value);
  if (source_op == NULL ||
      iree_any_bit_set(source_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate source producer is unavailable");
  }
  const loom_attribute_t witness_attr =
      loom_scf_residency_candidate_source_witness(candidate_op);
  if (witness_attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate has no complete source legality witness");
  }
  const loom_named_attr_slice_t witness = loom_attr_as_dict(witness_attr);
  if (witness.count != 10) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate has no complete source legality witness");
  }

  int64_t version = 0;
  int64_t result_index = 0;
  int64_t instance_flags = 0;
  int64_t operand_count = 0;
  int64_t result_count = 0;
  int64_t tied_result_count = 0;
  int64_t region_count = 0;
  int64_t successor_count = 0;
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessVersion, &version));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessResultIndex, &result_index));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessInstanceFlags, &instance_flags));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessOperandCount, &operand_count));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessResultCount, &result_count));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessTiedResultCount, &tied_result_count));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessRegionCount, &region_count));
  IREE_RETURN_IF_ERROR(loom_scf_residency_witness_i64(
      module, witness, kWitnessSuccessorCount, &successor_count));

  const loom_named_attr_t* op_name_entry =
      loom_scf_residency_find_witness_entry(module, witness, kWitnessOpName);
  const loom_named_attr_t* attributes_entry =
      loom_scf_residency_find_witness_entry(module, witness,
                                            kWitnessAttributes);
  if (op_name_entry == NULL || op_name_entry->value.kind != LOOM_ATTR_STRING ||
      op_name_entry->value.string_id >= module->strings.count ||
      attributes_entry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate source witness has malformed identity fields");
  }

  const loom_value_slice_t captures =
      loom_scf_residency_candidate_captures(candidate_op);
  const bool shape_matches =
      version == LOOM_SCF_RESIDENCY_SOURCE_WITNESS_VERSION &&
      result_index >= 0 && result_index < source_op->result_count &&
      loom_op_const_results(source_op)[result_index] == source &&
      instance_flags == source_op->instance_flags &&
      operand_count == source_op->operand_count &&
      result_count == source_op->result_count &&
      tied_result_count == source_op->tied_result_count &&
      region_count == source_op->region_count &&
      successor_count == source_op->successor_count &&
      captures.count == source_op->operand_count &&
      iree_string_view_equal(
          module->strings.entries[op_name_entry->value.string_id],
          loom_op_name(module, source_op)) &&
      loom_scf_residency_source_attributes_equal(module, source_op,
                                                 attributes_entry->value);
  if (!shape_matches) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency candidate source producer changed after placement");
  }
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  for (iree_host_size_t i = 0; i < captures.count; ++i) {
    if (captures.values[i] != source_operands[i]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "residency candidate source dependency closure changed after "
          "placement");
    }
  }

  return iree_ok_status();
}
