// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/materialization.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/projection.h"

static bool loom_target_record_projection_bundle_is_valid(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL;
}

static void loom_target_record_projection_initialize_storage(
    const loom_target_bundle_t* bundle, iree_string_view_t record_name,
    loom_target_bundle_storage_t* out_storage) {
  *out_storage = (loom_target_bundle_storage_t){
      .snapshot = *bundle->snapshot,
      .export_plan = *bundle->export_plan,
      .config = *bundle->config,
  };
  out_storage->snapshot.name = record_name;
  out_storage->export_plan.name = record_name;
  out_storage->config.name = record_name;
  out_storage->bundle.name = record_name;
  loom_target_bundle_storage_rebind(out_storage);
}

static void loom_target_record_projection_apply_attr(
    const loom_module_t* module, const loom_op_t* op,
    const loom_target_projection_t* projection,
    loom_target_bundle_storage_t* storage) {
  const loom_attribute_t attr = loom_op_const_attrs(op)[projection->attr_index];
  if (loom_attr_is_absent(attr)) {
    return;
  }

  uint8_t* storage_base = (uint8_t*)storage;
  void* destination = storage_base + projection->storage_offset;
  switch (projection->value_kind) {
    case LOOM_TARGET_PROJECTION_VALUE_ENUM_U8:
      *(uint8_t*)destination = (uint8_t)loom_attr_as_enum(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32:
      *(uint32_t*)destination = (uint32_t)loom_attr_as_i64(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64:
      *(uint64_t*)destination = (uint64_t)loom_attr_as_i64(attr);
      break;
    case LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW:
      *(iree_string_view_t*)destination =
          module->strings.entries[loom_attr_as_string_id(attr)];
      break;
  }
}

static void loom_target_record_projection_apply_authored_specialization(
    const loom_module_t* module, const loom_op_t* authored_target_op,
    const loom_target_like_descriptor_t* descriptor,
    loom_target_bundle_storage_t* storage) {
  if (authored_target_op == NULL) {
    return;
  }
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    const loom_target_projection_t* projection = &descriptor->projections[i];
    if (projection->specialization !=
        LOOM_TARGET_PROJECTION_SPECIALIZATION_AUTHORED) {
      continue;
    }
    loom_target_record_projection_apply_attr(module, authored_target_op,
                                             projection, storage);
  }
}

bool loom_target_record_projection_resolve(
    const loom_module_t* module, loom_target_like_t target,
    iree_string_view_t record_name, loom_target_bundle_storage_t* out_storage,
    loom_target_fact_field_set_t* out_authored_fields) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_target_bundle_storage_t){0};
  if (out_authored_fields != NULL) {
    *out_authored_fields = 0;
  }
  if (!loom_target_like_isa(target)) {
    return false;
  }

  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  const uint8_t selector =
      (uint8_t)loom_attr_as_enum(loom_target_like_selector(target));
  const loom_target_bundle_t* row_bundle =
      loom_target_bundle_table_lookup(descriptor->bundle_table, selector);
  if (!loom_target_record_projection_bundle_is_valid(row_bundle)) {
    return false;
  }

  loom_target_record_projection_initialize_storage(row_bundle, record_name,
                                                   out_storage);
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    const loom_target_projection_t* projection = &descriptor->projections[i];
    if (out_authored_fields != NULL &&
        !loom_attr_is_absent(
            loom_op_const_attrs(target.op)[projection->attr_index])) {
      loom_target_fact_field_set_insert(out_authored_fields,
                                        projection->fact_field);
    }
    loom_target_record_projection_apply_attr(module, target.op, projection,
                                             out_storage);
  }
  return true;
}

static bool loom_target_record_projection_values_equal(
    const loom_target_projection_t* projection,
    const loom_target_bundle_storage_t* lhs,
    const loom_target_bundle_storage_t* rhs) {
  const uint8_t* lhs_base = (const uint8_t*)lhs;
  const uint8_t* rhs_base = (const uint8_t*)rhs;
  const void* lhs_value = lhs_base + projection->storage_offset;
  const void* rhs_value = rhs_base + projection->storage_offset;
  switch (projection->value_kind) {
    case LOOM_TARGET_PROJECTION_VALUE_ENUM_U8:
      return *(const uint8_t*)lhs_value == *(const uint8_t*)rhs_value;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32:
      return *(const uint32_t*)lhs_value == *(const uint32_t*)rhs_value;
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64:
      return *(const uint64_t*)lhs_value == *(const uint64_t*)rhs_value;
    case LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW:
      return iree_string_view_equal(*(const iree_string_view_t*)lhs_value,
                                    *(const iree_string_view_t*)rhs_value);
    default:
      return false;
  }
}

bool loom_target_record_projection_matches_bundle(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_bundle_t* selected_bundle,
    const loom_op_t* authored_target_op) {
  if (!module || !target_op ||
      !loom_target_record_projection_bundle_is_valid(selected_bundle)) {
    return false;
  }
  const loom_target_like_t target = loom_target_like_cast(module, target_op);
  if (!loom_target_like_isa(target)) {
    return false;
  }

  loom_target_bundle_storage_t existing_storage = {0};
  if (!loom_target_record_projection_resolve(
          module, target, iree_string_view_empty(), &existing_storage,
          /*out_authored_fields=*/NULL)) {
    return false;
  }
  loom_target_bundle_storage_t selected_storage = {0};
  loom_target_record_projection_initialize_storage(
      selected_bundle, iree_string_view_empty(), &selected_storage);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  if (authored_target_op != NULL &&
      authored_target_op->kind != target_op->kind) {
    return false;
  }
  loom_target_record_projection_apply_authored_specialization(
      module, authored_target_op, descriptor, &selected_storage);

  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    if (!loom_target_record_projection_values_equal(&descriptor->projections[i],
                                                    &existing_storage,
                                                    &selected_storage)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_target_record_projection_encode_attr(
    loom_module_t* module, const loom_target_projection_t* projection,
    const loom_target_bundle_storage_t* selected_storage,
    loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  const uint8_t* storage_base = (const uint8_t*)selected_storage;
  const void* value = storage_base + projection->storage_offset;
  switch (projection->value_kind) {
    case LOOM_TARGET_PROJECTION_VALUE_ENUM_U8:
      *out_attr = loom_attr_enum(*(const uint8_t*)value);
      return iree_ok_status();
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U32:
      *out_attr = loom_attr_i64(*(const uint32_t*)value);
      return iree_ok_status();
    case LOOM_TARGET_PROJECTION_VALUE_I64_TO_U64: {
      const uint64_t integer_value = *(const uint64_t*)value;
      if (integer_value > (uint64_t)INT64_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "target materialization attribute %u value %" PRIu64
            " exceeds the target-record i64 range",
            (unsigned)projection->attr_index, integer_value);
      }
      *out_attr = loom_attr_i64((int64_t)integer_value);
      return iree_ok_status();
    }
    case LOOM_TARGET_PROJECTION_VALUE_STRING_VIEW: {
      loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_module_intern_string(
          module, *(const iree_string_view_t*)value, &string_id));
      *out_attr = loom_attr_string(string_id);
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target materialization attribute %u has unknown projection kind %u",
          (unsigned)projection->attr_index, (unsigned)projection->value_kind);
  }
}

iree_status_t loom_target_record_projection_build(
    loom_builder_t* builder, loom_op_kind_t op_kind, uint8_t selector,
    loom_symbol_ref_t symbol, const loom_target_bundle_t* selected_bundle,
    const loom_op_t* authored_target_op,
    const loom_target_record_extension_attr_t* extension_attrs,
    iree_host_size_t extension_attr_count, loom_location_id_t location,
    loom_op_t** out_target_op) {
  if (!builder || !builder->module || !out_target_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target record projection requires a builder and output");
  }
  *out_target_op = NULL;
  if (!loom_target_record_projection_bundle_is_valid(selected_bundle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target record projection requires a complete selected bundle");
  }
  if (extension_attr_count != 0 && extension_attrs == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target record projection extension attributes are missing");
  }
  if (authored_target_op != NULL && authored_target_op->kind != op_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "authored target op kind 0x%04X does not match specialized op kind "
        "0x%04X",
        (unsigned)authored_target_op->kind, (unsigned)op_kind);
  }

  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(builder->module->context, op_kind);
  if (!vtable || !vtable->target_like) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target record projection op kind 0x%04X is not target-like",
        (unsigned)op_kind);
  }
  const loom_op_vtable_flags_t structural_flags =
      LOOM_OP_VTABLE_VARIADIC_OPERANDS | LOOM_OP_VTABLE_VARIADIC_RESULTS |
      LOOM_OP_VTABLE_VARIADIC_REGIONS | LOOM_OP_VTABLE_SEGMENTED_OPERANDS;
  if (vtable->fixed_operand_count != 0 || vtable->fixed_result_count != 0 ||
      vtable->region_count != 0 ||
      iree_any_bit_set(vtable->vtable_flags, structural_flags)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target record projection op kind 0x%04X is not metadata-only",
        (unsigned)op_kind);
  }

  const loom_target_like_vtable_t* target_like = vtable->target_like;
  const loom_target_like_descriptor_t* descriptor = target_like->descriptor;
  const loom_target_bundle_t* row_bundle =
      loom_target_bundle_table_lookup(descriptor->bundle_table, selector);
  if (!loom_target_record_projection_bundle_is_valid(row_bundle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target record projection selector %u has no generated row",
        (unsigned)selector);
  }

  loom_target_bundle_storage_t row_storage = {0};
  loom_target_record_projection_initialize_storage(
      row_bundle, iree_string_view_empty(), &row_storage);
  loom_target_bundle_storage_t selected_storage = {0};
  loom_target_record_projection_initialize_storage(
      selected_bundle, iree_string_view_empty(), &selected_storage);
  loom_target_record_projection_apply_authored_specialization(
      builder->module, authored_target_op, descriptor, &selected_storage);

  loom_attribute_t attrs[UINT8_MAX + 1] = {0};
  attrs[target_like->symbol_attr_index] = loom_attr_symbol(symbol);
  attrs[target_like->selector_attr_index] = loom_attr_enum(selector);
  for (uint8_t i = 0; i < descriptor->projection_count; ++i) {
    const loom_target_projection_t* projection = &descriptor->projections[i];
    if (loom_target_record_projection_values_equal(projection, &row_storage,
                                                   &selected_storage)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_target_record_projection_encode_attr(
        builder->module, projection, &selected_storage,
        &attrs[projection->attr_index]));
  }

  for (iree_host_size_t i = 0; i < extension_attr_count; ++i) {
    const loom_target_record_extension_attr_t extension = extension_attrs[i];
    if (extension.attr_index >= vtable->attribute_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target record extension attribute index %u exceeds op attribute "
          "count %u",
          (unsigned)extension.attr_index, (unsigned)vtable->attribute_count);
    }
    if (loom_attr_is_absent(extension.value)) {
      continue;
    }
    if (!loom_attr_is_absent(attrs[extension.attr_index])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target record extension attribute index %u overlaps a common or "
          "identity attribute",
          (unsigned)extension.attr_index);
    }
    attrs[extension.attr_index] = extension.value;
  }

  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, op_kind, 0, 0, 0, 0,
                                                vtable->attribute_count,
                                                location, out_target_op));
  memcpy(loom_op_attrs(*out_target_op), attrs,
         vtable->attribute_count * sizeof(attrs[0]));
  return loom_builder_finalize_op(builder, *out_target_op);
}
