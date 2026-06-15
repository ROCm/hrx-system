// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "loom/codegen/low/text_asm_internal.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static const uint8_t kLowAsmResourceIndexName[] =
    LOOM_BSTRING_LITERAL(5, "index");
static const uint8_t kLowAsmResourceSemanticTypeName[] =
    LOOM_BSTRING_LITERAL(13, "source_type");
static const uint8_t kLowAsmResourceExtentName[] =
    LOOM_BSTRING_LITERAL(6, "extent");
static const uint8_t kLowAsmResourceCacheSwizzleStrideName[] =
    LOOM_BSTRING_LITERAL(20, "cache_swizzle_stride");
static const uint8_t kLowAsmStorageReserveByteLengthName[] =
    LOOM_BSTRING_LITERAL(11, "byte_length");
static const uint8_t kLowAsmStorageReserveByteAlignmentName[] =
    LOOM_BSTRING_LITERAL(14, "byte_alignment");
static const uint8_t kLowAsmStorageAddressOffsetName[] =
    LOOM_BSTRING_LITERAL(6, "offset");

static const loom_attr_descriptor_t kLowAsmResourceIndexAttr = {
    .name = kLowAsmResourceIndexName,
    .attr_kind = LOOM_ATTR_I64,
};
static const loom_attr_descriptor_t kLowAsmResourceSemanticTypeAttr = {
    .name = kLowAsmResourceSemanticTypeName,
    .attr_kind = LOOM_ATTR_TYPE,
};
static const loom_attr_descriptor_t kLowAsmResourceExtentAttr = {
    .name = kLowAsmResourceExtentName,
    .attr_kind = LOOM_ATTR_I64,
    .flags = LOOM_ATTR_OPTIONAL,
};
static const loom_attr_descriptor_t kLowAsmResourceCacheSwizzleStrideAttr = {
    .name = kLowAsmResourceCacheSwizzleStrideName,
    .attr_kind = LOOM_ATTR_I64,
    .flags = LOOM_ATTR_OPTIONAL,
};
static const loom_attr_descriptor_t kLowAsmStorageReserveByteLengthAttr = {
    .name = kLowAsmStorageReserveByteLengthName,
    .attr_kind = LOOM_ATTR_I64,
};
static const loom_attr_descriptor_t kLowAsmStorageReserveByteAlignmentAttr = {
    .name = kLowAsmStorageReserveByteAlignmentName,
    .attr_kind = LOOM_ATTR_I64,
};
static const loom_attr_descriptor_t kLowAsmStorageAddressOffsetAttr = {
    .name = kLowAsmStorageAddressOffsetName,
    .attr_kind = LOOM_ATTR_I64,
    .flags = LOOM_ATTR_ELIDE_DEFAULT,
};

static iree_status_t loom_low_descriptor_text_asm_resource_key_to_kind(
    iree_string_view_t key, uint8_t* out_kind) {
  if (iree_string_view_equal(key, IREE_SV("native_pointer"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("vm_state"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("vm_import"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("hal_binding"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown low asm resource kind '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_low_descriptor_text_asm_resource_kind_to_key(
    uint8_t kind, iree_string_view_t* out_key) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      *out_key = IREE_SV("native_pointer");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
      *out_key = IREE_SV("vm_state");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
      *out_key = IREE_SV("vm_import");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
      *out_key = IREE_SV("hal_binding");
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low resource import kind %u",
                              (unsigned)kind);
  }
}

iree_status_t loom_low_descriptor_text_asm_structural_attr_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor) {
  (void)state;
  *out_descriptor = NULL;
  if (kind != LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE) {
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE &&
        iree_string_view_equal(attr_name, IREE_SV("byte_length"))) {
      *out_descriptor = &kLowAsmStorageReserveByteLengthAttr;
      return iree_ok_status();
    }
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE &&
        iree_string_view_equal(attr_name, IREE_SV("byte_alignment"))) {
      *out_descriptor = &kLowAsmStorageReserveByteAlignmentAttr;
      return iree_ok_status();
    }
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS &&
        iree_string_view_equal(attr_name, IREE_SV("offset"))) {
      *out_descriptor = &kLowAsmStorageAddressOffsetAttr;
    }
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW &&
        iree_string_view_equal(attr_name, IREE_SV("offset"))) {
      *out_descriptor = &kLowAsmStorageAddressOffsetAttr;
    }
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW &&
        iree_string_view_equal(attr_name, IREE_SV("byte_length"))) {
      *out_descriptor = &kLowAsmStorageReserveByteLengthAttr;
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("index"))) {
    *out_descriptor = &kLowAsmResourceIndexAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("source_type"))) {
    *out_descriptor = &kLowAsmResourceSemanticTypeAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("extent"))) {
    *out_descriptor = &kLowAsmResourceExtentAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("cache_swizzle_stride"))) {
    *out_descriptor = &kLowAsmResourceCacheSwizzleStrideAttr;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_required_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const loom_named_attr_t** out_attr) {
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_lookup_attr(module, attrs, name, out_attr));
  if (*out_attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural intrinsic is missing "
                            "required attribute '%.*s'",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_build_resource(
    loom_builder_t* builder, iree_string_view_t key,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  uint8_t import_kind = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_resource_key_to_kind(key, &import_kind));

  const loom_named_attr_t* index_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("index"), &index_attr));
  if (index_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm resource index must be an i64 attr");
  }

  const loom_named_attr_t* source_type_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("source_type"), &source_type_attr));
  if (source_type_attr->value.kind != LOOM_ATTR_TYPE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm resource source_type must be a type attr");
  }

  loom_low_resource_build_flags_t flags = 0;
  int64_t extent = 0;
  const loom_named_attr_t* extent_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("extent"), &extent_attr));
  if (extent_attr != NULL) {
    if (extent_attr->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm resource extent must be an "
                              "i64 attr");
    }
    flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_EXTENT;
    extent = extent_attr->value.i64;
  }

  int64_t cache_swizzle_stride = 0;
  const loom_named_attr_t* cache_swizzle_stride_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("cache_swizzle_stride"),
      &cache_swizzle_stride_attr));
  if (cache_swizzle_stride_attr != NULL) {
    if (cache_swizzle_stride_attr->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm resource cache_swizzle_stride must be an i64 attr");
    }
    flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_CACHE_SWIZZLE_STRIDE;
    cache_swizzle_stride = cache_swizzle_stride_attr->value.i64;
  }

  return loom_low_resource_build(
      builder, flags, import_kind, LOOM_VALUE_ID_INVALID, index_attr->value.i64,
      source_type_attr->value.type_id, extent, cache_swizzle_stride,
      result_type, location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_storage_reserve(
    loom_builder_t* builder, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  const loom_named_attr_t* byte_length_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("byte_length"), &byte_length_attr));
  if (byte_length_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm storage byte_length must be an i64 attr");
  }

  const loom_named_attr_t* byte_alignment_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("byte_alignment"), &byte_alignment_attr));
  if (byte_alignment_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm storage byte_alignment must be an i64 attr");
  }

  return loom_low_storage_reserve_build(builder, byte_length_attr->value.i64,
                                        byte_alignment_attr->value.i64,
                                        result_type, location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_storage_address(
    loom_builder_t* builder, loom_value_id_t storage,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  const loom_named_attr_t* offset_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("offset"), &offset_attr));
  if (offset_attr && offset_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm storage_address offset must be an i64 attr");
  }
  int64_t offset = 0;
  if (offset_attr) {
    offset = offset_attr->value.i64;
  }

  return loom_low_storage_address_build(builder, storage, offset, result_type,
                                        location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_storage_view(
    loom_builder_t* builder, loom_value_id_t storage,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  const loom_named_attr_t* byte_length_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("byte_length"), &byte_length_attr));
  if (byte_length_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm storage_view byte_length must be an i64 attr");
  }

  const loom_named_attr_t* offset_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("offset"), &offset_attr));
  if (offset_attr != NULL && offset_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm storage_view offset must be an i64 attr");
  }
  int64_t offset = 0;
  if (offset_attr != NULL) {
    offset = offset_attr->value.i64;
  }

  return loom_low_storage_view_build(builder, storage, offset,
                                     byte_length_attr->value.i64, result_type,
                                     location, out_op);
}

iree_status_t loom_low_descriptor_text_asm_build_structural(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t key,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, int64_t offset, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  (void)state;
  switch (kind) {
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE:
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm resource takes no operands");
      }
      return loom_low_descriptor_text_asm_build_resource(
          builder, key, attributes, result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN: {
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm live_in takes no operands");
      }
      loom_string_id_t source = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_string(builder->module, key, &source));
      return loom_low_live_in_build(builder, source, attributes, result_type,
                                    location, out_op);
    }
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT:
      return loom_low_concat_build(builder, operands, operand_count,
                                   result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE:
      if (operand_count != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm slice takes one operand");
      }
      return loom_low_slice_build(builder, operands[0], offset, result_type,
                                  location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE:
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm storage takes no operands");
      }
      return loom_low_descriptor_text_asm_build_storage_reserve(
          builder, attributes, result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS:
      if (operand_count != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm storage_address takes one operand");
      }
      return loom_low_descriptor_text_asm_build_storage_address(
          builder, operands[0], attributes, result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW:
      if (operand_count != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm storage_view takes one operand");
      }
      return loom_low_descriptor_text_asm_build_storage_view(
          builder, operands[0], attributes, result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY:
      if (operand_count != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm copy takes one operand");
      }
      return loom_low_copy_build(builder, operands[0], result_type, location,
                                 out_op);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm structural kind %u",
                              (unsigned)kind);
  }
}

static iree_status_t loom_low_descriptor_text_asm_set_structural_attr(
    const loom_module_t* module, const loom_op_t* op, uint8_t attr_index,
    iree_string_view_t name, const loom_attr_descriptor_t* descriptor,
    loom_text_low_asm_statement_t* statement) {
  if (attr_index >= op->attribute_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural attr is out of range");
  }
  const loom_attribute_t* attr = &loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(*attr)) {
    return iree_ok_status();
  }
  if (statement->structural_attribute_count >=
      IREE_ARRAYSIZE(statement->structural_attributes)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low asm structural attr array is full");
  }
  statement->structural_attributes[statement->structural_attribute_count++] =
      (loom_text_low_asm_structural_attribute_t){
          .name = name,
          .value = attr,
          .descriptor = descriptor,
      };
  (void)module;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_resource(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_resource_kind_to_key(
      loom_low_resource_import_kind(op), &key));
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE,
      .structural_key = key,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_index_ATTR_INDEX, IREE_SV("index"),
      &kLowAsmResourceIndexAttr, out_statement));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_source_type_ATTR_INDEX,
      IREE_SV("source_type"), &kLowAsmResourceSemanticTypeAttr, out_statement));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_extent_ATTR_INDEX, IREE_SV("extent"),
      &kLowAsmResourceExtentAttr, out_statement));
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_cache_swizzle_stride_ATTR_INDEX,
      IREE_SV("cache_swizzle_stride"), &kLowAsmResourceCacheSwizzleStrideAttr,
      out_statement);
}

static iree_status_t loom_low_descriptor_text_asm_describe_live_in(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  loom_string_id_t source_id = loom_low_live_in_source(op);
  if (source_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.live_in source string is out of range");
  }
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN,
      .structural_key = module->strings.entries[source_id],
      .results = loom_op_const_results(op),
      .result_count = 1,
      .attributes = loom_low_live_in_attrs(op),
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_concat(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = sources.values,
      .operand_count = sources.count,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_slice(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE,
      .structural_offset = loom_low_slice_offset(op),
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = loom_op_const_operands(op),
      .operand_count = 1,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_copy(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_COPY,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = loom_op_const_operands(op),
      .operand_count = 1,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_storage_reserve(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_RESERVE,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_storage_reserve_byte_alignment_ATTR_INDEX,
      IREE_SV("byte_alignment"), &kLowAsmStorageReserveByteAlignmentAttr,
      out_statement));
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_storage_reserve_byte_length_ATTR_INDEX,
      IREE_SV("byte_length"), &kLowAsmStorageReserveByteLengthAttr,
      out_statement);
}

static iree_status_t loom_low_descriptor_text_asm_describe_storage_address(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_ADDRESS,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = loom_op_const_operands(op),
      .operand_count = 1,
      .location = op->location,
  };
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_storage_address_offset_ATTR_INDEX, IREE_SV("offset"),
      &kLowAsmStorageAddressOffsetAttr, out_statement);
}

static iree_status_t loom_low_descriptor_text_asm_describe_storage_view(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_STORAGE_VIEW,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = loom_op_const_operands(op),
      .operand_count = 1,
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_storage_view_offset_ATTR_INDEX, IREE_SV("offset"),
      &kLowAsmStorageAddressOffsetAttr, out_statement));
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_storage_view_byte_length_ATTR_INDEX,
      IREE_SV("byte_length"), &kLowAsmStorageReserveByteLengthAttr,
      out_statement);
}

iree_status_t loom_low_descriptor_text_asm_describe_structural_operation(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  if (loom_low_resource_isa(op)) {
    return loom_low_descriptor_text_asm_describe_resource(module, op,
                                                          out_statement);
  }
  if (loom_low_live_in_isa(op)) {
    return loom_low_descriptor_text_asm_describe_live_in(module, op,
                                                         out_statement);
  }
  if (loom_low_concat_isa(op)) {
    return loom_low_descriptor_text_asm_describe_concat(module, op,
                                                        out_statement);
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_descriptor_text_asm_describe_slice(module, op,
                                                       out_statement);
  }
  if (loom_low_copy_isa(op)) {
    return loom_low_descriptor_text_asm_describe_copy(module, op,
                                                      out_statement);
  }
  if (loom_low_storage_reserve_isa(op)) {
    return loom_low_descriptor_text_asm_describe_storage_reserve(module, op,
                                                                 out_statement);
  }
  if (loom_low_storage_address_isa(op)) {
    return loom_low_descriptor_text_asm_describe_storage_address(module, op,
                                                                 out_statement);
  }
  if (loom_low_storage_view_isa(op)) {
    return loom_low_descriptor_text_asm_describe_storage_view(module, op,
                                                              out_statement);
  }
  return iree_ok_status();
}
