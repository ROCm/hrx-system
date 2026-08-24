// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/body.h"

#include "loom/format/bytecode/writer/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// IR body serialization
//===----------------------------------------------------------------------===//

static void loom_bytecode_count_region_tree(
    const loom_region_t* region, loom_bytecode_body_counts_t* counts) {
  ++counts->region_count;
  counts->block_count += region->block_count;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    counts->value_count += block->arg_count;
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++counts->op_count;
      counts->value_count += op->result_count;
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        if (regions[region_index]) {
          loom_bytecode_count_region_tree(regions[region_index], counts);
        }
      }
    }
  }
}

static void loom_bytecode_count_op_regions(
    const loom_op_t* op, loom_bytecode_body_counts_t* counts) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (!regions[i]) continue;
    loom_bytecode_count_region_tree(regions[i], counts);
  }
}

static uint8_t loom_bytecode_count_root_regions(const loom_op_t* op) {
  uint8_t count = 0;
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      ++count;
    }
  }
  return count;
}

iree_status_t loom_bytecode_count_serialized_bodies(
    loom_bytecode_numbering_t* numbering, loom_bytecode_body_counts_t* counts) {
  const loom_module_t* module = numbering->module;
  *counts = (loom_bytecode_body_counts_t){
      .value_count = 0,
      .region_count = 0,
      .block_count = 0,
      .op_count = 0,
  };
  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < module->symbols.count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_t* symbol = &module->symbols.entries[module_symbol_id];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    bool is_function_like =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(bytecode_kind);
    bool is_global =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        bytecode_kind == LOOM_SYMBOL_GLOBAL;
    bool is_record =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD;
    if (!is_function_like || !symbol->defining_op) {
      if (is_global && symbol->defining_op) {
        loom_bytecode_global_value_list_t local_values = {0};
        IREE_RETURN_IF_ERROR(loom_bytecode_collect_global_values(
            numbering->arena, module, symbol->defining_op, &local_values));
        counts->value_count += local_values.count;
      } else if (is_record && symbol->defining_op &&
                 symbol->defining_op->region_count == 1) {
        loom_region_t* body = loom_op_regions(symbol->defining_op)[0];
        if (body) {
          loom_bytecode_count_region_tree(body, counts);
        }
      }
      continue;
    }
    loom_func_like_t func_like =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(func_like)) {
      continue;
    }
    counts->value_count += func_like.op->result_count;
    if (loom_bytecode_count_root_regions(func_like.op) != 0) {
      loom_bytecode_count_op_regions(func_like.op, counts);
    } else {
      loom_value_slice_t workload_args =
          loom_kernel_workload_arg_ids(module, func_like.op);
      uint16_t arg_count = 0;
      loom_func_like_arg_ids(func_like, &arg_count);
      counts->value_count += (uint64_t)workload_args.count + arg_count;
    }
  }
  return iree_ok_status();
}

// Forward declarations for recursive IR writing.
static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth);

static iree_status_t loom_bytecode_write_value_def(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value) {
  uint32_t name_writer_id = 0;
  if (value->name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, value->name_id, &name_writer_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, name_writer_id));

  uint32_t type_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
      numbering, value->type, &type_writer_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, type_writer_id));

  // Dim bindings: count dynamic dims, then emit value refs.
  loom_type_t type = value->type;
  uint8_t rank = loom_type_rank(type);
  uint32_t dynamic_count = 0;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) ++dynamic_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, dynamic_count));
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed = loom_type_dim(type, i);
    if (!loom_dim_is_dynamic(packed)) continue;
    loom_value_id_t dim_value_id = loom_dim_value_id(packed);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, dim_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_svarint(writer, (int64_t)value_number));
  }

  // Encoding binding.
  if (loom_type_has_ssa_encoding(type)) {
    uint16_t encoding_value_id = loom_type_encoding_value_id(type);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, encoding_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, 1 + value_number));
  } else {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_emit_value_def(
    iree_string_builder_t* builder, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_value_t* value) {
  uint32_t name_writer_id = 0;
  if (value->name_id != LOOM_STRING_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, value->name_id, &name_writer_id));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)name_writer_id));

  uint32_t type_writer_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
      numbering, value->type, &type_writer_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)type_writer_id));

  loom_type_t type = value->type;
  uint8_t rank = loom_type_rank(type);
  uint32_t dynamic_count = 0;
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) ++dynamic_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_emit_uvarint(builder, (uint64_t)dynamic_count));
  for (uint8_t i = 0; i < rank; ++i) {
    uint64_t packed = loom_type_dim(type, i);
    if (!loom_dim_is_dynamic(packed)) continue;
    loom_value_id_t dim_value_id = loom_dim_value_id(packed);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, dim_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_svarint(builder, (int64_t)value_number));
  }

  if (loom_type_has_ssa_encoding(type)) {
    uint16_t encoding_value_id = loom_type_encoding_value_id(type);
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, encoding_value_id, &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_emit_uvarint(builder, 1 + (uint64_t)value_number));
  } else {
    IREE_RETURN_IF_ERROR(loom_bytecode_emit_uvarint(builder, 0));
  }
  return iree_ok_status();
}

static uint8_t loom_bytecode_instance_flags_mask(
    const loom_op_vtable_t* vtable) {
  if (!iree_all_bits_set(vtable->vtable_flags,
                         LOOM_OP_VTABLE_HAS_INSTANCE_FLAGS)) {
    return 0;
  }
  if (vtable->instance_flags_case_count >= 8) return UINT8_MAX;
  return (uint8_t)((1u << vtable->instance_flags_case_count) - 1u);
}

static iree_status_t loom_bytecode_find_successor_block_index(
    const loom_op_t* op, const loom_block_t* target,
    uint16_t* out_block_index) {
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation successor target is NULL");
  }
  if (!op->parent_block || !op->parent_block->parent_region) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation with successors is not attached to a region");
  }
  const loom_region_t* region = op->parent_block->parent_region;
  if (target->parent_region && target->parent_region != region) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "operation successor target belongs to a different region");
  }
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (loom_region_const_block(region, i) == target) {
      *out_block_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "operation successor target is not in its region");
}

static iree_status_t loom_bytecode_write_operation(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering, const loom_op_t* op,
    uint32_t depth) {
  const loom_module_t* module = numbering->module;
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op kind 0x%04x has no registered vtable",
                            (unsigned)op->kind);
  }

  // Operation table index, plus one so 0 remains an invalid reference.
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, writer_op_id + 1));

  uint8_t instance_flags_mask = loom_bytecode_instance_flags_mask(vtable);
  if (iree_any_bit_set(op->instance_flags, (uint8_t)~instance_flags_mask)) {
    iree_string_view_t name = loom_op_vtable_name(vtable);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "op %.*s has undeclared instance flag bits 0x%02x", (int)name.size,
        name.data,
        (unsigned)(op->instance_flags & (uint8_t)~instance_flags_mask));
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, op->instance_flags));

  loom_location_id_t location =
      numbering->location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS
          ? LOOM_LOCATION_UNKNOWN
          : op->location;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, location));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_source_trivia(
      writer, iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  // Operands.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->operand_count));
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    uint32_t value_number = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
        value_numbering, operands[i], &value_number));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, value_number));
  }
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  const uint16_t* operand_segment_counts =
      loom_op_const_operand_segment_counts(op);
  for (uint8_t i = 0; i < operand_segment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        writer, operand_segment_counts[i]));
  }

  // Successors are encoded as region-local block ordinals. This keeps bytecode
  // independent from optional text labels and makes forward edges cheap.
  loom_block_t* const* successors = loom_op_const_successors(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->successor_count));
  for (uint8_t i = 0; i < op->successor_count; ++i) {
    uint16_t block_index = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_find_successor_block_index(
        op, successors[i], &block_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, block_index));
  }

  // Results.
  const loom_value_id_t* results = loom_op_const_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->result_count));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_t* value = loom_module_value(module, results[i]);
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  // Tied results.
  const loom_tied_result_t* tied = loom_op_tied_results(op);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->tied_result_count));
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].result_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, tied[i].operand_index));
  }

  // Attributes: each has a key name from the vtable descriptor and a
  // tagged value from the op's trailing data.
  const loom_attribute_t* attrs = loom_op_attrs(op);
  if (op->attribute_count > 0 && !vtable->attr_descriptors) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "op kind 0x%04x has %d attributes but no attr_descriptors in vtable",
        (unsigned)op->kind, (int)op->attribute_count);
  }
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        op, &vtable->attr_descriptors[i], attrs[i], &present));
    if (present) ++present_attr_count;
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        op, &vtable->attr_descriptors[i], attrs[i], &present));
    if (!present) continue;

    iree_string_view_t key_name =
        loom_attr_descriptor_name(&vtable->attr_descriptors[i]);
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, key_name, &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    // Value.
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (descriptor->attr_kind == LOOM_ATTR_SCOPED_ENUM) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_write_scoped_enum(writer, numbering, attrs[i]));
    } else {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
          writer, numbering, value_numbering, attrs[i], descriptor));
    }
  }

  // Regions.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->region_count));
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_bytecode_write_region(
          writer, numbering, value_numbering, regions[i], depth + 1));
    } else {
      // Empty region: no source flags and 0 blocks.
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(writer, 0));
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_block(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_block_t* block, uint32_t depth) {
  const loom_module_t* module = numbering->module;

  // Label.
  bool has_label = block->label_id != LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, has_label ? 1 : 0));
  if (has_label) {
    uint32_t label_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, block->label_id, &label_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, label_writer_id));
  }

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_block_comments(module, block, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_source_trivia(
      writer,
      iree_any_bit_set(block->flags, LOOM_BLOCK_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  // Block args.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, block->arg_count));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(writer, numbering,
                                                       value_numbering, value));
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, block->op_count));
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    IREE_RETURN_IF_ERROR(loom_bytecode_write_operation(
        writer, numbering, value_numbering, op, depth));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_region(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_value_numbering_t* value_numbering,
    const loom_region_t* region, uint32_t depth) {
  if (depth >= LOOM_BYTECODE_WRITER_MAX_REGION_DEPTH) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region nesting exceeds maximum depth %d",
                            LOOM_BYTECODE_WRITER_MAX_REGION_DEPTH);
  }
  if (region->source_flags & ~LOOM_REGION_SOURCE_FLAG_MASK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "region source flags contain unsupported bits 0x%04X",
        (unsigned)region->source_flags);
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, region->source_flags));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, region->block_count));
  for (uint16_t i = 0; i < region->block_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_write_block(writer, numbering, value_numbering,
                                  loom_region_const_block(region, i), depth));
  }
  return iree_ok_status();
}

// Writes the IR section and returns per-symbol root-region ranges.
iree_status_t loom_bytecode_write_ir_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering,
    loom_bytecode_ir_region_list_t* ir_regions) {
  const loom_module_t* module = numbering->module;
  iree_host_size_t section_start = page_writer->total_written;

  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < module->symbols.count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_t* symbol = &module->symbols.entries[module_symbol_id];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    bool is_function_like =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(bytecode_kind);
    bool is_record =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD;
    if (!symbol->defining_op) {
      continue;
    }

    uint8_t root_region_count = 0;
    const loom_low_repr_descriptor_set_t* low_descriptor_set = NULL;
    if (is_function_like) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (!loom_func_like_isa(func_like)) {
        continue;
      }
      root_region_count = loom_bytecode_count_root_regions(func_like.op);
      if (root_region_count == 0) {
        continue;
      }

      IREE_RETURN_IF_ERROR(loom_bytecode_resolve_function_low_descriptor_set(
          numbering, func_like, &low_descriptor_set));

      // Intern function signature and regions (matching Python walk order).
      numbering->low_repr.active_descriptor_set = low_descriptor_set;
      IREE_RETURN_IF_ERROR(loom_bytecode_number_function(numbering, func_like));
    } else if (is_record && symbol->defining_op->region_count == 1) {
      root_region_count = loom_bytecode_count_root_regions(symbol->defining_op);
      if (root_region_count == 0) {
        continue;
      }

      // Intern record metadata and body (matching Python walk order).
      IREE_RETURN_IF_ERROR(
          loom_bytecode_number_record(numbering, symbol->defining_op));
    } else {
      continue;
    }

    loom_bytecode_ir_region_list_t* region_list = &ir_regions[module_symbol_id];
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        numbering->arena, root_region_count, sizeof(*region_list->values),
        (void**)&region_list->values));
    loom_region_t** regions = loom_op_regions(symbol->defining_op);
    for (uint8_t i = 0; i < symbol->defining_op->region_count; ++i) {
      if (!regions[i]) continue;

      loom_bytecode_body_counts_t region_counts = {0};
      loom_bytecode_count_region_tree(regions[i], &region_counts);

      loom_bytecode_value_numbering_t value_numbering;
      loom_bytecode_value_numbering_initialize(&value_numbering, module,
                                               numbering->arena);
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_ensure_capacity(
          &value_numbering, region_counts.value_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_region(
          &value_numbering, regions[i]));

      const iree_host_size_t payload_start = page_writer->total_written;
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, region_counts.value_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, region_counts.region_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, region_counts.block_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, region_counts.op_count));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_region(
          page_writer, numbering, &value_numbering, regions[i], 0));
      const iree_host_size_t payload_length =
          page_writer->total_written - payload_start;
      if (payload_length > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "root region payload length %" PRIhsz
                                " exceeds uint32 maximum",
                                payload_length);
      }
      region_list->values[region_list->count++] =
          (loom_bytecode_ir_region_payload_t){
              .offset = payload_start - section_start,
              .length = (uint32_t)payload_length,
              .region_index = i,
          };
    }
    IREE_ASSERT(region_list->count == root_region_count);
    numbering->low_repr.active_descriptor_set = NULL;
  }

  return iree_ok_status();
}
