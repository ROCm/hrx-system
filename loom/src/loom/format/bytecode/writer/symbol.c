// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/symbol.h"

#include "loom/format/bytecode/writer/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Symbol metadata and dependency facets
//===----------------------------------------------------------------------===//

// Number of fixed-width entry offsets retained in each patch chunk.
#define LOOM_BYTECODE_SYMBOL_OFFSET_CHUNK_CAPACITY 256u

// Number of common-case offsets retained without an arena allocation.
#define LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY 16u

// Arena-owned wire bytes for a sequential portion of an offset table.
typedef struct loom_bytecode_symbol_offset_chunk_t {
  // Next chunk in table order, or NULL at the end.
  struct loom_bytecode_symbol_offset_chunk_t* next;
  // Little-endian entry offsets ready for direct stream writes.
  uint8_t values[LOOM_BYTECODE_SYMBOL_OFFSET_CHUNK_CAPACITY][sizeof(uint64_t)];
} loom_bytecode_symbol_offset_chunk_t;

static_assert(sizeof(loom_bytecode_symbol_offset_chunk_t) <=
                  LOOM_BYTECODE_WRITER_PAGE_SIZE,
              "symbol offset chunks must fit in recyclable arena blocks");

// Append-only patch bytes for one import or export offset table.
typedef struct loom_bytecode_symbol_offset_list_t {
  // Common-case little-endian entry offsets stored with the list header.
  uint8_t inline_values[LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY]
                       [sizeof(uint64_t)];
  // First chunk in table order.
  loom_bytecode_symbol_offset_chunk_t* first;
  // Last chunk receiving new offsets.
  loom_bytecode_symbol_offset_chunk_t* last;
  // Number of populated offsets across all chunks.
  uint32_t count;
} loom_bytecode_symbol_offset_list_t;

static iree_status_t loom_bytecode_symbol_offset_list_append(
    iree_arena_allocator_t* arena, uint64_t value,
    loom_bytecode_symbol_offset_list_t* list) {
  uint8_t* bytes = NULL;
  if (list->count < LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY) {
    bytes = list->inline_values[list->count];
  } else {
    const uint32_t chunk_offset =
        (list->count - LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY) %
        LOOM_BYTECODE_SYMBOL_OFFSET_CHUNK_CAPACITY;
    if (chunk_offset == 0) {
      loom_bytecode_symbol_offset_chunk_t* chunk = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*chunk), (void**)&chunk));
      chunk->next = NULL;
      if (list->last) {
        list->last->next = chunk;
      } else {
        list->first = chunk;
      }
      list->last = chunk;
    }
    bytes = list->last->values[chunk_offset];
  }
  for (uint32_t i = 0; i < sizeof(value); ++i) {
    bytes[i] = (uint8_t)(value >> (i * 8));
  }
  ++list->count;
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_symbol_offset_list(
    iree_io_stream_t* stream, const loom_bytecode_symbol_offset_list_t* list) {
  uint32_t remaining = list->count;
  const uint32_t inline_count =
      remaining < LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY
          ? remaining
          : LOOM_BYTECODE_SYMBOL_OFFSET_INLINE_CAPACITY;
  if (inline_count > 0) {
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, (iree_host_size_t)inline_count * sizeof(list->inline_values[0]),
        list->inline_values));
    remaining -= inline_count;
  }
  const loom_bytecode_symbol_offset_chunk_t* chunk = list->first;
  while (remaining > 0) {
    const uint32_t count =
        remaining < LOOM_BYTECODE_SYMBOL_OFFSET_CHUNK_CAPACITY
            ? remaining
            : LOOM_BYTECODE_SYMBOL_OFFSET_CHUNK_CAPACITY;
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, (iree_host_size_t)count * sizeof(chunk->values[0]),
        chunk->values));
    remaining -= count;
    chunk = chunk->next;
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_patch_symbol_offsets(
    loom_bytecode_page_writer_t* writer, uint64_t table_offset,
    const loom_bytecode_symbol_offset_list_t* imports,
    const loom_bytecode_symbol_offset_list_t* exports) {
  if (imports->count == 0 && exports->count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_flush(writer));
  IREE_RETURN_IF_ERROR(iree_io_stream_seek(writer->stream,
                                           IREE_IO_STREAM_SEEK_SET,
                                           (iree_io_stream_pos_t)table_offset));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_write_symbol_offset_list(writer->stream, imports));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_write_symbol_offset_list(writer->stream, exports));
  return iree_io_stream_seek(writer->stream, IREE_IO_STREAM_SEEK_SET,
                             (iree_io_stream_pos_t)writer->total_written);
}

static iree_status_t loom_bytecode_symbol_kind_byte(loom_symbol_kind_t kind,
                                                    uint8_t* out_byte) {
  switch (kind) {
    case LOOM_SYMBOL_FUNC_DEF:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DEF;
      return iree_ok_status();
    case LOOM_SYMBOL_FUNC_DECL:
      *out_byte = LOOM_BYTECODE_SYMBOL_FUNC_DECL;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_DECL:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_DEF:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF;
      return iree_ok_status();
    case LOOM_SYMBOL_TEMPLATE_UKERNEL:
      *out_byte = LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL;
      return iree_ok_status();
    case LOOM_SYMBOL_GLOBAL:
      *out_byte = LOOM_BYTECODE_SYMBOL_GLOBAL;
      return iree_ok_status();
    case LOOM_SYMBOL_EXECUTABLE:
      *out_byte = LOOM_BYTECODE_SYMBOL_EXECUTABLE;
      return iree_ok_status();
    case LOOM_SYMBOL_RECORD:
      *out_byte = LOOM_BYTECODE_SYMBOL_RECORD;
      return iree_ok_status();
    case LOOM_SYMBOL_NONE:
      break;
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown symbol kind %u", (unsigned)kind);
}

// Writes the function-like metadata fields for a single symbol entry.
// Called from loom_bytecode_write_symbols_section for each function-like
// symbol that has a defining op.
static bool loom_bytecode_func_metadata_attr_is_shared(
    const loom_op_vtable_t* vtable, const loom_func_like_vtable_t* func_like,
    uint8_t attr_index) {
  if (loom_bytecode_attr_is_symbol_identity(vtable, attr_index)) {
    return true;
  }
  if (vtable->symbol_def &&
      attr_index ==
          loom_symbol_definition_linkage_attr_index(vtable->symbol_def)) {
    return true;
  }
  iree_string_view_t name =
      loom_attr_descriptor_name(&vtable->attr_descriptors[attr_index]);
  if (iree_string_view_equal(name, IREE_SV("import_module")) ||
      iree_string_view_equal(name, IREE_SV("import_symbol")) ||
      attr_index == func_like->visibility_attr_index ||
      attr_index == func_like->cc_attr_index ||
      attr_index == func_like->effect_contract_attr_index ||
      attr_index == func_like->predicates_attr_index) {
    return true;
  }
  if (attr_index == func_like->template_family_attr_index ||
      attr_index == func_like->priority_attr_index) {
    return true;
  }
  return false;
}

static iree_status_t loom_bytecode_write_func_payload_attrs(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, func_like.op);
  const loom_attribute_t* attrs = loom_op_attrs(func_like.op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_func_metadata_attr_is_shared(
                       vtable, func_like.vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_attr_count));
  for (uint8_t i = 0; i < func_like.op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(loom_bytecode_op_attr_is_present(
        func_like.op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_func_metadata_attr_is_shared(
                        vtable, func_like.vtable, i)) {
      continue;
    }
    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
        writer, numbering, signature_numbering, attrs[i], descriptor));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_region_payload_references(
    loom_bytecode_page_writer_t* writer,
    const loom_bytecode_ir_region_list_t* region_list) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, region_list->count));
  for (uint8_t i = 0; i < region_list->count; ++i) {
    const loom_bytecode_ir_region_payload_t* payload = &region_list->values[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(writer, payload->region_index));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u64_le(writer, payload->offset));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u32_le(writer, payload->length));
  }
  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_func_metadata(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, loom_func_like_t func_like,
    loom_bytecode_value_numbering_t* signature_numbering,
    const loom_bytecode_ir_region_list_t* region_list) {
  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_op(
      numbering, func_like.op, &writer_op_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      writer, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, func_like.op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_source_trivia(
      writer,
      iree_any_bit_set(func_like.op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_u8(writer, loom_func_like_cc(func_like)));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
      writer, loom_func_like_effect_contract(func_like)));

  loom_value_slice_t workload_args =
      loom_kernel_workload_arg_ids(module, func_like.op);
  uint16_t arg_count = 0;
  const loom_value_id_t* arg_ids =
      loom_func_like_arg_ids(func_like, &arg_count);
  uint16_t result_count = func_like.op->result_count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, workload_args.count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, arg_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, result_count));

  for (uint16_t i = 0; i < workload_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, workload_args.values[i]));
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, arg_ids[i]));
  }
  const loom_value_id_t* result_ids = loom_op_const_results(func_like.op);
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_value_numbering_assign_value(
        signature_numbering, result_ids[i]));
  }

  // Kernel workload and ordinary FuncLike argument value definitions.
  for (uint16_t i = 0; i < workload_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(
        writer, numbering, signature_numbering,
        loom_module_value(module, workload_args.values[i])));
  }
  for (uint16_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_write_value_def(writer, numbering, signature_numbering,
                                      loom_module_value(module, arg_ids[i])));
  }

  // Result value definitions with tied info.
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_like.op);
  uint16_t tied_result_count = func_like.op->tied_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    bool is_tied = false;
    uint16_t tied_operand_index = 0;
    for (uint16_t t = 0; t < tied_result_count; ++t) {
      if (tied_results[t].result_index == i) {
        is_tied = true;
        tied_operand_index = tied_results[t].operand_index;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(writer, is_tied ? 1 : 0));
    IREE_RETURN_IF_ERROR(loom_bytecode_write_value_def(
        writer, numbering, signature_numbering,
        loom_module_value(module, result_ids[i])));
    if (is_tied) {
      IREE_RETURN_IF_ERROR(
          loom_bytecode_page_writer_write_uvarint(writer, tied_operand_index));
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, tied_result_count));

  // Predicates.
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(func_like, &predicate_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, predicate_count));
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(writer, predicate->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(writer, predicate->arg_count));
    for (uint8_t arg_index = 0; arg_index < predicate->arg_count; ++arg_index) {
      uint8_t tag = predicate->arg_tags[arg_index];
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, tag));
      switch (tag) {
        case LOOM_PRED_ARG_VALUE: {
          uint32_t value_number = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_resolve_value_number(
              signature_numbering, (loom_value_id_t)predicate->args[arg_index],
              &value_number));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(writer, value_number));
          break;
        }
        case LOOM_PRED_ARG_CONST: {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_svarint(
              writer, predicate->args[arg_index]));
          break;
        }
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown predicate arg tag %d", (int)tag);
      }
    }
  }

  // Template provider metadata references the family by module-local symbol
  // ordinal. This preserves symbol identity across private families and lets
  // selected materialization project the reference without string lookup.
  loom_symbol_ref_t template_family = loom_func_like_template_family(func_like);
  if (func_like.vtable->template_family_attr_index != LOOM_ATTR_INDEX_NONE) {
    if (!loom_symbol_ref_is_valid(template_family) ||
        template_family.module_id != 0 ||
        template_family.symbol_id >= module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template provider symbol must reference a module-local family");
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        writer, loom_bytecode_wire_symbol_ordinal(numbering,
                                                  template_family.symbol_id)));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        writer, (uint64_t)loom_func_like_priority(func_like)));
  }

  IREE_RETURN_IF_ERROR(loom_bytecode_write_func_payload_attrs(
      writer, numbering, module, func_like, signature_numbering));

  return loom_bytecode_write_region_payload_references(writer, region_list);
}

static iree_status_t loom_bytecode_write_global_metadata(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    const loom_bytecode_global_value_list_t* local_values,
    loom_bytecode_value_numbering_t* value_numbering) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_number_global(numbering, op, local_values));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      writer, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_source_trivia(
      writer, iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, op->result_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, local_values->count));
  loom_bytecode_global_value_iterator_t iterator =
      loom_bytecode_global_value_iterator_begin(local_values);
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  while (loom_bytecode_global_value_iterator_next(&iterator, &value_id)) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_value_numbering_assign_value(value_numbering, value_id));
  }
  iterator = loom_bytecode_global_value_iterator_begin(local_values);
  while (loom_bytecode_global_value_iterator_next(&iterator, &value_id)) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_write_value_def(writer, numbering, value_numbering,
                                      loom_module_value(module, value_id)));
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
        writer, numbering, value_numbering, attrs[i], descriptor));
  }

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_record_metadata(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_module_t* module, const loom_op_t* op,
    const loom_bytecode_ir_region_list_t* region_list) {
  IREE_RETURN_IF_ERROR(loom_bytecode_validate_record_symbol_op(module, op));
  IREE_RETURN_IF_ERROR(loom_bytecode_number_record(numbering, op));

  uint32_t writer_op_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_numbering_intern_op(numbering, op, &writer_op_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      writer, (uint64_t)writer_op_id + 1));

  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(module, op, &comment_count);
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_source_trivia(
      writer, iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
      comments, comment_count));

  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_attribute_t* attrs = loom_op_attrs(op);
  uint8_t present_attr_count = 0;
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (present && !loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      ++present_attr_count;
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, present_attr_count));
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    bool present = false;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_op_attr_is_present(op, descriptor, attrs[i], &present));
    if (!present || loom_bytecode_attr_is_symbol_identity(vtable, i)) {
      continue;
    }

    uint32_t key_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
        numbering, loom_attr_descriptor_name(descriptor), &key_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, key_writer_id));
    IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(writer, numbering, NULL,
                                                        attrs[i], descriptor));
  }

  return loom_bytecode_write_region_payload_references(writer, region_list);
}

static loom_attribute_t loom_bytecode_find_op_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name) {
  if (!vtable || !vtable->attr_descriptors) {
    return loom_attr_absent();
  }
  const loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (iree_string_view_equal(
            loom_attr_descriptor_name(&vtable->attr_descriptors[i]), name)) {
      return attrs[i];
    }
  }
  return loom_attr_absent();
}

static iree_status_t loom_bytecode_find_string_attr_by_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    iree_string_view_t name, loom_string_id_t* out_string_id) {
  loom_attribute_t attr = loom_bytecode_find_op_attr_by_name(vtable, op, name);
  if (loom_attr_is_absent(attr)) {
    *out_string_id = LOOM_STRING_ID_INVALID;
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function symbol attribute %.*s must be a string",
                            (int)name.size, name.data);
  }
  *out_string_id = loom_attr_as_string_id(attr);
  return iree_ok_status();
}

typedef struct loom_bytecode_symbol_linkage_t {
  // True when the symbol has public source visibility.
  bool is_public;
  // True when the symbol is available for cross-module static linkage.
  bool is_export;
  // True when the symbol resolves to another module.
  bool is_import;
  // True when the source symbol name was explicitly authored.
  bool has_import_symbol;
  // Source module for an imported symbol.
  loom_string_id_t import_module_id;
  // Source symbol name for an imported symbol.
  loom_string_id_t import_symbol_id;
} loom_bytecode_symbol_linkage_t;

static bool loom_bytecode_symbol_has_visibility_attr(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!symbol->defining_op) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, symbol->defining_op);
  if (!vtable || !vtable->attr_descriptors) {
    return false;
  }
  const loom_attribute_t* attrs = loom_op_const_attrs(symbol->defining_op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("visibility"))) {
      continue;
    }
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        i >= symbol->defining_op->attribute_count) {
      return false;
    }
    return loom_attr_as_enum(attrs[i]) != 0;
  }
  return false;
}

static iree_status_t loom_bytecode_symbol_linkage(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_bytecode_symbol_linkage_t* out_linkage) {
  *out_linkage = (loom_bytecode_symbol_linkage_t){
      .is_public = iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC) ||
                   loom_bytecode_symbol_has_visibility_attr(module, symbol),
      .is_export = false,
      .is_import = false,
      .has_import_symbol = false,
      .import_module_id = LOOM_STRING_ID_INVALID,
      .import_symbol_id = LOOM_STRING_ID_INVALID,
  };
  out_linkage->is_export = out_linkage->is_public;
  if (!symbol->defining_op) {
    return iree_ok_status();
  }

  loom_func_like_t func_like = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func_like)) {
    return iree_ok_status();
  }
  if (loom_func_like_is_exported(func_like)) {
    out_linkage->is_export = true;
  }

  const loom_op_vtable_t* op_vtable = loom_op_vtable(module, func_like.op);
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_module"),
      &out_linkage->import_module_id));
  IREE_RETURN_IF_ERROR(loom_bytecode_find_string_attr_by_name(
      op_vtable, func_like.op, IREE_SV("import_symbol"),
      &out_linkage->import_symbol_id));
  out_linkage->has_import_symbol =
      out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID;

  if (out_linkage->import_module_id == LOOM_STRING_ID_INVALID) {
    if (out_linkage->import_symbol_id != LOOM_STRING_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "function symbol import_symbol requires import_module");
    }
    return iree_ok_status();
  }

  out_linkage->is_import = true;
  out_linkage->is_export = false;
  if (out_linkage->import_symbol_id == LOOM_STRING_ID_INVALID) {
    out_linkage->import_symbol_id = symbol->name_id;
  }
  return iree_ok_status();
}

// Streams the SYMBOLS section and patches its leading offset tables in place.
iree_status_t loom_bytecode_write_symbols_section(
    loom_bytecode_page_writer_t* writer, loom_bytecode_numbering_t* numbering,
    const loom_bytecode_ir_region_index_t* ir_region_index) {
  const loom_module_t* module = numbering->module;

  // Classify symbols.
  uint32_t import_count = 0;
  uint32_t export_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_linkage(
        module, &module->symbols.entries[i], &linkage));
    if (linkage.is_import) {
      ++import_count;
    } else if (linkage.is_export) {
      ++export_count;
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, module->symbols.count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, import_count));
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(writer, export_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      writer, ir_region_index->payload_count));

  // Reserve import/export offset tables (patched after writing entries).
  const uint64_t offset_table_start = writer->total_written;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_zeros(
      writer,
      ((iree_host_size_t)import_count + export_count) * sizeof(uint64_t)));
  const uint64_t entries_start = writer->total_written;
  loom_bytecode_symbol_offset_list_t import_offsets = {0};
  loom_bytecode_symbol_offset_list_t export_offsets = {0};

  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < module->symbols.count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_t* symbol = &module->symbols.entries[module_symbol_id];
    loom_bytecode_symbol_linkage_t linkage;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_symbol_linkage(module, symbol, &linkage));
    loom_symbol_kind_t metadata_kind = loom_symbol_bytecode_kind(symbol);
    bool has_function_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
        loom_symbol_kind_is_function_like(metadata_kind);
    bool has_global_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        metadata_kind == LOOM_SYMBOL_GLOBAL;
    bool has_record_metadata =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        metadata_kind == LOOM_SYMBOL_RECORD;
    const uint64_t entry_offset = writer->total_written - entries_start;

    // Track import/export offsets.
    if (linkage.is_import) {
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_offset_list_append(
          numbering->arena, entry_offset, &import_offsets));
    } else if (linkage.is_export) {
      IREE_RETURN_IF_ERROR(loom_bytecode_symbol_offset_list_append(
          numbering->arena, entry_offset, &export_offsets));
    }

    // Name.
    uint32_t name_writer_id = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
        numbering, symbol->name_id, &name_writer_id));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, name_writer_id));

    // Kind.
    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_symbol_kind_byte(
        loom_symbol_bytecode_kind(symbol), &kind_byte));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(writer, kind_byte));

    // Visibility.
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
        writer, linkage.is_public ? LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC
                                  : LOOM_BYTECODE_SYMBOL_VISIBILITY_PRIVATE));

    // Flags.
    uint16_t bytecode_flags =
        linkage.is_public ? LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC : 0;
    if (linkage.is_export) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_EXPORT;
    }
    if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_RETAIN)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_RETAIN;
    }
    if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_LINK_ONCE)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_LINK_ONCE;
    }
    if (loom_symbol_definition_is_declaration(symbol->definition)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION;
    }
    if (loom_symbol_definition_is_test_only(symbol->definition)) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY;
    }
    if (linkage.is_import) {
      bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT;
      if (linkage.has_import_symbol) {
        bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_IMPORT_SYMBOL;
      }
    }
    if (has_function_metadata && symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like) &&
          func_like.vtable->predicates_attr_index != LOOM_ATTR_INDEX_NONE &&
          !loom_attr_is_absent(loom_op_const_attrs(
              func_like.op)[func_like.vtable->predicates_attr_index])) {
        bytecode_flags |= LOOM_BYTECODE_SYMBOL_FLAG_PREDICATES;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u16_le(writer, bytecode_flags));
    const loom_location_id_t location =
        numbering->location_mode == LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS ||
                !symbol->defining_op
            ? LOOM_LOCATION_UNKNOWN
            : symbol->defining_op->location;
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(writer, location));
    if (linkage.is_import) {
      uint32_t import_module_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_module_id, &import_module_string_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          writer, import_module_string_id));
      uint32_t import_symbol_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, linkage.import_symbol_id, &import_symbol_string_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          writer, import_symbol_string_id));
    }

    // Function metadata.
    if (has_function_metadata && symbol->defining_op) {
      loom_func_like_t func_like =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(func_like)) {
        loom_bytecode_value_numbering_t signature_numbering;
        loom_bytecode_value_numbering_initialize(&signature_numbering,
                                                 numbering);
        const loom_bytecode_ir_region_list_t region_list =
            loom_bytecode_ir_region_index_list(ir_region_index,
                                               module_symbol_id);
        IREE_RETURN_IF_ERROR(loom_bytecode_write_func_metadata(
            writer, numbering, module, func_like, &signature_numbering,
            &region_list));
      }
    } else if (has_global_metadata && symbol->defining_op) {
      loom_bytecode_value_numbering_t signature_numbering;
      loom_bytecode_value_numbering_initialize(&signature_numbering, numbering);
      const loom_bytecode_global_value_list_t* local_values =
          loom_bytecode_global_values_for_symbol(numbering, module_symbol_id);
      IREE_RETURN_IF_ERROR(loom_bytecode_write_global_metadata(
          writer, numbering, module, symbol->defining_op, local_values,
          &signature_numbering));
    } else if (has_record_metadata && symbol->defining_op) {
      const loom_bytecode_ir_region_list_t region_list =
          loom_bytecode_ir_region_index_list(ir_region_index, module_symbol_id);
      IREE_RETURN_IF_ERROR(loom_bytecode_write_record_metadata(
          writer, numbering, module, symbol->defining_op, &region_list));
    }
  }

  return loom_bytecode_patch_symbol_offsets(writer, offset_table_start,
                                            &import_offsets, &export_offsets);
}

//===----------------------------------------------------------------------===//
// Symbol reference index
//===----------------------------------------------------------------------===//

// Prepared analysis and aggregate counts written to SYMBOL_REFERENCES.
static uint32_t loom_bytecode_count_dependency_occurrences(
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id) {
  uint32_t dependency_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(table, occurrence_id);
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      ++dependency_count;
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return dependency_count;
}

iree_status_t loom_bytecode_symbol_reference_plan_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_symbol_reference_plan_t* out_plan) {
  *out_plan = (loom_bytecode_symbol_reference_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(module, arena, &out_plan->table));

  for (iree_host_size_t i = 0; i < out_plan->table.occurrence_count; ++i) {
    if (loom_symbol_reference_occurrence_is_dependency(
            loom_symbol_reference_table_occurrence(&out_plan->table, i))) {
      ++out_plan->dependency_count;
    }
  }
  out_plan->module_dependency_count =
      loom_bytecode_count_dependency_occurrences(
          &out_plan->table, out_plan->table.first_module_occurrence_id);

  return iree_ok_status();
}

static iree_status_t loom_bytecode_write_dependency_row(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering,
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id,
    uint32_t dependency_count) {
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, dependency_count));
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(table, occurrence_id);
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, occurrence->source_root_region_index_plus_one));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, loom_bytecode_wire_symbol_ordinal(
                           numbering, occurrence->target_symbol_id)));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, loom_symbol_reference_contract_make(
                           occurrence->target_interfaces, occurrence->role)));
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_symbol_references_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering,
    const loom_bytecode_symbol_reference_plan_t* plan) {
  const loom_symbol_reference_table_t* table = &plan->table;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, table->symbol_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, plan->dependency_count));
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, table->template_demands.count));
  IREE_RETURN_IF_ERROR(loom_bytecode_write_dependency_row(
      page_writer, numbering, table, table->first_module_occurrence_id,
      plan->module_dependency_count));
  for (loom_symbol_id_t wire_ordinal = 0; wire_ordinal < table->symbol_count;
       ++wire_ordinal) {
    const loom_symbol_id_t module_symbol_id =
        loom_bytecode_module_symbol_id(numbering, wire_ordinal);
    const loom_symbol_reference_symbol_occurrences_t symbol =
        loom_symbol_reference_table_symbol(table, module_symbol_id);
    const uint32_t dependency_count =
        loom_bytecode_count_dependency_occurrences(
            table, symbol.first_outgoing_occurrence_id);
    IREE_RETURN_IF_ERROR(loom_bytecode_write_dependency_row(
        page_writer, numbering, table, symbol.first_outgoing_occurrence_id,
        dependency_count));
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, symbol.template_demand_count));
    loom_template_demand_id_t demand_id = symbol.first_template_demand_id;
    while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID) {
      const loom_template_demand_t* demand =
          &table->template_demands.values[demand_id];
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, demand->source_root_region_index_plus_one));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, loom_bytecode_wire_symbol_ordinal(
                           numbering, demand->family_symbol_id)));
      demand_id = demand->next_source_demand_id;
    }
  }
  return iree_ok_status();
}
