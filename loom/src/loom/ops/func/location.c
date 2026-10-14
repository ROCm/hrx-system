// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/location.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/format/location.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"

static iree_status_t loom_func_location_invalid(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field, int64_t value, iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field),
      loom_param_i64(value),
      loom_param_string(constraint),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_014,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static bool loom_func_location_range_valid(loom_i64_array_t range) {
  if (range.count != 4) {
    return false;
  }
  for (iree_host_size_t i = 0; i < range.count; ++i) {
    if (range.values[i] < 0 || range.values[i] > UINT32_MAX) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_func_location_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  const loom_parameterized_attr_array_t nodes = loom_func_location_nodes(op);
  if (!nodes.count) {
    return loom_func_location_invalid(
        op, emitter, IREE_SV("nodes"), 0,
        IREE_SV("nonempty postorder location graph"));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < nodes.count && iree_status_is_ok(status);
       ++i) {
    const loom_attribute_t node = nodes.values[i];
    if (loom_func_location_file_attr_isa(node)) {
      if (!loom_func_location_range_valid(
              loom_func_location_file_attr_range(node))) {
        status = loom_func_location_invalid(
            op, emitter, IREE_SV("range"), i,
            IREE_SV("four unsigned 32-bit source coordinates"));
      }
      const loom_parameterized_attr_array_t fields =
          loom_func_location_file_attr_fields(node);
      for (iree_host_size_t j = 0;
           j < fields.count && iree_status_is_ok(status); ++j) {
        const loom_attribute_t field = fields.values[j];
        const int64_t index = loom_func_location_field_attr_index(field);
        if (index < 0 || index > UINT32_MAX ||
            !loom_func_location_range_valid(
                loom_func_location_field_attr_range(field))) {
          status = loom_func_location_invalid(
              op, emitter, IREE_SV("fields"), j,
              IREE_SV("source field kind, unsigned index, and four unsigned "
                      "32-bit coordinates"));
        }
      }
    } else if (loom_func_location_fused_attr_isa(node)) {
      const loom_i64_array_t children =
          loom_func_location_fused_attr_children(node);
      for (iree_host_size_t j = 0;
           j < children.count && iree_status_is_ok(status); ++j) {
        if (children.values[j] < 0 || (uint64_t)children.values[j] >= i) {
          status = loom_func_location_invalid(
              op, emitter, IREE_SV("children"), children.values[j],
              IREE_SV("index of a preceding location node"));
        }
      }
    } else if (loom_func_location_tagged_attr_isa(node)) {
      const int64_t tag = loom_func_location_tagged_attr_tag(node);
      const int64_t child = loom_func_location_tagged_attr_child(node);
      if (tag <= 0 || tag > UINT32_MAX) {
        status =
            loom_func_location_invalid(op, emitter, IREE_SV("tag"), tag,
                                       IREE_SV("nonzero unsigned 32-bit tag"));
      } else if (loom_func_location_tagged_attr_has_child(node) &&
                 (child < 0 || (uint64_t)child >= i)) {
        status = loom_func_location_invalid(
            op, emitter, IREE_SV("child"), child,
            IREE_SV("index of a preceding location node"));
      }
    } else if (!loom_func_location_unknown_attr_isa(node) &&
               !loom_func_location_opaque_attr_isa(node)) {
      const loom_parameterized_attr_descriptor_t* descriptor =
          loom_context_resolve_parameterized_attr(
              module->context, loom_attr_as_parameterized_kind(node));
      const loom_diagnostic_param_t params[] = {
          loom_param_string(IREE_SV("nodes")),
          loom_param_string(loom_bstring_view(descriptor->name)),
          loom_param_string(IREE_SV("func.location node")),
      };
      const loom_diagnostic_emission_t emission = {
          .op = op,
          .error = LOOM_ERR_STRUCTURE_042,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
      };
      status = iree_diagnostic_emit(emitter, &emission);
    }
  }
  return status;
}

// The encoder consumes verified nodes. The wire layout is computed once before
// allocating its exact output size; each node retains its payload extent.
typedef struct loom_func_location_layout_t {
  // Stable runtime node kind.
  loom_location_value_kind_t kind;
  // Source flags retained without interpreting their provenance meaning.
  loom_location_value_flags_t flags;
  // Byte offset of the node's fixed payload (or fused child array).
  uint32_t offset;
  // Byte length of that payload, excluding referenced strings and data.
  uint32_t length;
} loom_func_location_layout_t;

static void loom_func_location_write_range(uint8_t* target,
                                           loom_i64_array_t range) {
  for (iree_host_size_t i = 0; i < 4; ++i) {
    iree_unaligned_store_le_u32(target + i * 4, (uint32_t)range.values[i]);
  }
}

static void loom_func_location_write_bytes(uint8_t* bytes, uint32_t* cursor,
                                           uint8_t* reference,
                                           iree_const_byte_span_t source) {
  iree_unaligned_store_le_u32(reference, *cursor);
  iree_unaligned_store_le_u32(reference + 4, (uint32_t)source.data_length);
  if (source.data_length) {
    memcpy(bytes + *cursor, source.data, source.data_length);
  }
  *cursor += (uint32_t)source.data_length;
}

static void loom_func_location_write_string(const loom_module_t* module,
                                            uint8_t* bytes, uint32_t* cursor,
                                            uint8_t* reference,
                                            loom_string_id_t string_id) {
  const iree_string_view_t string =
      loom_string_table_get(&module->strings, string_id);
  loom_func_location_write_bytes(
      bytes, cursor, reference,
      iree_make_const_byte_span(string.data, string.size));
}

iree_status_t loom_func_location_encode(const loom_module_t* module,
                                        loom_parameterized_attr_array_t nodes,
                                        iree_arena_allocator_t* arena,
                                        iree_const_byte_span_t* out_bytes) {
  *out_bytes = iree_const_byte_span_empty();
  loom_func_location_layout_t* layouts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, nodes.count, sizeof(*layouts), (void**)&layouts));
  uint64_t size = LOOM_LOCATION_VALUE_HEADER_LENGTH +
                  (uint64_t)nodes.count * LOOM_LOCATION_VALUE_NODE_LENGTH;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < nodes.count && iree_status_is_ok(status);
       ++i) {
    const loom_attribute_t node = nodes.values[i];
    loom_func_location_layout_t* layout = &layouts[i];
    *layout = (loom_func_location_layout_t){.offset = (uint32_t)size};
    uint64_t payload_length = 0;
    if (loom_func_location_file_attr_isa(node)) {
      layout->kind = LOOM_LOCATION_VALUE_FILE;
      layout->flags = loom_func_location_file_attr_synthetic(node)
                          ? LOOM_LOCATION_VALUE_FLAG_SYNTHETIC
                          : 0;
      payload_length = LOOM_LOCATION_VALUE_FILE_LENGTH;
      size += (uint64_t)loom_string_table_get(
                  &module->strings, loom_func_location_file_attr_source(node))
                  .size +
              (uint64_t)loom_func_location_file_attr_fields(node).count *
                  LOOM_LOCATION_VALUE_FIELD_LENGTH +
              loom_func_location_file_attr_text(node).data_length;
    } else if (loom_func_location_fused_attr_isa(node)) {
      layout->kind = LOOM_LOCATION_VALUE_FUSED;
      layout->flags = loom_func_location_fused_attr_synthetic(node)
                          ? LOOM_LOCATION_VALUE_FLAG_SYNTHETIC
                          : 0;
      payload_length =
          (uint64_t)loom_func_location_fused_attr_children(node).count * 4;
    } else if (loom_func_location_opaque_attr_isa(node)) {
      layout->kind = LOOM_LOCATION_VALUE_OPAQUE;
      layout->flags = loom_func_location_opaque_attr_synthetic(node)
                          ? LOOM_LOCATION_VALUE_FLAG_SYNTHETIC
                          : 0;
      payload_length = LOOM_LOCATION_VALUE_OPAQUE_LENGTH;
      size += (uint64_t)loom_string_table_get(
                  &module->strings, loom_func_location_opaque_attr_source(node))
                  .size +
              loom_func_location_opaque_attr_data(node).data_length;
    } else if (loom_func_location_tagged_attr_isa(node)) {
      layout->kind = LOOM_LOCATION_VALUE_TAGGED;
      layout->flags = loom_func_location_tagged_attr_synthetic(node)
                          ? LOOM_LOCATION_VALUE_FLAG_SYNTHETIC
                          : 0;
      payload_length = LOOM_LOCATION_VALUE_TAGGED_LENGTH;
      size += loom_func_location_tagged_attr_data(node).data_length;
    } else {
      layout->kind = LOOM_LOCATION_VALUE_UNKNOWN;
      layout->offset = 0;
    }
    size += payload_length;
    if (size > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "captured location exceeds 4 GiB");
    }
    layout->length = (uint32_t)payload_length;
  }
  IREE_RETURN_IF_ERROR(status);
  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, (iree_host_size_t)size, (void**)&bytes));
  memset(bytes, 0, (iree_host_size_t)size);
  memcpy(bytes, "LLOC", 4);
  iree_unaligned_store_le_u32(bytes + 4, LOOM_LOCATION_VALUE_VERSION);
  iree_unaligned_store_le_u32(bytes + 8, (uint32_t)size);
  iree_unaligned_store_le_u32(bytes + 12, (uint32_t)nodes.count);
  for (iree_host_size_t i = 0; i < nodes.count; ++i) {
    const loom_attribute_t node = nodes.values[i];
    const loom_func_location_layout_t* layout = &layouts[i];
    uint8_t* record = bytes + LOOM_LOCATION_VALUE_HEADER_LENGTH +
                      i * LOOM_LOCATION_VALUE_NODE_LENGTH;
    record[0] = (uint8_t)layout->kind;
    record[1] = layout->flags;
    iree_unaligned_store_le_u32(record + 4, layout->offset);
    iree_unaligned_store_le_u32(record + 8, layout->length);
    uint8_t* payload = bytes + layout->offset;
    uint32_t cursor = layout->offset + layout->length;
    switch (layout->kind) {
      case LOOM_LOCATION_VALUE_UNKNOWN:
        break;
      case LOOM_LOCATION_VALUE_FILE: {
        loom_func_location_write_string(
            module, bytes, &cursor, payload,
            loom_func_location_file_attr_source(node));
        loom_func_location_write_range(
            payload + 8, loom_func_location_file_attr_range(node));
        const loom_parameterized_attr_array_t fields =
            loom_func_location_file_attr_fields(node);
        iree_unaligned_store_le_u32(payload + 24, cursor);
        iree_unaligned_store_le_u32(payload + 28, (uint32_t)fields.count);
        for (iree_host_size_t j = 0; j < fields.count; ++j) {
          const loom_attribute_t field = fields.values[j];
          iree_unaligned_store_le_u32(
              bytes + cursor,
              (uint32_t)loom_func_location_field_attr_kind(field));
          iree_unaligned_store_le_u32(
              bytes + cursor + 4,
              (uint32_t)loom_func_location_field_attr_index(field));
          loom_func_location_write_range(
              bytes + cursor + 8, loom_func_location_field_attr_range(field));
          cursor += LOOM_LOCATION_VALUE_FIELD_LENGTH;
        }
        if (loom_func_location_file_attr_has_text(node)) {
          loom_func_location_write_bytes(
              bytes, &cursor, payload + 32,
              loom_func_location_file_attr_text(node));
        }
        break;
      }
      case LOOM_LOCATION_VALUE_FUSED: {
        const loom_i64_array_t children =
            loom_func_location_fused_attr_children(node);
        for (iree_host_size_t j = 0; j < children.count; ++j) {
          iree_unaligned_store_le_u32(payload + j * 4,
                                      (uint32_t)children.values[j]);
        }
        break;
      }
      case LOOM_LOCATION_VALUE_OPAQUE:
        loom_func_location_write_string(
            module, bytes, &cursor, payload,
            loom_func_location_opaque_attr_source(node));
        loom_func_location_write_bytes(
            bytes, &cursor, payload + 8,
            loom_func_location_opaque_attr_data(node));
        break;
      case LOOM_LOCATION_VALUE_TAGGED:
        iree_unaligned_store_le_u32(
            payload, (uint32_t)loom_func_location_tagged_attr_tag(node));
        iree_unaligned_store_le_u32(
            payload + 4,
            loom_func_location_tagged_attr_has_child(node)
                ? (uint32_t)loom_func_location_tagged_attr_child(node)
                : UINT32_MAX);
        loom_func_location_write_bytes(
            bytes, &cursor, payload + 8,
            loom_func_location_tagged_attr_data(node));
        break;
    }
  }
  *out_bytes = iree_make_const_byte_span(bytes, (iree_host_size_t)size);
  return iree_ok_status();
}
