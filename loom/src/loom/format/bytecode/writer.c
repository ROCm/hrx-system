// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer.h"

#include <string.h>

#include "loom/format/bytecode/writer/attribute.h"
#include "loom/format/bytecode/writer/body.h"
#include "loom/format/bytecode/writer/catalog.h"
#include "loom/format/bytecode/writer/encoder.h"
#include "loom/format/bytecode/writer/numbering.h"
#include "loom/format/bytecode/writer/symbol.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/module_record.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ops/op_defs.h"

#define LOOM_BYTECODE_DEFAULT_PRODUCER "loom-c"

//===----------------------------------------------------------------------===//
// Type kind mapping (C enum → bytecode kind byte)
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_type_kind_byte(loom_type_kind_t kind,
                                                  uint8_t* out_byte) {
  switch (kind) {
    case LOOM_TYPE_NONE:
      *out_byte = LOOM_BYTECODE_TYPE_NONE;
      return iree_ok_status();
    case LOOM_TYPE_SCALAR:
      *out_byte = LOOM_BYTECODE_TYPE_SCALAR;
      return iree_ok_status();
    case LOOM_TYPE_TILE:
      *out_byte = LOOM_BYTECODE_TYPE_TILE;
      return iree_ok_status();
    case LOOM_TYPE_TENSOR:
      *out_byte = LOOM_BYTECODE_TYPE_TENSOR;
      return iree_ok_status();
    case LOOM_TYPE_VECTOR:
      *out_byte = LOOM_BYTECODE_TYPE_VECTOR;
      return iree_ok_status();
    case LOOM_TYPE_VIEW:
      *out_byte = LOOM_BYTECODE_TYPE_VIEW;
      return iree_ok_status();
    case LOOM_TYPE_BUFFER:
      *out_byte = LOOM_BYTECODE_TYPE_BUFFER;
      return iree_ok_status();
    case LOOM_TYPE_FUNCTION:
      *out_byte = LOOM_BYTECODE_TYPE_FUNCTION;
      return iree_ok_status();
    case LOOM_TYPE_DIALECT:
      *out_byte = LOOM_BYTECODE_TYPE_DIALECT;
      return iree_ok_status();
    case LOOM_TYPE_REGISTER:
      *out_byte = LOOM_BYTECODE_TYPE_REGISTER;
      return iree_ok_status();
    case LOOM_TYPE_STORAGE:
      *out_byte = LOOM_BYTECODE_TYPE_STORAGE;
      return iree_ok_status();
    case LOOM_TYPE_PARAMETERIZED:
      *out_byte = LOOM_BYTECODE_TYPE_PARAMETERIZED;
      return iree_ok_status();
    case LOOM_TYPE_ENCODING:
      *out_byte = LOOM_BYTECODE_TYPE_ENCODING;
      return iree_ok_status();
    case LOOM_TYPE_POOL:
      *out_byte = LOOM_BYTECODE_TYPE_POOL;
      return iree_ok_status();
    default:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown type kind %d",
                          (int)kind);
}

static iree_status_t loom_bytecode_encoding_role_byte(loom_encoding_role_t role,
                                                      uint8_t* out_byte) {
  switch (role) {
    case LOOM_ENCODING_ROLE_UNKNOWN:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_UNKNOWN;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_ADDRESS_LAYOUT:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_LAYOUT;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_STORAGE_SCHEMA:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_SCHEMA;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_PHYSICAL_STORAGE:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_STORAGE;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM:
      *out_byte = LOOM_BYTECODE_ENCODING_ROLE_TRANSFORM;
      return iree_ok_status();
    case LOOM_ENCODING_ROLE_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown encoding role %d", (int)role);
}

// Writes the STRINGS section through the page writer.
static iree_status_t loom_bytecode_write_strings_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->string_count));
  for (iree_host_size_t i = 0; i < numbering->string_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, numbering->string_entries[i]));
  }
  return iree_ok_status();
}

// Writes the SOURCES section through the page writer.
static iree_status_t loom_bytecode_write_sources_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t source_count = module->sources.count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, source_count));
  for (iree_host_size_t i = 0; i < source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, module->sources.entries[i]));
  }
  return iree_ok_status();
}

// Writes the TYPES section through the page writer.
static iree_status_t loom_bytecode_write_types_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->type_count));

  for (uint32_t writer_type_id = 0; writer_type_id < numbering->type_count;
       ++writer_type_id) {
    iree_host_size_t module_index = numbering->type_order[writer_type_id];
    loom_type_t type = module->types.entries[module_index];
    loom_type_kind_t kind = loom_type_kind(type);

    uint8_t kind_byte = 0;
    IREE_RETURN_IF_ERROR(loom_bytecode_type_kind_byte(kind, &kind_byte));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, kind_byte));

    switch (kind) {
      case LOOM_TYPE_NONE:
        break;
      case LOOM_TYPE_SCALAR: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        break;
      }
      case LOOM_TYPE_TILE:
      case LOOM_TYPE_TENSOR:
      case LOOM_TYPE_VECTOR:
      case LOOM_TYPE_VIEW: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_element_type(type)));
        uint8_t rank = loom_type_rank(type);
        if (kind == LOOM_TYPE_VECTOR) {
          if (rank == 0) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "vector types must have rank >= 1");
          }
          if (type.encoding_id != 0 || type.encoding_flags != 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "vector types must not carry encoding or layout attachments");
          }
        }
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, rank));
        // Encoding.
        if (loom_type_has_ssa_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_SSA));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        } else if (loom_type_has_static_encoding(type)) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_STATIC));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, type.encoding_id));
        } else {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
              page_writer, LOOM_BYTECODE_ENCODING_ATTACHMENT_NONE));
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_uvarint(page_writer, 0));
        }
        // Dims.
        for (uint8_t i = 0; i < rank; ++i) {
          if (loom_type_dim_is_dynamic_at(type, i)) {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 1));
          } else {
            IREE_RETURN_IF_ERROR(
                loom_bytecode_page_writer_write_u8(page_writer, 0));
            IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
                page_writer, (uint64_t)loom_type_dim_static_size_at(type, i)));
          }
        }
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        const loom_func_type_data_t* func_data = loom_type_func_data(type);
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->arg_count));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, func_data->result_count));
        iree_host_size_t type_count =
            (iree_host_size_t)func_data->arg_count + func_data->result_count;
        for (iree_host_size_t i = 0; i < type_count; ++i) {
          uint32_t sub_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, func_data->types[i], &sub_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, sub_type_id));
        }
        break;
      }
      case LOOM_TYPE_DIALECT: {
        loom_string_id_t name_id = loom_type_dialect_name_id(type);
        uint32_t name_writer_id = 0;
        if (name_id < numbering->module->strings.count) {
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
              numbering, numbering->module->strings.entries[name_id],
              &name_writer_id));
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, name_writer_id));
        uint16_t param_count = loom_type_dialect_param_count(type);
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_uvarint(page_writer, param_count));
        const loom_type_t* params = loom_type_dialect_params(type);
        for (uint16_t i = 0; i < param_count; ++i) {
          uint32_t param_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, params[i], &param_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, param_type_id));
        }
        break;
      }
      case LOOM_TYPE_PARAMETERIZED: {
        const loom_parameterized_type_descriptor_t* descriptor =
            loom_type_parameterized_descriptor(type);
        const uint8_t parameter_count =
            loom_type_parameterized_parameter_count(type);
        const loom_attribute_t* parameters =
            loom_type_parameterized_parameters(type);
        uint32_t family_name_id = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
            numbering, loom_bstring_view(descriptor->name), &family_name_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, family_name_id));
        uint8_t present_count = 0;
        for (uint8_t i = 0; i < parameter_count; ++i) {
          if (!loom_attr_is_absent(parameters[i])) ++present_count;
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, present_count));
        for (uint8_t i = 0; i < parameter_count; ++i) {
          if (loom_attr_is_absent(parameters[i])) continue;
          const loom_attr_descriptor_t* parameter_descriptor =
              &descriptor->parameter_descriptors[i];
          uint32_t parameter_name_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_string_view(
              numbering, loom_attr_descriptor_name(parameter_descriptor),
              &parameter_name_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, parameter_name_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
              page_writer, numbering, /*value_numbering=*/NULL, parameters[i],
              parameter_descriptor));
        }
        break;
      }
      case LOOM_TYPE_REGISTER: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, loom_type_register_payload0(type)));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, loom_type_register_payload1(type)));
        const loom_type_t* value_type = loom_type_register_value_type(type);
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, value_type ? 1 : 0));
        if (value_type) {
          uint32_t value_type_id = 0;
          IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_type(
              numbering, *value_type, &value_type_id));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, value_type_id));
        }
        break;
      }
      case LOOM_TYPE_STORAGE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_u8(
            page_writer, (uint8_t)loom_type_storage_space(type)));
        break;
      }
      case LOOM_TYPE_ENCODING: {
        uint8_t role_byte = 0;
        IREE_RETURN_IF_ERROR(loom_bytecode_encoding_role_byte(
            loom_type_encoding_role(type), &role_byte));
        IREE_RETURN_IF_ERROR(
            loom_bytecode_page_writer_write_u8(page_writer, role_byte));
        break;
      }
      case LOOM_TYPE_BUFFER:
        // No additional data.
        break;
      case LOOM_TYPE_POOL: {
        uint64_t dim = loom_type_dim(type, 0);
        if (loom_dim_is_dynamic(dim)) {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 1));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_bytecode_page_writer_write_u8(page_writer, 0));
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, (uint64_t)loom_dim_static_size(dim)));
        }
        break;
      }
      default:
        // Unreachable: loom_bytecode_type_kind_byte above rejects
        // unknown kinds before we get here.
        break;
    }
  }

  return iree_ok_status();
}

// Writes the ENCODINGS section: encoding family registry + instances.
static iree_status_t loom_bytecode_write_encodings_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;

  // Build the encoding family registry from unique encoding names.
  // Small (typically <10 families), so linear dedup is fine.
  typedef struct {
    loom_string_id_t name_id;
    uint32_t writer_string_id;
  } encoding_family_entry_t;
  encoding_family_entry_t family_entries[256];
  iree_host_size_t family_count = 0;

  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    loom_string_id_t name_id = module->encodings.entries[i].name_id;
    bool found = false;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == name_id) {
        found = true;
        break;
      }
    }
    if (!found) {
      if (family_count >= 256) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "more than 256 unique encoding families");
      }
      uint32_t writer_string_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, name_id, &writer_string_id));
      family_entries[family_count++] = (encoding_family_entry_t){
          .name_id = name_id, .writer_string_id = writer_string_id};
    }
  }

  // Encoding family count and family name string IDs.
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, family_count));
  for (iree_host_size_t k = 0; k < family_count; ++k) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, family_entries[k].writer_string_id));
  }

  // Encoding instances.
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, module->encodings.count));
  for (iree_host_size_t i = 0; i < module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &module->encodings.entries[i];

    // Find the family index for this encoding's name.
    uint32_t family_index = 0;
    for (iree_host_size_t k = 0; k < family_count; ++k) {
      if (family_entries[k].name_id == encoding->name_id) {
        family_index = (uint32_t)k;
        break;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_uvarint(page_writer, family_index));

    // Alias string ID plus one (0 = no alias).
    uint32_t alias_string_id_plus1 = 0;
    if (encoding->alias_id != LOOM_STRING_ID_INVALID) {
      uint32_t alias_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, encoding->alias_id, &alias_writer_id));
      alias_string_id_plus1 = alias_writer_id + 1;
    }
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, alias_string_id_plus1));

    // Parameters as structured named attributes, using the same
    // attribute serialization as the IR section.
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, encoding->attribute_count));
    for (uint8_t p = 0; p < encoding->attribute_count; ++p) {
      const loom_named_attr_t* attr = &encoding->attributes[p];
      uint32_t param_name_writer_id = 0;
      IREE_RETURN_IF_ERROR(loom_bytecode_numbering_intern_module_string(
          numbering, attr->name_id, &param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
          page_writer, param_name_writer_id));
      IREE_RETURN_IF_ERROR(loom_bytecode_write_attr_value(
          page_writer, numbering, NULL, attr->value, NULL));
    }
  }

  return iree_ok_status();
}

// Writes the OPS section through the page writer.
static iree_status_t loom_bytecode_write_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->op_count));
  for (uint32_t i = 0; i < numbering->op_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, numbering->op_entries[i].string_writer_id));
  }
  return iree_ok_status();
}

// Writes the LOCATIONS section: location_count + per-entry serialization.
static iree_status_t loom_bytecode_write_locations_section(
    loom_bytecode_page_writer_t* page_writer, const loom_module_t* module) {
  iree_host_size_t location_count = module->locations.count;
  IREE_RETURN_IF_ERROR(
      loom_bytecode_page_writer_write_uvarint(page_writer, location_count));
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    const loom_location_entry_t* entry = &module->locations.entries[i];
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, (uint8_t)entry->kind));
    IREE_RETURN_IF_ERROR(
        loom_bytecode_page_writer_write_u8(page_writer, entry->flags));
    switch (entry->kind) {
      case LOOM_LOCATION_NONE:
        break;
      case LOOM_LOCATION_FILE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.start_col));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_line));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->file.end_col));
        break;
      }
      case LOOM_LOCATION_FUSED: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->fused.count));
        for (uint32_t c = 0; c < entry->fused.count; ++c) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
              page_writer, entry->fused.children[c]));
        }
        break;
      }
      case LOOM_LOCATION_OPAQUE: {
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.source_id));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->opaque.data_length));
        if (entry->opaque.data_length > 0) {
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
              page_writer, entry->opaque.data, entry->opaque.data_length));
        }
        break;
      }
      case LOOM_LOCATION_TAGGED: {
        if (entry->tagged.tag == LOOM_LOCATION_TAG_INVALID) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "tagged location has invalid tag 0");
        }
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->tagged.tag));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->tagged.child));
        IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
            page_writer, entry->tagged.data_length));
        if (entry->tagged.data_length > 0) {
          if (!entry->tagged.data) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "tagged location has data_length %u but NULL data",
                entry->tagged.data_length);
          }
          IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write(
              page_writer, entry->tagged.data, entry->tagged.data_length));
        }
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown location kind %d", (int)entry->kind);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Top-level writer
//===----------------------------------------------------------------------===//

static iree_status_t loom_bytecode_validate_module(
    const loom_module_t* module) {
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (needed for op vtables)");
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_symbol_kind_t bytecode_kind = loom_symbol_bytecode_kind(symbol);
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL) ||
        bytecode_kind == LOOM_SYMBOL_GLOBAL) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz " has no defining op",
                                i);
      }
      const loom_op_t* op = symbol->defining_op;
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable ||
          !iree_all_bits_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE) ||
          !vtable->symbol_def ||
          vtable->symbol_def->bytecode_kind != LOOM_SYMBOL_GLOBAL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GLOBAL symbol %" PRIhsz
                                " defining op does not use the GLOBAL "
                                "bytecode payload",
                                i);
      }
      if (op->operand_count != 0 || op->region_count != 0 ||
          op->tied_result_count != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op must not have operands, regions, or tied results",
            i);
      }
      if (op->result_count == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz " defining op must have results", i);
      }
      if (op->attribute_count > 0 && !vtable->attr_descriptors) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "GLOBAL symbol %" PRIhsz
            " defining op has attributes but no descriptors",
            i);
      }
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_EXECUTABLE) ||
        bytecode_kind == LOOM_SYMBOL_EXECUTABLE) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "EXECUTABLE symbols not yet supported");
    }
    if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RECORD) ||
        bytecode_kind == LOOM_SYMBOL_RECORD) {
      if (!symbol->defining_op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "RECORD symbol %" PRIhsz " has no defining op",
                                i);
      }
      IREE_RETURN_IF_ERROR(
          loom_bytecode_validate_record_symbol_op(module, symbol->defining_op));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_bytecode_write_module(
    const loom_module_t* module, iree_io_stream_t* stream,
    const loom_bytecode_write_options_t* options,
    iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, loom_bytecode_validate_module(module));

  // Check stream capabilities.
  iree_io_stream_mode_t mode = iree_io_stream_mode(stream);
  if (!(mode & IREE_IO_STREAM_MODE_WRITABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not writable");
  }
  if (!(mode & IREE_IO_STREAM_MODE_SEEKABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "stream is not seekable (needed for directory "
                            "patching)");
  }

  iree_string_view_t producer = (options && options->producer.size > 0)
                                    ? options->producer
                                    : IREE_SV(LOOM_BYTECODE_DEFAULT_PRODUCER);
  loom_bytecode_location_mode_t location_mode =
      options ? options->location_mode
              : LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS;
  if (location_mode > LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported bytecode location mode %u",
                            (unsigned)location_mode);
  }
  if (location_mode == LOOM_BYTECODE_LOCATION_MODE_FULL_LOCATIONS) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "FULL_LOCATIONS bytecode mode requires field span emission");
  }

  // Temporary arena for all working memory. All numbering tables,
  // value maps, and scratch allocations come from here. Deinitialized
  // at the end, returning all blocks to the shared pool in O(1).
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  // Initialize the page writer.
  loom_bytecode_page_writer_t page_writer;
  loom_bytecode_page_writer_initialize(&page_writer, stream);

  // Initialize numbering context.
  loom_bytecode_numbering_t numbering;
  iree_status_t status =
      loom_bytecode_numbering_initialize(&numbering, module, &arena);
  numbering.location_mode = location_mode;
  numbering.low_repr_environment = options ? options->low_repr_environment
                                           : (loom_low_repr_environment_t){0};

  loom_module_record_plan_t record_plan = {0};
  bool record_plan_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_module_record_plan_initialize(module, &record_plan);
    record_plan_initialized = iree_status_is_ok(status);
  }
  loom_bytecode_provider_import_plan_t provider_import_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_provider_import_plan_initialize(
        module, &record_plan, &arena, &provider_import_plan);
  }

  // Pass 1: Number module metadata. Function signatures and bodies are numbered
  // during IR section writing.
  if (iree_status_is_ok(status)) {
    uint32_t unused_id = 0;
    status = loom_bytecode_numbering_intern_module_string(
        &numbering, module->name_id, &unused_id);
  }
  if (iree_status_is_ok(status)) {
    for (loom_symbol_id_t wire_ordinal = 0;
         wire_ordinal < module->symbols.count && iree_status_is_ok(status);
         ++wire_ordinal) {
      const loom_symbol_id_t module_symbol_id =
          loom_bytecode_module_symbol_id(&numbering, wire_ordinal);
      uint32_t unused_id = 0;
      status = loom_bytecode_numbering_intern_module_string(
          &numbering, module->symbols.entries[module_symbol_id].name_id,
          &unused_id);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_number_provider_imports(&numbering, &record_plan,
                                                   &provider_import_plan);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0;
         i < module->encodings.count && iree_status_is_ok(status); ++i) {
      status = loom_bytecode_number_encoding(&numbering, (uint16_t)(i + 1));
    }
  }
  loom_bytecode_symbol_reference_plan_t symbol_reference_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_symbol_reference_plan_initialize(
        module, &arena, &symbol_reference_plan);
  }

  // File header: magic, version, location mode, module count, producer string.
  iree_string_view_t module_name = module->strings.entries[module->name_id];

  if (iree_status_is_ok(status)) {
    // Magic.
    status = loom_bytecode_page_writer_write(&page_writer, LOOM_BYTECODE_MAGIC,
                                             LOOM_BYTECODE_MAGIC_LENGTH);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer,
                                                LOOM_BYTECODE_FORMAT_VERSION);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u8(&page_writer, location_mode);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    1);  // module_count
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u32_le(
        &page_writer, (uint32_t)module_name.size);  // string_pool_length
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // reserved
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_null_terminated_string(
        &page_writer, producer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module directory entry. module_offset and module_length are
  // written as placeholders and patched after all sections are written.
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_u32_le(&page_writer, 0);  // name_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    (uint16_t)module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u16_le(&page_writer,
                                                    0);  // module_flags
  }
  // module_offset placeholder: we'll patch this as part of the directory entry.
  iree_host_size_t module_dir_offset_position = 0;
  if (iree_status_is_ok(status)) {
    module_dir_offset_position = page_writer.total_written;
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_offset
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_u64_le(&page_writer,
                                                    0);  // module_length
  }

  // File string pool: module name(s).
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write(&page_writer, module_name.data,
                                             module_name.size);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_pad_to_alignment(&page_writer, 8);
  }

  // Module data starts at this offset.
  iree_host_size_t module_start = page_writer.total_written;

  loom_bytecode_section_kind_t section_write_order[LOOM_BYTECODE_SECTION_COUNT];
  iree_host_size_t section_count = 0;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_IR;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SYMBOLS;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS;
  section_write_order[section_count++] =
      LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_STRINGS;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SOURCES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_TYPES;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_ENCODINGS;
  section_write_order[section_count++] = LOOM_BYTECODE_SECTION_OPS;
  if (location_mode != LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    section_write_order[section_count++] = LOOM_BYTECODE_SECTION_LOCATIONS;
  }
  iree_host_size_t file_header_line_count = 0;
  (void)loom_module_file_header(module, &file_header_line_count);
  if (file_header_line_count > 0) {
    section_write_order[section_count++] = LOOM_BYTECODE_SECTION_SOURCE_TRIVIA;
  }
  loom_bytecode_body_counts_t module_counts = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_count_serialized_bodies(&numbering, &module_counts);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_bytecode_page_writer_write_uvarint(&page_writer, section_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.value_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(
        &page_writer, module_counts.region_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.block_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_uvarint(&page_writer,
                                                     module_counts.op_count);
  }

  // Section directory placeholder — patched after all sections are written.
  iree_host_size_t section_dir_patch_position = page_writer.total_written;
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_write_zeros(
        &page_writer,
        section_count * sizeof(loom_bytecode_section_dir_entry_t));
  }

  // Track section offsets and lengths.
  uint64_t section_offsets[LOOM_BYTECODE_SECTION_COUNT] = {0};
  uint64_t section_lengths[LOOM_BYTECODE_SECTION_COUNT] = {0};

  // Allocate root-region payload tracking from the arena.
  loom_bytecode_ir_region_list_t* ir_regions = NULL;
  if (iree_status_is_ok(status) && module->symbols.count > 0) {
    status =
        iree_arena_allocate_array(&arena, module->symbols.count,
                                  sizeof(*ir_regions), (void**)&ir_regions);
    if (iree_status_is_ok(status)) {
      memset(ir_regions, 0, module->symbols.count * sizeof(*ir_regions));
    }
  }

  // IR section: independently bounded root regions streamed through the page
  // writer.
  // Written first so the numbering tables grow as entities are encountered.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_IR] =
        page_writer.total_written - module_start;
    status =
        loom_bytecode_write_ir_section(&page_writer, &numbering, ir_regions);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_IR] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_IR];
    }
  }

  // Symbols section: buffered in a string builder because the import/export
  // offset tables at the start reference entry positions that come later.
  // The SYMBOLS section uses a string_builder (which needs realloc, so it
  // can't use the arena). Use the module's context allocator.
  iree_string_builder_t symbols_builder;
  iree_string_builder_initialize(module->context->allocator, &symbols_builder);
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_write_symbols_section(&symbols_builder, &numbering,
                                                 ir_regions);
  }
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOLS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_page_writer_write(
        &page_writer, iree_string_builder_buffer(&symbols_builder),
        iree_string_builder_size(&symbols_builder));
    section_lengths[LOOM_BYTECODE_SECTION_SYMBOLS] =
        iree_string_builder_size(&symbols_builder);
  }
  iree_string_builder_deinitialize(&symbols_builder);

  // Provider imports use direct SYMBOLS ordinals and the canonical
  // keyed-module-record projection.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_provider_imports_section(
        &page_writer, &numbering, &record_plan, &provider_import_plan);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_PROVIDER_IMPORTS];
    }
  }

  // Symbol references preserve the direct metadata-only dependency graph.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_symbol_references_section(
        &page_writer, &numbering, &symbol_reference_plan);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SYMBOL_REFERENCES];
    }
  }

  // Strings section: all interned strings from the numbering context.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_STRINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_strings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_STRINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_STRINGS];
    }
  }

  // Sources section: module-local source identifiers (filenames, tags).
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_sources_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCES];
    }
  }

  // Types section: interned type table in topological order.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_TYPES] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_types_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_TYPES] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_TYPES];
    }
  }

  // Encodings section: kind registry + parameterized instances.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_encodings_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_ENCODINGS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_ENCODINGS];
    }
  }

  // Ops section: op kind name registry.
  if (iree_status_is_ok(status)) {
    section_offsets[LOOM_BYTECODE_SECTION_OPS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_ops_section(&page_writer, &numbering);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_OPS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_OPS];
    }
  }

  // Locations section.
  if (iree_status_is_ok(status) &&
      location_mode != LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS) {
    section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_locations_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_LOCATIONS] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_LOCATIONS];
    }
  }

  // Module-owned source presentation.
  if (iree_status_is_ok(status) && file_header_line_count > 0) {
    section_offsets[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA] =
        page_writer.total_written - module_start;
    status = loom_bytecode_write_source_trivia_section(&page_writer, module);
    if (iree_status_is_ok(status)) {
      section_lengths[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA] =
          page_writer.total_written - module_start -
          section_offsets[LOOM_BYTECODE_SECTION_SOURCE_TRIVIA];
    }
  }

  // Flush remaining page buffer, then seek back to patch the section
  // directory and module directory with correct offsets and lengths.
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_page_writer_flush(&page_writer);
  }

  // Patch section directory.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)section_dir_patch_position);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < section_count && iree_status_is_ok(status);
         ++i) {
      loom_bytecode_section_kind_t kind = section_write_order[i];
      uint8_t entry[sizeof(loom_bytecode_section_dir_entry_t)] = {0};
      entry[0] = (uint8_t)kind;
      entry[1] = (uint8_t)((uint16_t)kind >> 8);
      uint64_t section_offset = section_offsets[kind];
      uint64_t section_length = section_lengths[kind];
      for (int byte_index = 0; byte_index < 8; ++byte_index) {
        entry[8 + byte_index] = (uint8_t)(section_offset >> (byte_index * 8));
        entry[16 + byte_index] = (uint8_t)(section_length >> (byte_index * 8));
      }
      status = iree_io_stream_write(stream, sizeof(entry), entry);
    }
  }

  // Patch module directory: module_offset and module_length.
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET,
                            (iree_io_stream_pos_t)module_dir_offset_position);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_offset = module_start;
    status = iree_io_stream_write(stream, 8, &module_offset);
  }
  if (iree_status_is_ok(status)) {
    uint64_t module_length = page_writer.total_written - module_start;
    status = iree_io_stream_write(stream, 8, &module_length);
  }

  // All numbering tables, value maps, and IR region lists were arena-allocated.
  // One call returns all blocks to the shared pool.
  if (record_plan_initialized) {
    loom_module_record_plan_deinitialize(&record_plan);
  }
  iree_arena_deinitialize(&arena);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
