// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/tables.h"

#include "loom/format/bytecode/writer/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ops/op_defs.h"

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
iree_status_t loom_bytecode_write_strings_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->strings.count));
  for (iree_host_size_t i = 0; i < numbering->strings.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_string(
        page_writer, numbering->strings.values[i]));
  }
  return iree_ok_status();
}

// Writes the SOURCES section through the page writer.
iree_status_t loom_bytecode_write_sources_section(
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
iree_status_t loom_bytecode_write_types_section(
    loom_bytecode_page_writer_t* page_writer,
    loom_bytecode_numbering_t* numbering) {
  const loom_module_t* module = numbering->module;
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->types.count));

  for (uint32_t writer_type_id = 0; writer_type_id < numbering->types.count;
       ++writer_type_id) {
    iree_host_size_t module_index =
        numbering->types.module_indices_by_writer_id[writer_type_id];
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
iree_status_t loom_bytecode_write_encodings_section(
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
iree_status_t loom_bytecode_write_ops_section(
    loom_bytecode_page_writer_t* page_writer,
    const loom_bytecode_numbering_t* numbering) {
  IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
      page_writer, numbering->ops.count));
  for (uint32_t i = 0; i < numbering->ops.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_bytecode_page_writer_write_uvarint(
        page_writer, numbering->ops.values[i].string_writer_id));
  }
  return iree_ok_status();
}

// Writes the LOCATIONS section: location_count + per-entry serialization.
iree_status_t loom_bytecode_write_locations_section(
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
