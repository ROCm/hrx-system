// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/text_writer.h"

#include <inttypes.h>

#include "loom/target/emit/llvmir/types.h"

static const loom_llvmir_type_t* loom_llvmir_text_type(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id) {
  return type_id < module->type_count ? &module->types[type_id] : NULL;
}

static const loom_llvmir_value_t* loom_llvmir_text_value(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  return value_id < module->value_count ? &module->values[value_id] : NULL;
}

static loom_llvmir_type_id_t loom_llvmir_text_value_type(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  const loom_llvmir_value_t* value = loom_llvmir_text_value(module, value_id);
  return value ? value->type_id : LOOM_LLVMIR_TYPE_ID_INVALID;
}

static loom_llvmir_function_t* loom_llvmir_text_function(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id) {
  return function_id < module->function_count ? module->functions[function_id]
                                              : NULL;
}

static loom_llvmir_global_t* loom_llvmir_text_global(
    const loom_llvmir_module_t* module, loom_llvmir_global_id_t global_id) {
  return global_id < module->global_count ? module->globals[global_id] : NULL;
}

static iree_status_t loom_llvmir_write_label_ref(
    const loom_llvmir_function_t* function, loom_llvmir_block_id_t block_id,
    loom_output_stream_t* stream) {
  if (block_id >= function->block_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM text writer saw unknown block");
  }
  const loom_llvmir_block_t* block = function->blocks[block_id];
  if (!iree_string_view_is_empty(block->name)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
    return loom_output_stream_write(stream, block->name);
  }
  return loom_output_stream_write_format(stream, "%%bb%u", block_id);
}

static iree_status_t loom_llvmir_write_label_def(
    const loom_llvmir_function_t* function, const loom_llvmir_block_t* block,
    loom_output_stream_t* stream) {
  if (!iree_string_view_is_empty(block->name)) {
    return loom_output_stream_write(stream, block->name);
  }
  return loom_output_stream_write_format(stream, "bb%u", block->id);
}

static iree_status_t loom_llvmir_write_escaped_string(
    loom_output_stream_t* stream, iree_string_view_t string) {
  static const char hex_digits[] = "0123456789ABCDEF";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  for (iree_host_size_t i = 0; i < string.size; ++i) {
    uint8_t character = (uint8_t)string.data[i];
    if (character == '"' || character == '\\' || character < 0x20 ||
        character >= 0x7F) {
      char escaped[3] = {
          '\\',
          hex_digits[character >> 4],
          hex_digits[character & 0xF],
      };
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write(stream, iree_make_string_view(escaped, 3)));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, iree_make_string_view((const char*)&character, 1)));
    }
  }
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_llvmir_write_type(const loom_llvmir_module_t* module,
                                            loom_llvmir_type_id_t type_id,
                                            loom_output_stream_t* stream) {
  const loom_llvmir_type_t* type = loom_llvmir_text_type(module, type_id);
  if (!type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM text writer saw unknown type");
  }
  switch (type->kind) {
    case LOOM_LLVMIR_TYPE_VOID:
      return loom_output_stream_write_cstring(stream, "void");
    case LOOM_LLVMIR_TYPE_INTEGER:
      return loom_output_stream_write_format(stream, "i%u", type->bit_width);
    case LOOM_LLVMIR_TYPE_FLOAT:
      switch (type->float_kind) {
        case LOOM_LLVMIR_FLOAT_F16:
          return loom_output_stream_write_cstring(stream, "half");
        case LOOM_LLVMIR_FLOAT_BF16:
          return loom_output_stream_write_cstring(stream, "bfloat");
        case LOOM_LLVMIR_FLOAT_F32:
          return loom_output_stream_write_cstring(stream, "float");
        case LOOM_LLVMIR_FLOAT_F64:
          return loom_output_stream_write_cstring(stream, "double");
        default:
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unknown LLVM float kind");
      }
    case LOOM_LLVMIR_TYPE_POINTER:
      if (type->address_space == 0) {
        return loom_output_stream_write_cstring(stream, "ptr");
      }
      return loom_output_stream_write_format(stream, "ptr addrspace(%u)",
                                             type->address_space);
    case LOOM_LLVMIR_TYPE_VECTOR: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "<%u x ", type->element_count));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_type(module, type->element_type, stream));
      return loom_output_stream_write_char(stream, '>');
    }
    case LOOM_LLVMIR_TYPE_STRUCT: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{ "));
      for (uint32_t i = 0; i < type->element_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(
            loom_llvmir_write_type(module, type->element_types[i], stream));
      }
      return loom_output_stream_write_cstring(stream, " }");
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM type kind");
  }
}

static iree_status_t loom_llvmir_write_fallback_value_name(
    loom_llvmir_value_id_t value_id, loom_output_stream_t* stream) {
  return loom_output_stream_write_format(stream, "%%v%u", value_id);
}

static iree_status_t loom_llvmir_write_integer_vector_constant(
    const loom_llvmir_module_t* module, const loom_llvmir_value_t* value,
    loom_output_stream_t* stream) {
  const loom_llvmir_type_t* vector_type =
      loom_llvmir_text_type(module, value->type_id);
  if (!vector_type || vector_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      vector_type->element_type >= module->type_count ||
      value->integer_vector.value_count != vector_type->element_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM text writer saw invalid integer vector constant");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '<'));
  for (iree_host_size_t i = 0; i < value->integer_vector.value_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_type(module, vector_type->element_type, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, " %" PRIu64, value->integer_vector.values[i]));
  }
  return loom_output_stream_write_char(stream, '>');
}

static iree_status_t loom_llvmir_write_value_ref(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id,
    loom_output_stream_t* stream) {
  const loom_llvmir_value_t* value = loom_llvmir_text_value(module, value_id);
  if (!value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM text writer saw unknown value");
  }
  switch (value->kind) {
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER:
      return loom_output_stream_write_format(stream, "%" PRIu64,
                                             value->integer_value);
    case LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS:
      return loom_output_stream_write_format(stream, "0x%016" PRIX64,
                                             value->float_bits);
    case LOOM_LLVMIR_VALUE_CONSTANT_NULL:
      return loom_output_stream_write_cstring(stream, "null");
    case LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR:
      return loom_llvmir_write_integer_vector_constant(module, value, stream);
    case LOOM_LLVMIR_VALUE_CONSTANT_POISON:
      return loom_output_stream_write_cstring(stream, "poison");
    case LOOM_LLVMIR_VALUE_GLOBAL: {
      loom_llvmir_global_t* global =
          loom_llvmir_text_global(module, value->global.global_id);
      if (!global) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM text writer saw unknown global");
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '@'));
      return loom_output_stream_write(stream, global->name);
    }
    case LOOM_LLVMIR_VALUE_PARAMETER:
    case LOOM_LLVMIR_VALUE_INSTRUCTION:
      if (!iree_string_view_is_empty(value->name)) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
        return loom_output_stream_write(stream, value->name);
      }
      return loom_llvmir_write_fallback_value_name(value_id, stream);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM value kind");
  }
}

static iree_status_t loom_llvmir_write_typed_value_ref(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id,
    loom_output_stream_t* stream) {
  loom_llvmir_type_id_t type_id = loom_llvmir_text_value_type(module, value_id);
  if (type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM text writer saw value with unknown type");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_write_type(module, type_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
  return loom_llvmir_write_value_ref(module, value_id, stream);
}

static iree_status_t loom_llvmir_write_attr(const loom_llvmir_module_t* module,
                                            const loom_llvmir_attr_t* attr,
                                            loom_output_stream_t* stream) {
  switch (attr->kind) {
    case LOOM_LLVMIR_ATTR_ALIGN:
      return loom_output_stream_write_format(stream, "align %" PRIu64,
                                             attr->value);
    case LOOM_LLVMIR_ATTR_NOALIAS:
      return loom_output_stream_write_cstring(stream, "noalias");
    case LOOM_LLVMIR_ATTR_READONLY:
      return loom_output_stream_write_cstring(stream, "readonly");
    case LOOM_LLVMIR_ATTR_WRITEONLY:
      return loom_output_stream_write_cstring(stream, "writeonly");
    case LOOM_LLVMIR_ATTR_READNONE:
      return loom_output_stream_write_cstring(stream, "readnone");
    case LOOM_LLVMIR_ATTR_NOUNDEF:
      return loom_output_stream_write_cstring(stream, "noundef");
    case LOOM_LLVMIR_ATTR_NONNULL:
      return loom_output_stream_write_cstring(stream, "nonnull");
    case LOOM_LLVMIR_ATTR_INREG:
      return loom_output_stream_write_cstring(stream, "inreg");
    case LOOM_LLVMIR_ATTR_ALWAYSINLINE:
      return loom_output_stream_write_cstring(stream, "alwaysinline");
    case LOOM_LLVMIR_ATTR_NOCAPTURE:
      return loom_output_stream_write_cstring(stream, "nocapture");
    case LOOM_LLVMIR_ATTR_IMMARG:
      return loom_output_stream_write_cstring(stream, "immarg");
    case LOOM_LLVMIR_ATTR_RANGE: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "range("));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_type(module, attr->type_id, stream));
      return loom_output_stream_write_format(
          stream, " %" PRIu64 ", %" PRIu64 ")", attr->value, attr->value2);
    }
    case LOOM_LLVMIR_ATTR_STRING_KEY:
      return loom_llvmir_write_escaped_string(stream, attr->key);
    case LOOM_LLVMIR_ATTR_STRING_KEY_VALUE: {
      IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(stream, attr->key));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '='));
      return loom_llvmir_write_escaped_string(stream, attr->string_value);
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM attribute kind");
  }
}

static iree_status_t loom_llvmir_write_attr_list(
    const loom_llvmir_module_t* module, loom_llvmir_attr_list_t attrs,
    loom_output_stream_t* stream) {
  for (iree_host_size_t i = 0; i < attrs.attr_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_attr(module, &attrs.attrs[i], stream));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_metadata_attachments(
    const loom_llvmir_metadata_attachment_storage_t* attachments,
    iree_host_size_t attachment_count, const char* separator,
    loom_output_stream_t* stream) {
  if (attachment_count != 0 && attachments == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM text writer saw null metadata attachment storage");
  }
  for (iree_host_size_t i = 0; i < attachment_count; ++i) {
    const loom_llvmir_metadata_attachment_storage_t* attachment =
        &attachments[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, separator));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '!'));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, attachment->name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, " !%u", attachment->metadata_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_parameter(
    const loom_llvmir_module_t* module,
    const loom_llvmir_parameter_t* parameter, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_type(module, parameter->type_id, stream));
  if (parameter->attrs.attr_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_attr_list(module, parameter->attrs, stream));
  }
  if (!iree_string_view_is_empty(parameter->name)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " %"));
    return loom_output_stream_write(stream, parameter->name);
  }
  return iree_ok_status();
}

static const char* loom_llvmir_linkage_spelling(loom_llvmir_linkage_t linkage) {
  switch (linkage) {
    case LOOM_LLVMIR_LINKAGE_DEFAULT:
      return "";
    case LOOM_LLVMIR_LINKAGE_DSO_LOCAL:
      return "dso_local ";
    case LOOM_LLVMIR_LINKAGE_INTERNAL:
      return "internal ";
    case LOOM_LLVMIR_LINKAGE_PRIVATE:
      return "private ";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_calling_convention_spelling(
    loom_llvmir_calling_convention_t calling_convention) {
  switch (calling_convention) {
    case LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT:
      return "";
    case LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL:
      return "amdgpu_kernel ";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_binop_spelling(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
      return "add";
    case LOOM_LLVMIR_BINOP_SUB:
      return "sub";
    case LOOM_LLVMIR_BINOP_MUL:
      return "mul";
    case LOOM_LLVMIR_BINOP_UDIV:
      return "udiv";
    case LOOM_LLVMIR_BINOP_SDIV:
      return "sdiv";
    case LOOM_LLVMIR_BINOP_UREM:
      return "urem";
    case LOOM_LLVMIR_BINOP_SREM:
      return "srem";
    case LOOM_LLVMIR_BINOP_AND:
      return "and";
    case LOOM_LLVMIR_BINOP_OR:
      return "or";
    case LOOM_LLVMIR_BINOP_XOR:
      return "xor";
    case LOOM_LLVMIR_BINOP_SHL:
      return "shl";
    case LOOM_LLVMIR_BINOP_LSHR:
      return "lshr";
    case LOOM_LLVMIR_BINOP_ASHR:
      return "ashr";
    case LOOM_LLVMIR_BINOP_FADD:
      return "fadd";
    case LOOM_LLVMIR_BINOP_FSUB:
      return "fsub";
    case LOOM_LLVMIR_BINOP_FMUL:
      return "fmul";
    case LOOM_LLVMIR_BINOP_FDIV:
      return "fdiv";
    case LOOM_LLVMIR_BINOP_FREM:
      return "frem";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_atomic_rmw_op_spelling(
    loom_llvmir_atomic_rmw_op_t op) {
  switch (op) {
    case LOOM_LLVMIR_ATOMIC_RMW_XCHG:
      return "xchg";
    case LOOM_LLVMIR_ATOMIC_RMW_ADD:
      return "add";
    case LOOM_LLVMIR_ATOMIC_RMW_SUB:
      return "sub";
    case LOOM_LLVMIR_ATOMIC_RMW_AND:
      return "and";
    case LOOM_LLVMIR_ATOMIC_RMW_OR:
      return "or";
    case LOOM_LLVMIR_ATOMIC_RMW_XOR:
      return "xor";
    case LOOM_LLVMIR_ATOMIC_RMW_MAX:
      return "max";
    case LOOM_LLVMIR_ATOMIC_RMW_MIN:
      return "min";
    case LOOM_LLVMIR_ATOMIC_RMW_UMAX:
      return "umax";
    case LOOM_LLVMIR_ATOMIC_RMW_UMIN:
      return "umin";
    case LOOM_LLVMIR_ATOMIC_RMW_FADD:
      return "fadd";
    case LOOM_LLVMIR_ATOMIC_RMW_FMAX:
      return "fmax";
    case LOOM_LLVMIR_ATOMIC_RMW_FMIN:
      return "fmin";
    case LOOM_LLVMIR_ATOMIC_RMW_FMAXIMUM:
      return "fmaximum";
    case LOOM_LLVMIR_ATOMIC_RMW_FMINIMUM:
      return "fminimum";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_atomic_ordering_spelling(
    loom_llvmir_atomic_ordering_t ordering) {
  switch (ordering) {
    case LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC:
      return "monotonic";
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQUIRE:
      return "acquire";
    case LOOM_LLVMIR_ATOMIC_ORDERING_RELEASE:
      return "release";
    case LOOM_LLVMIR_ATOMIC_ORDERING_ACQ_REL:
      return "acq_rel";
    case LOOM_LLVMIR_ATOMIC_ORDERING_SEQ_CST:
      return "seq_cst";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_unop_spelling(loom_llvmir_unop_t op) {
  switch (op) {
    case LOOM_LLVMIR_UNOP_FNEG:
      return "fneg";
    default:
      return NULL;
  }
}

static bool loom_llvmir_binop_is_float(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_FADD:
    case LOOM_LLVMIR_BINOP_FSUB:
    case LOOM_LLVMIR_BINOP_FMUL:
    case LOOM_LLVMIR_BINOP_FDIV:
    case LOOM_LLVMIR_BINOP_FREM:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_binop_allows_no_wrap_flags(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_ADD:
    case LOOM_LLVMIR_BINOP_SUB:
    case LOOM_LLVMIR_BINOP_MUL:
    case LOOM_LLVMIR_BINOP_SHL:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_binop_allows_exact_flag(loom_llvmir_binop_t op) {
  switch (op) {
    case LOOM_LLVMIR_BINOP_UDIV:
    case LOOM_LLVMIR_BINOP_SDIV:
    case LOOM_LLVMIR_BINOP_LSHR:
    case LOOM_LLVMIR_BINOP_ASHR:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_write_integer_arithmetic_flags(
    loom_llvmir_binop_t op, loom_llvmir_integer_arithmetic_flags_t flags,
    loom_output_stream_t* stream) {
  const uint32_t known_flags = LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
                               LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP |
                               LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT;
  uint32_t flags_bits = (uint32_t)flags;
  if (flags_bits & ~known_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM integer arithmetic flags");
  }
  const uint32_t no_wrap_flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP |
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  if ((flags_bits & no_wrap_flags) &&
      !loom_llvmir_binop_allows_no_wrap_flags(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry no-wrap flags");
  }
  if ((flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT) &&
      !loom_llvmir_binop_allows_exact_flag(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop cannot carry exact flag");
  }
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " nuw"));
  }
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " nsw"));
  }
  if (flags_bits & LOOM_LLVMIR_INTEGER_ARITHMETIC_EXACT) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " exact"));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_fast_math_flags(
    loom_llvmir_fast_math_flags_t flags, loom_output_stream_t* stream) {
  const uint32_t known_flags = LOOM_LLVMIR_FAST_MATH_FAST;
  uint32_t flags_bits = (uint32_t)flags;
  if (flags_bits & ~known_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM fast-math flags");
  }
  if (flags_bits == LOOM_LLVMIR_FAST_MATH_FAST) {
    return loom_output_stream_write_cstring(stream, " fast");
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_REASSOC) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " reassoc"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_NANS) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " nnan"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_INFS) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " ninf"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " nsz"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_RECIPROCAL) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " arcp"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_ALLOW_CONTRACT) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " contract"));
  }
  if (flags_bits & LOOM_LLVMIR_FAST_MATH_APPROX_FUNC) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " afn"));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_binop_flags(
    const loom_llvmir_instruction_t* instruction,
    loom_output_stream_t* stream) {
  if (instruction->binop.integer_flags != LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE &&
      instruction->binop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binop mixes integer and fast-math flags");
  }
  if (instruction->binop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE) {
    if (!loom_llvmir_binop_is_float(instruction->binop.op)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM integer binop cannot carry fast-math "
                              "flags");
    }
    return loom_llvmir_write_fast_math_flags(instruction->binop.fast_math_flags,
                                             stream);
  }
  return loom_llvmir_write_integer_arithmetic_flags(
      instruction->binop.op, instruction->binop.integer_flags, stream);
}

static const char* loom_llvmir_icmp_predicate_spelling(
    loom_llvmir_icmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_ICMP_EQ:
      return "eq";
    case LOOM_LLVMIR_ICMP_NE:
      return "ne";
    case LOOM_LLVMIR_ICMP_UGT:
      return "ugt";
    case LOOM_LLVMIR_ICMP_UGE:
      return "uge";
    case LOOM_LLVMIR_ICMP_ULT:
      return "ult";
    case LOOM_LLVMIR_ICMP_ULE:
      return "ule";
    case LOOM_LLVMIR_ICMP_SGT:
      return "sgt";
    case LOOM_LLVMIR_ICMP_SGE:
      return "sge";
    case LOOM_LLVMIR_ICMP_SLT:
      return "slt";
    case LOOM_LLVMIR_ICMP_SLE:
      return "sle";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_fcmp_predicate_spelling(
    loom_llvmir_fcmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_FCMP_FALSE:
      return "false";
    case LOOM_LLVMIR_FCMP_OEQ:
      return "oeq";
    case LOOM_LLVMIR_FCMP_OGT:
      return "ogt";
    case LOOM_LLVMIR_FCMP_OGE:
      return "oge";
    case LOOM_LLVMIR_FCMP_OLT:
      return "olt";
    case LOOM_LLVMIR_FCMP_OLE:
      return "ole";
    case LOOM_LLVMIR_FCMP_ONE:
      return "one";
    case LOOM_LLVMIR_FCMP_ORD:
      return "ord";
    case LOOM_LLVMIR_FCMP_UNO:
      return "uno";
    case LOOM_LLVMIR_FCMP_UEQ:
      return "ueq";
    case LOOM_LLVMIR_FCMP_UGT:
      return "ugt";
    case LOOM_LLVMIR_FCMP_UGE:
      return "uge";
    case LOOM_LLVMIR_FCMP_ULT:
      return "ult";
    case LOOM_LLVMIR_FCMP_ULE:
      return "ule";
    case LOOM_LLVMIR_FCMP_UNE:
      return "une";
    case LOOM_LLVMIR_FCMP_TRUE:
      return "true";
    default:
      return NULL;
  }
}

static const char* loom_llvmir_cast_op_spelling(loom_llvmir_cast_op_t op) {
  switch (op) {
    case LOOM_LLVMIR_CAST_TRUNCATE:
      return "trunc";
    case LOOM_LLVMIR_CAST_ZERO_EXTEND:
      return "zext";
    case LOOM_LLVMIR_CAST_SIGN_EXTEND:
      return "sext";
    case LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT:
      return "fptoui";
    case LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT:
      return "fptosi";
    case LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP:
      return "uitofp";
    case LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP:
      return "sitofp";
    case LOOM_LLVMIR_CAST_FP_TRUNCATE:
      return "fptrunc";
    case LOOM_LLVMIR_CAST_FP_EXTEND:
      return "fpext";
    case LOOM_LLVMIR_CAST_PTR_TO_INT:
      return "ptrtoint";
    case LOOM_LLVMIR_CAST_INT_TO_PTR:
      return "inttoptr";
    case LOOM_LLVMIR_CAST_BITCAST:
      return "bitcast";
    case LOOM_LLVMIR_CAST_ADDRESS_SPACE_CAST:
      return "addrspacecast";
    case LOOM_LLVMIR_CAST_PTR_TO_ADDR:
      return "ptrtoaddr";
    default:
      return NULL;
  }
}

static iree_status_t loom_llvmir_write_result_prefix(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction,
    loom_output_stream_t* stream) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_write_value_ref(
      module, instruction->result_value_id, stream));
  return loom_output_stream_write_cstring(stream, " = ");
}

static iree_status_t loom_llvmir_write_phi(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    const loom_llvmir_instruction_t* instruction,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_result_prefix(module, instruction, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "phi "));
  IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
      module, loom_llvmir_text_value_type(module, instruction->result_value_id),
      stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
  for (iree_host_size_t i = 0; i < instruction->phi.incoming_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "[ "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_value_ref(
        module, instruction->phi.incoming[i].value, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_label_ref(
        function, instruction->phi.incoming[i].predecessor, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " ]"));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_alloca(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction,
    loom_output_stream_t* stream) {
  const loom_llvmir_type_t* result_type = loom_llvmir_text_type(
      module,
      loom_llvmir_text_value_type(module, instruction->result_value_id));
  if (!result_type || result_type->kind != LOOM_LLVMIR_TYPE_POINTER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca result must be a pointer");
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_result_prefix(module, instruction, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "alloca "));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_type(module, instruction->alloca.element_type, stream));
  if (instruction->alloca.has_explicit_count) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
        module, instruction->alloca.count, stream));
  }
  if (instruction->alloca.alignment != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ", align %u", instruction->alloca.alignment));
  }
  if (result_type->address_space != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ", addrspace(%u)", result_type->address_space));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_write_instruction(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    const loom_llvmir_instruction_t* instruction,
    loom_output_stream_t* stream) {
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI:
      return loom_llvmir_write_phi(module, function, instruction, stream);
    case LOOM_LLVMIR_INST_BINOP: {
      const char* op = loom_llvmir_binop_spelling(instruction->binop.op);
      if (!op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM binop");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, op));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_binop_flags(instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, loom_llvmir_text_value_type(module, instruction->binop.lhs),
          stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_value_ref(module, instruction->binop.lhs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_value_ref(module, instruction->binop.rhs,
                                         stream);
    }
    case LOOM_LLVMIR_INST_UNOP: {
      const char* op = loom_llvmir_unop_spelling(instruction->unop.op);
      if (!op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM unary op");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, op));
      if (instruction->unop.fast_math_flags != LOOM_LLVMIR_FAST_MATH_NONE) {
        if (instruction->unop.op != LOOM_LLVMIR_UNOP_FNEG) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM unary op cannot carry fast-math flags");
        }
        IREE_RETURN_IF_ERROR(loom_llvmir_write_fast_math_flags(
            instruction->unop.fast_math_flags, stream));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      return loom_llvmir_write_typed_value_ref(module, instruction->unop.value,
                                               stream);
    }
    case LOOM_LLVMIR_INST_ICMP: {
      const char* predicate =
          loom_llvmir_icmp_predicate_spelling(instruction->icmp.predicate);
      if (!predicate) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM icmp predicate");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "icmp "));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, predicate));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, loom_llvmir_text_value_type(module, instruction->icmp.lhs),
          stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_value_ref(module, instruction->icmp.lhs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_value_ref(module, instruction->icmp.rhs, stream);
    }
    case LOOM_LLVMIR_INST_FCMP: {
      const char* predicate =
          loom_llvmir_fcmp_predicate_spelling(instruction->fcmp.predicate);
      if (!predicate) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM fcmp predicate");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "fcmp "));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, predicate));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, loom_llvmir_text_value_type(module, instruction->fcmp.lhs),
          stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_value_ref(module, instruction->fcmp.lhs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_value_ref(module, instruction->fcmp.rhs, stream);
    }
    case LOOM_LLVMIR_INST_SELECT: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "select "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->select.condition, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->select.true_value, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_typed_value_ref(
          module, instruction->select.false_value, stream);
    }
    case LOOM_LLVMIR_INST_CAST: {
      const char* op = loom_llvmir_cast_op_spelling(instruction->cast.op);
      if (!op) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM cast op");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, op));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->cast.value, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " to "));
      return loom_llvmir_write_type(
          module,
          loom_llvmir_text_value_type(module, instruction->result_value_id),
          stream);
    }
    case LOOM_LLVMIR_INST_GEP: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "getelementptr "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, instruction->gep.element_type, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->gep.base, stream));
      for (iree_host_size_t i = 0; i < instruction->gep.index_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
            module, instruction->gep.indices[i], stream));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_ALLOCA:
      return loom_llvmir_write_alloca(module, instruction, stream);
    case LOOM_LLVMIR_INST_LOAD: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "load "));
      if (instruction->load.flags & LOOM_LLVMIR_MEMORY_VOLATILE) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "volatile "));
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, instruction->load.result_type, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->load.pointer, stream));
      if (instruction->load.alignment != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, ", align %u", instruction->load.alignment));
      }
      return loom_llvmir_write_metadata_attachments(
          instruction->load.metadata_attachments,
          instruction->load.metadata_attachment_count, ", ", stream);
    }
    case LOOM_LLVMIR_INST_STORE: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "store "));
      if (instruction->store.flags & LOOM_LLVMIR_MEMORY_VOLATILE) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "volatile "));
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->store.value, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->store.pointer, stream));
      if (instruction->store.alignment != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, ", align %u", instruction->store.alignment));
      }
      return loom_llvmir_write_metadata_attachments(
          instruction->store.metadata_attachments,
          instruction->store.metadata_attachment_count, ", ", stream);
    }
    case LOOM_LLVMIR_INST_ATOMIC_RMW: {
      const char* op_spelling =
          loom_llvmir_atomic_rmw_op_spelling(instruction->atomic_rmw.op);
      if (!op_spelling) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM atomicrmw op");
      }
      const char* ordering_spelling = loom_llvmir_atomic_ordering_spelling(
          instruction->atomic_rmw.ordering);
      if (!ordering_spelling) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM atomic ordering");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "atomicrmw "));
      if (instruction->atomic_rmw.flags & LOOM_LLVMIR_MEMORY_VOLATILE) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "volatile "));
      }
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, op_spelling));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->atomic_rmw.pointer, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->atomic_rmw.value, stream));
      if (!iree_string_view_is_empty(instruction->atomic_rmw.sync_scope)) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, " syncscope("));
        IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
            stream, instruction->atomic_rmw.sync_scope));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ')'));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ordering_spelling));
      if (instruction->atomic_rmw.alignment != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, ", align %u", instruction->atomic_rmw.alignment));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_CMPXCHG: {
      const char* success_ordering_spelling =
          loom_llvmir_atomic_ordering_spelling(
              instruction->cmpxchg.success_ordering);
      const char* failure_ordering_spelling =
          loom_llvmir_atomic_ordering_spelling(
              instruction->cmpxchg.failure_ordering);
      if (!success_ordering_spelling || !failure_ordering_spelling) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM atomic ordering");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "cmpxchg "));
      if (instruction->cmpxchg.is_weak) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "weak "));
      }
      if (instruction->cmpxchg.flags & LOOM_LLVMIR_MEMORY_VOLATILE) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "volatile "));
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->cmpxchg.pointer, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->cmpxchg.expected, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->cmpxchg.replacement, stream));
      if (!iree_string_view_is_empty(instruction->cmpxchg.sync_scope)) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, " syncscope("));
        IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
            stream, instruction->cmpxchg.sync_scope));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ')'));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, success_ordering_spelling));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, failure_ordering_spelling));
      if (instruction->cmpxchg.alignment != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, ", align %u", instruction->cmpxchg.alignment));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_EXTRACT_ELEMENT: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "extractelement "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->extract_element.vector, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_typed_value_ref(
          module, instruction->extract_element.index, stream);
    }
    case LOOM_LLVMIR_INST_EXTRACT_VALUE: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "extractvalue "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->extract_value.aggregate, stream));
      return loom_output_stream_write_format(stream, ", %u",
                                             instruction->extract_value.index);
    }
    case LOOM_LLVMIR_INST_INSERT_ELEMENT: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "insertelement "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->insert_element.vector, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->insert_element.element, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_typed_value_ref(
          module, instruction->insert_element.index, stream);
    }
    case LOOM_LLVMIR_INST_SHUFFLE_VECTOR: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "shufflevector "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->shuffle_vector.lhs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->shuffle_vector.rhs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      return loom_llvmir_write_typed_value_ref(
          module, instruction->shuffle_vector.mask, stream);
    }
    case LOOM_LLVMIR_INST_CALL: {
      loom_llvmir_function_t* callee =
          loom_llvmir_text_function(module, instruction->call.callee);
      if (!callee) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM text writer saw unknown callee");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "call "));
      if (instruction->call.result_attrs.attr_count > 0) {
        IREE_RETURN_IF_ERROR(loom_llvmir_write_attr_list(
            module, instruction->call.result_attrs, stream));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_type(module, callee->return_type, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " @"));
      IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, callee->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
      for (iree_host_size_t i = 0; i < instruction->call.arg_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
            module, instruction->call.args[i], stream));
      }
      return loom_output_stream_write_char(stream, ')');
    }
    case LOOM_LLVMIR_INST_INLINE_ASM: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_write_result_prefix(module, instruction, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "call "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_type(
          module, instruction->inline_asm.result_type, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " asm"));
      if (instruction->inline_asm.flags & LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, " sideeffect"));
      }
      if (instruction->inline_asm.flags & LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, " alignstack"));
      }
      if (instruction->inline_asm.flags &
          LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, " inteldialect"));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
          stream, instruction->inline_asm.asm_template));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
          stream, instruction->inline_asm.constraints));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
      for (iree_host_size_t i = 0; i < instruction->inline_asm.arg_count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
            module, instruction->inline_asm.args[i], stream));
      }
      return loom_output_stream_write_char(stream, ')');
    }
    case LOOM_LLVMIR_INST_RET:
      if (!instruction->ret.has_value) {
        return loom_output_stream_write_cstring(stream, "ret void");
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "ret "));
      return loom_llvmir_write_typed_value_ref(module, instruction->ret.value,
                                               stream);
    case LOOM_LLVMIR_INST_BR: {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "br label "));
      return loom_llvmir_write_label_ref(function, instruction->br.target,
                                         stream);
    }
    case LOOM_LLVMIR_INST_COND_BR: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "br "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_typed_value_ref(
          module, instruction->cond_br.condition, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ", label "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_label_ref(
          function, instruction->cond_br.true_block, stream));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ", label "));
      return loom_llvmir_write_label_ref(
          function, instruction->cond_br.false_block, stream);
    }
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return loom_output_stream_write_cstring(stream, "unreachable");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_write_function_signature(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, function->kind == LOOM_LLVMIR_FUNCTION_DEFINITION ? "define "
                                                                : "declare "));
  const char* linkage = loom_llvmir_linkage_spelling(function->linkage);
  const char* calling_convention =
      loom_llvmir_calling_convention_spelling(function->calling_convention);
  if (!linkage || !calling_convention) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM function signature option");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, linkage));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, calling_convention));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_type(module, function->return_type, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " @"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, function->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '('));
  for (iree_host_size_t i = 0; i < function->parameter_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_parameter(module, &function->parameters[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ')'));
  if (function->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, " #%u", function->attr_group_id));
  }
  return loom_llvmir_write_metadata_attachments(
      function->metadata_attachments, function->metadata_attachment_count, " ",
      stream);
}

static iree_status_t loom_llvmir_write_global(
    const loom_llvmir_module_t* module, const loom_llvmir_global_t* global,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '@'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, global->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " = "));
  const char* linkage = loom_llvmir_linkage_spelling(global->linkage);
  if (!linkage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM global linkage");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, linkage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, global->is_constant ? "constant " : "global "));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_type(module, global->value_type, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ' '));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_value_ref(module, global->initializer, stream));
  if (global->alignment != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ", align %u",
                                                         global->alignment));
  }
  return loom_output_stream_write_char(stream, '\n');
}

static iree_status_t loom_llvmir_write_function(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_write_function_signature(module, function, stream));
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return loom_output_stream_write_char(stream, '\n');
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " {\n"));
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    IREE_RETURN_IF_ERROR(loom_llvmir_write_label_def(function, block, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":\n"));
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "  "));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_instruction(
          module, function, &block->instructions[j], stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
  }
  return loom_output_stream_write_cstring(stream, "}\n");
}

iree_status_t loom_llvmir_text_write_module(const loom_llvmir_module_t* module,
                                            loom_output_stream_t* stream) {
  bool wrote_header = false;
  if (!iree_string_view_is_empty(module->target_config.source_name)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "source_filename = "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
        stream, module->target_config.source_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    wrote_header = true;
  }
  if (!iree_string_view_is_empty(module->target_config.data_layout)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "target datalayout = "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
        stream, module->target_config.data_layout));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    wrote_header = true;
  }
  if (!iree_string_view_is_empty(module->target_config.target_triple)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "target triple = "));
    IREE_RETURN_IF_ERROR(loom_llvmir_write_escaped_string(
        stream, module->target_config.target_triple));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    wrote_header = true;
  }
  if (wrote_header &&
      (module->global_count > 0 || module->function_count > 0)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
  }

  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_global(module, module->globals[i], stream));
  }

  if (module->global_count > 0 && module->function_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
  }

  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_write_function(module, module->functions[i], stream));
  }

  if (module->attr_group_count > 0) {
    if (module->global_count > 0 || module->function_count > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
    for (iree_host_size_t i = 0; i < module->attr_group_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "attributes #%u = { ", (uint32_t)i));
      IREE_RETURN_IF_ERROR(loom_llvmir_write_attr_list(
          module, module->attr_groups[i].attrs, stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, " }\n"));
    }
  }

  if (module->metadata_node_count > 0) {
    if (module->global_count > 0 || module->function_count > 0 ||
        module->attr_group_count > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '\n'));
    }
    for (iree_host_size_t i = 0; i < module->metadata_node_count; ++i) {
      const loom_llvmir_metadata_node_t* node = &module->metadata_nodes[i];
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_format(stream, "!%u = !{", (uint32_t)i));
      for (iree_host_size_t j = 0; j < node->i32_value_count; ++j) {
        if (j > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "i32 %" PRIi32, node->i32_values[j]));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}\n"));
    }
  }

  return iree_ok_status();
}
