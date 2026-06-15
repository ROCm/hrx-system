// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_types.h"

#include "loom/target/emit/spirv/binary_format.h"

typedef enum loom_spirv_type_key_kind_e {
  LOOM_SPIRV_TYPE_KEY_VOID = 1,
  LOOM_SPIRV_TYPE_KEY_INT = 2,
  LOOM_SPIRV_TYPE_KEY_FLOAT = 3,
  LOOM_SPIRV_TYPE_KEY_POINTER = 4,
  LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT = 5,
  LOOM_SPIRV_TYPE_KEY_FUNCTION = 6,
  LOOM_SPIRV_TYPE_KEY_RUNTIME_ARRAY = 7,
  LOOM_SPIRV_TYPE_KEY_STRUCT = 8,
  LOOM_SPIRV_TYPE_KEY_ARRAY = 9,
  LOOM_SPIRV_TYPE_KEY_BOOL = 10,
  LOOM_SPIRV_TYPE_KEY_VECTOR = 11,
  LOOM_SPIRV_TYPE_KEY_COOPERATIVE_MATRIX = 12,
} loom_spirv_type_key_kind_t;

enum {
  LOOM_SPIRV_TYPE_CACHE_INITIAL_CAPACITY = 16,
  LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT = 8,
  LOOM_SPIRV_INTEGER_CONSTANT_CACHE_INITIAL_CAPACITY = 16,
};

typedef struct loom_spirv_type_key_t {
  // SPIR-V type declaration family.
  loom_spirv_type_key_kind_t kind;
  // ArrayStride decoration for pointer/array types, or zero when undecorated.
  uint32_t array_stride;
  // Number of entries in |operands|.
  uint8_t operand_count;
  // Kind-specific SPIR-V type operands following the result ID.
  uint32_t operands[LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT];
} loom_spirv_type_key_t;

struct loom_spirv_type_cache_entry_t {
  // Numeric type key.
  loom_spirv_type_key_t key;
  // SPIR-V result ID assigned to |key|.
  uint32_t type_id;
};

typedef struct loom_spirv_integer_constant_key_t {
  // SPIR-V integer type ID of the constant.
  uint32_t type_id;
  // Number of literal words in |words|.
  uint8_t word_count;
  // Little-endian integer literal words following the result ID.
  uint32_t words[2];
} loom_spirv_integer_constant_key_t;

struct loom_spirv_integer_constant_cache_entry_t {
  // Numeric integer constant key.
  loom_spirv_integer_constant_key_t key;
  // SPIR-V result ID assigned to |key|.
  uint32_t id;
};

void loom_spirv_type_context_initialize(
    loom_spirv_module_builder_t* builder, iree_arena_allocator_t* scratch_arena,
    loom_spirv_type_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = (loom_spirv_type_context_t){
      .scratch_arena = scratch_arena,
      .builder = builder,
  };
}

static loom_spirv_binary_writer_t* loom_spirv_type_context_section(
    loom_spirv_type_context_t* context, loom_spirv_module_section_t section) {
  return loom_spirv_module_builder_section(context->builder, section);
}

static uint32_t loom_spirv_type_context_allocate_id(
    loom_spirv_type_context_t* context) {
  return loom_spirv_module_builder_allocate_id(context->builder);
}

static bool loom_spirv_type_key_equal(const loom_spirv_type_key_t* lhs,
                                      const loom_spirv_type_key_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->array_stride != rhs->array_stride ||
      lhs->operand_count != rhs->operand_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->operand_count; ++i) {
    if (lhs->operands[i] != rhs->operands[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_spirv_emit_type_cache_reserve(
    loom_spirv_type_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->type_cache_capacity) {
    return iree_ok_status();
  }
  if (context->type_cache_capacity == 0) {
    minimum_capacity =
        iree_max(minimum_capacity, LOOM_SPIRV_TYPE_CACHE_INITIAL_CAPACITY);
  }
  return iree_arena_grow_array(
      context->scratch_arena, context->type_cache_count, minimum_capacity,
      sizeof(*context->type_cache_entries), &context->type_cache_capacity,
      (void**)&context->type_cache_entries);
}

static iree_status_t loom_spirv_emit_type_declaration(
    loom_spirv_type_context_t* context, const loom_spirv_type_key_t* key,
    uint32_t type_id) {
  switch (key->kind) {
    case LOOM_SPIRV_TYPE_KEY_VOID: {
      const uint32_t operands[] = {type_id};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_VOID, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_BOOL: {
      const uint32_t operands[] = {type_id};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_BOOL, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_INT: {
      const uint32_t operands[] = {type_id, key->operands[0], key->operands[1]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_INT, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_FLOAT: {
      uint32_t operands[] = {type_id, key->operands[0], key->operands[1]};
      const uint8_t operand_count =
          key->operand_count == 2 ? IREE_ARRAYSIZE(operands) : 2;
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_FLOAT, operands, operand_count);
    }
    case LOOM_SPIRV_TYPE_KEY_VECTOR: {
      const uint32_t operands[] = {type_id, key->operands[0], key->operands[1]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_VECTOR, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_COOPERATIVE_MATRIX: {
      const uint32_t operands[] = {
          type_id,          key->operands[0], key->operands[1],
          key->operands[2], key->operands[3], key->operands[4],
      };
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_COOPERATIVE_MATRIX_KHR, operands,
          IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_POINTER: {
      const uint32_t operands[] = {
          type_id,
          key->operands[0],
          key->operands[1],
      };
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_POINTER, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_RUNTIME_ARRAY: {
      const uint32_t operands[] = {type_id, key->operands[0]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_RUNTIME_ARRAY, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_ARRAY: {
      const uint32_t operands[] = {type_id, key->operands[0], key->operands[1]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_ARRAY, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT: {
      const uint32_t operands[] = {type_id, key->operands[0]};
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_STRUCT, operands, IREE_ARRAYSIZE(operands));
    }
    case LOOM_SPIRV_TYPE_KEY_STRUCT: {
      uint32_t operands[1 + LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT] = {0};
      operands[0] = type_id;
      for (uint8_t i = 0; i < key->operand_count; ++i) {
        operands[1 + i] = key->operands[i];
      }
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_STRUCT, operands, 1 + key->operand_count);
    }
    case LOOM_SPIRV_TYPE_KEY_FUNCTION: {
      uint32_t operands[1 + LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT] = {0};
      operands[0] = type_id;
      for (uint8_t i = 0; i < key->operand_count; ++i) {
        operands[1 + i] = key->operands[i];
      }
      return loom_spirv_binary_write_instruction(
          loom_spirv_type_context_section(
              context, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
          LOOM_SPIRV_OP_TYPE_FUNCTION, operands, 1 + key->operand_count);
    }
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unknown SPIR-V type key kind %u",
                          (uint32_t)key->kind);
}

static iree_status_t loom_spirv_emit_type_id(loom_spirv_type_context_t* context,
                                             const loom_spirv_type_key_t* key,
                                             uint32_t* out_type_id,
                                             bool* out_emitted) {
  for (iree_host_size_t i = 0; i < context->type_cache_count; ++i) {
    const loom_spirv_type_cache_entry_t* entry =
        &context->type_cache_entries[i];
    if (loom_spirv_type_key_equal(key, &entry->key)) {
      *out_type_id = entry->type_id;
      if (out_emitted != NULL) {
        *out_emitted = false;
      }
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_cache_reserve(
      context, context->type_cache_count + 1));

  const uint32_t type_id = loom_spirv_type_context_allocate_id(context);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_declaration(context, key, type_id));
  context->type_cache_entries[context->type_cache_count++] =
      (loom_spirv_type_cache_entry_t){
          .key = *key,
          .type_id = type_id,
      };
  *out_type_id = type_id;
  if (out_emitted != NULL) {
    *out_emitted = true;
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_type_void(loom_spirv_type_context_t* context,
                                        uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_VOID,
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_bool(loom_spirv_type_context_t* context,
                                        uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_BOOL,
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_int(loom_spirv_type_context_t* context,
                                       uint32_t bit_width, uint32_t signedness,
                                       uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_INT,
      .operand_count = 2,
      .operands = {bit_width, signedness},
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_float(loom_spirv_type_context_t* context,
                                         uint32_t bit_width,
                                         loom_spirv_fp_encoding_t fp_encoding,
                                         uint32_t* out_type_id) {
  const bool has_encoding = fp_encoding != LOOM_SPIRV_FP_ENCODING_MAX;
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_FLOAT,
      .operand_count = has_encoding ? 2 : 1,
      .operands = {bit_width, fp_encoding},
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_scalar(loom_spirv_type_context_t* context,
                                          loom_spirv_scalar_type_t scalar_type,
                                          uint32_t* out_type_id) {
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(scalar_type);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unknown SPIR-V scalar type %u",
                            (uint32_t)scalar_type);
  }
  switch (descriptor->kind) {
    case LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT:
      return loom_spirv_emit_type_float(context, descriptor->bit_width,
                                        descriptor->fp_encoding, out_type_id);
    case LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT:
      return loom_spirv_emit_type_int(context, descriptor->bit_width,
                                      /*signedness=*/1, out_type_id);
    case LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT:
      return loom_spirv_emit_type_int(context, descriptor->bit_width,
                                      /*signedness=*/0, out_type_id);
    case LOOM_SPIRV_SCALAR_TYPE_KIND_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unknown SPIR-V scalar type kind %u",
                          (uint32_t)descriptor->kind);
}

iree_status_t loom_spirv_emit_type_vector(loom_spirv_type_context_t* context,
                                          uint32_t element_type_id,
                                          uint32_t component_count,
                                          uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_VECTOR,
      .operand_count = 2,
      .operands = {element_type_id, component_count},
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_cooperative_matrix(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t value_type,
    uint32_t* out_type_id) {
  uint32_t component_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_scalar(
      context, value_type.scalar_type, &component_type_id));
  uint32_t scope_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_u32_constant(context, value_type.scope, &scope_id));
  uint32_t rows_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_u32_constant(context, value_type.rows, &rows_id));
  uint32_t columns_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_u32_constant(context, value_type.columns, &columns_id));
  uint32_t use_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      context, value_type.cooperative_matrix_use, &use_id));
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_COOPERATIVE_MATRIX,
      .operand_count = 5,
      .operands = {component_type_id, scope_id, rows_id, columns_id, use_id},
  };
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_i32(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id) {
  return loom_spirv_emit_type_int(context, 32, 1, out_type_id);
}

iree_status_t loom_spirv_emit_type_u32(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id) {
  return loom_spirv_emit_type_int(context, 32, 0, out_type_id);
}

iree_status_t loom_spirv_emit_type_u64(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id) {
  return loom_spirv_emit_type_int(context, 64, 0, out_type_id);
}

iree_status_t loom_spirv_emit_type_runtime_array(
    loom_spirv_type_context_t* context, uint32_t element_type_id,
    uint32_t array_stride, uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_RUNTIME_ARRAY,
      .array_stride = array_stride,
      .operand_count = 1,
      .operands = {element_type_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(context, &key, out_type_id, &emitted));
  if (emitted && array_stride != 0) {
    const uint32_t decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        array_stride,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_type_array(loom_spirv_type_context_t* context,
                                         uint32_t element_type_id,
                                         uint32_t length_id,
                                         uint32_t array_stride,
                                         uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_ARRAY,
      .array_stride = array_stride,
      .operand_count = 2,
      .operands = {element_type_id, length_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(context, &key, out_type_id, &emitted));
  if (emitted && array_stride != 0) {
    const uint32_t decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        array_stride,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_type_struct(loom_spirv_type_context_t* context,
                                          const uint32_t* member_type_ids,
                                          uint8_t member_count,
                                          uint32_t* out_type_id,
                                          bool* out_emitted) {
  if (member_count > LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V struct type exceeds the emitter type-key "
                            "operand capacity");
  }
  loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_STRUCT,
      .operand_count = member_count,
  };
  for (uint8_t i = 0; i < member_count; ++i) {
    key.operands[i] = member_type_ids[i];
  }
  return loom_spirv_emit_type_id(context, &key, out_type_id, out_emitted);
}

iree_status_t loom_spirv_emit_decorate_block(loom_spirv_type_context_t* context,
                                             uint32_t type_id) {
  const uint32_t decoration_operands[] = {
      type_id,
      LOOM_SPIRV_DECORATION_BLOCK,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_type_context_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, decoration_operands,
      IREE_ARRAYSIZE(decoration_operands));
}

iree_status_t loom_spirv_emit_decorate_member_offset(
    loom_spirv_type_context_t* context, uint32_t type_id,
    uint32_t member_ordinal, uint32_t byte_offset) {
  const uint32_t decoration_operands[] = {
      type_id,
      member_ordinal,
      LOOM_SPIRV_DECORATION_OFFSET,
      byte_offset,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_type_context_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_MEMBER_DECORATE, decoration_operands,
      IREE_ARRAYSIZE(decoration_operands));
}

iree_status_t loom_spirv_emit_type_function(loom_spirv_type_context_t* context,
                                            uint32_t result_type_id,
                                            const uint32_t* parameter_type_ids,
                                            uint8_t parameter_count,
                                            uint32_t* out_type_id) {
  if ((iree_host_size_t)parameter_count + 1 >
      LOOM_SPIRV_TYPE_KEY_MAX_OPERAND_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V function type exceeds the emitter "
                            "type-key operand capacity");
  }
  loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_FUNCTION,
      .operand_count = (uint8_t)(parameter_count + 1),
      .operands = {result_type_id},
  };
  for (uint8_t i = 0; i < parameter_count; ++i) {
    key.operands[1 + i] = parameter_type_ids[i];
  }
  return loom_spirv_emit_type_id(context, &key, out_type_id,
                                 /*out_emitted=*/NULL);
}

iree_status_t loom_spirv_emit_type_pointer(loom_spirv_type_context_t* context,
                                           uint32_t storage_class,
                                           uint32_t pointee_type_id,
                                           uint32_t pointer_array_stride,
                                           uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_POINTER,
      .array_stride = pointer_array_stride,
      .operand_count = 2,
      .operands = {storage_class, pointee_type_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(context, &key, out_type_id, &emitted));
  if (emitted && pointer_array_stride != 0) {
    const uint32_t decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_ARRAY_STRIDE,
        pointer_array_stride,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, decoration_operands,
        IREE_ARRAYSIZE(decoration_operands)));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_type_descriptor_struct(
    loom_spirv_type_context_t* context, uint32_t field_type_id,
    uint32_t* out_type_id) {
  const loom_spirv_type_key_t key = {
      .kind = LOOM_SPIRV_TYPE_KEY_DESCRIPTOR_STRUCT,
      .operand_count = 1,
      .operands = {field_type_id},
  };
  bool emitted = false;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_id(context, &key, out_type_id, &emitted));
  if (emitted) {
    const uint32_t member_decoration_operands[] = {
        *out_type_id,
        0,
        LOOM_SPIRV_DECORATION_OFFSET,
        0,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_MEMBER_DECORATE, member_decoration_operands,
        IREE_ARRAYSIZE(member_decoration_operands)));
    const uint32_t block_decoration_operands[] = {
        *out_type_id,
        LOOM_SPIRV_DECORATION_BLOCK,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
        LOOM_SPIRV_OP_DECORATE, block_decoration_operands,
        IREE_ARRAYSIZE(block_decoration_operands)));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
    loom_spirv_type_context_t* context, uint32_t field_type_id,
    uint32_t* out_type_id) {
  uint32_t struct_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_descriptor_struct(
      context, field_type_id, &struct_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, struct_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
    loom_spirv_type_context_t* context, uint32_t field_type_id,
    uint32_t* out_type_id) {
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_STORAGE_BUFFER, field_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t loom_spirv_emit_type_bda_root_struct(
    loom_spirv_type_context_t* context, uint16_t bda_constant_word_count,
    uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(context, &u64_type_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u32(context, &u32_type_id));
  uint32_t member_type_ids[7] = {
      u64_type_id, u64_type_id, u32_type_id,
      u32_type_id, u32_type_id, u32_type_id,
  };
  uint8_t member_count = 6;
  if (bda_constant_word_count != 0) {
    const uint32_t constant_word_count_id =
        loom_spirv_type_context_allocate_id(context);
    const uint32_t constant_operands[] = {
        u32_type_id,
        constant_word_count_id,
        bda_constant_word_count,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_type_context_section(context,
                                        LOOM_SPIRV_MODULE_SECTION_DECLARATION),
        LOOM_SPIRV_OP_CONSTANT, constant_operands,
        IREE_ARRAYSIZE(constant_operands)));
    IREE_RETURN_IF_ERROR(loom_spirv_emit_type_array(
        context, u32_type_id, constant_word_count_id, /*array_stride=*/4,
        &member_type_ids[member_count++]));
  }
  bool emitted = false;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_struct(
      context, member_type_ids, member_count, out_type_id, &emitted));
  if (emitted) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_decorate_block(context, *out_type_id));
    const uint32_t member_offsets[] = {0, 8, 16, 20, 24, 28};
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(member_offsets); ++i) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_decorate_member_offset(
          context, *out_type_id, i, member_offsets[i]));
    }
    if (bda_constant_word_count != 0) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_decorate_member_offset(
          context, *out_type_id, LOOM_SPIRV_BDA_ROOT_CONSTANT_MEMBER_INDEX,
          LOOM_SPIRV_BDA_ROOT_CONSTANT_BYTE_OFFSET));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_type_ptr_push_constant_bda_root(
    loom_spirv_type_context_t* context, uint16_t bda_constant_word_count,
    uint32_t* out_type_id) {
  uint32_t root_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_bda_root_struct(
      context, bda_constant_word_count, &root_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PUSH_CONSTANT, root_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_push_constant_u32(
    loom_spirv_type_context_t* context, uint32_t* out_type_id) {
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u32(context, &u32_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PUSH_CONSTANT, u32_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_push_constant_u64(
    loom_spirv_type_context_t* context, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(context, &u64_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PUSH_CONSTANT, u64_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static iree_status_t loom_spirv_emit_type_bda_address_table_struct(
    loom_spirv_type_context_t* context, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(context, &u64_type_id));
  uint32_t runtime_array_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_runtime_array(
      context, u64_type_id, /*array_stride=*/8, &runtime_array_type_id));
  const uint32_t member_type_ids[] = {runtime_array_type_id};
  bool emitted = false;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_struct(
      context, member_type_ids, IREE_ARRAYSIZE(member_type_ids), out_type_id,
      &emitted));
  if (emitted) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_decorate_block(context, *out_type_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_decorate_member_offset(context, *out_type_id, 0, 0));
  }
  return iree_ok_status();
}

iree_status_t
loom_spirv_emit_type_ptr_physical_storage_buffer_bda_address_table(
    loom_spirv_type_context_t* context, uint32_t* out_type_id) {
  uint32_t table_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bda_address_table_struct(context, &table_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, table_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_u64(
    loom_spirv_type_context_t* context, uint32_t* out_type_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(context, &u64_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, u64_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_scalar(
    loom_spirv_type_context_t* context, loom_spirv_scalar_type_t scalar_type,
    uint32_t* out_type_id) {
  uint32_t scalar_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_scalar(context, scalar_type, &scalar_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER, scalar_type_id,
      /*pointer_array_stride=*/1, out_type_id);
}

iree_status_t loom_spirv_emit_type_ptr_workgroup_scalar(
    loom_spirv_type_context_t* context, loom_spirv_scalar_type_t scalar_type,
    uint32_t* out_type_id) {
  uint32_t scalar_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_scalar(context, scalar_type, &scalar_type_id));
  return loom_spirv_emit_type_pointer(
      context, LOOM_SPIRV_STORAGE_CLASS_WORKGROUP, scalar_type_id,
      /*pointer_array_stride=*/0, out_type_id);
}

static bool loom_spirv_integer_constant_key_equal(
    const loom_spirv_integer_constant_key_t* lhs,
    const loom_spirv_integer_constant_key_t* rhs) {
  if (lhs->type_id != rhs->type_id || lhs->word_count != rhs->word_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->word_count; ++i) {
    if (lhs->words[i] != rhs->words[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_spirv_emit_integer_constant_cache_reserve(
    loom_spirv_type_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->integer_constant_cache_capacity) {
    return iree_ok_status();
  }
  if (context->integer_constant_cache_capacity == 0) {
    minimum_capacity = iree_max(
        minimum_capacity, LOOM_SPIRV_INTEGER_CONSTANT_CACHE_INITIAL_CAPACITY);
  }
  return iree_arena_grow_array(
      context->scratch_arena, context->integer_constant_cache_count,
      minimum_capacity, sizeof(*context->integer_constant_cache_entries),
      &context->integer_constant_cache_capacity,
      (void**)&context->integer_constant_cache_entries);
}

static iree_status_t loom_spirv_emit_cached_integer_constant(
    loom_spirv_type_context_t* context,
    const loom_spirv_integer_constant_key_t* key, uint32_t* out_constant_id) {
  for (iree_host_size_t i = 0; i < context->integer_constant_cache_count; ++i) {
    const loom_spirv_integer_constant_cache_entry_t* entry =
        &context->integer_constant_cache_entries[i];
    if (loom_spirv_integer_constant_key_equal(key, &entry->key)) {
      *out_constant_id = entry->id;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_emit_integer_constant_cache_reserve(
      context, context->integer_constant_cache_count + 1));

  const uint32_t constant_id = loom_spirv_type_context_allocate_id(context);
  uint32_t operands[2 + IREE_ARRAYSIZE(key->words)] = {0};
  operands[0] = key->type_id;
  operands[1] = constant_id;
  for (uint8_t i = 0; i < key->word_count; ++i) {
    operands[2 + i] = key->words[i];
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_type_context_section(context,
                                      LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_CONSTANT, operands, 2 + key->word_count));
  context->integer_constant_cache_entries
      [context->integer_constant_cache_count++] =
      (loom_spirv_integer_constant_cache_entry_t){
          .key = *key,
          .id = constant_id,
      };
  *out_constant_id = constant_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_i32_constant(loom_spirv_type_context_t* context,
                                           int32_t value,
                                           uint32_t* out_constant_id) {
  uint32_t i32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_i32(context, &i32_type_id));
  const loom_spirv_integer_constant_key_t key = {
      .type_id = i32_type_id,
      .word_count = 1,
      .words = {(uint32_t)value},
  };
  return loom_spirv_emit_cached_integer_constant(context, &key,
                                                 out_constant_id);
}

iree_status_t loom_spirv_emit_u32_constant(loom_spirv_type_context_t* context,
                                           uint32_t value,
                                           uint32_t* out_constant_id) {
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u32(context, &u32_type_id));
  const loom_spirv_integer_constant_key_t key = {
      .type_id = u32_type_id,
      .word_count = 1,
      .words = {value},
  };
  return loom_spirv_emit_cached_integer_constant(context, &key,
                                                 out_constant_id);
}

iree_status_t loom_spirv_emit_u64_constant(loom_spirv_type_context_t* context,
                                           uint64_t value,
                                           uint32_t* out_constant_id) {
  uint32_t u64_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u64(context, &u64_type_id));
  const loom_spirv_integer_constant_key_t key = {
      .type_id = u64_type_id,
      .word_count = 2,
      .words = {(uint32_t)value, (uint32_t)(value >> 32)},
  };
  return loom_spirv_emit_cached_integer_constant(context, &key,
                                                 out_constant_id);
}

iree_status_t loom_spirv_emit_type_id_for_value_type(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t type,
    uint32_t* out_type_id) {
  switch (type.value_class) {
    case LOOM_SPIRV_VALUE_CLASS_SCALAR:
      return loom_spirv_emit_type_scalar(context, type.scalar_type,
                                         out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_BOOL:
      return loom_spirv_emit_type_bool(context, out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_OFFSET64:
    case LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS:
      return loom_spirv_emit_type_u64(context, out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER:
      return loom_spirv_emit_type_ptr_physical_storage_buffer_scalar(
          context, type.scalar_type, out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP:
      return loom_spirv_emit_type_ptr_workgroup_scalar(
          context, type.scalar_type, out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX:
      return loom_spirv_emit_type_cooperative_matrix(context, type,
                                                     out_type_id);
    case LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY:
      break;
    case LOOM_SPIRV_VALUE_CLASS_UNKNOWN:
      break;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "SPIR-V packet row does not select a result type");
}
