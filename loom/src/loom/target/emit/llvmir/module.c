// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/module.h"

#include <string.h>

#include "loom/target/emit/llvmir/types.h"

#define LOOM_LLVMIR_ARENA_BLOCK_SIZE 4096

static iree_status_t loom_llvmir_module_append_entry(
    iree_arena_allocator_t* arena, void** entries,
    iree_host_size_t* inout_count, iree_host_size_t* inout_capacity,
    iree_host_size_t element_size, void** out_entry) {
  if (*inout_count == *inout_capacity) {
    iree_host_size_t minimum_capacity =
        *inout_capacity == 0 ? 8 : *inout_capacity + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(arena, *inout_count,
                                               minimum_capacity, element_size,
                                               inout_capacity, entries));
  }
  uint8_t* entry_bytes = (uint8_t*)(*entries) + (*inout_count * element_size);
  memset(entry_bytes, 0, element_size);
  *inout_count += 1;
  *out_entry = entry_bytes;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_module_copy_string(
    loom_llvmir_module_t* module, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  if (source.size == 0) {
    *out_copy = iree_string_view_empty();
    return iree_ok_status();
  }
  if (source.data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty string has null storage");
  }
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "string length overflows allocation size");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, source.size + 1, (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_copy = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_copy_attr(loom_llvmir_module_t* module,
                                           const loom_llvmir_attr_t* source,
                                           loom_llvmir_attr_t* target) {
  *target = *source;
  target->key = iree_string_view_empty();
  target->string_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_string(module, source->key, &target->key));
  return loom_llvmir_module_copy_string(module, source->string_value,
                                        &target->string_value);
}

static iree_status_t loom_llvmir_module_copy_attrs(
    loom_llvmir_module_t* module, const loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_count, loom_llvmir_attr_list_t* out_list) {
  out_list->attrs = NULL;
  out_list->attr_count = 0;
  if (attr_count == 0) return iree_ok_status();
  if (attrs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty LLVM attribute list has null storage");
  }
  loom_llvmir_attr_t* copied_attrs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, attr_count,
                                                 sizeof(*copied_attrs),
                                                 (void**)&copied_attrs));
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_copy_attr(module, &attrs[i], &copied_attrs[i]));
  }
  out_list->attrs = copied_attrs;
  out_list->attr_count = attr_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_module_define_value(
    loom_llvmir_module_t* module, loom_llvmir_value_kind_t kind,
    loom_llvmir_type_id_t type_id, iree_string_view_t name,
    loom_llvmir_value_t** out_value, loom_llvmir_value_id_t* out_value_id) {
  if (type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM value references unknown type");
  }
  loom_llvmir_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->values, &module->value_count,
      &module->value_capacity, sizeof(*module->values), (void**)&value));
  value->kind = kind;
  value->type_id = type_id;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_string(module, name, &value->name));
  if (out_value_id) {
    *out_value_id = (loom_llvmir_value_id_t)(module->value_count - 1);
  }
  if (out_value) *out_value = value;
  return iree_ok_status();
}

static bool loom_llvmir_type_equal(const loom_llvmir_type_t* lhs,
                                   const loom_llvmir_type_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->bit_width != rhs->bit_width ||
      lhs->address_space != rhs->address_space ||
      lhs->element_count != rhs->element_count ||
      lhs->element_type != rhs->element_type ||
      lhs->float_kind != rhs->float_kind) {
    return false;
  }
  if (lhs->kind != LOOM_LLVMIR_TYPE_STRUCT) return true;
  for (uint32_t i = 0; i < lhs->element_count; ++i) {
    if (lhs->element_types[i] != rhs->element_types[i]) return false;
  }
  return true;
}

static bool loom_llvmir_module_is_constant_value_kind(
    loom_llvmir_value_kind_t kind) {
  return kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_NULL ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR ||
         kind == LOOM_LLVMIR_VALUE_CONSTANT_POISON;
}

static bool loom_llvmir_module_is_power_of_two_u32(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_llvmir_module_check_global_linkage(
    loom_llvmir_linkage_t linkage) {
  switch (linkage) {
    case LOOM_LLVMIR_LINKAGE_DEFAULT:
    case LOOM_LLVMIR_LINKAGE_DSO_LOCAL:
    case LOOM_LLVMIR_LINKAGE_INTERNAL:
    case LOOM_LLVMIR_LINKAGE_PRIVATE:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM global linkage");
  }
}

static iree_status_t loom_llvmir_module_get_type(
    loom_llvmir_module_t* module, const loom_llvmir_type_t* type,
    loom_llvmir_type_id_t* out_type_id) {
  for (iree_host_size_t i = 0; i < module->type_count; ++i) {
    if (loom_llvmir_type_equal(&module->types[i], type)) {
      *out_type_id = (loom_llvmir_type_id_t)i;
      return iree_ok_status();
    }
  }
  loom_llvmir_type_t* stored_type = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->types, &module->type_count,
      &module->type_capacity, sizeof(*module->types), (void**)&stored_type));
  *stored_type = *type;
  if (type->kind == LOOM_LLVMIR_TYPE_STRUCT && type->element_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&module->arena, type->element_count,
                                  sizeof(*stored_type->element_types),
                                  (void**)&stored_type->element_types));
    memcpy(stored_type->element_types, type->element_types,
           type->element_count * sizeof(*stored_type->element_types));
  }
  *out_type_id = (loom_llvmir_type_id_t)(module->type_count - 1);
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_allocate(
    const loom_llvmir_target_config_t* target_config,
    iree_allocator_t allocator, loom_llvmir_module_t** out_module) {
  *out_module = NULL;

  loom_llvmir_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->allocator = allocator;
  iree_arena_block_pool_initialize(LOOM_LLVMIR_ARENA_BLOCK_SIZE, allocator,
                                   &module->block_pool);
  iree_arena_initialize(&module->block_pool, &module->arena);

  iree_status_t status = iree_ok_status();
  if (target_config != NULL) {
    status = loom_llvmir_module_copy_string(module, target_config->source_name,
                                            &module->target_config.source_name);
    if (iree_status_is_ok(status)) {
      status =
          loom_llvmir_module_copy_string(module, target_config->target_triple,
                                         &module->target_config.target_triple);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_llvmir_module_copy_string(module, target_config->data_layout,
                                         &module->target_config.data_layout);
    }
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_module_copy_string(module, target_config->producer,
                                              &module->target_config.producer);
    }
    module->target_config.default_pointer_bitwidth =
        target_config->default_pointer_bitwidth;
    module->target_config.index_bitwidth = target_config->index_bitwidth;
    module->target_config.offset_bitwidth = target_config->offset_bitwidth;
  }

  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    loom_llvmir_module_free(module);
  }
  return status;
}

void loom_llvmir_module_free(loom_llvmir_module_t* module) {
  if (module == NULL) return;
  iree_arena_deinitialize(&module->arena);
  iree_arena_block_pool_deinitialize(&module->block_pool);
  iree_allocator_free(module->allocator, module);
}

iree_status_t loom_llvmir_module_get_void_type(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t* out_type_id) {
  loom_llvmir_type_t type = {.kind = LOOM_LLVMIR_TYPE_VOID};
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_integer_type(
    loom_llvmir_module_t* module, uint32_t bit_width,
    loom_llvmir_type_id_t* out_type_id) {
  if (bit_width == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM integer type needs non-zero bit width");
  }
  loom_llvmir_type_t type = {
      .kind = LOOM_LLVMIR_TYPE_INTEGER,
      .bit_width = bit_width,
  };
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_float_type(
    loom_llvmir_module_t* module, loom_llvmir_float_kind_t float_kind,
    loom_llvmir_type_id_t* out_type_id) {
  if (float_kind != LOOM_LLVMIR_FLOAT_F16 &&
      float_kind != LOOM_LLVMIR_FLOAT_BF16 &&
      float_kind != LOOM_LLVMIR_FLOAT_F32 &&
      float_kind != LOOM_LLVMIR_FLOAT_F64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM float kind");
  }
  loom_llvmir_type_t type = {
      .kind = LOOM_LLVMIR_TYPE_FLOAT,
      .float_kind = float_kind,
  };
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_pointer_type(
    loom_llvmir_module_t* module, uint32_t address_space,
    loom_llvmir_type_id_t* out_type_id) {
  loom_llvmir_type_t type = {
      .kind = LOOM_LLVMIR_TYPE_POINTER,
      .address_space = address_space,
  };
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_vector_type(
    loom_llvmir_module_t* module, uint32_t element_count,
    loom_llvmir_type_id_t element_type_id, loom_llvmir_type_id_t* out_type_id) {
  if (element_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM vector type needs non-zero element count");
  }
  if (element_type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM vector references unknown element type");
  }
  loom_llvmir_type_t type = {
      .kind = LOOM_LLVMIR_TYPE_VECTOR,
      .element_count = element_count,
      .element_type = element_type_id,
  };
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_struct_type(
    loom_llvmir_module_t* module, const loom_llvmir_type_id_t* element_type_ids,
    iree_host_size_t element_count, loom_llvmir_type_id_t* out_type_id) {
  if (element_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM struct type needs at least one element");
  }
  if (element_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM struct element count exceeds 32 bits");
  }
  if (element_type_ids == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM struct has null element type storage");
  }
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    if (element_type_ids[i] >= module->type_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM struct references unknown element type");
    }
    if (module->types[element_type_ids[i]].kind == LOOM_LLVMIR_TYPE_VOID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM struct element type must not be void");
    }
  }
  loom_llvmir_type_t type = {
      .kind = LOOM_LLVMIR_TYPE_STRUCT,
      .element_count = (uint32_t)element_count,
      .element_types = (loom_llvmir_type_id_t*)element_type_ids,
  };
  return loom_llvmir_module_get_type(module, &type, out_type_id);
}

iree_status_t loom_llvmir_module_get_vector_type_info(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    uint32_t* out_element_count, loom_llvmir_type_id_t* out_element_type_id) {
  *out_element_count = 0;
  *out_element_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
  if (type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM vector info references unknown type");
  }
  const loom_llvmir_type_t* type = &module->types[type_id];
  if (type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM vector info requires a vector type");
  }
  *out_element_count = type->element_count;
  *out_element_type_id = type->element_type;
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_integer_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id, uint64_t value,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_value_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_CONSTANT_INTEGER, type_id,
      iree_string_view_empty(), &constant, out_value_id));
  constant->integer_value = value;
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_integer_vector_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t vector_type_id,
    const uint64_t* values, iree_host_size_t value_count,
    loom_llvmir_value_id_t* out_value_id) {
  if (vector_type_id >= module->type_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM integer vector constant references unknown type");
  }
  const loom_llvmir_type_t* vector_type = &module->types[vector_type_id];
  if (vector_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      vector_type->element_type >= module->type_count ||
      module->types[vector_type->element_type].kind !=
          LOOM_LLVMIR_TYPE_INTEGER) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM integer vector constant type must be an integer vector");
  }
  if (value_count != vector_type->element_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM integer vector constant element count does not match type");
  }
  if (value_count > 0 && values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM integer vector constant has null value storage");
  }
  loom_llvmir_value_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_CONSTANT_INTEGER_VECTOR, vector_type_id,
      iree_string_view_empty(), &constant, out_value_id));
  if (value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, value_count, sizeof(*constant->integer_vector.values),
        (void**)&constant->integer_vector.values));
    memcpy(constant->integer_vector.values, values,
           value_count * sizeof(*constant->integer_vector.values));
  }
  constant->integer_vector.value_count = value_count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_float_bits_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id, uint64_t bits,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_value_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_CONSTANT_FLOAT_BITS, type_id,
      iree_string_view_empty(), &constant, out_value_id));
  constant->float_bits = bits;
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_null_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    loom_llvmir_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_CONSTANT_NULL, type_id,
      iree_string_view_empty(), NULL, out_value_id));
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_poison_constant(
    loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id,
    loom_llvmir_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_CONSTANT_POISON, type_id,
      iree_string_view_empty(), NULL, out_value_id));
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_count, loom_llvmir_attr_group_id_t* out_group_id) {
  if (attr_count > 0 && attrs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty LLVM attr group has null storage");
  }
  loom_llvmir_attr_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->attr_groups, &module->attr_group_count,
      &module->attr_group_capacity, sizeof(*module->attr_groups),
      (void**)&group));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_attrs(module, attrs, attr_count, &group->attrs));
  *out_group_id = (loom_llvmir_attr_group_id_t)(module->attr_group_count - 1);
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_metadata_i32_tuple(
    loom_llvmir_module_t* module,
    const loom_llvmir_metadata_i32_tuple_t* metadata,
    loom_llvmir_metadata_id_t* out_metadata_id) {
  if (metadata->value_count > 0 && metadata->values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM metadata tuple has null value storage");
  }
  loom_llvmir_metadata_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->metadata_nodes,
      &module->metadata_node_count, &module->metadata_node_capacity,
      sizeof(*module->metadata_nodes), (void**)&node));
  if (metadata->value_count > 0) {
    loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_module_get_integer_type(module, 32, &i32_type));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, metadata->value_count, sizeof(*node->i32_values),
        (void**)&node->i32_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, metadata->value_count, sizeof(*node->i32_value_ids),
        (void**)&node->i32_value_ids));
    memcpy(node->i32_values, metadata->values,
           metadata->value_count * sizeof(*node->i32_values));
    for (iree_host_size_t i = 0; i < metadata->value_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
          module, i32_type, (uint32_t)metadata->values[i],
          &node->i32_value_ids[i]));
    }
  }
  node->i32_value_count = metadata->value_count;
  *out_metadata_id =
      (loom_llvmir_metadata_id_t)(module->metadata_node_count - 1);
  return iree_ok_status();
}

iree_status_t loom_llvmir_module_add_global(
    loom_llvmir_module_t* module, const loom_llvmir_global_desc_t* desc,
    loom_llvmir_global_t** out_global) {
  *out_global = NULL;
  if (iree_string_view_is_empty(desc->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global name must not be empty");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_module_check_global_linkage(desc->linkage));
  if (desc->value_type >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global references unknown value type");
  }
  if (module->types[desc->value_type].kind == LOOM_LLVMIR_TYPE_VOID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global value type must not be void");
  }
  if (desc->initializer >= module->value_count ||
      !loom_llvmir_module_is_constant_value_kind(
          module->values[desc->initializer].kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global initializer must be a constant");
  }
  if (module->values[desc->initializer].type_id != desc->value_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global initializer type mismatch");
  }
  if (desc->alignment != 0 &&
      !loom_llvmir_module_is_power_of_two_u32(desc->alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM global alignment is not a power of two");
  }

  loom_llvmir_type_id_t pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      module, desc->address_space, &pointer_type));

  loom_llvmir_global_t* global = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, sizeof(*global), (void**)&global));
  memset(global, 0, sizeof(*global));
  global->module = module;
  global->id = (loom_llvmir_global_id_t)module->global_count;
  global->linkage = desc->linkage;
  global->value_type = desc->value_type;
  global->address_space = desc->address_space;
  global->is_constant = desc->is_constant;
  global->initializer = desc->initializer;
  global->alignment = desc->alignment;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_string(module, desc->name, &global->name));

  loom_llvmir_global_t** global_slot = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->globals, &module->global_count,
      &module->global_capacity, sizeof(*module->globals),
      (void**)&global_slot));
  *global_slot = global;

  loom_llvmir_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      module, LOOM_LLVMIR_VALUE_GLOBAL, pointer_type, desc->name, &value,
      &global->value_id));
  value->global.global_id = global->id;
  *out_global = global;
  return iree_ok_status();
}

loom_llvmir_global_id_t loom_llvmir_global_id(
    const loom_llvmir_global_t* global) {
  return global ? global->id : LOOM_LLVMIR_GLOBAL_ID_INVALID;
}

loom_llvmir_value_id_t loom_llvmir_global_value_id(
    const loom_llvmir_global_t* global) {
  return global ? global->value_id : LOOM_LLVMIR_VALUE_ID_INVALID;
}

iree_status_t loom_llvmir_module_add_function(
    loom_llvmir_module_t* module, const loom_llvmir_function_desc_t* desc,
    loom_llvmir_function_t** out_function) {
  *out_function = NULL;
  if (iree_string_view_is_empty(desc->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function name must not be empty");
  }
  if (desc->kind != LOOM_LLVMIR_FUNCTION_DECLARATION &&
      desc->kind != LOOM_LLVMIR_FUNCTION_DEFINITION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM function kind");
  }
  if (desc->return_type >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function references unknown return type");
  }
  if (desc->attr_group_id != LOOM_LLVMIR_ATTR_GROUP_ID_INVALID &&
      desc->attr_group_id >= module->attr_group_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function references unknown attr group");
  }

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&module->arena, sizeof(*function),
                                           (void**)&function));
  memset(function, 0, sizeof(*function));
  function->module = module;
  function->id = (loom_llvmir_function_id_t)module->function_count;
  function->kind = desc->kind;
  function->return_type = desc->return_type;
  function->linkage = desc->linkage;
  function->calling_convention = desc->calling_convention;
  function->attr_group_id = desc->attr_group_id;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_string(module, desc->name, &function->name));

  loom_llvmir_function_t** function_slot = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &module->arena, (void**)&module->functions, &module->function_count,
      &module->function_capacity, sizeof(*module->functions),
      (void**)&function_slot));
  *function_slot = function;
  *out_function = function;
  return iree_ok_status();
}

loom_llvmir_function_t* loom_llvmir_module_find_function(
    loom_llvmir_module_t* module, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    loom_llvmir_function_t* function = module->functions[i];
    if (iree_string_view_equal(function->name, name)) return function;
  }
  return NULL;
}

loom_llvmir_function_id_t loom_llvmir_function_id(
    const loom_llvmir_function_t* function) {
  return function ? function->id : LOOM_LLVMIR_FUNCTION_ID_INVALID;
}

loom_llvmir_module_t* loom_llvmir_function_module(
    const loom_llvmir_function_t* function) {
  return function ? function->module : NULL;
}

iree_status_t loom_llvmir_function_add_parameter(
    loom_llvmir_function_t* function, const loom_llvmir_parameter_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  if (function->kind != LOOM_LLVMIR_FUNCTION_DEFINITION &&
      function->kind != LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM function kind");
  }
  if (desc->type_id >= function->module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM parameter references unknown type");
  }
  if (desc->attr_count > 0 && desc->attrs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty LLVM parameter attrs have null storage");
  }
  loom_llvmir_parameter_t* parameter = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &function->module->arena, (void**)&function->parameters,
      &function->parameter_count, &function->parameter_capacity,
      sizeof(*function->parameters), (void**)&parameter));
  parameter->type_id = desc->type_id;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_copy_string(
      function->module, desc->name, &parameter->name));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_copy_attrs(
      function->module, desc->attrs, desc->attr_count, &parameter->attrs));
  loom_llvmir_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_define_value(
      function->module, LOOM_LLVMIR_VALUE_PARAMETER, desc->type_id, desc->name,
      &value, out_value_id));
  parameter->value_id = *out_value_id;
  value->parameter.function_id = function->id;
  value->parameter.parameter_ordinal =
      (uint32_t)(function->parameter_count - 1);
  return iree_ok_status();
}

iree_status_t loom_llvmir_function_add_metadata_attachment(
    loom_llvmir_function_t* function,
    const loom_llvmir_metadata_attachment_t* attachment) {
  if (iree_string_view_is_empty(attachment->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM metadata attachment name must not be empty");
  }
  if (attachment->metadata_id >= function->module->metadata_node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function references unknown metadata node");
  }
  loom_llvmir_metadata_attachment_storage_t* stored_attachment = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &function->module->arena, (void**)&function->metadata_attachments,
      &function->metadata_attachment_count,
      &function->metadata_attachment_capacity,
      sizeof(*function->metadata_attachments), (void**)&stored_attachment));
  stored_attachment->metadata_id = attachment->metadata_id;
  return loom_llvmir_module_copy_string(function->module, attachment->name,
                                        &stored_attachment->name);
}

iree_status_t loom_llvmir_function_add_block(loom_llvmir_function_t* function,
                                             iree_string_view_t name,
                                             loom_llvmir_block_t** out_block) {
  *out_block = NULL;
  if (function->kind != LOOM_LLVMIR_FUNCTION_DEFINITION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM declarations cannot contain blocks");
  }
  loom_llvmir_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&function->module->arena,
                                           sizeof(*block), (void**)&block));
  memset(block, 0, sizeof(*block));
  block->function = function;
  block->id = (loom_llvmir_block_id_t)function->block_count;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_copy_string(function->module, name, &block->name));

  loom_llvmir_block_t** block_slot = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_append_entry(
      &function->module->arena, (void**)&function->blocks,
      &function->block_count, &function->block_capacity,
      sizeof(*function->blocks), (void**)&block_slot));
  *block_slot = block;
  *out_block = block;
  return iree_ok_status();
}

loom_llvmir_block_id_t loom_llvmir_block_id(const loom_llvmir_block_t* block) {
  return block ? block->id : LOOM_LLVMIR_BLOCK_ID_INVALID;
}
