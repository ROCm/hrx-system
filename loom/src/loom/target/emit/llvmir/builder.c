// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/builder.h"

#include <string.h>

#include "loom/target/emit/llvmir/types.h"

static iree_status_t loom_llvmir_builder_append_entry(
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

static iree_status_t loom_llvmir_builder_copy_string(
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

static iree_status_t loom_llvmir_builder_copy_attr(
    loom_llvmir_module_t* module, const loom_llvmir_attr_t* source,
    loom_llvmir_attr_t* target) {
  *target = *source;
  target->key = iree_string_view_empty();
  target->string_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_llvmir_builder_copy_string(module, source->key, &target->key));
  return loom_llvmir_builder_copy_string(module, source->string_value,
                                         &target->string_value);
}

static iree_status_t loom_llvmir_builder_copy_attrs(
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
        loom_llvmir_builder_copy_attr(module, &attrs[i], &copied_attrs[i]));
  }
  out_list->attrs = copied_attrs;
  out_list->attr_count = attr_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_builder_copy_metadata_attachments(
    loom_llvmir_module_t* module,
    const loom_llvmir_metadata_attachment_t* attachments,
    iree_host_size_t attachment_count,
    loom_llvmir_metadata_attachment_storage_t** out_attachments) {
  *out_attachments = NULL;
  if (attachment_count == 0) return iree_ok_status();
  if (attachments == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty LLVM metadata attachment list has null storage");
  }
  loom_llvmir_metadata_attachment_storage_t* copied_attachments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, attachment_count, sizeof(*copied_attachments),
      (void**)&copied_attachments));
  for (iree_host_size_t i = 0; i < attachment_count; ++i) {
    if (iree_string_view_is_empty(attachments[i].name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM metadata attachment name must not be empty");
    }
    if (attachments[i].metadata_id >= module->metadata_node_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LLVM instruction references unknown metadata node");
    }
    copied_attachments[i].metadata_id = attachments[i].metadata_id;
    IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_string(
        module, attachments[i].name, &copied_attachments[i].name));
  }
  *out_attachments = copied_attachments;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_builder_copy_values(
    loom_llvmir_module_t* module, const loom_llvmir_value_id_t* values,
    iree_host_size_t value_count, loom_llvmir_value_id_t** out_values) {
  *out_values = NULL;
  if (value_count == 0) return iree_ok_status();
  if (values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty LLVM value list has null storage");
  }
  loom_llvmir_value_id_t* copied_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&module->arena, value_count,
                                                 sizeof(*copied_values),
                                                 (void**)&copied_values));
  memcpy(copied_values, values, value_count * sizeof(*copied_values));
  *out_values = copied_values;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_check_value(loom_llvmir_module_t* module,
                                             loom_llvmir_value_id_t value_id) {
  if (value_id >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction references unknown value");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_check_block(loom_llvmir_function_t* function,
                                             loom_llvmir_block_id_t block_id) {
  if (block_id >= function->block_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction references unknown block");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_builder_append_instruction(
    loom_llvmir_block_t* block, const loom_llvmir_instruction_t* instruction) {
  loom_llvmir_instruction_t* stored_instruction = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_append_entry(
      &block->function->module->arena, (void**)&block->instructions,
      &block->instruction_count, &block->instruction_capacity,
      sizeof(*block->instructions), (void**)&stored_instruction));
  *stored_instruction = *instruction;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_define_instruction_value(
    loom_llvmir_block_t* block, loom_llvmir_type_id_t type_id,
    iree_string_view_t name, loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  if (type_id >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM value references unknown type");
  }
  loom_llvmir_value_t* value = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_append_entry(
      &module->arena, (void**)&module->values, &module->value_count,
      &module->value_capacity, sizeof(*module->values), (void**)&value));
  value->kind = LOOM_LLVMIR_VALUE_INSTRUCTION;
  value->type_id = type_id;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_builder_copy_string(module, name, &value->name));
  *out_value_id = (loom_llvmir_value_id_t)(module->value_count - 1);
  value->instruction.function_id = block->function->id;
  value->instruction.block_id = block->id;
  value->instruction.instruction_ordinal = (uint32_t)block->instruction_count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_build_phi(loom_llvmir_block_t* block,
                                    const loom_llvmir_phi_desc_t* desc,
                                    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  if (desc->incoming_count > 0 && desc->incoming == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM phi has null incoming edge storage");
  }
  for (iree_host_size_t i = 0; i < desc->incoming_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_check_value(module, desc->incoming[i].value));
    IREE_RETURN_IF_ERROR(loom_llvmir_check_block(
        block->function, desc->incoming[i].predecessor));
  }
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_PHI,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
  };
  if (desc->incoming_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, desc->incoming_count, sizeof(*instruction.phi.incoming),
        (void**)&instruction.phi.incoming));
    memcpy(instruction.phi.incoming, desc->incoming,
           desc->incoming_count * sizeof(*instruction.phi.incoming));
  }
  instruction.phi.incoming_count = desc->incoming_count;
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_set_phi_incoming(
    loom_llvmir_block_t* block, loom_llvmir_value_id_t phi_value_id,
    const loom_llvmir_phi_incoming_t* incoming,
    iree_host_size_t incoming_count) {
  loom_llvmir_module_t* module = block->function->module;
  if (incoming_count > 0 && incoming == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM phi has null incoming edge storage");
  }
  if (phi_value_id >= module->value_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM phi incoming update references unknown "
                            "result value");
  }
  loom_llvmir_value_t* phi_value = &module->values[phi_value_id];
  if (phi_value->kind != LOOM_LLVMIR_VALUE_INSTRUCTION ||
      phi_value->instruction.function_id != block->function->id ||
      phi_value->instruction.block_id != block->id ||
      phi_value->instruction.instruction_ordinal >= block->instruction_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM phi incoming update result is not defined "
                            "by the target block");
  }
  loom_llvmir_instruction_t* instruction =
      &block->instructions[phi_value->instruction.instruction_ordinal];
  if (instruction->kind != LOOM_LLVMIR_INST_PHI ||
      instruction->result_value_id != phi_value_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM phi incoming update result is not a phi");
  }
  if (instruction->phi.incoming_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "LLVM phi incoming update can only fill an empty "
                            "incoming list");
  }
  for (iree_host_size_t i = 0; i < incoming_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, incoming[i].value));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_check_block(block->function, incoming[i].predecessor));
    if (module->values[incoming[i].value].type_id != phi_value->type_id) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM phi incoming value type mismatch");
    }
  }
  if (incoming_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, incoming_count, sizeof(*instruction->phi.incoming),
        (void**)&instruction->phi.incoming));
    memcpy(instruction->phi.incoming, incoming,
           incoming_count * sizeof(*instruction->phi.incoming));
  }
  instruction->phi.incoming_count = incoming_count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_build_binop(loom_llvmir_block_t* block,
                                      const loom_llvmir_binop_desc_t* desc,
                                      loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->rhs));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_BINOP,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .binop =
          {
              .op = desc->op,
              .lhs = desc->lhs,
              .rhs = desc->rhs,
              .integer_flags = desc->integer_flags,
              .fast_math_flags = desc->fast_math_flags,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_unop(loom_llvmir_block_t* block,
                                     const loom_llvmir_unop_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->value));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_UNOP,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .unop =
          {
              .op = desc->op,
              .value = desc->value,
              .fast_math_flags = desc->fast_math_flags,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_icmp(loom_llvmir_block_t* block,
                                     const loom_llvmir_icmp_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->rhs));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_ICMP,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .icmp =
          {
              .predicate = desc->predicate,
              .lhs = desc->lhs,
              .rhs = desc->rhs,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_fcmp(loom_llvmir_block_t* block,
                                     const loom_llvmir_fcmp_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->rhs));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_FCMP,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .fcmp =
          {
              .predicate = desc->predicate,
              .lhs = desc->lhs,
              .rhs = desc->rhs,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_cast(loom_llvmir_block_t* block,
                                     const loom_llvmir_cast_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->value));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_CAST,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .cast =
          {
              .op = desc->op,
              .value = desc->value,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_gep(loom_llvmir_block_t* block,
                                    const loom_llvmir_gep_desc_t* desc,
                                    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->base));
  if (desc->index_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM getelementptr needs at least one index");
  }
  if (desc->indices == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM getelementptr has null index storage");
  }
  if (desc->element_type >= module->type_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM getelementptr references unknown element type");
  }
  for (iree_host_size_t i = 0; i < desc->index_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->indices[i]));
  }
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_GEP,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .gep =
          {
              .element_type = desc->element_type,
              .base = desc->base,
              .index_count = desc->index_count,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_values(
      module, desc->indices, desc->index_count, &instruction.gep.indices));
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_alloca(loom_llvmir_block_t* block,
                                       const loom_llvmir_alloca_desc_t* desc,
                                       loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  if (desc->element_type >= module->type_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM alloca references unknown element type");
  }

  bool has_explicit_count = true;
  loom_llvmir_value_id_t count = desc->count;
  if (count == LOOM_LLVMIR_VALUE_ID_INVALID) {
    has_explicit_count = false;
    loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_module_get_integer_type(module, 32, &i32_type));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_module_add_integer_constant(module, i32_type, 1, &count));
  } else {
    IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, count));
  }

  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_ALLOCA,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .alloca =
          {
              .element_type = desc->element_type,
              .count = count,
              .has_explicit_count = has_explicit_count,
              .alignment = desc->alignment,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_load(loom_llvmir_block_t* block,
                                     const loom_llvmir_load_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->pointer));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_LOAD,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .load =
          {
              .result_type = desc->result_type,
              .pointer = desc->pointer,
              .alignment = desc->alignment,
              .flags = desc->flags,
              .metadata_attachment_count = desc->metadata_attachment_count,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_metadata_attachments(
      module, desc->metadata_attachments, desc->metadata_attachment_count,
      &instruction.load.metadata_attachments));
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_store(loom_llvmir_block_t* block,
                                      const loom_llvmir_store_desc_t* desc) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->value));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->pointer));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_STORE,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .store =
          {
              .value = desc->value,
              .pointer = desc->pointer,
              .alignment = desc->alignment,
              .flags = desc->flags,
              .metadata_attachment_count = desc->metadata_attachment_count,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_metadata_attachments(
      module, desc->metadata_attachments, desc->metadata_attachment_count,
      &instruction.store.metadata_attachments));
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_atomic_rmw(
    loom_llvmir_block_t* block, const loom_llvmir_atomic_rmw_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->pointer));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->value));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_ATOMIC_RMW,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .atomic_rmw =
          {
              .result_type = desc->result_type,
              .op = desc->op,
              .pointer = desc->pointer,
              .value = desc->value,
              .ordering = desc->ordering,
              .alignment = desc->alignment,
              .flags = desc->flags,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_string(
      module, desc->sync_scope, &instruction.atomic_rmw.sync_scope));
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_cmpxchg(loom_llvmir_block_t* block,
                                        const loom_llvmir_cmpxchg_desc_t* desc,
                                        loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->pointer));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->expected));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->replacement));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_CMPXCHG,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .cmpxchg =
          {
              .result_type = desc->result_type,
              .value_type = desc->value_type,
              .pointer = desc->pointer,
              .expected = desc->expected,
              .replacement = desc->replacement,
              .success_ordering = desc->success_ordering,
              .failure_ordering = desc->failure_ordering,
              .alignment = desc->alignment,
              .flags = desc->flags,
              .is_weak = desc->is_weak,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_string(
      module, desc->sync_scope, &instruction.cmpxchg.sync_scope));
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_call(loom_llvmir_block_t* block,
                                     const loom_llvmir_call_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  loom_llvmir_function_t* callee = desc->callee < module->function_count
                                       ? module->functions[desc->callee]
                                       : NULL;
  if (callee == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM call references unknown callee");
  }
  if (callee->parameter_count != desc->arg_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM call arg count does not match callee");
  }
  if (desc->arg_count > 0 && desc->args == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM call has null arg storage");
  }
  for (iree_host_size_t i = 0; i < desc->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->args[i]));
  }
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_CALL,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .call =
          {
              .callee = desc->callee,
              .arg_count = desc->arg_count,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_values(
      module, desc->args, desc->arg_count, &instruction.call.args));
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_attrs(
      module, desc->result_attrs, desc->result_attr_count,
      &instruction.call.result_attrs));
  const loom_llvmir_type_t* return_type =
      callee->return_type < module->type_count
          ? &module->types[callee->return_type]
          : NULL;
  if (!return_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM call callee has unknown return type");
  }
  if (return_type->kind == LOOM_LLVMIR_TYPE_VOID) {
    if (out_value_id) *out_value_id = LOOM_LLVMIR_VALUE_ID_INVALID;
  } else {
    if (out_value_id == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM non-void call needs an output value");
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
        block, callee->return_type, desc->result_name, out_value_id));
    instruction.result_value_id = *out_value_id;
  }
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_inline_asm(
    loom_llvmir_block_t* block, const loom_llvmir_inline_asm_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  if (iree_string_view_is_empty(desc->asm_template)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM inline asm template must not be empty");
  }
  if (desc->arg_count > 0 && desc->args == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM inline asm has null arg storage");
  }
  loom_llvmir_type_id_t inline_asm_pointer_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_pointer_type(module, 0, &inline_asm_pointer_type));
  (void)inline_asm_pointer_type;
  for (iree_host_size_t i = 0; i < desc->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->args[i]));
  }
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_INLINE_ASM,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .inline_asm =
          {
              .result_type = desc->result_type,
              .flags = desc->flags,
              .arg_count = desc->arg_count,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_string(
      module, desc->asm_template, &instruction.inline_asm.asm_template));
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_string(
      module, desc->constraints, &instruction.inline_asm.constraints));
  IREE_RETURN_IF_ERROR(loom_llvmir_builder_copy_values(
      module, desc->args, desc->arg_count, &instruction.inline_asm.args));
  const loom_llvmir_type_t* result_type =
      desc->result_type < module->type_count ? &module->types[desc->result_type]
                                             : NULL;
  if (!result_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM inline asm references unknown result type");
  }
  if (result_type->kind == LOOM_LLVMIR_TYPE_VOID) {
    if (out_value_id) *out_value_id = LOOM_LLVMIR_VALUE_ID_INVALID;
  } else {
    if (out_value_id == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM non-void inline asm needs an output value");
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
        block, desc->result_type, desc->result_name, out_value_id));
    instruction.result_value_id = *out_value_id;
  }
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_select(loom_llvmir_block_t* block,
                                       const loom_llvmir_select_desc_t* desc,
                                       loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->condition));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->true_value));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->false_value));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_SELECT,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .select =
          {
              .condition = desc->condition,
              .true_value = desc->true_value,
              .false_value = desc->false_value,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_extract_element(
    loom_llvmir_block_t* block, const loom_llvmir_extract_element_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->vector));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->index));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_EXTRACT_ELEMENT,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .extract_element =
          {
              .vector = desc->vector,
              .index = desc->index,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_extract_value(
    loom_llvmir_block_t* block, const loom_llvmir_extract_value_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->aggregate));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_EXTRACT_VALUE,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .extract_value =
          {
              .aggregate = desc->aggregate,
              .index = desc->index,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_insert_element(
    loom_llvmir_block_t* block, const loom_llvmir_insert_element_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->vector));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->element));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->index));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_INSERT_ELEMENT,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .insert_element =
          {
              .vector = desc->vector,
              .element = desc->element,
              .index = desc->index,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_shuffle_vector(
    loom_llvmir_block_t* block, const loom_llvmir_shuffle_vector_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_module_t* module = block->function->module;
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->rhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(module, desc->mask));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_SHUFFLE_VECTOR,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .shuffle_vector =
          {
              .lhs = desc->lhs,
              .rhs = desc->rhs,
              .mask = desc->mask,
          },
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_define_instruction_value(
      block, desc->result_type, desc->result_name, out_value_id));
  instruction.result_value_id = *out_value_id;
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_ret_void(loom_llvmir_block_t* block) {
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_RET,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .ret = {.has_value = false, .value = LOOM_LLVMIR_VALUE_ID_INVALID},
  };
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_ret(loom_llvmir_block_t* block,
                                    loom_llvmir_value_id_t value) {
  IREE_RETURN_IF_ERROR(loom_llvmir_check_value(block->function->module, value));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_RET,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .ret = {.has_value = true, .value = value},
  };
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_br(loom_llvmir_block_t* block,
                                   loom_llvmir_block_id_t target) {
  IREE_RETURN_IF_ERROR(loom_llvmir_check_block(block->function, target));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_BR,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .br = {.target = target},
  };
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_cond_br(loom_llvmir_block_t* block,
                                        loom_llvmir_value_id_t condition,
                                        loom_llvmir_block_id_t true_block,
                                        loom_llvmir_block_id_t false_block) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_check_value(block->function->module, condition));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_block(block->function, true_block));
  IREE_RETURN_IF_ERROR(loom_llvmir_check_block(block->function, false_block));
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_COND_BR,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
      .cond_br =
          {
              .condition = condition,
              .true_block = true_block,
              .false_block = false_block,
          },
  };
  return loom_llvmir_builder_append_instruction(block, &instruction);
}

iree_status_t loom_llvmir_build_unreachable(loom_llvmir_block_t* block) {
  loom_llvmir_instruction_t instruction = {
      .kind = LOOM_LLVMIR_INST_UNREACHABLE,
      .result_value_id = LOOM_LLVMIR_VALUE_ID_INVALID,
  };
  return loom_llvmir_builder_append_instruction(block, &instruction);
}
