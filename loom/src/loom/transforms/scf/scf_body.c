// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_body.h"

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/encoding.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/walk.h"

typedef struct loom_scf_body_builder_t {
  // Scheduling admission, or serial cloning without additional admission rules.
  loom_scf_body_mode_t mode;
  // Function owned immutable-space records, extended as children are cloned.
  const loom_scf_memory_t* spaces;
  // Allocated access correspondence slots.
  iree_host_size_t memory_capacity;
  // Module providing maintained type uses and interned attribute payloads.
  const loom_module_t* module;
  // Block defining the source iteration's local values.
  const loom_block_t* block;
  // Optional enclosing block whose direct definitions are schedule captures.
  const loom_block_t* capture_block;
  // Current operation, whose result-type self references need no dependency.
  const loom_op_t* op;
  // Destination plan populated in authored order.
  loom_scf_body_t* body;
  // Arena owning all plan storage.
  iree_arena_allocator_t* arena;
  // Allocated reference slots in the packed dependency array.
  iree_host_size_t reference_capacity;
  // Allocated outer operation slots.
  iree_host_size_t operation_capacity;
  // Allocated top-level source-order boundary slots.
  iree_host_size_t source_order_boundary_capacity;
  // Outer scheduling unit receiving the current operation's captures/effects.
  loom_scf_body_operation_t* operation;
  // Optional memory metadata for the current scheduling unit.
  loom_scf_body_access_unit_t* access_unit;
  // First control operation outside the supported structured body.
  const loom_op_t* unstructured_op;
} loom_scf_body_builder_t;

static iree_status_t loom_scf_body_append_reference(loom_value_id_t value_id,
                                                    void* user_data) {
  loom_scf_body_builder_t* builder = (loom_scf_body_builder_t*)user_data;
  const loom_value_t* value = loom_module_value(builder->module, value_id);
  bool allow_identity_mapping = false;
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* definition_block = loom_value_def_block(value);
    if (definition_block != builder->block &&
        definition_block != builder->capture_block) {
      return iree_ok_status();
    }
  } else {
    const loom_op_t* definition = loom_value_def_op(value);
    if (!definition || definition == builder->op) {
      return iree_ok_status();
    }
    if (!definition->parent_block) {
      allow_identity_mapping = true;
    } else if (definition->parent_block != builder->block &&
               definition->parent_block != builder->capture_block) {
      return iree_ok_status();
    }
  }
  loom_scf_body_t* body = builder->body;
  if (body->reference_count == builder->reference_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, body->reference_count, body->reference_count + 1,
        sizeof(*body->references), &builder->reference_capacity,
        (void**)&body->references));
  }
  body->references[body->reference_count++] = (loom_scf_body_reference_t){
      .value_id = value_id,
      .allow_identity_mapping = allow_identity_mapping,
  };
  return iree_ok_status();
}

static iree_status_t loom_scf_body_append_attribute(
    loom_scf_body_builder_t* builder, const loom_attribute_t* attribute) {
  switch ((loom_attr_kind_t)attribute->kind) {
    case LOOM_ATTR_TYPE:
      return loom_type_walk_value_refs(
          builder->module,
          loom_type_table_get(&builder->module->types, attribute->type_id),
          loom_scf_body_append_reference, builder);
    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        const loom_predicate_t* predicate = &attribute->predicate_list[i];
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          if (predicate->arg_tags[j] != LOOM_PRED_ARG_VALUE) {
            continue;
          }
          IREE_RETURN_IF_ERROR(loom_scf_body_append_reference(
              (loom_value_id_t)predicate->args[j], builder));
        }
      }
      break;
    case LOOM_ATTR_DICT:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_scf_body_append_attribute(
            builder, &attribute->dict_entries[i].value));
      }
      break;
    case LOOM_ATTR_PARAMETERIZED:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_scf_body_append_attribute(
            builder, &attribute->parameterized_slots[i]));
      }
      break;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_scf_body_append_attribute(
            builder, &attribute->parameterized_array[i]));
      }
      break;
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding = loom_module_encoding(
          builder->module, (uint16_t)attribute->encoding_id);
      for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_scf_body_append_attribute(
            builder, &encoding->attributes[i].value));
      }
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

loom_scf_body_effect_flags_t loom_scf_body_operation_effects(
    const loom_module_t* module, const loom_op_t* op) {
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  loom_scf_body_effect_flags_t flags = 0;
  if (loom_traits_may_read(traits)) {
    flags |= LOOM_SCF_BODY_EFFECT_READ;
    loom_memory_access_t access = loom_memory_access_cast(module, op);
    if (!loom_memory_access_isa(access) ||
        loom_memory_access_operation_kind(access) !=
            LOOM_MEMORY_ACCESS_OPERATION_LOAD) {
      flags |= LOOM_SCF_BODY_EFFECT_NON_LOAD_READ;
    }
  }
  if (loom_traits_may_write(traits)) {
    flags |= LOOM_SCF_BODY_EFFECT_WRITE;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT)) {
    flags |= LOOM_SCF_BODY_EFFECT_SOURCE_ORDER;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_CONVERGENT)) {
    flags |= LOOM_SCF_BODY_EFFECT_CONVERGENT;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                   LOOM_TRAIT_UNKNOWN_EFFECTS |
                                   LOOM_TRAIT_POISON_BOUNDARY |
                                   LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    flags |= LOOM_SCF_BODY_EFFECT_ORDERED;
  }
  if (flags == 0 && !iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    flags |= LOOM_SCF_BODY_EFFECT_ORDERED;
  }
  return flags;
}

static iree_status_t loom_scf_body_capture_payload(
    loom_scf_body_builder_t* builder, const loom_op_t* op) {
  builder->op = op;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_scf_body_append_reference(operands[i], builder));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_use_iterator_t dependencies;
    loom_module_value_type_dependencies(builder->module, results[i],
                                        &dependencies);
    for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
         provider != LOOM_VALUE_ID_INVALID;
         provider = loom_type_dependencies_next(&dependencies)) {
      IREE_RETURN_IF_ERROR(loom_scf_body_append_reference(provider, builder));
    }
  }
  const loom_attribute_t* attributes = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_scf_body_append_attribute(builder, &attributes[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_scf_body_append_memory(
    loom_scf_body_builder_t* builder, const loom_op_t* op) {
  loom_scf_body_t* body = builder->body;
  if (body->accesses.count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SCF memory access count exceeds uint32");
  }
  if (body->accesses.count == builder->memory_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, body->accesses.count, body->accesses.count + 1,
        sizeof(*body->accesses.operations), &builder->memory_capacity,
        (void**)&body->accesses.operations));
  }
  body->accesses.operations[body->accesses.count++] =
      (loom_ir_remap_op_projection_t){.source_op = op};
  return iree_ok_status();
}

static iree_status_t loom_scf_body_capture_operation(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_scf_body_builder_t* builder = user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    *out_result = LOOM_WALK_SKIP;
    return iree_ok_status();
  }
  if (builder->mode == LOOM_SCF_BODY_MODE_PROJECT_MEMORY) {
    if (loom_memory_access_isa(loom_memory_access_cast(builder->module, op))) {
      return loom_scf_body_append_memory(builder, op);
    }
    return iree_ok_status();
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
  const bool structured = loom_scf_if_isa(op) || loom_scf_for_isa(op);
  if (op->successor_count != 0 ||
      (!structured &&
       (op->region_count != 0 || (vtable && vtable->region_count != 0)))) {
    builder->unstructured_op = op;
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }
  loom_scf_body_t* body = builder->body;
  if (context->depth == 0) {
    if (op == builder->block->last_op) {
      builder->operation = &body->terminator;
    } else {
      if (body->count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "SCF body operation count exceeds uint32");
      }
      if (body->count == builder->operation_capacity) {
        if (builder->spaces) {
          iree_host_size_t capacity = builder->operation_capacity;
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              builder->arena, body->count, (iree_host_size_t)body->count + 1,
              sizeof(*body->accesses.units), &capacity,
              (void**)&body->accesses.units));
        }
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            builder->arena, body->count, (iree_host_size_t)body->count + 1,
            sizeof(*body->operations), &builder->operation_capacity,
            (void**)&body->operations));
      }
      builder->operation = &body->operations[body->count++];
    }
    *builder->operation = (loom_scf_body_operation_t){
        .op = op,
        .reference_begin = body->reference_count,
    };
    if (builder->spaces && op != builder->block->last_op) {
      builder->access_unit = &body->accesses.units[body->count - 1];
      *builder->access_unit = (loom_scf_body_access_unit_t){
          .begin = body->accesses.count,
      };
    }
  }
  IREE_RETURN_IF_ERROR(loom_scf_body_capture_payload(builder, op));
  builder->operation->reference_count =
      body->reference_count - builder->operation->reference_begin;
  // Structured control and yields contribute captures, while their nested
  // operations establish the scheduling unit's complete effects.
  if (!structured && !loom_scf_yield_isa(op)) {
    loom_scf_body_effect_flags_t effects =
        loom_scf_body_operation_effects(builder->module, op);
    builder->operation->effects |= effects;
    if (builder->spaces) {
      loom_memory_access_t access =
          loom_memory_access_cast(builder->module, op);
      uint8_t memory_effects = 0;
      if (loom_memory_access_isa(access)) {
        IREE_RETURN_IF_ERROR(loom_scf_body_append_memory(builder, op));
        builder->access_unit->count++;
        uint8_t space = loom_scf_memory_lookup(builder->spaces, op);
        const bool ordinary = !(effects & ~(LOOM_SCF_BODY_EFFECT_READ |
                                            LOOM_SCF_BODY_EFFECT_WRITE));
        if (ordinary && space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
          memory_effects = LOOM_SCF_BODY_MEMORY_WORKGROUP;
        } else if (ordinary && effects == LOOM_SCF_BODY_EFFECT_READ) {
          memory_effects = space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL
                               ? LOOM_SCF_BODY_MEMORY_GLOBAL_LOAD
                               : LOOM_SCF_BODY_MEMORY_OTHER_LOAD;
        } else {
          memory_effects = LOOM_SCF_BODY_MEMORY_UNSUPPORTED;
        }
      } else if (loom_kernel_barrier_isa(op) &&
                 loom_kernel_barrier_memory_space(op) ==
                     LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
        memory_effects = LOOM_SCF_BODY_MEMORY_WORKGROUP;
      } else if (effects && effects != LOOM_SCF_BODY_EFFECT_CONVERGENT) {
        memory_effects = LOOM_SCF_BODY_MEMORY_UNSUPPORTED;
      }
      builder->access_unit->effects |= memory_effects;
    }
    if (context->depth == 0 &&
        iree_any_bit_set(effects, LOOM_SCF_BODY_EFFECT_SOURCE_ORDER)) {
      if (body->source_order_boundary_count ==
          builder->source_order_boundary_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            builder->arena, body->source_order_boundary_count,
            (iree_host_size_t)body->source_order_boundary_count + 1,
            sizeof(*body->source_order_boundaries),
            &builder->source_order_boundary_capacity,
            (void**)&body->source_order_boundaries));
      }
      body->source_order_boundaries[body->source_order_boundary_count++] =
          body->count - 1;
    }
    if (iree_any_bit_set(effects, LOOM_SCF_BODY_EFFECT_READ) &&
        !iree_any_bit_set(effects, LOOM_SCF_BODY_EFFECT_NON_LOAD_READ)) {
      ++builder->operation->load_count;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scf_body_build(
    const loom_module_t* module, const loom_block_t* block,
    const loom_block_t* capture_block, const loom_scf_memory_t* spaces,
    loom_scf_body_mode_t mode, iree_arena_allocator_t* arena,
    loom_scf_body_t* out_body, const loom_op_t** out_unstructured_op) {
  *out_body = (loom_scf_body_t){0};
  *out_unstructured_op = NULL;
  loom_scf_body_builder_t builder = {
      .spaces = spaces,
      .mode = mode,
      .module = module,
      .block = block,
      .capture_block = capture_block,
      .body = out_body,
      .arena = arena,
  };
  loom_walk_result_t result = LOOM_WALK_CONTINUE;
  iree_status_t status = loom_walk_region(
      module, block->parent_region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){.fn = loom_scf_body_capture_operation,
                             .user_data = &builder},
      &result);
  *out_unstructured_op = builder.unstructured_op;
  return status;
}
