// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/abi/c/projection.h"

#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_c_abi_emission_state_t {
  loom_value_id_t result_storage;
  loom_value_id_t result_address;
} loom_low_c_abi_emission_state_t;

typedef uint16_t loom_low_c_abi_required_ops_t;
enum loom_low_c_abi_required_op_bits_e {
  LOOM_LOW_C_ABI_REQUIRED_OP_ADD_ADDRESS = 1u << 0,
  LOOM_LOW_C_ABI_REQUIRED_OP_MASK_U32 = 1u << 1,
  LOOM_LOW_C_ABI_REQUIRED_OP_WIDEN_U32 = 1u << 2,
  LOOM_LOW_C_ABI_REQUIRED_OP_TRUNCATE_U64 = 1u << 3,
  LOOM_LOW_C_ABI_REQUIRED_OP_SHIFT_U64 = 1u << 4,
  LOOM_LOW_C_ABI_REQUIRED_OP_OR_U64 = 1u << 5,
  LOOM_LOW_C_ABI_REQUIRED_OP_CONSTANT_U64 = 1u << 6,
  LOOM_LOW_C_ABI_REQUIRED_OP_MEMORY_OFFSET = 1u << 7,
};

static iree_status_t loom_low_c_abi_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  return count == 0
             ? iree_ok_status()
             : iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static void loom_low_c_abi_initialize_objects(
    const loom_c_abi_object_t* source_objects, uint16_t object_count,
    loom_low_c_abi_object_projection_t* objects) {
  for (uint16_t i = 0; i < object_count; ++i) {
    IREE_ASSERT_LE(source_objects[i].byte_length, UINT32_MAX);
    IREE_ASSERT_LE(source_objects[i].byte_alignment, UINT32_MAX);
    objects[i] = (loom_low_c_abi_object_projection_t){
        .storage_byte_length = (uint32_t)source_objects[i].byte_length,
        .storage_byte_alignment = (uint32_t)source_objects[i].byte_alignment,
        .physical_offset = LOOM_LOW_C_ABI_INDEX_NONE,
        .transport = LOOM_LOW_C_ABI_OBJECT_TRANSPORT_VALUE,
    };
  }
}

static void loom_low_c_abi_initialize_leaves(
    const loom_c_abi_leaf_t* source_leaves, uint16_t leaf_count,
    loom_low_c_abi_leaf_projection_t* leaves) {
  for (uint16_t i = 0; i < leaf_count; ++i) {
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[i];
    loom_low_c_abi_leaf_flags_t flags = 0;
    if (source_leaf->flags & LOOM_C_ABI_LEAF_FLAG_POINTER) {
      flags |= LOOM_LOW_C_ABI_LEAF_FLAG_POINTER;
    } else if (source_leaf->flags & LOOM_C_ABI_LEAF_FLAG_VECTOR) {
      flags |= LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR;
    } else if (source_leaf->flags & LOOM_C_ABI_LEAF_FLAG_BOOLEAN) {
      flags |= LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN;
    }
    leaves[i] = (loom_low_c_abi_leaf_projection_t){
        .load_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        .store_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        .pack_move_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        .unpack_move_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        .physical_index = LOOM_LOW_C_ABI_INDEX_NONE,
        .flags = flags,
    };
  }
}

iree_status_t loom_low_c_abi_projection_plan_allocate(
    const loom_c_abi_signature_layout_t* layout,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    uint16_t argument_carrier_capacity, uint16_t result_carrier_capacity,
    iree_arena_allocator_t* arena,
    loom_low_c_abi_projection_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_LE(layout->signature.arguments.count, UINT16_MAX);
  IREE_ASSERT_LE(layout->signature.results.count, UINT16_MAX);
  IREE_ASSERT_EQ(layout->argument_logical_count, logical_argument_count);
  IREE_ASSERT_EQ(layout->result_logical_count, logical_result_count);
  *out_plan = NULL;

  loom_low_c_abi_projection_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->layout = layout;
  plan->logical_argument_types = logical_argument_types;
  plan->logical_result_types = logical_result_types;
  plan->argument_object_count = (uint16_t)layout->signature.arguments.count;
  plan->result_object_count = (uint16_t)layout->signature.results.count;
  plan->argument_carrier_capacity = argument_carrier_capacity;
  plan->result_carrier_capacity = result_carrier_capacity;
  plan->indirect_result_object = LOOM_LOW_C_ABI_INDEX_NONE;
  plan->indirect_result_argument = LOOM_LOW_C_ABI_INDEX_NONE;
  plan->indirect_result_echo = LOOM_LOW_C_ABI_INDEX_NONE;

  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, plan->argument_object_count, sizeof(*plan->arguments),
      (void**)&plan->arguments));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, plan->result_object_count, sizeof(*plan->results),
      (void**)&plan->results));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, layout->argument_leaf_count, sizeof(*plan->argument_leaves),
      (void**)&plan->argument_leaves));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, layout->result_leaf_count, sizeof(*plan->result_leaves),
      (void**)&plan->result_leaves));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, argument_carrier_capacity, sizeof(*plan->argument_carriers),
      (void**)&plan->argument_carriers));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_allocate_array(
      arena, result_carrier_capacity, sizeof(*plan->result_carriers),
      (void**)&plan->result_carriers));

  loom_low_c_abi_initialize_objects(
      layout->arguments, plan->argument_object_count, plan->arguments);
  loom_low_c_abi_initialize_objects(layout->results, plan->result_object_count,
                                    plan->results);
  loom_low_c_abi_initialize_leaves(layout->argument_leaves,
                                   layout->argument_leaf_count,
                                   plan->argument_leaves);
  loom_low_c_abi_initialize_leaves(
      layout->result_leaves, layout->result_leaf_count, plan->result_leaves);
  *out_plan = plan;
  return iree_ok_status();
}

static uint16_t loom_low_c_abi_append_carrier(
    loom_low_c_abi_carrier_projection_t* carriers, uint16_t* count,
    uint16_t capacity, loom_low_c_abi_object_projection_t* objects,
    uint16_t object_count, uint16_t object_index, uint32_t byte_offset,
    uint32_t storage_byte_length) {
  IREE_ASSERT_LT(*count, capacity);
  const uint16_t index = (*count)++;
  carriers[index] = (loom_low_c_abi_carrier_projection_t){
      .byte_offset = byte_offset,
      .pack_move_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
      .unpack_move_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
  };
  if (object_index != LOOM_LOW_C_ABI_INDEX_NONE) {
    IREE_ASSERT_LT(object_index, object_count);
    loom_low_c_abi_object_projection_t* object = &objects[object_index];
    if (object->physical_count == 0) {
      object->physical_offset = index;
    } else {
      IREE_ASSERT_EQ(object->physical_offset + object->physical_count, index);
    }
    ++object->physical_count;
    IREE_ASSERT_LE((uint64_t)byte_offset + storage_byte_length, UINT32_MAX);
    object->storage_byte_length = iree_max(object->storage_byte_length,
                                           byte_offset + storage_byte_length);
  }
  return index;
}

uint16_t loom_low_c_abi_projection_append_argument_carrier(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint32_t byte_offset, uint32_t storage_byte_length) {
  return loom_low_c_abi_append_carrier(
      plan->argument_carriers, &plan->argument_carrier_count,
      plan->argument_carrier_capacity, plan->arguments,
      plan->argument_object_count, object_index, byte_offset,
      storage_byte_length);
}

uint16_t loom_low_c_abi_projection_append_result_carrier(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint32_t byte_offset, uint32_t storage_byte_length) {
  return loom_low_c_abi_append_carrier(
      plan->result_carriers, &plan->result_carrier_count,
      plan->result_carrier_capacity, plan->results, plan->result_object_count,
      object_index, byte_offset, storage_byte_length);
}

void loom_low_c_abi_projection_set_argument_address(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index) {
  IREE_ASSERT_LT(object_index, plan->argument_object_count);
  loom_low_c_abi_object_projection_t* object = &plan->arguments[object_index];
  IREE_ASSERT_EQ(object->physical_count, 1);
  object->transport = LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS;
}

void loom_low_c_abi_projection_set_indirect_result(
    loom_low_c_abi_projection_plan_t* plan, uint16_t object_index,
    uint16_t address_argument, uint16_t echo_result) {
  IREE_ASSERT_LT(object_index, plan->result_object_count);
  IREE_ASSERT_LT(address_argument, plan->argument_carrier_count);
  IREE_ASSERT(echo_result == LOOM_LOW_C_ABI_INDEX_NONE ||
              echo_result < plan->result_carrier_count);
  IREE_ASSERT_EQ(plan->indirect_result_object, LOOM_LOW_C_ABI_INDEX_NONE);
  plan->results[object_index].transport =
      LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS;
  plan->indirect_result_object = object_index;
  plan->indirect_result_argument = address_argument;
  plan->indirect_result_echo = echo_result;
}

static bool loom_low_c_abi_select_move(
    const loom_low_c_abi_projection_target_t* target, loom_type_t source_type,
    loom_type_t result_type, uint32_t* out_descriptor) {
  *out_descriptor = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT(loom_low_type_is_register(source_type));
  IREE_ASSERT(loom_low_type_is_register(result_type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(source_type),
                 target->ops.register_descriptor_set_stable_id);
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(result_type),
                 target->ops.register_descriptor_set_stable_id);
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(source_type), 1);
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(result_type), 1);
  if (loom_low_register_type_same_class(source_type, result_type)) {
    return true;
  }
  const uint16_t source_class = loom_low_register_type_class_id(source_type);
  const uint16_t result_class = loom_low_register_type_class_id(result_type);
  for (uint16_t i = 0; i < target->move_count; ++i) {
    if (target->moves[i].source_register_class == source_class &&
        target->moves[i].result_register_class == result_class) {
      *out_descriptor = target->moves[i].descriptor;
      return *out_descriptor != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
    }
  }
  return false;
}

static const loom_low_c_abi_vector_memory_descriptor_t*
loom_low_c_abi_select_vector_memory(
    const loom_low_c_abi_projection_target_t* target, loom_type_t type) {
  IREE_ASSERT(loom_low_type_is_register(type));
  IREE_ASSERT_EQ(loom_low_register_type_descriptor_set_stable_id(type),
                 target->ops.register_descriptor_set_stable_id);
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(type), 1);
  const uint16_t register_class = loom_low_register_type_class_id(type);
  for (uint16_t i = 0; i < target->vector_memory_count; ++i) {
    if (target->vector_memory[i].register_class == register_class) {
      return &target->vector_memory[i];
    }
  }
  return NULL;
}

static loom_type_t loom_low_c_abi_register_type(
    const loom_low_c_abi_projection_ops_t* ops, uint16_t register_class) {
  return loom_low_register_type(ops->register_descriptor_set_stable_id,
                                register_class, 1);
}

static bool loom_low_c_abi_register_type_matches(
    const loom_low_c_abi_projection_ops_t* ops, uint16_t register_class,
    loom_type_t type) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_unit_count(type) == 1 &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             ops->register_descriptor_set_stable_id &&
         loom_low_register_type_class_id(type) == register_class;
}

static bool loom_low_c_abi_select_scalar_memory(
    const loom_low_c_abi_projection_ops_t* ops, uint64_t byte_length,
    uint32_t* out_load, uint32_t* out_store, loom_type_t* out_bits_type) {
  switch (byte_length) {
    case 1:
      *out_load = ops->load_u8;
      *out_store = ops->store_u8;
      *out_bits_type =
          loom_low_c_abi_register_type(ops, ops->bits32_register_class);
      break;
    case 2:
      *out_load = ops->load_u16;
      *out_store = ops->store_u16;
      *out_bits_type =
          loom_low_c_abi_register_type(ops, ops->bits32_register_class);
      break;
    case 4:
      *out_load = ops->load_u32;
      *out_store = ops->store_u32;
      *out_bits_type =
          loom_low_c_abi_register_type(ops, ops->bits32_register_class);
      break;
    case 8:
      *out_load = ops->load_u64;
      *out_store = ops->store_u64;
      *out_bits_type =
          loom_low_c_abi_register_type(ops, ops->bits64_register_class);
      break;
    default:
      return false;
  }
  return true;
}

static bool loom_low_c_abi_finalize_leaf_ops(
    const loom_low_c_abi_projection_target_t* target,
    const loom_c_abi_object_t* source_objects,
    loom_low_c_abi_object_projection_t* objects, uint16_t object_count,
    const loom_c_abi_leaf_t* source_leaves,
    loom_low_c_abi_leaf_projection_t* leaves, const loom_type_t* logical_types,
    iree_string_view_t* out_reason) {
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_object_t* source_object = &source_objects[i];
    loom_low_c_abi_object_projection_t* object = &objects[i];
    for (uint16_t j = 0; j < source_object->leaf_count; ++j) {
      const uint16_t leaf_index = source_object->leaf_offset + j;
      const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
      loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
      IREE_ASSERT_LE(source_leaf->byte_offset, UINT32_MAX);
      IREE_ASSERT_LE(source_leaf->byte_length, UINT32_MAX);
      const loom_type_t logical_type =
          logical_types[source_leaf->logical_offset];
      if (!loom_low_type_is_register(logical_type) ||
          loom_low_register_type_unit_count(logical_type) != 1) {
        *out_reason =
            IREE_SV("source leaf has no singular Low register carrier");
        return false;
      }
      IREE_ASSERT_EQ(
          loom_low_register_type_descriptor_set_stable_id(logical_type),
          target->ops.register_descriptor_set_stable_id);
      if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR) {
        const loom_low_c_abi_vector_memory_descriptor_t* memory =
            loom_low_c_abi_select_vector_memory(target, logical_type);
        if (memory) {
          leaf->load_descriptor = memory->load_descriptor;
          leaf->store_descriptor = memory->store_descriptor;
          IREE_ASSERT_LE(
              source_leaf->byte_offset + memory->register_byte_length,
              UINT32_MAX);
          object->storage_byte_length = iree_max(
              object->storage_byte_length, (uint32_t)source_leaf->byte_offset +
                                               memory->register_byte_length);
        }
        continue;
      }

      uint64_t byte_length = source_leaf->byte_length;
      if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
        IREE_ASSERT_EQ(source_leaf->logical_count, 2);
        if (byte_length != target->ops.address_byte_length ||
            !loom_low_c_abi_register_type_matches(
                &target->ops, target->ops.address_register_class,
                logical_type) ||
            !loom_low_c_abi_register_type_matches(
                &target->ops, target->ops.address_register_class,
                logical_types[source_leaf->logical_offset + 1])) {
          *out_reason = IREE_SV(
              "source pointer width or carriers do not match target address");
          return false;
        }
      }
      loom_type_t bits_type;
      if (!loom_low_c_abi_select_scalar_memory(
              &target->ops, byte_length, &leaf->load_descriptor,
              &leaf->store_descriptor, &bits_type)) {
        *out_reason =
            IREE_SV("target cannot address one source scalar leaf width");
        return false;
      }
      if (!(leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER)) {
        loom_low_c_abi_select_move(target, logical_type, bits_type,
                                   &leaf->pack_move_descriptor);
        loom_low_c_abi_select_move(target, bits_type, logical_type,
                                   &leaf->unpack_move_descriptor);
      }
    }
  }
  return true;
}

static bool loom_low_c_abi_validate_leaf_ops(
    const loom_low_c_abi_projection_ops_t* ops,
    const loom_c_abi_object_t* source_objects,
    const loom_low_c_abi_object_projection_t* objects, uint16_t object_count,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_type_t* logical_types, iree_string_view_t* out_reason) {
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_object_t* source_object = &source_objects[i];
    const loom_low_c_abi_object_projection_t* object = &objects[i];
    const bool uses_storage =
        object->transport == LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS ||
        !object->has_direct_projection;
    for (uint16_t j = 0; j < source_object->leaf_count; ++j) {
      const uint16_t leaf_index = source_object->leaf_offset + j;
      const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
      const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
      if (uses_storage &&
          (leaf->load_descriptor == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
           leaf->store_descriptor == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)) {
        *out_reason = IREE_SV(
            "target has no address load/store for one projected source leaf");
        return false;
      }
      if (leaf->flags &
          (LOOM_LOW_C_ABI_LEAF_FLAG_POINTER | LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR |
           LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY)) {
        continue;
      }
      const loom_type_t logical_type =
          logical_types[source_leaf->logical_offset];
      const loom_type_t bits_type = loom_low_c_abi_register_type(
          ops, source_leaf->byte_length <= 4 ? ops->bits32_register_class
                                             : ops->bits64_register_class);
      if (!loom_low_register_type_same_class(logical_type, bits_type) &&
          (leaf->pack_move_descriptor == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
           leaf->unpack_move_descriptor == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)) {
        *out_reason = IREE_SV(
            "target cannot move one source scalar to its integer bit carrier");
        return false;
      }
    }
  }
  return true;
}

static void loom_low_c_abi_clear_direct_projection(
    const loom_c_abi_object_t* source_object,
    loom_low_c_abi_object_projection_t* object,
    loom_low_c_abi_leaf_projection_t* leaves,
    loom_low_c_abi_carrier_projection_t* carriers) {
  object->has_direct_projection = false;
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    loom_low_c_abi_leaf_projection_t* leaf =
        &leaves[source_object->leaf_offset + i];
    leaf->physical_index = LOOM_LOW_C_ABI_INDEX_NONE;
    leaf->flags &= ~LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY;
  }
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    carriers[object->physical_offset + i].leaf_count = 0;
  }
}

static void loom_low_c_abi_bind_direct_projection(
    const loom_low_c_abi_projection_ops_t* ops,
    const loom_c_abi_object_t* source_object,
    loom_low_c_abi_object_projection_t* object,
    const loom_c_abi_leaf_t* source_leaves,
    loom_low_c_abi_leaf_projection_t* leaves, const loom_type_t* logical_types,
    loom_low_c_abi_carrier_projection_t* carriers,
    const loom_type_t* physical_types) {
  if (object->transport == LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS) {
    return;
  }
  object->has_direct_projection = true;
  uint16_t local_carrier_index = 0;
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    const uint32_t leaf_byte_length =
        leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER
            ? ops->address_byte_length
            : (uint32_t)source_leaf->byte_length;
    while (local_carrier_index < object->physical_count) {
      const loom_low_c_abi_carrier_projection_t* carrier =
          &carriers[object->physical_offset + local_carrier_index];
      const uint32_t carrier_byte_end =
          local_carrier_index + 1 < object->physical_count
              ? carriers[object->physical_offset + local_carrier_index + 1]
                    .byte_offset
              : (uint32_t)source_object->byte_length;
      if (source_leaf->byte_offset >= carrier->byte_offset &&
          source_leaf->byte_offset + leaf_byte_length <= carrier_byte_end) {
        break;
      }
      ++local_carrier_index;
    }
    if (local_carrier_index == object->physical_count) {
      loom_low_c_abi_clear_direct_projection(source_object, object, leaves,
                                             carriers);
      return;
    }
    const uint16_t physical_index =
        object->physical_offset + local_carrier_index;
    loom_low_c_abi_carrier_projection_t* carrier = &carriers[physical_index];
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR) {
      const loom_type_t logical_type =
          logical_types[source_leaf->logical_offset];
      if (!loom_low_type_is_register(logical_type) ||
          loom_low_register_type_unit_count(logical_type) != 1 ||
          !loom_low_register_type_same_class(logical_type,
                                             physical_types[physical_index]) ||
          source_leaf->byte_offset != carrier->byte_offset) {
        loom_low_c_abi_clear_direct_projection(source_object, object, leaves,
                                               carriers);
        return;
      }
    }
    leaf->physical_index = physical_index;
    ++carrier->leaf_count;
  }

  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    const loom_low_c_abi_carrier_projection_t* carrier =
        &carriers[leaf->physical_index];
    const loom_type_t logical_type = logical_types[source_leaf->logical_offset];
    if (!(leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) &&
        carrier->leaf_count == 1 &&
        source_leaf->byte_offset == carrier->byte_offset &&
        loom_low_type_is_register(logical_type) &&
        loom_low_register_type_unit_count(logical_type) == 1 &&
        loom_low_register_type_same_class(
            logical_type, physical_types[leaf->physical_index])) {
      leaf->flags |= LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY;
    }
  }
}

static bool loom_low_c_abi_finalize_carrier_moves(
    const loom_low_c_abi_projection_target_t* target,
    const loom_c_abi_object_t* source_objects,
    const loom_low_c_abi_object_projection_t* objects, uint16_t object_count,
    const loom_low_c_abi_leaf_projection_t* leaves,
    loom_low_c_abi_carrier_projection_t* carriers,
    const loom_type_t* physical_types, iree_string_view_t* out_reason) {
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_object_t* source_object = &source_objects[i];
    const loom_low_c_abi_object_projection_t* object = &objects[i];
    if (!object->has_direct_projection) {
      continue;
    }
    for (uint16_t j = 0; j < object->physical_count; ++j) {
      const uint16_t physical_index = object->physical_offset + j;
      loom_low_c_abi_carrier_projection_t* carrier = &carriers[physical_index];
      bool identity = false;
      for (uint16_t k = 0; k < source_object->leaf_count; ++k) {
        const loom_low_c_abi_leaf_projection_t* leaf =
            &leaves[source_object->leaf_offset + k];
        if (leaf->physical_index == physical_index &&
            (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY)) {
          identity = true;
          break;
        }
      }
      if (identity) {
        continue;
      }
      if (carrier->leaf_count == 0) {
        if (!loom_low_c_abi_register_type_matches(
                &target->ops, target->ops.bits64_register_class,
                physical_types[physical_index])) {
          *out_reason = IREE_SV(
              "target cannot materialize padding in one physical carrier");
          return false;
        }
        continue;
      }
      const loom_type_t bits64_type = loom_low_c_abi_register_type(
          &target->ops, target->ops.bits64_register_class);
      if (!loom_low_c_abi_select_move(target, bits64_type,
                                      physical_types[physical_index],
                                      &carrier->pack_move_descriptor) ||
          !loom_low_c_abi_select_move(target, physical_types[physical_index],
                                      bits64_type,
                                      &carrier->unpack_move_descriptor)) {
        *out_reason = IREE_SV(
            "target cannot move packed object bits to one physical carrier");
        return false;
      }
    }
  }
  return true;
}

static loom_low_c_abi_required_ops_t loom_low_c_abi_collect_required_ops(
    const loom_c_abi_object_t* source_objects,
    const loom_low_c_abi_object_projection_t* objects, uint16_t object_count,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_low_c_abi_carrier_projection_t* carriers) {
  loom_low_c_abi_required_ops_t required = 0;
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_object_t* source_object = &source_objects[i];
    const loom_low_c_abi_object_projection_t* object = &objects[i];
    const bool uses_storage =
        object->transport == LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS ||
        !object->has_direct_projection;
    for (uint16_t j = 0; j < source_object->leaf_count; ++j) {
      const uint16_t leaf_index = source_object->leaf_offset + j;
      const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
      const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
      if (uses_storage) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_MEMORY_OFFSET;
      }
      if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_ADD_ADDRESS |
                    LOOM_LOW_C_ABI_REQUIRED_OP_CONSTANT_U64;
      }
      if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_MASK_U32;
      }
      if (!object->has_direct_projection ||
          (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY)) {
        continue;
      }
      const loom_low_c_abi_carrier_projection_t* carrier =
          &carriers[leaf->physical_index];
      if (!(leaf->flags & (LOOM_LOW_C_ABI_LEAF_FLAG_POINTER |
                           LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR)) &&
          source_leaf->byte_length <= 4) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_WIDEN_U32 |
                    LOOM_LOW_C_ABI_REQUIRED_OP_TRUNCATE_U64;
        if (source_leaf->byte_length < 4) {
          required |= LOOM_LOW_C_ABI_REQUIRED_OP_MASK_U32;
        }
      }
      if (source_leaf->byte_offset != carrier->byte_offset) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_SHIFT_U64;
      }
    }
    if (!object->has_direct_projection) {
      continue;
    }
    for (uint16_t j = 0; j < object->physical_count; ++j) {
      const loom_low_c_abi_carrier_projection_t* carrier =
          &carriers[object->physical_offset + j];
      if (carrier->leaf_count == 0) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_CONSTANT_U64;
      } else if (carrier->leaf_count > 1) {
        required |= LOOM_LOW_C_ABI_REQUIRED_OP_OR_U64;
      }
    }
  }
  return required;
}

static bool loom_low_c_abi_validate_required_ops(
    loom_low_c_abi_required_ops_t required,
    const loom_low_c_abi_projection_ops_t* ops,
    iree_string_view_t* out_reason) {
  IREE_ASSERT(!(required & LOOM_LOW_C_ABI_REQUIRED_OP_MEMORY_OFFSET) ||
              ops->memory_offset_name != NULL);
  IREE_ASSERT(!(required & LOOM_LOW_C_ABI_REQUIRED_OP_MASK_U32) ||
              ops->mask_name != NULL);
  IREE_ASSERT(!(required & LOOM_LOW_C_ABI_REQUIRED_OP_SHIFT_U64) ||
              ops->shift_name != NULL);
  IREE_ASSERT(!(required & LOOM_LOW_C_ABI_REQUIRED_OP_CONSTANT_U64) ||
              ops->constant_name != NULL);
  if (((required & LOOM_LOW_C_ABI_REQUIRED_OP_ADD_ADDRESS) &&
       ops->add_address == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_MASK_U32) &&
       ops->mask_u32 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_WIDEN_U32) &&
       ops->widen_u32 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_TRUNCATE_U64) &&
       ops->truncate_u64 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_SHIFT_U64) &&
       (ops->shift_left_u64 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
        ops->shift_right_u64 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_OR_U64) &&
       ops->or_u64 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) ||
      ((required & LOOM_LOW_C_ABI_REQUIRED_OP_CONSTANT_U64) &&
       ops->constant_u64 == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)) {
    *out_reason = IREE_SV(
        "target has no fixed operation required by C source projection");
    return false;
  }
  return true;
}

static bool loom_low_c_abi_finalize_side(
    loom_low_c_abi_projection_plan_t* plan,
    const loom_low_c_abi_projection_target_t* target,
    const loom_c_abi_object_t* source_objects,
    loom_low_c_abi_object_projection_t* objects, uint16_t object_count,
    const loom_c_abi_leaf_t* source_leaves,
    loom_low_c_abi_leaf_projection_t* leaves, const loom_type_t* logical_types,
    loom_low_c_abi_carrier_projection_t* carriers,
    const loom_type_t* physical_types, iree_string_view_t* out_reason) {
  if (!loom_low_c_abi_finalize_leaf_ops(target, source_objects, objects,
                                        object_count, source_leaves, leaves,
                                        logical_types, out_reason)) {
    return false;
  }
  for (uint16_t i = 0; i < object_count; ++i) {
    loom_low_c_abi_bind_direct_projection(
        &target->ops, &source_objects[i], &objects[i], source_leaves, leaves,
        logical_types, carriers, physical_types);
  }
  if (!loom_low_c_abi_validate_leaf_ops(&target->ops, source_objects, objects,
                                        object_count, source_leaves, leaves,
                                        logical_types, out_reason)) {
    return false;
  }
  return loom_low_c_abi_finalize_carrier_moves(target, source_objects, objects,
                                               object_count, leaves, carriers,
                                               physical_types, out_reason);
}

static void loom_low_c_abi_assert_physical_types(
    const loom_low_c_abi_projection_target_t* target,
    const loom_type_t* physical_types, uint16_t physical_count) {
  for (uint16_t i = 0; i < physical_count; ++i) {
    IREE_ASSERT(loom_low_type_is_register(physical_types[i]));
    IREE_ASSERT_EQ(
        loom_low_register_type_descriptor_set_stable_id(physical_types[i]),
        target->ops.register_descriptor_set_stable_id);
    IREE_ASSERT_EQ(loom_low_register_type_unit_count(physical_types[i]), 1);
  }
}

bool loom_low_c_abi_projection_plan_finalize(
    loom_low_c_abi_projection_plan_t* plan,
    const loom_low_c_abi_projection_target_t* target,
    const loom_type_t* physical_argument_types,
    uint16_t physical_argument_count, const loom_type_t* physical_result_types,
    uint16_t physical_result_count, iree_string_view_t* out_reason) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_reason);
  *out_reason = iree_string_view_empty();
  IREE_ASSERT_EQ(physical_argument_count, plan->argument_carrier_count);
  IREE_ASSERT_EQ(physical_result_count, plan->result_carrier_count);
  if (target->ops.byte_order != LOOM_LOW_C_ABI_BYTE_ORDER_LITTLE) {
    *out_reason =
        IREE_SV("C source projection requires little-endian target byte order");
    return false;
  }
  if (target->ops.address_byte_length != 8 ||
      target->ops.address_register_class != target->ops.bits64_register_class) {
    *out_reason = IREE_SV(
        "C source projection requires a 64-bit integer address carrier");
    return false;
  }
  if (target->ops.register_descriptor_set_stable_id == 0 ||
      target->ops.bits32_register_class == LOOM_LOW_REGISTER_CLASS_ID_INVALID ||
      target->ops.bits64_register_class == LOOM_LOW_REGISTER_CLASS_ID_INVALID ||
      target->ops.address_register_class ==
          LOOM_LOW_REGISTER_CLASS_ID_INVALID) {
    *out_reason = IREE_SV("target has no native C projection carrier types");
    return false;
  }
  loom_low_c_abi_assert_physical_types(target, physical_argument_types,
                                       physical_argument_count);
  loom_low_c_abi_assert_physical_types(target, physical_result_types,
                                       physical_result_count);
  plan->physical_argument_types = physical_argument_types;
  plan->physical_result_types = physical_result_types;
  plan->ops = target->ops;
  if (!loom_low_c_abi_finalize_side(
          plan, target, plan->layout->arguments, plan->arguments,
          plan->argument_object_count, plan->layout->argument_leaves,
          plan->argument_leaves, plan->logical_argument_types,
          plan->argument_carriers, physical_argument_types, out_reason)) {
    return false;
  }
  if (!loom_low_c_abi_finalize_side(
          plan, target, plan->layout->results, plan->results,
          plan->result_object_count, plan->layout->result_leaves,
          plan->result_leaves, plan->logical_result_types,
          plan->result_carriers, physical_result_types, out_reason)) {
    return false;
  }
  const loom_low_c_abi_required_ops_t required =
      loom_low_c_abi_collect_required_ops(
          plan->layout->arguments, plan->arguments, plan->argument_object_count,
          plan->layout->argument_leaves, plan->argument_leaves,
          plan->argument_carriers) |
      loom_low_c_abi_collect_required_ops(
          plan->layout->results, plan->results, plan->result_object_count,
          plan->layout->result_leaves, plan->result_leaves,
          plan->result_carriers);
  return loom_low_c_abi_validate_required_ops(required, &target->ops,
                                              out_reason);
}

static loom_location_id_t loom_low_c_abi_location(const loom_op_t* source_op) {
  return source_op ? source_op->location : LOOM_LOCATION_UNKNOWN;
}

static iree_status_t loom_low_c_abi_emit_packet(
    loom_low_lower_context_t* context, uint32_t descriptor_ref,
    const loom_value_id_t* operands, uint16_t operand_count,
    iree_string_view_t immediate_name, loom_attribute_t immediate,
    loom_type_t result_type, bool tie_first_operand,
    loom_location_id_t location, loom_value_id_t* out_value) {
  IREE_ASSERT_NE(descriptor_ref, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  if (out_value) {
    *out_value = LOOM_VALUE_ID_INVALID;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ref);
  loom_named_attr_t attr = {.value = immediate};
  const uint16_t attr_count = iree_string_view_is_empty(immediate_name) ? 0 : 1;
  if (attr_count) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context), immediate_name, &attr.name_id));
  }
  const uint16_t result_count =
      loom_type_kind(result_type) == LOOM_TYPE_NONE ? 0 : 1;
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* op = NULL;
  if (loom_low_descriptor_layout(descriptor_set, descriptor)->op_kind ==
      LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
    IREE_ASSERT_EQ(result_count, 1);
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
        loom_low_lower_context_builder(context), descriptor_set, descriptor,
        loom_make_named_attr_slice(&attr, attr_count), result_type, location,
        &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
        loom_low_lower_context_builder(context), descriptor_set, descriptor,
        /*access_flags=*/0, operands, operand_count,
        loom_make_named_attr_slice(&attr, attr_count),
        result_count ? &result_type : NULL, result_count,
        tie_first_operand ? &tie : NULL, tie_first_operand ? 1 : 0, location,
        &op));
  }
  if (out_value) {
    *out_value = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_emit_move(loom_low_lower_context_t* context,
                                              uint32_t descriptor,
                                              loom_value_id_t source,
                                              loom_type_t result_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_value) {
  if (descriptor == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    *out_value = source;
    return iree_ok_status();
  }
  return loom_low_c_abi_emit_packet(
      context, descriptor, &source, 1, iree_string_view_empty(),
      loom_attr_absent(), result_type, /*tie_first_operand=*/false, location,
      out_value);
}

static iree_status_t loom_low_c_abi_emit_copy(loom_low_lower_context_t* context,
                                              loom_value_id_t source,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_value) {
  loom_op_t* copy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), source, /*detached=*/false,
      loom_module_value_type(loom_low_lower_context_module(context), source),
      location, &copy));
  *out_value = loom_low_copy_result(copy);
  return iree_ok_status();
}

static loom_type_t loom_low_c_abi_bits_type(
    const loom_low_c_abi_projection_plan_t* plan, uint64_t byte_length) {
  return loom_low_c_abi_register_type(
      &plan->ops, byte_length <= 4 ? plan->ops.bits32_register_class
                                   : plan->ops.bits64_register_class);
}

static iree_string_view_t loom_low_c_abi_immediate_name(const char* name) {
  return name ? iree_make_cstring_view(name) : iree_string_view_empty();
}

static iree_status_t loom_low_c_abi_emit_load(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, uint32_t descriptor,
    loom_value_id_t base, uint32_t byte_offset, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, descriptor, &base, 1,
      loom_low_c_abi_immediate_name(plan->ops.memory_offset_name),
      loom_attr_i64(byte_offset), result_type, /*tie_first_operand=*/false,
      location, out_value);
}

static iree_status_t loom_low_c_abi_emit_store(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, uint32_t descriptor,
    loom_value_id_t value, loom_value_id_t base, uint32_t byte_offset,
    loom_location_id_t location) {
  const loom_value_id_t operands[] = {value, base};
  return loom_low_c_abi_emit_packet(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_low_c_abi_immediate_name(plan->ops.memory_offset_name),
      loom_attr_i64(byte_offset), loom_type_none(), /*tie_first_operand=*/false,
      location,
      /*out_value=*/NULL);
}

static iree_status_t loom_low_c_abi_emit_mask_u32(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_value_id_t value,
    uint32_t mask, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, plan->ops.mask_u32, &value, 1,
      loom_low_c_abi_immediate_name(plan->ops.mask_name), loom_attr_i64(mask),
      result_type, /*tie_first_operand=*/true, location, out_value);
}

static iree_status_t loom_low_c_abi_emit_widen_u32(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_value_id_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, plan->ops.widen_u32, &value, 1, iree_string_view_empty(),
      loom_attr_absent(),
      loom_low_c_abi_register_type(&plan->ops, plan->ops.bits64_register_class),
      /*tie_first_operand=*/false, location, out_value);
}

static iree_status_t loom_low_c_abi_emit_truncate_u64(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_value_id_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, plan->ops.truncate_u64, &value, 1, iree_string_view_empty(),
      loom_attr_absent(),
      loom_low_c_abi_register_type(&plan->ops, plan->ops.bits32_register_class),
      /*tie_first_operand=*/false, location, out_value);
}

static iree_status_t loom_low_c_abi_emit_shift_u64(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, bool left,
    loom_value_id_t value, uint32_t shift, loom_location_id_t location,
    loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, left ? plan->ops.shift_left_u64 : plan->ops.shift_right_u64,
      &value, 1, loom_low_c_abi_immediate_name(plan->ops.shift_name),
      loom_attr_i64(shift),
      loom_low_c_abi_register_type(&plan->ops, plan->ops.bits64_register_class),
      /*tie_first_operand=*/true, location, out_value);
}

static iree_status_t loom_low_c_abi_emit_or_u64(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {lhs, rhs};
  return loom_low_c_abi_emit_packet(
      context, plan->ops.or_u64, operands, IREE_ARRAYSIZE(operands),
      iree_string_view_empty(), loom_attr_absent(),
      loom_low_c_abi_register_type(&plan->ops, plan->ops.bits64_register_class),
      /*tie_first_operand=*/true, location, out_value);
}

static iree_status_t loom_low_c_abi_emit_pointer_address(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_value_id_t base,
    loom_value_id_t byte_offset, loom_location_id_t location,
    loom_value_id_t* out_address) {
  loom_value_id_t base_copy;
  IREE_RETURN_IF_ERROR(
      loom_low_c_abi_emit_copy(context, base, location, &base_copy));
  const loom_value_id_t operands[] = {base_copy, byte_offset};
  return loom_low_c_abi_emit_packet(
      context, plan->ops.add_address, operands, IREE_ARRAYSIZE(operands),
      iree_string_view_empty(), loom_attr_absent(),
      loom_low_c_abi_register_type(&plan->ops,
                                   plan->ops.address_register_class),
      /*tie_first_operand=*/true, location, out_address);
}

static iree_status_t loom_low_c_abi_emit_zero(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_low_c_abi_emit_packet(
      context, plan->ops.constant_u64, /*operands=*/NULL, /*operand_count=*/0,
      loom_low_c_abi_immediate_name(plan->ops.constant_name), loom_attr_i64(0),
      result_type,
      /*tie_first_operand=*/false, location, out_value);
}

static iree_status_t loom_low_c_abi_reserve_storage(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_low_c_abi_object_projection_t* object,
    loom_location_id_t location, loom_value_id_t* out_storage,
    loom_value_id_t* out_address) {
  loom_builder_t* ir = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      ir, object->storage_byte_length, object->storage_byte_alignment,
      loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), location, &storage_op));
  *out_storage = loom_low_storage_reserve_storage(storage_op);
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      ir, *out_storage, /*offset=*/0,
      loom_low_c_abi_register_type(&plan->ops,
                                   plan->ops.address_register_class),
      location, &address_op));
  *out_address = loom_low_storage_address_result(address_op);
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_pack_leaves_to_address(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_value_id_t* logical_values, loom_value_id_t address,
    loom_location_id_t location) {
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    loom_value_id_t bits = logical_values[source_leaf->logical_offset];
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_store(
          context, plan, leaf->store_descriptor, bits, address,
          (uint32_t)source_leaf->byte_offset, location));
      continue;
    }
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_pointer_address(
          context, plan, bits, logical_values[source_leaf->logical_offset + 1],
          location, &bits));
    } else {
      const loom_type_t bits_type =
          loom_low_c_abi_bits_type(plan, source_leaf->byte_length);
      IREE_RETURN_IF_ERROR(
          loom_low_c_abi_emit_move(context, leaf->pack_move_descriptor, bits,
                                   bits_type, location, &bits));
    }
    IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_store(
        context, plan, leaf->store_descriptor, bits, address,
        (uint32_t)source_leaf->byte_offset, location));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_unpack_leaves_from_address(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_type_t* logical_types, loom_value_id_t storage,
    loom_value_id_t address, loom_value_id_t* logical_values,
    loom_location_id_t location) {
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_VECTOR) {
      if (storage != LOOM_VALUE_ID_INVALID) {
        loom_op_t* reload = NULL;
        IREE_RETURN_IF_ERROR(loom_low_reload_build(
            loom_low_lower_context_builder(context), storage,
            (uint32_t)source_leaf->byte_offset,
            logical_types[source_leaf->logical_offset], location, &reload));
        logical_values[source_leaf->logical_offset] =
            loom_low_reload_result(reload);
      } else {
        IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_load(
            context, plan, leaf->load_descriptor, address,
            (uint32_t)source_leaf->byte_offset,
            logical_types[source_leaf->logical_offset], location,
            &logical_values[source_leaf->logical_offset]));
      }
      continue;
    }

    const loom_type_t bits_type =
        loom_low_c_abi_bits_type(plan, source_leaf->byte_length);
    loom_value_id_t bits;
    IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_load(
        context, plan, leaf->load_descriptor, address,
        (uint32_t)source_leaf->byte_offset, bits_type, location, &bits));
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
      logical_values[source_leaf->logical_offset] = bits;
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_zero(
          context, plan, logical_types[source_leaf->logical_offset + 1],
          location, &logical_values[source_leaf->logical_offset + 1]));
      continue;
    }
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_mask_u32(
          context, plan, bits, 1, logical_types[source_leaf->logical_offset],
          location, &bits));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_move(
          context, leaf->unpack_move_descriptor, bits,
          logical_types[source_leaf->logical_offset], location, &bits));
    }
    logical_values[source_leaf->logical_offset] = bits;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_pack_direct_object(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_low_c_abi_object_projection_t* object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_value_id_t* logical_values,
    const loom_low_c_abi_carrier_projection_t* carriers,
    const loom_type_t* physical_types, loom_value_id_t* physical_values,
    loom_location_id_t location) {
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    physical_values[object->physical_offset + i] = LOOM_VALUE_ID_INVALID;
  }
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    IREE_ASSERT_NE(leaf->physical_index, LOOM_LOW_C_ABI_INDEX_NONE);
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY) {
      physical_values[leaf->physical_index] =
          logical_values[source_leaf->logical_offset];
      continue;
    }

    const loom_low_c_abi_carrier_projection_t* carrier =
        &carriers[leaf->physical_index];
    loom_value_id_t fragment = logical_values[source_leaf->logical_offset];
    const uint32_t byte_length = leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER
                                     ? plan->ops.address_byte_length
                                     : (uint32_t)source_leaf->byte_length;
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_pointer_address(
          context, plan, fragment,
          logical_values[source_leaf->logical_offset + 1], location,
          &fragment));
    } else {
      const loom_type_t fragment_type =
          loom_low_c_abi_bits_type(plan, byte_length);
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_move(
          context, leaf->pack_move_descriptor, fragment, fragment_type,
          location, &fragment));
      if (byte_length < 4) {
        const uint32_t mask =
            leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN
                ? 1u
                : (1u << (byte_length * 8u)) - 1u;
        IREE_RETURN_IF_ERROR(
            loom_low_c_abi_emit_copy(context, fragment, location, &fragment));
        IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_mask_u32(
            context, plan, fragment, mask, fragment_type, location, &fragment));
      }
      if (byte_length <= 4) {
        IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_widen_u32(
            context, plan, fragment, location, &fragment));
      }
    }

    const uint32_t shift =
        ((uint32_t)source_leaf->byte_offset - carrier->byte_offset) * 8u;
    if (shift != 0) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_shift_u64(
          context, plan, /*left=*/true, fragment, shift, location, &fragment));
    }
    loom_value_id_t* accumulator = &physical_values[leaf->physical_index];
    if (*accumulator == LOOM_VALUE_ID_INVALID) {
      *accumulator = fragment;
    } else {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_or_u64(
          context, plan, *accumulator, fragment, location, accumulator));
    }
  }

  for (uint16_t i = 0; i < object->physical_count; ++i) {
    const uint16_t physical_index = object->physical_offset + i;
    const loom_low_c_abi_carrier_projection_t* carrier =
        &carriers[physical_index];
    loom_value_id_t value = physical_values[physical_index];
    if (value == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_zero(
          context, plan, physical_types[physical_index], location, &value));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_move(
          context, carrier->pack_move_descriptor, value,
          physical_types[physical_index], location, &value));
    }
    physical_values[physical_index] = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_unpack_direct_object(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_low_c_abi_object_projection_t* object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_type_t* logical_types,
    const loom_low_c_abi_carrier_projection_t* carriers,
    const loom_value_id_t* physical_values, loom_value_id_t* logical_values,
    loom_location_id_t location) {
  for (uint16_t i = 0; i < source_object->leaf_count; ++i) {
    const uint16_t leaf_index = source_object->leaf_offset + i;
    const loom_c_abi_leaf_t* source_leaf = &source_leaves[leaf_index];
    const loom_low_c_abi_leaf_projection_t* leaf = &leaves[leaf_index];
    IREE_ASSERT_NE(leaf->physical_index, LOOM_LOW_C_ABI_INDEX_NONE);
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_DIRECT_IDENTITY) {
      logical_values[source_leaf->logical_offset] =
          physical_values[leaf->physical_index];
      continue;
    }

    const loom_low_c_abi_carrier_projection_t* carrier =
        &carriers[leaf->physical_index];
    loom_value_id_t fragment = physical_values[leaf->physical_index];
    IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_move(
        context, carrier->unpack_move_descriptor, fragment,
        loom_low_c_abi_register_type(&plan->ops,
                                     plan->ops.bits64_register_class),
        location, &fragment));
    const uint32_t shift =
        ((uint32_t)source_leaf->byte_offset - carrier->byte_offset) * 8u;
    if (shift != 0) {
      IREE_RETURN_IF_ERROR(
          loom_low_c_abi_emit_copy(context, fragment, location, &fragment));
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_shift_u64(
          context, plan, /*left=*/false, fragment, shift, location, &fragment));
    }
    if (leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_POINTER) {
      logical_values[source_leaf->logical_offset] = fragment;
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_zero(
          context, plan, logical_types[source_leaf->logical_offset + 1],
          location, &logical_values[source_leaf->logical_offset + 1]));
      continue;
    }

    if (source_leaf->byte_length <= 4) {
      IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_truncate_u64(
          context, plan, fragment, location, &fragment));
      if (source_leaf->byte_length < 4) {
        const uint32_t mask =
            leaf->flags & LOOM_LOW_C_ABI_LEAF_FLAG_NORMALIZE_BOOLEAN
                ? 1u
                : (1u << (source_leaf->byte_length * 8u)) - 1u;
        IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_mask_u32(
            context, plan, fragment, mask,
            loom_low_c_abi_register_type(&plan->ops,
                                         plan->ops.bits32_register_class),
            location, &fragment));
      }
    }
    IREE_RETURN_IF_ERROR(loom_low_c_abi_emit_move(
        context, leaf->unpack_move_descriptor, fragment,
        logical_types[source_leaf->logical_offset], location, &fragment));
    logical_values[source_leaf->logical_offset] = fragment;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_pack_object(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_low_c_abi_object_projection_t* object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_value_id_t* logical_values,
    const loom_low_c_abi_carrier_projection_t* carriers,
    const loom_type_t* physical_types, loom_value_id_t* physical_values,
    loom_location_id_t location) {
  if (object->transport == LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS) {
    loom_value_id_t storage;
    loom_value_id_t address;
    IREE_RETURN_IF_ERROR(loom_low_c_abi_reserve_storage(
        context, plan, object, location, &storage, &address));
    IREE_RETURN_IF_ERROR(loom_low_c_abi_pack_leaves_to_address(
        context, plan, source_object, source_leaves, leaves, logical_values,
        address, location));
    IREE_ASSERT_EQ(object->physical_count, 1);
    physical_values[object->physical_offset] = address;
    return iree_ok_status();
  }
  if (object->has_direct_projection) {
    return loom_low_c_abi_pack_direct_object(
        context, plan, source_object, object, source_leaves, leaves,
        logical_values, carriers, physical_types, physical_values, location);
  }
  loom_value_id_t storage;
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_low_c_abi_reserve_storage(
      context, plan, object, location, &storage, &address));
  IREE_RETURN_IF_ERROR(loom_low_c_abi_pack_leaves_to_address(
      context, plan, source_object, source_leaves, leaves, logical_values,
      address, location));
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    const uint16_t physical_index = object->physical_offset + i;
    loom_op_t* reload = NULL;
    IREE_RETURN_IF_ERROR(loom_low_reload_build(
        loom_low_lower_context_builder(context), storage,
        carriers[physical_index].byte_offset, physical_types[physical_index],
        location, &reload));
    physical_values[physical_index] = loom_low_reload_result(reload);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_c_abi_unpack_object(
    loom_low_lower_context_t* context,
    const loom_low_c_abi_projection_plan_t* plan,
    const loom_c_abi_object_t* source_object,
    const loom_low_c_abi_object_projection_t* object,
    const loom_c_abi_leaf_t* source_leaves,
    const loom_low_c_abi_leaf_projection_t* leaves,
    const loom_type_t* logical_types,
    const loom_low_c_abi_carrier_projection_t* carriers,
    const loom_value_id_t* physical_values, loom_value_id_t* logical_values,
    loom_location_id_t location) {
  if (object->transport == LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS) {
    IREE_ASSERT_EQ(object->physical_count, 1);
    return loom_low_c_abi_unpack_leaves_from_address(
        context, plan, source_object, source_leaves, leaves, logical_types,
        LOOM_VALUE_ID_INVALID, physical_values[object->physical_offset],
        logical_values, location);
  }
  if (object->has_direct_projection) {
    return loom_low_c_abi_unpack_direct_object(
        context, plan, source_object, object, source_leaves, leaves,
        logical_types, carriers, physical_values, logical_values, location);
  }
  loom_value_id_t storage;
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_low_c_abi_reserve_storage(
      context, plan, object, location, &storage, &address));
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    const uint16_t physical_index = object->physical_offset + i;
    loom_op_t* spill = NULL;
    IREE_RETURN_IF_ERROR(loom_low_spill_build(
        loom_low_lower_context_builder(context),
        physical_values[physical_index], storage,
        carriers[physical_index].byte_offset, location, &spill));
  }
  return loom_low_c_abi_unpack_leaves_from_address(
      context, plan, source_object, source_leaves, leaves, logical_types,
      storage, address, logical_values, location);
}

static iree_status_t loom_low_c_abi_emission_state_allocate(
    iree_arena_allocator_t* arena,
    loom_low_c_abi_emission_state_t** out_state) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_state), (void**)out_state));
  **out_state = (loom_low_c_abi_emission_state_t){
      .result_storage = LOOM_VALUE_ID_INVALID,
      .result_address = LOOM_VALUE_ID_INVALID,
  };
  return iree_ok_status();
}

iree_status_t loom_low_c_abi_projection_emit_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state) {
  (void)user_data;
  const loom_low_c_abi_projection_plan_t* plan = target_plan;
  const loom_location_id_t location = loom_low_c_abi_location(source_op);
  switch (phase) {
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_DEFINITION_ENTRY: {
      IREE_ASSERT_EQ(input_count, plan->argument_carrier_count);
      IREE_ASSERT_EQ(output_count, plan->layout->argument_logical_count);
      for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_c_abi_unpack_object(
            context, plan, &plan->layout->arguments[i], &plan->arguments[i],
            plan->layout->argument_leaves, plan->argument_leaves,
            plan->logical_argument_types, plan->argument_carriers, input_values,
            output_values, location));
      }
      if (plan->indirect_result_object != LOOM_LOW_C_ABI_INDEX_NONE) {
        loom_low_c_abi_emission_state_t* state = NULL;
        IREE_RETURN_IF_ERROR(
            loom_low_c_abi_emission_state_allocate(state_arena, &state));
        state->result_address = input_values[plan->indirect_result_argument];
        *inout_state = state;
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_DEFINITION_EXIT: {
      IREE_ASSERT_EQ(input_count, plan->layout->result_logical_count);
      IREE_ASSERT_EQ(output_count, plan->result_carrier_count);
      for (uint16_t i = 0; i < plan->result_object_count; ++i) {
        if (i == plan->indirect_result_object) {
          const loom_low_c_abi_emission_state_t* state = *inout_state;
          IREE_ASSERT_NE(state, NULL);
          IREE_RETURN_IF_ERROR(loom_low_c_abi_pack_leaves_to_address(
              context, plan, &plan->layout->results[i],
              plan->layout->result_leaves, plan->result_leaves, input_values,
              state->result_address, location));
          if (plan->indirect_result_echo != LOOM_LOW_C_ABI_INDEX_NONE) {
            output_values[plan->indirect_result_echo] = state->result_address;
          }
        } else {
          IREE_RETURN_IF_ERROR(loom_low_c_abi_pack_object(
              context, plan, &plan->layout->results[i], &plan->results[i],
              plan->layout->result_leaves, plan->result_leaves, input_values,
              plan->result_carriers, plan->physical_result_types, output_values,
              location));
        }
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_CALL_ARGUMENTS: {
      IREE_ASSERT_EQ(input_count, plan->layout->argument_logical_count);
      IREE_ASSERT_EQ(output_count, plan->argument_carrier_count);
      if (plan->indirect_result_object != LOOM_LOW_C_ABI_INDEX_NONE) {
        loom_low_c_abi_emission_state_t* state = NULL;
        IREE_RETURN_IF_ERROR(
            loom_low_c_abi_emission_state_allocate(state_arena, &state));
        IREE_RETURN_IF_ERROR(loom_low_c_abi_reserve_storage(
            context, plan, &plan->results[plan->indirect_result_object],
            location, &state->result_storage, &state->result_address));
        output_values[plan->indirect_result_argument] = state->result_address;
        *inout_state = state;
      }
      for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_c_abi_pack_object(
            context, plan, &plan->layout->arguments[i], &plan->arguments[i],
            plan->layout->argument_leaves, plan->argument_leaves, input_values,
            plan->argument_carriers, plan->physical_argument_types,
            output_values, location));
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_SOURCE_SIGNATURE_EMISSION_CALL_RESULTS: {
      IREE_ASSERT_EQ(input_count, plan->result_carrier_count);
      IREE_ASSERT_EQ(output_count, plan->layout->result_logical_count);
      for (uint16_t i = 0; i < plan->result_object_count; ++i) {
        if (i == plan->indirect_result_object) {
          loom_low_c_abi_emission_state_t* state = *inout_state;
          IREE_ASSERT_NE(state, NULL);
          if (plan->indirect_result_echo != LOOM_LOW_C_ABI_INDEX_NONE) {
            state->result_address = input_values[plan->indirect_result_echo];
          }
          IREE_RETURN_IF_ERROR(loom_low_c_abi_unpack_leaves_from_address(
              context, plan, &plan->layout->results[i],
              plan->layout->result_leaves, plan->result_leaves,
              plan->logical_result_types, state->result_storage,
              state->result_address, output_values, location));
        } else {
          IREE_RETURN_IF_ERROR(loom_low_c_abi_unpack_object(
              context, plan, &plan->layout->results[i], &plan->results[i],
              plan->layout->result_leaves, plan->result_leaves,
              plan->logical_result_types, plan->result_carriers, input_values,
              output_values, location));
        }
      }
      return iree_ok_status();
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown C source signature emission phase");
  IREE_BUILTIN_UNREACHABLE();
}
