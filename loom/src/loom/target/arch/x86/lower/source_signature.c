// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/source_signature.h"

#include "loom/codegen/low/lower/abi/c/projection.h"
#include "loom/target/arch/x86/call_abi.h"
#include "loom/target/arch/x86/descriptors/entry.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/arch/x86/source_signature.h"

enum {
  LOOM_X86_SOURCE_SIGNATURE_INTEGER_ARGUMENT_REGISTER_COUNT = 6,
  LOOM_X86_SOURCE_SIGNATURE_SSE_ARGUMENT_REGISTER_COUNT = 8,
  LOOM_X86_SOURCE_SIGNATURE_RESULT_REGISTER_COUNT = 2,
};

static const int64_t kLoomX86SourceSignatureIntegerArgumentRegisters[] = {
    7, 6, 2, 1, 8, 9};
static const int64_t kLoomX86SourceSignatureIntegerResultRegisters[] = {0, 2};

typedef struct loom_x86_source_signature_plan_builder_t {
  // Active source-signature lowering context.
  loom_low_lower_context_t* context;
  // Arena retaining physical signatures and the common projection plan.
  iree_arena_allocator_t* arena;
  // Common source-object projection under construction.
  loom_low_c_abi_projection_plan_t* projection;
  // Physical Low argument types under construction.
  loom_type_t* argument_types;
  // Physical Low result types under construction.
  loom_type_t* result_types;
  // Source-semantic argument types parallel to argument_types.
  loom_type_t* semantic_argument_types;
  // Source-semantic result types parallel to result_types.
  loom_type_t* semantic_result_types;
  // Exact register placements parallel to argument_types.
  int64_t* argument_registers;
  // Exact stack placements parallel to argument_types.
  int64_t* argument_offsets;
  // Exact register placements parallel to result_types.
  int64_t* result_registers;
  // Number of populated physical arguments.
  uint16_t argument_count;
  // Number of populated physical results.
  uint16_t result_count;
  // Consumed integer argument registers.
  uint8_t integer_argument_count;
  // Consumed SIMD argument registers.
  uint8_t sse_argument_count;
  // Consumed integer result registers.
  uint8_t integer_result_count;
  // Consumed SIMD result registers.
  uint8_t sse_result_count;
  // Current stack argument byte length.
  uint32_t stack_bytes;
  // Maximum stack argument alignment.
  uint8_t stack_alignment;
} loom_x86_source_signature_plan_builder_t;

static iree_status_t loom_x86_source_signature_unsupported(
    loom_low_lower_context_t* context, iree_string_view_t reason) {
  return loom_low_lower_emit_source_signature_unsupported(
      context, loom_low_lower_context_source_function(context).op, reason);
}

static iree_status_t loom_x86_source_signature_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  return count == 0
             ? iree_ok_status()
             : iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static iree_status_t loom_x86_source_signature_classify_objects(
    loom_low_lower_context_t* context,
    const loom_c_abi_signature_layout_t* layout,
    const loom_c_abi_object_t* objects, uint16_t object_count,
    const loom_c_abi_leaf_t* leaves, uint16_t maximum_vector_bytes,
    iree_arena_allocator_t* arena,
    loom_x86_source_signature_classification_t** out_classifications) {
  *out_classifications = NULL;
  loom_x86_source_signature_classification_t* classifications = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, object_count, sizeof(*classifications), (void**)&classifications));
  for (uint16_t i = 0; i < object_count; ++i) {
    const loom_c_abi_leaf_t* object_leaves =
        objects[i].leaf_count ? leaves + objects[i].leaf_offset : NULL;
    if (!loom_x86_source_signature_classify(layout, &objects[i], object_leaves,
                                            maximum_vector_bytes,
                                            &classifications[i])) {
      return loom_x86_source_signature_unsupported(
          context, IREE_SV("source object is outside the x86-64 SysV family"));
    }
  }
  *out_classifications = classifications;
  return iree_ok_status();
}

static uint16_t loom_x86_source_signature_direct_carrier_count(
    const loom_x86_source_signature_classification_t* classification,
    uint8_t* out_integer_count, uint8_t* out_sse_count) {
  *out_integer_count = 0;
  *out_sse_count = 0;
  uint16_t carrier_count = 0;
  for (uint8_t i = 0; i < classification->eightbyte_count; ++i) {
    switch (classification->eightbytes[i]) {
      case LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER:
        ++*out_integer_count;
        ++carrier_count;
        break;
      case LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE:
        ++*out_sse_count;
        ++carrier_count;
        break;
      case LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP:
      case LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS:
        break;
      case LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY:
        IREE_ASSERT_UNREACHABLE("direct source object has memory eightbyte");
        break;
    }
  }
  return carrier_count;
}

static loom_type_t loom_x86_source_signature_vector_semantic_type(
    uint32_t byte_length) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                             loom_dim_pack_static(byte_length), 0);
}

static iree_status_t loom_x86_source_signature_append_carrier(
    loom_x86_source_signature_plan_builder_t* builder, bool argument,
    uint16_t object_index, uint32_t byte_offset, uint16_t register_class,
    loom_type_t semantic_type, int64_t register_location,
    int64_t stack_offset) {
  loom_type_t physical_type;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      builder->context, register_class, 1, &physical_type));
  const uint32_t register_bytes =
      loom_x86_call_abi_register_byte_length(register_class);
  IREE_ASSERT_NE(register_bytes, 0);
  if (argument) {
    const uint16_t index = loom_low_c_abi_projection_append_argument_carrier(
        builder->projection, object_index, byte_offset, register_bytes);
    IREE_ASSERT_EQ(index, builder->argument_count);
    builder->argument_types[index] = physical_type;
    builder->semantic_argument_types[index] = semantic_type;
    builder->argument_registers[index] = register_location;
    builder->argument_offsets[index] = stack_offset;
    ++builder->argument_count;
  } else {
    const uint16_t index = loom_low_c_abi_projection_append_result_carrier(
        builder->projection, object_index, byte_offset, register_bytes);
    IREE_ASSERT_EQ(index, builder->result_count);
    builder->result_types[index] = physical_type;
    builder->semantic_result_types[index] = semantic_type;
    builder->result_registers[index] = register_location;
    ++builder->result_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_signature_append_direct_object(
    loom_x86_source_signature_plan_builder_t* builder, bool argument,
    uint16_t object_index,
    const loom_x86_source_signature_classification_t* classification) {
  for (uint8_t i = 0; i < classification->eightbyte_count; ++i) {
    const loom_x86_source_signature_class_t abi_class =
        classification->eightbytes[i];
    if (abi_class == LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS ||
        abi_class == LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP) {
      continue;
    }
    const uint32_t byte_offset = (uint32_t)i * 8u;
    if (abi_class == LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER) {
      const uint8_t bank_index = argument ? builder->integer_argument_count++
                                          : builder->integer_result_count++;
      const int64_t location =
          argument ? kLoomX86SourceSignatureIntegerArgumentRegisters[bank_index]
                   : kLoomX86SourceSignatureIntegerResultRegisters[bank_index];
      IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_carrier(
          builder, argument, object_index, byte_offset,
          LOOM_X86_REGISTER_CLASS_GPR64, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
          location,
          /*stack_offset=*/-1));
      continue;
    }
    IREE_ASSERT_EQ(abi_class, LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE);
    uint8_t run_count = 1;
    while (i + run_count < classification->eightbyte_count &&
           classification->eightbytes[i + run_count] ==
               LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP) {
      ++run_count;
    }
    const uint32_t run_bytes = (uint32_t)run_count * 8u;
    uint16_t register_class = LOOM_X86_REGISTER_CLASS_XMM;
    if (run_bytes == 32) {
      register_class = LOOM_X86_REGISTER_CLASS_YMM;
    } else if (run_bytes == 64) {
      register_class = LOOM_X86_REGISTER_CLASS_ZMM;
    }
    const uint32_t remaining_bytes = classification->byte_length - byte_offset;
    const loom_type_t semantic_type =
        run_count > 1
            ? loom_x86_source_signature_vector_semantic_type(run_bytes)
            : loom_type_scalar(remaining_bytes <= 4 ? LOOM_SCALAR_TYPE_F32
                                                    : LOOM_SCALAR_TYPE_F64);
    const uint8_t bank_index =
        argument ? builder->sse_argument_count++ : builder->sse_result_count++;
    IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_carrier(
        builder, argument, object_index, byte_offset, register_class,
        semantic_type, bank_index, /*stack_offset=*/-1));
    i += run_count - 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_signature_append_memory_argument(
    loom_x86_source_signature_plan_builder_t* builder, uint16_t object_index,
    const loom_x86_source_signature_classification_t* classification) {
  const uint32_t alignment = iree_max(8u, classification->byte_alignment);
  if (alignment > UINT8_MAX) {
    return loom_x86_source_signature_unsupported(
        builder->context,
        IREE_SV("x86 source signature stack alignment exceeds 255 bytes"));
  }
  const uint64_t aligned_offset =
      iree_host_align(builder->stack_bytes, alignment);
  const uint64_t object_bytes =
      iree_host_align(classification->byte_length, 8u);
  if (aligned_offset + object_bytes > UINT32_MAX) {
    return loom_x86_source_signature_unsupported(
        builder->context,
        IREE_SV("x86 source signature stack argument area exceeds 32 bits"));
  }
  const uint16_t carrier_count =
      (uint16_t)((classification->byte_length + 7u) / 8u);
  for (uint16_t i = 0; i < carrier_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_carrier(
        builder, /*argument=*/true, object_index, (uint32_t)i * 8u,
        LOOM_X86_REGISTER_CLASS_GPR64, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        /*register_location=*/-1,
        (int64_t)(aligned_offset + (uint64_t)i * 8u)));
  }
  builder->stack_bytes = (uint32_t)(aligned_offset + object_bytes);
  builder->stack_alignment = (uint8_t)iree_max(
      (uint32_t)builder->stack_alignment, iree_max(16u, alignment));
  return iree_ok_status();
}

static iree_status_t loom_x86_source_signature_allocate_physical_arrays(
    loom_x86_source_signature_plan_builder_t* builder,
    uint16_t argument_capacity, uint16_t result_capacity) {
  iree_arena_allocator_t* arena = builder->arena;
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_types),
      (void**)&builder->argument_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, result_capacity, sizeof(*builder->result_types),
      (void**)&builder->result_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, argument_capacity, sizeof(*builder->semantic_argument_types),
      (void**)&builder->semantic_argument_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, result_capacity, sizeof(*builder->semantic_result_types),
      (void**)&builder->semantic_result_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_registers),
      (void**)&builder->argument_registers));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_offsets),
      (void**)&builder->argument_offsets));
  return loom_x86_source_signature_allocate_array(
      arena, result_capacity, sizeof(*builder->result_registers),
      (void**)&builder->result_registers);
}

static void loom_x86_source_signature_projection_target_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_x86_entry_descriptors_t* entry,
    loom_low_c_abi_move_descriptor_t moves[4], uint16_t* out_move_count,
    loom_low_c_abi_vector_memory_descriptor_t vector_memory[3],
    uint16_t* out_vector_memory_count,
    loom_low_c_abi_projection_target_t* out_target) {
  *out_move_count = 0;
  if (entry->maximum_simd_byte_length >= 16) {
    moves[(*out_move_count)++] = (loom_low_c_abi_move_descriptor_t){
        .source_register_class = LOOM_X86_REGISTER_CLASS_XMM,
        .result_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
        .descriptor = entry->move_xmm_u32,
    };
    moves[(*out_move_count)++] = (loom_low_c_abi_move_descriptor_t){
        .source_register_class = LOOM_X86_REGISTER_CLASS_XMM,
        .result_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
        .descriptor = entry->move_xmm_u64,
    };
    moves[(*out_move_count)++] = (loom_low_c_abi_move_descriptor_t){
        .source_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
        .result_register_class = LOOM_X86_REGISTER_CLASS_XMM,
        .descriptor = entry->move_u32_xmm,
    };
    moves[(*out_move_count)++] = (loom_low_c_abi_move_descriptor_t){
        .source_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
        .result_register_class = LOOM_X86_REGISTER_CLASS_XMM,
        .descriptor = entry->move_u64_xmm,
    };
  }
  *out_vector_memory_count = 0;
  if (entry->maximum_simd_byte_length >= 16) {
    vector_memory[(*out_vector_memory_count)++] =
        (loom_low_c_abi_vector_memory_descriptor_t){
            .register_class = LOOM_X86_REGISTER_CLASS_XMM,
            .register_byte_length = 16,
            .load_descriptor = entry->load_128_xmm,
            .store_descriptor = entry->store_128_xmm,
        };
  }
  if (entry->maximum_simd_byte_length >= 32) {
    vector_memory[(*out_vector_memory_count)++] =
        (loom_low_c_abi_vector_memory_descriptor_t){
            .register_class = LOOM_X86_REGISTER_CLASS_YMM,
            .register_byte_length = 32,
            .load_descriptor = entry->load_256_ymm,
            .store_descriptor = entry->store_256_ymm,
        };
  }
  if (entry->maximum_simd_byte_length >= 64) {
    vector_memory[(*out_vector_memory_count)++] =
        (loom_low_c_abi_vector_memory_descriptor_t){
            .register_class = LOOM_X86_REGISTER_CLASS_ZMM,
            .register_byte_length = 64,
            .load_descriptor = entry->load_512_zmm,
            .store_descriptor = entry->store_512_zmm,
        };
  }
  *out_target = (loom_low_c_abi_projection_target_t){
      .ops =
          {
              .register_descriptor_set_stable_id = descriptor_set->stable_id,
              .bits32_register_class = LOOM_X86_REGISTER_CLASS_GPR32,
              .bits64_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
              .address_register_class = LOOM_X86_REGISTER_CLASS_GPR64,
              .address_byte_length = 8,
              .byte_order = LOOM_LOW_C_ABI_BYTE_ORDER_LITTLE,
              .load_u8 = entry->load_u8,
              .load_u16 = entry->load_u16,
              .load_u32 = entry->load_u32,
              .load_u64 = entry->load_u64,
              .store_u8 = entry->store_u8,
              .store_u16 = entry->store_u16,
              .store_u32 = entry->store_u32,
              .store_u64 = entry->store_u64,
              .add_address = entry->add_u64,
              .mask_u32 = entry->mask_u32,
              .widen_u32 = entry->widen_u32,
              .truncate_u64 = entry->truncate_u64,
              .shift_left_u64 = entry->shift_left_u64,
              .shift_right_u64 = entry->shift_right_u64,
              .or_u64 = entry->or_u64,
              .constant_u64 = entry->constant_u64,
              .memory_offset_name = "disp32",
              .mask_name = "imm32",
              .shift_name = "shift",
              .constant_name = "imm64",
          },
      .moves = moves,
      .move_count = *out_move_count,
      .vector_memory = vector_memory,
      .vector_memory_count = *out_vector_memory_count,
  };
}

iree_status_t loom_x86_plan_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    const loom_c_abi_signature_layout_t* layout,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* result_arena,
    loom_low_lower_source_signature_target_plan_t* out_plan) {
  (void)user_data;
  *out_plan = (loom_low_lower_source_signature_target_plan_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_x86_entry_descriptors_t* entry =
      loom_x86_entry_descriptors(descriptor_set);
  if (!entry) {
    return loom_x86_source_signature_unsupported(
        context,
        IREE_SV("x86 representation has no source signature vocabulary"));
  }
  if (layout->signature.arguments.count > UINT16_MAX ||
      layout->signature.results.count > UINT16_MAX) {
    return loom_x86_source_signature_unsupported(
        context, IREE_SV("x86 source signature object count exceeds 65535"));
  }
  if (layout->signature.results.count > 1) {
    return loom_x86_source_signature_unsupported(
        context,
        IREE_SV("C source signature requires at most one result object"));
  }
  const uint16_t argument_object_count =
      (uint16_t)layout->signature.arguments.count;
  const uint16_t result_object_count =
      (uint16_t)layout->signature.results.count;

  loom_x86_source_signature_classification_t* argument_classifications = NULL;
  loom_x86_source_signature_classification_t* result_classifications = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_classify_objects(
      context, layout, layout->arguments, argument_object_count,
      layout->argument_leaves, entry->maximum_simd_byte_length, scratch_arena,
      &argument_classifications));
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_classify_objects(
      context, layout, layout->results, result_object_count,
      layout->result_leaves, entry->maximum_simd_byte_length, scratch_arena,
      &result_classifications));

  const bool has_indirect_result =
      result_object_count != 0 && result_classifications[0].is_memory;
  uint64_t argument_capacity = has_indirect_result ? 1u : 0u;
  for (uint16_t i = 0; i < argument_object_count; ++i) {
    argument_capacity +=
        ((uint64_t)argument_classifications[i].byte_length + 7u) / 8u;
  }
  uint64_t result_capacity = has_indirect_result ? 1u : 0u;
  if (result_object_count != 0 && !has_indirect_result) {
    result_capacity =
        ((uint64_t)result_classifications[0].byte_length + 7u) / 8u;
  }
  if (argument_capacity > UINT16_MAX || result_capacity > UINT16_MAX) {
    return loom_x86_source_signature_unsupported(
        context,
        IREE_SV("x86 source signature physical signature exceeds 65535"));
  }

  loom_low_c_abi_projection_plan_t* projection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_c_abi_projection_plan_allocate(
      layout, logical_argument_types, logical_argument_count,
      logical_result_types, logical_result_count, (uint16_t)argument_capacity,
      (uint16_t)result_capacity, result_arena, &projection));
  loom_x86_source_signature_plan_builder_t builder = {
      .context = context,
      .arena = result_arena,
      .projection = projection,
  };
  IREE_RETURN_IF_ERROR(loom_x86_source_signature_allocate_physical_arrays(
      &builder, (uint16_t)argument_capacity, (uint16_t)result_capacity));

  if (has_indirect_result) {
    IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_carrier(
        &builder, /*argument=*/true, LOOM_LOW_C_ABI_INDEX_NONE,
        /*byte_offset=*/0, LOOM_X86_REGISTER_CLASS_GPR64, loom_type_buffer(),
        kLoomX86SourceSignatureIntegerArgumentRegisters[0],
        /*stack_offset=*/-1));
    builder.integer_argument_count = 1;
  }

  for (uint16_t i = 0; i < argument_object_count; ++i) {
    const loom_x86_source_signature_classification_t* classification =
        &argument_classifications[i];
    uint8_t required_integer = 0;
    uint8_t required_sse = 0;
    loom_x86_source_signature_direct_carrier_count(
        classification, &required_integer, &required_sse);
    const bool registers_available =
        !classification->is_memory &&
        builder.integer_argument_count + required_integer <=
            LOOM_X86_SOURCE_SIGNATURE_INTEGER_ARGUMENT_REGISTER_COUNT &&
        builder.sse_argument_count + required_sse <=
            LOOM_X86_SOURCE_SIGNATURE_SSE_ARGUMENT_REGISTER_COUNT;
    if (registers_available) {
      IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_direct_object(
          &builder, /*argument=*/true, i, classification));
    } else {
      IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_memory_argument(
          &builder, i, classification));
    }
  }

  if (result_object_count != 0) {
    if (has_indirect_result) {
      IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_carrier(
          &builder, /*argument=*/false, LOOM_LOW_C_ABI_INDEX_NONE,
          /*byte_offset=*/0, LOOM_X86_REGISTER_CLASS_GPR64, loom_type_buffer(),
          kLoomX86SourceSignatureIntegerResultRegisters[0],
          /*stack_offset=*/-1));
      loom_low_c_abi_projection_set_indirect_result(
          projection, /*object_index=*/0, /*address_argument=*/0,
          /*echo_result=*/0);
    } else {
      uint8_t required_integer = 0;
      uint8_t required_sse = 0;
      loom_x86_source_signature_direct_carrier_count(
          &result_classifications[0], &required_integer, &required_sse);
      IREE_ASSERT_LE(required_integer,
                     LOOM_X86_SOURCE_SIGNATURE_RESULT_REGISTER_COUNT);
      IREE_ASSERT_LE(required_sse,
                     LOOM_X86_SOURCE_SIGNATURE_RESULT_REGISTER_COUNT);
      IREE_RETURN_IF_ERROR(loom_x86_source_signature_append_direct_object(
          &builder, /*argument=*/false, /*object_index=*/0,
          &result_classifications[0]));
    }
  }

  loom_low_c_abi_move_descriptor_t moves[4];
  loom_low_c_abi_vector_memory_descriptor_t vector_memory[3];
  uint16_t move_count = 0;
  uint16_t vector_memory_count = 0;
  loom_low_c_abi_projection_target_t projection_target;
  loom_x86_source_signature_projection_target_initialize(
      descriptor_set, entry, moves, &move_count, vector_memory,
      &vector_memory_count, &projection_target);
  iree_string_view_t unsupported_reason;
  if (!loom_low_c_abi_projection_plan_finalize(
          projection, &projection_target, builder.argument_types,
          builder.argument_count, builder.result_types, builder.result_count,
          &unsupported_reason)) {
    return loom_x86_source_signature_unsupported(context, unsupported_reason);
  }

  if (builder.stack_bytes != 0) {
    builder.stack_bytes =
        (uint32_t)iree_host_align(builder.stack_bytes, builder.stack_alignment);
  }
  loom_named_attr_slice_t abi_layout;
  IREE_RETURN_IF_ERROR(loom_x86_call_abi_layout_make(
      loom_low_lower_context_module(context), builder.semantic_argument_types,
      builder.argument_count, builder.semantic_result_types,
      builder.result_count, builder.argument_registers,
      builder.argument_offsets, builder.result_registers, builder.stack_bytes,
      builder.stack_alignment, &abi_layout));

  const loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  if (has_indirect_result) {
    loom_tied_result_t* tie = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(result_arena, sizeof(*tie), (void**)&tie));
    *tie = (loom_tied_result_t){
        .result_index = 0,
        .operand_index = 0,
        .has_type_change = false,
    };
    tied_results = tie;
    tied_result_count = 1;
  }
  *out_plan = (loom_low_lower_source_signature_target_plan_t){
      .argument_types = builder.argument_types,
      .argument_count = builder.argument_count,
      .result_types = builder.result_types,
      .result_count = builder.result_count,
      .tied_results = tied_results,
      .tied_result_count = tied_result_count,
      .abi_layout = abi_layout,
      .target_data = projection,
  };
  return iree_ok_status();
}

iree_status_t loom_x86_emit_source_signature(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_signature_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state) {
  return loom_low_c_abi_projection_emit_source_signature(
      user_data, context, phase, source_op, target_plan, input_values,
      input_count, output_values, output_count, state_arena, inout_state);
}
