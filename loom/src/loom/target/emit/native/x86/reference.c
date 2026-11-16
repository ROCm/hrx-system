// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/reference.h"

#include "iree/vm/ref.h"
#include "loom/target/arch/x86/register_classes.h"

static_assert(IREE_VM_REF_OBJECT_SIZE == 4,
              "native owner adjustments address the 32-bit object prefix");
static_assert(offsetof(iree_vm_ref_type_descriptor_t, destroy) == 0,
              "native final release loads the descriptor's first pointer");

uint32_t loom_x86_reference_label(loom_x86_function_t* function) {
  return function->block_count + 1 + function->local_label_count++;
}

void loom_x86_reference_bind(loom_x86_function_t* function, uint32_t label) {
  loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_LABEL, 0,
                           (loom_x86_encoding_operands_t){0}, label);
}

static void loom_x86_reference_branch(loom_x86_function_t* function,
                                      uint16_t form, uint16_t encoding,
                                      uint8_t input, uint32_t label) {
  loom_x86_function_append(function, form, encoding,
                           (loom_x86_encoding_operands_t){.inputs = {input}},
                           label);
  ++function->local_fixup_count;
}

static void loom_x86_reference_word(loom_x86_function_t* function, bool store,
                                    uint8_t reg,
                                    loom_x86_reference_address_t address) {
  loom_x86_function_append(
      function,
      store ? LOOM_X86_ENCODING_FORM_STORE : LOOM_X86_ENCODING_FORM_LOAD,
      (store ? 0x89 : 0x8b) | LOOM_X86_ENCODING_REX_W,
      (loom_x86_encoding_operands_t){
          .immediate = address.displacement,
          .result = reg,
          .inputs = {store ? reg : address.base, store ? address.base : 0}},
      UINT32_MAX);
}

static void loom_x86_reference_pair(loom_x86_function_t* function, bool store,
                                    loom_x86_reference_address_t address) {
  loom_x86_reference_word(function, store, 10, address);
  address.displacement += 8;
  loom_x86_reference_word(function, store, 11, address);
}

static void loom_x86_reference_scratch(
    loom_x86_function_t* function, const loom_x86_reference_context_t* context,
    bool store) {
  loom_x86_reference_pair(
      function, store,
      (loom_x86_reference_address_t){.base = 4,
                                     .displacement = context->scratch_offset});
}

void loom_x86_reference_initialize(loom_x86_function_t* function,
                                   loom_x86_reference_address_t address) {
  for (uint32_t word = 0; word < 2; ++word) {
    loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_ZERO_MEMORY,
                             LOOM_X86_ENCODING_REX_W,
                             (loom_x86_encoding_operands_t){
                                 .immediate = address.displacement + 8 * word,
                                 .inputs = {address.base}},
                             UINT32_MAX);
  }
}

// R10/R11 contain an old cell already detached from its storage. The caller
// preserves those scratch registers; every other register survives this path.
static void loom_x86_reference_release(
    loom_x86_function_t* function,
    const loom_x86_reference_context_t* context) {
  const uint32_t done = loom_x86_reference_label(function);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_BRANCH_ZERO,
                            LOOM_X86_ENCODING_REX_W, 10, done);
  // All borrowed states have bit zero set. TEST does not destroy the
  // descriptor.
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_TEST_IMMEDIATE_BRANCH,
      0xf7 | LOOM_X86_ENCODING_REX_W,
      (loom_x86_encoding_operands_t){.immediate = IREE_VM_REF_STATE_BORROWED,
                                     .inputs = {11},
                                     .secondary = 5},
      done);
  ++function->local_fixup_count;
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_LOCK_UNARY_MEMORY, 1u << 9,
      (loom_x86_encoding_operands_t){.inputs = {10}}, UINT32_MAX);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_BRANCH_FLAGS, 5, 0,
                            done);
  loom_x86_reference_word(function, false, 11,
                          (loom_x86_reference_address_t){.base = 11});
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_BRANCH_ZERO,
                            LOOM_X86_ENCODING_REX_W, 11, done);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_CALL_LOCAL, 0, 0,
                            context->destroy_label);
  loom_x86_reference_bind(function, done);
}

void loom_x86_reference_discard(loom_x86_function_t* function,
                                const loom_x86_reference_context_t* context,
                                loom_x86_reference_address_t address) {
  loom_x86_reference_scratch(function, context, true);
  loom_x86_reference_pair(function, false, address);
  loom_x86_reference_initialize(function, address);
  loom_x86_reference_release(function, context);
  loom_x86_reference_scratch(function, context, false);
}

void loom_x86_reference_transfer(loom_x86_function_t* function,
                                 const loom_x86_reference_context_t* context,
                                 loom_x86_reference_transfer_t transfer,
                                 loom_x86_reference_address_t destination,
                                 loom_x86_reference_address_t source) {
  if (source.base == destination.base &&
      source.displacement == destination.displacement) {
    return;
  }
  const loom_x86_reference_address_t payload = {
      .base = 4, .displacement = context->scratch_offset + 16};
  loom_x86_reference_scratch(function, context, true);
  loom_x86_reference_pair(function, false, source);
  if (transfer == LOOM_X86_REFERENCE_COPY ||
      transfer == LOOM_X86_REFERENCE_BORROW ||
      transfer == LOOM_X86_REFERENCE_RETAIN) {
    const uint32_t ready = loom_x86_reference_label(function);
    loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_BRANCH_ZERO,
                              LOOM_X86_ENCODING_REX_W, 10, ready);
    if (transfer == LOOM_X86_REFERENCE_BORROW) {
      loom_x86_function_append(
          function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
          0x81 | (1u << 9) | LOOM_X86_ENCODING_REX_W,
          (loom_x86_encoding_operands_t){
              .result = 11, .immediate = IREE_VM_REF_STATE_BORROWED},
          UINT32_MAX);
    } else {
      if (transfer == LOOM_X86_REFERENCE_COPY) {
        loom_x86_function_append(function,
                                 LOOM_X86_ENCODING_FORM_TEST_IMMEDIATE_BRANCH,
                                 0xf7 | LOOM_X86_ENCODING_REX_W,
                                 (loom_x86_encoding_operands_t){
                                     .immediate = IREE_VM_REF_STATE_BORROWED,
                                     .inputs = {11},
                                     .secondary = 5},
                                 ready);
        ++function->local_fixup_count;
      } else {
        loom_x86_function_append(
            function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
            0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
            (loom_x86_encoding_operands_t){
                .result = 11, .immediate = ~IREE_VM_REF_STATE_MASK},
            UINT32_MAX);
      }
      loom_x86_function_append(
          function, LOOM_X86_ENCODING_FORM_LOCK_UNARY_MEMORY, 0,
          (loom_x86_encoding_operands_t){.inputs = {10}}, UINT32_MAX);
    }
    loom_x86_reference_bind(function, ready);
  }
  loom_x86_reference_pair(function, true, payload);
  if (transfer == LOOM_X86_REFERENCE_TRANSFER) {
    loom_x86_reference_initialize(function, source);
  }
  loom_x86_reference_pair(function, false, destination);
  if (transfer == LOOM_X86_REFERENCE_EXCHANGE) {
    loom_x86_reference_pair(function, true, source);
  } else {
    // Detach before invoking native destruction, which may reenter the caller.
    loom_x86_reference_initialize(function, destination);
    loom_x86_reference_release(function, context);
  }
  loom_x86_reference_pair(function, false, payload);
  loom_x86_reference_pair(function, true, destination);
  loom_x86_reference_scratch(function, context, false);
}

void loom_x86_reference_prepare_return(
    loom_x86_function_t* function, const loom_x86_reference_context_t* context,
    loom_x86_reference_address_t address, bool consume,
    uint32_t failure_label) {
  const uint32_t ready = loom_x86_reference_label(function);
  const uint32_t rejected = loom_x86_reference_label(function);
  const uint32_t done = loom_x86_reference_label(function);
  loom_x86_reference_scratch(function, context, true);
  loom_x86_reference_pair(function, false, address);
  // Null and owned cells have no borrow bit. Borrowed-only has bit one set.
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_TEST_IMMEDIATE_BRANCH,
      LOOM_X86_ENCODING_REX_W,
      (loom_x86_encoding_operands_t){.immediate = IREE_VM_REF_STATE_BORROWED,
                                     .inputs = {11},
                                     .secondary = 4},
      ready);
  ++function->local_fixup_count;
  loom_x86_function_append(function,
                           LOOM_X86_ENCODING_FORM_TEST_IMMEDIATE_BRANCH,
                           LOOM_X86_ENCODING_REX_W,
                           (loom_x86_encoding_operands_t){
                               .immediate = IREE_VM_REF_STATE_BORROWED_ONLY ^
                                            IREE_VM_REF_STATE_BORROWED,
                               .inputs = {11},
                               .secondary = 5},
                           rejected);
  ++function->local_fixup_count;
  if (consume) {
    loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_LOCK_UNARY_MEMORY,
                             0, (loom_x86_encoding_operands_t){.inputs = {10}},
                             UINT32_MAX);
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (4u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.result = 11,
                                       .immediate = ~IREE_VM_REF_STATE_MASK},
        UINT32_MAX);
    loom_x86_reference_pair(function, true, address);
  }
  loom_x86_reference_bind(function, ready);
  loom_x86_reference_scratch(function, context, false);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_JUMP, 0, 0, done);
  loom_x86_reference_bind(function, rejected);
  loom_x86_reference_scratch(function, context, false);
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_CONSTANT, 0xb8,
      (loom_x86_encoding_operands_t){
          .result = 0, .immediate = IREE_STATUS_FAILED_PRECONDITION},
      UINT32_MAX);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_JUMP, 0, 0,
                            failure_label);
  loom_x86_reference_bind(function, done);
}

void loom_x86_reference_compare(loom_x86_function_t* function,
                                const loom_x86_reference_context_t* context,
                                uint8_t result,
                                loom_x86_reference_address_t left,
                                const loom_x86_reference_address_t* right) {
  loom_x86_reference_scratch(function, context, true);
  loom_x86_reference_word(function, false, 10, left);
  if (right) {
    loom_x86_reference_word(function, false, 11, *right);
  }
  loom_x86_function_append(
      function,
      right ? LOOM_X86_ENCODING_FORM_COMPARE
            : LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE,
      4 | LOOM_X86_ENCODING_REX_W,
      (loom_x86_encoding_operands_t){.result = result, .inputs = {10, 11}},
      UINT32_MAX);
  const loom_x86_reference_address_t payload = {
      .base = 4, .displacement = context->scratch_offset + 16};
  loom_x86_reference_word(function, true, result, payload);
  loom_x86_reference_scratch(function, context, false);
  loom_x86_reference_word(function, false, result, payload);
}

void loom_x86_reference_select(loom_x86_function_t* function,
                               const loom_x86_reference_context_t* context,
                               uint8_t condition,
                               loom_x86_reference_address_t destination,
                               loom_x86_reference_address_t true_value,
                               loom_x86_reference_address_t false_value) {
  const uint32_t false_label = loom_x86_reference_label(function);
  const uint32_t done = loom_x86_reference_label(function);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_BRANCH_ZERO, 0,
                            condition, false_label);
  loom_x86_reference_transfer(function, context, LOOM_X86_REFERENCE_COPY,
                              destination, true_value);
  loom_x86_reference_branch(function, LOOM_X86_ENCODING_FORM_JUMP, 0, 0, done);
  loom_x86_reference_bind(function, false_label);
  loom_x86_reference_transfer(function, context, LOOM_X86_REFERENCE_COPY,
                              destination, false_value);
  loom_x86_reference_bind(function, done);
}

void loom_x86_reference_destroy_thunk(
    loom_x86_function_t* function, const loom_x86_reference_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_table_t* allocation) {
  // The local CALL puts RSP at eight modulo sixteen. Nine saved GPRs restore
  // call alignment; every SIMD/mask area is rounded to sixteen bytes.
  static const uint8_t registers[] = {0, 1, 2, 6, 7, 8, 9, 10, 11};
  const loom_low_call_clobber_list_t clobbers =
      loom_x86_function_common_call_clobbers(NULL, descriptor_set);
  // Physical extents already include allocated values and transport scratch.
  // SIMD classes alias the same register indices, so preserve their union at
  // the profile's widest width. Scalar orchestration needs no SIMD saves.
  uint32_t vector_count = 0;
  for (uint16_t register_class = LOOM_X86_REGISTER_CLASS_XMM;
       register_class <= LOOM_X86_REGISTER_CLASS_ZMM &&
       register_class < allocation->physical_extents.count;
       ++register_class) {
    vector_count = iree_max(
        vector_count,
        allocation->physical_extents.ends_by_reg_class[register_class]);
  }
  const uint32_t mask_count =
      LOOM_X86_REGISTER_CLASS_K < allocation->physical_extents.count
          ? allocation->physical_extents
                .ends_by_reg_class[LOOM_X86_REGISTER_CLASS_K]
          : 0;
  loom_low_call_clobber_t groups[2];
  uint32_t vector_bytes = 0;
  for (iree_host_size_t i = 0; i < clobbers.count; ++i) {
    groups[i] = clobbers.values[i];
    groups[i].count = groups[i].register_class == LOOM_X86_REGISTER_CLASS_K
                          ? mask_count
                          : vector_count;
    vector_bytes += groups[i].count *
                    loom_x86_transport_byte_length(groups[i].register_class);
  }
  vector_bytes = iree_host_align(vector_bytes, 16);
  loom_x86_reference_bind(function, context->destroy_label);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(registers); ++i) {
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_PUSH, 0,
        (loom_x86_encoding_operands_t){.inputs = {registers[i]}}, UINT32_MAX);
  }
  if (vector_bytes) {
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | (5u << 9) | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.result = 4, .immediate = vector_bytes},
        UINT32_MAX);
  }
  uint32_t offset = 0;
  for (iree_host_size_t i = 0; i < clobbers.count; ++i) {
    const loom_low_call_clobber_t* group = &groups[i];
    for (uint32_t j = 0; j < group->count; ++j) {
      loom_x86_transport_instruction_t instruction;
      loom_x86_transport_select_storage(
          LOOM_X86_STORAGE_TRANSFER_STORE, group->register_class,
          group->location + j, 4, offset, &instruction);
      loom_x86_function_append_transport(function, &instruction);
      offset += loom_x86_transport_byte_length(group->register_class);
    }
  }
  loom_x86_function_append(
      function, LOOM_X86_ENCODING_FORM_UNARY_R_RM,
      0x8b | LOOM_X86_ENCODING_REX_W,
      (loom_x86_encoding_operands_t){.result = 7, .inputs = {10}}, UINT32_MAX);
  loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_CALL_INDIRECT, 0,
                           (loom_x86_encoding_operands_t){.inputs = {11}},
                           UINT32_MAX);
  offset = 0;
  for (iree_host_size_t i = 0; i < clobbers.count; ++i) {
    const loom_low_call_clobber_t* group = &groups[i];
    for (uint32_t j = 0; j < group->count; ++j) {
      loom_x86_transport_instruction_t instruction;
      loom_x86_transport_select_storage(
          LOOM_X86_STORAGE_TRANSFER_LOAD, group->register_class,
          group->location + j, 4, offset, &instruction);
      loom_x86_function_append_transport(function, &instruction);
      offset += loom_x86_transport_byte_length(group->register_class);
    }
  }
  if (vector_bytes) {
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE,
        0x81 | LOOM_X86_ENCODING_REX_W,
        (loom_x86_encoding_operands_t){.result = 4, .immediate = vector_bytes},
        UINT32_MAX);
  }
  for (iree_host_size_t i = IREE_ARRAYSIZE(registers); i > 0; --i) {
    loom_x86_function_append(
        function, LOOM_X86_ENCODING_FORM_POP, 0,
        (loom_x86_encoding_operands_t){.inputs = {registers[i - 1]}},
        UINT32_MAX);
  }
  loom_x86_function_append(function, LOOM_X86_ENCODING_FORM_RETURN, 0,
                           (loom_x86_encoding_operands_t){0}, UINT32_MAX);
}
