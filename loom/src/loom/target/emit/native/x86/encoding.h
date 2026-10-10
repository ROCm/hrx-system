// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct x86-64 encoding of allocated instructions.
//
// Descriptor encoding_format_id selects either a scalar operand form below or
// a packed vector recipe marked by LOOM_X86_ENCODING_FORMAT_VECTOR. Python
// descriptor declarations supply all immutable opcode and recipe facts;
// encoding never interprets a mnemonic or resolves SSA values. Vector
// encoding_id fields hold opcode, map, mandatory prefix, W, a prefix/map
// extension, and vector length in that order from low to high bits. EVEX maps
// 5 and 6 use the two map-zero extension states; all other maps retain their
// architectural number and use the extension as the EVEX selector.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_ENCODING_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_ENCODING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_x86_encoding_form_e {
  LOOM_X86_ENCODING_FORM_NONE = 0,
  LOOM_X86_ENCODING_FORM_BINARY_RM_R = 1,
  LOOM_X86_ENCODING_FORM_BINARY_R_RM = 2,
  LOOM_X86_ENCODING_FORM_MULTIPLY_HIGH = 3,
  LOOM_X86_ENCODING_FORM_MULTIPLY_IMMEDIATE = 4,
  LOOM_X86_ENCODING_FORM_BINARY_IMMEDIATE = 5,
  LOOM_X86_ENCODING_FORM_SHIFT_IMMEDIATE = 6,
  LOOM_X86_ENCODING_FORM_SHIFT_COUNT = 7,
  LOOM_X86_ENCODING_FORM_MOVE = 8,
  LOOM_X86_ENCODING_FORM_SELECT = 9,
  LOOM_X86_ENCODING_FORM_SUBTRACT_IF_UGE = 10,
  LOOM_X86_ENCODING_FORM_COMPARE = 11,
  LOOM_X86_ENCODING_FORM_COMPARE_IMMEDIATE = 12,
  LOOM_X86_ENCODING_FORM_CONSTANT = 13,
  LOOM_X86_ENCODING_FORM_LOAD = 14,
  LOOM_X86_ENCODING_FORM_STORE = 15,
  LOOM_X86_ENCODING_FORM_ADDRESS_ADD = 16,
  LOOM_X86_ENCODING_FORM_ADDRESS_DISPLACEMENT = 17,
  LOOM_X86_ENCODING_FORM_ADDRESS_SCALE = 18,
  LOOM_X86_ENCODING_FORM_ADDRESS_ADD_SCALE = 19,
  LOOM_X86_ENCODING_FORM_JUMP = 20,
  // Structural control and ABI transport use the same encoder without an IR
  // descriptor. Conditional branches test the predicate at its register width.
  LOOM_X86_ENCODING_FORM_BRANCH_NONZERO = 21,
  LOOM_X86_ENCODING_FORM_PUSH = 22,
  LOOM_X86_ENCODING_FORM_POP = 23,
  LOOM_X86_ENCODING_FORM_RETURN = 24,
  LOOM_X86_ENCODING_FORM_BRANCH_ZERO = 25,
  LOOM_X86_ENCODING_FORM_CALL = 26,
  // RIP-relative address with a signed displacement in the last four bytes.
  LOOM_X86_ENCODING_FORM_ADDRESS_PC_RELATIVE = 27,
} loom_x86_encoding_form_t;

// Native fixup identities. Object adapters supply their format's mapping.
typedef enum loom_x86_relocation_kind_e {
  LOOM_X86_RELOCATION_NONE = 0,
  // Signed rel32 call displacement, relative to the end of its four-byte field.
  LOOM_X86_RELOCATION_CALL = 1,
  // Signed rel32 data address, relative to the end of its four-byte field.
  LOOM_X86_RELOCATION_ADDRESS = 2,
  // Absolute 64-bit image pointer, relocated by the loader's load bias.
  LOOM_X86_RELOCATION_POINTER = 3,
} loom_x86_relocation_kind_t;

enum loom_x86_encoding_format_bits_e {
  // Marks a packed VEX/EVEX operand recipe. Scalar forms leave this bit clear.
  LOOM_X86_ENCODING_FORMAT_VECTOR = 1u << 15,
};

// Packed vector recipes use four-bit selectors for ModRM.reg in bits 0..3,
// VEX.vvvv in bits 4..7, and ModRM.r/m in bits 8..11. Memory recipes select
// their base through ModRM.r/m; an optional index immediately follows the base
// in the concrete input array. Form-specific selector bit 3 modifiers encode
// EVEX zeroing in ModRM.reg, a full-vector displacement tuple in VEX.vvvv, and
// 32-bit embedded broadcast in ModRM.r/m. Bits 12..14 select the encoding
// behavior and bit 15 marks the recipe. Selectors name allocated results and
// inputs or the bounded ModRM opcode extensions used by immediate shift forms.
typedef enum loom_x86_vector_register_selector_e {
  LOOM_X86_VECTOR_REGISTER_RESULT = 0,
  LOOM_X86_VECTOR_REGISTER_INPUT_0 = 1,
  LOOM_X86_VECTOR_REGISTER_INPUT_1 = 2,
  LOOM_X86_VECTOR_REGISTER_INPUT_2 = 3,
  LOOM_X86_VECTOR_REGISTER_NONE = 4,
  LOOM_X86_VECTOR_REGISTER_FIXED_2 = 5,
  LOOM_X86_VECTOR_REGISTER_FIXED_3 = 6,
  LOOM_X86_VECTOR_REGISTER_FIXED_4 = 7,
  LOOM_X86_VECTOR_REGISTER_FIXED_6 = 8,
} loom_x86_vector_register_selector_t;

// Bits 12..14 of a packed vector recipe select bounded encoding behavior.
typedef enum loom_x86_vector_encoding_behavior_e {
  LOOM_X86_VECTOR_ENCODING_REGISTERS = 0,
  LOOM_X86_VECTOR_ENCODING_IMMEDIATE = 1,
  LOOM_X86_VECTOR_ENCODING_BLEND_MASK = 2,
  LOOM_X86_VECTOR_ENCODING_EVEX_MASK = 3,
  LOOM_X86_VECTOR_ENCODING_LOAD = 4,
  LOOM_X86_VECTOR_ENCODING_STORE = 5,
  LOOM_X86_VECTOR_ENCODING_RIP_LOAD = 6,
  LOOM_X86_VECTOR_ENCODING_EVEX_MASK_LOAD = 7,
} loom_x86_vector_encoding_behavior_t;

enum loom_x86_encoding_flag_bits_e {
  LOOM_X86_ENCODING_OPCODE_0F = 1u << 8,
  LOOM_X86_ENCODING_REX_W = 1u << 12,
  LOOM_X86_ENCODING_OPERAND_16 = 1u << 13,
  // The r/m register for register forms, or the value register for memory.
  LOOM_X86_ENCODING_BYTE = 1u << 14,
  LOOM_X86_ENCODING_INDEXED = 1u << 15,
};

// Native register numbers use architectural encoding order: RAX=0, RCX=1,
// RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7, and R8..R15=8..15. Width aliases
// share their number. Destructive operands have already been reconciled by
// shared allocation transport; the encoder does not insert repair moves.
typedef struct loom_x86_encoding_operands_t {
  // Immediate or memory displacement, interpreted by the selected form.
  int64_t immediate;
  // Architectural result register; unused for stores and control transfers.
  uint8_t result;
  // Architectural input registers in descriptor operand order.
  uint8_t inputs[5];
  // SIB scale exponent (0, 1, 2, or 3), not a byte multiplier.
  uint8_t scale;
  // True when the input immediately after a selected memory base is an index.
  bool has_index;
} loom_x86_encoding_operands_t;

static_assert(sizeof(loom_x86_encoding_operands_t) == 16,
              "x86 encoding operands must remain compact");

// A descriptor can expand into a bounded sequence (for example CMP, SETcc,
// MOVZX). The longest supported sequence fits in 32 bytes. Preparation appends
// concrete encoding inputs before any bytes are emitted.
typedef struct loom_x86_encoded_instruction_t {
  // Native instruction sequence, with no references to compiler storage.
  uint8_t bytes[32];
  // Number of live bytes in |bytes|.
  uint8_t length;
} loom_x86_encoded_instruction_t;

// Returns architectural GPR writes, including implicit writes, for a concrete
// instruction. Instruction preparation retains their union before frame layout;
// byte encoding has no frame or preservation policy. Width aliases share bits.
uint16_t loom_x86_encoding_gpr_writes(
    loom_x86_encoding_form_t form,
    const loom_x86_encoding_operands_t* operands);

// Encodes one supported form with verified, allocated operands. Immediate
// ranges, legal register classes, and destructive ties are producer invariants.
// The caller rejects descriptors with FORM_NONE at the target-support boundary.
void loom_x86_encode_instruction(
    uint16_t encoding_format_id, uint16_t encoding_id,
    const loom_x86_encoding_operands_t* operands,
    loom_x86_encoded_instruction_t* out_instruction);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_ENCODING_H_
