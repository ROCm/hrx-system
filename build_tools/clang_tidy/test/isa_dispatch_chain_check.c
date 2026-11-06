// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef _Bool bool;

typedef struct iree_clang_tidy_dispatch_op_t {
  int kind;
} iree_clang_tidy_dispatch_op_t;

#define DEFINE_ISA(name) \
  static bool name(const iree_clang_tidy_dispatch_op_t* op) { return op; }

DEFINE_ISA(chain_first_isa)
DEFINE_ISA(chain_second_isa)
DEFINE_ISA(chain_third_isa)
DEFINE_ISA(chain_fourth_isa)
DEFINE_ISA(compound_first_isa)
DEFINE_ISA(compound_second_isa)
DEFINE_ISA(compound_third_isa)
DEFINE_ISA(compound_fourth_isa)
DEFINE_ISA(macro_first_isa)
DEFINE_ISA(macro_second_isa)
DEFINE_ISA(macro_third_isa)
DEFINE_ISA(macro_fourth_isa)
DEFINE_ISA(five_first_isa)
DEFINE_ISA(five_second_isa)
DEFINE_ISA(five_third_isa)
DEFINE_ISA(five_fourth_isa)
DEFINE_ISA(five_fifth_isa)
DEFINE_ISA(short_first_isa)
DEFINE_ISA(short_second_isa)
DEFINE_ISA(short_third_isa)
DEFINE_ISA(body_first_isa)
DEFINE_ISA(body_second_isa)
DEFINE_ISA(body_third_isa)
DEFINE_ISA(body_fourth_isa)
DEFINE_ISA(suppressed_first_isa)
DEFINE_ISA(suppressed_second_isa)
DEFINE_ISA(suppressed_third_isa)
DEFINE_ISA(suppressed_fourth_isa)

#define MACRO_PAIR_ISA(op) (macro_first_isa(op) || macro_second_isa(op))

void iree_clang_tidy_dispatch_c_chain(const iree_clang_tidy_dispatch_op_t* op) {
  if (chain_first_isa(op)) {
  } else if (chain_second_isa(op) || chain_third_isa(op)) {
  } else if (chain_fourth_isa(op)) {
  }
}

void iree_clang_tidy_dispatch_c_compound(
    const iree_clang_tidy_dispatch_op_t* op) {
  if (compound_first_isa(op) || compound_second_isa(op) ||
      compound_third_isa(op) || compound_fourth_isa(op)) {
  }
}

void iree_clang_tidy_dispatch_c_macro(const iree_clang_tidy_dispatch_op_t* op) {
  if (MACRO_PAIR_ISA(op) || macro_third_isa(op) || macro_fourth_isa(op)) {
  }
}

void iree_clang_tidy_dispatch_c_five(const iree_clang_tidy_dispatch_op_t* op) {
  if (five_first_isa(op) || five_second_isa(op) || five_third_isa(op) ||
      five_fourth_isa(op) || five_fifth_isa(op)) {
  }
}

void iree_clang_tidy_dispatch_c_short(const iree_clang_tidy_dispatch_op_t* op) {
  if (short_first_isa(op)) {
  } else if (short_second_isa(op)) {
  } else if (short_third_isa(op)) {
  }
}

void iree_clang_tidy_dispatch_c_body_calls(
    const iree_clang_tidy_dispatch_op_t* op) {
  if (body_first_isa(op)) {
    (void)body_second_isa(op);
    (void)body_third_isa(op);
    (void)body_fourth_isa(op);
  }
}

void iree_clang_tidy_dispatch_c_suppressed(
    const iree_clang_tidy_dispatch_op_t* op) {
  if (suppressed_first_isa(op) ||  // NOLINT(iree-isa-dispatch-chain)
      suppressed_second_isa(op) || suppressed_third_isa(op) ||
      suppressed_fourth_isa(op)) {
  }
}
