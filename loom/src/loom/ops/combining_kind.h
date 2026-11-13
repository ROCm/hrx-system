// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place

#ifndef LOOM_OPS_COMBINING_KIND_H_
#define LOOM_OPS_COMBINING_KIND_H_

// Algebraic combining operations for reductions and scans.
typedef enum loom_combining_kind_e {
  // Integer addition.
  LOOM_COMBINING_KIND_ADDI = 0,
  // Floating-point addition.
  LOOM_COMBINING_KIND_ADDF = 1,
  // Integer multiplication.
  LOOM_COMBINING_KIND_MULI = 2,
  // Floating-point multiplication.
  LOOM_COMBINING_KIND_MULF = 3,
  // Signed integer minimum.
  LOOM_COMBINING_KIND_MINSI = 4,
  // Signed integer maximum.
  LOOM_COMBINING_KIND_MAXSI = 5,
  // Unsigned integer minimum.
  LOOM_COMBINING_KIND_MINUI = 6,
  // Unsigned integer maximum.
  LOOM_COMBINING_KIND_MAXUI = 7,
  // Bitwise AND.
  LOOM_COMBINING_KIND_ANDI = 8,
  // Bitwise OR.
  LOOM_COMBINING_KIND_ORI = 9,
  // Bitwise XOR.
  LOOM_COMBINING_KIND_XORI = 10,
  // IEEE 754 floating-point minimum.
  LOOM_COMBINING_KIND_MINIMUMF = 11,
  // IEEE 754 floating-point maximum.
  LOOM_COMBINING_KIND_MAXIMUMF = 12,
  // C99 fmin-style floating-point minimum.
  LOOM_COMBINING_KIND_MINNUMF = 13,
  // C99 fmax-style floating-point maximum.
  LOOM_COMBINING_KIND_MAXNUMF = 14,
  LOOM_COMBINING_KIND_COUNT_ = 15,
} loom_combining_kind_t;

#endif  // LOOM_OPS_COMBINING_KIND_H_
