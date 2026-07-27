// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scalar/combining.h"

#include "loom/ops/scalar/ops.h"

bool loom_scalar_combining_kind_op(loom_combining_kind_t kind,
                                   loom_op_kind_t* out_op_kind) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      *out_op_kind = LOOM_OP_SCALAR_ADDI;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
      *out_op_kind = LOOM_OP_SCALAR_ADDF;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_op_kind = LOOM_OP_SCALAR_MULI;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out_op_kind = LOOM_OP_SCALAR_MULF;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      *out_op_kind = LOOM_OP_SCALAR_MINSI;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      *out_op_kind = LOOM_OP_SCALAR_MAXSI;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      *out_op_kind = LOOM_OP_SCALAR_MINUI;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      *out_op_kind = LOOM_OP_SCALAR_MAXUI;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_op_kind = LOOM_OP_SCALAR_ANDI;
      return true;
    case LOOM_COMBINING_KIND_ORI:
      *out_op_kind = LOOM_OP_SCALAR_ORI;
      return true;
    case LOOM_COMBINING_KIND_XORI:
      *out_op_kind = LOOM_OP_SCALAR_XORI;
      return true;
    case LOOM_COMBINING_KIND_MINIMUMF:
      *out_op_kind = LOOM_OP_SCALAR_MINIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXIMUMF:
      *out_op_kind = LOOM_OP_SCALAR_MAXIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      *out_op_kind = LOOM_OP_SCALAR_MINNUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out_op_kind = LOOM_OP_SCALAR_MAXNUMF;
      return true;
    case LOOM_COMBINING_KIND_COUNT_:
      break;
  }
  *out_op_kind = LOOM_OP_KIND_UNKNOWN;
  return false;
}
