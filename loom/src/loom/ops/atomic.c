// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/atomic.h"

bool loom_atomic_kind_is_valid(uint8_t kind) {
  return kind < LOOM_ATOMIC_KIND_COUNT_;
}

bool loom_atomic_ordering_is_valid(uint8_t ordering) {
  return ordering < LOOM_ATOMIC_ORDERING_COUNT_;
}

bool loom_atomic_scope_is_valid(uint8_t scope) {
  return scope < LOOM_ATOMIC_SCOPE_COUNT_;
}

static bool loom_atomic_cmpxchg_failure_no_stronger_than_success(
    uint8_t success_ordering, uint8_t failure_ordering) {
  switch (success_ordering) {
    case LOOM_ATOMIC_ORDERING_RELAXED:
      return failure_ordering == LOOM_ATOMIC_ORDERING_RELAXED;
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
      return failure_ordering == LOOM_ATOMIC_ORDERING_RELAXED ||
             failure_ordering == LOOM_ATOMIC_ORDERING_ACQUIRE;
    case LOOM_ATOMIC_ORDERING_RELEASE:
      return failure_ordering == LOOM_ATOMIC_ORDERING_RELAXED;
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
      return failure_ordering == LOOM_ATOMIC_ORDERING_RELAXED ||
             failure_ordering == LOOM_ATOMIC_ORDERING_ACQUIRE;
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      return failure_ordering == LOOM_ATOMIC_ORDERING_RELAXED ||
             failure_ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
             failure_ordering == LOOM_ATOMIC_ORDERING_SEQ_CST;
    default:
      return true;
  }
}

loom_atomic_cmpxchg_ordering_error_t loom_atomic_cmpxchg_ordering_validate(
    uint8_t success_ordering, uint8_t failure_ordering) {
  if (!loom_atomic_ordering_is_valid(success_ordering) ||
      !loom_atomic_ordering_is_valid(failure_ordering)) {
    return LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE;
  }
  if (failure_ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
      failure_ordering == LOOM_ATOMIC_ORDERING_ACQ_REL) {
    return LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_RELEASE;
  }
  if (!loom_atomic_cmpxchg_failure_no_stronger_than_success(success_ordering,
                                                            failure_ordering)) {
    return LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_STRONGER;
  }
  return LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE;
}

iree_string_view_t loom_atomic_cmpxchg_ordering_error_attr_name(
    loom_atomic_cmpxchg_ordering_error_t error) {
  switch (error) {
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_RELEASE:
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_STRONGER:
      return IREE_SV("failure_ordering");
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE:
    default:
      return IREE_SV("");
  }
}

iree_string_view_t loom_atomic_cmpxchg_ordering_error_expected_constraint(
    loom_atomic_cmpxchg_ordering_error_t error) {
  switch (error) {
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_RELEASE:
      return IREE_SV("relaxed, acquire, or seq_cst failure ordering");
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_FAILURE_STRONGER:
      return IREE_SV("failure ordering no stronger than success ordering");
    case LOOM_ATOMIC_CMPXCHG_ORDERING_ERROR_NONE:
    default:
      return IREE_SV("");
  }
}

bool loom_atomic_kind_accepts_integer(uint8_t kind) {
  switch (kind) {
    case LOOM_ATOMIC_KIND_XCHGI:
    case LOOM_ATOMIC_KIND_ADDI:
    case LOOM_ATOMIC_KIND_SUBI:
    case LOOM_ATOMIC_KIND_ANDI:
    case LOOM_ATOMIC_KIND_ORI:
    case LOOM_ATOMIC_KIND_XORI:
    case LOOM_ATOMIC_KIND_MINSI:
    case LOOM_ATOMIC_KIND_MAXSI:
    case LOOM_ATOMIC_KIND_MINUI:
    case LOOM_ATOMIC_KIND_MAXUI:
      return true;
    default:
      return false;
  }
}

bool loom_atomic_kind_accepts_float(uint8_t kind) {
  switch (kind) {
    case LOOM_ATOMIC_KIND_XCHGF:
    case LOOM_ATOMIC_KIND_ADDF:
    case LOOM_ATOMIC_KIND_MINIMUMF:
    case LOOM_ATOMIC_KIND_MAXIMUMF:
    case LOOM_ATOMIC_KIND_MINNUMF:
    case LOOM_ATOMIC_KIND_MAXNUMF:
      return true;
    default:
      return false;
  }
}

bool loom_atomic_kind_is_exchange(uint8_t kind) {
  return kind == LOOM_ATOMIC_KIND_XCHGI || kind == LOOM_ATOMIC_KIND_XCHGF;
}
