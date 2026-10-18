// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/combining.h"

#include <math.h>

typedef enum loom_combining_identity_e {
  LOOM_COMBINING_IDENTITY_NONE = 0,
  LOOM_COMBINING_IDENTITY_ZERO,
  LOOM_COMBINING_IDENTITY_ONE,
  LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM,
  LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM,
  LOOM_COMBINING_IDENTITY_ALL_ONES,
  LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO,
  LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY,
  LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY,
} loom_combining_identity_t;

enum loom_combining_property_flag_bits_e {
  LOOM_COMBINING_PROPERTY_FLAG_INTEGER = 1u << 0,
  LOOM_COMBINING_PROPERTY_FLAG_FLOAT = 1u << 1,
  LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT = 1u << 2,
};
typedef uint8_t loom_combining_property_flags_t;

typedef struct loom_combining_descriptor_t {
  // Accepted operand family and algebraic properties.
  loom_combining_property_flags_t flags;
  // Algebraic identity used when padding reductions.
  uint8_t identity;
  // Empty-prefix seed used by exclusive scans.
  uint8_t scan_seed;
} loom_combining_descriptor_t;
static_assert(sizeof(loom_combining_descriptor_t) == 3,
              "combining descriptors must stay compact");

// One row owns the algebraic properties of each public combining kind. Keep
// selection and materialization keyed by this table so extending the family
// cannot silently omit an identity or idempotence classification.
static const loom_combining_descriptor_t kCombiningDescriptors[] = {
    [LOOM_COMBINING_KIND_ADDI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER,
                                  LOOM_COMBINING_IDENTITY_ZERO,
                                  LOOM_COMBINING_IDENTITY_ZERO},
    [LOOM_COMBINING_KIND_ADDF] = {LOOM_COMBINING_PROPERTY_FLAG_FLOAT,
                                  LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO,
                                  LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO},
    [LOOM_COMBINING_KIND_MULI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER,
                                  LOOM_COMBINING_IDENTITY_ONE,
                                  LOOM_COMBINING_IDENTITY_ONE},
    [LOOM_COMBINING_KIND_MULF] = {LOOM_COMBINING_PROPERTY_FLAG_FLOAT,
                                  LOOM_COMBINING_IDENTITY_ONE,
                                  LOOM_COMBINING_IDENTITY_ONE},
    [LOOM_COMBINING_KIND_MINSI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                       LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                   LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM,
                                   LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM},
    [LOOM_COMBINING_KIND_MAXSI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                       LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                   LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM,
                                   LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM},
    [LOOM_COMBINING_KIND_MINUI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                       LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                   LOOM_COMBINING_IDENTITY_ALL_ONES,
                                   LOOM_COMBINING_IDENTITY_ALL_ONES},
    [LOOM_COMBINING_KIND_MAXUI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                       LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                   LOOM_COMBINING_IDENTITY_ZERO,
                                   LOOM_COMBINING_IDENTITY_ZERO},
    [LOOM_COMBINING_KIND_ANDI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                      LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                  LOOM_COMBINING_IDENTITY_ALL_ONES,
                                  LOOM_COMBINING_IDENTITY_ALL_ONES},
    [LOOM_COMBINING_KIND_ORI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER |
                                     LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
                                 LOOM_COMBINING_IDENTITY_ZERO,
                                 LOOM_COMBINING_IDENTITY_ZERO},
    [LOOM_COMBINING_KIND_XORI] = {LOOM_COMBINING_PROPERTY_FLAG_INTEGER,
                                  LOOM_COMBINING_IDENTITY_ZERO,
                                  LOOM_COMBINING_IDENTITY_ZERO},
    [LOOM_COMBINING_KIND_MINIMUMF] =
        {LOOM_COMBINING_PROPERTY_FLAG_FLOAT |
             LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
         LOOM_COMBINING_IDENTITY_NONE,
         LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY},
    [LOOM_COMBINING_KIND_MAXIMUMF] =
        {LOOM_COMBINING_PROPERTY_FLAG_FLOAT |
             LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
         LOOM_COMBINING_IDENTITY_NONE,
         LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY},
    [LOOM_COMBINING_KIND_MINNUMF] =
        {LOOM_COMBINING_PROPERTY_FLAG_FLOAT |
             LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
         LOOM_COMBINING_IDENTITY_NONE,
         LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY},
    [LOOM_COMBINING_KIND_MAXNUMF] =
        {LOOM_COMBINING_PROPERTY_FLAG_FLOAT |
             LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT,
         LOOM_COMBINING_IDENTITY_NONE,
         LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY},
};
static_assert(IREE_ARRAYSIZE(kCombiningDescriptors) ==
                  LOOM_COMBINING_KIND_COUNT_,
              "combining descriptor table must cover every kind");

static const loom_combining_descriptor_t* loom_combining_descriptor(
    loom_combining_kind_t kind) {
  return loom_combining_kind_is_valid(kind) ? &kCombiningDescriptors[kind]
                                            : NULL;
}

bool loom_combining_kind_accepts_integer(loom_combining_kind_t kind) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  return descriptor && iree_any_bit_set(descriptor->flags,
                                        LOOM_COMBINING_PROPERTY_FLAG_INTEGER);
}

bool loom_combining_kind_accepts_float(loom_combining_kind_t kind) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  return descriptor && iree_any_bit_set(descriptor->flags,
                                        LOOM_COMBINING_PROPERTY_FLAG_FLOAT);
}

bool loom_combining_kind_is_idempotent(loom_combining_kind_t kind) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  return descriptor &&
         iree_any_bit_set(descriptor->flags,
                          LOOM_COMBINING_PROPERTY_FLAG_IDEMPOTENT);
}

bool loom_combining_kind_i1_equivalent(loom_combining_kind_t kind,
                                       loom_combining_kind_t* out_kind) {
  if (out_kind == NULL) {
    return false;
  }
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_XORI:
      *out_kind = LOOM_COMBINING_KIND_XORI;
      return true;
    case LOOM_COMBINING_KIND_MULI:
    case LOOM_COMBINING_KIND_MAXSI:
    case LOOM_COMBINING_KIND_MINUI:
    case LOOM_COMBINING_KIND_ANDI:
      *out_kind = LOOM_COMBINING_KIND_ANDI;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
    case LOOM_COMBINING_KIND_MAXUI:
    case LOOM_COMBINING_KIND_ORI:
      *out_kind = LOOM_COMBINING_KIND_ORI;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
    case LOOM_COMBINING_KIND_MULF:
    case LOOM_COMBINING_KIND_MINIMUMF:
    case LOOM_COMBINING_KIND_MAXIMUMF:
    case LOOM_COMBINING_KIND_MINNUMF:
    case LOOM_COMBINING_KIND_MAXNUMF:
    case LOOM_COMBINING_KIND_COUNT_:
      return false;
  }
  return false;
}

bool loom_combining_kind_integer_identity(loom_combining_kind_t kind,
                                          loom_scalar_type_t element_type,
                                          int64_t* out_identity) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  if (!out_identity || descriptor == NULL ||
      !iree_any_bit_set(descriptor->flags,
                        LOOM_COMBINING_PROPERTY_FLAG_INTEGER) ||
      descriptor->identity == LOOM_COMBINING_IDENTITY_NONE ||
      !loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER,
                                     element_type)) {
    return false;
  }
  const int32_t bit_count = loom_scalar_type_bitwidth(element_type);
  if (bit_count < 1 || bit_count > 64) {
    return false;
  }
  const int64_t signed_minimum = bit_count == 1 ? 1
                                 : bit_count == 64
                                     ? INT64_MIN
                                     : -(INT64_C(1) << (bit_count - 1));
  const int64_t signed_maximum =
      bit_count == 64 ? INT64_MAX
                      : (int64_t)((UINT64_C(1) << (bit_count - 1)) - 1u);
  switch (descriptor->identity) {
    case LOOM_COMBINING_IDENTITY_ZERO:
      *out_identity = 0;
      return true;
    case LOOM_COMBINING_IDENTITY_ONE:
      *out_identity = 1;
      return true;
    case LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM:
      *out_identity = signed_maximum;
      return true;
    case LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM:
      *out_identity = signed_minimum;
      return true;
    case LOOM_COMBINING_IDENTITY_ALL_ONES:
      *out_identity = element_type == LOOM_SCALAR_TYPE_I1 ? 1 : -1;
      return true;
    case LOOM_COMBINING_IDENTITY_NONE:
    case LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO:
    case LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY:
    case LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY:
      return false;
  }
  return false;
}

bool loom_combining_kind_float_identity(loom_combining_kind_t kind,
                                        double* out_identity) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  if (!out_identity || descriptor == NULL ||
      !iree_any_bit_set(descriptor->flags,
                        LOOM_COMBINING_PROPERTY_FLAG_FLOAT)) {
    return false;
  }
  switch (descriptor->identity) {
    case LOOM_COMBINING_IDENTITY_ONE:
      *out_identity = 1.0;
      return true;
    case LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO:
      *out_identity = -0.0;
      return true;
    case LOOM_COMBINING_IDENTITY_NONE:
    case LOOM_COMBINING_IDENTITY_ZERO:
    case LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM:
    case LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM:
    case LOOM_COMBINING_IDENTITY_ALL_ONES:
    case LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY:
    case LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY:
      return false;
  }
  return false;
}

bool loom_combining_kind_float_scan_seed(loom_combining_kind_t kind,
                                         double* out_seed) {
  const loom_combining_descriptor_t* descriptor =
      loom_combining_descriptor(kind);
  if (!out_seed || descriptor == NULL ||
      !iree_any_bit_set(descriptor->flags,
                        LOOM_COMBINING_PROPERTY_FLAG_FLOAT)) {
    return false;
  }
  switch (descriptor->scan_seed) {
    case LOOM_COMBINING_IDENTITY_ONE:
      *out_seed = 1.0;
      return true;
    case LOOM_COMBINING_IDENTITY_NEGATIVE_ZERO:
      *out_seed = -0.0;
      return true;
    case LOOM_COMBINING_IDENTITY_POSITIVE_INFINITY:
      *out_seed = INFINITY;
      return true;
    case LOOM_COMBINING_IDENTITY_NEGATIVE_INFINITY:
      *out_seed = -INFINITY;
      return true;
    case LOOM_COMBINING_IDENTITY_NONE:
    case LOOM_COMBINING_IDENTITY_ZERO:
    case LOOM_COMBINING_IDENTITY_SIGNED_MINIMUM:
    case LOOM_COMBINING_IDENTITY_SIGNED_MAXIMUM:
    case LOOM_COMBINING_IDENTITY_ALL_ONES:
      return false;
  }
  return false;
}
