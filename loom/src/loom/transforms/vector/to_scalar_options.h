// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared options for vector-to-scalar reference lowerings.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_OPTIONS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_OPTIONS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_matrix_fragment_layout_t loom_matrix_fragment_layout_t;

typedef enum loom_vector_to_scalar_flag_bits_e {
  // No optional vector-to-scalar behavior is enabled.
  LOOM_VECTOR_TO_SCALAR_FLAG_NONE = 0u,
  // Lowering may emit source-level subgroup communication operations.
  LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION = 1u << 0,
} loom_vector_to_scalar_flag_bits_t;

// Bitset of loom_vector_to_scalar_flag_bits_t values.
typedef uint32_t loom_vector_to_scalar_flags_t;

typedef struct loom_vector_mma_to_scalar_options_t {
  // Matrix-fragment layout selected by the target contract, or NULL when the
  // fallback has no target-shaped physical layout to interpret.
  const loom_matrix_fragment_layout_t* matrix_fragment_layout;
  // Optional lowering behaviors enabled by the caller.
  loom_vector_to_scalar_flags_t flags;
} loom_vector_mma_to_scalar_options_t;

static inline loom_vector_mma_to_scalar_options_t
loom_vector_mma_to_scalar_options_empty(void) {
  return (loom_vector_mma_to_scalar_options_t){0};
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_OPTIONS_H_
