// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/structure.h"

#include <stddef.h>
#include <string.h>

typedef struct amdf_structure_prefix_t {
  amdf_structure_type_t type;
  uint32_t structure_size;
} amdf_structure_prefix_t;

_Static_assert(offsetof(amdf_input_structure_t, next) ==
                   offsetof(amdf_output_structure_t, next),
               "input and output structure headers must have one layout");

static amdf_status_t amdf_structure_validate_header(
    const void* structure, amdf_structure_type_t expected_type,
    uint32_t minimum_size) {
  if (structure == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_structure_prefix_t prefix;
  memcpy(&prefix, structure, sizeof(prefix));
  if (prefix.type != expected_type || prefix.structure_size < minimum_size) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const void* next = NULL;
  memcpy(&next,
         (const uint8_t*)structure + offsetof(amdf_input_structure_t, next),
         sizeof(next));
  if (next != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_structure_validate_input(const void* structure,
                                            amdf_structure_type_t expected_type,
                                            uint32_t minimum_size) {
  return amdf_structure_validate_header(structure, expected_type, minimum_size);
}

amdf_status_t amdf_structure_validate_output(
    const void* structure, amdf_structure_type_t expected_type,
    uint32_t minimum_size) {
  return amdf_structure_validate_header(structure, expected_type, minimum_size);
}
