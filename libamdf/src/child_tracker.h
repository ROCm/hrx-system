// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_CHILD_TRACKER_H_
#define AMDF_SRC_CHILD_TRACKER_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/atomics.h"

// Tracks children that borrow an explicitly owned parent object.
typedef struct amdf_child_tracker_t {
  // Number of currently registered children.
  amdf_atomic_uint32_t count;
} amdf_child_tracker_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Initializes an empty tracker before its parent is published.
void amdf_child_tracker_initialize(amdf_child_tracker_t* tracker);

// Returns the number of registered children with acquire ordering.
uint32_t amdf_child_tracker_count(const amdf_child_tracker_t* tracker);

// Registers one child or reports that the counter is exhausted.
amdf_status_t amdf_child_tracker_register(amdf_child_tracker_t* tracker);

// Unregisters one child and diagnoses an unbalanced release.
void amdf_child_tracker_unregister(amdf_child_tracker_t* tracker);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_CHILD_TRACKER_H_
