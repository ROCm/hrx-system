// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_BOOTSTRAP_H_
#define AMDF_SRC_XDNA_BOOTSTRAP_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable target bootstrap consumed by native execution providers.
typedef struct amdf_xdna_bootstrap_t {
  // Native context-admission identity and accounting for this bootstrap.
  struct {
    // UUID naming the bootstrap, independent of application executable images.
    uint8_t uuid[16];
    // Nominal native admission accounting, not the application's operation
    // count.
    uint32_t operations_per_cycle;
  } context;
  // Provider-independent PDI bytes copied into native device storage. Fabric
  // effects belong only to interpreter transport and leave application data
  // routes unclaimed after admission.
  const void* pdi_bytes;
  // Number of bytes in `pdi_bytes`.
  uint32_t pdi_byte_length;
  // Transaction bytes used to admit the interpreter before application work.
  const void* admission_transaction_bytes;
  // Number of bytes in `admission_transaction_bytes`.
  uint32_t admission_transaction_byte_length;
} amdf_xdna_bootstrap_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_BOOTSTRAP_H_
