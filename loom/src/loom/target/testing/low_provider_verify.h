// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only target-low provider table audits.
//
// Target-low provider lists are linked static package data. Production legality
// and diagnostic paths trust them as executable construction invariants; these
// verifiers are for package tests that validate authored provider tables.

#ifndef LOOM_TARGET_TESTING_LOW_PROVIDER_VERIFY_H_
#define LOOM_TARGET_TESTING_LOW_PROVIDER_VERIFY_H_

#include "iree/base/api.h"
#include "loom/target/low_legality.h"
#include "loom/target/low_packet_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies target-low legality provider list shape.
iree_status_t loom_target_low_legality_provider_list_verify(
    loom_target_low_legality_provider_list_t list);

// Verifies target-low packet diagnostic provider list shape.
iree_status_t loom_target_low_packet_diagnostic_provider_list_verify(
    loom_target_low_packet_diagnostic_provider_list_t list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TESTING_LOW_PROVIDER_VERIFY_H_
