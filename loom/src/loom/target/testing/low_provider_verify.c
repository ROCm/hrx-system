// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/testing/low_provider_verify.h"

#include "loom/ir/context.h"

iree_status_t loom_target_low_legality_provider_list_verify(
    loom_target_low_legality_provider_list_t list) {
  if (loom_target_low_legality_provider_list_is_empty(list)) {
    return iree_ok_status();
  }
  if (list.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality provider list is required");
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    const loom_target_low_legality_provider_t* provider = list.values[i];
    if (provider == NULL || provider->try_verify_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target-low legality provider is invalid");
    }
    if (provider->builtin_dialect_bits == 0 ||
        iree_any_bit_set(provider->builtin_dialect_bits,
                         ~((1u << LOOM_DIALECT_BUILTIN_COUNT_) - 1u))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target-low legality provider dialect mask is "
                              "invalid");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_target_low_packet_diagnostic_provider_list_verify(
    loom_target_low_packet_diagnostic_provider_list_t list) {
  if (loom_target_low_packet_diagnostic_provider_list_is_empty(list)) {
    return iree_ok_status();
  }
  if (list.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low packet diagnostic provider list is required");
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    const loom_target_low_packet_diagnostic_provider_t* provider =
        list.values[i];
    if (provider == NULL || provider->try_diagnose_packet == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target-low packet diagnostic provider is invalid");
    }
  }
  return iree_ok_status();
}
