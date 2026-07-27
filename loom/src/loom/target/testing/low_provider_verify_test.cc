// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/testing/low_provider_verify.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_status_t IgnoreProviderOp(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  (void)context;
  (void)op;
  *out_handled = false;
  return iree_ok_status();
}

static iree_status_t IgnorePacketDiagnostic(
    const loom_target_low_packet_diagnostic_provider_t* provider,
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, bool* out_handled) {
  (void)provider;
  (void)context;
  (void)packet;
  *out_handled = false;
  return iree_ok_status();
}

TEST(TargetLowProviderVerifyTest, AcceptsEmptyLists) {
  const loom_target_low_legality_provider_list_t legality_list =
      loom_target_low_legality_provider_list_empty();
  EXPECT_TRUE(loom_target_low_legality_provider_list_is_empty(legality_list));
  IREE_EXPECT_OK(loom_target_low_legality_provider_list_verify(legality_list));

  const loom_target_low_packet_diagnostic_provider_list_t diagnostic_list =
      loom_target_low_packet_diagnostic_provider_list_empty();
  EXPECT_TRUE(loom_target_low_packet_diagnostic_provider_list_is_empty(
      diagnostic_list));
  IREE_EXPECT_OK(
      loom_target_low_packet_diagnostic_provider_list_verify(diagnostic_list));
}

TEST(TargetLowProviderVerifyTest, AcceptsProviderRows) {
  const loom_target_low_legality_provider_t legality_provider = {
      /*.name=*/IREE_SVL("test-legality-provider"),
      /*.builtin_dialect_bits=*/1u << LOOM_DIALECT_VECTOR,
      /*.try_verify_op=*/IgnoreProviderOp,
  };
  const loom_target_low_legality_provider_t* legality_values[] = {
      &legality_provider,
  };
  const loom_target_low_legality_provider_list_t legality_list =
      loom_target_low_legality_provider_list_make(
          legality_values, IREE_ARRAYSIZE(legality_values));
  IREE_EXPECT_OK(loom_target_low_legality_provider_list_verify(legality_list));

  const loom_target_low_packet_diagnostic_provider_t diagnostic_provider = {
      /*.name=*/IREE_SVL("test-packet-provider"),
      /*.try_diagnose_packet=*/IgnorePacketDiagnostic,
  };
  const loom_target_low_packet_diagnostic_provider_t* diagnostic_values[] = {
      &diagnostic_provider,
  };
  const loom_target_low_packet_diagnostic_provider_list_t diagnostic_list =
      loom_target_low_packet_diagnostic_provider_list_make(
          diagnostic_values, IREE_ARRAYSIZE(diagnostic_values));
  IREE_EXPECT_OK(
      loom_target_low_packet_diagnostic_provider_list_verify(diagnostic_list));
}

TEST(TargetLowProviderVerifyTest, RejectsMissingTables) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_low_legality_provider_list_verify(
          (loom_target_low_legality_provider_list_t){/*.count=*/1,
                                                     /*.values=*/NULL}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_low_packet_diagnostic_provider_list_verify(
                            (loom_target_low_packet_diagnostic_provider_list_t){
                                /*.count=*/1,
                                /*.values=*/NULL}));
}

}  // namespace
}  // namespace loom
