// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/legalizer_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

namespace loom {
namespace {

static iree_status_t TargetLegalize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  (void)context;
  (void)op;
  *out_result = {};
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t* LookupOnlyEntry(
    const loom_target_legalizer_registry_t* registry, loom_op_kind_t op_kind) {
  const loom_target_legalizer_op_entry_t op_entry =
      loom_target_legalizer_registry_lookup_kind(registry, op_kind);
  EXPECT_EQ(op_entry.entry_count, 1u);
  if (op_entry.entry_count != 1) return nullptr;
  return &registry->entries[op_entry.entry_start];
}

static void ExpectReferenceProvider(
    const loom_target_legalizer_registry_t* registry, loom_op_kind_t op_kind,
    iree_string_view_t expected_name) {
  const loom_target_legalizer_entry_t* entry =
      LookupOnlyEntry(registry, op_kind);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(iree_string_view_equal(entry->provider_name, expected_name));
  EXPECT_EQ(entry->provider_strategy, LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE);
}

TEST(LowLegalizerRegistryTest, TargetProvidersPrecedeGenericProviders) {
  const loom_target_legalizer_rule_t target_rules[] = {
      {/*.flags=*/LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL,
       /*.root_kind=*/LOOM_OP_SCALAR_EXTF,
       /*.legalize=*/TargetLegalize},
  };
  const loom_target_legalizer_provider_t target_provider = {
      /*.name=*/IREE_SVL("target"),
      /*.strategy=*/LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
      /*.rules=*/target_rules,
      /*.rule_count=*/IREE_ARRAYSIZE(target_rules),
  };
  const loom_target_legalizer_provider_t* target_providers[] = {
      &target_provider,
  };

  loom_target_legalizer_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_low_legalizer_registry_storage_initialize(
      loom_target_legalizer_provider_list_make(
          target_providers, IREE_ARRAYSIZE(target_providers)),
      iree_allocator_system(), &storage));
  const loom_target_legalizer_registry_t* registry =
      loom_target_legalizer_registry_storage_registry(&storage);

  const loom_target_legalizer_op_entry_t extf_entry =
      loom_target_legalizer_registry_lookup_kind(registry, LOOM_OP_SCALAR_EXTF);
  ASSERT_EQ(extf_entry.entry_count, 2u);
  const loom_target_legalizer_entry_t* extf_entries =
      &registry->entries[extf_entry.entry_start];
  EXPECT_TRUE(
      iree_string_view_equal(extf_entries[0].provider_name, IREE_SV("target")));
  EXPECT_EQ(extf_entries[0].provider_strategy,
            LOOM_TARGET_LEGALIZER_STRATEGY_TARGET);
  EXPECT_EQ(extf_entries[0].flags,
            LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL);
  EXPECT_EQ(extf_entries[0].legalize, &TargetLegalize);
  EXPECT_TRUE(
      iree_string_view_equal(extf_entries[1].provider_name, IREE_SV("scalar")));
  EXPECT_EQ(extf_entries[1].provider_strategy,
            LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE);

  ExpectReferenceProvider(registry, LOOM_OP_BUFFER_COPY, IREE_SV("buffer"));
  ExpectReferenceProvider(registry, LOOM_OP_VECTOR_REDUCE, IREE_SV("vector"));
  ExpectReferenceProvider(registry, LOOM_OP_VIEW_ATOMIC_RMW, IREE_SV("view"));

  loom_target_legalizer_registry_storage_deinitialize(&storage);
}

}  // namespace
}  // namespace loom
