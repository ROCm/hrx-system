// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/legalization.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const loom_target_legalizer_rule_t kTargetRules[] = {
    {/*.flags=*/0,
     /*.root_kind=*/LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3),
     /*.legalize=*/nullptr},
    {/*.flags=*/0,
     /*.root_kind=*/LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 2),
     /*.legalize=*/nullptr},
};

static const loom_target_legalizer_rule_t kReferenceRules[] = {
    {/*.flags=*/0,
     /*.root_kind=*/LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3),
     /*.legalize=*/nullptr},
};

static const loom_target_legalizer_provider_t kTargetProvider = {
    /*.name=*/IREE_SVL("target"),
    /*.strategy=*/LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
    /*.rules=*/kTargetRules,
    /*.rule_count=*/IREE_ARRAYSIZE(kTargetRules),
};

static const loom_target_legalizer_provider_t kReferenceProvider = {
    /*.name=*/IREE_SVL("reference"),
    /*.strategy=*/LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    /*.rules=*/kReferenceRules,
    /*.rule_count=*/IREE_ARRAYSIZE(kReferenceRules),
};

TEST(TargetLegalizerRegistryTest, ComposesOrderedProviderListsIntoOneSlab) {
  const loom_target_legalizer_provider_t* target_providers[] = {
      &kTargetProvider,
  };
  const loom_target_legalizer_provider_t* reference_providers[] = {
      &kReferenceProvider,
  };
  const loom_target_legalizer_provider_list_t provider_lists[] = {
      loom_target_legalizer_provider_list_make(
          target_providers, IREE_ARRAYSIZE(target_providers)),
      loom_target_legalizer_provider_list_make(
          reference_providers, IREE_ARRAYSIZE(reference_providers)),
  };

  loom_target_legalizer_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_legalizer_registry_storage_initialize(
      provider_lists, IREE_ARRAYSIZE(provider_lists), iree_allocator_system(),
      &storage));
  const loom_target_legalizer_registry_t* registry =
      loom_target_legalizer_registry_storage_registry(&storage);

  ASSERT_EQ(registry->entry_count, 3u);
  const loom_target_legalizer_op_entry_t scalar_entry =
      loom_target_legalizer_registry_lookup_kind(
          registry, LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3));
  ASSERT_EQ(scalar_entry.entry_count, 2u);
  const loom_target_legalizer_entry_t* scalar_entries =
      &registry->entries[scalar_entry.entry_start];
  EXPECT_TRUE(iree_string_view_equal(scalar_entries[0].provider_name,
                                     IREE_SV("target")));
  EXPECT_EQ(scalar_entries[0].provider_strategy,
            LOOM_TARGET_LEGALIZER_STRATEGY_TARGET);
  EXPECT_TRUE(iree_string_view_equal(scalar_entries[1].provider_name,
                                     IREE_SV("reference")));
  EXPECT_EQ(scalar_entries[1].provider_strategy,
            LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE);

  EXPECT_EQ(loom_target_legalizer_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0))
                .entry_count,
            0u);
  EXPECT_EQ(loom_target_legalizer_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_TILE, 0))
                .entry_count,
            0u);
  EXPECT_EQ(loom_target_legalizer_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 4))
                .entry_count,
            0u);

  const uintptr_t allocation_begin =
      reinterpret_cast<uintptr_t>(storage.allocation.data);
  const uintptr_t allocation_end =
      allocation_begin + storage.allocation.data_length;
  const uintptr_t dialects = reinterpret_cast<uintptr_t>(registry->dialects);
  const uintptr_t op_entries =
      reinterpret_cast<uintptr_t>(registry->dialects[0].op_entries);
  const uintptr_t entries = reinterpret_cast<uintptr_t>(registry->entries);
  EXPECT_GE(dialects, allocation_begin);
  EXPECT_LT(dialects, allocation_end);
  EXPECT_GE(op_entries, allocation_begin);
  EXPECT_LT(op_entries, allocation_end);
  EXPECT_GE(entries, allocation_begin);
  EXPECT_LT(entries, allocation_end);

  loom_target_legalizer_registry_storage_deinitialize(&storage);
  EXPECT_EQ(storage.allocation.data, nullptr);
}

TEST(TargetLegalizerRegistryTest, EmptyProviderSetNeedsNoAllocation) {
  loom_target_legalizer_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_legalizer_registry_storage_initialize(
      /*provider_lists=*/nullptr, /*provider_list_count=*/0,
      iree_allocator_system(), &storage));
  const loom_target_legalizer_registry_t* registry =
      loom_target_legalizer_registry_storage_registry(&storage);
  EXPECT_EQ(registry->entry_count, 0u);
  EXPECT_EQ(storage.allocation.data, nullptr);
  EXPECT_EQ(storage.allocation.data_length, 0u);
  loom_target_legalizer_registry_storage_deinitialize(&storage);
}

}  // namespace
}  // namespace loom
