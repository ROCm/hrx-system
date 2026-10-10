// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/pattern_registry.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct ApplyContext {
  std::vector<int> calls;
  int rewrite_tag = 0;
};

static iree_status_t RecordPattern(const loom_rewrite_pattern_t* pattern,
                                   void* context, loom_op_t*, loom_rewriter_t*,
                                   bool* out_changed) {
  ApplyContext* apply_context = static_cast<ApplyContext*>(context);
  const int tag = *static_cast<const int*>(pattern->user_data);
  apply_context->calls.push_back(tag);
  *out_changed = tag == apply_context->rewrite_tag;
  return iree_ok_status();
}

TEST(RewritePatternRegistryTest, IndexesSparseKindsAndPreservesProviderOrder) {
  static const int kTagOne = 1;
  static const int kTagTwo = 2;
  static const int kTagThree = 3;
  static const int kTagFour = 4;
  const loom_op_kind_t scalar_kind = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3);
  const loom_op_kind_t vector_kind = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 2);
  const loom_op_kind_t external_kind = LOOM_OP_KIND(0x80, 255);
  const loom_rewrite_pattern_t first_patterns[] = {
      {scalar_kind, RecordPattern, &kTagOne},
      {external_kind, RecordPattern, &kTagFour},
  };
  const loom_rewrite_pattern_t second_patterns[] = {
      {scalar_kind, RecordPattern, &kTagTwo},
      {vector_kind, RecordPattern, &kTagThree},
  };
  const loom_rewrite_pattern_provider_t first_provider = {
      .name = IREE_SVL("first"),
      .patterns = first_patterns,
      .pattern_count = IREE_ARRAYSIZE(first_patterns),
  };
  const loom_rewrite_pattern_provider_t second_provider = {
      .name = IREE_SVL("second"),
      .patterns = second_patterns,
      .pattern_count = IREE_ARRAYSIZE(second_patterns),
  };
  const loom_rewrite_pattern_provider_t* provider_values[] = {
      &first_provider,
      &second_provider,
  };

  loom_rewrite_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_storage_initialize(
      loom_rewrite_pattern_provider_list_make(provider_values,
                                              IREE_ARRAYSIZE(provider_values)),
      iree_allocator_system(), &storage));
  const loom_rewrite_pattern_registry_t* registry =
      loom_rewrite_pattern_registry_storage_registry(&storage);

  EXPECT_EQ(registry->pattern_count, 4u);
  const loom_rewrite_pattern_span_t scalar_span =
      loom_rewrite_pattern_registry_lookup_kind(registry, scalar_kind);
  ASSERT_EQ(scalar_span.count, 2u);
  EXPECT_EQ(scalar_span.patterns[0], &first_patterns[0]);
  EXPECT_EQ(scalar_span.patterns[1], &second_patterns[0]);

  const loom_rewrite_pattern_span_t vector_span =
      loom_rewrite_pattern_registry_lookup_kind(registry, vector_kind);
  ASSERT_EQ(vector_span.count, 1u);
  EXPECT_EQ(vector_span.patterns[0], &second_patterns[1]);

  const loom_rewrite_pattern_span_t external_span =
      loom_rewrite_pattern_registry_lookup_kind(registry, external_kind);
  ASSERT_EQ(external_span.count, 1u);
  EXPECT_EQ(external_span.patterns[0], &first_patterns[1]);

  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_TEST, 0))
                .count,
            0u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_TILE, 0))
                .count,
            0u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 4))
                .count,
            0u);

  const uintptr_t allocation_begin =
      reinterpret_cast<uintptr_t>(storage.allocation.data);
  const uintptr_t allocation_end =
      allocation_begin + storage.allocation.data_length;
  EXPECT_GE(reinterpret_cast<uintptr_t>(registry->dialects), allocation_begin);
  EXPECT_LT(reinterpret_cast<uintptr_t>(registry->dialects), allocation_end);
  EXPECT_GE(reinterpret_cast<uintptr_t>(registry->patterns), allocation_begin);
  EXPECT_LT(reinterpret_cast<uintptr_t>(registry->patterns), allocation_end);

  loom_rewrite_pattern_registry_storage_deinitialize(&storage);
  EXPECT_EQ(storage.allocation.data, nullptr);
}

TEST(RewritePatternRegistryTest, AppliesOnlyTheRootedSpanUntilARewrite) {
  static const int kTagOne = 1;
  static const int kTagTwo = 2;
  static const int kTagThree = 3;
  const loom_op_kind_t scalar_kind = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3);
  const loom_rewrite_pattern_t patterns[] = {
      {scalar_kind, RecordPattern, &kTagOne},
      {LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 2), RecordPattern, &kTagThree},
      {scalar_kind, RecordPattern, &kTagTwo},
  };
  const loom_rewrite_pattern_provider_t provider = {
      .name = IREE_SVL("test"),
      .patterns = patterns,
      .pattern_count = IREE_ARRAYSIZE(patterns),
  };
  const loom_rewrite_pattern_provider_t* provider_values[] = {&provider};
  loom_rewrite_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_storage_initialize(
      loom_rewrite_pattern_provider_list_make(provider_values,
                                              IREE_ARRAYSIZE(provider_values)),
      iree_allocator_system(), &storage));
  const loom_rewrite_pattern_registry_t* registry =
      loom_rewrite_pattern_registry_storage_registry(&storage);

  loom_op_t op = {
      .kind = scalar_kind,
  };
  ApplyContext context;
  context.rewrite_tag = kTagTwo;
  bool changed = false;
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_apply(
      registry, &context, &op, /*rewriter=*/nullptr, &changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(context.calls, (std::vector<int>{kTagOne, kTagTwo}));

  context.calls.clear();
  op.kind = LOOM_OP_KIND(LOOM_DIALECT_INDEX, 7);
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_apply(
      registry, &context, &op, /*rewriter=*/nullptr, &changed));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(context.calls.empty());

  loom_rewrite_pattern_registry_storage_deinitialize(&storage);
}

TEST(RewritePatternRegistryTest, EmptyProviderSetNeedsNoAllocation) {
  loom_rewrite_pattern_registry_storage_t storage = {};
  IREE_ASSERT_OK(loom_rewrite_pattern_registry_storage_initialize(
      loom_rewrite_pattern_provider_list_empty(), iree_allocator_system(),
      &storage));
  const loom_rewrite_pattern_registry_t* registry =
      loom_rewrite_pattern_registry_storage_registry(&storage);
  EXPECT_EQ(registry->pattern_count, 0u);
  EXPECT_EQ(storage.allocation.data, nullptr);
  EXPECT_EQ(storage.allocation.data_length, 0u);
  EXPECT_EQ(loom_rewrite_pattern_registry_lookup_kind(
                registry, LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 0))
                .count,
            0u);
  loom_rewrite_pattern_registry_storage_deinitialize(&storage);
}

TEST(RewritePatternRegistryTest, RejectsMalformedProviders) {
  loom_rewrite_pattern_registry_storage_t storage = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewrite_pattern_registry_storage_initialize(
          loom_rewrite_pattern_provider_list_make(/*values=*/nullptr, 1),
          iree_allocator_system(), &storage));

  const loom_rewrite_pattern_provider_t* null_provider[] = {nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewrite_pattern_registry_storage_initialize(
          loom_rewrite_pattern_provider_list_make(null_provider, 1),
          iree_allocator_system(), &storage));

  const loom_rewrite_pattern_provider_t missing_table = {
      .name = IREE_SVL("missing-table"),
      .patterns = nullptr,
      .pattern_count = 1,
  };
  const loom_rewrite_pattern_provider_t* missing_table_values[] = {
      &missing_table,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewrite_pattern_registry_storage_initialize(
          loom_rewrite_pattern_provider_list_make(missing_table_values, 1),
          iree_allocator_system(), &storage));

  const loom_rewrite_pattern_t unknown_root[] = {
      {LOOM_OP_KIND_UNKNOWN, RecordPattern, nullptr},
  };
  const loom_rewrite_pattern_provider_t unknown_root_provider = {
      .name = IREE_SVL("unknown-root"),
      .patterns = unknown_root,
      .pattern_count = IREE_ARRAYSIZE(unknown_root),
  };
  const loom_rewrite_pattern_provider_t* unknown_root_values[] = {
      &unknown_root_provider,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewrite_pattern_registry_storage_initialize(
          loom_rewrite_pattern_provider_list_make(unknown_root_values, 1),
          iree_allocator_system(), &storage));

  const loom_rewrite_pattern_t missing_callback[] = {
      {LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 0), nullptr, nullptr},
  };
  const loom_rewrite_pattern_provider_t missing_callback_provider = {
      .name = IREE_SVL("missing-callback"),
      .patterns = missing_callback,
      .pattern_count = IREE_ARRAYSIZE(missing_callback),
  };
  const loom_rewrite_pattern_provider_t* missing_callback_values[] = {
      &missing_callback_provider,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewrite_pattern_registry_storage_initialize(
          loom_rewrite_pattern_provider_list_make(missing_callback_values, 1),
          iree_allocator_system(), &storage));
}

}  // namespace
}  // namespace loom
