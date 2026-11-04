// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/cache_policy.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TextCachePolicy, DefaultTargetUsesConfiguredBudgetNotAvailableMemory) {
  EXPECT_EQ(loom_serve_text_cache_default_target(0), UINT64_C(2) << 30);
  EXPECT_EQ(loom_serve_text_cache_default_target(UINT64_C(1) << 30),
            UINT64_C(64) << 20);
  EXPECT_EQ(loom_serve_text_cache_default_target(UINT64_C(128) << 30),
            UINT64_C(2) << 30);
}

struct CacheStorage {
  // Counts requested from the source, including the final retained candidate.
  std::vector<iree_host_size_t> evaluations;
  // Initial resident state plus COW headroom when checkpoints are present.
  uint64_t residents = 4;
  // Source layout bucket, intentionally not a linear cost function.
  uint64_t alignment = 1;
  // A source error, independent of whether its size would fit the target.
  iree_host_size_t rejected = IREE_HOST_SIZE_MAX;

  static iree_status_t Evaluate(void* user_data, iree_host_size_t count,
                                loom_serve_text_cache_cost_t* out_cost) {
    auto* storage = static_cast<CacheStorage*>(user_data);
    storage->evaluations.push_back(count);
    if (count == storage->rejected) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "source failure");
    }
    const uint64_t slots =
        count ? 2 * storage->residents + count : storage->residents;
    *out_cost = {iree_align_uint64(slots, storage->alignment) * 100};
    return iree_ok_status();
  }
};

TEST(TextCachePolicy, ExactCountsIgnoreSoftTarget) {
  CacheStorage storage;
  const loom_serve_text_cache_options_t options = {1, 19, 5};
  loom_serve_text_cache_selection_t selection;
  IREE_ASSERT_OK(loom_serve_text_cache_select(
      &options, 3, CacheStorage::Evaluate, &storage, &selection));
  EXPECT_EQ(selection.session_capacity, 19u);
  EXPECT_EQ(selection.prefix_capacity, 5u);
  EXPECT_EQ(selection.checkpoint_capacity, 28u);
  EXPECT_EQ(selection.additional_bytes, 0u);
  EXPECT_EQ(storage.evaluations, std::vector<iree_host_size_t>({28}));
}

TEST(TextCachePolicy, PrefixPriorityThenDetachedHistoryIncludingExchange) {
  CacheStorage storage;
  const loom_serve_text_cache_options_t options = {
      1100, LOOM_SERVE_TEXT_CACHE_AUTOMATIC, LOOM_SERVE_TEXT_CACHE_AUTOMATIC};
  loom_serve_text_cache_selection_t selection;
  IREE_ASSERT_OK(loom_serve_text_cache_select(
      &options, 0, CacheStorage::Evaluate, &storage, &selection));
  EXPECT_EQ(selection.prefix_capacity, 2u);
  EXPECT_EQ(selection.session_capacity, 4u);
  EXPECT_EQ(selection.checkpoint_capacity, 7u);
  EXPECT_EQ(selection.additional_bytes, 1100u);
  EXPECT_EQ(storage.evaluations.back(), 7u);
}

TEST(TextCachePolicy, SmallTargetSelectsOneOrNoPrefixWithoutFailure) {
  for (const uint64_t target : {0u, 499u, 500u, 600u}) {
    CacheStorage storage;
    const loom_serve_text_cache_options_t options = {
        target, LOOM_SERVE_TEXT_CACHE_AUTOMATIC,
        LOOM_SERVE_TEXT_CACHE_AUTOMATIC};
    loom_serve_text_cache_selection_t selection;
    IREE_ASSERT_OK(loom_serve_text_cache_select(
        &options, 0, CacheStorage::Evaluate, &storage, &selection));
    const uint64_t expected = target < 500 ? 0 : target / 100 - 4;
    EXPECT_EQ(selection.prefix_capacity, expected);
    EXPECT_EQ(selection.session_capacity, 0u);
    EXPECT_LE(selection.additional_bytes, target);
    EXPECT_EQ(storage.evaluations.back(), expected);
  }
}

TEST(TextCachePolicy, ExplicitZeroAndPinsSurviveAutomaticSizing) {
  CacheStorage storage;
  const loom_serve_text_cache_options_t options = {
      400, LOOM_SERVE_TEXT_CACHE_AUTOMATIC, 0};
  loom_serve_text_cache_selection_t selection;
  IREE_ASSERT_OK(loom_serve_text_cache_select(
      &options, 2, CacheStorage::Evaluate, &storage, &selection));
  EXPECT_EQ(selection.prefix_capacity, 0u);
  EXPECT_EQ(selection.session_capacity, 3u);
  EXPECT_EQ(selection.checkpoint_capacity, 6u);
  EXPECT_EQ(selection.additional_bytes, 400u);
}

TEST(TextCachePolicy, NonlinearSourceBucketsAreEvaluatedExactly) {
  CacheStorage storage;
  storage.alignment = 8;
  const loom_serve_text_cache_options_t options = {
      800, LOOM_SERVE_TEXT_CACHE_AUTOMATIC, LOOM_SERVE_TEXT_CACHE_AUTOMATIC};
  loom_serve_text_cache_selection_t selection;
  IREE_ASSERT_OK(loom_serve_text_cache_select(
      &options, 0, CacheStorage::Evaluate, &storage, &selection));
  EXPECT_EQ(selection.prefix_capacity, 2u);
  EXPECT_EQ(selection.session_capacity, 5u);
  EXPECT_EQ(selection.checkpoint_capacity, 8u);
  EXPECT_EQ(selection.additional_bytes, 800u);
  EXPECT_EQ(storage.evaluations.back(), 8u);
}

TEST(TextCachePolicy, AutomaticDirectoryIsBoundedByBytesNotPhysicalPages) {
  CacheStorage storage;
  storage.residents = 0;
  const loom_serve_text_cache_options_t options = {
      600, LOOM_SERVE_TEXT_CACHE_AUTOMATIC, 0};
  loom_serve_text_cache_selection_t selection;
  IREE_ASSERT_OK(loom_serve_text_cache_select(
      &options, 0, CacheStorage::Evaluate, &storage, &selection));
  EXPECT_EQ(selection.session_capacity, 5u);
  EXPECT_EQ(selection.checkpoint_capacity, 6u);
  EXPECT_EQ(selection.additional_bytes, 600u);
  EXPECT_EQ(storage.evaluations.back(), 6u);
}

TEST(TextCachePolicy, SourceFailureIsNotAReasonToSilentlyDisableCaching) {
  CacheStorage storage;
  storage.rejected = 2;
  const loom_serve_text_cache_options_t options = {
      1000, LOOM_SERVE_TEXT_CACHE_AUTOMATIC, LOOM_SERVE_TEXT_CACHE_AUTOMATIC};
  loom_serve_text_cache_selection_t selection;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_cache_select(&options, 0, CacheStorage::Evaluate,
                                   &storage, &selection));
  EXPECT_EQ(storage.evaluations, std::vector<iree_host_size_t>({0, 2}));
}

}  // namespace
