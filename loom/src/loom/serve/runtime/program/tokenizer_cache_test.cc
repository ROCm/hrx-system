// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/tokenizer_cache.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loomc/iree.h"

namespace {

class TokenizerCacheTest : public ::testing::Test {
 protected:
  struct Completion {
    loom_serve_tokenizer_subscription_t subscription = {};
    iree_status_t status = iree_ok_status();
    loom_serve_tokenizer_t* tokenizer = nullptr;
    int count = 0;

    Completion() {
      subscription.completion = [](void* user_data, iree_status_t status,
                                   loom_serve_tokenizer_t* tokenizer) {
        auto* self = static_cast<Completion*>(user_data);
        self->status = status;
        self->tokenizer = tokenizer;
        ++self->count;
      };
      subscription.user_data = this;
    }

    ~Completion() {
      iree_status_ignore(status);
      loom_serve_tokenizer_release(tokenizer);
    }
  };

  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_tokenizer_cache_allocate(iree_allocator_system(), &cache_));
  }

  void TearDown() override {
    loom_serve_tokenizer_cache_free(cache_);
    for (loomc_source_t* source : sources_) {
      loomc_source_release(source);
    }
  }

  loomc_source_t* Source(iree_string_view_t value) {
    const loomc_source_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(options),
        .identifier = loomc_make_string_view("tokenizer.json", 14),
        .contents = loomc_make_byte_span(value.data, value.size),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    IREE_CHECK_OK(iree_status_from_loomc(
        loomc_source_create(&options, loomc_allocator_system(), &source)));
    sources_.push_back(source);
    return source;
  }

  static iree_tokenizer_t* BuildTokenizer() {
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
        IREE_SV(R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]}})"),
        iree_allocator_system(), &tokenizer));
    return tokenizer;
  }

  loom_serve_tokenizer_cache_t* cache_ = nullptr;
  std::vector<loomc_source_t*> sources_;
};

TEST_F(TokenizerCacheTest, JoinsInflightAndTrimsOnlyAfterBorrowersRelease) {
  loomc_source_t* producer_source = Source(IREE_SV("same"));
  loomc_source_t* subscriber_source = Source(IREE_SV("same"));
  Completion producer_subscription;
  Completion pending;
  loom_serve_tokenizer_acquisition_t acquisition;
  loom_serve_tokenizer_t* producer = nullptr;
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(
      cache_, producer_source, &producer_subscription.subscription,
      &acquisition, &producer));
  EXPECT_EQ(acquisition, LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER);
  EXPECT_EQ(producer, nullptr);

  loom_serve_tokenizer_t* subscriber = nullptr;
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(cache_, subscriber_source,
                                                    &pending.subscription,
                                                    &acquisition, &subscriber));
  EXPECT_EQ(acquisition, LOOM_SERVE_TOKENIZER_ACQUISITION_PENDING);
  EXPECT_EQ(subscriber, nullptr);
  EXPECT_EQ(pending.count, 0);

  IREE_ASSERT_OK(loom_serve_tokenizer_cache_complete(
      cache_, producer_source, iree_ok_status(), BuildTokenizer(), &producer));
  ASSERT_NE(producer, nullptr);
  EXPECT_EQ(pending.count, 1);
  IREE_EXPECT_OK(pending.status);
  EXPECT_EQ(pending.tokenizer, producer);

  loom_serve_tokenizer_cache_trim(cache_);
  Completion late_subscription;
  loom_serve_tokenizer_t* late = nullptr;
  loomc_source_t* late_source = Source(IREE_SV("same"));
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(
      cache_, late_source, &late_subscription.subscription, &acquisition,
      &late));
  EXPECT_EQ(acquisition, LOOM_SERVE_TOKENIZER_ACQUISITION_COMPLETE);
  EXPECT_EQ(late, producer);

  loom_serve_tokenizer_release(late);
  loom_serve_tokenizer_release(producer);
  producer = nullptr;
  loom_serve_tokenizer_release(pending.tokenizer);
  pending.tokenizer = nullptr;
  loom_serve_tokenizer_cache_trim(cache_);

  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(
      cache_, producer_source, &producer_subscription.subscription,
      &acquisition, &producer));
  EXPECT_EQ(acquisition, LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER);
  iree_status_t failure =
      iree_make_status(IREE_STATUS_CANCELLED, "retire test producer");
  IREE_EXPECT_OK(loom_serve_tokenizer_cache_complete(
      cache_, producer_source, failure, nullptr, &producer));
  iree_status_ignore(failure);
}

TEST_F(TokenizerCacheTest, FailureBroadcastRemovesIdentityForRetry) {
  loomc_source_t* producer_source = Source(IREE_SV("retry"));
  loomc_source_t* subscriber_source = Source(IREE_SV("retry"));
  Completion producer_subscription;
  Completion pending;
  loom_serve_tokenizer_acquisition_t acquisition;
  loom_serve_tokenizer_t* producer = nullptr;
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(
      cache_, producer_source, &producer_subscription.subscription,
      &acquisition, &producer));
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(cache_, subscriber_source,
                                                    &pending.subscription,
                                                    &acquisition, &producer));

  iree_status_t failure =
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid tokenizer");
  IREE_EXPECT_OK(loom_serve_tokenizer_cache_complete(
      cache_, producer_source, failure, nullptr, &producer));
  iree_status_ignore(failure);
  EXPECT_EQ(pending.count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, pending.status);
  pending.status = iree_ok_status();
  EXPECT_EQ(pending.tokenizer, nullptr);

  IREE_ASSERT_OK(loom_serve_tokenizer_cache_acquire(
      cache_, producer_source, &producer_subscription.subscription,
      &acquisition, &producer));
  EXPECT_EQ(acquisition, LOOM_SERVE_TOKENIZER_ACQUISITION_PRODUCER);
  IREE_ASSERT_OK(loom_serve_tokenizer_cache_complete(
      cache_, producer_source, iree_ok_status(), BuildTokenizer(), &producer));
  loom_serve_tokenizer_release(producer);
}

}  // namespace
