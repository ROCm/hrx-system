// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/tokenizer_load.h"

#include <atomic>
#include <string>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "loom/model/card.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/program/package.h"
#include "loomc/iree.h"

namespace {

class TokenizerLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto json = IREE_SV(
        R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}})");
    const loom_serve_package_file_t files[] = {
        {IREE_SV("tokenizer.json"),
         iree_make_const_byte_span(json.data, json.size)},
        {IREE_SV("vocabulary.data"),
         iree_make_const_byte_span(json.data, json.size)},
        {IREE_SV("asset/tokenizer.json"),
         iree_make_const_byte_span(json.data, json.size)},
    };
    IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model.json"), 3, files,
                                             iree_allocator_system(),
                                             &sources_.package));
  }

  void TearDown() override {
    // Accepted work must retire even when the test's assertions fail.
    RunQueued();
    EXPECT_TRUE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
    IREE_EXPECT_OK(status_);
    iree_tokenizer_free(tokenizer_);
    loom_serve_package_release(sources_.package);
    loom_model_card_release(sources_.card);
  }

  static loomc_status_t Submit(void* user_data, loomc_task_t* task) {
    auto* self = static_cast<TokenizerLoadTest*>(user_data);
    if (self->reject_) {
      return loomc_status_from_code(LOOMC_STATUS_RESOURCE_EXHAUSTED);
    }
    if (self->inline_) {
      loomc_task_execute(task, 0);
    } else {
      self->queued_ = task;
    }
    return loomc_ok_status();
  }

  iree_status_t Start(iree_string_view_t locator = IREE_SV("tokenizer.json"),
                      iree_string_view_t directory = iree_string_view_empty()) {
    return loom_serve_tokenizer_load_start(
        {Submit, this}, &sources_, directory, locator,
        [](void* self) { ++static_cast<TokenizerLoadTest*>(self)->wakes_; },
        this, iree_allocator_system(), &load_);
  }

  iree_status_t SelectTokenizer(const std::string& tokenizer) {
    const std::string card =
        R"({"schema_version":1,"model":{"id":"test","name":"test"},
            "programs":{"source":{"inputs":["unused.loom"],"libraries":[]}},
            "services":{"transcription":{"kind":"recognition","version":1}},
            "variants":{"fixture":{"services":{"transcription":{
              "programs":["source"],"tokenizers":{"vocabulary":)" +
        tokenizer + R"(}}}}}})";
    return loom_model_card_parse(iree_make_cstring_view(card.c_str()), nullptr,
                                 iree_allocator_system(), &sources_.card);
  }

  iree_status_t StartNamed(iree_string_view_t name = IREE_SV("vocabulary")) {
    return loom_serve_tokenizer_load_start(
        {Submit, this}, &sources_, IREE_SV("ignored-for-card"), name,
        [](void* self) { ++static_cast<TokenizerLoadTest*>(self)->wakes_; },
        this, iree_allocator_system(), &load_);
  }

  void RunQueued() {
    if (queued_) {
      auto* task = queued_;
      queued_ = nullptr;
      loomc_task_execute(task, 0);
    }
  }

  void CheckResult() {
    ASSERT_TRUE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
    IREE_ASSERT_OK(status_);
    ASSERT_EQ(load_, nullptr);
    ASSERT_NE(tokenizer_, nullptr);
    EXPECT_EQ(iree_tokenizer_vocab_lookup(iree_tokenizer_vocab(tokenizer_),
                                          IREE_SV("b")),
              1);
    EXPECT_EQ(wakes_, 1);
  }

  void CheckSingleWorker();

  // Immutable source bytes borrowed through terminal take.
  loom_serve_sources_t sources_ = {};
  // Operation owned until its nonblocking take succeeds.
  loom_serve_tokenizer_load_t* load_ = nullptr;
  // Tokenizer transferred from completed work.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Owned load outcome consumed by each test or teardown.
  iree_status_t status_ = iree_ok_status();
  // Task accepted by the deterministic sink, not the tokenizer implementation.
  loomc_task_t* queued_ = nullptr;
  // Submission policy exercising the public sink's inline-completion contract.
  bool inline_ = false;
  // Submission policy exercising rejection without task ownership transfer.
  bool reject_ = false;
  // Number of terminal notifications received by the serialized owner.
  int wakes_ = 0;
};

TEST_F(TokenizerLoadTest, LegacyPathResolvesAgainstItsDeclaredDirectory) {
  IREE_ASSERT_OK(Start(IREE_SV("tokenizer.json"), IREE_SV("asset")));
  RunQueued();
  CheckResult();
}

TEST_F(TokenizerLoadTest, NamedTokenizerUsesExplicitFormatAndIndependentPath) {
  IREE_ASSERT_OK(SelectTokenizer(
      R"({"path":"vocabulary.data","format":"huggingface-json"})"));
  IREE_ASSERT_OK(StartNamed());
  RunQueued();
  CheckResult();
}

TEST_F(TokenizerLoadTest, NamedTokenizerInfersJsonEncoding) {
  IREE_ASSERT_OK(SelectTokenizer(R"("tokenizer.json")"));
  IREE_ASSERT_OK(StartNamed());
  RunQueued();
  CheckResult();
}

TEST_F(TokenizerLoadTest, MissingIdentityFailsWithoutSubmission) {
  IREE_ASSERT_OK(SelectTokenizer(R"("tokenizer.json")"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, StartNamed(IREE_SV("other")));
  EXPECT_EQ(load_, nullptr);
  EXPECT_EQ(queued_, nullptr);
  EXPECT_EQ(wakes_, 0);
}

TEST_F(TokenizerLoadTest, AmbiguousFormatFailsWithoutSubmission) {
  IREE_ASSERT_OK(SelectTokenizer(R"("vocabulary.data")"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, StartNamed());
  EXPECT_EQ(load_, nullptr);
  EXPECT_EQ(queued_, nullptr);
  EXPECT_EQ(wakes_, 0);
}

TEST_F(TokenizerLoadTest, ExplicitUnsupportedFormatNeverFallsBackToFilename) {
  IREE_ASSERT_OK(
      SelectTokenizer(R"({"path":"tokenizer.json","format":"other"})"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, StartNamed());
  EXPECT_EQ(load_, nullptr);
  EXPECT_EQ(queued_, nullptr);
  EXPECT_EQ(wakes_, 0);
}

TEST_F(TokenizerLoadTest, PendingTakePreservesOwnershipAndCopiesLocator) {
  std::string locator = "tokenizer.json";
  IREE_ASSERT_OK(Start(iree_make_cstring_view(locator.c_str())));
  locator.assign(locator.size(), '?');
  EXPECT_FALSE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
  EXPECT_EQ(tokenizer_, nullptr);
  EXPECT_EQ(wakes_, 0);
  RunQueued();
  CheckResult();
  // A second take does not overwrite the transferred result.
  EXPECT_TRUE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
  EXPECT_NE(tokenizer_, nullptr);
}

TEST_F(TokenizerLoadTest, InlineCompletionPublishesOneOwnedResult) {
  inline_ = true;
  IREE_ASSERT_OK(Start());
  CheckResult();
}

TEST_F(TokenizerLoadTest, RejectedSubmissionHasNoWakeOrOperation) {
  reject_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Start());
  EXPECT_EQ(load_, nullptr);
  EXPECT_EQ(wakes_, 0);
  EXPECT_EQ(queued_, nullptr);
}

TEST_F(TokenizerLoadTest, MissingSourcePublishesItsFailure) {
  IREE_ASSERT_OK(Start(IREE_SV("missing.json")));
  RunQueued();
  EXPECT_TRUE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status_);
  status_ = iree_ok_status();
  EXPECT_EQ(tokenizer_, nullptr);
  EXPECT_EQ(wakes_, 1);
}

TEST_F(TokenizerLoadTest, CancelledOwnerStillTakesAcceptedConstruction) {
  IREE_ASSERT_OK(Start());
  status_ = iree_make_status(IREE_STATUS_CANCELLED, "owner cancelled");
  EXPECT_FALSE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
  RunQueued();
  EXPECT_TRUE(loom_serve_tokenizer_load_take(&load_, &status_, &tokenizer_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status_);
  status_ = iree_ok_status();
  EXPECT_NE(tokenizer_, nullptr);
}

void TokenizerLoadTest::CheckSingleWorker() {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  struct loom_serve_test_tokenizer_load_parent_t {
    // Intrusive shared-worker task.
    loomc_task_t task;
    // Shared sink also used by the child.
    loomc_task_sink_t sink;
    // Immutable test source owner.
    TokenizerLoadTest* test;
    // Outer test notification, never waited on by the worker.
    iree_notification_t notification;
    // Release publication of the parent's final access to its result fields.
    std::atomic<bool> retired{false};
  } parent;
  parent.sink = loom_serve_host_compiler_sink(host);
  parent.test = this;
  iree_notification_initialize(&parent.notification);
  const loomc_task_vtable_t vtable = {
      [](loomc_task_t* task, loomc_host_size_t worker_ordinal) {
        auto* parent =
            reinterpret_cast<loom_serve_test_tokenizer_load_parent_t*>(task);
        auto* test = parent->test;
        test->status_ = loom_serve_tokenizer_load_start(
            parent->sink, &test->sources_, iree_string_view_empty(),
            IREE_SV("tokenizer.json"),
            [](void* data) {
              iree_notification_post(static_cast<iree_notification_t*>(data),
                                     IREE_ALL_WAITERS);
            },
            &parent->notification, iree_allocator_system(), &test->load_);
      },
      [](loomc_task_t* task) {
        auto* parent =
            reinterpret_cast<loom_serve_test_tokenizer_load_parent_t*>(task);
        parent->retired.store(true, std::memory_order_release);
        iree_notification_post(&parent->notification, IREE_ALL_WAITERS);
      }};
  loomc_task_initialize(&vtable, &parent.task);
  IREE_ASSERT_OK(iree_status_from_loomc(
      loomc_task_sink_submit(parent.sink, &parent.task)));
  iree_notification_await(
      &parent.notification,
      [](void* data) {
        return static_cast<loom_serve_test_tokenizer_load_parent_t*>(data)
            ->retired.load(std::memory_order_acquire);
      },
      &parent, iree_infinite_timeout());
  iree_notification_await(
      &parent.notification,
      [](void* data) {
        auto* test = static_cast<TokenizerLoadTest*>(data);
        return loom_serve_tokenizer_load_take(&test->load_, &test->status_,
                                              &test->tokenizer_);
      },
      this, iree_infinite_timeout());
  // No parent synchronously waits for its child, including with one worker.
  // Only the outer owner joins and releases the shared host after both retire.
  loom_serve_host_release(host);
  IREE_EXPECT_OK(status_);
  ASSERT_NE(tokenizer_, nullptr);
  iree_notification_deinitialize(&parent.notification);
}

TEST_F(TokenizerLoadTest, SharedSingleWorkerRunsChildAfterParentRetires) {
  CheckSingleWorker();
}

}  // namespace
