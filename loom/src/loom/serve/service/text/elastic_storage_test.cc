// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/maintenance_gate.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class ElasticTextServiceTest : public SourceTextServiceTest {
 protected:
  static bool HasPendingBacking(const std::string& status) {
    const std::string key = "\"pending_bytes\":";
    size_t position = 0;
    while ((position = status.find(key, position)) != std::string::npos) {
      position += key.size();
      if (status[position] != '0') {
        return true;
      }
    }
    return false;
  }

  void CheckPrefixBackingRetirement(bool trim_first) {
    const iree_hal_physical_memory_params_t parameters = {
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    iree_device_size_t minimum = 0;
    iree_device_size_t slab_size = 0;
    IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
        iree_hal_device_allocator(loom_serve_device_handle(device_)),
        parameters, &minimum, &slab_size));
    auto options = ModelOptions();
    options.context_capacity = slab_size / 8;
    options.pool_capacity = 4 * options.context_capacity;
    loom_serve_text_model_t* source = nullptr;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                  host_allocator_, &source));
    const auto release_model = [](loom_serve_text_model_t* model) {
      IREE_EXPECT_OK(loom_serve_text_model_free(model));
    };
    const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)>
        owner(source, release_model);
    // The fixture places four 768-byte private rows and ten 256-byte recurrent
    // slots before its four-byte-per-token KV plane. Fill exactly one slab;
    // a new COW tail must acquire native backing, not reuse an existing page.
    const size_t private_bytes = 4 * 768 + 10 * 256;
    const size_t extents[] = {
        4, options.context_capacity,
        (slab_size - private_bytes) / 4 - 4 - options.context_capacity};
    for (size_t i = 0; i < IREE_ARRAYSIZE(extents); ++i) {
      bool admitted = false;
      IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
          loom_serve_text_model_row(source, i), LOOM_SERVE_TEXT_RESERVE_FRESH,
          nullptr, extents[i], &admitted));
      ASSERT_TRUE(admitted);
    }
    auto* row = loom_serve_text_model_row(source, 0);
    auto* warm = loom_serve_text_model_row(model_, 0);
    const int32_t tokens[] = {0, 1, 2};
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(warm, IREE_ARRAYSIZE(tokens), tokens));
    ASSERT_EQ(loom_serve_text_model_memory_statistics(source).committed_bytes,
              slab_size);
    loom_serve_text_checkpoint_t* checkpoint = nullptr;
    loom_serve_execution_frontier_t frontier;
    {
      loom_serve::testing::MaintenanceGate gate(
          loom_serve_device_storage_pool(device_)->maintenance);
      IREE_ASSERT_OK(
          loom_serve_text_row_submit_pin(row, &checkpoint, &frontier));
      ASSERT_NE(checkpoint, nullptr);
      EXPECT_EQ(loom_serve_text_model_memory_statistics(source).pending_bytes,
                slab_size);
      const int32_t token = 1;
      IREE_ASSERT_OK(loom_serve_text_row_prefill(warm, 1, &token));
      Oracle peer;
      peer.Consume("abxb");
      EXPECT_EQ(loom_serve_text_row_token(warm), peer.Predict());
    }
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {IREE_ARRAYSIZE(frontier.semaphores), frontier.semaphores,
         frontier.values},
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    if (trim_first) {
      loom_serve_text_trim_result_t result;
      IREE_ASSERT_OK(loom_serve_text_model_trim(source, &result));
    }
    // Reuse the state-owned operation slot without an intervening epoch. A
    // second pin must preserve the authoritative reader of the first fork.
    loom_serve_text_checkpoint_t* alias = nullptr;
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &alias));
    ASSERT_NE(alias, nullptr);
    auto* fork = loom_serve_text_model_row(source, 3);
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        fork, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, alias, 8, &admitted));
    ASSERT_TRUE(admitted);
    loom_serve_text_checkpoint_release(alias);
    Oracle reference;
    reference.Consume("abx");
    const int32_t next = 0;
    for (auto* branch : {row, fork}) {
      IREE_ASSERT_OK(loom_serve_text_row_prefill(branch, 1, &next));
      Oracle continued = reference;
      continued.Consume(next);
      EXPECT_EQ(loom_serve_text_row_token(branch), continued.Predict());
    }
    loom_serve_text_checkpoint_release(checkpoint);
  }

  void CheckColdStateAdmission(bool cancel) {
    StartManaged();
    for (const char* name : {"fixture-text", "cold"}) {
      ASSERT_EQ(
          Read(Send(Declaration(name, directory_), "", "/v1/models/register"))
              .find("HTTP/1.1 202"),
          0u);
      ASSERT_NO_FATAL_FAILURE(WaitModel(name, R"("registration":"ready")"));
    }
    // Explicit parameter activation leaves the cold model's KV untouched.
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "", "/v1/models/drain"))
                  .find("HTTP/1.1 200"),
              0u);
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "", "/v1/models/activate"))
                  .find("HTTP/1.1 202"),
              0u);
    const auto activated = WaitModel("cold", R"("parameter_operation":"idle")");
    ASSERT_NE(activated.find(R"("parameters":{"active":true)"),
              std::string::npos)
        << activated;
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "", "/v1/models/undrain"))
                  .find("HTTP/1.1 200"),
              0u);
    const std::string prompt = "abxa";
    const std::string warm = Request(Message("user", prompt), 4);
    std::string cold_request = warm;
    cold_request.replace(cold_request.find("fixture-text"), 12, "cold");
    ASSERT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))),
              Oracle::Complete(prompt, 4));
    ASSERT_NO_FATAL_FAILURE(WaitModel("fixture-text", R"("phase":"idle")"));
    ASSERT_FALSE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
    int cold = -1;
    {
      // The application has one task worker. Only the HAL's existing native
      // cold executor is held; the normal router and warm model must progress.
      loom_serve::testing::MaintenanceGate gate(
          loom_serve_device_storage_pool(device_)->maintenance);
      cold = Send(cold_request);
      while (!http_.failed() && !HasFatalFailure()) {
        if (HasPendingBacking(Read(Send("", "", "/v1/status", "GET")))) {
          break;
        }
      }
      ASSERT_FALSE(http_.failed());
      if (cancel) {
        // A disconnected peer does not release native mapping borrowers.
        ASSERT_EQ(close(cold), 0);
        ASSERT_EQ(Read(Send(R"({"model":"cold","mode":"abort"})", "",
                            "/v1/models/unregister"))
                      .find("HTTP/1.1 202"),
                  0u);
        ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("admission":"removing")"));
        EXPECT_TRUE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
      }
      EXPECT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))),
                Oracle::Complete(prompt, 4));
      EXPECT_EQ(Read(Send("", "", "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    }
    if (cancel) {
      ASSERT_NO_FATAL_FAILURE(WaitModel("cold", {}));
    } else {
      EXPECT_EQ(Content(Read(cold)), Oracle::Complete(prompt, 4));
      ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("phase":"idle")"));
    }
    EXPECT_FALSE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
  }
};

class RegisteredElasticTextServiceTest : public ElasticTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

TEST_F(ElasticTextServiceTest,
       PrefixNativeBackingRetiresBeforeNextReservation) {
  CheckPrefixBackingRetirement(/*trim_first=*/false);
}

TEST_F(ElasticTextServiceTest, PrefixNativeBackingRetiresBeforeTrim) {
  CheckPrefixBackingRetirement(/*trim_first=*/true);
}

TEST_F(RegisteredElasticTextServiceTest,
       ColdStateCommitDoesNotBlockWarmHttpInferenceOnOneWorker) {
  CheckColdStateAdmission(/*cancel=*/false);
}

TEST_F(RegisteredElasticTextServiceTest,
       UnregisterJoinsPendingStateWithoutBlockingWarmInference) {
  CheckColdStateAdmission(/*cancel=*/true);
}

TEST_F(RegisteredElasticTextServiceTest,
       IdleParametersReleaseWithoutBlockingPeersOrDiscardingHistory) {
  StartManaged({.parameter_timeout = iree_make_duration_ms(1)});
  for (const char* name : {"fixture-text", "warm"}) {
    ASSERT_EQ(
        Read(Send(Declaration(name, directory_), "", "/v1/models/register"))
            .find("HTTP/1.1 202"),
        0u);
    ASSERT_NO_FATAL_FAILURE(WaitModel(name, R"("registration":"ready")"));
    ASSERT_EQ(Read(Send("{\"model\":\"" + std::string(name) + "\"}", "",
                        "/v1/models/pin"))
                  .find("HTTP/1.1 200"),
              0u);
  }
  const std::string prompt = "abxa";
  const std::string request = Request(Message("user", prompt), 2);
  const std::string first =
      Content(Read(Send(request, "X-Loom-Session: retained\r\n")));
  ASSERT_EQ(first, Oracle::Complete(prompt, 2));
  ASSERT_EQ(
      Read(Send("",
                "X-Loom-Model: fixture-text\r\nX-Loom-Session: retained\r\n",
                "/v1/checkpoints/base"))
          .find("HTTP/1.1 201"),
      0u);
  std::string warm = request;
  warm.replace(warm.find("fixture-text"), 12, "warm");
  ASSERT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))), first);
  ASSERT_NO_FATAL_FAILURE(WaitModel("fixture-text", R"("phase":"idle")"));
  ASSERT_NO_FATAL_FAILURE(WaitModel("warm", R"("phase":"idle")"));
  {
    loom_serve::testing::MaintenanceGate gate(
        loom_serve_device_storage_pool(device_)->maintenance);
    ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unpin"))
                  .find("HTTP/1.1 200"),
              0u);
    // This poll does not keep parameters hot. Expiry schedules native release
    // without closing admission or blocking the sole serving task worker.
    const auto expired =
        WaitModel("fixture-text", R"("parameters":{"active":false)");
    ASSERT_NE(expired.find(R"("deactivations":1)"), std::string::npos)
        << expired;
    EXPECT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))), first);
    EXPECT_EQ(Read(Send("", "", "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    const auto peer = WaitModel("warm", R"("phase":"idle")");
    EXPECT_NE(peer.find(R"("parameters":{"active":true)"), std::string::npos);
    EXPECT_NE(peer.find(R"("deactivations":0)"), std::string::npos);
  }
  const auto continuation =
      Request(Message("user", prompt) + "," + Message("assistant", first) +
                  "," + Message("user", "b"),
              2);
  const auto resumed = Read(Send(continuation, "X-Loom-Session: retained\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(prompt + first + "#b", 2));
  EXPECT_NE(resumed.find(R"("cached_tokens":5)"), std::string::npos) << resumed;
  const auto rewind = Read(Send(
      continuation, "X-Loom-Session: fork\r\nX-Loom-Checkpoint: base\r\n"));
  EXPECT_EQ(Content(rewind), Oracle::Complete(prompt + first + "#b", 2));
  EXPECT_NE(rewind.find(R"("cached_tokens":5)"), std::string::npos) << rewind;
  EXPECT_EQ(Read(Send("", "X-Loom-Model: fixture-text\r\n",
                      "/v1/checkpoints/base", "DELETE"))
                .find("HTTP/1.1 200"),
            0u);
  const auto status = Read(Send("", "", "/v1/status", "GET"));
  EXPECT_NE(status.find(R"("idle":{"workspace_ms":0,"parameter_ms":1})"),
            std::string::npos);
}

TEST_F(ElasticTextServiceTest,
       CompactionYieldsNativeReleasesAndPreservesSharedCheckpoint) {
  const iree_hal_physical_memory_params_t parameters = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_device_size_t minimum = 0;
  iree_device_size_t slab_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      iree_hal_device_allocator(loom_serve_device_handle(device_)), parameters,
      &minimum, &slab_size));
  auto options = ModelOptions();
  // Four full reservations cover two slabs of KV plus the private prefix.
  // Only three tokens in the upper half survive; no long prompt is executed.
  options.context_capacity = slab_size / 8;
  options.pool_capacity = 4 * options.context_capacity;
  loom_serve_text_model_t* sparse = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &sparse));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)>
      sparse_owner(sparse, release_model);
  for (size_t i = 0; i < 4; ++i) {
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        loom_serve_text_model_row(sparse, i), LOOM_SERVE_TEXT_RESERVE_FRESH,
        nullptr, options.context_capacity, &admitted));
    ASSERT_TRUE(admitted);
  }
  auto* row = loom_serve_text_model_row(sparse, 2);
  const int32_t tokens[] = {0, 1, 2};
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
  Oracle reference;
  reference.Consume("abx");
  ASSERT_EQ(loom_serve_text_row_token(row), reference.Predict());
  for (size_t i : {0u, 1u, 3u}) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(sparse, i)));
  }
  loom_serve_text_row_release_reservation(row);
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  const auto before = loom_serve_text_model_memory_statistics(sparse);
  ASSERT_EQ(before.committed_bytes, 3 * slab_size);

  auto* warm = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(warm, IREE_ARRAYSIZE(tokens), tokens));
  Oracle warm_reference = reference;
  const auto advance_warm = [&] {
    const int32_t token = 1;
    IREE_ASSERT_OK(loom_serve_text_row_prefill(warm, 1, &token));
    warm_reference.Consume(token);
    EXPECT_EQ(loom_serve_text_row_token(warm), warm_reference.Predict());
  };
  loom_serve_text_trim_result_t result;
  loom_serve_execution_frontier_t frontier;
  const auto wait_frontier = [&] {
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        {IREE_ARRAYSIZE(frontier.semaphores), frontier.semaphores,
         frontier.values},
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  };
  {
    loom_serve::testing::MaintenanceGate gate(
        loom_serve_device_storage_pool(device_)->maintenance);
    IREE_ASSERT_OK(
        loom_serve_text_model_submit_trim(sparse, &result, &frontier));
    const auto pending = loom_serve_text_model_memory_statistics(sparse);
    EXPECT_EQ(pending.committed_bytes, before.committed_bytes);
    EXPECT_EQ(pending.releasing_bytes, slab_size);
    ASSERT_NO_FATAL_FAILURE(advance_warm());
  }
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  bool complete = true;
  IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result, &complete,
                                                     &frontier));
  ASSERT_FALSE(complete);
  EXPECT_EQ(result.moved_blocks, 1u);
  EXPECT_EQ(result.copied_bytes, 16u);
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  {
    loom_serve::testing::MaintenanceGate gate(
        loom_serve_device_storage_pool(device_)->maintenance);
    IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result,
                                                       &complete, &frontier));
    ASSERT_FALSE(complete);
    const auto pending = loom_serve_text_model_memory_statistics(sparse);
    EXPECT_EQ(pending.committed_bytes, 2 * slab_size);
    EXPECT_EQ(pending.releasing_bytes, slab_size);
    ASSERT_NO_FATAL_FAILURE(advance_warm());
  }
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result, &complete,
                                                     &frontier));
  ASSERT_TRUE(complete);
  EXPECT_EQ(result.released_bytes, 2 * slab_size);
  EXPECT_EQ(loom_serve_text_model_memory_statistics(sparse).committed_bytes,
            slab_size);

  // The compacted page is still shared. Appending detaches its partial tail;
  // restoring the immutable endpoint must retain the original prediction.
  const int32_t a = 0;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &a));
  Oracle continued = reference;
  continued.Consume(a);
  EXPECT_EQ(loom_serve_text_row_token(row), continued.Predict());
  auto* fork = loom_serve_text_model_row(sparse, 0);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  EXPECT_EQ(loom_serve_text_row_token(fork), reference.Predict());
  const int32_t b = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(fork, 1, &b));
  reference.Consume(b);
  EXPECT_EQ(loom_serve_text_row_token(fork), reference.Predict());
  loom_serve_text_checkpoint_release(checkpoint);
}

TEST_F(ElasticTextServiceTest, RetirementJoinsAcceptedStateTrim) {
  auto options = ModelOptions();
  loom_serve_text_model_t* retiring = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &retiring));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> owner(
      retiring, release_model);
  const int32_t tokens[] = {0, 1, 2};
  auto* row = loom_serve_text_model_row(retiring, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
  IREE_ASSERT_OK(loom_serve_text_row_reset(row));
  auto* warm = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(warm, IREE_ARRAYSIZE(tokens), tokens));
  loom_serve_text_trim_result_t result;
  loom_serve_execution_frontier_t frontier;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  {
    loom_serve::testing::MaintenanceGate gate(
        loom_serve_device_storage_pool(device_)->maintenance);
    IREE_ASSERT_OK(
        loom_serve_text_model_submit_trim(retiring, &result, &frontier));
    EXPECT_GT(loom_serve_text_model_memory_statistics(retiring).releasing_bytes,
              0u);
    loom_serve_text_model_retire(
        retiring,
        [](void* payload) {
          iree_notification_post(static_cast<iree_notification_t*>(payload),
                                 IREE_ALL_WAITERS);
        },
        &notification);
    EXPECT_FALSE(loom_serve_text_model_is_retired(retiring));
    const int32_t token = 1;
    IREE_ASSERT_OK(loom_serve_text_row_prefill(warm, 1, &token));
    Oracle reference;
    reference.Consume("abxb");
    EXPECT_EQ(loom_serve_text_row_token(warm), reference.Predict());
  }
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      retiring, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(owner.release()));
  iree_notification_deinitialize(&notification);
}

TEST_F(ElasticTextServiceTest, RetirementOwnsAcceptedCheckpointCapture) {
  auto options = ModelOptions();
  const std::string prompt = "abxabbaaxbabx";
  std::vector<int32_t> tokens;
  for (const char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  auto* row = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, tokens.size(), tokens.data()));
  Oracle reference;
  reference.Consume(prompt);
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_text_checkpoint_submit_suspend(checkpoint, &frontier));

  // Retire after acceptance, without completing the capture or resetting its
  // idle row. Model teardown owns the snapshot and its queued buffer views.
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  loom_serve_text_model_retire(
      model_,
      [](void* payload) {
        iree_notification_post(static_cast<iree_notification_t*>(payload),
                               IREE_ALL_WAITERS);
      },
      &notification);
  EXPECT_TRUE(iree_notification_await(
      &notification,
      [](void* payload) {
        return loom_serve_text_model_is_retired(
            static_cast<loom_serve_text_model_t*>(payload));
      },
      model_, iree_infinite_timeout()));
  IREE_ASSERT_OK(loom_serve_text_model_free_retired(model_));
  model_ = nullptr;
  iree_notification_deinitialize(&notification);
  const auto memory = loom_serve_device_memory_statistics(device_);
  EXPECT_EQ(memory.retained.reserved_bytes, 0u);
  EXPECT_EQ(memory.retained.committed_bytes, 0u);

  // Same host, queues, and parameter pool remain available to the replacement.
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &model_));
  row = loom_serve_text_model_row(model_, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, tokens.size(), tokens.data()));
  EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
}

TEST_F(ElasticTextServiceTest,
       SuspendedCheckpointSurvivesRestoreReleaseAndNameReuse) {
  Start();
  const std::string base = "abxa";
  const std::string first = Content(Read(
      Send(Request(Message("user", base), 2), "X-Loom-Session: root\r\n")));
  ASSERT_EQ(first, Oracle::Complete(base, 2));
  const std::string history =
      Message("user", base) + "," + Message("assistant", first);
  const std::string pinned =
      Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base"));
  ASSERT_EQ(pinned.find("HTTP/1.1 201"), 0u) << pinned;
  EXPECT_NE(pinned.find("\"host_snapshot_bytes\":0}"), std::string::npos);
  std::string snapshot;
  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const std::string suspended =
        Read(Send("", "", "/v1/checkpoints/base/suspend"));
    ASSERT_EQ(suspended.find("HTTP/1.1 200"), 0u) << suspended;
    EXPECT_NE(suspended.find("\"host_snapshot_bytes\":"), std::string::npos);
    EXPECT_EQ(suspended.find("\"host_snapshot_bytes\":0}"), std::string::npos);
    if (i == 0) {
      snapshot = suspended;
    }
    EXPECT_EQ(suspended, snapshot);
    EXPECT_EQ(Read(Send("", "", "/v1/checkpoints/base/suspend")), snapshot);
    const std::string response =
        Read(Send(Request(history + "," + Message("user", "bax")),
                  "X-Loom-Session: fork\r\nX-Loom-Checkpoint: base\r\n"));
    EXPECT_EQ(Content(response), Oracle::Complete(base + first + "#bax", 16));
    EXPECT_NE(response.find("\"cached_tokens\":5"), std::string::npos);
  }
  // Release a cold entry, then reuse its public name and native slot. The new
  // pin has no stale host snapshot even though the preceding one was suspended.
  EXPECT_EQ(Read(Send("", "", "/v1/checkpoints/base/suspend")), snapshot);
  const std::string released =
      Read(Send("", "", "/v1/checkpoints/base", "DELETE"));
  ASSERT_EQ(released.find("HTTP/1.1 200"), 0u) << released;
  EXPECT_NE(released.find("\"pinned\":false,\"host_snapshot_bytes\":0}"),
            std::string::npos);
  EXPECT_EQ(Read(Send("", "X-Loom-Session: root\r\n", "/v1/checkpoints/base")),
            pinned);
}

TEST_F(ElasticTextServiceTest, EndpointAliasesShareOneSuspendedPayload) {
  auto* source = loom_serve_text_model_row(model_, 0);
  auto* target = loom_serve_text_model_row(model_, 1);
  const std::vector<int32_t> tokens = {0, 1, 2, 0};
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(source, tokens.size(), tokens.data()));
  const auto prediction = loom_serve_text_row_token(source);
  // Repeat after the final release to exercise actual endpoint slot reuse.
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    loom_serve_text_checkpoint_t* endpoint = nullptr;
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &endpoint));
    ASSERT_NE(endpoint, nullptr);
    loom_serve_text_checkpoint_retain(endpoint);
    IREE_ASSERT_OK(loom_serve_text_checkpoint_suspend(endpoint));
    const auto bytes = loom_serve_text_checkpoint_suspended_bytes(endpoint);
    ASSERT_GT(bytes, 0u);
    EXPECT_EQ(loom_serve_text_model_checkpoint_suspended_bytes(model_), bytes);
    loom_serve_text_checkpoint_release(endpoint);
    EXPECT_EQ(loom_serve_text_model_checkpoint_suspended_bytes(model_), bytes);
    bool restored = false;
    IREE_ASSERT_OK(
        loom_serve_text_row_try_restore(target, endpoint, &restored));
    ASSERT_TRUE(restored);
    EXPECT_EQ(loom_serve_text_row_position(target), tokens.size());
    EXPECT_EQ(loom_serve_text_row_token(target), prediction);
    EXPECT_EQ(loom_serve_text_model_checkpoint_suspended_bytes(model_), 0u);
    loom_serve_text_checkpoint_release(endpoint);
    IREE_ASSERT_OK(loom_serve_text_row_reset(target));
  }
}

}  // namespace
