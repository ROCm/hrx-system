// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/testing/native_allocator.h"
#include "loom/serve/testing/service_application.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::OwnerTurn;
using loom_serve::testing::text::SourceTextServiceTest;

class ElasticTextServiceTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    // Cold-allocation holds must not depend on a previous test's cached slabs.
    auto* pool = loom_serve_pool_memory(storage_owner_);
    loom_serve_memory_pool_trim(pool, 0);
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }

  void TearDown() override {
    SourceTextServiceTest::TearDown();
    // Pool ownership returns to the outer fixture only after the router joins.
    loom_serve_memory_operation_await(filler_operation_);
    IREE_EXPECT_OK(loom_serve_memory_operation_query_status(filler_operation_));
    loom_serve_memory_operation_free(filler_operation_);
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(filler_));
    iree_hal_semaphore_release(filler_ready_);
  }

  void WaitFrontier(const loom_serve_execution_frontier_t& frontier) {
    auto* workspace = loom_serve_pool_workspace(storage_owner_);
    for (size_t i = 0; i < IREE_ARRAYSIZE(frontier.semaphores); ++i) {
      IREE_ASSERT_OK(loom_serve_workspace_wait(
          1, &workspace, frontier.semaphores[i], frontier.values[i]));
    }
  }

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
    const iree_device_size_t slab_size =
        loom_serve_test_application_options_default().slab_size;
    const iree_device_size_t native_size =
        loom_serve_pool_state_geometry(storage_owner_).minimum_size;
    auto options = ModelOptions();
    options.context_capacity = slab_size / 8;
    options.pool_capacity = 4 * options.context_capacity;
    loom_serve_text_model_t* source = nullptr;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &source));
    const auto release_model = [](loom_serve_text_model_t* model) {
      IREE_EXPECT_OK(loom_serve_text_model_free(model));
    };
    const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)>
        owner(source, release_model);
    // Fill exactly one four-byte-per-token KV slab. A new COW tail must acquire
    // another full KV slab; recurrent state occupies one native extent instead
    // of being rounded to the KV replacement size.
    const size_t extents[] = {4, options.context_capacity,
                              slab_size / 4 - 4 - options.context_capacity};
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
              slab_size + native_size);
    // Occupy the shared parent's remaining slices without growing physical
    // commitment. The COW page below must then acquire a new native parent.
    auto* memory_pool = loom_serve_pool_memory(storage_owner_);
    const auto before_filler = loom_serve_memory_pool_statistics(memory_pool);
    ASSERT_GT(before_filler.cached_bytes, 0u);
    ASSERT_EQ(before_filler.cached_bytes % native_size, 0u);
    loom_runtime_storage_statistics_t filler_statistics = {0};
    loom_serve_virtual_buffer_t* filler = nullptr;
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        memory_pool, loom_serve_pool_state_geometry(storage_owner_),
        before_filler.cached_bytes, 256, &filler_statistics, &filler));
    const auto free_filler = [](loom_serve_virtual_buffer_t* buffer) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
    };
    std::unique_ptr<loom_serve_virtual_buffer_t, decltype(free_filler)>
        filler_owner(filler, free_filler);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(
        filler, 0, before_filler.cached_bytes));
    const auto after_filler = loom_serve_memory_pool_statistics(memory_pool);
    EXPECT_EQ(after_filler.committed_bytes, before_filler.committed_bytes);
    EXPECT_EQ(after_filler.cached_bytes, 0u);
    loom_serve_text_checkpoint_t* checkpoint = nullptr;
    loom_serve_execution_frontier_t frontier;
    {
      loom_serve::testing::ScopedNativeHold gate(device_);
      IREE_ASSERT_OK(
          loom_serve_text_row_submit_pin(row, &checkpoint, &frontier));
      gate.WaitEntered();
      ASSERT_NE(checkpoint, nullptr);
      EXPECT_EQ(loom_serve_text_model_memory_statistics(source).pending_bytes,
                native_size);
      const int32_t token = 1;
      IREE_ASSERT_OK(loom_serve_text_row_prefill(warm, 1, &token));
      Oracle peer;
      peer.Consume("abxb");
      EXPECT_EQ(loom_serve_text_row_token(warm), peer.Predict());
    }
    ASSERT_NO_FATAL_FAILURE(WaitFrontier(frontier));
    filler_owner.reset();
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
      ASSERT_NO_FATAL_FAILURE(
          WaitModelState(name, R"("registration":"ready")"));
    }
    // Explicit parameter activation leaves the cold model's KV untouched.
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "", "/v1/models/drain"))
                  .find("HTTP/1.1 200"),
              0u);
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "", "/v1/models/activate"))
                  .find("HTTP/1.1 202"),
              0u);
    const auto activated =
        WaitModelState("cold", R"("parameter_operation":"idle")");
    ASSERT_NE(activated.find(
                  R"("primary":{"phase":"ready","pinned":false,"active":true)"),
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
    ASSERT_NO_FATAL_FAILURE(
        WaitModelState("fixture-text", R"("phase":"idle")"));
    ASSERT_FALSE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
    // Keep the cold request on the native allocation path even when warm
    // models leave reusable slices in a shared physical parent.
    auto* memory_pool = loom_serve_pool_memory(storage_owner_);
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &filler_ready_));
    loom_runtime_storage_statistics_t before_filler = {};
    // The fixture has one worker: owner turns exclude router pool accesses,
    // while native readiness is awaited only by the outer test thread.
    ASSERT_EQ(WorkerCount(), 1u);
    ASSERT_NO_FATAL_FAILURE(
        OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
          before_filler = loom_serve_memory_pool_statistics(memory_pool);
          ASSERT_GT(before_filler.cached_bytes, 0u);
          IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
              memory_pool, loom_serve_pool_state_geometry(storage_owner_),
              before_filler.cached_bytes, 256, &filler_statistics_, &filler_));
          const loom_serve_memory_range_t range = {filler_, 0,
                                                   before_filler.cached_bytes};
          loom_serve_memory_admission_t admission;
          IREE_ASSERT_OK(loom_serve_memory_pool_commit_async(
              memory_pool, 1, &range, filler_ready_, 1, &admission,
              &filler_operation_));
          ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
        }));
    if (HasFatalFailure()) {
      return;
    }
    IREE_ASSERT_OK(iree_hal_semaphore_wait(
        filler_ready_, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    ASSERT_NO_FATAL_FAILURE(
        OwnerTurn::Run(loom_serve_host_executor(host_), [&] {
          ASSERT_TRUE(
              loom_serve_memory_operation_is_complete(filler_operation_));
          IREE_ASSERT_OK(
              loom_serve_memory_operation_query_status(filler_operation_));
          const auto after_filler =
              loom_serve_memory_pool_statistics(memory_pool);
          EXPECT_EQ(after_filler.committed_bytes,
                    before_filler.committed_bytes);
          EXPECT_EQ(after_filler.cached_bytes, 0u);
        }));
    if (HasFatalFailure()) {
      return;
    }
    int cold = -1;
    {
      // The application has one task worker. Only the HAL's existing native
      // cold executor is held; the normal router and warm model must progress.
      loom_serve::testing::ScopedNativeHold gate(device_);
      cold = Send(cold_request);
      gate.WaitEntered();
      ASSERT_FALSE(http_.failed());
      ASSERT_TRUE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
      if (cancel) {
        // A disconnected peer does not release native mapping borrowers.
        ASSERT_EQ(close(cold), 0);
        ASSERT_EQ(Read(Send(R"({"model":"cold","mode":"abort"})", "",
                            "/v1/models/unregister"))
                      .find("HTTP/1.1 202"),
                  0u);
        ASSERT_NO_FATAL_FAILURE(
            WaitModelState("cold", R"("admission":"removing")"));
        EXPECT_TRUE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
      }
      EXPECT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))),
                Oracle::Complete(prompt, 4));
      EXPECT_EQ(Read(Send("", "", "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    }
    if (cancel) {
      ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("cold"));
    } else {
      EXPECT_EQ(Content(Read(cold)), Oracle::Complete(prompt, 4));
      ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("phase":"idle")"));
    }
    EXPECT_FALSE(HasPendingBacking(Read(Send("", "", "/v1/status", "GET"))));
  }

  // Spare shared backing held through router shutdown in cold-admission cases.
  loom_serve_virtual_buffer_t* filler_ = nullptr;
  // Accounting outlives the filler's asynchronous commitment and destruction.
  loom_runtime_storage_statistics_t filler_statistics_ = {};
  // Accepted commitment retained until the serialized router owner has joined.
  loom_serve_memory_operation_t* filler_operation_ = nullptr;
  // Native readiness waited by the outer test without publishing pool state.
  iree_hal_semaphore_t* filler_ready_ = nullptr;
};

class RegisteredElasticTextServiceTest : public ElasticTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }
};

TEST_F(ElasticTextServiceTest,
       PrefixNativeBackingRetiresBeforeNextReservation) {
  CheckPrefixBackingRetirement(/*trim_first=*/false);
}

TEST_F(ElasticTextServiceTest,
       AutomaticAddressabilityDoesNotScaleLiveBackingOrDirectories) {
  struct Observation {
    // Address space retained independently of physical backing.
    uint64_t reserved_bytes;
    // Live mapped backing before the model retires.
    uint64_t mapped_bytes;
    // Materialized page identities, not the maximum page namespace.
    uint32_t directory_capacity;
    // Cold compaction scratch remains absent before compaction is requested.
    uint32_t compaction_capacity;
  };
  const auto observe = [&](iree_host_size_t capacity,
                           Observation* observation) {
    IREE_ASSERT_OK(loom_serve_test_pool_trim_and_drain(storage_owner_, 0));
    auto* memory_pool = loom_serve_pool_memory(storage_owner_);
    auto options = ModelOptions();
    options.pool_capacity = 0;
    // The source fixture declares four payload bytes per logical token. The
    // real preparation path, not this caller, converts the byte boundary.
    options.cache_limit_bytes = uint64_t{capacity} * sizeof(int32_t);
    loom_serve_text_model_t* source = nullptr;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &source));
    const auto release_model = [](loom_serve_text_model_t* model) {
      IREE_EXPECT_OK(loom_serve_text_model_free(model));
    };
    std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> owner(
        source, release_model);
    EXPECT_EQ(loom_serve_text_model_pool_usage(source).available, capacity);
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        loom_serve_text_model_row(source, 0), LOOM_SERVE_TEXT_RESERVE_FRESH,
        nullptr, 4, &admitted));
    ASSERT_TRUE(admitted);
    const int32_t tokens[] = {0, 1, 2};
    auto* row = loom_serve_text_model_row(source, 0);
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
    Oracle reference;
    reference.Consume("abx");
    EXPECT_EQ(loom_serve_text_row_token(row), reference.Predict());
    const auto after = loom_serve_text_model_memory_statistics(source);
    EXPECT_EQ(loom_serve_text_model_pool_usage(source).available, capacity - 4);
    IREE_ASSERT_OK(loom_serve_memory_pool_await(memory_pool));
    *observation = {
        .reserved_bytes = after.reserved_bytes,
        .mapped_bytes = after.committed_bytes,
        .directory_capacity = source->state.cache.pool.capacity,
        .compaction_capacity = source->state.cache.destination_capacity,
    };
    owner.reset();
    IREE_ASSERT_OK(loom_serve_test_pool_trim_and_drain(storage_owner_, 0));
  };

  Observation small, maximum;
  ASSERT_NO_FATAL_FAILURE(observe(256, &small));
  ASSERT_NO_FATAL_FAILURE(observe(128 * 1024 * 1024, &maximum));
  const auto geometry = loom_serve_pool_state_geometry(storage_owner_);
  EXPECT_GT(maximum.reserved_bytes, small.reserved_bytes);
  // The tiny source can co-locate cache and private bytes in one mapping at
  // the smaller stride. Separating them at the maximum may add a mapping, but
  // both live footprints still fit the same physical parent.
  EXPECT_EQ(iree_device_align(maximum.mapped_bytes, geometry.backing_size),
            iree_device_align(small.mapped_bytes, geometry.backing_size));
  EXPECT_EQ(maximum.directory_capacity, small.directory_capacity);
  EXPECT_EQ(maximum.compaction_capacity, 0u);
  EXPECT_EQ(small.compaction_capacity, 0u);
}

TEST_F(ElasticTextServiceTest, PrefixNativeBackingRetiresBeforeTrim) {
  CheckPrefixBackingRetirement(/*trim_first=*/true);
}

TEST_F(ElasticTextServiceTest, TrimPreservesSparseCacheWithoutPhysicalSavings) {
  const uint64_t slab_size =
      loom_serve_test_application_options_default().slab_size;
  const uint64_t mapping_size =
      loom_serve_pool_state_geometry(storage_owner_).mapping_size;
  // Packing within one slab and replacing one live slab with another both
  // leave the same footprint. Neither justifies moving a shared cache page.
  for (uint64_t hole_extent : {UINT64_C(32), slab_size / 4}) {
    auto options = ModelOptions();
    options.context_capacity = slab_size / 8;
    options.pool_capacity = slab_size / 2;
    loom_serve_text_model_t* sparse = nullptr;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(
        device_, storage_owner_, &options, host_allocator_, &sparse));
    const auto release_model = [](loom_serve_text_model_t* model) {
      IREE_EXPECT_OK(loom_serve_text_model_free(model));
    };
    const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)>
        owner(sparse, release_model);
    auto* hole = loom_serve_text_model_row(sparse, 0);
    auto* source = loom_serve_text_model_row(sparse, 1);
    auto* upper_hole = loom_serve_text_model_row(sparse, 2);
    const uint64_t first_hole_extent =
        iree_min(hole_extent, options.context_capacity);
    bool admitted = false;
    IREE_ASSERT_OK(
        loom_serve_text_row_try_reserve(hole, LOOM_SERVE_TEXT_RESERVE_FRESH,
                                        nullptr, first_hole_extent, &admitted));
    ASSERT_TRUE(admitted);
    if (hole_extent > first_hole_extent) {
      IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
          upper_hole, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr,
          hole_extent - first_hole_extent, &admitted));
      ASSERT_TRUE(admitted);
    }
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        source, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &admitted));
    ASSERT_TRUE(admitted);
    const int32_t tokens[] = {0, 1, 2};
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(source, IREE_ARRAYSIZE(tokens), tokens));
    loom_serve_text_checkpoint_t* checkpoint = nullptr;
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &checkpoint));
    ASSERT_NE(checkpoint, nullptr);
    std::unique_ptr<loom_serve_text_checkpoint_t,
                    decltype(&loom_serve_text_checkpoint_release)>
        endpoint(checkpoint, loom_serve_text_checkpoint_release);
    IREE_ASSERT_OK(loom_serve_text_row_reset(hole));
    IREE_ASSERT_OK(loom_serve_text_row_reset(upper_hole));
    loom_serve_text_trim_result_t trim;
    IREE_ASSERT_OK(loom_serve_text_model_trim(sparse, &trim));
    EXPECT_EQ(trim.moved_blocks, 0u);
    EXPECT_EQ(trim.copied_bytes, 0u);
    EXPECT_EQ(trim.released_bytes, hole_extent == 32 ? 0 : slab_size);
    EXPECT_EQ(loom_serve_text_model_memory_statistics(sparse).committed_bytes,
              2 * mapping_size);
    IREE_ASSERT_OK(
        loom_serve_text_row_try_restore(hole, checkpoint, &admitted));
    ASSERT_TRUE(admitted);
    Oracle oracle;
    oracle.Consume("abxb");
    const int32_t suffix = 1;
    for (auto* branch : {source, hole}) {
      IREE_ASSERT_OK(loom_serve_text_row_prefill(branch, 1, &suffix));
      EXPECT_EQ(loom_serve_text_row_token(branch), oracle.Predict());
    }
  }
}

TEST_F(ElasticTextServiceTest, PrefixTailRetiresBeforeCompactionAndPeerReuse) {
  const uint64_t slab_size =
      loom_serve_test_application_options_default().slab_size;
  auto options = ModelOptions();
  options.context_capacity = slab_size / 8;
  options.pool_capacity = slab_size / 2;
  loom_serve_text_model_t* sparse = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &sparse));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> owner(
      sparse, release_model);
  auto* anchor = loom_serve_text_model_row(sparse, 3);
  bool anchored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      anchor, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 4, &anchored));
  ASSERT_TRUE(anchored);
  auto* hole = loom_serve_text_model_row(sparse, 0);
  auto* source = loom_serve_text_model_row(sparse, 1);
  auto* peer = loom_serve_text_model_row(sparse, 2);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      hole, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, options.context_capacity,
      &admitted));
  ASSERT_TRUE(admitted);
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      peer, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr,
      options.context_capacity - 4, &admitted));
  ASSERT_TRUE(admitted);
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      source, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &admitted));
  ASSERT_TRUE(admitted);
  const int32_t tokens[] = {0, 1, 2};
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(source, IREE_ARRAYSIZE(tokens), tokens));
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  std::unique_ptr<loom_serve_text_checkpoint_t,
                  decltype(&loom_serve_text_checkpoint_release)>
      endpoint(checkpoint, loom_serve_text_checkpoint_release);

  // The active pin copies the partial page into a private continuation. Free
  // lower IDs so compaction moves both the endpoint and the reserved writer
  // before the writer's next epoch. Its completed copy-source reference must
  // not survive that relocation under the old identity.
  IREE_ASSERT_OK(loom_serve_text_row_reset(hole));
  IREE_ASSERT_OK(loom_serve_text_row_reset(peer));
  loom_serve_text_trim_result_t trim;
  IREE_ASSERT_OK(loom_serve_text_model_trim(sparse, &trim));
  ASSERT_GT(trim.moved_blocks, 0u);
  EXPECT_EQ(trim.released_bytes, slab_size);
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      peer, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &admitted));
  ASSERT_TRUE(admitted);
  const auto available = loom_serve_text_model_pool_usage(sparse).available;
  const int32_t suffix = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(source, 1, &suffix));
  Oracle oracle;
  oracle.Consume("abxb");
  EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  // Retirement must not release the old physical ID now owned by the peer.
  ASSERT_EQ(loom_serve_text_model_pool_usage(sparse).available, available);

  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(peer, IREE_ARRAYSIZE(tokens), tokens));
  Oracle prefix;
  prefix.Consume("abx");
  EXPECT_EQ(loom_serve_text_row_token(peer), prefix.Predict());
  IREE_ASSERT_OK(loom_serve_text_model_trim(sparse, &trim));
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      hole, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint, 16, &admitted));
  ASSERT_TRUE(admitted);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(hole, 1, &suffix));
  EXPECT_EQ(loom_serve_text_row_token(hole), oracle.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_reset(hole));
  IREE_ASSERT_OK(loom_serve_text_row_reset(source));
  IREE_ASSERT_OK(loom_serve_text_row_reset(peer));
  IREE_ASSERT_OK(loom_serve_text_row_reset(anchor));
  EXPECT_EQ(loom_serve_text_model_pool_usage(sparse).available,
            options.pool_capacity - 4);
  endpoint.reset();
  EXPECT_EQ(loom_serve_text_model_pool_usage(sparse).available,
            options.pool_capacity);
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
    ASSERT_NO_FATAL_FAILURE(WaitModelState(name, R"("registration":"ready")"));
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
  ASSERT_NO_FATAL_FAILURE(WaitModelState("fixture-text", R"("phase":"idle")"));
  ASSERT_NO_FATAL_FAILURE(WaitModelState("warm", R"("phase":"idle")"));
  {
    ASSERT_EQ(Read(Send(R"({"model":"fixture-text"})", "", "/v1/models/unpin"))
                  .find("HTTP/1.1 200"),
              0u);
    // This poll does not keep parameters hot. Expiry returns backing to the
    // shared cache without closing admission or losing retained history.
    const auto expired = WaitModelState(
        "fixture-text",
        R"("primary":{"phase":"ready","pinned":false,"active":false)");
    ASSERT_NE(expired.find(R"("deactivations":1)"), std::string::npos)
        << expired;
    EXPECT_EQ(Content(Read(Send(warm, "X-Loom-Session: warm\r\n"))), first);
    EXPECT_EQ(Read(Send("", "", "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    const auto peer = WaitModelState("warm", R"("phase":"idle")");
    EXPECT_NE(
        peer.find(R"("primary":{"phase":"ready","pinned":true,"active":true)"),
        std::string::npos);
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
       CompactionReusesPackedParentAndPreservesSharedCheckpoint) {
  const iree_device_size_t slab_size =
      loom_serve_test_application_options_default().slab_size;
  const iree_device_size_t mapping_size =
      loom_serve_pool_state_geometry(storage_owner_).mapping_size;
  auto options = ModelOptions();
  // Eight full reservations touch four KV slabs. Retain one low page and one
  // upper page: dead bulk backing retires while both short live tails remain
  // packed in the shared parent. Compaction returns one mapping for reuse
  // without acquiring or releasing another physical parent.
  options.context_capacity = slab_size / 8;
  options.pool_capacity = slab_size;
  options.row_count = 8;
  loom_serve_text_model_t* sparse = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &sparse));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  const std::unique_ptr<loom_serve_text_model_t, decltype(release_model)>
      sparse_owner(sparse, release_model);
  for (size_t i = 0; i < options.row_count; ++i) {
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        loom_serve_text_model_row(sparse, i), LOOM_SERVE_TEXT_RESERVE_FRESH,
        nullptr, options.context_capacity, &admitted));
    ASSERT_TRUE(admitted);
  }
  auto* row = loom_serve_text_model_row(sparse, 4);
  const int32_t tokens[] = {0, 1, 2};
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
  Oracle reference;
  reference.Consume("abx");
  ASSERT_EQ(loom_serve_text_row_token(row), reference.Predict());
  auto* anchor = loom_serve_text_model_row(sparse, 0);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(anchor, IREE_ARRAYSIZE(tokens), tokens));
  loom_serve_text_row_release_reservation(anchor);
  for (size_t i : {1u, 2u, 3u, 5u, 6u, 7u}) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(sparse, i)));
  }
  loom_serve_text_row_release_reservation(row);
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  const auto before = loom_serve_text_model_memory_statistics(sparse);
  ASSERT_EQ(before.committed_bytes, 4 * slab_size + 2 * mapping_size);

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
    ASSERT_NO_FATAL_FAILURE(WaitFrontier(frontier));
  };
  auto* pool = loom_serve_pool_memory(storage_owner_);
  IREE_ASSERT_OK(loom_serve_text_model_submit_trim(sparse, &result, &frontier));
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  {
    loom_serve::testing::ScopedNativeHold gate(device_,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    loom_serve_memory_pool_trim(pool, 0);
    gate.WaitEntered();
    const auto physical = loom_serve_memory_pool_statistics(
        loom_serve_pool_memory(storage_owner_));
    const auto returned = loom_serve_text_model_memory_statistics(sparse);
    EXPECT_EQ(returned.committed_bytes, 4 * mapping_size);
    EXPECT_EQ(returned.releasing_bytes, 0u);
    EXPECT_GE(physical.releasing_bytes, slab_size);
    ASSERT_NO_FATAL_FAILURE(advance_warm());
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  const uint64_t physical_after_bulk_trim =
      loom_serve_memory_pool_statistics(pool).committed_bytes;
  bool complete = false;
  IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result, &complete,
                                                     &frontier));
  ASSERT_FALSE(complete);
  EXPECT_EQ(result.moved_blocks, 1u);
  EXPECT_EQ(result.copied_bytes, 16u);
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result, &complete,
                                                     &frontier));
  ASSERT_FALSE(complete);
  ASSERT_NO_FATAL_FAILURE(wait_frontier());
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  IREE_ASSERT_OK(loom_serve_text_model_complete_trim(sparse, &result, &complete,
                                                     &frontier));
  ASSERT_TRUE(complete);
  const auto after = loom_serve_text_model_memory_statistics(sparse);
  EXPECT_EQ(result.released_bytes,
            before.committed_bytes - after.committed_bytes);
  EXPECT_EQ(after.committed_bytes, 3 * mapping_size);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool).committed_bytes,
            physical_after_bulk_trim);

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

TEST_F(ElasticTextServiceTest,
       RetirementJoinsStateTrimWithoutWaitingForNativeFree) {
  auto options = ModelOptions();
  // The reservation spans a full physical class so its first page enters the
  // retained cache. Native disposal then occurs at explicit pool trim.
  options.pool_capacity =
      loom_serve_test_application_options_default().slab_size / 4;
  loom_serve_text_model_t* retiring = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &retiring));
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
  IREE_ASSERT_OK(
      loom_serve_text_model_submit_trim(retiring, &result, &frontier));
  ASSERT_NO_FATAL_FAILURE(WaitFrontier(frontier));
  auto* pool = loom_serve_pool_memory(storage_owner_);
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  {
    loom_serve::testing::ScopedNativeHold gate(device_,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    loom_serve_memory_pool_trim(pool, 0);
    gate.WaitEntered();
    loom_serve_text_model_retire(
        retiring,
        [](void* payload) {
          iree_notification_post(static_cast<iree_notification_t*>(payload),
                                 IREE_ALL_WAITERS);
        },
        &notification);
    const int32_t token = 1;
    IREE_ASSERT_OK(loom_serve_text_row_prefill(warm, 1, &token));
    Oracle reference;
    reference.Consume("abxb");
    EXPECT_EQ(loom_serve_text_row_token(warm), reference.Predict());
    EXPECT_TRUE(iree_notification_await(
        &notification,
        [](void* payload) {
          return loom_serve_text_model_is_retired(
              static_cast<loom_serve_text_model_t*>(payload));
        },
        retiring, iree_infinite_timeout()));
    IREE_ASSERT_OK(loom_serve_text_model_free_retired(owner.release()));
    EXPECT_GT(loom_serve_memory_pool_statistics(
                  loom_serve_pool_memory(storage_owner_))
                  .releasing_bytes,
              0u);
  }
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
  // Physical capacity returns at the outer pool join, not model retirement.
  IREE_ASSERT_OK(loom_serve_test_pool_trim_and_drain(storage_owner_, 0));
  const auto memory = loom_serve_pool_statistics(storage_owner_);
  EXPECT_FALSE(loom_serve_pool_is_in_use(storage_owner_));
  EXPECT_EQ(memory.retained.pending_bytes, 0u);
  EXPECT_EQ(memory.retained.committed_bytes, 0u);
  EXPECT_EQ(memory.retained.releasing_bytes, 0u);

  // Same host, queues, and parameter pool remain available to the replacement.
  IREE_ASSERT_OK(loom_serve_text_model_allocate(
      device_, storage_owner_, &options, host_allocator_, &model_));
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
