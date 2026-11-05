// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <utility>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"

namespace loom_serve::testing::embedding {

class PoolBindingTest : public SourceEmbeddingServiceTest {
 protected:
  iree_host_size_t DeviceCount() const override { return 2; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceEmbeddingServiceTest::SetUp());
    interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
    loom_serve_finite_service_free(std::exchange(service_, nullptr));
    IREE_ASSERT_OK(
        loom_serve_finite_model_free(std::exchange(model_, nullptr)));
    ASSERT_NO_FATAL_FAILURE(CreatePool(&shared_));
    ASSERT_NO_FATAL_FAILURE(CreatePool(&isolated_));
    const std::string package =
        iree::testing::MakeTempFilePath("embedding_pool_binding");
    ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
    ASSERT_NO_FATAL_FAILURE(Load(package, loom_serve_device_group_at(group_, 0),
                                 shared_, &producer_));
    ASSERT_NO_FATAL_FAILURE(Load(package, device_, shared_, &peer_));
    ASSERT_NO_FATAL_FAILURE(Load(package, device_, isolated_, &model_));
    IREE_ASSERT_OK(loom_serve_finite_service_allocate(
        host_, model_, loom_serve_embedding_protocol(), IREE_SV("fixture"), 16,
        16, 0, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_finite_service_interface(service_);
    EXPECT_EQ(interface_.runtime.capacity_domain,
              loom_serve_pool_residency_cache(isolated_));
  }

  void TearDown() override {
    loom_serve_finite_result_release(std::exchange(result_, nullptr));
    if (http_.joinable()) {
      loom_serve_http_server_request_stop(server_);
      IREE_EXPECT_OK(http_.Join());
    }
    IREE_EXPECT_OK(loom_serve_finite_model_free(producer_));
    IREE_EXPECT_OK(loom_serve_finite_model_free(peer_));
    SourceEmbeddingServiceTest::TearDown();
  }

  void CreatePool(loom_serve_pool_t** out_pool) {
    const std::string name =
        "pool" + std::to_string(loom_serve_device_group_pool_count(group_));
    const loom_serve_pool_options_t options = {
        .elastic = true,
        .weight_slab_size = 4096,
        .state_slab_size = 4096,
        .memory_limit = 8192,
    };
    IREE_ASSERT_OK(loom_serve_device_group_create_pool(
        group_, {iree_make_cstring_view(name.c_str()), 3}, &options, out_pool));
  }

  void Load(const std::string& package, loom_serve_device_t* device,
            loom_serve_pool_t* pool, loom_serve_finite_model_t** out_model) {
    ModelDocument document;
    IREE_ASSERT_OK(document.Load(package + "/model.json"));
    const loom_serve_finite_model_options_t options = {
        .sources = document.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(
        device, pool, &options, host_allocator_, out_model));
    EXPECT_EQ(loom_serve_finite_model_memory_limit(*out_model), 8192u);
    EXPECT_EQ(loom_serve_residency_cache(
                  loom_serve_finite_model_residency(*out_model)),
              loom_serve_pool_residency_cache(pool));
  }

  void Prepare(loom_serve_finite_model_t* model) {
    loom_serve_embedding_request_t request = {};
    IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
        IREE_SV("{\"model\":\"fixture\",\"input\":\"a\"}"), host_allocator_,
        &request));
    loom_serve_prepared_request_t prepared = {};
    iree_status_t status = loom_serve_finite_model_prepare_request(
        model, request.base.input, &prepared);
    if (iree_status_is_ok(status)) {
      status = loom_serve_finite_model_prepare_batch(model, 1, &prepared,
                                                     &request.base.count);
    }
    iree_vm_buffer_release(prepared.bytes);
    loom_serve_embedding_request_deinitialize(&request);
    IREE_ASSERT_OK(status);
    bool complete = false;
    IREE_ASSERT_OK(
        loom_serve_finite_model_advance_batch(model, false, &complete));
    ASSERT_TRUE(complete);  // This source uses its already-compiled cold stage.
  }

  void Submit(loom_serve_finite_model_t* model, loom_serve_pool_t* pool) {
    loom_serve_execution_frontier_t ready;
    IREE_ASSERT_OK(loom_serve_finite_model_submit(model, &result_, &ready));
    auto* workspace = loom_serve_pool_workspace(pool);
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_ASSERT_OK(loom_serve_workspace_wait(workspace, ready.semaphores[i],
                                               ready.values[i]));
    }
    loom_serve_finite_model_complete(model);
    EXPECT_EQ(loom_serve_finite_result_output_length(result_), 1024u);
    loom_serve_finite_snapshot_t* snapshot = nullptr;
    IREE_ASSERT_OK(
        loom_serve_finite_result_download(result_, &snapshot, &ready));
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      status = iree_status_join(
          status, loom_serve_workspace_wait(workspace, ready.semaphores[i],
                                            ready.values[i]));
    }
    if (iree_status_is_ok(status)) {
      const auto row = loom_serve_finite_snapshot_row(snapshot, 0, 0);
      const std::array<float, 4> expected = {2, 2, 2, 2};
      EXPECT_EQ(row.data_length, sizeof(expected));
      if (row.data_length == sizeof(expected)) {
        EXPECT_EQ(memcmp(row.data, expected.data(), sizeof(expected)), 0);
      }
    }
    loom_serve_finite_snapshot_release(snapshot);
    IREE_ASSERT_OK(status);
  }

  // Explicit owner shared by producer and peer on separate execution devices.
  loom_serve_pool_t* shared_ = nullptr;
  // Second allowance; no capacity is inferred from its physical placement.
  loom_serve_pool_t* isolated_ = nullptr;
  // Source producing retained output before the HTTP service runs.
  loom_serve_finite_model_t* producer_ = nullptr;
  // Prepared source sharing the producer's admission and storage owner.
  loom_serve_finite_model_t* peer_ = nullptr;
  // Final consumer remains live after producer model retirement.
  loom_serve_finite_result_t* result_ = nullptr;
};

TEST_F(PoolBindingTest, RetainedResultsChargeOnlyTheirExplicitPool) {
  EXPECT_EQ(loom_serve_device_group_count(group_), 2u);
  EXPECT_NE(loom_serve_pool_memory(shared_), loom_serve_pool_memory(isolated_));
  ASSERT_NO_FATAL_FAILURE(Prepare(producer_));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(
      loom_serve_finite_model_residency(producer_), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Submit(producer_, shared_));
  IREE_ASSERT_OK(
      loom_serve_finite_model_free(std::exchange(producer_, nullptr)));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(loom_serve_pool_memory(shared_)));
  auto* cache = loom_serve_pool_residency_cache(shared_);
  EXPECT_TRUE(loom_serve_residency_cache_is_pending(cache));
  ASSERT_NO_FATAL_FAILURE(Prepare(peer_));
  auto* residency = loom_serve_finite_model_residency(peer_);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residency, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).admission.waiting);

  // Cancelling an unsubmitted cohort withdraws its interest, without releasing
  // anyone else's output. The next cohort must encounter the same pressure.
  loom_serve_finite_model_release_batch(peer_);
  EXPECT_FALSE(loom_serve_residency_snapshot(residency).admission.waiting);
  ASSERT_NO_FATAL_FAILURE(Prepare(peer_));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residency, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);

  StartRouter();
  const auto response =
      Read(Send(R"({"model":"fixture","input":["a","bbb"]})"));
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
  EXPECT_NE(response.find(R"("index":0,"embedding":[2,2,2,4])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("index":1,"embedding":[12,6,4,4])"),
            std::string::npos);

  // Release the result while the real router owns pool progress. Its join must
  // include these explicit owners, not just the default per-device binding.
  loom_serve_finite_result_release(std::exchange(result_, nullptr));
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  EXPECT_FALSE(loom_serve_residency_cache_is_pending(cache));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residency, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Submit(peer_, shared_));
  EXPECT_LE(loom_serve_memory_pool_statistics(loom_serve_pool_memory(shared_))
                .peak_bytes,
            8192u);
  EXPECT_LE(loom_serve_memory_pool_statistics(loom_serve_pool_memory(isolated_))
                .peak_bytes,
            8192u);
}

TEST_F(PoolBindingTest, LastResultOutlivesEveryModelInItsPool) {
  ASSERT_NO_FATAL_FAILURE(Prepare(producer_));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(
      loom_serve_finite_model_residency(producer_), &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Submit(producer_, shared_));
  IREE_ASSERT_OK(
      loom_serve_finite_model_free(std::exchange(producer_, nullptr)));
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(peer_, nullptr)));
  EXPECT_TRUE(loom_serve_pool_is_in_use(shared_));
  loom_serve_finite_result_release(std::exchange(result_, nullptr));
  EXPECT_FALSE(loom_serve_pool_is_in_use(shared_));
  // The independent model still owns its own pool on the peer's device.
  EXPECT_TRUE(loom_serve_pool_is_in_use(isolated_));
}

TEST_F(PoolBindingTest, TrimReclaimsOnlyTheSelectedPoolAndPreservesReload) {
  const auto* tokenizer = loom_serve_finite_model_tokenizer(producer_);
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(loom_serve_finite_model_tokenizer(peer_), tokenizer);
  EXPECT_EQ(loom_serve_finite_model_tokenizer(model_), tokenizer);
  for (const auto& [model, pool] :
       {std::pair{producer_, shared_}, std::pair{model_, isolated_}}) {
    ASSERT_NO_FATAL_FAILURE(Prepare(model));
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(
        loom_serve_finite_model_residency(model), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    ASSERT_NO_FATAL_FAILURE(Submit(model, pool));
    loom_serve_finite_result_release(std::exchange(result_, nullptr));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(loom_serve_pool_memory(pool)));
    EXPECT_GT(loom_serve_pool_statistics(pool).retained.committed_bytes, 0u);
  }
  auto* shared_residency = loom_serve_finite_model_residency(producer_);
  auto* isolated_residency = loom_serve_finite_model_residency(model_);
  const auto before = loom_serve_pool_statistics(isolated_);
  const auto activity = loom_serve_residency_activity(shared_residency);

  IREE_ASSERT_OK(loom_serve_pool_trim(shared_, 0));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(loom_serve_pool_memory(shared_)));
  EXPECT_FALSE(loom_serve_residency_snapshot(shared_residency).active);
  EXPECT_TRUE(loom_serve_residency_snapshot(isolated_residency).active);
  EXPECT_EQ(loom_serve_pool_statistics(shared_).retained.committed_bytes, 0u);
  EXPECT_EQ(loom_serve_pool_statistics(isolated_).retained.committed_bytes,
            before.retained.committed_bytes);

  // Storage eviction and code trimming preserve live tokenizer borrowers,
  // including models on another execution device or in another storage pool.
  loom_serve_device_trim_programs(device_);
  EXPECT_EQ(loom_serve_finite_model_tokenizer(producer_), tokenizer);
  EXPECT_EQ(loom_serve_finite_model_tokenizer(peer_), tokenizer);
  EXPECT_EQ(loom_serve_finite_model_tokenizer(model_), tokenizer);

  ASSERT_NO_FATAL_FAILURE(Prepare(producer_));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(shared_residency, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Submit(producer_, shared_));
  EXPECT_EQ(loom_serve_residency_activity(shared_residency).activations,
            activity.activations + 1);
  EXPECT_EQ(loom_serve_pool_statistics(isolated_).retained.committed_bytes,
            before.retained.committed_bytes);
}

}  // namespace loom_serve::testing::embedding
