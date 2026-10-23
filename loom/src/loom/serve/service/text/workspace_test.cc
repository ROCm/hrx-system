// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <future>

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(SourceTextServiceTest, NativeWorkspaceReplacementPreservesAdmission) {
  auto* row = loom_serve_text_model_row(model_, 0);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 32, &admitted));
  ASSERT_TRUE(admitted);
  const int32_t token = 1;
  Oracle oracle;
  oracle.Consume(token);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());

  const auto path =
      iree::testing::MakeTempFilePath("workspace_variant", ".loom");
  // Keep the exact numerical program, but route its intermediate through a
  // command-owned allocation instead of the model's retained residual root.
  ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
      directory_ + "/advance.loom", path,
      "  kernel.launch @consume(%bias, %residual, %metadata, %origins, %arena, "
      "%tokens) : (view<1xi32>, buffer, buffer, buffer, buffer, buffer)\n"
      "  kernel.launch @select(%residual, %metadata, %origins, %arena, "
      "%outputs)",
      "  %bytes = index.constant 4194304 : offset\n"
      "  %temporary = buffer.alloca<global> align(16) %bytes : buffer\n"
      "  kernel.launch @consume(%bias, %temporary, %metadata, %origins, "
      "%arena, "
      "%tokens) : (view<1xi32>, buffer, buffer, buffer, buffer, buffer)\n"
      "  kernel.launch @select(%temporary, %metadata, %origins, %arena, "
      "%outputs)"));
  const auto provider = iree_make_cstring_view(path.c_str());
  const loom_serve_sources_t sources = {.count = 1, .values = &provider};
  struct Compilation {
    // Owned completion status, transferred before publishing the candidate.
    iree_status_t status = iree_ok_status();
    // Publishes the candidate and completion status to the outer test thread.
    std::promise<loom_serve_text_variant_t*> ready;
  } completion;
  IREE_ASSERT_OK(loom_serve_text_model_compile_variant_async(
      model_, &sources, nullptr,
      [](void* user_data, iree_status_t status, void* variant,
         const loom_serve_compilation_report_t* report) {
        auto* completion = static_cast<Compilation*>(user_data);
        completion->status = status;
        completion->ready.set_value(
            static_cast<loom_serve_text_variant_t*>(variant));
      },
      &completion));
  std::unique_ptr<loom_serve_text_variant_t,
                  decltype(&loom_serve_text_variant_free)>
      candidate(completion.ready.get_future().get(),
                loom_serve_text_variant_free);
  IREE_ASSERT_OK(completion.status);

  auto* pool = loom_serve_device_memory_pool(device_);
  loom_serve_memory_statistics_t peer_statistics = {};
  const auto release_peer = [](loom_serve_virtual_buffer_t* buffer) {
    IREE_EXPECT_OK(loom_serve_virtual_buffer_free(buffer));
  };
  std::unique_ptr<loom_serve_virtual_buffer_t, decltype(release_peer)> peer(
      nullptr, release_peer);
  const auto activity = loom_serve_text_model_weight_activity(model_);
  if (pool) {
    // The existing rows and parameters fit. The replacement's 4 MiB
    // transient exceeds both reusable workspace backing and the 4 KiB left
    // by this independently retained peer.
    const uint64_t peer_bytes = loom_serve_memory_pool_limit(pool) -
                                loom_serve_memory_pool_live_bytes(pool) - 4096;
    loom_serve_virtual_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_device_state_geometry(device_), peer_bytes, 16,
        &peer_statistics, &buffer));
    peer.reset(buffer);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(buffer, 0, peer_bytes));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        loom_serve_text_model_select_variant(model_, candidate.get()));
    IREE_ASSERT_OK(loom_serve_text_row_decode(row));
    oracle.Consume(oracle.Predict());
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
    peer.reset();
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool));
  }
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, candidate.get()));
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  oracle.Consume(oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());

  loom_serve_execution_t* peer_execution = nullptr;
  IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &peer_execution));
  std::unique_ptr<loom_serve_execution_t,
                  decltype(&loom_serve_execution_release)>
      peer_timeline(peer_execution, loom_serve_execution_release);
  loom_serve_residency_t* peer_residency = nullptr;
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      loom_serve_device_residency_cache(device_), peer_execution, 0, nullptr,
      host_allocator_, &peer_residency));
  std::unique_ptr<loom_serve_residency_t, decltype(&loom_serve_residency_free)>
      peer_claim(peer_residency, loom_serve_residency_free);
  const auto peer_bytes = [&] {
    const auto workspace =
        loom_serve_workspace_memory_usage(loom_serve_device_workspace(device_));
    // Logical claims reuse workspace backing; only retained model state and
    // parameters consume capacity independently of those claims.
    const uint64_t available = loom_serve_memory_pool_limit(pool) -
                               loom_serve_memory_pool_live_bytes(pool) +
                               workspace.committed_bytes +
                               workspace.pending_bytes;
    const uint64_t slab_size =
        loom_serve_device_weight_geometry(device_).slab_size;
    // A partial physical slab cannot back the peer. Leave room for the
    // original command inside the complete slabs available to both claims.
    return (available / slab_size) * slab_size - 4096;
  };
  if (pool) {
    // Independently prepared work uses ordinary admission. The selected text
    // transient must remain charged between epochs, not only during selection.
    IREE_ASSERT_OK(loom_serve_residency_prepare_working_set(peer_residency,
                                                            peer_bytes(), 0));
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(peer_residency, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  }
  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, nullptr));
  if (pool) {
    // Restoring the smaller original releases only private-workspace credit.
    // The identical waiting peer now fits; rows and parameters never moved.
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(peer_residency, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    loom_serve_residency_cancel_admission(peer_residency);
    loom_serve_residency_release(peer_residency);
  }
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  oracle.Consume(oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
  const auto after = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(after.activations, activity.activations);
  EXPECT_EQ(after.read_bytes, activity.read_bytes);

  // The same authored commands loaded initially, instead of hot-swapped, must
  // publish the same private peak. Reuse this device and the pressure claimant.
  auto options = ModelOptions();
  options.sources.count = 1;
  options.sources.values = &provider;
  loom_serve_text_model_t* cold_model = nullptr;
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &cold_model));
  const auto release_model = [](loom_serve_text_model_t* model) {
    IREE_EXPECT_OK(loom_serve_text_model_free(model));
  };
  std::unique_ptr<loom_serve_text_model_t, decltype(release_model)> cold(
      cold_model, release_model);
  auto* cold_row = loom_serve_text_model_row(cold_model, 0);
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      cold_row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 32, &admitted));
  ASSERT_TRUE(admitted);
  if (pool) {
    IREE_ASSERT_OK(loom_serve_residency_prepare_working_set(peer_residency,
                                                            peer_bytes(), 0));
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(peer_residency, &admission));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    loom_serve_residency_cancel_admission(peer_residency);
  }
  IREE_ASSERT_OK(loom_serve_text_row_prefill(cold_row, 1, &token));
  Oracle cold_oracle;
  cold_oracle.Consume(token);
  EXPECT_EQ(loom_serve_text_row_token(cold_row), cold_oracle.Predict());
}

}  // namespace
