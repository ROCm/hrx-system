// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/service_test_fixture.h"

namespace loom_serve::testing::image {

class ImageSpecializationTest : public SourceImageServiceTest {
 protected:
  void TearDown() override {
    ReleaseCompiler();
    if (prepared_) {
      IREE_EXPECT_OK(Resolve(true));
      loom_serve_image_model_discard_batch(model_);
    }
    SourceImageServiceTest::TearDown();
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ImageSpecializationTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        std::this_thread::get_id() != test->owner_) {
      ++test->compiler_allocations_;
      std::unique_lock<std::mutex> lock(test->mutex_);
      if (test->holding_) {
        test->entered_ = true;
        test->condition_.notify_all();
        test->condition_.wait(lock, [&] { return !test->holding_; });
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  static void Wake(void* self) {
    auto* test = static_cast<ImageSpecializationTest*>(self);
    std::lock_guard<std::mutex> lock(test->mutex_);
    ++test->wakes_;
    test->condition_.notify_all();
  }

  iree_status_t Resolve(bool cancel) {
    bool complete = false;
    while (!complete) {
      const auto before = wakes_.load();
      IREE_RETURN_IF_ERROR(
          loom_serve_image_model_advance_batch(model_, cancel, &complete));
      if (!complete) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return wakes_.load() != before; });
      }
    }
    return iree_ok_status();
  }

  void Run(uint32_t width, uint32_t height, bool reuse) {
    const auto before = compiler_allocations_.load();
    loom_service_image_prepared_t prepared;
    iree_const_byte_span_t rgba;
    ImageRequest input;
    IREE_ASSERT_OK(input.Initialize(IREE_SV(R"({"prompt":"aba","seed":1})"),
                                    width, height));
    IREE_ASSERT_OK(loom_serve_image_model_generate(model_, &input.value,
                                                   &prepared, &rgba));
    EXPECT_EQ(compiler_allocations_.load() == before, reuse);
    const auto expected = Pixels(3, 1, (size_t)width * height);
    ASSERT_EQ(rgba.data_length, expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(rgba.data[i], expected[i]);
    }
  }

  void RunCohort(float first_strength, bool cancel, bool reuse) {
    const auto before = compiler_allocations_.load();
    const std::string first_body =
        R"({"prompt":"ab","loras":[{"id":"red","scale":)" +
        std::to_string(first_strength - 1.0f) +
        R"(},{"id":"green","scale":-0.5}]})";
    std::array<ImageRequest, 2> inputs;
    IREE_ASSERT_OK(inputs[0].Initialize(
        iree_make_string_view(first_body.data(), first_body.size()), kWidth,
        kHeight));
    IREE_ASSERT_OK(inputs[1].Initialize(
        IREE_SV(
            R"({"prompt":"aba","seed":1,"loras":[{"id":"green","scale":-0.25},{"id":"red","scale":-1.5}]})"),
        kWidth, kHeight));
    struct Prepared {
      ~Prepared() {
        for (const auto& request : requests) {
          iree_vm_buffer_release(request.bytes);
        }
      }
      // Independent CPU payloads retained until the batch captures its copy.
      std::array<loom_service_prepared_request_t, 2> requests = {};
    } prepared;
    IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
        model_, &inputs[0].value, &prepared.requests[0]));
    // Rejection of another request cannot release or replace a prepared peer.
    for (
        const char* body : {
            R"({"prompt":"ab","size":"65x5"})",
            R"({"prompt":"ab","loras":[{"id":"missing"}]})",
            R"({"prompt":"ab","loras":[{"id":"red","scale":0},{"id":"\u0072ed"}]})",
        }) {
      SCOPED_TRACE(body);
      ImageRequest input;
      IREE_ASSERT_OK(
          input.Initialize(iree_make_cstring_view(body), kWidth, kHeight));
      loom_service_prepared_request_t rejected = {};
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                            loom_serve_image_model_prepare_request(
                                model_, &input.value, &rejected));
      EXPECT_EQ(rejected.bytes, nullptr);
      iree_vm_buffer_release(rejected.bytes);
    }
    IREE_ASSERT_OK(loom_serve_image_model_prepare_request(
        model_, &inputs[1].value, &prepared.requests[1]));
    EXPECT_EQ(prepared.requests[0].units, 1u);
    EXPECT_EQ(prepared.requests[0].tag, prepared.requests[1].tag);
    const iree_host_size_t lengths[] = {kPixels * 4, kPixels * 4};
    loom_service_image_prepared_t batch;
    IREE_ASSERT_OK(loom_serve_image_model_prepare_batch(
        model_, prepared.requests.size(), prepared.requests.data(), lengths,
        &batch));
    prepared_ = true;
    EXPECT_EQ(batch.input_bytes, 2u * (16u + 32u));
    loom_serve_image_model_set_observer(model_, {Wake, this});
    IREE_ASSERT_OK(Resolve(false));
    EXPECT_EQ(compiler_allocations_.load() == before, reuse);
    const auto generator = loom_serve_image_model_generator(model_);
    IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(
        loom_serve_image_model_residency(model_)));
    // From admission onward, teardown joins the model's accepted ownership.
    prepared_ = false;
    uint32_t parts = 0;
    bool complete = false;
    while (!complete) {
      loom_runtime_execution_frontier_t frontier;
      IREE_ASSERT_OK(generator.submit(
          generator.self,
          cancel && parts ? LOOM_SERVE_IMAGE_SUBMIT_FLAG_CANCEL : 0, &frontier,
          &complete));
      ASSERT_NO_FATAL_FAILURE(Wait(frontier));
      ++parts;
    }
    EXPECT_EQ(parts, GetParam() ? 2u : 1u);
    const auto rgba = generator.complete(generator.self);
    prepared_ = false;
    loom_serve_image_model_set_observer(model_, {});
    if (cancel && GetParam()) {
      return;
    }
    ASSERT_EQ(rgba.data_length, 2 * kPixels * 4);
    for (size_t item = 0; item < 2; ++item) {
      const auto expected =
          Pixels(item ? 3 : 2, item, kPixels, item ? -0.5f : first_strength,
                 item ? 0.75f : 0.5f);
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(rgba.data[item * lengths[item] + i], expected[i]);
      }
    }
  }

  void ReleaseCompiler() {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = false;
    condition_.notify_all();
  }

  // This outer thread performs preparation and drives the real native model.
  const std::thread::id owner_ = std::this_thread::get_id();
  // Model allocations on actual compiler workers distinguish misses from hits.
  std::atomic<size_t> compiler_allocations_ = 0;
  // Published compiler completions observed by the outer test owner.
  std::atomic<uint64_t> wakes_ = 0;
  // Protects the explicit dependency used to test accepted-compile retirement.
  std::mutex mutex_;
  // Signals worker entry, dependency release, and terminal publication.
  std::condition_variable condition_;
  // An accepted worker is held only for the cancellation experiment.
  bool holding_ = false;
  // A real compiler allocation has reached the held boundary.
  bool entered_ = false;
  // Teardown must collect and release an outstanding prepared request.
  bool prepared_ = false;
};

TEST_P(ImageSpecializationTest, ExactReuseAndLruEvictionPreserveOutput) {
  ASSERT_NO_FATAL_FAILURE(
      CreateStandaloneModel({2, 16 * 1024 * 1024}, {this, Allocate}));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  // This source specializes pixel count, so rotation reuses the same code.
  ASSERT_NO_FATAL_FAILURE(Run(9, 15, true));
  ASSERT_NO_FATAL_FAILURE(Run(16, 11, false));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, true));
  ASSERT_NO_FATAL_FAILURE(Run(1, 1, false));
  ASSERT_NO_FATAL_FAILURE(Run(16, 11, false));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

TEST_P(ImageSpecializationTest, CohortPrivateValuesSurvivePartsAndReuse) {
  ASSERT_NO_FATAL_FAILURE(
      CreateStandaloneModel({2, 16 * 1024 * 1024}, {this, Allocate}));
  ASSERT_NO_FATAL_FAILURE(RunCohort(0.25f, false, false));
  ASSERT_NO_FATAL_FAILURE(RunCohort(0.75f, false, true));
  ASSERT_NO_FATAL_FAILURE(RunCohort(0.0f, true, true));
  ASSERT_NO_FATAL_FAILURE(RunCohort(-0.25f, false, true));
}

TEST_P(ImageSpecializationTest, PeerAdmissionRetainsWholeInvocationWorkingSet) {
  constexpr uint32_t kExtent = 48;
  const auto prepare = [this] {
    loom_service_image_prepared_t prepared;
    IREE_ASSERT_OK(PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"),
                                 kExtent, kExtent, &prepared));
    prepared_ = true;
    loom_serve_image_model_set_observer(model_, {Wake, this});
    IREE_ASSERT_OK(Resolve(false));
    loom_serve_image_model_set_observer(model_, {});
  };
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  ASSERT_NO_FATAL_FAILURE(prepare());
  IREE_ASSERT_OK(loom_runtime_residency_acquire_wait(
      loom_serve_image_model_residency(model_)));
  prepared_ = false;
  const auto destroy = [](loom_serve_image_model_t* model) {
    IREE_EXPECT_OK(loom_serve_image_model_free(model));
  };
  std::unique_ptr<loom_serve_image_model_t, decltype(destroy)> first(
      std::exchange(model_, nullptr), destroy);
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel());
  ASSERT_NO_FATAL_FAILURE(prepare());
  const auto peer = loom_serve_image_model_generator(model_);
  const auto expect_peer_capacity = [&] {
    loom_runtime_residency_admission_t admission;
    IREE_ASSERT_OK(peer.try_admit(peer.self, &admission));
    // Each 9,216-byte output fits individually. Two outputs exceed the
    // bounded domain's 16 KiB even before payload and workspace.
    EXPECT_EQ(admission, loom_runtime_pool_memory(storage_owner_)
                             ? LOOM_RUNTIME_RESIDENCY_ADMISSION_CAPACITY
                             : LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
    if (admission == LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED) {
      loom_runtime_residency_release(loom_serve_image_model_residency(model_));
    }
  };
  ASSERT_NO_FATAL_FAILURE(expect_peer_capacity());
  const auto finish = [&](loom_serve_image_model_t* model, bool check_peer) {
    const auto generator = loom_serve_image_model_generator(model);
    bool complete = false;
    while (!complete) {
      loom_runtime_execution_frontier_t frontier;
      IREE_ASSERT_OK(generator.submit(generator.self, 0, &frontier, &complete));
      ASSERT_NO_FATAL_FAILURE(Wait(frontier));
      if (check_peer) {
        // Intermediate command parts and final device readiness both retain
        // the accepted invocation until its owner collects the output.
        ASSERT_NO_FATAL_FAILURE(expect_peer_capacity());
      }
    }
    const auto output = generator.complete(generator.self);
    const auto expected = Pixels(3, 1, kExtent * kExtent);
    ASSERT_EQ(output.data_length, expected.size());
    EXPECT_EQ(
        std::vector<uint8_t>(output.data, output.data + output.data_length),
        expected);
  };
  ASSERT_NO_FATAL_FAILURE(finish(first.get(), true));

  // The identical prepared peer now fits, without another JIT or preparation.
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(peer.try_admit(peer.self, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  prepared_ = false;
  ASSERT_NO_FATAL_FAILURE(finish(model_, false));
}

TEST_P(ImageSpecializationTest, DisabledReuseStillAllowsLargerRequests) {
  ASSERT_NO_FATAL_FAILURE(CreateStandaloneModel({0, 0}, {this, Allocate}));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, false));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

TEST_P(ImageSpecializationTest, CancellationJoinsAcceptedCompilerBeforeReuse) {
  ASSERT_NO_FATAL_FAILURE(
      CreateStandaloneModel({2, 16 * 1024 * 1024}, {this, Allocate}));
  loom_serve_image_model_set_observer(model_, {Wake, this});
  loom_service_image_prepared_t prepared;
  IREE_ASSERT_OK(
      PrepareSingle(IREE_SV(R"({"prompt":"aba","seed":1})"), 15, 9, &prepared));
  prepared_ = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = true;
  }
  bool complete = true;
  IREE_ASSERT_OK(
      loom_serve_image_model_advance_batch(model_, false, &complete));
  EXPECT_FALSE(complete);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  IREE_ASSERT_OK(loom_serve_image_model_advance_batch(model_, true, &complete));
  EXPECT_FALSE(complete);
  ReleaseCompiler();
  IREE_ASSERT_OK(Resolve(true));
  loom_serve_image_model_discard_batch(model_);
  prepared_ = false;
  loom_serve_image_model_set_observer(model_, {});
  ASSERT_NO_FATAL_FAILURE(Run(15, 9, true));
  ASSERT_NO_FATAL_FAILURE(Run(kWidth, kHeight, true));
}

INSTANTIATE_TEST_SUITE_P(CommandQuanta, ImageSpecializationTest,
                         ::testing::Values(0u, 1u));

}  // namespace loom_serve::testing::image
