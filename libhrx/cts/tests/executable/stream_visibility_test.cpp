// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

// The HAL executable registry also serves GTest suites. Keep Catch's assertion
// names while using that registry's linked Loom artifacts.
#define GTEST_DONT_DEFINE_FAIL 1
#define GTEST_DONT_DEFINE_SUCCEED 1

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <cstdio>

#include "hrx_test_fixture.hpp"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"

namespace {

enum class Observation { kWait, kQuery, kEvent };

class StreamVisibilityTest : public HrxTestFixture {
 protected:
  void Initialize() {
    std::array<char, 64> architecture = {};
    REQUIRE_OK(
        hrx().device_get_property(device_, HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                  architecture.data(), architecture.size()));
    iree_hal_amd_gpu_target_identity_t identity;
    iree_status_t status = iree_hal_amd_gpu_target_identity_parse_artifact_key(
        iree_make_cstring_view(architecture.data()), &identity);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      FAIL("cannot parse GPU artifact identity");
    }
    const auto images = iree::hal::cts::CtsRegistry::ListExecutableTargets(
        "hrx_dependent_chain");
    const auto* image =
        iree::hal::amd::gpu::cts::SelectExecutableTarget(images, identity);
    REQUIRE(image != nullptr);
    const auto data = image->data_fn(IREE_SV("dependent_chain.bin"));
    REQUIRE_OK(hrx().executable_load_data(device_, data.data, data.data_length,
                                          image->family, image->target_key,
                                          &executable_));
    REQUIRE_OK(hrx().executable_lookup_export_by_name(
        executable_, "dependent_chain", &ordinal_));
    REQUIRE_OK(hrx().stream_create(
        FindProvisionedQueue(device_,
                             HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER),
        0, &streams_[0]));
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const bool mapped = i != 1;
      const hrx_buffer_params_t params = {
          .type = mapped ? HRX_MEMORY_TYPE_HOST_LOCAL |
                               HRX_MEMORY_TYPE_HOST_COHERENT |
                               HRX_MEMORY_TYPE_DEVICE_VISIBLE
                         : HRX_MEMORY_TYPE_DEVICE_LOCAL,
          .access = HRX_MEMORY_ACCESS_ALL,
          .usage = HRX_BUFFER_USAGE_DEFAULT |
                   (mapped ? HRX_BUFFER_USAGE_MAPPING_PERSISTENT : 0u),
      };
      REQUIRE_OK(hrx().allocator_allocate_buffer(
          hrx().device_allocator(device_), params, kByteLength, &buffers_[i]));
      if (mapped) {
        void* data = nullptr;
        REQUIRE_OK(hrx().buffer_map(buffers_[i], HRX_MAP_READ | HRX_MAP_WRITE,
                                    0, kByteLength, &data));
        mapped_[i] = static_cast<uint32_t*>(data);
      }
    }
    REQUIRE_OK(
        hrx().event_create(device_, HRX_EVENT_FLAG_DISABLE_TIMING, &event_));
  }

  ~StreamVisibilityTest() {
    // Assertion unwinding joins accepted work before mappings or code
    // disappear.
    for (hrx_stream_t stream : streams_) {
      if (stream) {
        ReportHrxTestFailure("stream visibility completion",
                             hrx().stream_synchronize(stream));
        hrx().stream_release(stream);
      }
    }
    hrx().event_release(event_);
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (mapped_[i]) {
        ReportHrxTestFailure("stream visibility unmap",
                             hrx().buffer_unmap(buffers_[i]));
      }
      hrx().buffer_release(buffers_[i]);
    }
    hrx().executable_release(executable_);
  }

  void InitializeInput(uint32_t epoch) {
    for (uint32_t i = 0; i < kElementCount; ++i) {
      expected_[i] = mapped_[0][i] = epoch * 131 + i * 747796405u;
      mapped_[2][i] = ~expected_[i];
    }
  }

  void Dispatch(hrx_stream_t stream, size_t source, size_t target,
                uint32_t step) {
    const hrx_dispatch_config_t config = {
        {kElementCount / 256, 1, 1}, {256, 1, 1}, 0};
    const uint32_t constants[] = {kElementCount, step};
    const hrx_buffer_ref_t bindings[] = {
        {buffers_[source], 0, kByteLength},
        {buffers_[target], 0, kByteLength},
    };
    REQUIRE_OK(hrx().stream_dispatch(stream, executable_, ordinal_, &config,
                                     constants, sizeof(constants), bindings, 2,
                                     HRX_DISPATCH_FLAG_NONE));
    std::array<uint32_t, kElementCount> next;
    for (uint32_t i = 0; i < kElementCount; ++i) {
      const uint32_t neighbor = expected_[(i + 257) & (kElementCount - 1)];
      next[i] = (expected_[i] ^ ((neighbor << 1) | (neighbor >> 31))) + step;
    }
    expected_ = next;
  }

  void Observe(hrx_stream_t stream, Observation observation) {
    REQUIRE_OK(hrx().stream_flush(stream));
    switch (observation) {
      case Observation::kWait:
        REQUIRE_OK(hrx().stream_wait(stream));
        break;
      case Observation::kQuery: {
        bool complete = false;
        while (!complete) {
          REQUIRE_OK(hrx().stream_query(stream, &complete));
        }
        break;
      }
      case Observation::kEvent:
        REQUIRE_OK(hrx().event_record(event_, stream));
        REQUIRE_OK(hrx().event_synchronize(event_));
        break;
    }
    // No transfer or extra queue operation may repair the host visibility.
    for (uint32_t i = 0; i < kElementCount; ++i) {
      REQUIRE(mapped_[2][i] == expected_[i]);
    }
  }

  void CreateConsumer() {
    const uint32_t ordinal =
        hrx().queue_get_family_ordinal(hrx().stream_get_queue(streams_[0]));
    hrx_queue_family_info_t family;
    REQUIRE_OK(hrx().device_query_queue_family(device_, ordinal, &family));
    if (!family.supports_acquisition) {
      SKIP("selected family cannot acquire an independent consumer queue");
    }
    hrx_queue_t queue = nullptr;
    REQUIRE_OK(hrx().queue_acquire(device_, ordinal, nullptr, &queue));
    const hrx_status_t status = hrx().stream_create(queue, 0, &streams_[1]);
    hrx().queue_release(queue);
    REQUIRE_OK(status);
    REQUIRE(hrx().stream_get_queue(streams_[0]) !=
            hrx().stream_get_queue(streams_[1]));
  }

  static constexpr uint32_t kElementCount = 4096;
  static constexpr size_t kByteLength = kElementCount * sizeof(uint32_t);
  // Executable containing the same cross-workgroup recurrence as the benchmark.
  hrx_executable_t executable_ = nullptr;
  // Reflected recurrence export index.
  uint32_t ordinal_ = 0;
  // Producer and optional independently acquired consumer timelines.
  std::array<hrx_stream_t, 2> streams_ = {};
  // Coherent input, device-local intermediate, and coherent output allocations.
  std::array<hrx_buffer_t, 3> buffers_ = {};
  // Persistent mappings kept alive through the final public observation.
  std::array<uint32_t*, 3> mapped_ = {};
  // Event capturing the producer or final consumer's flushed timeline.
  hrx_event_t event_ = nullptr;
  // Independent host recurrence; no device readback participates in this value.
  std::array<uint32_t, kElementCount> expected_ = {};
};

TEST_CASE_METHOD(StreamVisibilityTest,
                 "Stream boundaries publish mapped dispatch payloads",
                 "[stream][visibility]") {
  const auto observation =
      GENERATE(Observation::kWait, Observation::kQuery, Observation::kEvent);
  const bool explicit_barrier = GENERATE(false, true);
  Initialize();
  for (uint32_t epoch = 1; epoch <= 3; ++epoch) {
    InitializeInput(epoch);
    Dispatch(streams_[0], 0, 1, 1);
    Dispatch(streams_[0], 1, 2, 2);
    if (explicit_barrier) {
      REQUIRE_OK(hrx().stream_barrier(streams_[0], nullptr));
    }
    Dispatch(streams_[0], 2, 1, 3);
    Dispatch(streams_[0], 1, 2, 4);
    Observe(streams_[0], observation);
  }
}

TEST_CASE_METHOD(StreamVisibilityTest,
                 "Stream dependencies order transfers and dispatches",
                 "[stream][visibility]") {
  Initialize();
  const uint32_t pattern = 0xABCD1234u;
  expected_.fill(pattern);
  REQUIRE_OK(hrx().stream_fill_buffer(streams_[0], buffers_[1], 0, kByteLength,
                                      &pattern, sizeof(pattern)));
  Dispatch(streams_[0], 1, 2, 1);
  REQUIRE_OK(hrx().stream_copy_buffer(streams_[0], buffers_[2], 0, buffers_[1],
                                      0, kByteLength));
  Dispatch(streams_[0], 1, 2, 2);
  Observe(streams_[0], Observation::kQuery);
}

TEST_CASE_METHOD(StreamVisibilityTest,
                 "Stream event hands dispatch data to another queue",
                 "[stream][visibility][event]") {
  Initialize();
  CreateConsumer();
  for (uint32_t epoch = 1; epoch <= 3; ++epoch) {
    InitializeInput(epoch);
    Dispatch(streams_[0], 0, 1, 1);
    REQUIRE_OK(hrx().event_record(event_, streams_[0]));
    REQUIRE_OK(hrx().stream_wait_event(streams_[1], event_));
    Dispatch(streams_[1], 1, 2, 2);
    Observe(streams_[1], Observation::kWait);
  }
}

}  // namespace
