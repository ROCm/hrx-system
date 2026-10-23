// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch_upload.h"

#include <algorithm>
#include <array>
#include <vector>

#include "loom/serve/service/text/model_instance.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class SourceEpochUploadTest : public SourceTextServiceTest {
 protected:
  loom_serve_text_options_t ModelOptions() const override {
    auto options = SourceTextServiceTest::ModelOptions();
    options.prefill_capacity = 4097;
    options.context_capacity = 8192;
    options.pool_capacity = 8192;
    options.epoch_shapes = wide_shapes_;
    options.epoch_count = IREE_ARRAYSIZE(wide_shapes_);
    return options;
  }

  // Retained packet capacity is independent of this epoch's live token count.
  const loom_serve_packing_shape_t wide_shapes_[2] = {{4, 1}, {4097, 4}};
  // Distinct sentinels reveal uploads of unwritten host capacity.
  const uint8_t host_canary_ = 0x31;
  // Borrowed fill pattern remains alive through failed submission retirement.
  const uint8_t device_canary_ = 0xA7;
  // Download payloads outlive terminal queue cleanup in the fixture teardown.
  std::array<std::vector<uint8_t>, 2> downloaded_;
};

TEST_F(SourceEpochUploadTest, LivePacketsPreserveUnusedHostAndDeviceCapacity) {
  // Inspect native packet storage without introducing a serving observation
  // hook. All encoding, uploads, numerical execution, and publication below
  // use the real model path, shared by CPU and native GPU suites.
  const std::array<iree_byte_span_t, 2> host = {
      model_->control.host.bytes[TEXT_HOST_METADATA],
      model_->control.host.bytes[TEXT_HOST_TOKENS],
  };
  const std::array<iree_hal_buffer_t*, 2> device = {
      model_->state.epoch.buffers[1], model_->state.epoch.buffers[4]};
  std::array<Oracle, 4> reference;
  for (auto [span_count, tokens_per_span] :
       {std::pair{4u, 1024u}, {1u, 1u}, {3u, 341u}, {1u, 1u}}) {
    SCOPED_TRACE(::testing::Message()
                 << "spans=" << span_count << " tokens=" << tokens_per_span);
    uint64_t completion = 0;
    for (size_t slot = 0; slot < host.size(); ++slot) {
      std::memset(host[slot].data, host_canary_, host[slot].data_length);
      const iree_hal_transfer_operation_t fill = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = device[slot],
                  .length = host[slot].data_length,
                  .pattern = &device_canary_,
                  .pattern_length = sizeof(device_canary_),
              },
      };
      IREE_ASSERT_OK(loom_serve_execution_transfer(model_->execution, 1, &fill,
                                                   &completion));
    }
    std::array<std::vector<int32_t>, 4> inputs;
    std::array<loom_serve_text_span_t, 4> spans;
    for (size_t row = 0; row < span_count; ++row) {
      inputs[row].resize(tokens_per_span);
      for (size_t i = 0; i < tokens_per_span; ++i) {
        inputs[row][i] = (i + row) % 3;
        reference[row].Consume(inputs[row][i]);
      }
      spans[row] = {row, tokens_per_span, inputs[row].data(),
                    LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
    }
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 1, span_count, spans.data()));
    const std::array<size_t, 2> live_lengths = {
        16 + span_count * 36, span_count * tokens_per_span * sizeof(int32_t)};
    for (size_t row = 0; row < span_count; ++row) {
      EXPECT_EQ(
          loom_serve_text_row_token(loom_serve_text_model_row(model_, row)),
          reference[row].Predict());
    }
    for (size_t slot = 0; slot < host.size(); ++slot) {
      downloaded_[slot].resize(host[slot].data_length);
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = device[slot],
                  .target = downloaded_[slot].data(),
                  .length = downloaded_[slot].size(),
              },
      };
      IREE_ASSERT_OK(loom_serve_execution_transfer(model_->execution, 1,
                                                   &download, &completion));
    }
    IREE_ASSERT_OK(loom_serve_execution_wait(model_->execution, completion));
    for (size_t slot = 0; slot < host.size(); ++slot) {
      EXPECT_EQ(std::memcmp(downloaded_[slot].data(), host[slot].data,
                            live_lengths[slot]),
                0);
      for (size_t i = live_lengths[slot]; i < host[slot].data_length; ++i) {
        ASSERT_EQ(host[slot].data[i], host_canary_)
            << "host slot=" << slot << " byte=" << i;
        ASSERT_EQ(downloaded_[slot][i], device_canary_)
            << "device slot=" << slot << " byte=" << i;
      }
    }
  }
}

class EpochUploadTest : public SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &execution_));
    IREE_ASSERT_OK(
        loom_serve_execution_reserve_workspace(execution_, kCapacity, 256));
    uint64_t completion = 0;
    for (size_t slot = 0; slot < targets_.size(); ++slot) {
      host_[slot].resize(kCapacity, static_cast<uint8_t>(0x31 + slot));
      result_[slot].resize(kCapacity);
      packets_[slot] =
          iree_make_const_byte_span(host_[slot].data(), host_[slot].size());
      const auto request =
          iree_hal_make_pool_reservation_request(kCapacity, 256);
      IREE_ASSERT_OK(loom_serve_execution_alloca(execution_, 1, &request,
                                                 &targets_[slot]));
      const iree_hal_transfer_operation_t fill = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill =
              {
                  .target_buffer = targets_[slot],
                  .length = kCapacity,
                  .pattern = &canary_,
                  .pattern_length = sizeof(canary_),
              },
      };
      IREE_ASSERT_OK(
          loom_serve_execution_transfer(execution_, 1, &fill, &completion));
    }
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    for (auto* target : targets_) {
      iree_hal_buffer_release(target);
    }
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    loom_serve_execution_release(execution_);
    iree_hal_semaphore_release(gate_);
    SourceTextServiceTest::TearDown();
  }

  void Range(uint64_t source, uint64_t source_offset, uint64_t target,
             uint64_t target_offset, uint64_t length) {
    const size_t begin = plan_.size();
    plan_.resize(begin + 40);
    iree_unaligned_store_le_u64(plan_.data() + begin, source);
    iree_unaligned_store_le_u64(plan_.data() + begin + 8, source_offset);
    iree_unaligned_store_le_u64(plan_.data() + begin + 16, target);
    iree_unaligned_store_le_u64(plan_.data() + begin + 24, target_offset);
    iree_unaligned_store_le_u64(plan_.data() + begin + 32, length);
  }

  void Range(uint64_t slot, uint64_t offset, uint64_t length) {
    Range(slot, offset, slot, offset, length);
  }

  iree_status_t Submit() {
    return loom_serve_text_epoch_upload(
        execution_, iree_make_const_byte_span(plan_.data(), plan_.size()),
        packets_.data(), targets_.data());
  }

  void Download() {
    iree_hal_transfer_operation_t operations[2];
    for (size_t slot = 0; slot < targets_.size(); ++slot) {
      operations[slot] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = targets_[slot],
                  .target = result_[slot].data(),
                  .length = result_[slot].size(),
              },
      };
    }
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        execution_, IREE_ARRAYSIZE(operations), operations, &completion));
    IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  }

  // Capacity includes a discontiguous payload and untouched trailing storage.
  static constexpr size_t kCapacity = 256 * 1024;
  // Borrowed fill pattern survives asynchronous initialization.
  uint8_t canary_ = 0xA7;
  // Host payloads remain alive through every accepted queue access.
  std::array<std::vector<uint8_t>, 2> host_;
  // Stable packet descriptors point into the retained host payloads.
  std::array<iree_const_byte_span_t, 2> packets_;
  // Actual downloaded contents, including holes and inactive capacity.
  std::array<std::vector<uint8_t>, 2> result_;
  // Independent device packet capacity reused across submissions.
  std::array<iree_hal_buffer_t*, 2> targets_ = {};
  // Source plan storage need not survive the submission call.
  std::vector<uint8_t> plan_;
  // Real command/transfer timeline, with no mock submission layer.
  loom_serve_execution_t* execution_ = nullptr;
  // Explicitly controlled queue readiness for borrowed lifetime coverage.
  iree_hal_semaphore_t* gate_ = nullptr;
};

TEST_F(EpochUploadTest, SparseRangesPreserveHolesAndReuseCapacity) {
  std::array<std::vector<uint8_t>, 2> expected;
  for (auto& bytes : expected) {
    bytes.resize(kCapacity, canary_);
  }
  uint8_t generation = 0x41;
  for (size_t rows : {4u, 1u, 3u, 1u}) {
    plan_.clear();
    for (size_t slot = 0; slot < host_.size(); ++slot) {
      const size_t begin = slot ? 65536 : 0;
      const size_t length = slot ? rows * 40976 : 12 + rows * 20;
      std::fill(host_[slot].begin(), host_[slot].end(), generation++);
      std::copy_n(host_[slot].begin() + begin, length,
                  expected[slot].begin() + begin);
      Range(slot, begin, length);
    }
    IREE_ASSERT_OK(Submit());
    ASSERT_NO_FATAL_FAILURE(Download());
    for (size_t slot = 0; slot < result_.size(); ++slot) {
      // Changing every host byte catches accidental uploads beyond the current
      // live extent, including storage used by a larger previous invocation.
      for (size_t i = 0; i < kCapacity; ++i) {
        ASSERT_EQ(result_[slot][i], expected[slot][i])
            << "slot=" << slot << "byte=" << i << "rows=" << rows;
      }
    }
  }
}

TEST_F(EpochUploadTest, ValidatesWholePlanBeforePublishingAnyRange) {
  const std::array<std::array<uint64_t, 5>, 7> invalid = {{
      {2, 0, 1, 0, 1},
      {0, 0, 2, 0, 1},
      {0, kCapacity + 1, 1, 0, 0},
      {1, 0, 1, kCapacity, 1},
      {1, 1, 1, 0, UINT64_MAX},
      {0, 0, 0, 64, 1},
      {0, 0, 0, 0, 1},
  }};
  const std::array<iree_status_code_t, 7> expected = {
      IREE_STATUS_INVALID_ARGUMENT, IREE_STATUS_INVALID_ARGUMENT,
      IREE_STATUS_OUT_OF_RANGE,     IREE_STATUS_OUT_OF_RANGE,
      IREE_STATUS_OUT_OF_RANGE,     IREE_STATUS_INVALID_ARGUMENT,
      IREE_STATUS_INVALID_ARGUMENT,
  };
  for (size_t index = 0; index < invalid.size(); ++index) {
    const auto& range = invalid[index];
    plan_.clear();
    for (size_t i = 0; i < 65; ++i) {
      Range(0, i, 1);
    }
    Range(range[0], range[1], range[2], range[3], range[4]);
    const auto before = loom_serve_execution_frontier(execution_);
    IREE_EXPECT_STATUS_IS(expected[index], Submit());
    const auto after = loom_serve_execution_frontier(execution_);
    EXPECT_EQ(before.values[0], after.values[0]);
  }
  plan_.assign(39, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Submit());
  ASSERT_NO_FATAL_FAILURE(Download());
  for (const auto& bytes : result_) {
    for (uint8_t value : bytes) {
      ASSERT_EQ(value, canary_);
    }
  }
}

TEST_F(EpochUploadTest, EmptyRangesDoNotSubmitWork) {
  const auto before = loom_serve_execution_frontier(execution_);
  IREE_ASSERT_OK(Submit());
  Range(0, kCapacity, 0);
  Range(1, 0, 0);
  IREE_ASSERT_OK(Submit());
  const auto after = loom_serve_execution_frontier(execution_);
  EXPECT_EQ(before.values[0], after.values[0]);
  ASSERT_NO_FATAL_FAILURE(Download());
  for (const auto& bytes : result_) {
    for (uint8_t value : bytes) {
      ASSERT_EQ(value, canary_);
    }
  }
}

TEST_F(EpochUploadTest, HostAndDeviceBoundsAreIndependent) {
  // Both host and device capacity are independently established boundaries.
  packets_[0].data_length = 17;
  Range(0, 16, 2);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Submit());
  plan_.clear();
  host_[1].resize(kCapacity + 16);
  packets_[1] = iree_make_const_byte_span(host_[1].data(), host_[1].size());
  Range(1, kCapacity, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Submit());
  ASSERT_NO_FATAL_FAILURE(Download());
  for (const auto& bytes : result_) {
    for (uint8_t value : bytes) {
      ASSERT_EQ(value, canary_);
    }
  }
}

TEST_F(EpochUploadTest, MultipleBatchesBorrowPayloadButNotPlan) {
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  uint64_t value = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(execution_, {1, &gate_, &value}));
  for (size_t slot = 0; slot < targets_.size(); ++slot) {
    for (size_t i = slot; i < 130; i += 2) {
      Range(slot, i * 2, 1);
    }
  }
  Range(0, kCapacity, 0);
  IREE_ASSERT_OK(Submit());
  std::fill(plan_.begin(), plan_.end(), 0xFF);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, value, nullptr));
  ASSERT_NO_FATAL_FAILURE(Download());
  for (size_t slot = 0; slot < result_.size(); ++slot) {
    for (size_t i = 0; i < kCapacity; ++i) {
      const bool written = i < 260 && i % 2 == 0 && (i / 2) % 2 == slot;
      ASSERT_EQ(result_[slot][i], written ? host_[slot][i] : canary_);
    }
  }
}

TEST_F(EpochUploadTest, CompactSourcesScatterAndBroadcastWithoutPacking) {
  std::array<std::vector<uint8_t>, 2> expected;
  for (auto& bytes : expected) {
    bytes.resize(kCapacity, canary_);
  }
  for (size_t slot = 0; slot < host_.size(); ++slot) {
    for (size_t i = 0; i < kCapacity; ++i) {
      host_[slot][i] = static_cast<uint8_t>(i * 17 + (i >> 8) + slot * 31);
    }
  }
  // Source ordinals and offsets need not mirror destination placement. The
  // second copy broadcasts immutable bytes without duplicating the source.
  Range(1, 31, 0, 64, 17);
  Range(1, 31, 0, 192, 17);
  Range(0, 0, 1, 65536, 40976);
  std::copy_n(host_[1].begin() + 31, 17, expected[0].begin() + 64);
  std::copy_n(host_[1].begin() + 31, 17, expected[0].begin() + 192);
  std::copy_n(host_[0].begin(), 40976, expected[1].begin() + 65536);
  IREE_ASSERT_OK(Submit());
  ASSERT_NO_FATAL_FAILURE(Download());
  EXPECT_EQ(result_, expected);
}

}  // namespace
