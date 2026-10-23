// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>

#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/service/text/epoch.h"
#include "loom/serve/service/text/epoch_upload.h"
#include "loom/serve/service/text/media_epoch.h"
#include "loom/serve/service/text/media_projector.h"
#include "loom/serve/service/text/readiness.h"
#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

class MediaExecutionTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const char* source = std::getenv("LOOM_SERVE_TEST_MEDIA_SOURCE");
    ASSERT_NE(source, nullptr);
    IREE_ASSERT_OK(producer_document_.Load(source));
    const loom_serve_finite_model_options_t options = {
        .sources = producer_document_.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    IREE_ASSERT_OK(loom_serve_finite_model_allocate(
        device_, &options, host_allocator_, &producer_));
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    if (consumer_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(consumer_));
      loom_serve_execution_release(consumer_);
    }
    iree_hal_semaphore_release(gate_);
    loom_serve_text_media_epoch_deinitialize(&packet_);
    SourceTextServiceTest::TearDown();
    IREE_EXPECT_OK(loom_serve_finite_model_free(producer_));
    loom_serve_finite_result_release(result_);
    for (size_t i = 0; i < requests_.size(); ++i) {
      iree_vm_buffer_release(prepared_[i].bytes);
      loom_serve_embedding_request_deinitialize(&requests_[i]);
    }
  }

  static void Wait(loom_serve_execution_frontier_t ready) {
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          ready.semaphores[i], ready.values[i], iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }

  void Submit(std::array<const char*, 3> bodies = {
                  R"({"model":"fixture","input":["a","b"]})",
                  R"({"model":"fixture","input":["bb","a","xxx"]})",
                  R"({"model":"fixture","input":["b","a"]})"}) {
    // Ragged requests share one dispatch; the source deliberately stores its
    // physical rows backwards. Original request and row ordinals are required.
    std::array<iree_host_size_t, 3> rows;
    for (size_t i = 0; i < requests_.size(); ++i) {
      IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
          iree_make_cstring_view(bodies[i]), host_allocator_, &requests_[i]));
      IREE_ASSERT_OK(loom_serve_finite_model_prepare_request(
          producer_, requests_[i].base.input, &prepared_[i]));
      rows[i] = requests_[i].base.count;
    }
    IREE_ASSERT_OK(loom_serve_finite_model_prepare_batch(
        producer_, prepared_.size(), prepared_.data(), rows.data()));
    bool code_ready = false;
    IREE_ASSERT_OK(
        loom_serve_finite_model_advance_batch(producer_, false, &code_ready));
    ASSERT_TRUE(code_ready);
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(
        loom_serve_finite_model_residency(producer_), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    loom_serve_execution_frontier_t ready;
    IREE_ASSERT_OK(loom_serve_finite_model_submit(producer_, &result_, &ready));
    EXPECT_EQ(ready.values[1], 0u);
  }

  void RemoveProducer() {
    ASSERT_NO_FATAL_FAILURE(Wait(loom_serve_finite_result_ready(result_)));
    loom_serve_finite_model_complete(producer_);
    IREE_ASSERT_OK(
        loom_serve_finite_model_free(std::exchange(producer_, nullptr)));
    loom_serve_finite_result_release(std::exchange(result_, nullptr));
  }

  static int32_t Fingerprint(float value, uint32_t image_row = 0) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits + image_row;
  }

  // Production source document outliving the projector.
  loom_serve::testing::ModelDocument producer_document_;
  // Independent finite model sharing the text device and workspace.
  loom_serve_finite_model_t* producer_ = nullptr;
  // Original cohort ownership shared by the text requests.
  loom_serve_finite_result_t* result_ = nullptr;
  // HTTP protocol and source input storage retained through producer work.
  std::array<loom_serve_embedding_request_t, 3> requests_ = {};
  // Source-prepared request bytes, independent of cohort result ownership.
  std::array<loom_serve_prepared_request_t, 3> prepared_ = {};
  // Explicitly gated consumer for delayed-capture retirement coverage.
  loom_serve_execution_t* consumer_ = nullptr;
  // Owner-signalled readiness; no sleep or deadline models queue retirement.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Consumer-owned metadata and mixed input allocations.
  std::array<iree_hal_buffer_t*, 2> buffers_ = {};
  // Real epoch gather metadata retained through consumer retirement.
  loom_serve_text_media_epoch_t packet_ = {};
  // Host control upload retained even when a test exits before queue release.
  uint32_t control_ = 17;
  // Final consumer output, read only by the test oracle.
  std::vector<uint8_t> output_;
};

TEST_F(MediaExecutionTest, ProjectorAlignsAndReturnsItsHostAllocation) {
  constexpr size_t alignment = alignof(loom_serve_text_readiness_t);
  alignas(alignment) uint8_t storage[4096];
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host_), 0, &proactor));
  // A valid host allocator need not provide the task process's extended
  // alignment. Exercise each naturally aligned offset within one cache line.
  for (size_t offset = 0; offset < alignment; offset += iree_max_align_t) {
    SCOPED_TRACE(offset);
    iree_allocator_inline_storage_t arena = {
        .capacity = sizeof(storage) - offset,
        .buffer = storage + offset,
    };
    loom_serve_text_media_projector_t* projector = nullptr;
    IREE_ASSERT_OK(loom_serve_text_media_projector_allocate(
        producer_, 1, loom_serve_host_executor(host_), proactor, {},
        iree_allocator_inline_arena(&arena), &projector));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(projector) % alignment, 0u);
    EXPECT_TRUE(loom_serve_text_media_projector_is_collecting(projector));
    EXPECT_FALSE(loom_serve_text_media_projector_is_pending(projector));
    loom_serve_text_media_projector_free(projector);
    EXPECT_EQ(arena.length, 0u);
  }
}

TEST_F(MediaExecutionTest, RaggedCohortConditionsPackedAndIsolatedTextChunks) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  const loom_serve_text_media_input_span_t images_a[] = {
      {
          .input_offset = 1,
          .input_count = 2,
          .request_index = 2,
          .result = result_,
      },
      {
          .input_offset = 4,
          .input_count = 2,
          .request_index = 0,
          .result = result_,
      },
  };
  const loom_serve_text_media_input_span_t images_b[] = {
      {
          .input_offset = 0,
          .input_count = 3,
          .request_index = 1,
          .result = result_,
      },
  };
  const loom_serve_text_media_input_t media[] = {{images_a, 2}, {images_b, 1}};
  const std::array<std::vector<int32_t>, 3> tokens = {
      {{2, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 2, 1}, {0, 1, 2, 1, 0}}};
  const std::array<std::vector<int32_t>, 3> expected = {
      {{2, Fingerprint(2), Fingerprint(1, 1), 0, Fingerprint(1),
        Fingerprint(2, 1), 1},
       {Fingerprint(4), Fingerprint(1, 1), Fingerprint(9, 2), 2, 1},
       tokens[2]}};
  for (auto mode :
       {LOOM_SERVE_TEXT_SCHEDULE_PACKED, LOOM_SERVE_TEXT_SCHEDULE_ISOLATED,
        LOOM_SERVE_TEXT_SCHEDULE_MATCHED}) {
    SCOPED_TRACE(mode);
    const loom_serve_text_epoch_options_t options = {
        .context_capacity = 96,
        .chunk_size = 4,
        .continuation_epochs = 1,
        .schedule_mode = mode,
        .shapes = shapes_,
        .shape_count = IREE_ARRAYSIZE(shapes_),
    };
    std::array<size_t, 3> offsets = {};
    for (size_t chunk = 0; chunk < 2; ++chunk) {
      loom_serve_text_epoch_row_t rows[3] = {};
      loom_serve_packed_span_t scheduled[3];
      for (size_t row = 0; row < 3; ++row) {
        rows[row] = {
            .input_tokens = tokens[row].data() + offsets[row],
            .input_count = tokens[row].size() - offsets[row],
            .input_offset = offsets[row],
            .media = row < 2 ? &media[row] : nullptr,
            .output_count = 1,
            .position = offsets[row],
        };
        scheduled[row] = {row, chunk ? rows[row].input_count : (row ? 2u : 3u)};
      }
      loom_serve_text_epoch_t epoch;
      loom_serve_text_epoch_continuation_t continuation;
      loom_serve_text_epoch_prepare(&options, rows, 1, 3, scheduled, &epoch,
                                    &continuation);
      do {
        loom_serve_execution_frontier_t ready;
        IREE_ASSERT_OK(loom_serve_text_epoch_submit(
            model_, mode, &epoch, &continuation, nullptr, &ready));
        if (mode == LOOM_SERVE_TEXT_SCHEDULE_MATCHED && chunk == 1 &&
            epoch.cursor == 1) {
          // Both image requests have submitted their last readers. Their
          // metadata and result owner may disappear before text completion.
          ASSERT_NO_FATAL_FAILURE(RemoveProducer());
        }
        ASSERT_NO_FATAL_FAILURE(Wait(ready));
        IREE_ASSERT_OK(loom_serve_text_epoch_complete(model_, mode, &epoch));
      } while (mode != LOOM_SERVE_TEXT_SCHEDULE_PACKED && epoch.cursor < 3);
      for (size_t row = 0; row < 3; ++row) {
        EXPECT_EQ(epoch.results[row].known_count, scheduled[row].token_count);
        offsets[row] += scheduled[row].token_count;
      }
    }
    for (size_t row = 0; row < 3; ++row) {
      Oracle oracle;
      for (int32_t value : expected[row]) {
        oracle.Consume(value);
      }
      auto* state = loom_serve_text_model_row(model_, row);
      EXPECT_EQ(loom_serve_text_row_token(state), oracle.Predict());
      // Subsequent text-only decode consumes the same retained image state.
      oracle.Consume(oracle.Predict());
      IREE_ASSERT_OK(loom_serve_text_row_decode(state));
      EXPECT_EQ(loom_serve_text_row_token(state), oracle.Predict());
      IREE_ASSERT_OK(loom_serve_text_row_reset(state));
    }
  }
}

TEST_F(MediaExecutionTest,
       GatedGatherRetainsOnlyDeviceRowsPastProducerRemoval) {
  ASSERT_NO_FATAL_FAILURE(Submit());
  const uint8_t source_metadata[] = {9, 8, 7};
  const loom_serve_text_media_input_span_t image = {
      .input_offset = 30,
      .input_count = 3,
      .request_index = 1,
      .result = result_,
      .metadata =
          iree_make_const_byte_span(source_metadata, sizeof(source_metadata)),
  };
  const loom_serve_text_media_input_t media = {&image, 1};
  const int32_t tokens[] = {0, 0};
  const loom_serve_text_span_t span = {
      .row_index = 0,
      .token_count = 2,
      .token_ids = tokens,
      .input_offset = 31,
      .media = &media,
  };
  IREE_ASSERT_OK(loom_serve_text_media_epoch_initialize(
      1, &span, 4, 0, nullptr, 0, host_allocator_, &packet_));
  EXPECT_EQ(packet_.first_feature_count, 2u);
  EXPECT_EQ(packet_.transfer.result_count, 1u);
  EXPECT_EQ(packet_.transfer.operation_count, 2u);
  EXPECT_EQ(packet_.device_length, 160u);
  for (size_t i = 0; i < packet_.transfer.operation_count; ++i) {
    EXPECT_EQ(packet_.transfer.operations[i].type,
              IREE_HAL_TRANSFER_OPERATION_TYPE_COPY);
  }
  iree_const_byte_span_t plan;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      packet_.plan, 0, iree_vm_buffer_length(packet_.plan), &plan));
  EXPECT_EQ(plan.data_length, 2 * 28u + sizeof(source_metadata));
  EXPECT_EQ(iree_unaligned_load_le_u32(plan.data + 24), 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(plan.data + 28 + 24), 2u);
  EXPECT_EQ(
      std::memcmp(plan.data + 56, source_metadata, sizeof(source_metadata)), 0);
  std::memset(packet_.packet_record_bytes.data, 0x5A,
              packet_.packet_record_bytes.data_length);

  IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &consumer_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(consumer_, {1, &gate_, &gate_value}));
  const auto metadata_request = iree_hal_make_pool_reservation_request(16, 16);
  const auto packet_request =
      iree_hal_make_pool_reservation_request(packet_.device_length, 16);
  IREE_ASSERT_OK(loom_serve_execution_alloca(consumer_, 1, &metadata_request,
                                             &buffers_[0]));
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(consumer_, 1, &packet_request, &buffers_[1]));
  uint8_t ranges[80];
  const uint64_t fields[] = {
      0, 0, 0, 0, sizeof(control_),
      1, 0, 1, 0, packet_.packet_record_bytes.data_length};
  for (size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u64(ranges + i * sizeof(uint64_t), fields[i]);
  }
  const iree_const_byte_span_t packets[] = {
      iree_make_const_byte_span(&control_, sizeof(control_)),
      iree_const_cast_byte_span(packet_.packet_record_bytes),
  };
  IREE_ASSERT_OK(loom_serve_text_epoch_upload(
      consumer_, iree_make_const_byte_span(ranges, sizeof(ranges)), packets,
      buffers_.data()));
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_serve_text_media_epoch_enqueue(&packet_, consumer_,
                                                     buffers_[1], &value));
  EXPECT_EQ(packet_.transfer.operations, nullptr);
  ASSERT_NO_FATAL_FAILURE(RemoveProducer());
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  output_.resize(packet_.device_length);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = buffers_[1],
              .target = output_.data(),
              .length = output_.size(),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(consumer_, 1, &download, &value));
  for (auto*& buffer : buffers_) {
    IREE_ASSERT_OK(loom_serve_execution_dealloca(consumer_, buffer, &value));
    iree_hal_buffer_release(std::exchange(buffer, nullptr));
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_));
  EXPECT_EQ(output_[0], 0x5A);
  EXPECT_EQ(iree_unaligned_load_le_u32(output_.data() + 128), Fingerprint(1));
  EXPECT_EQ(iree_unaligned_load_le_u32(output_.data() + 144), Fingerprint(9));
}

TEST_F(MediaExecutionTest, DenseRangesCoalesceAcrossRequestsAndContinuation) {
  ASSERT_NO_FATAL_FAILURE(Submit({
      R"({"model":"fixture","input":"a"})",
      R"({"model":"fixture","input":"b"})",
      R"({"model":"fixture","input":"x"})",
  }));
  const loom_serve_text_media_input_span_t images_a[] = {
      {
          .input_offset = 0,
          .input_count = 1,
          .request_index = 2,
          .result = result_,
      },
      {
          .input_offset = 1,
          .input_count = 1,
          .request_index = 0,
          .result = result_,
      },
  };
  const loom_serve_text_media_input_span_t image_b = {
      .input_offset = 0,
      .input_count = 1,
      .request_index = 1,
      .result = result_,
  };
  const loom_serve_text_media_input_t media_a = {images_a, 2};
  const loom_serve_text_media_input_t media_b = {&image_b, 1};
  const int32_t token = 0;
  const loom_serve_text_span_t first[] = {
      {
          .row_index = 0,
          .token_count = 1,
          .token_ids = &token,
          .media = &media_a,
      },
      {
          .row_index = 1,
          .token_count = 1,
          .token_ids = &token,
          .media = &media_b,
      },
  };
  loom_serve_text_span_t next = {
      .row_index = 0,
      .token_count = 1,
      .token_ids = &token,
      .input_offset = 1,
      .media = &media_a,
  };
  IREE_ASSERT_OK(loom_serve_text_media_epoch_initialize(
      2, first, 2, 1, &next, 1, host_allocator_, &packet_));
  EXPECT_EQ(packet_.first_feature_count, 2u);
  EXPECT_EQ(packet_.next_feature_count, 1u);
  EXPECT_EQ(packet_.device_length, 144u);
  EXPECT_EQ(packet_.transfer.result_count, 3u);
  ASSERT_EQ(packet_.transfer.operation_count, 1u);
  const auto& copy = packet_.transfer.operations[0];
  EXPECT_EQ(copy.type, IREE_HAL_TRANSFER_OPERATION_TYPE_COPY);
  EXPECT_EQ(copy.copy.source_offset, 0u);
  EXPECT_EQ(copy.copy.target_offset, 96u);
  EXPECT_EQ(copy.copy.length, 48u);
  iree_const_byte_span_t plan;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      packet_.plan, 0, iree_vm_buffer_length(packet_.plan), &plan));
  EXPECT_EQ(iree_unaligned_load_le_u32(plan.data + 2 * 28),
            LOOM_SERVE_TEXT_ROW_CAPACITY);
  loom_serve_text_media_epoch_deinitialize(&packet_);

  // Already consumed images leave no metadata, packet, or transfer allocation.
  next.input_offset = 2;
  IREE_ASSERT_OK(loom_serve_text_media_epoch_initialize(
      1, &next, 4, 0, nullptr, 0, host_allocator_, &packet_));
  EXPECT_EQ(packet_.plan, nullptr);
  EXPECT_EQ(packet_.packet_records, nullptr);
  EXPECT_EQ(packet_.transfer.operations, nullptr);
}

}  // namespace
