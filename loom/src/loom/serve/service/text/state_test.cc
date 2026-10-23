// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/state.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/environment.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/testing/service_application.h"

namespace {

// Draft models independently choose a cache, origin table, and bounded carry.
// Sequence models retain the same opaque carry without speculative storage.
class TextStateTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    const loom_serve_text_options_t options = {
        .mode = (GetParam() & 8) ? LOOM_SERVE_TEXT_MODE_ENCODE
                                 : LOOM_SERVE_TEXT_MODE_GENERATE,
        .context_capacity = 17,
        .pool_capacity = 32,
        .epoch_count = 1,
        .draft_depth = (GetParam() & 8) ? 0u : 5u,
        .row_count = 2,
        .checkpoint_capacity = 2,
    };
    loom_serve_text_state_initialize(&options, host_allocator, &state);
    auto application_options = loom_serve_test_application_options_default();
    if (application_options.backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
      application_options.memory_limit = 64 * 1024 * 1024;
    }
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &application_options, host_allocator, &application));
    device = loom_serve_test_application_device_at(application, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
    IREE_ASSERT_OK(iree_vm_environment_allocate(host_allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));

    const bool cache = (GetParam() & 1) != 0;
    const bool table = (GetParam() & 2) != 0;
    const uint64_t carry_stride = (GetParam() & 4) ? 198 * 1024 : 0;
    const iree_host_size_t lengths[] = {11 * 24, 2 * 4 * 16, 16, 16,
                                        64 + (cache ? 2u : 1u) * 40};
    for (size_t i = 0; i < IREE_ARRAYSIZE(lengths); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(
          iree_vm_buffer_create(lengths[i], 8, host_allocator, &buffer));
      IREE_ASSERT_OK(
          iree_vm_buffer_map_write(buffer, 0, lengths[i], &bytes[i]));
      memset(bytes[i].data, 0, lengths[i]);
      results[i] = iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
    }
    const uint64_t allocations[] = {64,
                                    576,
                                    64,
                                    56,
                                    64,
                                    64,
                                    4 * carry_stride,
                                    (GetParam() & 8) ? 0u : 64u,
                                    (GetParam() & 8) ? 0u : 64u,
                                    cache ? 128u : 0u,
                                    table ? 56u : 0u};
    for (size_t i = 0; i < IREE_ARRAYSIZE(allocations); ++i) {
      iree_unaligned_store_le_u64(bytes[0].data + i * 24, allocations[i]);
      iree_unaligned_store_le_u64(bytes[0].data + i * 24 + 8, 256);
      iree_unaligned_store_le_u64(bytes[0].data + i * 24 + 16,
                                  i == 1   ? 448
                                  : i == 6 ? allocations[i]
                                           : 0);
    }
    for (size_t row = 0; row < 2; ++row) {
      const uint64_t views[] = {row * 128,      8, 0, 0, row * 128 + 16, 16,
                                row * 128 + 48, 32};
      StoreWords(bytes[1].data + row * 64, views, IREE_ARRAYSIZE(views));
    }
    // Unallocated tables may still have source payloads, but no upload
    // consumer. Allocated tables preserve these bytes before their page maps.
    memset(bytes[2].data, 0x31, 16);
    memset(bytes[3].data, 0x72, 16);
    const uint64_t geometry[] = {4, 16, carry_stride, 32, 256, 32, 16,
                                 6, 1,  448,          1,  128, 16, 9,
                                 0, 1,  128,          16};
    StoreWords(bytes[4].data, geometry, lengths[4] / sizeof(uint64_t));
  }

  void TearDown() override {
    loom_serve_text_state_retire(&state, nullptr, nullptr);
    if (execution) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution));
    }
    loom_serve_text_state_await_retirement(&state);
    IREE_EXPECT_OK(loom_serve_text_state_deinitialize(&state));
    loom_serve_execution_release(execution);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    iree_vm_environment_free(environment);
  }

  static void StoreWords(uint8_t* target, const uint64_t* values,
                         size_t count) {
    for (size_t i = 0; i < count; ++i) {
      iree_unaligned_store_le_u64(target + i * sizeof(*values), values[i]);
    }
  }

  iree_status_t Allocate() {
    IREE_RETURN_IF_ERROR(loom_serve_text_storage_initialize(
        &types, results, 2, 17, 32, host_allocator, &state.storage));
    return loom_serve_text_state_allocate(&state, device, execution);
  }

  iree_status_t Upload(iree_hal_buffer_t* buffer, uint64_t offset,
                       const std::vector<uint8_t>& data) {
    const iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .upload =
            {
                .source = data.data(),
                .target_buffer = buffer,
                .target_offset = offset,
                .length = data.size(),
            },
    };
    uint64_t completion = 0;
    IREE_RETURN_IF_ERROR(
        loom_serve_execution_transfer(execution, 1, &upload, &completion));
    return loom_serve_execution_wait(execution, completion);
  }

  void CheckBytes(iree_hal_buffer_t* buffer, uint64_t offset,
                  const std::vector<uint8_t>& expected) {
    std::vector<uint8_t> actual(expected.size());
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffer,
                .source_offset = offset,
                .target = actual.data(),
                .length = actual.size(),
            },
    };
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(execution, 1, &download, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution, completion));
    EXPECT_EQ(actual, expected);
  }

  void CheckMaps() {
    for (iree_hal_buffer_t* table :
         {state.epoch.buffers[2], state.draft.row_table}) {
      if (!table) {
        continue;
      }
      for (size_t row = 0; row < state.row_count; ++row) {
        const size_t block_count = state.rows[row].block_count;
        if (!block_count) {
          continue;
        }
        const size_t offset = row * state.storage.blocks_per_row;
        std::vector<uint8_t> expected(block_count * sizeof(uint32_t));
        for (size_t i = 0; i < block_count; ++i) {
          iree_unaligned_store_le_u32(expected.data() + i * sizeof(uint32_t),
                                      state.cache.maps[offset + i]);
        }
        ASSERT_NO_FATAL_FAILURE(CheckBytes(
            table, state.storage.map_origin + offset * sizeof(uint32_t),
            expected));
      }
    }
  }

  void CacheContents(loom_serve_text_state_row_t* row, bool upload) {
    const uint32_t* blocks =
        state.cache.maps + (row - state.rows) * state.storage.blocks_per_row;
    for (size_t region = 0; region < state.storage.region_count; ++region) {
      const auto& layout = state.storage.regions[region];
      auto* buffer =
          layout.allocation == 1 ? state.row_arena : state.draft.cache;
      for (uint32_t block = 0; block < row->block_count; ++block) {
        const uint64_t offset = layout.blocks.origin + blocks[block] * 16;
        std::vector<uint8_t> data(16, 1 + region * 37 + block * 13);
        if (upload) {
          IREE_ASSERT_OK(Upload(buffer, offset, data));
        } else {
          ASSERT_NO_FATAL_FAILURE(CheckBytes(buffer, offset, data));
        }
      }
    }
  }

  // Allocator enclosing bootstrap metadata and state retirement.
  const iree_allocator_t host_allocator = iree_allocator_system();
  // Borrowed process-wide device used by this physical-target test suite.
  loom_serve_device_t* device = nullptr;
  // Owned serialized model execution domain.
  loom_serve_execution_t* execution = nullptr;
  // VM buffer type owner for the source bootstrap values.
  iree_vm_environment_t* environment = nullptr;
  // Resolved canonical reference handles for bootstrap buffers.
  iree_vm_ref_types_t types = {};
  // Source bootstrap values moved into storage by Allocate.
  iree_vm_variant_t results[5] = {};
  // Borrowed mutable records used before crossing the storage boundary.
  iree_byte_span_t bytes[5] = {};
  // Production native state, including its actual pool and queue ownership.
  loom_serve_text_state_t state = {};
};

TEST_P(TextStateTest, OptionalRootsRetainOriginsMapsAndCheckpointCarry) {
  IREE_ASSERT_OK(Allocate());
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  EXPECT_EQ(state.draft.cache != nullptr, (GetParam() & 1) != 0);
  EXPECT_EQ(state.draft.row_table != nullptr, (GetParam() & 2) != 0);
  EXPECT_EQ(state.carry != nullptr, (GetParam() & 4) != 0);
  EXPECT_EQ(state.draft.committed != nullptr, (GetParam() & 8) == 0);
  EXPECT_EQ(state.draft.results != nullptr, (GetParam() & 8) == 0);
  ASSERT_NO_FATAL_FAILURE(
      CheckBytes(state.epoch.buffers[2], 0, std::vector<uint8_t>(16, 0x31)));
  if (state.draft.row_table) {
    ASSERT_NO_FATAL_FAILURE(
        CheckBytes(state.draft.row_table, 0, std::vector<uint8_t>(16, 0x72)));
  }
  loom_serve_text_storage_release_payloads(&state.storage);

  const int32_t tokens[] = {0, 1, 2, 1, 0};
  const loom_serve_text_span_t span = {0, 5, tokens};
  const uint32_t extent = 5;
  IREE_ASSERT_OK(loom_serve_text_state_grow(&state, 1, &span, &extent));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  ASSERT_NO_FATAL_FAILURE(CheckMaps());
  auto* row = &state.rows[0];
  ASSERT_NO_FATAL_FAILURE(CacheContents(row, true));
  const std::vector<uint8_t> recurrent(16, 0x58);
  IREE_ASSERT_OK(Upload(row->buffers[TEXT_RECURRENT], 0, recurrent));
  std::vector<uint8_t> carry(state.storage.carry_stride);
  for (size_t i = 0; i < carry.size(); ++i) {
    carry[i] = static_cast<uint8_t>((i * 17 + i / 256 + 3) % 251);
  }
  if (!carry.empty()) {
    IREE_ASSERT_OK(Upload(state.carry, 0, carry));
  }
  // Publish a consumed frontier after its writes, as model_complete_epoch
  // does. Source position need not equal the number of consumed inputs.
  row->position = 5;
  row->source_position = 9;
  loom_serve_text_state_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_state_row_submit_pin(row, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  const std::vector<uint8_t> changed(carry.size(), 0xA5);
  if (!carry.empty()) {
    IREE_ASSERT_OK(Upload(state.carry, 0, changed));
  }
  auto* fork = &state.rows[1];
  bool restored = false;
  IREE_ASSERT_OK(
      loom_serve_text_state_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  EXPECT_EQ(fork->position, 5u);
  EXPECT_EQ(fork->source_position, 9u);
  ASSERT_NO_FATAL_FAILURE(CheckMaps());
  ASSERT_NO_FATAL_FAILURE(CacheContents(fork, false));
  if (!carry.empty()) {
    ASSERT_NO_FATAL_FAILURE(CheckBytes(state.carry, 0, changed));
    ASSERT_NO_FATAL_FAILURE(
        CheckBytes(state.carry, state.storage.carry_stride, carry));
  }
  if (state.memory.pool) {
    IREE_ASSERT_OK(loom_serve_text_state_checkpoint_submit_suspend(checkpoint));
    IREE_ASSERT_OK(loom_serve_execution_drain(execution));
    loom_serve_text_state_checkpoint_complete_suspend(checkpoint);
    EXPECT_EQ(loom_serve_snapshot_size(checkpoint->snapshot),
              16 + carry.size() + state.storage.region_count * 32);
    // Destroy the live endpoint copy so only the host snapshot can recover it.
    if (!carry.empty()) {
      const auto index =
          state.row_count + (checkpoint - state.checkpoints.values);
      IREE_ASSERT_OK(
          Upload(state.carry, index * state.storage.carry_stride, changed));
    }
    IREE_ASSERT_OK(
        loom_serve_text_state_row_try_restore(fork, checkpoint, &restored));
    ASSERT_TRUE(restored);
    ASSERT_NO_FATAL_FAILURE(CacheContents(fork, false));
    ASSERT_NO_FATAL_FAILURE(
        CheckBytes(fork->buffers[TEXT_RECURRENT], 0, recurrent));
    if (!carry.empty()) {
      ASSERT_NO_FATAL_FAILURE(
          CheckBytes(state.carry, state.storage.carry_stride, carry));
    }
    IREE_ASSERT_OK(loom_serve_text_state_row_suspend(fork));
    if (!carry.empty()) {
      IREE_ASSERT_OK(Upload(state.carry, state.storage.carry_stride, changed));
    }
    bool resumed = false;
    IREE_ASSERT_OK(loom_serve_text_state_row_try_resume(fork, &resumed));
    ASSERT_TRUE(resumed);
    ASSERT_NO_FATAL_FAILURE(CheckMaps());
    ASSERT_NO_FATAL_FAILURE(CacheContents(fork, false));
    if (!carry.empty()) {
      ASSERT_NO_FATAL_FAILURE(
          CheckBytes(state.carry, state.storage.carry_stride, carry));
    }
    // The restored endpoint uses newer physical IDs. Retiring the original
    // prefix leaves holes, forcing compaction and publication of remapped IDs.
    loom_serve_text_state_row_reset(row);
    loom_serve_text_trim_result_t result;
    IREE_ASSERT_OK(loom_serve_text_state_submit_trim(&state, &result));
    bool complete = false;
    while (!complete) {
      IREE_ASSERT_OK(loom_serve_execution_drain(execution));
      IREE_ASSERT_OK(
          loom_serve_text_state_complete_trim(&state, &result, &complete));
    }
    EXPECT_GT(result.moved_blocks, 0u);
    ASSERT_NO_FATAL_FAILURE(CheckMaps());
    ASSERT_NO_FATAL_FAILURE(CacheContents(fork, false));
  }
  // Capacity reservation detaches shared tails and publishes the same maps
  // to whichever origin tables the source actually allocated.
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_text_state_row_submit_reserve(
      fork, LOOM_SERVE_TEXT_RESERVE_CONTINUE, nullptr, 9, nullptr, nullptr,
      &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  loom_serve_text_state_row_complete_reserve(fork);
  ASSERT_NO_FATAL_FAILURE(CheckMaps());
  loom_serve_text_state_row_release_reservation(fork);
  loom_serve_text_state_checkpoint_release(checkpoint);
}

TEST_P(TextStateTest, CarryFollowsSourceStride) {
  const auto offset = 6 * 24;
  iree_unaligned_store_le_u64(bytes[0].data + offset,
                              (GetParam() & 4) ? 0 : 64);
  iree_unaligned_store_le_u64(bytes[0].data + offset + 16, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Allocate());
}

TEST_P(TextStateTest, CarryRequiresPackedEpochs) {
  if (!(GetParam() & 4)) {
    return;
  }
  state.flags &= ~LOOM_SERVE_TEXT_STATE_FLAG_PACKED;
  const iree::Status status = Allocate();
  EXPECT_EQ(status.code(), iree::StatusCode::kInvalidArgument);
  EXPECT_NE(status.ToString().find("source carry requires packed epochs"),
            std::string::npos);
}

TEST_P(TextStateTest, CommittedSpansFollowDraftScheduling) {
  iree_unaligned_store_le_u64(bytes[0].data + 7 * 24,
                              (GetParam() & 8) ? 64 : 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Allocate());
}

TEST_P(TextStateTest, FeedbackFollowsDraftScheduling) {
  iree_unaligned_store_le_u64(bytes[0].data + 8 * 24,
                              (GetParam() & 8) ? 64 : 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Allocate());
}

INSTANTIATE_TEST_SUITE_P(DraftStorage, TextStateTest, ::testing::Range(0u, 8u));
INSTANTIATE_TEST_SUITE_P(SequenceStorage, TextStateTest,
                         ::testing::Values(8u, 12u));

}  // namespace
