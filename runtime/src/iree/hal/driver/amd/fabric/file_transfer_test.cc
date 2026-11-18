// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/file_transfer.h"

#include <array>
#include <vector>

#include "iree/hal/driver/amd/fabric/buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct BufferStorage {
  struct CacheOperation {
    // Native cache operation requested by the production buffer view.
    amdf_host_cache_operation_t operation;
    // Byte position within the native mapping.
    uint64_t offset;
    // Exact byte count of the requested maintenance.
    uint64_t length;
  };

  static amdf_status_t AMDF_CALL CacheControl(
      amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
      uint64_t offset, uint64_t length) {
    auto* self = reinterpret_cast<BufferStorage*>(mapping);
    self->cache_operations.push_back({operation, offset, length});
    return self->cache_status;
  }

  static void Release(void* user_data, iree_hal_buffer_t* buffer) {
    ++static_cast<BufferStorage*>(user_data)->release_count;
  }

  // Native backing bytes, with guard space around the logical buffer.
  alignas(64) std::array<uint8_t, 256> contents;
  // Mapping information borrowed by the production buffer view.
  iree_hal_amd_fabric_memory_t memory = {};
  // Prepared physical-view identity, independent of a parent-buffer lookup.
  iree_hal_buffer_backing_facts_t backing = {};
  // Native dependency providing cache maintenance.
  amdf_api_t api = {};
  // Persistent host execution binding independent of public map grants.
  iree_hal_buffer_native_binding_t binding = {};
  // Native source suffix consumed by the shared physical-range resolver.
  iree_hal_amd_fabric_memory_binding_facts_t binding_facts = {};
  // Observed native cache operations.
  std::vector<CacheOperation> cache_operations;
  // Native cache result under test.
  amdf_status_t cache_status = AMDF_STATUS_OK;
  // Number of final backing-view releases.
  size_t release_count = 0;
};

static_assert(offsetof(BufferStorage, binding_facts) ==
              offsetof(BufferStorage, binding) +
                  sizeof(iree_hal_buffer_native_binding_t));

class FileTransferTest
    : public ::testing::TestWithParam<iree_hal_memory_access_t> {
 protected:
  void SetUp() override {
    storage_.contents.fill(0xA5);
    storage_.memory.contents =
        iree_make_byte_span(storage_.contents.data(), storage_.contents.size());
    storage_.memory.mapping = reinterpret_cast<amdf_host_mapping_t*>(&storage_);
    storage_.api.host_mapping_cache_control = BufferStorage::CacheControl;
    storage_.binding.host_pointer = storage_.contents.data();
    iree_hal_amd_fabric_buffer_storage_t backing = {
        .memory = &storage_.memory,
        .bindings = &storage_.binding,
        .host_binding_index = 0,
        .offset = 8,
    };
    iree_hal_buffer_t* root = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_fabric_buffer_wrap(
        {}, &storage_.api, backing, 192, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
        {BufferStorage::Release, &storage_}, iree_allocator_system(), &root));
    iree_status_t status = iree_hal_buffer_subspan(
        root, 16, 96, iree_allocator_system(), &buffer_);
    iree_hal_buffer_release(root);
    IREE_ASSERT_OK(status);
  }
  void TearDown() override {
    iree_hal_buffer_release(buffer_);
    iree_hal_memory_contract_release(contract_);
    EXPECT_EQ(storage_.release_count, 1u);
  }
  // Logical range inside a nested native buffer view.
  static constexpr uint64_t kBufferOffset = 4;
  // Exact useful range, excluding surrounding guard bytes.
  static constexpr uint64_t kLength = 32;
  // Physical allocation origin plus both view-relative offsets.
  static constexpr uint64_t kNativeOffset = 8 + 16 + kBufferOffset;
  // Native mapping dependency used by the production physical-source query.
  BufferStorage storage_;
  // Logical view borrowing the native mapping until teardown.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Backing contract outliving the view.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // Dense facts indexed by queue scope, with no lookup during qualification.
  std::array<iree_hal_amd_fabric_memory_scope_facts_t, 4> scope_facts_ = {};
};

TEST_P(FileTransferTest, NativeAdmissionUsesAchievedBackingCapabilities) {
  enum class Route { kHost, kCopy };
  struct Case {
    // Achieved physical storage class, independent of requested placement.
    amdf_memory_class_t memory_class;
    // Native qualification of the prepared host mapping.
    amdf_host_mapping_features_t features;
    // Expected native route for both buffered and DIRECT files.
    Route route;
  };
  const Case cases[] = {
      {AMDF_MEMORY_CLASS_SYSTEM, 0, Route::kCopy},
      {AMDF_MEMORY_CLASS_SYSTEM, AMDF_HOST_MAPPING_FEATURE_FILE_IO,
       Route::kHost},
      {AMDF_MEMORY_CLASS_LOCAL, 0, Route::kCopy},
      {AMDF_MEMORY_CLASS_LOCAL, AMDF_HOST_MAPPING_FEATURE_FILE_IO,
       Route::kHost},
      {AMDF_MEMORY_CLASS_PRIVATE, 0, Route::kCopy},
      {AMDF_MEMORY_CLASS_PRIVATE, AMDF_HOST_MAPPING_FEATURE_FILE_IO,
       Route::kHost},
  };
  const uint16_t type = IREE_HAL_BUFFER_INTERFACE_HOST;
  const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(storage_.binding) + sizeof(storage_.binding_facts),
      .binding_count = 1,
      .host_binding_index = 0,
      .types = &type,
  };
  IREE_ASSERT_OK(
      iree_hal_memory_contract_create(&storage_, scope_facts_.size(), &layout,
                                      iree_allocator_system(), &contract_));
  contract_->scopes[2].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_->memory.contract = contract_;
  buffer_->memory.backing = &storage_.backing;
  buffer_->memory.offset = 8 + 16;
  storage_.binding_facts = {&storage_.memory, scope_facts_.data()};
  storage_.memory.host_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  storage_.api.memory_query_pair_info =
      +[](const amdf_memory_site_t*, const amdf_memory_site_t*,
          amdf_memory_pair_info_t* pair) -> amdf_status_t {
    pair->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    pair->release.kind = pair->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    return AMDF_STATUS_OK;
  };
  iree_hal_queue_family_t family = {};
  family.memory.domain = &storage_;
  family.memory.queue_scope_id = 2;
  for (const auto& item : cases) {
    SCOPED_TRACE(::testing::Message()
                 << item.memory_class << ':' << item.features);
    storage_.memory.info.memory_class = item.memory_class;
    storage_.memory.host_features = item.features;
    iree_hal_memory_effects_t before, after;
    iree_hal_amd_fabric_file_source_t source;
    iree::Status status = iree_hal_amd_fabric_file_source_initialize(
        &storage_.api, &family, 0, GetParam(),
        {buffer_, kBufferOffset, kLength}, &source, &before, &after);
    IREE_ASSERT_OK(status);
    const auto& range = source;
    EXPECT_EQ(range.storage, &storage_.memory);
    EXPECT_EQ(range.source_offset, kNativeOffset);
    EXPECT_EQ(range.contents.data_length, kLength);
    EXPECT_EQ(range.contents.data,
              item.route == Route::kHost
                  ? storage_.contents.data() + kNativeOffset
                  : nullptr);
    EXPECT_EQ(before.bits, 0u);
    EXPECT_EQ(after.bits, 0u);
    EXPECT_TRUE(storage_.cache_operations.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, FileTransferTest,
                         ::testing::Values(IREE_HAL_MEMORY_ACCESS_READ,
                                           IREE_HAL_MEMORY_ACCESS_WRITE));

}  // namespace
