// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/host_mapping.h"

#include "gtest/gtest.h"
#include "libamdf/src/memory.h"

namespace {

struct NativeMapping {
  // Common production mapping whose native dependency is recorded below.
  amdf_host_mapping_t base;
  // Number of native cache operations dispatched.
  uint32_t operation_count = 0;
  // Last operation requested at the native boundary.
  amdf_host_cache_operation_t operation = AMDF_HOST_CACHE_OPERATION_NONE;
  // View-relative offset received by the native adapter.
  uint64_t byte_offset = 0;
  // Exact logical byte range passed to the native adapter.
  uint64_t byte_length = 0;
  // Fallible native result propagated by the public boundary.
  amdf_status_t status = AMDF_STATUS_OK;
};

amdf_status_t RecordCacheControl(amdf_host_mapping_t* base,
                                 amdf_host_cache_operation_t operation,
                                 uint64_t byte_offset, uint64_t byte_length) {
  auto* mapping = reinterpret_cast<NativeMapping*>(base);
  ++mapping->operation_count;
  mapping->operation = operation;
  mapping->byte_offset = byte_offset;
  mapping->byte_length = byte_length;
  return mapping->status;
}

const amdf_host_mapping_vtable_t kNativeMappingVtable = {
    .cache_control = RecordCacheControl,
};

class HostMappingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // The memory dependency is host-coherent for its GPU. Its CPU mapping
    // still advertises real operations needed by a noncoherent consumer.
    memory_.accesses = &access_;
    memory_.info.access_count = 1;
    access_.info.flags = AMDF_MEMORY_FLAG_HOST_COHERENT;
    ASSERT_EQ(amdf_host_mapping_initialize(&mapping_.base,
                                           &kNativeMappingVtable, &memory_),
              AMDF_STATUS_OK);
    mapping_.base.info.byte_length = 256;
    mapping_.base.info.memory_byte_offset = 4096;
    mapping_.base.info.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    mapping_.base.info.flags =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    mapping_.base.info.flush.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
    mapping_.base.info.invalidate.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
  }

  void TearDown() override { amdf_host_mapping_deinitialize(&mapping_.base); }

  // Backing dependency with no native allocation required by this boundary.
  amdf_memory_t memory_ = {};
  // GPU coherence facts are separate from CPU cache-control mechanisms.
  amdf_memory_access_state_t access_ = {};
  // Recording native dependency behind the real common mapping operations.
  NativeMapping mapping_ = {};
};

TEST_F(HostMappingTest, ExplicitOperationsAreIndependentOfDeviceCoherence) {
  EXPECT_EQ(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 17, 83),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_.operation_count, 1u);
  EXPECT_EQ(mapping_.operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
  EXPECT_EQ(mapping_.byte_offset, 17u);
  EXPECT_EQ(mapping_.byte_length, 83u);
  EXPECT_EQ(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 255, 1),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_.operation_count, 2u);
  EXPECT_EQ(mapping_.operation, AMDF_HOST_CACHE_OPERATION_INVALIDATE);
  EXPECT_EQ(mapping_.byte_offset, 255u);
}

TEST_F(HostMappingTest, RejectsInvalidAccessAndRangesBeforeNativeWork) {
  EXPECT_EQ(amdf_status_code(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 255, 2)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(
      amdf_status_code(amdf_host_mapping_cache_control(
          &mapping_.base, AMDF_HOST_CACHE_OPERATION_INVALIDATE, UINT64_MAX, 1)),
      AMDF_STATUS_CODE_INVALID_ARGUMENT);
  mapping_.base.info.flags = AMDF_MEMORY_MAP_FLAG_READ;
  EXPECT_EQ(amdf_status_code(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 1)),
            AMDF_STATUS_CODE_FAILED_PRECONDITION);
  EXPECT_EQ(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 256, 0),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapping_.operation_count, 0u);
}

TEST_F(HostMappingTest, DistinguishesNoOperationUnsupportedAndNativeFailure) {
  mapping_.base.info.flush.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  EXPECT_EQ(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 256),
            AMDF_STATUS_OK);
  mapping_.base.info.flush.kind = AMDF_CACHE_TRANSITION_KIND_UNKNOWN;
  EXPECT_EQ(amdf_status_code(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, 256)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  EXPECT_EQ(mapping_.operation_count, 0u);
  mapping_.status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_host_mapping_cache_control(
                &mapping_.base, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0, 256),
            mapping_.status);
  EXPECT_EQ(mapping_.operation_count, 1u);
}

}  // namespace
