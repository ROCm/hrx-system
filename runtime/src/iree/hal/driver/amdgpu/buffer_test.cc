// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/buffer.h"

#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using BufferPtr =
    std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>;

class ForeignBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // A foreign provider publishes only generic binding slots. No AMDGPU
    // private capability extension follows this array.
    static const uint16_t types[] = {
        IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
        IREE_HAL_BUFFER_INTERFACE_HOST,
    };
    const iree_hal_buffer_binding_layout_t layout = {
        sizeof(bindings_), IREE_ARRAYSIZE(types), 1, types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 4, &layout, iree_allocator_system(), &contract_));
    contract_->scopes[2].interfaces =
        1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    contract_->scopes[2].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    bindings_[0].device_address = reinterpret_cast<uintptr_t>(storage_);
    bindings_[1].host_pointer = reinterpret_cast<uint8_t*>(storage_);
    backing_.allocation_alignment = alignof(uint64_t);
    backing_.maintenance_alignment = 1;
  }

  void TearDown() override { iree_hal_memory_contract_release(contract_); }

  BufferPtr CreateBuffer(
      iree_hal_atomic_operation_capabilities_t capabilities) {
    backing_.atomic_operations = capabilities;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE,
        sizeof(storage_), iree_make_byte_span(storage_, sizeof(storage_)),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &buffer));
    buffer->memory.contract = contract_;
    buffer->memory.bindings = bindings_;
    buffer->memory.backing = &backing_;
    return BufferPtr(buffer, iree_hal_buffer_release);
  }

  // Borrowed allocation bytes retained until all views are released.
  uint64_t storage_[4] = {};
  // Generic native slots intentionally contain no backend-private extension.
  iree_hal_buffer_native_binding_t bindings_[2] = {};
  // Immutable backing capabilities published before a buffer is consumed.
  iree_hal_buffer_backing_facts_t backing_ = {};
  // Shared access contract retained independently by the foreign provider.
  iree_hal_memory_contract_t* contract_ = nullptr;
};

TEST_F(ForeignBufferTest,
       CompleteCapabilitiesReachAtomicValidationThroughViews) {
  for (uint32_t cells = 0;
       cells <= IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL; ++cells) {
    auto buffer =
        CreateBuffer(iree_hal_amdgpu_atomic_memory_expand_capabilities(cells));
    EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(buffer.get()), cells);
    iree_hal_buffer_t* subspan = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer.get(), 8, 16,
                                           iree_allocator_system(), &subspan));
    auto view = BufferPtr(subspan, iree_hal_buffer_release);
    const auto view_cells =
        iree_hal_amdgpu_buffer_atomic_memory_cells(view.get());
    EXPECT_EQ(view_cells, cells);
    auto* view_pointer =
        static_cast<uint8_t*>(iree_hal_amdgpu_buffer_device_pointer(
            iree_hal_buffer_allocated_buffer(view.get()))) +
        iree_hal_buffer_byte_offset(view.get());
    EXPECT_EQ(view_pointer, reinterpret_cast<uint8_t*>(&storage_[1]));
    IREE_EXPECT_OK(iree_hal_amdgpu_atomic_memory_validate_required_cells(
        view_cells, view_pointer, cells,
        IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT));
  }
}

TEST_F(ForeignBufferTest,
       PartialOperationFamiliesCannotAdvertiseCompleteCells) {
  for (uint32_t omitted = 1; omitted <= IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL;
       omitted <<= 1) {
    if (!(omitted & IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL)) {
      continue;
    }
    const auto partial = IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL & ~omitted;
    const iree_hal_atomic_operation_capabilities_t capabilities = {
        partial, partial, partial, partial};
    auto buffer = CreateBuffer(capabilities);
    EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(buffer.get()), 0u);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INCOMPATIBLE,
        iree_hal_amdgpu_atomic_memory_validate_target(
            iree_hal_amdgpu_buffer_atomic_memory_cells(buffer.get()), storage_,
            IREE_HAL_ATOMIC_WIDTH_32, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
            IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT));
  }
}

TEST_F(ForeignBufferTest, AbsentBackingDoesNotReadPrivateBindingExtensions) {
  auto buffer = CreateBuffer({});
  buffer->memory.backing = nullptr;
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(buffer.get()), 0u);
}

TEST_F(ForeignBufferTest, UnscopedForeignStorageIsNotQualified) {
  auto buffer = CreateBuffer(iree_hal_amdgpu_atomic_memory_expand_capabilities(
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL));
  buffer->memory.contract = nullptr;
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(buffer.get()), 0u);
}

}  // namespace
}  // namespace iree::hal::amdgpu
