// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/memory.h"
#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/fabric/transient_buffer.h"
#include "iree/hal/driver/amd/fabric/transient_buffer_pool.h"

namespace iree::hal::amd::fabric {

TEST_F(MemoryBackendTest, TransientSlotsPreserveMixedNativeBindingsAndFacts) {
  binding_descriptions_[0] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                              AMDF_MEMORY_ADDRESS_GPU};
  native_.address_kinds[0] = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreatePool(IREE_HAL_BUFFER_USAGE_STORAGE, /*interfaces=*/0, &pool));
  const auto request = iree_hal_make_pool_reservation_request(64, 64);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, request, iree_infinite_timeout(), &backing));
  const iree_hal_pool_reservation_view_t view = {
      .buffer = backing,
      .byte_offset = iree_hal_buffer_byte_offset(backing),
      .byte_length = iree_hal_buffer_byte_length(backing),
      .memory = iree_hal_buffer_memory_view(backing),
  };
  iree_hal_amd_fabric_transient_buffer_pool_t buffers;
  iree_hal_amd_fabric_transient_buffer_pool_initialize(allocator_, &buffers);
  iree_hal_buffer_t* root = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_fabric_transient_buffer_create_batch(
      {}, iree_hal_buffer_memory_type(backing),
      iree_hal_buffer_allowed_access(backing),
      iree_hal_buffer_allowed_usage(backing), pool, &buffers, 1, &request,
      &root));
  iree_hal_buffer_t* child = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(root, 8, 16, allocator_, &child));
  IREE_ASSERT_OK(
      iree_hal_amd_fabric_transient_buffer_stage_backing(root, &view));
  iree_hal_amd_fabric_transient_buffer_commit(root);
  iree_hal_buffer_release(backing);
  iree_hal_buffer_release(root);
  iree_hal_amd_fabric_transient_buffer_pool_trim(&buffers);

  const auto memory = iree_hal_buffer_memory_view(child);
  const auto* facts =
      reinterpret_cast<const iree_hal_amd_fabric_memory_binding_facts_t*>(
          memory.bindings + memory.contract->binding_layout.binding_count);
  EXPECT_NE(facts->memory->mapping, nullptr);
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_buffer_native_binding_slot_t slot;
    IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
        pool, ResolveScope(IREE_HAL_MEMORY_SITE_PROGRAM, families_[i]),
        binding_descriptions_[i].type, &slot));
    EXPECT_EQ(iree_hal_buffer_native_binding(child, slot).device_address,
              UINT64_C(0x100008) + i * UINT64_C(0x10000));
    EXPECT_EQ(
        facts->scope_facts[families_[i]->memory.queue_scope_id].access_ordinal,
        i);
  }
  EXPECT_EQ(native_.destroy_count, 0);
  iree_hal_buffer_release(child);
  iree_hal_amd_fabric_transient_buffer_pool_deinitialize(&buffers);
  iree_hal_pool_release(pool);
  native_.AwaitEmpty();
  EXPECT_EQ(native_.unmap_count, 1);
  EXPECT_EQ(native_.destroy_count, 1);
}

}  // namespace iree::hal::amd::fabric
