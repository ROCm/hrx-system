// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/buffer.h"

#include <array>
#include <cstdarg>
#include <map>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

namespace {
struct NativeBuffers {
  // Native GEM table: repeated PRIME import returns the same unrefcounted
  // handle.
  std::map<int, uint32_t> imports;
  // Next native handle assigned by the modeled kernel.
  uint32_t next_handle = 1;
  // Per-handle number of map, unmap, and close operations.
  std::map<uint32_t, std::array<uint32_t, 3>> operations;
  // Native allocation domain returned by GEM_OP.
  uint32_t domain = AMDGPU_GEM_DOMAIN_GTT;
  // Injected map failure; only the native syscall dependency is replaced.
  int map_error = 0;
  // Injected unmap failure used to check terminal ownership preservation.
  int unmap_error = 0;
};
thread_local NativeBuffers* native_buffers = nullptr;
}  // namespace

extern "C" int __real_ioctl(int descriptor, unsigned long request, ...);
extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_buffers == nullptr) {
    return __real_ioctl(descriptor, request, argument);
  }
  EXPECT_EQ(descriptor, 17);
  auto& native = *native_buffers;
  switch (request) {
    case DRM_IOCTL_PRIME_FD_TO_HANDLE: {
      auto* import = static_cast<drm_prime_handle*>(argument);
      auto [entry, inserted] =
          native.imports.emplace(import->fd, native.next_handle);
      if (inserted) {
        ++native.next_handle;
      }
      import->handle = entry->second;
      return 0;
    }
    case DRM_IOCTL_AMDGPU_GEM_OP: {
      const auto* query = static_cast<drm_amdgpu_gem_op*>(argument);
      EXPECT_EQ(query->op, AMDGPU_GEM_OP_GET_GEM_CREATE_INFO);
      auto* information =
          reinterpret_cast<drm_amdgpu_gem_create_in*>(query->value);
      information->domains = native.domain;
      return 0;
    }
    case DRM_IOCTL_AMDGPU_GEM_VA: {
      const auto* mapping = static_cast<drm_amdgpu_gem_va*>(argument);
      const bool unmap = mapping->operation == AMDGPU_VA_OP_UNMAP;
      EXPECT_TRUE(unmap || mapping->operation == AMDGPU_VA_OP_MAP);
      ++native.operations[mapping->handle][unmap ? 1 : 0];
      const int error = unmap ? native.unmap_error : native.map_error;
      if (error != 0) {
        errno = error;
        return -1;
      }
      return 0;
    }
    case DRM_IOCTL_GEM_CLOSE: {
      const auto* close = static_cast<drm_gem_close*>(argument);
      ++native.operations[close->handle][2];
      for (auto it = native.imports.begin(); it != native.imports.end();) {
        if (it->second == close->handle) {
          it = native.imports.erase(it);
        } else {
          ++it;
        }
      }
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected native operation " << request;
      errno = ENOTTY;
      return -1;
  }
}

namespace {
class MediaBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_buffers = &native_;
    device_.base.base.host_allocator = amdf_allocator_system();
    device_.descriptor = 17;
    device_.page_size = 4096;
    device_.address_begin = 4096;
    device_.address_end = 4096 * 5;
    ASSERT_EQ(pthread_mutex_init(&device_.memory_mutex, nullptr), 0);
  }
  void TearDown() override {
    for (auto*& binding : bindings_) {
      if (binding != nullptr) {
        const auto status = amdf_media_drm_buffer_release(&binding);
        EXPECT_EQ(status, device_.memory_status);
        amdf_media_drm_buffer_abandon(&binding);
      }
    }
    EXPECT_EQ(device_.buffers, nullptr);
    EXPECT_EQ(pthread_mutex_destroy(&device_.memory_mutex), 0);
    native_buffers = nullptr;
  }
  // Production native owner and VM index, with only ioctl supplied by the
  // model.
  amdf_media_drm_device_t device_ = {};
  // Inert native dependency; contains no actual GEM/VA resources.
  NativeBuffers native_;
  // Live attachment owners, including partially prepared failures.
  std::array<amdf_media_drm_buffer_t*, 4> bindings_ = {};
};

TEST_F(MediaBufferTest, DuplicateImportsCloseOneHandleAndReuseItsAddressGap) {
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[0]),
      AMDF_STATUS_OK);
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[1]),
      AMDF_STATUS_OK);
  ASSERT_EQ(bindings_[0], bindings_[1]);
  const uint32_t first = bindings_[0]->handle;
  const uint64_t address = bindings_[0]->address;
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 31, 4096, false, &bindings_[2]),
      AMDF_STATUS_OK);
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[0]), AMDF_STATUS_OK);
  EXPECT_EQ(native_.operations[first], (std::array<uint32_t, 3>{1, 0, 0}));
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[1]), AMDF_STATUS_OK);
  EXPECT_EQ(native_.operations[first], (std::array<uint32_t, 3>{1, 1, 1}));
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 32, 4096, false, &bindings_[3]),
      AMDF_STATUS_OK);
  EXPECT_EQ(bindings_[3]->address, address);
}

TEST_F(MediaBufferTest, FailedPreparationCannotPublishAnAliasDuringRollback) {
  native_.map_error = ENOMEM;
  const auto error = amdf_linux_error(ENOMEM);
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[0]),
      error);
  const uint32_t handle = bindings_[0]->handle;
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[1]),
      error);
  ASSERT_EQ(bindings_[0], bindings_[1]);
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[0]), AMDF_STATUS_OK);
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[1]), AMDF_STATUS_OK);
  EXPECT_EQ(native_.operations[handle], (std::array<uint32_t, 3>{1, 0, 1}));
}

TEST_F(MediaBufferTest,
       FailedUnmapQuarantinesNativeOwnershipAndConsumesMetadata) {
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[0]),
      AMDF_STATUS_OK);
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 31, 4096, false, &bindings_[1]),
      AMDF_STATUS_OK);
  const uint32_t handle = bindings_[0]->handle;
  const uint32_t peer = bindings_[1]->handle;
  native_.unmap_error = EIO;
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[0]),
            amdf_linux_error(EIO));
  ASSERT_NE(bindings_[0], nullptr);
  amdf_media_drm_buffer_abandon(&bindings_[0]);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(device_.memory_status, lost);
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[1]), lost);
  amdf_media_drm_buffer_abandon(&bindings_[1]);
  EXPECT_EQ(native_.operations[handle], (std::array<uint32_t, 3>{1, 1, 0}));
  EXPECT_EQ(native_.operations[peer], (std::array<uint32_t, 3>{1, 0, 0}));
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[2]),
      lost);
  EXPECT_EQ(native_.next_handle, 3u);
}

TEST_F(MediaBufferTest, SystemProfileRejectsLocalPlacementBeforeMapping) {
  native_.domain = AMDGPU_GEM_DOMAIN_VRAM;
  ASSERT_EQ(
      amdf_media_drm_buffer_prepare(&device_, 30, 4096, false, &bindings_[0]),
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  const uint32_t handle = bindings_[0]->handle;
  ASSERT_EQ(amdf_media_drm_buffer_release(&bindings_[0]), AMDF_STATUS_OK);
  EXPECT_EQ(native_.operations[handle], (std::array<uint32_t, 3>{0, 0, 1}));
}
}  // namespace
