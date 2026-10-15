// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/fence.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>

#include "iree/async/platform/posix/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(PosixFenceTest, NativeRegistrationFailureFailsAcceptedImport) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_posix(iree_async_proactor_options_default(),
                                       iree_allocator_system(), &proactor));
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, 0, iree_allocator_system(), &semaphore));
  int descriptors[2];
  ASSERT_EQ(pipe(descriptors), 0);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(
      proactor, iree_async_primitive_from_fd(descriptors[0]), semaphore, 1));

  // The import has already accepted ownership. Inject the native registration
  // failure at the event-set boundary while running the actual poll path.
  auto* event_set = iree_async_proactor_posix_cast(proactor)->event_set;
  const auto* original_vtable = event_set->vtable;
  auto failing_vtable = *original_vtable;
  failing_vtable.add = [](iree_async_posix_event_set_t*, int,
                          short) -> iree_status_t {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  };
  event_set->vtable = &failing_vtable;
  IREE_EXPECT_OK(
      iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  event_set->vtable = original_vtable;

  EXPECT_EQ(iree_async_semaphore_query_status(semaphore),
            IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);
  EXPECT_EQ(fcntl(descriptors[0], F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(iree_async_proactor_posix_cast(proactor)->fd_map.entry_count, 0u);
  close(descriptors[1]);
  iree_async_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST(PosixFenceTest, IndexAllocationFailurePreservesOtherImports) {
  struct Allocator {
    // Enabled only after every import has accepted its tracker allocation.
    bool fail = false;
    static iree_status_t Control(void* self, iree_allocator_command_t command,
                                 const void* parameters, void** pointer) {
      if (static_cast<Allocator*>(self)->fail &&
          command != IREE_ALLOCATOR_COMMAND_FREE) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
      auto system = iree_allocator_system();
      return system.ctl(system.self, command, parameters, pointer);
    }
  } allocator;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_posix(
      iree_async_proactor_options_default(), {&allocator, Allocator::Control},
      &proactor));
  constexpr size_t kCount = 32;
  std::array<iree_async_semaphore_t*, kCount> semaphores{};
  std::array<std::array<int, 2>, kCount> descriptors;
  for (size_t i = 0; i < kCount; ++i) {
    IREE_ASSERT_OK(iree_async_semaphore_create(
        proactor, 0, 0, iree_allocator_system(), &semaphores[i]));
    ASSERT_EQ(pipe(descriptors[i].data()), 0);
    IREE_ASSERT_OK(iree_async_semaphore_import_fence(
        proactor, iree_async_primitive_from_fd(descriptors[i][0]),
        semaphores[i], 1));
  }

  allocator.fail = true;
  IREE_EXPECT_OK(
      iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  allocator.fail = false;
  size_t failed = 0;
  size_t pending = 0;
  for (size_t i = 0; i < kCount; ++i) {
    if (fcntl(descriptors[i][0], F_GETFD) == -1) {
      EXPECT_EQ(errno, EBADF);
      EXPECT_EQ(iree_async_semaphore_query_status(semaphores[i]),
                IREE_STATUS_RESOURCE_EXHAUSTED);
      ++failed;
    } else {
      EXPECT_EQ(iree_async_semaphore_query_status(semaphores[i]),
                IREE_STATUS_OK);
      const char value = 1;
      ASSERT_EQ(write(descriptors[i][1], &value, sizeof(value)), sizeof(value));
      ++pending;
    }
  }
  EXPECT_GT(failed, 0u);
  EXPECT_GT(pending, 0u);

  size_t completed = 0;
  while (completed < pending) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), &count));
    completed += count;
  }
  for (size_t i = 0; i < kCount; ++i) {
    if (iree_async_semaphore_query_status(semaphores[i]) == IREE_STATUS_OK) {
      EXPECT_EQ(iree_async_semaphore_query(semaphores[i]), 1u);
    }
    close(descriptors[i][1]);
    iree_async_semaphore_release(semaphores[i]);
  }
  EXPECT_EQ(iree_async_proactor_posix_cast(proactor)->fd_map.entry_count, 0u);
  iree_async_proactor_release(proactor);
}

}  // namespace
