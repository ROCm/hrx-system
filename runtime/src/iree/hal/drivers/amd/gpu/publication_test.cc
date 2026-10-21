// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/publication.h"

#include <array>
#include <atomic>
#include <barrier>
#include <memory>
#include <semaphore>
#include <thread>

#include "iree/hal/drivers/amd/gpu/queue/publisher.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(PublicationTest, IdleOwnerInvokesNoSource) {
  uint32_t count = 0;
  const iree_hal_amd_gpu_publication_source_t source = {
      .publish =
          [](void* user_data) {
            ++*static_cast<uint32_t*>(user_data);
            return false;
          },
      .user_data = &count,
  };
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      1, &source, {}, iree_allocator_system(), &publication));
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  EXPECT_EQ(count, 0u);
}

TEST(PublicationTest, OnlyNotifiedSourcesRunAcrossEveryReadyWord) {
  struct Source {
    // Number of observed native publication turns.
    uint32_t count = 0;
    // Publishes a completed source turn to the test controller.
    std::counting_semaphore<256>* completed;
  };
  std::counting_semaphore<256> completed(0);
  std::array<Source, IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES> queues;
  std::array<iree_hal_amd_gpu_publication_source_t,
             IREE_HAL_AMD_GPU_PUBLICATION_MAX_SOURCES>
      sources;
  for (size_t i = 0; i < sources.size(); ++i) {
    queues[i].completed = &completed;
    sources[i] = {
        .publish =
            [](void* user_data) {
              auto* source = static_cast<Source*>(user_data);
              ++source->count;
              source->completed->release();
              return false;
            },
        .user_data = &queues[i],
    };
  }
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      sources.size(), sources.data(), {}, iree_allocator_system(),
      &publication));
  for (uint32_t index :
       {0u, 63u, 64u, 127u, 128u, 191u, 192u, 255u, 256u, 257u}) {
    iree_hal_amd_gpu_publication_notify(&publication, index);
  }
  for (uint32_t i = 0; i < 10; ++i) {
    completed.acquire();
  }
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  for (size_t i = 0; i < queues.size(); ++i) {
    EXPECT_EQ(queues[i].count,
              i % 64 == 0 || i % 64 == 63 || i == 257 ? 1u : 0u);
  }
}

TEST(PublicationTest, NotificationDuringFinalObservationIsRetained) {
  struct Source {
    // Number of turns, observed only after joining the worker.
    uint32_t count = 0;
    // First publication has observed its final native readiness predicate.
    std::binary_semaphore entered{0};
    // Controller releases that last publisher access.
    std::binary_semaphore resume{0};
    // A second publication consumed the racing ready bit.
    std::binary_semaphore completed{0};
  } queue;
  const iree_hal_amd_gpu_publication_source_t source = {
      .publish =
          [](void* user_data) {
            auto* queue = static_cast<Source*>(user_data);
            if (++queue->count == 1) {
              queue->entered.release();
              queue->resume.acquire();
            } else {
              queue->completed.release();
            }
            return false;
          },
      .user_data = &queue,
  };
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      1, &source, {}, iree_allocator_system(), &publication));
  iree_hal_amd_gpu_publication_notify(&publication, 0);
  queue.entered.acquire();
  for (uint32_t i = 0; i < 1000; ++i) {
    iree_hal_amd_gpu_publication_notify(&publication, 0);
  }
  queue.resume.release();
  queue.completed.acquire();
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  EXPECT_EQ(queue.count, 2u);
}

TEST(PublicationTest, ReplacementJoinsLastSourceAccess) {
  struct Source {
    // Separately owned handoff gates outlive source reclamation.
    std::binary_semaphore* entered;
    // Authorizes the callback's final access to this source.
    std::binary_semaphore* resume;
    // Used once after the controller begins removing this source.
    uint32_t count = 0;
  };
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  std::binary_semaphore replacing{0};
  auto queue = std::make_unique<Source>(Source{&entered, &resume});
  auto source = std::make_unique<iree_hal_amd_gpu_publication_source_t>(
      iree_hal_amd_gpu_publication_source_t{
          .publish =
              [](void* user_data) {
                auto* source = static_cast<Source*>(user_data);
                source->entered->release();
                source->resume->acquire();
                ++source->count;
                return false;
              },
          .user_data = queue.get(),
      });
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      1, source.get(), {}, iree_allocator_system(), &publication));
  iree_hal_amd_gpu_publication_notify(&publication, 0);
  entered.acquire();
  std::atomic<bool> replaced = false;
  std::thread owner([&] {
    replacing.release();
    iree_hal_amd_gpu_publication_set_sources(&publication, 0, nullptr);
    EXPECT_EQ(queue->count, 1u);
    queue.reset();
    source.reset();
    replaced.store(true, std::memory_order_release);
  });
  replacing.acquire();
  EXPECT_FALSE(replaced.load(std::memory_order_acquire));
  resume.release();
  owner.join();
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  EXPECT_TRUE(replaced.load());
}

TEST(PublicationTest, ReplacementSupportsVacantSlotsAndGrowingReadyExtent) {
  std::counting_semaphore<2> first_completed{0};
  std::counting_semaphore<2> second_completed{0};
  const iree_hal_amd_gpu_publication_source_t source = {
      .publish =
          [](void* user_data) {
            static_cast<std::counting_semaphore<2>*>(user_data)->release();
            return false;
          },
      .user_data = &first_completed,
  };
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      0, nullptr, {}, iree_allocator_system(), &publication));
  std::array<iree_hal_amd_gpu_publication_source_t, 65> first = {};
  first[64] = source;
  iree_hal_amd_gpu_publication_set_sources(&publication, first.size(),
                                           first.data());
  iree_hal_amd_gpu_publication_notify(&publication, 64);
  first_completed.acquire();

  // A removed ordinal can leave a coalesced hint. Its vacant slot has no
  // callback, while its neighbor continues to publish normally.
  std::array<iree_hal_amd_gpu_publication_source_t, 129> second = {};
  second[128] = source;
  second[128].user_data = &second_completed;
  iree_hal_amd_gpu_publication_notify(&publication, 64);
  iree_hal_amd_gpu_publication_set_sources(&publication, second.size(),
                                           second.data());
  iree_hal_amd_gpu_publication_notify(&publication, 128);
  second_completed.acquire();
  iree_hal_amd_gpu_publication_set_sources(&publication, 0, nullptr);
  iree_hal_amd_gpu_publication_deinitialize(&publication);
}

TEST(PublicationTest, ReadySuffixYieldsToAnotherQueue) {
  struct Source {
    // Shared ready-set owner, initialized before notification.
    iree_hal_amd_gpu_publication_t* publication;
    // Number of bounded turns for the large ready prefix.
    uint32_t count = 0;
    // Prefix progress when the other queue receives its turn.
    uint32_t peer_observed = 0;
    // Both queues publish their terminal observation.
    std::counting_semaphore<2> completed{0};
  } queue;
  iree_hal_amd_gpu_publication_t publication;
  queue.publication = &publication;
  const iree_hal_amd_gpu_publication_source_t sources[] = {
      {
          .publish =
              [](void* user_data) {
                auto* queue = static_cast<Source*>(user_data);
                if (++queue->count == 1) {
                  iree_hal_amd_gpu_publication_notify(queue->publication, 1);
                }
                if (queue->count != 128) {
                  return true;
                }
                queue->completed.release();
                return false;
              },
          .user_data = &queue,
      },
      {
          .publish =
              [](void* user_data) {
                auto* queue = static_cast<Source*>(user_data);
                queue->peer_observed = queue->count;
                queue->completed.release();
                return false;
              },
          .user_data = &queue,
      },
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      IREE_ARRAYSIZE(sources), sources, {}, iree_allocator_system(),
      &publication));
  iree_hal_amd_gpu_publication_notify(&publication, 0);
  queue.completed.acquire();
  queue.completed.acquire();
  iree_hal_amd_gpu_publication_deinitialize(&publication);
  EXPECT_EQ(queue.count, 128u);
  EXPECT_EQ(queue.peer_observed, 2u);
}

TEST(PublicationTest, ConcurrentCallersAndDeferredPublisherRetireEveryEntry) {
  constexpr uint32_t kThreadCount = 8;
  constexpr uint32_t kEntryCount = 128;
  struct Queue {
    // Real queue publication ownership, shared by API callers and the worker.
    iree_hal_amd_gpu_publisher_t publisher;
    // Deferred turn owner, initialized before API submission starts.
    iree_hal_amd_gpu_publication_t publication;
    // Detached entries, counted while exclusive publication is held.
    uint32_t consumed = 0;
    // Terminal observation after the last accepted entry is detached.
    std::binary_semaphore completed{0};

    void ConsumeOne() {
      if (iree_hal_amd_gpu_publisher_next(&publisher)) {
        if (++consumed == kThreadCount * kEntryCount) {
          completed.release();
        }
      }
    }

    void Submit(iree_hal_amd_gpu_publication_entry_t* entry) {
      if (iree_hal_amd_gpu_publisher_enqueue(&publisher, entry)) {
        ConsumeOne();
        if (iree_hal_amd_gpu_publisher_handoff(&publisher, nullptr)) {
          iree_hal_amd_gpu_publication_notify(&publication, 0);
        }
      }
    }
  } queue;
  iree_hal_amd_gpu_publisher_initialize(&queue.publisher);
  const iree_hal_amd_gpu_publication_source_t source = {
      .publish =
          [](void* user_data) {
            auto* queue = static_cast<Queue*>(user_data);
            if (!iree_hal_amd_gpu_publisher_acquire_service(
                    &queue->publisher)) {
              return false;
            }
            queue->ConsumeOne();
            return iree_hal_amd_gpu_publisher_handoff(&queue->publisher,
                                                      nullptr);
          },
      .user_data = &queue,
  };
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      1, &source, {}, iree_allocator_system(), &queue.publication));
  std::array<std::array<iree_hal_amd_gpu_publication_entry_t, kEntryCount>,
             kThreadCount>
      entries = {};
  std::array<std::thread, kThreadCount> threads;
  std::barrier<> start(kThreadCount);
  for (uint32_t i = 0; i < kThreadCount; ++i) {
    threads[i] = std::thread([&, i] {
      start.arrive_and_wait();
      for (auto& entry : entries[i]) {
        queue.Submit(&entry);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  queue.completed.acquire();
  iree_hal_amd_gpu_publication_deinitialize(&queue.publication);
  iree_hal_amd_gpu_publisher_deinitialize(&queue.publisher);
  EXPECT_EQ(queue.consumed, kThreadCount * kEntryCount);
}

TEST(PublicationTest, NativeResultReleasesQueueBeforeCallbackThreadJoins) {
  struct Queue {
    // Native result publication and final ownership boundary under test.
    iree_hal_amd_gpu_publisher_t publisher;
    // Last result required for this queue to become idle.
    iree_hal_amd_gpu_publication_entry_t entry = {};
  };
  struct Source {
    // Queue lifetime ends when its last result is consumed.
    std::unique_ptr<Queue> queue;
    // The enclosing cold owner can remove this now-empty source.
    std::binary_semaphore retired{0};
  };
  iree_hal_amd_gpu_publication_t publication;
  IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
      0, nullptr, {}, iree_allocator_system(), &publication));
  for (uint32_t i = 0; i < 256; ++i) {
    Source source{std::make_unique<Queue>()};
    iree_hal_amd_gpu_publisher_initialize(&source.queue->publisher);
    const iree_hal_amd_gpu_publication_source_t binding = {
        .publish =
            [](void* user_data) {
              auto* source = static_cast<Source*>(user_data);
              if (!source->queue || !iree_hal_amd_gpu_publisher_acquire_service(
                                        &source->queue->publisher)) {
                return false;
              }
              EXPECT_EQ(
                  iree_hal_amd_gpu_publisher_next(&source->queue->publisher),
                  &source->queue->entry);
              EXPECT_FALSE(iree_hal_amd_gpu_publisher_handoff(
                  &source->queue->publisher, nullptr));
              iree_hal_amd_gpu_publisher_deinitialize(
                  &source->queue->publisher);
              source->queue.reset();
              source->retired.release();
              return false;
            },
        .user_data = &source,
    };
    iree_hal_amd_gpu_publication_set_sources(&publication, 1, &binding);
    std::binary_semaphore native_tail{0};
    auto* queue = source.queue.get();
    std::thread callback([&, queue] {
      iree_hal_amd_gpu_publication_enqueue(&publication, 0, &queue->publisher,
                                           &queue->entry);
      // Native callback bookkeeping may continue after queue retirement. It
      // borrows only the device; queue release need not drain that thread.
      native_tail.acquire();
    });
    source.retired.acquire();
    iree_hal_amd_gpu_publication_set_sources(&publication, 0, nullptr);
    EXPECT_EQ(source.queue, nullptr);
    native_tail.release();
    callback.join();
  }
  iree_hal_amd_gpu_publication_deinitialize(&publication);
}

}  // namespace
