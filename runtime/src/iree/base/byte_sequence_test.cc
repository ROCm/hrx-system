// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/byte_sequence.h"

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;

using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, void (*)(iree_byte_sequence_t*)>;

typedef struct test_byte_sequence_t {
  // Byte sequence interface exposed to the implementation under test.
  iree_byte_sequence_t base;
  // Ordered segments returned during enumeration.
  const iree_const_byte_span_t* segments;
  // Number of entries in |segments|.
  iree_host_size_t segment_count;
  // Optional counter incremented when the sequence is destroyed.
  int* destroy_count;
} test_byte_sequence_t;

static const test_byte_sequence_t* test_byte_sequence_const_cast(
    const iree_byte_sequence_t* base_sequence) {
  return (const test_byte_sequence_t*)base_sequence;
}

static test_byte_sequence_t* test_byte_sequence_cast(
    iree_byte_sequence_t* base_sequence) {
  return (test_byte_sequence_t*)base_sequence;
}

static void test_byte_sequence_destroy(iree_byte_sequence_t* base_sequence) {
  test_byte_sequence_t* sequence = test_byte_sequence_cast(base_sequence);
  if (sequence->destroy_count) ++*sequence->destroy_count;
}

static iree_status_t test_byte_sequence_enumerate(
    const iree_byte_sequence_t* base_sequence,
    iree_byte_sequence_segment_callback_t callback) {
  const test_byte_sequence_t* sequence =
      test_byte_sequence_const_cast(base_sequence);
  for (iree_host_size_t i = 0; i < sequence->segment_count; ++i) {
    IREE_RETURN_IF_ERROR(
        callback.fn(callback.user_data, sequence->segments[i]));
  }
  return iree_ok_status();
}

static const iree_byte_sequence_vtable_t test_byte_sequence_vtable = {
    test_byte_sequence_destroy,
    test_byte_sequence_enumerate,
};

static void test_byte_sequence_initialize(
    const iree_const_byte_span_t* segments, iree_host_size_t segment_count,
    uint64_t length, int* destroy_count, test_byte_sequence_t* out_sequence) {
  iree_byte_sequence_initialize(&test_byte_sequence_vtable, length,
                                &out_sequence->base);
  out_sequence->segments = segments;
  out_sequence->segment_count = segment_count;
  out_sequence->destroy_count = destroy_count;
}

static iree_status_t append_segment(void* user_data,
                                    iree_const_byte_span_t segment) {
  auto* contents = static_cast<std::vector<uint8_t>*>(user_data);
  contents->insert(contents->end(), segment.data,
                   segment.data + segment.data_length);
  return iree_ok_status();
}

static iree_byte_sequence_segment_callback_t make_append_callback(
    std::vector<uint8_t>* contents) {
  return {
      append_segment,
      contents,
  };
}

TEST(ByteSequenceTest, OwnedSpanMovesStorage) {
  iree_byte_span_t source_span = iree_byte_span_empty();
  source_span.data_length = 4;
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), source_span.data_length,
      (void**)&source_span.data));
  const uint8_t expected[] = {1, 2, 3, 4};
  memcpy(source_span.data, expected, sizeof(expected));

  iree_byte_sequence_t* sequence = NULL;
  IREE_ASSERT_OK(iree_byte_sequence_create_from_span_move(
      &source_span, iree_allocator_system(), &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);

  EXPECT_EQ(source_span.data, nullptr);
  EXPECT_EQ(source_span.data_length, 0u);
  EXPECT_EQ(iree_byte_sequence_length(sequence), sizeof(expected));
  std::vector<uint8_t> actual;
  IREE_EXPECT_OK(
      iree_byte_sequence_enumerate(sequence, make_append_callback(&actual)));
  EXPECT_THAT(actual, ElementsAre(1, 2, 3, 4));
}

TEST(ByteSequenceTest, OwnedEmptySpanProducesRealSequence) {
  iree_byte_span_t source_span = iree_byte_span_empty();
  iree_byte_sequence_t* sequence = NULL;
  IREE_ASSERT_OK(iree_byte_sequence_create_from_span_move(
      &source_span, iree_allocator_system(), &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);

  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(iree_byte_sequence_length(sequence), 0u);
  std::vector<uint8_t> actual;
  IREE_EXPECT_OK(
      iree_byte_sequence_enumerate(sequence, make_append_callback(&actual)));
  EXPECT_TRUE(actual.empty());
}

TEST(ByteSequenceTest, RetainReleaseControlsLifetime) {
  int destroy_count = 0;
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(NULL, 0, 0, &destroy_count, &sequence);

  iree_byte_sequence_retain(&sequence.base);
  iree_byte_sequence_release(&sequence.base);
  EXPECT_EQ(destroy_count, 0);
  iree_byte_sequence_release(&sequence.base);
  EXPECT_EQ(destroy_count, 1);

  iree_byte_sequence_retain(NULL);
  iree_byte_sequence_release(NULL);
}

TEST(ByteSequenceTest, EnumeratesMultipleSegmentsInOrder) {
  const uint8_t segment0[] = {0, 1};
  const uint8_t segment1[] = {2};
  const uint8_t segment2[] = {3, 4, 5};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(segment0, sizeof(segment0)),
      iree_make_const_byte_span(segment1, sizeof(segment1)),
      iree_make_const_byte_span(segment2, sizeof(segment2)),
  };
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 6, NULL,
                                &sequence);

  std::vector<uint8_t> actual;
  IREE_EXPECT_OK(iree_byte_sequence_enumerate(&sequence.base,
                                              make_append_callback(&actual)));
  EXPECT_THAT(actual, ElementsAre(0, 1, 2, 3, 4, 5));

  iree_byte_sequence_release(&sequence.base);
}

typedef struct failing_callback_state_t {
  // Number of callbacks observed before the failure was returned.
  int callback_count;
} failing_callback_state_t;

static iree_status_t fail_on_first_segment(void* user_data,
                                           iree_const_byte_span_t segment) {
  (void)segment;
  failing_callback_state_t* state = (failing_callback_state_t*)user_data;
  ++state->callback_count;
  return iree_make_status(IREE_STATUS_ABORTED, "intentional callback failure");
}

TEST(ByteSequenceTest, CallbackFailureStopsEnumeration) {
  const uint8_t segment0[] = {0};
  const uint8_t segment1[] = {1};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(segment0, sizeof(segment0)),
      iree_make_const_byte_span(segment1, sizeof(segment1)),
  };
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 2, NULL,
                                &sequence);

  failing_callback_state_t state = {0};
  iree_byte_sequence_segment_callback_t callback = {
      fail_on_first_segment,
      &state,
  };
  Status status(iree_byte_sequence_enumerate(&sequence.base, callback));
  EXPECT_THAT(status, StatusIs(StatusCode::kAborted));
  EXPECT_THAT(status.ToString(), HasSubstr("intentional callback failure"));
  EXPECT_EQ(state.callback_count, 1);

  iree_byte_sequence_release(&sequence.base);
}

TEST(ByteSequenceTest, SupportsConcurrentEnumeration) {
  const uint8_t segment0[] = {0, 1};
  const uint8_t segment1[] = {2, 3};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(segment0, sizeof(segment0)),
      iree_make_const_byte_span(segment1, sizeof(segment1)),
  };
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 4, NULL,
                                &sequence);

  std::atomic<bool> all_match(true);
  const std::vector<uint8_t> expected = {0, 1, 2, 3};
  std::array<std::thread, 4> threads;
  for (std::thread& thread : threads) {
    thread = std::thread([&]() {
      for (int i = 0; i < 100; ++i) {
        std::vector<uint8_t> actual;
        Status status(iree_byte_sequence_enumerate(
            &sequence.base, make_append_callback(&actual)));
        if (!status.ok() || actual != expected) {
          all_match.store(false);
          return;
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_TRUE(all_match.load());

  iree_byte_sequence_release(&sequence.base);
}

TEST(ByteSequenceTest, CloneCombinesSegmentsAndOwnsResult) {
  const uint8_t segment0[] = {0, 1};
  const uint8_t segment1[] = {2};
  const uint8_t segment2[] = {3, 4, 5};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(segment0, sizeof(segment0)),
      iree_make_const_byte_span(segment1, sizeof(segment1)),
      iree_make_const_byte_span(segment2, sizeof(segment2)),
  };
  int destroy_count = 0;
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 6,
                                &destroy_count, &sequence);

  iree_byte_span_t clone = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(&sequence.base,
                                          iree_allocator_system(), &clone));
  iree_byte_sequence_release(&sequence.base);

  EXPECT_EQ(destroy_count, 1);
  EXPECT_THAT(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
              ElementsAre(0, 1, 2, 3, 4, 5));
  iree_allocator_free(iree_allocator_system(), clone.data);
}

TEST(ByteSequenceTest, CloneEmptySequenceDoesNotAllocate) {
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(NULL, 0, 0, NULL, &sequence);

  uint8_t sentinel = 0;
  iree_byte_span_t clone = iree_make_byte_span(&sentinel, 1);
  IREE_EXPECT_OK(
      iree_byte_sequence_clone(&sequence.base, iree_allocator_null(), &clone));
  EXPECT_EQ(clone.data, nullptr);
  EXPECT_EQ(clone.data_length, 0u);

  iree_byte_sequence_release(&sequence.base);
}

typedef struct controlled_allocator_t {
  // Allocator receiving commands that are not intentionally failed.
  iree_allocator_t delegate;
  // Whether allocation commands return resource exhausted.
  bool fail_allocations;
  // Number of successful allocation commands forwarded to |delegate|.
  int allocation_count;
  // Total requested byte length across successful allocation commands.
  iree_host_size_t allocation_bytes;
  // Number of successful free commands forwarded to |delegate|.
  int free_count;
} controlled_allocator_t;

static iree_status_t controlled_allocator_ctl(void* self,
                                              iree_allocator_command_t command,
                                              const void* params,
                                              void** inout_ptr) {
  controlled_allocator_t* allocator = (controlled_allocator_t*)self;
  if (allocator->fail_allocations &&
      (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
       command == IREE_ALLOCATOR_COMMAND_CALLOC ||
       command == IREE_ALLOCATOR_COMMAND_REALLOC)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "intentional allocation failure");
  }
  iree_status_t status = allocator->delegate.ctl(allocator->delegate.self,
                                                 command, params, inout_ptr);
  if (iree_status_is_ok(status)) {
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      ++allocator->allocation_count;
      allocator->allocation_bytes +=
          ((const iree_allocator_alloc_params_t*)params)->byte_length;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->free_count;
    }
  }
  return status;
}

TEST(ByteSequenceTest, CloneAllocatesOneExactLengthBuffer) {
  const uint8_t segment0[] = {0, 1};
  const uint8_t segment1[] = {2};
  const uint8_t segment2[] = {3, 4, 5};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(segment0, sizeof(segment0)),
      iree_make_const_byte_span(segment1, sizeof(segment1)),
      iree_make_const_byte_span(segment2, sizeof(segment2)),
  };
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 6, NULL,
                                &sequence);
  controlled_allocator_t allocator_state = {
      iree_allocator_system(), false, 0, 0, 0,
  };
  iree_allocator_t allocator = {
      &allocator_state,
      controlled_allocator_ctl,
  };

  iree_byte_span_t clone = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(&sequence.base, allocator, &clone));
  EXPECT_EQ(allocator_state.allocation_count, 1);
  EXPECT_EQ(allocator_state.allocation_bytes, 6u);
  EXPECT_THAT(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
              ElementsAre(0, 1, 2, 3, 4, 5));

  iree_allocator_free(allocator, clone.data);
  EXPECT_EQ(allocator_state.free_count, 1);
  iree_byte_sequence_release(&sequence.base);
}

TEST(ByteSequenceTest, MoveFailurePreservesSource) {
  controlled_allocator_t allocator_state = {
      iree_allocator_system(), false, 0, 0, 0,
  };
  iree_allocator_t allocator = {
      &allocator_state,
      controlled_allocator_ctl,
  };
  iree_byte_span_t source_span = iree_byte_span_empty();
  source_span.data_length = 4;
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      allocator, source_span.data_length, (void**)&source_span.data));
  uint8_t* original_data = source_span.data;
  allocator_state.fail_allocations = true;

  test_byte_sequence_t sentinel;
  iree_byte_sequence_t* sequence = &sentinel.base;
  EXPECT_THAT(Status(iree_byte_sequence_create_from_span_move(
                  &source_span, allocator, &sequence)),
              StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(sequence, nullptr);
  EXPECT_EQ(source_span.data, original_data);
  EXPECT_EQ(source_span.data_length, 4u);

  allocator_state.fail_allocations = false;
  iree_allocator_free(allocator, source_span.data);
  EXPECT_EQ(allocator_state.free_count, 1);
}

TEST(ByteSequenceTest, CloneAllocationFailureLeavesOutputEmpty) {
  const uint8_t data[] = {0, 1};
  const iree_const_byte_span_t segments[] = {
      iree_make_const_byte_span(data, sizeof(data)),
  };
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(segments, IREE_ARRAYSIZE(segments), 2, NULL,
                                &sequence);
  controlled_allocator_t allocator_state = {
      iree_allocator_system(), true, 0, 0, 0,
  };
  iree_allocator_t allocator = {
      &allocator_state,
      controlled_allocator_ctl,
  };

  uint8_t sentinel = 0;
  iree_byte_span_t clone = iree_make_byte_span(&sentinel, 1);
  EXPECT_THAT(
      Status(iree_byte_sequence_clone(&sequence.base, allocator, &clone)),
      StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(clone.data, nullptr);
  EXPECT_EQ(clone.data_length, 0u);
  EXPECT_EQ(allocator_state.free_count, 0);

  iree_byte_sequence_release(&sequence.base);
}

TEST(ByteSequenceTest, CloneRejectsOversizedLength) {
#if defined(IREE_PTR_SIZE_32)
  test_byte_sequence_t sequence;
  test_byte_sequence_initialize(NULL, 0, (uint64_t)IREE_HOST_SIZE_MAX + 1, NULL,
                                &sequence);

  uint8_t sentinel = 0;
  iree_byte_span_t clone = iree_make_byte_span(&sentinel, 1);
  EXPECT_THAT(Status(iree_byte_sequence_clone(&sequence.base,
                                              iree_allocator_null(), &clone)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_EQ(clone.data, nullptr);
  EXPECT_EQ(clone.data_length, 0u);

  iree_byte_sequence_release(&sequence.base);
#else
  GTEST_SKIP() << "uint64_t sequences cannot exceed a 64-bit host size";
#endif  // IREE_PTR_SIZE_32
}

}  // namespace
