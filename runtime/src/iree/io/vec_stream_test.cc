// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/vec_stream.h"

#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::StatusOr;
using iree::testing::status::StatusIs;
using testing::ElementsAre;

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;
using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, void (*)(iree_byte_sequence_t*)>;

typedef struct segment_view_t {
  // Borrowed segment data pointer.
  const uint8_t* data;
  // Segment length in bytes.
  iree_host_size_t data_length;
} segment_view_t;

static iree_status_t collect_segment_view(void* user_data,
                                          iree_const_byte_span_t segment) {
  auto* segments = static_cast<std::vector<segment_view_t>*>(user_data);
  segments->push_back({segment.data, segment.data_length});
  return iree_ok_status();
}

typedef struct segment_comparison_state_t {
  // Segment views captured from the source stream.
  const std::vector<segment_view_t>* expected_segments;
  // Ordinal of the next expected segment.
  iree_host_size_t index;
  // Whether every segment observed so far matched.
  bool all_match;
} segment_comparison_state_t;

static iree_status_t compare_segment_view(void* user_data,
                                          iree_const_byte_span_t segment) {
  auto* state = static_cast<segment_comparison_state_t*>(user_data);
  if (state->index >= state->expected_segments->size()) {
    state->all_match = false;
    return iree_ok_status();
  }
  const segment_view_t expected = (*state->expected_segments)[state->index++];
  if (segment.data != expected.data ||
      segment.data_length != expected.data_length) {
    state->all_match = false;
  }
  return iree_ok_status();
}

static iree_status_t count_segment(void* user_data,
                                   iree_const_byte_span_t segment) {
  (void)segment;
  ++*(iree_host_size_t*)user_data;
  return iree_ok_status();
}

typedef struct tracking_allocator_t {
  // Allocator receiving commands that are not intentionally failed.
  iree_allocator_t delegate;
  // Whether allocation commands return resource exhausted.
  bool fail_allocations;
  // Number of successful allocation commands.
  iree_host_size_t allocation_count;
  // Number of successful free commands.
  iree_host_size_t free_count;
} tracking_allocator_t;

static iree_status_t tracking_allocator_ctl(void* self,
                                            iree_allocator_command_t command,
                                            const void* params,
                                            void** inout_ptr) {
  tracking_allocator_t* allocator = (tracking_allocator_t*)self;
  const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                             command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                             command == IREE_ALLOCATOR_COMMAND_REALLOC;
  if (allocator->fail_allocations && is_allocation) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "intentional allocation failure");
  }
  iree_status_t status = allocator->delegate.ctl(allocator->delegate.self,
                                                 command, params, inout_ptr);
  if (iree_status_is_ok(status)) {
    if (is_allocation) {
      ++allocator->allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->free_count;
    }
  }
  return status;
}

static iree_allocator_t make_tracking_allocator(tracking_allocator_t* state) {
  return {
      state,
      tracking_allocator_ctl,
  };
}

static StatusOr<StreamPtr> CreateStream(iree_io_stream_mode_t mode,
                                        size_t block_size = 1 * 1024) {
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      mode, block_size, iree_allocator_system(), &stream);
  if (!iree_status_is_ok(status)) return status;
  return StreamPtr(stream, iree_io_stream_release);
}

template <typename T, size_t N>
static StatusOr<StreamPtr> CreateStreamWithContents(iree_io_stream_mode_t mode,
                                                    T (&elements)[N],
                                                    size_t block_size = 1024) {
  iree_io_stream_t* stream = NULL;
  iree_status_t status =
      iree_io_vec_stream_create(mode | IREE_IO_STREAM_MODE_WRITABLE, block_size,
                                iree_allocator_system(), &stream);
  if (!iree_status_is_ok(status)) return status;
  StreamPtr stream_owner(stream, iree_io_stream_release);
  status = iree_io_stream_write(stream, sizeof(T) * N, elements);
  if (!iree_status_is_ok(status)) return status;
  status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  if (!iree_status_is_ok(status)) return status;
  return stream_owner;
}

TEST(VecStreamTest, Empty) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE));
  EXPECT_EQ(iree_io_stream_mode(stream.get()), IREE_IO_STREAM_MODE_READABLE);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekSet) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // No-op seek to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to end-of-stream.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET,
                                     iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte).
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Try seeking out of bounds (off the front of the list).
  EXPECT_THAT(
      Status(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, -1)),
      StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 6));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 6);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 6);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekFromCurrent) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to end-of-stream by jumping the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(),
                                     IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(),
                                     IREE_IO_STREAM_SEEK_FROM_CURRENT,
                                     -iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek forward to absolute position 1.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // No-op seek to current location (absolute 1).
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte) - here (5-1) - 1 = 3.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 3));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek forward 1 to absolute end-of-stream.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(
                  stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, -100)),
              StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_CURRENT, 600));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 600);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 600);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekFromEnd) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Jump to end-of-stream.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin by seeking back the full length.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END,
                                     -iree_io_stream_length(stream.get())));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Seek to absolute offset length-1 (last valid byte) - here 5 - 1 = 4.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Reset back to origin.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Try seeking out of bounds.
  EXPECT_THAT(Status(iree_io_stream_seek(stream.get(),
                                         IREE_IO_STREAM_SEEK_FROM_END, -100)),
              StatusIs(StatusCode::kOutOfRange));

  // Seek off the end of the stream to extend it.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 100));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 105);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 105);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, SeekToAlignment) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  // Alignment must be a power of two.
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), 63)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_seek_to_alignment(stream.get(), -2)),
              StatusIs(StatusCode::kInvalidArgument));

  // Alignment at 0 should always be ok.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 0));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Seek forward to an unaligned absolute offset 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);

  // Seek forward to alignment 2, which should be absolute offset 2.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 2);

  // Alignment that matches the current offset (2) should be a no-op.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 2);

  // Align up from an aligned value.
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 4);

  // Align off the end of the stream to extend.
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_EXPECT_OK(iree_io_stream_seek_to_alignment(stream.get(), 16));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 16);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
}

TEST(VecStreamTest, ReadUpTo) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  uint8_t read_buffer[64] = {0xDD};
  iree_host_size_t read_length = 0;

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 0, read_buffer, &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 1, read_buffer, &read_length));
  EXPECT_EQ(read_length, 1);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(
      iree_io_stream_read(stream.get(), 2, read_buffer, &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                     read_buffer, &read_length));
  EXPECT_EQ(read_length, 2);
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream should be a no-op.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                     read_buffer, &read_length));
  EXPECT_EQ(read_length, 0);
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 0xDD);
}

TEST(VecStreamTest, ReadExact) {
  uint8_t data[5] = {0, 1, 2, 3, 4};
  IREE_ASSERT_OK_AND_ASSIGN(
      auto stream,
      CreateStreamWithContents(IREE_IO_STREAM_MODE_READABLE, data));

  // Streams start at origin 0.
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), sizeof(data));
  EXPECT_FALSE(iree_io_stream_is_eos(stream.get()));

  uint8_t read_buffer[64] = {0xDD};

  // Reads of zero length should no-op.
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 0, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);

  // Reads should advance the stream offset.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 1, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(read_buffer[0], 0);
  EXPECT_EQ(read_buffer[1], 0xDD);

  // Read another chunk of 2 bytes.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(read_buffer[0], 1);
  EXPECT_EQ(read_buffer[1], 2);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Read up to the end of the stream (2 bytes remaining) by reading over.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  IREE_EXPECT_OK(iree_io_stream_read(stream.get(), 2, read_buffer, NULL));
  EXPECT_EQ(iree_io_stream_offset(stream.get()),
            iree_io_stream_length(stream.get()));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  EXPECT_EQ(read_buffer[0], 3);
  EXPECT_EQ(read_buffer[1], 4);
  EXPECT_EQ(read_buffer[2], 0xDD);

  // Reading from the end of the stream fails with no read length arg.
  memset(read_buffer, 0xDD, sizeof(read_buffer));
  EXPECT_THAT(Status(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));

  // Reset back to the origin and try reading off the end.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  EXPECT_THAT(Status(iree_io_stream_read(stream.get(), sizeof(read_buffer),
                                         read_buffer, NULL)),
              StatusIs(StatusCode::kOutOfRange));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
}

TEST(VecStreamTest, Write) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t data[5] = {0xDD};
  const uint8_t write_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Writes of zero length should be a no-op.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 0, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  EXPECT_EQ(data[0], 0xDD);

  // Writes should advance the stream.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 1, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 1);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), 1, data, NULL));
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 0xDD);

  // Write 2 more bytes and ensure only those are mutated.
  memset(data, 0xDD, sizeof(data));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 2, &write_buffer[1]));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 2);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 1 + 2);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), 3, data, NULL));
  EXPECT_EQ(data[0], 0);
  EXPECT_EQ(data[1], 1);
  EXPECT_EQ(data[2], 2);
  EXPECT_EQ(data[3], 0xDD);

  // Seek to the end of the stream and try to write 0 bytes (should be a no-op).
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, 0));
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));
  IREE_EXPECT_OK(iree_io_stream_write(stream.get(), 0, write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 3);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 3);
  EXPECT_TRUE(iree_io_stream_is_eos(stream.get()));

  // Overwrite the entire contents of the storage.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_EXPECT_OK(
      iree_io_stream_write(stream.get(), sizeof(data), write_buffer));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 5);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 5);
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(write_buffer[0], write_buffer[1], write_buffer[2],
                          write_buffer[3], write_buffer[4]));
}

TEST(VecStreamTest, MoveEmptyContentsLeavesReusableStream) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  iree_byte_sequence_t* sequence = NULL;
  IREE_ASSERT_OK(iree_io_vec_stream_move_contents(stream.get(), &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);

  ASSERT_NE(sequence, nullptr);
  EXPECT_EQ(iree_byte_sequence_length(sequence), 0u);
  iree_const_byte_span_t empty_span = iree_make_const_byte_span("x", 1);
  EXPECT_TRUE(
      iree_byte_sequence_try_get_contiguous_span(sequence, &empty_span));
  EXPECT_EQ(empty_span.data, nullptr);
  EXPECT_EQ(empty_span.data_length, 0u);
  iree_host_size_t segment_count = 0;
  iree_byte_sequence_segment_callback_t callback = {
      count_segment,
      &segment_count,
  };
  IREE_EXPECT_OK(iree_byte_sequence_enumerate(sequence, callback));
  EXPECT_EQ(segment_count, 0u);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);

  const uint8_t new_contents[] = {4, 5, 6};
  IREE_ASSERT_OK(
      iree_io_stream_write(stream.get(), sizeof(new_contents), new_contents));
  iree_byte_sequence_t* second_sequence = NULL;
  IREE_ASSERT_OK(
      iree_io_vec_stream_move_contents(stream.get(), &second_sequence));
  ByteSequencePtr second_sequence_owner(second_sequence,
                                        iree_byte_sequence_release);

  iree_const_byte_span_t contiguous_span = iree_const_byte_span_empty();
  EXPECT_TRUE(iree_byte_sequence_try_get_contiguous_span(second_sequence,
                                                         &contiguous_span));
  EXPECT_THAT(
      std::vector<uint8_t>(contiguous_span.data,
                           contiguous_span.data + contiguous_span.data_length),
      ElementsAre(4, 5, 6));

  iree_byte_span_t clone = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(second_sequence,
                                          iree_allocator_system(), &clone));
  EXPECT_THAT(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
              ElementsAre(4, 5, 6));
  iree_allocator_free(iree_allocator_system(), clone.data);
}

TEST(VecStreamTest, MoveContentsTransfersBlocksWithoutCopying) {
  tracking_allocator_t allocator_state = {
      iree_allocator_system(),
      false,
      0,
      0,
  };
  iree_allocator_t allocator = make_tracking_allocator(&allocator_state);
  iree_io_stream_t* raw_stream = NULL;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE, 1024,
      allocator, &raw_stream));
  StreamPtr stream(raw_stream, iree_io_stream_release);

  std::vector<uint8_t> expected(2503);
  for (iree_host_size_t i = 0; i < expected.size(); ++i) {
    expected[i] = (uint8_t)i;
  }
  IREE_ASSERT_OK(
      iree_io_stream_write(stream.get(), expected.size(), expected.data()));

  std::vector<segment_view_t> initial_segments;
  IREE_ASSERT_OK(iree_io_vec_stream_enumerate_blocks(
      stream.get(), collect_segment_view, &initial_segments));
  ASSERT_GT(initial_segments.size(), 2u);
  ASSERT_GT(initial_segments.front().data_length, 4u);
  EXPECT_LT(initial_segments.back().data_length,
            initial_segments.front().data_length);

  const iree_io_stream_pos_t patch_offset =
      initial_segments.front().data_length - 2;
  const uint8_t patch[] = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4};
  IREE_ASSERT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, patch_offset));
  IREE_ASSERT_OK(iree_io_stream_write(stream.get(), sizeof(patch), patch));
  memcpy(expected.data() + patch_offset, patch, sizeof(patch));

  std::vector<segment_view_t> source_segments;
  IREE_ASSERT_OK(iree_io_vec_stream_enumerate_blocks(
      stream.get(), collect_segment_view, &source_segments));
  const iree_host_size_t allocation_count_before_move =
      allocator_state.allocation_count;

  iree_byte_sequence_t* sequence = NULL;
  IREE_ASSERT_OK(iree_io_vec_stream_move_contents(stream.get(), &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);
  EXPECT_EQ(allocator_state.allocation_count, allocation_count_before_move + 1);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
  EXPECT_EQ(iree_io_stream_length(stream.get()), 0);
  EXPECT_EQ(iree_byte_sequence_length(sequence), expected.size());
  iree_const_byte_span_t contiguous_span = iree_const_byte_span_empty();
  EXPECT_FALSE(
      iree_byte_sequence_try_get_contiguous_span(sequence, &contiguous_span));

  stream.reset();
  segment_comparison_state_t comparison_state = {
      &source_segments,
      0,
      true,
  };
  iree_byte_sequence_segment_callback_t callback = {
      compare_segment_view,
      &comparison_state,
  };
  IREE_ASSERT_OK(iree_byte_sequence_enumerate(sequence, callback));
  EXPECT_TRUE(comparison_state.all_match);
  EXPECT_EQ(comparison_state.index, source_segments.size());

  iree_byte_span_t clone = iree_byte_span_empty();
  IREE_ASSERT_OK(
      iree_byte_sequence_clone(sequence, iree_allocator_system(), &clone));
  EXPECT_EQ(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
            expected);
  iree_allocator_free(iree_allocator_system(), clone.data);

  sequence_owner.reset();
  EXPECT_EQ(allocator_state.free_count, allocator_state.allocation_count);
}

TEST(VecStreamTest, MoveContentsUpdatesRetainedAliases) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));
  const uint8_t original_contents[] = {1, 2, 3};
  IREE_ASSERT_OK(iree_io_stream_write(stream.get(), sizeof(original_contents),
                                      original_contents));
  iree_io_stream_retain(stream.get());
  StreamPtr alias(stream.get(), iree_io_stream_release);

  iree_byte_sequence_t* sequence = NULL;
  IREE_ASSERT_OK(iree_io_vec_stream_move_contents(stream.get(), &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);
  EXPECT_EQ(iree_io_stream_offset(alias.get()), 0);
  EXPECT_EQ(iree_io_stream_length(alias.get()), 0);

  stream.reset();
  const uint8_t alias_contents[] = {8, 9};
  IREE_EXPECT_OK(iree_io_stream_write(alias.get(), sizeof(alias_contents),
                                      alias_contents));
  EXPECT_EQ(iree_io_stream_length(alias.get()), sizeof(alias_contents));

  iree_byte_span_t clone = iree_byte_span_empty();
  IREE_ASSERT_OK(
      iree_byte_sequence_clone(sequence, iree_allocator_system(), &clone));
  EXPECT_THAT(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
              ElementsAre(1, 2, 3));
  iree_allocator_free(iree_allocator_system(), clone.data);
}

TEST(VecStreamTest, MoveContentsFailurePreservesStream) {
  tracking_allocator_t allocator_state = {
      iree_allocator_system(),
      false,
      0,
      0,
  };
  iree_allocator_t allocator = make_tracking_allocator(&allocator_state);
  iree_io_stream_t* raw_stream = NULL;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE, 1024,
      allocator, &raw_stream));
  StreamPtr stream(raw_stream, iree_io_stream_release);

  std::vector<uint8_t> expected(2503);
  for (iree_host_size_t i = 0; i < expected.size(); ++i) {
    expected[i] = (uint8_t)i;
  }
  IREE_ASSERT_OK(
      iree_io_stream_write(stream.get(), expected.size(), expected.data()));
  const iree_io_stream_pos_t original_offset =
      iree_io_stream_offset(stream.get());
  const iree_io_stream_pos_t original_length =
      iree_io_stream_length(stream.get());
  std::vector<segment_view_t> original_segments;
  IREE_ASSERT_OK(iree_io_vec_stream_enumerate_blocks(
      stream.get(), collect_segment_view, &original_segments));
  const iree_host_size_t original_allocation_count =
      allocator_state.allocation_count;
  const iree_host_size_t original_free_count = allocator_state.free_count;
  allocator_state.fail_allocations = true;

  iree_byte_sequence_t sentinel;
  iree_byte_sequence_t* sequence = &sentinel;
  EXPECT_THAT(Status(iree_io_vec_stream_move_contents(stream.get(), &sequence)),
              StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(sequence, nullptr);
  EXPECT_EQ(iree_io_stream_offset(stream.get()), original_offset);
  EXPECT_EQ(iree_io_stream_length(stream.get()), original_length);
  EXPECT_EQ(allocator_state.allocation_count, original_allocation_count);
  EXPECT_EQ(allocator_state.free_count, original_free_count);

  std::vector<segment_view_t> preserved_segments;
  IREE_ASSERT_OK(iree_io_vec_stream_enumerate_blocks(
      stream.get(), collect_segment_view, &preserved_segments));
  ASSERT_EQ(preserved_segments.size(), original_segments.size());
  for (iree_host_size_t i = 0; i < original_segments.size(); ++i) {
    EXPECT_EQ(preserved_segments[i].data, original_segments[i].data);
    EXPECT_EQ(preserved_segments[i].data_length,
              original_segments[i].data_length);
  }

  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  std::vector<uint8_t> actual(expected.size());
  IREE_ASSERT_OK(
      iree_io_stream_read(stream.get(), actual.size(), actual.data(), NULL));
  EXPECT_EQ(actual, expected);
}

TEST(VecStreamTest, FillSizes) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Fill patterns must be 1,2,4,8 bytes.
  EXPECT_THAT(Status(iree_io_stream_fill(stream.get(), 1, pattern, 3)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_io_stream_fill(stream.get(), 1, pattern, 9)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 0);
}

TEST(VecStreamTest, Fill1) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Fill with pattern size 1.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 3, pattern, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 3);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -2));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 1));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 2 + 2);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x80));
}

TEST(VecStreamTest, Fill2) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  // Fill with pattern size 2.
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 3, pattern, 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 3 * 2);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 2));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 4 + 2 * 2);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0x80, 0x90, 0x80, 0x90, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x80, 0x90, 0x80, 0x90));
}

TEST(VecStreamTest, Fill4) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 2, pattern, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 2 * 4);
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -4));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 4));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16 - 4 + 1 * 4);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0xA0, 0xB0, 0x80, 0x90, 0xA0, 0xB0,
                          0x00, 0x00, 0x00, 0x80, 0x90, 0xA0, 0xB0));
}

TEST(VecStreamTest, Fill8Unaligned) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 1));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 8));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 1 + 1 * 8);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
}

TEST(VecStreamTest, Fill8End) {
  IREE_ASSERT_OK_AND_ASSIGN(auto stream,
                            CreateStream(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE));

  uint8_t pattern[] = {0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0};

  // Extend to 16 bytes for easy fill testing.
  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 16));
  IREE_EXPECT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));

  IREE_EXPECT_OK(
      iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_FROM_END, -8));
  IREE_EXPECT_OK(iree_io_stream_fill(stream.get(), 1, pattern, 8));
  EXPECT_EQ(iree_io_stream_offset(stream.get()), 16);

  uint8_t data[16] = {0xDD};
  IREE_ASSERT_OK(iree_io_stream_seek(stream.get(), IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream.get(), sizeof(data), data, NULL));
  EXPECT_THAT(data,
              ElementsAre(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
                          0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0));
}

}  // namespace
