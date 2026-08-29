// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/io/file.h"

#include <cstring>
#include <memory>

#include "iree/io/file_contents.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, decltype(&iree_io_stream_release)>;
using ByteSequencePtr = std::unique_ptr<iree_byte_sequence_t,
                                        decltype(&iree_byte_sequence_release)>;

static iree_status_t CountSegment(void* user_data,
                                  iree_const_byte_span_t segment) {
  (void)segment;
  ++*static_cast<iree_host_size_t*>(user_data);
  return iree_ok_status();
}

TEST(FileTest, WritesSegmentedByteSequenceInLogicalOrder) {
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(IREE_IO_STREAM_MODE_WRITABLE, 64,
                                           iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream, iree_io_stream_release);

  uint8_t expected[4097];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    expected[i] = static_cast<uint8_t>(i);
  }
  IREE_ASSERT_OK(iree_io_stream_write(stream, sizeof(expected), expected));

  iree_byte_sequence_t* sequence = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_move_contents(stream, &sequence));
  ByteSequencePtr sequence_owner(sequence, iree_byte_sequence_release);
  iree_host_size_t segment_count = 0;
  IREE_ASSERT_OK(iree_byte_sequence_enumerate(
      sequence, (iree_byte_sequence_segment_callback_t){
                    /*.fn=*/CountSegment,
                    /*.user_data=*/&segment_count,
                }));
  ASSERT_GT(segment_count, 1u);

  iree::testing::TempFilePath output_path("loom_segmented_artifact", ".bin");
  IREE_ASSERT_OK(loom_tooling_write_output_byte_sequence(
      output_path.path_view(), sequence, iree_allocator_system()));

  iree_io_file_contents_t* actual = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(output_path.path_view(),
                                            iree_allocator_system(), &actual));
  ASSERT_EQ(actual->const_buffer.data_length, sizeof(expected));
  EXPECT_EQ(std::memcmp(actual->const_buffer.data, expected, sizeof(expected)),
            0);
  iree_io_file_contents_free(actual);
}

}  // namespace
}  // namespace loom
