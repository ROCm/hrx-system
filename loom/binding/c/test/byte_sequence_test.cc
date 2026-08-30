// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/byte_sequence.h"

#include <array>
#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using testing::ElementsAre;

using ByteSequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;

loomc_status_t AppendSegment(void* user_data, loomc_byte_span_t segment) {
  auto* bytes = static_cast<std::vector<uint8_t>*>(user_data);
  bytes->insert(bytes->end(), segment.data, segment.data + segment.data_length);
  return loomc_ok_status();
}

TEST(ByteSequenceTest, CopiesAndEnumeratesImmutableContents) {
  std::array<uint8_t, 4> source = {1, 2, 3, 4};
  loomc_byte_sequence_t* sequence = nullptr;
  LOOMC_ASSERT_OK(loomc_byte_sequence_create_copy(
      loomc_make_byte_span(source.data(), source.size()),
      loomc_allocator_system(), &sequence));
  ByteSequencePtr sequence_owner(sequence);
  source.fill(0);

  EXPECT_EQ(loomc_byte_sequence_length(sequence), 4u);
  loomc_byte_span_t contiguous_span = loomc_byte_span_empty();
  ASSERT_TRUE(
      loomc_byte_sequence_try_get_contiguous_span(sequence, &contiguous_span));
  EXPECT_THAT(
      std::vector<uint8_t>(contiguous_span.data,
                           contiguous_span.data + contiguous_span.data_length),
      ElementsAre(1, 2, 3, 4));

  std::vector<uint8_t> enumerated;
  loomc_byte_sequence_callback_t callback = {
      /*.fn=*/AppendSegment,
      /*.user_data=*/&enumerated,
  };
  LOOMC_EXPECT_OK(loomc_byte_sequence_enumerate(sequence, callback));
  EXPECT_THAT(enumerated, ElementsAre(1, 2, 3, 4));
}

TEST(ByteSequenceTest, NullHasZeroLength) {
  EXPECT_EQ(loomc_byte_sequence_length(nullptr), 0u);
}

TEST(ByteSequenceTest, ClonesIntoCallerOwnedStorage) {
  const uint8_t source[] = {5, 6, 7};
  loomc_byte_sequence_t* sequence = nullptr;
  LOOMC_ASSERT_OK(loomc_byte_sequence_create_copy(
      loomc_make_byte_span(source, sizeof(source)), loomc_allocator_system(),
      &sequence));
  ByteSequencePtr sequence_owner(sequence);

  loomc_byte_span_t clone = loomc_byte_span_empty();
  LOOMC_ASSERT_OK(
      loomc_byte_sequence_clone(sequence, loomc_allocator_system(), &clone));
  EXPECT_THAT(std::vector<uint8_t>(clone.data, clone.data + clone.data_length),
              ElementsAre(5, 6, 7));
  loomc_allocator_free(loomc_allocator_system(), (void*)clone.data);
}

}  // namespace
