// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/encoding.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(EncodingTest, EqualIgnoresAliasAndComparesArrayContents) {
  int64_t block_a[] = {16, 32, 64};
  int64_t block_b[] = {16, 32, 64};
  loom_named_attr_t attrs_a[] = {{
      /*.name_id=*/7,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64_array(block_a, IREE_ARRAYSIZE(block_a)),
  }};
  loom_named_attr_t attrs_b[] = {{
      /*.name_id=*/7,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64_array(block_b, IREE_ARRAYSIZE(block_b)),
  }};

  loom_encoding_t q8_a = {
      /*.name_id=*/42,
      /*.alias_id=*/100,
      /*.attribute_count=*/1,
      /*.reserved=*/{},       /*.attributes=*/attrs_a,
  };
  loom_encoding_t q8_b = {
      /*.name_id=*/42,
      /*.alias_id=*/200,
      /*.attribute_count=*/1,
      /*.reserved=*/{},       /*.attributes=*/attrs_b,
  };

  EXPECT_TRUE(loom_encoding_equal(&q8_a, &q8_b));
  EXPECT_EQ(loom_encoding_hash(&q8_a), loom_encoding_hash(&q8_b));
}

TEST(EncodingTest, NotEqualForDifferentParams) {
  loom_named_attr_t attrs_a[] = {{
      /*.name_id=*/7,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64(32),
  }};
  loom_named_attr_t attrs_b[] = {{
      /*.name_id=*/7,
      /*.reserved=*/{},
      /*.value=*/loom_attr_i64(64),
  }};

  loom_encoding_t q8_a = {
      /*.name_id=*/42,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/attrs_a,
  };
  loom_encoding_t q8_b = {
      /*.name_id=*/42,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/1,
      /*.reserved=*/{},
      /*.attributes=*/attrs_b,
  };

  EXPECT_FALSE(loom_encoding_equal(&q8_a, &q8_b));
}

TEST(EncodingTest, AttrsEmptySliceForNoParams) {
  loom_encoding_t encoding = {
      /*.name_id=*/42,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/0,
      /*.reserved=*/{},
      /*.attributes=*/NULL,
  };

  loom_named_attr_slice_t attrs = loom_encoding_attrs(&encoding);
  EXPECT_EQ(attrs.entries, nullptr);
  EXPECT_EQ(attrs.count, 0u);
}

}  // namespace
}  // namespace loom
