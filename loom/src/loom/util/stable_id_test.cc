// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/stable_id.h"

#include "iree/testing/gtest.h"

namespace {

TEST(StableIdTest, ProducesDurableNonZeroIds) {
  uint64_t id = loom_stable_id_from_string(IREE_SV("amdgpu.v_add_u32"));
  EXPECT_NE(id, LOOM_STABLE_ID_NONE);
  EXPECT_EQ(id >> 63, 0u);
  EXPECT_EQ(id, loom_stable_id_from_string(IREE_SV("amdgpu.v_add_u32")));
}

TEST(StableIdTest, DistinguishesDifferentKeys) {
  EXPECT_NE(loom_stable_id_from_string(IREE_SV("amdgpu.v_add_u32")),
            loom_stable_id_from_string(IREE_SV("amdgpu.v_sub_u32")));
}

}  // namespace
