// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/testing/gtest.h"

namespace iree::hal::cts {
namespace {

TEST(TestdataTest, LinkedCorpusIsAvailableThroughEveryBackendView) {
  for (const char* backend : {"fixture_primary", "fixture_alias"}) {
    SCOPED_TRACE(backend);
    const auto targets = CtsRegistry::ListExecutableTargets(backend);
    ASSERT_EQ(targets.size(), 1u);
    const auto& target = targets.front();
    EXPECT_EQ(target.name, "fixture_target");
    EXPECT_STREQ(target.family, "fixture_family");
    EXPECT_STREQ(target.target_key, "fixture:target");
    for (const char* name : {"first", "second"}) {
      const std::string file_name = std::string(name) + ".txt";
      const auto data = target.data_fn(
          iree_make_string_view(file_name.data(), file_name.size()));
      ASSERT_NE(data.data, nullptr);
      EXPECT_EQ(std::string(reinterpret_cast<const char*>(data.data),
                            data.data_length),
                std::string(name) + "\n");
    }
    const auto missing = target.data_fn(IREE_SV("missing.txt"));
    EXPECT_EQ(missing.data, nullptr);
    EXPECT_EQ(missing.data_length, 0u);
  }
}

}  // namespace
}  // namespace iree::hal::cts
