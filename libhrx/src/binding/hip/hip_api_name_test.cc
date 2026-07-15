// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "iree/testing/gtest.h"

namespace {

using hipApiNameFn = const char* (*)(uint32_t id);

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return "libamdhip64.so";
#endif
}

class HipApiNameTest : public testing::Test {
 protected:
  void SetUp() override {
    library_ = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) {
      GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                   << dlerror();
    }
    api_name_ = reinterpret_cast<hipApiNameFn>(dlsym(library_, "hipApiName"));
    ASSERT_NE(api_name_, nullptr) << "hipApiName is not exported";
  }

  void TearDown() override {
    if (library_ != nullptr) {
      dlclose(library_);
    }
  }

  void* library_ = nullptr;
  hipApiNameFn api_name_ = nullptr;
};

TEST_F(HipApiNameTest, ReturnsStableNamesForDefinedIds) {
  EXPECT_STREQ("__hipPopCallConfiguration", api_name_(1));
  EXPECT_STREQ("hipLibraryGetManaged", api_name_(479));
}

TEST_F(HipApiNameTest, ReturnsUnknownForReservedAndOutOfRangeIds) {
  EXPECT_STREQ("unknown", api_name_(0));
  EXPECT_STREQ("unknown", api_name_(50));
  EXPECT_STREQ("unknown", api_name_(std::numeric_limits<uint32_t>::max()));
}

TEST_F(HipApiNameTest, AlwaysReturnsANonNullName) {
  for (uint32_t id = 0; id < 1024; ++id) {
    EXPECT_NE(api_name_(id), nullptr) << "id=" << id;
  }
}

}  // namespace
