// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/package.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/program/sources.h"

namespace {

TEST(PackageTest, ImmutableBytesAndSourcesOutlivePackage) {
  std::string binary("LBC\0\r\n\xFF", 7);
  const auto expected = binary;
  const loom_runtime_package_file_t files[] = {
      {IREE_SV("model/provider.loombc"),
       iree_make_const_byte_span(binary.data(), binary.size())},
      {IREE_SV("shared/tokenizer.json"), iree_make_const_byte_span("{}", 2)},
  };
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_create(
      IREE_SV("model/model.json"), IREE_ARRAYSIZE(files), files,
      iree_allocator_system(), &package));
  binary.assign(binary.size(), '?');
  loomc_source_t* source = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_open(
      package, IREE_SV("./provider.loombc"), &source));
  loomc_source_t* tokenizer = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_open(
      package, IREE_SV("../shared/tokenizer.json"), &tokenizer));
  loomc_source_release(tokenizer);
  loom_runtime_package_retain(package);
  loom_runtime_package_release(package);
  loom_runtime_package_release(package);
  const auto bytes = loomc_source_contents(source);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes.data), bytes.data_length),
      expected);
  const auto name = loomc_source_identifier(source);
  EXPECT_EQ(std::string(name.data, name.size), "model/provider.loombc");
  loomc_source_release(source);
}

TEST(PackageTest, RejectsAmbiguousNamesAndDuplicateMembers) {
  for (const auto* name : {"", "/root.loom", "C:/root.loom", "a\\b.loom",
                           "../a.loom", "a/../b.loom", "./a.loom", "a//b.loom",
                           "a/", "a/./b.loom", "a\nb.loom"}) {
    SCOPED_TRACE(name);
    const loom_runtime_package_file_t file = {iree_make_cstring_view(name),
                                              iree_make_const_byte_span("", 0)};
    loom_runtime_package_t* package = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_runtime_package_create(IREE_SV("model.json"), 1, &file,
                                    iree_allocator_system(), &package));
    EXPECT_EQ(package, nullptr);
  }
  const loom_runtime_package_file_t duplicate[] = {
      {IREE_SV("a.loom"), iree_make_const_byte_span("a", 1)},
      {IREE_SV("a.loom"), iree_make_const_byte_span("b", 1)},
  };
  loom_runtime_package_t* package = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_package_create(IREE_SV("model.json"), 2, duplicate,
                                  iree_allocator_system(), &package));
  EXPECT_EQ(package, nullptr);
}

TEST(PackageTest, LookupNeverEscapesOrFallsBackToLocalFiles) {
  loom_runtime_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_runtime_package_create(IREE_SV("model/variant.json"), 0,
                                             nullptr, iree_allocator_system(),
                                             &package));
  for (const auto* locator : {"../../escape.loom", "a/../../../escape.loom",
                              "/absolute.loom", "C:drive.loom"}) {
    loomc_source_t* source = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_runtime_package_open(package, iree_make_cstring_view(locator),
                                  &source));
    EXPECT_EQ(source, nullptr);
  }
  const loom_runtime_sources_t sources = {
      .base_directory = IREE_SV("/ignored"),
      .package = package,
  };
  loomc_source_t* source = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_runtime_sources_open(&sources, IREE_SV("missing.loom"),
                                iree_allocator_system(), &source));
  loom_runtime_package_release(package);
}

}  // namespace
