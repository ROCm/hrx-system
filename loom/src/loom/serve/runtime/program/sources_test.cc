// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/sources.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SourcesTest, CloneOwnsAllLocatorsInOneAllocation) {
  size_t allocations = 0;
  const iree_allocator_t allocator = {
      &allocations,
      +[](void* self, iree_allocator_command_t command, const void* parameters,
          void** pointer) -> iree_status_t {
        if (command != IREE_ALLOCATOR_COMMAND_FREE) {
          ++*static_cast<size_t*>(self);
        }
        const auto system = iree_allocator_system();
        return system.ctl(system.self, command, parameters, pointer);
      }};
  std::string base = "model/directory";
  std::string prepare = "prepare.loom";
  std::string control = "control.loom";
  std::string first = "../shared/first.loom";
  std::string second = "/models/second.loom";
  std::string partition = "node-1";
  iree_string_view_t providers[] = {iree_make_cstring_view(first.c_str()),
                                    iree_make_cstring_view(second.c_str())};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(base.c_str()),
      .prepare = iree_make_cstring_view(prepare.c_str()),
      .control = iree_make_cstring_view(control.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
      .reference =
          {
              .selector =
                  {
                      .partition = iree_make_cstring_view(partition.c_str()),
                  },
          },
  };
  loom_serve_sources_t* copy = nullptr;
  IREE_ASSERT_OK(loom_serve_sources_clone(&sources, allocator, &copy));
  EXPECT_EQ(allocations, 1u);
  for (auto* text : {&base, &prepare, &control, &first, &second, &partition}) {
    text->assign(text->size(), '?');
  }
  providers[0] = {};
  providers[1] = {};
  const auto string = [](iree_string_view_t value) {
    return std::string(value.data, value.size);
  };
  EXPECT_EQ(string(copy->base_directory), "model/directory");
  EXPECT_EQ(string(copy->prepare), "prepare.loom");
  EXPECT_EQ(string(copy->control), "control.loom");
  ASSERT_EQ(copy->count, 2u);
  EXPECT_EQ(string(copy->values[0]), "../shared/first.loom");
  EXPECT_EQ(string(copy->values[1]), "/models/second.loom");
  EXPECT_EQ(string(copy->reference.selector.partition), "node-1");
  loom_serve_sources_free(copy, allocator);
}

TEST(SourcesTest, ClonedNamespaceRetainsPackageAndRelativeResolution) {
  const auto allocator = iree_allocator_system();
  const loom_serve_package_file_t files[] = {
      {IREE_SV("model/policy.loom"), iree_make_const_byte_span("policy", 6)}};
  loom_serve_package_t* package = nullptr;
  IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model/model.json"), 1,
                                           files, allocator, &package));
  const loom_serve_sources_t sources = {.package = package};
  loom_serve_sources_t* copy = nullptr;
  IREE_ASSERT_OK(loom_serve_sources_clone(&sources, allocator, &copy));
  loom_serve_package_release(package);
  loomc_source_t* source = nullptr;
  IREE_ASSERT_OK(loom_serve_sources_open(copy, IREE_SV("policy.loom"),
                                         allocator, &source));
  loom_serve_sources_free(copy, allocator);
  const auto bytes = loomc_source_contents(source);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(bytes.data), bytes.data_length),
      "policy");
  loomc_source_release(source);
}

TEST(SourcesTest, EmptyAndUnrepresentableNamespaces) {
  const auto allocator = iree_allocator_system();
  loom_serve_sources_t sources = {};
  loom_serve_sources_t* copy = nullptr;
  IREE_ASSERT_OK(loom_serve_sources_clone(&sources, allocator, &copy));
  EXPECT_EQ(copy->count, 0u);
  EXPECT_EQ(copy->values, nullptr);
  EXPECT_EQ(copy->base_directory.size, 0u);
  EXPECT_EQ(copy->prepare.size, 0u);
  EXPECT_EQ(copy->control.size, 0u);
  loom_serve_sources_free(copy, allocator);
  sources.count = IREE_HOST_SIZE_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_sources_clone(&sources, allocator, &copy));
  EXPECT_EQ(copy, nullptr);
  sources.count = 0;
  sources.base_directory.size = IREE_HOST_SIZE_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_sources_clone(&sources, allocator, &copy));
  EXPECT_EQ(copy, nullptr);
  loom_serve_sources_free(copy, allocator);
}

TEST(SourcesTest, RelativePathsRequireABase) {
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV("")));
  EXPECT_FALSE(loom_serve_path_is_absolute(IREE_SV("model/prepare.loombc")));
  char* path = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_path_resolve({}, IREE_SV("prepare.loombc"),
                              iree_allocator_system(), &path));
  EXPECT_EQ(path, nullptr);
  IREE_ASSERT_OK(loom_serve_path_resolve(IREE_SV("models/local"),
                                         IREE_SV("../shared/kernel.loom"),
                                         iree_allocator_system(), &path));
  EXPECT_STREQ(path, "models/local/../shared/kernel.loom");
  iree_allocator_free(iree_allocator_system(), path);
}

TEST(SourcesTest, AbsolutePathsIgnoreBaseAndRemainAuthored) {
  // A share path is absolute in both supported local filesystem namespaces.
  const auto locator = IREE_SV("//files/models/model.loombc");
  EXPECT_TRUE(loom_serve_path_is_absolute(locator));
  for (const auto base : {iree_string_view_empty(), IREE_SV("ignored")}) {
    char* path = nullptr;
    IREE_ASSERT_OK(
        loom_serve_path_resolve(base, locator, iree_allocator_system(), &path));
    EXPECT_STREQ(path, "//files/models/model.loombc");
    EXPECT_NE(path, locator.data);
    iree_allocator_free(iree_allocator_system(), path);
  }
}

}  // namespace
