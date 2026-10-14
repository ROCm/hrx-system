// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/config/input.h"

#include <string>

#include "iree/base/api.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace loom {
namespace {

TEST(ConfigInputTest, AssignmentAdapterOwnsValuesAndRejectsDuplicates) {
  loom_config_text_binding_set_t text_set;
  loom_config_text_binding_set_initialize(iree_allocator_system(), &text_set);

  IREE_ASSERT_OK(loom_tooling_config_text_binding_set_append_assignment(
      &text_set, IREE_SV(" @model36.model.hidden_size = 4096 ")));
  ASSERT_EQ(text_set.binding_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      text_set.bindings[0].key,
      iree_make_cstring_view("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(text_set.bindings[0].value,
                                     iree_make_cstring_view("4096")));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_tooling_config_text_binding_set_append_assignment(
          &text_set, IREE_SV("model36.model.hidden_size=8192")));

  loom_config_text_binding_set_deinitialize(&text_set);
}

TEST(ConfigInputTest, LoadsJsonObjectFiles) {
  iree::testing::TempFilePath config_file("loom_config_test", ".json");
  const std::string json = R"({
    "model36": {
      "model": {"hidden_size": 4096}
    }
  })";
  IREE_ASSERT_OK(iree_io_file_contents_write(
      iree_make_cstring_view(config_file.path().c_str()),
      iree_make_const_byte_span(json.data(), json.size()),
      iree_allocator_system()));

  loom_config_text_binding_set_t text_set;
  loom_config_text_binding_set_initialize(iree_allocator_system(), &text_set);
  IREE_ASSERT_OK(loom_tooling_config_text_binding_set_append_json_file(
      &text_set, iree_make_cstring_view(config_file.path().c_str()),
      iree_allocator_system()));
  ASSERT_EQ(text_set.binding_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      text_set.bindings[0].key,
      iree_make_cstring_view("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(text_set.bindings[0].value,
                                     iree_make_cstring_view("4096")));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tooling_config_text_binding_set_append_json_file(
                            &text_set, IREE_SV("-"), iree_allocator_system()));
  loom_config_text_binding_set_deinitialize(&text_set);
}

}  // namespace
}  // namespace loom
