// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/config/config.h"

#include <string>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ConfigMaterializeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("config_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  std::string PrintSchema(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_EXPECT_OK(loom_tooling_config_format_schema_json(module, &stream));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  iree_status_t Materialize(
      loom_module_t* module, const loom_tooling_config_binding_t* bindings,
      iree_host_size_t binding_count,
      loom_tooling_config_materialize_flags_t flags,
      loom_tooling_config_materialize_result_t* out_result) {
    loom_tooling_config_set_t config_set;
    loom_tooling_config_set_initialize(iree_allocator_system(), &config_set);
    loom_tooling_config_materialize_options_t options;
    loom_tooling_config_materialize_options_initialize(&options);
    options.flags = flags;
    options.config_set = &config_set;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
         ++i) {
      status = loom_tooling_config_set_append(&config_set, bindings[i].key,
                                              bindings[i].value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_tooling_config_materialize_module(module, &options,
                                                      &block_pool_, out_result);
    }
    loom_tooling_config_set_deinitialize(&config_set);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ConfigMaterializeTest, ConfigSetOwnsAssignmentsAndRejectsDuplicates) {
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(iree_allocator_system(), &config_set);

  IREE_ASSERT_OK(loom_tooling_config_set_append_assignment(
      &config_set, IREE_SV(" @model36.model.hidden_size = 4096 ")));
  EXPECT_EQ(config_set.binding_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      config_set.bindings[0].key,
      iree_make_cstring_view("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[0].value,
                                     iree_make_cstring_view("4096")));

  iree_status_t status = loom_tooling_config_set_append_assignment(
      &config_set, IREE_SV("model36.model.hidden_size=8192"));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  loom_tooling_config_set_deinitialize(&config_set);
}

TEST_F(ConfigMaterializeTest, ConfigSetAppendsJsonObjectBindings) {
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(iree_allocator_system(), &config_set);

  IREE_ASSERT_OK(
      loom_tooling_config_set_append_json_object(&config_set, IREE_SV(R"({
        // JSONC comments are accepted because config files are edited by hand.
        "model36": {
          "model": {"hidden_size": 4096},
          "features": {"enable_mtp": true}
        },
        "@direct": "4\u0032",
        "looks_object": "{not_object}",
      })")));

  ASSERT_EQ(config_set.binding_count, 4u);
  EXPECT_TRUE(iree_string_view_equal(
      config_set.bindings[0].key,
      iree_make_cstring_view("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[0].value,
                                     iree_make_cstring_view("4096")));
  EXPECT_TRUE(iree_string_view_equal(
      config_set.bindings[1].key,
      iree_make_cstring_view("model36.features.enable_mtp")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[1].value,
                                     iree_make_cstring_view("true")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[2].key,
                                     iree_make_cstring_view("direct")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[2].value,
                                     iree_make_cstring_view("42")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[3].key,
                                     iree_make_cstring_view("looks_object")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[3].value,
                                     iree_make_cstring_view("{not_object}")));

  loom_tooling_config_set_deinitialize(&config_set);
}

TEST_F(ConfigMaterializeTest, ConfigSetAppendsJsonFileBindings) {
  iree::testing::TempFilePath config_file("loom_config_test", ".json");
  std::string json = R"({
    "model36": {
      "model": {"hidden_size": 4096}
    }
  })";
  IREE_ASSERT_OK(iree_io_file_contents_write(
      iree_make_cstring_view(config_file.path().c_str()),
      iree_make_const_byte_span(json.data(), json.size()),
      iree_allocator_system()));

  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(iree_allocator_system(), &config_set);

  IREE_ASSERT_OK(loom_tooling_config_set_append_json_file(
      &config_set, iree_make_cstring_view(config_file.path().c_str()),
      iree_allocator_system()));
  ASSERT_EQ(config_set.binding_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      config_set.bindings[0].key,
      iree_make_cstring_view("model36.model.hidden_size")));
  EXPECT_TRUE(iree_string_view_equal(config_set.bindings[0].value,
                                     iree_make_cstring_view("4096")));

  iree_status_t status = loom_tooling_config_set_append_json_file(
      &config_set, IREE_SV("-"), iree_allocator_system());
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);

  loom_tooling_config_set_deinitialize(&config_set);
}

TEST_F(ConfigMaterializeTest, IgnoresNonSensitiveBindings) {
  ModulePtr module = Parse(R"(
func.def @no_config(%x: i32) -> (i32) {
  func.return %x : i32
}
)");

  loom_tooling_config_binding_t binding = {
      /*.key=*/IREE_SV("model36.model.hidden_size"),
      /*.value=*/IREE_SV("4096"),
  };
  loom_tooling_config_materialize_result_t result = {0};
  IREE_ASSERT_OK(Materialize(module.get(), &binding, 1, 0, &result));
  EXPECT_EQ(result.materialized_count, 0u);
  EXPECT_EQ(result.ignored_count, 1u);

  std::string printed = Print(module.get());
  EXPECT_NE(printed.find("func.def @no_config"), std::string::npos);
  EXPECT_EQ(printed.find("config.def"), std::string::npos);
}

TEST_F(ConfigMaterializeTest, MaterializesConstrainedDecl) {
  ModulePtr module = Parse(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]

func.def @read_config() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}
)");

  loom_tooling_config_binding_t binding = {
      /*.key=*/IREE_SV("model36.model.hidden_size"),
      /*.value=*/IREE_SV("4096"),
  };
  loom_tooling_config_materialize_result_t result = {0};
  IREE_ASSERT_OK(Materialize(module.get(), &binding, 1,
                             LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES,
                             &result));
  EXPECT_EQ(result.materialized_count, 1u);
  EXPECT_EQ(result.ignored_count, 0u);

  std::string printed = Print(module.get());
  EXPECT_NE(
      printed.find("config.def @model36.model.hidden_size = 4096 : index"),
      std::string::npos);
  EXPECT_EQ(printed.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
}

TEST_F(ConfigMaterializeTest, RejectsConstraintViolation) {
  ModulePtr module = Parse(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]
)");

  loom_tooling_config_binding_t binding = {
      /*.key=*/IREE_SV("model36.model.hidden_size"),
      /*.value=*/IREE_SV("4103"),
  };
  iree_status_t status =
      Materialize(module.get(), &binding, 1,
                  LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES, nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST_F(ConfigMaterializeTest, MaterializesEncodingValue) {
  ModulePtr module = Parse(R"(
config.decl @model36.layout : encoding<layout>
)");

  loom_tooling_config_binding_t binding = {
      /*.key=*/IREE_SV("@model36.layout"),
      /*.value=*/IREE_SV(" #dense "),
  };
  loom_tooling_config_materialize_result_t result = {0};
  IREE_ASSERT_OK(Materialize(module.get(), &binding, 1,
                             LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES,
                             &result));
  EXPECT_EQ(result.materialized_count, 1u);

  std::string printed = Print(module.get());
  EXPECT_NE(printed.find("config.def @model36.layout = #dense : "
                         "encoding<layout>"),
            std::string::npos);
}

TEST_F(ConfigMaterializeTest, RejectsWrongEncodingRole) {
  ModulePtr module = Parse(R"(
config.decl @model36.layout : encoding<layout>
)");

  loom_tooling_config_binding_t binding = {
      /*.key=*/IREE_SV("model36.layout"),
      /*.value=*/IREE_SV("#ggml_q4_0<block_elems=32, storage_bytes=18>"),
  };
  iree_status_t status =
      Materialize(module.get(), &binding, 1,
                  LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES, nullptr);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
}

TEST_F(ConfigMaterializeTest, RequireResolvedRejectsRemainingDecls) {
  ModulePtr module = Parse(R"(
config.decl @model36.model.hidden_size : index
)");

  loom_tooling_config_resolution_result_t result = {0};
  iree_status_t status =
      loom_tooling_config_require_resolved_module(module.get(), &result);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(result.unresolved_count, 1u);
  iree_status_free(status);
}

TEST_F(ConfigMaterializeTest, RequireResolvedAcceptsDefs) {
  ModulePtr module = Parse(R"(
config.def @model36.model.hidden_size = 4096 : index
)");

  loom_tooling_config_resolution_result_t result = {0};
  IREE_ASSERT_OK(
      loom_tooling_config_require_resolved_module(module.get(), &result));
  EXPECT_EQ(result.unresolved_count, 0u);
}

TEST_F(ConfigMaterializeTest, FormatsConfigSchemaJson) {
  ModulePtr module = Parse(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 0, 8192), mul(%value, 16)]
config.def @model36.features.enable_mtp = true : i1
)");

  std::string schema = PrintSchema(module.get());
  EXPECT_NE(schema.find("\"count\":2"), std::string::npos);
  EXPECT_NE(schema.find("\"name\":\"model36.model.hidden_size\""),
            std::string::npos);
  EXPECT_NE(schema.find("\"state\":\"decl\""), std::string::npos);
  EXPECT_NE(schema.find("\"required\":true"), std::string::npos);
  EXPECT_NE(schema.find("\"type\":\"index\""), std::string::npos);
  EXPECT_NE(schema.find("\"kind\":\"range\""), std::string::npos);
  EXPECT_NE(schema.find("\"kind\":\"mul\""), std::string::npos);
  EXPECT_NE(schema.find("\"value\":8192"), std::string::npos);
  EXPECT_NE(schema.find("\"name\":\"model36.features.enable_mtp\""),
            std::string::npos);
  EXPECT_NE(schema.find("\"state\":\"def\""), std::string::npos);
  EXPECT_NE(schema.find("\"required\":false"), std::string::npos);
  EXPECT_NE(schema.find("\"default\":\"true\""), std::string::npos);
}

}  // namespace
}  // namespace loom
