// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/index_materializer.h"
#include "loom/link/module_index.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kMarkerParameter = {
    .name = LOOM_BSTRING_REF(6, "marker"),
    .attr_kind = LOOM_ATTR_SYMBOL,
};
static const loom_encoding_family_descriptor_t kLayoutDescriptor = {
    .name = LOOM_BSTRING_REF(11, "test.layout"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
    .parameter_count = 1,
    .parameter_descriptors = &kMarkerParameter,
};
static const loom_encoding_vtable_t kLayoutVtable = {
    .descriptor = &kLayoutDescriptor,
};

class TemplateInputBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kLayoutVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }
  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  loom_module_t* Parse(const std::string& text) {
    const loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_string_view(text.data(), text.size()), IREE_SV("inputs.loom"),
        &context_, &pool_, &options, &module));
    IREE_ASSERT(module);
    return module;
  }
  std::vector<uint8_t> Write(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_RESIZABLE | IREE_IO_STREAM_MODE_SEEKABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream, nullptr, &pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }
  // Shared pool backs source admission and each independently linked caller.
  iree_arena_block_pool_t pool_;
  // Registry owns the custom encoding descriptor through source retirement.
  loom_context_t context_;
};

TEST_F(TemplateInputBindingTest, PreservesSymbolIdentityAcrossLibraryOwners) {
  const std::string family = R"(
template.decl @read(%layout: encoding<layout>, %input: view<16xf32, %layout>) -> (i32)
)";
  const std::string library = family + R"(
func.def public @marker() { func.return }
func.def @private_marker() { func.return }
template.def<@read> priority(20) @global_provider(%layout: encoding<layout>, %input: view<16xf32, #test.layout<marker=@marker>>) -> (i32) {
  %value = scalar.constant 20 : i32
  template.return %value : i32
}
template.def<@read> priority(30) @private_provider(%layout: encoding<layout>, %input: view<16xf32, #test.layout<marker=@private_marker>>) -> (i32) {
  %value = scalar.constant 30 : i32
  template.return %value : i32
}
template.def<@read> priority(1) @fallback(%layout: encoding<layout>, %input: view<16xf32, %layout>) -> (i32) {
  %value = scalar.constant 1 : i32
  template.return %value : i32
}
)";
  const std::string roots[] = {
      family + R"(
func.decl @marker()
func.def public @entry(%buffer: buffer) -> (i32) {
  %layout = encoding.define #test.layout<marker=@marker> : encoding<layout>
  %offset = index.constant 0 : offset
  %input = buffer.view %buffer[%offset] : buffer -> view<16xf32, %layout>
  %value = template.apply<@read>(%layout, %input) : (encoding<layout>, view<16xf32, %layout>) -> (i32)
  func.return %value : i32
}
)",
      family + R"(
func.def @private_marker() { func.return }
func.def public @entry(%buffer: buffer) -> (i32) {
  %layout = encoding.define #test.layout<marker=@private_marker> : encoding<layout>
  %offset = index.constant 0 : offset
  %input = buffer.view %buffer[%offset] : buffer -> view<16xf32, %layout>
  %value = template.apply<@read>(%layout, %input) : (encoding<layout>, view<16xf32, %layout>) -> (i32)
  func.return %value : i32
}
)",
      family + R"(
func.def public @entry(%layout: encoding<layout>, %input: view<16xf32, %layout>) -> (i32) {
  %value = template.apply<@read>(%layout, %input) : (encoding<layout>, view<16xf32, %layout>) -> (i32)
  func.return %value : i32
}
)",
  };
  for (bool bytecode : {false, true}) {
    for (size_t case_index = 0; case_index < IREE_ARRAYSIZE(roots);
         ++case_index) {
      SCOPED_TRACE(::testing::Message() << bytecode << ":" << case_index);
      loom_module_t* source = Parse(library);
      const auto bytes = Write(source);
      loom_link_module_index_t* base = nullptr;
      IREE_ASSERT_OK(loom_link_module_index_allocate(
          &context_, &pool_, iree_allocator_system(), &base));
      const loom_link_module_index_add_options_t library_options = {
          .provider_name = IREE_SV("library"),
          .role = LOOM_LINK_PROVIDER_ROLE_LIBRARY,
      };
      if (bytecode) {
        IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
            base, iree_make_const_byte_span(bytes.data(), bytes.size()),
            IREE_SV("library.loombc"), nullptr, &library_options, nullptr));
      } else {
        IREE_ASSERT_OK(loom_link_module_index_add_materialized(
            base, source, &library_options, nullptr));
      }
      loom_link_module_index_t* index = nullptr;
      IREE_ASSERT_OK(loom_link_module_index_allocate_overlay(
          base, &pool_, iree_allocator_system(), &index));
      loom_module_t* root = Parse(roots[case_index]);
      IREE_ASSERT_OK(loom_link_module_index_add_materialized(index, root,
                                                             nullptr, nullptr));
      const iree_string_view_t entry = IREE_SV("entry");
      const loom_link_plan_options_t options = {
          .mode = LOOM_LINK_PLAN_LINK,
          .root_symbols = {.count = 1, .values = &entry},
      };
      const loom_link_plan_materialization_environment_t environment = {
          .context = &context_,
          .block_pool = &pool_,
          .allocator = iree_allocator_system(),
      };
      loom_link_index_materialization_t linked = {};
      IREE_ASSERT_OK(loom_link_index_materialize(index, &options, &environment,
                                                 IREE_SV("linked"), &linked));
      const auto* library_module = loom_link_module_index_module_at(index, 0);
      for (const char* name :
           {"global_provider", "private_provider", "fallback"}) {
        const auto* provider = loom_link_module_index_lookup_private(
            index, library_module, iree_make_cstring_view(name));
        ASSERT_NE(provider, nullptr);
        // An unknown caller preserves the alternatives for later binding.
        const bool selected =
            case_index == 2 ||
            (case_index == 0 ? std::string(name) == "global_provider"
                             : std::string(name) == "fallback");
        EXPECT_EQ(
            loom_link_plan_contains_symbol(linked.plan, provider->ordinal),
            selected);
      }
      loom_link_module_index_free(index);
      loom_link_module_index_free(base);
      loom_module_free(root);
      loom_module_free(source);
      const loom_verify_options_t verify_options = {
          .sink = {loom_diagnostic_stderr_sink, nullptr},
      };
      loom_verify_result_t verification = {};
      IREE_ASSERT_OK(loom_verify_module(linked.product.module, &verify_options,
                                        &verification));
      EXPECT_EQ(verification.error_count, 0u);
      loom_link_index_materialization_deinitialize(&linked);
    }
  }
}

}  // namespace
}  // namespace loom
