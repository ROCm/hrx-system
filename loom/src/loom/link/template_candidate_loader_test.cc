// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/template_candidate_loader.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/link/index_materializer.h"
#include "loom/link/testdata/link_index/link_index_testdata.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

static std::string Fixture(iree_string_view_t name) {
  const auto* files = loom_link_index_testdata_create();
  for (size_t i = 0; i < loom_link_index_testdata_size(); ++i) {
    if (iree_string_view_equal(iree_make_cstring_view(files[i].name), name)) {
      return std::string(files[i].data, files[i].size);
    }
  }
  ADD_FAILURE() << "missing link fixture";
  return {};
}

class TemplateCandidateLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }
  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  testing::ModulePtr Parse(const std::string& source, iree_string_view_t name) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        name, &context_, &block_pool_, &options, &module));
    return testing::ModulePtr(module);
  }

  std::vector<uint8_t> Write(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    const loom_bytecode_write_options_t options = {};
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, &options, &block_pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  // Shared storage for source modules and invocation arenas.
  iree_arena_block_pool_t block_pool_ = {};
  // Finalized dialect definitions used by every provider form.
  loom_context_t context_ = {};
};

struct SignatureDiagnostic {
  // Signature axis changed in the authored input.
  bool arguments;
  // Whether the diagnostic sink fails with an infrastructure status.
  bool fail;
  // Number of delivered diagnostics in the current invocation.
  uint32_t count = 0;
};

static iree_status_t CaptureSignature(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* capture = static_cast<SignatureDiagnostic*>(user_data);
  ++capture->count;
  EXPECT_EQ(emission->error, LOOM_ERR_SYMBOL_014);
  EXPECT_TRUE(iree_string_view_equal(
      emission->params[1].string,
      capture->arguments ? IREE_SV("arguments") : IREE_SV("results")));
  EXPECT_EQ(emission->params[2].u64, capture->arguments ? 0u : 1u);
  EXPECT_EQ(emission->params[3].u64, capture->arguments ? 1u : 0u);
  EXPECT_NE(emission->origin, nullptr);
  if (emission->origin) {
    EXPECT_TRUE(iree_string_view_equal(emission->origin->filename,
                                       IREE_SV("root.loom")));
    EXPECT_EQ(emission->origin->start_line, 3u);
  }
  EXPECT_EQ(emission->related_op_count, 1u);
  if (emission->related_op_count) {
    const auto* related = emission->related_ops[0].origin;
    EXPECT_NE(related, nullptr);
    if (related) {
      EXPECT_TRUE(
          iree_string_view_equal(related->filename, IREE_SV("provider.loom")));
      EXPECT_EQ(related->start_line, 10u);
    }
  }
  return capture->fail ? iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                          "diagnostic sink exhausted")
                       : iree_ok_status();
}

TEST_F(TemplateCandidateLoaderTest,
       RejectsMismatchedSignaturesWithoutPublishingOrCachingCallbacks) {
  const auto root_source = Fixture(IREE_SV("target_roots.loom"));
  auto root = Parse(root_source, IREE_SV("root.loom"));
  ASSERT_NE(root, nullptr);
  for (bool arguments : {false, true}) {
    SCOPED_TRACE(arguments);
    auto source = Fixture(IREE_SV("target_providers.loom"));
    const std::string original = "@generic() -> (index)";
    const auto position = source.find(original);
    ASSERT_NE(position, std::string::npos);
    source.replace(
        position, original.size(),
        arguments ? "@generic(%unused: index) -> (index)" : "@generic() -> ()");
    if (!arguments) {
      const std::string original_return = "template.return %count : index";
      const auto return_position = source.find(original_return, position);
      ASSERT_NE(return_position, std::string::npos);
      source.replace(return_position, original_return.size(),
                     "template.return");
    }
    auto provider = Parse(source, IREE_SV("provider.loom"));
    ASSERT_NE(provider, nullptr);
    const auto bytes = Write(provider.get());
    for (int form = 0; form < 3; ++form) {
      SCOPED_TRACE(form);
      loom_link_module_index_t* index_pointer = nullptr;
      IREE_ASSERT_OK(loom_link_module_index_allocate(
          &context_, &block_pool_, iree_allocator_system(), &index_pointer));
      std::unique_ptr<loom_link_module_index_t,
                      decltype(&loom_link_module_index_free)>
          index(index_pointer, loom_link_module_index_free);
      IREE_ASSERT_OK(loom_link_module_index_add_materialized(
          index.get(), root.get(), nullptr, nullptr));
      const loom_link_module_index_add_options_t add_options = {
          .role = LOOM_LINK_PROVIDER_ROLE_LIBRARY,
      };
      if (form == 0) {
        IREE_ASSERT_OK(loom_link_module_index_add_materialized(
            index.get(), provider.get(), &add_options, nullptr));
      } else if (form == 1) {
        IREE_ASSERT_OK(loom_link_module_index_add_text(
            index.get(), iree_make_string_view(source.data(), source.size()),
            IREE_SV("provider.loom"), nullptr, &add_options, nullptr));
      } else {
        IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
            index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
            IREE_SV("provider.loombc"), nullptr, &add_options, nullptr));
      }
      const iree_string_view_t roots[] = {IREE_SV("unknown")};
      const loom_link_plan_options_t options = {
          .mode = LOOM_LINK_PLAN_LINK,
          .root_symbols = {.count = 1, .values = roots},
      };
      loom_link_plan_t* plan_pointer = nullptr;
      IREE_ASSERT_OK(loom_link_plan_build(
          index.get(), &options, iree_allocator_system(), &plan_pointer));
      std::unique_ptr<loom_link_plan_t, decltype(&loom_link_plan_free)> plan(
          plan_pointer, loom_link_plan_free);
      ASSERT_NE(plan, nullptr);
      loom_link_plan_materialization_environment_t environment = {
          .context = &context_,
          .block_pool = &block_pool_,
          .allocator = iree_allocator_system(),
      };
      loom_link_template_candidate_loader_t* loader_pointer = nullptr;
      IREE_ASSERT_OK(loom_link_template_candidate_loader_allocate(
          index.get(), &environment, &loader_pointer));
      std::unique_ptr<loom_link_template_candidate_loader_t,
                      decltype(&loom_link_template_candidate_loader_free)>
          loader(loader_pointer, loom_link_template_candidate_loader_free);
      iree_arena_allocator_t arena;
      iree_arena_initialize(&block_pool_, &arena);
      loom_link_plan_materialization_t product = {};
      IREE_ASSERT_OK(loom_link_plan_materialize(
          plan.get(), &environment, IREE_SV("candidate"), &arena, &product));
      ASSERT_NE(product.module, nullptr);
      // Reuse the cache after a missing sink and after a failing sink. Each
      // invocation owns its callback and must reject independently.
      for (int mode : {0, 1, 2, 1}) {
        SCOPED_TRACE(mode);
        SignatureDiagnostic capture = {arguments, mode == 2};
        environment.diagnostics.emitter =
            mode ? iree_diagnostic_emitter_t{CaptureSignature, &capture}
                 : iree_diagnostic_emitter_t{};
        loom_template_provider_slice_t candidates = {};
        bool accepted = true;
        iree_status_t status = loom_link_template_candidate_loader_load(
            loader.get(), plan.get(), &product, environment.diagnostics, &arena,
            &candidates, &accepted);
        if (mode == 2) {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
        } else {
          IREE_EXPECT_OK(status);
        }
        EXPECT_FALSE(accepted);
        EXPECT_EQ(candidates.count, 0u);
        EXPECT_EQ(candidates.providers, nullptr);
        EXPECT_EQ(capture.count, mode ? 1u : 0u);

        capture.count = 0;
        loom_link_index_materialization_t linked = {};
        status = loom_link_index_materialize(
            index.get(), &options, &environment, IREE_SV("linked"), &linked);
        if (mode == 2) {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
        } else {
          IREE_EXPECT_OK(status);
        }
        EXPECT_EQ(linked.product.module, nullptr);
        EXPECT_EQ(linked.plan, nullptr);
        EXPECT_EQ(capture.count, mode ? 1u : 0u);
        loom_link_index_materialization_deinitialize(&linked);
      }
      loom_module_free(product.module);
      iree_arena_deinitialize(&arena);
    }
  }
}

}  // namespace
}  // namespace loom
