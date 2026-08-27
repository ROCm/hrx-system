// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/dependency_analysis.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/testdata/dependency_analysis_testdata.h"
#include "loom/ops/op_registry.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

enum ProviderOrdinal {
  kInputProvider = 0,
  kLocalInputProvider,
  kDirectProvider,
  kDuplicateProvider,
  kTransitiveProvider,
  kUnusedProvider,
  kProviderCount,
};

enum class DirectRepresentation {
  kText,
  kBytecode,
};

using ProviderOrdinals = std::array<iree_host_size_t, kProviderCount>;
using RequirementSnapshot =
    std::tuple<std::string, loom_link_dependency_requirement_kind_t,
               loom_link_dependency_ownership_t,
               loom_link_dependency_resolution_t, bool, iree_host_size_t,
               iree_host_size_t>;

class LinkDependencyAnalysisTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    bytecode_buffers_.reserve(4);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_string_view_t FindSource(std::string_view name) {
    const iree_file_toc_t* files =
        loom_link_dependency_analysis_testdata_create();
    for (iree_host_size_t i = 0;
         i < loom_link_dependency_analysis_testdata_size(); ++i) {
      if (name == files[i].name) {
        return iree_make_string_view(files[i].data, files[i].size);
      }
    }
    return iree_string_view_empty();
  }

  std::vector<uint8_t> WriteSource(iree_string_view_t source,
                                   iree_string_view_t filename) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    IREE_CHECK_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                  &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream,
                                             /*options=*/nullptr,
                                             &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(),
                                      /*out_buffer_length=*/nullptr));
    iree_io_stream_release(stream);
    loom_module_free(module);
    return bytes;
  }

  void AddText(loom_link_module_index_t* index, std::string_view filename,
               std::string_view provider_name, loom_link_provider_role_t role,
               iree_host_size_t* out_provider_ordinal) {
    const iree_string_view_t source = FindSource(filename);
    ASSERT_FALSE(iree_string_view_is_empty(source));
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/
        iree_make_string_view(provider_name.data(), provider_name.size()),
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_text(
        index, source, iree_make_string_view(filename.data(), filename.size()),
        /*parse_options=*/nullptr, &options, out_provider_ordinal));
  }

  void AddBytecode(loom_link_module_index_t* index, std::string_view filename,
                   std::string_view provider_name,
                   loom_link_provider_role_t role,
                   iree_host_size_t* out_provider_ordinal) {
    const iree_string_view_t source = FindSource(filename);
    ASSERT_FALSE(iree_string_view_is_empty(source));
    bytecode_buffers_.push_back(WriteSource(
        source, iree_make_string_view(filename.data(), filename.size())));
    const std::vector<uint8_t>& bytes = bytecode_buffers_.back();
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/
        iree_make_string_view(provider_name.data(), provider_name.size()),
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytes.data(), bytes.size()),
        iree_make_string_view(filename.data(), filename.size()),
        /*index_options=*/nullptr, &options, out_provider_ordinal));
  }

  IndexPtr CreateIndex(DirectRepresentation direct_representation,
                       ProviderOrdinals* out_provider_ordinals) {
    loom_link_module_index_t* raw_index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &raw_index));
    IndexPtr index(raw_index);
    AddText(index.get(), "input.loom", "input", LOOM_LINK_PROVIDER_ROLE_INPUT,
            &(*out_provider_ordinals)[kInputProvider]);
    AddText(index.get(), "input_local.loom", "input_local",
            LOOM_LINK_PROVIDER_ROLE_INPUT,
            &(*out_provider_ordinals)[kLocalInputProvider]);
    if (direct_representation == DirectRepresentation::kBytecode) {
      AddBytecode(index.get(), "direct.loom", "direct",
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY,
                  &(*out_provider_ordinals)[kDirectProvider]);
    } else {
      AddText(index.get(), "direct.loom", "direct",
              LOOM_LINK_PROVIDER_ROLE_LIBRARY,
              &(*out_provider_ordinals)[kDirectProvider]);
    }
    AddBytecode(index.get(), "direct_duplicate.loom", "duplicate",
                LOOM_LINK_PROVIDER_ROLE_LIBRARY,
                &(*out_provider_ordinals)[kDuplicateProvider]);
    AddText(index.get(), "transitive.loom", "transitive",
            LOOM_LINK_PROVIDER_ROLE_LIBRARY,
            &(*out_provider_ordinals)[kTransitiveProvider]);
    AddText(index.get(), "unused.loom", "unused",
            LOOM_LINK_PROVIDER_ROLE_LIBRARY,
            &(*out_provider_ordinals)[kUnusedProvider]);
    return index;
  }

  loom_link_dependency_analysis_t Analyze(
      const loom_link_module_index_t* index,
      const std::vector<iree_host_size_t>& direct_providers) {
    const loom_link_dependency_analysis_options_t options = {
        /*.direct_provider_ordinals=*/direct_providers.data(),
        /*.direct_provider_count=*/direct_providers.size(),
    };
    loom_link_dependency_analysis_t analysis = {};
    IREE_CHECK_OK(loom_link_dependency_analyze(
        index, &options, &block_pool_, &analysis_arena_,
        iree_allocator_system(), &analysis));
    return analysis;
  }

  const loom_link_dependency_requirement_t* FindExactRequirement(
      const loom_link_dependency_analysis_t& analysis, std::string_view name) {
    for (iree_host_size_t i = 0; i < analysis.requirements.count; ++i) {
      const loom_link_dependency_requirement_t* requirement =
          &analysis.requirements.values[i];
      if (requirement->kind != LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
        continue;
      }
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(analysis.index,
                                           requirement->target.symbol_ordinal);
      if (std::string_view(symbol->name.data, symbol->name.size) == name) {
        return requirement;
      }
    }
    return nullptr;
  }

  const loom_link_dependency_requirement_t* FindTemplateRequirement(
      const loom_link_dependency_analysis_t& analysis, std::string_view name) {
    for (iree_host_size_t i = 0; i < analysis.requirements.count; ++i) {
      const loom_link_dependency_requirement_t* requirement =
          &analysis.requirements.values[i];
      if (requirement->kind !=
          LOOM_LINK_DEPENDENCY_REQUIREMENT_TEMPLATE_FAMILY) {
        continue;
      }
      const loom_link_module_index_template_family_t* family =
          loom_link_module_index_template_family_at(
              analysis.index, requirement->target.template_family_ordinal);
      if (std::string_view(family->name.data, family->name.size) == name) {
        return requirement;
      }
    }
    return nullptr;
  }

  std::vector<RequirementSnapshot> Snapshot(
      const loom_link_dependency_analysis_t& analysis) {
    std::vector<RequirementSnapshot> snapshot;
    snapshot.reserve(analysis.requirements.count);
    for (iree_host_size_t i = 0; i < analysis.requirements.count; ++i) {
      const loom_link_dependency_requirement_t& requirement =
          analysis.requirements.values[i];
      iree_string_view_t name = iree_string_view_empty();
      if (requirement.kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
        name = loom_link_module_index_symbol_at(
                   analysis.index, requirement.target.symbol_ordinal)
                   ->name;
      } else {
        name = loom_link_module_index_template_family_at(
                   analysis.index, requirement.target.template_family_ordinal)
                   ->name;
      }
      snapshot.emplace_back(std::string(name.data, name.size), requirement.kind,
                            requirement.ownership, requirement.resolution,
                            requirement.exported, requirement.occurrence_count,
                            requirement.candidates.count);
    }
    return snapshot;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  iree_arena_allocator_t analysis_arena_;
  std::vector<std::vector<uint8_t>> bytecode_buffers_;
};

TEST_F(LinkDependencyAnalysisTest,
       ClassifiesExactAndTemplateRequirementsByDirectOwnership) {
  ProviderOrdinals provider_ordinals = {};
  IndexPtr index =
      CreateIndex(DirectRepresentation::kBytecode, &provider_ordinals);
  const std::vector<iree_host_size_t> direct_providers = {
      provider_ordinals[kDirectProvider], provider_ordinals[kUnusedProvider]};
  const loom_link_dependency_analysis_t analysis =
      Analyze(index.get(), direct_providers);

  EXPECT_EQ(analysis.exact_occurrence_count, 7u);
  EXPECT_EQ(analysis.template_demand_occurrence_count, 3u);
  ASSERT_EQ(analysis.requirements.count, 11u);

  const loom_link_dependency_requirement_t* direct =
      FindExactRequirement(analysis, "direct");
  ASSERT_NE(direct, nullptr);
  EXPECT_EQ(direct->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT);
  EXPECT_EQ(direct->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS);
  EXPECT_FALSE(direct->exported);
  EXPECT_EQ(direct->occurrence_count, 1u);
  EXPECT_EQ(direct->candidates.count, 2u);

  const loom_link_dependency_requirement_t* transitive =
      FindExactRequirement(analysis, "transitive");
  ASSERT_NE(transitive, nullptr);
  EXPECT_EQ(transitive->ownership,
            LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT);
  EXPECT_EQ(transitive->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE);

  const loom_link_dependency_requirement_t* reexported =
      FindExactRequirement(analysis, "reexported");
  ASSERT_NE(reexported, nullptr);
  EXPECT_EQ(reexported->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT);
  EXPECT_EQ(reexported->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE);
  EXPECT_TRUE(reexported->exported);
  EXPECT_EQ(reexported->occurrence_count, 1u);

  const loom_link_dependency_requirement_t* incompatible =
      FindExactRequirement(analysis, "bad");
  ASSERT_NE(incompatible, nullptr);
  EXPECT_EQ(incompatible->ownership,
            LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE);
  EXPECT_EQ(incompatible->resolution,
            LOOM_LINK_DEPENDENCY_RESOLUTION_INCOMPATIBLE);
  ASSERT_EQ(incompatible->candidates.count, 1u);
  const loom_link_dependency_candidate_t& incompatible_candidate =
      analysis.candidates.values[incompatible->candidates.first];
  EXPECT_FALSE(incompatible_candidate.compatible);
  EXPECT_EQ(incompatible_candidate.mismatch.kind,
            LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE);

  const loom_link_dependency_requirement_t* missing =
      FindExactRequirement(analysis, "missing");
  ASSERT_NE(missing, nullptr);
  EXPECT_EQ(missing->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED);
  EXPECT_EQ(missing->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_UNRESOLVED);

  const loom_link_dependency_requirement_t* private_only =
      FindExactRequirement(analysis, "private_only");
  ASSERT_NE(private_only, nullptr);
  EXPECT_EQ(private_only->ownership,
            LOOM_LINK_DEPENDENCY_OWNERSHIP_INACCESSIBLE);
  EXPECT_EQ(private_only->resolution,
            LOOM_LINK_DEPENDENCY_RESOLUTION_UNRESOLVED);
  ASSERT_EQ(private_only->candidates.count, 1u);
  EXPECT_TRUE(
      analysis.candidates.values[private_only->candidates.first].compatible);

  const loom_link_dependency_requirement_t* local =
      FindExactRequirement(analysis, "local");
  ASSERT_NE(local, nullptr);
  EXPECT_EQ(local->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL);
  EXPECT_EQ(local->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_LOCAL);

  const loom_link_dependency_requirement_t* export_only =
      FindExactRequirement(analysis, "export_only");
  ASSERT_NE(export_only, nullptr);
  EXPECT_TRUE(export_only->exported);
  EXPECT_EQ(export_only->occurrence_count, 0u);
  EXPECT_EQ(export_only->ownership,
            LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT);
  EXPECT_EQ(export_only->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE);

  const loom_link_dependency_requirement_t* direct_template =
      FindTemplateRequirement(analysis, "family.direct");
  ASSERT_NE(direct_template, nullptr);
  EXPECT_EQ(direct_template->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT);
  EXPECT_EQ(direct_template->resolution,
            LOOM_LINK_DEPENDENCY_RESOLUTION_NOT_APPLICABLE);
  EXPECT_EQ(direct_template->candidates.count, 2u);

  const loom_link_dependency_requirement_t* transitive_template =
      FindTemplateRequirement(analysis, "family.transitive");
  ASSERT_NE(transitive_template, nullptr);
  EXPECT_EQ(transitive_template->ownership,
            LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT);
  EXPECT_EQ(transitive_template->resolution,
            LOOM_LINK_DEPENDENCY_RESOLUTION_NOT_APPLICABLE);

  const loom_link_dependency_requirement_t* open_template =
      FindTemplateRequirement(analysis, "family.open");
  ASSERT_NE(open_template, nullptr);
  EXPECT_EQ(open_template->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN);
  EXPECT_EQ(open_template->resolution,
            LOOM_LINK_DEPENDENCY_RESOLUTION_NOT_APPLICABLE);
  EXPECT_EQ(open_template->candidates.count, 0u);

  ASSERT_EQ(analysis.used_direct_providers.count, 1u);
  EXPECT_EQ(analysis.used_direct_providers.values[0],
            provider_ordinals[kDirectProvider]);
  ASSERT_EQ(analysis.unused_direct_providers.count, 1u);
  EXPECT_EQ(analysis.unused_direct_providers.values[0],
            provider_ordinals[kUnusedProvider]);
}

TEST_F(LinkDependencyAnalysisTest, ReportsAmbiguousDirectExactDefinitions) {
  ProviderOrdinals provider_ordinals = {};
  IndexPtr index = CreateIndex(DirectRepresentation::kText, &provider_ordinals);
  const std::vector<iree_host_size_t> direct_providers = {
      provider_ordinals[kDirectProvider],
      provider_ordinals[kDuplicateProvider]};
  const loom_link_dependency_analysis_t analysis =
      Analyze(index.get(), direct_providers);

  const loom_link_dependency_requirement_t* direct =
      FindExactRequirement(analysis, "direct");
  ASSERT_NE(direct, nullptr);
  EXPECT_EQ(direct->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT);
  EXPECT_EQ(direct->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS);
  EXPECT_FALSE(loom_link_dependency_requirement_satisfied(direct));
  ASSERT_EQ(direct->candidates.count, 2u);
  EXPECT_TRUE(analysis.candidates.values[direct->candidates.first].compatible);
  EXPECT_TRUE(
      analysis.candidates.values[direct->candidates.first + 1].compatible);
}

TEST_F(LinkDependencyAnalysisTest,
       CompatibleInterfaceDeclarationsDoNotCompeteAsDefinitions) {
  loom_link_module_index_t* raw_index = nullptr;
  IREE_ASSERT_OK(loom_link_module_index_allocate(
      &context_, &block_pool_, iree_allocator_system(), &raw_index));
  IndexPtr index(raw_index);
  iree_host_size_t input_provider = 0;
  iree_host_size_t declaration_a_provider = 0;
  iree_host_size_t declaration_b_provider = 0;
  iree_host_size_t definition_provider = 0;
  AddText(index.get(), "declaration_input.loom", "input",
          LOOM_LINK_PROVIDER_ROLE_INPUT, &input_provider);
  AddText(index.get(), "direct_declaration_a.loom", "declaration_a",
          LOOM_LINK_PROVIDER_ROLE_LIBRARY, &declaration_a_provider);
  AddText(index.get(), "direct_declaration_b.loom", "declaration_b",
          LOOM_LINK_PROVIDER_ROLE_LIBRARY, &declaration_b_provider);
  AddText(index.get(), "transitive_definition.loom", "definition",
          LOOM_LINK_PROVIDER_ROLE_LIBRARY, &definition_provider);

  const loom_link_dependency_analysis_t analysis =
      Analyze(index.get(), {declaration_a_provider, declaration_b_provider});
  const loom_link_dependency_requirement_t* requirement =
      FindExactRequirement(analysis, "interface_only");
  ASSERT_NE(requirement, nullptr);
  EXPECT_EQ(requirement->ownership, LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT);
  EXPECT_EQ(requirement->resolution, LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE);
  EXPECT_TRUE(loom_link_dependency_requirement_satisfied(requirement));
  ASSERT_EQ(requirement->candidates.count, 3u);
  for (iree_host_size_t i = 0; i < requirement->candidates.count; ++i) {
    EXPECT_TRUE(analysis.candidates.values[requirement->candidates.first + i]
                    .compatible);
  }
}

TEST_F(LinkDependencyAnalysisTest, TextAndBytecodeProvidersProduceSameReport) {
  ProviderOrdinals text_provider_ordinals = {};
  IndexPtr text_index =
      CreateIndex(DirectRepresentation::kText, &text_provider_ordinals);
  const loom_link_dependency_analysis_t text_analysis =
      Analyze(text_index.get(), {text_provider_ordinals[kDirectProvider],
                                 text_provider_ordinals[kUnusedProvider]});
  const std::vector<RequirementSnapshot> text_snapshot =
      Snapshot(text_analysis);

  iree_arena_reset(&analysis_arena_);
  ProviderOrdinals bytecode_provider_ordinals = {};
  IndexPtr bytecode_index =
      CreateIndex(DirectRepresentation::kBytecode, &bytecode_provider_ordinals);
  const loom_link_dependency_analysis_t bytecode_analysis = Analyze(
      bytecode_index.get(), {bytecode_provider_ordinals[kDirectProvider],
                             bytecode_provider_ordinals[kUnusedProvider]});
  EXPECT_EQ(Snapshot(bytecode_analysis), text_snapshot);
}

TEST_F(LinkDependencyAnalysisTest, ValidatesDirectProviderSelection) {
  ProviderOrdinals provider_ordinals = {};
  IndexPtr index = CreateIndex(DirectRepresentation::kText, &provider_ordinals);
  loom_link_dependency_analysis_t analysis = {};

  const loom_link_dependency_analysis_options_t missing_ordinals = {
      /*.direct_provider_ordinals=*/nullptr,
      /*.direct_provider_count=*/1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_dependency_analyze(index.get(), &missing_ordinals, &block_pool_,
                                   &analysis_arena_, iree_allocator_system(),
                                   &analysis));

  iree_arena_reset(&analysis_arena_);
  const iree_host_size_t input_provider = provider_ordinals[kInputProvider];
  const loom_link_dependency_analysis_options_t input_as_library = {
      /*.direct_provider_ordinals=*/&input_provider,
      /*.direct_provider_count=*/1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_dependency_analyze(index.get(), &input_as_library, &block_pool_,
                                   &analysis_arena_, iree_allocator_system(),
                                   &analysis));

  iree_arena_reset(&analysis_arena_);
  const std::array<iree_host_size_t, 2> duplicate_provider = {
      provider_ordinals[kDirectProvider],
      provider_ordinals[kDirectProvider],
  };
  const loom_link_dependency_analysis_options_t duplicated = {
      /*.direct_provider_ordinals=*/duplicate_provider.data(),
      /*.direct_provider_count=*/duplicate_provider.size(),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_link_dependency_analyze(index.get(), &duplicated, &block_pool_,
                                   &analysis_arena_, iree_allocator_system(),
                                   &analysis));

  iree_arena_reset(&analysis_arena_);
  const iree_host_size_t out_of_range_provider =
      loom_link_module_index_provider_count(index.get());
  const loom_link_dependency_analysis_options_t out_of_range = {
      /*.direct_provider_ordinals=*/&out_of_range_provider,
      /*.direct_provider_count=*/1,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_link_dependency_analyze(index.get(), &out_of_range, &block_pool_,
                                   &analysis_arena_, iree_allocator_system(),
                                   &analysis));
}

}  // namespace
}  // namespace loom
