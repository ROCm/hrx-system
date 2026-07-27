// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/linker.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};
using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

class LinkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables,
                    loom_config_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables,
                    loom_target_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables,
                    loom_test_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  ModulePtr ParseOwned(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_module_t* Link(std::initializer_list<loom_module_t*> source_modules) {
    loom_module_t* linked_module = nullptr;
    IREE_CHECK_OK(LinkStatus(source_modules, &linked_module));
    modules_.push_back(linked_module);
    return linked_module;
  }

  loom_module_t* LinkRoots(
      std::initializer_list<loom_module_t*> source_modules,
      std::initializer_list<iree_string_view_t> root_symbols) {
    loom_module_t* linked_module = nullptr;
    IREE_CHECK_OK(LinkStatus(source_modules, root_symbols, &linked_module));
    modules_.push_back(linked_module);
    return linked_module;
  }

  iree_status_t LinkStatus(std::initializer_list<loom_module_t*> source_modules,
                           loom_module_t** out_module) {
    return LinkStatus(source_modules, {}, out_module);
  }

  iree_status_t LinkStatus(
      std::initializer_list<loom_module_t*> source_modules,
      std::initializer_list<iree_string_view_t> root_symbols,
      loom_module_t** out_module) {
    std::vector<const loom_module_t*> inputs;
    inputs.reserve(source_modules.size());
    for (loom_module_t* module : source_modules) {
      inputs.push_back(module);
    }
    std::vector<iree_string_view_t> roots(root_symbols);
    loom_link_options_t options = {
        /*.module_name=*/IREE_SV("linked"),
        /*.root_symbols=*/{/*.count=*/roots.size(), /*.values=*/roots.data()},
    };
    iree_status_t status = loom_link_materialized_modules(
        inputs.data(), inputs.size(), &options, &block_pool_,
        iree_allocator_system(), out_module);
    return status;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t options = {
        /*.sink=*/{},
        /*.max_errors=*/100,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkerTest, ResolvesForwardReferenceFromLaterCorpusModule) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
}

TEST_F(LinkerTest, ConcreteDefinitionSupersedesDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)

func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootMaterializesReachableSymbols) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @test_target

func.decl target(@test_target) @identity(%x: i32) -> (i32)
func.decl @unused_decl(%x: i32) -> (i32)

func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = LinkRoots({harness, corpus}, {IREE_SV("@caller")});
  Verify(linked);
  EXPECT_EQ(linked->symbols.count, 3u);

  std::string text = Print(linked);
  EXPECT_NE(text.find("test.target<low_core> @test_target"), std::string::npos);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) @identity"),
            std::string::npos);
  EXPECT_EQ(text.find("func.decl @unused_decl"), std::string::npos);
  EXPECT_EQ(text.find("func.def @unused"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootMaterializesApplyContractProviders) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.template<demo.apply> @provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.unused> @unused_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @caller(%x: i32) -> (i32) {
  %y = func.apply<demo.apply>(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));

  loom_module_t* linked = LinkRoots({module}, {IREE_SV("@caller")});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.apply<demo.apply>"), std::string::npos);
  EXPECT_NE(text.find("func.template<demo.apply>"), std::string::npos);
  EXPECT_EQ(text.find("func.template<demo.unused>"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootMaterializesLibraryApplyContractProviders) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @caller(%x: i32) -> (i32) {
  %y = func.apply<demo.apply>(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.template<demo.apply> @provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.unused> @unused_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = LinkRoots({harness, library}, {IREE_SV("@caller")});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.apply<demo.apply>"), std::string::npos);
  EXPECT_NE(text.find("func.template<demo.apply>"), std::string::npos);
  EXPECT_EQ(text.find("func.template<demo.unused>"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootReplacesDeclarationAtStructuralPosition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @before(%x: i32) -> (i32) {
  func.return %x : i32
}

func.decl @identity(%x: i32) -> (i32)

func.def @after(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = LinkRoots({harness, corpus}, {IREE_SV("@after")});
  Verify(linked);

  std::string text = Print(linked);
  size_t before = text.find("func.def @before");
  size_t identity = text.find("func.def @identity");
  size_t after = text.find("func.def @after");
  EXPECT_NE(identity, std::string::npos);
  EXPECT_NE(after, std::string::npos);
  EXPECT_LT(identity, after);
  EXPECT_EQ(before, std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootCanResolveProviderBeforeRootModule) {
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)

func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));

  loom_module_t* linked = LinkRoots({corpus, harness}, {IREE_SV("@caller")});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootIgnoresUnreachableDuplicateDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @caller(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* first = Parse(IREE_SV(R"(
func.def @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked =
      LinkRoots({harness, first, second}, {IREE_SV("@caller")});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_EQ(text.find("func.def @unused"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootRejectsMissingRoot) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def @caller(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        LinkStatus({harness}, {IREE_SV("@missing")}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, SelectiveRootOutputIgnoresUnrelatedFunctionOrder) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* corpus_with_unrelated = Parse(IREE_SV(R"(
func.def @unrelated(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = LinkRoots({harness, corpus}, {IREE_SV("@identity")});
  loom_module_t* linked_with_unrelated =
      LinkRoots({harness, corpus_with_unrelated}, {IREE_SV("@identity")});

  EXPECT_EQ(linked->symbols.count, 1u);
  EXPECT_EQ(linked_with_unrelated->symbols.count, 1u);
  EXPECT_EQ(Print(linked), Print(linked_with_unrelated));
}

TEST_F(LinkerTest, SelectiveRootMaterializesConfigDependency) {
  loom_module_t* module = Parse(IREE_SV(R"(
config.decl @model36.model.hidden_size : index

func.def @read_config() -> (index) {
  %hidden = config.get @model36.model.hidden_size : index
  func.return %hidden : index
}

func.def @unrelated(%x: index) -> (index) {
  func.return %x : index
}
)"));

  loom_module_t* linked = LinkRoots({module}, {IREE_SV("@read_config")});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
  EXPECT_NE(text.find("func.def @read_config"), std::string::npos);
  EXPECT_EQ(text.find("func.def @unrelated"), std::string::npos);
}

TEST_F(LinkerTest, MergesDeclarationTargetContractIntoDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @test_target

func.decl target(@test_target) abi(object_function) export("identity_export") @identity(%x: i32) -> (i32)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) abi(object_function) "
                      "export(\"identity_export\") @identity"),
            std::string::npos);
}

TEST_F(LinkerTest, MergesDeclarationPredicatesIntoDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @test_target

func.decl target(@test_target) @bounded(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)]
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @bounded(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) {
  func.return %x : tensor<[%m]xf32>
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @bounded"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) @bounded"),
            std::string::npos);
  EXPECT_NE(text.find("where [mul(%m, 16)]"), std::string::npos);
}

TEST_F(LinkerTest, RejectsDeclarationDefinitionTargetConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @decl_target

func.decl target(@decl_target) @identity(%x: i32) -> (i32)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
test.target<low_core> @def_target

func.def target(@def_target) @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, RejectsDeclarationDefinitionSignatureConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i64) -> (i64)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, KeepsDeclarationWhenNoDefinitionExists) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @external_identity(%m: index, %x: tensor<[%m]xf32>) -> (tensor<[%m]xf32>) where [mul(%m, 16)]
)"));

  loom_module_t* linked = Link({harness});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.decl @external_identity"), std::string::npos);
  EXPECT_NE(text.find("tensor<[%m]xf32"), std::string::npos);
  EXPECT_NE(text.find("where [mul(%m, 16)]"), std::string::npos);
}

TEST_F(LinkerTest, UsesConfigDefinitionOverDeclaration) {
  loom_module_t* root = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 4096 : index
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 1, 8192), mul(%value, 16)]
)"));

  loom_module_t* linked = Link({root, library});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("config.def @model36.model.hidden_size = 4096 : index"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
}

TEST_F(LinkerTest, MergesIdenticalConfigDefinitions) {
  loom_module_t* first = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 4096 : index
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 4096 : index
)"));

  loom_module_t* linked = Link({first, second});
  Verify(linked);

  std::string text = Print(linked);
  size_t first_def =
      text.find("config.def @model36.model.hidden_size = 4096 : index");
  EXPECT_NE(first_def, std::string::npos);
  EXPECT_EQ(text.find("config.def @model36.model.hidden_size", first_def + 1),
            std::string::npos);
}

TEST_F(LinkerTest, RejectsConflictingConfigDefinitions) {
  loom_module_t* first = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 4096 : index
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 2048 : index
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        LinkStatus({first, second}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, RejectsConfigDefinitionViolatingDeclaration) {
  loom_module_t* root = Parse(IREE_SV(R"(
config.def @model36.model.hidden_size = 4103 : index
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 1, 8192), mul(%value, 16)]
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({root, library}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, MergesConfigDeclarations) {
  loom_module_t* first = Parse(IREE_SV(R"(
config.decl @model36.model.hidden_size : %value: index where [range(%value, 1, 8192)]
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
config.decl @model36.model.hidden_size : %value: index where [mul(%value, 16)]
)"));

  loom_module_t* linked = Link({first, second});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("config.decl @model36.model.hidden_size"),
            std::string::npos);
  EXPECT_NE(text.find("range(%value, 1, 8192)"), std::string::npos);
  EXPECT_NE(text.find("mul(%value, 16)"), std::string::npos);
}

TEST_F(LinkerTest, RenamesPrivateDefinitionConflictsAndRewritesCalls) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @entry_a(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @entry_b(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({first, second});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("func.def @helper("), std::string::npos);
  EXPECT_NE(text.find("func.def @helper$link0("), std::string::npos);
  EXPECT_NE(text.find("func.def public @entry_a"), std::string::npos);
  EXPECT_NE(text.find("func.call @helper(%x)"), std::string::npos);
  EXPECT_NE(text.find("func.def public @entry_b"), std::string::npos);
  EXPECT_NE(text.find("func.call @helper$link0(%x)"), std::string::npos);
}

TEST_F(LinkerTest, PrivateRenameOutputIsStable) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @entry_a(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @entry_b(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked_a = Link({first, second});
  loom_module_t* linked_b = Link({first, second});
  Verify(linked_a);
  Verify(linked_b);
  EXPECT_EQ(Print(linked_a), Print(linked_b));
}

TEST_F(LinkerTest, IncrementalAddDoesNotRetainPreviousSourceModule) {
  loom_linker_t* linker = nullptr;
  loom_linker_options_t linker_options = {
      /*.module_name=*/IREE_SV("linked"),
  };
  IREE_ASSERT_OK(loom_linker_create(&context_, &linker_options, &block_pool_,
                                    iree_allocator_system(), &linker));

  iree_string_view_t roots[] = {IREE_SV("@caller")};
  loom_linker_add_options_t add_options = {
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  {
    ModulePtr harness = ParseOwned(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)

func.def @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
    IREE_ASSERT_OK(loom_linker_add_module(linker, harness.get(), &add_options));
  }
  {
    ModulePtr corpus = ParseOwned(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
    IREE_ASSERT_OK(loom_linker_add_module(linker, corpus.get(), &add_options));
  }

  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
}

TEST_F(LinkerTest, RejectsDuplicatePublicConcreteDefinitions) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  const loom_module_t* inputs[] = {first, second};
  loom_module_t* linked = nullptr;
  loom_link_options_t options = {
      /*.module_name=*/IREE_SV("linked"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_link_materialized_modules(
                            inputs, IREE_ARRAYSIZE(inputs), &options,
                            &block_pool_, iree_allocator_system(), &linked));
  EXPECT_EQ(linked, nullptr);
}

}  // namespace
}  // namespace loom
