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
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/target/test/alt_descriptors.h"
#include "loom/target/test/descriptors.h"
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
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    low_registry_.descriptor_set_providers = low_descriptor_set_providers_;
    low_registry_.descriptor_set_provider_count =
        IREE_ARRAYSIZE(low_descriptor_set_providers_);
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
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_, &parse_options.low_asm_environment);
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
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_, &parse_options.low_asm_environment);
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

  loom_linker_t* CreateIncrementalLinker(
      iree_host_size_t provider_import_count = 0,
      iree_host_size_t provider_import_anchor_count = 0) {
    loom_linker_t* linker = nullptr;
    loom_linker_options_t options = {};
    options.module_name = IREE_SV("linked");
    options.provider_imports.count = provider_import_count;
    options.provider_imports.anchor_count = provider_import_anchor_count;
    IREE_CHECK_OK(loom_linker_create(&context_, &options, &block_pool_,
                                     iree_allocator_system(), &linker));
    return linker;
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

  // Synthetic representation contracts accepted by Low parser fixtures.
  loom_low_descriptor_set_provider_t low_descriptor_set_providers_[2] = {
      loom_test_low_core_descriptor_set,
      loom_test_low_alt_descriptor_set,
  };

  // Descriptor registry projected into the text parser environment.
  loom_low_descriptor_registry_t low_registry_ = {};

  std::vector<loom_module_t*> modules_;
};

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

TEST_F(LinkerTest, CommandDefinitionSupersedesMatchingDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @test_target

command.program.decl target(@test_target) @decode(%token_count: index) launch(%parameters: buffer)

command.program.def public @entry(%token_count: index) launch(%parameters: buffer) {
  command.program.launch @decode[%token_count](%parameters) : [index](buffer)
  command.return
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
command.program.def @decode(%token_count: index) launch(%parameters: buffer) {
  command.return
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("command.program.decl @decode"), std::string::npos);
  EXPECT_NE(text.find("command.program.def target(@test_target) @decode"),
            std::string::npos);
  EXPECT_NE(text.find("command.program.launch @decode"), std::string::npos);
}

TEST_F(LinkerTest, ConcreteTargetRecordSupersedesDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
target.decl @gpu

func.def target(@gpu) @entry() {
  func.return
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
test.target<low_core> @gpu
)"));

  Verify(harness);
  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("target.decl @gpu"), std::string::npos);
  EXPECT_NE(text.find("test.target<low_core> @gpu"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@gpu) @entry"), std::string::npos);
}

TEST_F(LinkerTest, ConcreteRodataSupersedesDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
global.rodata.decl @metadata
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
global.rodata.def @metadata = align(4) bytes("4c4f4f4d")
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("global.rodata.decl @metadata"), std::string::npos);
  EXPECT_NE(
      text.find("global.rodata.def @metadata = align(4) bytes(\"4c4f4f4d\")"),
      std::string::npos);
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
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def target(@test_target) @identity"),
            std::string::npos);
  EXPECT_EQ(text.find("func.decl @unused_decl"), std::string::npos);
  EXPECT_EQ(text.find("func.def @unused"), std::string::npos);
}

TEST_F(LinkerTest, SelectiveRootMaterializesApplyContractProviders) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.template<demo.apply> requires [#target.subgroup.size<64>] @provider(%x: i32) -> (i32) {
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
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
  EXPECT_NE(text.find("func.apply<demo.apply>"), std::string::npos);
  EXPECT_NE(text.find("func.template<demo.apply> requires "
                      "[#target.subgroup.size<64>] @provider"),
            std::string::npos);
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
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
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
  size_t after = text.find("func.def retain @after");
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
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
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
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
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
  EXPECT_NE(text.find("func.def retain @read_config"), std::string::npos);
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

TEST_F(LinkerTest, RejectsDeclarationDefinitionRepresentationConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
low.func.decl target<test.low.core> @identity()
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
low.func.def target<test.low.alt> @identity() {
  low.return
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

TEST_F(LinkerTest, RejectsCommandSignaturePartitionConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
command.program.decl @program(%fixed: buffer) launch(%dynamic: buffer)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
command.program.def @program() launch(%fixed: buffer, %dynamic: buffer) {
  command.return
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, MatchesKernelWorkloadAndLaunchSignatures) {
  loom_module_t* harness = Parse(IREE_SV(R"(
kernel.decl @dynamic_copy(%element_count: index) launch(%row_count: index, %output: tensor<[%row_count]xi32>)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
kernel.def @dynamic_copy(%length: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%length, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%rows: index, %destination: tensor<[%rows]xi32>) {
  kernel.return
}
)"));

  loom_module_t* linked = Link({harness, corpus});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("kernel.decl @dynamic_copy"), std::string::npos);
  EXPECT_NE(text.find("kernel.def @dynamic_copy"), std::string::npos);
}

TEST_F(LinkerTest, RejectsKernelWorkloadCountConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
kernel.decl @fill(%element_count: index, %row_count: index) launch(%output: buffer)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
kernel.def @fill(%element_count: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%output: buffer) {
  kernel.return
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, RejectsKernelWorkloadTypeConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
kernel.decl @fill(%element_count: i64) launch(%output: buffer)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
kernel.def @fill(%element_count: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%output: buffer) {
  kernel.return
}
)"));

  loom_module_t* linked = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        LinkStatus({harness, corpus}, &linked));
  EXPECT_EQ(linked, nullptr);
}

TEST_F(LinkerTest, RejectsKernelLaunchSignatureConflict) {
  loom_module_t* harness = Parse(IREE_SV(R"(
kernel.decl @fill(%element_count: index) launch(%output: buffer)
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
kernel.def @fill(%element_count: index) {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%output: buffer, %value: i32) {
  kernel.return
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

TEST_F(LinkerTest, PrivateRenameRecanonicalizesIndexedImportAnchors) {
  loom_module_t* first = Parse(IREE_SV(R"(
module.import "provider" [@helper, @helper$link0]

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @helper$link0(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({first, second});
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("module.import \"provider\" "
                      "[@helper$link0, @helper$link1]"),
            std::string::npos);
  EXPECT_NE(text.find("func.def @helper$link1("), std::string::npos);
  EXPECT_NE(text.find("func.def public @helper("), std::string::npos);
}

TEST_F(LinkerTest, MergesProviderImportsAcrossModules) {
  loom_module_t* zeta = Parse(IREE_SV(R"(
// Zeta candidate.
module.import "provider" [@zeta]

func.def @zeta(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* alpha = Parse(IREE_SV(R"(
// Alpha candidate.
module.import "provider" [@alpha]

func.def @alpha(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_module_t* linked = Link({zeta, alpha});
  Verify(linked);

  const std::string text = Print(linked);
  const std::string import = "module.import \"provider\" [@alpha, @zeta]";
  const size_t import_position = text.find(import);
  ASSERT_NE(import_position, std::string::npos) << text;
  EXPECT_EQ(text.find("module.import \"provider\"", import_position + 1),
            std::string::npos);
  const size_t alpha_comment = text.find("// Alpha candidate.");
  const size_t zeta_comment = text.find("// Zeta candidate.");
  ASSERT_NE(alpha_comment, std::string::npos);
  ASSERT_NE(zeta_comment, std::string::npos);
  EXPECT_LT(alpha_comment, zeta_comment);
  EXPECT_LT(zeta_comment, import_position);
}

TEST_F(LinkerTest, ProjectedImportFollowsExactPrivateRename) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* projected = Parse(IREE_SV(R"(
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @zeta(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_linker_t* linker = CreateIncrementalLinker(
      /*provider_import_count=*/1, /*provider_import_anchor_count=*/2);
  IREE_ASSERT_OK(loom_linker_add_exact_module(
      linker, first, loom_linker_source_provider_import_list_empty()));
  const iree_host_size_t projected_symbols[] = {0, 1};
  const uint32_t import_anchors[] = {0, 1};
  const iree_string_view_t import_comments[] = {
      IREE_SV("Projected provider."),
  };
  const loom_linker_source_provider_import_t provider_imports[] = {{
      /*.provider=*/IREE_SV("provider"),
      /*.anchors=*/
      {
          /*.count=*/IREE_ARRAYSIZE(import_anchors),
          /*.ordinals=*/import_anchors,
      },
      /*.comments=*/
      {
          /*.count=*/IREE_ARRAYSIZE(import_comments),
          /*.values=*/import_comments,
      },
      /*.leading_blank_line=*/true,
  }};
  IREE_ASSERT_OK(loom_linker_add_module_symbols(
      linker, projected,
      (loom_linker_source_symbol_list_t){
          /*.count=*/IREE_ARRAYSIZE(projected_symbols),
          /*.ordinals=*/projected_symbols,
      },
      (loom_linker_source_provider_import_list_t){
          /*.count=*/IREE_ARRAYSIZE(provider_imports),
          /*.values=*/provider_imports,
      }));

  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  const std::string text = Print(linked);
  EXPECT_NE(text.find("// Projected provider.\n"
                      "module.import \"provider\" "
                      "[@helper$link0, @zeta]"),
            std::string::npos);
  const loom_op_t* import_op = nullptr;
  const loom_op_t* candidate_op = nullptr;
  loom_block_for_each_op(loom_region_const_entry_block(linked->body),
                         candidate_op) {
    if (loom_module_import_isa(candidate_op)) {
      import_op = candidate_op;
      break;
    }
  }
  ASSERT_NE(import_op, nullptr) << text;
  EXPECT_TRUE(
      iree_any_bit_set(import_op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE));
}

TEST_F(LinkerTest, ProjectedImportRequiresReservedCapacity) {
  loom_module_t* source = Parse(IREE_SV(R"(
func.def @helper() {
  func.return
}
)"));
  const iree_host_size_t symbols[] = {0};
  const uint32_t anchors[] = {0};
  const loom_linker_source_provider_import_t provider_imports[] = {{
      /*.provider=*/IREE_SV("provider"),
      /*.anchors=*/
      {
          /*.count=*/IREE_ARRAYSIZE(anchors),
          /*.ordinals=*/anchors,
      },
  }};

  loom_linker_t* linker = CreateIncrementalLinker();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/IREE_ARRAYSIZE(symbols),
                                /*.ordinals=*/symbols,
                            },
                            (loom_linker_source_provider_import_list_t){
                                /*.count=*/IREE_ARRAYSIZE(provider_imports),
                                /*.values=*/provider_imports,
                            }));
  loom_linker_free(linker);

  linker = CreateIncrementalLinker(
      /*provider_import_count=*/1, /*provider_import_anchor_count=*/1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_linker_add_module_symbols(
                            linker, source, /*source_symbols=*/{},
                            (loom_linker_source_provider_import_list_t){
                                /*.count=*/IREE_ARRAYSIZE(provider_imports),
                                /*.values=*/provider_imports,
                            }));
  loom_linker_free(linker);
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

TEST_F(LinkerTest, IncrementalLinkDoesNotReferenceReleasedSourceModules) {
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

  IREE_ASSERT_OK(loom_linker_finalize_roots(linker, add_options.root_symbols));
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def retain @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
}

TEST_F(LinkerTest, ExactSelectionsLinkPrecomputedCrossModuleClosure) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @identity(%x: i32) -> (i32)

func.def public @caller(%x: i32) -> (i32) {
  %y = func.call @identity(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* corpus = Parse(IREE_SV(R"(
func.def @identity(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @unused_corpus(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  loom_linker_t* linker = CreateIncrementalLinker();
  const iree_host_size_t harness_symbols[] = {0, 1};
  IREE_ASSERT_OK(loom_linker_add_module_symbols(
      linker, harness,
      (loom_linker_source_symbol_list_t){
          /*.count=*/IREE_ARRAYSIZE(harness_symbols),
          /*.ordinals=*/harness_symbols,
      },
      loom_linker_source_provider_import_list_empty()));
  const iree_host_size_t corpus_symbols[] = {0};
  IREE_ASSERT_OK(loom_linker_add_module_symbols(
      linker, corpus,
      (loom_linker_source_symbol_list_t){
          /*.count=*/IREE_ARRAYSIZE(corpus_symbols),
          /*.ordinals=*/corpus_symbols,
      },
      loom_linker_source_provider_import_list_empty()));
  const iree_string_view_t roots[] = {IREE_SV("@caller")};
  IREE_ASSERT_OK(
      loom_linker_finalize_roots(linker, (iree_string_view_list_t){
                                             /*.count=*/IREE_ARRAYSIZE(roots),
                                             /*.values=*/roots,
                                         }));
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_EQ(text.find("func.decl @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def @identity"), std::string::npos);
  EXPECT_NE(text.find("func.def public retain @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @identity"), std::string::npos);
  EXPECT_EQ(text.find("@unused"), std::string::npos);
}

TEST_F(LinkerTest, ExactModuleLinksCompleteDenseProjection) {
  loom_module_t* source = Parse(IREE_SV(R"(
test.module_metadata
module.import "consumed" [@helper]

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @caller(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));

  loom_linker_t* linker = CreateIncrementalLinker();
  IREE_ASSERT_OK(loom_linker_add_exact_module(
      linker, source, loom_linker_source_provider_import_list_empty()));
  const iree_string_view_t roots[] = {IREE_SV("@caller")};
  IREE_ASSERT_OK(
      loom_linker_finalize_roots(linker, (iree_string_view_list_t){
                                             /*.count=*/IREE_ARRAYSIZE(roots),
                                             /*.values=*/roots,
                                         }));
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  std::string text = Print(linked);
  EXPECT_NE(text.find("test.module_metadata"), std::string::npos);
  EXPECT_EQ(text.find("module.import"), std::string::npos);
  EXPECT_NE(text.find("func.def @helper"), std::string::npos);
  EXPECT_NE(text.find("func.def public retain @caller"), std::string::npos);
  EXPECT_NE(text.find("func.call @helper"), std::string::npos);
}

TEST_F(LinkerTest, ExactModuleLinksMetadataWithoutSymbols) {
  loom_module_t* source = Parse(IREE_SV("test.module_metadata\n"));

  loom_linker_t* linker = CreateIncrementalLinker();
  IREE_ASSERT_OK(loom_linker_add_exact_module(
      linker, source, loom_linker_source_provider_import_list_empty()));
  loom_module_t* linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(linker, &linked));
  loom_linker_free(linker);
  modules_.push_back(linked);
  Verify(linked);

  EXPECT_NE(Print(linked).find("test.module_metadata"), std::string::npos);
}

TEST_F(LinkerTest, ExactModulePreservesSourceOperationOrder) {
  loom_module_t* source = Parse(IREE_SV(R"(
func.def @first() {
  func.return
}

func.def @second() {
  func.return
}
)"));
  loom_block_t* source_block = loom_module_block(source);
  loom_op_t* first_op = source_block->first_op;
  ASSERT_NE(first_op, nullptr);
  loom_op_t* second_op = first_op->next_op;
  ASSERT_NE(second_op, nullptr);
  loom_block_unlink_op(source, second_op);
  IREE_ASSERT_OK(
      loom_block_insert_before_op(source, source_block, first_op, second_op));
  loom_module_record_op_effects(source, second_op);

  loom_linker_t* dense_linker = CreateIncrementalLinker();
  IREE_ASSERT_OK(loom_linker_add_exact_module(
      dense_linker, source, loom_linker_source_provider_import_list_empty()));
  loom_module_t* dense_linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(dense_linker, &dense_linked));
  loom_linker_free(dense_linker);
  modules_.push_back(dense_linked);
  Verify(dense_linked);

  const iree_host_size_t source_symbols[] = {0, 1};
  loom_linker_t* sparse_linker = CreateIncrementalLinker();
  IREE_ASSERT_OK(loom_linker_add_module_symbols(
      sparse_linker, source,
      (loom_linker_source_symbol_list_t){
          /*.count=*/IREE_ARRAYSIZE(source_symbols),
          /*.ordinals=*/source_symbols,
      },
      loom_linker_source_provider_import_list_empty()));
  loom_module_t* sparse_linked = nullptr;
  IREE_ASSERT_OK(loom_linker_finish(sparse_linker, &sparse_linked));
  loom_linker_free(sparse_linker);
  modules_.push_back(sparse_linked);
  Verify(sparse_linked);

  for (const loom_module_t* linked : {dense_linked, sparse_linked}) {
    const std::string text = Print(linked);
    const size_t second_position = text.find("func.def @second");
    const size_t first_position = text.find("func.def @first");
    ASSERT_NE(second_position, std::string::npos);
    ASSERT_NE(first_position, std::string::npos);
    EXPECT_LT(second_position, first_position);
  }
}

TEST_F(LinkerTest, ExactSelectionRejectsMissingDependency) {
  loom_module_t* source = Parse(IREE_SV(R"(
func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @caller(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));

  loom_linker_t* linker = CreateIncrementalLinker();
  const iree_host_size_t source_symbols[] = {1};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/IREE_ARRAYSIZE(source_symbols),
                                /*.ordinals=*/source_symbols,
                            },
                            loom_linker_source_provider_import_list_empty()));
  loom_linker_free(linker);
}

TEST_F(LinkerTest, ExactSelectionRejectsMalformedOrdinals) {
  loom_module_t* source = Parse(IREE_SV(R"(
func.def @first() {
  func.return
}

func.def @second() {
  func.return
}
)"));

  loom_linker_t* linker = CreateIncrementalLinker();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/1,
                                /*.ordinals=*/nullptr,
                            },
                            loom_linker_source_provider_import_list_empty()));
  const iree_host_size_t duplicate_symbols[] = {0, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/IREE_ARRAYSIZE(duplicate_symbols),
                                /*.ordinals=*/duplicate_symbols,
                            },
                            loom_linker_source_provider_import_list_empty()));
  const iree_host_size_t descending_symbols[] = {1, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/IREE_ARRAYSIZE(descending_symbols),
                                /*.ordinals=*/descending_symbols,
                            },
                            loom_linker_source_provider_import_list_empty()));
  const iree_host_size_t out_of_range_symbols[] = {2};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_linker_add_module_symbols(
                            linker, source,
                            (loom_linker_source_symbol_list_t){
                                /*.count=*/IREE_ARRAYSIZE(out_of_range_symbols),
                                /*.ordinals=*/out_of_range_symbols,
                            },
                            loom_linker_source_provider_import_list_empty()));
  loom_linker_free(linker);
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

TEST_F(LinkerTest, ProviderDefinitionSatisfiesProviderDeclaration) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.provider.decl<demo.effect> @provider(%x: i32) -> (i32)
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.template<demo.effect> @provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  const loom_module_t* inputs[] = {harness, library};
  loom_module_t* linked = nullptr;
  loom_link_options_t options = {
      /*.module_name=*/IREE_SV("linked"),
  };
  IREE_ASSERT_OK(loom_link_materialized_modules(
      inputs, IREE_ARRAYSIZE(inputs), &options, &block_pool_,
      iree_allocator_system(), &linked));
  modules_.push_back(linked);
  Verify(linked);

  const std::string text = Print(linked);
  EXPECT_EQ(text.find("func.provider.decl"), std::string::npos);
  EXPECT_NE(text.find("func.template<demo.effect> @provider"),
            std::string::npos);
}

TEST_F(LinkerTest, RejectsMismatchedProviderDeclarationContract) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.provider.decl<demo.expected> @provider(%x: i32) -> (i32)
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.template<demo.other> @provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  const loom_module_t* inputs[] = {harness, library};
  loom_module_t* linked = nullptr;
  loom_link_options_t options = {
      /*.module_name=*/IREE_SV("linked"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_link_materialized_modules(
                            inputs, IREE_ARRAYSIZE(inputs), &options,
                            &block_pool_, iree_allocator_system(), &linked));
  EXPECT_EQ(linked, nullptr);
}

}  // namespace
}  // namespace loom
