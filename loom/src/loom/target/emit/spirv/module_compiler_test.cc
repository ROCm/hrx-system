// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_compiler.h"

#include <string.h>

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/provider.h"
#include "loom/target/arch/spirv/records/target_records.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_version.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/testing/byte_sequence.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::ByteSequenceClone;
using ::loom::testing::ModulePtr;

static constexpr iree_host_size_t kSpirvHeaderWordCount = 5;

static const loom_target_fact_type_t kSpirvCompilerTestTargetFactType = {
    .name = IREE_SVL("spirv-test"),
    .storage_size = sizeof(loom_target_facts_t),
};

static bool SpirvModuleHasCapability(const loom_spirv_module_binary_t& module,
                                     uint32_t capability) {
  if (module.word_count < kSpirvHeaderWordCount ||
      module.words[0] != LOOM_SPIRV_MAGIC_NUMBER) {
    ADD_FAILURE() << "emitter produced an invalid SPIR-V module header";
    return false;
  }
  for (iree_host_size_t i = kSpirvHeaderWordCount; i < module.word_count;) {
    const uint32_t instruction = module.words[i];
    const uint16_t word_count = (uint16_t)(instruction >> 16);
    const uint16_t opcode = (uint16_t)instruction;
    if (word_count == 0 || word_count > module.word_count - i) {
      ADD_FAILURE() << "emitter produced a malformed SPIR-V instruction";
      return false;
    }
    if (opcode == LOOM_SPIRV_OP_CAPABILITY && word_count == 2 &&
        module.words[i + 1] == capability) {
      return true;
    }
    i += word_count;
  }
  return false;
}

struct EncodingDiagnostic {
  // First source operation rejected during emission.
  const loom_op_t* op = nullptr;
  // First structured rejection code.
  const loom_error_def_t* error = nullptr;
  // Original export context forwarded to the sink.
  std::string export_name;
  // Number of emissions delivered to this sink.
  size_t count = 0;

  static iree_status_t Emit(void* user_data,
                            const loom_diagnostic_emission_t* emission) {
    auto* self = static_cast<EncodingDiagnostic*>(user_data);
    if (self->count++ == 0) {
      self->op = emission->op;
      self->error = emission->error;
      if (!iree_string_view_is_empty(emission->export_name)) {
        self->export_name.assign(emission->export_name.data,
                                 emission->export_name.size);
      }
    }
    return iree_ok_status();
  }

  iree_diagnostic_emitter_t emitter() { return {Emit, this}; }
};

class SpirvModuleCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_spirv_target_provider.register_context(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] =
        {
            loom_spirv_logical_core_descriptor_set,
        };
    loom_target_low_descriptor_registry_initialize_from_tables(
        kDescriptorSetProviders, IREE_ARRAYSIZE(kDescriptorSetProviders),
        &low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(iree_string_view_t source) {
    loom_text_parse_options_t options = {.max_errors = 20};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("module_compiler_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  void InitializeTargetFacts(const loom_target_bundle_t* bundle,
                             loom_target_facts_t* out_facts) {
    loom_target_facts_builder_initialize(&kSpirvCompilerTestTargetFactType,
                                         bundle, out_facts);
  }

  loom_target_function_version_t MakeFunctionVersion(
      loom_module_t* module, iree_string_view_t function_name,
      const loom_target_facts_t* target_facts) {
    loom_op_t* function_op =
        module->symbols.entries[FindSymbol(module, function_name)].defining_op;
    loom_func_like_t function = loom_func_like_cast(module, function_op);
    IREE_ASSERT(loom_func_like_isa(function));
    loom_target_function_version_t version = {};
    version.base.type = &loom_target_function_version_type;
    version.base.function = function;
    version.function_target_facts = target_facts;
    return version;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(SpirvModuleCompilerTest, ValueNameInstructionLimit) {
  // OpName's header and target ID leave 65533 words for its NUL-padded name.
  constexpr size_t kMaximumNameLength = (UINT16_MAX - 2u) * 4u - 1u;
  for (bool argument : {false, true}) {
    for (size_t length : {kMaximumNameLength, kMaximumNameLength + 1}) {
      SCOPED_TRACE(::testing::Message()
                   << "argument=" << argument << " length=" << length);
      std::string name(length, 'a');
      std::string source =
          "spirv.target<vulkan1_3> @target\n"
          "low.func.def public target<spirv.logical.core>(@target) "
          "abi(shader_entry_point) @entry(";
      if (argument) {
        source += "%" + name + ": reg<spirv.id : i32>";
      }
      source += ") asm {\n";
      if (!argument) {
        source += "  %" + name + " = OpConstant.i32 1\n";
      }
      source += "  return\n}\n";
      ModulePtr module =
          ParseModule(iree_make_string_view(source.data(), source.size()));
      const loom_op_t* function_op =
          module->symbols.entries[FindSymbol(module.get(), IREE_SV("entry"))]
              .defining_op;
      const loom_op_t* expected_op =
          argument ? function_op
                   : loom_region_const_entry_block(
                         loom_low_function_const_body(function_op))
                         ->first_op;
      for (bool capture : {false, true}) {
        EncodingDiagnostic diagnostic;
        loom_spirv_module_binary_t binary = {};
        bool emitted = false;
        IREE_ASSERT_OK(loom_spirv_compile_module_binary(
            module.get(), &low_registry_.registry,
            capture ? diagnostic.emitter() : iree_diagnostic_emitter_t{},
            &arena_, nullptr, iree_allocator_system(), &emitted, &binary));
        EXPECT_EQ(emitted, length == kMaximumNameLength);
        if (!emitted) {
          EXPECT_EQ(binary.words, nullptr);
          EXPECT_EQ(binary.word_count, 0u);
          if (capture) {
            EXPECT_EQ(diagnostic.error, LOOM_ERR_BACKEND_060);
            EXPECT_EQ(diagnostic.op, expected_op);
            EXPECT_NE(diagnostic.op->location, 0u);
            EXPECT_EQ(diagnostic.export_name, "entry");
          }
        } else {
          EXPECT_EQ(diagnostic.count, 0u);
        }
        loom_spirv_module_binary_deinitialize(&binary, iree_allocator_system());
      }
    }
  }
}

TEST_F(SpirvModuleCompilerTest, EntryNameAndMetadataInstructionLimits) {
  constexpr size_t kMaximumEntryNameLength = (UINT16_MAX - 3u) * 4u - 1u;
  for (bool hal_kernel : {false, true}) {
    for (size_t length :
         {kMaximumEntryNameLength, kMaximumEntryNameLength + 1}) {
      SCOPED_TRACE(::testing::Message()
                   << "hal_kernel=" << hal_kernel << " length=" << length);
      ModulePtr module = ParseModule(IREE_SV(R"(
low.func.def target<spirv.logical.core> @entry() asm {
  return
}
)"));
      std::string name(length, 'e');
      loom_target_facts_t facts = {};
      InitializeTargetFacts(&loom_spirv_low_target_bundle_vulkan1_3, &facts);
      facts.storage.export_plan.name = IREE_SV("source_entry");
      facts.storage.export_plan.export_symbol =
          iree_make_string_view(name.data(), name.size());
      facts.storage.export_plan.abi_kind =
          hal_kernel ? LOOM_TARGET_ABI_HAL_KERNEL
                     : LOOM_TARGET_ABI_SHADER_ENTRY_POINT;
      loom_target_function_version_t version =
          MakeFunctionVersion(module.get(), IREE_SV("entry"), &facts);
      loom_function_version_t* version_values[] = {&version.base};
      loom_function_version_list_t versions = {version_values, 1};
      loom_spirv_compile_options_t options = {.function_versions = &versions};
      EncodingDiagnostic diagnostic;
      loom_spirv_module_binary_t binary = {};
      bool emitted = false;
      IREE_ASSERT_OK(loom_spirv_compile_module_binary(
          module.get(), &low_registry_.registry, diagnostic.emitter(), &arena_,
          &options, iree_allocator_system(), &emitted, &binary));
      // HAL metadata includes the entry name plus its parameter-key syntax,
      // so it can exceed the encoding even while OpEntryPoint itself fits.
      EXPECT_EQ(emitted, !hal_kernel && length == kMaximumEntryNameLength);
      if (!emitted) {
        EXPECT_EQ(binary.words, nullptr);
        EXPECT_EQ(diagnostic.error, LOOM_ERR_BACKEND_060);
        EXPECT_EQ(diagnostic.op, version.base.function.op);
        EXPECT_EQ(diagnostic.export_name, "source_entry");
      }
      loom_spirv_module_binary_deinitialize(&binary, iree_allocator_system());
    }
  }
}

TEST_F(SpirvModuleCompilerTest, InternalSignatureInstructionLimits) {
  const std::string type = "reg<spirv.id : i32>";
  for (bool parameters : {false, true}) {
    const size_t maximum = UINT16_MAX - (parameters ? 4u : 3u);
    for (size_t count : {maximum, maximum + 1}) {
      SCOPED_TRACE(::testing::Message()
                   << "parameters=" << parameters << " count=" << count);
      std::string names;
      std::string types;
      std::string declarations;
      std::string repeated_values;
      for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
          names += ", ";
          types += ", ";
          declarations += ", ";
          repeated_values += ", ";
        }
        const std::string name = "%p" + std::to_string(i);
        names += name;
        types += type;
        declarations += name + ": " + type;
        repeated_values += "%value";
      }
      std::string source =
          "spirv.target<vulkan1_3> @target\n"
          "low.func.def target<spirv.logical.core>(@target) @helper(";
      if (parameters) {
        source += declarations;
      }
      source += ")";
      if (!parameters) {
        source += " -> (" + types + ")";
      }
      source += " asm {\n";
      if (!parameters) {
        source += "  %value = OpConstant.i32 1\n";
      }
      source += "  return";
      if (!parameters) {
        source += " " + repeated_values;
      }
      source +=
          "\n}\nlow.func.def public target<spirv.logical.core>(@target) "
          "abi(shader_entry_point) @entry() asm {\n";
      if (parameters) {
        source += "  %value = OpConstant.i32 1\n";
      }
      source += "  ";
      if (!parameters) {
        source += names + " = ";
      }
      source += "low.func.call @helper(";
      if (parameters) {
        source += repeated_values;
      }
      source += ") : (";
      if (parameters) {
        source += types;
      }
      source += ")";
      if (!parameters) {
        source += " -> (" + types + ")";
      }
      source += "\n  return\n}\n";
      ModulePtr module =
          ParseModule(iree_make_string_view(source.data(), source.size()));
      loom_target_facts_t facts = {};
      InitializeTargetFacts(&loom_spirv_low_target_bundle_vulkan1_3, &facts);
      facts.storage.export_plan.name = IREE_SV("entry");
      const loom_spirv_compile_entry_t entry = {
          .function_op =
              module->symbols
                  .entries[FindSymbol(module.get(), IREE_SV("entry"))]
                  .defining_op,
          .target_facts = &facts,
      };
      const loom_spirv_compile_options_t options = {
          .entries = &entry,
          .entry_count = 1,
      };
      EncodingDiagnostic diagnostic;
      loom_spirv_module_binary_t binary = {};
      bool emitted = false;
      IREE_ASSERT_OK(loom_spirv_compile_module_binary(
          module.get(), &low_registry_.registry, diagnostic.emitter(), &arena_,
          &options, iree_allocator_system(), &emitted, &binary));
      EXPECT_EQ(emitted, count == maximum);
      if (!emitted) {
        EXPECT_EQ(binary.words, nullptr);
        EXPECT_EQ(diagnostic.error, LOOM_ERR_BACKEND_060);
        EXPECT_EQ(
            diagnostic.op,
            module->symbols.entries[FindSymbol(module.get(), IREE_SV("helper"))]
                .defining_op);
      }
      loom_spirv_module_binary_deinitialize(&binary, iree_allocator_system());
    }
  }
}

TEST_F(SpirvModuleCompilerTest, TargetEncodingRequirementsAreDiagnostics) {
  for (int rejection = 0; rejection < 3; ++rejection) {
    SCOPED_TRACE(rejection);
    ModulePtr module = ParseModule(IREE_SV(R"(
low.func.def target<spirv.logical.core> abi(shader_entry_point) @entry() asm {
  return
}
)"));
    loom_target_facts_t facts = {};
    InitializeTargetFacts(&loom_spirv_low_target_bundle_vulkan1_3, &facts);
    facts.storage.export_plan.name = IREE_SV("source_entry");
    if (rejection == 2) {
      facts.storage.config.contract_feature_bits = 0;
    } else if (rejection == 1) {
      facts.storage.config.contract_feature_bits |= UINT64_C(1) << 63;
    } else {
      facts.storage.snapshot.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    }
    loom_target_function_version_t version =
        MakeFunctionVersion(module.get(), IREE_SV("entry"), &facts);
    loom_function_version_t* version_values[] = {&version.base};
    loom_function_version_list_t versions = {version_values, 1};
    loom_spirv_compile_options_t options = {.function_versions = &versions};
    EncodingDiagnostic diagnostic;
    loom_spirv_module_binary_t binary = {};
    bool emitted = true;
    IREE_ASSERT_OK(loom_spirv_compile_module_binary(
        module.get(), &low_registry_.registry, diagnostic.emitter(), &arena_,
        &options, iree_allocator_system(), &emitted, &binary));
    EXPECT_FALSE(emitted);
    EXPECT_EQ(binary.words, nullptr);
    EXPECT_EQ(diagnostic.error, LOOM_ERR_BACKEND_055);
    EXPECT_EQ(diagnostic.op,
              rejection == 2 ? nullptr : version.base.function.op);
    EXPECT_EQ(diagnostic.export_name, rejection == 2 ? "" : "source_entry");
    loom_spirv_module_binary_deinitialize(&binary, iree_allocator_system());
  }
}

TEST_F(SpirvModuleCompilerTest, UnboundCallRejectsWithoutDiagnosticSink) {
  ModulePtr module = ParseModule(IREE_SV(R"(
spirv.target<vulkan1_3> @target
low.func.decl target<spirv.logical.core>(@target) @external()
low.func.def public target<spirv.logical.core>(@target) abi(shader_entry_point) @caller() asm {
  low.func.call @external() : ()
  return
}
)"));
  loom_spirv_module_binary_t binary = {};
  bool emitted = true;
  IREE_ASSERT_OK(loom_spirv_compile_module_binary(
      module.get(), &low_registry_.registry, iree_diagnostic_emitter_t{},
      &arena_, nullptr, iree_allocator_system(), &emitted, &binary));
  EXPECT_FALSE(emitted);
  EXPECT_EQ(binary.words, nullptr);
  EXPECT_EQ(binary.word_count, 0u);
  loom_spirv_module_binary_deinitialize(&binary, iree_allocator_system());
}

TEST_F(SpirvModuleCompilerTest,
       FunctionTargetFactsSelectCapabilitiesWithoutMutatingIR) {
  ModulePtr module = ParseModule(IREE_SV(R"(
low.func.def target<spirv.logical.core> abi(shader_entry_point) @kernel() asm {
  return
}
)"));
  const loom_symbol_id_t function_symbol_id =
      FindSymbol(module.get(), IREE_SV("kernel"));
  loom_func_like_t function = loom_func_like_cast(
      module.get(), module->symbols.entries[function_symbol_id].defining_op);
  ASSERT_TRUE(loom_func_like_isa(function));
  const loom_symbol_ref_t authored_target = loom_func_like_target(function);
  const iree_host_size_t authored_symbol_count = module->symbols.count;

  loom_target_facts_t generic_target_facts = {};
  InitializeTargetFacts(&loom_spirv_low_target_bundle_vulkan1_3,
                        &generic_target_facts);
  loom_target_function_version_t function_version = MakeFunctionVersion(
      module.get(), IREE_SV("kernel"), &generic_target_facts);
  loom_function_version_t* version_values[] = {&function_version.base};
  loom_function_version_list_t function_versions = {
      .values = version_values,
      .count = IREE_ARRAYSIZE(version_values),
  };
  loom_spirv_compile_options_t options = {
      .function_versions = &function_versions,
  };

  loom_spirv_module_binary_t generic_module = {};
  bool generic_emitted = false;
  IREE_ASSERT_OK(loom_spirv_compile_module_binary(
      module.get(), &low_registry_.registry, iree_diagnostic_emitter_t{},
      &arena_, &options, iree_allocator_system(), &generic_emitted,
      &generic_module));
  ASSERT_TRUE(generic_emitted);
  EXPECT_FALSE(
      SpirvModuleHasCapability(generic_module, LOOM_SPIRV_CAPABILITY_FLOAT16));
  EXPECT_FALSE(
      SpirvModuleHasCapability(generic_module, LOOM_SPIRV_CAPABILITY_INT8));
  EXPECT_FALSE(
      SpirvModuleHasCapability(generic_module, LOOM_SPIRV_CAPABILITY_INT16));
  loom_spirv_module_binary_deinitialize(&generic_module,
                                        iree_allocator_system());

  loom_target_facts_t exact_target_facts = {};
  InitializeTargetFacts(&loom_spirv_low_target_bundle_extended_types,
                        &exact_target_facts);
  function_version.function_target_facts = &exact_target_facts;

  loom_spirv_module_binary_t exact_module = {};
  bool exact_emitted = false;
  IREE_ASSERT_OK(loom_spirv_compile_module_binary(
      module.get(), &low_registry_.registry, iree_diagnostic_emitter_t{},
      &arena_, &options, iree_allocator_system(), &exact_emitted,
      &exact_module));
  ASSERT_TRUE(exact_emitted);
  EXPECT_TRUE(
      SpirvModuleHasCapability(exact_module, LOOM_SPIRV_CAPABILITY_FLOAT16));
  EXPECT_TRUE(
      SpirvModuleHasCapability(exact_module, LOOM_SPIRV_CAPABILITY_INT8));
  EXPECT_TRUE(
      SpirvModuleHasCapability(exact_module, LOOM_SPIRV_CAPABILITY_INT16));
  loom_spirv_module_binary_deinitialize(&exact_module, iree_allocator_system());

  EXPECT_EQ(module->symbols.count, authored_symbol_count);
  const loom_symbol_ref_t emitted_target = loom_func_like_target(function);
  EXPECT_EQ(emitted_target.module_id, authored_target.module_id);
  EXPECT_EQ(emitted_target.symbol_id, authored_target.symbol_id);
}

TEST_F(SpirvModuleCompilerTest, RetainsRequestedArtifactMetadata) {
  ModulePtr module = ParseModule(IREE_SV(R"(
low.func.def target<spirv.logical.core> @helper() asm {
  return
}

low.kernel.def target<spirv.logical.core> workgroup_size(1, 1, 1) @loom_kernel() asm {
  low.func.call @helper() : ()
  return
}
)"));

  loom_target_facts_t target_facts = {};
  InitializeTargetFacts(&loom_spirv_low_target_bundle_hal_kernel,
                        &target_facts);
  char export_symbol[] = "retained_entry";
  target_facts.storage.export_plan.export_symbol =
      iree_make_cstring_view(export_symbol);
  loom_target_function_version_t function_version =
      MakeFunctionVersion(module.get(), IREE_SV("loom_kernel"), &target_facts);
  function_version.base.flags = LOOM_FUNCTION_VERSION_FLAG_RETAIN;
  loom_target_facts_t helper_target_facts = {};
  InitializeTargetFacts(&loom_spirv_low_target_bundle_hal_kernel,
                        &helper_target_facts);
  helper_target_facts.storage.export_plan.abi_kind = LOOM_TARGET_ABI_UNKNOWN;
  loom_target_function_version_t helper_version = MakeFunctionVersion(
      module.get(), IREE_SV("helper"), &helper_target_facts);
  loom_function_version_t* version_values[] = {
      &helper_version.base,
      &function_version.base,
  };
  const loom_function_version_list_t function_versions = {
      .values = version_values,
      .count = IREE_ARRAYSIZE(version_values),
  };

  loom_target_emit_request_t request = {
      .low_descriptor_registry = &low_registry_.registry,
      .module = module.get(),
      .function_versions = &function_versions,
      .identifier = IREE_SV("module.spv"),
      .scratch_arena = &arena_,
      .allocator = iree_allocator_system(),
  };
  loom_target_emit_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_spirv_module_emitter.emit(&request, &emitted, &artifact));
  ASSERT_TRUE(emitted);
  EXPECT_EQ(artifact.target_bundle, nullptr);
  EXPECT_EQ(artifact.storage, nullptr);
  loom_target_emit_artifact_release(&artifact);

  request.flags = LOOM_TARGET_EMIT_REQUEST_FLAG_RETAIN_TARGET_BUNDLE;
  request.artifact_manifest.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
  char manifest_identifier[] = "module.manifest.json";
  request.artifact_manifest.identifier =
      iree_make_cstring_view(manifest_identifier);
  IREE_ASSERT_OK(loom_spirv_module_emitter.emit(&request, &emitted, &artifact));
  memset(export_symbol, '!', sizeof(export_symbol));
  memset(manifest_identifier, '!', sizeof(manifest_identifier));

  ASSERT_TRUE(emitted);
  ASSERT_NE(artifact.target_bundle, nullptr);
  EXPECT_EQ(artifact.target_bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(artifact.target_bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(artifact.target_bundle->export_plan->abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_TRUE(
      iree_string_view_equal(artifact.target_bundle->export_plan->export_symbol,
                             IREE_SV("retained_entry")));
  EXPECT_EQ(artifact.target_artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  ASSERT_NE(artifact.contents, nullptr);
  ASSERT_EQ(artifact.sidecar_count, 1u);
  ASSERT_NE(artifact.sidecars, nullptr);
  EXPECT_EQ(artifact.sidecars[0].kind,
            LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST);
  EXPECT_TRUE(iree_string_view_equal(artifact.sidecars[0].identifier,
                                     IREE_SV("module.manifest.json")));

  ByteSequenceClone executable_data(iree_allocator_system());
  IREE_ASSERT_OK(executable_data.Clone(artifact.contents));
  const iree_const_byte_span_t executable_contents = executable_data.contents();
  ASSERT_GE(executable_contents.data_length, sizeof(uint32_t));
  uint32_t magic = 0;
  memcpy(&magic, executable_contents.data, sizeof(magic));
  EXPECT_EQ(magic, LOOM_SPIRV_MAGIC_NUMBER);

  ByteSequenceClone manifest_data(iree_allocator_system());
  IREE_ASSERT_OK(manifest_data.Clone(artifact.sidecars[0].contents));
  const iree_const_byte_span_t manifest_contents = manifest_data.contents();
  const std::string manifest_text(
      reinterpret_cast<const char*>(manifest_contents.data),
      manifest_contents.data_length);
  EXPECT_NE(manifest_text.find("\"format\":\"spirv-binary\""),
            std::string::npos)
      << manifest_text;
  EXPECT_NE(manifest_text.find("\"name\":\"module.spv\""), std::string::npos)
      << manifest_text;
  EXPECT_EQ(manifest_text.find("helper"), std::string::npos) << manifest_text;

  loom_target_emit_artifact_release(&artifact);
}

}  // namespace
}  // namespace loom
