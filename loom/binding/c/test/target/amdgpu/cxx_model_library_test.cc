// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/amdgpu/cxx_model_library_sources.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/import/cxx.h"
#include "loomc/link.h"
#include "loomc/target/amdgpu.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static constexpr std::array<const char*, 3> kLibraryRoots = {
    "ffn_routed_gate_up_swiglu_q4k_q8",
    "nvfp4_matrix",
    "qwen38_iq4xs_gate_up_swiglu",
};

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string text = ToString(contents);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return text;
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        ToString(artifact->format) == format) {
      return artifact;
    }
  }
  return nullptr;
}

SourcePtr CreateSource(const iree_file_toc_t& file) {
  const loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_UNKNOWN,
      /*.identifier=*/loomc_make_cstring_view(file.name),
      /*.contents=*/loomc_make_byte_span(file.data, file.size),
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

// Immutable source set supplied to one frontend invocation.
class EmbeddedSourceSet {
 public:
  EmbeddedSourceSet() {
    const iree_file_toc_t* files = loomc_cxx_model_library_sources_create();
    const iree_host_size_t file_count = loomc_cxx_model_library_sources_size();
    sources_.reserve(file_count);
    for (iree_host_size_t i = 0; i < file_count; ++i) {
      sources_.push_back({CreateSource(files[i]), false});
    }
  }

  loomc_source_t* Find(std::string_view name) const {
    for (const Entry& entry : sources_) {
      if (ToString(loomc_source_identifier(entry.source.get())) == name) {
        return entry.source.get();
      }
    }
    return nullptr;
  }

  void ExpectRequiredIncludes() const {
    static constexpr std::array<const char*, 5> kRequiredIncludes = {
        "iq4xs_gate_up.cxx",          "nvfp4_matrix.cxx",  "nvfp4_matrix.h",
        "nvfp4_matrix_providers.cxx", "q4k_q8_swiglu.cxx",
    };
    for (const char* required : kRequiredIncludes) {
      const Entry* found = nullptr;
      for (const Entry& entry : sources_) {
        if (ToString(loomc_source_identifier(entry.source.get())) == required) {
          found = &entry;
          break;
        }
      }
      ASSERT_NE(found, nullptr) << required;
      EXPECT_TRUE(found->requested) << required;
    }
  }

  static loomc_status_t Resolve(void* user_data, loomc_string_view_t path,
                                loomc_source_t** out_source) {
    auto& self = *static_cast<EmbeddedSourceSet*>(user_data);
    *out_source = nullptr;
    std::string_view name(path.data, path.size);
    const size_t separator = name.find_last_of('/');
    if (separator != std::string_view::npos) {
      name.remove_prefix(separator + 1);
    }
    for (Entry& entry : self.sources_) {
      if (ToString(loomc_source_identifier(entry.source.get())) == name) {
        entry.requested = true;
        loomc_source_retain(entry.source.get());
        *out_source = entry.source.get();
        break;
      }
    }
    return loomc_ok_status();
  }

 private:
  struct Entry {
    // Immutable source retained for the frontend invocation.
    SourcePtr source;
    // Whether preprocessing requested this source by include path.
    bool requested;
  };

  // Embedded translation-unit members indexed by basename.
  std::vector<Entry> sources_;
};

struct ConfigBindingSpec {
  // Authored configuration key.
  const char* key;
  // Exact value supplied to this product.
  const char* value;
};

struct ProductSpec {
  // Stable product label used in test traces and artifact identifiers.
  const char* name;
  // Exported kernel selected from the reusable library.
  const char* root;
  // Concrete AMDGPU processor profile.
  const char* target;
  // Per-product configuration bindings.
  const ConfigBindingSpec* configs;
  // Number of configuration bindings.
  size_t config_count;
  // Target instruction proving the expected semantic provider reached codegen.
  const char* listing_marker;
};

static constexpr ConfigBindingSpec kQ4KQ8Configs[] = {
    {"ffn_routed_gate_up.input_size", "512"},
};

static constexpr ConfigBindingSpec kIq4XsSingleConfigs[] = {
    {"qwen38.iq4xs.async_staging", "true"},
    {"qwen38.iq4xs.staging_depth", "1"},
    {"qwen38.iq4xs.packet_unroll_factor", "4"},
    {"qwen38.iq4xs.block_unroll_factor", "2"},
    {"qwen38.iq4xs.tile_unroll_factor", "1"},
};

static constexpr ConfigBindingSpec kIq4XsDoubleConfigs[] = {
    {"qwen38.iq4xs.async_staging", "true"},
    {"qwen38.iq4xs.staging_depth", "2"},
    {"qwen38.iq4xs.packet_unroll_factor", "4"},
    {"qwen38.iq4xs.block_unroll_factor", "2"},
    {"qwen38.iq4xs.tile_unroll_factor", "1"},
};

static constexpr ProductSpec kProducts[] = {
    {
        "q4k_q8_gfx1250",
        "ffn_routed_gate_up_swiglu_q4k_q8",
        "gfx1250",
        kQ4KQ8Configs,
        std::size(kQ4KQ8Configs),
        nullptr,
    },
    {
        "nvfp4_wave32",
        "nvfp4_matrix",
        "gfx1100",
        nullptr,
        0,
        "v_wmma_f32_16x16x16_f16",
    },
    {
        "nvfp4_wave64",
        "nvfp4_matrix",
        "gfx942",
        nullptr,
        0,
        "v_mfma_f32_16x16x16_f16",
    },
    {
        "iq4xs_async_single",
        "qwen38_iq4xs_gate_up_swiglu",
        "gfx942",
        kIq4XsSingleConfigs,
        std::size(kIq4XsSingleConfigs),
        "global_load_lds_dword",
    },
    {
        "iq4xs_async_double",
        "qwen38_iq4xs_gate_up_swiglu",
        "gfx942",
        kIq4XsDoubleConfigs,
        std::size(kIq4XsDoubleConfigs),
        "global_load_lds_dword",
    },
};

class CxxModelLibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_target_environment_t* environment = nullptr;
    LOOMC_ASSERT_OK(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment));
    environment_.reset(environment);

    const loomc_context_target_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.target_environment=*/environment_.get(),
    };
    const loomc_context_options_t context_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        /*.structure_size=*/sizeof(context_options),
        /*.next=*/&target_options,
    };
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                         loomc_allocator_system(), &context));
    context_.reset(context);

    SourcePtr library_bytecode;
    {
      EmbeddedSourceSet source_set;
      loomc_source_t* main_source =
          source_set.Find("specializing_model_library.cxx");
      ASSERT_NE(main_source, nullptr);

      loomc_workspace_t* workspace = nullptr;
      LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                             &workspace));
      WorkspacePtr import_workspace(workspace);

      std::array<loomc_string_view_t, kLibraryRoots.size()> roots;
      for (size_t i = 0; i < roots.size(); ++i) {
        roots[i] = loomc_make_cstring_view(kLibraryRoots[i]);
      }
      loomc_cxx_import_options_t import_options = {};
      import_options.type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS;
      import_options.structure_size = sizeof(import_options);
      import_options.flags = LOOMC_CXX_IMPORT_FLAG_APPROXIMATE_FUNCTIONS;
      import_options.source_provider = {EmbeddedSourceSet::Resolve,
                                        &source_set};
      import_options.roots = roots.data();
      import_options.root_count = roots.size();

      loomc_module_t* module = nullptr;
      loomc_result_t* result = nullptr;
      LOOMC_ASSERT_OK(loomc_module_import_cxx(
          context_.get(), import_workspace.get(), main_source, &import_options,
          loomc_allocator_system(), &module, &result));
      ModulePtr imported_module(module);
      ResultPtr import_result(result);
      ExpectSucceededResult(import_result.get());
      ASSERT_NE(imported_module, nullptr);
      source_set.ExpectRequiredIncludes();

      loomc_module_serialize_options_t serialize_options = {};
      serialize_options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
      serialize_options.structure_size = sizeof(serialize_options);
      serialize_options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
      serialize_options.identifier =
          loomc_make_cstring_view("specializing_model_library.loombc");
      loomc_source_t* serialized = nullptr;
      LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
          imported_module.get(), &serialize_options, loomc_allocator_system(),
          &serialized));
      library_bytecode.reset(serialized);
    }

    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_create(
        context_.get(), nullptr, loomc_allocator_system(), &builder));
    LinkIndexBuilderPtr builder_owner(builder);
    const loomc_link_index_source_options_t source_options = {
        /*.provider_name=*/loomc_make_cstring_view("model-library"),
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        builder, library_bytecode.get(), &source_options, nullptr));
    loomc_link_index_t* link_index = nullptr;
    loomc_result_t* index_result = nullptr;
    LOOMC_ASSERT_OK(
        loomc_link_index_builder_finish(builder, &link_index, &index_result));
    link_index_.reset(link_index);
    ResultPtr index_result_owner(index_result);
    ExpectSucceededResult(index_result_owner.get());
    ASSERT_GT(loomc_link_index_symbol_count(link_index_.get()),
              kLibraryRoots.size());
    library_bytecode.reset();

    loomc_linker_t* linker = nullptr;
    LOOMC_ASSERT_OK(loomc_linker_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &linker));
    linker_.reset(linker);
    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context_.get(), nullptr,
                                          loomc_allocator_system(), &compiler));
    compiler_.reset(compiler);
  }

  void LinkProduct(const ProductSpec& product, WorkspacePtr* out_workspace,
                   ModulePtr* out_module) {
    out_workspace->reset();
    out_module->reset();
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    out_workspace->reset(workspace);

    const loomc_string_view_t root = loomc_make_cstring_view(product.root);
    const loomc_link_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.link_index=*/link_index_.get(),
        /*.module_name=*/loomc_make_cstring_view(product.name),
        /*.mode=*/LOOMC_LINK_MODE_LINK,
        /*.root_symbols=*/&root,
        /*.root_symbol_count=*/1,
        /*.flags=*/LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS,
        /*.config=*/{},
        /*.root_provider_ordinals=*/nullptr,
        /*.root_provider_count=*/0,
        /*.module_providers=*/nullptr,
        /*.module_provider_count=*/0,
    };
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_module(linker_.get(), out_workspace->get(),
                                      &options, &module, &result));
    ResultPtr result_owner(result);
    ExpectSucceededResult(result_owner.get());
    out_module->reset(module);
  }

  TargetProfilePtr CreateProfile(const char* target) {
    loomc_amdgpu_profile_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    options.structure_size = sizeof(options);
    options.identifier = loomc_make_cstring_view(target);
    options.identity.target = options.identifier;
    loomc_target_profile_t* profile = nullptr;
    LOOMC_EXPECT_OK(loomc_target_profile_create_amdgpu(
        environment_.get(), &options, loomc_allocator_system(), &profile));
    return TargetProfilePtr(profile);
  }

  void CompileProduct(const ProductSpec& product, loomc_workspace_t* workspace,
                      loomc_module_t* module, ResultPtr* out_result) {
    out_result->reset();
    TargetProfilePtr profile = CreateProfile(product.target);
    ASSERT_NE(profile, nullptr);
    const loomc_compile_report_options_t report_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        /*.structure_size=*/sizeof(report_options),
        /*.next=*/nullptr,
        /*.mode=*/LOOMC_COMPILE_REPORT_MODE_DETAILS,
        /*.format=*/LOOMC_COMPILE_REPORT_FORMAT_JSON,
        /*.identifier=*/loomc_string_view_empty(),
    };
    const std::string artifact_name = std::string(product.name) + ".hsaco";
    const loomc_emit_options_t emit_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        /*.structure_size=*/sizeof(emit_options),
        /*.next=*/&report_options,
        /*.artifact_format=*/
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
        /*.identifier=*/
        loomc_make_string_view(artifact_name.data(), artifact_name.size()),
        /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY |
            LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
    };
    std::vector<loomc_config_binding_t> bindings;
    bindings.reserve(product.config_count);
    for (size_t i = 0; i < product.config_count; ++i) {
      bindings.push_back({
          loomc_make_cstring_view(product.configs[i].key),
          loomc_make_cstring_view(product.configs[i].value),
      });
    }
    const loomc_config_options_t config_options = {
        /*.bindings=*/bindings.data(),
        /*.binding_count=*/bindings.size(),
        /*.json_object=*/{},
        /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
    };
    const loomc_string_view_t root = loomc_make_cstring_view(product.root);
    const loomc_compile_artifact_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.roots=*/&root,
        /*.root_count=*/1,
        /*.excluded_roots=*/nullptr,
        /*.excluded_root_count=*/0,
        /*.target_profile=*/profile.get(),
        /*.config=*/product.config_count != 0 ? &config_options : nullptr,
        /*.emit_options=*/&emit_options,
        /*.artifact_flags=*/0,
    };
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_artifact(
        compiler_.get(), workspace, /*pass_program=*/nullptr, module, &options,
        loomc_allocator_system(), &result));
    out_result->reset(result);
    ExpectSucceededResult(out_result->get());
  }

  // Target providers and emitter registry shared by every product.
  TargetEnvironmentPtr environment_;
  // Immutable descriptor and dialect context shared by import and compilation.
  ContextPtr context_;
  // Frozen one-import library index reused across product links.
  LinkIndexPtr link_index_;
  // Prepared linker reused with product-local workspaces.
  LinkerPtr linker_;
  // Prepared compiler reused with product-local modules and workspaces.
  CompilerPtr compiler_;
};

TEST_F(CxxModelLibraryTest, SpecializesIndependentHsacoProducts) {
  for (const ProductSpec& product : kProducts) {
    SCOPED_TRACE(product.name);
    WorkspacePtr workspace;
    ModulePtr module;
    LinkProduct(product, &workspace, &module);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(module, nullptr);

    loomc_source_t* text_source = nullptr;
    LOOMC_ASSERT_OK(loomc_module_serialize_text_to_source(
        module.get(), nullptr, loomc_allocator_system(), &text_source));
    SourcePtr text_source_owner(text_source);
    const std::string linked_text =
        ToString(loomc_source_contents(text_source_owner.get()));
    EXPECT_NE(linked_text.find(std::string("@") + product.root + "("),
              std::string::npos);
    for (const char* other_root : kLibraryRoots) {
      if (std::string_view(other_root) != product.root) {
        EXPECT_EQ(linked_text.find(std::string("@") + other_root + "("),
                  std::string::npos);
      }
    }
    for (size_t i = 0; i < product.config_count; ++i) {
      EXPECT_NE(linked_text.find(std::string("@") + product.configs[i].key),
                std::string::npos);
    }

    ResultPtr result;
    CompileProduct(product, workspace.get(), module.get(), &result);
    ASSERT_NE(result, nullptr);

    const loomc_artifact_t* hsaco =
        FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                     LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    ASSERT_NE(hsaco, nullptr);
    const std::string hsaco_contents = ToString(hsaco->contents);
    ASSERT_GE(hsaco_contents.size(), 4u);
    EXPECT_EQ(hsaco_contents.compare(0, 4,
                                     "\x7f"
                                     "ELF",
                                     4),
              0);

    const loomc_artifact_t* report =
        FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                     LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
    ASSERT_NE(report, nullptr);
    const std::string report_text = ToString(report->contents);
    EXPECT_NE(report_text.find(std::string("\"target_key\":\"") +
                               product.target + "\""),
              std::string::npos);
    for (size_t i = 0; i < product.config_count; ++i) {
      const std::string binding = std::string("\"key\":\"") +
                                  product.configs[i].key + "\",\"value\":\"" +
                                  product.configs[i].value + "\"";
      EXPECT_NE(report_text.find(binding), std::string::npos) << binding;
    }

    const loomc_artifact_t* listing =
        FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_TEXT, "amdgpu-assembly");
    ASSERT_NE(listing, nullptr);
    const std::string listing_text = ToString(listing->contents);
    EXPECT_NE(listing_text.find(product.root), std::string::npos);
    if (product.listing_marker != nullptr) {
      EXPECT_NE(listing_text.find(product.listing_marker), std::string::npos)
          << product.listing_marker;
    }
  }
}

}  // namespace
