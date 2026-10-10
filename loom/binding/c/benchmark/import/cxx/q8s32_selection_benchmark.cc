// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Measures closed linking and template selection from reusable bytecode
// indexes. C++ import and Loom text parsing happen during fixture setup so the
// timed region isolates the common production selector and linker path.

#include <string>
#include <string_view>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"
#include "loom/binding/c/benchmark/import/cxx/q8s32_sources.h"
#include "loomc/import/cxx.h"
#include "loomc/target/amdgpu.h"

namespace loomc::bench {
namespace {

enum class Q8S32ProviderForm {
  kImportedCxx,
  kAuthoredLoom,
};

enum class CxxImportRole {
  kRoot,
  kProvider,
};

static bool SkipOnError(benchmark::State& state, iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return false;
  }
  const std::string message = FormatStatus(status);
  iree_status_free(status);
  state.SkipWithError(message.c_str());
  return true;
}

static iree_status_t CreateSource(EmbeddedSource embedded,
                                  loomc_source_format_t format,
                                  SourcePtr* out_source) {
  out_source->reset();
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = format,
      .identifier = embedded.identifier,
      .contents = embedded.contents,
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
      .release = nullptr,
      .release_user_data = nullptr,
  };
  loomc_source_t* source = nullptr;
  IREE_RETURN_IF_ERROR(
      to_iree_status(loomc_source_create(&options, loom_allocator(), &source)));
  out_source->reset(source);
  return iree_ok_status();
}

static iree_status_t SerializeModule(const loomc_module_t* module,
                                     const char* identifier,
                                     SourcePtr* out_source) {
  out_source->reset();
  const loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
      .identifier = loomc_make_cstring_view(identifier),
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
  };
  loomc_source_t* source = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_module_serialize_to_source(
      module, &options, loom_allocator(), &source)));
  out_source->reset(source);
  return iree_ok_status();
}

class Q8S32SelectionFixture {
 public:
  explicit Q8S32SelectionFixture(Q8S32ProviderForm provider_form)
      : provider_form_(provider_form) {}

  iree_status_t Initialize() {
    loomc_target_environment_t* raw_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
        loom_allocator(), &raw_environment)));
    target_environment_.reset(raw_environment);

    loomc_amdgpu_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .identifier = loomc_make_cstring_view("gfx1151"),
    };
    profile_options.identity.target = profile_options.identifier;
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
        target_environment_.get(), &profile_options, loom_allocator(),
        &raw_profile)));
    target_profile_.reset(raw_profile);

    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .target_environment = target_environment_.get(),
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    loomc_context_t* raw_context = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_context_create(
        &context_options, loom_allocator(), &raw_context)));
    context_.reset(raw_context);

    loomc_workspace_t* raw_setup_workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &raw_setup_workspace)));
    WorkspacePtr setup_workspace(raw_setup_workspace);

    const auto* sources = loomc_cxx_benchmark_q8s32_sources_create();
    const size_t source_count = loomc_cxx_benchmark_q8s32_sources_size();
    const EmbeddedSource header =
        FindEmbeddedSource(sources, source_count, "q8s32_specialization.h");
    const EmbeddedSource root =
        FindEmbeddedSource(sources, source_count, "q8s32_specialization.cxx");
    const EmbeddedSource cxx_providers = FindEmbeddedSource(
        sources, source_count, "q8s32_specialization_providers.cxx");
    const EmbeddedSource loom_providers = FindEmbeddedSource(
        sources, source_count, "q8s32_specialization_providers.loom");
    shared_header_source_bytes_ = header.contents.data_length;
    root_main_source_bytes_ = root.contents.data_length;
    provider_main_source_bytes_ =
        provider_form_ == Q8S32ProviderForm::kImportedCxx
            ? cxx_providers.contents.data_length
            : loom_providers.contents.data_length;
    provider_admitted_source_bytes_ =
        provider_main_source_bytes_ +
        (provider_form_ == Q8S32ProviderForm::kImportedCxx
             ? shared_header_source_bytes_
             : 0);

    SourcePtr root_source;
    IREE_RETURN_IF_ERROR(
        CreateSource(header, LOOMC_SOURCE_FORMAT_UNKNOWN, &header_source_));
    IREE_RETURN_IF_ERROR(
        CreateSource(root, LOOMC_SOURCE_FORMAT_UNKNOWN, &root_source));
    ModulePtr root_module;
    IREE_RETURN_IF_ERROR(ImportCxx(setup_workspace.get(), root_source.get(),
                                   CxxImportRole::kRoot, &root_module));
    IREE_RETURN_IF_ERROR(SerializeModule(root_module.get(), "q8s32_root.loombc",
                                         &root_bytecode_source_));
    root_bytecode_bytes_ =
        loomc_source_contents(root_bytecode_source_.get()).data_length;

    ModulePtr provider_module;
    if (provider_form_ == Q8S32ProviderForm::kImportedCxx) {
      SourcePtr provider_source;
      IREE_RETURN_IF_ERROR(CreateSource(
          cxx_providers, LOOMC_SOURCE_FORMAT_UNKNOWN, &provider_source));
      IREE_RETURN_IF_ERROR(
          ImportCxx(setup_workspace.get(), provider_source.get(),
                    CxxImportRole::kProvider, &provider_module));
    } else {
      SourcePtr provider_source;
      IREE_RETURN_IF_ERROR(CreateSource(
          loom_providers, LOOMC_SOURCE_FORMAT_TEXT, &provider_source));
      IREE_RETURN_IF_ERROR(Deserialize(
          setup_workspace.get(), provider_source.get(), &provider_module));
    }
    IREE_RETURN_IF_ERROR(SerializeModule(provider_module.get(),
                                         "q8s32_providers.loombc",
                                         &provider_bytecode_source_));
    provider_bytecode_bytes_ =
        loomc_source_contents(provider_bytecode_source_.get()).data_length;

    IREE_RETURN_IF_ERROR(BuildIndex());
    loomc_linker_t* raw_linker = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_linker_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &raw_linker)));
    linker_.reset(raw_linker);

    loomc_workspace_t* raw_workspace = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
        /*options=*/nullptr, loom_allocator(), &raw_workspace)));
    workspace_.reset(raw_workspace);
    return iree_ok_status();
  }

  iree_status_t Link(ModulePtr* out_module, ResultPtr* out_result) const {
    out_module->reset();
    out_result->reset();
    const loomc_string_view_t root_symbol =
        loomc_make_cstring_view("@q8s32_specialize");
    const loomc_target_specialization_t specialization = {
        .function_symbol = root_symbol,
        .target_profile = target_profile_.get(),
    };
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .specializations = &specialization,
        .specialization_count = 1,
        .target_bindings = nullptr,
        .target_binding_count = 0,
    };
    const loomc_config_binding_t input_capacity = {
        .key = loomc_make_cstring_view("model.q8s32.input_capacity"),
        .value = loomc_make_cstring_view("1280"),
    };
    const loomc_link_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .link_index = link_index_.get(),
        .module_name = loomc_make_cstring_view("q8s32_selected"),
        .mode = LOOMC_LINK_MODE_LINK,
        .root_symbols = &root_symbol,
        .root_symbol_count = 1,
        .flags = LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS,
        .config =
            {
                .bindings = &input_capacity,
                .binding_count = 1,
                .json_object = {},
                .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
        .root_provider_ordinals = nullptr,
        .root_provider_count = 0,
    };
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_module(
        linker_.get(), workspace_.get(), &options, &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "Q8S32 closed linking"));
    if (!module) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Q8S32 closed linking produced no module");
    }
    out_module->reset(module.release());
    out_result->reset(result.release());
    return iree_ok_status();
  }

  iree_status_t MeasureSelectedBytecode(const loomc_module_t* module) {
    SourcePtr source;
    IREE_RETURN_IF_ERROR(
        SerializeModule(module, "q8s32_selected.loombc", &source));
    selected_bytecode_bytes_ = loomc_source_contents(source.get()).data_length;
    return iree_ok_status();
  }

  loomc_workspace_t* workspace() const { return workspace_.get(); }
  size_t shared_header_source_bytes() const {
    return shared_header_source_bytes_;
  }
  size_t root_main_source_bytes() const { return root_main_source_bytes_; }
  size_t root_bytecode_bytes() const { return root_bytecode_bytes_; }
  size_t provider_main_source_bytes() const {
    return provider_main_source_bytes_;
  }
  size_t provider_admitted_source_bytes() const {
    return provider_admitted_source_bytes_;
  }
  size_t provider_bytecode_bytes() const { return provider_bytecode_bytes_; }
  size_t selected_bytecode_bytes() const { return selected_bytecode_bytes_; }
  loomc_host_size_t indexed_symbol_count() const {
    return loomc_link_index_symbol_count(link_index_.get());
  }

 private:
  static loomc_status_t ProvideHeader(void* user_data, loomc_string_view_t path,
                                      loomc_source_t** out_source) {
    auto& self = *static_cast<Q8S32SelectionFixture*>(user_data);
    *out_source = nullptr;
    std::string_view candidate(path.data, path.size);
    const size_t separator = candidate.find_last_of('/');
    if (separator != std::string_view::npos) {
      candidate.remove_prefix(separator + 1);
    }
    const loomc_string_view_t identifier =
        loomc_source_identifier(self.header_source_.get());
    if (candidate == std::string_view(identifier.data, identifier.size)) {
      loomc_source_retain(self.header_source_.get());
      *out_source = self.header_source_.get();
    }
    return loomc_ok_status();
  }

  iree_status_t ImportCxx(loomc_workspace_t* workspace,
                          const loomc_source_t* source, CxxImportRole role,
                          ModulePtr* out_module) {
    out_module->reset();
    const loomc_string_view_t root =
        loomc_make_cstring_view("q8s32_specialize");
    loomc_cxx_import_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS,
        .structure_size = sizeof(options),
        .source_provider = {ProvideHeader, this},
        .roots = role == CxxImportRole::kRoot ? &root : nullptr,
    };
    options.root_count = role == CxxImportRole::kRoot ? 1 : 0;
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_module_import_cxx(context_.get(), workspace, source, &options,
                                loom_allocator(), &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "Q8S32 C++ import"));
    out_module->reset(module.release());
    return iree_ok_status();
  }

  iree_status_t Deserialize(loomc_workspace_t* workspace,
                            const loomc_source_t* source,
                            ModulePtr* out_module) const {
    out_module->reset();
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_module_deserialize_from_source(
        context_.get(), workspace, source, /*options=*/nullptr,
        loom_allocator(), &raw_module, &raw_result));
    ModulePtr module(raw_module);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "Q8S32 Loom parsing"));
    out_module->reset(module.release());
    return iree_ok_status();
  }

  iree_status_t BuildIndex() {
    loomc_link_index_builder_t* raw_builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &raw_builder)));
    LinkIndexBuilderPtr builder(raw_builder);
    IREE_RETURN_IF_ERROR(
        AddSourceToIndex(builder.get(), root_bytecode_source_.get(),
                         "q8s32-root", LOOMC_LINK_PROVIDER_ROLE_INPUT));
    IREE_RETURN_IF_ERROR(
        AddSourceToIndex(builder.get(), provider_bytecode_source_.get(),
                         "q8s32-providers", LOOMC_LINK_PROVIDER_ROLE_LIBRARY));
    loomc_link_index_t* raw_index = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_index_builder_finish(
        builder.get(), &raw_index, &raw_result));
    LinkIndexPtr index(raw_index);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "Q8S32 index construction"));
    link_index_.reset(index.release());
    return iree_ok_status();
  }

  Q8S32ProviderForm provider_form_;
  size_t shared_header_source_bytes_ = 0;
  size_t root_main_source_bytes_ = 0;
  size_t root_bytecode_bytes_ = 0;
  size_t provider_main_source_bytes_ = 0;
  size_t provider_admitted_source_bytes_ = 0;
  size_t provider_bytecode_bytes_ = 0;
  size_t selected_bytecode_bytes_ = 0;
  TargetEnvironmentPtr target_environment_;
  TargetProfilePtr target_profile_;
  ContextPtr context_;
  SourcePtr header_source_;
  SourcePtr root_bytecode_source_;
  SourcePtr provider_bytecode_source_;
  LinkIndexPtr link_index_;
  LinkerPtr linker_;
  WorkspacePtr workspace_;
};

static void SetWorkspaceCounters(
    benchmark::State& state, const loomc_workspace_statistics_t& cold_before,
    const loomc_workspace_statistics_t& cold_after,
    const loomc_workspace_statistics_t& timed_before,
    const loomc_workspace_statistics_t& timed_after) {
  state.counters["workspace_block_size"] =
      static_cast<double>(timed_after.total_block_size);
  state.counters["cold_workspace_block_allocations"] =
      static_cast<double>(cold_after.block_system_allocation_count -
                          cold_before.block_system_allocation_count);
  state.counters["cold_workspace_block_bytes"] =
      static_cast<double>(cold_after.block_system_allocation_bytes -
                          cold_before.block_system_allocation_bytes);
  state.counters["timed_workspace_block_allocations"] =
      static_cast<double>(timed_after.block_system_allocation_count -
                          timed_before.block_system_allocation_count);
  state.counters["timed_workspace_block_bytes"] =
      static_cast<double>(timed_after.block_system_allocation_bytes -
                          timed_before.block_system_allocation_bytes);
  state.counters["workspace_oversized_allocations"] =
      static_cast<double>(timed_after.oversized_allocation_count);
}

static void SelectQ8S32Providers(benchmark::State& state,
                                 Q8S32ProviderForm provider_form) {
  Q8S32SelectionFixture fixture(provider_form);
  if (SkipOnError(state, fixture.Initialize())) {
    return;
  }

  loomc_workspace_statistics_t cold_before = {};
  loomc_workspace_query_statistics(fixture.workspace(), &cold_before);
  ModulePtr module;
  ResultPtr result;
  if (SkipOnError(state, fixture.Link(&module, &result))) {
    return;
  }
  loomc_workspace_statistics_t cold_after = {};
  loomc_workspace_query_statistics(fixture.workspace(), &cold_after);
  if (SkipOnError(state, fixture.MeasureSelectedBytecode(module.get()))) {
    return;
  }
  result.reset();
  module.reset();
  loomc_workspace_trim(fixture.workspace());

  if (SkipOnError(state, fixture.Link(&module, &result))) {
    return;
  }
  result.reset();
  module.reset();

  loomc_workspace_statistics_t timed_before = {};
  loomc_workspace_query_statistics(fixture.workspace(), &timed_before);
  int64_t link_count = 0;
  for (auto _ : state) {
    if (SkipOnError(state, fixture.Link(&module, &result))) {
      break;
    }
    benchmark::DoNotOptimize(module.get());
    ++link_count;
    state.PauseTiming();
    result.reset();
    module.reset();
    state.ResumeTiming();
  }
  loomc_workspace_statistics_t timed_after = {};
  loomc_workspace_query_statistics(fixture.workspace(), &timed_after);

  state.counters["indexed_symbols"] =
      static_cast<double>(fixture.indexed_symbol_count());
  state.counters["shared_header_source_bytes"] =
      static_cast<double>(fixture.shared_header_source_bytes());
  state.counters["root_main_source_bytes"] =
      static_cast<double>(fixture.root_main_source_bytes());
  state.counters["root_bytecode_bytes"] =
      static_cast<double>(fixture.root_bytecode_bytes());
  state.counters["provider_main_source_bytes"] =
      static_cast<double>(fixture.provider_main_source_bytes());
  state.counters["provider_admitted_source_bytes"] =
      static_cast<double>(fixture.provider_admitted_source_bytes());
  state.counters["provider_bytecode_bytes"] =
      static_cast<double>(fixture.provider_bytecode_bytes());
  state.counters["selected_bytecode_bytes"] =
      static_cast<double>(fixture.selected_bytecode_bytes());
  state.counters["links/s"] =
      benchmark::Counter(link_count, benchmark::Counter::kIsRate);
  SetWorkspaceCounters(state, cold_before, cold_after, timed_before,
                       timed_after);
  state.SetItemsProcessed(link_count);
}

static void SelectQ8S32CxxProviders(benchmark::State& state) {
  SelectQ8S32Providers(state, Q8S32ProviderForm::kImportedCxx);
}

static void SelectQ8S32AuthoredLoomProviders(benchmark::State& state) {
  SelectQ8S32Providers(state, Q8S32ProviderForm::kAuthoredLoom);
}

BENCHMARK(SelectQ8S32CxxProviders)->Unit(benchmark::kMicrosecond);
BENCHMARK(SelectQ8S32AuthoredLoomProviders)->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace loomc::bench
