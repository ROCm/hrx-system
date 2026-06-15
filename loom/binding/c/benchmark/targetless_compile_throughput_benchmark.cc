// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

namespace {

using loomc::bench::AddSourceToIndex;
using loomc::bench::CompileScenario;
using loomc::bench::CreateTextSource;
using loomc::bench::DeserializeSource;
using loomc::bench::LinkerPtr;
using loomc::bench::LinkIndexBuilderPtr;
using loomc::bench::LinkIndexPtr;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::RunCompileBenchmark;
using loomc::bench::SourcePtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateModuleBytecodeArtifact;
using loomc::bench::WorkspacePtr;

class TunerFlowScenario final : public CompileScenario {
 public:
  explicit TunerFlowScenario(iree_host_size_t job_count)
      : job_count_(job_count) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(CompileScenario::SetUp(worker_count));
    return CreateTextSource("tuner_kernel.loom", source_text_, &source_);
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);

    ModulePtr module;
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(), workspace.get(),
                                           source_.get(), &module));

    char tile_m_value[16] = {0};
    char tile_n_value[16] = {0};
    char unroll_value[16] = {0};
    std::snprintf(tile_m_value, sizeof(tile_m_value), "%d",
                  16 + (int)(job_ordinal % 16) * 16);
    std::snprintf(tile_n_value, sizeof(tile_n_value), "%d",
                  16 + (int)((job_ordinal / 16) % 16) * 16);
    std::snprintf(unroll_value, sizeof(unroll_value), "%d",
                  1 + (int)(job_ordinal % 8));
    loomc_config_binding_t bindings[] = {
        {
            /*.key=*/loomc_make_cstring_view("@tuner.tile_m"),
            /*.value=*/loomc_make_cstring_view(tile_m_value),
        },
        {
            /*.key=*/loomc_make_cstring_view("@tuner.tile_n"),
            /*.value=*/loomc_make_cstring_view(tile_n_value),
        },
        {
            /*.key=*/loomc_make_cstring_view("@tuner.unroll"),
            /*.value=*/loomc_make_cstring_view(unroll_value),
        },
    };
    loomc_compile_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.module_name=*/loomc_make_cstring_view("tuner_kernel"),
        /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
        /*.config=*/
        {
            /*.bindings=*/bindings,
            /*.binding_count=*/IREE_ARRAYSIZE(bindings),
            /*.json_object=*/
            loomc_make_cstring_view("{\"@tuner.model.hidden_size\":\"4096\"}"),
            /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        },
    };

    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
        &options, loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(result.get(), "compilation"));

    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(
        ValidateModuleBytecodeArtifact(result.get(), &artifact_bytes));
    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

 private:
  static const char* source_text_;

  iree_host_size_t job_count_ = 0;
  SourcePtr source_;
};

const char* TunerFlowScenario::source_text_ =
    "config.decl @tuner.model.hidden_size : %value: index where [range(%value, "
    "0, 8192), mul(%value, 16)]\n"
    "config.decl @tuner.tile_m : %value: index where [range(%value, 0, 512), "
    "mul(%value, 16)]\n"
    "config.decl @tuner.tile_n : %value: index where [range(%value, 0, 512), "
    "mul(%value, 16)]\n"
    "config.decl @tuner.unroll : %value: index where [range(%value, 1, 16)]\n"
    "func.def public @entry() -> (index) {\n"
    "  %hidden = config.get @tuner.model.hidden_size : index\n"
    "  %tile_m = config.get @tuner.tile_m : index\n"
    "  %tile_n = config.get @tuner.tile_n : index\n"
    "  %unroll = config.get @tuner.unroll : index\n"
    "  %sum0 = index.add %hidden, %tile_m : index\n"
    "  %sum1 = index.add %sum0, %tile_n : index\n"
    "  %scaled = index.mul %sum1, %unroll : index\n"
    "  func.return %scaled : index\n"
    "}\n";

class ModelFlowScenario final : public CompileScenario {
 public:
  ModelFlowScenario(iree_host_size_t kernel_count,
                    iree_host_size_t repeat_count)
      : kernel_count_(kernel_count), repeat_count_(repeat_count) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(CompileScenario::SetUp(worker_count));

    loomc_linker_t* linker = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_linker_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &linker)));
    linker_.reset(linker);

    IREE_RETURN_IF_ERROR(CreateModelSources());
    IREE_RETURN_IF_ERROR(BuildLinkIndex());
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override {
    return kernel_count_ * repeat_count_;
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);

    iree_host_size_t kernel_ordinal = job_ordinal % kernel_count_;
    iree_host_size_t repeat_ordinal = job_ordinal / kernel_count_;

    char hidden_size_value[16] = {0};
    char bias_value[16] = {0};
    std::snprintf(hidden_size_value, sizeof(hidden_size_value), "%d",
                  1024 + (int)(repeat_ordinal % 8) * 16);
    std::snprintf(bias_value, sizeof(bias_value), "%d",
                  (int)(kernel_ordinal * 4));

    loomc_config_binding_t bindings[] = {
        {
            /*.key=*/
            loomc_make_string_view(config_keys_[kernel_ordinal].data(),
                                   config_keys_[kernel_ordinal].size()),
            /*.value=*/loomc_make_cstring_view(hidden_size_value),
        },
        {
            /*.key=*/
            loomc_make_string_view(bias_keys_[kernel_ordinal].data(),
                                   bias_keys_[kernel_ordinal].size()),
            /*.value=*/loomc_make_cstring_view(bias_value),
        },
    };
    loomc_string_view_t root_symbol =
        loomc_make_string_view(root_symbols_[kernel_ordinal].data(),
                               root_symbols_[kernel_ordinal].size());
    loomc_link_options_t link_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
        /*.structure_size=*/sizeof(link_options),
        /*.next=*/nullptr,
        /*.link_index=*/link_index_.get(),
        /*.module_name=*/loomc_make_cstring_view("model_kernel"),
        /*.root_symbols=*/&root_symbol,
        /*.root_symbol_count=*/1,
        /*.flags=*/0,
        /*.config=*/
        {
            /*.bindings=*/bindings,
            /*.binding_count=*/IREE_ARRAYSIZE(bindings),
            /*.json_object=*/loomc_string_view_empty(),
            /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
                LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
        },
    };

    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_link_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_link_module(linker_.get(), workspace.get(), &link_options,
                          &raw_module, &raw_link_result));
    ModulePtr module(raw_module);
    ResultPtr link_result(raw_link_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(link_result.get(), "linking"));

    loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/nullptr,
        /*.module_name=*/loomc_make_cstring_view("model_kernel"),
        /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
        /*.config=*/
        {
            /*.bindings=*/nullptr,
            /*.binding_count=*/0,
            /*.json_object=*/loomc_string_view_empty(),
            /*.flags=*/0,
        },
    };

    loomc_result_t* raw_compile_result = nullptr;
    status = to_iree_status(loomc_compile_module(
        compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
        &compile_options, loom_allocator(), &raw_compile_result));
    ResultPtr compile_result(raw_compile_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(compile_result.get(), "compilation"));

    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(
        ValidateModuleBytecodeArtifact(compile_result.get(), &artifact_bytes));
    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

 private:
  iree_status_t CreateModelSources() {
    source_storage_.clear();
    sources_.clear();
    provider_names_.clear();
    root_symbols_.clear();
    config_keys_.clear();
    bias_keys_.clear();

    source_storage_.reserve(kernel_count_ + 1);
    sources_.reserve(kernel_count_ + 1);
    provider_names_.reserve(kernel_count_ + 1);
    root_symbols_.reserve(kernel_count_);
    config_keys_.reserve(kernel_count_);
    bias_keys_.reserve(kernel_count_);

    for (iree_host_size_t i = 0; i < kernel_count_; ++i) {
      std::ostringstream root_name;
      root_name << "@caller_" << i;
      root_symbols_.push_back(root_name.str());

      std::ostringstream config_key;
      config_key << "@model.hidden_size_" << i;
      config_keys_.push_back(config_key.str());

      std::ostringstream bias_key;
      bias_key << "@model.bias_" << i;
      bias_keys_.push_back(bias_key.str());

      std::ostringstream source;
      source << "config.decl " << config_keys_.back()
             << " : %value: index where [range(%value, 0, 8192), "
                "mul(%value, 16)]\n";
      source << "config.decl " << bias_keys_.back()
             << " : %value: index where [range(%value, 0, 8192)]\n";
      source << "func.decl @identity(%x: index) -> (index)\n";
      source << "func.def public " << root_symbols_.back()
             << "() -> (index) {\n";
      source << "  %hidden = config.get " << config_keys_.back()
             << " : index\n";
      source << "  %bias = config.get " << bias_keys_.back() << " : index\n";
      source << "  %sum = index.add %hidden, %bias : index\n";
      source << "  %y = func.call @identity(%sum) : (index) -> (index)\n";
      source << "  func.return %y : index\n";
      source << "}\n";
      source_storage_.push_back(source.str());

      std::ostringstream identifier;
      identifier << "model_kernel_" << i << ".loom";
      SourcePtr source_ptr;
      IREE_RETURN_IF_ERROR(CreateTextSource(
          identifier.str(), source_storage_.back(), &source_ptr));
      sources_.push_back(std::move(source_ptr));

      std::ostringstream provider_name;
      provider_name << "model_kernel_" << i;
      provider_names_.push_back(provider_name.str());
    }

    source_storage_.push_back(
        "func.def public @identity(%x: index) -> (index) {\n"
        "  func.return %x : index\n"
        "}\n");
    SourcePtr library_source;
    IREE_RETURN_IF_ERROR(CreateTextSource(
        "model_library.loom", source_storage_.back(), &library_source));
    sources_.push_back(std::move(library_source));
    provider_names_.push_back("model_library");
    return iree_ok_status();
  }

  iree_status_t BuildLinkIndex() {
    loomc_link_index_builder_t* builder = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_link_index_builder_create(
        context_.get(), /*options=*/nullptr, loom_allocator(), &builder)));
    LinkIndexBuilderPtr builder_ptr(builder);

    for (iree_host_size_t i = 0; i < kernel_count_; ++i) {
      IREE_RETURN_IF_ERROR(
          AddSourceToIndex(builder_ptr.get(), sources_[i].get(),
                           provider_names_[i], LOOMC_LINK_PROVIDER_ROLE_INPUT));
    }
    IREE_RETURN_IF_ERROR(AddSourceToIndex(
        builder_ptr.get(), sources_.back().get(), provider_names_.back(),
        LOOMC_LINK_PROVIDER_ROLE_LIBRARY));

    loomc_link_index_t* link_index = nullptr;
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(loomc_link_index_builder_finish(
        builder_ptr.get(), &link_index, &raw_result));
    LinkIndexPtr link_index_ptr(link_index);
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(
        RequireSucceededResult(result.get(), "link index construction"));
    link_index_.reset(link_index_ptr.release());
    return iree_ok_status();
  }

  iree_host_size_t kernel_count_ = 0;
  iree_host_size_t repeat_count_ = 0;
  LinkerPtr linker_;
  LinkIndexPtr link_index_;
  std::vector<std::string> source_storage_;
  std::vector<SourcePtr> sources_;
  std::vector<std::string> provider_names_;
  std::vector<std::string> root_symbols_;
  std::vector<std::string> config_keys_;
  std::vector<std::string> bias_keys_;
};

static std::unique_ptr<CompileScenario> CreateTunerFlowScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<TunerFlowScenario>((iree_host_size_t)state.range(1));
}

static void BM_TunerFlowSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateTunerFlowScenario, nullptr);
}
BENCHMARK(BM_TunerFlowSmoke)->Args({1, 2})->Args({2, 4})->UseRealTime();

static void BM_TunerFlow(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateTunerFlowScenario, nullptr);
}
BENCHMARK(BM_TunerFlow)
    ->Args({1, 64})
    ->Args({2, 128})
    ->Args({4, 256})
    ->Args({8, 512})
    ->Args({16, 1024})
    ->Args({32, 2048})
    ->Args({64, 4096})
    ->Args({96, 6144})
    ->UseRealTime();

static std::unique_ptr<CompileScenario> CreateModelFlowScenario(
    const ::benchmark::State& state, void* user_data) {
  (void)user_data;
  return std::make_unique<ModelFlowScenario>((iree_host_size_t)state.range(1),
                                             (iree_host_size_t)state.range(2));
}

static void BM_ModelFlowSmoke(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateModelFlowScenario, nullptr);
}
BENCHMARK(BM_ModelFlowSmoke)->Args({1, 4, 1})->Args({2, 4, 1})->UseRealTime();

static void BM_ModelFlow(::benchmark::State& state) {
  RunCompileBenchmark(state, CreateModelFlowScenario, nullptr);
}
BENCHMARK(BM_ModelFlow)
    ->Args({1, 32, 10})
    ->Args({2, 32, 10})
    ->Args({4, 32, 10})
    ->Args({8, 32, 10})
    ->Args({16, 32, 10})
    ->Args({32, 32, 10})
    ->Args({64, 32, 10})
    ->Args({96, 32, 10})
    ->UseRealTime();

}  // namespace
