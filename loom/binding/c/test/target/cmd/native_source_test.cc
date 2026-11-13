// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <barrier>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/cmd/native_source_testdata.h"
#include "loomc/target/cmd.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using IndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using CompilerPtr = HandlePtr<loomc_cmd_compiler_t, loomc_cmd_compiler_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;

// Observes the compiler owner's real allocation boundary. Invocation results
// and workspaces use their ordinary independent allocator.
class CompilerAllocator {
 public:
  loomc_allocator_t allocator() { return {this, Control}; }

  size_t live_bytes() {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_bytes_;
  }

  size_t allocations() {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_;
  }

  // Both workers must reach their private allocation before either source can
  // complete. Readiness comes from allocation entry, never elapsed time.
  void JoinNextAllocations(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    arrivals_remaining_ = count;
  }

  void FailAfter(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_countdown_ = count;
  }

  // Pause retirement of an existing retained allocation, excluding construction
  // scratch allocated after this snapshot. Other compiler calls remain live.
  void PauseRetirement() {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_allocations_ = live_;
    retirement_waiting_ = false;
    retirement_released_ = false;
  }

  void WaitForRetirement() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return retirement_waiting_; });
  }

  void ResumeRetirement() {
    std::lock_guard<std::mutex> lock(mutex_);
    retirement_released_ = true;
    condition_.notify_all();
  }

 private:
  static loomc_status_t Control(void* self, loomc_allocator_command_t command,
                                const void* params, void** inout_ptr) {
    auto* tracker = static_cast<CompilerAllocator*>(self);
    std::unique_lock<std::mutex> lock(tracker->mutex_);
    if (command == LOOMC_ALLOCATOR_COMMAND_TRIM) {
      const auto backing = loomc_allocator_system();
      return backing.ctl(backing.self, command, params, inout_ptr);
    }
    if (command == LOOMC_ALLOCATOR_COMMAND_FREE &&
        tracker->paused_allocations_.contains(*inout_ptr)) {
      tracker->paused_allocations_.clear();
      tracker->retirement_waiting_ = true;
      tracker->condition_.notify_all();
      tracker->condition_.wait(lock,
                               [&] { return tracker->retirement_released_; });
    }
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE &&
        tracker->failure_countdown_ != LOOMC_HOST_SIZE_MAX) {
      if (tracker->failure_countdown_ == 0) {
        tracker->failure_countdown_ = LOOMC_HOST_SIZE_MAX;
        return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                                 "source preparation allocation failed");
      }
      --tracker->failure_countdown_;
    }
    void* original = *inout_ptr;
    const auto backing = loomc_allocator_system();
    loomc_status_t status =
        backing.ctl(backing.self, command, params, inout_ptr);
    if (!loomc_status_is_ok(status)) {
      return status;
    }
    if (original != nullptr) {
      const auto found = tracker->live_.find(original);
      EXPECT_NE(found, tracker->live_.end());
      if (found != tracker->live_.end()) {
        tracker->live_bytes_ -= found->second;
        tracker->live_.erase(found);
      }
    }
    if (command != LOOMC_ALLOCATOR_COMMAND_FREE) {
      const auto* request =
          static_cast<const loomc_allocator_alloc_params_t*>(params);
      tracker->live_[*inout_ptr] = request->byte_length;
      tracker->live_bytes_ += request->byte_length;
      ++tracker->allocations_;
      if (tracker->arrivals_remaining_ != 0) {
        --tracker->arrivals_remaining_;
        tracker->condition_.notify_all();
        tracker->condition_.wait(
            lock, [&] { return tracker->arrivals_remaining_ == 0; });
      }
    }
    return status;
  }

  // Serializes accounting and the optional two-builder readiness barrier.
  std::mutex mutex_;
  // Completion notification for the allocation-entry barrier.
  std::condition_variable condition_;
  // Requested length of every live owner allocation.
  std::unordered_map<void*, size_t> live_;
  // Sum of requested lengths in live_.
  size_t live_bytes_ = 0;
  // Number of successful allocating calls, including reallocations.
  size_t allocations_ = 0;
  // Allocations that must enter before joined builders resume.
  size_t arrivals_remaining_ = 0;
  // Allocations preceding a single injected failure, or MAX when disabled.
  size_t failure_countdown_ = LOOMC_HOST_SIZE_MAX;
  // Allocations live when retirement was armed; later scratch is excluded.
  std::unordered_map<void*, size_t> paused_allocations_;
  // True when an actual retained allocation has entered destruction.
  bool retirement_waiting_ = false;
  // Allows the paused deallocation to complete after competing publication.
  bool retirement_released_ = false;
};

struct Output {
  // Independent command product returned by the public operation.
  ProductPtr product;
  // Independent diagnostics returned by the public operation.
  ResultPtr result;
  // Native requests transferred synchronously by publication.
  std::vector<RequestPtr> requests;
};

loomc_status_t Capture(void* user_data, loomc_request_t* request) {
  static_cast<Output*>(user_data)->requests.emplace_back(request);
  return loomc_ok_status();
}

std::string Diagnostics(const loomc_result_t* result) {
  std::string text;
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    text.append(diagnostic->message.data, diagnostic->message.size);
    text.push_back('\n');
  }
  return text;
}

void ExpectSucceeded(const Output& output, size_t request_count = 1) {
  ASSERT_NE(output.result, nullptr);
  for (size_t i = 0; i < loomc_result_diagnostic_count(output.result.get());
       ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(output.result.get(), i);
    EXPECT_NE(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR)
        << std::string(diagnostic->message.data, diagnostic->message.size);
  }
  EXPECT_TRUE(loomc_result_succeeded(output.result.get()));
  ASSERT_NE(output.product, nullptr);
  EXPECT_EQ(output.requests.size(), request_count);
}

std::string RequestBytes(const Output& output, size_t ordinal = 0) {
  EXPECT_GT(output.requests.size(), ordinal);
  if (output.requests.size() <= ordinal) {
    return {};
  }
  const auto bytes = loomc_source_contents(
      loomc_request_source(output.requests[ordinal].get()));
  return std::string(reinterpret_cast<const char*>(bytes.data),
                     bytes.data_length);
}

TEST(CmdCompilerTest, RetainsAnEmptySourceIndex) {
  loomc_context_t* context_raw = nullptr;
  LOOMC_ASSERT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context_raw));
  ContextPtr context(context_raw);
  loomc_link_index_builder_t* builder_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
      context.get(), nullptr, loomc_allocator_system(), &builder_raw));
  BuilderPtr builder(builder_raw);
  loomc_link_index_t* index_raw = nullptr;
  loomc_result_t* result_raw = nullptr;
  LOOMC_ASSERT_OK(
      loomc_link_index_builder_finish(builder.get(), &index_raw, &result_raw));
  IndexPtr index(index_raw);
  ResultPtr result(result_raw);
  ASSERT_TRUE(loomc_result_succeeded(result.get()))
      << Diagnostics(result.get());

  CompilerAllocator allocator;
  loomc_cmd_compiler_t* compiler_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_compiler_create(
      index.get(), nullptr, allocator.allocator(), &compiler_raw));
  CompilerPtr compiler(compiler_raw);
  index.reset();
  builder.reset();
  context.reset();
  compiler.reset();
  EXPECT_EQ(allocator.live_bytes(), 0u);
}

class NativeSourceTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(nullptr, loomc_allocator_system(), &context));
    context_.reset(context);
    workspace_ = CreateWorkspace();
    loomc_link_index_builder_t* builder_raw = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
        context, nullptr, loomc_allocator_system(), &builder_raw));
    BuilderPtr builder(builder_raw);
    const auto* files = loomc_cmd_native_source_testdata_create();
    for (size_t i = 0; i < loomc_cmd_native_source_testdata_size(); ++i) {
      const loomc_source_options_t source_options = {
          .format = LOOMC_SOURCE_FORMAT_TEXT,
          .identifier = loomc_make_cstring_view(files[i].name),
          .contents = loomc_make_byte_span(files[i].data, files[i].size),
          .storage = LOOMC_SOURCE_STORAGE_BORROWED,
      };
      loomc_source_t* source_raw = nullptr;
      LOOMC_ASSERT_OK(loomc_source_create(
          &source_options, loomc_allocator_system(), &source_raw));
      SourcePtr source(source_raw);
      if (GetParam()) {
        loomc_module_t* module_raw = nullptr;
        loomc_result_t* result_raw = nullptr;
        LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
            context, workspace_.get(), source.get(), nullptr,
            loomc_allocator_system(), &module_raw, &result_raw));
        ModulePtr module(module_raw);
        ResultPtr result(result_raw);
        ASSERT_TRUE(loomc_result_succeeded(result.get()))
            << Diagnostics(result.get());
        const loomc_module_serialize_options_t options = {
            .format = LOOMC_SOURCE_FORMAT_BYTECODE,
            .identifier = loomc_make_cstring_view(files[i].name),
        };
        LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
            module.get(), &options, loomc_allocator_system(), &source_raw));
        source.reset(source_raw);
      }
      const loomc_link_index_source_options_t options = {
          .provider_name = loomc_make_cstring_view(files[i].name),
          .role = std::strcmp(files[i].name, "command.loom") == 0
                      ? LOOMC_LINK_PROVIDER_ROLE_INPUT
                      : LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
      };
      LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
          builder.get(), source.get(), &options, nullptr));
    }
    loomc_link_index_t* index = nullptr;
    loomc_result_t* result_raw = nullptr;
    LOOMC_ASSERT_OK(
        loomc_link_index_builder_finish(builder.get(), &index, &result_raw));
    index_.reset(index);
    ResultPtr result(result_raw);
    ASSERT_TRUE(loomc_result_succeeded(result.get()))
        << Diagnostics(result.get());
    ResetCompiler(64u * 1024u * 1024u);
  }

  void TearDown() override {
    compiler_.reset();
    EXPECT_EQ(allocator_.live_bytes(), 0u);
  }

 public:
  WorkspacePtr CreateWorkspace() {
    loomc_workspace_t* workspace = nullptr;
    LOOMC_EXPECT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    return WorkspacePtr(workspace);
  }

  void ResetCompiler(size_t byte_limit) {
    compiler_.reset();
    EXPECT_EQ(allocator_.live_bytes(), 0u);
    const loomc_cmd_compiler_options_t options = {
        .retained_byte_limit = byte_limit,
    };
    loomc_cmd_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_cmd_compiler_create(
        index_.get(), &options, allocator_.allocator(), &compiler));
    compiler_.reset(compiler);
  }

  loomc_status_t Build(const char* json, Output* output,
                       loomc_config_policy_flags_t flags = 0,
                       loomc_request_sink_t sink = {},
                       loomc_workspace_t* workspace = nullptr,
                       const char* root_name = "dispatch") {
    *output = {};
    loomc_link_index_symbol_t root = {};
    EXPECT_TRUE(loomc_link_index_lookup_global(
        index_.get(), loomc_make_cstring_view(root_name), &root));
    const loomc_cmd_program_product_options_t options = {
        .compiler = compiler_.get(),
        .root_symbol_ordinals = &root.ordinal,
        .root_symbol_count = 1,
        .config =
            {
                .json_object = loomc_make_cstring_view(json),
                .flags = flags,
            },
        .request_sink =
            sink.publish ? sink : loomc_request_sink_t{Capture, output},
    };
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_cmd_program_product_build(
        workspace ? workspace : workspace_.get(), &options,
        loomc_allocator_system(), &product, &result);
    output->product.reset(product);
    output->result.reset(result);
    return status;
  }

 protected:
  // Allocator state outlives the compiler's complete source/cache ownership.
  CompilerAllocator allocator_;
  // Immutable parser and dialect environment.
  ContextPtr context_;
  // Reused serial invocation scratch.
  WorkspacePtr workspace_;
  // Frozen source universe for all requests in the test.
  IndexPtr index_;
  // Real public compiler whose reuse and residency are observed.
  CompilerPtr compiler_;
};

TEST_P(NativeSourceTest, CompleteClassBudgetSurvivesReuseAndOwnerRetirement) {
  Output cold;
  LOOMC_ASSERT_OK(Build("{}", &cold, 0, {}, nullptr, "class_budget"));
  ExpectSucceeded(cold, 128);
  ASSERT_EQ(cold.requests.size(), 128u);
  Output warm;
  LOOMC_ASSERT_OK(Build("{}", &warm, 0, {}, nullptr, "class_budget"));
  ExpectSucceeded(warm, 128);
  ASSERT_EQ(warm.requests.size(), cold.requests.size());
  compiler_.reset();
  EXPECT_EQ(allocator_.live_bytes(), 0u);
  for (size_t ordinal = 0; ordinal < cold.requests.size(); ++ordinal) {
    EXPECT_EQ(RequestBytes(cold, ordinal), RequestBytes(warm, ordinal));
  }
}

TEST_P(NativeSourceTest, DeferredProvidersRetainOnlyPossibleStructure) {
  for (size_t byte_limit : {size_t{0}, size_t{64 * 1024 * 1024}}) {
    ResetCompiler(byte_limit);
    for (const char* root : {"residual_computed", "residual_guarded",
                             "residual_target", "residual_false"}) {
      SCOPED_TRACE(root);
      std::vector<std::string> previous;
      for (unsigned round = 0; round < 2; ++round) {
        Output output;
        LOOMC_ASSERT_OK(Build("{}", &output, 0, {}, nullptr, root));
        ExpectSucceeded(output,
                        std::strcmp(root, "residual_false") == 0 ? 1 : 2);
        std::vector<std::string> current;
        for (size_t ordinal = 0; ordinal < output.requests.size(); ++ordinal) {
          current.push_back(RequestBytes(output, ordinal));
        }
        if (round) {
          EXPECT_EQ(current, previous);
        }
        previous = std::move(current);
      }
    }
  }
}

TEST_P(NativeSourceTest, MatchesSuccessfulClosureAndConfigurationPolicy) {
  const char* configurations[] = {
      "{}",
      "{\"extra\":7}",
      "{\"extra\":9}",
      "{\"mode\":1,\"extra\":7}",
      "{\"mode\":1,\"extra\":9}",
      "{\"mode\":1,\"extra\":7,\"unused\":\"not an integer\"}",
      "{\"mode\":1}",
  };
  std::vector<std::string> fresh;
  ResetCompiler(0);
  for (const char* json : configurations) {
    Output output;
    LOOMC_ASSERT_OK(Build(json, &output));
    ExpectSucceeded(output);
    fresh.push_back(RequestBytes(output));
  }
  EXPECT_EQ(fresh[0], fresh[1]);
  EXPECT_EQ(fresh[0], fresh[2]);
  EXPECT_NE(fresh[3], fresh[4]);
  EXPECT_EQ(fresh[3], fresh[5]);
  EXPECT_NE(fresh[3], fresh[6]);
  ResetCompiler(64u * 1024u * 1024u);
  for (size_t round = 0; round < 2; ++round) {
    for (size_t i = 0; i < fresh.size(); ++i) {
      SCOPED_TRACE(configurations[i]);
      const auto allocations = allocator_.allocations();
      Output output;
      LOOMC_ASSERT_OK(Build(configurations[i], &output));
      ExpectSucceeded(output);
      EXPECT_EQ(RequestBytes(output), fresh[i]);
      if (round != 0 || i == 1 || i == 2 || i == 5) {
        EXPECT_EQ(allocator_.allocations(), allocations);
      }
    }
  }
  Output unresolved;
  LOOMC_ASSERT_OK(Build("{\"mode\":1}", &unresolved,
                        LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED));
  ASSERT_NE(unresolved.result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(unresolved.result.get()));
  EXPECT_EQ(unresolved.product, nullptr);
  EXPECT_TRUE(unresolved.requests.empty());
  Output resolved;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &resolved,
                        LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED));
  ExpectSucceeded(resolved);
  EXPECT_EQ(RequestBytes(resolved), fresh[3]);
}

TEST_P(NativeSourceTest,
       ReentrantPressureKeepsActiveSourceAndEvictsIdleSource) {
  const size_t owner_bytes = allocator_.live_bytes();
  Output initial;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &initial));
  ExpectSucceeded(initial);
  const size_t source_bytes = allocator_.live_bytes() - owner_bytes;
  ASSERT_GT(source_bytes, 0u);
  const auto expected = RequestBytes(initial);
  ResetCompiler(source_bytes);
  const size_t bounded_owner_bytes = allocator_.live_bytes();
  struct Reentry {
    // Fixture exposing the same compiler with independent nested scratch.
    NativeSourceTest* fixture;
    // Outer accepted request destination.
    Output* outer;
    // Nested output kept past the active source lease.
    Output nested;
  };
  Output outer;
  Reentry reentry{this, &outer, {}};
  const loomc_request_sink_t sink = {
      .publish = [](void* user_data,
                    loomc_request_t* request) -> loomc_status_t {
        auto* state = static_cast<Reentry*>(user_data);
        state->outer->requests.emplace_back(request);
        auto workspace = state->fixture->CreateWorkspace();
        return state->fixture->Build("{\"mode\":1,\"extra\":9}", &state->nested,
                                     0, {}, workspace.get());
      },
      .user_data = &reentry,
  };
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &outer, 0, sink));
  ExpectSucceeded(outer);
  ExpectSucceeded(reentry.nested);
  EXPECT_EQ(RequestBytes(outer), expected);
  EXPECT_EQ(allocator_.live_bytes(), bounded_owner_bytes + source_bytes);
  const auto allocations = allocator_.allocations();
  Output hit;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &hit));
  EXPECT_EQ(allocator_.allocations(), allocations);
  Output displaced;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":9}", &displaced));
  EXPECT_EQ(RequestBytes(displaced), RequestBytes(reentry.nested));
  const auto after_displacement = allocator_.allocations();
  Output rebuilt;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &rebuilt));
  EXPECT_GT(allocator_.allocations(), after_displacement);
  EXPECT_EQ(RequestBytes(rebuilt), expected);
  EXPECT_LE(allocator_.live_bytes(), bounded_owner_bytes + source_bytes);
  compiler_.reset();
  EXPECT_EQ(allocator_.live_bytes(), 0u);
  EXPECT_EQ(RequestBytes(outer), expected);
  EXPECT_EQ(RequestBytes(reentry.nested), RequestBytes(displaced));
}

TEST_P(NativeSourceTest, DuplicateBuildersRetainOneSourceAndReleaseLoser) {
  const size_t owner_bytes = allocator_.live_bytes();
  Output baseline;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &baseline));
  ExpectSucceeded(baseline);
  const size_t source_bytes = allocator_.live_bytes() - owner_bytes;
  ResetCompiler(source_bytes);
  const size_t fixed_bytes = allocator_.live_bytes();
  auto first_workspace = CreateWorkspace();
  auto second_workspace = CreateWorkspace();
  Output first;
  Output second;
  allocator_.JoinNextAllocations(2);
  std::thread worker([&] {
    LOOMC_EXPECT_OK(Build("{\"mode\":1,\"extra\":7}", &first, 0, {},
                          first_workspace.get()));
  });
  LOOMC_EXPECT_OK(Build("{\"mode\":1,\"extra\":7}", &second, 0, {},
                        second_workspace.get()));
  worker.join();
  ExpectSucceeded(first);
  ExpectSucceeded(second);
  EXPECT_EQ(RequestBytes(first), RequestBytes(baseline));
  EXPECT_EQ(RequestBytes(second), RequestBytes(baseline));
  EXPECT_EQ(allocator_.live_bytes(), fixed_bytes + source_bytes);
}

TEST_P(NativeSourceTest, FailedPreparationKeepsIndependentExactDiagnostics) {
  const size_t owner_bytes = allocator_.live_bytes();
  Output first;
  LOOMC_ASSERT_OK(
      Build("{\"broken_count\":2}", &first, 0, {}, nullptr, "broken_dispatch"));
  ASSERT_NE(first.result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(first.result.get()));
  EXPECT_TRUE(first.requests.empty());
  EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
  const size_t allocations = allocator_.allocations();
  Output retry;
  LOOMC_ASSERT_OK(
      Build("{\"broken_count\":2}", &retry, 0, {}, nullptr, "broken_dispatch"));
  ASSERT_NE(retry.result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(retry.result.get()));
  EXPECT_GT(allocator_.allocations(), allocations);
  EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
  Output repaired;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &repaired));
  ExpectSucceeded(repaired);
  compiler_.reset();
  index_.reset();
  workspace_.reset();
  context_.reset();
  EXPECT_EQ(allocator_.live_bytes(), 0u);
  bool found_source = false;
  for (size_t i = 0; i < loomc_result_diagnostic_count(first.result.get());
       ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(first.result.get(), i);
    EXPECT_GT(diagnostic->message.size, 0u);
    if (diagnostic->range.source != nullptr) {
      found_source = true;
      const auto identifier = loomc_source_identifier(diagnostic->range.source);
      EXPECT_EQ(std::string(identifier.data, identifier.size),
                "providers.loom-test");
      EXPECT_GT(diagnostic->range.start_line, 0u);
    }
  }
  EXPECT_TRUE(found_source);
}

TEST_P(NativeSourceTest, OversizedSourcesPublishWithoutRetention) {
  ResetCompiler(1);
  const size_t owner_bytes = allocator_.live_bytes();
  Output first;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &first));
  ExpectSucceeded(first);
  EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
  const auto allocations = allocator_.allocations();
  Output second;
  LOOMC_ASSERT_OK(Build("{\"mode\":1,\"extra\":7}", &second));
  ExpectSucceeded(second);
  EXPECT_GT(allocator_.allocations(), allocations);
  EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
  EXPECT_EQ(RequestBytes(first), RequestBytes(second));
}

TEST_P(NativeSourceTest, AllocationFailuresReleasePartialPreparations) {
  for (const char* root : {"dispatch", "residual_computed"}) {
    SCOPED_TRACE(root);
    ResetCompiler(64u * 1024u * 1024u);
    const size_t request_count = std::strcmp(root, "dispatch") == 0 ? 1 : 2;
    const size_t initial_allocations = allocator_.allocations();
    Output baseline;
    LOOMC_ASSERT_OK(
        Build("{\"mode\":1,\"extra\":7}", &baseline, 0, {}, nullptr, root));
    ExpectSucceeded(baseline, request_count);
    const size_t preparation_allocations =
        allocator_.allocations() - initial_allocations;
    ASSERT_GT(preparation_allocations, 0u);
    for (size_t i = 0; i < preparation_allocations; ++i) {
      SCOPED_TRACE(i);
      ResetCompiler(64u * 1024u * 1024u);
      const size_t owner_bytes = allocator_.live_bytes();
      allocator_.FailAfter(i);
      Output failed;
      const loomc_status_t status =
          Build("{\"mode\":1,\"extra\":7}", &failed, 0, {}, nullptr, root);
      EXPECT_EQ(loomc_status_consume_code(status),
                LOOMC_STATUS_RESOURCE_EXHAUSTED);
      EXPECT_EQ(failed.product, nullptr);
      EXPECT_TRUE(failed.requests.empty());
      EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
      Output recovered;
      LOOMC_ASSERT_OK(
          Build("{\"mode\":1,\"extra\":7}", &recovered, 0, {}, nullptr, root));
      ExpectSucceeded(recovered, request_count);
      for (size_t ordinal = 0; ordinal < request_count; ++ordinal) {
        EXPECT_EQ(RequestBytes(recovered, ordinal),
                  RequestBytes(baseline, ordinal));
      }
    }
  }
}

struct StagedRoots {
  // Staged tuple whose provider requires two structural classes.
  const char* matching;
  // Different tuple choosing one structurally generic class.
  const char* other;
};

constexpr StagedRoots kStagedRoots[] = {
    {"staged_scalar_match", "staged_scalar_other"},
    {"staged_schema_match", "staged_schema_other"},
    {"staged_storage_match", "staged_storage_other"},
    {"staged_layout_match", "staged_layout_other"},
    {"staged_vector_match", "staged_vector_other"},
    {"staged_buffer_match", "staged_buffer_other"},
    {"staged_collision_match", "staged_collision_other"},
};

TEST_P(NativeSourceTest, StagedPlansPreserveOutputAndIndependentOwnership) {
  for (const auto& roots : kStagedRoots) {
    SCOPED_TRACE(roots.matching);
    std::array<std::vector<std::string>, 2> expected;
    for (size_t byte_limit : {size_t{0}, size_t{1}, size_t{64 * 1024 * 1024}}) {
      ResetCompiler(byte_limit);
      const size_t owner_bytes = allocator_.live_bytes();
      for (unsigned round = 0; round < 2; ++round) {
        const size_t allocations = allocator_.allocations();
        for (unsigned variant = 0; variant < 2; ++variant) {
          Output output;
          LOOMC_ASSERT_OK(Build("{}", &output, 0, {}, nullptr,
                                variant ? roots.other : roots.matching));
          ExpectSucceeded(output, variant ? 1 : 2);
          std::vector<std::string> bytes;
          for (size_t i = 0; i < output.requests.size(); ++i) {
            bytes.push_back(RequestBytes(output, i));
          }
          if (byte_limit == 0 && round == 0) {
            expected[variant] = bytes;
          } else {
            EXPECT_EQ(bytes, expected[variant]);
          }
        }
        if (byte_limit <= 1) {
          EXPECT_EQ(allocator_.live_bytes(), owner_bytes);
        } else if (round != 0) {
          EXPECT_EQ(allocator_.allocations(), allocations);
        }
      }
      Output surviving;
      LOOMC_ASSERT_OK(Build("{}", &surviving, 0, {}, nullptr, roots.matching));
      compiler_.reset();
      EXPECT_EQ(allocator_.live_bytes(), 0u);
      ASSERT_EQ(surviving.requests.size(), expected[0].size());
      for (size_t i = 0; i < surviving.requests.size(); ++i) {
        EXPECT_EQ(RequestBytes(surviving, i), expected[0][i]);
      }
    }
  }
}

TEST_P(NativeSourceTest, RequiredStagedProvidersCloseBeforePublication) {
  for (size_t limit : {size_t{0}, size_t{64 * 1024 * 1024}}) {
    ResetCompiler(limit);
    Output first;
    LOOMC_ASSERT_OK(
        Build("{}", &first, 0, {}, nullptr, "staged_required_interleaved"));
    ExpectSucceeded(first, 4);
    Output rejected;
    EXPECT_EQ(loomc_status_consume_code(Build("{}", &rejected, 0, {}, nullptr,
                                              "staged_required_invalid")),
              LOOMC_STATUS_NOT_FOUND);
    EXPECT_EQ(rejected.product, nullptr);
    EXPECT_TRUE(rejected.requests.empty());
    Output again;
    LOOMC_ASSERT_OK(
        Build("{}", &again, 0, {}, nullptr, "staged_required_interleaved"));
    ExpectSucceeded(again, 4);
    compiler_.reset();
    EXPECT_EQ(allocator_.live_bytes(), 0u);
    for (size_t i = 0; i < first.requests.size(); ++i) {
      EXPECT_EQ(RequestBytes(first, i), RequestBytes(again, i));
    }
  }
}

TEST_P(NativeSourceTest, PlanLeasesSurviveCallbacksAndDetachedDestruction) {
  // Exercise shared sources, nested payloads, a real hash collision, and two
  // independent source owners competing for the same byte budget.
  const StagedRoots cases[] = {
      kStagedRoots[0],
      kStagedRoots[2],
      kStagedRoots[6],
      {"staged_scalar_match", "staged_layout_other"},
  };
  for (const auto& roots : cases) {
    SCOPED_TRACE(roots.matching);
    size_t one_entry = 0;
    std::array<std::vector<std::string>, 2> expected;
    for (unsigned variant = 0; variant < 2; ++variant) {
      ResetCompiler(64 * 1024 * 1024);
      const size_t owner_bytes = allocator_.live_bytes();
      Output output;
      LOOMC_ASSERT_OK(Build("{}", &output, 0, {}, nullptr,
                            variant ? roots.other : roots.matching));
      ExpectSucceeded(output, variant ? 1 : 2);
      one_entry = std::max(one_entry, allocator_.live_bytes() - owner_bytes);
      for (size_t i = 0; i < output.requests.size(); ++i) {
        expected[variant].push_back(RequestBytes(output, i));
      }
    }
    ResetCompiler(one_entry);
    const size_t owner_bytes = allocator_.live_bytes();
    Output first;
    LOOMC_ASSERT_OK(Build("{}", &first, 0, {}, nullptr, roots.matching));
    std::barrier<> gate(2);
    struct HeldCallback {
      // Explicit callback entry/release rendezvous with the competing caller.
      std::barrier<>* gate;
      // Transferred requests kept past the callback and source lifetime.
      Output* output;
    };
    Output held_output;
    HeldCallback held{&gate, &held_output};
    const loomc_request_sink_t sink = {
        .publish = [](void* user_data,
                      loomc_request_t* request) -> loomc_status_t {
          auto* held = static_cast<HeldCallback*>(user_data);
          if (held->output->requests.empty()) {
            held->gate->arrive_and_wait();
            held->gate->arrive_and_wait();
          }
          return Capture(held->output, request);
        },
        .user_data = &held,
    };
    auto workspace = CreateWorkspace();
    std::thread worker([&] {
      LOOMC_EXPECT_OK(
          Build("{}", &held_output, 0, sink, workspace.get(), roots.matching));
    });
    gate.arrive_and_wait();
    Output competing;
    LOOMC_EXPECT_OK(Build("{}", &competing, 0, {}, nullptr, roots.other));
    ExpectSucceeded(competing);
    const auto calls = allocator_.allocations();
    Output hit;
    LOOMC_EXPECT_OK(Build("{}", &hit, 0, {}, nullptr, roots.matching));
    const bool reused = allocator_.allocations() == calls;
    gate.arrive_and_wait();
    worker.join();
    // Establish actual retention before waiting for its retirement below.
    ASSERT_TRUE(reused);
    ExpectSucceeded(held_output, 2);
    EXPECT_LE(allocator_.live_bytes(), owner_bytes + one_entry);

    allocator_.PauseRetirement();
    std::thread retiring([&] {
      LOOMC_EXPECT_OK(
          Build("{}", &competing, 0, {}, workspace.get(), roots.other));
    });
    allocator_.WaitForRetirement();
    LOOMC_EXPECT_OK(Build("{}", &hit, 0, {}, nullptr, roots.matching));
    ExpectSucceeded(hit, 2);
    allocator_.ResumeRetirement();
    retiring.join();
    ExpectSucceeded(competing);
    EXPECT_LE(allocator_.live_bytes(), owner_bytes + one_entry);
    compiler_.reset();
    EXPECT_EQ(allocator_.live_bytes(), 0u);
    for (size_t i = 0; i < expected[0].size(); ++i) {
      EXPECT_EQ(RequestBytes(held_output, i), expected[0][i]);
      EXPECT_EQ(RequestBytes(hit, i), expected[0][i]);
    }
    EXPECT_EQ(RequestBytes(competing), expected[1][0]);
  }
}

TEST_P(NativeSourceTest, ConcurrentPlanBuildersRetainOneWinner) {
  for (const auto& roots : kStagedRoots) {
    SCOPED_TRACE(roots.matching);
    ResetCompiler(64 * 1024 * 1024);
    Output matching;
    LOOMC_ASSERT_OK(Build("{}", &matching, 0, {}, nullptr, roots.matching));
    Output baseline;
    LOOMC_ASSERT_OK(Build("{}", &baseline, 0, {}, nullptr, roots.other));
    const size_t expected_bytes = allocator_.live_bytes();
    ResetCompiler(64 * 1024 * 1024);
    LOOMC_ASSERT_OK(Build("{}", &matching, 0, {}, nullptr, roots.matching));
    allocator_.JoinNextAllocations(4);
    std::array<Output, 4> results;
    std::vector<std::thread> workers;
    for (auto& result : results) {
      workers.emplace_back([&, output = &result] {
        auto workspace = CreateWorkspace();
        LOOMC_EXPECT_OK(
            Build("{}", output, 0, {}, workspace.get(), roots.other));
        for (unsigned i = 0; i < 8; ++i) {
          Output warm;
          const bool match = i % 2 == 0;
          LOOMC_EXPECT_OK(Build("{}", &warm, 0, {}, workspace.get(),
                                match ? roots.matching : roots.other));
          ExpectSucceeded(warm, match ? 2 : 1);
          EXPECT_EQ(RequestBytes(warm),
                    RequestBytes(match ? matching : baseline));
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    EXPECT_EQ(allocator_.live_bytes(), expected_bytes);
    for (const auto& output : results) {
      ExpectSucceeded(output);
      EXPECT_EQ(RequestBytes(output), RequestBytes(baseline));
    }
  }
}

TEST_P(NativeSourceTest, FailedPlanConstructionAndSinkLeaveReusableSource) {
  for (const auto& roots :
       {kStagedRoots[0], kStagedRoots[2], kStagedRoots[6]}) {
    SCOPED_TRACE(roots.matching);
    ResetCompiler(64 * 1024 * 1024);
    Output matching;
    LOOMC_ASSERT_OK(Build("{}", &matching, 0, {}, nullptr, roots.matching));
    const size_t initial_allocations = allocator_.allocations();
    Output baseline;
    LOOMC_ASSERT_OK(Build("{}", &baseline, 0, {}, nullptr, roots.other));
    const size_t allocations = allocator_.allocations() - initial_allocations;
    ASSERT_GT(allocations, 0u);
    for (size_t failure = 0; failure < allocations; ++failure) {
      SCOPED_TRACE(failure);
      ResetCompiler(64 * 1024 * 1024);
      LOOMC_ASSERT_OK(Build("{}", &matching, 0, {}, nullptr, roots.matching));
      const size_t retained_bytes = allocator_.live_bytes();
      allocator_.FailAfter(failure);
      Output failed;
      EXPECT_EQ(loomc_status_consume_code(
                    Build("{}", &failed, 0, {}, nullptr, roots.other)),
                LOOMC_STATUS_RESOURCE_EXHAUSTED);
      EXPECT_EQ(failed.product, nullptr);
      EXPECT_TRUE(failed.requests.empty());
      EXPECT_EQ(allocator_.live_bytes(), retained_bytes);
      Output retry;
      LOOMC_ASSERT_OK(Build("{}", &retry, 0, {}, nullptr, roots.other));
      ExpectSucceeded(retry);
      EXPECT_EQ(RequestBytes(retry), RequestBytes(baseline));
    }
    Output failed;
    const loomc_request_sink_t sink = {
        .publish = [](void*, loomc_request_t* request) -> loomc_status_t {
          loomc_request_release(request);
          return loomc_make_status(LOOMC_STATUS_ABORTED,
                                   "publication rejected");
        },
    };
    EXPECT_EQ(loomc_status_consume_code(
                  Build("{}", &failed, 0, sink, nullptr, roots.matching)),
              LOOMC_STATUS_ABORTED);
    EXPECT_EQ(failed.product, nullptr);
    const size_t calls = allocator_.allocations();
    Output retry;
    LOOMC_ASSERT_OK(Build("{}", &retry, 0, {}, nullptr, roots.matching));
    ExpectSucceeded(retry, 2);
    EXPECT_EQ(allocator_.allocations(), calls);
    EXPECT_EQ(RequestBytes(retry), RequestBytes(matching));
  }
}

TEST_P(NativeSourceTest, PlanIndexRemainsExactAcrossRepeatedBoundedEviction) {
  constexpr size_t kValueCount = 64;
  std::array<std::string, kValueCount> expected;
  const size_t owner_bytes = allocator_.live_bytes();
  size_t eight_plans = 0;
  for (size_t value = 0; value < kValueCount; ++value) {
    Output output;
    const auto json = "{\"staged_value\":" + std::to_string(value) + "}";
    LOOMC_ASSERT_OK(Build(json.c_str(), &output, 0, {}, nullptr,
                          "staged_scalar_configured"));
    ExpectSucceeded(output, value == 4 ? 2 : 1);
    expected[value] = RequestBytes(output);
    if (value == 7) {
      eight_plans = allocator_.live_bytes() - owner_bytes;
    }
  }
  const size_t calls = allocator_.allocations();
  for (size_t value = 0; value < kValueCount; ++value) {
    Output output;
    const auto json = "{\"staged_value\":" + std::to_string(value) + "}";
    LOOMC_ASSERT_OK(Build(json.c_str(), &output, 0, {}, nullptr,
                          "staged_scalar_configured"));
    EXPECT_EQ(RequestBytes(output), expected[value]);
  }
  EXPECT_EQ(allocator_.allocations(), calls);
  ResetCompiler(eight_plans);
  const size_t fixed_bytes = allocator_.live_bytes();
  for (size_t i = 0; i < 2 * kValueCount; ++i) {
    // Visit every key in a different order than its hash-bit insertion path.
    const size_t value = (i * 37) % kValueCount;
    Output output;
    const auto json = "{\"staged_value\":" + std::to_string(value) + "}";
    LOOMC_ASSERT_OK(Build(json.c_str(), &output, 0, {}, nullptr,
                          "staged_scalar_configured"));
    EXPECT_EQ(RequestBytes(output), expected[value]);
    EXPECT_LE(allocator_.live_bytes(), fixed_bytes + eight_plans);
  }
}

INSTANTIATE_TEST_SUITE_P(ProviderStorage, NativeSourceTest, ::testing::Bool());

}  // namespace
