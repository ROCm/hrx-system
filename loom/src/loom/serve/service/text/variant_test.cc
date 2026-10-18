// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <condition_variable>
#include <mutex>

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

// A real executor owner submits the candidate while the outer test keeps the
// original model active. One worker catches accidental worker-side JIT waits.
class VariantCompilation {
 public:
  VariantCompilation(loom_serve_text_model_t* model,
                     loom_serve_sources_t sources)
      : model_(model), sources_(sources) {}

  ~VariantCompilation() {
    if (started_) {
      IREE_EXPECT_OK(Await());
    }
    loom_serve_text_variant_free(variant_);
  }

  void Start(iree_task_executor_t* executor) {
    started_ = true;
    iree_task_process_initialize(Submit, 0, 1, &process_);
    process_.user_data = this;
    process_.release_fn = Release;
    iree_task_executor_schedule_process(executor, &process_);
  }

  iree_status_t Await() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return completed_ && released_; });
    started_ = false;
    const auto status = status_;
    status_ = iree_ok_status();
    return status;
  }

  loom_serve_text_variant_t* variant() const { return variant_; }

  loom_serve_text_variant_t* TakeVariant() {
    auto* result = variant_;
    variant_ = nullptr;
    return result;
  }

 private:
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t* context,
                              iree_task_process_drain_result_t* result) {
    auto* self = static_cast<VariantCompilation*>(process->user_data);
    auto status = loom_serve_text_model_compile_variant_async(
        self->model_, &self->sources_, nullptr, Complete, self);
    if (!iree_status_is_ok(status)) {
      Complete(self, status, nullptr, nullptr);
    }
    result->completed = true;
    result->did_work = true;
    return iree_ok_status();
  }

  static void Complete(void* user_data, iree_status_t status, void* variant,
                       const loom_serve_compilation_report_t* report) {
    auto* self = static_cast<VariantCompilation*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->status_ = status;
    self->variant_ = static_cast<loom_serve_text_variant_t*>(variant);
    self->completed_ = true;
    self->condition_.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<VariantCompilation*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->released_ = true;
    self->condition_.notify_one();
  }

  // Borrowed live model through completion and candidate release.
  loom_serve_text_model_t* model_;
  // Borrowed immutable source paths through completion.
  loom_serve_sources_t sources_;
  // Owner task with independent final-release notification.
  iree_task_process_t process_ = {};
  // Protects callback results and final process release.
  std::mutex mutex_;
  // Only the outer test thread waits; no device or worker waits here.
  std::condition_variable condition_;
  // True until the outer thread consumes the accepted operation's outcome.
  bool started_ = false;
  // Compiler completion published its final access.
  bool completed_ = false;
  // The submitting process has retired its final access.
  bool released_ = false;
  // Owned terminal status, transferred by Await.
  iree_status_t status_ = iree_ok_status();
  // Owned successful or failed candidate.
  loom_serve_text_variant_t* variant_ = nullptr;
};

// Mutating a copy models an author editing a native provider between JITs.
// The full authored program remains the same fixture used by HTTP serving.
void EditNativeProvider(const std::string& source_path,
                        const std::string& target_path,
                        const std::string& original,
                        const std::string& replacement) {
  std::ifstream input(source_path);
  ASSERT_TRUE(input.good());
  std::string source((std::istreambuf_iterator<char>(input)), {});
  const size_t position = source.find(original);
  ASSERT_NE(position, std::string::npos);
  source.replace(position, original.size(), replacement);
  std::ofstream output(target_path);
  output << source;
  output.close();
  ASSERT_TRUE(output.good());
}

TEST_F(SourceTextServiceTest, NativeVariantPreservesLiveRowsAndPinnedPrefix) {
  auto* source = loom_serve_text_model_row(model_, 0);
  auto* branch = loom_serve_text_model_row(model_, 1);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      source, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 32, &admitted));
  ASSERT_TRUE(admitted);
  const int32_t tokens[] = {0, 1, 2};
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(source, IREE_ARRAYSIZE(tokens), tokens));
  EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  loom_serve_text_checkpoint_t* endpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &endpoint));
  ASSERT_NE(endpoint, nullptr);
  std::unique_ptr<loom_serve_text_checkpoint_t,
                  decltype(&loom_serve_text_checkpoint_release)>
      checkpoint(endpoint, loom_serve_text_checkpoint_release);
  const Oracle prefix = oracle;
  const auto weights = loom_serve_text_model_weight_activity(model_);
  const auto* chat_policy = loom_serve_text_model_chat_policy(model_);
  ASSERT_GT(weights.read_bytes, 0u);

  const auto path = iree::testing::MakeTempFilePath("native_variant", ".loom");
  // Select token zero without changing the retained-state update. This makes
  // publication observable independently of source hashes or compile counters.
  ASSERT_NO_FATAL_FAILURE(EditNativeProvider(directory_ + "/advance.loom", path,
                                             "scalar.remui %total, %c3_i32",
                                             "scalar.remui %total, %c1_i32"));
  const auto provider = iree_make_cstring_view(path.c_str());
  const loom_serve_sources_t sources = {.count = 1, .values = &provider};
  VariantCompilation candidate(model_, sources);
  candidate.Start(loom_serve_host_executor(host_));

  // The active native table keeps working while its replacement compiles.
  const int32_t during = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(source, 1, &during));
  oracle.Consume(during);
  EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  IREE_ASSERT_OK(candidate.Await());
  ASSERT_NE(candidate.variant(), nullptr);
  EXPECT_GT(loom_serve_text_variant_report(candidate.variant())->kernels, 0u);
  const auto before = loom_serve_text_model_pool_usage(model_);
  IREE_ASSERT_OK(
      loom_serve_text_model_select_variant(model_, candidate.variant()));
  const auto after = loom_serve_text_model_pool_usage(model_);
  EXPECT_EQ(after.available, before.available);
  EXPECT_EQ(after.pending, before.pending);
  EXPECT_EQ(loom_serve_text_model_chat_policy(model_), chat_policy);
  // Selection owns code independently of the candidate and its reflection.
  loom_serve_text_variant_free(candidate.TakeVariant());
  const int32_t next = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(source, 1, &next));
  oracle.Consume(next);
  EXPECT_NE(oracle.Predict(), 0);
  EXPECT_EQ(loom_serve_text_row_token(source), 0);

  // A pinned endpoint created by the original code remains independently
  // advanceable by the replacement, including its pending prediction.
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      branch, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint.get(), 32,
      &admitted));
  ASSERT_TRUE(admitted);
  EXPECT_EQ(loom_serve_text_row_token(branch), prefix.Predict());
  const int32_t branch_token = 0;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(branch, 1, &branch_token));
  Oracle branch_oracle = prefix;
  branch_oracle.Consume(branch_token);
  EXPECT_NE(branch_oracle.Predict(), 0);
  EXPECT_EQ(loom_serve_text_row_token(branch), 0);

  IREE_ASSERT_OK(loom_serve_text_model_select_variant(model_, nullptr));
  IREE_ASSERT_OK(loom_serve_text_row_decode(source));
  oracle.Consume(0);
  EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  IREE_ASSERT_OK(loom_serve_text_row_decode(branch));
  branch_oracle.Consume(0);
  EXPECT_EQ(loom_serve_text_row_token(branch), branch_oracle.Predict());
  const auto retained = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_F(SourceTextServiceTest, FailedNativeVariantsLeaveCurrentModelUsable) {
  auto* row = loom_serve_text_model_row(model_, 0);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 32, &admitted));
  ASSERT_TRUE(admitted);
  Oracle oracle;
  const int32_t token = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
  oracle.Consume(token);
  const auto weights = loom_serve_text_model_weight_activity(model_);
  for (bool invalid_source : {true, false}) {
    SCOPED_TRACE(invalid_source);
    const auto path =
        iree::testing::MakeTempFilePath("rejected_variant", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/advance.loom", path,
        invalid_source ? "scalar.remui %total" : "\"bias\"",
        invalid_source ? "scalar.not_an_operation %total" : "\"other\""));
    const auto provider = iree_make_cstring_view(path.c_str());
    const loom_serve_sources_t sources = {.count = 1, .values = &provider};
    VariantCompilation candidate(model_, sources);
    candidate.Start(loom_serve_host_executor(host_));
    IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, candidate.Await());
    ASSERT_NE(candidate.variant(), nullptr);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_serve_text_model_select_variant(model_, candidate.variant()));
    if (invalid_source) {
      EXPECT_GT(
          loom_serve_text_variant_report(candidate.variant())->diagnostic_count,
          0u);
    }
    IREE_ASSERT_OK(loom_serve_text_row_prefill(row, 1, &token));
    oracle.Consume(token);
    EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
  }
  const auto retained = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_F(SourceTextServiceTest, IndependentCandidatesShareOnlyModelResources) {
  const auto modified_path =
      iree::testing::MakeTempFilePath("independent_variant", ".loom");
  const std::string original_path = directory_ + "/advance.loom";
  ASSERT_NO_FATAL_FAILURE(EditNativeProvider(original_path, modified_path,
                                             "scalar.remui %total, %c3_i32",
                                             "scalar.remui %total, %c1_i32"));
  const auto modified_provider = iree_make_cstring_view(modified_path.c_str());
  VariantCompilation modified(model_,
                              {.count = 1, .values = &modified_provider});
  VariantCompilation original(model_, document_.value().sources);
  modified.Start(loom_serve_host_executor(host_));
  original.Start(loom_serve_host_executor(host_));
  IREE_ASSERT_OK(modified.Await());
  IREE_ASSERT_OK(original.Await());
  ASSERT_NE(modified.variant(), original.variant());
  const auto* modified_report =
      loom_serve_text_variant_report(modified.variant());
  const auto* original_report =
      loom_serve_text_variant_report(original.variant());
  EXPECT_NE(modified_report, original_report);
  EXPECT_GT(modified_report->kernels, 0u);
  EXPECT_GT(original_report->kernels, 0u);

  auto* row = loom_serve_text_model_row(model_, 0);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 32, &admitted));
  ASSERT_TRUE(admitted);
  const int32_t tokens[] = {0, 1, 2, 1, 1};
  Oracle oracle;
  for (int32_t token : tokens) {
    oracle.Consume(token);
  }
  IREE_ASSERT_OK(
      loom_serve_text_model_select_variant(model_, modified.variant()));
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(row, IREE_ARRAYSIZE(tokens), tokens));
  EXPECT_EQ(loom_serve_text_row_token(row), 0);
  EXPECT_NE(oracle.Predict(), 0);
  IREE_ASSERT_OK(
      loom_serve_text_model_select_variant(model_, original.variant()));
  IREE_ASSERT_OK(loom_serve_text_row_decode(row));
  oracle.Consume(0);
  EXPECT_EQ(loom_serve_text_row_token(row), oracle.Predict());
}

// Drives the actual HTTP service owner while compilation and device execution
// use its normal shared worker pool. Teardown joins accepted work even when a
// test assertion exits early; callback storage lives through that join.
class ReconfigurationTextServiceTest : public SourceTextServiceTest {
 protected:
  void SetUp() override {
    SourceTextServiceTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    auto options = ServiceOptions();
    options.variant_capacity = 6;
    IREE_ASSERT_OK(loom_serve_text_service_allocate(
        model_, &options, nullptr, host_allocator_, &service_));
    interface_ = loom_serve_text_service_interface(service_);
    configuration_ = interface_.reconfiguration;
    ASSERT_NE(configuration_.pool, nullptr);
    original_ = configuration_.selected(configuration_.self);
    Observe(interface_);
  }

  void TearDown() override {
    if (service_) {
      AdvanceUntil(
          interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING);
      interface_.set_observer(interface_.self, {});
      loom_serve_text_service_free(service_, IREE_STATUS_OK);
      IREE_EXPECT_OK(TakeSelectionStatus());
    }
    SourceTextServiceTest::TearDown();
  }

  bool Idle() const {
    return !interface_.is_pending(interface_.self) &&
           !interface_.is_active(interface_.self) &&
           !interface_.snapshot(interface_.self).queued_requests;
  }

  void Stage(std::string path, loom_serve_variant_id_t base,
             loom_serve_variant_id_t* out_id) {
    // Both the locator array and its string disappear before JIT completion.
    // The service captures inputs instead of borrowing the management request.
    path = std::filesystem::absolute(path).string();
    const auto provider = iree_make_cstring_view(path.c_str());
    const loom_serve_sources_t sources = {.count = 1, .values = &provider};
    IREE_ASSERT_OK(configuration_.stage_native(configuration_.self, base,
                                               &sources, out_id));
  }

  void AwaitCandidates() {
    AdvanceUntil(interface_, [&] {
      return !loom_serve_variant_pool_pending(configuration_.pool);
    });
  }

  void Select(loom_serve_variant_id_t id) {
    selection_.called = false;
    IREE_ASSERT_OK(configuration_.select(
        configuration_.self, id, configuration_.selected(configuration_.self),
        Selected, this));
  }

  iree_status_t TakeSelectionStatus() {
    const auto status = selection_.status;
    selection_.status = iree_ok_status();
    return status;
  }

  static void Selected(void* user_data, iree_status_t status,
                       loom_serve_variant_id_t id) {
    auto* self = static_cast<ReconfigurationTextServiceTest*>(user_data);
    EXPECT_FALSE(self->selection_.called);
    self->selection_.status = status;
    self->selection_.id = id;
    self->selection_.called = true;
  }

  // Owned service; the inherited model and transport outlive its retirement.
  loom_serve_text_service_t* service_ = nullptr;
  // Borrowed owner callbacks, invoked without a concurrent router owner.
  loom_serve_service_t interface_ = {};
  // Borrowed modality policy; its pool has no preferred-choice field.
  loom_serve_reconfiguration_t configuration_ = {};
  // Initially preferred executable identity, used for explicit rollback.
  loom_serve_variant_id_t original_ = 0;
  // Owner-thread terminal selection delivery, alive through TearDown.
  struct {
    // Exactly one callback is owed for an accepted selection.
    bool called = false;
    // Owned status, consumed by the test or teardown.
    iree_status_t status = iree_ok_status();
    // Actual preferred identity after success, failure or cancellation.
    loom_serve_variant_id_t id = 0;
  } selection_;
};

TEST_F(ReconfigurationTextServiceTest,
       SwitchWaitsForBorrowersAndPreservesHttpSessionAcrossRollback) {
  const auto unloaded = loom_serve_text_model_weight_activity(model_);
  const auto path = iree::testing::MakeTempFilePath("service_variant", ".loom");
  ASSERT_NO_FATAL_FAILURE(EditNativeProvider(directory_ + "/advance.loom", path,
                                             "scalar.remui %total, %c3_i32",
                                             "scalar.remui %total, %c1_i32"));
  loom_serve_variant_id_t id = 0;
  ASSERT_NO_FATAL_FAILURE(Stage(path, original_, &id));
  ASSERT_NO_FATAL_FAILURE(AwaitCandidates());
  loom_serve_variant_snapshot_t snapshot = {};
  ASSERT_TRUE(
      loom_serve_variant_pool_snapshot(configuration_.pool, id, &snapshot));
  ASSERT_EQ(snapshot.phase, LOOM_SERVE_VARIANT_READY);
  EXPECT_EQ(configuration_.selected(configuration_.self), original_);
  const auto compiled = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(compiled.read_bytes, unloaded.read_bytes);
  EXPECT_EQ(compiled.activations, unloaded.activations);

  const std::string prompt = "abxabba";
  const int first =
      Send(Request(Message("user", prompt), 4), "X-Loom-Session: first\r\n");
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(first, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] {
    return std::string(interface_.snapshot(interface_.self).phase) == "execute";
  }));
  EXPECT_EQ(interface_.telemetry.counters->batches, 0u);
  // First admission activates a demand-loaded model. Establish the resident
  // baseline here so the experiment measures replacement, not initial load.
  const auto weights = loom_serve_text_model_weight_activity(model_);
  EXPECT_GT(weights.read_bytes, 0u);
  const int second =
      Send(Request(Message("user", "baxab"), 3), "X-Loom-Session: second\r\n");
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(second, &connection, &request));
  loom_serve::testing::text::OwnerTurn::Run(
      loom_serve_host_executor(host_), [&] {
        IREE_ASSERT_OK(
            interface_.accept(interface_.self, connection, request, 0));
        EXPECT_EQ(interface_.snapshot(interface_.self).queued_requests, 1u);
        ASSERT_NO_FATAL_FAILURE(Select(id));
        EXPECT_FALSE(selection_.called);
        EXPECT_EQ(configuration_.selected(configuration_.self), original_);
        bool progress = false;
        IREE_ASSERT_OK(interface_.advance(interface_.self, 0, &progress));
        // The accepted CPU task cannot retire while its sole worker runs this
        // turn, even if the outstanding device epoch is already ready.
        EXPECT_FALSE(selection_.called);
        EXPECT_EQ(configuration_.selected(configuration_.self), original_);
      });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  ASSERT_TRUE(selection_.called);
  IREE_ASSERT_OK(TakeSelectionStatus());
  EXPECT_EQ(selection_.id, id);
  EXPECT_EQ(Content(Read(first)), "aaaa");
  EXPECT_EQ(Content(Read(second)), "aaa");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_variant_pool_remove(configuration_.pool, id));

  ASSERT_NO_FATAL_FAILURE(Select(original_));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil(interface_, [&] { return selection_.called; }));
  IREE_ASSERT_OK(TakeSelectionStatus());
  EXPECT_EQ(selection_.id, original_);
  IREE_ASSERT_OK(loom_serve_variant_pool_remove(configuration_.pool, id));
  const int resumed = Send(
      Request(Message("user", prompt) + "," + Message("assistant", "aaaa") +
                  "," + Message("user", "bx"),
              2),
      "X-Loom-Session: first\r\n");
  ASSERT_NO_FATAL_FAILURE(Accept(resumed, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  const auto response = Read(resumed);
  EXPECT_EQ(Content(response), Oracle::Complete(prompt + "aaaa#bx", 2));
  EXPECT_NE(response.find(R"("cached_tokens":10)"), std::string::npos)
      << response;
  const auto retained = loom_serve_text_model_weight_activity(model_);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
  EXPECT_EQ(retained.activations, weights.activations);
}

TEST_F(ReconfigurationTextServiceTest,
       CandidateFailuresAndStaleSelectionsLeaveInferenceHealthy) {
  const auto changed =
      iree::testing::MakeTempFilePath("service_choice", ".loom");
  ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
      directory_ + "/advance.loom", changed, "scalar.remui %total, %c3_i32",
      "scalar.remui %total, %c1_i32"));
  loom_serve_variant_id_t ready = 0, alternative = 0;
  ASSERT_NO_FATAL_FAILURE(Stage(changed, original_, &ready));
  ASSERT_NO_FATAL_FAILURE(
      Stage(directory_ + "/advance.loom", original_, &alternative));
  loom_serve_variant_id_t failures[2] = {};
  for (unsigned i = 0; i < 2; ++i) {
    const auto rejected =
        iree::testing::MakeTempFilePath("service_rejected", ".loom");
    ASSERT_NO_FATAL_FAILURE(EditNativeProvider(
        directory_ + "/advance.loom", rejected,
        i ? "\"bias\"" : "scalar.remui %total",
        i ? "\"missing\"" : "scalar.not_an_operation %total"));
    ASSERT_NO_FATAL_FAILURE(Stage(rejected, original_, &failures[i]));
  }
  ASSERT_NO_FATAL_FAILURE(AwaitCandidates());
  EXPECT_EQ(configuration_.selected(configuration_.self), original_);
  for (unsigned i = 0; i < 2; ++i) {
    loom_serve_variant_snapshot_t snapshot = {};
    ASSERT_TRUE(loom_serve_variant_pool_snapshot(configuration_.pool,
                                                 failures[i], &snapshot));
    EXPECT_EQ(snapshot.phase, LOOM_SERVE_VARIANT_FAILED);
    EXPECT_EQ(snapshot.code, IREE_STATUS_INVALID_ARGUMENT);
    EXPECT_GT(snapshot.message.size, 0u);
    if (!i) {
      EXPECT_GT(snapshot.compilation->diagnostic_count, 0u);
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        configuration_.select(configuration_.self, failures[i], original_,
                              Selected, this));
  }
  EXPECT_FALSE(selection_.called);
  // A request after failed JIT/reflection still executes the unchanged model.
  const int baseline = Send(Request(Message("user", "baxab"), 3));
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(baseline, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(baseline)), Oracle::Complete("baxab", 3));

  ASSERT_NO_FATAL_FAILURE(Select(ready));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        configuration_.select(configuration_.self, alternative,
                                              original_, Selected, this));
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil(interface_, [&] { return selection_.called; }));
  IREE_ASSERT_OK(TakeSelectionStatus());
  EXPECT_EQ(selection_.id, ready);
  // A second author cannot publish over a choice different from its expected
  // current base. Merely acquiring another choice is not a policy change.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        configuration_.select(configuration_.self, alternative,
                                              original_, Selected, this));
  loom_serve_variant_entry_t* pinned = nullptr;
  IREE_ASSERT_OK(loom_serve_variant_pool_acquire(configuration_.pool,
                                                 alternative, &pinned));
  EXPECT_EQ(configuration_.selected(configuration_.self), ready);
  loom_serve_variant_entry_release(pinned);
  const int updated = Send(Request(Message("user", "baxab"), 3));
  ASSERT_NO_FATAL_FAILURE(Accept(updated, interface_, &connection));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  EXPECT_EQ(Content(Read(updated)), "aaa");
}

TEST_F(ReconfigurationTextServiceTest,
       SelectionWaitsForTheInputCountingSourceBorrow) {
  loom_serve_variant_id_t id = 0;
  ASSERT_NO_FATAL_FAILURE(Stage(directory_ + "/advance.loom", original_, &id));
  ASSERT_NO_FATAL_FAILURE(AwaitCandidates());
  const int client = Send(Request(Message("user", "abxa")), "",
                          "/v1/chat/completions/input_tokens");
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client, &connection, &request));
  loom_serve::testing::text::OwnerTurn::Run(
      loom_serve_host_executor(host_), [&] {
        IREE_ASSERT_OK(
            interface_.accept(interface_.self, connection, request, 0));
        ASSERT_NO_FATAL_FAILURE(Select(id));
        bool progress = false;
        IREE_ASSERT_OK(interface_.advance(interface_.self, 0, &progress));
        EXPECT_FALSE(selection_.called);
        EXPECT_EQ(configuration_.selected(configuration_.self), original_);
      });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(interface_, [&] { return Idle(); }));
  ASSERT_TRUE(selection_.called);
  IREE_ASSERT_OK(TakeSelectionStatus());
  EXPECT_EQ(selection_.id, id);
  EXPECT_NE(Read(client).find("{\"input_tokens\":4}"), std::string::npos);
  EXPECT_EQ(loom_serve_text_model_weight_activity(model_).activations, 0u);
}

TEST_F(ReconfigurationTextServiceTest,
       StopCancelsSelectionButJoinsAcceptedCompilerBorrow) {
  loom_serve_variant_id_t id = 0;
  loom_serve::testing::text::OwnerTurn::Run(
      loom_serve_host_executor(host_), [&] {
        ASSERT_NO_FATAL_FAILURE(
            Stage(directory_ + "/advance.loom", original_, &id));
        ASSERT_NO_FATAL_FAILURE(Select(original_));
        bool progress = false;
        IREE_ASSERT_OK(interface_.advance(
            interface_.self, LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress));
        EXPECT_TRUE(selection_.called);
        EXPECT_EQ(selection_.id, original_);
        IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, TakeSelectionStatus());
        EXPECT_TRUE(interface_.is_pending(interface_.self));
        loom_serve_variant_snapshot_t snapshot = {};
        ASSERT_TRUE(loom_serve_variant_pool_snapshot(configuration_.pool, id,
                                                     &snapshot));
        EXPECT_EQ(snapshot.phase, LOOM_SERVE_VARIANT_COMPILING);
      });
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil(
      interface_, [&] { return Idle(); }, LOOM_SERVE_SERVICE_FLAG_STOPPING));
  loom_serve_variant_snapshot_t snapshot = {};
  ASSERT_TRUE(
      loom_serve_variant_pool_snapshot(configuration_.pool, id, &snapshot));
  EXPECT_EQ(snapshot.phase, LOOM_SERVE_VARIANT_READY);
  const auto provider = IREE_SV("ignored.loom");
  const loom_serve_sources_t sources = {.count = 1, .values = &provider};
  loom_serve_variant_id_t rejected = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      configuration_.stage_native(configuration_.self, original_, &sources,
                                  &rejected));
  EXPECT_EQ(rejected, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        configuration_.select(configuration_.self, id,
                                              original_, Selected, this));
}

}  // namespace
