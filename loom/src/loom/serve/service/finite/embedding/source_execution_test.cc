// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/runtime/program/package.h"
#include "loom/serve/service/finite/embedding/protocol.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/service/finite/result.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/stalled_stderr.h"

namespace loom_serve::testing::embedding {

using loom_serve::testing::StalledStderr;

TEST_F(SourceEmbeddingServiceTest, ReconfiguresCommandToEliminateWorkspace) {
  StartRouter();
  const auto inventory =
      Read(Send({}, "/v1/models/variants", "GET", "X-Loom-Model: fixture\r\n"));
  ASSERT_EQ(inventory.find("HTTP/1.1 200"), 0u) << inventory;
  const uint64_t original = Number(inventory, "selected");
  const std::string request =
      R"({"model":"fixture","input":["a","bbb"],"dimensions":2})";
  const auto baseline = Read(Send(request));
  EXPECT_NE(baseline.find(R"("index":0,"embedding":[1,1])"), std::string::npos)
      << baseline;
  EXPECT_NE(baseline.find(R"("index":1,"embedding":[6,3])"), std::string::npos)
      << baseline;
  loom_serve::testing::SourcePackage package;
  IREE_ASSERT_OK(
      package.AddFile("model/config.loom", directory_ + "/config.loom"));
  IREE_ASSERT_OK(package.AddFile("model/direct.loom",
                                 directory_ + "/variants/direct.loom"));
  IREE_ASSERT_OK(
      package.AddFile("model/fused.loom", directory_ + "/variants/fused.loom"));
  int client = -1;
  IREE_ASSERT_OK(http_.Send(
      package.Request(
          "/v1/models/variants",
          "{\"model\":\"fixture\",\"base\":" + std::to_string(original) +
              ",\"sources\":[\"config.loom\",\"direct.loom\",\"fused.loom\"],"
              "\"wait\":true}"),
      &client));
  const auto staged = Read(client);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = Number(staged, "id", "variant");
  ASSERT_EQ(SelectVariant(candidate, original).find("HTTP/1.1 200"), 0u);
  const auto fused = Read(Send(request));
  EXPECT_NE(fused.find(R"("index":0,"embedding":[1,1])"), std::string::npos)
      << fused;
  EXPECT_NE(fused.find(R"("index":1,"embedding":[6,3])"), std::string::npos)
      << fused;
  ASSERT_EQ(SelectVariant(original, candidate).find("HTTP/1.1 200"), 0u);
  const auto restored = Read(Send(request));
  EXPECT_NE(restored.find(R"("index":0,"embedding":[1,1])"), std::string::npos)
      << restored;
  EXPECT_NE(restored.find(R"("index":1,"embedding":[6,3])"), std::string::npos)
      << restored;
}

TEST_F(SourceEmbeddingServiceTest,
       CoalescesRaggedRequestsAndDropsOnlyResetPeer) {
  const int a = Send(R"({"model":"fixture","input":["a","bbb"]})");
  ASSERT_NO_FATAL_FAILURE(Accept(a));
  const int b = Send(R"({"model":"fixture","input":[2]})");
  loom_serve_http_connection_t* cancelled = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(b, &cancelled));
  const int c = Send(R"({"model":"fixture","input":[[2,0]],"dimensions":2})");
  ASSERT_NO_FATAL_FAILURE(Accept(c));
  const int bad = Send(R"({"model":"fixture","input":"a","dimensions":3})");
  ASSERT_NO_FATAL_FAILURE(Accept(bad));
  for (iree_host_size_t i = 0; i < 4; ++i) {
    loom_serve_http_request_snapshot_t snapshot = {};
    ASSERT_TRUE(
        interface_.inspection.request_at(interface_.self, i, &snapshot));
    EXPECT_EQ(snapshot.id, i + 1);
    EXPECT_STREQ(snapshot.phase, "queued");
    EXPECT_EQ(snapshot.counters.requests, 1u);
    EXPECT_EQ(snapshot.counters.input_tokens, 0u);
  }
  EXPECT_TRUE(Step());  // All independent requests prepared.
  // The executor may still own prepared packets; only immutable receipt and
  // owner-published progress are inspected before that release is consumed.
  loom_serve_http_request_snapshot_t preparing = {};
  ASSERT_TRUE(interface_.inspection.request_at(interface_.self, 0, &preparing));
  EXPECT_EQ(preparing.id, 1u);
  EXPECT_STREQ(preparing.phase, "preparing");
  EXPECT_TRUE(Step());  // Rejection collected; valid cohort constructed.
  EXPECT_TRUE(Step());  // One command submitted, with compact feedback.
  for (iree_host_size_t i = 0; i < 3; ++i) {
    loom_serve_http_request_snapshot_t snapshot = {};
    ASSERT_TRUE(
        interface_.inspection.request_at(interface_.self, i, &snapshot));
    EXPECT_EQ(snapshot.id, i + 1);
    EXPECT_STREQ(snapshot.phase, "running");
    EXPECT_EQ(snapshot.counters.input_tokens, 0u);
  }
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(b, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(b), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* value) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(value));
      },
      cancelled, iree_infinite_timeout());
  EXPECT_TRUE(Step());  // Capture only live result views for CPU encoding.
  EXPECT_STREQ(interface_.snapshot(interface_.self).phase, "encoding");
  EXPECT_EQ(interface_.snapshot(interface_.self).active_requests, 2u);
  EXPECT_EQ(interface_.telemetry.counters->cancelled, 1u);
  Drain();
  const std::string result_a = Read(a);
  EXPECT_EQ(result_a.find("HTTP/1.1 200"), 0u);
  // Fourth component was computed on device from the *whole* cohort's four
  // rows, including the reset peer. Physical result rows were stored backwards.
  EXPECT_NE(result_a.find(R"("index":0,"embedding":[1,1,1,4])"),
            std::string::npos);
  EXPECT_NE(result_a.find(R"("index":1,"embedding":[6,3,2,4])"),
            std::string::npos);
  EXPECT_NE(result_a.find(R"("prompt_tokens":4,"total_tokens":4)"),
            std::string::npos);
  const std::string result_c = Read(c);
  EXPECT_NE(result_c.find(R"("index":0,"embedding":[4,2])"), std::string::npos);
  EXPECT_NE(result_c.find(R"("prompt_tokens":2,"total_tokens":2)"),
            std::string::npos);
  EXPECT_EQ(Read(bad).find("HTTP/1.1 400"), 0u);
  // The same source/queue/pool is reusable after feedback and CPU release.
  const int next = Send(R"({"model":"fixture","input":[0]})");
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  EXPECT_NE(Read(next).find(R"("embedding":[1,1,1,1])"), std::string::npos);
  const auto& counters = *interface_.telemetry.counters;
  EXPECT_EQ(counters.requests, 5u);
  EXPECT_EQ(counters.completed, 3u);
  EXPECT_EQ(counters.rejected, 1u);
  EXPECT_EQ(counters.cancelled, 1u);
  EXPECT_EQ(counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens, 8u);
  EXPECT_EQ(counters.batches, 2u);
}

TEST_F(SourceEmbeddingServiceTest, CancelledDeviceCohortSkipsEncoding) {
  CheckAbandonedCohort(0);
}

TEST_F(SourceEmbeddingServiceTest, StoppingDeviceCohortSkipsEncoding) {
  CheckAbandonedCohort(LOOM_SERVE_SERVICE_FLAG_STOPPING);
}

TEST_F(SourceEmbeddingServiceTest,
       SourceCapacitySplitsWithoutChangingClientRows) {
  std::vector<int> clients;
  for (size_t i = 0; i < 4; ++i) {
    clients.push_back(Send("{\"model\":\"fixture\",\"input\":\"" +
                           std::string(31, 'b') + "\"}"));
    ASSERT_NO_FATAL_FAILURE(Accept(clients.back()));
  }
  Drain();
  for (int client : clients) {
    EXPECT_NE(Read(client).find(R"("embedding":[62,31,2,2])"),
              std::string::npos);
  }
}

TEST_F(SourceEmbeddingServiceTest,
       InputLimitsRejectOnlyTheirRequestsAndPreserveFullCapacity) {
  std::string long_ids = "[0";
  for (size_t i = 1; i < 32; ++i) {
    long_ids += ",0";
  }
  long_ids += "]";
  std::string many_rows = "[\"a\"";
  for (size_t i = 1; i < 65; ++i) {
    many_rows += ",\"a\"";
  }
  many_rows += "]";
  const std::string costly_row = "\"" + std::string(22, 'a') + "\"";
  const std::string invalid_inputs[] = {
      "\"" + std::string(32, 'b') + "\"",
      "\"" + std::string(4096, 'a') + "\"",
      long_ids,
      "[3]",
      many_rows,
      "[" + costly_row + "," + costly_row + "," + costly_row + "]",
  };
  std::vector<int> rejected;
  for (const auto& input : invalid_inputs) {
    rejected.push_back(Send("{\"model\":\"fixture\",\"input\":" + input + "}"));
    ASSERT_NO_FATAL_FAILURE(Accept(rejected.back()));
  }
  // Every row fits individually, and these two consume the entire 64-unit
  // cohort. Rejected preparation must retain neither units nor device rows.
  const int valid =
      Send("{\"model\":\"fixture\",\"input\":[\"" + std::string(31, 'b') +
           "\",\"" + std::string(31, 'a') + "\"]}");
  ASSERT_NO_FATAL_FAILURE(Accept(valid));
  Drain();
  for (size_t i = 0; i < rejected.size(); ++i) {
    SCOPED_TRACE(i);
    const auto response = Read(rejected[i]);
    EXPECT_EQ(response.find("HTTP/1.1 400"), 0u);
    EXPECT_NE(response.find("fixture requires"), std::string::npos);
  }
  const auto response = Read(valid);
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(response.find(R"("index":0,"embedding":[62,31,2,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("index":1,"embedding":[31,31,1,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("prompt_tokens":62,"total_tokens":62)"),
            std::string::npos);
  EXPECT_FALSE(interface_.is_pending(interface_.self));

  const int next = Send(R"({"model":"fixture","input":[0]})");
  ASSERT_NO_FATAL_FAILURE(Accept(next));
  Drain();
  const auto next_response = Read(next);
  EXPECT_EQ(next_response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(next_response.find(R"("embedding":[1,1,1,1])"), std::string::npos);
  const auto& counters = *interface_.telemetry.counters;
  EXPECT_EQ(counters.requests, 8u);
  EXPECT_EQ(counters.completed, 2u);
  EXPECT_EQ(counters.rejected, 6u);
  EXPECT_EQ(counters.cancelled, 0u);
  EXPECT_EQ(counters.failed, 0u);
  EXPECT_EQ(counters.input_tokens, 63u);
  EXPECT_EQ(counters.batches, 2u);
}

TEST_F(SourceEmbeddingServiceTest,
       SharedRouterServesUtilitiesAndBothEncodings) {
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  // Exercise heartbeat cancellation during normal router shutdown. The test
  // waits for ownership, not the deadline, so even an hour-long timer retires.
  ASSERT_NO_FATAL_FAILURE(CreateService(INT64_C(3600000000000)));
  StartRouter();
  EXPECT_NE(Read(Send("", "/v1/models", "GET")).find("fixture"),
            std::string::npos);
  EXPECT_NE(Read(Send(R"({"model":"fixture","content":"aba"})", "/tokenize"))
                .find(R"({"tokens":[0,1,0]})"),
            std::string::npos);
  const int a =
      Send(R"({"model":"fixture","input":["a","bbb"],"dimensions":2})");
  const int b = Send(
      R"({"model":"fixture","input":[[0],[1,1,1]],"dimensions":2,"encoding_format":"base64"})");
  EXPECT_NE(Read(Send("", "/healthz", "GET")).find("\"status\":\"ready\""),
            std::string::npos);
  const std::string floats = Read(a);
  EXPECT_NE(floats.find(R"("embedding":[1,1])"), std::string::npos);
  EXPECT_NE(floats.find(R"("embedding":[6,3])"), std::string::npos);
  const std::string encoded = Read(b);
  EXPECT_NE(encoded.find(R"("embedding":"AACAPwAAgD8=")"), std::string::npos);
  EXPECT_NE(encoded.find(R"("embedding":"AADAQAAAQEA=")"), std::string::npos);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":[-1]})")).find("HTTP/1.1 400"),
      0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture","input":"a","dimensions":3})"))
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":"a"})")).find("HTTP/1.1 200"),
      0u);
}

TEST_F(SourceEmbeddingServiceTest, TokenizerFailureRetiresCompiledModel) {
  const auto assets =
      iree::testing::MakeTempFilePath("missing_embedding_tokenizer");
  ASSERT_TRUE(std::filesystem::create_directory(assets));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(directory_ + "/model.json"));
  const loom_serve_finite_model_options_t options = {
      .sources = document.value().sources,
      .asset_directory = iree_make_cstring_view(assets.c_str()),
      .options = IREE_SV("{}"),
  };
  const int resident = Send(R"({"model":"fixture","input":"a"})");
  ASSERT_NO_FATAL_FAILURE(Accept(resident));
  loom_serve_test_embedding_model_lifecycle_t load;
  load.service = &interface_;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, load.Run(device_, options));
  EXPECT_EQ(load.model, nullptr);
  EXPECT_GT(load.report.commands, 0u);
  EXPECT_NE(Read(resident).find(R"("embedding":[1,1,1,1])"), std::string::npos);
}

TEST_F(SourceEmbeddingServiceTest,
       CancelledAndFailedLoadsKeepTheirResidentPeerUsable) {
  const std::string package =
      iree::testing::MakeTempFilePath("invalid_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  // copy_file preserves read-only runfile permissions on this owned copy.
  std::filesystem::permissions(package + "/control.loom",
                               std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  std::ofstream invalid(package + "/control.loom");
  invalid << "this is not a Loom program\n";
  invalid.close();
  ASSERT_TRUE(invalid.good());
  for (bool cancel : {true, false}) {
    const auto& source = cancel ? directory_ : package;
    loom_serve::testing::ModelDocument document;
    IREE_ASSERT_OK(document.Load(source + "/model.json"));
    const loom_serve_finite_model_options_t options = {
        .sources = document.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{}"),
    };
    const int resident = Send(R"({"model":"fixture","input":"a"})");
    ASSERT_NO_FATAL_FAILURE(Accept(resident));
    loom_serve_test_embedding_model_lifecycle_t load;
    load.service = &interface_;
    load.cancel_after_start = cancel;
    iree_status_t status;
    {
      StalledStderr stalled;
      ASSERT_NO_FATAL_FAILURE(stalled.Begin());
      status = load.Run(device_, options);
    }
    IREE_EXPECT_STATUS_IS(
        cancel ? IREE_STATUS_CANCELLED : IREE_STATUS_INVALID_ARGUMENT, status);
    EXPECT_EQ(load.model, nullptr);
    EXPECT_EQ(load.report.commands, 0u);
    if (!cancel) {
      EXPECT_GT(load.report.diagnostic_count, 0u);
      EXPECT_GT(load.report.diagnostics[0].message_length, 0u);
    }
    EXPECT_NE(Read(resident).find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
  }
}

class DeviceResultTest : public SourceEmbeddingServiceTest {
 protected:
  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    iree_hal_buffer_release(target_);
    if (consumer_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(consumer_));
      loom_serve_execution_release(consumer_);
    }
    iree_hal_semaphore_release(gate_);
    for (auto* result : results_) {
      loom_serve_finite_result_release(result);
    }
    SourceEmbeddingServiceTest::TearDown();
    ReleaseInputs();
  }

  void Prepare(std::initializer_list<const char*> bodies) {
    requests_.resize(bodies.size());
    prepared_.resize(bodies.size());
    std::vector<iree_host_size_t> rows(bodies.size());
    size_t i = 0;
    for (const char* body : bodies) {
      IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
          iree_make_cstring_view(body), host_allocator_, &requests_[i]));
      IREE_ASSERT_OK(loom_serve_finite_model_prepare_request(
          model_, requests_[i].base.input, &prepared_[i]));
      rows[i] = requests_[i].base.count;
      ++i;
    }
    IREE_ASSERT_OK(loom_serve_finite_model_prepare_batch(
        model_, prepared_.size(), prepared_.data(), rows.data()));
  }

  void SubmitPrepared(size_t ordinal) {
    bool code_ready = false;
    IREE_ASSERT_OK(
        loom_serve_finite_model_advance_batch(model_, false, &code_ready));
    ASSERT_TRUE(code_ready);
    loom_serve_residency_admission_t admission;
    IREE_ASSERT_OK(loom_serve_residency_try_admit(
        loom_serve_finite_model_residency(model_), &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    loom_serve_execution_frontier_t frontier;
    IREE_ASSERT_OK(
        loom_serve_finite_model_submit(model_, &results_[ordinal], &frontier));
    EXPECT_EQ(frontier.values[1], transfer_value_);
  }

  void Submit(std::initializer_list<const char*> bodies, size_t ordinal) {
    ASSERT_NO_FATAL_FAILURE(Prepare(bodies));
    ASSERT_NO_FATAL_FAILURE(SubmitPrepared(ordinal));
  }

  void Complete(size_t ordinal) {
    auto ready = loom_serve_finite_result_ready(results_[ordinal]);
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(
          ready.semaphores[i], ready.values[i], iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
    }
    loom_serve_finite_model_complete(model_);
    ReleaseInputs();
  }

  void ReleaseInputs() {
    for (auto& prepared : prepared_) {
      iree_vm_buffer_release(prepared.bytes);
    }
    prepared_.clear();
    for (auto& request : requests_) {
      loom_serve_embedding_request_deinitialize(&request);
    }
    requests_.clear();
  }

  // Independent device consumer, held past producer model destruction.
  loom_serve_execution_t* consumer_ = nullptr;
  // Explicit readiness control for the final device reader.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Consumer-owned pooled destination, independent of finite output geometry.
  iree_hal_buffer_t* target_ = nullptr;
  // Source cohorts retained concurrently without retaining the source model.
  std::array<loom_serve_finite_result_t*, 2> results_ = {};
  // Source input ownership through successful completion or failed teardown.
  std::vector<loom_serve_embedding_request_t> requests_;
  // Per-request source preparation, with the same lifetime as requests_.
  std::vector<loom_serve_finite_prepared_t> prepared_;
  // Only the final consumer output crosses to the host for the oracle.
  std::array<float, 14> output_ = {};
  // Explicit result downloads advance the producer's separate transfer queue.
  uint64_t transfer_value_ = 0;
};

class GenerationSelectionTest : public DeviceResultTest,
                                public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    DeviceResultTest::SetUp();
    if (GetParam()) {
      ASSERT_NO_FATAL_FAILURE(UseDirectPackage());
    }
    update_ = loom_serve_finite_model_execution_update(model_);
  }

  void TearDown() override {
    if (candidate_) {
      update_.dispose(candidate_);
    }
    loom_serve_package_release(package_);
    IREE_EXPECT_OK(status_);
    DeviceResultTest::TearDown();
  }

  void CompileReplacement() {
    const std::string stage = GetParam()
                                  ? (SourceServiceDirectStageSource()[0]
                                         ? SourceServiceDirectStageSource()
                                         : directory_ + "/variants/direct.loom")
                                  : directory_ + "/embed.loom";
    const std::string paths[] = {directory_ + "/config.loom", stage};
    std::string contents[2];
    for (size_t i = 0; i < 2; ++i) {
      std::ifstream input(paths[i]);
      ASSERT_TRUE(input.good());
      contents[i].assign(std::istreambuf_iterator<char>(input), {});
    }
    const std::string original = "scalar.constant 1 : i32";
    const size_t position = contents[1].find(original);
    ASSERT_NE(position, std::string::npos);
    contents[1].replace(position, original.size(), "scalar.constant 2 : i32");
    const loom_serve_package_file_t files[] = {
        {IREE_SV("model/config.loom"),
         iree_make_const_byte_span(contents[0].data(), contents[0].size())},
        {IREE_SV("model/stage.loom"),
         iree_make_const_byte_span(contents[1].data(), contents[1].size())}};
    IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model/model.json"),
                                             IREE_ARRAYSIZE(files), files,
                                             host_allocator_, &package_));
    const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                            IREE_SV("stage.loom")};
    const loom_serve_sources_t sources = {
        .count = IREE_ARRAYSIZE(providers),
        .values = providers,
        .package = package_,
    };
    IREE_ASSERT_OK(update_.compile(
        update_.self, &sources,
        [](void* self, iree_status_t status, void* value,
           const loom_serve_compilation_report_t* report) {
          auto* test = static_cast<GenerationSelectionTest*>(self);
          std::lock_guard<std::mutex> lock(test->mutex_);
          test->candidate_ = value;
          test->status_ = status;
          test->ready_ = true;
          test->condition_.notify_one();
        },
        this));
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return ready_; });
    IREE_ASSERT_OK(std::exchange(status_, iree_ok_status()));
    loom_serve_package_release(std::exchange(package_, nullptr));
  }

  // Public model replacement capability, valid through candidate disposal.
  loom_serve_execution_update_t update_ = {};
  // Terminal compiler payload, independently retained by model selections.
  void* candidate_ = nullptr;
  // Provider package retired immediately after compilation finishes.
  loom_serve_package_t* package_ = nullptr;
  // Joins the real asynchronous compiler's final callback with the outer test.
  std::mutex mutex_;
  // Outer-thread notification; no serving worker waits for compilation.
  std::condition_variable condition_;
  // Callback has relinquished all borrowed inputs.
  bool ready_ = false;
  // Owned compiler outcome, consumed after callback publication.
  iree_status_t status_ = iree_ok_status();
};

TEST_P(GenerationSelectionTest,
       PreparedBatchPinsCodeAcrossSelectionAndRemoval) {
  ASSERT_NO_FATAL_FAILURE(CompileReplacement());
  ASSERT_NO_FATAL_FAILURE(Prepare({R"({"model":"fixture","input":"a"})"}));
  IREE_ASSERT_OK(update_.select(update_.self, candidate_));
  ASSERT_NO_FATAL_FAILURE(SubmitPrepared(0));
  ASSERT_NO_FATAL_FAILURE(Complete(0));

  ASSERT_NO_FATAL_FAILURE(Prepare({R"({"model":"fixture","input":"a"})"}));
  IREE_ASSERT_OK(update_.select(update_.self, nullptr));
  update_.dispose(std::exchange(candidate_, nullptr));
  ASSERT_NO_FATAL_FAILURE(SubmitPrepared(1));
  ASSERT_NO_FATAL_FAILURE(Complete(1));

  IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &consumer_));
  for (size_t i = 0; i < results_.size(); ++i) {
    auto ready = loom_serve_finite_result_ready(results_[i]);
    IREE_ASSERT_OK(loom_serve_execution_barrier(
        consumer_, {2, ready.semaphores, ready.values}));
    const auto row = loom_serve_finite_result_row(results_[i], 0, 0);
    ASSERT_EQ(row.length, 4 * sizeof(float));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = row.buffer,
                .source_offset = row.offset,
                .target = output_.data() + i * 4,
                .length = row.length,
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(consumer_, 1, &download, &value));
  }
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_));
  const float expected[] = {1, 1, 1, 1, 2, 1, 2, 1};
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(output_[i], expected[i]);
  }
  StartRouter();
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(CommandAndDirect, GenerationSelectionTest,
                         ::testing::Values(false, true));

class RequestChoiceTest : public GenerationSelectionTest {
 protected:
  void SetUp() override {
    DeviceResultTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    if (SourceServiceDirectStageSource()[0] && !GetParam()) {
      GTEST_SKIP() << "This target provides resident direct stages only.";
    }
    ASSERT_NO_FATAL_FAILURE(Load({2, 16 * 1024 * 1024}));
  }

  void TearDown() override {
    ReleaseCompiler();
    if (batch_active_) {
      IREE_EXPECT_OK(Resolve(true));
      loom_serve_finite_model_release_batch(model_);
      ReleaseInputs();
    }
    loom_serve_package_release(package_);
    GenerationSelectionTest::TearDown();
  }

  void Load(loom_serve_variant_cache_options_t cache,
            bool reject_two_units = false,
            std::array<uint32_t, 2> capacity_factors = {20, 16}) {
    interface_.set_observer(interface_.self, {});
    loom_serve_finite_service_free(std::exchange(service_, nullptr));
    IREE_ASSERT_OK(
        loom_serve_finite_model_free(std::exchange(model_, nullptr)));
    const std::string stage = GetParam()
                                  ? (SourceServiceDirectStageSource()[0]
                                         ? SourceServiceDirectStageSource()
                                         : directory_ + "/variants/direct.loom")
                                  : directory_ + "/embed.loom";
    const std::string paths[] = {
        directory_ + "/config.loom",  stage,
        directory_ + "/control.loom", directory_ + "/variants/specialize.loom",
        directory_ + "/prepare.loom", directory_ + "/tokenizer.json"};
    const iree_string_view_t names[] = {
        IREE_SV("model/config.loom"),       IREE_SV("model/stage.loom"),
        IREE_SV("model/control_base.loom"), IREE_SV("model/control.loom"),
        IREE_SV("model/prepare.loom"),      IREE_SV("model/tokenizer.json")};
    std::string contents[IREE_ARRAYSIZE(paths)];
    loom_serve_package_file_t files[IREE_ARRAYSIZE(paths)] = {};
    for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
      std::ifstream input(paths[i]);
      ASSERT_TRUE(input.good()) << paths[i];
      contents[i].assign(std::istreambuf_iterator<char>(input), {});
    }
    // Only cold admission capacities change. Packet preparation, row plans,
    // stage configuration, and actual command workspace remain identical.
    const std::array<uint32_t, 2> default_factors = {20, 16};
    for (size_t i = 0; i < capacity_factors.size(); ++i) {
      const std::string original =
          "scalar.constant " + std::to_string(default_factors[i]) + " : i64";
      const auto offset = contents[4].find(original);
      ASSERT_NE(offset, std::string::npos);
      contents[4].replace(
          offset, original.size(),
          "scalar.constant " + std::to_string(capacity_factors[i]) + " : i64");
    }
    const auto position = contents[2].find("@prepare_batch(");
    ASSERT_NE(position, std::string::npos);
    contents[2].replace(position, 15, "@prepare_batch_base(");
    if (reject_two_units) {
      const std::string binding =
          "  func.call @config_i64(%stage, %key, %units64)";
      const auto offset = contents[3].find(binding);
      ASSERT_NE(offset, std::string::npos);
      contents[3].replace(offset, binding.size(), R"(
  %c2_i64 = scalar.constant 2 : i64
  %bad = scalar.cmpi eq, %units64, %c2_i64 : i64
  %compile_units = scf.if %bad -> (i64) {
    scf.yield %c0_i64 : i64
  } else {
    scf.yield %units64 : i64
  }
  func.call @config_i64(%stage, %key, %compile_units))");
    }
    if (GetParam()) {
      for (size_t i : {3u, 4u}) {
        size_t offset = 0;
        while ((offset = contents[i].find("@stage(", offset)) !=
               std::string::npos) {
          contents[i].replace(offset, 7, "@dispatch(");
          offset += 10;
        }
        const std::string root = "bytes(\"656d626564\")";
        offset = contents[i].find(root);
        ASSERT_NE(offset, std::string::npos);
        contents[i].replace(offset, root.size(),
                            "bytes(\"7265736964656e745f656d626564\")");
      }
    }
    for (size_t i = 0; i < IREE_ARRAYSIZE(paths); ++i) {
      files[i] = {names[i], iree_make_const_byte_span(contents[i].data(),
                                                      contents[i].size())};
    }
    IREE_ASSERT_OK(loom_serve_package_create(IREE_SV("model/model.json"),
                                             IREE_ARRAYSIZE(files), files,
                                             host_allocator_, &package_));
    const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                            IREE_SV("stage.loom"),
                                            IREE_SV("control_base.loom")};
    const loom_serve_finite_model_options_t options = {
        .sources =
            {
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = IREE_ARRAYSIZE(providers),
                .values = providers,
                .package = package_,
            },
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}"),
        .specialization_cache = cache,
    };
    loom_serve_test_embedding_model_lifecycle_t load;
    IREE_ASSERT_OK(load.Run(device_, options, {this, Allocate}));
    model_ = load.model;
    transfer_value_ = 0;
    update_ = loom_serve_finite_model_execution_update(model_);
    loom_serve_package_release(std::exchange(package_, nullptr));
    ASSERT_NO_FATAL_FAILURE(CreateService(0));
    loom_serve_finite_model_set_observer(model_, {Wake, this});
  }

  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<RequestChoiceTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        std::this_thread::get_id() != test->owner_) {
      ++test->worker_allocations_;
      std::unique_lock<std::mutex> lock(test->mutex_);
      if (test->holding_) {
        test->entered_ = true;
        test->condition_.notify_all();
        test->condition_.wait(lock, [&] { return !test->holding_; });
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  static void Wake(void* self) {
    auto* test = static_cast<RequestChoiceTest*>(self);
    std::lock_guard<std::mutex> lock(test->mutex_);
    ++test->wakes_;
    test->condition_.notify_all();
  }

  void PrepareChoice(const char* body) {
    ASSERT_NO_FATAL_FAILURE(Prepare({body}));
    batch_active_ = true;
  }

  iree_status_t Resolve(bool cancel = false) {
    bool complete = false;
    while (!complete) {
      const auto previous = wakes_.load();
      IREE_RETURN_IF_ERROR(
          loom_serve_finite_model_advance_batch(model_, cancel, &complete));
      if (!complete) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return wakes_.load() != previous; });
      }
    }
    return iree_ok_status();
  }

  void RunChoice(const char* body, bool reuse,
                 std::initializer_list<float> expected) {
    const auto before = worker_allocations_.load();
    ASSERT_NO_FATAL_FAILURE(PrepareChoice(body));
    IREE_ASSERT_OK(Resolve());
    EXPECT_EQ(worker_allocations_.load() == before, reuse);
    ASSERT_NO_FATAL_FAILURE(CheckChoice(expected));
  }

  void CheckChoice(std::initializer_list<float> expected) {
    ASSERT_NO_FATAL_FAILURE(SubmitPrepared(0));
    loom_serve_finite_snapshot_t* snapshot = nullptr;
    loom_serve_execution_frontier_t ready;
    IREE_ASSERT_OK(
        loom_serve_finite_result_download(results_[0], &snapshot, &ready));
    transfer_value_ = ready.values[1];
    for (size_t i = 0; i < IREE_ARRAYSIZE(ready.semaphores); ++i) {
      IREE_EXPECT_OK(iree_hal_semaphore_wait(
          ready.semaphores[i], ready.values[i], iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE));
    }
    const auto row = loom_serve_finite_snapshot_row(snapshot, 0, 0);
    EXPECT_EQ(row.data_length, expected.size() * sizeof(float));
    if (row.data_length == expected.size() * sizeof(float)) {
      size_t i = 0;
      for (float value : expected) {
        float actual = 0;
        memcpy(&actual, row.data + i++ * sizeof(float), sizeof(actual));
        EXPECT_EQ(actual, value);
      }
    }
    loom_serve_finite_snapshot_release(snapshot);
    ASSERT_NO_FATAL_FAILURE(Complete(0));
    batch_active_ = false;
    loom_serve_finite_result_release(std::exchange(results_[0], nullptr));
  }

  void ReleaseCompiler() {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = false;
    condition_.notify_all();
  }

  // Outer test thread drives serialized preparation/admission, not compilation.
  const std::thread::id owner_ = std::this_thread::get_id();
  // Actual asynchronous compiler allocations, used to distinguish reuse/misses.
  std::atomic<size_t> worker_allocations_ = 0;
  // Compiler completion wakes, joined outside the serving worker pool.
  std::atomic<uint64_t> wakes_ = 0;
  // Publication and a deliberate allocation-boundary dependency gate.
  std::mutex mutex_;
  // Notifies compiler entry, dependency release, and terminal publication.
  std::condition_variable condition_;
  // Compiler allocations are held only during explicit cancellation coverage.
  bool holding_ = false;
  // A real compiler worker has reached the held allocation boundary.
  bool entered_ = false;
  // Preparation succeeded; teardown must collect/release this invocation.
  bool batch_active_ = false;
  // Temporary input package, released before the first request specialization.
  loom_serve_package_t* package_ = nullptr;
};

TEST_P(RequestChoiceTest, ShapeReuseAndLruEvictionSurviveProviderRetirement) {
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"a"})", false, {1, 1, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"bbb"})", false, {6, 3, 2, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"a"})", true, {1, 1, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"ab"})", false, {3, 2, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"a"})", true, {1, 1, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"bbb"})", false, {6, 3, 2, 1}));
  StartRouter();
  const auto response =
      Read(Send(R"({"model":"fixture","input":["a","bbb"]})"));
  EXPECT_NE(response.find(R"("embedding":[1,1,1,2])"), std::string::npos);
  EXPECT_NE(response.find(R"("embedding":[6,3,2,2])"), std::string::npos);
}

TEST_P(RequestChoiceTest, OversizeChoiceRunsWithoutBecomingIdleCacheStorage) {
  ASSERT_NO_FATAL_FAILURE(Load({2, 1}));
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        RunChoice(R"({"model":"fixture","input":"a"})", false, {1, 1, 1, 1}));
  }
}

TEST_P(RequestChoiceTest, AdmissionCeilingsDoNotSizeWorkspaceBacking) {
  constexpr uint32_t kCapacityFactor = 2048;
  constexpr uint64_t kCapacityBytes = 64 * kCapacityFactor;
  const auto run = [&] {
    ASSERT_NO_FATAL_FAILURE(
        RunChoice(R"({"model":"fixture","input":"a"})", false, {1, 1, 1, 1}));
    const auto workspace =
        loom_serve_device_memory_statistics(device_).workspace;
    EXPECT_GT(workspace.bytes_committed, 0u);
    EXPECT_LT(workspace.bytes_committed, kCapacityBytes);
  };
  ASSERT_NO_FATAL_FAILURE(run());
  // Enlarge each independent admission ceiling without changing the work.
  ASSERT_NO_FATAL_FAILURE(
      Load({2, 16 * 1024 * 1024}, false, {kCapacityFactor, 16}));
  ASSERT_NO_FATAL_FAILURE(run());
  ASSERT_NO_FATAL_FAILURE(
      Load({2, 16 * 1024 * 1024}, false, {20, kCapacityFactor}));
  ASSERT_NO_FATAL_FAILURE(run());
  // Actual cohort growth still acquires its required backing on demand.
  ASSERT_NO_FATAL_FAILURE(PrepareChoice(
      R"({"model":"fixture","input":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]})"));
  IREE_ASSERT_OK(Resolve());
  ASSERT_NO_FATAL_FAILURE(CheckChoice({31, 31, 1, 2}));
  EXPECT_LT(
      loom_serve_device_memory_statistics(device_).workspace.bytes_committed,
      kCapacityBytes);
}

TEST_P(RequestChoiceTest, CancellationCollectsAcceptedCompilerBeforeRelease) {
  ASSERT_NO_FATAL_FAILURE(PrepareChoice(R"({"model":"fixture","input":"a"})"));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = true;
  }
  bool complete = false;
  IREE_ASSERT_OK(
      loom_serve_finite_model_advance_batch(model_, false, &complete));
  EXPECT_FALSE(complete);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  IREE_ASSERT_OK(
      loom_serve_finite_model_advance_batch(model_, true, &complete));
  EXPECT_FALSE(complete);
  ReleaseCompiler();
  IREE_ASSERT_OK(Resolve(true));
  loom_serve_finite_model_release_batch(model_);
  ReleaseInputs();
  batch_active_ = false;
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"a"})", true, {1, 1, 1, 1}));
}

TEST_P(RequestChoiceTest, CapturedGenerationSurvivesReplacementAndRemoval) {
  ASSERT_NO_FATAL_FAILURE(CompileReplacement());
  ASSERT_NO_FATAL_FAILURE(PrepareChoice(R"({"model":"fixture","input":"a"})"));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    holding_ = true;
  }
  bool complete = false;
  IREE_ASSERT_OK(
      loom_serve_finite_model_advance_batch(model_, false, &complete));
  EXPECT_FALSE(complete);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  IREE_ASSERT_OK(update_.select(update_.self, candidate_));
  ReleaseCompiler();
  IREE_ASSERT_OK(Resolve());
  ASSERT_NO_FATAL_FAILURE(CheckChoice({1, 1, 1, 1}));

  ASSERT_NO_FATAL_FAILURE(PrepareChoice(R"({"model":"fixture","input":"a"})"));
  IREE_ASSERT_OK(update_.select(update_.self, nullptr));
  update_.dispose(std::exchange(candidate_, nullptr));
  IREE_ASSERT_OK(Resolve());
  ASSERT_NO_FATAL_FAILURE(CheckChoice({2, 1, 2, 1}));
  ASSERT_NO_FATAL_FAILURE(
      RunChoice(R"({"model":"fixture","input":"a"})", true, {1, 1, 1, 1}));
}

TEST_P(RequestChoiceTest, FailedCompilationReleasesChoiceAndKeepsSourceIndex) {
  ASSERT_NO_FATAL_FAILURE(Load({2, 16 * 1024 * 1024}, true));
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        PrepareChoice(R"({"model":"fixture","input":"a"})"));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Resolve());
    loom_serve_finite_model_release_batch(model_);
    ReleaseInputs();
    batch_active_ = false;
    ASSERT_NO_FATAL_FAILURE(RunChoice(R"({"model":"fixture","input":"bbb"})",
                                      i != 0, {6, 3, 2, 1}));
  }
}

TEST_P(RequestChoiceTest, ZeroCapacityRunsWithoutIdleRetention) {
  ASSERT_NO_FATAL_FAILURE(Load({0, 16 * 1024 * 1024}));
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        RunChoice(R"({"model":"fixture","input":"a"})", false, {1, 1, 1, 1}));
  }
}

INSTANTIATE_TEST_SUITE_P(CommandAndDirect, RequestChoiceTest,
                         ::testing::Values(false, true));

TEST_F(DeviceResultTest, RaggedCohortsComposeBeforeProducerRetirement) {
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  ASSERT_NO_FATAL_FAILURE(
      Submit({R"({"model":"fixture","input":["a","bbb"]})",
              R"({"model":"fixture","input":[[2,0]],"dimensions":2})"},
             0));
  EXPECT_EQ(loom_serve_finite_result_length(results_[0]), 40u);
  EXPECT_EQ(loom_serve_finite_result_output_length(results_[0]), 1024u);
  EXPECT_EQ(loom_serve_finite_result_input_tokens(results_[0], 0), 4u);
  EXPECT_EQ(loom_serve_finite_result_input_tokens(results_[0], 1), 2u);
  ASSERT_NO_FATAL_FAILURE(Complete(0));
  ASSERT_NO_FATAL_FAILURE(Submit({R"({"model":"fixture","input":[0]})"}, 1));

  IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &consumer_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  auto ready = loom_serve_finite_result_ready(results_[1]);
  iree_hal_semaphore_t* waits[] = {ready.semaphores[0], gate_};
  uint64_t values[] = {ready.values[0], 1};
  IREE_ASSERT_OK(loom_serve_execution_barrier(consumer_, {2, waits, values}));
  const auto reservation =
      iree_hal_make_pool_reservation_request(sizeof(output_), 16);
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(consumer_, &reservation, &target_));

  std::array<iree_hal_transfer_operation_t, 3> copies = {};
  const std::array<std::array<size_t, 2>, 3> rows = {{{0, 0}, {0, 1}, {1, 0}}};
  iree_device_size_t offset = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto row =
        loom_serve_finite_result_row(results_[0], rows[i][0], rows[i][1]);
    copies[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
        .copy =
            {
                .source_buffer = row.buffer,
                .source_offset = row.offset,
                .target_buffer = target_,
                .target_offset = offset,
                .length = row.length,
            },
    };
    offset += row.length;
  }
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(consumer_, copies.size(),
                                               copies.data(), &value));

  // Enqueue the consumer before observing the second producer's completion.
  // Retaining the first cohort did not prevent producing the next one.
  const auto row = loom_serve_finite_result_row(results_[1], 0, 0);
  const iree_hal_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = row.buffer,
              .source_offset = row.offset,
              .target_buffer = target_,
              .target_offset = offset,
              .length = row.length,
          },
  };
  IREE_ASSERT_OK(loom_serve_execution_transfer(consumer_, 1, &copy, &value));
  ASSERT_NO_FATAL_FAILURE(Complete(1));
  EXPECT_EQ(loom_serve_finite_result_input_tokens(results_[0], 0), 4u);
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(model_, nullptr)));
  for (auto*& result : results_) {
    loom_serve_finite_result_release(std::exchange(result, nullptr));
  }
  // Neither destruction of the producer nor final caller result release may
  // release the allocation captured by the still-blocked consumer.
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = target_,
              .target = output_.data(),
              .length = sizeof(output_),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(consumer_, 1, &download, &value));
  IREE_ASSERT_OK(loom_serve_execution_dealloca(consumer_, target_, &value));
  iree_hal_buffer_release(std::exchange(target_, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_));
  const std::array<float, 14> expected = {1, 1, 1, 3, 6, 3, 2,
                                          3, 4, 2, 1, 1, 1, 1};
  EXPECT_EQ(output_, expected);
  IREE_ASSERT_OK(loom_serve_device_trim(device_, 0));
  EXPECT_EQ(
      loom_serve_device_memory_statistics(device_).workspace.reservation_count,
      0u);
}

}  // namespace loom_serve::testing::embedding
