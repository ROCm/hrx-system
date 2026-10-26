// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/stalled_stderr.h"

namespace loom_serve::testing::embedding {

using loom_serve::testing::StalledStderr;

TEST_F(SourceEmbeddingServiceTest,
       ReconfiguresCommandsWithoutPreparingParametersAgain) {
  const std::string directory =
      iree::testing::MakeTempFilePath("variant_weights");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(directory));
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(model_, nullptr)));
  ASSERT_NO_FATAL_FAILURE(CreateModel(directory));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  auto* residency = loom_serve_finite_model_residency(model_);
  IREE_ASSERT_OK(loom_serve_residency_activate(residency));
  const auto weights = loom_serve_residency_activity(residency);
  StartRouter();
  const auto inventory =
      Read(Send({}, "/v1/models/variants", "GET", "X-Loom-Model: fixture\r\n"));
  ASSERT_EQ(inventory.find("HTTP/1.1 200"), 0u) << inventory;
  const uint64_t original = Number(inventory, "selected");
  const std::string request = R"({"model":"fixture","input":"a"})";
  EXPECT_NE(Read(Send(request)).find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  const auto staged = UploadVariant(original, directory, "embed.loom",
                                    "scalar.constant 2 : i32", true);
  ASSERT_EQ(staged.find("HTTP/1.1 200"), 0u) << staged;
  const uint64_t candidate = Number(staged, "id", "variant");
  EXPECT_EQ(Number(staged, "selected"), original);
  EXPECT_NE(Read(Send(request)).find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  const auto selected = SelectVariant(candidate, original);
  ASSERT_EQ(selected.find("HTTP/1.1 200"), 0u) << selected;
  const auto changed = Read(Send(request));
  EXPECT_NE(changed.find(R"("embedding":[4,2,4,2])"), std::string::npos)
      << changed;
  EXPECT_EQ(SelectVariant(original, original).find("HTTP/1.1 409"), 0u);
  ASSERT_EQ(SelectVariant(original, candidate).find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(Send(request)).find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  const auto failed = UploadVariant(original, directory, "embed.loom",
                                    "scalar.not_an_operation 1 : i32", true);
  EXPECT_EQ(failed.find("HTTP/1.1 422"), 0u) << failed;
  EXPECT_NE(failed.find("not_an_operation"), std::string::npos);
  EXPECT_EQ(Number(failed, "selected"), original);
  EXPECT_NE(Read(Send(request)).find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto retained = loom_serve_residency_activity(residency);
  EXPECT_EQ(retained.activations, weights.activations);
  EXPECT_EQ(retained.read_bytes, weights.read_bytes);
  EXPECT_EQ(retained.prepared_bytes, weights.prepared_bytes);
}

TEST_F(SourceEmbeddingServiceTest,
       UploadedPolicyPreparesServerLocalCheckpoint) {
  const std::string directory =
      iree::testing::MakeTempFilePath("uploaded_checkpoint_package");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(directory));
  const std::string checkpoints =
      iree::testing::MakeTempFilePath("uploaded_checkpoint");
  ASSERT_TRUE(std::filesystem::create_directory(checkpoints));
  ASSERT_TRUE(std::filesystem::copy_file(directory + "/weights.safetensors",
                                         checkpoints + "/weights.safetensors"));
  StartManagedRouter();
  {
    loom_serve::testing::SourcePackage package;
    for (const auto* file :
         {"prepare.loom", "control.loom", "config.loom", "embed.loom",
          "weights.loom", "policy.loom", "tokenizer.json"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory + "/" + file));
    }
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request(
            "/v1/models/register",
            Declaration("checkpoint", directory, "drained", checkpoints)),
        &client));
    const auto registered = Read(client);
    ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
  }
  ASSERT_EQ(Read(Send(R"({"model":"checkpoint"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  const auto response = Read(Send(R"({"model":"checkpoint","input":"a"})"));
  EXPECT_NE(response.find(R"("embedding":[2,2,2,2])"), std::string::npos)
      << response << ModelStatus("checkpoint");
  // A model unregister joins accepted compiler borrowers even after the
  // staging response and uploaded source package have left the client.
  const uint64_t base = Number(Read(Send({}, "/v1/models/variants", "GET",
                                         "X-Loom-Model: checkpoint\r\n")),
                               "selected");
  {
    loom_serve::testing::SourcePackage package;
    for (const char* file : {"config.loom", "embed.loom", "weights.loom"}) {
      IREE_ASSERT_OK(package.AddFile(std::string("model/") + file,
                                     directory + "/" + file));
    }
    int client = -1;
    IREE_ASSERT_OK(http_.Send(
        package.Request(
            "/v1/models/variants",
            "{\"model\":\"checkpoint\",\"base\":" + std::to_string(base) +
                ",\"sources\":[\"config.loom\",\"embed.loom\",\"weights.loom\"]"
                ",\"wait\":false}"),
        &client));
    const auto accepted = Read(client);
    ASSERT_EQ(accepted.find("HTTP/1.1 202"), 0u) << accepted;
  }
  ASSERT_EQ(Read(Send(R"({"model":"checkpoint"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("checkpoint");
}

TEST_F(SourceEmbeddingServiceTest,
       ImpossibleWorkingSetRejectsBeforeActivation) {
  const std::string package =
      iree::testing::MakeTempFilePath("working_set_checkpoint");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  auto declaration = Declaration("large", package);
  const std::string cohort = "\"batch_units\":" + std::to_string(BatchUnits());
  const auto offset = declaration.find(cohort);
  ASSERT_NE(offset, std::string::npos);
  declaration.replace(offset, cohort.size(), R"("batch_units":256)");
  const auto registered = Read(Send(declaration, "/v1/models/register"));
  ASSERT_EQ(registered.find("HTTP/1.1 202"), 0u) << registered;
  const auto response = Read(Send(R"({"model":"large","input":"a"})"));
  if (loom_serve_device_memory_pool(device_)) {
    ASSERT_EQ(response.find("HTTP/1.1 503"), 0u) << response;
    EXPECT_NE(response.find("invocation working set"), std::string::npos);
    const auto state = ModelStatus("large");
    EXPECT_EQ(state.find(R"("error":)"), std::string::npos) << state;
    EXPECT_NE(state.find(R"("activations":0)"), std::string::npos) << state;
    EXPECT_NE(state.find(R"("read_bytes":0)"), std::string::npos) << state;
  } else {
    ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    EXPECT_NE(response.find(R"("embedding":[2,2,2,2])"), std::string::npos);
  }
  // The same checkpoint and kernels fit with the ordinary cohort bound. A
  // capacity rejection leaves registration and the other service usable.
  ASSERT_EQ(Read(Send(Declaration("small", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const auto next = Read(Send(R"({"model":"small","input":"a"})"));
  ASSERT_EQ(next.find("HTTP/1.1 200"), 0u) << next;
  EXPECT_NE(next.find(R"("embedding":[2,2,2,2])"), std::string::npos);
}

TEST_F(SourceEmbeddingServiceTest,
       CheckpointRegistrationOwnsItsOperationAndIncarnation) {
  const std::string package =
      iree::testing::MakeTempFilePath("dynamic_checkpoint");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  const std::string declaration = Declaration("checkpoint", package, "drained");
  const auto registered = Read(Send(declaration, "/v1/models/register"));
  EXPECT_EQ(registered.find("HTTP/1.1 202"), 0u);
  EXPECT_NE(registered.find(R"("registration":"registered")"),
            std::string::npos);
  EXPECT_NE(registered.find(R"("admission":"drained")"), std::string::npos);
  EXPECT_EQ(Read(Send(declaration, "/v1/models/register")).find("HTTP/1.1 409"),
            0u);
  EXPECT_EQ(
      Read(Send(Declaration("cold", package, "drained", package + "/absent"),
                "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_EQ(Read(Send(Declaration("full", package), "/v1/models/register"))
                .find("HTTP/1.1 429"),
            0u);
  EXPECT_NE(ModelStatus("cold").find(R"("registration":"registered")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("cold");

  const int held = Send(R"({"model":"checkpoint","input":"a"})");
  WaitModelState("checkpoint", R"("held_requests":1)");
  const int tokenizer =
      Send(R"({"model":"checkpoint","content":"ab"})", "/tokenize");
  WaitModelState("checkpoint", R"("held_requests":2)");
  // The accepted operation belongs to the registry, not this unread response.
  const int admin = Send(R"({"model":"checkpoint"})", "/v1/models/undrain");
  WaitModelState("checkpoint", R"("admission":"open")");
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(admin, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)), 0);
  ASSERT_EQ(close(admin), 0);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  const std::string embedding_response = Read(held);
  const std::string tokenizer_response = Read(tokenizer);
  ASSERT_NE(embedding_response.find(R"("embedding":[2,2,2,2])"),
            std::string::npos)
      << embedding_response << ModelStatus("checkpoint");
  ASSERT_NE(tokenizer_response.find(R"({"tokens":[0,1]})"), std::string::npos)
      << tokenizer_response;
  const auto ready = WaitModelState("checkpoint", R"("registration":"ready")");
  EXPECT_NE(ready.find("\"workers\":1"), std::string::npos);
  const uint64_t completed = CompletedRequests();
  EXPECT_EQ(completed, 2u);
  EXPECT_NE(Read(Send({}, "/metrics", "GET"))
                .find("loom_serve_model_completed_total{model=\"checkpoint\","
                      "kind=\"embedding\",incarnation=\"1\"} 1\n"),
            std::string::npos);
  const std::string logs =
      Read(Send({}, "/logz", "GET", "X-Loom-Model: checkpoint\r\n"));
  const std::string prefix = R"({"event":"cursor","value":")";
  const size_t cursor_begin = logs.find(prefix);
  ASSERT_NE(cursor_begin, std::string::npos);
  const size_t value_begin = cursor_begin + prefix.size();
  const std::string cursor =
      logs.substr(value_begin, logs.find('"', value_begin) - value_begin);
  EXPECT_EQ(Read(Send(R"({"model":"checkpoint","mode":"drain"})",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("checkpoint");
  EXPECT_EQ(CompletedRequests(), completed);
  EXPECT_EQ(
      Read(Send(Declaration("checkpoint", package), "/v1/models/register"))
          .find("HTTP/1.1 202"),
      0u);
  EXPECT_NE(Read(Send(R"({"model":"checkpoint","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  EXPECT_EQ(Read(Send({}, "/logz", "GET",
                      "X-Loom-Model: checkpoint\r\nX-Loom-Cursor: " + cursor +
                          "\r\n"))
                .find("HTTP/1.1 409"),
            0u);
  EXPECT_EQ(CompletedRequests(), completed + 1);
  const std::string metrics = Read(Send({}, "/metrics", "GET"));
  EXPECT_EQ(metrics.find("incarnation=\"1\""), std::string::npos);
  EXPECT_NE(
      metrics.find("loom_serve_model_completed_total{model=\"checkpoint\","
                   "kind=\"embedding\",incarnation=\"3\"} 1\n"),
      std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"checkpoint","mode":"abort"})",
                      "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("checkpoint");
}

TEST_F(SourceEmbeddingServiceTest,
       InvalidStageAfterExecutionRetiresManagedRequests) {
  const std::string package = iree::testing::MakeTempFilePath("invalid_stage");
  ASSERT_TRUE(std::filesystem::create_directory(package));
  for (const char* file : {"config.loom", "control.loom", "embed.loom",
                           "prepare.loom", "model.json", "tokenizer.json"}) {
    ASSERT_TRUE(std::filesystem::copy_file(directory_ + "/" + file,
                                           package + "/" + file));
  }
  {
    const std::string path = package + "/control.loom";
    std::ifstream input(path);
    ASSERT_TRUE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const std::string original =
        "  %work = func.call @execute_2(%stage, %input, %output) : "
        "(i32, hal.buffer, hal.buffer) -> (i64)";
    const size_t position = source.find(original);
    ASSERT_NE(position, std::string::npos);
    // One valid command has been accepted before the source selects an
    // undeclared second stage. Retirement must join that accepted command.
    source.insert(position + original.size(),
                  "\n  %invalid = scalar.constant 4096 : i32\n"
                  "  %rejected = func.call @execute_2(%invalid, %input, "
                  "%output) : (i32, hal.buffer, hal.buffer) -> (i64)");
    input.close();
    std::filesystem::permissions(path, std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::ofstream output(path, std::ios::trunc);
    output << source;
    ASSERT_TRUE(output.good());
  }
  CheckInvalidSubmissionRetirement(package);
}

TEST_F(SourceEmbeddingServiceTest,
       FailedRegistrationTransportDoesNotWaitForUnsubmittedWork) {
  fail_listener_allocations_.store(true);
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::
            INTERCEPT_ONLY_CURRENT_THREAD,
        &failures);
    CheckInvalidSubmissionRetirement(directory_);
  }
  ASSERT_EQ(failures.size(), 2);
  EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
  EXPECT_NE(std::string(failures.GetTestPartResult(0).message())
                .find("registration transport witness"),
            std::string::npos);
  EXPECT_TRUE(failures.GetTestPartResult(1).fatally_failed());
  EXPECT_TRUE(http_.failed());
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  EXPECT_EQ(retired_counters_.requests, 0u);
}

TEST_F(SourceEmbeddingServiceTest,
       FailedRegistrationPublishesEvidenceAndReleasesHeldClaims) {
  const std::string package = iree::testing::MakeTempFilePath("failed_dynamic");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  {
    std::filesystem::permissions(package + "/embed.loom",
                                 std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::add);
    std::ofstream source(package + "/embed.loom", std::ios::trunc);
    source << "this is not a source program\n";
    ASSERT_TRUE(source.good());
  }
  StartManagedRouter();
  StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(Declaration("broken", package, "drained"),
                      "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  const int held = Send(R"({"model":"broken","input":"a"})");
  WaitModelState("broken", R"("held_requests":1)");
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 503"), 0u);
  const auto failed = WaitModelState("broken", R"("registration":"failed")");
  EXPECT_NE(failed.find("embed.loom"), std::string::npos);
  EXPECT_NE(failed.find(R"("diagnostics":[{)"), std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/undrain"))
                .find("HTTP/1.1 409"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"broken"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("broken");
}

TEST_F(SourceEmbeddingServiceTest,
       LogReadersWrapIndependentlyWhileStderrIsFull) {
  StartRouter();
  auto poll = [&](const std::string& cursor) {
    return Read(
        Send("", "/logz", "GET",
             "X-Loom-Model: fixture\r\n" +
                 (cursor.empty() ? std::string()
                                 : "X-Loom-Cursor: " + cursor + "\r\n")));
  };
  auto cursor_of = [](const std::string& response) {
    const std::string marker = R"({"event":"cursor","value":")";
    const size_t start = response.find(marker);
    EXPECT_NE(start, std::string::npos);
    if (start == std::string::npos) {
      return std::string();
    }
    const size_t first = start + marker.size();
    return response.substr(first, response.find('"', first) - first);
  };
  const std::string initial = poll("");
  EXPECT_EQ(initial.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(initial.find(R"("event":"stream_begin","sequence":0)"),
            std::string::npos);
  EXPECT_NE(initial.find(R"("partial":false)"), std::string::npos);
  const std::string slow = cursor_of(initial);
  std::string fast = slow;
  StalledStderr stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Begin());
  EXPECT_EQ(Read(Send(R"({"model":7})")).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Read(Send(R"({"model":"fixture","tokens":[-1]})", "/detokenize"))
                .find("HTTP/1.1 400"),
            0u);
  const std::string diagnostics = Read(Send("", "/logz", "GET"));
  EXPECT_NE(diagnostics.find(R"("scope":"server","counters_available":true)"),
            std::string::npos);
  EXPECT_NE(diagnostics.find(R"("event":"error","sequence":1)"),
            std::string::npos);
  EXPECT_NE(diagnostics.find(R"("event":"error","sequence":2)"),
            std::string::npos);
  EXPECT_NE(
      diagnostics.find(
          R"("values":{"requests":0,"completed":0,"rejected":0,"cancelled":0,"failed":0,"input_tokens":0,"output_tokens":0,"batches":0})"),
      std::string::npos);
  EXPECT_EQ(Read(Send("", "/logz", "GET", "X-Loom-Cursor: " + slow + "\r\n"))
                .find("HTTP/1.1 409"),
            0u);
  uint64_t next = 1;
  std::string latest;
  for (uint64_t i = 0; i < 40; ++i) {
    const std::string inference =
        Read(Send(R"({"model":"fixture","input":["a","bbb"],"dimensions":2})"));
    ASSERT_EQ(inference.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(inference.find(R"("embedding":[1,1])"), std::string::npos);
    EXPECT_NE(inference.find(R"("embedding":[6,3])"), std::string::npos);
    latest = poll(fast);
    EXPECT_EQ(latest.find(R"("event":"gap")"), std::string::npos);
    // Publication is independent of device completion and terminal retirement.
    // Counters and HTTP polls are not inserted into the producer's stream.
    size_t offset = 0;
    for (const char* event : {"request_begin", "batch_submit", "batch_end",
                              "first_output", "request_end"}) {
      const std::string marker = "\"event\":\"" + std::string(event) +
                                 "\",\"sequence\":" + std::to_string(next++) +
                                 ",";
      const auto found = latest.find(marker, offset);
      ASSERT_NE(found, std::string::npos);
      offset = found + marker.size();
    }
    EXPECT_EQ(latest.find("\"sequence\":", offset), std::string::npos);
    fast = cursor_of(latest);
    if (i % 8 == 0) {
      // Observation transport cancellation must not write a blocking diagnostic
      // or poison subsequent inference. The model stream remains unchanged.
      const int peer = Send("", "/logz", "GET");
      const linger reset = {1, 0};
      ASSERT_EQ(setsockopt(peer, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
                0);
      ASSERT_EQ(close(peer), 0);
    }
  }
  EXPECT_EQ(next, 201u);
  const std::string totals =
      R"("requests":40,"completed":40,"rejected":0,"cancelled":0,"failed":0,"input_tokens":160,"output_tokens":0,"batches":40)";
  EXPECT_NE(latest.find(totals), std::string::npos);
  EXPECT_NE(Read(Send("", "/logz", "GET")).find(totals), std::string::npos);
  const std::string resumed = poll(slow);
  EXPECT_NE(resumed.find(R"({"event":"gap","from":1,"to":73})"),
            std::string::npos);
  EXPECT_NE(resumed.find(totals), std::string::npos);
  const std::string joined = poll("");
  EXPECT_NE(joined.find(R"("partial":true)"), std::string::npos);
  EXPECT_NE(joined.find(R"({"event":"gap","from":0,"to":73})"),
            std::string::npos);
  const std::string caught_up = poll(fast);
  EXPECT_EQ(caught_up.find("\"sequence\":"), std::string::npos);
  EXPECT_EQ(cursor_of(caught_up), fast);
  EXPECT_NE(caught_up.find(totals), std::string::npos);
  EXPECT_EQ(poll("bad").find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(poll("0:0:1").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(poll(fast.substr(0, fast.find_last_of(':') + 1) + "999")
                .find("HTTP/1.1 400"),
            0u);
  EXPECT_EQ(Read(Send("", "/logz", "GET", "X-Loom-Model: missing\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Shutdown itself must also retire while no stderr consumer is running.
  loom_serve_http_server_request_stop(server_);
  IREE_EXPECT_OK(http_.Join());
}

}  // namespace loom_serve::testing::embedding
