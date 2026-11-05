// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/testing/stalled_stderr.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"

namespace loom_serve::testing::embedding {

using loom_runtime::testing::StalledStderr;

TEST_F(SourceEmbeddingServiceTest,
       CheckpointLoadSharesTheServingWorkerAndNeverWritesStderr) {
  const std::string package =
      iree::testing::MakeTempFilePath("live_checkpoint_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(package + "/model.json"));
  const loom_serve_finite_model_options_t options = {
      .sources = document.value().sources,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{}"),
  };
  const int resident = Send(R"({"model":"fixture","input":"a"})");
  ASSERT_NO_FATAL_FAILURE(Accept(resident));
  loom_serve_test_embedding_model_lifecycle_t load;
  load.service = &interface_;
  iree_status_t status;
  {
    StalledStderr stalled;
    ASSERT_NO_FATAL_FAILURE(stalled.Begin());
    status = load.Run(device_, storage_owner_, options);
  }
  retained_ = load.model;
  IREE_ASSERT_OK(status);
  ASSERT_NE(retained_, nullptr);
  EXPECT_NE(Read(resident).find(R"("embedding":[1,1,1,1])"), std::string::npos);
  EXPECT_EQ(load.report.commands, 2u);  // Consumer and in-place preparer.
  EXPECT_EQ(load.report.workers, 1u);
  EXPECT_EQ(load.report.diagnostic_count, 0u);

  // Loading indexes the checkpoint without activating it. First use doubles
  // the factor from 1.0 to 2.0 in place; later requests reuse that value.
  EXPECT_EQ(loom_runtime_residency_activity(
                loom_serve_finite_model_residency(retained_))
                .activations,
            0u);
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  std::swap(model_, retained_);
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  const int prepared = Send(R"({"model":"fixture","input":"a"})");
  ASSERT_NO_FATAL_FAILURE(Accept(prepared));
  Drain();
  EXPECT_NE(Read(prepared).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  const int changed =
      Send(R"({"model":"fixture","input":"bbb","dimensions":2})");
  ASSERT_NO_FATAL_FAILURE(Accept(changed));
  Drain();
  EXPECT_NE(Read(changed).find(R"("embedding":[12,6])"), std::string::npos);
  EXPECT_EQ(
      loom_runtime_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      1u);
}

TEST_F(SourceEmbeddingServiceTest,
       PreparedParameterCommandRetiresCancelledCohort) {
  const std::string package =
      iree::testing::MakeTempFilePath("cancelled_checkpoint_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  CheckAbandonedCohort(0, R"("embedding":[2,2,2,2])");
  EXPECT_EQ(
      loom_runtime_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      1u);
}

TEST_F(SourceEmbeddingServiceTest,
       AdmissionDrainPreservesPreparedParametersAndResumesHeldHttpRequests) {
  const std::string package =
      iree::testing::MakeTempFilePath("drained_checkpoint_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  EXPECT_EQ(
      loom_runtime_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      0u);
  StartRouter();
  const auto control = [this](const char* target) {
    return Read(Send(R"({"model":"fixture"})", target));
  };
  EXPECT_NE(control("/v1/models/drain").find(R"("admission":"drained")"),
            std::string::npos);
  std::vector<int> clients;
  for (int i = 0; i < 7; ++i) {
    clients.push_back(Send(R"({"model":"fixture","input":"a"})"));
  }
  const int tokens =
      Send(R"({"model":"fixture","content":"aba"})", "/tokenize");
  // The control reply acknowledges the actual owner cutoff; transport
  // arrivals may precede or follow each other across independent connections.
  std::string state;
  do {
    state = control("/v1/models/drain");
    ASSERT_EQ(state.find("HTTP/1.1 200"), 0u);
  } while (state.find(R"("held_requests":8,)") == std::string::npos);
  EXPECT_NE(state.find(R"("queued_requests":0,"active_requests":0)"),
            std::string::npos);
  EXPECT_NE(state.find(R"("tokenizer_requests":0)"), std::string::npos);
  EXPECT_EQ(
      Read(Send(R"({"model":"fixture","input":"a"})")).find("HTTP/1.1 429"),
      0u);
  EXPECT_NE(Read(Send("", "/metrics", "GET"))
                .find("loom_serve_server_completed_total 0\n"),
            std::string::npos);
  EXPECT_EQ(control("/v1/models/undrain").find("HTTP/1.1 200"), 0u);
  for (const int client : clients) {
    const auto response = Read(client);
    // The fourth oracle component reports the live cohort width. All seven
    // parked inference requests enter the same model batch on undrain.
    EXPECT_NE(response.find(R"("embedding":[2,2,2,14])"), std::string::npos)
        << response;
  }
  EXPECT_NE(Read(tokens).find(R"({"tokens":[0,1,0]})"), std::string::npos);

  EXPECT_NE(control("/v1/models/drain").find(R"("admission":"drained")"),
            std::string::npos);
  const int resumed = Send(R"({"model":"fixture","input":"a"})");
  do {
    state = control("/v1/models/drain");
    ASSERT_EQ(state.find("HTTP/1.1 200"), 0u);
  } while (state.find(R"("held_requests":1,)") == std::string::npos);
  EXPECT_EQ(control("/v1/models/undrain").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(resumed).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  EXPECT_NE(Read(Send("", "/metrics", "GET"))
                .find("loom_serve_server_completed_total 8\n"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  IREE_EXPECT_OK(http_.Join());
  // Reversible admission never evicted/reloaded the in-place prepared
  // parameter. Applying preparation twice would also change the exact output.
  EXPECT_EQ(
      loom_runtime_residency_activity(loom_serve_finite_model_residency(model_))
          .activations,
      1u);
}

}  // namespace loom_serve::testing::embedding
