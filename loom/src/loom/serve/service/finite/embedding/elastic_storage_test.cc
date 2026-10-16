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
#include "loom/serve/testing/native_allocator.h"
#include "loom/serve/testing/package.h"
#include "loom/serve/testing/stalled_stderr.h"

namespace loom_serve::testing::embedding {

using loom_serve::testing::StalledStderr;

class ElasticEmbeddingServiceTest : public SourceEmbeddingServiceTest {};

TEST_F(ElasticEmbeddingServiceTest,
       ParameterDeactivationDoesNotBlockWarmRequestsAndReloadsExactly) {
  const std::string package =
      iree::testing::MakeTempFilePath("parameter_release_progress");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NE(Read(Send(R"({"model":"cold","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  ASSERT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/drain"))
                .find("HTTP/1.1 200"),
            0u);
  {
    loom_serve::testing::ScopedNativeHold gate(device_,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    ASSERT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/deactivate"))
                  .find("HTTP/1.1 202"),
              0u);
    gate.WaitEntered();
    const auto retired = WaitModel("cold", R"("parameter_operation":"idle")");
    EXPECT_NE(retired.find(R"("committed_bytes":0)"), std::string::npos);
    // Model ownership can end while its native capacity remains charged.
    EXPECT_NE(Read(Send({}, "/v1/status", "GET"))
                  .find(R"("releasing_bytes":2097152)"),
              std::string::npos);
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
  }
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("parameter_operation":"idle")"));
  EXPECT_NE(ModelStatus("cold").find(R"("committed_bytes":0)"),
            std::string::npos);
  ASSERT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"cold","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
}

TEST_F(ElasticEmbeddingServiceTest,
       PressureReleaseDefersAdmissionWithoutRetainingTheRetiredModel) {
  const std::string package =
      iree::testing::MakeTempFilePath("parameter_release_pressure");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  for (const char* name : {"first", "second"}) {
    ASSERT_EQ(Read(Send(Declaration(name, package), "/v1/models/register"))
                  .find("HTTP/1.1 202"),
              0u);
    ASSERT_NO_FATAL_FAILURE(WaitModel(name, R"("registration":"ready")"));
  }
  ASSERT_NE(Read(Send(R"({"model":"first","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  int waiting = -1;
  {
    loom_serve::testing::ScopedNativeHold gate(device_,
                                               IREE_ALLOCATOR_COMMAND_FREE);
    waiting = Send(R"({"model":"second","input":"a"})");
    gate.WaitEntered();
    ASSERT_NO_FATAL_FAILURE(WaitModel("first", R"("committed_bytes":0)"));
    EXPECT_NE(Read(Send({}, "/v1/status", "GET"))
                  .find(R"("releasing_bytes":2097152)"),
              std::string::npos);
    EXPECT_NE(ModelStatus("second").find(R"("committed_bytes":0)"),
              std::string::npos);
    EXPECT_EQ(Read(Send(R"({"model":"first","mode":"abort"})",
                        "/v1/models/unregister"))
                  .find("HTTP/1.1 202"),
              0u);
    ASSERT_NO_FATAL_FAILURE(WaitModel("first", {}));
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    EXPECT_NE(ModelStatus("second").find(R"("pending_bytes":0)"),
              std::string::npos);
  }
  // No polling request drives the server between release and final inference.
  // The native capacity wake must resume the waiting request itself.
  EXPECT_NE(Read(waiting).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(WaitModel("first", {}));
}

TEST_F(ElasticEmbeddingServiceTest,
       ColdParameterCommitDoesNotBlockWarmHttpInferenceOnOneWorker) {
  const std::string package =
      iree::testing::MakeTempFilePath("cold_parameter_progress");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("registration":"ready")"));
  // Populate the shared workspace and the warm service's feedback path before
  // holding cold memory. Its next request needs no additional physical storage.
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  int cold = -1;
  {
    loom_serve::testing::ScopedNativeHold gate(device_);
    cold = Send(R"({"model":"cold","input":"a"})");
    gate.WaitEntered();
    const auto waiting = WaitModel("cold", R"("pending_bytes":2097152)");
    EXPECT_NE(waiting.find(R"("committed_bytes":0)"), std::string::npos);
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
  }
  EXPECT_NE(Read(cold).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", {}));
}

TEST_F(ElasticEmbeddingServiceTest,
       UnregisterJoinsPendingBackingWithoutBlockingWarmInference) {
  const std::string package =
      iree::testing::MakeTempFilePath("cancel_cold_parameters");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("registration":"ready")"));
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  int cold = -1;
  {
    loom_serve::testing::ScopedNativeHold gate(device_);
    cold = Send(R"({"model":"cold","input":"a"})");
    gate.WaitEntered();
    ASSERT_NO_FATAL_FAILURE(WaitModel("cold", R"("pending_bytes":2097152)"));
    EXPECT_EQ(Read(Send(R"({"model":"cold","mode":"abort"})",
                        "/v1/models/unregister"))
                  .find("HTTP/1.1 202"),
              0u);
    // Abort closes the peer, but accepted device work still owns the model.
    // Registration enters retirement only after that service work has drained.
    EXPECT_TRUE(Read(cold, /*allow_empty=*/true).empty());
    const auto removing = ModelStatus("cold");
    EXPECT_NE(removing.find(R"("pending_bytes":2097152)"), std::string::npos);
    EXPECT_NE(removing.find(R"("admission":"removing")"), std::string::npos);
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_FALSE(ModelStatus("cold").empty());
  }
  ASSERT_NO_FATAL_FAILURE(WaitModel("cold", {}));
  const auto status = Read(Send({}, "/v1/status", "GET"));
  EXPECT_NE(status.find(R"("pending_bytes":0)"), std::string::npos);
}

TEST_F(ElasticEmbeddingServiceTest,
       EmptyServerReusesOneParameterSlabAcrossManagedLifetimes) {
  const std::string package =
      iree::testing::MakeTempFilePath("dynamic_pressure");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(model_, nullptr)));
  StartManagedRouter();
  EXPECT_NE(Read(Send({}, "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  for (const char* name : {"first", "second"}) {
    EXPECT_EQ(Read(Send(Declaration(name, package), "/v1/models/register"))
                  .find("HTTP/1.1 202"),
              0u);
    EXPECT_NE(
        Read(Send(std::string("{\"model\":\"") + name + "\",\"input\":\"a\"}"))
            .find(R"("embedding":[2,2,2,2])"),
        std::string::npos);
  }
  EXPECT_EQ(Read(Send(R"({"model":"first"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("first", {});
  EXPECT_NE(Read(Send(R"({"model":"second","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"second"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModel("second", {});
  EXPECT_EQ(CompletedRequests(), 3u);
  EXPECT_NE(Read(Send({}, "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto memory = loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(memory.reserved_bytes, 0u);
  EXPECT_EQ(memory.committed_bytes, 0u);
  EXPECT_EQ(retired_counters_.completed, 3u);
}

TEST_F(ElasticEmbeddingServiceTest,
       CheckpointPackageRetainsPreparedCohortUnderPressure) {
  const std::string package =
      iree::testing::MakeTempFilePath("checkpoint_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(loom_serve_finite_model_free(std::exchange(model_, nullptr)));
  constexpr uint64_t kSlabBytes = 2 * 1024 * 1024;
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  auto* retained_residency = loom_serve_finite_model_residency(retained_);
  auto* residency = loom_serve_finite_model_residency(model_);
  IREE_ASSERT_OK(loom_serve_residency_acquire(retained_residency));
  IREE_ASSERT_OK(loom_serve_residency_await(retained_residency));
  EXPECT_EQ(loom_serve_residency_statistics(retained_residency).committed_bytes,
            kSlabBytes);
  const int cancelled = Send(R"({"model":"fixture","input":["a","bbb"]})");
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(cancelled, &connection));
  const auto queued = interface_.snapshot(interface_.self);
  EXPECT_EQ(queued.queued_requests, 1u);
  EXPECT_EQ(queued.active_requests, 0u);
  EXPECT_GE(queued.oldest_queued_age, 0);
  EXPECT_EQ(queued.oldest_active_age, 0);
  // The outer owner holds another model's pin and will release it. Queue the
  // cohort behind that activity instead of treating capacity as unreachable.
  constexpr auto kShared = LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY;
  EXPECT_TRUE(Step(kShared));  // Prepare requests.
  EXPECT_TRUE(Step(kShared));  // Prepare cohort.
  EXPECT_TRUE(Step(kShared));  // Consume preparation; admission denied.
  const auto waiting = interface_.snapshot(interface_.self);
  EXPECT_STREQ(waiting.phase, "waiting");
  EXPECT_EQ(waiting.queued_requests, 0u);
  EXPECT_EQ(waiting.oldest_queued_age, 0);
  EXPECT_EQ(waiting.active_requests, 1u);
  EXPECT_GT(waiting.oldest_active_age, 0);
  EXPECT_GE(waiting.oldest_active_age, queued.oldest_queued_age);
  const auto pressure = loom_serve_residency_snapshot(residency);
  EXPECT_TRUE(pressure.admission.waiting);
  EXPECT_EQ(pressure.admission.count, 1u);
  EXPECT_EQ(pressure.admission.duration_ns, pressure.admission.age_ns);
  EXPECT_EQ(pressure.pin_count, 0u);
  EXPECT_EQ(loom_serve_residency_snapshot(retained_residency).pin_count, 1u);
  const uint64_t wakes = wakes_.load();
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(Step(kShared));
    EXPECT_FALSE(interface_.is_active(interface_.self));
    EXPECT_GE(interface_.snapshot(interface_.self).oldest_active_age,
              waiting.oldest_active_age);
    const auto retry = loom_serve_residency_snapshot(residency);
    EXPECT_TRUE(retry.admission.waiting);
    EXPECT_EQ(retry.admission.count, 1u);
    EXPECT_EQ(retry.admission.duration_ns, retry.admission.age_ns);
    EXPECT_GE(retry.admission.age_ns, pressure.admission.age_ns);
  }
  // No new CPU process or device submission occurred during those turns. The
  // source packet and result association were retained, not prepared again.
  EXPECT_EQ(wakes_.load(), wakes);
  EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes, 0u);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(
      loom_serve_residency_try_admit(retained_residency, &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(cancelled, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(cancelled), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connection, iree_infinite_timeout());
  EXPECT_TRUE(Step(kShared));
  // Reset withdraws the older pressure interest. The still-pinned peer may
  // accept again without waiting for its existing invocation to finish.
  const auto withdrawn = loom_serve_residency_snapshot(residency);
  EXPECT_FALSE(withdrawn.admission.waiting);
  EXPECT_EQ(withdrawn.admission.count, 1u);
  EXPECT_EQ(withdrawn.admission.age_ns, 0u);
  EXPECT_GE(withdrawn.admission.duration_ns, pressure.admission.duration_ns);
  EXPECT_EQ(withdrawn.pin_count, 0u);
  IREE_ASSERT_OK(
      loom_serve_residency_try_admit(retained_residency, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_cancel_admission(retained_residency);
  loom_serve_residency_release(retained_residency);
  const auto cancelled_state = interface_.snapshot(interface_.self);
  EXPECT_EQ(cancelled_state.oldest_queued_age, 0);
  EXPECT_EQ(cancelled_state.oldest_active_age, 0);
  EXPECT_EQ(loom_serve_residency_snapshot(residency).admission.duration_ns,
            withdrawn.admission.duration_ns);
  const int request = Send(R"({"model":"fixture","input":["a","bbb"]})");
  ASSERT_NO_FATAL_FAILURE(Accept(request));
  EXPECT_TRUE(Step(kShared));
  EXPECT_TRUE(Step(kShared));
  EXPECT_TRUE(Step(kShared));
  const auto second_wait = loom_serve_residency_snapshot(residency);
  EXPECT_TRUE(second_wait.admission.waiting);
  EXPECT_EQ(second_wait.admission.count, 2u);
  EXPECT_EQ(second_wait.admission.duration_ns,
            withdrawn.admission.duration_ns + second_wait.admission.age_ns);
  loom_serve_residency_release(retained_residency);
  Drain();
  const std::string result = Read(request);
  const auto idle = interface_.snapshot(interface_.self);
  EXPECT_EQ(idle.queued_requests, 0u);
  EXPECT_EQ(idle.active_requests, 0u);
  EXPECT_EQ(idle.oldest_queued_age, 0);
  EXPECT_EQ(idle.oldest_active_age, 0);
  const auto admitted = loom_serve_residency_snapshot(residency);
  EXPECT_FALSE(admitted.admission.waiting);
  EXPECT_EQ(admitted.admission.count, 2u);
  EXPECT_EQ(admitted.admission.age_ns, 0u);
  EXPECT_GE(admitted.admission.duration_ns, second_wait.admission.duration_ns);
  EXPECT_EQ(admitted.pin_count, 0u);
  EXPECT_EQ(result.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(result.find(R"("index":0,"embedding":[2,2,2,4])"),
            std::string::npos);
  EXPECT_NE(result.find(R"("index":1,"embedding":[12,6,4,4])"),
            std::string::npos);
  EXPECT_EQ(loom_serve_residency_statistics(retained_residency).committed_bytes,
            0u);
  EXPECT_EQ(loom_serve_residency_statistics(residency).committed_bytes,
            kSlabBytes);

  const uint64_t reservation_bytes =
      loom_serve_residency_statistics(residency).reserved_bytes;
  const auto before_removal =
      loom_serve_device_memory_statistics(device_).retained;
  // Retire the model with prepared parameters on the serialized owner that
  // must admit its peer. The one-slab budget cannot hold both parameter sets.
  interface_.set_observer(interface_.self, {});
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  std::swap(model_, retained_);
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  const int peer = Send(R"({"model":"fixture","input":"a"})");
  ASSERT_NO_FATAL_FAILURE(Accept(peer));
  loom_serve_test_embedding_model_lifecycle_t removal;
  removal.model = std::exchange(retained_, nullptr);
  removal.service = &interface_;
  iree_status_t removal_status;
  {
    StalledStderr stalled;
    ASSERT_NO_FATAL_FAILURE(stalled.Begin());
    removal_status = removal.Retire(device_);
  }
  retained_ = removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(retained_, nullptr);
  EXPECT_NE(Read(peer).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  const auto after_removal =
      loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(after_removal.reserved_bytes,
            before_removal.reserved_bytes - reservation_bytes);
  EXPECT_EQ(after_removal.committed_bytes, kSlabBytes);

  // Final removal returns the shared model pool to empty without tearing down
  // its device or host executor. Only private command cache backing may remain.
  loom_serve_finite_service_free(std::exchange(service_, nullptr));
  loom_serve_test_embedding_model_lifecycle_t final_removal;
  final_removal.model = std::exchange(model_, nullptr);
  removal_status = final_removal.Retire(device_);
  model_ = final_removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(model_, nullptr);
  const auto empty = loom_serve_device_memory_statistics(device_).retained;
  EXPECT_EQ(empty.reserved_bytes, 0u);
  EXPECT_EQ(empty.committed_bytes, 0u);
}

}  // namespace loom_serve::testing::embedding
