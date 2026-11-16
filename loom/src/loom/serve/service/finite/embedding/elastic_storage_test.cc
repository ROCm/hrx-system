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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/testing/native_allocator.h"
#include "loom/runtime/testing/stalled_stderr.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"
#include "loom/service/finite/embedding/protocol.h"
#include "loom/service/finite/registration.h"

namespace loom_serve::testing::embedding {

using loom_runtime::testing::StalledStderr;

class ElasticEmbeddingServiceTest : public SourceEmbeddingServiceTest {
 protected:
  // Two (16 + 52 * 32)-byte invocations share one 4096-byte workspace slab.
  // A second parameter slab cannot coexist with that live workspace.
  uint64_t BatchUnits() const override { return 32; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceEmbeddingServiceTest::SetUp());
    // Cold-allocation holds must not depend on a previous test's cached slabs.
    auto* pool = loom_runtime_pool_memory(storage_owner_);
    loom_runtime_memory_pool_trim(pool, 0);
    IREE_ASSERT_OK(loom_runtime_memory_pool_await(pool));
  }

  template <typename Predicate>
  iree_status_t ProgressRegistrationUntil(
      loom_service_finite_registration_t* registration, Predicate done) {
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_runtime_device_group_advance(group_);
      const bool progress =
          loom_service_finite_registration_progress(registration, 0);
      const auto snapshot =
          loom_service_finite_registration_snapshot(registration);
      if (snapshot.phase == LOOM_SERVICE_FINITE_REGISTRATION_RELEASE_FAILED) {
        status = iree_status_join(status, iree_status_clone(snapshot.status));
      }
      const bool complete = done(snapshot);
      if (complete || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      if (complete || !iree_status_is_ok(status)) {
        return status;
      }
    }
  }
};

TEST_F(ElasticEmbeddingServiceTest,
       FailedNativeWarmupRetainsAcceptedProtectionUntilRetirement) {
  const std::string package =
      iree::testing::MakeTempFilePath("native_warmup_retirement");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  loom_serve::testing::ModelDocument document;
  IREE_ASSERT_OK(document.Load(package + "/model.json"));
  const loom_service_finite_model_options_t model_options = {
      .sources = document.value().sources,
      .asset_directory = IREE_SV("."),
      .options = document.value().options,
  };
  const loom_service_finite_registration_options_t options = {
      .incarnation = 1,
      .protocol = loom_service_embedding_protocol(),
      .name = IREE_SV("native"),
      .capacity = 1,
  };
  const loom_runtime_wake_callback_t observer = {
      [](void* self) {
        iree_notification_post(static_cast<iree_notification_t*>(self),
                               IREE_ALL_WAITERS);
      },
      &notification_,
  };
  loom_service_finite_registration_t* raw_registration = nullptr;
  IREE_ASSERT_OK(loom_service_finite_registration_allocate(
      device_, storage_owner_, &model_options, &options, observer,
      host_allocator_, &raw_registration));
  auto retire = [this](loom_service_finite_registration_t* registration) {
    loom_service_finite_registration_request_retirement(
        registration, LOOM_SERVICE_FINITE_RETIRE_FLAG_ABORT);
    IREE_CHECK_OK(
        ProgressRegistrationUntil(registration, [](const auto& state) {
          return state.phase == LOOM_SERVICE_FINITE_REGISTRATION_RETIRED;
        }));
    loom_service_finite_registration_free(registration);
  };
  std::unique_ptr<loom_service_finite_registration_t, decltype(retire)>
      registration(raw_registration, retire);
  const auto* control =
      loom_service_finite_registration_control(registration.get());
  auto schedule = [&](loom_runtime_service_work_flags_t work) {
    const auto layout = loom_runtime_service_control_layout(control);
    for (uint64_t i = 0; i < layout.offer_capacity; ++i) {
      loom_runtime_service_offer_t offer = {};
      EXPECT_EQ(loom_runtime_service_read_offer(control, i, &offer),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      if (offer.choice_count && offer.work == work) {
        return loom_service_finite_registration_try_schedule(registration.get(),
                                                             offer.key, 0, 0);
      }
    }
    return loom_runtime_service_schedule_outcome_t{};
  };
  ASSERT_TRUE(schedule(LOOM_RUNTIME_SERVICE_WORK_PREPARATION).accepted);
  IREE_ASSERT_OK(
      ProgressRegistrationUntil(registration.get(), [](const auto& state) {
        return state.phase == LOOM_SERVICE_FINITE_REGISTRATION_READY ||
               state.phase == LOOM_SERVICE_FINITE_REGISTRATION_FAILED;
      }));
  ASSERT_EQ(loom_service_finite_registration_snapshot(registration.get()).phase,
            LOOM_SERVICE_FINITE_REGISTRATION_READY);
  ASSERT_TRUE(
      loom_service_finite_registration_parameters(registration.get()).elastic);
  loom_runtime::testing::NativeAllocator::ForDevice(device_).FailNextAllocation(
      IREE_ALLOCATOR_COMMAND_CALLOC, /*native_only=*/true);
  IREE_ASSERT_OK(loom_service_finite_registration_request_parameters(
      registration.get(), LOOM_SERVICE_FINITE_PARAMETER_OPERATION_ACTIVATE));
  const auto accepted = schedule(LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  ASSERT_TRUE(accepted.accepted);
  IREE_ASSERT_OK(
      ProgressRegistrationUntil(registration.get(), [](const auto& state) {
        return state.phase == LOOM_SERVICE_FINITE_REGISTRATION_FAILED;
      }));
  const auto failed =
      loom_service_finite_registration_snapshot(registration.get());
  EXPECT_EQ(iree_status_code(failed.status), IREE_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(failed.parameter_operation,
            LOOM_SERVICE_FINITE_PARAMETER_OPERATION_NONE);
  loom_runtime_service_operation_t operation = {};
  ASSERT_EQ(loom_runtime_service_read_operation(
                control, accepted.operation.ordinal, &operation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(operation.key.serial, accepted.operation.serial);
  EXPECT_EQ(operation.work, LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  // The callback joined; protection still lives.
  EXPECT_EQ(operation.waits, 0u);
  loom_runtime_service_use_t use = {};
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);

  // No application views remain. Failure cleanup requires no new policy grant,
  // and only actual native retirement invalidates the accepted use.
  loom_service_finite_registration_request_retirement(registration.get(), 0);
  IREE_ASSERT_OK(
      ProgressRegistrationUntil(registration.get(), [](const auto& state) {
        return state.phase == LOOM_SERVICE_FINITE_REGISTRATION_RETIRED;
      }));
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  EXPECT_EQ(observation.accepted_devices, 0u);
}

TEST_F(ElasticEmbeddingServiceTest,
       ParameterDeactivationCachesBackingAndReloadsExactly) {
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
  ASSERT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/deactivate"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
  const auto retired =
      WaitModelState("cold", R"("parameter_operation":"idle")");
  EXPECT_NE(retired.find(R"("committed_bytes":0)"), std::string::npos);
  // Slab accounting distinguishes retained physical storage from temporary
  // host metadata that an unmap may allocate and free.
  EXPECT_NE(Read(Send({}, "/v1/status", "GET")).find(R"("cached_bytes":4096)"),
            std::string::npos);
  ASSERT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/undrain"))
                .find("HTTP/1.1 200"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"cold","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
}

TEST_F(ElasticEmbeddingServiceTest,
       PressureReusesBackingWithoutRetainingTheRetiredModel) {
  const std::string package =
      iree::testing::MakeTempFilePath("parameter_release_pressure");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  for (const char* name : {"first", "second"}) {
    ASSERT_EQ(Read(Send(Declaration(name, package), "/v1/models/register"))
                  .find("HTTP/1.1 202"),
              0u);
    ASSERT_NO_FATAL_FAILURE(WaitModelState(name, R"("registration":"ready")"));
  }
  ASSERT_NE(Read(Send(R"({"model":"first","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  {
    // Both models fit separately in the one-slab budget. Pressure must transfer
    // backing without needing another native allocation or a polling request.
    loom_runtime::testing::ScopedNativeHold gate(device_);
    EXPECT_NE(Read(Send(R"({"model":"second","input":"a"})"))
                  .find(R"("embedding":[2,2,2,2])"),
              std::string::npos);
    ASSERT_NO_FATAL_FAILURE(WaitModelState("first", R"("committed_bytes":0)"));
    EXPECT_NE(ModelStatus("second").find(R"("committed_bytes":4096)"),
              std::string::npos);
    EXPECT_EQ(Read(Send(R"({"model":"first","mode":"abort"})",
                        "/v1/models/unregister"))
                  .find("HTTP/1.1 202"),
              0u);
    ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("first"));
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
    EXPECT_NE(ModelStatus("second").find(R"("pending_bytes":0)"),
              std::string::npos);
  }
  ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("first"));
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
  ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("registration":"ready")"));
  // Populate the shared workspace and the warm service's feedback path before
  // holding cold memory. Its next request needs no additional physical storage.
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  int cold = -1;
  {
    loom_runtime::testing::ScopedNativeHold gate(device_);
    cold = Send(R"({"model":"cold","input":"a"})");
    gate.WaitEntered();
    const auto waiting = WaitModelState("cold", R"("pending_bytes":4096)");
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
  ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("cold"));
}

TEST_F(ElasticEmbeddingServiceTest,
       ColdAllocationFailureDoesNotPublishAnOutput) {
  const std::string package =
      iree::testing::MakeTempFilePath("failed_parameter_output");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("registration":"ready")"));
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);

  // Leave native failure free to complete during submission. Holding it until
  // another request queues would exclude this dependency-publication ordering.
  loom_runtime::testing::NativeAllocator::ForDevice(device_).FailNextAllocation(
      IREE_ALLOCATOR_COMMAND_CALLOC, /*native_only=*/true);
  const auto failed = Read(Send(R"({"model":"cold","input":"a"})"));
  ASSERT_EQ(failed.find("HTTP/1.1 503"), 0u) << failed;
  EXPECT_NE(failed.find(R"("type":"loom_execution_error")"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("registration":"failed")"));
  EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
}

TEST_F(ElasticEmbeddingServiceTest,
       ColdAllocationFailurePreservesWarmHttpInference) {
  const std::string package =
      iree::testing::MakeTempFilePath("failed_parameter_allocation");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  StartManagedRouter();
  ASSERT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("registration":"ready")"));
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);

  loom_runtime::testing::NativeAllocator::ForDevice(device_).FailNextAllocation(
      IREE_ALLOCATOR_COMMAND_CALLOC, /*native_only=*/true);
  int active = -1;
  int pending = -1;
  {
    loom_runtime::testing::ScopedNativeHold allocation(device_);
    active = Send(R"({"model":"cold","input":"a"})");
    allocation.WaitEntered();
    pending = Send(R"({"model":"cold","input":"b"})");
    ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("queued_requests":1)"));
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
  }
  const auto failed = Read(active);
  const auto rejected = Read(pending);
  ASSERT_EQ(failed.find("HTTP/1.1 503"), 0u) << failed;
  EXPECT_NE(failed.find(R"("type":"loom_execution_error")"), std::string::npos);
  EXPECT_EQ(rejected.find("HTTP/1.1 503"), 0u) << rejected;
  EXPECT_NE(rejected.find(R"("type":"loom_execution_error")"),
            std::string::npos);
  const auto state = WaitModelState("cold", R"("registration":"failed")");
  EXPECT_NE(state.find(R"("error":{"code":8,)"), std::string::npos) << state;
  EXPECT_EQ(Read(Send({}, "/v1/models", "GET")).find(R"("id":"cold")"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"cold","input":"a"})")).find("HTTP/1.1 503"),
            0u);
  EXPECT_EQ(Read(Send({}, "/healthz", "GET")).find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"cold"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("cold"));
  EXPECT_EQ(Read(Send(Declaration("cold", package), "/v1/models/register"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_NE(Read(Send(R"({"model":"cold","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
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
  ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("registration":"ready")"));
  ASSERT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                .find(R"("embedding":[1,1,1,1])"),
            std::string::npos);
  int cold = -1;
  {
    loom_runtime::testing::ScopedNativeHold gate(device_);
    cold = Send(R"({"model":"cold","input":"a"})");
    gate.WaitEntered();
    ASSERT_NO_FATAL_FAILURE(WaitModelState("cold", R"("pending_bytes":4096)"));
    EXPECT_EQ(Read(Send(R"({"model":"cold","mode":"abort"})",
                        "/v1/models/unregister"))
                  .find("HTTP/1.1 202"),
              0u);
    // Abort closes the peer, but accepted device work still owns the model.
    // Registration enters retirement only after that service work has drained.
    EXPECT_TRUE(Read(cold, /*allow_empty=*/true).empty());
    const auto removing = ModelStatus("cold");
    EXPECT_NE(removing.find(R"("pending_bytes":4096)"), std::string::npos);
    EXPECT_NE(removing.find(R"("admission":"removing")"), std::string::npos);
    EXPECT_NE(Read(Send(R"({"model":"fixture","input":"a"})"))
                  .find(R"("embedding":[1,1,1,1])"),
              std::string::npos);
    EXPECT_FALSE(ModelStatus("cold").empty());
  }
  ASSERT_NO_FATAL_FAILURE(WaitModelAbsent("cold"));
  const auto status = Read(Send({}, "/v1/status", "GET"));
  EXPECT_NE(status.find(R"("pending_bytes":0)"), std::string::npos);
}

TEST_F(ElasticEmbeddingServiceTest,
       EmptyServerReusesOneParameterSlabAcrossManagedLifetimes) {
  const std::string package =
      iree::testing::MakeTempFilePath("dynamic_pressure");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
  loom_service_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(
      loom_service_finite_model_free(std::exchange(model_, nullptr)));
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
  WaitModelAbsent("first");
  EXPECT_NE(Read(Send(R"({"model":"second","input":"a"})"))
                .find(R"("embedding":[2,2,2,2])"),
            std::string::npos);
  EXPECT_EQ(Read(Send(R"({"model":"second"})", "/v1/models/unregister"))
                .find("HTTP/1.1 202"),
            0u);
  WaitModelAbsent("second");
  EXPECT_EQ(CompletedRequests(), 3u);
  EXPECT_NE(Read(Send({}, "/v1/models", "GET")).find(R"("data":[])"),
            std::string::npos);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  const auto memory = loom_runtime_pool_statistics(storage_owner_).retained;
  const auto workspace = loom_runtime_workspace_memory_usage(
      loom_runtime_pool_workspace(storage_owner_));
  EXPECT_FALSE(loom_runtime_pool_is_in_use(storage_owner_));
  EXPECT_EQ(memory.committed_bytes,
            memory.cached_bytes + workspace.committed_bytes);
  EXPECT_EQ(retired_counters_.completed, 3u);
}

TEST_F(ElasticEmbeddingServiceTest,
       CheckpointPackageRetainsPreparedCohortUnderPressure) {
  const std::string package =
      iree::testing::MakeTempFilePath("checkpoint_embedding");
  ASSERT_NO_FATAL_FAILURE(WriteCheckpointPackage(package));
  interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
  loom_service_finite_service_free(std::exchange(service_, nullptr));
  IREE_ASSERT_OK(
      loom_service_finite_model_free(std::exchange(model_, nullptr)));
  constexpr uint64_t kSlabBytes = 4096;
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  retained_ = std::exchange(model_, nullptr);
  ASSERT_NO_FATAL_FAILURE(CreateModel(package));
  ASSERT_NO_FATAL_FAILURE(CreateService(0));
  auto* retained_residency = loom_service_finite_model_residency(retained_);
  auto* residency = loom_service_finite_model_residency(model_);
  IREE_ASSERT_OK(loom_runtime_residency_acquire(retained_residency));
  IREE_ASSERT_OK(loom_runtime_residency_await(retained_residency));
  EXPECT_EQ(
      loom_runtime_residency_statistics(retained_residency).committed_bytes,
      kSlabBytes);
  const int cancelled = Send(R"({"model":"fixture","input":["a","bbb"]})");
  loom_serve_http_connection_t* connection = nullptr;
  ASSERT_NO_FATAL_FAILURE(Accept(cancelled, &connection));
  const auto queued = interface_.snapshot(interface_.runtime.self);
  EXPECT_EQ(queued.queued_requests, 1u);
  EXPECT_EQ(queued.active_requests, 0u);
  EXPECT_GE(queued.oldest_queued_age, 0);
  EXPECT_EQ(queued.oldest_active_age, 0);
  // The outer owner holds another model's pin and will release it. Queue the
  // cohort behind that activity instead of treating capacity as unreachable.
  constexpr auto kShared = LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY;
  while (std::strcmp(interface_.snapshot(interface_.runtime.self).phase,
                     "waiting") != 0) {
    ASSERT_TRUE(Step(kShared));
  }
  const auto waiting = interface_.snapshot(interface_.runtime.self);
  EXPECT_STREQ(waiting.phase, "waiting");
  EXPECT_EQ(waiting.queued_requests, 0u);
  EXPECT_EQ(waiting.oldest_queued_age, 0);
  EXPECT_EQ(waiting.active_requests, 1u);
  EXPECT_GT(waiting.oldest_active_age, 0);
  EXPECT_GE(waiting.oldest_active_age, queued.oldest_queued_age);
  const auto pressure = loom_runtime_residency_snapshot(residency);
  EXPECT_TRUE(pressure.admission.waiting);
  EXPECT_EQ(pressure.admission.count, 1u);
  EXPECT_EQ(pressure.admission.duration_ns, pressure.admission.age_ns);
  EXPECT_EQ(pressure.pin_count, 0u);
  EXPECT_EQ(loom_runtime_residency_snapshot(retained_residency).pin_count, 1u);
  const uint64_t wakes = wakes_.load();
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(Step(kShared));
    EXPECT_FALSE(interface_.runtime.vtable->is_active(interface_.runtime.self));
    EXPECT_GE(interface_.snapshot(interface_.runtime.self).oldest_active_age,
              waiting.oldest_active_age);
    const auto retry = loom_runtime_residency_snapshot(residency);
    EXPECT_TRUE(retry.admission.waiting);
    EXPECT_EQ(retry.admission.count, 1u);
    EXPECT_EQ(retry.admission.duration_ns, retry.admission.age_ns);
    EXPECT_GE(retry.admission.age_ns, pressure.admission.age_ns);
  }
  // No new CPU process or device submission occurred during those turns. The
  // source packet and result association were retained, not prepared again.
  EXPECT_EQ(wakes_.load(), wakes);
  EXPECT_EQ(loom_runtime_residency_statistics(residency).committed_bytes, 0u);
  loom_runtime_residency_admission_t admission;
  IREE_ASSERT_OK(
      loom_runtime_residency_try_admit(retained_residency, &admission));
  EXPECT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_DEFERRED);
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
  const auto withdrawn = loom_runtime_residency_snapshot(residency);
  EXPECT_FALSE(withdrawn.admission.waiting);
  EXPECT_EQ(withdrawn.admission.count, 1u);
  EXPECT_EQ(withdrawn.admission.age_ns, 0u);
  EXPECT_GE(withdrawn.admission.duration_ns, pressure.admission.duration_ns);
  EXPECT_EQ(withdrawn.pin_count, 0u);
  IREE_ASSERT_OK(
      loom_runtime_residency_try_admit(retained_residency, &admission));
  ASSERT_EQ(admission, LOOM_RUNTIME_RESIDENCY_ADMISSION_ACQUIRED);
  loom_runtime_residency_cancel_admission(retained_residency);
  loom_runtime_residency_release(retained_residency);
  const auto cancelled_state = interface_.snapshot(interface_.runtime.self);
  EXPECT_EQ(cancelled_state.oldest_queued_age, 0);
  EXPECT_EQ(cancelled_state.oldest_active_age, 0);
  EXPECT_EQ(loom_runtime_residency_snapshot(residency).admission.duration_ns,
            withdrawn.admission.duration_ns);
  const int request = Send(R"({"model":"fixture","input":["a","bbb"]})");
  ASSERT_NO_FATAL_FAILURE(Accept(request));
  while (std::strcmp(interface_.snapshot(interface_.runtime.self).phase,
                     "waiting") != 0) {
    ASSERT_TRUE(Step(kShared));
  }
  const auto second_wait = loom_runtime_residency_snapshot(residency);
  EXPECT_TRUE(second_wait.admission.waiting);
  EXPECT_EQ(second_wait.admission.count, 2u);
  EXPECT_EQ(second_wait.admission.duration_ns,
            withdrawn.admission.duration_ns + second_wait.admission.age_ns);
  loom_runtime_residency_release(retained_residency);
  Drain();
  const std::string result = Read(request);
  const auto idle = interface_.snapshot(interface_.runtime.self);
  EXPECT_EQ(idle.queued_requests, 0u);
  EXPECT_EQ(idle.active_requests, 0u);
  EXPECT_EQ(idle.oldest_queued_age, 0);
  EXPECT_EQ(idle.oldest_active_age, 0);
  const auto admitted = loom_runtime_residency_snapshot(residency);
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
  EXPECT_EQ(
      loom_runtime_residency_statistics(retained_residency).committed_bytes,
      0u);
  EXPECT_EQ(loom_runtime_residency_statistics(residency).committed_bytes,
            kSlabBytes);

  const uint64_t reservation_bytes =
      loom_runtime_residency_statistics(residency).reserved_bytes;
  const auto before_removal =
      loom_runtime_pool_statistics(storage_owner_).retained;
  // Retire the model with prepared parameters on the serialized owner that
  // must admit its peer. The one-slab budget cannot hold both parameter sets.
  interface_.runtime.vtable->set_observer(interface_.runtime.self, {});
  loom_service_finite_service_free(std::exchange(service_, nullptr));
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
    removal_status = removal.Retire(device_, storage_owner_);
  }
  retained_ = removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(retained_, nullptr);
  EXPECT_NE(Read(peer).find(R"("embedding":[2,2,2,2])"), std::string::npos);
  const auto after_removal =
      loom_runtime_pool_statistics(storage_owner_).retained;
  EXPECT_EQ(after_removal.reserved_bytes,
            before_removal.reserved_bytes - reservation_bytes);
  EXPECT_EQ(after_removal.committed_bytes,
            kSlabBytes + loom_runtime_workspace_memory_usage(
                             loom_runtime_pool_workspace(storage_owner_))
                             .committed_bytes);

  // Final removal retains reusable physical backing without any model owning
  // it. The same device and host executor remain available for replacements.
  loom_service_finite_service_free(std::exchange(service_, nullptr));
  loom_serve_test_embedding_model_lifecycle_t final_removal;
  final_removal.model = std::exchange(model_, nullptr);
  removal_status = final_removal.Retire(device_, storage_owner_);
  model_ = final_removal.model;
  IREE_ASSERT_OK(removal_status);
  EXPECT_EQ(model_, nullptr);
  const auto empty = loom_runtime_pool_statistics(storage_owner_).retained;
  const auto workspace = loom_runtime_workspace_memory_usage(
      loom_runtime_pool_workspace(storage_owner_));
  EXPECT_FALSE(loom_runtime_pool_is_in_use(storage_owner_));
  EXPECT_EQ(empty.committed_bytes,
            empty.cached_bytes + workspace.committed_bytes);
}

}  // namespace loom_serve::testing::embedding
