// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/serving/router_test_fixture.h"

namespace {

using loom_serve::testing::HttpRouterTest;
using loom_serve::testing::ModelService;
using loom_serve::testing::ObservationClient;

// A model dependency with independently controlled service and final device
// ownership. HTTP admission, the registry, task owner and retirement policy are
// the production implementations under test.
struct RegistrationDependency {
  enum class PreparationFailure { kNone, kLocal, kShared };

  // Requests reach the same held-completion service used by the routing tests.
  ModelService service = {.name = "dynamic", .hold_completion = true};
  // Model-protocol counters supplied by the dependency, not synthesized by
  // routing.
  loom_serve_telemetry_counters_t counters = {};
  // Serializes retirement readiness and wake detachment with the owner.
  std::mutex mutex;
  // Outer-test synchronization on actual retirement entry.
  std::condition_variable condition;
  // Final device-borrower wake, independent of the service observer.
  loom_serve_http_observer_t observer = {};
  // Registry-owned history borrowed by the prepared service.
  loom_serve_telemetry_stream_t* stream = nullptr;
  // Retire has been called after all service and utility borrowers completed.
  bool retiring = false;
  // The test has returned final model ownership.
  bool retired = false;
  // Preparation allocation count, observable without reading owner-only state.
  std::atomic<unsigned> created{0};
  // Operation destruction count, independent of name/slot reuse.
  std::atomic<unsigned> destroyed{0};
  // Injects one quiescent platform release failure, preserving owned storage.
  std::atomic<bool> fail_release{false};
  // Counts final release attempts, not asynchronous ownership polling.
  std::atomic<unsigned> release_attempts{0};
  // Preparation failure scope, configured under mutex before factory creation.
  PreparationFailure preparation_failure = PreparationFailure::kNone;
  // Parameter state belongs to the model dependency and changes on owner turns.
  loom_serve_residency_snapshot_t parameters = {
      .active = true,
      .elastic = true,
      .memory = {.reserved_bytes = 4096, .committed_bytes = 4096},
      .activity = {.activations = 1}};
  // Injects an explicit native parameter release failure after quiescence.
  std::atomic<bool> fail_deactivation{false};
  // Eviction policy supplied by the real registry, independent of readiness.
  std::atomic<bool> pinned{false};
  // Adapter capabilities configured before factory publication.
  struct {
    // Whether the dependency implements eviction policy.
    bool pinning = true;
    // Whether the dependency implements explicit parameter warmup.
    bool activation = true;
  } capabilities;
  // Optional shared pressure domain, configured before factory publication.
  const void* capacity_domain = nullptr;
  // Independently controlled parameter-readiness dependency.
  struct {
    // An accepted warmup still borrows the model until readiness is consumed.
    bool submitted = false;
    // The test has published terminal readiness.
    bool ready = false;
    // Unregister has requested cancellation while the callback is outstanding.
    bool cancelling = false;
    // Number of accepted operations, not repeated owner polls.
    unsigned starts = 0;
    // Terminal readiness result transferred to the production registry.
    iree_status_t status = iree_ok_status();
  } activation;

  struct Operation {
    // Test-owned model dependency, live beyond the owning registry.
    RegistrationDependency* dependency;
  };

  static iree_status_t Prepare(void* self, bool cancel, bool* complete,
                               loom_serve_http_service_t* service,
                               loom_serve_compilation_report_t*) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    *complete = true;
    if (cancel) {
      return iree_make_status(IREE_STATUS_CANCELLED, "preparation cancelled");
    }
    std::lock_guard<std::mutex> lock(dependency->mutex);
    if (dependency->preparation_failure != PreparationFailure::kNone) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "model prepare failed");
    }
    *service =
        dependency->service.Interface(nullptr, dependency->capacity_domain);
    service->telemetry.stream = dependency->stream;
    return iree_ok_status();
  }
  static iree_status_t Retire(void* self, bool* complete) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    EXPECT_FALSE(ModelService::IsPending(&dependency->service));
    EXPECT_TRUE(dependency->service.pending.empty());
    std::lock_guard<std::mutex> lock(dependency->mutex);
    EXPECT_FALSE(dependency->activation.submitted);
    dependency->retiring = true;
    *complete = dependency->retired;
    if (*complete) {
      dependency->observer = {};
      ++dependency->release_attempts;
      if (dependency->fail_release.exchange(false)) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "platform release failed");
      }
    }
    dependency->condition.notify_all();
    return iree_ok_status();
  }
  static void Destroy(void* self) {
    auto* operation = static_cast<Operation*>(self);
    ++operation->dependency->destroyed;
    delete operation;
  }
  static loom_serve_residency_snapshot_t Parameters(void* self) {
    return static_cast<Operation*>(self)->dependency->parameters;
  }
  static void SetPinned(void* self, bool pinned) {
    static_cast<Operation*>(self)->dependency->pinned = pinned;
  }
  static iree_status_t Deactivate(void* self) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    EXPECT_FALSE(ModelService::IsPending(&dependency->service));
    EXPECT_TRUE(dependency->service.pending.empty());
    if (dependency->fail_deactivation.exchange(false)) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "parameter release failed");
    }
    auto& parameters = dependency->parameters;
    parameters.activity.deactivations += parameters.active;
    parameters.active = false;
    parameters.memory.committed_bytes = 0;
    return iree_ok_status();
  }
  static iree_status_t Activate(void* self, loom_serve_activation_flags_t flags,
                                loom_serve_activation_phase_t* phase) {
    auto* dependency = static_cast<Operation*>(self)->dependency;
    std::lock_guard<std::mutex> lock(dependency->mutex);
    auto& activation = dependency->activation;
    const bool cancel =
        iree_any_bit_set(flags, LOOM_SERVE_ACTIVATION_FLAG_CANCEL);
    if (!activation.submitted && cancel) {
      *phase = LOOM_SERVE_ACTIVATION_COMPLETE;
      return iree_ok_status();
    }
    if (!activation.submitted) {
      activation.submitted = true;
      ++activation.starts;
    }
    activation.cancelling |= cancel;
    dependency->condition.notify_all();
    if (!activation.ready) {
      *phase = LOOM_SERVE_ACTIVATION_LOADING;
      return iree_ok_status();
    }
    activation.submitted = false;
    activation.ready = false;
    *phase = LOOM_SERVE_ACTIVATION_COMPLETE;
    if (iree_status_is_ok(activation.status)) {
      auto& parameters = dependency->parameters;
      parameters.activity.activations += !parameters.active;
      parameters.active = true;
      parameters.memory.committed_bytes = parameters.memory.reserved_bytes;
    }
    return std::exchange(activation.status, iree_ok_status());
  }
  static iree_status_t Create(
      void* self, const loom_serve_model_configuration_t* configuration,
      loom_serve_http_observer_t observer,
      loom_serve_telemetry_stream_t* stream,
      loom_serve_registration_t* registration, iree_allocator_t) {
    auto* dependency = static_cast<RegistrationDependency*>(self);
    // Import fixture configuration before transferring it to the model owner.
    std::lock_guard<std::mutex> lock(dependency->mutex);
    dependency->service.name = configuration->name.data;
    dependency->service.counters = &dependency->counters;
    dependency->observer = observer;
    dependency->stream = stream;
    ++dependency->created;
    *registration = {
        .self = new Operation{dependency},
        .prepare = Prepare,
        .retire = Retire,
        .parameters = Parameters,
        .set_pinned = dependency->capabilities.pinning ? SetPinned : nullptr,
        .activate = dependency->capabilities.activation ? Activate : nullptr,
        .deactivate = Deactivate,
        .destroy = Destroy};
    return iree_ok_status();
  }
  static iree_status_t QueryStatus(void* self) {
    auto* dependency = static_cast<RegistrationDependency*>(self);
    std::lock_guard<std::mutex> lock(dependency->mutex);
    return dependency->preparation_failure == PreparationFailure::kShared
               ? iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "shared reclamation failed")
               : iree_ok_status();
  }
  void WaitRetiring() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return retiring; });
  }
  void WaitActivating() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return activation.submitted; });
  }
  void WaitActivationCancellation() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return activation.cancelling; });
  }
  void ReleaseActivation(iree_status_t status = iree_ok_status()) {
    std::lock_guard<std::mutex> lock(mutex);
    EXPECT_TRUE(activation.submitted);
    activation.ready = true;
    activation.status = status;
    observer.wake(observer.user_data);
  }
  void ReleaseRetirement() {
    std::lock_guard<std::mutex> lock(mutex);
    retired = true;
    observer.wake(observer.user_data);
  }
};

class HttpRouterRegistrationTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    managed_capacity_ = 1;
    factory_ = {.self = &managed_,
                .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
                .create = RegistrationDependency::Create,
                .query_status = RegistrationDependency::QueryStatus};
    HttpRouterTest::SetUp();
  }
  void Declare() {
    EXPECT_EQ(
        Read(
            Send(
                "POST", "/v1/models/register",
                R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets","admission":"drained"})"))
            .find("HTTP/1.1 202"),
        0u);
    EXPECT_EQ(managed_.created.load(), 0u);
  }
  void Register() {
    Declare();
    EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
    while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
           std::string::npos) {
    }
  }
  void Remove() {
    EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
    managed_.WaitRetiring();
    managed_.ReleaseRetirement();
    while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
           std::string::npos) {
    }
  }
  std::string WaitParameterOperation(const char* phase) {
    const std::string expected =
        std::string("\"parameter_operation\":\"") + phase + "\"";
    for (;;) {
      const auto state = Read(Send("GET", "/v1/status"));
      if (state.find(expected) != std::string::npos) {
        return state;
      }
    }
  }
  // The model alone is a dependency; every control/transport path is
  // production.
  RegistrationDependency managed_;
};

TEST_P(HttpRouterRegistrationTest, StalledObserverDoesNotRetainRemovedModel) {
  Register();
  ObservationClient observer;
  observer.Open(server_, "X-Loom-Model: dynamic\r\n");
  observer.WaitCredit(false);
  // The first response is already copied, but its only send credit remains
  // occupied throughout native model retirement and independent peer output.
  Remove();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "X-Loom-Model: dynamic\r\n"))
                .find("HTTP/1.1 404"),
            0u);
  // Reuse the registration slot before returning the old reader's credit.
  EXPECT_EQ(
      Read(
          Send(
              "POST", "/v1/models/register",
              R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets","admission":"drained"})"))
          .find("HTTP/1.1 202"),
      0u);
  const auto first = observer.Frame();
  const auto cursor = ObservationClient::Cursor(first);
  EXPECT_NE(first.find("\"model\":\"dynamic\""), std::string::npos);
  const auto ended = observer.Frame();
  EXPECT_EQ(ended.find("event: stream_end\n"), 0u);
  EXPECT_NE(ended.find("\"reason\":\"model_unregistered\""), std::string::npos);
  observer.Join();
  // The replacement registration never accepts the old incarnation's cursor.
  EXPECT_EQ(
      Read(Send("GET", "/logz/stream", "",
                "X-Loom-Model: dynamic\r\nLast-Event-ID: " + cursor + "\r\n"))
          .find("HTTP/1.1 409"),
      0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
}

TEST_P(HttpRouterRegistrationTest,
       DrainedActivationSurvivesDisconnectAndPreservesPeerProgress) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.parameters.active = false;
    managed_.parameters.memory.committed_bytes = 0;
    managed_.capacity_domain = &a_;
  }
  Declare();
  const int operation =
      Send("POST", "/v1/models/activate", R"({"model":"dynamic"})");
  managed_.WaitActivating();
  EXPECT_EQ(close(operation), 0);
  EXPECT_EQ(managed_.created.load(), 1u);
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  WaitHeld("dynamic", 1);
  const int peer = Send("POST", "/v1/test", R"({"model":"b"})");
  b_.WaitAccepted(1);
  // This peer treats accepted activation as releasable shared pressure. It
  // remains queued, while independent requests and health still complete.
  EXPECT_NE(Control("drain", "b").find(R"("queued_requests":1)"),
            std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_NE(
      WaitParameterOperation("activating").find(R"("admission":"drained")"),
      std::string::npos);
  managed_.ReleaseActivation();
  const auto ready = WaitParameterOperation("idle");
  EXPECT_NE(ready.find(R"("active":true,"backing":"elastic")"),
            std::string::npos);
  EXPECT_NE(ready.find(R"("held_requests":1)"), std::string::npos);
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_EQ(Read(request).find("HTTP/1.1 200"), 0u);
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_EQ(managed_.activation.starts, 1u);
  }
  Remove();
}

TEST_P(HttpRouterRegistrationTest, PinningDoesNotPrepareOrActivateAModel) {
  Declare();
  for (int i = 0; i < 2; ++i) {
    const auto pinned = Control("pin", "dynamic");
    EXPECT_EQ(pinned.find("HTTP/1.1 200"), 0u);
    EXPECT_NE(pinned.find(R"("parameter_pinned":true)"), std::string::npos);
    EXPECT_NE(pinned.find(R"("registration":"registered")"), std::string::npos);
    EXPECT_EQ(managed_.created.load(), 0u);
  }
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_NE(Control("unpin", "dynamic").find(R"("parameter_pinned":false)"),
            std::string::npos);
  EXPECT_EQ(managed_.created.load(), 0u);
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
         std::string::npos) {
  }
  EXPECT_TRUE(managed_.pinned.load());
  EXPECT_EQ(Control("unpin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_FALSE(managed_.pinned.load());
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_TRUE(managed_.pinned.load());
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, UnregisterJoinsAcceptedActivationFirst) {
  Declare();
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  EXPECT_TRUE(managed_.pinned.load());
  EXPECT_EQ(Control("unpin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivationCancellation();
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_FALSE(managed_.retiring);
  }
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  loom_serve_http_server_request_stop(server_);
  EXPECT_FALSE(returned_.load());
  managed_.ReleaseActivation();
  managed_.WaitRetiring();
  managed_.ReleaseRetirement();
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       UnsupportedDeclaredPinRetiresWithoutPublishing) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capabilities.pinning = false;
  }
  Declare();
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.WaitRetiring();
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find("registration cannot pin parameters"),
            std::string::npos);
  EXPECT_NE(state.find(R"("registration":"retiring")"), std::string::npos);
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("unpin", "dynamic").find("HTTP/1.1 409"), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, UnsupportedReadyPinKeepsTheModelUsable) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capabilities.pinning = false;
  }
  Register();
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 501"), 0u);
  EXPECT_EQ(Control("unpin", "dynamic").find("HTTP/1.1 501"), 0u);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest,
       RemovalWithdrawsWarmupBeforeAdmittedWorkEnds) {
  Register();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 409"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  WaitParameterOperation("waiting");
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    EXPECT_EQ(managed_.activation.starts, 0u);
  }
  managed_.service.Release();
  EXPECT_TRUE(Read(request).empty());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, UnsupportedWarmupKeepsFailureModelLocal) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capabilities.activation = false;
  }
  Declare();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  EXPECT_NE(Read(Send("GET", "/v1/status"))
                .find("registration cannot activate parameters"),
            std::string::npos);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_FALSE(returned_.load());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, ActivationFailureDrainsIndependentOwners) {
  Declare();
  const int peer = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  expected_code_ = IREE_STATUS_UNAVAILABLE;
  managed_.ReleaseActivation(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "parameter read failed"));
  held_.WaitStopping();
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_FALSE(returned_.load());
  held_.Release();
  EXPECT_TRUE(Read(peer).empty());
  EXPECT_FALSE(returned_.load());
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.ReleaseRetirement();
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
  EXPECT_NE(iree::Status::ToString(status_).find("parameter read failed"),
            std::string::npos);
}

TEST_P(HttpRouterRegistrationTest,
       DeactivationFinishesAdmittedWorkAndSurvivesClientDisconnect) {
  Register();
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  const int first =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":1})");
  managed_.service.WaitSubmitted();
  const int second =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":2})");
  managed_.service.WaitAccepted(2);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic","value":3})");
  WaitHeld("dynamic", 1);
  const int operation =
      Send("POST", "/v1/models/deactivate", R"({"model":"dynamic"})");
  while (Read(Send("GET", "/v1/status"))
             .find(R"("parameter_operation":"deactivating")") ==
         std::string::npos) {
  }
  EXPECT_EQ(close(operation), 0);
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.service.Release();
  EXPECT_NE(Read(first).find(R"("value":1)"), std::string::npos);
  managed_.service.WaitSubmitted();
  const auto waiting = Read(Send("GET", "/v1/status"));
  EXPECT_NE(waiting.find(R"("parameter_operation":"deactivating")"),
            std::string::npos);
  EXPECT_NE(waiting.find(R"("deactivations":0)"), std::string::npos);
  managed_.service.Release();
  EXPECT_NE(Read(second).find(R"("value":2)"), std::string::npos);
  std::string cold;
  do {
    cold = Read(Send("GET", "/v1/status"));
  } while (cold.find(R"("active":false,"backing":"elastic")") ==
           std::string::npos);
  EXPECT_NE(cold.find(R"("committed_bytes":0)"), std::string::npos);
  EXPECT_NE(cold.find(R"("deactivations":1)"), std::string::npos);
  EXPECT_NE(cold.find(R"("held_requests":1)"), std::string::npos);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_NE(Read(held).find(R"("value":3)"), std::string::npos);
  Remove();
}

TEST_P(HttpRouterRegistrationTest,
       FixedDeactivationRejectionPreservesTheModel) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.parameters.elastic = false;
  }
  Register();
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const auto rejected = Control("deactivate", "dynamic");
  EXPECT_EQ(rejected.find("HTTP/1.1 409"), 0u);
  EXPECT_NE(rejected.find("fixed parameter backing"), std::string::npos);
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find(R"("active":true,"backing":"fixed")"),
            std::string::npos);
  EXPECT_EQ(state.find("committed_bytes"), std::string::npos);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_EQ(Read(request).find("HTTP/1.1 200"), 0u);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, DeactivationFailureDrainsIndependentOwners) {
  managed_.fail_deactivation = true;
  Register();
  const int peer = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  Control("drain", "dynamic");
  expected_code_ = IREE_STATUS_UNAVAILABLE;
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  held_.WaitStopping();
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.ReleaseRetirement();
  EXPECT_FALSE(returned_.load());
  held_.Release();
  EXPECT_TRUE(Read(peer).empty());
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
  EXPECT_NE(iree::Status::ToString(status_).find("parameter release failed"),
            std::string::npos);
}

TEST_P(HttpRouterRegistrationTest, LocalPreparationFailureKeepsPeersServing) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation_failure =
        RegistrationDependency::PreparationFailure::kLocal;
  }
  Declare();
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.WaitRetiring();
  EXPECT_NE(Read(Send("GET", "/v1/status")).find("model prepare failed"),
            std::string::npos);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_FALSE(returned_.load());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, SharedPreparationFailureDrainsEveryOwner) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation_failure =
        RegistrationDependency::PreparationFailure::kShared;
  }
  Declare();
  const int peer = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  expected_code_ = IREE_STATUS_UNAVAILABLE;
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  held_.WaitStopping();
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.ReleaseRetirement();
  EXPECT_FALSE(returned_.load());
  held_.Release();
  EXPECT_TRUE(Read(peer).empty());
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
  EXPECT_NE(iree::Status::ToString(status_).find("shared reclamation failed"),
            std::string::npos);
}

TEST_P(HttpRouterRegistrationTest, RemovalSupersedesPendingDeactivation) {
  Register();
  const int request = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find(R"("parameter_operation":"idle")"), std::string::npos);
  EXPECT_NE(state.find(R"("deactivations":0)"), std::string::npos);
  managed_.service.Release();
  EXPECT_TRUE(Read(request).empty());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, EmptyHeaderCannotSelectAnUnusedSlot) {
  EXPECT_EQ(
      Read(Send("POST", "/v1/checkpoints/empty", "", "X-Loom-Model: \r\n"))
          .find("HTTP/1.1 404"),
      0u);
  EXPECT_EQ(
      Read(Send("GET", "/logz", "", "X-Loom-Model: \r\n")).find("HTTP/1.1 404"),
      0u);
}

TEST_P(HttpRouterRegistrationTest,
       ParameterOperationsValidateTheirSelectorAndOwnership) {
  for (const auto* operation : {"activate", "deactivate", "pin", "unpin"}) {
    for (const char* body :
         {R"({})", R"({"model":""})", R"({"model":"dynamic","extra":true})",
          R"({"model":"dynamic","model":"a"})"}) {
      EXPECT_EQ(Read(Send("POST", std::string("/v1/models/") + operation, body))
                    .find("HTTP/1.1 400"),
                0u);
    }
    EXPECT_EQ(Control(operation, "absent").find("HTTP/1.1 404"), 0u);
    EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
    EXPECT_EQ(Control(operation, "a").find("HTTP/1.1 409"), 0u);
    EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  }
}

TEST_P(HttpRouterRegistrationTest,
       RetirementKeepsSharedCapacityWaitersQueuedUntilOwnershipReturns) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capacity_domain = &a_;
  }
  Register();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  const int peer = Send("POST", "/v1/test", R"({"model":"b"})");
  EXPECT_TRUE(iree_any_bit_set(b_.WaitAdmission(),
                               LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY));
  EXPECT_NE(Control("drain", "b").find(R"("queued_requests":1)"),
            std::string::npos);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Read(Send("GET", "/healthz")).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.ReleaseRetirement();
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       RetirementDoesNotInventActivityInAnotherCapacityDomain) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capacity_domain = &managed_;
  }
  Register();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  const int peer = Send("POST", "/v1/test", R"({"model":"b"})");
  EXPECT_FALSE(iree_any_bit_set(b_.WaitAdmission(),
                                LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY));
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       GracefulRemovalFinishesAdmittedQueuesBeforeRetirement) {
  Register();
  const int first =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":1})");
  managed_.service.WaitSubmitted();
  const int second =
      Send("POST", "/v1/test", R"({"model":"dynamic","value":2})");
  managed_.service.WaitAccepted(2);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic","value":3})");
  WaitHeld("dynamic", 1);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  managed_.service.Release();
  EXPECT_NE(Read(first).find(R"("value":1)"), std::string::npos);
  managed_.service.WaitSubmitted();
  managed_.service.Release();
  EXPECT_NE(Read(second).find(R"("value":2)"), std::string::npos);
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(
          Send(
              "POST", "/v1/models/register",
              R"({"kind":"embedding","name":"dynamic","source":"model","assets":"assets"})"))
          .find("HTTP/1.1 409"),
      0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       AbortAndServerStopStillWaitForActualModelOwnership) {
  Register();
  const int first = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitSubmitted();
  const int queued = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  managed_.service.WaitAccepted(2);
  Control("drain", "dynamic");
  const int held = Send("POST", "/v1/test", R"({"model":"dynamic"})");
  WaitHeld("dynamic", 1);
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  managed_.service.WaitStopping();
  EXPECT_EQ(Read(held).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  managed_.service.Release();
  EXPECT_TRUE(Read(first).empty());
  EXPECT_TRUE(Read(queued).empty());
  managed_.WaitRetiring();
  loom_serve_http_server_request_stop(server_);
  EXPECT_FALSE(returned_.load());
  managed_.ReleaseRetirement();
  owner_.join();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest,
       DiscoveryExcludesUnpreparedAndRemovingModels) {
  Declare();
  EXPECT_EQ(Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")"),
            std::string::npos);
  EXPECT_NE(Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")"),
            std::string::npos);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  while (Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")") ==
         std::string::npos) {
  }
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_NE(Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")"),
            std::string::npos);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  EXPECT_EQ(Read(Send("GET", "/v1/models")).find(R"("id":"dynamic")"),
            std::string::npos);
  EXPECT_NE(
      Read(Send("GET", "/v1/status")).find(R"("registration":"retiring")"),
      std::string::npos);
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
}

TEST_P(HttpRouterRegistrationTest,
       ReleaseFailureRetainsOwnershipUntilExplicitRetry) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.capacity_domain = &a_;
  }
  managed_.fail_release = true;
  Register();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  managed_.ReleaseRetirement();
  std::string status;
  do {
    status = Read(Send("GET", "/v1/status"));
  } while (status.find(R"("registration":"release_failed")") ==
           std::string::npos);
  EXPECT_NE(status.find("platform release failed"), std::string::npos);
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_EQ(managed_.release_attempts.load(), 1u);
  // Quiescent failure retains backing but promises no progress. A peer must
  // not park indefinitely waiting for an ownership wake that cannot arrive.
  const int peer = Send("POST", "/v1/test", R"({"model":"b"})");
  EXPECT_FALSE(iree_any_bit_set(b_.WaitAdmission(),
                                LOOM_SERVE_HTTP_SERVICE_FLAG_SHARED_ACTIVITY));
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.release_attempts.load(), 2u);
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterRegistrationTest,
                         ::testing::Values(1, 4));

}  // namespace
