// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/server/router_test_fixture.h"

namespace {

const iree_string_view_t kDeviceNames[] = {IREE_SV("device")};

using loom_serve::testing::HttpRouterTest;
using loom_serve::testing::loom_serve_test_model_service_t;
using loom_serve::testing::ObservationClient;

// A model dependency with independently controlled service and final device
// ownership. HTTP admission, the registry, task owner and retirement policy are
// the production implementations under test.
struct loom_serve_test_registration_dependency_t {
  enum class loom_serve_test_registration_preparation_failure_t {
    kNone,
    kLocal,
    kShared
  };

  // Requests reach the same held-completion service used by the routing tests.
  loom_serve_test_model_service_t service = {
      .name = "dynamic",
      .hold_completion = true,
  };
  // Immutable deployment role supplied by the real configuration parser.
  loom_serve_model_kind_t kind = LOOM_SERVE_MODEL_KIND_EMBEDDING;
  // Model-protocol counters supplied by the dependency, not synthesized by
  // routing.
  loom_serve_telemetry_counters_t counters = {};
  // Historical terminal observations owned by the dependency until retirement.
  loom_serve_telemetry_latency_t latency = {};
  // Serializes retirement readiness and wake detachment with the owner.
  std::mutex mutex;
  // Outer-test synchronization on actual retirement entry.
  std::condition_variable condition;
  // Final device-borrower wake, independent of the service observer.
  loom_serve_wake_callback_t observer = {};
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
  loom_serve_test_registration_preparation_failure_t preparation_failure =
      loom_serve_test_registration_preparation_failure_t::kNone;
  // Independently held compiler/tokenizer dependency, guarded by mutex.
  struct {
    // Preparation can publish its service after this dependency completes.
    bool ready = true;
    // Owner observations, used to join an explicit test wake without polling.
    uint64_t advances = 0;
    // Most recent preparation policy supplied by the production registry.
    loom_serve_registration_prepare_flags_t flags = 0;
  } preparation;
  // Terminal compiler evidence supplied independently of lifecycle timing.
  loom_serve_compilation_report_t compilation = {};
  // Parameter state belongs to the model dependency and changes on owner turns.
  loom_serve_residency_snapshot_t parameters = {
      .active = true,
      .elastic = true,
      .memory = {.reserved_bytes = 4096, .committed_bytes = 4096},
      .activity = {.activations = 1},
  };
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

  struct loom_serve_test_registration_operation_t {
    // Test-owned model dependency, live beyond the owning registry.
    loom_serve_test_registration_dependency_t* dependency;
  };

  static iree_status_t Prepare(void* self,
                               loom_serve_registration_prepare_flags_t flags,
                               bool* complete, loom_serve_service_t* service,
                               loom_serve_compilation_report_t* out_report) {
    auto* dependency =
        static_cast<loom_serve_test_registration_operation_t*>(self)
            ->dependency;
    std::lock_guard<std::mutex> lock(dependency->mutex);
    dependency->preparation.flags = flags;
    ++dependency->preparation.advances;
    dependency->condition.notify_all();
    *complete = true;
    if (iree_any_bit_set(flags, LOOM_SERVE_REGISTRATION_PREPARE_FLAG_CANCEL)) {
      return iree_make_status(IREE_STATUS_CANCELLED, "preparation cancelled");
    }
    if (!dependency->preparation.ready) {
      *complete = false;
      return iree_ok_status();
    }
    *out_report = dependency->compilation;
    if (dependency->preparation_failure !=
        loom_serve_test_registration_preparation_failure_t::kNone) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE, "model prepare failed");
    }
    *service =
        dependency->service.Interface(nullptr, dependency->capacity_domain);
    service->kind = dependency->kind;
    if (service->kind == LOOM_SERVE_MODEL_KIND_ACTIVITY) {
      service->endpoint = nullptr;
      service->accept = nullptr;
    }
    service->telemetry.stream = dependency->stream;
    return iree_ok_status();
  }
  static iree_status_t Retire(void* self, bool* complete) {
    auto* dependency =
        static_cast<loom_serve_test_registration_operation_t*>(self)
            ->dependency;
    EXPECT_FALSE(
        loom_serve_test_model_service_t::IsPending(&dependency->service));
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
    auto* operation =
        static_cast<loom_serve_test_registration_operation_t*>(self);
    ++operation->dependency->destroyed;
    delete operation;
  }
  static loom_serve_registration_parameters_t Parameters(
      void* self, loom_serve_model_component_t component) {
    const auto parameters =
        static_cast<loom_serve_test_registration_operation_t*>(self)
            ->dependency->parameters;
    return {
        .phase = "ready",
        .prepared = true,
        .deactivatable = parameters.elastic,
        .parameters = parameters,
    };
  }
  static void SetPinned(void* self, loom_serve_model_component_t component,
                        bool pinned) {
    static_cast<loom_serve_test_registration_operation_t*>(self)
        ->dependency->pinned = pinned;
  }
  static iree_status_t Deactivate(void* self,
                                  loom_serve_model_component_t component,
                                  bool* out_complete) {
    *out_complete = true;
    auto* dependency =
        static_cast<loom_serve_test_registration_operation_t*>(self)
            ->dependency;
    EXPECT_FALSE(
        loom_serve_test_model_service_t::IsPending(&dependency->service));
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
  static iree_status_t Activate(void* self,
                                loom_serve_model_component_t component,
                                loom_serve_activation_flags_t flags,
                                loom_serve_activation_phase_t* phase,
                                iree_status_t* component_status) {
    *component_status = iree_ok_status();
    auto* dependency =
        static_cast<loom_serve_test_registration_operation_t*>(self)
            ->dependency;
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
      loom_serve_wake_callback_t observer,
      loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
      loom_serve_registration_t* out_registration) {
    auto* dependency =
        static_cast<loom_serve_test_registration_dependency_t*>(self);
    // Import fixture configuration before transferring it to the model owner.
    std::lock_guard<std::mutex> lock(dependency->mutex);
    dependency->service.name = configuration->name.data;
    dependency->kind = configuration->kind;
    dependency->service.counters = &dependency->counters;
    dependency->service.latency = &dependency->latency;
    dependency->observer = observer;
    dependency->stream = stream;
    ++dependency->created;
    *out_registration = {
        .self = new loom_serve_test_registration_operation_t{dependency},
        .prepare = Prepare,
        .retire = Retire,
        .parameters = Parameters,
        .set_pinned = dependency->capabilities.pinning ? SetPinned : nullptr,
        .activate = dependency->capabilities.activation ? Activate : nullptr,
        .deactivate = Deactivate,
        .destroy = Destroy,
    };
    return iree_ok_status();
  }
  static iree_status_t QueryStatus(void* self) {
    auto* dependency =
        static_cast<loom_serve_test_registration_dependency_t*>(self);
    std::lock_guard<std::mutex> lock(dependency->mutex);
    return dependency->preparation_failure ==
                   loom_serve_test_registration_preparation_failure_t::kShared
               ? iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "shared reclamation failed")
               : iree_ok_status();
  }
  void WaitRetiring() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return retiring; });
  }
  loom_serve_registration_prepare_flags_t ObservePreparation() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this] { return preparation.advances != 0; });
    const auto previous = preparation.advances;
    observer.wake(observer.user_data);
    condition.wait(
        lock, [this, previous] { return preparation.advances != previous; });
    return preparation.flags;
  }
  void ReleasePreparation() {
    std::lock_guard<std::mutex> lock(mutex);
    preparation.ready = true;
    observer.wake(observer.user_data);
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
    factory_ = {
        .self = &managed_,
        .kinds = (1u << LOOM_SERVE_MODEL_KIND_EMBEDDING) |
                 (1u << LOOM_SERVE_MODEL_KIND_TEXT) |
                 (1u << LOOM_SERVE_MODEL_KIND_ACTIVITY),
        .device_names = {1, kDeviceNames},
        .create = loom_serve_test_registration_dependency_t::Create,
        .query_status = loom_serve_test_registration_dependency_t::QueryStatus,
    };
    HttpRouterTest::SetUp();
  }
  void Declare(const std::string& kind = "embedding") {
    EXPECT_EQ(
        Read(
            Send(
                "POST", "/v1/models/register",
                R"({"name":"dynamic","admission":"drained","definition":{"kind":")" +
                    kind +
                    R"(","prepare":"/model/prepare.loom","control":"/model/control.loom","sources":["/model/kernel.loom"],"assets":"/assets"}})"))
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
  static uint64_t PreparationDuration(const std::string& status) {
    const std::string key = "\"preparation_duration_ns\":";
    const auto position = status.find(key);
    EXPECT_NE(position, std::string::npos);
    return position == std::string::npos
               ? 0
               : std::stoull(status.substr(position + key.size()));
  }
  // The model alone is a dependency; every control/transport path is
  // production.
  loom_serve_test_registration_dependency_t managed_;
};

TEST_P(HttpRouterRegistrationTest, NativeOnlyMetadataAndHttpRejectionAgree) {
  Declare("activity");
  const auto declared = Read(Send("GET", "/v1/status"));
  EXPECT_NE(declared.find(R"("kind":"activity","tokenize":false)"),
            std::string::npos);
  const auto cold =
      Read(Send("POST", "/v1/audio/transcriptions", R"({"model":"dynamic"})"));
  EXPECT_EQ(cold.find("HTTP/1.1 404"), 0u);
  EXPECT_NE(cold.find("model has no HTTP inference endpoint"),
            std::string::npos);
  EXPECT_EQ(managed_.created.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
         std::string::npos) {
  }
  const auto models = Read(Send("GET", "/v1/models"));
  EXPECT_NE(models.find(R"("kind":"activity","tokenize":false)"),
            std::string::npos);
  const auto ready =
      Read(Send("POST", "/v1/audio/transcriptions", R"({"model":"dynamic"})"));
  EXPECT_EQ(ready.find("HTTP/1.1 404"), 0u);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find("model=\"dynamic\",kind=\"activity\""),
            std::string::npos);
  EXPECT_NE(metrics.find("loom_serve_model_held_requests{model=\"dynamic\","
                         "kind=\"activity\",incarnation=\"1\"} 0"),
            std::string::npos);
  Remove();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest, StalledObserverDoesNotRetainRemovedModel) {
  Register();
  ObservationClient observer;
  ASSERT_NO_FATAL_FAILURE(observer.Open(server_, "X-Loom-Model: dynamic\r\n"));
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
              R"({"name":"dynamic","admission":"drained","definition":{"kind":"embedding","prepare":"/model/prepare.loom","control":"/model/control.loom","sources":["/model/kernel.loom"],"assets":"/assets"}})"))
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
       ParameterPressureUsesModelObservationsAndRetiresWithIncarnation) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.parameters.pin_count = 3;
    managed_.parameters.admission = {
        .waiting = true,
        .count = 7,
        .duration_ns = 5000000000,
        .age_ns = 2500000000,
    };
  }
  Register();
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(
      state.find(
          R"("pins":3,"admission":{"waiting":true,"count":7,"duration_seconds":5.000000000,"age_seconds":2.500000000})"),
      std::string::npos);
  const auto metrics = Read(Send("GET", "/metrics"));
  EXPECT_NE(metrics.find(
                "loom_serve_model_parameter_pins{model=\"dynamic\",kind="
                "\"embedding\",incarnation=\"1\",component=\"primary\"} 3\n"
                "loom_serve_model_parameter_admission_waiting{model="
                "\"dynamic\",kind=\"embedding\",incarnation=\"1\",component="
                "\"primary\"} 1\n"
                "loom_serve_model_parameter_admission_waits_total{model="
                "\"dynamic\",kind=\"embedding\",incarnation=\"1\",component="
                "\"primary\"} 7\n"
                "loom_serve_model_parameter_admission_wait_seconds_total{model="
                "\"dynamic\",kind=\"embedding\",incarnation=\"1\",component="
                "\"primary\"} 5\n"
                "loom_serve_model_parameter_admission_wait_age_seconds{model="
                "\"dynamic\",kind=\"embedding\",incarnation=\"1\",component="
                "\"primary\"} 2.5\n"),
            std::string::npos);
  // A dependency without parameter observations is absent, not a zero sample.
  EXPECT_EQ(metrics.find("loom_serve_model_parameter_pins{model=\"a\""),
            std::string::npos);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  EXPECT_EQ(Read(Send("GET", "/metrics"))
                .find("loom_serve_model_parameter_pins{model=\"dynamic\""),
            std::string::npos);
  EXPECT_EQ(Read(Send("GET", "/v1/status")).find("\"pins\":"),
            std::string::npos);
  managed_.ReleaseRetirement();
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
    EXPECT_NE(pinned.find(R"("pinned":true)"), std::string::npos);
    EXPECT_NE(pinned.find(R"("registration":"registered")"), std::string::npos);
    EXPECT_EQ(managed_.created.load(), 0u);
  }
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 409"), 0u);
  EXPECT_NE(Control("unpin", "dynamic").find(R"("pinned":false)"),
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

TEST_P(HttpRouterRegistrationTest,
       PreparationDemandTracksInferenceRatherThanHeldUtilities) {
  // Observe held claims without the ordinary WaitHeld helper's drain action:
  // this test's preparation remains open behind an unresolved dependency.
  const auto wait_held = [this](size_t count) {
    const std::string expected =
        "\"held_requests\":" + std::to_string(count) + ",";
    while (Read(Send("GET", "/v1/status")).find(expected) ==
           std::string::npos) {
    }
  };
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation.ready = false;
  }
  Declare();
  EXPECT_EQ(Control("pin", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(managed_.ObservePreparation(), 0u);
  const int utility = Send("POST", "/tokenize", R"({"model":"dynamic"})");
  wait_held(1);
  EXPECT_EQ(managed_.ObservePreparation(), 0u);

  const int inference =
      Send("POST", "/v1/embeddings", R"({"model":"dynamic"})");
  wait_held(2);
  EXPECT_EQ(managed_.ObservePreparation(),
            LOOM_SERVE_REGISTRATION_PREPARE_FLAG_DEMAND);
  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(managed_.ObservePreparation(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(managed_.ObservePreparation(),
            LOOM_SERVE_REGISTRATION_PREPARE_FLAG_DEMAND);

  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(inference, SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
            0);
  ASSERT_EQ(close(inference), 0);
  wait_held(1);
  EXPECT_EQ(managed_.ObservePreparation(), 0u);
  // Cancel the outstanding dependency through real removal. Utilities never
  // turn this cancellation into permission to activate parameters.
  Remove();
  EXPECT_EQ(Read(utility).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(managed_.preparation.flags,
            LOOM_SERVE_REGISTRATION_PREPARE_FLAG_CANCEL);
}

TEST_P(HttpRouterRegistrationTest, ExplicitWarmupSuppliesPreparationDemand) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation.ready = false;
  }
  Declare();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(managed_.ObservePreparation(),
            LOOM_SERVE_REGISTRATION_PREPARE_FLAG_DEMAND);
  managed_.ReleasePreparation();
  managed_.WaitActivating();
  managed_.ReleaseActivation();
  WaitParameterOperation("idle");
  Remove();
}

TEST_P(HttpRouterRegistrationTest,
       PreparationElapsedExcludesDormancyAndLaterActivation) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation.ready = false;
    managed_.compilation.commands = 3;
    managed_.compilation.stage_duration_sum_ns = 123;
  }
  Declare();
  EXPECT_EQ(PreparationDuration(Read(Send("GET", "/v1/status"))), 0u);
  EXPECT_EQ(managed_.created.load(), 0u);
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.ObservePreparation();
  const auto preparing = PreparationDuration(Read(Send("GET", "/v1/status")));
  EXPECT_GT(preparing, 0u);
  EXPECT_GE(PreparationDuration(Read(Send("GET", "/v1/status"))), preparing);

  managed_.ReleasePreparation();
  std::string ready;
  do {
    ready = Read(Send("GET", "/v1/status"));
  } while (ready.find(R"("registration":"ready")") == std::string::npos);
  const auto completed = PreparationDuration(ready);
  EXPECT_GE(completed, preparing);
  EXPECT_NE(ready.find(R"("stage_duration_sum_ns":123)"), std::string::npos);
  EXPECT_EQ(ready.find(R"("duration_ns":)"), std::string::npos);
  EXPECT_EQ(PreparationDuration(Read(Send("GET", "/v1/status"))), completed);

  EXPECT_EQ(Control("drain", "dynamic").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  EXPECT_EQ(PreparationDuration(Read(Send("GET", "/v1/status"))), completed);
  managed_.ReleaseActivation();
  EXPECT_EQ(PreparationDuration(WaitParameterOperation("idle")), completed);
  Remove();
}

TEST_P(HttpRouterRegistrationTest, RetiringComponentTimingIsUnavailable) {
  const auto response = Read(Send(
      "POST", "/v1/models/register",
      R"({"name":"dynamic","definition":{"kind":"text",)"
      R"("prepare":"/model/prepare.loom","control":"/model/control.loom",)"
      R"("weights":"/model/weights.safetensors","tokenizer":"/model/tokenizer.json",)"
      R"("sources":["/model/kernel.loom"],"image":{)"
      R"("prepare":"/image/prepare.loom","control":"/image/control.loom",)"
      R"("assets":"/image"}}})"));
  ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
  while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
         std::string::npos) {
  }
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  const auto state = Read(Send("GET", "/v1/status"));
  EXPECT_NE(state.find(R"("image":{"phase":"retiring","pinned":false})"),
            std::string::npos)
      << state;
  EXPECT_GT(PreparationDuration(state), 0u);
  managed_.ReleaseRetirement();
}

TEST_P(HttpRouterRegistrationTest, CancelledPreparationStopsBeforeRetirement) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation.ready = false;
  }
  Declare();
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.ObservePreparation();
  const auto preparing = PreparationDuration(Read(Send("GET", "/v1/status")));
  EXPECT_GT(preparing, 0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  const auto cancelled = PreparationDuration(Read(Send("GET", "/v1/status")));
  EXPECT_GE(cancelled, preparing);
  EXPECT_EQ(PreparationDuration(Read(Send("GET", "/v1/status"))), cancelled);
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
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

TEST_P(HttpRouterRegistrationTest, ActivationFailureRetiresOnlyItsModel) {
  Declare();
  const int peer = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  EXPECT_EQ(Control("activate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitActivating();
  managed_.ReleaseActivation(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "parameter read failed"));
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_FALSE(returned_.load());
  EXPECT_NE(Read(Send("GET", "/v1/status")).find("parameter read failed"),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/test", R"({"model":"dynamic"})"))
                .find("HTTP/1.1 503"),
            0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  held_.Release();
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_FALSE(returned_.load());
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  Remove();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
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
  EXPECT_NE(rejected.find("cannot release its parameter backing"),
            std::string::npos);
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

TEST_P(HttpRouterRegistrationTest, DeactivationFailureRetiresOnlyItsModel) {
  managed_.fail_deactivation = true;
  Register();
  const int peer = Send("POST", "/v1/test", R"({"model":"held"})");
  held_.WaitSubmitted();
  Control("drain", "dynamic");
  EXPECT_EQ(Control("deactivate", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  EXPECT_EQ(managed_.destroyed.load(), 0u);
  EXPECT_NE(Read(Send("GET", "/v1/status")).find("parameter release failed"),
            std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/test", R"({"model":"dynamic"})"))
                .find("HTTP/1.1 503"),
            0u);
  EXPECT_FALSE(returned_.load());
  held_.Release();
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  Remove();
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

TEST_P(HttpRouterRegistrationTest, LocalPreparationFailureKeepsPeersServing) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation_failure = loom_serve_test_registration_dependency_t::
        loom_serve_test_registration_preparation_failure_t::kLocal;
  }
  Declare();
  EXPECT_EQ(Control("undrain", "dynamic").find("HTTP/1.1 200"), 0u);
  managed_.WaitRetiring();
  const auto failed = Read(Send("GET", "/v1/status"));
  EXPECT_NE(failed.find("model prepare failed"), std::string::npos);
  const auto duration = PreparationDuration(failed);
  EXPECT_GT(duration, 0u);
  EXPECT_EQ(PreparationDuration(Read(Send("GET", "/v1/status"))), duration);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  EXPECT_FALSE(returned_.load());
  Remove();
}

TEST_P(HttpRouterRegistrationTest, SharedPreparationFailureDrainsEveryOwner) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.preparation_failure = loom_serve_test_registration_dependency_t::
        loom_serve_test_registration_preparation_failure_t::kShared;
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
                               LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY));
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
                                LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY));
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
              R"({"name":"dynamic","definition":{"kind":"embedding","prepare":"/model/prepare.loom","control":"/model/control.loom","sources":["/model/kernel.loom"],"assets":"/assets"}})"))
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
                                LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY));
  EXPECT_EQ(Read(peer).find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  EXPECT_EQ(managed_.release_attempts.load(), 2u);
  EXPECT_EQ(managed_.destroyed.load(), 1u);
}

class HttpRouterRegistrationLatencyTest : public HttpRouterRegistrationTest {
 protected:
  void SetUp() override {
    for (auto* service : {&a_, &b_, &held_, &other_, &escaped_}) {
      service->latency = &empty_latency_;
    }
    HttpRouterRegistrationTest::SetUp();
  }

  // Immutable empty history for the externally owned peer dependencies.
  loom_serve_telemetry_latency_t empty_latency_ = {};
};

TEST_P(HttpRouterRegistrationLatencyTest,
       TerminalDistributionsSurviveDestructionAndNameReuse) {
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.counters = {.requests = 3, .completed = 2, .rejected = 1};
    loom_serve_telemetry_latency_observe(
        &managed_.latency.outcomes[LOOM_SERVE_TELEMETRY_COMPLETED], 100000000);
    loom_serve_telemetry_latency_observe(
        &managed_.latency.outcomes[LOOM_SERVE_TELEMETRY_COMPLETED], 2000000000);
    loom_serve_telemetry_latency_observe(
        &managed_.latency.outcomes[LOOM_SERVE_TELEMETRY_REJECTED], 3000000);
    loom_serve_telemetry_latency_observe(&managed_.latency.first_output,
                                         10000000);
    loom_serve_telemetry_latency_observe(&managed_.latency.first_output,
                                         500000000);
  }
  Register();
  const auto live = Read(Send("GET", "/metrics"));
  EXPECT_NE(live.find("loom_serve_model_request_duration_seconds_count{"
                      "model=\"dynamic\",kind=\"embedding\",incarnation=\"1\","
                      "outcome=\"completed\"} 2\n"),
            std::string::npos);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  managed_.WaitRetiring();
  // The registry copied the distributions before entering model retirement.
  // Reusing dependency storage cannot mutate that terminal snapshot.
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.latency = {};
  }
  for (const auto& response : {Read(Send("GET", "/metrics")), live}) {
    EXPECT_NE(response.find("loom_serve_server_request_duration_seconds_count{"
                            "outcome=\"completed\"} 2\n"),
              std::string::npos);
    EXPECT_NE(response.find("loom_serve_server_request_duration_seconds_sum{"
                            "outcome=\"completed\"} 2.1\n"),
              std::string::npos);
    EXPECT_NE(response.find("loom_serve_server_request_duration_seconds_count{"
                            "outcome=\"rejected\"} 1\n"),
              std::string::npos);
    EXPECT_NE(
        response.find("loom_serve_server_first_output_seconds_count{} 2\n"),
        std::string::npos);
    EXPECT_NE(
        response.find("loom_serve_server_first_output_seconds_sum{} 0.51\n"),
        std::string::npos);
  }
  managed_.ReleaseRetirement();
  while (Read(Send("GET", "/v1/status")).find(R"("id":"dynamic")") !=
         std::string::npos) {
  }
  {
    std::lock_guard<std::mutex> lock(managed_.mutex);
    managed_.retired = false;
    managed_.retiring = false;
    managed_.counters = {.requests = 1, .completed = 1};
    loom_serve_telemetry_latency_observe(
        &managed_.latency.outcomes[LOOM_SERVE_TELEMETRY_COMPLETED], 7000000000);
    loom_serve_telemetry_latency_observe(&managed_.latency.first_output,
                                         1000000000);
  }
  // Declare() expects the initial factory creation count. Register explicitly
  // here to exercise reuse of a name without resetting its history.
  EXPECT_EQ(
      Read(
          Send(
              "POST", "/v1/models/register",
              R"({"name":"dynamic","definition":{"kind":"embedding","prepare":"/model/prepare.loom","control":"/model/control.loom","sources":["/model/kernel.loom"],"assets":"/assets"}})"))
          .find("HTTP/1.1 202"),
      0u);
  while (managed_.created.load() != 2) {
    Read(Send("GET", "/v1/status"));
  }
  while (Read(Send("GET", "/v1/status")).find(R"("registration":"ready")") ==
         std::string::npos) {
  }
  EXPECT_NE(Read(Send("GET", "/metrics"))
                .find("loom_serve_model_request_duration_seconds_count{"
                      "model=\"dynamic\",kind=\"embedding\",incarnation=\"2\","
                      "outcome=\"completed\"} 1\n"),
            std::string::npos);
  Remove();
  const auto removed = Read(Send("GET", "/metrics"));
  EXPECT_NE(removed.find("loom_serve_server_request_duration_seconds_count{"
                         "outcome=\"completed\"} 3\n"),
            std::string::npos);
  EXPECT_NE(removed.find("loom_serve_server_request_duration_seconds_sum{"
                         "outcome=\"completed\"} 9.1\n"),
            std::string::npos);
  EXPECT_NE(removed.find("loom_serve_server_request_duration_seconds_count{"
                         "outcome=\"rejected\"} 1\n"),
            std::string::npos);
  EXPECT_EQ(removed.find("model=\"dynamic\""), std::string::npos);
  EXPECT_NE(removed.find("loom_serve_server_first_output_seconds_count{} 3\n"),
            std::string::npos);
  EXPECT_NE(removed.find("loom_serve_server_first_output_seconds_sum{} 1.51\n"),
            std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterRegistrationLatencyTest,
                         ::testing::Values(1, 4));
INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterRegistrationTest,
                         ::testing::Values(1, 4));

}  // namespace
