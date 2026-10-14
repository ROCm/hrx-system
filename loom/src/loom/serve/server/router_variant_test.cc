// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/variant_pool.h"
#include "loom/serve/server/router_test_fixture.h"

namespace {

using loom_serve::testing::HttpRouterTest;
using loom_serve::testing::ModelService;

// Only compiler/device readiness is controlled here. Real transport, routing,
// registry retirement and variant ownership must handle every intervening
// request while these dependencies remain outstanding.
struct VariantDependency : ModelService {
  // Real cache, accessed only by the serving owner except producer completion.
  loom_serve_variant_pool_t* variants = nullptr;
  // Acquired policy choice; its lifetime belongs to the dependency.
  loom_serve_variant_entry_t* selected = nullptr;
  // Published identity copied for the outer test, never a borrowed pool query.
  std::atomic<uint64_t> original{0};
  // Terminal model destruction, independent of connection completion.
  std::atomic<unsigned> destroyed{0};
  // Service counters retained until the registry copies terminal totals.
  loom_serve_telemetry_counters_t totals = {};
  // Registration-owned stream borrowed through service retirement.
  loom_serve_telemetry_stream_t* stream = nullptr;
  // Simulated compiler borrows protected by the inherited mutex.
  std::vector<loom_serve_variant_entry_t*> compilers;
  // Total accepted producer jobs, also protected by the mutex.
  size_t started = 0;
  // Test cleanup lets any just-accepted job complete before owner shutdown.
  bool complete_immediately = false;
  // One accepted policy change, independently controlled by the outer test.
  struct {
    // Acquired candidate, accessed by the owner only.
    loom_serve_variant_entry_t* entry = nullptr;
    // Borrowed completion storage until owner publication.
    loom_serve_variant_selection_fn_t completion = nullptr;
    // Opaque caller state whose lifetime is the subject of these tests.
    void* user_data = nullptr;
    // Device-boundary readiness, protected by the inherited mutex.
    bool ready = false;
    // Submission observation for the outer test, protected by the mutex.
    bool started = false;
  } selection;

  static VariantDependency* From(void* self) {
    return static_cast<VariantDependency*>(static_cast<ModelService*>(self));
  }
  static void SetObserver(void* self, loom_serve_wake_callback_t observer) {
    ModelService::SetObserver(self, observer);
    loom_serve_variant_pool_set_observer(From(self)->variants, observer);
  }
  static bool IsPending(void* self) {
    auto* dependency = From(self);
    return ModelService::IsPending(self) || dependency->selection.entry ||
           loom_serve_variant_pool_pending(dependency->variants);
  }
  static iree_status_t Advance(void* self,
                               loom_serve_http_service_flags_t flags,
                               bool* out_progress) {
    IREE_RETURN_IF_ERROR(ModelService::Advance(self, flags, out_progress));
    auto* dependency = From(self);
    *out_progress |= loom_serve_variant_pool_collect(dependency->variants);
    std::lock_guard<std::mutex> lock(dependency->mutex);
    auto& selection = dependency->selection;
    if (selection.entry && (selection.ready || dependency->stopping)) {
      const bool cancel = dependency->stopping;
      if (cancel) {
        loom_serve_variant_entry_release(selection.entry);
      } else {
        loom_serve_variant_entry_release(dependency->selected);
        dependency->selected = selection.entry;
      }
      selection.entry = nullptr;
      selection.completion(
          selection.user_data,
          cancel ? iree_make_status(IREE_STATUS_CANCELLED, "model is stopping")
                 : iree_ok_status(),
          Selected(dependency));
      *out_progress = true;
    }
    return iree_ok_status();
  }
  static iree_status_t Stage(void* self, loom_serve_variant_id_t base,
                             const loom_serve_sources_t* sources,
                             loom_serve_variant_id_t* out_id) {
    auto* dependency = static_cast<VariantDependency*>(self);
    loom_serve_variant_entry_t* entry = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_serve_variant_pool_begin(dependency->variants, base, &entry));
    *out_id = loom_serve_variant_entry_id(entry);
    std::lock_guard<std::mutex> lock(dependency->mutex);
    if (dependency->complete_immediately) {
      loom_serve_variant_entry_complete(entry, iree_ok_status(), nullptr,
                                        nullptr);
    } else {
      dependency->compilers.push_back(entry);
    }
    ++dependency->started;
    dependency->condition.notify_all();
    return iree_ok_status();
  }
  static uint64_t Selected(void* self) {
    return loom_serve_variant_entry_id(
        static_cast<VariantDependency*>(self)->selected);
  }
  static iree_status_t Select(void* self, loom_serve_variant_id_t id,
                              loom_serve_variant_id_t expected,
                              loom_serve_variant_selection_fn_t completion,
                              void* user_data) {
    auto* dependency = static_cast<VariantDependency*>(self);
    if (expected != Selected(self) || dependency->selection.entry) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "selection conflict");
    }
    loom_serve_variant_entry_t* entry = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_serve_variant_pool_acquire(dependency->variants, id, &entry));
    std::lock_guard<std::mutex> lock(dependency->mutex);
    dependency->selection = {entry, completion, user_data, false, true};
    dependency->condition.notify_all();
    return iree_ok_status();
  }
  static iree_status_t Prepare(void* self, bool cancel, bool* out_complete,
                               loom_serve_http_service_t* out_service,
                               loom_serve_compilation_report_t* out_report) {
    auto* dependency = static_cast<VariantDependency*>(self);
    *out_complete = true;
    if (cancel) {
      return iree_make_status(IREE_STATUS_CANCELLED, "preparation cancelled");
    }
    *out_service = dependency->Interface(nullptr, nullptr);
    out_service->set_observer = SetObserver;
    out_service->is_pending = IsPending;
    out_service->advance = Advance;
    out_service->telemetry.stream = dependency->stream;
    out_service->reconfiguration = {dependency, dependency->variants, Stage,
                                    Select, Selected};
    return iree_ok_status();
  }
  static iree_status_t Retire(void* self, bool* out_complete) {
    auto* dependency = static_cast<VariantDependency*>(self);
    EXPECT_FALSE(IsPending(static_cast<ModelService*>(dependency)));
    loom_serve_variant_entry_release(dependency->selected);
    dependency->selected = nullptr;
    loom_serve_variant_pool_free(dependency->variants);
    dependency->variants = nullptr;
    *out_complete = true;
    return iree_ok_status();
  }
  static void Destroy(void* self) {
    ++static_cast<VariantDependency*>(self)->destroyed;
  }
  static iree_status_t Create(
      void* self, const loom_serve_model_configuration_t* configuration,
      loom_serve_wake_callback_t observer,
      loom_serve_telemetry_stream_t* stream, iree_allocator_t host_allocator,
      loom_serve_registration_t* out_registration) {
    auto* dependency = static_cast<VariantDependency*>(self);
    IREE_RETURN_IF_ERROR(loom_serve_variant_pool_allocate(
        8, nullptr, [](void* value) {}, host_allocator, &dependency->selected,
        &dependency->variants));
    loom_serve_variant_entry_retain(dependency->selected);
    dependency->original.store(Selected(self));
    dependency->name = configuration->name.data;
    dependency->counters = &dependency->totals;
    dependency->stream = stream;
    dependency->stopping = false;
    *out_registration = {
        .self = self,
        .prepare = Prepare,
        .retire = Retire,
        .destroy = Destroy,
    };
    return iree_ok_status();
  }
  void WaitCompilers(size_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return started >= count; });
  }
  void CompleteCompilers(bool remaining = false) {
    std::lock_guard<std::mutex> lock(mutex);
    complete_immediately |= remaining;
    for (auto* entry : compilers) {
      loom_serve_variant_entry_complete(entry, iree_ok_status(), nullptr,
                                        nullptr);
    }
    compilers.clear();
  }
  void WaitSelection() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return selection.started; });
  }
  void ReleaseSelection() {
    std::lock_guard<std::mutex> lock(mutex);
    selection.ready = true;
    observer.wake(observer.user_data);
  }
};

class HttpRouterVariantTest : public HttpRouterTest {
 protected:
  void SetUp() override {
    managed_capacity_ = 1;
    factory_ = {
        .self = &model_,
        .kinds = 1u << LOOM_SERVE_MODEL_KIND_EMBEDDING,
        .device_count = 1,
        .create = VariantDependency::Create,
    };
    HttpRouterTest::SetUp();
  }
  void TearDown() override {
    model_.CompleteCompilers(true);
    HttpRouterTest::TearDown();
  }
  std::string Register() {
    return Read(Send(
        "POST", "/v1/models/register",
        R"({"name":"dynamic","definition":{"kind":"embedding","prepare":"/model/prepare.loom","control":"/model/control.loom","sources":["/model/kernel.loom"],"assets":"/assets"}})"));
  }
  void Declare() {
    const auto response = Register();
    ASSERT_EQ(response.find("HTTP/1.1 202"), 0u) << response;
    while (!returned_.load() && !HasFatalFailure()) {
      const auto state = Read(Send("GET", "/v1/status"));
      ASSERT_EQ(state.find("HTTP/1.1 200"), 0u) << state;
      if (state.find("\"registration\":\"ready\"") != std::string::npos) {
        return;
      }
    }
    FAIL() << "Server stopped before registration readiness was observed";
  }
  int Stage(bool wait = true) {
    return Send("POST", "/v1/models/variants",
                "{\"model\":\"dynamic\",\"base\":" +
                    std::to_string(model_.original.load()) +
                    ",\"sources\":[\"/model/candidate.loom\"],\"wait\":" +
                    (wait ? "true}" : "false}"));
  }
  std::string Inventory(const std::string& headers = "") {
    return Read(Send("GET", "/v1/models/variants", "",
                     "X-Loom-Model: dynamic\r\n" + headers));
  }
  void WaitRemoval() {
    while (!returned_.load() && !HasFatalFailure()) {
      // Discovery withdraws on removal intent; status retains the registration
      // until every borrower retires and the registry releases its name.
      const auto state = Read(Send("GET", "/v1/status"));
      ASSERT_EQ(state.find("HTTP/1.1 200"), 0u) << state;
      if (state.find("\"id\":\"dynamic\"") == std::string::npos) {
        return;
      }
    }
    FAIL() << "Server stopped before registration retirement was observed";
  }
  // Dependencies outlive the joined serving owner, including teardown failure.
  VariantDependency model_;
};

TEST_P(HttpRouterVariantTest, UnregisterRetiresReadersBeforeCompilerOwnership) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const uint64_t original = model_.original.load();
  const int stage = Stage();
  model_.WaitCompilers(1);
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(stage).find("HTTP/1.1 410"), 0u);

  // The compiler still owns the registration even though inference discovery
  // and the waiting compilation reader have both withdrawn.
  const auto discovery = Read(Send("GET", "/v1/models"));
  ASSERT_EQ(discovery.find("HTTP/1.1 200"), 0u) << discovery;
  EXPECT_EQ(discovery.find("\"id\":\"dynamic\""), std::string::npos)
      << discovery;
  const auto state = Read(Send("GET", "/v1/status"));
  ASSERT_EQ(state.find("HTTP/1.1 200"), 0u) << state;
  EXPECT_NE(state.find("\"id\":\"dynamic\""), std::string::npos) << state;
  EXPECT_NE(state.find("\"admission\":\"removing\""), std::string::npos)
      << state;
  const auto duplicate = Register();
  EXPECT_EQ(duplicate.find("HTTP/1.1 409"), 0u) << duplicate;
  EXPECT_EQ(model_.destroyed.load(), 0u);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  model_.CompleteCompilers();
  ASSERT_NO_FATAL_FAILURE(WaitRemoval());
  EXPECT_EQ(model_.destroyed.load(), 1u);
  ASSERT_NO_FATAL_FAILURE(Declare());
  EXPECT_NE(model_.original.load(), original);
  EXPECT_EQ(Inventory("X-Loom-Variant: " + std::to_string(original) + "\r\n")
                .find("HTTP/1.1 404"),
            0u);
}

TEST_P(HttpRouterVariantTest, DisconnectedReaderDoesNotCancelConstruction) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const int stage = Stage();
  model_.WaitCompilers(1);
  EXPECT_EQ(shutdown(stage, SHUT_RDWR), 0);
  EXPECT_EQ(close(stage), 0);
  const auto pending = Inventory();
  EXPECT_NE(pending.find("\"count\":2"), std::string::npos) << pending;
  EXPECT_NE(pending.find("\"phase\":\"compiling\""), std::string::npos);
  model_.CompleteCompilers();
  while (!returned_.load() && !HasFatalFailure()) {
    const auto response = Inventory();
    ASSERT_EQ(response.find("HTTP/1.1 200"), 0u) << response;
    if (response.find("\"phase\":\"compiling\"") == std::string::npos) {
      EXPECT_NE(response.find("\"count\":2"), std::string::npos);
      break;
    }
  }
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  ASSERT_NO_FATAL_FAILURE(WaitRemoval());
}

TEST_P(HttpRouterVariantTest, DisconnectedSelectionKeepsCallbackUntilBoundary) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const std::string id = std::to_string(model_.original.load());
  const int selecting = Send(
      "POST", "/v1/models/variants/select",
      "{\"model\":\"dynamic\",\"variant\":" + id + ",\"expected\":" + id + "}");
  model_.WaitSelection();
  EXPECT_EQ(shutdown(selecting, SHUT_RDWR), 0);
  EXPECT_EQ(close(selecting), 0);
  EXPECT_EQ(
      Read(Send("POST", "/v1/test", R"({"model":"a"})")).find("HTTP/1.1 200"),
      0u);
  model_.ReleaseSelection();
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  ASSERT_NO_FATAL_FAILURE(WaitRemoval());
}

TEST_P(HttpRouterVariantTest, AbortCompletesAcceptedSelectionBeforeRetirement) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const std::string id = std::to_string(model_.original.load());
  const int selecting = Send(
      "POST", "/v1/models/variants/select",
      "{\"model\":\"dynamic\",\"variant\":" + id + ",\"expected\":" + id + "}");
  model_.WaitSelection();
  EXPECT_EQ(Read(Send("POST", "/v1/models/unregister",
                      R"({"model":"dynamic","mode":"abort"})"))
                .find("HTTP/1.1 202"),
            0u);
  EXPECT_EQ(Read(selecting).find("HTTP/1.1 409"), 0u);
  ASSERT_NO_FATAL_FAILURE(WaitRemoval());
  EXPECT_EQ(model_.destroyed.load(), 1u);
}

TEST_P(HttpRouterVariantTest, ShutdownJoinsCompilerAfterReaderLeaves) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const int stage = Stage();
  model_.WaitCompilers(1);
  loom_serve_http_server_request_stop(server_);
  model_.WaitStopping();
  EXPECT_TRUE(Read(stage).empty());
  EXPECT_FALSE(returned_.load());
  EXPECT_EQ(model_.destroyed.load(), 0u);
  model_.CompleteCompilers();
}

TEST_P(HttpRouterVariantTest, InvalidInputLeavesPoolAndServiceUnchanged) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const std::string prefix = "{\"model\":\"dynamic\",\"base\":" +
                             std::to_string(model_.original.load());
  for (const auto& suffix :
       {"}", ",\"sources\":[]}", ",\"sources\":[null]}", ",\"sources\":[\"\"]}",
        ",\"sources\":[\"bad\\u0000.loom\"]}",
        ",\"sources\":[\"/model.loom\"],\"wait\":1}",
        ",\"sources\":[\"/model.loom\"],\"base_directory\":\"relative\"}",
        ",\"sources\":[\"/model.loom\"],\"prepare\":\"/prepare.loom\"}",
        ",\"sources\":[\"/model.loom\"],\"model\":\"dynamic\"}",
        ",\"sources\":[\"/model.loom\"]} null"}) {
    SCOPED_TRACE(suffix);
    const auto response =
        Read(Send("POST", "/v1/models/variants", prefix + suffix));
    EXPECT_EQ(response.find("HTTP/1.1 400"), 0u) << response;
  }
  EXPECT_EQ(Read(Send("GET", "/v1/models/variants")).find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Inventory("X-Loom-Variant: 0\r\n").find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Inventory("X-Loom-Offset: -1\r\n").find("HTTP/1.1 400"), 0u);
  EXPECT_EQ(Read(Send("GET", "/v1/models/variants", "", "X-Loom-Model: a\r\n"))
                .find("HTTP/1.1 501"),
            0u);
  EXPECT_EQ(Read(Send("DELETE", "/v1/models/variants")).find("HTTP/1.1 405"),
            0u);
  EXPECT_NE(Inventory().find("\"count\":1"), std::string::npos);
  EXPECT_EQ(Read(Send("POST", "/v1/test", R"({"model":"dynamic"})"))
                .find("HTTP/1.1 200"),
            0u);
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterVariantTest,
                         ::testing::Values(1, 4));

class HttpRouterVariantCreditTest : public HttpRouterVariantTest {
 protected:
  void SetUp() override {
    connection_capacity_ = 3;
    HttpRouterVariantTest::SetUp();
  }
};

TEST_P(HttpRouterVariantCreditTest, CompilationAndHoldingLeaveControlCredit) {
  ASSERT_NO_FATAL_FAILURE(Declare());
  const int stage = Stage();
  model_.WaitCompilers(1);
  EXPECT_EQ(Control("drain", "a").find("HTTP/1.1 200"), 0u);
  const int held = Send("POST", "/v1/test", R"({"model":"a"})");
  WaitHeld("a", 1);
  EXPECT_EQ(Read(Stage()).find("HTTP/1.1 429"), 0u);
  EXPECT_EQ(Read(Send("GET", "/logz/stream", "", "X-Loom-Model: dynamic\r\n"))
                .find("HTTP/1.1 429"),
            0u);
  const auto listing = Inventory();
  EXPECT_EQ(listing.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(listing.find("\"count\":2"), std::string::npos) << listing;
  EXPECT_EQ(Control("unregister", "dynamic").find("HTTP/1.1 202"), 0u);
  EXPECT_EQ(Read(stage).find("HTTP/1.1 410"), 0u);
  EXPECT_EQ(Control("undrain", "a").find("HTTP/1.1 200"), 0u);
  EXPECT_EQ(Read(held).find("HTTP/1.1 200"), 0u);
  model_.CompleteCompilers();
  ASSERT_NO_FATAL_FAILURE(WaitRemoval());
}

INSTANTIATE_TEST_SUITE_P(Workers, HttpRouterVariantCreditTest,
                         ::testing::Values(1, 4));

}  // namespace
