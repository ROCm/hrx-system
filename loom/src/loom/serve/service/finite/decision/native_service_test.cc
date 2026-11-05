// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/internal/json.h"
#include "iree/base/threading/notification.h"
#include "iree/task/owner.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/server/driver.h"
#include "loom/serve/service/finite/decision/protocol.h"
#include "loom/serve/service/finite/registration.h"
#include "loom/serve/service/finite/service.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/service_application.h"

namespace {

constexpr char kRequest[] =
    R"({"model":"decision","state":"a","questions":{"truth":{"type":"noul","instructions":"???","criteria":{"false":"","true":"x"}}}})";

class NativeDecisionServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    const auto options = loom_serve_test_application_options_default();
    loom_serve_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_serve_test_application_acquire(
        &options, iree_allocator_system(), &application));
    group_ = loom_serve_test_application_device_group(application);
    device_ = loom_serve_test_application_device_at(application, 0);
    pool_ = loom_serve_device_group_pool_at(group_, 0);
    const loom_runtime_wake_callback_t observer = {Wake, this};
    loom_serve_device_group_set_observer(group_, observer);
    const char* source = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source, nullptr);
    const std::string file = source;
    IREE_ASSERT_OK(
        document_.Load(file.substr(0, file.find_last_of('/')) + "/model.json"));
    const loom_serve_finite_model_options_t model_options = {
        .sources = document_.value().sources,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}"),
    };
    IREE_ASSERT_OK(AllocateRegistration(model_options));
    IREE_ASSERT_OK(Prepare(0));
    const auto finite = loom_serve_finite_service_interface(service_);
    interface_ = {
        .name = finite.name,
        .runtime = finite.runtime,
        .kind = finite.kind,
        .snapshot = finite.snapshot,
        .inspection = finite.inspection,
        .reconfiguration = finite.reconfiguration,
        .telemetry =
            {
                .stream = finite.telemetry.stream,
                .counters = finite.telemetry.counters,
                .latency = finite.telemetry.latency,
            },
    };
    loom_serve_registry_t* registry = nullptr;
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        1, &interface_, 0, {}, iree_allocator_system(), &registry));
    loom_serve_driver_initialize(registry, &driver_);
    loom_serve_driver_set_observer(&driver_, observer);
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
    }
    for (auto*& call : calls_) {
      loom_serve_finite_call_close(std::exchange(call, nullptr));
    }
    iree_hal_buffer_release(std::exchange(target_, nullptr));
    if (registration_) {
      RetireService();
    }
    if (consumer_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(consumer_));
    }
    loom_serve_execution_release(consumer_);
    iree_hal_semaphore_release(gate_);
    loom_serve_device_group_set_observer(group_, {});
    iree_notification_deinitialize(&notification_);
    iree_status_free(driver_.status);
  }

  static void Wake(void* self) {
    iree_notification_post(
        &static_cast<NativeDecisionServiceTest*>(self)->notification_,
        IREE_ALL_WAITERS);
  }

  iree_status_t AllocateRegistration(
      const loom_serve_finite_model_options_t& model_options) {
    preparation_complete_ = false;
    const loom_serve_finite_registration_options_t service_options = {
        .protocol = loom_serve_decision_protocol(),
        .name = IREE_SV("decision"),
        .capacity = 3,
    };
    return loom_serve_finite_registration_allocate(
        device_, pool_, &model_options, &service_options, {Wake, this},
        iree_allocator_system(), &registration_);
  }

  iree_status_t Prepare(loom_serve_finite_load_flags_t flags) {
    iree_status_t status = iree_ok_status();
    loom_serve_compilation_report_t report = {};
    while (!preparation_complete_) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      status =
          iree_status_join(status, loom_serve_device_group_advance(group_));
      status = iree_status_join(
          status, loom_serve_finite_registration_prepare(
                      registration_,
                      flags | (iree_status_is_ok(status)
                                   ? 0
                                   : LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL),
                      &preparation_complete_, &service_, &report));
      if (preparation_complete_) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    return status;
  }

  template <typename Predicate>
  void DriveUntil(Predicate done, loom_runtime_service_flags_t flags = 0) {
    driver_.stopping |= (flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING) != 0;
    while (true) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      driver_.status = iree_status_join(
          driver_.status, loom_serve_device_group_advance(group_));
      bool progress = false;
      loom_serve_driver_advance(&driver_, {Wake, this}, {}, &progress);
      loom_serve_driver_collect(&driver_, &progress);
      const bool complete = done();
      if (complete || progress || !iree_status_is_ok(driver_.status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(iree_status_clone(driver_.status));
      if (complete) {
        break;
      }
    }
  }

  void RetireService() {
    if (!preparation_complete_) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                            Prepare(LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL));
    }
    if (driver_.registry) {
      ASSERT_NO_FATAL_FAILURE(DriveUntil(
          [&] {
            return !interface_.runtime.vtable->is_pending(
                interface_.runtime.self);
          },
          LOOM_RUNTIME_SERVICE_FLAG_STOPPING));
      loom_serve_driver_set_observer(&driver_, {});
      IREE_ASSERT_OK(
          loom_serve_registry_free(std::exchange(driver_.registry, nullptr)));
    }
    bool complete = false;
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_serve_device_group_advance(group_);
      status = iree_status_join(status, loom_serve_finite_registration_retire(
                                            registration_, &complete));
      if (complete || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
    service_ = nullptr;
    loom_serve_finite_registration_free(std::exchange(registration_, nullptr));
  }

  // Real asynchronous composition: application threads enter for admission and
  // observation; only the shared executor advances resources and services.
  class Owner {
   public:
    explicit Owner(NativeDecisionServiceTest& fixture) : fixture_(fixture) {
      iree_task_owner_initialize(
          loom_serve_host_executor(loom_serve_device_host(fixture.device_)),
          Advance, this, &owner_);
      Call([&] {
        const loom_runtime_wake_callback_t observer = {Wake, this};
        loom_serve_driver_set_observer(&fixture_.driver_, observer);
        loom_serve_device_group_set_observer(fixture_.group_, observer);
      });
    }

    ~Owner() {
      Call([&] {
        // Device results may outlive the model, but their pool credit must be
        // returned before joining the whole resource owner's retirement.
        for (auto*& call : fixture_.calls_) {
          loom_serve_finite_call_close(std::exchange(call, nullptr));
        }
        fixture_.driver_.stopping = true;
      });
      iree_task_owner_await(&owner_);
      iree_task_owner_deinitialize(&owner_);
      const loom_runtime_wake_callback_t observer = {
          NativeDecisionServiceTest::Wake, &fixture_};
      loom_serve_driver_set_observer(&fixture_.driver_, observer);
      loom_serve_device_group_set_observer(fixture_.group_, observer);
    }

    template <typename Fn>
    void Call(Fn&& call) {
      using Callback = std::remove_reference_t<Fn>;
      iree_task_owner_call(
          &owner_, [](void* self) { (*static_cast<Callback*>(self))(); },
          &call);
    }

    template <typename Predicate>
    void WaitUntil(Predicate done) {
      bool complete = false;
      while (!complete) {
        const auto ticket =
            iree_notification_prepare_wait(&fixture_.notification_);
        Call([&] { complete = done(); });
        if (complete) {
          iree_notification_cancel_wait(&fixture_.notification_);
        } else {
          iree_notification_commit_wait(&fixture_.notification_, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
      }
    }

   private:
    static void Wake(void* self) {
      iree_task_owner_wake(&static_cast<Owner*>(self)->owner_);
    }

    static void Advance(void* self, const iree_task_worker_context_t*,
                        iree_task_process_drain_result_t* result) {
      auto& owner = *static_cast<Owner*>(self);
      auto& fixture = owner.fixture_;
      fixture.driver_.status =
          iree_status_join(fixture.driver_.status,
                           loom_serve_device_group_advance(fixture.group_));
      loom_serve_driver_advance(&fixture.driver_, {Wake, self}, {},
                                &result->did_work);
      bool pending =
          loom_serve_driver_collect(&fixture.driver_, &result->did_work);
      pending |= loom_serve_device_group_is_pending(fixture.group_);
      if (fixture.driver_.stopping && !pending) {
        loom_serve_driver_set_observer(&fixture.driver_, {});
        loom_serve_device_group_set_observer(fixture.group_, {});
        result->completed = true;
      }
    }

    // Resource/model fixture, alive until this executor owner has retired.
    NativeDecisionServiceTest& fixture_;
    // Same enterable compute process used by the HTTP router.
    iree_task_owner_t owner_;
  };

  void Open(size_t index, const char* text = kRequest,
            loom_serve_finite_output_format_t output_format =
                LOOM_SERVE_FINITE_OUTPUT_DEVICE) {
    iree_string_builder_t body;
    iree_string_builder_initialize(iree_allocator_system(), &body);
    auto status = iree_string_builder_append_cstring(&body, text);
    if (iree_status_is_ok(status)) {
      status = loom_serve_finite_service_try_open(
          service_, &body, output_format, &calls_[index]);
    }
    if (iree_status_is_ok(status) && calls_[index]) {
      EXPECT_EQ(iree_string_builder_size(&body), 0u);
    }
    iree_string_builder_deinitialize(&body);
    IREE_ASSERT_OK(status);
    ASSERT_NE(calls_[index], nullptr);
  }

  // Shared application resource owners, alive through every escaped result.
  loom_serve_device_group_t* group_ = nullptr;
  // Execution placement of this real source service.
  loom_serve_device_t* device_ = nullptr;
  // Shared physical admission independent of execution placement.
  loom_serve_pool_t* pool_ = nullptr;
  // Owns source views until every preparation/cancellation attempt retires.
  loom_serve::testing::ModelDocument document_;
  // Shared production preparation/service/retirement owner.
  loom_serve_finite_registration_t* registration_ = nullptr;
  // Preparation loans have retired, including on terminal failure.
  bool preparation_complete_ = false;
  // Borrowed real admission, batching, and CPU/device progression.
  loom_serve_finite_service_t* service_ = nullptr;
  // Owner-side operations, with no listener or HTTP notification.
  loom_serve_service_t interface_ = {};
  // Shared production traversal and retirement policy, also used by HTTP.
  loom_serve_driver_t driver_ = {};
  // Joined wake target for resource and service publications.
  iree_notification_t notification_;
  // Caller-owned receipts, including after service removal.
  std::array<loom_serve_finite_call_t*, 4> calls_ = {};
  // Independent downstream work, never submitted through the source model.
  loom_serve_execution_t* consumer_ = nullptr;
  // Explicit consumer dependency proving retirement does not wait for peers.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Downstream logical output, released after its queue captures it.
  iree_hal_buffer_t* target_ = nullptr;
  // Host destination alive through the downstream feedback operation.
  std::array<float, 2> values_ = {};
};

TEST_F(NativeDecisionServiceTest, PreparationFailurePublishesNoService) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  loom_serve_finite_model_options_t options = {
      .sources = document_.value().sources,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  options.sources.prepare = IREE_SV("missing-prepare.loom");
  IREE_ASSERT_OK(AllocateRegistration(options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare(0));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, UnstartedCancellationDoesNotReadSource) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_serve_finite_model_options_t options = {
      .sources = {.prepare = IREE_SV("missing-prepare.loom")},
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, PreparationCancellationJoinsAcceptedLoad) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_serve_finite_model_options_t options = {
      .sources = document_.value().sources,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  loom_serve_compilation_report_t report = {};
  IREE_ASSERT_OK(loom_serve_finite_registration_prepare(
      registration_, 0, &preparation_complete_, &service_, &report));
  EXPECT_FALSE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(LOOM_SERVE_FINITE_LOAD_FLAG_CANCEL));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, DeviceOutputEscapesBeforeServiceRetirement) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_serve_finite_call_output(calls_[0]) != nullptr ||
           loom_serve_finite_call_is_complete(calls_[0]);
  }));
  if (loom_serve_finite_call_is_complete(calls_[0])) {
    IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[0]));
  }
  EXPECT_FALSE(loom_serve_finite_call_is_complete(calls_[0]));
  EXPECT_TRUE(interface_.runtime.vtable->has_requests(interface_.runtime.self));
  const auto* first = loom_serve_finite_call_output(calls_[0]);
  ASSERT_NE(first, nullptr);
  const auto* second = loom_serve_finite_call_output(calls_[1]);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->result, second->result);
  EXPECT_EQ(first->request_index, 0u);
  EXPECT_EQ(second->request_index, 1u);
  EXPECT_EQ(second->row_count, 1u);
  const auto ready = loom_serve_finite_result_ready(second->result);
  EXPECT_EQ(ready.values[1], 0u);

  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_, pool_, &consumer_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  iree_hal_semaphore_t* waits[] = {ready.semaphores[0], ready.semaphores[1],
                                   gate_};
  uint64_t wait_values[] = {ready.values[0], ready.values[1], 1};
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(consumer_, {3, waits, wait_values}));
  const auto allocation =
      iree_hal_make_pool_reservation_request(sizeof(values_), 16);
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(consumer_, 1, &allocation, &target_));
  const auto row =
      loom_serve_finite_result_row(second->result, second->request_index, 0);
  ASSERT_EQ(row.length, sizeof(values_));
  const iree_hal_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = row.buffer,
              .source_offset = row.offset,
              .target_buffer = target_,
              .length = row.length,
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(consumer_, 1, &copy, &value));
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = target_,
              .target = values_.data(),
              .length = sizeof(values_),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(consumer_, 1, &download, &value));
  iree_hal_buffer_release(std::exchange(target_, nullptr));
  for (auto*& call : calls_) {
    loom_serve_finite_call_close(std::exchange(call, nullptr));
  }
  ASSERT_NO_FATAL_FAILURE(RetireService());

  // Accepted downstream captures, not the source model or caller receipts,
  // now own the result. Releasing the gate must still yield the packed output.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  iree_hal_semaphore_release(std::exchange(gate_, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_));
  EXPECT_FLOAT_EQ(values_[0], 0.75f);
  EXPECT_FLOAT_EQ(values_[1], 0.25f);
}

TEST_F(NativeDecisionServiceTest, PressurePreservesInputAndRejectionIsLocal) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1, "{}", LOOM_SERVE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(2));
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(&body, kRequest));
  const auto original = iree_string_builder_view(&body);
  for (auto format :
       {LOOM_SERVE_FINITE_OUTPUT_DEVICE, LOOM_SERVE_FINITE_OUTPUT_JSON}) {
    IREE_EXPECT_OK(loom_serve_finite_service_try_open(service_, &body, format,
                                                      &calls_[3]));
    EXPECT_EQ(calls_[3], nullptr);
    EXPECT_EQ(iree_string_builder_view(&body).data, original.data);
    EXPECT_EQ(iree_string_builder_size(&body), original.size);
  }
  iree_string_builder_deinitialize(&body);
  loom_serve_finite_call_close(std::exchange(calls_[2], nullptr));

  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_serve_finite_call_is_complete(calls_[0]) &&
           loom_serve_finite_call_is_complete(calls_[1]);
  }));
  IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_finite_call_status(calls_[1]));
  EXPECT_EQ(loom_serve_finite_call_output(calls_[1]), nullptr);
  ASSERT_NO_FATAL_FAILURE(Open(3));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_serve_finite_call_is_complete(calls_[3]); }));
  IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[3]));
  const auto* output = loom_serve_finite_call_output(calls_[3]);
  ASSERT_NE(output, nullptr);
  // No earlier native cohort appended a host feedback operation.
  EXPECT_EQ(loom_serve_finite_result_ready(output->result).values[1], 0u);
  uint32_t notifications = 0;
  loom_serve_finite_call_set_observer(
      calls_[3],
      {[](void* self) { ++*static_cast<uint32_t*>(self); }, &notifications});
  EXPECT_EQ(notifications, 1u);
  loom_serve_finite_call_set_observer(calls_[3], {});
}

TEST_F(NativeDecisionServiceTest, StopDuringPreparationJoinsInputBorrowers) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return std::string(interface_.snapshot(interface_.runtime.self).phase) ==
           "preparing";
  }));
  ASSERT_NO_FATAL_FAILURE(
      DriveUntil([&] { return loom_serve_finite_call_is_complete(calls_[0]); },
                 LOOM_RUNTIME_SERVICE_FLAG_STOPPING));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_finite_call_status(calls_[0]));
  EXPECT_EQ(loom_serve_finite_call_output(calls_[0]), nullptr);
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(&body, kRequest));
  const auto original = iree_string_builder_view(&body);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_finite_service_try_open(
          service_, &body, LOOM_SERVE_FINITE_OUTPUT_DEVICE, &calls_[1]));
  EXPECT_EQ(calls_[1], nullptr);
  EXPECT_EQ(iree_string_builder_view(&body).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body), original.size);
  iree_string_builder_deinitialize(&body);
}

TEST_F(NativeDecisionServiceTest, NativeCallerUsesExecutorWithoutHostPolling) {
  Owner owner(*this);
  ASSERT_NO_FATAL_FAILURE(owner.Call([&] {
    EXPECT_FALSE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    ASSERT_NO_FATAL_FAILURE(Open(0));
    ASSERT_NO_FATAL_FAILURE(Open(1));
    EXPECT_TRUE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    for (size_t i = 0; i < 2; ++i) {
      loom_serve_finite_call_set_observer(calls_[i], {Wake, this});
    }
  }));
  owner.WaitUntil([&] {
    return loom_serve_finite_call_is_complete(calls_[0]) &&
           loom_serve_finite_call_is_complete(calls_[1]);
  });
  ASSERT_NO_FATAL_FAILURE(owner.Call([&] {
    IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[0]));
    IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[1]));
    EXPECT_FALSE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    const auto* first = loom_serve_finite_call_output(calls_[0]);
    const auto* second = loom_serve_finite_call_output(calls_[1]);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->result, second->result);
    EXPECT_EQ(first->request_index, 0u);
    EXPECT_EQ(second->request_index, 1u);
    EXPECT_EQ(loom_serve_finite_result_ready(first->result).values[1], 0u);
    ASSERT_NO_FATAL_FAILURE(Open(2));
    loom_serve_finite_call_set_observer(calls_[2], {Wake, this});
  }));
  owner.WaitUntil(
      [&] { return loom_serve_finite_call_is_complete(calls_[2]); });
  owner.Call([&] { IREE_EXPECT_OK(loom_serve_finite_call_status(calls_[2])); });
}

TEST_F(NativeDecisionServiceTest, JsonResponseOutlivesModelAndResourceOwner) {
  std::unique_ptr<loom_serve_finite_call_t,
                  decltype(&loom_serve_finite_call_close)>
      retained(nullptr, loom_serve_finite_call_close);
  {
    Owner owner(*this);
    ASSERT_NO_FATAL_FAILURE(owner.Call([&] {
      for (size_t i = 0; i < 2; ++i) {
        ASSERT_NO_FATAL_FAILURE(
            Open(i, kRequest, LOOM_SERVE_FINITE_OUTPUT_JSON));
        loom_serve_finite_call_set_observer(calls_[i], {Wake, this});
      }
    }));
    owner.WaitUntil([&] {
      return loom_serve_finite_call_is_complete(calls_[0]) &&
             loom_serve_finite_call_is_complete(calls_[1]);
    });
    owner.Call([&] {
      IREE_EXPECT_OK(loom_serve_finite_call_status(calls_[1]));
      EXPECT_EQ(loom_serve_finite_call_output(calls_[1]), nullptr);
      loom_serve_finite_call_set_observer(calls_[1], {});
      retained.reset(std::exchange(calls_[1], nullptr));
    });
  }
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const auto response = loom_serve_finite_call_response(retained.get());
  EXPECT_TRUE(iree_string_view_equal(
      response,
      IREE_SV(
          "{\"model\":\"decision\",\"answers\":{\"truth\":{\"type\":\"noul\","
          "\"noul\":0.25}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}"
          "}")));
}

TEST_F(NativeDecisionServiceTest, MixedDeviceAndJsonCallsShareOneDownload) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1, kRequest, LOOM_SERVE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(2, kRequest, LOOM_SERVE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_serve_finite_call_is_complete(calls_[0]) &&
           loom_serve_finite_call_is_complete(calls_[1]) &&
           loom_serve_finite_call_is_complete(calls_[2]);
  }));
  IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[0]));
  const auto* device_output = loom_serve_finite_call_output(calls_[0]);
  ASSERT_NE(device_output, nullptr);
  EXPECT_EQ(device_output->request_index, 0u);
  EXPECT_EQ(loom_serve_finite_result_ready(device_output->result).values[1],
            0u);
  for (size_t i = 1; i < 3; ++i) {
    IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[i]));
    EXPECT_EQ(loom_serve_finite_call_output(calls_[i]), nullptr);
    auto value = loom_serve_finite_call_response(calls_[i]);
    for (const char* key : {"answers", "truth", "noul"}) {
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          value, iree_make_cstring_view(key), &value));
    }
    double probability = 0;
    IREE_ASSERT_OK(iree_json_parse_double(value, &probability));
    // Three requests share one cohort; separate launches would produce 2/3.
    // Vulkan F32 division allows 2.5 ULP at this normal divisor.
    EXPECT_NEAR(probability, 1.0 / 3.0, 2.5 * 0x1p-25);
  }
  ASSERT_NO_FATAL_FAILURE(Open(3));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_serve_finite_call_is_complete(calls_[3]); }));
  IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[3]));
  const auto* next_output = loom_serve_finite_call_output(calls_[3]);
  ASSERT_NE(next_output, nullptr);
  // Both JSON requests used one feedback operation, and the following device
  // call adds none. This queries the next cohort's captured producer frontier.
  EXPECT_EQ(loom_serve_finite_result_ready(next_output->result).values[1], 1u);
}

TEST_F(NativeDecisionServiceTest, JsonCancellationJoinsEncodingBorrowers) {
  ASSERT_NO_FATAL_FAILURE(Open(0, kRequest, LOOM_SERVE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(1, kRequest, LOOM_SERVE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return std::string(interface_.snapshot(interface_.runtime.self).phase) ==
           "encoding";
  }));
  loom_serve_finite_call_close(std::exchange(calls_[0], nullptr));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_serve_finite_call_is_complete(calls_[1]); }));
  IREE_ASSERT_OK(loom_serve_finite_call_status(calls_[1]));
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const auto response = loom_serve_finite_call_response(calls_[1]);
  EXPECT_TRUE(iree_string_view_equal(
      response,
      IREE_SV(
          "{\"model\":\"decision\",\"answers\":{\"truth\":{\"type\":\"noul\","
          "\"noul\":0.25}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}"
          "}")));
}

}  // namespace
