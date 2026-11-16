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
#include <vector>

#include "iree/base/internal/json.h"
#include "iree/base/threading/notification.h"
#include "iree/task/owner.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/driver.h"
#include "loom/runtime/program/sources.h"
#include "loom/runtime/program/variant_pool.h"
#include "loom/runtime/service_control.h"
#include "loom/runtime/testing/application.h"
#include "loom/runtime/testing/worker_gate.h"
#include "loom/service/finite/decision/protocol.h"
#include "loom/service/finite/registration.h"
#include "loom/service/finite/service.h"
#include "loom/service/registration.h"

namespace {

constexpr char kRequest[] =
    R"({"model":"decision","state":"a","questions":{"truth":{"type":"noul","instructions":"???","criteria":{"false":"","true":"x"}}}})";

class NativeDecisionServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    const auto options = loom_runtime_test_application_options_default();
    loom_runtime_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_runtime_test_application_acquire(
        &options, iree_allocator_system(), &application));
    group_ = loom_runtime_test_application_device_group(application);
    device_ = loom_runtime_test_application_device_at(application, 0);
    pool_ = loom_runtime_device_group_pool_at(group_, 0);
    const loom_runtime_wake_callback_t observer = {Wake, this};
    loom_runtime_device_group_set_observer(group_, observer);
    const char* source = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source, nullptr);
    const std::string file = source;
    const std::string directory = file.substr(0, file.find_last_of('/'));
    const loom_runtime_sources_t unresolved = {
        .base_directory =
            iree_make_string_view(directory.data(), directory.size()),
        .reference =
            {
                .path = IREE_SV("card.json"),
                .selector = {.preset = IREE_SV("check")},
            },
    };
    IREE_ASSERT_OK(loom_runtime_sources_resolve(
        &unresolved, iree_allocator_system(), &sources_));
    const loom_service_finite_model_options_t model_options = {
        .sources = *sources_,
        .asset_directory = IREE_SV("."),
        .options = IREE_SV("{\"batch_units\":64}"),
    };
    IREE_ASSERT_OK(AllocateRegistration(model_options));
    IREE_ASSERT_OK(Prepare(0));
    interface_ = loom_service_finite_service_interface(service_);
    loom_runtime_driver_initialize(1, &interface_.runtime, &driver_);
    loom_runtime_driver_set_observer(&driver_, observer);
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
    }
    for (auto*& call : calls_) {
      loom_service_finite_call_close(std::exchange(call, nullptr));
    }
    iree_hal_buffer_release(std::exchange(target_, nullptr));
    if (registration_) {
      RetireService();
    }
    if (consumer_) {
      IREE_EXPECT_OK(loom_runtime_execution_drain(consumer_));
    }
    loom_runtime_execution_release(consumer_);
    iree_hal_semaphore_release(gate_);
    loom_runtime_device_group_set_observer(group_, {});
    iree_notification_deinitialize(&notification_);
    iree_status_free(driver_.status);
    loom_runtime_sources_free(sources_, iree_allocator_system());
  }

  static void Wake(void* self) {
    iree_notification_post(
        &static_cast<NativeDecisionServiceTest*>(self)->notification_,
        IREE_ALL_WAITERS);
  }

  iree_status_t AllocateRegistration(
      const loom_service_finite_model_options_t& model_options,
      iree_host_size_t variant_capacity = 0) {
    preparation_complete_ = false;
    const loom_service_finite_registration_options_t service_options = {
        .incarnation = ++incarnation_,
        .protocol = loom_service_decision_protocol(),
        .name = IREE_SV("decision"),
        .capacity = 3,
        .variant_capacity = variant_capacity,
    };
    return loom_service_finite_registration_allocate(
        device_, pool_, &model_options, &service_options, {Wake, this},
        iree_allocator_system(), &registration_);
  }

  // Each native lifecycle occupying this fixture's slot has a distinct key.
  uint64_t incarnation_ = 0;

  iree_status_t Prepare(loom_service_registration_prepare_flags_t flags) {
    iree_status_t status = iree_ok_status();
    while (!preparation_complete_) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      status =
          iree_status_join(status, loom_runtime_device_group_advance(group_));
      status = iree_status_join(
          status,
          loom_service_registration_prepare(
              registration_,
              flags | (iree_status_is_ok(status)
                           ? 0
                           : LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL),
              &preparation_complete_));
      if (preparation_complete_) {
        service_ = loom_service_finite_registration_service(registration_);
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
          driver_.status, loom_runtime_device_group_advance(group_));
      bool progress = false;
      loom_runtime_driver_advance(&driver_, &progress);
      loom_runtime_driver_collect(&driver_, &progress);
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

  // Embedded owner progression with no permission to begin another unit.
  template <typename Predicate>
  iree_status_t ProgressUntil(
      Predicate done, loom_runtime_service_flags_t flags = 0,
      iree_status_code_t expected_status = IREE_STATUS_OK) {
    while (true) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_runtime_device_group_advance(group_);
      const bool progress =
          loom_service_registration_progress(registration_, flags);
      const auto snapshot = loom_service_registration_snapshot(registration_);
      if (!preparation_complete_ &&
          snapshot.phase != LOOM_SERVICE_REGISTRATION_DECLARED &&
          snapshot.phase != LOOM_SERVICE_REGISTRATION_PREPARING) {
        preparation_complete_ = true;
        if (snapshot.service.self) {
          service_ = loom_service_finite_registration_service(registration_);
          interface_ = loom_service_finite_service_interface(service_);
          loom_runtime_driver_initialize(1, &interface_.runtime, &driver_);
        }
      }
      if (snapshot.phase == LOOM_SERVICE_REGISTRATION_RETIRED) {
        service_ = nullptr;
      }
      const bool complete = done();
      const bool unexpected_failure =
          !iree_status_is_ok(snapshot.status) &&
          iree_status_code(snapshot.status) != expected_status;
      if (complete || unexpected_failure) {
        status = iree_status_join(status, iree_status_clone(snapshot.status));
      }
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

  loom_runtime_service_offer_t Offer() {
    loom_runtime_service_offer_t offer = {};
    EXPECT_EQ(loom_runtime_service_read_offer(
                  loom_service_registration_control(registration_), 0, &offer),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    return offer;
  }

  loom_runtime_service_offer_t OfferedWork(
      loom_runtime_service_work_flags_t kind) {
    const auto* control = loom_service_registration_control(registration_);
    const auto layout = loom_runtime_service_control_layout(control);
    for (uint64_t i = 0; i < layout.offer_capacity; ++i) {
      loom_runtime_service_offer_t offer = {};
      EXPECT_EQ(loom_runtime_service_read_offer(control, i, &offer),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      if (offer.choice_count && offer.work == kind) {
        return offer;
      }
    }
    return {};
  }

  loom_runtime_service_operation_key_t Schedule(
      loom_runtime_service_offer_key_t offer) {
    const auto outcome =
        loom_service_registration_try_schedule(registration_, offer, 0, 0);
    EXPECT_EQ(outcome.status_code, IREE_STATUS_OK);
    EXPECT_EQ(outcome.accepted, outcome.operation.serial != 0);
    return outcome.operation;
  }

  std::vector<loom_runtime_service_operation_t> AcceptedOperations() {
    const auto* control = loom_service_registration_control(registration_);
    const auto layout = loom_runtime_service_control_layout(control);
    std::vector<loom_runtime_service_operation_t> operations;
    for (uint64_t i = 0; i < layout.operation_capacity; ++i) {
      loom_runtime_service_operation_t operation = {};
      EXPECT_EQ(loom_runtime_service_read_operation(control, i, &operation),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      if (operation.key.serial) {
        operations.push_back(operation);
      }
    }
    return operations;
  }

  void RetireModel() {
    if (!preparation_complete_) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_CANCELLED,
          Prepare(LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL));
    }
    if (driver_.capacity) {
      ASSERT_NO_FATAL_FAILURE(DriveUntil(
          [&] {
            return !interface_.runtime.vtable->is_pending(
                interface_.runtime.self);
          },
          LOOM_RUNTIME_SERVICE_FLAG_STOPPING));
      loom_runtime_driver_set_observer(&driver_, {});
      driver_.services = nullptr;
      driver_.capacity = 0;
    }
    bool complete = false;
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_runtime_device_group_advance(group_);
      status = iree_status_join(
          status, loom_service_registration_retire(registration_, &complete));
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
  }

  void RetireService() {
    ASSERT_NO_FATAL_FAILURE(RetireModel());
    loom_service_registration_free(std::exchange(registration_, nullptr));
  }

  // Real asynchronous composition: application threads enter for admission and
  // observation; only the shared executor advances resources and services.
  class Owner {
   public:
    explicit Owner(NativeDecisionServiceTest& fixture) : fixture_(fixture) {
      iree_task_owner_initialize(
          loom_runtime_host_executor(loom_runtime_device_host(fixture.device_)),
          Advance, this, &owner_);
      Call([&] {
        const loom_runtime_wake_callback_t observer = {Wake, this};
        loom_runtime_driver_set_observer(&fixture_.driver_, observer);
        loom_runtime_device_group_set_observer(fixture_.group_, observer);
      });
    }

    ~Owner() {
      Call([&] {
        // Device results may outlive the model, but their pool credit must be
        // returned before joining the whole resource owner's retirement.
        for (auto*& call : fixture_.calls_) {
          loom_service_finite_call_close(std::exchange(call, nullptr));
        }
        fixture_.driver_.stopping = true;
      });
      iree_task_owner_await(&owner_);
      iree_task_owner_deinitialize(&owner_);
      const loom_runtime_wake_callback_t observer = {
          NativeDecisionServiceTest::Wake, &fixture_};
      loom_runtime_driver_set_observer(&fixture_.driver_, observer);
      loom_runtime_device_group_set_observer(fixture_.group_, observer);
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
                           loom_runtime_device_group_advance(fixture.group_));
      loom_runtime_driver_advance(&fixture.driver_, &result->did_work);
      bool pending =
          loom_runtime_driver_collect(&fixture.driver_, &result->did_work);
      pending |= loom_runtime_device_group_is_pending(fixture.group_);
      if (fixture.driver_.stopping && !pending) {
        loom_runtime_driver_set_observer(&fixture.driver_, {});
        loom_runtime_device_group_set_observer(fixture.group_, {});
        result->completed = true;
      }
    }

    // Resource/model fixture, alive until this executor owner has retired.
    NativeDecisionServiceTest& fixture_;
    // Same enterable compute process used by the HTTP router.
    iree_task_owner_t owner_;
  };

  void Open(size_t index, const char* text = kRequest,
            loom_service_finite_output_format_t output_format =
                LOOM_SERVICE_FINITE_OUTPUT_DEVICE) {
    iree_string_builder_t body;
    iree_string_builder_initialize(iree_allocator_system(), &body);
    auto status = iree_string_builder_append_cstring(&body, text);
    if (iree_status_is_ok(status)) {
      status = loom_service_finite_service_try_open(
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
  loom_runtime_device_group_t* group_ = nullptr;
  // Execution placement of this real source service.
  loom_runtime_device_t* device_ = nullptr;
  // Shared physical admission independent of execution placement.
  loom_runtime_pool_t* pool_ = nullptr;
  // Owns source views until every preparation/cancellation attempt retires.
  loom_runtime_sources_t* sources_ = nullptr;
  // Shared production preparation/service/retirement owner.
  loom_service_registration_t* registration_ = nullptr;
  // Preparation loans have retired, including on terminal failure.
  bool preparation_complete_ = false;
  // Borrowed real admission, batching, and CPU/device progression.
  loom_service_finite_service_t* service_ = nullptr;
  // Owner-side operations, with no listener or HTTP notification.
  loom_service_finite_service_interface_t interface_ = {};
  // Shared production traversal and retirement policy, also used by HTTP.
  loom_runtime_driver_t driver_ = {};
  // Joined wake target for resource and service publications.
  iree_notification_t notification_;
  // Caller-owned receipts, including after service removal.
  std::array<loom_service_finite_call_t*, 4> calls_ = {};
  // Independent downstream work, never submitted through the source model.
  loom_runtime_execution_t* consumer_ = nullptr;
  // Explicit consumer dependency proving retirement does not wait for peers.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Downstream logical output, released after its queue captures it.
  iree_hal_buffer_t* target_ = nullptr;
  // Host destination alive through the downstream feedback operation.
  std::array<float, 2> values_ = {};
};

TEST_F(NativeDecisionServiceTest,
       SelectionControlsBatchingAndAcceptedWorkOwnsProgress) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_service_finite_model_options_t options = {
      .sources = *sources_,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  IREE_ASSERT_OK(AllocateRegistration(options, 3));
  const auto* control = loom_service_registration_control(registration_);
  const auto load = OfferedWork(LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  ASSERT_EQ(load.choice_count, 1u);
  EXPECT_FALSE(loom_service_registration_progress(registration_, 0));
  EXPECT_EQ(loom_service_registration_snapshot(registration_).phase,
            LOOM_SERVICE_REGISTRATION_DECLARED);
  const auto load_operation = Schedule(load.key);
  ASSERT_NE(load_operation.serial, 0u);
  EXPECT_EQ(Schedule(load.key).serial, 0u);
  IREE_ASSERT_OK(ProgressUntil([&] {
    return loom_service_registration_snapshot(registration_).phase ==
           LOOM_SERVICE_REGISTRATION_READY;
  }));
  ASSERT_NE(service_, nullptr);

  // Explicit management grants own independent compiler lifetimes. A failed
  // candidate stays local while another candidate becomes usable, and neither
  // needs an inference request or another scheduling grant to complete.
  const auto configuration = interface_.reconfiguration;
  const auto original = configuration.selected(configuration.self);
  const iree_string_view_t providers[] = {IREE_SV("config.loom"),
                                          IREE_SV("decide.loom")};
  const loom_runtime_sources_t replacement = {
      .base_directory = sources_->base_directory,
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  loom_runtime_variant_id_t candidate = 0;
  IREE_ASSERT_OK(configuration.stage_native(configuration.self, original,
                                            &replacement, &candidate));
  const iree_string_view_t missing = IREE_SV("missing-native-provider.loom");
  const loom_runtime_sources_t invalid = {
      .base_directory = sources_->base_directory,
      .count = 1,
      .values = &missing,
  };
  loom_runtime_variant_id_t failed = 0;
  IREE_ASSERT_OK(configuration.stage_native(configuration.self, original,
                                            &invalid, &failed));
  EXPECT_NE(candidate, failed);
  EXPECT_TRUE(loom_runtime_variant_pool_pending(configuration.pool));
  const auto compilations = AcceptedOperations();
  ASSERT_EQ(compilations.size(), 2u);
  EXPECT_NE(compilations[0].key.ordinal, compilations[1].key.ordinal);
  loom_runtime_service_observation_t management = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &management),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(management.pending_operation_count, 2u);
  EXPECT_EQ(management.owned_request_count, 0u);
  EXPECT_EQ(management.offer_count, 0u);
  EXPECT_EQ(management.accepted_devices, 1u);
  for (const auto& compilation : compilations) {
    EXPECT_EQ(compilation.origin.revision, 0u);
    EXPECT_EQ(compilation.waits, LOOM_RUNTIME_SERVICE_WAIT_CPU);
    EXPECT_EQ(compilation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
    loom_runtime_service_use_t use = {};
    ASSERT_EQ(loom_runtime_service_read_operation_use(
                  control, compilation.key, compilation.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(use.resource_ordinal, 0u);
    EXPECT_EQ(use.estimates, 0u);
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_runtime_variant_pool_remove(configuration.pool, candidate));
  loom_runtime_variant_id_t rejected = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        configuration.stage_native(configuration.self, original,
                                                   &replacement, &rejected));
  EXPECT_EQ(rejected, 0u);
  IREE_ASSERT_OK(ProgressUntil(
      [&] { return !loom_runtime_variant_pool_pending(configuration.pool); }));
  EXPECT_TRUE(AcceptedOperations().empty());
  for (const auto& compilation : compilations) {
    loom_runtime_service_use_t use = {};
    EXPECT_EQ(loom_runtime_service_read_operation_use(
                  control, compilation.key, compilation.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_CHANGED);
  }
  loom_runtime_variant_snapshot_t variant = {};
  ASSERT_TRUE(loom_runtime_variant_pool_snapshot(configuration.pool, candidate,
                                                 &variant));
  EXPECT_EQ(variant.phase, LOOM_RUNTIME_VARIANT_READY);
  ASSERT_TRUE(
      loom_runtime_variant_pool_snapshot(configuration.pool, failed, &variant));
  EXPECT_EQ(variant.phase, LOOM_RUNTIME_VARIANT_FAILED);
  EXPECT_EQ(variant.code, IREE_STATUS_NOT_FOUND);
  IREE_ASSERT_OK(loom_runtime_variant_pool_remove(configuration.pool, failed));
  struct Selection {
    // Owner callback has relinquished its borrowed result storage.
    bool complete = false;
    // Actual preferred code after the selection attempt.
    loom_runtime_variant_id_t selected = 0;
    // Terminal selection outcome, independent of compilation diagnostics.
    iree_status_code_t code = IREE_STATUS_UNKNOWN;
  } selection;
  IREE_ASSERT_OK(configuration.select(
      configuration.self, candidate, original,
      [](void* self, iree_status_t status, loom_runtime_variant_id_t selected) {
        auto& result = *static_cast<Selection*>(self);
        result.selected = selected;
        result.code = iree_status_code(status);
        iree_status_free(status);
        result.complete = true;
      },
      &selection));
  EXPECT_FALSE(selection.complete);
  const auto selecting = AcceptedOperations();
  ASSERT_EQ(selecting.size(), 1u);
  EXPECT_NE(selecting[0].key.ordinal, compilations[0].key.ordinal);
  EXPECT_NE(selecting[0].key.ordinal, compilations[1].key.ordinal);
  EXPECT_EQ(selecting[0].waits, LOOM_RUNTIME_SERVICE_WAIT_QUIESCENCE);
  IREE_ASSERT_OK(ProgressUntil([&] { return selection.complete; }));
  EXPECT_TRUE(AcceptedOperations().empty());
  EXPECT_EQ(selection.selected, candidate);
  EXPECT_EQ(selection.code, IREE_STATUS_OK);
  EXPECT_EQ(configuration.selected(configuration.self), candidate);

  // Explicit warmup is another deferred native choice. Withdrawing it makes
  // the old decision stale; accepted warmup later joins with no new grants.
  IREE_ASSERT_OK(loom_service_registration_request_parameters(
      registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
      LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE));
  const auto withdrawn = OfferedWork(LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  ASSERT_EQ(withdrawn.choice_count, 1u);
  EXPECT_FALSE(loom_service_registration_progress(registration_, 0));
  IREE_ASSERT_OK(loom_service_registration_request_parameters(
      registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
      LOOM_SERVICE_PARAMETER_OPERATION_NONE));
  EXPECT_EQ(Schedule(withdrawn.key).serial, 0u);
  IREE_ASSERT_OK(loom_service_registration_request_parameters(
      registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
      LOOM_SERVICE_PARAMETER_OPERATION_ACTIVATE));
  const auto warmup = OfferedWork(LOOM_RUNTIME_SERVICE_WORK_ACQUISITION);
  const auto warmup_operation = Schedule(warmup.key);
  ASSERT_NE(warmup_operation.serial, 0u);
  EXPECT_NE(warmup_operation.serial, load_operation.serial);
  IREE_ASSERT_OK(ProgressUntil([&] {
    return loom_service_registration_snapshot(registration_)
               .parameter_operation == LOOM_SERVICE_PARAMETER_OPERATION_NONE;
  }));
  if (loom_service_registration_parameters(registration_,
                                           LOOM_SERVICE_MODEL_COMPONENT_PRIMARY)
          .parameters.elastic) {
    IREE_ASSERT_OK(loom_service_registration_request_parameters(
        registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
        LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE));
    const auto unpinned = OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE);
    ASSERT_EQ(unpinned.choice_count, 1u);
    loom_service_registration_set_pinned(
        registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY, true);
    EXPECT_EQ(OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE).choice_count, 0u);
    EXPECT_EQ(Schedule(unpinned.key).serial, 0u);
    loom_service_registration_set_pinned(
        registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY, false);
    ASSERT_NE(
        Schedule(OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE).key).serial,
        0u);
    IREE_ASSERT_OK(ProgressUntil([&] {
      return loom_service_registration_snapshot(registration_)
                 .parameter_operation == LOOM_SERVICE_PARAMETER_OPERATION_NONE;
    }));
    EXPECT_FALSE(loom_service_registration_parameters(
                     registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY)
                     .parameters.active);
  } else {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_service_registration_request_parameters(
            registration_, LOOM_SERVICE_MODEL_COMPONENT_PRIMARY,
            LOOM_SERVICE_PARAMETER_OPERATION_DEACTIVATE));
  }
  EXPECT_EQ(loom_service_registration_snapshot(registration_).phase,
            LOOM_SERVICE_REGISTRATION_READY);
  IREE_ASSERT_OK(
      loom_runtime_device_create_execution(device_, pool_, &consumer_));
  const float expected_probabilities[] = {2.0f / 3.0f, 0.25f, 1.0f / 3.0f};
  uint64_t preceding_serial = 0;
  for (size_t count = 1; count <= 3; ++count) {
    ASSERT_NO_FATAL_FAILURE(Open(0));
    const auto initial = Offer();
    for (size_t i = 1; i < count; ++i) {
      ASSERT_NO_FATAL_FAILURE(Open(i));
    }
    const auto preparation = Offer();
    EXPECT_EQ(preparation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
    loom_runtime_service_operation_key_t accepted = {};
    if (count > 1) {
      accepted = Schedule(initial.key);
      EXPECT_EQ(accepted.serial, 0u);  // The candidate window changed.
    }
    bool progress = loom_service_registration_progress(registration_, 0);
    loom_runtime_service_observation_t observation = {};
    ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(observation.owned_request_count, count);
    EXPECT_EQ(observation.waiting_request_count, count);
    EXPECT_EQ(observation.offer_count, 1u);
    EXPECT_EQ(observation.pending_operation_count, 0u);
    EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);

    accepted = Schedule(preparation.key);
    ASSERT_GT(accepted.serial, preceding_serial);
    const auto prepared_operation = accepted;
    EXPECT_EQ(Offer().choice_count, 0u);
    IREE_ASSERT_OK(ProgressUntil(
        [&] { return Offer().work == LOOM_RUNTIME_SERVICE_WORK_EXECUTION; }));
    EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
    const auto execution = Offer();
    loom_runtime_service_choice_t choice = {};
    ASSERT_EQ(
        loom_runtime_service_read_choice(control, execution.key, 0, &choice),
        LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(choice.request_count, count);
    EXPECT_EQ(choice.devices, 1u);
    EXPECT_EQ(choice.estimates, 0u);
    // Progress with no accepted work preserves the selected choice's use join.
    progress = loom_service_registration_progress(registration_, 0);
    EXPECT_FALSE(progress);
    loom_runtime_service_use_t offered_use = {};
    EXPECT_EQ(
        loom_runtime_service_read_choice_use(
            control, execution.key, 0, choice.publication, 0, &offered_use),
        LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    accepted = Schedule(preparation.key);
    EXPECT_EQ(accepted.serial, 0u);
    accepted = Schedule(execution.key);
    ASSERT_GT(accepted.serial, prepared_operation.serial);
    preceding_serial = accepted.serial;
    loom_runtime_service_operation_t operation = {};
    ASSERT_EQ(loom_runtime_service_read_operation(control, accepted.ordinal,
                                                  &operation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(operation.key.serial, accepted.serial);
    EXPECT_EQ(operation.origin.revision, execution.key.revision);
    EXPECT_EQ(operation.work, LOOM_RUNTIME_SERVICE_WORK_EXECUTION);
    EXPECT_EQ(Offer().choice_count, 0u);
    loom_runtime_service_use_t use = {};
    ASSERT_EQ(loom_runtime_service_read_operation_use(
                  control, operation.key, operation.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(use.resource_ordinal, 0u);
    accepted = Schedule(execution.key);
    EXPECT_EQ(accepted.serial, 0u);

    loom_runtime_variant_id_t overlapping_candidate = 0;
    if (count == 1) {
      IREE_ASSERT_OK(configuration.stage_native(
          configuration.self, candidate, &replacement, &overlapping_candidate));
      ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
      EXPECT_EQ(observation.pending_operation_count, 2u);
      EXPECT_EQ(observation.owned_request_count, 1u);
      // Another producer's publication cannot invalidate this cohort's use.
      EXPECT_EQ(loom_runtime_service_read_operation_use(
                    control, operation.key, operation.publication, 0, &use),
                LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    }
    // Only accepted obligations progress; no automatic selection drives this.
    IREE_ASSERT_OK(ProgressUntil([&] {
      return loom_service_finite_call_is_complete(calls_[count - 1]) &&
             !loom_runtime_variant_pool_pending(configuration.pool);
    }));
    if (overlapping_candidate) {
      ASSERT_TRUE(loom_runtime_variant_pool_snapshot(
          configuration.pool, overlapping_candidate, &variant));
      EXPECT_EQ(variant.phase, LOOM_RUNTIME_VARIANT_READY);
      IREE_ASSERT_OK(loom_runtime_variant_pool_remove(configuration.pool,
                                                      overlapping_candidate));
    }
    for (size_t i = 0; i < count; ++i) {
      IREE_ASSERT_OK(loom_service_finite_call_status(calls_[i]));
      ASSERT_NE(loom_service_finite_call_output(calls_[i]), nullptr);
    }
    const auto* output = loom_service_finite_call_output(calls_[0]);
    auto ready = loom_service_finite_result_ready(output->result);
    EXPECT_EQ(ready.values[1], 0u);  // Observation adds no feedback submission.
    IREE_ASSERT_OK(loom_runtime_execution_barrier(
        consumer_, {2, ready.semaphores, ready.values}));
    const auto row = loom_service_finite_result_row(output->result,
                                                    output->request_index, 0);
    ASSERT_EQ(row.length, sizeof(values_));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = row.buffer,
                .source_offset = row.offset,
                .target = values_.data(),
                .length = row.length,
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_runtime_execution_feedback(consumer_, 1, &download, &value));
    IREE_ASSERT_OK(loom_runtime_execution_drain(consumer_));
    EXPECT_NEAR(values_[1], expected_probabilities[count - 1], 2.5 * 0x1p-25);
    EXPECT_NEAR(values_[0], 1.0f - expected_probabilities[count - 1],
                2.5 * 0x1p-25);
    for (size_t i = 0; i < count; ++i) {
      loom_service_finite_call_close(std::exchange(calls_[i], nullptr));
    }
    ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    EXPECT_EQ(observation.owned_request_count, 0u);
    EXPECT_EQ(observation.pending_operation_count, 0u);
    EXPECT_EQ(loom_runtime_service_read_operation_use(
                  control, operation.key, operation.publication, 0, &use),
              LOOM_RUNTIME_SERVICE_READ_CHANGED);
  }
  // Reuse a collected compiler slot, then remove while its replacement is
  // accepted. Relinquishing frontend views cannot release that compiler borrow.
  loom_runtime::testing::WorkerGate compiler_gate(
      loom_runtime_host_executor(loom_runtime_device_host(device_)));
  loom_runtime_variant_id_t final_candidate = 0;
  IREE_ASSERT_OK(configuration.stage_native(configuration.self, candidate,
                                            &replacement, &final_candidate));
  EXPECT_TRUE(loom_runtime_variant_pool_pending(configuration.pool));
  const auto retiring_compilations = AcceptedOperations();
  ASSERT_EQ(retiring_compilations.size(), 1u);
  const auto retiring_compilation = retiring_compilations[0];
  for (const auto& compilation : compilations) {
    if (compilation.key.ordinal == retiring_compilation.key.ordinal) {
      EXPECT_GT(retiring_compilation.key.serial, compilation.key.serial);
    }
  }
  // Relinquish the embedding application's engine views before granting
  // release.
  loom_runtime_driver_set_observer(&driver_, {});
  driver_.services = nullptr;
  driver_.capacity = 0;
  interface_ = {};
  service_ = nullptr;
  loom_service_registration_request_retirement(registration_, 0);
  EXPECT_EQ(OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE).choice_count, 0u);
  loom_runtime_service_use_t retiring_use = {};
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, retiring_compilation.key,
                retiring_compilation.publication, 0, &retiring_use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_FALSE(loom_service_registration_progress(registration_, 0));
  // Prepare the wait before releasing the compiler: its completion must wake
  // the native lifecycle after the application's traversal has detached.
  const auto retirement_ticket = iree_notification_prepare_wait(&notification_);
  compiler_gate.Open();
  iree_notification_commit_wait(&notification_, retirement_ticket,
                                IREE_DURATION_ZERO, IREE_TIME_INFINITE_FUTURE);
  // Worker completion transfers ownership; only native collection retires the
  // published operation, independently of the earlier withdrawal of admission.
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, retiring_compilation.key,
                retiring_compilation.publication, 0, &retiring_use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  IREE_ASSERT_OK(ProgressUntil([&] {
    return OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE).choice_count != 0;
  }));
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, retiring_compilation.key,
                retiring_compilation.publication, 0, &retiring_use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
  const auto release = OfferedWork(LOOM_RUNTIME_SERVICE_WORK_RELEASE);
  ASSERT_EQ(release.choice_count, 1u);
  EXPECT_FALSE(loom_service_registration_progress(registration_, 0));
  EXPECT_EQ(loom_service_registration_snapshot(registration_).phase,
            LOOM_SERVICE_REGISTRATION_READY);
  ASSERT_NE(Schedule(release.key).serial, 0u);
  IREE_ASSERT_OK(ProgressUntil([&] {
    return loom_service_registration_snapshot(registration_).phase ==
           LOOM_SERVICE_REGISTRATION_RETIRED;
  }));
  EXPECT_EQ(control, loom_service_registration_control(registration_));
  EXPECT_EQ(loom_runtime_service_control_layout(control).incarnation,
            incarnation_);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.pending_operation_count, 0u);
}

TEST_F(NativeDecisionServiceTest, PreparationFailurePublishesNoService) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  loom_service_finite_model_options_t options = {
      .sources =
          {
              .base_directory = sources_->base_directory,
              .prepare = IREE_SV("missing-prepare.loom"),
          },
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  ASSERT_NE(
      Schedule(OfferedWork(LOOM_RUNTIME_SERVICE_WORK_PREPARATION).key).serial,
      0u);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      ProgressUntil(
          [&] {
            return loom_service_registration_snapshot(registration_).phase ==
                   LOOM_SERVICE_REGISTRATION_FAILED;
          },
          0, IREE_STATUS_NOT_FOUND));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  const auto failed = loom_service_registration_snapshot(registration_);
  EXPECT_EQ(failed.service.self, nullptr);
  EXPECT_EQ(iree_status_code(failed.status), IREE_STATUS_NOT_FOUND);
  loom_runtime_service_observation_t observation;
  ASSERT_EQ(loom_runtime_service_read_observation(
                loom_service_registration_control(registration_), &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.offer_count, 0u);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  EXPECT_EQ(observation.accepted_devices, 0u);
  loom_service_registration_request_retirement(registration_, 0);
  // A retained error is observable while accepted cleanup still progresses.
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      ProgressUntil(
          [&] {
            return loom_service_registration_snapshot(registration_).phase ==
                   LOOM_SERVICE_REGISTRATION_RETIRED;
          },
          0, IREE_STATUS_NOT_FOUND));
  EXPECT_EQ(loom_service_registration_snapshot(registration_).phase,
            LOOM_SERVICE_REGISTRATION_RETIRED);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, UnstartedCancellationDoesNotReadSource) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_service_finite_model_options_t options = {
      .sources = {.prepare = IREE_SV("missing-prepare.loom")},
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(LOOM_SERVICE_REGISTRATION_PREPARE_FLAG_CANCEL));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, UnstartedRetirementWithdrawsPreparation) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_service_finite_model_options_t options = {
      .sources = *sources_,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  const auto* control = loom_service_registration_control(registration_);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.offer_count, 1u);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  // Exercise the native unstarted disposal path without the fixture's normal
  // cancellation-and-join step for preparation that has already begun.
  preparation_complete_ = true;
  ASSERT_NO_FATAL_FAILURE(RetireModel());
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.offer_count, 0u);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  EXPECT_EQ(observation.offered_devices, 0u);
  EXPECT_EQ(observation.accepted_devices, 0u);
}

TEST_F(NativeDecisionServiceTest, PreparationCancellationJoinsAcceptedLoad) {
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const loom_service_finite_model_options_t options = {
      .sources = *sources_,
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{\"batch_units\":64}"),
  };
  IREE_ASSERT_OK(AllocateRegistration(options));
  const auto* control = loom_service_registration_control(registration_);
  const auto layout = loom_runtime_service_control_layout(control);
  loom_runtime_service_observation_t observation = {};
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.owned_request_count, 0u);
  EXPECT_EQ(observation.offer_count, 1u);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  loom_runtime_service_offer_t preparation = {};
  for (uint64_t i = 0; i < layout.offer_capacity; ++i) {
    loom_runtime_service_offer_t candidate = {};
    ASSERT_EQ(loom_runtime_service_read_offer(control, i, &candidate),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    if (candidate.choice_count) {
      preparation = candidate;
    }
  }
  EXPECT_EQ(preparation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  IREE_ASSERT_OK(loom_service_registration_prepare(registration_, 0,
                                                   &preparation_complete_));
  service_ = loom_service_finite_registration_service(registration_);
  EXPECT_FALSE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.owned_request_count, 0u);
  EXPECT_EQ(observation.offer_count, 0u);
  EXPECT_EQ(observation.pending_operation_count, 1u);
  EXPECT_EQ(observation.accepted_devices, 1u);
  loom_runtime_service_operation_t operation = {};
  for (uint64_t i = 0; i < layout.operation_capacity; ++i) {
    loom_runtime_service_operation_t candidate = {};
    ASSERT_EQ(loom_runtime_service_read_operation(control, i, &candidate),
              LOOM_RUNTIME_SERVICE_READ_COMPLETE);
    if (candidate.key.serial) {
      operation = candidate;
    }
  }
  EXPECT_NE(operation.key.serial, 0u);
  EXPECT_EQ(operation.origin.ordinal, preparation.key.ordinal);
  EXPECT_EQ(operation.origin.revision, preparation.key.revision);
  EXPECT_EQ(operation.work, LOOM_RUNTIME_SERVICE_WORK_PREPARATION);
  loom_runtime_service_use_t use = {};
  ASSERT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(use.resource_ordinal, 0u);
  loom_service_registration_request_retirement(registration_,
                                               LOOM_SERVICE_RETIRE_FLAG_ABORT);
  // Cancellation intent cannot release the accepted preparation borrow.
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      ProgressUntil(
          [&] {
            return loom_service_registration_snapshot(registration_).phase ==
                   LOOM_SERVICE_REGISTRATION_RETIRED;
          },
          0, IREE_STATUS_CANCELLED));
  EXPECT_TRUE(preparation_complete_);
  EXPECT_EQ(service_, nullptr);
  ASSERT_EQ(loom_runtime_service_read_observation(control, &observation),
            LOOM_RUNTIME_SERVICE_READ_COMPLETE);
  EXPECT_EQ(observation.pending_operation_count, 0u);
  EXPECT_EQ(observation.accepted_devices, 0u);
  EXPECT_EQ(loom_runtime_service_read_operation_use(
                control, operation.key, operation.publication, 0, &use),
            LOOM_RUNTIME_SERVICE_READ_CHANGED);
  ASSERT_NO_FATAL_FAILURE(RetireService());
}

TEST_F(NativeDecisionServiceTest, DeviceOutputEscapesBeforeServiceRetirement) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_service_finite_call_output(calls_[0]) != nullptr ||
           loom_service_finite_call_is_complete(calls_[0]);
  }));
  if (loom_service_finite_call_is_complete(calls_[0])) {
    IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  }
  EXPECT_FALSE(loom_service_finite_call_is_complete(calls_[0]));
  EXPECT_TRUE(interface_.runtime.vtable->has_requests(interface_.runtime.self));
  const auto* first = loom_service_finite_call_output(calls_[0]);
  ASSERT_NE(first, nullptr);
  const auto* second = loom_service_finite_call_output(calls_[1]);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->result, second->result);
  EXPECT_EQ(first->request_index, 0u);
  EXPECT_EQ(second->request_index, 1u);
  EXPECT_EQ(second->row_count, 1u);
  const auto ready = loom_service_finite_result_ready(second->result);
  EXPECT_EQ(ready.values[1], 0u);

  IREE_ASSERT_OK(
      loom_runtime_device_create_execution(device_, pool_, &consumer_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_runtime_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  iree_hal_semaphore_t* waits[] = {ready.semaphores[0], ready.semaphores[1],
                                   gate_};
  uint64_t wait_values[] = {ready.values[0], ready.values[1], 1};
  IREE_ASSERT_OK(
      loom_runtime_execution_barrier(consumer_, {3, waits, wait_values}));
  const auto allocation =
      iree_hal_make_pool_reservation_request(sizeof(values_), 16);
  IREE_ASSERT_OK(
      loom_runtime_execution_alloca(consumer_, 1, &allocation, &target_));
  const auto row =
      loom_service_finite_result_row(second->result, second->request_index, 0);
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
  IREE_ASSERT_OK(loom_runtime_execution_transfer(consumer_, 1, &copy, &value));
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
      loom_runtime_execution_feedback(consumer_, 1, &download, &value));
  iree_hal_buffer_release(std::exchange(target_, nullptr));
  for (auto*& call : calls_) {
    loom_service_finite_call_close(std::exchange(call, nullptr));
  }
  ASSERT_NO_FATAL_FAILURE(RetireService());

  // Accepted downstream captures, not the source model or caller receipts,
  // now own the result. Releasing the gate must still yield the packed output.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  iree_hal_semaphore_release(std::exchange(gate_, nullptr));
  IREE_ASSERT_OK(loom_runtime_execution_drain(consumer_));
  EXPECT_FLOAT_EQ(values_[0], 0.75f);
  EXPECT_FLOAT_EQ(values_[1], 0.25f);
}

TEST_F(NativeDecisionServiceTest, PressurePreservesInputAndRejectionIsLocal) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1, "{}", LOOM_SERVICE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(2));
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(&body, kRequest));
  const auto original = iree_string_builder_view(&body);
  for (auto format :
       {LOOM_SERVICE_FINITE_OUTPUT_DEVICE, LOOM_SERVICE_FINITE_OUTPUT_JSON}) {
    IREE_EXPECT_OK(loom_service_finite_service_try_open(service_, &body, format,
                                                        &calls_[3]));
    EXPECT_EQ(calls_[3], nullptr);
    EXPECT_EQ(iree_string_builder_view(&body).data, original.data);
    EXPECT_EQ(iree_string_builder_size(&body), original.size);
  }
  iree_string_builder_deinitialize(&body);
  loom_service_finite_call_close(std::exchange(calls_[2], nullptr));

  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]);
  }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_service_finite_call_status(calls_[1]));
  EXPECT_EQ(loom_service_finite_call_output(calls_[1]), nullptr);
  ASSERT_NO_FATAL_FAILURE(Open(3));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_service_finite_call_is_complete(calls_[3]); }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[3]));
  const auto* output = loom_service_finite_call_output(calls_[3]);
  ASSERT_NE(output, nullptr);
  // No earlier native cohort appended a host feedback operation.
  EXPECT_EQ(loom_service_finite_result_ready(output->result).values[1], 0u);
  uint32_t notifications = 0;
  loom_service_finite_call_set_observer(
      calls_[3],
      {[](void* self) { ++*static_cast<uint32_t*>(self); }, &notifications});
  EXPECT_EQ(notifications, 1u);
  loom_service_finite_call_set_observer(calls_[3], {});
}

TEST_F(NativeDecisionServiceTest, StopDuringPreparationJoinsInputBorrowers) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NE(Schedule(Offer().key).serial, 0u);
  IREE_ASSERT_OK(ProgressUntil(
      [&] { return loom_service_finite_call_is_complete(calls_[0]); },
      LOOM_RUNTIME_SERVICE_FLAG_STOPPING));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_service_finite_call_status(calls_[0]));
  EXPECT_EQ(loom_service_finite_call_output(calls_[0]), nullptr);
  const auto stopped = loom_service_registration_snapshot(registration_);
  EXPECT_EQ(stopped.phase, LOOM_SERVICE_REGISTRATION_READY);
  EXPECT_EQ(stopped.service.self, service_);
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_EXPECT_OK(iree_string_builder_append_cstring(&body, kRequest));
  const auto original = iree_string_builder_view(&body);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_service_finite_service_try_open(
          service_, &body, LOOM_SERVICE_FINITE_OUTPUT_DEVICE, &calls_[1]));
  EXPECT_EQ(calls_[1], nullptr);
  EXPECT_EQ(iree_string_builder_view(&body).data, original.data);
  EXPECT_EQ(iree_string_builder_size(&body), original.size);
  iree_string_builder_deinitialize(&body);
  // The stopped engine remains usable for final observation until the
  // application relinquishes its views. Cleanup then needs no new grant.
  loom_runtime_driver_set_observer(&driver_, {});
  driver_.services = nullptr;
  driver_.capacity = 0;
  interface_ = {};
  service_ = nullptr;
  loom_service_registration_request_retirement(registration_,
                                               LOOM_SERVICE_RETIRE_FLAG_ABORT);
  IREE_ASSERT_OK(ProgressUntil([&] {
    return loom_service_registration_snapshot(registration_).phase ==
           LOOM_SERVICE_REGISTRATION_RETIRED;
  }));
}

TEST_F(NativeDecisionServiceTest, NativeCallerUsesExecutorWithoutHostPolling) {
  Owner owner(*this);
  auto admit_requests = [&] {
    EXPECT_FALSE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    ASSERT_NO_FATAL_FAILURE(Open(0));
    ASSERT_NO_FATAL_FAILURE(Open(1));
    EXPECT_TRUE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    for (size_t i = 0; i < 2; ++i) {
      loom_service_finite_call_set_observer(calls_[i], {Wake, this});
    }
  };
  ASSERT_NO_FATAL_FAILURE(owner.Call(admit_requests));
  owner.WaitUntil([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]);
  });
  auto verify_results_and_admit = [&] {
    IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
    IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
    EXPECT_FALSE(
        interface_.runtime.vtable->has_requests(interface_.runtime.self));
    const auto* first = loom_service_finite_call_output(calls_[0]);
    const auto* second = loom_service_finite_call_output(calls_[1]);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->result, second->result);
    EXPECT_EQ(first->request_index, 0u);
    EXPECT_EQ(second->request_index, 1u);
    EXPECT_EQ(loom_service_finite_result_ready(first->result).values[1], 0u);
    ASSERT_NO_FATAL_FAILURE(Open(2));
    loom_service_finite_call_set_observer(calls_[2], {Wake, this});
  };
  ASSERT_NO_FATAL_FAILURE(owner.Call(verify_results_and_admit));
  owner.WaitUntil(
      [&] { return loom_service_finite_call_is_complete(calls_[2]); });
  owner.Call(
      [&] { IREE_EXPECT_OK(loom_service_finite_call_status(calls_[2])); });
}

TEST_F(NativeDecisionServiceTest, JsonResponseOutlivesModelAndResourceOwner) {
  std::unique_ptr<loom_service_finite_call_t,
                  decltype(&loom_service_finite_call_close)>
      retained(nullptr, loom_service_finite_call_close);
  {
    Owner owner(*this);
    auto admit_requests = [&] {
      for (size_t i = 0; i < 2; ++i) {
        ASSERT_NO_FATAL_FAILURE(
            Open(i, kRequest, LOOM_SERVICE_FINITE_OUTPUT_JSON));
        loom_service_finite_call_set_observer(calls_[i], {Wake, this});
      }
    };
    ASSERT_NO_FATAL_FAILURE(owner.Call(admit_requests));
    owner.WaitUntil([&] {
      return loom_service_finite_call_is_complete(calls_[0]) &&
             loom_service_finite_call_is_complete(calls_[1]);
    });
    owner.Call([&] {
      IREE_EXPECT_OK(loom_service_finite_call_status(calls_[1]));
      EXPECT_EQ(loom_service_finite_call_output(calls_[1]), nullptr);
      loom_service_finite_call_set_observer(calls_[1], {});
      retained.reset(std::exchange(calls_[1], nullptr));
    });
  }
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const auto response = loom_service_finite_call_response(retained.get());
  EXPECT_TRUE(iree_string_view_equal(
      response,
      IREE_SV(
          "{\"model\":\"decision\",\"answers\":{\"truth\":{\"type\":\"noul\","
          "\"noul\":0.25}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}"
          "}")));
}

TEST_F(NativeDecisionServiceTest, MixedDeviceAndJsonCallsShareOneDownload) {
  ASSERT_NO_FATAL_FAILURE(Open(0));
  ASSERT_NO_FATAL_FAILURE(Open(1, kRequest, LOOM_SERVICE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(2, kRequest, LOOM_SERVICE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_service_finite_call_is_complete(calls_[0]) &&
           loom_service_finite_call_is_complete(calls_[1]) &&
           loom_service_finite_call_is_complete(calls_[2]);
  }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[0]));
  const auto* device_output = loom_service_finite_call_output(calls_[0]);
  ASSERT_NE(device_output, nullptr);
  EXPECT_EQ(device_output->request_index, 0u);
  EXPECT_EQ(loom_service_finite_result_ready(device_output->result).values[1],
            0u);
  for (size_t i = 1; i < 3; ++i) {
    IREE_ASSERT_OK(loom_service_finite_call_status(calls_[i]));
    EXPECT_EQ(loom_service_finite_call_output(calls_[i]), nullptr);
    auto value = loom_service_finite_call_response(calls_[i]);
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
      [&] { return loom_service_finite_call_is_complete(calls_[3]); }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[3]));
  const auto* next_output = loom_service_finite_call_output(calls_[3]);
  ASSERT_NE(next_output, nullptr);
  // Both JSON requests used one feedback operation, and the following device
  // call adds none. This queries the next cohort's captured producer frontier.
  EXPECT_EQ(loom_service_finite_result_ready(next_output->result).values[1],
            1u);
}

TEST_F(NativeDecisionServiceTest, JsonCancellationJoinsEncodingBorrowers) {
  ASSERT_NO_FATAL_FAILURE(Open(0, kRequest, LOOM_SERVICE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(Open(1, kRequest, LOOM_SERVICE_FINITE_OUTPUT_JSON));
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return std::string(interface_.snapshot(interface_.runtime.self).phase) ==
           "collecting";
  }));
  loom_service_finite_call_close(std::exchange(calls_[0], nullptr));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_service_finite_call_is_complete(calls_[1]); }));
  IREE_ASSERT_OK(loom_service_finite_call_status(calls_[1]));
  ASSERT_NO_FATAL_FAILURE(RetireService());
  const auto response = loom_service_finite_call_response(calls_[1]);
  EXPECT_TRUE(iree_string_view_equal(
      response,
      IREE_SV(
          "{\"model\":\"decision\",\"answers\":{\"truth\":{\"type\":\"noul\","
          "\"noul\":0.25}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}"
          "}")));
}

}  // namespace
