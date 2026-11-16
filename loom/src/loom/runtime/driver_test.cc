// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/driver.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/service_control.h"
#include "loom/runtime/service_schedule.h"

namespace {

// Controlled dependencies of the real driver, with distinct request ownership,
// asynchronous borrowers, and capacity activity.
struct Service {
  // Stable ordinal for exact traversal observations.
  size_t index = 0;
  // Borrowed traversal log.
  std::vector<size_t>* visits = nullptr;
  // Installed owner wake, never invoked synchronously here.
  loom_runtime_wake_callback_t observer = {};
  // Most recently delivered progression flags.
  loom_runtime_service_flags_t flags = 0;
  // Accepted request storage, including queued work.
  bool requests = false;
  // Accepted asynchronous loan awaiting an external completion.
  bool pending = false;
  // Work capable of releasing shared backing.
  bool active = false;
  // Persistent terminal failure, not a one-shot test event.
  bool failed = false;
  // Optional publication cleared by the service during its terminal turn.
  loom_runtime_service_t* retirement_slot = nullptr;

  loom_runtime_service_t Interface(const void* capacity_domain) {
    static const loom_runtime_service_vtable_t vtable = {
        .set_observer =
            [](void* self, loom_runtime_wake_callback_t observer) {
              static_cast<Service*>(self)->observer = observer;
            },
        .has_requests =
            [](void* self) { return static_cast<Service*>(self)->requests; },
        .is_pending =
            [](void* self) { return static_cast<Service*>(self)->pending; },
        .is_active =
            [](void* self) { return static_cast<Service*>(self)->active; },
        .advance = [](void* self, loom_runtime_service_flags_t flags,
                      bool* out_progress) -> iree_status_t {
          auto& service = *static_cast<Service*>(self);
          service.visits->push_back(service.index);
          service.flags = flags;
          *out_progress = false;
          if (iree_any_bit_set(flags, LOOM_RUNTIME_SERVICE_FLAG_STOPPING) &&
              !service.pending) {
            *out_progress = service.requests;
            service.requests = false;
          }
          if (service.retirement_slot) {
            *service.retirement_slot = {};
            *out_progress = true;
          }
          return service.failed
                     ? iree_make_status(IREE_STATUS_DATA_LOSS, "failed service")
                     : iree_ok_status();
        },
    };
    return {this, &vtable, capacity_domain};
  }
};

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < services_.size(); ++i) {
      services_[i].index = i;
      services_[i].visits = &visits_;
      slots_[i] =
          services_[i].Interface(i < 2 ? &first_domain_ : &second_domain_);
    }
    loom_runtime_driver_initialize(slots_.size(), slots_.data(), &driver_);
  }

  void TearDown() override {
    loom_runtime_driver_set_observer(&driver_, {});
    iree_status_free(driver_.status);
  }

  // Production driver under test.
  loom_runtime_driver_t driver_ = {};
  // Stable publications, including two vacant slots.
  std::array<loom_runtime_service_t, 5> slots_ = {};
  // Service implementations retained independently of their publications.
  std::array<Service, 3> services_;
  // Exact traversal observations.
  std::vector<size_t> visits_;
  // Shared capacity of the first two services.
  int first_domain_ = 0;
  // Independent capacity of the third service.
  int second_domain_ = 0;
};

TEST_F(DriverTest, RotatesAcrossVacantSlotsAndInstallsObserver) {
  const loom_runtime_wake_callback_t observer = {[](void*) {}, this};
  loom_runtime_driver_set_observer(&driver_, observer);
  for (const auto& service : services_) {
    EXPECT_EQ(service.observer.user_data, this);
  }
  const std::vector<size_t> expected[] = {
      {0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2}, {0, 1, 2}};
  for (const auto& turn : expected) {
    visits_.clear();
    bool progress = true;
    loom_runtime_driver_advance(&driver_, &progress);
    EXPECT_EQ(visits_, turn);
    EXPECT_FALSE(progress);
    EXPECT_FALSE(loom_runtime_driver_collect(&driver_, &progress));
  }
  EXPECT_EQ(driver_.cursor, 0u);
}

TEST_F(DriverTest, PendingLoansAndQueuedRequestsDoNotPromiseSharedCapacity) {
  services_[1].pending = true;
  services_[1].requests = true;
  EXPECT_EQ(loom_runtime_driver_shared_activity(&driver_, 0), 0u);
  services_[1].active = true;
  EXPECT_EQ(loom_runtime_driver_shared_activity(&driver_, 0),
            LOOM_RUNTIME_SERVICE_FLAG_SHARED_ACTIVITY);
  EXPECT_EQ(loom_runtime_driver_shared_activity(&driver_, 1), 0u);
  EXPECT_EQ(loom_runtime_driver_shared_activity(&driver_, 2), 0u);
  slots_[0].capacity_domain = nullptr;
  slots_[1].capacity_domain = nullptr;
  EXPECT_EQ(loom_runtime_driver_shared_activity(&driver_, 0), 0u);
}

TEST_F(DriverTest, LastFailureStopsEarlierQueuedWorkBeforeDetachment) {
  services_[0].requests = true;
  services_[2].failed = true;
  bool progress = false;
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_TRUE(driver_.stopping);
  EXPECT_TRUE(services_[0].requests);
  EXPECT_FALSE(services_[0].flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  progress = false;
  EXPECT_TRUE(loom_runtime_driver_collect(&driver_, &progress));
  EXPECT_TRUE(progress);
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_FALSE(services_[0].requests);
  for (const auto& service : services_) {
    EXPECT_TRUE(service.flags & LOOM_RUNTIME_SERVICE_FLAG_STOPPING);
  }
  progress = false;
  EXPECT_FALSE(loom_runtime_driver_collect(&driver_, &progress));
  EXPECT_FALSE(progress);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_status_clone(driver_.status));
}

TEST_F(DriverTest, RepeatedFailureWaitsForAcceptedLoansWithoutPolling) {
  services_[0].pending = true;
  services_[0].failed = true;
  bool progress = false;
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_TRUE(progress);
  EXPECT_TRUE(loom_runtime_driver_collect(&driver_, &progress));
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_FALSE(progress);
  EXPECT_TRUE(loom_runtime_driver_collect(&driver_, &progress));
  EXPECT_FALSE(progress);
  services_[0].pending = false;
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_runtime_driver_collect(&driver_, &progress));
}

TEST_F(DriverTest, LateAdapterFailureRequiresACompleteStoppingTurn) {
  services_[0].requests = true;
  bool progress = false;
  loom_runtime_driver_advance(&driver_, &progress);
  driver_.status = iree_make_status(IREE_STATUS_DATA_LOSS, "adapter failed");
  EXPECT_TRUE(loom_runtime_driver_collect(&driver_, &progress));
  EXPECT_TRUE(progress);
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_FALSE(services_[0].requests);
  EXPECT_FALSE(loom_runtime_driver_collect(&driver_, &progress));
}

TEST_F(DriverTest, CurrentSlotCanRetireAndBeRepublished) {
  services_[1].retirement_slot = &slots_[1];
  bool progress = false;
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_TRUE(progress);
  EXPECT_EQ(slots_[1].vtable, nullptr);
  visits_.clear();
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_EQ(visits_, (std::vector<size_t>{2, 0}));
  services_[1].retirement_slot = nullptr;
  slots_[1] = services_[1].Interface(&first_domain_);
  visits_.clear();
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_EQ(visits_, (std::vector<size_t>{2, 0, 1}));
}

TEST_F(DriverTest, StaleDecisionsSpendBothIndependentRoundBudgets) {
  std::array<uint64_t, 17> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      1, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  size_t evaluations = 0;
  const loom_runtime_service_policy_t policy = {
      .self = &evaluations,
      .select =
          [](void* self, const loom_runtime_driver_t*,
             loom_runtime_service_schedule_context_t*,
             loom_runtime_service_plan_t* output) {
            ++*static_cast<size_t*>(self);
            loom_runtime_service_plan_reset(output);
            const loom_runtime_service_decision_t stale = {
                .service_ordinal = 4,
                .service_incarnation = 1,
            };
            EXPECT_TRUE(loom_runtime_service_plan_append(output, &stale));
          },
  };
  for (const auto initial : {loom_runtime_service_schedule_budget_t{3, 8},
                             loom_runtime_service_schedule_budget_t{8, 3}}) {
    auto budget = initial;
    evaluations = 0;
    loom_runtime_service_schedule_context_t context = {.now_ns = 123};
    bool progress = false;
    loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                                 &progress);
    EXPECT_FALSE(progress);
    EXPECT_EQ(evaluations, 3u);
    EXPECT_EQ(budget.evaluations, initial.evaluations - 3);
    EXPECT_EQ(budget.decisions, initial.decisions - 3);
    EXPECT_EQ(context.wake_time_ns, context.now_ns);
    EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 1u);
    const auto outcome = loom_runtime_service_plan_outcome(plan, 0);
    EXPECT_FALSE(outcome.accepted);
    EXPECT_EQ(outcome.status_code, IREE_STATUS_OK);
    EXPECT_EQ(outcome.operation.serial, 0u);
  }
}

TEST_F(DriverTest, WithheldBudgetsPreservePlansWithoutPolling) {
  std::array<uint64_t, 17> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      1, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  size_t evaluations = 0;
  const loom_runtime_service_policy_t policy = {
      .self = &evaluations,
      .select =
          [](void* self, const loom_runtime_driver_t*,
             loom_runtime_service_schedule_context_t* context,
             loom_runtime_service_plan_t* output) {
            ++*static_cast<size_t*>(self);
            loom_runtime_service_plan_reset(output);
            const loom_runtime_service_decision_t stale = {
                .service_ordinal = 4,
                .service_incarnation = 1,
            };
            EXPECT_TRUE(loom_runtime_service_plan_append(output, &stale));
            context->wake_time_ns = 500;
          },
  };
  services_[0].pending = true;
  bool progress = false;
  loom_runtime_driver_advance(&driver_, &progress);
  EXPECT_TRUE(loom_runtime_driver_collect(&driver_, &progress));
  EXPECT_FALSE(progress);
  loom_runtime_service_schedule_context_t context = {.now_ns = 123};
  loom_runtime_service_schedule_budget_t budget = {0, 4};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(evaluations, 0u);
  EXPECT_EQ(context.wake_time_ns, IREE_TIME_INFINITE_FUTURE);
  EXPECT_FALSE(progress);

  budget = {3, 0};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(evaluations, 1u);
  EXPECT_EQ(context.wake_time_ns, 500);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).decision_count, 1u);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 0u);
  EXPECT_FALSE(progress);

  budget = {0, 0};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(evaluations, 1u);
  EXPECT_EQ(context.wake_time_ns, IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 0u);
  EXPECT_FALSE(progress);

  // Existing decisions can consume granted application credit independently
  // of policy evaluation. The prefix is attempted once, then remains inert.
  budget = {0, 1};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(loom_runtime_service_plan_layout(plan).applied_count, 1u);
  EXPECT_EQ(evaluations, 1u);
  EXPECT_EQ(context.wake_time_ns, context.now_ns);
  EXPECT_FALSE(progress);
  budget = {0, 1};
  loom_runtime_driver_schedule(&driver_, policy, plan, &budget, &context,
                               &progress);
  EXPECT_EQ(budget.decisions, 1u);
  EXPECT_EQ(context.wake_time_ns, IREE_TIME_INFINITE_FUTURE);
  EXPECT_FALSE(progress);
}

TEST(DriverPolicyTest, FullOutputVisitsPeersBeforeWrappingOffers) {
  const iree_host_size_t use_capacity = 0;
  const loom_runtime_service_offer_capacity_t offers[] = {
      {1, &use_capacity}, {1, &use_capacity}, {1, &use_capacity}};
  const loom_runtime_service_control_capacity_t capacity = {
      .offer_capacity = 3,
      .offers = offers,
  };
  iree_host_size_t length = 0;
  IREE_ASSERT_OK(loom_runtime_service_control_size(&capacity, &length));
  std::array<std::vector<uint64_t>, 2> controls;
  std::array<loom_runtime_service_t, 2> slots = {};
  size_t attempts = 0;
  const loom_runtime_service_vtable_t vtable = {
      .try_schedule =
          [](void* self, loom_runtime_service_offer_key_t, uint64_t,
             loom_runtime_service_flags_t,
             loom_runtime_service_schedule_outcome_t* outcome) {
            ++*static_cast<size_t*>(self);
            *outcome = {};
            return iree_ok_status();
          },
  };
  for (size_t i = 0; i < controls.size(); ++i) {
    controls[i].resize(length / sizeof(uint64_t));
    loom_runtime_service_control_t* control = nullptr;
    IREE_ASSERT_OK(loom_runtime_service_control_initialize(
        &capacity, i + 1, iree_make_byte_span(controls[i].data(), length),
        &control));
    slots[i] = {.self = &attempts, .vtable = &vtable, .control = control};
    for (uint64_t j = 0; j < 3; ++j) {
      const loom_runtime_service_offer_t offer = {
          .key = {j, 1},
          .choice_count = 1,
      };
      const loom_runtime_service_choice_publication_t choice = {};
      loom_runtime_service_publish_offer(control, &offer, &choice);
    }
  }
  loom_runtime_driver_t driver;
  loom_runtime_driver_initialize(slots.size(), slots.data(), &driver);
  std::array<uint64_t, 17> storage;
  loom_runtime_service_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_runtime_service_plan_initialize(
      1, iree_make_byte_span(storage.data(), sizeof(storage)), &plan));
  loom_runtime_service_default_policy_t state = {};
  const auto policy = loom_runtime_service_default_policy(&state);
  loom_runtime_service_schedule_context_t context = {};
  // Keep every opportunity offered. A full output must still reach the second
  // service instead of repeatedly wrapping the first one's three choices.
  for (uint64_t i = 0; i < 12; ++i) {
    policy.select(policy.self, &driver, &context, plan);
    const auto decision = loom_runtime_service_plan_decision(plan, 0);
    EXPECT_EQ(decision.service_ordinal, (i / 3) % 2);
    EXPECT_EQ(decision.offer.ordinal, i % 3);
    EXPECT_EQ(loom_runtime_service_plan_layout(plan).selection_flags,
              LOOM_RUNTIME_SERVICE_SELECTION_CAPACITY);
    loom_runtime_service_plan_complete(plan, IREE_STATUS_OK, {});
  }
  EXPECT_EQ(attempts, 0u);
  IREE_EXPECT_OK(driver.status);
}

TEST(EmptyDriverTest, StopsWithoutWorkOrPolling) {
  loom_runtime_driver_t driver;
  loom_runtime_driver_initialize(0, nullptr, &driver);
  driver.stopping = true;
  bool progress = true;
  loom_runtime_driver_advance(&driver, &progress);
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_runtime_driver_collect(&driver, &progress));
  EXPECT_EQ(driver.cursor, 0u);
  IREE_EXPECT_OK(driver.status);
}

}  // namespace
