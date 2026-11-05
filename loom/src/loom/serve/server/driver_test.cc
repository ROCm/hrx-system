// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/driver.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controlled service dependencies, not a replacement for the production driver.
// Real model/cancellation paths also use this driver in the decision and HTTP
// suites; these small dependencies make traversal and capacity facts exact.
struct Service {
  // Stable slot ordinal used to observe traversal.
  size_t index = 0;
  // Borrowed trace of service and adapter visits.
  std::vector<size_t>* visits = nullptr;
  // Installed nonblocking owner wake.
  loom_serve_wake_callback_t observer = {};
  // Received flags from the last service turn.
  loom_serve_service_flags_t flags = 0;
  // Accepted callback storage, independent of capacity-releasing activity.
  bool pending = false;
  // Work that can release shared physical capacity.
  bool active = false;
  // Terminal dependency failure published on its next turn.
  bool fail = false;

  loom_serve_service_t Interface(iree_string_view_t name,
                                 const void* capacity_domain) {
    return {
        .name = name,
        .self = this,
        .set_observer =
            [](void* self, loom_serve_wake_callback_t value) {
              static_cast<Service*>(self)->observer = value;
            },
        .is_pending =
            [](void* self) { return static_cast<Service*>(self)->pending; },
        .is_active =
            [](void* self) { return static_cast<Service*>(self)->active; },
        .advance = [](void* self, loom_serve_service_flags_t flags,
                      bool* out_progress) -> iree_status_t {
          auto& service = *static_cast<Service*>(self);
          service.visits->push_back(2 * service.index);
          service.flags = flags;
          *out_progress = false;
          if (service.fail) {
            service.fail = false;
            return iree_make_status(IREE_STATUS_DATA_LOSS, "dependency failed");
          }
          return iree_ok_status();
        },
        .capacity_domain = capacity_domain,
    };
  }
};

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::array<loom_serve_service_t, 3> interfaces;
    const iree_string_view_t names[] = {IREE_SV("a"), IREE_SV("b"),
                                        IREE_SV("c")};
    for (size_t i = 0; i < services_.size(); ++i) {
      services_[i].index = i;
      services_[i].visits = &visits_;
      interfaces[i] = services_[i].Interface(
          names[i], i < 2 ? &first_domain_ : &second_domain_);
    }
    loom_serve_registry_t* registry = nullptr;
    IREE_ASSERT_OK(
        loom_serve_registry_allocate(interfaces.size(), interfaces.data(), 2,
                                     {}, iree_allocator_system(), &registry));
    loom_serve_driver_initialize(registry, &driver_);
  }

  void TearDown() override {
    loom_serve_driver_set_observer(&driver_, {});
    IREE_EXPECT_OK(loom_serve_registry_free(driver_.registry));
    iree_status_free(driver_.status);
  }

  // Driver under test, borrowing the real registry.
  loom_serve_driver_t driver_ = {};
  // Exact service callbacks feeding the shared driver.
  std::array<Service, 3> services_;
  // Service visits have even ordinals; pre-retirement observations are odd.
  std::vector<size_t> visits_;
  // Distinct opaque capacity identities; no physical pool is required here.
  int first_domain_ = 0;
  // Independent capacity owner for the third service.
  int second_domain_ = 0;
};

TEST_F(DriverTest, RotatesAcrossVacantSlotsAndObservesEachTurn) {
  const loom_serve_driver_observer_t observer = {
      .self = &visits_,
      .observe =
          [](void* self, iree_host_size_t index, bool stopping,
             bool* out_progress) {
            EXPECT_FALSE(stopping);
            static_cast<std::vector<size_t>*>(self)->push_back(2 * index + 1);
            *out_progress = false;
            return iree_ok_status();
          },
  };
  const std::vector<size_t> expected[] = {
      {0, 1, 2, 3, 4, 5}, {2, 3, 4, 5, 0, 1}, {4, 5, 0, 1, 2, 3},
      {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5},
  };
  for (const auto& turn : expected) {
    visits_.clear();
    bool progress = false;
    loom_serve_driver_advance(&driver_, {}, observer, &progress);
    EXPECT_EQ(visits_, turn);
    EXPECT_FALSE(progress);
    EXPECT_FALSE(loom_serve_driver_collect(&driver_, &progress));
    IREE_EXPECT_OK(iree_status_clone(driver_.status));
  }
  EXPECT_EQ(driver_.cursor, 0u);
}

TEST_F(DriverTest, CapacityWaitingRequiresActivityInTheSameNonnullDomain) {
  services_[1].pending = true;
  EXPECT_EQ(loom_serve_driver_shared_activity(&driver_, 0), 0u);
  services_[1].active = true;
  EXPECT_EQ(loom_serve_driver_shared_activity(&driver_, 0),
            LOOM_SERVE_SERVICE_FLAG_SHARED_ACTIVITY);
  EXPECT_EQ(loom_serve_driver_shared_activity(&driver_, 1), 0u);
  EXPECT_EQ(loom_serve_driver_shared_activity(&driver_, 2), 0u);
  auto* first = loom_serve_registry_at(driver_.registry, 0);
  auto* second = loom_serve_registry_at(driver_.registry, 1);
  first->service.capacity_domain = nullptr;
  second->service.capacity_domain = nullptr;
  EXPECT_EQ(loom_serve_driver_shared_activity(&driver_, 0), 0u);
  services_[1].pending = false;
}

TEST_F(DriverTest, ApplicationFailureStopsPeersButRetainsPendingOwnership) {
  const loom_serve_wake_callback_t wake = {[](void*) {}, this};
  loom_serve_driver_set_observer(&driver_, wake);
  for (const auto& service : services_) {
    EXPECT_EQ(service.observer.user_data, this);
  }
  services_[0].fail = true;
  services_[0].pending = true;
  bool progress = false;
  loom_serve_driver_advance(&driver_, wake, {}, &progress);
  EXPECT_TRUE(driver_.stopping);
  EXPECT_TRUE(progress);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_status_clone(driver_.status));
  EXPECT_TRUE(services_[1].flags & LOOM_SERVE_SERVICE_FLAG_STOPPING);
  EXPECT_TRUE(services_[2].flags & LOOM_SERVE_SERVICE_FLAG_STOPPING);
  EXPECT_TRUE(loom_serve_driver_collect(&driver_, &progress));
  services_[0].pending = false;
  progress = false;
  loom_serve_driver_advance(&driver_, wake, {}, &progress);
  EXPECT_FALSE(loom_serve_driver_collect(&driver_, &progress));
  EXPECT_FALSE(progress);
}

TEST(EmptyDriverTest, ZeroSlotsCanStopWithoutWorkOrPolling) {
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(loom_serve_registry_allocate(
      0, nullptr, 0, {}, iree_allocator_system(), &registry));
  loom_serve_driver_t driver;
  loom_serve_driver_initialize(registry, &driver);
  driver.stopping = true;
  bool progress = true;
  loom_serve_driver_advance(&driver, {}, {}, &progress);
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_driver_collect(&driver, &progress));
  EXPECT_EQ(driver.cursor, 0u);
  IREE_EXPECT_OK(driver.status);
  IREE_EXPECT_OK(loom_serve_registry_free(registry));
}

}  // namespace
