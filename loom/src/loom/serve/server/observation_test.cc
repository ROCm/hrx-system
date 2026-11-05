// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/observation.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"
#include "loom/runtime/telemetry/sample.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/serve/service/service.h"
#include "loom/service/telemetry/record.h"

namespace {

class ObservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_runtime_telemetry_stream_allocate(
        4, iree_allocator_system(), &stream_));
    const loom_serve_service_t service = {
        .name = IREE_SV("model"),
        .self = this,
        .kind = LOOM_SERVE_MODEL_KIND_EMBEDDING,
        .snapshot =
            [](void* self) {
              auto* test = static_cast<ObservationTest*>(self);
              return loom_serve_service_snapshot_t{
                  .phase = "idle",
                  .active_requests = test->active_,
              };
            },
        .telemetry =
            {
                .stream = stream_,
                .counters = &counters_,
                .sample =
                    [](void* self, loom_runtime_telemetry_sample_sink_t sink) {
                      auto* test = static_cast<ObservationTest*>(self);
                      ++test->samples_;
                      static const loom_runtime_telemetry_metric_t fields[] = {
                          {"resident", "bytes", LOOM_RUNTIME_TELEMETRY_GAUGE}};
                      const loom_runtime_telemetry_topic_t topic = {
                          "storage", 1, 1, fields};
                      return sink.emit(sink.self, &topic, &test->resident_);
                    },
            },
    };
    IREE_ASSERT_OK(loom_serve_registry_allocate(
        1, &service, 130, {}, iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(loom_serve_observation_allocate(
        registry_, &resources_, allocator_.allocator(), &observation_));
    iree_string_builder_initialize_with_storage(storage_, sizeof(storage_),
                                                &output_);
  }

  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    loom_serve_observation_free(observation_);
    IREE_EXPECT_OK(loom_serve_registry_free(registry_));
    loom_runtime_telemetry_stream_free(stream_);
  }

  void Sample(iree_time_t now) {
    iree_host_size_t cursor = 0;
    do {
      IREE_ASSERT_OK(
          loom_serve_observation_sample(observation_, now, &cursor, &output_));
    } while (cursor);
  }

  std::string Page(iree_host_size_t& cursor,
                   loom_serve_observation_version_t& version) {
    iree_string_builder_reset(&output_);
    IREE_CHECK_OK(loom_serve_observation_write_page(observation_, &cursor,
                                                    &output_, &version));
    const auto view = iree_string_builder_view(&output_);
    return std::string(view.data, view.size);
  }

  // Real registry and observation owner; the service dependency supplies facts.
  loom_serve_registry_t* registry_ = nullptr;
  // Real borrowed telemetry lifetime, owned until registry retirement.
  loom_runtime_telemetry_stream_t* stream_ = nullptr;
  // Copied source cache independent of service storage and reader count.
  loom_serve_observation_t* observation_ = nullptr;
  // Borrowed application resources; service-only tests have no device binding.
  loom_serve_resources_t resources_ = {};
  // Allocation commands from the shared cache, excluding registry ownership.
  iree::vm::testing::CountingAllocator allocator_;
  // Caller-owned fixed wire-size formatting extent.
  char storage_[16368];
  // Reusable view over storage_, never a growable output.
  iree_string_builder_t output_ = {};
  // Service-owned common counters.
  loom_service_telemetry_counters_t counters_ = {};
  // Source-defined gauge, independent of common request counters.
  uint64_t resident_ = 4096;
  // Scheduler-owned current activity.
  iree_host_size_t active_ = 0;
  // Counts provider sampling, independently of observation-page reads.
  size_t samples_ = 0;
};

TEST_F(ObservationTest, LifetimeEpochIsCopiedAndIndependentOfReaderAdmission) {
  const auto since = loom_runtime_telemetry_stream_identity(stream_).created_ns;
  EXPECT_EQ(loom_serve_registry_counters_since_ns(registry_), since);
  counters_.requests = counters_.completed = 17;
  Sample(since + 1000000000);
  iree_host_size_t cursor = 0;
  loom_serve_observation_version_t version;
  const std::string epoch = "\"since_ns\":\"" + std::to_string(since) + "\"";
  EXPECT_NE(Page(cursor, version).find(epoch), std::string::npos);
  EXPECT_NE(Page(cursor, version).find(epoch), std::string::npos);
  Sample(since + 100000000000);
  cursor = 0;
  EXPECT_NE(Page(cursor, version).find(epoch), std::string::npos);
  EXPECT_NE(Page(cursor, version).find(epoch), std::string::npos);
}

TEST_F(ObservationTest, MissingModelEpochIsUnavailableNotSamplingTime) {
  loom_serve_registry_at(registry_, 0)->service.telemetry.stream = nullptr;
  Sample(100);
  iree_host_size_t cursor = 1;
  loom_serve_observation_version_t version;
  EXPECT_NE(Page(cursor, version).find("\"since_ns\":null"), std::string::npos);
}

TEST_F(ObservationTest, IdleSamplesRetainRevisionsAndAdvanceCaptureTime) {
  Sample(100);
  const uint64_t revision = loom_serve_observation_revision(observation_);
  EXPECT_EQ(samples_, 1u);
  iree_host_size_t cursor = 0;
  loom_serve_observation_version_t server, model;
  const auto server_page = Page(cursor, server);
  const auto model_page = Page(cursor, model);
  EXPECT_EQ(server.time_ns, 100);
  EXPECT_NE(server.revision, model.revision);
  for (iree_time_t now = 200; now <= 1000; now += 100) {
    Sample(now);
  }
  EXPECT_EQ(loom_serve_observation_revision(observation_), revision);
  cursor = 0;
  EXPECT_EQ(Page(cursor, server), server_page);
  EXPECT_EQ(Page(cursor, model), model_page);
  EXPECT_EQ(server.time_ns, 1000);
  EXPECT_EQ(model.time_ns, 1000);
  EXPECT_EQ(samples_, 10u);
}

TEST_F(ObservationTest, ReaderCountDoesNotMultiplyProviderSampling) {
  Sample(100);
  const auto allocations = allocator_.allocation_count();
  for (size_t reader = 0; reader < 64; ++reader) {
    iree_host_size_t cursor = 0;
    loom_serve_observation_version_t version;
    do {
      EXPECT_FALSE(Page(cursor, version).empty());
    } while (cursor);
  }
  EXPECT_EQ(samples_, 1u);
  for (iree_time_t now = 200; now < 10000; now += 100) {
    resident_ = 4096 + (now % 3);
    Sample(now);
  }
  EXPECT_EQ(allocator_.allocation_count(), allocations);
}

TEST_F(ObservationTest, ChangedSourceRetainsThePreviousUnchangedBoundary) {
  Sample(100);
  iree_host_size_t cursor = 1;
  loom_serve_observation_version_t before, after;
  Page(cursor, before);
  Sample(200);
  resident_ = 8192;
  Sample(300);
  cursor = 1;
  EXPECT_NE(Page(cursor, after).find("\"value\":\"8192\""), std::string::npos);
  EXPECT_NE(after.revision, before.revision);
  EXPECT_EQ(after.previous_revision, before.revision);
  EXPECT_EQ(after.previous_time_ns, 200);
  EXPECT_EQ(after.time_ns, 300);
  Sample(400);
  cursor = 1;
  Page(cursor, after);
  EXPECT_EQ(after.previous_time_ns, 200);
  EXPECT_EQ(after.time_ns, 400);
}

TEST_F(ObservationTest, CountersActivityAndAdmissionAreAllChangeSources) {
  Sample(100);
  auto revision = loom_serve_observation_revision(observation_);
  counters_.output_tokens = UINT64_MAX;
  Sample(200);
  EXPECT_GT(loom_serve_observation_revision(observation_), revision);
  revision = loom_serve_observation_revision(observation_);
  active_ = 1;
  Sample(300);
  EXPECT_GT(loom_serve_observation_revision(observation_), revision);
  revision = loom_serve_observation_revision(observation_);
  loom_serve_registry_at(registry_, 0)->closed = true;
  Sample(400);
  EXPECT_GT(loom_serve_observation_revision(observation_), revision);
  iree_host_size_t cursor = 1;
  loom_serve_observation_version_t version;
  const auto page = Page(cursor, version);
  EXPECT_NE(page.find("\"active\":1"), std::string::npos);
  EXPECT_NE(page.find("\"admission_open\":false"), std::string::npos);
  EXPECT_NE(page.find("\"output_tokens\":\"18446744073709551615\""),
            std::string::npos);
}

TEST_F(ObservationTest, EmptyRangesStayBoundedAndDoNotRepeatAFormerSource) {
  Sample(100);
  iree_host_size_t cursor = 1;
  loom_serve_observation_version_t version;
  EXPECT_NE(Page(cursor, version).find("\"name\":\"model\""),
            std::string::npos);
  EXPECT_EQ(cursor, 2u);
  EXPECT_EQ(Page(cursor, version),
            "{\"type\":\"models\",\"first\":1,\"next\":65,\"done\":false,"
            "\"model\":null}");
  EXPECT_EQ(version.revision, 0u);
  EXPECT_EQ(version.time_ns, 100);
  EXPECT_EQ(Page(cursor, version),
            "{\"type\":\"models\",\"first\":65,\"next\":129,\"done\":false,"
            "\"model\":null}");
  EXPECT_EQ(Page(cursor, version),
            "{\"type\":\"models\",\"first\":129,\"next\":131,\"done\":true,"
            "\"model\":null}");
  EXPECT_EQ(cursor, 0u);
  EXPECT_EQ(version.time_ns, 100);
}

TEST_F(ObservationTest, FailedFormattingDoesNotPublishAPartialSource) {
  Sample(100);
  iree_host_size_t cursor = 1;
  loom_serve_observation_version_t version;
  const auto before = Page(cursor, version);
  const auto revision = loom_serve_observation_revision(observation_);
  char tiny[8];
  iree_string_builder_t output;
  iree_string_builder_initialize_with_storage(tiny, sizeof(tiny), &output);
  cursor = 1;
  resident_ = 8192;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_observation_sample(observation_, 200, &cursor, &output));
  EXPECT_EQ(cursor, 1u);
  EXPECT_EQ(loom_serve_observation_revision(observation_), revision);
  EXPECT_EQ(Page(cursor, version), before);
  EXPECT_EQ(version.time_ns, 100);
  iree_string_builder_deinitialize(&output);
}

}  // namespace
