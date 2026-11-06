// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/resources.h"

#include <map>
#include <memory>
#include <string>

#include "iree/base/internal/json.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"
#include "loom/runtime/execution/execution.h"
#include "loom/serve/server/observation.h"

namespace {

class ResourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &output_);
  }
  void TearDown() override {
    iree_hal_buffer_release(storage_buffer_);
    iree_hal_buffer_release(workspace_buffer_);
    if (execution_) {
      IREE_EXPECT_OK(loom_runtime_execution_drain(execution_));
      loom_runtime_execution_release(execution_);
    }
    iree_string_builder_deinitialize(&output_);
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
    loom_runtime_host_release(host_);
  }
  std::string Write(const loom_serve_resources_t& resources) {
    iree_string_builder_reset(&output_);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&output_, &stream);
    IREE_EXPECT_OK(loom_serve_resources_write_json(&resources, &stream));
    auto value = iree_string_builder_view(&output_);
    return std::string(value.data, value.size);
  }
  iree_status_t CreateCpuGroup(
      loom_runtime_pool_backing_t backing = LOOM_RUNTIME_POOL_BACKING_FIXED,
      iree_string_view_t first_name = IREE_SV("first"),
      iree_string_view_t second_name = IREE_SV("second")) {
    IREE_RETURN_IF_ERROR(
        loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host_));
    const loom_runtime_device_options_t options[] = {
        {
            .driver = loom_runtime_host_task_driver(host_),
            .uri = IREE_SV("task"),
            .name = first_name,
        },
        {
            .driver = loom_runtime_host_task_driver(host_),
            .uri = IREE_SV("task"),
            .name = second_name,
        }};
    IREE_RETURN_IF_ERROR(loom_runtime_device_group_allocate(
        host_, IREE_ARRAYSIZE(options), options, iree_allocator_system(),
        &group_));
    const loom_runtime_pool_options_t pool_options = {
        .elastic = backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC,
        .weight_slab_size =
            backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC ? 65536u : 0u,
        .state_slab_size =
            backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC ? 65536u : 0u,
        .memory_limit =
            backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC ? 1048576u : 0u,
    };
    loom_runtime_pool_t* pool = nullptr;
    IREE_RETURN_IF_ERROR(loom_runtime_device_group_create_pool(
        group_, {first_name, 3}, &pool_options, &pool));
    const loom_runtime_pool_options_t fixed_options = {};
    return loom_runtime_device_group_create_pool(group_, {second_name, 2},
                                                 &fixed_options, &pool);
  }

  std::map<std::string, uint64_t> Observe(iree_host_size_t pool_index) {
    iree_string_builder_reset(&output_);
    const loom_serve_resources_t resources = {.device_group = group_};
    IREE_CHECK_OK(loom_serve_resources_write_observation(
        &resources, loom_runtime_device_group_count(group_) + pool_index,
        &output_));
    const auto json = iree_string_builder_view(&output_);
    iree_string_view_t topics;
    IREE_CHECK_OK(
        iree_json_lookup_object_value(json, IREE_SV("topics"), &topics));
    iree_host_size_t topic_count = 0;
    IREE_CHECK_OK(iree_json_array_length(topics, &topic_count));
    std::map<std::string, uint64_t> values;
    for (iree_host_size_t i = 0; i < topic_count; ++i) {
      iree_string_view_t topic, name, fields;
      IREE_CHECK_OK(iree_json_array_get(topics, i, &topic));
      IREE_CHECK_OK(
          iree_json_lookup_object_value(topic, IREE_SV("topic"), &name));
      IREE_CHECK_OK(
          iree_json_lookup_object_value(topic, IREE_SV("fields"), &fields));
      iree_host_size_t field_count = 0;
      IREE_CHECK_OK(iree_json_array_length(fields, &field_count));
      for (iree_host_size_t j = 0; j < field_count; ++j) {
        iree_string_view_t field, key, value;
        IREE_CHECK_OK(iree_json_array_get(fields, j, &field));
        IREE_CHECK_OK(
            iree_json_lookup_object_value(field, IREE_SV("name"), &key));
        IREE_CHECK_OK(
            iree_json_lookup_object_value(field, IREE_SV("value"), &value));
        EXPECT_EQ(value.data[-1], '"');
        EXPECT_EQ(value.data[value.size], '"');
        uint64_t number = 0;
        IREE_CHECK_OK(iree_json_parse_uint64(value, &number));
        EXPECT_TRUE(values
                        .emplace(std::string(name.data, name.size) + "." +
                                     std::string(key.data, key.size),
                                 number)
                        .second);
      }
    }
    return values;
  }

  // Application owner outliving all borrowed records.
  loom_runtime_host_t* host_ = nullptr;
  // One immutable group with independently identifiable logical devices.
  loom_runtime_device_group_t* group_ = nullptr;
  // Response storage detached from all resource owners.
  iree_string_builder_t output_;
  // Real model execution borrowing the first device's workspace.
  loom_runtime_execution_t* execution_ = nullptr;
  // Caller-owned fixed persistent storage, released before execution join.
  iree_hal_buffer_t* storage_buffer_ = nullptr;
  // Caller-owned transient view, released before execution join.
  iree_hal_buffer_t* workspace_buffer_ = nullptr;
};

TEST_F(ResourcesTest, NoPublishedResourcesDoesNotInventHardware) {
  EXPECT_EQ(Write({}), R"({"listeners":[],"devices":[],"pools":[]})");
  const loom_serve_resources_t resources = {};
  EXPECT_EQ(loom_serve_resources_observation_count(&resources), 0u);
}

TEST_F(ResourcesTest, FixedBackingObservesNativeStorageAndNestedWorkspace) {
  IREE_ASSERT_OK(CreateCpuGroup());
  auto* device = loom_runtime_device_group_at(group_, 0);
  auto* storage_owner = loom_runtime_device_group_pool_at(group_, 0);
  IREE_ASSERT_OK(
      loom_runtime_device_create_execution(device, storage_owner, &execution_));
  IREE_ASSERT_OK(
      loom_runtime_execution_reserve_workspace(execution_, 65536, 16));
  const auto request = iree_hal_make_pool_reservation_request(4096, 16);
  IREE_ASSERT_OK(loom_runtime_execution_alloca_from_pool(
      execution_, loom_runtime_pool_storage(storage_owner), 1, &request,
      &storage_buffer_));
  IREE_ASSERT_OK(loom_runtime_execution_alloca(execution_, 1, &request,
                                               &workspace_buffer_));
  IREE_ASSERT_OK(loom_runtime_execution_wait(
      execution_, loom_runtime_execution_frontier(execution_).values[0]));
  const auto values = Observe(0);
  EXPECT_EQ(values.size(), 34u);
  iree_hal_pool_stats_t native;
  iree_hal_pool_query_stats(loom_runtime_pool_storage(storage_owner), &native);
  const auto workspace = loom_runtime_pool_statistics(storage_owner).workspace;
  EXPECT_EQ(values.at("storage.committed_bytes"), native.bytes_committed);
  EXPECT_EQ(values.at("workspace.committed_bytes"), workspace.bytes_committed);
  EXPECT_GT(values.at("storage.committed_bytes"),
            values.at("workspace.committed_bytes"));
  EXPECT_GE(values.at("workspace.reserved_bytes"), 4096u);
  EXPECT_EQ(values.count("elastic.committed_bytes"), 0u);
  EXPECT_EQ(Observe(1).at("storage.committed_bytes"), 0u);
}

TEST_F(ResourcesTest, ElasticBackingHasAnIndependentBudgetAndStorageOwner) {
  IREE_ASSERT_OK(CreateCpuGroup(LOOM_RUNTIME_POOL_BACKING_ELASTIC));
  const auto values = Observe(0);
  EXPECT_EQ(values.size(), 43u);
  EXPECT_EQ(values.at("elastic.budget_limit"), 1048576u);
  EXPECT_EQ(values.at("elastic.committed_bytes"), 0u);
  EXPECT_EQ(values.at("storage.committed_bytes"), 0u);
  EXPECT_EQ(values.at("storage.budget_limit"), 0u);
  EXPECT_EQ(values.at("workspace.committed_bytes"), 0u);
  const auto fixed = Observe(1);
  EXPECT_EQ(fixed.count("elastic.budget_limit"), 0u);
}

TEST_F(ResourcesTest, TruncatedDisplayKeysDoNotMergePoolIdentities) {
  const std::string prefix(256, 'a');
  const std::string first = prefix + "first";
  const std::string second = prefix + "second";
  IREE_ASSERT_OK(
      CreateCpuGroup(LOOM_RUNTIME_POOL_BACKING_FIXED,
                     iree_make_string_view(first.data(), first.size()),
                     iree_make_string_view(second.data(), second.size())));
  for (iree_host_size_t index = 0; index < 2; ++index) {
    Observe(index);
    const auto json = iree_string_builder_view(&output_);
    iree_string_view_t key, ordinal;
    IREE_ASSERT_OK(iree_json_lookup_object_value(json, IREE_SV("key"), &key));
    EXPECT_EQ(std::string(key.data, key.size), prefix);
    bool truncated = false;
    IREE_ASSERT_OK(
        iree_json_lookup_bool(json, IREE_SV("key_truncated"), &truncated));
    EXPECT_TRUE(truncated);
    IREE_ASSERT_OK(
        iree_json_lookup_object_value(json, IREE_SV("index"), &ordinal));
    uint64_t value = 0;
    IREE_ASSERT_OK(iree_json_parse_uint64(ordinal, &value));
    EXPECT_EQ(value, index);
    EXPECT_LT(json.size, 16368u);
  }
}

TEST_F(ResourcesTest, DevicePagesHaveBoundedSamplingAndSharedCopiedStorage) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  EXPECT_EQ(loom_serve_resources_observation_count(&resources), 4u);
  loom_serve_registry_t* registry = nullptr;
  IREE_ASSERT_OK(loom_serve_registry_allocate(
      0, nullptr, 130, {}, iree_allocator_system(), &registry));
  auto free_registry = [](loom_serve_registry_t* value) {
    IREE_EXPECT_OK(loom_serve_registry_free(value));
  };
  std::unique_ptr<loom_serve_registry_t, decltype(free_registry)>
      registry_owner(registry, free_registry);
  iree::vm::testing::CountingAllocator allocator;
  loom_serve_observation_t* observation = nullptr;
  IREE_ASSERT_OK(loom_serve_observation_allocate(
      registry, &resources, allocator.allocator(), &observation));
  std::unique_ptr<loom_serve_observation_t,
                  decltype(&loom_serve_observation_free)>
      observation_owner(observation, loom_serve_observation_free);
  char storage[16368];
  iree_string_builder_t scratch;
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &scratch);
  iree_host_size_t cursor = 0;
  for (iree_time_t now = 100; now <= 500; now += 100) {
    IREE_ASSERT_OK(
        loom_serve_observation_sample(observation, now, &cursor, &scratch));
    EXPECT_EQ(cursor, static_cast<iree_host_size_t>(now / 100));
  }
  // Devices and pools each consumed separate scheduler steps. Empty model
  // ranges still advance by at most 64 slots per subsequent step.
  IREE_ASSERT_OK(
      loom_serve_observation_sample(observation, 600, &cursor, &scratch));
  EXPECT_EQ(cursor, 69u);
  while (cursor) {
    IREE_ASSERT_OK(
        loom_serve_observation_sample(observation, 600, &cursor, &scratch));
  }
  const auto allocations = allocator.allocation_count();
  const auto revision = loom_serve_observation_revision(observation);
  for (size_t reader = 0; reader < 64; ++reader) {
    cursor = 0;
    for (iree_host_size_t index = 0; index < 5; ++index) {
      iree_string_builder_reset(&scratch);
      loom_serve_observation_version_t version;
      IREE_ASSERT_OK(loom_serve_observation_write_page(observation, &cursor,
                                                       &scratch, &version));
      EXPECT_EQ(version.time_ns, static_cast<iree_time_t>((index + 1) * 100));
      iree_string_view_t identity;
      IREE_ASSERT_OK(iree_json_lookup_object_value(
          iree_string_builder_view(&scratch),
          index ? IREE_SV("key") : IREE_SV("type"), &identity));
      EXPECT_EQ(std::string(identity.data, identity.size), index == 0 ? "server"
                                                           : index % 2 == 1
                                                               ? "first"
                                                               : "second");
    }
  }
  cursor = 0;
  do {
    IREE_ASSERT_OK(
        loom_serve_observation_sample(observation, 700, &cursor, &scratch));
  } while (cursor);
  EXPECT_EQ(loom_serve_observation_revision(observation), revision);
  EXPECT_EQ(allocator.allocation_count(), allocations);
  // Failed formatting preserves the copied page and its capture boundary.
  char tiny[8];
  iree_string_builder_t small;
  iree_string_builder_initialize_with_storage(tiny, sizeof(tiny), &small);
  cursor = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_observation_sample(observation, 800, &cursor, &small));
  EXPECT_EQ(cursor, 1u);
  iree_string_builder_reset(&scratch);
  loom_serve_observation_version_t version;
  IREE_ASSERT_OK(loom_serve_observation_write_page(observation, &cursor,
                                                   &scratch, &version));
  EXPECT_EQ(version.time_ns, 700);
  iree_string_builder_deinitialize(&small);
  iree_string_builder_deinitialize(&scratch);
}

TEST_F(ResourcesTest, EveryRecordOfOneCpuGroupIsPublishedWithoutModelWork) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  EXPECT_EQ(
      Write(resources),
      R"({"listeners":[],"devices":[{"key":"first","id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}]},{"key":"second","id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}]}],"pools":[{"key":"first","devices":[0,1],"idle":{"workspace_ms":0,"parameter_ms":0},"memory":{"backing":"fixed","retained":null,"storage":{"reserved_bytes":0,"committed_bytes":0,"abandoned_bytes":0},"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0,"slab_count":0,"reserve_count":0,"release_count":0,"reuse_count":0,"reuse_miss_count":0,"fresh_count":0,"exhausted_count":0,"over_budget_count":0,"wait_count":0,"quarantine_eviction_count":0}}},{"key":"second","devices":[1],"idle":{"workspace_ms":0,"parameter_ms":0},"memory":{"backing":"fixed","retained":null,"storage":{"reserved_bytes":0,"committed_bytes":0,"abandoned_bytes":0},"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0,"slab_count":0,"reserve_count":0,"release_count":0,"reuse_count":0,"reuse_miss_count":0,"fresh_count":0,"exhausted_count":0,"over_budget_count":0,"wait_count":0,"quarantine_eviction_count":0}}}]})");
}

TEST_F(ResourcesTest, FixedStorageIsVisibleWithoutWorkspaceOrElasticBacking) {
  IREE_ASSERT_OK(CreateCpuGroup());
  auto* storage_owner = loom_runtime_device_group_pool_at(group_, 0);
  auto* peer = loom_runtime_device_group_pool_at(group_, 1);
  iree_hal_buffer_t* raw_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      loom_runtime_pool_storage(storage_owner),
      iree_hal_make_pool_reservation_request(4096, 256),
      iree_infinite_timeout(), &raw_buffer));
  std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)> buffer(
      raw_buffer, iree_hal_buffer_release);

  const auto memory = loom_runtime_pool_statistics(storage_owner);
  EXPECT_GE(memory.storage.bytes_reserved, 4096u);
  EXPECT_GE(memory.storage.bytes_committed, memory.storage.bytes_reserved);
  EXPECT_EQ(memory.storage.bytes_abandoned, 0u);
  EXPECT_EQ(memory.workspace.bytes_committed, 0u);
  EXPECT_EQ(memory.retained.committed_bytes, 0u);
  EXPECT_EQ(loom_runtime_pool_statistics(peer).storage.bytes_committed, 0u);
  const auto response = Write({.device_group = group_});
  EXPECT_NE(response.find(
                "\"retained\":null,\"storage\":{\"reserved_bytes\":" +
                std::to_string(memory.storage.bytes_reserved) +
                ",\"committed_bytes\":" +
                std::to_string(memory.storage.bytes_committed) +
                ",\"abandoned_bytes\":0},\"workspace\":{\"reserved_bytes\":0"),
            std::string::npos);

  buffer.reset();
  EXPECT_EQ(loom_runtime_pool_statistics(storage_owner).storage.bytes_reserved,
            0u);
}

TEST_F(ResourcesTest, TrimRequiresExplicitPoolAndIntegerTarget) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  for (const auto* value : {
           "",
           "[]",
           "null",
           "{}",
           R"({"pool":"first"})",
           R"({"target_bytes":0})",
           R"({"pool":0,"target_bytes":0})",
           R"({"pool":"first","target_bytes":null})",
           R"({"pool":true,"target_bytes":0})",
           R"({"pool":"first","target_bytes":[]})",
           R"({"pool":"first","target_bytes":0,"unknown":1})",
           R"({"pool":"first","pool":"second","target_bytes":0})",
           R"({"pool":"first","target_bytes":0,"target_bytes":1})",
           R"({"pool":0.0,"target_bytes":0})",
           R"({"pool":-1,"target_bytes":0})",
           R"({"pool":"first","target_bytes":-1})",
           R"({"pool":"first","target_bytes":0.5})",
           R"({"pool":"first","target_bytes":1e3})",
           R"({"pool":"first","target_bytes":0} trailing)",
       }) {
    SCOPED_TRACE(value);
    iree_host_size_t index = 99;
    uint64_t target = 99;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_resources_parse_trim(
                              &resources, iree_make_cstring_view(value),
                              iree_allocator_system(), &index, &target));
    EXPECT_EQ(index, 99u);
    EXPECT_EQ(target, 99u);
  }
}

TEST_F(ResourcesTest, TrimResolvesOnlyPublishedPoolNames) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t empty = {};
  const loom_serve_resources_t populated = {.device_group = group_};
  iree_host_size_t index = 99;
  uint64_t target = 99;
  for (const auto* resources : {&empty, &populated}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_NOT_FOUND,
        loom_serve_resources_parse_trim(
            resources, IREE_SV(R"({"pool":"absent","target_bytes":0})"),
            iree_allocator_system(), &index, &target));
    EXPECT_EQ(index, 99u);
    EXPECT_EQ(target, 99u);
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      loom_serve_resources_parse_trim(
          &populated,
          IREE_SV(R"({"pool":"first","target_bytes":18446744073709551616})"),
          iree_allocator_system(), &index, &target));
  EXPECT_EQ(index, 99u);
  EXPECT_EQ(target, 99u);
}

TEST_F(ResourcesTest, TrimReportsSelectedResourceWithoutModelOwnership) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  iree_host_size_t index = 0;
  uint64_t target = 0;
  IREE_ASSERT_OK(loom_serve_resources_parse_trim(
      &resources,
      IREE_SV(R"({"target_bytes":18446744073709551615,"pool":"sec\u006fnd"})"),
      iree_allocator_system(), &index, &target));
  EXPECT_EQ(index, 1u);
  EXPECT_EQ(target, UINT64_MAX);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&output_, &stream);
  IREE_ASSERT_OK(loom_serve_resources_trim(&resources, index, target, &stream));
  auto value = iree_string_builder_view(&output_);
  EXPECT_EQ(
      std::string(value.data, value.size),
      R"({"target_bytes":18446744073709551615,"pool":{"key":"second","devices":[1],"idle":{"workspace_ms":0,"parameter_ms":0},"memory":{"backing":"fixed","retained":null,"storage":{"reserved_bytes":0,"committed_bytes":0,"abandoned_bytes":0},"workspace":{"reserved_bytes":0,"committed_bytes":0,"free_bytes":0,"quarantined_bytes":0,"budget_limit":0,"reservation_count":0,"slab_count":0,"reserve_count":0,"release_count":0,"reuse_count":0,"reuse_miss_count":0,"fresh_count":0,"exhausted_count":0,"over_budget_count":0,"wait_count":0,"quarantine_eviction_count":0}}}})");
}

TEST_F(ResourcesTest, ProgramTrimRequiresOnlyAnExplicitDevice) {
  IREE_ASSERT_OK(CreateCpuGroup());
  const loom_serve_resources_t resources = {.device_group = group_};
  for (const auto* value : {
           "",
           "[]",
           "null",
           "{}",
           R"({"device":0})",
           R"({"device":""})",
           R"({"pool":"first"})",
           R"({"device":"first","target_bytes":0})",
           R"({"device":"first","device":"second"})",
           R"({"device":"first"} trailing)",
       }) {
    SCOPED_TRACE(value);
    iree_host_size_t index = 99;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_resources_parse_trim_programs(
                              &resources, iree_make_cstring_view(value),
                              iree_allocator_system(), &index));
    EXPECT_EQ(index, 99u);
  }
  iree_host_size_t index = 99;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_resources_parse_trim_programs(
                            &resources, IREE_SV(R"({"device":"absent"})"),
                            iree_allocator_system(), &index));
  EXPECT_EQ(index, 99u);
  IREE_ASSERT_OK(loom_serve_resources_parse_trim_programs(
      &resources, IREE_SV(R"({"device":"sec\u006fnd"})"),
      iree_allocator_system(), &index));
  EXPECT_EQ(index, 1u);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&output_, &stream);
  IREE_ASSERT_OK(
      loom_serve_resources_trim_programs(&resources, index, &stream));
  const auto value = iree_string_builder_view(&output_);
  EXPECT_EQ(
      std::string(value.data, value.size),
      R"({"device":{"key":"second","id":"task","name":"task","driver":"task","backend":"task","physical_devices":[{"ordinal":0,"name":"task","path":"task"}]}})");
}

}  // namespace
