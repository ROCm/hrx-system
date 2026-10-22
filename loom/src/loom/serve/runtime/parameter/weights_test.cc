// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/base/tooling/flags.h"
#include "loom/serve/runtime/parameter/weights_test_base.h"

IREE_FLAG(string, weight_sources, "", "Weight loader source catalog.");
IREE_FLAG(string, weight_device_uri, "", "Weight loader test device URI.");

namespace loom_serve::testing {

const char* WeightSourcePath() { return FLAG_weight_sources; }
const char* WeightDeviceUri() { return FLAG_weight_device_uri; }

namespace {

struct WeightDeviceGroup {
  // Immutable physical backing policy.
  bool elastic;
  // Requested slab bytes, zero for fixed backing.
  uint64_t slab_size;
  // Committed-byte budget, zero for fixed backing.
  uint64_t memory_limit;
  // Native allocator dependency outliving all resources allocated through it.
  NativeAllocator allocator;
  // Process-owned device group, shared across suites with matching options.
  loom_serve_device_group_t* group = nullptr;
  // Retained initialization result; a failed creation is never retried.
  iree_status_t status = iree_ok_status();
};

class WeightDeviceEnvironment : public ::testing::Environment {
 public:
  iree_status_t Acquire(bool elastic, uint64_t slab_size, uint64_t memory_limit,
                        loom_serve_device_group_t** out_group) {
    *out_group = nullptr;
    if (!elastic) {
      slab_size = 0;
      memory_limit = 0;
    }
    for (auto& entry : groups_) {
      if (entry->elastic == elastic && entry->slab_size == slab_size &&
          entry->memory_limit == memory_limit) {
        if (iree_status_is_ok(entry->status)) {
          *out_group = entry->group;
        }
        return iree_status_clone(entry->status);
      }
    }

    auto& entry = groups_.emplace_back(std::make_unique<WeightDeviceGroup>());
    entry->elastic = elastic;
    entry->slab_size = slab_size;
    entry->memory_limit = memory_limit;
    auto& status = entry->status;
    if (!host_) {
      status = loom_serve_test_host_create(1, iree_allocator_system(), &host_);
    }
    if (iree_status_is_ok(status) && !driver_) {
      status = loom_serve_test_driver_acquire(host_, &driver_);
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_device_options_t options = {
          .driver = driver_,
          .uri = WeightDeviceUri()[0]
                     ? iree_make_cstring_view(WeightDeviceUri())
                     : loom_serve_test_device()->device_uri,
          .backing = elastic ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                             : LOOM_SERVE_DEVICE_BACKING_FIXED,
          .weight_slab_size = slab_size,
          .state_slab_size = slab_size,
          .memory_limit = memory_limit,
      };
      status = loom_serve_device_group_allocate(
          host_, 1, &options, entry->allocator.allocator(), &entry->group);
    }
    if (iree_status_is_ok(status)) {
      status =
          entry->allocator.Observe(loom_serve_device_group_at(entry->group, 0));
    }
    if (iree_status_is_ok(status)) {
      *out_group = entry->group;
    }
    return iree_status_clone(status);
  }

  void TearDown() override {
    for (auto& entry : groups_) {
      IREE_EXPECT_OK(loom_serve_device_group_free(entry->group));
      entry->allocator.AwaitReleased();
      iree_status_free(entry->status);
    }
    groups_.clear();
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    loom_serve_host_release(host_);
    host_ = nullptr;
  }

 private:
  // Shared host services retained through every native device's final release.
  loom_serve_host_t* host_ = nullptr;
  // Retained driver supplying every immutable fixture configuration.
  iree_hal_driver_t* driver_ = nullptr;
  // Stable allocator storage for process-owned configurations.
  std::vector<std::unique_ptr<WeightDeviceGroup>> groups_;
};

auto* const weight_device_environment = static_cast<WeightDeviceEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new WeightDeviceEnvironment));

}  // namespace

iree_status_t AcquireWeightDeviceGroup(bool elastic, uint64_t slab_size,
                                       uint64_t memory_limit,
                                       loom_serve_device_group_t** out_group) {
  return weight_device_environment->Acquire(elastic, slab_size, memory_limit,
                                            out_group);
}

TEST_P(WeightsTest, ConfiguredPreparersAndSharedRootsPrepareEachTensorOnce) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  IREE_ASSERT_OK(Load(2, 3));
  // Three consumers and two configurations of the same preparation root.
  // Matching tensor policies reuse the recorded command; matching names with
  // distinct configurations must not reuse each other's values.
  EXPECT_EQ(compilation_.commands, 5u);
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

TEST_P(WeightsTest, SplitFilesPopulateOneSharedCheckpointDomain) {
  WriteCheckpoint(path_.path(), 0, 0, 1);
  WriteCheckpoint(adapter_path_.path(), 0, 1, 3);
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  const loom_serve_weight_root_t roots[] = {Root(0, 0), Root(1, 0), Root(2, 0),
                                            Root(2, 1)};
  IREE_ASSERT_OK(LoadRoots(2, IREE_ARRAYSIZE(roots), roots,
                           {path_.path(), adapter_path_.path()}));
  EXPECT_EQ(compilation_.commands, 5u);
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

TEST_P(WeightsTest, EmptyFirstFileDoesNotHideLaterParameters) {
  WriteCheckpoint(path_.path(), 0, 0, 0);
  WriteCheckpoint(adapter_path_.path(), 0);
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(LoadRoots(0, 1, &root, {path_.path(), adapter_path_.path()}));
  Run(0, 70);
}

TEST_P(WeightsTest, OneWorkerActivatesAndConsumesThroughAsyncTimelineWait) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_jit_stage_record(
      compiled_[0], iree_hal_queue_family(dispatch_),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, buffers_[0].data(),
      &commands_[0]));
  auto* host = loom_serve_device_host(device_owner_);
  loom_serve_test_weight_activation_t activation;
  activation.executor = loom_serve_host_executor(host);
  IREE_ASSERT_OK(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &activation.proactor));
  activation.residency = residencies_[0];
  activation.execution = execution_;
  activation.command = commands_[0];
  activation.output = output_buffer_;
  activation.readback = &output_;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &activation.gate));
  uint64_t held = 1;
  auto status =
      loom_serve_execution_barrier(execution_, {1, &activation.gate, &held});
  if (iree_status_is_ok(status)) {
    status = activation.Run();
  }
  iree_hal_semaphore_release(activation.gate);
  IREE_ASSERT_OK(status);
  ASSERT_TRUE(activation.admitted);
  for (int32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(output_[i], 70 + 3 * i);
  }
  // Readiness publication retains the pin. Pressure cannot evict the result's
  // parameters before the owning scheduler consumes completion.
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  DrainNativeReturns();
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  // The same compiled consumer survives a complete asynchronous reload.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
}

TEST_P(WeightsTest, CanonicalWeightsNeedNoPreparationOrLeadingSharedGroup) {
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Load(0, 1));
  Run(0, 90);
}

TEST_P(WeightsTest, PreparationPublishesOnlyRecordableRootAddresses) {
  IREE_ASSERT_OK(Compile(0, "raw"));
  const auto root = Root(0, 0);
  auto allocate = [&] {
    return loom_serve_weights_allocate(
        device_, transfer_, dispatch_,
        loom_serve_device_storage_pool(device_owner_), pool_,
        loom_serve_device_weight_geometry(device_owner_),
        loom_serve_device_workspace(device_owner_), 0, 1, &root,
        host_allocator_, &plans_[plan_count_++]);
  };
  if (!pool_) {
    // Allocation is accepted while its native dependency is held. Its later
    // failure belongs to preparation completion, before consumer recording.
    auto& allocator =
        loom_serve::testing::NativeAllocator::ForDevice(device_owner_);
    allocator.FailNextAllocation(IREE_ALLOCATOR_COMMAND_CALLOC,
                                 /*native_only=*/true);
    loom_serve::testing::ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(allocate());
    hold.WaitEntered();
  } else {
    IREE_ASSERT_OK(allocate());
  }
  auto status = Prepare(plans_[0], {path_.path()});
  if (!pool_) {
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    // Failed backing belongs to this plan; the shared compiler and execution
    // queues remain usable by another parameter owner.
    IREE_ASSERT_OK(Compile(1, "target"));
    const auto peer = Root(1, 0);
    IREE_ASSERT_OK(LoadRoots(0, 1, &peer, {path_.path()}));
    Run(1, 70);
  } else {
    // Elastic roots already have stable virtual addresses. Recording needs
    // no physical allocation, file read, or parameter activation.
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(loom_serve_jit_stage_record(
        compiled_[0], iree_hal_queue_family(dispatch_),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, buffers_[0].data(),
        &commands_[0]));
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 0u);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 0u);
  }
}

TEST_P(WeightsTest, SharedActivationImportsReadinessWithoutReloading) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &shared_execution_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &consumer_gate_));
  loom_serve_weight_activity_t activity = {};
  {
    // Elastic backing cannot finish while the actual native allocator is
    // held. Both consumers must inherit that dependency without host waits.
    loom_serve::testing::ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
    if (pool_) {
      hold.WaitEntered();
    }
    activity = loom_serve_weights_activity(plans_[0]);
    ASSERT_EQ(activity.activations, 1u);
    ASSERT_GT(activity.prepared_bytes, 0u);
    uint64_t gate_value = 1;
    IREE_ASSERT_OK(loom_serve_execution_barrier(
        execution_, {1, &consumer_gate_, &gate_value}));
    const auto producer = loom_serve_execution_frontier(execution_);
    const auto before = loom_serve_execution_frontier(shared_execution_);
    IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
    IREE_ASSERT_OK(
        loom_serve_weights_import_readiness(plans_[0], shared_execution_));
    const auto consumer = loom_serve_execution_frontier(shared_execution_);
    // Fail before issuing a consumer if no readiness dependency was imported;
    // executing against unbacked elastic roots would itself be invalid.
    ASSERT_GT(consumer.values[0], before.values[0]);
    EXPECT_EQ(loom_serve_execution_frontier(execution_).values[0],
              producer.values[0]);
    if (pool_) {
      uint64_t ready_value = 0;
      IREE_ASSERT_OK(
          iree_hal_semaphore_query(consumer.semaphores[0], &ready_value));
      EXPECT_LT(ready_value, consumer.values[0]);
    }
  }
  const auto shared = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(shared.activations, activity.activations);
  EXPECT_EQ(shared.read_bytes, activity.read_bytes);
  EXPECT_EQ(shared.prepared_bytes, activity.prepared_bytes);
  EXPECT_EQ(shared.groups, activity.groups);
  // The second consumer finishes before unrelated first-consumer work can
  // proceed. Importing that execution's frontier would serialize them.
  ASSERT_NO_FATAL_FAILURE(RunOn(shared_execution_, 0, 70));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(consumer_gate_, 1, nullptr));
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
}

TEST_P(WeightsTest, WarmResidencyDoesNotResubmitActivation) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  const auto before = loom_serve_execution_frontier(execution_);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  const auto after = loom_serve_execution_frontier(execution_);
  EXPECT_EQ(after.values[0], before.values[0]);
  EXPECT_EQ(after.values[1], before.values[1]);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  const auto next = loom_serve_execution_frontier(execution_);
  EXPECT_EQ(next.values[0], before.values[0]);
  EXPECT_EQ(next.values[1], before.values[1]);
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, SeparateCheckpointDomainsRetainSharedBaseStorage) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const loom_serve_weight_root_t base[] = {Root(1, 0), Root(0, 0)};
  IREE_ASSERT_OK(LoadRoots(2, 2, base, {path_.path()}));
  EXPECT_EQ(buffers_[0][0], buffers_[1][0]);
  EXPECT_EQ(buffers_[1][1], nullptr);
  Run(0, 90);

  const loom_serve_weight_root_t adapter = Root(1, 1);
  IREE_ASSERT_OK(LoadRoots(0, 1, &adapter, {adapter_path_.path()}));
  EXPECT_NE(buffers_[1][0], buffers_[1][1]);
  Run(1, 290);
  Run(1, 290);
  Run(0, 90);
}

TEST_P(WeightsTest, PolicySizeMismatchRejectsBeforeFileSubmission) {
  IREE_ASSERT_OK(Compile(0, "size"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Load(0, 1));
}

TEST_P(WeightsTest, MissingPreparationRootFailsCompilation) {
  WriteCheckpoint(path_.path(), 0, 4, 1);
  IREE_ASSERT_OK(Compile(0, "missing"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
}

TEST_P(WeightsTest, MultiplePreparationCommandsRejectBeforeCompilation) {
  IREE_ASSERT_OK(Compile(0, "size"));
  const auto root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, 1, &root, {path_.path()}, "boundaries.loom"));
  EXPECT_EQ(compilation_.commands, 1u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 0u);
}

TEST_P(WeightsTest, DirectPreparationStageRejectsBeforeCompilation) {
  IREE_ASSERT_OK(Compile(0, "missing"));
  const auto root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, 1, &root, {path_.path()}, "boundaries.loom"));
  EXPECT_EQ(compilation_.commands, 1u);
}

TEST_P(WeightsTest, PreparationCommandRequiresFinalExtent) {
  IREE_ASSERT_OK(Compile(0, "raw"));
  const auto root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, 1, &root, {path_.path()}, "boundaries.loom"));
  EXPECT_EQ(compilation_.commands, 1u);
}

TEST_P(WeightsTest, CanonicalPolicyRequiresZeroExtent) {
  IREE_ASSERT_OK(Compile(0, "unavailable"));
  const auto root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, 1, &root, {path_.path()}, "boundaries.loom"));
  EXPECT_EQ(compilation_.commands, 1u);
}

TEST_P(WeightsTest, WeightPolicyCannotIntroduceCheckpointDomains) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      CreateRoots(0, 1, &root, {path_.path()}, "boundaries.loom"));
  EXPECT_EQ(compilation_.commands, 1u);
}

TEST_P(WeightsTest, CheckpointIndexFailureLeavesTheSharedCompilerUsable) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto failed_root = Root(0, 0);
  IREE_ASSERT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      CreateRoots(0, 1, &failed_root,
                  {path_.path(), path_.path() + ".missing.safetensors"}));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).read_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  // The failed plan retains its own roots for normal owner-side cleanup while
  // another plan prepares and consumes through the same single-worker host.
  IREE_ASSERT_OK(Compile(1, "target"));
  const auto live_root = Root(1, 0);
  IREE_ASSERT_OK(LoadRoots(0, 1, &live_root, {path_.path()}));
  Run(1, 70);
}

TEST_P(WeightsTest, InitialBackingRetirementMatchesAcceptedWork) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  auto allocate_and_release_view = [&] {
    IREE_ASSERT_OK(loom_serve_weights_allocate(
        device_, transfer_, dispatch_,
        loom_serve_device_storage_pool(device_owner_), pool_,
        loom_serve_device_weight_geometry(device_owner_),
        loom_serve_device_workspace(device_owner_), 0, 1, &root,
        host_allocator_, &plans_[0]));
    iree_hal_buffer_release(buffers_[0][0]);
    buffers_[0][0] = nullptr;
  };
  if (!pool_) {
    loom_serve::testing::ScopedNativeHold hold(device_owner_);
    ASSERT_NO_FATAL_FAILURE(allocate_and_release_view());
    hold.WaitEntered();
    // No caller view remains, but accepted initial allocation still owns the
    // plan's pool and readiness state. Preparation failure must keep it alive.
    EXPECT_FALSE(loom_serve_weights_is_retired(plans_[0]));
  } else {
    // Elastic binding only reserves address space; it has no accepted native
    // backing operation until activation.
    ASSERT_NO_FATAL_FAILURE(allocate_and_release_view());
    EXPECT_TRUE(loom_serve_weights_is_retired(plans_[0]));
  }
  loom_serve_weights_await_retirement(plans_[0]);
  EXPECT_TRUE(loom_serve_weights_is_retired(plans_[0]));
}

TEST_P(WeightsTest, MissingFileParameterFailsTheLoad) {
  IREE_ASSERT_OK(Compile(0, "unavailable"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
}

TEST_P(WeightsTest, LateMissingParameterRejectsBeforeAnyActivationWork) {
  // A missing key after valid prepared tensors rejects the whole source plan,
  // before preparer compilation, reads, or physical activation begin.
  IREE_ASSERT_OK(Compile(0, "partial"));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND, Load(0, 1));
  EXPECT_EQ(compilation_.commands, 1u);
  const auto activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(activity.activations, 0u);
  EXPECT_EQ(activity.read_bytes, 0u);
  EXPECT_EQ(activity.prepared_bytes, 0u);
  EXPECT_EQ(activity.groups, 0u);
  IREE_ASSERT_OK(Compile(1, "target"));
  const auto live_root = Root(1, 0);
  IREE_ASSERT_OK(LoadRoots(0, 1, &live_root, {path_.path()}));
  Run(1, 70);
}

TEST_P(WeightsTest,
       ShortDirectInputsRejectBeforeUnchangedOrInPlacePreparation) {
  WriteCheckpoint(path_.path(), 0, 0, 4, 4);
  WriteCheckpoint(adapter_path_.path(), 0, 0, 4, 16);
  const char* roots[] = {"raw", "target"};
  for (size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
    IREE_ASSERT_OK(Compile(i, roots[i]));
    const auto root = Root(i, 0);
    iree::Status status(CreateRoots(0, 1, &root, {path_.path()}));
    EXPECT_THAT(status,
                iree::testing::status::StatusIs(iree::StatusCode::kOutOfRange));
    EXPECT_THAT(status.ToString(), ::testing::HasSubstr("requires 32 bytes"));
    EXPECT_THAT(status.ToString(), ::testing::HasSubstr("entry has 16"));
    EXPECT_EQ(compilation_.commands, i + 1);
    const auto activity = loom_serve_weights_activity(plans_[i]);
    EXPECT_EQ(activity.activations, 0u);
    EXPECT_EQ(activity.read_bytes, 0u);
    EXPECT_EQ(activity.prepared_bytes, 0u);
    EXPECT_EQ(activity.groups, 0u);
  }
  // The invalid models leave their shared device, worker, and peer timeline
  // usable. The valid peer reads only its requested prefix of each larger
  // entry.
  IREE_ASSERT_OK(Compile(2, "target"));
  const auto live_root = Root(2, 0);
  IREE_ASSERT_OK(LoadRoots(0, 1, &live_root, {adapter_path_.path()}));
  Run(2, 70);
}

TEST_P(WeightsTest, SharedWorkingSetsAdmitAtomicallyAndEvictOnlyIdleGroups) {
  auto* cache = loom_serve_device_residency_cache(device_owner_);
  WriteCheckpoint(adapter_path_.path(), 100);
  const char* stages[] = {"target", "raw", "raw"};
  for (iree_host_size_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, stages[i]));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root,
                               {i == 2 ? adapter_path_.path() : path_.path()}));
    EXPECT_EQ(
        loom_serve_weights_activation_workspace(plans_[i]).allocation_size, 0u);
    IREE_ASSERT_OK(loom_serve_residency_group_allocate(
        cache, 1, &plans_[i], host_allocator_, &parameter_groups_[i]));
  }
  IREE_ASSERT_OK(
      loom_serve_device_create_execution(device_owner_, &shared_execution_));
  loom_serve_residency_group_t* first[] = {parameter_groups_[0],
                                           parameter_groups_[1]};
  loom_serve_residency_group_t* second[] = {parameter_groups_[0],
                                            parameter_groups_[2]};
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      cache, execution_, 2, first, host_allocator_, &residencies_[0]));
  IREE_ASSERT_OK(loom_serve_residency_allocate(
      cache, shared_execution_, 2, second, host_allocator_, &residencies_[1]));
  IREE_ASSERT_OK(
      loom_serve_residency_allocate(cache, execution_, 1, &parameter_groups_[0],
                                    host_allocator_, &residencies_[2]));
  {
    loom_serve::testing::ScopedNativeHold hold(device_owner_);
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[2]));
    if (pool_) {
      hold.WaitEntered();
    }
    // The base's accepted backing is already charged even before native
    // completion. Adding the companion must not charge that base twice.
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  }
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  ASSERT_NO_FATAL_FAILURE(Run(1, 90));
  if (!pool_) {
    IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[1]));
    ASSERT_NO_FATAL_FAILURE(RunOn(shared_execution_, 0, 70));
    ASSERT_NO_FATAL_FAILURE(RunOn(shared_execution_, 2, 390));
    loom_serve_residency_release(residencies_[1]);
    loom_serve_residency_release(residencies_[2]);
    loom_serve_residency_release(residencies_[0]);
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
    return;
  }

  // The entire second set fits in the pool, but not alongside the live first
  // set. Denial starts no load and retains no partial working-set protection.
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[1]).pin_count, 0u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[2]).activations, 0u);
  loom_serve_residency_cancel_admission(residencies_[1]);
  EXPECT_FALSE(
      loom_serve_residency_snapshot(residencies_[1]).admission.waiting);
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(1));
  ASSERT_NO_FATAL_FAILURE(RunOn(shared_execution_, 0, 70));
  ASSERT_NO_FATAL_FAILURE(RunOn(shared_execution_, 2, 390));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).deactivations, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).committed_bytes, 0u);

  // One consumer retiring cannot unpin the base from the other working set.
  loom_serve_residency_release(residencies_[2]);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  bool complete = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_residency_deactivate(residencies_[0], &complete));
  loom_serve_residency_release(residencies_[1]);

  // Persistent retention is counted across borrowers, not last-writer-wins.
  loom_serve_residency_set_pinned(residencies_[0], true);
  loom_serve_residency_set_pinned(residencies_[2], true);
  loom_serve_residency_set_pinned(residencies_[0], false);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  DrainNativeReturns();
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).cached_bytes, kSlabBytes);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            kSlabBytes);
  loom_serve_residency_set_pinned(residencies_[2], false);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  DrainNativeReturns();
  EXPECT_FALSE(loom_serve_residency_snapshot(residencies_[0]).active);
  EXPECT_FALSE(loom_serve_residency_snapshot(residencies_[1]).active);
  EXPECT_FALSE(loom_serve_residency_snapshot(residencies_[2]).active);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).cached_bytes,
            2 * kSlabBytes);

  // The same recordings import newly prepared generations after eviction.
  const auto* command = commands_[0];
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  ASSERT_NO_FATAL_FAILURE(Run(1, 90));
  EXPECT_EQ(commands_[0], command);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 2u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).peak_bytes,
            2 * kSlabBytes);
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, CachedConsumersSurviveBackingReuseAndPreparationReplay) {
  IREE_ASSERT_OK(Compile(0, "target"));
  IREE_ASSERT_OK(Compile(1, "target"));
  IREE_ASSERT_OK(Compile(2, "auxiliary"));
  IREE_ASSERT_OK(Load(2, 3));
  const auto first_activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(first_activity.activations, 1u);
  EXPECT_EQ(first_activity.deactivations, 0u);
  EXPECT_GT(first_activity.read_bytes, 0u);
  EXPECT_GT(first_activity.prepared_bytes, 0u);
  EXPECT_GT(first_activity.groups, 0u);
  Run(0, 70);
  Run(2, 93);
  auto* const root = buffers_[0][0];
  auto* const command = commands_[0];
  bool complete = false;
  if (!GetParam()) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          loom_serve_weights_deactivate(plans_[0], &complete));
    Run(0, 70);
    return;
  }
  IREE_ASSERT_OK(loom_serve_weights_deactivate(plans_[0], &complete));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_weights_deactivate(plans_[0], &complete));
  ASSERT_TRUE(complete);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).cached_bytes,
            2 * kSlabBytes);
  loom_serve_virtual_buffer_t* other = nullptr;
  loom_serve_memory_statistics_t other_statistics = {};
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), 2 * kSlabBytes,
      256, &other_statistics, &other));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(other, 0, 2 * kSlabBytes));
  // Another allocation consumes every physical byte the model gave up.
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  const uint32_t poison = 0xDEADBEEF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = loom_serve_virtual_buffer_handle(other),
              .length = 2 * kSlabBytes,
              .pattern = &poison,
              .pattern_length = sizeof(poison),
          },
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &fill, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_free(other));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_ASSERT_OK(loom_serve_weights_activate(plans_[0]));
  IREE_ASSERT_OK(loom_serve_weights_import_readiness(plans_[0], execution_));
  const auto replay_activity = loom_serve_weights_activity(plans_[0]);
  EXPECT_EQ(replay_activity.activations, 2u);
  EXPECT_EQ(replay_activity.deactivations, 1u);
  EXPECT_EQ(replay_activity.read_bytes, first_activity.read_bytes * 2);
  EXPECT_EQ(replay_activity.prepared_bytes, first_activity.prepared_bytes * 2);
  EXPECT_EQ(replay_activity.groups, first_activity.groups * 2);
  EXPECT_EQ(buffers_[0][0], root);
  EXPECT_EQ(commands_[0], command);
  ExpectBacking(0, 2 * kSlabBytes);
  Run(0, 70);
  Run(1, 70);
  Run(2, 93);
}

TEST_P(WeightsTest, AdmissionUsesRecencyOnlyAmongUnpinnedGroups) {
  bool complete = false;
  for (iree_host_size_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
  }
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  // Touch the oldest entry, making the other resident entry the LRU victim.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  ASSERT_NO_FATAL_FAILURE(RunPinned(2, 70));
  if (!GetParam()) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        loom_serve_residency_deactivate(residencies_[0], &complete));
    ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
    return;
  }
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            kSlabBytes);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).committed_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).committed_bytes,
            kSlabBytes);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_serve_residency_deactivate(residencies_[0], &complete));
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).cached_bytes, kSlabBytes);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).peak_bytes,
            2 * kSlabBytes);
}

TEST_P(WeightsTest, ImpossibleParameterGroupCannotJoinPressureAdmission) {
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "raw"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  }
  if (!GetParam()) {
    IREE_ASSERT_OK(Register(0, 0, 3));
    return;
  }
  EXPECT_EQ(loom_serve_memory_pool_limit(pool_), 2 * kSlabBytes);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Register(0, 0, 3));
  EXPECT_EQ(residencies_[0], nullptr);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes, 0u);
  // Rejection does not poison the shared admission queue or borrowed plans.
  IREE_ASSERT_OK(Register(0, 0, 1));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, PersistentPinningOnlyChangesParameterEvictionEligibility) {
  bool complete = false;
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  auto* residency = residencies_[0];
  auto* cache = ResidencyCache();
  loom_serve_residency_set_pinned(residency, true);
  loom_serve_residency_set_pinned(residency, true);
  EXPECT_FALSE(loom_serve_residency_snapshot(residency).active);
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 0u);
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            GetParam() ? kSlabBytes : 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_serve_residency_deactivate(residency, &complete));
  loom_serve_residency_set_pinned(residency, false);
  loom_serve_residency_set_pinned(residency, false);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency).active);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, 0));
  EXPECT_EQ(loom_serve_residency_snapshot(residency).active, !GetParam());
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            GetParam() ? 2u : 1u);
}

TEST_P(WeightsTest, ExplicitActivationCancellationJoinsAcceptedReadiness) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(execution_, {1, &gate, &gate_value}));
  loom_serve_test_weight_activation_wake_t wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_allocate(
      residencies_[0],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      loom_serve_test_weight_activation_wake_t::Notify, &wake, host_allocator_,
      &activation));
  loom_serve_activation_phase_t phase;
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
  // Removing administrative retention cannot release accepted native work.
  loom_serve_residency_set_pinned(residencies_[0], true);
  loom_serve_residency_set_pinned(residencies_[0], false);
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_LOADING);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  ExpectBacking(0, GetParam() ? kSlabBytes : 0u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value, nullptr));
  wake.Await();
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_free(activation);
  iree_hal_semaphore_release(gate);
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 1u);
  // Warmup owns no persistent pin after completion.
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  DrainNativeReturns();
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
}

TEST_P(WeightsTest, CancelledWarmupWithdrawsCapacityInterestWithoutLoading) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  if (!GetParam()) {
    // Fixed allocations have no serving-imposed physical ceiling.
    return;
  }
  for (uint32_t i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(Compile(i, "raw"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  }
  IREE_ASSERT_OK(Register(0, 0, 2));
  IREE_ASSERT_OK(Register(1, 2, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[0]));
  loom_serve_test_weight_activation_wake_t wake;
  loom_serve_activation_t* activation = nullptr;
  IREE_ASSERT_OK(loom_serve_activation_allocate(
      residencies_[1],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      loom_serve_test_weight_activation_wake_t::Notify, &wake, host_allocator_,
      &activation));
  loom_serve_activation_phase_t phase;
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_WAITING);
  EXPECT_EQ(loom_serve_weights_activity(plans_[2]).activations, 0u);
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase));
  EXPECT_EQ(phase, LOOM_SERVE_ACTIVATION_COMPLETE);
  loom_serve_activation_free(activation);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_cancel_admission(residencies_[0]);
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_release(residencies_[0]);
  // A new operation can now reclaim the unpinned peer and finish normally.
  loom_serve_memory_pool_set_observer(
      pool_, loom_serve_test_weight_activation_wake_t::Notify, &wake);
  IREE_ASSERT_OK(loom_serve_activation_allocate(
      residencies_[1],
      loom_serve_host_proactor_pool(loom_serve_device_host(device_owner_)),
      loom_serve_test_weight_activation_wake_t::Notify, &wake, host_allocator_,
      &activation));
  IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  while (phase != LOOM_SERVE_ACTIVATION_COMPLETE) {
    wake.Await();
    IREE_ASSERT_OK(loom_serve_activation_advance(activation, 0, &phase));
  }
  loom_serve_memory_pool_set_observer(pool_, nullptr, nullptr);
  loom_serve_activation_free(activation);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[1]));
  ASSERT_NO_FATAL_FAILURE(Run(2, 90));
  loom_serve_residency_release(residencies_[1]);
}

TEST_P(WeightsTest, CheckpointGroupsAdmitAtomicallyAndRetentionPinsNest) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const auto first = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &first, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(1, i);
    IREE_ASSERT_OK(
        CreateRoots(0, 1, &root, {i ? adapter_path_.path() : path_.path()}));
  }
  IREE_ASSERT_OK(Register(1, 1, 2));
  auto check_activity = [&](uint64_t activations, uint64_t deactivations) {
    const auto total = loom_serve_residency_activity(residencies_[1]);
    const auto base = loom_serve_weights_activity(plans_[1]);
    const auto adapter = loom_serve_weights_activity(plans_[2]);
    EXPECT_EQ(total.activations, activations);
    EXPECT_EQ(total.deactivations, deactivations);
    EXPECT_EQ(total.read_bytes, base.read_bytes + adapter.read_bytes);
    EXPECT_EQ(total.prepared_bytes,
              base.prepared_bytes + adapter.prepared_bytes);
    EXPECT_EQ(total.groups, base.groups + adapter.groups);
  };
  check_activity(0, 0);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 90));
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(residencies_[1], &admitted));
  if (!GetParam()) {
    ASSERT_TRUE(admitted);
    check_activity(2, 0);
    ASSERT_NO_FATAL_FAILURE(Run(1, 290));
    loom_serve_residency_release(residencies_[1]);
    loom_serve_residency_release(residencies_[0]);
    return;
  }
  EXPECT_FALSE(admitted);
  check_activity(0, 0);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).peak_bytes, 0u);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[2]).peak_bytes, 0u);
  // Failed admission neither consumes its own pin nor releases retention.
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes,
            kSlabBytes);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[1]));
  check_activity(2, 0);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  ASSERT_NO_FATAL_FAILURE(Run(1, 290));
  loom_serve_residency_release(residencies_[1]);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  DrainNativeReturns();
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).cached_bytes,
            2 * kSlabBytes);
  check_activity(2, 2);
}

TEST_P(WeightsTest, PressureAdmissionDrainsNewPinsButNotAcceptedConsumers) {
  WriteCheckpoint(adapter_path_.path(), 100);
  IREE_ASSERT_OK(Compile(0, "raw"));
  IREE_ASSERT_OK(Compile(1, "domains"));
  const auto first = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &first, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(1, i);
    IREE_ASSERT_OK(
        CreateRoots(0, 1, &root, {i ? adapter_path_.path() : path_.path()}));
  }
  IREE_ASSERT_OK(Register(1, 1, 2));
  IREE_ASSERT_OK(Register(2, 0, 0));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  const auto initial = loom_serve_residency_snapshot(residencies_[0]);
  EXPECT_EQ(initial.pin_count, 1u);
  EXPECT_FALSE(initial.admission.waiting);
  EXPECT_EQ(initial.admission.count, 0u);
  EXPECT_EQ(initial.admission.duration_ns, 0u);
  EXPECT_EQ(initial.admission.age_ns, 0u);
  const iree_time_t first_wait_start = iree_time_now();
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  if (!GetParam()) {
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    const auto fixed = loom_serve_residency_snapshot(residencies_[1]);
    EXPECT_EQ(fixed.pin_count, 1u);
    EXPECT_FALSE(fixed.admission.waiting);
    EXPECT_EQ(fixed.admission.count, 0u);
    EXPECT_EQ(fixed.admission.duration_ns, 0u);
    ASSERT_NO_FATAL_FAILURE(Run(1, 290));
    loom_serve_residency_release(residencies_[1]);
    loom_serve_residency_release(residencies_[0]);
    return;
  }
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  const auto waiting = loom_serve_residency_snapshot(residencies_[1]);
  EXPECT_EQ(waiting.pin_count, 0u);
  EXPECT_TRUE(waiting.admission.waiting);
  EXPECT_EQ(waiting.admission.count, 1u);
  EXPECT_EQ(waiting.admission.duration_ns, waiting.admission.age_ns);
  EXPECT_LE(waiting.admission.age_ns, iree_time_now() - first_wait_start);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  // The already-admitted consumer still takes its nested execution pin and
  // produces exact device output. A parameter-free peer cannot obstruct it.
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 90));
  const auto retained = loom_serve_residency_snapshot(residencies_[0]);
  EXPECT_TRUE(retained.admission.waiting);
  EXPECT_EQ(retained.admission.count, 1u);
  EXPECT_EQ(retained.pin_count, 1u);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[2], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[2]).admission.count, 0u);
  loom_serve_residency_release(residencies_[2]);
  // Withdraw both tail and head, including repeated withdrawal. No owned
  // invocation pin is released by cancellation of a new request.
  loom_serve_residency_cancel_admission(residencies_[0]);
  const auto withdrawn = loom_serve_residency_snapshot(residencies_[0]);
  EXPECT_FALSE(withdrawn.admission.waiting);
  EXPECT_EQ(withdrawn.admission.count, 1u);
  EXPECT_EQ(withdrawn.admission.age_ns, 0u);
  EXPECT_GE(withdrawn.admission.duration_ns, retained.admission.duration_ns);
  EXPECT_EQ(withdrawn.pin_count, 1u);
  loom_serve_residency_cancel_admission(residencies_[0]);
  EXPECT_EQ(
      loom_serve_residency_snapshot(residencies_[0]).admission.duration_ns,
      withdrawn.admission.duration_ns);
  loom_serve_residency_cancel_admission(residencies_[1]);
  const auto first_wait = loom_serve_residency_snapshot(residencies_[1]);
  EXPECT_FALSE(first_wait.admission.waiting);
  EXPECT_EQ(first_wait.admission.count, 1u);
  EXPECT_EQ(first_wait.admission.age_ns, 0u);
  EXPECT_GE(first_wait.admission.duration_ns, waiting.admission.duration_ns);
  EXPECT_LE(first_wait.admission.duration_ns,
            iree_time_now() - first_wait_start);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(1));
  loom_serve_residency_cancel_admission(residencies_[1]);
  const auto admitted = loom_serve_residency_snapshot(residencies_[1]);
  EXPECT_FALSE(admitted.admission.waiting);
  EXPECT_EQ(admitted.admission.count, 2u);
  EXPECT_EQ(admitted.admission.age_ns, 0u);
  EXPECT_GE(admitted.admission.duration_ns, first_wait.admission.duration_ns);
  EXPECT_EQ(admitted.pin_count, 1u);
  ASSERT_NO_FATAL_FAILURE(Run(1, 290));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[0]).admission.count, 2u);
  loom_serve_residency_release(residencies_[1]);
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(0));
  loom_serve_residency_cancel_admission(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[0]).pin_count, 0u);
  EXPECT_EQ(
      loom_serve_residency_snapshot(residencies_[1]).admission.duration_ns,
      admitted.admission.duration_ns);
}

TEST_P(WeightsTest, PressureTurnSurvivesMutableStateNativeReturn) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  if (!GetParam()) {
    return;
  }
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
  }
  IREE_ASSERT_OK(loom_serve_residency_acquire_wait(residencies_[0]));
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), 2 * kSlabBytes,
      256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  loom_serve_residency_release(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(1));
  IREE_ASSERT_OK(loom_serve_residency_await(residencies_[1]));

  // Detach first so the explicit physical trim holds native disposal rather
  // than the return of a mapping's metadata. The new row cannot claim that
  // capacity until the source has actually disposed of the old backing.
  const loom_serve_memory_range_t next = {state_, kSlabBytes, kSlabBytes};
  bool admitted = false;
  loom_serve_virtual_buffer_begin_trim(state_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  {
    if (deferred_backing_) {
      deferred_backing_->DeferReturns();
    }
    loom_serve_memory_pool_trim(pool_, 0);
    deferred_backing_->WaitForReturn();
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).releasing_bytes,
              kSlabBytes);
    IREE_ASSERT_OK(
        loom_serve_memory_pool_try_commit(pool_, 1, &next, &admitted));
    EXPECT_FALSE(admitted);
    // No row was admitted, and activation has retired. Releasing that pin
    // must not surrender this request's turn to the younger parameter waiter.
    loom_serve_residency_release(residencies_[1]);
    EXPECT_TRUE(
        loom_serve_residency_snapshot(residencies_[1]).admission.waiting);
    IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
    EXPECT_TRUE(loom_serve_residency_snapshot(residencies_[1]).active);
    EXPECT_EQ(loom_serve_weights_activity(plans_[1]).deactivations, 0u);
    deferred_backing_->CompleteReturns();
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[1], &admission));
  ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool_, 1, &next, &admitted));
  ASSERT_TRUE(admitted);
  loom_serve_residency_cancel_admission(residencies_[1]);
  const auto completed = loom_serve_residency_snapshot(residencies_[1]);
  EXPECT_FALSE(completed.admission.waiting);
  EXPECT_EQ(completed.admission.count, 1u);
  EXPECT_EQ(completed.pin_count, 1u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[1]).activations, 1u);
  ASSERT_NO_FATAL_FAILURE(Run(1, 70));
  loom_serve_residency_release(residencies_[1]);
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(0));
  loom_serve_residency_cancel_admission(residencies_[0]);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).peak_bytes,
            2 * kSlabBytes);
}

TEST_P(WeightsTest, NewAdmissionsCoexistWithoutPhysicalPressure) {
  loom_serve_residency_admission_t admission;
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
    IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[i], &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  }
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[i], &admission));
    ASSERT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
    ASSERT_NO_FATAL_FAILURE(Run(i, 70));
    const auto active = loom_serve_residency_snapshot(residencies_[i]);
    EXPECT_EQ(active.pin_count, 2u);
    EXPECT_FALSE(active.admission.waiting);
    EXPECT_EQ(active.admission.count, 0u);
    EXPECT_EQ(active.admission.duration_ns, 0u);
    EXPECT_EQ(active.admission.age_ns, 0u);
    loom_serve_residency_release(residencies_[i]);
    loom_serve_residency_release(residencies_[i]);
  }
}

TEST_P(WeightsTest, WarmAdmissionCollectsCompletedPeerRelease) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  if (!GetParam()) {
    return;
  }
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
    ASSERT_NO_FATAL_FAILURE(RunPinned(i, 70));
  }
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), 2 * kSlabBytes,
      256, &state_statistics_, &state_));
  auto* cache = ResidencyCache();
  bool admitted = false;
  const loom_serve_memory_range_t first = {state_, 0, kSlabBytes};
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(cache, kSlabBytes));
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  {
    if (deferred_backing_) {
      deferred_backing_->DeferReturns();
    }
    loom_serve_memory_pool_trim(pool_, 0);
    deferred_backing_->WaitForReturn();
    IREE_ASSERT_OK(
        loom_serve_memory_pool_try_commit(pool_, 1, &first, &admitted));
    EXPECT_FALSE(admitted);
    EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(cache));
    // Native backing can outlive the weights' logical return. The pool owns
    // those bytes while the cache retains the uncollected release interest.
    EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).releasing_bytes,
              kSlabBytes);
    deferred_backing_->CompleteReturns();
  }
  IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &first, &admitted));
  ASSERT_TRUE(admitted);
  // A warm admission must retire the peer's logical release interest too.
  // The following state denial has no pending work that could wake a retry.
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED);
  const loom_serve_memory_range_t both = {state_, 0, 2 * kSlabBytes};
  IREE_ASSERT_OK(loom_serve_memory_pool_try_commit(pool_, 1, &both, &admitted));
  EXPECT_FALSE(admitted);
  EXPECT_FALSE(loom_serve_memory_pool_is_pending(pool_));
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  IREE_ASSERT_OK(loom_serve_residency_cache_query_status(cache));
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[0]).pin_count, 2u);
  EXPECT_EQ(loom_serve_weights_activity(plans_[1]).deactivations, 1u);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_release(residencies_[0]);
}

TEST_P(WeightsTest, IdleExpiryPreservesPinsAndReloadsWithoutRecompiling) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
    ASSERT_NO_FATAL_FAILURE(RunPinned(i, 70));
  }
  auto* cache = ResidencyCache();
  const auto timeout = iree_make_duration_ms(3600000);
  loom_serve_residency_set_pinned(residencies_[0], true);
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[1]));
  bool pending = true;
  IREE_ASSERT_OK(loom_serve_residency_cache_trim_idle(
      cache, iree_time_now() + 2 * timeout, timeout, &pending));
  EXPECT_FALSE(pending);
  EXPECT_TRUE(loom_serve_residency_snapshot(residencies_[0]).active);
  EXPECT_TRUE(loom_serve_residency_snapshot(residencies_[1]).active);
  loom_serve_residency_set_pinned(residencies_[0], false);
  const auto unpinned = iree_time_now();
  IREE_ASSERT_OK(
      loom_serve_residency_cache_trim_idle(cache, unpinned, timeout, &pending));
  EXPECT_EQ(pending, GetParam());
  IREE_ASSERT_OK(loom_serve_residency_cache_trim_idle(
      cache, unpinned + timeout / 2, timeout, &pending));
  EXPECT_EQ(pending, GetParam());
  IREE_ASSERT_OK(loom_serve_residency_cache_trim_idle(cache, unpinned + timeout,
                                                      timeout, &pending));
  EXPECT_FALSE(pending);
  EXPECT_EQ(loom_serve_residency_snapshot(residencies_[0]).active, !GetParam());
  EXPECT_TRUE(loom_serve_residency_snapshot(residencies_[1]).active);
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  }
  {
    if (deferred_backing_) {
      deferred_backing_->DeferReturns();
    }
    if (GetParam()) {
      loom_serve_memory_pool_trim(pool_, 0);
      deferred_backing_->WaitForReturn();
      EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).releasing_bytes,
                kSlabBytes);
    }
    // The independent consumer has no dependency on the held native release.
    ASSERT_NO_FATAL_FAILURE(Run(1, 70));
    if (deferred_backing_) {
      deferred_backing_->CompleteReturns();
    }
  }
  if (pool_) {
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
  }
  IREE_ASSERT_OK(loom_serve_residency_cache_query_status(cache));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations,
            GetParam() ? 2u : 1u);
  loom_serve_residency_release(residencies_[1]);
  const auto released = iree_time_now();
  IREE_ASSERT_OK(
      loom_serve_residency_cache_trim_idle(cache, released, timeout, &pending));
  EXPECT_EQ(pending, GetParam());
  EXPECT_TRUE(loom_serve_residency_snapshot(residencies_[1]).active);
}

TEST_P(WeightsTest, PinnedWeightsDeferIdleCacheReclamationUntilItCanHelp) {
  if (!GetParam()) {
    return;
  }
  IREE_ASSERT_OK(Compile(0, "domains"));
  for (uint32_t i = 0; i < 2; ++i) {
    const auto root = Root(0, i);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  }
  IREE_ASSERT_OK(Register(0, 0, 2));
  IREE_ASSERT_OK(Register(1, 0, 0));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), kSlabBytes, 256,
      &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, kSlabBytes));
  struct loom_serve_test_weight_cache_owner_t {
    // Logical ownership can be released once, independently of physical trim.
    bool retained = true;
    // Accepted maintenance keeps backing alive until its owner retires it.
    bool pending = false;
    // Counts requests, not repeated probes while retirement is pending.
    uint32_t requests = 0;
  } owner;
  loom_serve_residency_set_reclaimer(
      residencies_[1],
      {&owner,
       [](void* self) {
         auto* owner = static_cast<loom_serve_test_weight_cache_owner_t*>(self);
         if (!owner->retained) {
           return false;
         }
         owner->retained = false;
         owner->pending = true;
         ++owner->requests;
         return true;
       },
       [](void* self) {
         return static_cast<loom_serve_test_weight_cache_owner_t*>(self)
             ->pending;
       }});
  auto* cache = ResidencyCache();
  loom_serve_residency_admission_t admission;
  IREE_ASSERT_OK(Compile(1, "raw"));
  const auto peer_root = Root(1, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &peer_root, {path_.path()}));
  IREE_ASSERT_OK(Register(2, 2, 1));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[2]));
  ASSERT_NO_FATAL_FAILURE(Run(1, 90));
  // Even discarding every idle history cannot admit the two-slab group while
  // this one-slab peer is pinned. Neither kind of pin permits useful eviction.
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(owner.requests, 0u);
  EXPECT_TRUE(owner.retained);
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  loom_serve_residency_set_pinned(residencies_[2], true);
  loom_serve_residency_release(residencies_[2]);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
  EXPECT_EQ(owner.requests, 0u);
  EXPECT_TRUE(owner.retained);
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  loom_serve_residency_set_pinned(residencies_[2], false);
  // Once the peer's parameters become evictable, retiring the idle history is
  // necessary and sufficient. The regular asynchronous reclaimer must run.
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  EXPECT_EQ(loom_serve_residency_statistics(residencies_[0]).committed_bytes,
            0u);
  IREE_ASSERT_OK(loom_serve_residency_try_admit(residencies_[0], &admission));
  EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED);
  EXPECT_EQ(owner.requests, 1u);
  // Cancelling admission does not pretend accepted maintenance has retired.
  loom_serve_residency_cancel_admission(residencies_[0]);
  EXPECT_TRUE(loom_serve_residency_cache_is_reclaiming(cache));
  EXPECT_EQ(state_statistics_.committed_bytes, kSlabBytes);
  loom_serve_virtual_buffer_begin_trim(state_);
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  owner.pending = false;
  EXPECT_FALSE(loom_serve_residency_cache_is_reclaiming(cache));
  ASSERT_NO_FATAL_FAILURE(AdmitAfterRelease(0));
  ASSERT_NO_FATAL_FAILURE(Run(0, 90));
  loom_serve_residency_release(residencies_[0]);
  loom_serve_residency_set_reclaimer(residencies_[1], {});
  EXPECT_EQ(owner.requests, 1u);
}

TEST_P(WeightsTest, MutableGrowthReclaimsIdleParametersWithoutDiscardingState) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  if (!GetParam()) {
    return;
  }
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), 2 * kSlabBytes,
      256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_virtual_buffer_commit(state_, 0, 2 * kSlabBytes));
  EXPECT_EQ(state_statistics_.committed_bytes, 0u);
  loom_serve_residency_release(residencies_[0]);
  // Mutable growth reclaims idle parameters and reuses their backing without
  // requiring an intervening native free.
  const loom_serve_memory_range_t range = {state_, 0, 2 * kSlabBytes};
  bool state_admitted = false;
  IREE_ASSERT_OK(
      loom_serve_memory_pool_try_commit(pool_, 1, &range, &state_admitted));
  if (!state_admitted) {
    ASSERT_TRUE(loom_serve_memory_pool_is_pending(pool_));
    IREE_ASSERT_OK(loom_serve_memory_pool_await(pool_));
    IREE_ASSERT_OK(
        loom_serve_memory_pool_try_commit(pool_, 1, &range, &state_admitted));
  }
  ASSERT_TRUE(state_admitted);
  EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
  const uint32_t value = 0xA1B2C3D4;
  uint32_t observed = 0;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = loom_serve_virtual_buffer_handle(state_),
              .length = sizeof(value),
              .pattern = &value,
              .pattern_length = sizeof(value),
          },
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &fill, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution_, completion));
  bool admitted = true;
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(residencies_[0], &admitted));
  EXPECT_FALSE(admitted);
  IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
  EXPECT_EQ(state_statistics_.committed_bytes, 2 * kSlabBytes);
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = loom_serve_virtual_buffer_handle(state_),
              .target = &observed,
              .length = sizeof(observed),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(observed, value);
  loom_serve_virtual_buffer_begin_trim(state_);
  loom_serve_virtual_buffer_keep(state_, 0, sizeof(value));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(state_));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  EXPECT_EQ(loom_serve_memory_pool_statistics(pool_).committed_bytes,
            2 * kSlabBytes);
}

TEST_P(WeightsTest,
       DeviceTrimPrefersIdleWorkspaceAndPreservesPinnedParameters) {
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Compile(i, "target"));
    const auto root = Root(i, 0);
    IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
    IREE_ASSERT_OK(Register(i, i, 1));
    ASSERT_NO_FATAL_FAILURE(RunPinned(i, 70));
  }
  IREE_ASSERT_OK(loom_serve_residency_acquire(residencies_[0]));
  IREE_ASSERT_OK(
      loom_serve_execution_reserve_workspace(execution_, kSlabBytes, 256));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = kSlabBytes,
      .min_alignment = 256,
  };
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, 1, &request, &workspace_));
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution_, workspace_, &completion));
  iree_hal_buffer_release(workspace_);
  workspace_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  const auto before = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(before.retained.committed_bytes, GetParam() ? 2 * kSlabBytes : 0);
  // The shared workspace keeps its registered slab geometry after a trim.
  // Earlier streaming cases may have registered a larger geometry.
  EXPECT_GE(before.workspace.bytes_committed, kSlabBytes);
  IREE_ASSERT_OK(
      loom_serve_device_trim(device_owner_, before.retained.committed_bytes));
  const auto idle = loom_serve_device_memory_statistics(device_owner_);
  ASSERT_EQ(idle.retained.committed_bytes, before.retained.committed_bytes);
  DrainNativeReturns();
  EXPECT_EQ(loom_serve_device_memory_statistics(device_owner_)
                .workspace.bytes_committed,
            0u);
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  ASSERT_NO_FATAL_FAILURE(Run(1, 70));

  IREE_ASSERT_OK(
      loom_serve_execution_alloca(execution_, 1, &request, &workspace_));
  // The model's real command completion follows allocation on the work chain.
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, kSlabBytes - 1));
  DrainNativeReturns();
  const auto pinned = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(pinned.workspace.bytes_committed, before.workspace.bytes_committed);
  EXPECT_EQ(pinned.workspace.reservation_count, 1u);
  EXPECT_EQ(pinned.retained.committed_bytes, GetParam() ? kSlabBytes : 0);
  if (GetParam()) {
    EXPECT_EQ(loom_serve_weights_statistics(plans_[1]).committed_bytes, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(Run(0, 70));
  loom_serve_residency_release(residencies_[0]);
  IREE_ASSERT_OK(
      loom_serve_execution_dealloca(execution_, workspace_, &completion));
  iree_hal_buffer_release(workspace_);
  workspace_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  IREE_ASSERT_OK(loom_serve_device_trim(device_owner_, 0));
  DrainNativeReturns();
  const auto cold = loom_serve_device_memory_statistics(device_owner_);
  EXPECT_EQ(cold.retained.committed_bytes, 0u);
  EXPECT_EQ(cold.workspace.bytes_committed, 0u);
  ASSERT_NO_FATAL_FAILURE(RunPinned(1, 70));
  EXPECT_EQ(loom_serve_device_memory_statistics(device_owner_)
                .retained.committed_bytes,
            GetParam() ? kSlabBytes : 0);
}

class SmallRootWeightsTest : public WeightsTest {
 protected:
  uint64_t SlabSize() const override { return 2 * 1024 * 1024; }
};

TEST_P(SmallRootWeightsTest, SmallRootCommitsOnlyItsNativeExtent) {
  IREE_ASSERT_OK(Compile(0, "target"));
  const auto root = Root(0, 0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  if (GetParam()) {
    const auto geometry = loom_serve_device_weight_geometry(device_owner_);
    const auto memory = loom_serve_weights_statistics(plans_[0]);
    EXPECT_EQ(memory.reserved_bytes, geometry.minimum_size);
    EXPECT_EQ(memory.committed_bytes, geometry.minimum_size);
    EXPECT_LT(memory.committed_bytes, geometry.slab_size);
    IREE_ASSERT_OK(loom_serve_residency_cache_trim(ResidencyCache(), 0));
    DrainNativeReturns();
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
    ASSERT_NO_FATAL_FAILURE(RunPinned(0, 70));
  }
}

INSTANTIATE_TEST_SUITE_P(Backing, SmallRootWeightsTest,
                         ::testing::Values(false, true));

INSTANTIATE_TEST_SUITE_P(Backing, WeightsTest, ::testing::Values(false, true));

// The failed reclaimer belongs to a private pool; native devices remain shared.
class RejectedAdmissionWeightsTest : public WeightsTest {};

TEST_P(RejectedAdmissionWeightsTest, PreflightFailureReleasesAllPins) {
  ASSERT_NO_FATAL_FAILURE(UseDeferredBacking());
  IREE_ASSERT_OK(Compile(0, "raw"));
  const auto root = Root(0, 0);
  IREE_ASSERT_OK(CreateRoots(0, 1, &root, {path_.path()}));
  IREE_ASSERT_OK(Register(0, 0, 1));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
      pool_, loom_serve_device_state_geometry(device_owner_), 2 * kSlabBytes,
      256, &state_statistics_, &state_));
  IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(state_, 0, 2 * kSlabBytes));
  loom_serve_memory_pool_set_reclaimer(
      pool_, {nullptr, [](void*, uint64_t) {
                return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                        "parameter capacity reclaim failed");
              }});
  retirement_code_ = IREE_STATUS_UNAVAILABLE;
  for (int attempt = 0; attempt < 2; ++attempt) {
    loom_serve_residency_admission_t admission;
    IREE_EXPECT_STATUS_IS(retirement_code_, loom_serve_residency_try_admit(
                                                residencies_[0], &admission));
    EXPECT_EQ(admission, LOOM_SERVE_RESIDENCY_ADMISSION_CAPACITY);
    const auto snapshot = loom_serve_residency_snapshot(residencies_[0]);
    EXPECT_FALSE(snapshot.active);
    EXPECT_EQ(snapshot.pin_count, 0u);
    EXPECT_FALSE(loom_serve_residency_group_is_pinned(parameter_groups_[0]));
    EXPECT_EQ(loom_serve_weights_activity(plans_[0]).activations, 0u);
    EXPECT_EQ(loom_serve_weights_statistics(plans_[0]).committed_bytes, 0u);
    EXPECT_EQ(state_statistics_.committed_bytes, 2 * kSlabBytes);
  }
}

INSTANTIATE_TEST_SUITE_P(Elastic, RejectedAdmissionWeightsTest,
                         ::testing::Values(true));

}  // namespace loom_serve::testing
