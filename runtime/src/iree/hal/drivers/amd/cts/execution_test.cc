// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>
#include <tuple>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/cts/affine_images.h"
#include "iree/hal/drivers/amd/cts/cpu_executable.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::cts {
namespace {

using ::iree::hal::cts::Ref;

enum Engine { kCpu, kNpu, kGpu, kEngineCount };

enum class SubmissionMode {
  kDefaults,
  kExplicit,
  kRecordedIndirect,
  kRemappedDefaults,
  kPooledDefaults,
  kAtomicDefaults,
};

static bool UsesDefaultBarriers(SubmissionMode mode) {
  return mode == SubmissionMode::kDefaults ||
         mode == SubmissionMode::kRemappedDefaults ||
         mode == SubmissionMode::kPooledDefaults ||
         mode == SubmissionMode::kAtomicDefaults;
}

struct Boundary {
  // Combined queue-local execution and visibility actions.
  iree_hal_barrier_t barrier = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
  };
  // Per-edge range recipes captured alongside their combined global effects.
  std::array<iree_hal_buffer_barrier_t, kEngineCount> ranges = {};
};

struct Invocation {
  // Acquires from the actual producers of the invocation's inputs.
  Boundary before;
  // Releases to subsequent native and CPU consumers.
  Boundary after;
  // Optional reusable recording with only indirect buffer references.
  Ref<iree_hal_command_buffer_t> commands;
};

struct Participant {
  // Independently created device retained by this fixture and the sealed group.
  Ref<iree_hal_device_t> device;
  // Borrowed queue from the device's dispatch-capable family.
  iree_hal_queue_t* queue = nullptr;
  // Native implementation's semaphore, signaled only by its own queue.
  Ref<iree_hal_semaphore_t> timeline;
  // Loaded arithmetic executable owned until all submissions complete.
  Ref<iree_hal_executable_t> executable;
  // Reflected function in the executable.
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
};

class ExecutionTest
    : public ::testing::TestWithParam<std::tuple<bool, SubmissionMode>> {
 protected:
  static constexpr size_t kElementCount = 16;
  static constexpr size_t kBytes = kElementCount * sizeof(uint32_t);
  using Payload = std::array<uint32_t, 5 * kElementCount>;

  static void SetUpTestSuite() {
    // KFD's primary VM belongs to the process. Keep one canonical native
    // instance while independently constructing and releasing each device
    // group.
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("rdna"), &driver_));
  }

  static void TearDownTestSuite() {
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    iree_hal_driver_registry_free(registry_);
    registry_ = nullptr;
  }

  void SetUp() override {
    IREE_ASSERT_OK(context_.Initialize(iree_allocator_system(),
                                       iree_hal_device_event_sink_stderr()));
    iree_hal_driver_t* xdna_driver = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("xdna"), &xdna_driver));
    EXPECT_EQ(driver_, xdna_driver);
    iree_hal_driver_release(xdna_driver);
    ASSERT_NO_FATAL_FAILURE(CreateNativeDevice(kNpu, IREE_SV("xdna")));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(CreateNativeDevice(kGpu, IREE_SV("rdna")));
    if (IsSkipped()) {
      return;
    }

    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(4, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &cpu_executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &cpu_allocator_));
    iree_hal_task_device_params_t task_params;
    iree_hal_task_device_params_initialize(&task_params);
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("cpu"), &task_params, 1, &cpu_executor_, 0, nullptr,
        cpu_allocator_, context_.params(), iree_allocator_system(),
        participants_[kCpu].device.out()));

    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder,
                                             context_.frontier_tracker());
    status = iree_ok_status();
    for (size_t i = 0; i < kEngineCount && iree_status_is_ok(status); ++i) {
      status = iree_hal_device_group_builder_add_device(
          &builder, participants_[i].device);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), group_.out());
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    for (auto& participant : participants_) {
      participant.queue = iree_hal_device_queue(participant.device, 0, 0);
      ASSERT_NE(participant.queue, nullptr);
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          participant.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, participant.timeline.out()));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        participants_[kCpu].device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, start_.out()));
    ASSERT_NO_FATAL_FAILURE(CreatePool());
    ASSERT_NO_FATAL_FAILURE(LoadExecutables());
  }

  void TearDown() override {
    start_.reset();
    mapping_pool_.reset();
    pool_.reset();
    group_.reset();
    for (auto& participant : participants_) {
      participant.executable.reset();
      participant.timeline.reset();
      participant.device.reset();
    }
    iree_hal_allocator_release(cpu_allocator_);
    iree_task_executor_release(cpu_executor_);
    context_.Deinitialize();
  }

  void CreateNativeDevice(Engine engine, iree_string_view_t scheme) {
    iree_host_size_t count = 0;
    iree_hal_device_info_t* infos = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, scheme, iree_allocator_system(), &count, &infos));
    iree_allocator_free(iree_allocator_system(), infos);
    if (!count) {
      GTEST_SKIP() << "No native " << std::string(scheme.data, scheme.size)
                   << " endpoint";
    }
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_uri(
        driver_, scheme, context_.params(), iree_allocator_system(),
        participants_[engine].device.out()));
  }

  void CreatePool() {
    const iree_hal_buffer_interface_t interfaces[] = {
        IREE_HAL_BUFFER_INTERFACE_HOST, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
        IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    for (size_t i = 0; i < families_.size(); ++i) {
      families_[i].family = iree_hal_queue_family(participants_[i].queue);
      families_[i].usage =
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
      families_[i].interfaces = UINT64_C(1) << interfaces[i];
    }
    if (std::get<1>(GetParam()) == SubmissionMode::kAtomicDefaults) {
      for (Engine engine : {kCpu, kGpu}) {
        families_[engine].atomics.queue.system_scope_32 =
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
      }
    }
    iree_hal_pool_scope_t scope = {
        .family_count = families_.size(),
        .families = families_.data(),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
    if (std::get<1>(GetParam()) == SubmissionMode::kAtomicDefaults) {
      for (Engine engine : {kCpu, kGpu, kNpu}) {
        iree_hal_memory_scope_t memory_scope;
        IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
            group_, {IREE_HAL_MEMORY_SITE_QUEUE, families_[engine].family},
            &memory_scope));
        const auto atomics =
            iree_hal_pool_query_atomic_operations(pool_, memory_scope);
        if (engine == kNpu) {
          EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(atomics));
        } else {
          EXPECT_NE(
              atomics.system_scope_32 & IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
              0u);
        }
      }
    }
  }

  void CreateVirtualStorage(iree_hal_buffer_t** out_root,
                            std::array<Ref<iree_hal_buffer_t>, 2>* backings,
                            iree_device_size_t* out_unit) {
    iree_hal_pool_scope_t scope = {
        .family_count = families_.size(),
        .families = families_.data(),
        .host = {IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_MAPPING_MODE_PERSISTENT,
                 IREE_HAL_HOST_CACHEABILITY_UNKNOWN},
    };
    iree_hal_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        group_, scope, iree_allocator_system(), &capabilities));
    if (!capabilities.max_byte_length) {
      GTEST_SKIP() << "Native topology does not support joint CPU/NPU/GPU "
                      "address reservations";
    }
    auto source_families = families_;
    for (auto& family : source_families) {
      family.interfaces = UINT64_C(1)
                          << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
    }
    auto source_scope = scope;
    source_scope.families = source_families.data();
    source_scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, source_scope, &pool_options, iree_allocator_system(),
        mapping_pool_.out()));
    const auto source = iree_hal_pool_query_mapping_requirements(mapping_pool_);
    const auto unit = iree_max(source.minimum_granularity,
                               iree_max(capabilities.mapping_granularity,
                                        capabilities.reservation_granularity));
    ASSERT_GE(unit, sizeof(Payload));
    // Whole slabs start at source offset zero. The mapping unit constrains
    // source ranges and reserved addresses, not the backing's own addresses.
    for (auto& backing : *backings) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          mapping_pool_, iree_hal_make_pool_reservation_request(unit, 0),
          iree_infinite_timeout(), backing.out()));
    }
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = unit,
        .mapping_granularity = unit,
    };
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group_, scope, &options, iree_allocator_system(), out_root));
    *out_unit = unit;
  }

  void Load(Engine engine, iree_const_byte_span_t bytes,
            iree_string_view_t name) {
    auto& participant = participants_[engine];
    const auto* targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participant.device));
    ASSERT_EQ(targets->target_count, 1u);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data = bytes;
    IREE_ASSERT_OK(iree_hal_executable_load(
        iree_hal_queue_family(participant.queue), &targets->targets[0], &params,
        participant.executable.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        participant.executable, name, &participant.function));
  }

  void LoadExecutables() {
    auto& cpu = participants_[kCpu];
    cpu.executable.reset(
        CreateCpuAddExecutable(iree_hal_queue_family(cpu.queue)));
    cpu.function = iree_hal_executable_function_from_index(0);
    const auto* npu_targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participants_[kNpu].device));
    ASSERT_EQ(npu_targets->target_count, 1u);
    const bool halo =
        iree_string_view_equal(npu_targets->targets[0].target_key,
                               IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const auto* npu_image = halo ? iree_hal_amd_xdna_test_mul_i32_create()
                                 : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    ASSERT_NO_FATAL_FAILURE(Load(
        kNpu, iree_make_const_byte_span(npu_image[0].data, npu_image[0].size),
        IREE_SV("mul_i32")));
    const auto* gpu_targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participants_[kGpu].device));
    ASSERT_EQ(gpu_targets->target_count, 1u);
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        gpu_targets->targets[0].target_key, &identity));
    ASSERT_TRUE(identity.version.major == 11 || identity.version.major == 12);
    const char* selector =
        identity.version.major == 11 ? "gfx11-generic" : "gfx12-generic";
    const auto* gpu_images = iree_hal_amd_cts_affine_images_create();
    for (size_t i = 0; i < iree_hal_amd_cts_affine_images_size(); ++i) {
      if (strstr(gpu_images[i].name, selector)) {
        ASSERT_NO_FATAL_FAILURE(Load(
            kGpu,
            iree_make_const_byte_span(gpu_images[i].data, gpu_images[i].size),
            IREE_SV("affine_i32")));
        break;
      }
    }
    ASSERT_TRUE(participants_[kGpu].executable);
  }

  static iree_hal_dispatch_config_t Config(Engine engine) {
    auto config = iree_hal_make_static_dispatch_config(
        engine == kCpu ? kElementCount : 1, 1, 1);
    config.workgroup_size[0] = engine == kGpu ? 32 : 1;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    return config;
  }

  void AddTransition(Engine producer, Engine consumer,
                     iree_hal_memory_transition_action_t action,
                     iree_hal_buffer_ref_t reference, Boundary* boundary) {
    auto table = transition_table_;
    iree_hal_memory_scope_t scopes[2];
    const Engine engines[] = {producer, consumer};
    for (size_t i = 0; i < IREE_ARRAYSIZE(engines); ++i) {
      const iree_hal_memory_site_t site = {
          .kind = IREE_HAL_MEMORY_SITE_QUEUE,
          .family = iree_hal_queue_family(participants_[engines[i]].queue),
      };
      IREE_ASSERT_OK(
          iree_hal_device_group_resolve_memory_scope(group_, site, &scopes[i]));
    }
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, scopes[0], scopes[1], action, &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    const auto effects = action == IREE_HAL_MEMORY_TRANSITION_RELEASE
                             ? transition.release
                             : transition.acquire;
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(effects));
    boundary->barrier.effects =
        iree_hal_memory_effects_combine(boundary->barrier.effects, effects);
    const auto* recipe = iree_hal_memory_transition_recipe(table, pair, action);
    if (recipe) {
      auto& count = boundary->barrier.buffer_barrier_count;
      ASSERT_LT(count, boundary->ranges.size());
      boundary->ranges[count++] = {.buffer_ref = reference, .recipe = recipe};
      boundary->barrier.buffer_barriers = boundary->ranges.data();
    }
  }

  void Record(Engine engine, iree_hal_buffer_ref_list_t bindings,
              uint32_t addend, Invocation* invocation) {
    auto& participant = participants_[engine];
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(participant.queue),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/4, invocation->commands.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(invocation->commands));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(
        invocation->commands, &invocation->before.barrier));
    const auto constants =
        engine == kNpu ? iree_const_byte_span_empty()
                       : iree_make_const_byte_span(&addend, sizeof(addend));
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        invocation->commands, participant.executable, participant.function,
        Config(engine), constants, bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(invocation->commands,
                                                   &invocation->after.barrier));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(invocation->commands));
  }

  void Dispatch(Engine engine, iree_hal_semaphore_t* wait, uint64_t wait_value,
                uint64_t signal_value, iree_hal_buffer_ref_list_t bindings,
                uint32_t addend, const Invocation& invocation,
                iree_hal_buffer_binding_table_t binding_table) {
    auto& participant = participants_[engine];
    auto* signal = participant.timeline.get();
    if (invocation.commands) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          participant.queue, {1, &wait, &wait_value},
          {1, &signal, &signal_value}, invocation.commands, binding_table,
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
      return;
    }
    const iree_hal_barrier_list_t before = {1, &invocation.before.barrier};
    const iree_hal_barrier_list_t after = {1, &invocation.after.barrier};
    const iree_hal_queue_barriers_t barriers = {&before, &after};
    const auto constants =
        engine == kNpu ? iree_const_byte_span_empty()
                       : iree_make_const_byte_span(&addend, sizeof(addend));
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        participant.queue, {1, &wait, &wait_value}, {1, &signal, &signal_value},
        participant.executable, participant.function, Config(engine), constants,
        bindings,
        UsesDefaultBarriers(std::get<1>(GetParam())) ? nullptr : &barriers,
        IREE_HAL_DISPATCH_FLAG_NONE));
  }

  // CPU worker resources outlive the Task device using them.
  iree_task_executor_t* cpu_executor_ = nullptr;
  // Ordinary Task allocation source, independent of the shared physical pool.
  iree_hal_allocator_t* cpu_allocator_ = nullptr;
  // Proactors and frontier identities shared by all three device instances.
  ::iree::hal::cts::DeviceCreateContext context_;
  // Process-level registry holding the native VM across independent cases.
  static inline iree_hal_driver_registry_t* registry_ = nullptr;
  // Canonical AMD driver acquired through both public engine schemes.
  static inline iree_hal_driver_t* driver_ = nullptr;
  // Actual Task, XDNA, and PM4 implementations; no native submission bypass.
  std::array<Participant, kEngineCount> participants_;
  // Sealed topology carrying the complete participants and visibility scopes.
  Ref<iree_hal_device_group_t> group_;
  // One allocation source qualified for all three execution interfaces.
  Ref<iree_hal_pool_t> pool_;
  // Mapping sources borrow this pool until checked removal and buffer release.
  Ref<iree_hal_pool_t> mapping_pool_;
  // Same complete access set used for physical and stable-address storage.
  std::array<iree_hal_pool_family_access_t, kEngineCount> families_ = {};
  // Borrows the current test buffer's contract through its final submission.
  iree_hal_memory_transition_table_t transition_table_ = {};
  // Host-controlled gate keeping every consumer pending until capture finishes.
  Ref<iree_hal_semaphore_t> start_;
};

TEST_P(ExecutionTest, ConsumerFirstKernelsShareOneBacking) {
  const bool npu_first = std::get<0>(GetParam());
  const auto mode = std::get<1>(GetParam());
  const bool remapped = mode == SubmissionMode::kRemappedDefaults;
  const bool pooled = mode == SubmissionMode::kPooledDefaults;
  const bool recorded = mode == SubmissionMode::kRecordedIndirect;
  const bool atomic = mode == SubmissionMode::kAtomicDefaults;
  const bool defaults = UsesDefaultBarriers(mode);
  Ref<iree_hal_semaphore_t> memory_ready;
  IREE_ASSERT_OK(iree_hal_semaphore_create(participants_[kCpu].device,
                                           IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                           0, 0, memory_ready.out()));
  auto* memory_timeline = memory_ready.get();
  auto* start = start_.get();
  std::array<Ref<iree_hal_buffer_t>, 2> backings;
  std::array<Payload, 2> expected_backings = {};
  Ref<iree_hal_pool_t> block_pool;
  iree_device_size_t mapping_unit = 0;
  Ref<iree_hal_buffer_t> root;
  if (remapped) {
    ASSERT_NO_FATAL_FAILURE(
        CreateVirtualStorage(root.out(), &backings, &mapping_unit));
    if (IsSkipped()) {
      return;
    }
  } else if (pooled) {
    Ref<iree_hal_buffer_t> slab;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(512, 64),
        iree_infinite_timeout(), slab.out()));
    iree_hal_fixed_block_pool_options_t options;
    iree_hal_fixed_block_pool_options_initialize(&options);
    options.block_size = 512;
    options.alignment = 64;
    IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
        slab, 0, 512, &options, iree_allocator_system(), block_pool.out()));
    const auto request =
        iree_hal_make_pool_reservation_request(sizeof(Payload), 64);
    uint64_t gate = 1, allocated = 1;
    IREE_ASSERT_OK(iree_hal_queue_alloca(participants_[kCpu].queue,
                                         {1, &start, &gate},
                                         {1, &memory_timeline, &allocated},
                                         block_pool, 1, &request, root.out()));
  } else {
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(sizeof(Payload), 64),
        iree_infinite_timeout(), root.out()));
  }
  transition_table_ = iree_hal_buffer_transition_table(root);
  const iree_hal_memory_scope_t all_scopes = {
      iree_hal_device_group_memory_domain(group_), IREE_HAL_MEMORY_SCOPE_ANY};
  std::array<Ref<iree_hal_buffer_t>, 3> views;
  std::array<iree_hal_buffer_ref_t, 3> bindings;
  for (size_t i = 0; i < views.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, (i + 1) * kBytes, kBytes,
                                           iree_allocator_system(),
                                           views[i].out()));
    bindings[i] = iree_hal_make_buffer_ref(views[i], 0, kBytes);
  }
  const Engine first = npu_first ? kNpu : kGpu;
  const Engine second = npu_first ? kGpu : kNpu;
  const auto* gpu_binding = &bindings[npu_first ? 2 : 0];
  const iree_hal_buffer_binding_t table_bindings[] = {
      {views[0], 0, kBytes},
      {views[1], 0, kBytes},
      {views[2], 0, kBytes},
      {root, 0, sizeof(Payload)},
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      IREE_ARRAYSIZE(table_bindings), table_bindings};
  Invocation producer, native_first, native_second, consumer;
  Boundary upload_release, download_acquire;
  const auto root_ref = iree_hal_make_buffer_ref(root, 0, sizeof(Payload));
  const auto recorded_root_ref =
      recorded ? iree_hal_make_indirect_buffer_ref(3, 0, sizeof(Payload))
               : root_ref;
  if (!defaults) {
    // Upload also produces the unchanged RHS that the NPU reads in either
    // engine order, independently of the producer's later LHS update.
    for (Engine engine : {kNpu, kGpu}) {
      ASSERT_NO_FATAL_FAILURE(AddTransition(kCpu, engine,
                                            IREE_HAL_MEMORY_TRANSITION_RELEASE,
                                            root_ref, &upload_release));
    }
    ASSERT_NO_FATAL_FAILURE(AddTransition(kCpu, first,
                                          IREE_HAL_MEMORY_TRANSITION_RELEASE,
                                          recorded_root_ref, &producer.after));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(kCpu, first, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                      recorded_root_ref, &native_first.before));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(first, second, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                      recorded_root_ref, &native_first.after));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(first, second, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                      recorded_root_ref, &native_second.before));
    if (first == kGpu) {
      ASSERT_NO_FATAL_FAILURE(
          AddTransition(kCpu, kNpu, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                        recorded_root_ref, &native_second.before));
      // Final CPU verification reads the GPU-modified LHS as well as the NPU
      // result; the NPU's output release cannot publish somebody else's writes.
      ASSERT_NO_FATAL_FAILURE(
          AddTransition(kGpu, kCpu, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                        recorded_root_ref, &native_first.after));
      ASSERT_NO_FATAL_FAILURE(AddTransition(kGpu, kCpu,
                                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                                            root_ref, &download_acquire));
    }
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(second, kCpu, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                      recorded_root_ref, &native_second.after));
    ASSERT_NO_FATAL_FAILURE(AddTransition(second, kCpu,
                                          IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                                          recorded_root_ref, &consumer.before));
  }
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_barrier_list_t upload_after = {1, &upload_release.barrier};
  const iree_hal_barrier_list_t download_before = {1,
                                                   &download_acquire.barrier};
  const iree_hal_queue_barriers_t upload_barriers = {&empty, &upload_after};
  const iree_hal_queue_barriers_t download_barriers = {&download_before,
                                                       &empty};
  if (recorded) {
    std::array<iree_hal_buffer_ref_t, 3> references;
    for (size_t i = 0; i < references.size(); ++i) {
      references[i] = iree_hal_make_indirect_buffer_ref(i, 0, kBytes);
    }
    const auto gpu_reference = references[npu_first ? 2 : 0];
    ASSERT_NO_FATAL_FAILURE(Record(kCpu, {1, &references[0]}, 7, &producer));
    ASSERT_NO_FATAL_FAILURE(Record(
        first,
        first == kNpu
            ? iree_hal_buffer_ref_list_t{references.size(), references.data()}
            : iree_hal_buffer_ref_list_t{1, &gpu_reference},
        29, &native_first));
    ASSERT_NO_FATAL_FAILURE(Record(
        second,
        second == kNpu
            ? iree_hal_buffer_ref_list_t{references.size(), references.data()}
            : iree_hal_buffer_ref_list_t{1, &gpu_reference},
        29, &native_second));
    ASSERT_NO_FATAL_FAILURE(Record(kCpu, {1, &references[2]}, 100, &consumer));
  }
  auto* cpu_timeline = participants_[kCpu].timeline.get();
  for (uint32_t replay = 0; replay < (atomic ? 32u : 3u); ++replay) {
    SCOPED_TRACE(replay);
    uint64_t gate = replay + 1;
    uint64_t initialized = replay * 4 + 1, produced = initialized + 1,
             consumed = initialized + 2, downloaded = initialized + 3;
    uint64_t gpu_added = replay * 2 + 1, cpu_added = gpu_added + 1;
    Payload initial;
    initial.fill(0xA5A5A5A5u);
    for (size_t i = 0; i < kElementCount; ++i) {
      initial[kElementCount + i] =
          static_cast<uint32_t>(i * 97 + 13 + replay * 211);
      initial[2 * kElementCount + i] =
          static_cast<uint32_t>(0xFFFFFF00u + i * 11 + replay);
    }
    Payload result = {};
    Payload recycled = {};
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
    };
    download.download.source_buffer = root;
    download.download.target = result.data();
    download.download.length = sizeof(result);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        participants_[kCpu].queue, {1, &cpu_timeline, &consumed},
        {1, &cpu_timeline, &downloaded}, 1, &download,
        defaults ? nullptr : &download_barriers));
    ASSERT_NO_FATAL_FAILURE(Dispatch(kCpu, participants_[second].timeline, gate,
                                     consumed, {1, &bindings[2]}, 100, consumer,
                                     binding_table));
    // Submit every consumer before its producer. Each native engine signals its
    // own semaphore, so both native-to-foreign admission directions are
    // required.
    ASSERT_NO_FATAL_FAILURE(Dispatch(
        second, participants_[first].timeline, gate, gate,
        second == kNpu
            ? iree_hal_buffer_ref_list_t{bindings.size(), bindings.data()}
            : iree_hal_buffer_ref_list_t{1, gpu_binding},
        29, native_second, binding_table));
    ASSERT_NO_FATAL_FAILURE(Dispatch(
        first, atomic ? memory_timeline : cpu_timeline,
        atomic ? cpu_added : produced, gate,
        first == kNpu
            ? iree_hal_buffer_ref_list_t{bindings.size(), bindings.data()}
            : iree_hal_buffer_ref_list_t{1, gpu_binding},
        29, native_first, binding_table));
    if (atomic) {
      // CPU and GPU atomically update the first LHS element in the same
      // backing the NPU reads normally. Queue dependencies order both updates
      // before the native kernels, without granting NPU atomic capabilities.
      iree_hal_atomic_rmw_params_t params = {
          .operand = 3,
          .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
                   IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          .width = IREE_HAL_ATOMIC_WIDTH_32,
          .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
      };
      IREE_ASSERT_OK(iree_hal_queue_atomic_rmw(
          participants_[kCpu].queue, {1, &memory_timeline, &gpu_added},
          {1, &memory_timeline, &cpu_added}, views[0], 0, params, nullptr));
      params.operand = 5;
      IREE_ASSERT_OK(iree_hal_queue_atomic_rmw(
          participants_[kGpu].queue, {1, &cpu_timeline, &produced},
          {1, &memory_timeline, &gpu_added}, views[0], 0, params, nullptr));
    }
    ASSERT_NO_FATAL_FAILURE(Dispatch(kCpu, cpu_timeline, initialized, produced,
                                     {1, &bindings[0]}, 7, producer,
                                     binding_table));
    iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
    };
    upload.update.source_buffer = initial.data();
    upload.update.target_buffer = root;
    upload.update.length = sizeof(initial);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        participants_[kCpu].queue,
        {1, (remapped || (pooled && !replay)) ? &memory_timeline : &start,
         &gate},
        {1, &cpu_timeline, &initialized}, 1, &upload,
        defaults ? nullptr : &upload_barriers));
    if (remapped) {
      const iree_hal_memory_mapping_t mapping = {
          .target = root,
          .scope = all_scopes,
          .target_range = {0, mapping_unit},
          .source = backings[replay % backings.size()],
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = replay ? IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE
                              : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      };
      uint64_t previous_reader = replay * 4;
      iree_hal_semaphore_t* waits[] = {start, cpu_timeline};
      uint64_t values[] = {gate, previous_reader};
      // Every participant can submit the same typed batch. Address replacement
      // waits for the final CPU reader, not merely the earlier native producer.
      IREE_ASSERT_OK(iree_hal_queue_map(
          participants_[replay % kEngineCount].queue,
          {replay ? 2u : 1u, waits, values}, {1, &memory_timeline, &gate}, 1,
          &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
    }
    if (pooled && replay == 2) {
      uint64_t released = 2, acquired = 3, filled = 4, read = 5, returned = 6;
      auto* original = root.get();
      // The one-block pool cannot grow. Publish the final reader's frontier
      // while that reader is still gated, then acquire the same capacity on
      // GPU.
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          participants_[kNpu].queue, {1, &cpu_timeline, &downloaded},
          {1, &memory_timeline, &released}, 1, &original));
      Ref<iree_hal_buffer_t> reused;
      const auto request =
          iree_hal_make_pool_reservation_request(sizeof(Payload), 64);
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          participants_[kGpu].queue, {1, &memory_timeline, &released},
          {1, &memory_timeline, &acquired}, block_pool, 1, &request,
          reused.out()));
      const uint32_t pattern = 0x71727374u;
      IREE_ASSERT_OK(iree_hal_queue_fill(
          participants_[kGpu].queue, {1, &memory_timeline, &acquired},
          {1, &memory_timeline, &filled}, reused, 0, sizeof(Payload), &pattern,
          sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
      iree_hal_transfer_operation_t readback = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      };
      readback.download.source_buffer = reused;
      readback.download.target = recycled.data();
      readback.download.length = sizeof(recycled);
      IREE_ASSERT_OK(iree_hal_queue_transfer(
          participants_[kCpu].queue, {1, &memory_timeline, &filled},
          {1, &memory_timeline, &read}, 1, &readback, nullptr));
      auto* next = reused.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          participants_[kCpu].queue, {1, &memory_timeline, &read},
          {1, &memory_timeline, &returned}, 1, &next));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_signal(start, gate, nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(cpu_timeline, downloaded,
                                           iree_infinite_timeout(), 0));
    if (pooled && replay == 2) {
      IREE_ASSERT_OK(iree_hal_semaphore_wait(memory_timeline, 6,
                                             iree_infinite_timeout(), 0));
      Payload expected_recycled;
      expected_recycled.fill(0x71727374u);
      EXPECT_EQ(recycled, expected_recycled);
      iree_hal_pool_stats_t stats;
      iree_hal_pool_query_stats(block_pool, &stats);
      EXPECT_EQ(stats.reservation_count, 0u);
    }
    Payload expected = initial;
    for (size_t i = 0; i < kElementCount; ++i) {
      uint32_t lhs = initial[kElementCount + i] + 7;
      if (atomic && i == 0) {
        lhs += 8;
      }
      if (first == kGpu) {
        lhs = lhs * 3 + 29;
      }
      expected[kElementCount + i] = lhs;
      uint32_t value = lhs * initial[2 * kElementCount + i];
      if (second == kGpu) {
        value = value * 3 + 29;
      }
      expected[3 * kElementCount + i] = value + 100;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(result[i], expected[i]) << "element=" << i;
    }
    if (remapped) {
      expected_backings[replay % backings.size()] = expected;
      for (size_t i = 0; i < iree_min(replay + 1, backings.size()); ++i) {
        iree_hal_buffer_mapping_t mapping;
        IREE_ASSERT_OK(iree_hal_buffer_map_range(
            backings[i], IREE_HAL_MAPPING_MODE_PERSISTENT,
            IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
            sizeof(Payload), &mapping));
        EXPECT_EQ(memcmp(mapping.contents.data, expected_backings[i].data(),
                         sizeof(Payload)),
                  0)
            << "backing=" << i;
        IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
      }
    }
  }
  if (remapped) {
    uint64_t final_reader = 12, detached = 4;
    const iree_hal_memory_mapping_t mapping = {
        .target = root,
        .scope = all_scopes,
        .target_range = {0, mapping_unit},
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
    };
    IREE_ASSERT_OK(iree_hal_queue_map(participants_[kGpu].queue,
                                      {1, &cpu_timeline, &final_reader},
                                      {1, &memory_timeline, &detached}, 1,
                                      &mapping, IREE_HAL_QUEUE_MAP_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(memory_timeline, detached,
                                           iree_infinite_timeout(), 0));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SharedBacking, ExecutionTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(SubmissionMode::kDefaults,
                                         SubmissionMode::kExplicit,
                                         SubmissionMode::kRecordedIndirect,
                                         SubmissionMode::kRemappedDefaults,
                                         SubmissionMode::kPooledDefaults,
                                         SubmissionMode::kAtomicDefaults)),
    [](const ::testing::TestParamInfo<ExecutionTest::ParamType>& info) {
      std::string name =
          std::get<0>(info.param) ? "CpuNpuGpuCpu" : "CpuGpuNpuCpu";
      switch (std::get<1>(info.param)) {
        case SubmissionMode::kDefaults:
          return name + "Defaults";
        case SubmissionMode::kExplicit:
          return name + "Explicit";
        case SubmissionMode::kRecordedIndirect:
          return name + "RecordedIndirect";
        case SubmissionMode::kRemappedDefaults:
          return name + "RemappedDefaults";
        case SubmissionMode::kPooledDefaults:
          return name + "PooledDefaults";
        case SubmissionMode::kAtomicDefaults:
          return name + "AtomicDefaults";
      }
      return name;
    });

}  // namespace
}  // namespace iree::hal::amd::cts
