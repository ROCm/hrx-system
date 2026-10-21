// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/shm.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/cts/cpu_executable.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::cts {

enum class CpuHandoffMode {
  kQueueDefaults,
  kPublicMapping,
  kRecordedDirect,
  kRecordedDeferred,
  kRecordedIndirect,
  kRegisteredDirect,
  kRegisteredIndirect,
};

class CpuXdnaNativeTest : public ::testing::TestWithParam<CpuHandoffMode> {
 protected:
  void SetUp() override {
    SetUpXdna();
    if (IsSkipped() || HasFatalFailure()) {
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
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &cpu_proactor_pool_));
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = cpu_proactor_pool_;
    iree_hal_task_device_params_t task_params;
    iree_hal_task_device_params_initialize(&task_params);
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("cpu"), &task_params, 1, &cpu_executor_, 0, nullptr,
        cpu_allocator_, &create_params, iree_allocator_system(), &cpu_device_));
    cpu_queue_ = iree_hal_device_queue(cpu_device_, 0, 0);
    cpu_executable_ = CreateCpuAddExecutable(iree_hal_queue_family(cpu_queue_));
  }

  void TearDown() override {
    for (auto* commands : commands_) {
      iree_hal_command_buffer_release(commands);
    }
    iree_hal_executable_release(cpu_executable_);
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    iree_hal_pool_release(memory_pool_);
    iree_hal_executable_release(executable_);
    iree_hal_device_release(device_);
    iree_hal_device_release(cpu_device_);
    iree_hal_device_group_release(device_group_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_driver_release(driver_);
    iree_hal_driver_registry_free(registry_);
    if (registered_storage_.base) {
      EXPECT_EQ(registration_release_count_, 1);
      iree_shm_close(&registered_storage_);
    }
    iree_async_proactor_pool_release(cpu_proactor_pool_);
    iree_hal_allocator_release(cpu_allocator_);
    iree_task_executor_release(cpu_executor_);
  }

  void SetUpXdna() {
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("xdna"), &driver_));
    iree_host_size_t count = 0;
    iree_hal_device_info_t* infos = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, IREE_SV("xdna"), iree_allocator_system(), &count, &infos));
    std::string device_path;
    if (count) {
      device_path.assign(infos[0].path.data, infos[0].path.size);
    }
    iree_allocator_free(iree_allocator_system(), infos);
    if (!count) {
      GTEST_SKIP() << "No native XDNA endpoint";
    }
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = pool_;
    params.event_sink = iree_hal_device_event_sink_stderr();
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_path(
        driver_, IREE_SV("xdna"), iree_make_cstring_view(device_path.c_str()),
        0, nullptr, &params, iree_allocator_system(), &device_));
    // Native memory and execution retain the fabric without either discovery
    // owner. Every workload below runs after these releases.
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    iree_hal_driver_registry_free(registry_);
    registry_ = nullptr;
    // The device's selected entry keeps the runner alive independently.
    iree_async_proactor_pool_release(pool_);
    pool_ = nullptr;
    queue_ = iree_hal_device_queue(device_, 0, 0);
    ASSERT_NE(queue_, nullptr);
  }

  void LoadMultiply() {
    const auto* targets =
        iree_hal_device_spec_executables(iree_hal_device_spec(device_));
    ASSERT_EQ(targets->target_count, 1u);
    const bool halo = iree_string_view_equal(
        targets->targets[0].target_key, IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const auto* toc = halo ? iree_hal_amd_xdna_test_mul_i32_create()
                           : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    std::vector<uint8_t> bytes(toc[0].data, toc[0].data + toc[0].size);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes.data(), bytes.size());
    IREE_ASSERT_OK(
        iree_hal_executable_load(iree_hal_device_queue_family(device_, 0),
                                 &targets->targets[0], &params, &executable_));
    std::fill(bytes.begin(), bytes.end(), 0xCC);
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("mul_i32"), &function_));
  }

  void MakeSemaphore(iree_hal_semaphore_t** out_semaphore) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, out_semaphore));
    semaphores_.push_back(*out_semaphore);
  }

  void CreateSharedPool(iree_hal_memory_access_t host_access) {
    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker);
    iree_status_t status =
        iree_hal_device_group_builder_add_device(&builder, cpu_device_);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_add_device(&builder, device_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &device_group_);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    iree_async_frontier_tracker_release(tracker);
    IREE_ASSERT_OK(status);
    iree_hal_pool_family_access_t families[2] = {};
    families[0].family = iree_hal_queue_family(cpu_queue_);
    families[0].usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
    families[0].interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST;
    families[0].atomics.queue.system_scope_32 =
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
    families[1].family = iree_hal_queue_family(queue_);
    families[1].usage = families[0].usage;
    families[1].interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
    iree_hal_pool_scope_t scope = {
        .family_count = IREE_ARRAYSIZE(families),
        .families = families,
    };
    scope.host.access = host_access;
    scope.host.modes = host_access ? IREE_HAL_MAPPING_MODE_SCOPED : 0u;
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    const iree_hal_external_storage_class_t storage_class = {
        IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER, {0, 0}};
    const iree_hal_pool_import_class_t import_class = {storage_class,
                                                       {1, &storage_class}};
    if (GetParam() == CpuHandoffMode::kRegisteredDirect ||
        GetParam() == CpuHandoffMode::kRegisteredIndirect) {
      options.imports.class_count = 1;
      options.imports.classes = &import_class;
      if (GetParam() == CpuHandoffMode::kRegisteredIndirect) {
        options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
      }
    }
    IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, scope, &options,
                                             iree_allocator_system(),
                                             &memory_pool_));
  }

  iree_hal_memory_scope_t QueueScope(iree_hal_queue_t* queue) {
    iree_hal_memory_site_t site = {
        .kind = IREE_HAL_MEMORY_SITE_QUEUE,
        .family = iree_hal_queue_family(queue),
    };
    iree_hal_memory_scope_t scope;
    IREE_CHECK_OK(iree_hal_device_group_resolve_memory_scope(device_group_,
                                                             site, &scope));
    return scope;
  }

  void RecordCpuWork(iree_hal_buffer_ref_t reference, uint32_t addend,
                     iree_hal_memory_transition_action_t action,
                     iree_hal_command_buffer_t** out_commands) {
    const bool release = action == IREE_HAL_MEMORY_TRANSITION_RELEASE;
    const auto table = iree_hal_pool_transition_table(memory_pool_);
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, QueueScope(release ? cpu_queue_ : queue_),
        QueueScope(release ? queue_ : cpu_queue_), action, &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    const auto* recipe = iree_hal_memory_transition_recipe(table, pair, action);
    iree_hal_buffer_barrier_t range = {
        .buffer_ref = reference,
        .recipe = recipe,
    };
    iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
        .effects = release ? transition.release : transition.acquire,
    };
    barrier.buffer_barrier_count = recipe ? 1 : 0;
    barrier.buffer_barriers = recipe ? &range : nullptr;
    // The qualified XDNA DMA side requires no additional payload cache action.
    EXPECT_EQ((release ? transition.acquire : transition.release).bits, 0u);
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(cpu_queue_), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, UsesIndirectBindings() ? 1 : 0,
        out_commands));
    commands_.push_back(*out_commands);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_commands));
    if (!release) {
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(*out_commands, &barrier));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        *out_commands, cpu_executable_,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(16, 1, 1),
        iree_make_const_byte_span(&addend, sizeof(addend)), {1, &reference},
        IREE_HAL_DISPATCH_FLAG_NONE));
    if (release) {
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(*out_commands, &barrier));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(*out_commands));
  }

  bool UsesIndirectBindings() const {
    return GetParam() == CpuHandoffMode::kRecordedIndirect ||
           GetParam() == CpuHandoffMode::kRegisteredIndirect;
  }

  void ImportRegisteredStorage(iree_hal_buffer_t** out_buffer) {
    iree_hal_external_storage_class_t storage_class = {
        .type = IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
    };
    iree_hal_pool_import_requirements_t requirements;
    ASSERT_TRUE(iree_hal_pool_query_import_requirements(
        memory_pool_, storage_class, &requirements));
    const iree_host_size_t byte_length =
        ((iree_max(8192, requirements.min_byte_length) - 1) /
             requirements.byte_length_granularity +
         1) *
        requirements.byte_length_granularity;
    iree_shm_mapping_t originator = {};
    IREE_ASSERT_OK(iree_shm_create(nullptr, byte_length, &originator));
    memset(originator.base, 0xCC, originator.size);
    iree_status_t status = iree_shm_open_handle(
        originator.handle, originator.size, &registered_storage_);
    // Neither the creator's mapping nor its handle participates in native
    // registration or subsequent CPU/NPU work on the independent mapping.
    iree_shm_close(&originator);
    IREE_ASSERT_OK(status);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(registered_storage_.base) %
                  requirements.base_alignment,
              0u);
    iree_hal_external_storage_t storage = {
        .storage_class = storage_class,
    };
    storage.payload.host_pointer = registered_storage_.base;
    storage.byte_length = registered_storage_.size;
    storage.maintenance_range =
        iree_hal_make_buffer_range(0, registered_storage_.size);
    storage.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
    storage.release = +[](void* user_data, iree_hal_external_storage_class_t,
                          iree_hal_external_storage_payload_t) {
      ++*static_cast<int*>(user_data);
    };
    storage.release_user_data = &registration_release_count_;
    iree_hal_buffer_t* imported = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_import_buffer(
        memory_pool_, &storage, iree_hal_make_buffer_range(kBytes, 5 * kBytes),
        &imported));
    buffers_.push_back(imported);
    EXPECT_EQ(storage.byte_length, 0u);
    EXPECT_EQ(storage.release, nullptr);
    EXPECT_EQ(registration_release_count_, 0);
    iree_hal_buffer_range_t range;
    IREE_ASSERT_OK(iree_hal_buffer_export(
        imported, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
        storage_class, &storage, &range));
    EXPECT_EQ(storage.payload.host_pointer, registered_storage_.base);
    EXPECT_EQ(storage.byte_length, registered_storage_.size);
    EXPECT_EQ(storage.release, nullptr);
    EXPECT_EQ(range.byte_offset, kBytes);
    EXPECT_EQ(range.byte_length, 5 * kBytes);
    IREE_ASSERT_OK(
        iree_hal_pool_import_buffer(memory_pool_, &storage, range, out_buffer));
    EXPECT_EQ(storage.byte_length, 0u);
  }

  // Canonical compiler fixture binding extent.
  static constexpr size_t kBytes = 16 * sizeof(uint32_t);
  // Driver owning discovery until the XDNA device is constructed.
  iree_hal_driver_t* driver_ = nullptr;
  // Explicit registry released before native work begins.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Construction services released after the XDNA device retains its entry.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // XDNA execution device retained independently of the sealed group.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed XDNA queue used for arithmetic and semaphore creation.
  iree_hal_queue_t* queue_ = nullptr;
  // Sealed CPU/XDNA topology retaining both devices.
  iree_hal_device_group_t* device_group_ = nullptr;
  // Shared backing provider qualified for both execution families.
  iree_hal_pool_t* memory_pool_ = nullptr;
  // Loaded XDNA multiply kernel.
  iree_hal_executable_t* executable_ = nullptr;
  // Reflected XDNA entry point.
  iree_hal_executable_function_t function_ =
      iree_hal_executable_function_invalid();
  // Test-owned buffers released before the group and devices.
  std::vector<iree_hal_buffer_t*> buffers_;
  // Dependency timelines retained until all replays finish.
  std::vector<iree_hal_semaphore_t*> semaphores_;
  // Existing CPU services shared by all submissions in this test.
  iree_task_executor_t* cpu_executor_ = nullptr;
  // CPU device's ordinary allocation source.
  iree_hal_allocator_t* cpu_allocator_ = nullptr;
  // CPU asynchronous progress owner.
  iree_async_proactor_pool_t* cpu_proactor_pool_ = nullptr;
  // Task device retained independently of the sealed group.
  iree_hal_device_t* cpu_device_ = nullptr;
  // Borrowed provisioned CPU queue.
  iree_hal_queue_t* cpu_queue_ = nullptr;
  // Compiled CPU arithmetic kernel and immutable ABI metadata.
  iree_hal_executable_t* cpu_executable_ = nullptr;
  // Recorded programs retained across three changing-input replays.
  std::vector<iree_hal_command_buffer_t*> commands_;
  // Independently opened host mapping retained through native attachment
  // release.
  iree_shm_mapping_t registered_storage_ = {};
  // Original imported transport callback, independent of borrowed reimports.
  int registration_release_count_ = 0;
};

TEST_P(CpuXdnaNativeTest, CpuNpuCpuKernelsShareOneBacking) {
  const bool public_mapping = GetParam() == CpuHandoffMode::kPublicMapping;
  ASSERT_NO_FATAL_FAILURE(CreateSharedPool(public_mapping
                                               ? IREE_HAL_MEMORY_ACCESS_READ
                                               : IREE_HAL_MEMORY_ACCESS_NONE));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_pool_query_atomic_operations(
                                    memory_pool_, QueueScope(cpu_queue_))
                                    .system_scope_32,
                                IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD));
  EXPECT_TRUE(iree_hal_atomic_operation_capabilities_is_empty(
      iree_hal_pool_query_atomic_operations(memory_pool_, QueueScope(queue_))));
  ASSERT_NO_FATAL_FAILURE(LoadMultiply());
  iree_hal_semaphore_t *start = nullptr, *progress = nullptr;
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&start));
  ASSERT_NO_FATAL_FAILURE(MakeSemaphore(&progress));
  uint64_t start_value = 1, allocated = 1;
  const auto request = iree_hal_make_pool_reservation_request(5 * kBytes, 64);
  iree_hal_buffer_t* root = nullptr;
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t no_payload_barriers = {&empty, &empty};
  const bool deferred = GetParam() == CpuHandoffMode::kRecordedDeferred ||
                        GetParam() == CpuHandoffMode::kRecordedIndirect;
  const bool registered = GetParam() == CpuHandoffMode::kRegisteredDirect ||
                          GetParam() == CpuHandoffMode::kRegisteredIndirect;
  if (deferred) {
    IREE_ASSERT_OK(iree_hal_queue_alloca(cpu_queue_, {1, &start, &start_value},
                                         {1, &progress, &allocated},
                                         memory_pool_, 1, &request, &root));
  } else {
    if (registered) {
      ASSERT_NO_FATAL_FAILURE(ImportRegisteredStorage(&root));
    } else {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          memory_pool_, request, iree_infinite_timeout(), &root));
    }
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        cpu_queue_, {1, &start, &start_value}, {1, &progress, &allocated},
        &no_payload_barriers, IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  }
  buffers_.push_back(root);
  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_map_range(root, IREE_HAL_MAPPING_MODE_SCOPED,
                                public_mapping ? IREE_HAL_MEMORY_ACCESS_WRITE
                                               : IREE_HAL_MEMORY_ACCESS_READ,
                                IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, kBytes,
                                &mapping));
  std::array<iree_hal_buffer_ref_t, 3> npu_bindings;
  for (size_t i = 0; i < npu_bindings.size(); ++i) {
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, (i + 1) * kBytes, kBytes,
                                           iree_allocator_system(), &view));
    buffers_.push_back(view);
    npu_bindings[i] = iree_hal_make_buffer_ref(view, 0, kBytes);
  }
  const bool indirect = UsesIndirectBindings();
  const auto cpu_input =
      indirect ? iree_hal_make_indirect_buffer_ref(0, kBytes, kBytes)
               : iree_hal_make_buffer_ref(root, kBytes, kBytes);
  const auto cpu_output =
      indirect ? iree_hal_make_indirect_buffer_ref(0, 3 * kBytes, kBytes)
               : iree_hal_make_buffer_ref(root, 3 * kBytes, kBytes);
  const iree_hal_buffer_binding_t binding = {root, 0, 5 * kBytes};
  const iree_hal_buffer_binding_table_t binding_table = {
      indirect ? 1u : 0u, indirect ? &binding : nullptr};
  iree_hal_command_buffer_t *producer = nullptr, *consumer = nullptr;
  if (GetParam() != CpuHandoffMode::kQueueDefaults && !public_mapping) {
    ASSERT_NO_FATAL_FAILURE(RecordCpuWork(
        cpu_input, 7, IREE_HAL_MEMORY_TRANSITION_RELEASE, &producer));
    ASSERT_NO_FATAL_FAILURE(RecordCpuWork(
        cpu_output, 100, IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &consumer));
  }
  uint64_t previous = allocated;
  for (uint32_t replay = 0; replay < 3; ++replay) {
    SCOPED_TRACE(replay);
    uint64_t initialized = previous + 1, updated = previous + 2,
             produced = previous + 3, computed = previous + 4,
             consumed = previous + 5, downloaded = previous + 6;
    std::array<uint32_t, 80> initial;
    initial.fill(0xA5A5A5A5u);
    for (size_t i = 0; i < 16; ++i) {
      initial[16 + i] = static_cast<uint32_t>(i * 97 + 13 + replay * 211);
      initial[32 + i] = static_cast<uint32_t>(0xFFFFFF00u + i * 11 + replay);
    }
    std::array<uint32_t, 80> result = {};
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
    };
    download.download.source_buffer = root;
    download.download.target = result.data();
    download.download.length = sizeof(result);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        cpu_queue_, {1, &progress, &consumed}, {1, &progress, &downloaded}, 1,
        &download, &no_payload_barriers));
    // Submit consumers first: both queues must obey semaphore edges.
    if (consumer) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          cpu_queue_, {1, &progress, &computed}, {1, &progress, &consumed},
          consumer, binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    } else {
      const uint32_t addend = 100;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          cpu_queue_, {1, &progress, &computed}, {1, &progress, &consumed},
          cpu_executable_, iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(16, 1, 1),
          iree_make_const_byte_span(&addend, sizeof(addend)),
          {1, &npu_bindings[2]},
          /*barriers=*/nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue_, {1, &progress, &produced}, {1, &progress, &computed},
        executable_, function_, iree_hal_make_static_dispatch_config(1, 1, 1),
        {}, {npu_bindings.size(), npu_bindings.data()}, &no_payload_barriers,
        IREE_HAL_DISPATCH_FLAG_NONE));
    if (producer) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          cpu_queue_, {1, &progress, &updated}, {1, &progress, &produced},
          producer, binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    } else {
      const uint32_t addend = 7;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          cpu_queue_, {1, &progress, &updated}, {1, &progress, &produced},
          cpu_executable_, iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(16, 1, 1),
          iree_make_const_byte_span(&addend, sizeof(addend)),
          {1, &npu_bindings[0]},
          /*barriers=*/nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
    }
    const iree_hal_atomic_rmw_params_t atomic_update = {
        .operand = 5,
        .flags = IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
        .width = IREE_HAL_ATOMIC_WIDTH_32,
        .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
    };
    IREE_ASSERT_OK(iree_hal_queue_atomic_rmw(
        cpu_queue_, {1, &progress, &initialized}, {1, &progress, &updated},
        root, kBytes, atomic_update, &no_payload_barriers));
    iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
    };
    upload.update.source_buffer = initial.data();
    upload.update.target_buffer = root;
    upload.update.length = sizeof(initial);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        cpu_queue_, {1, &progress, &previous}, {1, &progress, &initialized}, 1,
        &upload, /*barriers=*/nullptr));
    if (!replay) {
      IREE_ASSERT_OK(iree_hal_semaphore_signal(start, start_value, nullptr));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_wait(progress, downloaded,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    for (size_t i = 0; i < 16; ++i) {
      const uint32_t input = initial[16 + i] + (i == 0 ? 5 : 0) + 7;
      EXPECT_EQ(result[i], initial[i]) << i;
      EXPECT_EQ(result[16 + i], input) << i;
      EXPECT_EQ(result[32 + i], initial[32 + i]) << i;
      EXPECT_EQ(result[48 + i], input * initial[32 + i] + 100) << i;
      EXPECT_EQ(result[64 + i], initial[64 + i]) << i;
    }
    if (public_mapping) {
      iree_hal_memory_site_t site = {
          .kind = IREE_HAL_MEMORY_SITE_HOST,
      };
      iree_hal_memory_scope_t host_scope;
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
          device_group_, site, &host_scope));
      const auto transition = iree_hal_pool_query_transition(
          memory_pool_, QueueScope(cpu_queue_), host_scope);
      EXPECT_EQ(transition.release.bits, 0u);
      EXPECT_EQ(transition.acquire.bits, 0u);
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          root, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, sizeof(result), &mapping));
      EXPECT_EQ(memcmp(mapping.contents.data, result.data(), sizeof(result)),
                0);
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    }
    previous = downloaded;
  }
  if (registered) {
    const auto* bytes = static_cast<const uint8_t*>(registered_storage_.base);
    for (size_t i = 0; i < kBytes; ++i) {
      EXPECT_EQ(bytes[i], 0xCC) << i;
    }
    for (size_t i = 6 * kBytes; i < registered_storage_.size; ++i) {
      EXPECT_EQ(bytes[i], 0xCC) << i;
    }
    EXPECT_EQ(registration_release_count_, 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SharedBacking, CpuXdnaNativeTest,
    ::testing::Values(
        CpuHandoffMode::kQueueDefaults, CpuHandoffMode::kPublicMapping,
        CpuHandoffMode::kRecordedDirect, CpuHandoffMode::kRecordedDeferred,
        CpuHandoffMode::kRecordedIndirect, CpuHandoffMode::kRegisteredDirect,
        CpuHandoffMode::kRegisteredIndirect),
    [](const ::testing::TestParamInfo<CpuHandoffMode>& info) {
      switch (info.param) {
        case CpuHandoffMode::kQueueDefaults:
          return "QueueDefaults";
        case CpuHandoffMode::kPublicMapping:
          return "PublicMapping";
        case CpuHandoffMode::kRecordedDirect:
          return "RecordedDirect";
        case CpuHandoffMode::kRecordedDeferred:
          return "RecordedDeferred";
        case CpuHandoffMode::kRecordedIndirect:
          return "RecordedIndirect";
        case CpuHandoffMode::kRegisteredDirect:
          return "RegisteredDirect";
        case CpuHandoffMode::kRegisteredIndirect:
          return "RegisteredIndirect";
      }
      return "";
    });

}  // namespace iree::hal::amd::cts
