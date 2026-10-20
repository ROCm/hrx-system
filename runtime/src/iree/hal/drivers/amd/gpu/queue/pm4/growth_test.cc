// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/growth.h"

#include <atomic>
#include <barrier>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/hal/device.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only native memory is substituted. Typed acquisition, its NUMA worker,
// publication ownership, captures, and backing initialization are production
// components. These tests qualify their ownership boundary, not GPU execution.
class GrowthTest : public ::testing::Test {
 protected:
  struct Memory {
    // Fixture owning the native allocation dependency.
    GrowthTest* owner;
    // Persistent host mapping requested by the real fabric memory adapter.
    std::vector<uint8_t> bytes;
    // Distinct aligned native address, independent of the host mapping.
    uint64_t address;
    // Native alignment promised by this allocation dependency.
    uint64_t alignment;
  };

  void SetUp() override {
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
      auto* self = reinterpret_cast<GrowthTest*>(scope);
      ++self->allocations_;
      if (self->hold_.exchange(false)) {
        self->entered_.release();
        self->resume_.acquire();
      }
      if (self->fail_.exchange(false)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
      const uint64_t alignment =
          iree_max(info->minimum_alignment, UINT64_C(4096));
      const uint64_t address =
          iree_align_uint64(self->next_address_, alignment);
      self->next_address_ =
          address + iree_align_uint64(info->byte_length, alignment);
      auto* memory = new Memory{self, std::vector<uint8_t>(info->byte_length),
                                address, alignment};
      ++self->live_;
      self->native_thread_ = std::this_thread::get_id();
      *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_info = [](amdf_memory_t* handle,
                                amdf_memory_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->byte_length = memory->bytes.size();
      info->alignment = memory->alignment;
      info->flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_COHERENT;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* handle, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* out_address) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      memory->owner->install_thread_ = std::this_thread::get_id();
      *out_address = memory->address;
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* handle, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** out_mapping) {
      *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(handle);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* handle,
                                      amdf_host_mapping_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->pointer = memory->bytes.data();
      info->byte_length = memory->bytes.size();
      info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* handle,
                             amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
      auto* memory = reinterpret_cast<Memory*>(handle);
      --memory->owner->live_;
      delete memory;
      return AMDF_STATUS_OK;
    };
    api_.instance_destroy = [](amdf_instance_t*) { return AMDF_STATUS_OK; };
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &api_;
    context_.fabric = fabric_;
    context_.api = &api_;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    memory_.scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    memory_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    memory_.profile.allocation.minimum_alignment = 8;
    memory_.profile.allocation.byte_length_granularity = 8;
    memory_.profile.allocation.maximum_byte_length = UINT32_MAX;
    scratch_memory_ = memory_;
    scratch_memory_.host_access = 0;
    iree_hal_amd_gpu_capture_pool_initialize(4096, iree_allocator_system(),
                                             &captures_);
    iree_hal_amd_gpu_publisher_initialize(&publisher_);
    iree_hal_amd_gpu_signal_pool_initialize(
        {iree_hal_amd_gpu_pm4_growth_signals_available, &growth_}, &signals_);
    growth_.context = &context_;
    growth_.memory = &memory_;
    growth_.scratch_memory = &scratch_memory_;
    growth_.scratch_profile = &scratch_profile_;
    growth_.captures = &captures_;
    growth_.publisher = &publisher_;
    growth_.publication = &publication_;
    growth_.queue_ordinal = 0;
    growth_.queue_count = 3;
    growth_.signal_capacity = 8;
    growth_.signal_pool = &signals_;
    growth_.reuse.entry.kind = IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS;
    source_ = {
        .publish =
            [](void* user_data) {
              return static_cast<GrowthTest*>(user_data)->Publish();
            },
        .user_data = this,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        1, &source_, {}, iree_allocator_system(), &publication_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_context_join_memory(&context_);
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    EXPECT_EQ(growth_.arguments, nullptr);
    EXPECT_EQ(growth_.signals, nullptr);
    EXPECT_EQ(growth_.scratch, nullptr);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 0);
    iree_hal_amd_gpu_signal_pool_deinitialize(&signals_);
    EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
    iree_hal_amd_gpu_publisher_deinitialize(&publisher_);
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    EXPECT_EQ(live_, 0u);
    iree_hal_amd_fabric_release(fabric_);
  }

  iree_hal_amd_gpu_pm4_submission_t* Capture() {
    iree_hal_amd_gpu_pm4_barrier_profile_t profile = {};
    const iree_hal_barrier_list_t empty = {};
    const iree_hal_queue_barriers_t barriers = {
        .before = &empty,
        .after = &empty,
    };
    iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_pm4_submission_create(
        &captures_, false, iree_hal_semaphore_list_empty(),
        iree_hal_semaphore_list_empty(), nullptr, nullptr, {}, {},
        iree_const_byte_span_empty(), iree_hal_buffer_ref_list_empty(),
        &profile, &barriers, 0, &submission));
    return submission;
  }

  void Handoff() {
    if (iree_hal_amd_gpu_publisher_handoff(&publisher_, nullptr)) {
      iree_hal_amd_gpu_publication_notify(&publication_, 0);
    } else {
      idle_.release();
    }
  }

  void JoinPublication() {
    // Producers have joined before this cold ownership check. The consumer
    // posts only after its final handoff, not merely after a capture retires.
    while (!iree_hal_amd_gpu_publisher_is_idle(&publisher_)) {
      idle_.acquire();
    }
  }

  void Park(iree_hal_amd_gpu_pm4_entry_kind_t kind) {
    auto* submission = Capture();
    ASSERT_TRUE(iree_hal_amd_gpu_publisher_enqueue(
        &publisher_, &submission->entry.publication));
    ASSERT_EQ(iree_hal_amd_gpu_publisher_next(&publisher_),
              &submission->entry.publication);
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_growth_begin(&growth_, kind, submission, 4096));
    Handoff();
  }

  void Complete(iree_hal_amd_gpu_publication_entry_t* entry,
                iree_status_code_t code) {
    auto* submission = iree_containerof(
        entry, iree_hal_amd_gpu_pm4_submission_t, entry.publication);
    iree_hal_amd_gpu_pm4_submission_release(submission);
    if (code != IREE_STATUS_OK) {
      ++failed_;
    }
    ++completed_;
    consumed_.release();
  }

  bool Publish() {
    if (!iree_hal_amd_gpu_publisher_acquire_service(&publisher_)) {
      return false;
    }
    auto* link = iree_hal_amd_gpu_publisher_next(&publisher_);
    if (link) {
      auto* entry =
          iree_containerof(link, iree_hal_amd_gpu_pm4_entry_t, publication);
      if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_SIGNALS) {
        iree_hal_amd_gpu_pm4_growth_reuse_signals(&growth_);
      } else if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_REUSE_PROGRAM) {
        iree_hal_amd_gpu_pm4_growth_reuse_program(entry);
      } else if (entry->kind >= IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS) {
        iree_hal_amd_gpu_pm4_growth_result_t result;
        iree::Status status =
            iree_hal_amd_gpu_pm4_growth_end(entry, 123456, &result);
        if (status.ok()) {
          if (result.arguments) {
            EXPECT_EQ(result.arguments->begin, 123456u);
            EXPECT_EQ(result.arguments->end, 123456u);
            EXPECT_EQ(result.arguments->capacity, 4096u);
            iree_hal_amd_gpu_pm4_argument_storage_free(&context_,
                                                       result.arguments);
          }
          if (result.signals) {
            EXPECT_EQ(result.signals->block.queue_count, 3u);
            auto* signal =
                iree_hal_amd_gpu_signal_pool_try_acquire(&signals_, 0);
            EXPECT_NE(signal, nullptr);
            iree_hal_amd_gpu_signal_release(signal);
            iree_hal_amd_gpu_pm4_signal_storage_free(&context_, result.signals);
          }
          if (result.scratch) {
            EXPECT_EQ(result.scratch->wave_byte_length, 4096u);
            auto* memory =
                reinterpret_cast<Memory*>(result.scratch->memory.handle);
            EXPECT_EQ(memory->bytes.size(), 4096u * 32 * 2);
            EXPECT_EQ(result.scratch->address, memory->address);
            EXPECT_EQ(result.scratch->memory.mapping, nullptr);
            iree_hal_amd_gpu_pm4_scratch_storage_free(&context_,
                                                      result.scratch);
          }
          if (result.first) {
            EXPECT_FALSE(iree_hal_amd_gpu_publisher_enqueue_batch(
                &publisher_, result.first, result.last));
          }
        } else {
          for (auto* current = result.first; current;) {
            auto* next =
                reinterpret_cast<iree_hal_amd_gpu_publication_entry_t*>(
                    iree_atomic_load(&current->next,
                                     iree_memory_order_relaxed));
            Complete(current, static_cast<iree_status_code_t>(status.code()));
            current = next;
          }
        }
        growth_completed_.release();
      } else {
        if (entry->kind == IREE_HAL_AMD_GPU_PM4_ENTRY_EXECUTE) {
          auto* submission =
              iree_containerof(entry, iree_hal_amd_gpu_pm4_submission_t, entry);
          auto* program = &submission->execute.program->resident;
          uint64_t frontier[3] = {};
          auto* instance = iree_hal_amd_gpu_pm4_program_try_acquire(
              program, growth_.queue_ordinal, frontier);
          EXPECT_NE(instance, nullptr);
          if (instance) {
            submission->execute.instance = instance;
            iree_hal_amd_gpu_pm4_program_publish(
                program, instance, growth_.queue_ordinal, completed_ + 1);
            submission->execute.published = true;
          }
        }
        if (require_signals_) {
          auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&signals_, 0);
          EXPECT_EQ(signal, expected_signal_);
          iree_hal_amd_gpu_signal_release(signal);
        }
        Complete(link, IREE_STATUS_OK);
      }
    }
    const bool more = iree_hal_amd_gpu_publisher_handoff(&publisher_, nullptr);
    if (!more) {
      idle_.release();
    }
    return more;
  }

  void ReuseDuringGrowth(bool fail_growth, bool return_before_enrollment) {
    uint64_t value = 0;
    uint64_t frontier[3] = {};
    iree_hal_amd_gpu_signal_t metadata = {};
    iree_hal_amd_gpu_signal_block_t block;
    iree_hal_amd_gpu_signal_block_initialize(&signals_, this, &value, 0x1000, 8,
                                             0, &metadata, 1, 3, frontier,
                                             &block);
    auto* held = iree_hal_amd_gpu_signal_pool_try_acquire(&signals_, 0);
    ASSERT_NE(held, nullptr);
    EXPECT_FALSE(iree_hal_amd_gpu_signal_pool_has_available(&signals_));
    expected_signal_ = held;
    require_signals_ = true;
    hold_ = true;
    fail_ = fail_growth;
    if (return_before_enrollment) {
      // The miss is proven above. A concurrent last release may finish before
      // publication has parked its capture and enrolled the capacity wake.
      iree_hal_amd_gpu_signal_release(held);
    }
    Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS);
    entered_.acquire();
    if (!return_before_enrollment) {
      iree_hal_amd_gpu_signal_release(held);
    }
    consumed_.acquire();
    EXPECT_EQ(completed_, 1u);
    EXPECT_EQ(allocations_, 1u);
    EXPECT_EQ(live_, 0u);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 1);
    // Public work has completed using the exact old cell. Native allocation
    // still owns its request and may now succeed or fail with an empty cohort.
    resume_.release();
    growth_completed_.acquire();
    iree_hal_amd_gpu_context_join_memory(&context_);
    JoinPublication();
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    EXPECT_EQ(failed_, 0u);
    EXPECT_EQ(completed_, 1u);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 0);
    iree_hal_amd_gpu_signal_block_deinitialize(&block);
  }

  // Native API dependency and its retained fabric domain registry.
  amdf_api_t api_ = {};
  // Owned fabric registry, released after callback and publication joins.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Production lifetime owner for typed native acquisition/release.
  iree_hal_amd_gpu_context_t context_ = {};
  // Coherent memory policy borrowed through each acquisition.
  iree_hal_amd_fabric_memory_options_t memory_ = {};
  // Physical scratch policy with no host mapping request.
  iree_hal_amd_fabric_memory_options_t scratch_memory_ = {};
  // Two native shader engines, each with 32 resident wave slots.
  iree_hal_amd_gpu_pm4_scratch_profile_t scratch_profile_ = {
      .waves_per_engine = 32,
      .engine_count = 2,
  };
  // Production reusable capture owner.
  iree_hal_amd_gpu_capture_pool_t captures_ = {};
  // Actual queue publication ownership and FIFO.
  iree_hal_amd_gpu_publisher_t publisher_ = {};
  // Exact cold acquisition state under test.
  iree_hal_amd_gpu_pm4_growth_context_t growth_ = {};
  // Signal cells across all completed native backing blocks.
  iree_hal_amd_gpu_signal_pool_t signals_ = {};
  // Device deferred publication owner.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Stable source borrowed by publication_.
  iree_hal_amd_gpu_publication_source_t source_ = {};
  // One-shot native allocation hold, armed before the request is submitted.
  std::atomic<bool> hold_{false};
  // One-shot native allocation failure.
  std::atomic<bool> fail_{false};
  // Entry into the native hold, independent of any wall-clock deadline.
  std::binary_semaphore entered_{0};
  // Release of the native hold.
  std::binary_semaphore resume_{0};
  // One token for each capture after its actual release.
  std::counting_semaphore<> consumed_{0};
  // One token after each native result is consumed, including empty cohorts.
  std::counting_semaphore<> growth_completed_{0};
  // Completed publisher handoffs, separate from terminal capture completion.
  std::counting_semaphore<> idle_{0};
  // Number of actual native allocation calls.
  std::atomic<uint32_t> allocations_{0};
  // Native address space allocated only by this fixture's NUMA owner.
  uint64_t next_address_ = 0x100000;
  // Native allocations whose release has not completed.
  std::atomic<uint32_t> live_{0};
  // Number of released capture owners.
  std::atomic<uint32_t> completed_{0};
  // Number of captures retired through failed native acquisition.
  std::atomic<uint32_t> failed_{0};
  // This consumer requires a real pool cell before retiring a parked capture.
  bool require_signals_ = false;
  // Exact existing cell expected by the held-growth reuse witness.
  iree_hal_amd_gpu_signal_t* expected_signal_ = nullptr;
  // Native acquisition thread, read after all callbacks join.
  std::thread::id native_thread_;
  // Address installation thread, read after publication joins.
  std::thread::id install_thread_;
};

// Uses the actual recorder, resident program, invocation capture, growth, and
// retirement ownership. The device is only a resource-lifetime dependency;
// native memory is supplied by GrowthTest. No GPU execution is simulated here.
class ProgramGrowthTest : public GrowthTest {
 protected:
  struct Device {
    // Resource identity retained by the production command buffer.
    iree_hal_resource_t resource;
    // Last-release observation after every production borrower is joined.
    bool destroyed = false;
  };

  struct ObservedBuffer {
    // HAL identity retained by the recording's direct-resource set.
    iree_hal_buffer_t base;
    // Set when the recording releases its final resource reference.
    bool destroyed = false;
  };

  void SetUp() override {
    GrowthTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* info) {
      info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
      info->memory_profile_count = 1;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t*, uint32_t, uint32_t,
           const amdf_memory_device_access_t*, amdf_memory_profile_t* profile,
           amdf_memory_access_capabilities_t*) {
          profile->roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                           AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
          profile->supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                     AMDF_MEMORY_FLAG_HOST_COHERENT |
                                     AMDF_MEMORY_FLAG_DEVICE_LOCAL;
          profile->host_mapping.supported_access =
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
          profile->allocation.minimum_alignment = 256;
          profile->allocation.byte_length_granularity = 256;
          profile->allocation.maximum_byte_length = UINT32_MAX;
          return AMDF_STATUS_OK;
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t*, const amdf_memory_profile_pair_query_t*,
           amdf_memory_pair_info_t* pair) {
          pair->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          pair->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          return AMDF_STATUS_OK;
        };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t*,
                                         amdf_host_cache_operation_t, uint64_t,
                                         uint64_t) { return AMDF_STATUS_OK; };
    context_.families = &native_family_;
    context_.target_info.gfx_ip.major = 11;
    fabric_->system_scope = memory_.scope;
    static const iree_hal_device_vtable_t vtable = {
        .destroy =
            [](iree_hal_device_t* device) {
              reinterpret_cast<Device*>(device)->destroyed = true;
            },
    };
    iree_hal_resource_initialize(&vtable, &device_.resource);
    iree_hal_queue_family_initialize(
        reinterpret_cast<iree_hal_device_t*>(&device_), 0, &family_spec_,
        &family_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_program_pool_initialize(
        &context_, 0, 3, false, iree_allocator_system(), &programs_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_context_join_memory(&context_);
    JoinPublication();
    iree_hal_command_buffer_release(command_buffer_);
    if (observed_buffer_caller_owned_) {
      iree_hal_buffer_release(&observed_buffer_.base);
    }
    iree_hal_memory_contract_release(observed_buffer_contract_);
    iree_hal_amd_gpu_pm4_program_pool_deinitialize(&programs_);
    iree_hal_device_release(reinterpret_cast<iree_hal_device_t*>(&device_));
    EXPECT_TRUE(device_.destroyed);
    GrowthTest::TearDown();
  }

  void RecordObservedResource() {
    const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    const iree_hal_buffer_binding_layout_t binding_layout = {
        sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(this, 2, &binding_layout,
                                                   iree_allocator_system(),
                                                   &observed_buffer_contract_));
    for (uint32_t scope = 0; scope < 2; ++scope) {
      observed_buffer_contract_->scopes[scope].interfaces =
          UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
      observed_buffer_contract_->scopes[scope]
          .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
      observed_buffer_contract_->scopes[scope].usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER;
    }
    family_.memory.domain = this;
    family_.memory.queue_scope_id = 0;
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy =
            [](iree_hal_buffer_t* buffer) {
              reinterpret_cast<ObservedBuffer*>(buffer)->destroyed = true;
            },
    };
    iree_hal_buffer_initialize(
        {}, &observed_buffer_.base, 4, 0, 4, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable,
        &observed_buffer_.base);
    observed_buffer_.base.memory = {
        .contract = observed_buffer_contract_,
        .bindings = &observed_buffer_binding_,
    };
    observed_buffer_caller_owned_ = true;
    const uint32_t pattern = 0;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer_, iree_hal_make_buffer_ref(&observed_buffer_.base, 0, 4),
        &pattern, sizeof(pattern), 0));
    iree_hal_buffer_release(&observed_buffer_.base);
    observed_buffer_caller_owned_ = false;
  }

  void ReuseDuringClone(bool fail_growth, bool before_handoff,
                        bool race_completion = false,
                        bool observe_recorded_resource = false) {
    const uint32_t completed = completed_ + 1;
    iree_hal_amd_gpu_pm4_barrier_profile_t profile;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0}, &profile));
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_command_buffer_create(
        &family_, &programs_, 0, IREE_HAL_COMMAND_CATEGORY_ANY, 0, 0, 0,
        &profile, iree_allocator_system(), &command_buffer_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer_));
    if (observe_recorded_resource) {
      ASSERT_NO_FATAL_FAILURE(RecordObservedResource());
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer_));
    iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_execute(
        &captures_, false, {}, {}, command_buffer_, {}, &profile, 0, &capture));
    iree_hal_amd_gpu_pm4_submission_t* previous = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_execute(
        &captures_, false, {}, {}, command_buffer_, {}, &profile, 0,
        &previous));
    auto* program = &capture->execute.program->resident;
    uint64_t frontier[3] = {};
    auto* initial =
        iree_hal_amd_gpu_pm4_program_try_acquire(program, 1, frontier);
    ASSERT_NE(initial, nullptr);
    previous->execute.instance = initial;
    iree_hal_amd_gpu_pm4_program_publish(program, initial, 1, 1);
    previous->execute.published = true;
    ASSERT_EQ(iree_hal_amd_gpu_pm4_program_try_acquire(program, 0, frontier),
              nullptr);

    hold_ = true;
    fail_ = fail_growth;
    ASSERT_TRUE(iree_hal_amd_gpu_publisher_enqueue(
        &publisher_, &capture->entry.publication));
    EXPECT_EQ(iree_hal_amd_gpu_publisher_next(&publisher_),
              &capture->entry.publication);
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_growth_begin(
        &growth_, IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_PROGRAM, capture, 0));
    entered_.acquire();
    // Only the two invocations and native growth now retain the recording.
    iree_hal_command_buffer_release(command_buffer_);
    command_buffer_ = nullptr;
    if (race_completion) {
      std::barrier<> start(2);
      std::thread retiring([&] {
        start.arrive_and_wait();
        iree_hal_amd_gpu_pm4_submission_release(previous);
      });
      Handoff();
      start.arrive_and_wait();
      resume_.release();
      consumed_.acquire();
      retiring.join();
    } else if (before_handoff) {
      // Publisher ownership remains with the caller while the return hint is
      // enqueued. Its later handoff must preserve that durable typed record.
      iree_hal_amd_gpu_pm4_submission_release(previous);
      Handoff();
    } else {
      Handoff();
      iree_hal_amd_gpu_pm4_submission_release(previous);
    }
    if (!race_completion) {
      consumed_.acquire();
      JoinPublication();
      EXPECT_EQ(completed_, completed);
      EXPECT_EQ(failed_, 0u);
      EXPECT_EQ(programs_.head, program);
      EXPECT_EQ(
          iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire),
          1);
      if (observe_recorded_resource) {
        EXPECT_TRUE(observed_buffer_.destroyed);
      }
      // The capture is gone, but the actual recorded program remains retained
      // while the clone allocation has not returned. A late failure is harmless
      // to that already-completed invocation.
      resume_.release();
    }
    growth_completed_.acquire();
    iree_hal_amd_gpu_context_join_memory(&context_);
    JoinPublication();
    iree_hal_amd_gpu_context_join_memory(&context_);
    EXPECT_EQ(programs_.head, nullptr);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 0);
    EXPECT_EQ(completed_, completed);
    EXPECT_EQ(failed_, 0u);
    EXPECT_EQ(live_, 1u);
  }

  // Device resource retained by real command-buffer construction/destruction.
  Device device_ = {};
  // Canonical family borrowed by the real recorder and invocation capture.
  iree_hal_queue_family_t family_ = {};
  // Immutable family row; this empty recording consumes no dispatch features.
  iree_hal_queue_family_spec_t family_spec_ = {};
  // Qualified native packet format used by builtin program construction.
  amdf_queue_family_info_t native_family_ = {};
  // Actual native placement, builtin code, and recorded-program registry.
  iree_hal_amd_gpu_pm4_program_pool_t programs_ = {};
  // Caller-owned recording reference, released while native growth is held.
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
  // Contract borrowed by the observed direct resource until recording release.
  iree_hal_memory_contract_t* observed_buffer_contract_ = nullptr;
  // Prepared program-site address retained only by the recording.
  iree_hal_buffer_native_binding_t observed_buffer_binding_ = {
      .device_address = 0x100000,
  };
  // Direct resource whose destruction establishes recording retirement.
  ObservedBuffer observed_buffer_ = {};
  // Whether fixture cleanup still owns the observed buffer's initial reference.
  bool observed_buffer_caller_owned_ = false;
};

TEST_F(ProgramGrowthTest, RecordedOwnerSurvivesEarlyReuseAndNativeSuccess) {
  ReuseDuringClone(false, false);
}

TEST_F(ProgramGrowthTest, RecordedOwnerSurvivesEarlyReuseAndNativeFailure) {
  ReuseDuringClone(true, false);
}

TEST_F(ProgramGrowthTest, ReturnDuringPublicationHandoffKeepsReuseReady) {
  ReuseDuringClone(false, true);
}

TEST_F(ProgramGrowthTest, ReturnDuringHandoffSurvivesLaterNativeFailure) {
  ReuseDuringClone(true, true);
}

TEST_F(ProgramGrowthTest, RecordedResourcesRetireBeforeUnusedCloneCompletes) {
  ReuseDuringClone(false, false, false, true);
}

TEST_F(ProgramGrowthTest,
       NativeResultRacesReturnedInstanceAndFinalRecordingRelease) {
  for (uint32_t i = 0; i < 128; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(ReuseDuringClone(false, false, true));
  }
}

TEST_F(GrowthTest, CoalescedArgumentsDoNotOwnTheReadyPublisher) {
  hold_ = true;
  Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS);
  entered_.acquire();
  Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS);
  EXPECT_EQ(allocations_, 1u);
  EXPECT_EQ(completed_, 0u);
  EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
  auto* warm = Capture();
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_enqueue(&publisher_,
                                                 &warm->entry.publication));
  Complete(iree_hal_amd_gpu_publisher_next(&publisher_), IREE_STATUS_OK);
  Handoff();
  consumed_.acquire();
  EXPECT_EQ(completed_, 1u);
  EXPECT_EQ(live_, 0u);
  resume_.release();
  consumed_.acquire();
  consumed_.acquire();
  iree_hal_amd_gpu_context_join_memory(&context_);
  iree_hal_amd_gpu_publication_deinitialize(&publication_);
  EXPECT_EQ(allocations_, 1u);
  EXPECT_EQ(completed_, 3u);
  EXPECT_NE(native_thread_, install_thread_);
}

TEST_F(GrowthTest, CoalescedSignalsAreInitializedOnlyAfterNativeCompletion) {
  hold_ = true;
  Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS);
  entered_.acquire();
  for (uint32_t i = 1; i < 64; ++i) {
    Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS);
  }
  EXPECT_EQ(allocations_, 1u);
  EXPECT_EQ(completed_, 0u);
  resume_.release();
  for (uint32_t i = 0; i < 64; ++i) {
    consumed_.acquire();
  }
  EXPECT_EQ(completed_, 64u);
  EXPECT_EQ(failed_, 0u);
}

TEST_F(GrowthTest, FailedNativeAcquisitionReturnsEveryCapturedOwner) {
  hold_ = true;
  fail_ = true;
  Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS);
  entered_.acquire();
  for (uint32_t i = 1; i < 1660; ++i) {
    Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS);
  }
  EXPECT_EQ(allocations_, 1u);
  resume_.release();
  for (uint32_t i = 0; i < 1660; ++i) {
    consumed_.acquire();
  }
  EXPECT_EQ(failed_, 1660u);
  EXPECT_EQ(completed_, 1660u);
}

TEST_F(GrowthTest, ScratchCohortRetiresOnNativeSuccessOrFailure) {
  for (bool fail : {false, true}) {
    SCOPED_TRACE(fail);
    hold_ = true;
    fail_ = fail;
    Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SCRATCH);
    entered_.acquire();
    for (uint32_t i = 1; i < 16; ++i) {
      Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SCRATCH);
    }
    EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
    resume_.release();
    for (uint32_t i = 0; i < 16; ++i) {
      consumed_.acquire();
    }
    iree_hal_amd_gpu_context_join_memory(&context_);
    JoinPublication();
    EXPECT_EQ(live_, 0u);
    EXPECT_EQ(allocations_, fail ? 2u : 1u);
    EXPECT_EQ(completed_, fail ? 32u : 16u);
    EXPECT_EQ(failed_, fail ? 16u : 0u);
  }
}

TEST_F(GrowthTest, ReturnedSignalResumesParkedWorkWhileAllocationIsHeld) {
  ReuseDuringGrowth(false, false);
}

TEST_F(GrowthTest, ReturnedSignalClosesMissBeforeEnrollmentRace) {
  ReuseDuringGrowth(false, true);
}

TEST_F(GrowthTest, LaterAllocationFailureDoesNotFailReusedWork) {
  ReuseDuringGrowth(true, false);
}

TEST_F(GrowthTest, EarlyReuseRetainsGrowthThroughLaterAllocationFailure) {
  ReuseDuringGrowth(true, true);
}

TEST_F(GrowthTest, CapacityReturnRacesNativeCompletionWithoutDuplicatingWork) {
  uint64_t value = 0;
  uint64_t frontier[3] = {};
  iree_hal_amd_gpu_signal_t metadata = {};
  iree_hal_amd_gpu_signal_block_t block;
  iree_hal_amd_gpu_signal_block_initialize(
      &signals_, this, &value, 0x1000, 8, 0, &metadata, 1, 3, frontier, &block);
  constexpr uint32_t kIterations = 128;
  constexpr uint32_t kCohortSize = 4;
  for (uint32_t i = 0; i < kIterations; ++i) {
    auto* held = iree_hal_amd_gpu_signal_pool_try_acquire(&signals_, 0);
    ASSERT_NE(held, nullptr);
    hold_ = true;
    for (uint32_t j = 0; j < kCohortSize; ++j) {
      Park(IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS);
    }
    entered_.acquire();
    std::barrier<> start(2);
    std::thread returning([&] {
      start.arrive_and_wait();
      iree_hal_amd_gpu_signal_release(held);
    });
    start.arrive_and_wait();
    resume_.release();
    for (uint32_t j = 0; j < kCohortSize; ++j) {
      consumed_.acquire();
    }
    growth_completed_.acquire();
    returning.join();
    iree_hal_amd_gpu_context_join_memory(&context_);
    JoinPublication();
    EXPECT_EQ(completed_, (i + 1) * kCohortSize);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 0);
  }
  EXPECT_EQ(failed_, 0u);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
}

TEST_F(GrowthTest, NativeCompletionRacesPublisherHandoffAndCaptureReuse) {
  for (uint32_t i = 0; i < 128; ++i) {
    Park(i % 2 ? IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_ARGUMENTS
               : IREE_HAL_AMD_GPU_PM4_ENTRY_GROW_SIGNALS);
    consumed_.acquire();
    // Capture release precedes the consumer's final FIFO handoff. The next
    // caller waits for ownership, without assuming the completion token joins
    // the entire deferred publisher. This is cold fixture synchronization.
    iree_hal_amd_gpu_context_join_memory(&context_);
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        1, &source_, {}, iree_allocator_system(), &publication_));
  }
  EXPECT_EQ(completed_, 128u);
  EXPECT_EQ(allocations_, 128u);
  EXPECT_EQ(failed_, 0u);
}

}  // namespace
