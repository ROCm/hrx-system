// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system_event.h"

#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/target_platform.h"
#include "iree/base/testing/dynamic_library_test_library_embed.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_LINUX)
#include <dlfcn.h>
#include <link.h>
#endif  // IREE_PLATFORM_LINUX

namespace iree::hal::amdgpu {
namespace {

// Lives in the test binary, so its module is the main program.
int PinProbeFunction() { return 0; }

// The pin is what lets the driver hand a callback pointer to a component that
// never gives it back, so its three outcomes have to stay distinguishable: a
// pin that happened, code the process has no way to unload, and a refusal.

// The main program cannot be unloaded, so an address in it needs no pin and
// must not be reported as one that was taken.
TEST(SystemEventPinTest, MainProgramNeedsNoPin) {
  iree_hal_amdgpu_module_pin_t pin = IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED;
  IREE_EXPECT_OK(iree_hal_amdgpu_system_event_pin_module_containing(
      reinterpret_cast<const void*>(&PinProbeFunction), &pin));
#if defined(IREE_PLATFORM_LINUX)
  EXPECT_EQ(pin, IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED);
#endif  // IREE_PLATFORM_LINUX
}

// An address in no loaded module at all is a refusal rather than a silent
// success: the caller asked for a lifetime guarantee that cannot be given.
TEST(SystemEventPinTest, AddressInNoModuleIsRefused) {
  int stack_local = 0;
  iree_hal_amdgpu_module_pin_t pin = IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_system_event_pin_module_containing(
                            reinterpret_cast<const void*>(&stack_local), &pin));
}

#if defined(IREE_PLATFORM_LINUX)

// dl_iterate_phdr callback stopping on the first object the loader named with
// an absolute path, storing the first byte of that object's first loadable
// segment through |user_data|. The loader names the main program with an empty
// string, and it names the vDSO, which it never read from a file, without a
// path, so a leading slash is enough to know the object came from a file the
// loader can name again.
static int VisitFirstObjectNamedByPath(struct dl_phdr_info* info,
                                       size_t info_size, void* user_data) {
  (void)info_size;
  if (info->dlpi_name == nullptr || info->dlpi_name[0] != '/') return 0;
  for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)* phdr = &info->dlpi_phdr[i];
    if (phdr->p_type != PT_LOAD) continue;
    *static_cast<const void**>(user_data) =
        reinterpret_cast<const void*>(info->dlpi_addr + phdr->p_vaddr);
    return 1;
  }
  return 0;
}

// An address in a shared object the loader mapped from a file classifies as a
// pin that was taken.
//
// The subject comes from the loader's own module list, not from a symbol
// lookup: a symbol resolved through the process-global scope can be answered by
// a definition in the main program, which is where a sanitizer runtime linked
// statically into the executable puts the dynamic linker interceptors it
// exports, and an address there is correctly classified as needing no pin. The
// module list also makes the subject's path a name the loader chose, so the pin
// has to re-resolve a path no caller here wrote.
TEST(SystemEventPinTest, SharedObjectIsPinned) {
  const void* address = nullptr;
  dl_iterate_phdr(VisitFirstObjectNamedByPath, &address);
  ASSERT_NE(address, nullptr)
      << "no loaded object is named with an absolute path, which is what a "
         "fully static link looks like from here";
  iree_hal_amdgpu_module_pin_t pin = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED;
  IREE_EXPECT_OK(
      iree_hal_amdgpu_system_event_pin_module_containing(address, &pin));
  EXPECT_EQ(pin, IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED);
}

// Writes a loadable shared object to |out_path| and returns an address inside
// it along with the handle holding it, so a caller can drop that handle and see
// whether the object is still mapped.
static void OpenDisposableSharedObject(const char* stem,
                                       iree::testing::TempFilePath* out_path,
                                       void** out_handle, void** out_address) {
  *out_handle = nullptr;
  *out_address = nullptr;
  *out_path = iree::testing::TempFilePath(stem, ".so");
  const struct iree_file_toc_t* file_toc =
      dynamic_library_test_library_create();
  IREE_ASSERT_OK(iree_io_file_contents_write(
      out_path->path_view(),
      iree_make_const_byte_span(file_toc->data, file_toc->size),
      iree_allocator_system()));
  void* handle = dlopen(out_path->path().c_str(), RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();
  void* address = dlsym(handle, "times_two");
  ASSERT_NE(address, nullptr);
  *out_handle = handle;
  *out_address = address;
}

// Returns whether the object at |path| is still mapped into this process.
// RTLD_NOLOAD answers without loading anything, so an unmapped object is a NULL
// rather than a fresh load.
static bool SharedObjectIsStillMapped(const iree::testing::TempFilePath& path) {
  void* handle = dlopen(path.path().c_str(), RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) return false;
  dlclose(handle);
  return true;
}

// The pin is a residency guarantee, not a classification: after it, the module
// stays mapped even though the code that asked for it holds no handle and the
// consumer has released the only one it had. The control half of this test is
// what makes that assertion mean something - an identical object that was never
// pinned is gone after the same release - so a pin that silently stopped
// pinning would fail here rather than pass on the classification alone.
TEST(SystemEventPinTest, PinnedSharedObjectSurvivesItsLastRelease) {
  iree::testing::TempFilePath control_path;
  void* control_handle = nullptr;
  void* control_address = nullptr;
  OpenDisposableSharedObject("iree_amdgpu_pin_control", &control_path,
                             &control_handle, &control_address);
  ASSERT_NE(control_handle, nullptr);
  EXPECT_EQ(dlclose(control_handle), 0);
  EXPECT_FALSE(SharedObjectIsStillMapped(control_path))
      << "an unpinned object outlived its last handle, so this test cannot "
         "tell a pin from a leak";

  iree::testing::TempFilePath pinned_path;
  void* pinned_handle = nullptr;
  void* pinned_address = nullptr;
  OpenDisposableSharedObject("iree_amdgpu_pin_subject", &pinned_path,
                             &pinned_handle, &pinned_address);
  ASSERT_NE(pinned_handle, nullptr);
  iree_hal_amdgpu_module_pin_t pin = IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_system_event_pin_module_containing(pinned_address, &pin));
  ASSERT_EQ(pin, IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED);
  EXPECT_EQ(dlclose(pinned_handle), 0);
  EXPECT_TRUE(SharedObjectIsStillMapped(pinned_path));

  control_path.Remove();
  pinned_path.Remove();
}

#endif  // IREE_PLATFORM_LINUX

}  // namespace
}  // namespace iree::hal::amdgpu

// The HSA callback is reached by capturing the pointer this driver hands to
// hsa_amd_register_system_event_handler, which requires a thunk in the libhsa
// function-pointer table. That table only exists in the default dynamic build.
#if !IREE_HAL_AMDGPU_LIBHSA_STATIC

namespace iree::hal::amdgpu {
namespace {

// Agent handles used by the fabricated devices below. Distinct from zero so a
// zero-filled struct never matches one by accident.
constexpr uint64_t kAgentHandleA = 0x1001u;
constexpr uint64_t kAgentHandleB = 0x1002u;
constexpr uint64_t kAgentHandleUnknown = 0x2001u;

constexpr uint64_t kFaultAddress = 0xDEAD0000u;
constexpr uint64_t kSecondFaultAddress = 0xBEEF0000u;

// The system event callback captured from the driver's registration call, plus
// the user data it registered with.
hsa_amd_system_event_callback_t g_captured_callback = nullptr;
void* g_captured_data = nullptr;
// Number of times the driver has installed its handler with the HSA runtime.
// The driver registers once per HSA lifetime, so this only advances when a
// shutdown event has told it the runtime's callback registry is gone.
int g_registration_count = 0;

static hsa_status_t HSA_API CaptureSystemEventHandler(
    hsa_amd_system_event_callback_t callback, void* data) {
  g_captured_callback = callback;
  g_captured_data = data;
  ++g_registration_count;
  return HSA_STATUS_SUCCESS;
}

// A libhsa whose only populated thunk is the system event registration this
// driver calls; every other entry stays NULL.
static iree_hal_amdgpu_libhsa_t MakeCapturingLibhsa() {
  iree_hal_amdgpu_libhsa_t libhsa = {};
  libhsa.hsa_amd_register_system_event_handler = CaptureSystemEventHandler;
  return libhsa;
}

static hsa_agent_t MakeAgent(uint64_t handle) {
  hsa_agent_t agent;
  agent.handle = handle;
  return agent;
}

// Fabricates the smallest logical device the registration path reads: a
// physical device per agent carrying only its agent handle, and the sticky
// failure status the callback latches.
//
// The delivery path itself reads none of this beyond the sticky status; queue
// delivery is bounded by the registration's own targets.
class FakeLogicalDevice {
 public:
  void Initialize(const uint64_t* agent_handles, iree_host_size_t agent_count) {
    const iree_host_size_t device_size =
        sizeof(iree_hal_amdgpu_logical_device_t) +
        agent_count * sizeof(iree_hal_amdgpu_physical_device_t*);
    IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(), device_size,
                                        (void**)&device_));
    memset(device_, 0, device_size);
    device_->host_allocator = iree_allocator_system();
    device_->physical_device_count = agent_count;
    for (iree_host_size_t i = 0; i < agent_count; ++i) {
      IREE_CHECK_OK(iree_allocator_malloc(
          iree_allocator_system(), sizeof(iree_hal_amdgpu_physical_device_t),
          (void**)&device_->physical_devices[i]));
      memset(device_->physical_devices[i], 0,
             sizeof(iree_hal_amdgpu_physical_device_t));
      device_->physical_devices[i]->device_agent = MakeAgent(agent_handles[i]);
    }
  }

  ~FakeLogicalDevice() {
    if (!device_) return;
    iree_status_free((iree_status_t)iree_atomic_exchange(
        &device_->failure_status, 0, iree_memory_order_acq_rel));
    for (iree_host_size_t i = 0; i < device_->physical_device_count; ++i) {
      iree_allocator_free(iree_allocator_system(),
                          device_->physical_devices[i]);
    }
    iree_allocator_free(iree_allocator_system(), device_);
  }

  iree_hal_amdgpu_logical_device_t* device() const { return device_; }

  iree_status_code_t FailureStatusCode() const {
    return iree_status_code((iree_status_t)iree_atomic_load(
        &device_->failure_status, iree_memory_order_acquire));
  }

  bool FailureStatusMentions(const char* text) const {
    return StatusMentions(
        (iree_status_t)iree_atomic_load(&device_->failure_status,
                                        iree_memory_order_acquire),
        text);
  }

  static bool StatusMentions(const iree_status_t status, const char* text) {
    if (iree_status_is_ok(status)) return false;
    iree_allocator_t host_allocator = iree_allocator_system();
    char* buffer = NULL;
    iree_host_size_t buffer_length = 0;
    if (!iree_status_to_string(status, &host_allocator, &buffer,
                               &buffer_length)) {
      return false;
    }
    const bool mentions = strstr(buffer, text) != nullptr;
    iree_allocator_free(host_allocator, buffer);
    return mentions;
  }

 private:
  // Fabricated device with its inline physical device pointer array.
  iree_hal_amdgpu_logical_device_t* device_ = NULL;
};

// A block of zero-filled host queues usable as failure delivery targets.
//
// Failing a queue is an atomic store plus a stop-signal store that is skipped
// when the queue has no stop signal, so a zero-filled queue records the failure
// without touching HSA.
class FakeHostQueues {
 public:
  explicit FakeHostQueues(iree_host_size_t count) : count_(count) {
    IREE_CHECK_OK(iree_allocator_malloc(
        iree_allocator_system(), count * sizeof(iree_hal_amdgpu_host_queue_t),
        (void**)&queues_));
    memset(queues_, 0, count * sizeof(iree_hal_amdgpu_host_queue_t));
  }

  ~FakeHostQueues() {
    for (iree_host_size_t i = 0; i < count_; ++i) {
      iree_status_free((iree_status_t)iree_atomic_exchange(
          &queues_[i].error_status, 0, iree_memory_order_acq_rel));
    }
    iree_allocator_free(iree_allocator_system(), queues_);
  }

  iree_hal_amdgpu_host_queue_t* queues() const { return queues_; }
  iree_host_size_t count() const { return count_; }

  iree_status_code_t ErrorStatusCode(iree_host_size_t index) const {
    return iree_status_code((iree_status_t)iree_atomic_load(
        &queues_[index].error_status, iree_memory_order_acquire));
  }

  bool ErrorStatusMentions(iree_host_size_t index, const char* text) const {
    return FakeLogicalDevice::StatusMentions(
        (iree_status_t)iree_atomic_load(&queues_[index].error_status,
                                        iree_memory_order_acquire),
        text);
  }

 private:
  // Zero-filled queue storage published as delivery targets.
  iree_hal_amdgpu_host_queue_t* queues_ = NULL;
  // Number of queues in |queues_|.
  iree_host_size_t count_ = 0;
};

class SystemEventTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // The driver registers with HSA once per process, so the callback can only
    // be captured by the first registration in this binary.
    const uint64_t agent_handles[] = {kAgentHandleA};
    FakeLogicalDevice device;
    device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
    iree_hal_amdgpu_libhsa_t libhsa = MakeCapturingLibhsa();
    iree_hal_amdgpu_system_event_registration_t* registration = NULL;
    IREE_ASSERT_OK(iree_hal_amdgpu_system_event_register_device(
        &libhsa, device.device(), iree_allocator_system(), &registration));
    iree_hal_amdgpu_system_event_unregister_device(registration);
    ASSERT_NE(g_captured_callback, nullptr);
  }

  // Registers |device| and returns its registration. Every registration a test
  // creates must be unregistered before the fabricated device is destroyed:
  // the registry is process-global and outlives the test.
  static iree_hal_amdgpu_system_event_registration_t* Register(
      const FakeLogicalDevice& device) {
    iree_hal_amdgpu_libhsa_t libhsa = MakeCapturingLibhsa();
    iree_hal_amdgpu_system_event_registration_t* registration = NULL;
    IREE_CHECK_OK(iree_hal_amdgpu_system_event_register_device(
        &libhsa, device.device(), iree_allocator_system(), &registration));
    return registration;
  }

  static hsa_status_t DispatchMemoryError(uint64_t agent_handle,
                                          uint64_t virtual_address) {
    hsa_amd_event_t event = {};
    event.event_type = HSA_AMD_GPU_MEMORY_ERROR_EVENT;
    event.memory_error.agent = MakeAgent(agent_handle);
    event.memory_error.virtual_address = virtual_address;
    event.memory_error.error_reason_mask = 0;
    return g_captured_callback(&event, g_captured_data);
  }

  static hsa_status_t DispatchSystemShutdown() {
    hsa_amd_event_t event = {};
    event.event_type = HSA_AMD_SYSTEM_SHUTDOWN_EVENT;
    return g_captured_callback(&event, g_captured_data);
  }

  static hsa_status_t DispatchMemoryFault(uint64_t agent_handle,
                                          uint64_t virtual_address) {
    hsa_amd_event_t event = {};
    event.event_type = HSA_AMD_GPU_MEMORY_FAULT_EVENT;
    event.memory_fault.agent = MakeAgent(agent_handle);
    event.memory_fault.virtual_address = virtual_address;
    event.memory_fault.fault_reason_mask = 0;
    return g_captured_callback(&event, g_captured_data);
  }
};

// An event for an agent no registration holds is not claimed, leaving the HSA
// runtime's abort fallback in place.
TEST_F(SystemEventTest, UnmatchedEventIsNotClaimed) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleUnknown, kFaultAddress),
            HSA_STATUS_ERROR);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_OK);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// A matched event fails every published queue of the agent and latches the
// device's sticky failure status, carrying the fault detail into both.
TEST_F(SystemEventTest, MatchedEventFailsPublishedQueues) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(3);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  for (iree_host_size_t i = 0; i < queues.count(); ++i) {
    EXPECT_EQ(queues.ErrorStatusCode(i), IREE_STATUS_ABORTED);
    EXPECT_TRUE(queues.ErrorStatusMentions(i, "00000000dead0000"));
  }
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_ABORTED);
  EXPECT_TRUE(device.FailureStatusMentions("00000000dead0000"));

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// A registration created before frontier assignment has no queue targets. It
// still claims the event, because the device's sticky status can observe it.
TEST_F(SystemEventTest, UnpublishedRegistrationClaimsWithoutFailingQueues) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_ABORTED);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// One event fails every published queue of the registration that holds its
// agent, not only the queues of that agent. Queues of one logical device share
// an epoch signal table spanning all of its agents, so a queue on a healthy
// agent can be waiting device-side on an epoch signal a faulted agent will
// never advance; leaving it live would strand its teardown forever.
TEST_F(SystemEventTest, FanoutCoversEveryAgentOfTheRegistration) {
  const uint64_t agent_handles[] = {kAgentHandleA, kAgentHandleB};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues_a(2);
  FakeHostQueues queues_b(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues_a.queues(), queues_a.count());
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleB)),
      queues_b.queues(), queues_b.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleB, kFaultAddress),
            HSA_STATUS_SUCCESS);
  for (iree_host_size_t i = 0; i < queues_a.count(); ++i) {
    EXPECT_EQ(queues_a.ErrorStatusCode(i), IREE_STATUS_ABORTED);
    EXPECT_TRUE(queues_a.ErrorStatusMentions(i, "00000000dead0000"));
  }
  for (iree_host_size_t i = 0; i < queues_b.count(); ++i) {
    EXPECT_EQ(queues_b.ErrorStatusCode(i), IREE_STATUS_ABORTED);
  }
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_ABORTED);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// The fanout stops at the registration boundary: a registration that holds no
// target for the event's agent is left entirely alone, queues and sticky status
// alike.
TEST_F(SystemEventTest, FanoutStopsAtTheRegistrationBoundary) {
  const uint64_t faulting_handles[] = {kAgentHandleA};
  const uint64_t other_handles[] = {kAgentHandleB};
  FakeLogicalDevice faulting_device;
  faulting_device.Initialize(faulting_handles,
                             IREE_ARRAYSIZE(faulting_handles));
  FakeLogicalDevice other_device;
  other_device.Initialize(other_handles, IREE_ARRAYSIZE(other_handles));
  FakeHostQueues faulting_queues(2);
  FakeHostQueues other_queues(2);
  iree_hal_amdgpu_system_event_registration_t* faulting_registration =
      Register(faulting_device);
  iree_hal_amdgpu_system_event_registration_t* other_registration =
      Register(other_device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          faulting_registration, MakeAgent(kAgentHandleA)),
      faulting_queues.queues(), faulting_queues.count());
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          other_registration, MakeAgent(kAgentHandleB)),
      other_queues.queues(), other_queues.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(faulting_queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);
  EXPECT_EQ(faulting_queues.ErrorStatusCode(1), IREE_STATUS_ABORTED);
  EXPECT_EQ(other_queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(other_queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(faulting_device.FailureStatusCode(), IREE_STATUS_ABORTED);
  EXPECT_EQ(other_device.FailureStatusCode(), IREE_STATUS_OK);

  iree_hal_amdgpu_system_event_unregister_device(other_registration);
  iree_hal_amdgpu_system_event_unregister_device(faulting_registration);
}

// Retiring an agent's queue targets stops queue delivery, but the device is
// still reachable through the HAL and its sticky status is still somewhere the
// fault can be reported, so the event is still claimed. That is the state a
// device sits in between a frontier deassignment and its next assignment.
TEST_F(SystemEventTest, RetiredTargetsStopQueueDeliveryAndStillClaim) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_agent_target_t* target =
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA));
  iree_hal_amdgpu_system_event_publish_queue_targets(target, queues.queues(),
                                                     queues.count());
  iree_hal_amdgpu_system_event_retire_queue_targets(target);
  // Retirement is idempotent without any record that it already ran.
  iree_hal_amdgpu_system_event_retire_queue_targets(target);

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_ABORTED);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// Device teardown retires the sticky status first and the queue targets after,
// because a fault arriving in between is what releases the teardown waits. In
// that window the queues are still the delivery target and the event is still
// claimed; only the status the device will never read again is given up.
TEST_F(SystemEventTest, RetiredDeviceStatusStillDeliversToPublishedQueues) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());
  iree_hal_amdgpu_system_event_retire_device_status(registration);

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_ABORTED);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_OK);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// A registration with both retired holds nowhere to deliver: its queues are
// about to be destroyed and its sticky slot is about to be emptied and freed.
// Latching that slot anyway would suppress the runtime's abort for the whole
// process while reporting the fault to nobody, so the event goes unclaimed -
// the same answer the registry gives once the registration is gone entirely.
TEST_F(SystemEventTest, FullyRetiredRegistrationDoesNotClaim) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_agent_target_t* target =
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA));
  iree_hal_amdgpu_system_event_publish_queue_targets(target, queues.queues(),
                                                     queues.count());
  iree_hal_amdgpu_system_event_retire_device_status(registration);
  iree_hal_amdgpu_system_event_retire_queue_targets(target);
  // Retirement is idempotent without any record that it already ran.
  iree_hal_amdgpu_system_event_retire_device_status(registration);

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_ERROR);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_OK);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// Once the last registration for an agent is removed nothing can observe the
// event, so it must not be claimed: the HSA runtime's abort is the only
// remaining report.
TEST_F(SystemEventTest, RemovedRegistrationDoesNotClaim) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_agent_target_t* target =
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA));
  iree_hal_amdgpu_system_event_publish_queue_targets(target, queues.queues(),
                                                     queues.count());
  iree_hal_amdgpu_system_event_retire_queue_targets(target);
  iree_hal_amdgpu_system_event_unregister_device(registration);

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_ERROR);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_OK);
}

// Both the per-queue slot and the sticky device slot keep the first failure
// they were given.
TEST_F(SystemEventTest, FirstFailureWinsAcrossRepeatedEvents) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(1);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kSecondFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_TRUE(queues.ErrorStatusMentions(0, "00000000dead0000"));
  EXPECT_FALSE(queues.ErrorStatusMentions(0, "00000000beef0000"));
  EXPECT_TRUE(device.FailureStatusMentions("00000000dead0000"));
  EXPECT_FALSE(device.FailureStatusMentions("00000000beef0000"));

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// Two logical devices can hold the same agent. Both receive delivery, and
// removing one leaves the other fully live.
TEST_F(SystemEventTest, RegistrationsSharingAnAgentAreIndependent) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice first_device;
  first_device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeLogicalDevice second_device;
  second_device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues first_queues(1);
  FakeHostQueues second_queues(1);
  iree_hal_amdgpu_system_event_registration_t* first_registration =
      Register(first_device);
  iree_hal_amdgpu_system_event_registration_t* second_registration =
      Register(second_device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          first_registration, MakeAgent(kAgentHandleA)),
      first_queues.queues(), first_queues.count());
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          second_registration, MakeAgent(kAgentHandleA)),
      second_queues.queues(), second_queues.count());

  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(first_queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);
  EXPECT_EQ(second_queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);
  EXPECT_EQ(first_device.FailureStatusCode(), IREE_STATUS_ABORTED);
  EXPECT_EQ(second_device.FailureStatusCode(), IREE_STATUS_ABORTED);

  // Removing the first registration must leave the second delivering.
  iree_hal_amdgpu_system_event_unregister_device(first_registration);
  FakeHostQueues late_queues(1);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          second_registration, MakeAgent(kAgentHandleA)),
      late_queues.queues(), late_queues.count());
  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kSecondFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(late_queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);

  iree_hal_amdgpu_system_event_unregister_device(second_registration);
}

// Lookup answers only for agents the registration actually holds.
TEST_F(SystemEventTest, AgentLookupAnswersOnlyForHeldAgents) {
  const uint64_t agent_handles[] = {kAgentHandleA, kAgentHandleB};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);

  EXPECT_NE(iree_hal_amdgpu_system_event_registration_lookup_agent(
                registration, MakeAgent(kAgentHandleA)),
            nullptr);
  EXPECT_NE(iree_hal_amdgpu_system_event_registration_lookup_agent(
                registration, MakeAgent(kAgentHandleB)),
            nullptr);
  EXPECT_EQ(iree_hal_amdgpu_system_event_registration_lookup_agent(
                registration, MakeAgent(kAgentHandleUnknown)),
            nullptr);
  EXPECT_EQ(iree_hal_amdgpu_system_event_registration_lookup_agent(
                NULL, MakeAgent(kAgentHandleA)),
            nullptr);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// Memory error events are allocator consistency failures raised inside a free,
// not GPU faults, so they are neither claimed nor converted into device
// failures even when they name a registered agent.
TEST_F(SystemEventTest, MemoryErrorEventIsNotClaimed) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));
  FakeHostQueues queues(2);
  iree_hal_amdgpu_system_event_registration_t* registration = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          registration, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());

  EXPECT_EQ(DispatchMemoryError(kAgentHandleA, kFaultAddress),
            HSA_STATUS_ERROR);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
  EXPECT_EQ(queues.ErrorStatusCode(1), IREE_STATUS_OK);
  EXPECT_EQ(device.FailureStatusCode(), IREE_STATUS_OK);

  iree_hal_amdgpu_system_event_unregister_device(registration);
}

// The HSA runtime destroys its callback registry in the final hsa_shut_down and
// does not restore it, so a process that reinitializes HSA must install the
// handler again. The shutdown event is the only notice of that, and clearing
// the driver's record of having registered is the only thing standing between a
// reinitialized process and permanently dead fault delivery.
TEST_F(SystemEventTest, HsaShutdownRearmsHandlerRegistration) {
  const uint64_t agent_handles[] = {kAgentHandleA};
  FakeLogicalDevice device;
  device.Initialize(agent_handles, IREE_ARRAYSIZE(agent_handles));

  // Registrations within one HSA lifetime share the one installed handler.
  const int baseline = g_registration_count;
  iree_hal_amdgpu_system_event_registration_t* first = Register(device);
  EXPECT_EQ(g_registration_count, baseline);
  iree_hal_amdgpu_system_event_unregister_device(first);

  EXPECT_EQ(DispatchSystemShutdown(), HSA_STATUS_SUCCESS);

  iree_hal_amdgpu_system_event_registration_t* second = Register(device);
  EXPECT_EQ(g_registration_count, baseline + 1);
  iree_hal_amdgpu_system_event_unregister_device(second);

  // Delivery still works through the reinstalled handler.
  FakeHostQueues queues(1);
  iree_hal_amdgpu_system_event_registration_t* third = Register(device);
  iree_hal_amdgpu_system_event_publish_queue_targets(
      iree_hal_amdgpu_system_event_registration_lookup_agent(
          third, MakeAgent(kAgentHandleA)),
      queues.queues(), queues.count());
  EXPECT_EQ(DispatchMemoryFault(kAgentHandleA, kFaultAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_ABORTED);
  iree_hal_amdgpu_system_event_unregister_device(third);
}

// Publication and retirement accept a NULL target or registration so a device
// that never registered still assigns, deassigns and tears down.
TEST_F(SystemEventTest, NullTargetOperationsAreNoOps) {
  FakeHostQueues queues(1);
  iree_hal_amdgpu_system_event_publish_queue_targets(NULL, queues.queues(),
                                                     queues.count());
  iree_hal_amdgpu_system_event_retire_queue_targets(NULL);
  iree_hal_amdgpu_system_event_retire_device_status(NULL);
  iree_hal_amdgpu_system_event_unregister_device(NULL);
  EXPECT_EQ(queues.ErrorStatusCode(0), IREE_STATUS_OK);
}

}  // namespace
}  // namespace iree::hal::amdgpu

#else

namespace iree::hal::amdgpu {
namespace {

// A static HSA link has no function-pointer table to install a thunk into, so
// the callback pointer cannot be captured and none of the delivery coverage
// above is buildable. Report that as a skip rather than leaving it looking as
// though it ran. The module pin coverage does not touch HSA and still runs.
TEST(SystemEventTest, DeliveryCoverageRequiresDynamicLibhsa) {
  GTEST_SKIP() << "system event delivery coverage requires the dynamic libhsa "
                  "function table";
}

}  // namespace
}  // namespace iree::hal::amdgpu

#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC
