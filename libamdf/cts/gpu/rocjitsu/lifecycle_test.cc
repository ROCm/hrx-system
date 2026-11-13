// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <signal.h>
#include <sys/resource.h>

#include <array>
#include <barrier>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "amdf/rocjitsu.h"
#include "libamdf/cts/gpu/aql/encoding/packets.h"
#include "libamdf/cts/gpu/aql/queue/publication.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/pm4/encoding/profile.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "libamdf/cts/gpu/util/memory.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace {

class ScopedRocjitsuLibraryPath {
 public:
  ScopedRocjitsuLibraryPath() {
    const char* value = std::getenv("LIBAMDF_ROCJITSU_PATH");
    if (value != nullptr) {
      value_ = value;
    }
  }

  ~ScopedRocjitsuLibraryPath() {
    EXPECT_EQ(value_.has_value()
                  ? setenv("LIBAMDF_ROCJITSU_PATH", value_->c_str(), 1)
                  : unsetenv("LIBAMDF_ROCJITSU_PATH"),
              0);
  }

 private:
  // Original override, preserving the distinction between absent and empty.
  std::optional<std::string> value_;
};

class ScopedDescriptorBudget {
 public:
  explicit ScopedDescriptorBudget(const struct rlimit& original)
      : original_(original) {}

  ~ScopedDescriptorBudget() {
    EXPECT_EQ(setrlimit(RLIMIT_NOFILE, &original_), 0);
  }

 private:
  // Host limit restored even when an assertion exits the test body.
  struct rlimit original_;
};

class RocjitsuTest : public ::testing::Test {
 protected:
  struct Device {
    // Endpoint retained until its device is destroyed.
    amdf_endpoint_t* endpoint = nullptr;
    // Live device retained through all queue and allocation releases.
    amdf_device_t* handle = nullptr;
    // Independent queues exercising shared doorbell backing.
    std::array<GpuUserQueue, 2> queues;
    // Native completion event retained until both queues retire.
    amdf_gpu_event_t* event = nullptr;
    // Immutable device-local native event producer operands.
    amdf_gpu_event_info_t event_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
        .structure_size = sizeof(amdf_gpu_event_info_t),
    };
    // Coherent output retained through final queue retirement.
    GpuMemory output;
    // Immutable packet recipe selected from public endpoint facts.
    const Pm4CommandProfile* profile = nullptr;
  };

  struct Machine {
    // Independent instance owning its configured VM.
    amdf_instance_t* instance = nullptr;
    // Stable device records, released before instance destruction.
    std::vector<std::unique_ptr<Device>> devices;
  };

  void SetUp() override {
    const auto& cache = GetCtsDeviceCache();
    if (!cache.has_rocjitsu_library()) {
      GTEST_SKIP()
          << "explicit rocjitsu library and multi-GPU configuration required";
    }
    library_ = cache.rocjitsu_library();
    std::ifstream file(cache.rocjitsu_configurations().front());
    ASSERT_TRUE(file.is_open());
    configuration_.assign(std::istreambuf_iterator<char>(file), {});
    ASSERT_FALSE(file.bad());
    for (size_t i = 0; i < signals_.size(); ++i) {
      ASSERT_EQ(sigaction(kSignals[i], nullptr, &signals_[i]), 0);
    }
    ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &descriptors_), 0);
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    const void* extension = nullptr;
    ASSERT_EQ(api_->query_extension(AMDF_EXTENSION_ROCJITSU, 1, 1, &extension),
              AMDF_STATUS_OK);
    simulator_ = static_cast<const amdf_rocjitsu_api_t*>(extension);
    ASSERT_EQ(
        api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_3,
                              AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
        AMDF_STATUS_OK);
    gpu_ = static_cast<const amdf_gpu_api_t*>(extension);
    AssertLibraryLoaded(false);
  }

  void TearDown() override {
    if (api_ == nullptr) {
      return;
    }
    for (auto& machine : machines_) {
      ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machine));
    }
    for (size_t i = 0; i < signals_.size(); ++i) {
      struct sigaction current = {};
      ASSERT_EQ(sigaction(kSignals[i], nullptr, &current), 0);
      EXPECT_EQ(current.sa_sigaction, signals_[i].sa_sigaction);
      EXPECT_EQ(current.sa_flags, signals_[i].sa_flags);
      for (int signal = 1; signal < NSIG; ++signal) {
        EXPECT_EQ(sigismember(&current.sa_mask, signal),
                  sigismember(&signals_[i].sa_mask, signal));
      }
    }
    struct rlimit current;
    ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &current), 0);
    EXPECT_EQ(current.rlim_cur, descriptors_.rlim_cur);
    EXPECT_EQ(current.rlim_max, descriptors_.rlim_max);
    AssertLibraryLoaded(false);
  }

  void AssertLibraryLoaded(bool expected) {
    void* handle = dlopen(library_.c_str(), RTLD_NOW | RTLD_NOLOAD);
    EXPECT_EQ(handle != nullptr, expected);
    if (handle != nullptr) {
      ASSERT_EQ(dlclose(handle), 0);
    }
  }

  amdf_rocjitsu_machine_create_info_t Configuration() const {
    return {
        .type = AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO,
        .structure_size = sizeof(amdf_rocjitsu_machine_create_info_t),
        .library_path = library_.c_str(),
        .configuration_json = configuration_.c_str(),
    };
  }

  void CreateInstance(Machine& machine) {
    const amdf_instance_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .structure_size = sizeof(create),
        .native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE,
    };
    ASSERT_EQ(api_->instance_create(&create, &machine.instance),
              AMDF_STATUS_OK);
  }

  void Enumerate(Machine& machine,
                 std::vector<amdf_endpoint_summary_t>& summaries) {
    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(machine.instance, 0, nullptr, &count),
              AMDF_STATUS_OK);
    summaries.resize(count);
    ASSERT_EQ(api_->endpoint_enumerate(machine.instance, count,
                                       summaries.data(), &count),
              AMDF_STATUS_OK);
    ASSERT_EQ(count, summaries.size());
  }

  void Activate(Machine& machine, amdf_queue_command_type_t command_type) {
    const auto configuration = Configuration();
    ASSERT_EQ(
        simulator_->instance_add_machine(machine.instance, &configuration),
        AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> summaries;
    ASSERT_NO_FATAL_FAILURE(Enumerate(machine, summaries));
    uint32_t scope_count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  machine.instance, 0, nullptr, &scope_count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    std::vector<amdf_memory_scope_t*> scopes(scope_count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(
                  machine.instance, scope_count, scopes.data(), &scope_count),
              AMDF_STATUS_OK);
    amdf_memory_scope_t* system = nullptr;
    for (auto* scope : scopes) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        system = scope;
      }
    }
    ASSERT_NE(system, nullptr);
    for (const auto& summary : summaries) {
      ASSERT_NE(summary.type_flags & AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE,
                0u);
      Device& device =
          *machine.devices.emplace_back(std::make_unique<Device>());
      ASSERT_EQ(
          api_->endpoint_open(machine.instance, &summary.id, &device.endpoint),
          AMDF_STATUS_OK);
      amdf_gpu_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(gpu_->endpoint_query_info(device.endpoint, &info),
                AMDF_STATUS_OK);
      amdf_queue_family_info_t family = {};
      bool matches = false;
      ASSERT_EQ(FindGpuQueueFamily(api_, device.endpoint,
                                   {
                                       .command_type = command_type,
                                       .roles = AMDF_QUEUE_ROLE_COMPUTE,
                                   },
                                   &family, &matches),
                AMDF_STATUS_OK);
      if (!matches) {
        GTEST_SKIP() << "configured GPU has no requested queue protocol";
      }
      if (command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
        device.profile = Pm4CommandProfile::Find(info);
        ASSERT_NE(device.profile, nullptr);
        EXPECT_EQ((family.format_features &
                   AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH) != 0,
                  info.topology.xcc_count == 1);
      }
      const amdf_gpu_device_create_info_t create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
          .structure_size = sizeof(create),
      };
      ASSERT_EQ(gpu_->device_create(device.endpoint, &create, &device.handle),
                AMDF_STATUS_OK);
      const amdf_gpu_event_create_info_t event_create = {
          .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
          .structure_size = sizeof(event_create),
      };
      ASSERT_EQ(gpu_->event_create(device.handle, &event_create, &device.event),
                AMDF_STATUS_OK);
      ASSERT_EQ(gpu_->event_query_info(device.event, &device.event_info),
                AMDF_STATUS_OK);
      ASSERT_EQ(device.event_info.format, AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL);
      ASSERT_EQ(device.event_info.format_version, 1u);
      for (auto& queue : device.queues) {
        ASSERT_NO_FATAL_FAILURE(
            queue.Initialize(api_, gpu_, device.handle, family,
                             AMDF_QUEUE_PRODUCER_MODE_SINGLE, {}));
      }
      const amdf_memory_device_access_t access = {
          device.handle,
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                       AMDF_MEMORY_FLAG_HOST_COHERENT,
          },
      };
      const uint32_t ordinal = FindGpuMemoryProfileOrdinal(
          api_, system, device.handle,
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
          AMDF_MEMORY_FLAG_HOST_VISIBLE, access.requirements);
      ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
      const amdf_memory_create_info_t allocation = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(allocation),
          .memory_profile_ordinal = ordinal,
          .access_count = 1,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .byte_length = 4096,
          .minimum_alignment = 65536,
          .accesses = &access,
      };
      ASSERT_NO_FATAL_FAILURE(
          device.output.Initialize(api_, system, allocation));
      std::memset(device.output.host.pointer, 0, 4096);
    }
    ASSERT_GE(machine.devices.size(), 2u);
  }

  void Execute(Machine& machine, uint32_t queue_ordinal, uint32_t epoch) {
    for (size_t i = 0; i < machine.devices.size(); ++i) {
      auto& device = *machine.devices[i];
      if (device.queues[queue_ordinal].info.command_type ==
          AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
        ASSERT_NO_FATAL_FAILURE(ExecuteAql(device, queue_ordinal, epoch));
      } else {
        const uint32_t value = epoch * 0x7139b25du + static_cast<uint32_t>(i);
        ASSERT_NO_FATAL_FAILURE(ExecutePm4(device, queue_ordinal, value));
      }
    }
  }

  void ExecuteAql(Device& device, uint32_t queue_ordinal, uint32_t epoch) {
    auto& queue = device.queues[queue_ordinal];
    auto& signal =
        static_cast<aql::Signal*>(device.output.host.pointer)[queue_ordinal];
    signal = aql::UserSignal(epoch);
    signal.event.mailbox_address = device.event_info.mailbox_address;
    signal.event.id = device.event_info.interrupt_context;
    const uint64_t first =
        GpuLoadAcquire<uint64_t>(queue.host.write_index_address);
    GpuStoreRelease(queue.host.write_index_address, first + 1);
    ASSERT_TRUE(aql::Publish(
        queue, first,
        aql::Barrier(
            aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
            device.output.device_address + queue_ordinal * sizeof(signal))));
    do {
      ASSERT_EQ(gpu_->event_wait(device.event, AMDF_TIMEOUT_INFINITE),
                AMDF_STATUS_OK);
    } while (GpuLoadAcquire<int64_t>(
                 reinterpret_cast<uintptr_t>(&signal.value)) != epoch - 1);
    ASSERT_NO_FATAL_FAILURE(queue.WaitConsumed(api_, first + 1));
  }

  void ExecutePm4(Device& device, uint32_t queue_ordinal, uint32_t value) {
    auto& queue = device.queues[queue_ordinal];
    auto* words = reinterpret_cast<uint32_t*>(queue.host.ring_address);
    const uint64_t first =
        GpuLoadAcquire<uint64_t>(queue.host.write_index_address);
    Pm4CommandWriter commands(words + first, *device.profile);
    commands.SystemBarrier();
    commands.ReleaseSystem32(device.output.device_address + queue_ordinal * 64,
                             value);
    commands.ReleaseSystemInterrupt32(device.event_info.mailbox_address,
                                      device.event_info.trigger_data,
                                      device.event_info.interrupt_context);
    commands.PadToEightWords();
    const uint64_t published = first + commands.word_count();
    ASSERT_LT(published * 4, queue.info.ring_byte_length);
    ASSERT_NO_FATAL_FAILURE(queue.PublishStream(published));
    do {
      ASSERT_EQ(gpu_->event_wait(device.event, AMDF_TIMEOUT_INFINITE),
                AMDF_STATUS_OK);
    } while (GpuLoadAcquire<uint32_t>(
                 reinterpret_cast<uintptr_t>(device.output.host.pointer) +
                 queue_ordinal * 64) != value);
    ASSERT_NO_FATAL_FAILURE(queue.WaitConsumed(api_, published));
  }

  void ReleaseMachine(Machine& machine) {
    for (auto& device : machine.devices) {
      for (auto& queue : device->queues) {
        ASSERT_TRUE(queue.Release(api_));
      }
      if (device->event != nullptr) {
        ASSERT_EQ(gpu_->event_destroy(device->event), AMDF_STATUS_OK);
        device->event = nullptr;
      }
      ASSERT_TRUE(device->output.Release(api_));
      if (device->handle != nullptr) {
        ASSERT_EQ(api_->device_destroy(device->handle), AMDF_STATUS_OK);
        device->handle = nullptr;
      }
      if (device->endpoint != nullptr) {
        ASSERT_EQ(api_->endpoint_close(device->endpoint), AMDF_STATUS_OK);
        device->endpoint = nullptr;
      }
    }
    machine.devices.clear();
    if (machine.instance != nullptr) {
      ASSERT_EQ(api_->instance_destroy(machine.instance), AMDF_STATUS_OK);
      machine.instance = nullptr;
    }
  }

  // Host handlers observed before explicit library loading.
  static constexpr std::array<int, 4> kSignals = {SIGSEGV, SIGBUS, SIGINT,
                                                  SIGTERM};
  // Snapshots compared after complete VM and DSO teardown.
  std::array<struct sigaction, kSignals.size()> signals_;
  // Host-owned descriptor budget, unchanged by bootstrap or activation.
  struct rlimit descriptors_ = {};
  // Provider API retained by the enclosing CTS executable.
  const amdf_api_t* api_ = nullptr;
  // Explicit embedding extension borrowed from the provider.
  const amdf_rocjitsu_api_t* simulator_ = nullptr;
  // Ordinary GPU operations borrowed from the provider.
  const amdf_gpu_api_t* gpu_ = nullptr;
  // Qualified external DSO path supplied by the launcher.
  std::string library_;
  // Multi-GPU configuration supplied by the launcher.
  std::string configuration_;
  // Independent VM owners, destroyed in index order.
  std::array<Machine, 2> machines_;
};

TEST_F(RocjitsuTest, LeavesDescriptorBudgetUnderHostControl) {
  ScopedDescriptorBudget restore_budget(descriptors_);
  struct rlimit budget = descriptors_;
  budget.rlim_cur = 8191;
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &budget), 0);

  auto& machine = machines_[0];
  ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
  const auto configuration = Configuration();
  ASSERT_EQ(simulator_->instance_add_machine(machine.instance, &configuration),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, summaries));
  ASSERT_FALSE(summaries.empty());
  Device& device = *machine.devices.emplace_back(std::make_unique<Device>());
  ASSERT_EQ(
      api_->endpoint_open(machine.instance, &summaries[0].id, &device.endpoint),
      AMDF_STATUS_OK);
  const amdf_gpu_device_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  amdf_device_t* sentinel = reinterpret_cast<amdf_device_t*>(uintptr_t{1});
  ASSERT_EQ(amdf_status_code(
                gpu_->device_create(device.endpoint, &create, &sentinel)),
            AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  ASSERT_EQ(sentinel, reinterpret_cast<amdf_device_t*>(uintptr_t{1}));
  struct rlimit observed;
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &observed), 0);
  ASSERT_EQ(observed.rlim_cur, budget.rlim_cur);
  ASSERT_EQ(observed.rlim_max, budget.rlim_max);

  budget.rlim_cur = 8192;
  ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &budget), 0);
  ASSERT_EQ(gpu_->device_create(device.endpoint, &create, &device.handle),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machine));
  ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &observed), 0);
  ASSERT_EQ(observed.rlim_cur, budget.rlim_cur);
  ASSERT_EQ(observed.rlim_max, budget.rlim_max);
}

TEST_F(RocjitsuTest, ValidatesBootstrapWithoutPublishingPartialMachines) {
  auto& machine = machines_[0];
  ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
  const void* extension = reinterpret_cast<const void*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(api_->query_extension(AMDF_EXTENSION_ROCJITSU, 2,
                                                   2, &extension)),
            AMDF_STATUS_CODE_VERSION_MISMATCH);
  EXPECT_EQ(extension, reinterpret_cast<const void*>(uintptr_t{1}));
  auto configuration = Configuration();
  EXPECT_EQ(amdf_status_code(
                simulator_->instance_add_machine(nullptr, &configuration)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(amdf_status_code(
                simulator_->instance_add_machine(machine.instance, nullptr)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  configuration.next = &configuration;
  EXPECT_EQ(amdf_status_code(simulator_->instance_add_machine(machine.instance,
                                                              &configuration)),
            AMDF_STATUS_CODE_UNSUPPORTED);
  configuration = Configuration();
  configuration.configuration_json = nullptr;
  EXPECT_EQ(amdf_status_code(simulator_->instance_add_machine(machine.instance,
                                                              &configuration)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  configuration = Configuration();
  configuration.library_path = "/nonexistent/librocjitsu.so";
  EXPECT_EQ(amdf_status_code(simulator_->instance_add_machine(machine.instance,
                                                              &configuration)),
            AMDF_STATUS_CODE_NOT_FOUND);
  configuration = Configuration();
  configuration.library_path = "libc.so.6";
  EXPECT_EQ(amdf_status_code(simulator_->instance_add_machine(machine.instance,
                                                              &configuration)),
            AMDF_STATUS_CODE_VERSION_MISMATCH);
  std::vector<amdf_endpoint_summary_t> before;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, before));
  configuration = Configuration();
  configuration.configuration_json = "{";
  EXPECT_FALSE(amdf_status_is_ok(
      simulator_->instance_add_machine(machine.instance, &configuration)));
  std::vector<amdf_endpoint_summary_t> after;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, after));
  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i) {
    EXPECT_TRUE(amdf_endpoint_id_is_equal(&before[i].id, &after[i].id));
  }
  AssertLibraryLoaded(false);

  configuration = Configuration();
  ASSERT_EQ(simulator_->instance_add_machine(machine.instance, &configuration),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, before));
  ASSERT_GE(before.size(), 2u);
  configuration.configuration_json = "{";
  EXPECT_FALSE(amdf_status_is_ok(
      simulator_->instance_add_machine(machine.instance, &configuration)));
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, after));
  ASSERT_EQ(before.size(), after.size());
  for (size_t i = 0; i < before.size(); ++i) {
    EXPECT_TRUE(amdf_endpoint_id_is_equal(&before[i].id, &after[i].id));
    EXPECT_NE(after[i].type_flags & AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE,
              0u);
  }
}

TEST_F(RocjitsuTest, ResolvesExplicitEnvironmentAndLoaderLibraryPaths) {
  ScopedRocjitsuLibraryPath restore_environment;
  const struct {
    // Selection rule identified in failure diagnostics.
    const char* name;
    // Explicit public API input; null and empty request default resolution.
    const char* library_path;
    // Environment override; null means the variable is absent.
    const char* environment_path;
    // Result of loading and negotiating the selected library.
    amdf_status_code_t expected_status;
  } cases[] = {
      {"explicit path overrides environment", library_.c_str(),
       "/nonexistent/librocjitsu.so", AMDF_STATUS_CODE_OK},
      {"explicit loader name overrides environment", "librocjitsu.so",
       "/nonexistent/librocjitsu.so", AMDF_STATUS_CODE_OK},
      {"missing explicit library does not fall back",
       "/nonexistent/librocjitsu.so", library_.c_str(),
       AMDF_STATUS_CODE_NOT_FOUND},
      {"incompatible explicit library does not fall back", "libc.so.6",
       library_.c_str(), AMDF_STATUS_CODE_VERSION_MISMATCH},
      {"null path uses environment", nullptr, library_.c_str(),
       AMDF_STATUS_CODE_OK},
      {"empty path uses environment", "", library_.c_str(),
       AMDF_STATUS_CODE_OK},
      {"missing environment library does not fall back", nullptr,
       "/nonexistent/librocjitsu.so", AMDF_STATUS_CODE_NOT_FOUND},
      {"incompatible environment library does not fall back", "", "libc.so.6",
       AMDF_STATUS_CODE_VERSION_MISMATCH},
      {"null path and absent environment use loader", nullptr, nullptr,
       AMDF_STATUS_CODE_OK},
      {"empty path and absent environment use loader", "", nullptr,
       AMDF_STATUS_CODE_OK},
      {"null path and empty environment use loader", nullptr, "",
       AMDF_STATUS_CODE_OK},
      {"empty path and empty environment use loader", "", "",
       AMDF_STATUS_CODE_OK},
  };
  auto& machine = machines_[0];
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ASSERT_EQ(
        test_case.environment_path != nullptr
            ? setenv("LIBAMDF_ROCJITSU_PATH", test_case.environment_path, 1)
            : unsetenv("LIBAMDF_ROCJITSU_PATH"),
        0);
    ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
    auto configuration = Configuration();
    configuration.library_path = test_case.library_path;
    ASSERT_EQ(amdf_status_code(simulator_->instance_add_machine(
                  machine.instance, &configuration)),
              test_case.expected_status);
    const bool loaded = test_case.expected_status == AMDF_STATUS_CODE_OK;
    AssertLibraryLoaded(loaded);
    if (loaded) {
      std::vector<amdf_endpoint_summary_t> summaries;
      ASSERT_NO_FATAL_FAILURE(Enumerate(machine, summaries));
      ASSERT_GE(summaries.size(), 2u);
      for (const auto& summary : summaries) {
        EXPECT_NE(summary.type_flags & AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE,
                  0u);
      }
    }
    ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machine));
    AssertLibraryLoaded(false);
  }
}

TEST_F(RocjitsuTest, ComposesPassiveMachinesAndPreservesEndpointOwnership) {
  auto& machine = machines_[0];
  ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
  const auto configuration = Configuration();
  ASSERT_EQ(simulator_->instance_add_machine(machine.instance, &configuration),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> first;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, first));
  ASSERT_GE(first.size(), 2u);
  ASSERT_EQ(simulator_->instance_add_machine(machine.instance, &configuration),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> all;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, all));
  ASSERT_EQ(all.size(), first.size() * 2);
  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_TRUE(amdf_endpoint_id_is_equal(&first[i].id, &all[i].id));
  }
  for (size_t i = 0; i < all.size(); ++i) {
    EXPECT_NE(all[i].type_flags & AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE, 0u);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_FALSE(amdf_endpoint_id_is_equal(&all[i].id, &all[j].id));
    }
  }
  std::array<amdf_endpoint_summary_t, 2> prefix;
  std::memset(prefix.data(), 0x5a, sizeof(prefix));
  const auto sentinel = prefix[1];
  uint32_t total = 0;
  ASSERT_EQ(amdf_status_code(api_->endpoint_enumerate(machine.instance, 1,
                                                      prefix.data(), &total)),
            AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  EXPECT_EQ(total, all.size());
  EXPECT_TRUE(amdf_endpoint_id_is_equal(&prefix[0].id, &all[0].id));
  EXPECT_EQ(std::memcmp(&prefix[1], &sentinel, sizeof(sentinel)), 0);
  Device& opened = *machine.devices.emplace_back(std::make_unique<Device>());
  ASSERT_EQ(
      api_->endpoint_open(machine.instance, &all.back().id, &opened.endpoint),
      AMDF_STATUS_OK);
  EXPECT_EQ(amdf_status_code(simulator_->instance_add_machine(machine.instance,
                                                              &configuration)),
            AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(amdf_status_code(api_->instance_destroy(machine.instance)),
            AMDF_STATUS_CODE_BUSY);
  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->endpoint_query_info(opened.endpoint, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.native_identity.type, AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_NONE);
  auto stale = all.back().id;
  stale.words[1] ^= 1;
  auto* unchanged = reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(
                api_->endpoint_open(machine.instance, &stale, &unchanged)),
            AMDF_STATUS_CODE_NOT_FOUND);
  EXPECT_EQ(unchanged, reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1}));
}

TEST_F(RocjitsuTest, NativeEndpointsRemainScopedToNativeInstances) {
  auto& machine = machines_[0];
  auto& native_machine = machines_[1];
  ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
  ASSERT_NO_FATAL_FAILURE(CreateInstance(native_machine));
  std::vector<amdf_endpoint_summary_t> native;
  ASSERT_NO_FATAL_FAILURE(Enumerate(native_machine, native));
  if (native.empty()) {
    GTEST_SKIP() << "native endpoints required to exercise fabric isolation";
  }
  const auto configuration = Configuration();
  ASSERT_EQ(simulator_->instance_add_machine(machine.instance, &configuration),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> configured;
  ASSERT_NO_FATAL_FAILURE(Enumerate(machine, configured));
  ASSERT_GE(configured.size(), 2u);
  for (const auto& summary : configured) {
    EXPECT_NE(summary.type_flags & AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE, 0u);
  }
  for (const auto& summary : native) {
    Device& opened =
        *native_machine.devices.emplace_back(std::make_unique<Device>());
    ASSERT_EQ(api_->endpoint_open(native_machine.instance, &summary.id,
                                  &opened.endpoint),
              AMDF_STATUS_OK);
    auto* unchanged = reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1});
    EXPECT_EQ(amdf_status_code(api_->endpoint_open(machine.instance,
                                                   &summary.id, &unchanged)),
              AMDF_STATUS_CODE_NOT_FOUND);
    EXPECT_EQ(unchanged, reinterpret_cast<amdf_endpoint_t*>(uintptr_t{1}));
  }
}

class RocjitsuQueueTest
    : public RocjitsuTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {};

TEST_P(RocjitsuQueueTest, IndependentMachinesSurviveQueueAndPeerTeardown) {
  for (uint32_t cycle = 0; cycle < 2; ++cycle) {
    for (auto& machine : machines_) {
      ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
      ASSERT_NO_FATAL_FAILURE(Activate(machine, GetParam()));
      if (IsSkipped()) {
        return;
      }
      ASSERT_NO_FATAL_FAILURE(Execute(machine, 0, 1));
      ASSERT_NO_FATAL_FAILURE(Execute(machine, 1, 2));
      for (auto& device : machine.devices) {
        ASSERT_TRUE(device->queues[0].Release(api_));
      }
      ASSERT_NO_FATAL_FAILURE(Execute(machine, 1, 3));
    }
    AssertLibraryLoaded(true);
    ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machines_[0]));
    AssertLibraryLoaded(true);
    ASSERT_NO_FATAL_FAILURE(Execute(machines_[1], 1, 4));
    ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machines_[1]));
    AssertLibraryLoaded(false);
  }
}

TEST_P(RocjitsuQueueTest, ConcurrentRetirementPreservesHostEventProgress) {
  auto& machine = machines_[0];
  ASSERT_NO_FATAL_FAILURE(CreateInstance(machine));
  ASSERT_NO_FATAL_FAILURE(Activate(machine, GetParam()));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(Execute(machine, 0, 1));
  ASSERT_NO_FATAL_FAILURE(Execute(machine, 1, 2));

  amdf_gpu_event_t* event = nullptr;
  const amdf_gpu_event_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  ASSERT_EQ(gpu_->event_create(machine.devices[0]->handle, &create, &event),
            AMDF_STATUS_OK);
  std::barrier<> start(static_cast<ptrdiff_t>(machine.devices.size() + 2));
  std::thread waiter([&] {
    start.arrive_and_wait();
    EXPECT_EQ(gpu_->event_wait(event, AMDF_TIMEOUT_INFINITE), AMDF_STATUS_OK);
  });
  std::vector<std::thread> retiring;
  for (auto& device : machine.devices) {
    retiring.emplace_back([&, device = device.get()] {
      start.arrive_and_wait();
      for (auto& queue : device->queues) {
        EXPECT_TRUE(queue.Release(api_));
      }
      EXPECT_TRUE(device->output.Release(api_));
    });
  }
  start.arrive_and_wait();
  EXPECT_EQ(gpu_->event_signal(event), AMDF_STATUS_OK);
  for (auto& thread : retiring) {
    thread.join();
  }
  // Release the waiter even when the overlapping signal reported a failure.
  // The failure above remains visible; cleanup begins after every retiree
  // joins.
  EXPECT_EQ(gpu_->event_signal(event), AMDF_STATUS_OK);
  waiter.join();
  EXPECT_EQ(gpu_->event_destroy(event), AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ReleaseMachine(machine));
}

INSTANTIATE_TEST_SUITE_P(
    Protocols, RocjitsuQueueTest,
    ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
    [](const ::testing::TestParamInfo<amdf_queue_command_type_t>& info) {
      return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL ? "Aql" : "Pm4";
    });

}  // namespace
