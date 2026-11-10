// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <signal.h>

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "rocjitsu/vm/rj_vm_config.h"

namespace {

const char* library_path = nullptr;

// A small explicit topology exercises allocation of the shared-memory keeper
// without requiring compiler output or an installed kernel driver.
constexpr const char kConfiguration[] = R"({
  "num_threads":1,"cpu_dispatch_threads":1,
  "vm":{"arch":"cdna5","gpu":{"device":{
    "gpu_id":1250,"gfx_target_version":120500,"unique_id":1250,
    "num_sdma_engines":0,"local_mem_size":1048576,"simd_count":16,
    "num_shader_engines":1,"num_shader_arrays_per_engine":1,"num_cu_per_sh":4}}},
  "topology":{"root":{"name":"soc","type":"soc","children":[
    {"name":"vram","type":"gpu_memory"},
    {"name":"xcd0","type":"xcd","children":[
      {"name":"l2","type":"l2_cache"},
      {"name":"cp","type":"command_processor"},
      {"name":"se0","type":"shader_engine","children":[
        {"name":"cu[0:4]","type":"compute_unit"}]}]}]}}
})";

template <typename Function>
Function LoadFunction(void* library, const char* name) {
  auto function = reinterpret_cast<Function>(dlsym(library, name));
  EXPECT_NE(function, nullptr) << name << ": " << dlerror();
  return function;
}

size_t KeeperLeaseCount() {
  std::ifstream mappings("/proc/self/maps");
  EXPECT_TRUE(mappings.is_open());
  size_t count = 0;
  for (std::string line; std::getline(mappings, line);) {
    if (line.find("/memfd:rocjitsu-owner") != std::string::npos) {
      ++count;
    }
  }
  EXPECT_TRUE(mappings.eof());
  return count;
}

TEST(RocjitsuRuntime, RelocatedPackageOwnsItsCompleteLifecycle) {
  const size_t initial_leases = KeeperLeaseCount();
  constexpr std::array signals{SIGSEGV, SIGBUS, SIGINT, SIGTERM};
  std::array<struct sigaction, signals.size()> original_actions{};
  for (size_t index = 0; index < signals.size(); ++index) {
    ASSERT_EQ(sigaction(signals[index], nullptr, &original_actions[index]), 0);
  }
  const auto check_actions = [&] {
    for (size_t index = 0; index < signals.size(); ++index) {
      struct sigaction action{};
      ASSERT_EQ(sigaction(signals[index], nullptr, &action), 0);
      EXPECT_EQ(action.sa_sigaction, original_actions[index].sa_sigaction);
      EXPECT_EQ(action.sa_flags, original_actions[index].sa_flags);
      for (int signal_number = 1; signal_number < NSIG; ++signal_number) {
        EXPECT_EQ(sigismember(&action.sa_mask, signal_number),
                  sigismember(&original_actions[index].sa_mask, signal_number));
      }
    }
  };
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    SCOPED_TRACE(cycle);
    std::unique_ptr<void, decltype(&dlclose)> library(
        dlopen(library_path, RTLD_NOW | RTLD_LOCAL), dlclose);
    ASSERT_NE(library, nullptr) << dlerror();
    ASSERT_NO_FATAL_FAILURE(check_actions());
    ASSERT_EQ(dlsym(RTLD_DEFAULT, "rj_vm_create_from_config"), nullptr);
    const auto create_config =
        LoadFunction<decltype(&rj_vm_config_create_from_string)>(
            library.get(), "rj_vm_config_create_from_string");
    const auto destroy_config = LoadFunction<decltype(&rj_vm_config_destroy)>(
        library.get(), "rj_vm_config_destroy");
    const auto gpu_info = LoadFunction<decltype(&rj_vm_config_gpu_info_at)>(
        library.get(), "rj_vm_config_gpu_info_at");
    const auto create = LoadFunction<decltype(&rj_vm_create_from_config)>(
        library.get(), "rj_vm_create_from_config");
    const auto destroy =
        LoadFunction<decltype(&rj_vm_destroy)>(library.get(), "rj_vm_destroy");
    const auto run =
        LoadFunction<decltype(&rj_vm_run)>(library.get(), "rj_vm_run");
    const auto request_exit = LoadFunction<decltype(&rj_vm_request_exit)>(
        library.get(), "rj_vm_request_exit");
    const auto open = LoadFunction<decltype(&rj_vm_device_open)>(
        library.get(), "rj_vm_device_open");
    const auto close = LoadFunction<decltype(&rj_vm_device_close)>(
        library.get(), "rj_vm_device_close");
    ASSERT_FALSE(::testing::Test::HasFailure());

    rj_vm_config_t* raw_config = nullptr;
    ASSERT_EQ(create_config(kConfiguration, &raw_config),
              ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_config_t, decltype(destroy_config)> config(
        raw_config, destroy_config);
    rj_vm_gpu_info_t info{};
    ASSERT_EQ(gpu_info(config.get(), 0, &info), ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(info.gpu_id, 1250u);
    EXPECT_EQ(KeeperLeaseCount(), initial_leases);

    rj_vm_t* raw_vm = nullptr;
    ASSERT_EQ(create(config.get(), RJ_VM_MODE_LOCAL, &raw_vm),
              ROCJITSU_STATUS_SUCCESS);
    std::unique_ptr<rj_vm_t, decltype(destroy)> vm(raw_vm, destroy);
    config.reset();
    uint32_t process_id = 0;
    ASSERT_EQ(open(vm.get(), 0, &process_id), ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(KeeperLeaseCount(), initial_leases + 1);
    ASSERT_NO_FATAL_FAILURE(check_actions());
    rj_status_t run_status = ROCJITSU_STATUS_ERROR;
    std::thread runner([&] { run_status = run(vm.get(), nullptr); });
    request_exit(vm.get(), "runtime package qualification");
    runner.join();
    EXPECT_EQ(run_status, ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(close(vm.get(), process_id), ROCJITSU_STATUS_SUCCESS);
    vm.reset();
    EXPECT_EQ(dlclose(library.release()), 0);
    EXPECT_EQ(KeeperLeaseCount(), initial_leases);
    std::unique_ptr<void, decltype(&dlclose)> retained(
        dlopen(library_path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD), dlclose);
    EXPECT_EQ(retained, nullptr);
    ASSERT_NO_FATAL_FAILURE(check_actions());
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc != 2) {
    return 1;
  }
  library_path = argv[1];
  return RUN_ALL_TESTS();
}
