// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Accounts for host and native storage retained by independent recordings of
// one compiled command, without checkpoint payload or command submission. Host
// observations exclude allocator metadata; native rows use HAL statistics. The
// observer adds synchronization, so these are storage, not timing, rows.

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/bank.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/stage_table.h"
#include "loom/serve/testing/compiler.h"
#include "loom/serve/testing/device.h"

namespace {

class Allocations {
 public:
  struct Snapshot {
    // Requested bytes still owned, excluding observer and allocator metadata.
    size_t live = 0;
    // Highest simultaneous requested bytes since the last reset.
    size_t peak = 0;
    // Successful allocation/reallocation calls since the last reset.
    size_t calls = 0;
  };

  iree_allocator_t allocator() { return {this, Control}; }

  Snapshot snapshot(bool reset = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot result = snapshot_;
    if (reset) {
      snapshot_.peak = snapshot_.live;
      snapshot_.calls = 0;
    }
    return result;
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto& owner = *static_cast<Allocations*>(self);
    std::lock_guard<std::mutex> lock(owner.mutex_);
    void* old = nullptr;
    size_t old_bytes = 0;
    if (command == IREE_ALLOCATOR_COMMAND_FREE ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      old = *pointer;
      if (old) {
        old_bytes = owner.sizes_.at(old);
      }
    }
    const auto system = iree_allocator_system();
    iree_status_t status = system.ctl(system.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (old) {
        owner.sizes_.erase(old);
      }
      owner.snapshot_.live -= old_bytes;
      if (command != IREE_ALLOCATOR_COMMAND_FREE && *pointer) {
        const size_t bytes =
            static_cast<const iree_allocator_alloc_params_t*>(params)
                ->byte_length;
        owner.sizes_.emplace(*pointer, bytes);
        owner.snapshot_.live += bytes;
        ++owner.snapshot_.calls;
      }
      owner.snapshot_.peak =
          std::max(owner.snapshot_.peak, owner.snapshot_.live);
    }
    return status;
  }

  // Protects observations made from device/compiler workers.
  std::mutex mutex_;
  // Exact sizes of live allocations made through this observer.
  std::unordered_map<void*, size_t> sizes_;
  // Requested storage accounting, independent of process RSS.
  Snapshot snapshot_;
};

static std::vector<std::string> ReadLines(const char* path) {
  std::ifstream file(path);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(2);
  }
  std::vector<std::string> result;
  for (std::string line; std::getline(file, line);) {
    if (!line.empty()) {
      result.push_back(std::move(line));
    }
  }
  if (!file.eof()) {
    std::fprintf(stderr, "failed reading %s\n", path);
    std::exit(2);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s DIRECTORY ROOT SOURCE_LIST CONFIG_LIST\n"
                 "Lists contain one source path or key=value per line.\n",
                 argv[0]);
    return 2;
  }
  auto source_names = ReadLines(argv[3]);
  auto config_lines = ReadLines(argv[4]);
  std::vector<iree_string_view_t> source_paths;
  for (const auto& name : source_names) {
    source_paths.push_back(iree_make_cstring_view(name.c_str()));
  }
  std::vector<loomc_config_binding_t> bindings;
  for (const auto& line : config_lines) {
    const size_t separator = line.find('=');
    if (separator == std::string::npos || !separator ||
        separator + 1 == line.size()) {
      std::fprintf(stderr, "invalid configuration: %s\n", line.c_str());
      return 2;
    }
    bindings.push_back(
        {{line.data(), separator},
         {line.data() + separator + 1, line.size() - separator - 1}});
  }
  const loomc_config_options_t config = {
      .bindings = bindings.data(),
      .binding_count = bindings.size(),
      .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
  };
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(argv[1]),
      .count = source_paths.size(),
      .values = source_paths.data(),
  };
  Allocations allocations;
  const auto tracked = allocations.allocator();
  const auto system = iree_allocator_system();
  loom_serve_host_t* host = nullptr;
  IREE_CHECK_OK(loom_serve_test_host_create(4, system, &host));
  iree_hal_driver_t* driver = nullptr;
  IREE_CHECK_OK(loom_serve_test_driver_acquire(host, &driver));
  const loom_serve_device_options_t options = {
      .driver = driver,
      .uri = loom_serve_test_device()->device_uri,
      .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
  };
  loom_serve_device_group_t* group = nullptr;
  IREE_CHECK_OK(
      loom_serve_device_group_allocate(host, 1, &options, tracked, &group));
  iree_hal_driver_release(driver);
  auto* device = loom_serve_device_group_at(group, 0);
  loom_serve_compilation_report_t report = {};
  loom_serve_jit_t* jit = nullptr;
  IREE_CHECK_OK(loom_serve_jit_create(loom_serve_device_handle(device),
                                      loom_serve_device_dispatch_queue(device),
                                      host, &sources, nullptr, &report, system,
                                      &jit));
  loom_serve_jit_stage_t* stage = nullptr;
  IREE_CHECK_OK(loom_serve_jit_compile(jit, iree_make_cstring_view(argv[2]),
                                       &config, &stage));
  auto* program = loom_serve_jit_stage_program(stage);
  if (!program) {
    std::fprintf(stderr, "root must produce a command program\n");
    return 2;
  }
  loom_serve_parameter_bank_t* bank = nullptr;
  IREE_CHECK_OK(
      loom_serve_parameter_bank_allocate(device, program, system, &bank));
  const auto baseline = allocations.snapshot(true);
  std::printf(
      "{\"stage_bytes\":%zu,\"parameters\":%u,\"baseline_bytes\":%zu}\n",
      loom_serve_jit_stage_storage_size(stage), program->parameters.count,
      baseline.live);
  for (uint32_t quantum : {0u, 256u}) {
    const auto before = allocations.snapshot(true);
#if IREE_STATISTICS_ENABLE
    auto* native = iree_hal_device_allocator(loom_serve_device_handle(device));
    iree_hal_allocator_statistics_t native_before = {};
    iree_hal_allocator_query_statistics(native, &native_before);
#endif
    const loom_serve_stage_record_options_t recording = {
        .queue_family =
            iree_hal_queue_family(loom_serve_device_dispatch_queue(device)),
        .parameters = loom_serve_parameter_bank_buffers(bank),
        .command_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        .binding_count = 2,
        .workspace = LOOM_SERVE_STAGE_WORKSPACE_INVOCATION,
        .command_quantum = quantum,
    };
    IREE_CHECK_OK(loom_serve_stage_record_check(&recording, stage));
    loom_serve_stage_table_t* tables[8] = {};
    size_t descriptors = 0;
    for (size_t i = 0; i < IREE_ARRAYSIZE(tables); ++i) {
      IREE_CHECK_OK(loom_serve_stage_table_allocate(
          &recording, nullptr, 1, &stage, tracked, &tables[i]));
      descriptors += loom_serve_stage_table_storage_size(tables[i]);
      const auto current = allocations.snapshot();
      std::printf(
          "{\"quantum\":%u,\"copies\":%zu,\"parts_per_copy\":%zu,"
          "\"live_delta\":%zu,\"peak_delta\":%zu,\"calls\":%zu,"
          "\"descriptors\":%zu",
          quantum, i + 1, tables[i]->part_count, current.live - before.live,
          current.peak - before.live, current.calls, descriptors);
#if IREE_STATISTICS_ENABLE
      iree_hal_allocator_statistics_t native_current = {};
      iree_hal_allocator_query_statistics(native, &native_current);
      std::printf(",\"native_host_delta\":%" PRIu64
                  ",\"native_device_delta\":%" PRIu64,
                  (native_current.host_bytes_allocated -
                   native_current.host_bytes_freed) -
                      (native_before.host_bytes_allocated -
                       native_before.host_bytes_freed),
                  (native_current.device_bytes_allocated -
                   native_current.device_bytes_freed) -
                      (native_before.device_bytes_allocated -
                       native_before.device_bytes_freed));
#else
      std::printf(",\"native_statistics\":\"disabled\"");
#endif
      std::printf("}\n");
    }
    for (auto* table : tables) {
      loom_serve_stage_table_free(table);
    }
    std::printf("{\"quantum\":%u,\"after_free_bytes\":%zu}\n", quantum,
                allocations.snapshot().live);
  }
  const auto memory = loom_serve_device_memory_statistics(device);
  std::printf("{\"parameter_reserved\":%" PRIu64
              ",\"parameter_committed\":%" PRIu64 "}\n",
              memory.retained.reserved_bytes, memory.retained.committed_bytes);
  loom_serve_parameter_bank_retire(bank, nullptr, nullptr);
  IREE_CHECK_OK(loom_serve_parameter_bank_await_retirement(bank));
  IREE_CHECK_OK(loom_serve_parameter_bank_free(bank));
  loom_serve_jit_stage_free(stage);
  loom_serve_jit_release(jit);
  IREE_CHECK_OK(loom_serve_device_group_free(group));
  loom_serve_host_release(host);
  std::printf("{\"final_tracked_bytes\":%zu}\n", allocations.snapshot().live);
  return allocations.snapshot().live ? 1 : 0;
}
