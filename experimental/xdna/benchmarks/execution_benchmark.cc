// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"
#include "experimental/xdna/executable.h"
#include "experimental/xdna/prepared_command.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

// Stop on native failure without retrying work or freeing potentially live DMA
// backing. Successful runs check retirement before caller-ordered teardown.
void CheckStatus(amdf_status_t status, const char* operation) {
  if (amdf_status_is_ok(status)) return;
  std::fprintf(stderr, "%s failed: domain=%u code=%u\n", operation,
               amdf_status_domain(status), amdf_status_code(status));
  std::exit(EXIT_FAILURE);
}

void CheckIreeStatus(iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  std::exit(EXIT_FAILURE);
}

void Check(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "%s\n", message);
  std::exit(EXIT_FAILURE);
}

constexpr size_t kElementCount = 16;
constexpr size_t kBindingByteLength = kElementCount * sizeof(uint32_t);
constexpr size_t kStorageByteLength = 3 * kBindingByteLength;
constexpr uint8_t kGuardValue = 0xA5;
using BindingValues = std::array<uint32_t, kElementCount>;
constexpr BindingValues kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFD,
    0xFFFFFFFE, 0xFFFFFFFF, 0x12345678, 0x87654321};

enum class CompletionTiming { kExcluded, kIncluded };

// The benchmark owns a real image consumer above libamdf. Every repetition
// shares one device, context and queue, with one explicitly retired command at
// a time. Cold preparation never enters a measured region.
class ExecutionBenchmark {
 public:
  void Initialize() {
    CheckStatus(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_1,
                                              AMDF_ABI_VERSION_LATEST, &api_),
                "query_api");
    const void* extension = nullptr;
    CheckStatus(api_->query_extension(
                    AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                    AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
                "query_extension");
    xdna_api_ = static_cast<const amdf_xdna_api_t*>(extension);
    amdf_instance_t* instance = nullptr;
    CheckStatus(GetCtsDeviceCache().GetInstance(&instance), "instance_create");
    uint32_t count = 0;
    CheckStatus(api_->endpoint_enumerate(instance, 0, nullptr, &count),
                "endpoint_count");
    std::vector<amdf_endpoint_summary_t> summaries(count);
    CheckStatus(
        api_->endpoint_enumerate(instance, count, summaries.data(), &count),
        "endpoint_enumerate");
    amdf_endpoint_t* endpoint = nullptr;
    for (const auto& summary : summaries) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
      CheckStatus(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                  "endpoint_open");
      break;
    }
    if (!endpoint) return;
    amdf_xdna_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    CheckStatus(xdna_api_->endpoint_query_info(endpoint, &info), "xdna_info");
    benchmark::AddCustomContext("xdna_target", info.target_id);
    skip_reason_ = "time-sliced XDNA contexts unavailable";
    if (!(info.context.scheduling_modes &
          AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED))
      return;
    const iree_file_toc_t* image = nullptr;
    if (std::strcmp(info.target_id, "amd.xdna.strix.17f0_10") == 0) {
      image = iree_hal_amd_xdna_test_mul_i32_npu4_create();
    } else if (std::strcmp(info.target_id, "amd.xdna.strix_halo.17f0_11") ==
               0) {
      image = iree_hal_amd_xdna_test_mul_i32_create();
    }
    skip_reason_ = "no canonical multiplication fixture for this XDNA target";
    if (!image) return;
    skip_reason_ = "native XDNA device materialization unavailable";
    const amdf_status_t status =
        GetCtsDeviceCache().GetXdnaDevice(endpoint, &device_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) return;
    CheckStatus(status, "device_create");

    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    CheckStatus(api_->endpoint_query_info(endpoint, &endpoint_info),
                "endpoint_info");
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < endpoint_info.queue_family_count; ++i) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      CheckStatus(api_->endpoint_query_queue_family_info(endpoint, i, &family),
                  "queue_family");
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL)) {
        family_ordinal = i;
        break;
      }
    }
    Check(family_ordinal != UINT32_MAX, "no native XDNA queue family");
    queue_family_spec_.name = IREE_SV("xdna");
    queue_family_spec_.physical_device_affinity = 1;
    queue_family_spec_.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
    iree_hal_queue_family_initialize(family_ordinal, &queue_family_spec_,
                                     &queue_family_);
    iree_hal_amd_xdna_aie2p_target_t target;
    CheckIreeStatus(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(info.target_id), 1, &target));
    iree_byte_span_t image_bytes = {nullptr, image->size};
    CheckIreeStatus(iree_allocator_clone(
        iree_allocator_system(),
        iree_make_const_byte_span(image->data, image->size),
        reinterpret_cast<void**>(&image_bytes.data)));
    iree_byte_sequence_t* sequence = nullptr;
    CheckIreeStatus(iree_byte_sequence_create_from_span_move(
        &image_bytes, iree_allocator_system(), &sequence));
    CheckIreeStatus(iree_hal_amd_xdna_executable_create(
        &queue_family_, sequence, &target, iree_allocator_system(),
        &executable_));
    iree_byte_sequence_release(sequence);
    iree_hal_executable_function_t function;
    CheckIreeStatus(iree_hal_executable_lookup_function_by_name(
        executable_, IREE_SV("mul_i32"), &function));
    CreateBindings(instance);
    PrepareExecution(function, info.instruction.address_alignment);

    WriteInputs();
    const uint64_t submission =
        Submit(iree_hal_amd_xdna_prepared_command_initialization(prepared_));
    Wait(submission);
    VerifyOutput(submission);
    VerifyInstructions();
    skip_reason_ = nullptr;
  }

  template <CompletionTiming completion_timing>
  void Run(benchmark::State& state) {
    if (skip_reason_) {
      state.SkipWithMessage(skip_reason_);
      return;
    }
    VerifyInstructions();
    const auto* command =
        iree_hal_amd_xdna_prepared_command_execution(prepared_);
    for (auto iteration : state) {
      (void)iteration;
      state.PauseTiming();
      WriteInputs();
      state.ResumeTiming();
      const uint64_t submission = Submit(command);
      if constexpr (completion_timing == CompletionTiming::kIncluded) {
        Wait(submission);
        state.PauseTiming();
      } else {
        state.PauseTiming();
        Wait(submission);
      }
      VerifyOutput(submission);
      state.ResumeTiming();
    }
    VerifyInstructions();
    state.SetItemsProcessed(state.iterations());
  }

  void Deinitialize() {
    if (queue_)
      CheckStatus(api_->kernel_queue_destroy(queue_), "queue_destroy");
    iree_hal_amd_xdna_prepared_command_destroy(prepared_);
    DestroyMemory(instructions_);
    if (context_)
      CheckStatus(xdna_api_->context_destroy(context_), "context_destroy");
    for (auto& binding : bindings_) {
      iree_hal_buffer_release(binding.buffer);
      DestroyMemory(binding.storage);
    }
    iree_hal_executable_release(executable_);
  }

 private:
  struct MappedMemory {
    // Native allocation owned until after its mapping and all commands retire.
    amdf_memory_t* memory = nullptr;
    // Explicit host view destroyed before memory.
    amdf_host_mapping_t* mapping = nullptr;
    // First mapped byte borrowed from mapping.
    uint8_t* pointer = nullptr;
  };

  void MapMemory(uint64_t byte_length, MappedMemory& memory) {
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = byte_length;
    CheckStatus(api_->memory_map(memory.memory, &map, &memory.mapping),
                "memory_map");
    amdf_host_mapping_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    CheckStatus(api_->host_mapping_query_info(memory.mapping, &info),
                "mapping_info");
    memory.pointer = static_cast<uint8_t*>(info.pointer);
  }

  void DestroyMemory(const MappedMemory& memory) {
    if (memory.mapping)
      CheckStatus(api_->host_mapping_destroy(memory.mapping),
                  "mapping_destroy");
    if (memory.memory)
      CheckStatus(api_->memory_destroy(memory.memory), "memory_destroy");
  }

  void CreateBindings(amdf_instance_t* instance) {
    uint32_t count = 0;
    Check(
        api_->instance_enumerate_memory_scopes(instance, 0, nullptr, &count) ==
            amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL),
        "no instance memory scopes");
    std::vector<amdf_memory_scope_t*> scopes(count);
    CheckStatus(api_->instance_enumerate_memory_scopes(instance, count,
                                                       scopes.data(), &count),
                "memory_scopes");
    amdf_memory_scope_t* system_scope = nullptr;
    for (auto* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      CheckStatus(api_->memory_scope_query_info(scope, &info), "scope_info");
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        system_scope = scope;
        break;
      }
    }
    Check(system_scope != nullptr, "no system memory scope");
    amdf_memory_device_access_t access = {};
    access.device = device_;
    access.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = uint64_t{1}
                                        << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    uint32_t profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0;; ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      amdf_memory_access_capabilities_t capabilities = {};
      capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      capabilities.structure_size = sizeof(capabilities);
      const auto status = api_->memory_scope_query_device_profile(
          system_scope, ordinal, 1, &access, &profile, &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE)) break;
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
        continue;
      CheckStatus(status, "data_profile");
      constexpr auto roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
      if ((profile.roles & roles) == roles &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
        profile_ordinal = ordinal;
        break;
      }
    }
    Check(profile_ordinal != AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN,
          "no host-visible XDNA data allocation profile");
    for (auto& binding : bindings_) {
      amdf_memory_create_info_t create = {};
      create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create.structure_size = sizeof(create);
      create.memory_profile_ordinal = profile_ordinal;
      create.access_count = 1;
      create.accesses = &access;
      create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      create.byte_length = kStorageByteLength;
      CheckStatus(
          api_->memory_create(system_scope, &create, &binding.storage.memory),
          "data_create");
      MapMemory(kStorageByteLength, binding.storage);
      std::memset(binding.storage.pointer, kGuardValue, kStorageByteLength);
      CheckIreeStatus(iree_hal_heap_buffer_wrap(
          iree_hal_buffer_placement_undefined(),
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE |
              IREE_HAL_MEMORY_ACCESS_UNALIGNED,
          IREE_HAL_BUFFER_USAGE_STORAGE, kBindingByteLength,
          iree_make_byte_span(binding.storage.pointer + kBindingByteLength,
                              kBindingByteLength),
          iree_hal_buffer_release_callback_null(), iree_allocator_system(),
          &binding.buffer));
    }
  }

  void PrepareExecution(iree_hal_executable_function_t function,
                        iree_host_size_t alignment) {
    amdf_xdna_context_create_info_t context_create = {};
    context_create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    context_create.structure_size = sizeof(context_create);
    context_create.logical_column_count = 1;
    context_create.physical_column_origin =
        AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    context_create.acceptable_scheduling_modes =
        AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    CheckStatus(xdna_api_->context_create(device_, &context_create, &context_),
                "context_create");
    CheckIreeStatus(iree_hal_amd_xdna_prepared_command_query_storage_size(
        executable_, function, alignment, &instruction_byte_length_));
    amdf_memory_scope_t* scope = nullptr;
    uint32_t count = 0;
    CheckStatus(
        xdna_api_->context_enumerate_memory_scopes(context_, 1, &scope, &count),
        "private_scope");
    Check(count == 1, "expected one instruction scope");
    amdf_memory_device_access_t access = {};
    access.device = device_;
    access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = uint64_t{1}
                                        << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    CheckStatus(api_->memory_scope_query_device_profile(
                    scope, 0, 1, &access, &profile, &capabilities),
                "instruction_profile");
    const uint64_t granularity = profile.allocation.byte_length_granularity;
    Check(granularity != 0, "zero instruction granularity");
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &access;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length =
        ((instruction_byte_length_ + granularity - 1) / granularity) *
        granularity;
    create.minimum_alignment = profile.allocation.minimum_alignment;
    CheckStatus(api_->memory_create(scope, &create, &instructions_.memory),
                "instructions_create");
    MapMemory(instruction_byte_length_, instructions_);
    std::array<iree_hal_amd_xdna_prepared_command_binding_t, 3>
        prepared_bindings = {};
    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& binding = prepared_bindings[i];
      binding.buffer_ref =
          iree_hal_make_buffer_ref(bindings_[i].buffer, 0, kBindingByteLength);
      binding.memory = bindings_[i].storage.memory;
      binding.memory_byte_offset = kBindingByteLength;
      CheckStatus(api_->memory_query_address(binding.memory, 0,
                                             AMDF_MEMORY_ADDRESS_XDNA_DMA,
                                             &binding.device_address),
                  "data_address");
      binding.device_address += kBindingByteLength;
    }
    amdf_xdna_kernel_command_t storage = {};
    storage.memory = instructions_.memory;
    storage.byte_length = instruction_byte_length_;
    CheckIreeStatus(iree_hal_amd_xdna_prepared_command_create(
        executable_, function, alignment, &storage,
        iree_make_byte_span(instructions_.pointer, instruction_byte_length_),
        prepared_bindings.size(), prepared_bindings.data(),
        iree_allocator_system(), &prepared_));
    original_instructions_.assign(
        instructions_.pointer,
        instructions_.pointer + instruction_byte_length_);
    CheckStatus(api_->host_mapping_cache_control(
                    instructions_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                    instruction_byte_length_),
                "instruction_publication");
    amdf_xdna_kernel_queue_create_info_t queue_create = {};
    queue_create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    queue_create.structure_size = sizeof(queue_create);
    queue_create.queue_family_ordinal = queue_family_.ordinal;
    CheckStatus(
        xdna_api_->kernel_queue_create(context_, &queue_create, &queue_),
        "queue_create");
  }

  uint64_t Submit(const amdf_xdna_kernel_command_t* command) {
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = command;
    uint64_t submission = 0;
    CheckStatus(xdna_api_->kernel_queue_submit(queue_, &submit, &submission),
                "queue_submit");
    return submission;
  }

  void Wait(uint64_t submission) {
    CheckStatus(
        api_->kernel_queue_wait(queue_, submission, AMDF_TIMEOUT_INFINITE, 0),
        "queue_wait");
  }

  void WriteInputs() {
    for (size_t i = 0; i < kElementCount; ++i) {
      expected_[0][i] = kValues[(i + input_iteration_) % kElementCount];
      expected_[1][i] = kValues[(i * 3 + input_iteration_ + 5) % kElementCount];
      expected_[2][i] = expected_[0][i] * expected_[1][i];
    }
    ++input_iteration_;
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      const auto& storage = bindings_[ordinal].storage;
      for (size_t i = 0; i < kElementCount; ++i) {
        const uint32_t value =
            ordinal == 2 ? ~expected_[ordinal][i] : expected_[ordinal][i];
        iree_unaligned_store_le_u32(
            storage.pointer + kBindingByteLength + i * sizeof(uint32_t), value);
      }
      CheckStatus(api_->host_mapping_cache_control(
                      storage.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                      kStorageByteLength),
                  "data_publication");
    }
  }

  void VerifyOutput(uint64_t submission) {
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    CheckStatus(api_->kernel_queue_query_status(queue_, &status),
                "queue_status");
    Check(status.retired_submission == submission, "submission not retired");
    CheckStatus(status.terminal_status, "queue_terminal_status");
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      const auto& storage = bindings_[ordinal].storage;
      CheckStatus(api_->host_mapping_cache_control(
                      storage.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                      kStorageByteLength),
                  "data_invalidation");
      for (size_t i = 0; i < kBindingByteLength; ++i) {
        Check(storage.pointer[i] == kGuardValue &&
                  storage.pointer[2 * kBindingByteLength + i] == kGuardValue,
              "binding guard changed");
      }
      for (size_t i = 0; i < kElementCount; ++i) {
        const uint32_t actual = iree_unaligned_load_le_u32(
            storage.pointer + kBindingByteLength + i * sizeof(uint32_t));
        if (actual != expected_[ordinal][i]) {
          std::fprintf(
              stderr,
              "input=%zu binding=%zu element=%zu actual=%u expected=%u\n",
              input_iteration_ - 1, ordinal, i, actual, expected_[ordinal][i]);
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }

  void VerifyInstructions() {
    CheckStatus(api_->host_mapping_cache_control(
                    instructions_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                    0, instruction_byte_length_),
                "instruction_invalidation");
    Check(std::memcmp(original_instructions_.data(), instructions_.pointer,
                      instruction_byte_length_) == 0,
          "prepared instructions changed");
  }

  // Negotiated core and XDNA API tables borrowed from the linked provider.
  const amdf_api_t* api_ = nullptr;
  // XDNA extension paired with api_.
  const amdf_xdna_api_t* xdna_api_ = nullptr;
  // Shared ordinary device owned by the CTS cache.
  amdf_device_t* device_ = nullptr;
  // Capability skip reason, null after complete native initialization.
  const char* skip_reason_ = "no XDNA endpoint present";
  // Executable-only family metadata; native queues are managed through libamdf.
  iree_hal_queue_family_spec_t queue_family_spec_ = {};
  // HAL queue-family descriptor borrowed by the executable.
  iree_hal_queue_family_t queue_family_ = {};
  // Immutable parsed image retaining its owned input bytes.
  iree_hal_executable_t* executable_ = nullptr;
  // Native context outliving its private instruction backing and queue.
  amdf_xdna_context_t* context_ = nullptr;
  // One resident instruction allocation and explicit host view.
  MappedMemory instructions_;
  // Used instruction prefix, independent of native allocation granularity.
  iree_host_size_t instruction_byte_length_ = 0;
  // Immutable prepared command retaining the executable and HAL wrappers.
  iree_hal_amd_xdna_prepared_command_t* prepared_ = nullptr;
  // Native publication lease borrowing context_.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Untimed byte oracle captured after cold relocation.
  std::vector<uint8_t> original_instructions_;
  struct Binding {
    // Native data backing with a guard region on either side of the payload.
    MappedMemory storage;
    // HAL wrapper borrowing the middle 64 bytes of storage.
    iree_hal_buffer_t* buffer = nullptr;
  };
  // Two inputs and one output retained across all repetitions.
  std::array<Binding, 3> bindings_;
  // Expected payloads for the current completed command.
  std::array<BindingValues, 3> expected_;
  // Input rotation shared across rows, warmups and repetitions.
  size_t input_iteration_ = 0;
};

}  // namespace

int main(int argument_count, char** argument_values) {
  if (!amdf_cts_provider_initialize(&argument_count, &argument_values))
    return EXIT_FAILURE;
  benchmark::Initialize(&argument_count, argument_values);
  if (benchmark::ReportUnrecognizedArguments(argument_count, argument_values))
    return EXIT_FAILURE;
  ExecutionBenchmark fixture;
  fixture.Initialize();
  benchmark::RegisterBenchmark(
      "XdnaExecution/Submit",
      [&fixture](benchmark::State& state) {
        fixture.Run<CompletionTiming::kExcluded>(state);
      })
      ->UseRealTime();
  benchmark::RegisterBenchmark(
      "XdnaExecution/SubmitAndWait",
      [&fixture](benchmark::State& state) {
        fixture.Run<CompletionTiming::kIncluded>(state);
      })
      ->UseRealTime();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  fixture.Deinitialize();
  CheckStatus(GetCtsDeviceCache().Deinitialize(), "device_cleanup");
  return amdf_cts_provider_deinitialize() ? EXIT_SUCCESS : EXIT_FAILURE;
}
