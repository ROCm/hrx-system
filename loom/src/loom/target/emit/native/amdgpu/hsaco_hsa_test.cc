// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hsa/hsa.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hsa/hsa_ext_amd.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/debugging.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/hal/binding_materialization.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/low_descriptor_registry.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

constexpr char kAsanConfigGlobalName[] = "iree_asan_config";
constexpr uint32_t kAsanConfigByteLength = 96;
constexpr char kFeedbackConfigGlobalName[] = "iree_feedback_config";
constexpr uint32_t kFeedbackConfigByteLength = 64;

iree_status_t PrintLowVerifyDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  fprintf(stderr, "low verifier diagnostic: %s: %s\n",
          emission->error ? emission->error->error_id : "<unknown>",
          emission->error ? emission->error->summary : "<unknown>");
  for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
    const loom_diagnostic_param_t* param = &emission->params[i];
    const char* name = emission->error && i < emission->error->param_count
                           ? emission->error->param_defs[i].name
                           : "<param>";
    switch (param->kind) {
      case LOOM_PARAM_STRING:
        fprintf(stderr, "  %s = %.*s\n", name, (int)param->string.size,
                param->string.data ? param->string.data : "");
        break;
      case LOOM_PARAM_I64:
        fprintf(stderr, "  %s = %" PRId64 "\n", name, param->i64);
        break;
      case LOOM_PARAM_U32:
        fprintf(stderr, "  %s = %" PRIu32 "\n", name, param->u32);
        break;
      case LOOM_PARAM_U64:
        fprintf(stderr, "  %s = %" PRIu64 "\n", name, param->u64);
        break;
      case LOOM_PARAM_BOOL:
        fprintf(stderr, "  %s = %s\n", name, param->boolean ? "true" : "false");
        break;
      case LOOM_PARAM_TYPE:
        fprintf(stderr, "  %s = <type>\n", name);
        break;
      case LOOM_PARAM_STRING_LIST:
        fprintf(stderr, "  %s = [", name);
        for (iree_host_size_t j = 0; j < param->string_list.count; ++j) {
          if (j != 0) fprintf(stderr, ", ");
          const iree_string_view_t value = param->string_list.values[j];
          fprintf(stderr, "%.*s", (int)value.size,
                  value.data ? value.data : "");
        }
        fprintf(stderr, "]\n");
        break;
      default:
        fprintf(stderr, "  %s = <unknown>\n", name);
        break;
    }
  }
  return iree_ok_status();
}

void RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                     DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, dialect_id, vtables,
                                               (uint16_t)count));
}

void InitializeLowKernelContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  IREE_ASSERT_OK(loom_amdgpu_ops_register_dialect(context));
  RegisterDialect(context, LOOM_DIALECT_LOW, loom_low_dialect_vtables);
  IREE_ASSERT_OK(loom_context_finalize(context));
}

iree_status_t FormatTargetRecordForProfile(
    const loom_amdgpu_target_profile_t* profile,
    std::string* out_target_record) {
  IREE_ASSERT_ARGUMENT(profile);
  IREE_ASSERT_ARGUMENT(out_target_record);
  *out_target_record = {};
  const loom_amdgpu_target_info_t* target = profile->identity.target;
  const loom_amdgpu_target_record_info_t* record_info =
      loom_amdgpu_target_record_info_for_target(target->name);
  if (record_info == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU HSA target '%.*s' has no target record",
                            (int)target->name.size, target->name.data);
  }
  std::string target_record = "amdgpu.target<";
  target_record.append(record_info->target_name.data,
                       record_info->target_name.size);
  target_record += "> @gfx_target";
  bool has_attrs = false;
  const auto append_attr = [&](const char* name, iree_string_view_t value) {
    target_record += has_attrs ? ", " : " {";
    target_record += name;
    target_record += " = ";
    target_record.append(value.data, value.size);
    has_attrs = true;
  };
  const loom_amdgpu_target_feature_state_t sramecc =
      profile->identity.amdhsa_features.sramecc;
  if (sramecc == LOOM_AMDGPU_TARGET_FEATURE_OFF ||
      sramecc == LOOM_AMDGPU_TARGET_FEATURE_ON) {
    append_attr("sramecc", loom_amdgpu_target_feature_state_attr_name(sramecc));
  }
  const loom_amdgpu_target_feature_state_t xnack =
      profile->identity.amdhsa_features.xnack;
  if (xnack == LOOM_AMDGPU_TARGET_FEATURE_OFF ||
      xnack == LOOM_AMDGPU_TARGET_FEATURE_ON) {
    append_attr("xnack", loom_amdgpu_target_feature_state_attr_name(xnack));
  }
  if (has_attrs) {
    target_record += "}";
  }
  target_record += "\n";
  *out_target_record = std::move(target_record);
  return iree_ok_status();
}

struct HsaApi {
  // Loaded HSA runtime shared library.
  iree_dynamic_library_t* library = nullptr;
  // `hsa_init` entry point.
  decltype(&::hsa_init) hsa_init = nullptr;
  // `hsa_shut_down` entry point.
  decltype(&::hsa_shut_down) hsa_shut_down = nullptr;
  // `hsa_status_string` entry point.
  decltype(&::hsa_status_string) hsa_status_string = nullptr;
  // `hsa_iterate_agents` entry point.
  decltype(&::hsa_iterate_agents) hsa_iterate_agents = nullptr;
  // `hsa_agent_get_info` entry point.
  decltype(&::hsa_agent_get_info) hsa_agent_get_info = nullptr;
  // `hsa_agent_iterate_isas` entry point.
  decltype(&::hsa_agent_iterate_isas) hsa_agent_iterate_isas = nullptr;
  // `hsa_isa_get_info_alt` entry point.
  decltype(&::hsa_isa_get_info_alt) hsa_isa_get_info_alt = nullptr;
  // `hsa_code_object_reader_create_from_memory` entry point.
  decltype(&::hsa_code_object_reader_create_from_memory)
      hsa_code_object_reader_create_from_memory = nullptr;
  // `hsa_code_object_reader_destroy` entry point.
  decltype(&::hsa_code_object_reader_destroy) hsa_code_object_reader_destroy =
      nullptr;
  // `hsa_executable_create_alt` entry point.
  decltype(&::hsa_executable_create_alt) hsa_executable_create_alt = nullptr;
  // `hsa_executable_destroy` entry point.
  decltype(&::hsa_executable_destroy) hsa_executable_destroy = nullptr;
  // `hsa_executable_load_agent_code_object` entry point.
  decltype(&::hsa_executable_load_agent_code_object)
      hsa_executable_load_agent_code_object = nullptr;
  // `hsa_executable_freeze` entry point.
  decltype(&::hsa_executable_freeze) hsa_executable_freeze = nullptr;
  // `hsa_executable_get_symbol_by_name` entry point.
  decltype(&::hsa_executable_get_symbol_by_name)
      hsa_executable_get_symbol_by_name = nullptr;
  // `hsa_executable_symbol_get_info` entry point.
  decltype(&::hsa_executable_symbol_get_info) hsa_executable_symbol_get_info =
      nullptr;
};

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient HSACO writer storage.
  iree_arena_allocator_t arena_ = {0};
};

StreamPtr CreateStream() {
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  return bytes;
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  std::string result = iree_status_code_string(iree_status_code(status));
  if (iree_status_to_string(status, &allocator, &buffer, &buffer_length)) {
    result.assign(buffer, buffer_length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return result;
}

std::string HsaStatusString(const HsaApi& api, hsa_status_t status) {
  const char* status_string = nullptr;
  if (api.hsa_status_string != nullptr &&
      api.hsa_status_string(status, &status_string) == HSA_STATUS_SUCCESS &&
      status_string != nullptr) {
    return status_string;
  }
  return std::to_string((int)status);
}

template <typename Fn, typename... Args>
hsa_status_t CallHsa(Fn fn, Args... args) {
  IREE_LEAK_CHECK_DISABLE_PUSH();
  hsa_status_t result = fn(args...);
  IREE_LEAK_CHECK_DISABLE_POP();
  return result;
}

template <typename Fn>
iree_status_t LoadHsaSymbol(HsaApi* api, const char* symbol_name, Fn* out_fn) {
  IREE_ASSERT_ARGUMENT(api);
  IREE_ASSERT_ARGUMENT(out_fn);
  void* symbol = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_dynamic_library_lookup_symbol(api->library, symbol_name, &symbol));
  *out_fn = reinterpret_cast<Fn>(symbol);
  return iree_ok_status();
}

void AppendHsaLibraryCandidate(std::vector<std::string>* candidates,
                               std::string candidate) {
  IREE_ASSERT_ARGUMENT(candidates);
  if (candidate.empty()) {
    return;
  }
  for (const std::string& existing_candidate : *candidates) {
    if (existing_candidate == candidate) {
      return;
    }
  }
  candidates->push_back(std::move(candidate));
}

void AppendHsaLibraryCandidates(std::vector<std::string>* candidates) {
  IREE_ASSERT_ARGUMENT(candidates);
  static const char* kLibraryNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
      "hsa-runtime64.dll",
#else
      "libhsa-runtime64.so.1",
      "libhsa-runtime64.so",
#endif  // IREE_PLATFORM_WINDOWS
  };
  const char* env_path = std::getenv("IREE_HAL_AMDGPU_LIBHSA_PATH");
  if (env_path != nullptr && env_path[0] != '\0') {
    const std::string path_fragment(env_path);
    AppendHsaLibraryCandidate(candidates, path_fragment);
    for (const char* library_name : kLibraryNames) {
      std::string candidate_path = path_fragment;
      if (candidate_path.back() != '/') {
        candidate_path.push_back('/');
      }
      candidate_path.append(library_name);
      AppendHsaLibraryCandidate(candidates, std::move(candidate_path));
    }
  }
  for (const char* library_name : kLibraryNames) {
    AppendHsaLibraryCandidate(candidates, library_name);
  }
}

iree_status_t LoadHsaLibrary(HsaApi* api) {
  IREE_ASSERT_ARGUMENT(api);
  std::vector<std::string> candidates;
  AppendHsaLibraryCandidates(&candidates);

  std::string failures;
  for (const std::string& candidate : candidates) {
    iree_status_t status = iree_dynamic_library_load_from_file(
        candidate.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
        iree_allocator_system(), &api->library);
    if (iree_status_is_ok(status)) {
      return iree_ok_status();
    }
    failures.append("\n  Tried: ");
    failures.append(candidate);
    failures.append("\n    ");
    failures.append(StatusToStringAndFree(status));
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "HSA runtime unavailable; specify "
                          "IREE_HAL_AMDGPU_LIBHSA_PATH if the runtime is "
                          "not on the loader path:%s",
                          failures.c_str());
}

void UnloadHsaApi(HsaApi* api) {
  IREE_ASSERT_ARGUMENT(api);
  iree_dynamic_library_release(api->library);
  *api = {};
}

iree_status_t LoadHsaApi(HsaApi* out_api) {
  IREE_ASSERT_ARGUMENT(out_api);
  *out_api = {};
  HsaApi api = {};
  iree_status_t status = LoadHsaLibrary(&api);
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_init", &api.hsa_init);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_shut_down", &api.hsa_shut_down);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_status_string", &api.hsa_status_string);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_iterate_agents", &api.hsa_iterate_agents);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_agent_get_info", &api.hsa_agent_get_info);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_agent_iterate_isas",
                           &api.hsa_agent_iterate_isas);
  }
  if (iree_status_is_ok(status)) {
    status =
        LoadHsaSymbol(&api, "hsa_isa_get_info_alt", &api.hsa_isa_get_info_alt);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_code_object_reader_create_from_memory",
                           &api.hsa_code_object_reader_create_from_memory);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_code_object_reader_destroy",
                           &api.hsa_code_object_reader_destroy);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_create_alt",
                           &api.hsa_executable_create_alt);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_destroy",
                           &api.hsa_executable_destroy);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_load_agent_code_object",
                           &api.hsa_executable_load_agent_code_object);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_freeze",
                           &api.hsa_executable_freeze);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_get_symbol_by_name",
                           &api.hsa_executable_get_symbol_by_name);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_symbol_get_info",
                           &api.hsa_executable_symbol_get_info);
  }
  if (iree_status_is_ok(status)) {
    *out_api = api;
  } else {
    UnloadHsaApi(&api);
  }
  return status;
}

class HsaRuntime {
 public:
  HsaRuntime() = default;
  HsaRuntime(const HsaRuntime&) = delete;
  HsaRuntime& operator=(const HsaRuntime&) = delete;

  ~HsaRuntime() {
    if (initialized_) {
      EXPECT_EQ(CallHsa(api_.hsa_shut_down), HSA_STATUS_SUCCESS);
    }
    UnloadHsaApi(&api_);
  }

  const HsaApi& api() const { return api_; }

  HsaApi* mutable_api() { return &api_; }

  void MarkInitialized() { initialized_ = true; }

 private:
  // Dynamically loaded HSA API surface.
  HsaApi api_ = {};
  // True once `hsa_init` has successfully incremented the runtime refcount.
  bool initialized_ = false;
};

class HsaCodeObjectReader {
 public:
  explicit HsaCodeObjectReader(const HsaApi* api) : api_(api) {}
  HsaCodeObjectReader(const HsaCodeObjectReader&) = delete;
  HsaCodeObjectReader& operator=(const HsaCodeObjectReader&) = delete;

  ~HsaCodeObjectReader() { Reset(); }

  hsa_code_object_reader_t get() const { return reader_; }

  hsa_code_object_reader_t* out() {
    Reset();
    return &reader_;
  }

  void Reset() {
    if (reader_.handle != 0) {
      EXPECT_EQ(CallHsa(api_->hsa_code_object_reader_destroy, reader_),
                HSA_STATUS_SUCCESS);
      reader_ = {};
    }
  }

 private:
  // HSA entry points used to release the reader.
  const HsaApi* api_ = nullptr;
  // Owned HSA code object reader handle.
  hsa_code_object_reader_t reader_ = {};
};

class HsaExecutable {
 public:
  explicit HsaExecutable(const HsaApi* api) : api_(api) {}
  HsaExecutable(const HsaExecutable&) = delete;
  HsaExecutable& operator=(const HsaExecutable&) = delete;

  ~HsaExecutable() { Reset(); }

  hsa_executable_t get() const { return executable_; }

  hsa_executable_t* out() {
    Reset();
    return &executable_;
  }

  void Reset() {
    if (executable_.handle != 0) {
      EXPECT_EQ(CallHsa(api_->hsa_executable_destroy, executable_),
                HSA_STATUS_SUCCESS);
      executable_ = {};
    }
  }

 private:
  // HSA entry points used to release the executable.
  const HsaApi* api_ = nullptr;
  // Owned HSA executable handle.
  hsa_executable_t executable_ = {};
};

struct GpuAgentSearch {
  // HSA API used while iterating agents.
  const HsaApi* api = nullptr;
  // First discovered matching agent.
  hsa_agent_t agent = {};
  // HSA-reported agent name used for diagnostics.
  std::string agent_name;
  // True once |agent| has been populated.
  bool found = false;
};

hsa_status_t FindFirstAgent(hsa_agent_t agent, void* user_data) {
  GpuAgentSearch* search = reinterpret_cast<GpuAgentSearch*>(user_data);
  hsa_device_type_t device_type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = CallHsa(search->api->hsa_agent_get_info, agent,
                                HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (device_type == HSA_DEVICE_TYPE_GPU) {
    char agent_name[64] = {0};
    status = CallHsa(search->api->hsa_agent_get_info, agent,
                     HSA_AGENT_INFO_NAME, agent_name);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    search->agent = agent;
    search->agent_name = agent_name;
    search->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

iree_status_t QueryHsaIsaName(const HsaApi& api, hsa_isa_t isa,
                              std::string* out_name) {
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = {};
  uint32_t name_length = 0;
  hsa_status_t status = CallHsa(api.hsa_isa_get_info_alt, isa,
                                HSA_ISA_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "hsa_isa_get_info_alt(HSA_ISA_INFO_NAME_LENGTH) failed: %s",
        HsaStatusString(api, status).c_str());
  }
  if (name_length == 0 || name_length > 1024) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HSA ISA name has invalid length %" PRIu32,
                            name_length);
  }

  std::string buffer((size_t)name_length + 1u, '\0');
  status =
      CallHsa(api.hsa_isa_get_info_alt, isa, HSA_ISA_INFO_NAME, buffer.data());
  if (status != HSA_STATUS_SUCCESS) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "hsa_isa_get_info_alt(HSA_ISA_INFO_NAME) failed: %s",
        HsaStatusString(api, status).c_str());
  }

  size_t actual_length = 0;
  while (actual_length < buffer.size() && buffer[actual_length] != '\0') {
    ++actual_length;
  }
  if (actual_length == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HSA ISA name is empty");
  }
  *out_name = buffer.substr(0, actual_length);
  return iree_ok_status();
}

struct AgentIsaSearch {
  // HSA API used while iterating agent ISAs.
  const HsaApi* api = nullptr;
  // First HSA ISA name reported for the selected GPU agent.
  std::string isa_name;
  // IREE status text captured if ISA name query fails inside the callback.
  std::string failure;
  // True once |isa_name| has been populated.
  bool found = false;
};

hsa_status_t FindFirstAgentIsa(hsa_isa_t isa, void* user_data) {
  AgentIsaSearch* search = reinterpret_cast<AgentIsaSearch*>(user_data);
  iree_status_t status = QueryHsaIsaName(*search->api, isa, &search->isa_name);
  if (!iree_status_is_ok(status)) {
    search->failure = StatusToStringAndFree(status);
    return HSA_STATUS_ERROR;
  }
  search->found = true;
  return HSA_STATUS_INFO_BREAK;
}

bool TryFindFirstGpuAgent(const HsaApi& api, hsa_agent_t* out_agent,
                          std::string* out_agent_name,
                          std::string* out_skip_reason) {
  IREE_ASSERT_ARGUMENT(out_agent);
  IREE_ASSERT_ARGUMENT(out_agent_name);
  IREE_ASSERT_ARGUMENT(out_skip_reason);
  GpuAgentSearch search = {/*.api=*/&api};
  hsa_status_t status =
      CallHsa(api.hsa_iterate_agents, FindFirstAgent, &search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    ADD_FAILURE() << "hsa_iterate_agents failed: "
                  << HsaStatusString(api, status);
    return false;
  }
  if (!search.found) {
    *out_skip_reason = "no HSA GPU agent found";
    return false;
  }
  *out_agent = search.agent;
  *out_agent_name = search.agent_name;
  return true;
}

struct AmdgpuHsaTarget {
  // HSA GPU agent that owns the queried target ISA.
  hsa_agent_t agent = {};
  // HSA-reported GPU agent name used for diagnostics.
  std::string agent_name;
  // Full HSA target id reported by the runtime for |agent|.
  std::string isa_name;
  // Canonical AMDGPU target identity resolved from physical HSA facts.
  loom_amdgpu_target_identity_t identity = {};
};

bool TryDiscoverCurrentAmdgpuTarget(const HsaApi& api,
                                    AmdgpuHsaTarget* out_target,
                                    std::string* out_skip_reason) {
  IREE_ASSERT_ARGUMENT(out_target);
  IREE_ASSERT_ARGUMENT(out_skip_reason);
  *out_target = {};
  hsa_agent_t agent = {};
  std::string agent_name;
  if (!TryFindFirstGpuAgent(api, &agent, &agent_name, out_skip_reason)) {
    return false;
  }

  AgentIsaSearch isa_search = {/*.api=*/&api};
  hsa_status_t status = CallHsa(api.hsa_agent_iterate_isas, agent,
                                FindFirstAgentIsa, &isa_search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    if (!isa_search.failure.empty()) {
      ADD_FAILURE() << isa_search.failure;
    } else {
      ADD_FAILURE() << "hsa_agent_iterate_isas failed: "
                    << HsaStatusString(api, status);
    }
    return false;
  }
  if (!isa_search.found) {
    *out_skip_reason =
        "HSA GPU agent '" + agent_name + "' did not report any ISAs";
    return false;
  }

  loom_amdgpu_amdhsa_target_id_t target_id = {};
  iree_status_t target_status = loom_amdgpu_target_info_parse_amdhsa_target_id(
      iree_make_string_view(isa_search.isa_name.data(),
                            isa_search.isa_name.size()),
      &target_id);
  if (!iree_status_is_ok(target_status)) {
    *out_skip_reason = StatusToStringAndFree(target_status);
    return false;
  }
  uint32_t physical_asic_revision = 0;
  if (loom_amdgpu_target_info_requires_physical_resolution(
          target_id.processor)) {
    status = CallHsa(api.hsa_agent_get_info, agent,
                     (hsa_agent_info_t)HSA_AMD_AGENT_INFO_ASIC_REVISION,
                     &physical_asic_revision);
    if (status != HSA_STATUS_SUCCESS) {
      ADD_FAILURE() << "hsa_agent_get_info(ASIC_REVISION) failed: "
                    << HsaStatusString(api, status);
      return false;
    }
  }
  const loom_amdgpu_target_info_t* target = nullptr;
  target_status = loom_amdgpu_target_info_lookup_physical_target(
      target_id.processor, physical_asic_revision, &target);
  if (!iree_status_is_ok(target_status)) {
    *out_skip_reason = StatusToStringAndFree(target_status);
    return false;
  }

  *out_target = {
      /*.agent=*/agent,
      /*.agent_name=*/std::move(agent_name),
      /*.isa_name=*/std::move(isa_search.isa_name),
      /*.identity=*/
      {
          /*.target=*/target,
          /*.amdhsa_features=*/target_id.features,
      },
  };
  return true;
}

loom_op_t* FindFirstLowFunction(loom_module_t* module) {
  loom_block_t* block = loom_module_block(module);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(block, op) {
    if (loom_low_function_def_isa(op)) {
      return op;
    }
  }
  return nullptr;
}

class LowKernelEmitter {
 public:
  LowKernelEmitter() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    InitializeLowKernelContext(&context_);
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
  }

  LowKernelEmitter(const LowKernelEmitter&) = delete;
  LowKernelEmitter& operator=(const LowKernelEmitter&) = delete;

  ~LowKernelEmitter() {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t EmitKernel(const loom_amdgpu_target_profile_t* target_profile,
                           const std::string& kernel_source,
                           std::string* out_hsaco,
                           iree_arena_allocator_t* arena) {
    IREE_ASSERT_ARGUMENT(out_hsaco);
    IREE_ASSERT_ARGUMENT(arena);
    *out_hsaco = {};
    std::string source;
    IREE_RETURN_IF_ERROR(FormatTargetRecordForProfile(target_profile, &source));
    source += kernel_source;
    IREE_RETURN_IF_ERROR(ParseSource(source));

    loom_op_t* low_function = FindFirstLowFunction(module_);
    if (low_function == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU HSA low kernel has no low func");
    }

    loom_symbol_fact_table_t symbol_facts = {};
    loom_symbol_fact_table_initialize(&symbol_facts, arena);
    loom_low_resolved_target_t target = {};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module_, &symbol_facts, low_function,
        /*function_target_facts=*/nullptr, &target_registry_.registry,
        iree_diagnostic_emitter_t{}, &target));
    const loom_low_descriptor_set_t* descriptor_set = target.descriptor_set;
    if (descriptor_set == nullptr) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU HSA low kernel target has no descriptor set");
    }
    loom_amdgpu_hal_binding_materialization_result_t materialization = {};
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_binding_materialize(
        module_, low_function, descriptor_set, &materialization, arena));

    loom_amdgpu_hal_kernel_abi_verify_result_t abi_verify_result = {};
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
        module_, low_function, descriptor_set, /*max_errors=*/20,
        iree_diagnostic_emitter_t{
            /*.fn=*/PrintLowVerifyDiagnostic,
            /*.user_data=*/nullptr,
        },
        &abi_verify_result, arena));
    if (abi_verify_result.error_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HSA low kernel failed HAL ABI verification");
    }

    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &target_registry_.registry;
    verify_options.emitter = {
        /*.fn=*/PrintLowVerifyDiagnostic,
        /*.user_data=*/nullptr,
    };
    verify_options.max_errors = 20;
    loom_low_verify_result_t verify_result = {};
    loom_low_verify_scratch_t verify_scratch =
        loom_low_verify_scratch_for_module(module_);
    IREE_RETURN_IF_ERROR(loom_low_verify_module(
        module_, &verify_options, &verify_scratch, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HSA low kernel failed low verification");
    }

    loom_low_storage_lease_provider_t storage_lease_provider = {};
    loom_amdgpu_storage_lease_provider(&storage_lease_provider);
    loom_low_emission_frame_options_t frame_options = {};
    frame_options.descriptor_registry = &target_registry_.registry;
    frame_options.allocation_fixed_values = abi_verify_result.fixed_values;
    frame_options.allocation_fixed_value_count =
        abi_verify_result.fixed_value_count;
    frame_options.storage_lease_provider = &storage_lease_provider;
    loom_low_emission_frame_t frame = {};
    IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
        module_, low_function, &frame_options, arena, &frame));

    loom_amdgpu_packet_plan_t packet_plan = {};
    IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
        &frame.schedule, &frame.allocation, arena, &packet_plan));

    StreamPtr stream = CreateStream();
    const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
        /*.abi_layout=*/&materialization.abi_layout,
        /*.abi_verify=*/&abi_verify_result,
        /*.preflight=*/{},
        /*.packet_plan=*/&packet_plan,
    };
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_kernel_hsaco(&frame.schedule, &frame.allocation,
                                      &hsaco_options, stream.get(), arena));
    *out_hsaco = StreamBytes(stream.get());
    return iree_ok_status();
  }

 private:
  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  iree_status_t ParseSource(const std::string& source) {
    ResetModule();
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &target_registry_.registry, &parse_options.low_asm_environment);
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_hsaco_hsa_smoke.loom"), &context_,
                        &block_pool_, &parse_options, &module_));
    if (module_ == nullptr) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HSA low kernel source failed to parse");
    }
    return iree_ok_status();
  }

  // Block pool backing parser and context allocations.
  iree_arena_block_pool_t block_pool_ = {0};
  // Loom context containing dialect/type registration for parsing and verify.
  loom_context_t context_ = {};
  // Parsed module owned by this emitter instance.
  loom_module_t* module_ = nullptr;
  // AMDGPU-only descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

iree_status_t PrepareTargetProfileForLowHsaco(
    const AmdgpuHsaTarget& target,
    loom_amdgpu_target_profile_t* out_target_profile) {
  IREE_ASSERT_ARGUMENT(out_target_profile);
  *out_target_profile = {};
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target.identity.target);
  IREE_ASSERT(processor != nullptr);
  if (!loom_amdgpu_processor_properties_support_hsaco(&processor->properties)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU target '%.*s' does not have native HSACO support",
        (int)target.identity.target->name.size,
        target.identity.target->name.data);
  }
  const loom_amdgpu_target_record_info_t* record_info =
      loom_amdgpu_target_record_info_for_target(target.identity.target->name);
  if (record_info == nullptr) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMDGPU target '%.*s' has no target-low record",
                            (int)target.identity.target->name.size,
                            target.identity.target->name.data);
  }
  return loom_amdgpu_target_profile_initialize(&target.identity,
                                               out_target_profile);
}

void AppendLowKernelTargetClause(iree_string_view_t representation_contract,
                                 std::string* source) {
  source->append("target<");
  source->append(representation_contract.data, representation_contract.size);
  source->append(">(@gfx_target)");
}

iree_status_t EmitWorkitemStoreKernelForAmdgpu(const AmdgpuHsaTarget& target,
                                               std::string* out_hsaco) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = {};
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_RETURN_IF_ERROR(
      PrepareTargetProfileForLowHsaco(target, &target_profile));

  TestArena arena;
  LowKernelEmitter emitter;
  std::string source = "low.kernel.def ";
  AppendLowKernelTargetClause(
      target_profile.identity.target->descriptor_set_key, &source);
  source.append(
      " @loom_kernel() {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %four = low.const<amdgpu.v_mov_b32> {imm32 = 4} : "
      "reg<amdgpu.vgpr>\n"
      "  %byte_offset = low.op<amdgpu.v_mul_lo_u32>(%tid, %four) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %binding = low.resource<hal_binding> {index = 0, "
      "source_type = hal.buffer} : reg<amdgpu.sgpr x2>\n"
      "  %descriptor = low.op<amdgpu.hal.buffer_descriptor>(%binding) "
      "{cache_swizzle_stride = 0, extent = 256} : "
      "(reg<amdgpu.sgpr x2>) -> reg<amdgpu.sgpr x4>\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%tid, %descriptor, %byte_offset, "
      "%zero) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n");
  return emitter.EmitKernel(&target_profile, source, out_hsaco, arena.arena());
}

iree_status_t EmitB128CopyKernelForAmdgpu(const AmdgpuHsaTarget& target,
                                          std::string* out_hsaco) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = {};
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_RETURN_IF_ERROR(
      PrepareTargetProfileForLowHsaco(target, &target_profile));
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target_profile.identity.target);
  IREE_ASSERT(processor != nullptr);

  std::string source = "low.kernel.def ";
  AppendLowKernelTargetClause(
      target_profile.identity.target->descriptor_set_key, &source);
  source.append(
      " @loom_kernel() {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %byte_offset = low.op<amdgpu.v_lshlrev_b32.src0_inline>(%tid) "
      "{imm32 = 4} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %source = low.resource<hal_binding> {index = 0, source_type "
      "= hal.buffer} : reg<amdgpu.sgpr x2>\n"
      "  %target = low.resource<hal_binding> {index = 1, source_type "
      "= hal.buffer} : reg<amdgpu.sgpr x2>\n");
  switch (target_profile.identity.target->descriptor_set_ordinal) {
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4:
      source +=
          "  %m0 = low.const<amdgpu.s_mov_b32_m0.imm> {imm32 = 0} : "
          "reg<amdgpu.m0>\n"
          "  %loaded = low.op<amdgpu.global_load_b128_saddr>(%byte_offset, "
          "%source, %m0) : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>, "
          "reg<amdgpu.m0>) -> reg<amdgpu.vgpr x4>\n"
          "  low.op<amdgpu.global_store_b128_saddr>(%byte_offset, %loaded, "
          "%target, %m0) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
          "reg<amdgpu.sgpr x2>, reg<amdgpu.m0>)\n";
      break;
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X:
      source +=
          "  %loaded = low.op<amdgpu.global_load_b128_saddr>(%byte_offset, "
          "%source) : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>) -> "
          "reg<amdgpu.vgpr x4>\n"
          "  low.op<amdgpu.global_store_b128_saddr>(%byte_offset, %loaded, "
          "%target) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
          "reg<amdgpu.sgpr x2>)\n";
      break;
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU B128 smoke kernel does not know the "
                              "descriptor-set operand shape");
  }
  source +=
      "  low.return\n"
      "}\n";
  TestArena arena;
  LowKernelEmitter emitter;
  return emitter.EmitKernel(&target_profile, source, out_hsaco, arena.arena());
}

std::string CodeObjectTargetIdForIdentity(
    const loom_amdgpu_target_identity_t& identity) {
  TestArena arena;
  iree_string_view_t target_id = iree_string_view_empty();
  IREE_CHECK_OK(loom_amdgpu_amdhsa_code_object_target_id_format(
      &identity, arena.arena(), &target_id));
  return std::string(target_id.data, target_id.size);
}

loom_amdgpu_metadata_kernel_t MinimalKernel(iree_string_view_t name,
                                            iree_string_view_t symbol,
                                            uint32_t wavefront_size) {
  return {
      /*.name=*/name,
      /*.descriptor_symbol=*/symbol,
      /*.kernarg_segment_size=*/0,
      /*.kernarg_segment_alignment=*/8,
      /*.wavefront_size=*/wavefront_size,
      /*.group_segment_fixed_size=*/0,
      /*.private_segment_fixed_size=*/0,
      /*.sgpr_count=*/4,
      /*.vgpr_count=*/1,
      /*.max_flat_workgroup_size=*/64,
      /*.required_workgroup_size=*/{/*.x=*/64, /*.y=*/1, /*.z=*/1},
      /*.has_required_workgroup_size=*/true,
      /*.workgroup_cluster_size=*/{},
      /*.has_workgroup_cluster_size=*/false,
      /*.target_extensions=*/{},
      /*.arguments=*/nullptr,
      /*.argument_count=*/0,
  };
}

iree_status_t EmitRuntimeGlobalKernelForAmdgpu(const AmdgpuHsaTarget& target,
                                               std::string* out_hsaco) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = {};
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_RETURN_IF_ERROR(
      PrepareTargetProfileForLowHsaco(target, &target_profile));
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target_profile.identity.target);
  IREE_ASSERT(processor != nullptr);

  const uint8_t s_endpgm[] = {0x00, 0x00, 0x81, 0xbf};
  const loom_amdgpu_hsaco_kernel_t kernel = {
      /*.metadata=*/MinimalKernel(IREE_SV("loom_kernel"),
                                  IREE_SV("loom_kernel.kd"),
                                  processor->properties.wavefront.default_size),
      /*.descriptor_options=*/{},
      /*.text=*/iree_make_const_byte_span(s_endpgm, sizeof(s_endpgm)),
  };
  const loom_amdgpu_hsaco_data_symbol_t data_symbols[] = {
      {
          /*.name=*/IREE_SV(kAsanConfigGlobalName),
          /*.initial_contents=*/{},
          /*.byte_length=*/kAsanConfigByteLength,
          /*.alignment=*/8,
          /*.flags=*/LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
      },
      {
          /*.name=*/IREE_SV(kFeedbackConfigGlobalName),
          /*.initial_contents=*/{},
          /*.byte_length=*/kFeedbackConfigByteLength,
          /*.alignment=*/8,
          /*.flags=*/LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
      },
  };
  const std::string target_id =
      CodeObjectTargetIdForIdentity(target_profile.identity);
  loom_amdgpu_hsaco_kernel_t revisioned_kernel = kernel;
  revisioned_kernel.metadata.target_extensions =
      target_profile.properties.kernel_metadata_extensions;
  const loom_amdgpu_hsaco_file_t file = {
      /*.target=*/iree_make_string_view(target_id.data(), target_id.size()),
      /*.processor=*/processor->name,
      /*.kernels=*/&revisioned_kernel,
      /*.kernel_count=*/1,
      /*.data_symbols=*/data_symbols,
      /*.data_symbol_count=*/IREE_ARRAYSIZE(data_symbols),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  *out_hsaco = StreamBytes(stream.get());
  return iree_ok_status();
}

iree_status_t CheckHsaStatus(const HsaApi& api, hsa_status_t status,
                             const char* call_name) {
  if (status == HSA_STATUS_SUCCESS) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "%s failed: %s", call_name,
                          HsaStatusString(api, status).c_str());
}

iree_status_t LoadExecutable(const HsaApi& api, const AmdgpuHsaTarget& target,
                             const std::string& hsaco,
                             HsaExecutable* executable) {
  IREE_ASSERT_ARGUMENT(executable);

  HsaCodeObjectReader reader(&api);
  IREE_RETURN_IF_ERROR(
      CheckHsaStatus(api,
                     CallHsa(api.hsa_code_object_reader_create_from_memory,
                             hsaco.data(), hsaco.size(), reader.out()),
                     "hsa_code_object_reader_create_from_memory"));

  IREE_RETURN_IF_ERROR(
      CheckHsaStatus(api,
                     CallHsa(api.hsa_executable_create_alt, HSA_PROFILE_FULL,
                             HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                             executable->out()),
                     "hsa_executable_create_alt"));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_load_agent_code_object, executable->get(),
              target.agent, reader.get(), nullptr, nullptr),
      "hsa_executable_load_agent_code_object"));
  reader.Reset();
  return CheckHsaStatus(
      api, CallHsa(api.hsa_executable_freeze, executable->get(), nullptr),
      "hsa_executable_freeze");
}

struct LoadedKernelInfo {
  // Kernel object value to put in AQL dispatch packets.
  uint64_t kernel_object = 0;
  // Kernarg segment size reported by the HSA loader.
  uint32_t kernarg_segment_size = 0;
  // Group segment size reported by the HSA loader.
  uint32_t group_segment_size = 0;
  // Private segment size reported by the HSA loader.
  uint32_t private_segment_size = 0;
};

struct LoadedVariableInfo {
  // HAL global publication relies on variable kind, address, and byte length.
  // Device-visible variable address reported after executable freeze.
  uint64_t address = 0;
  // HSA variable segment.
  hsa_variable_segment_t segment = HSA_VARIABLE_SEGMENT_READONLY;
  // Byte length reported by the HSA loader.
  uint32_t byte_length = 0;
};

iree_status_t LoadKernelExecutable(const HsaApi& api,
                                   const AmdgpuHsaTarget& target,
                                   const std::string& hsaco,
                                   const char* symbol_name,
                                   HsaExecutable* executable,
                                   LoadedKernelInfo* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = {};

  IREE_RETURN_IF_ERROR(LoadExecutable(api, target, hsaco, executable));

  hsa_executable_symbol_t symbol = {};
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_get_symbol_by_name, executable->get(),
              symbol_name, &target.agent, &symbol),
      "hsa_executable_get_symbol_by_name"));
  IREE_RETURN_IF_ERROR(
      CheckHsaStatus(api,
                     CallHsa(api.hsa_executable_symbol_get_info, symbol,
                             HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                             &out_info->kernel_object),
                     "hsa_executable_symbol_get_info(KERNEL_OBJECT)"));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
              &out_info->kernarg_segment_size),
      "hsa_executable_symbol_get_info(KERNARG_SEGMENT_SIZE)"));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
              &out_info->group_segment_size),
      "hsa_executable_symbol_get_info(GROUP_SEGMENT_SIZE)"));
  return CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
              &out_info->private_segment_size),
      "hsa_executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)");
}

iree_status_t QueryLoadedVariable(const HsaApi& api,
                                  const AmdgpuHsaTarget& target,
                                  HsaExecutable* executable,
                                  const char* symbol_name,
                                  LoadedVariableInfo* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = {};

  hsa_executable_symbol_t symbol = {};
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_get_symbol_by_name, executable->get(),
              symbol_name, &target.agent, &symbol),
      "hsa_executable_get_symbol_by_name"));

  hsa_symbol_kind_t symbol_kind = HSA_SYMBOL_KIND_KERNEL;
  IREE_RETURN_IF_ERROR(
      CheckHsaStatus(api,
                     CallHsa(api.hsa_executable_symbol_get_info, symbol,
                             HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbol_kind),
                     "hsa_executable_symbol_get_info(TYPE)"));
  if (symbol_kind != HSA_SYMBOL_KIND_VARIABLE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HSA symbol '%s' is not a variable", symbol_name);
  }
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &out_info->address),
      "hsa_executable_symbol_get_info(VARIABLE_ADDRESS)"));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT, &out_info->segment),
      "hsa_executable_symbol_get_info(VARIABLE_SEGMENT)"));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      api,
      CallHsa(api.hsa_executable_symbol_get_info, symbol,
              HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &out_info->byte_length),
      "hsa_executable_symbol_get_info(VARIABLE_SIZE)"));
  return iree_ok_status();
}

iree_status_t InitializeCurrentAmdgpuTarget(HsaRuntime* runtime,
                                            AmdgpuHsaTarget* out_target) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = {};
  IREE_RETURN_IF_ERROR(LoadHsaApi(runtime->mutable_api()));
  IREE_RETURN_IF_ERROR(CheckHsaStatus(
      runtime->api(), CallHsa(runtime->api().hsa_init), "hsa_init"));
  runtime->MarkInitialized();

  std::string skip_reason;
  if (TryDiscoverCurrentAmdgpuTarget(runtime->api(), out_target,
                                     &skip_reason)) {
    return iree_ok_status();
  }
  if (!skip_reason.empty()) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "%s", skip_reason.c_str());
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "AMDGPU HSA target discovery failed");
}

using EmitLowKernelForTargetFn =
    iree_status_t (*)(const AmdgpuHsaTarget& target, std::string* out_hsaco);

void LoadLowKernelForCurrentTargetOrSkip(
    EmitLowKernelForTargetFn emit_kernel,
    uint32_t expected_kernarg_segment_size) {
  HsaRuntime runtime;
  AmdgpuHsaTarget target = {};
  iree_status_t target_status =
      InitializeCurrentAmdgpuTarget(&runtime, &target);
  if (iree_status_is_not_found(target_status) ||
      iree_status_is_unavailable(target_status) ||
      iree_status_is_unimplemented(target_status)) {
    GTEST_SKIP() << StatusToStringAndFree(target_status);
  }
  IREE_ASSERT_OK(target_status);

  std::string hsaco;
  iree_status_t emit_status = emit_kernel(target, &hsaco);
  if (iree_status_is_unimplemented(emit_status)) {
    GTEST_SKIP() << "Loom AMDGPU HSACO emitter does not support current ISA '"
                 << target.isa_name << "' from HSA agent '" << target.agent_name
                 << "': " << StatusToStringAndFree(emit_status);
  }
  IREE_ASSERT_OK(emit_status);

  const HsaApi& api = runtime.api();
  HsaExecutable executable(&api);
  LoadedKernelInfo kernel = {};
  IREE_ASSERT_OK(LoadKernelExecutable(api, target, hsaco, "loom_kernel.kd",
                                      &executable, &kernel));
  EXPECT_NE(kernel.kernel_object, 0u);
  EXPECT_EQ(kernel.kernarg_segment_size, expected_kernarg_segment_size);
  EXPECT_EQ(kernel.group_segment_size, 0u);
  EXPECT_EQ(kernel.private_segment_size, 0u);
}

void LoadRuntimeGlobalsForCurrentTargetOrSkip() {
  HsaRuntime runtime;
  AmdgpuHsaTarget target = {};
  iree_status_t target_status =
      InitializeCurrentAmdgpuTarget(&runtime, &target);
  if (iree_status_is_not_found(target_status) ||
      iree_status_is_unavailable(target_status) ||
      iree_status_is_unimplemented(target_status)) {
    GTEST_SKIP() << StatusToStringAndFree(target_status);
  }
  IREE_ASSERT_OK(target_status);

  std::string hsaco;
  iree_status_t emit_status = EmitRuntimeGlobalKernelForAmdgpu(target, &hsaco);
  if (iree_status_is_unimplemented(emit_status)) {
    GTEST_SKIP() << "Loom AMDGPU HSACO emitter does not support current ISA '"
                 << target.isa_name << "' from HSA agent '" << target.agent_name
                 << "': " << StatusToStringAndFree(emit_status);
  }
  IREE_ASSERT_OK(emit_status);

  const HsaApi& api = runtime.api();
  HsaExecutable executable(&api);
  IREE_ASSERT_OK(LoadExecutable(api, target, hsaco, &executable));

  LoadedVariableInfo asan_config = {};
  IREE_ASSERT_OK(QueryLoadedVariable(api, target, &executable,
                                     kAsanConfigGlobalName, &asan_config));
  EXPECT_NE(asan_config.address, 0u);
  EXPECT_EQ(asan_config.segment, HSA_VARIABLE_SEGMENT_GLOBAL);
  EXPECT_EQ(asan_config.byte_length, kAsanConfigByteLength);

  LoadedVariableInfo feedback_config = {};
  IREE_ASSERT_OK(QueryLoadedVariable(
      api, target, &executable, kFeedbackConfigGlobalName, &feedback_config));
  EXPECT_NE(feedback_config.address, 0u);
  EXPECT_EQ(feedback_config.segment, HSA_VARIABLE_SEGMENT_GLOBAL);
  EXPECT_EQ(feedback_config.byte_length, kFeedbackConfigByteLength);
}

TEST(AmdgpuHsacoHsaTest, LoadsWorkitemIndexedStoreKernel) {
  LoadLowKernelForCurrentTargetOrSkip(EmitWorkitemStoreKernelForAmdgpu, 8);
}

TEST(AmdgpuHsacoHsaTest, LoadsB128CopyKernel) {
  LoadLowKernelForCurrentTargetOrSkip(EmitB128CopyKernelForAmdgpu, 16);
}

TEST(AmdgpuHsacoHsaTest, LoadsWritableRuntimeDataSymbols) {
  LoadRuntimeGlobalsForCurrentTargetOrSkip();
}

}  // namespace
}  // namespace loom
