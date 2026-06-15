// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif  // defined(_WIN32)

#ifndef __has_feature
#define __has_feature(x) 0
#endif  // __has_feature

#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
#define LOOMC_EXAMPLE_SANITIZER_ADDRESS 1
#endif  // __SANITIZE_ADDRESS__ || __has_feature(address_sanitizer)

#if defined(LOOMC_EXAMPLE_SANITIZER_ADDRESS) && defined(__has_include)
#if __has_include(<sanitizer/lsan_interface.h>)
#include <sanitizer/lsan_interface.h>
#define LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH() __lsan_disable()
#define LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP() __lsan_enable()
#endif  // __has_include(<sanitizer/lsan_interface.h>)
#endif  // LOOMC_EXAMPLE_SANITIZER_ADDRESS && __has_include

#if !defined(LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH)
#define LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH()
#define LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP()
#endif  // !LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH

#include "hsa/amd_hsa_queue.h"
#include "hsa/amd_hsa_signal.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ven_amd_loader.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

static const char kSourceText[] =
    "kernel.def export(\"targetless_store_i32\") @targetless_store_i32() {\n"
    "  %unit = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%unit, %unit, %unit) "
    "workgroup_size(%unit, %unit, %unit) : index\n"
    "} launch(%output: buffer) {\n"
    "  %zero_offset = index.constant 0 : offset\n"
    "  %zero_index = index.constant 0 : index\n"
    "  %value = scalar.constant 42 : i32\n"
    "  %global = buffer.assume.memory_space<global> %output : buffer\n"
    "  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32, "
    "#dense>\n"
    "  view.store %value, %view[%zero_index] : i32, view<1xi32, #dense>\n"
    "  kernel.return\n"
    "}\n";

static const char kDefaultModuleName[] = "targetless_store_i32";
static const char kDefaultKernelSymbolName[] = "targetless_store_i32";

#if defined(_WIN32)
typedef HMODULE hsa_library_t;
#else
typedef void* hsa_library_t;
#endif  // defined(_WIN32)

typedef hsa_status_t(HSA_API* hsa_init_fn_t)(void);
typedef hsa_status_t(HSA_API* hsa_shut_down_fn_t)(void);
typedef hsa_status_t(HSA_API* hsa_status_string_fn_t)(
    hsa_status_t status, const char** status_string);
typedef hsa_status_t(HSA_API* hsa_iterate_agents_fn_t)(
    hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
typedef hsa_status_t(HSA_API* hsa_agent_get_info_fn_t)(
    hsa_agent_t agent, hsa_agent_info_t attribute, void* value);
typedef hsa_status_t(HSA_API* hsa_agent_iterate_isas_fn_t)(
    hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t isa, void* data),
    void* data);
typedef hsa_status_t(HSA_API* hsa_isa_get_info_alt_fn_t)(
    hsa_isa_t isa, hsa_isa_info_t attribute, void* value);
typedef hsa_status_t(HSA_API* hsa_code_object_reader_create_from_memory_fn_t)(
    const void* code_object, size_t size,
    hsa_code_object_reader_t* code_object_reader);
typedef hsa_status_t(HSA_API* hsa_code_object_reader_destroy_fn_t)(
    hsa_code_object_reader_t code_object_reader);
typedef hsa_status_t(HSA_API* hsa_executable_create_alt_fn_t)(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char* options, hsa_executable_t* executable);
typedef hsa_status_t(HSA_API* hsa_executable_destroy_fn_t)(
    hsa_executable_t executable);
typedef hsa_status_t(HSA_API* hsa_executable_load_agent_code_object_fn_t)(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader, const char* options,
    hsa_loaded_code_object_t* loaded_code_object);
typedef hsa_status_t(HSA_API* hsa_executable_freeze_fn_t)(
    hsa_executable_t executable, const char* options);
typedef hsa_status_t(HSA_API* hsa_executable_get_symbol_by_name_fn_t)(
    hsa_executable_t executable, const char* symbol_name,
    const hsa_agent_t* agent, hsa_executable_symbol_t* symbol);
typedef hsa_status_t(HSA_API* hsa_executable_symbol_get_info_fn_t)(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute, void* value);
typedef hsa_status_t(HSA_API* hsa_amd_agent_iterate_memory_pools_fn_t)(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data);
typedef hsa_status_t(HSA_API* hsa_amd_memory_pool_get_info_fn_t)(
    hsa_amd_memory_pool_t memory_pool, hsa_amd_memory_pool_info_t attribute,
    void* value);
typedef hsa_status_t(HSA_API* hsa_amd_memory_pool_allocate_fn_t)(
    hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void** ptr);
typedef hsa_status_t(HSA_API* hsa_amd_memory_pool_free_fn_t)(void* ptr);
typedef hsa_status_t(HSA_API* hsa_amd_agents_allow_access_fn_t)(
    uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags,
    const void* ptr);
typedef hsa_status_t(HSA_API* hsa_queue_create_fn_t)(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data, uint32_t private_segment_size, uint32_t group_segment_size,
    hsa_queue_t** queue);
typedef hsa_status_t(HSA_API* hsa_queue_destroy_fn_t)(hsa_queue_t* queue);
typedef uint64_t(HSA_API* hsa_queue_load_read_index_scacquire_fn_t)(
    const hsa_queue_t* queue);
typedef uint64_t(HSA_API* hsa_queue_load_write_index_relaxed_fn_t)(
    const hsa_queue_t* queue);
typedef void(HSA_API* hsa_queue_store_write_index_screlease_fn_t)(
    const hsa_queue_t* queue, uint64_t value);
typedef hsa_status_t(HSA_API* hsa_amd_signal_create_fn_t)(
    hsa_signal_value_t initial_value, uint32_t num_consumers,
    const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal);
typedef hsa_status_t(HSA_API* hsa_signal_destroy_fn_t)(hsa_signal_t signal);
typedef void(HSA_API* hsa_signal_store_screlease_fn_t)(
    hsa_signal_t signal, hsa_signal_value_t value);
typedef hsa_signal_value_t(HSA_API* hsa_signal_wait_scacquire_fn_t)(
    hsa_signal_t signal, hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value, uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);

typedef struct emit_amdgpu_hsa_api_t {
  // Dynamically loaded HSA runtime library.
  hsa_library_t library;

  // `hsa_init` entry point.
  hsa_init_fn_t hsa_init;

  // `hsa_shut_down` entry point.
  hsa_shut_down_fn_t hsa_shut_down;

  // `hsa_status_string` entry point.
  hsa_status_string_fn_t hsa_status_string;

  // `hsa_iterate_agents` entry point.
  hsa_iterate_agents_fn_t hsa_iterate_agents;

  // `hsa_agent_get_info` entry point.
  hsa_agent_get_info_fn_t hsa_agent_get_info;

  // `hsa_agent_iterate_isas` entry point.
  hsa_agent_iterate_isas_fn_t hsa_agent_iterate_isas;

  // `hsa_isa_get_info_alt` entry point.
  hsa_isa_get_info_alt_fn_t hsa_isa_get_info_alt;

  // `hsa_code_object_reader_create_from_memory` entry point.
  hsa_code_object_reader_create_from_memory_fn_t
      hsa_code_object_reader_create_from_memory;

  // `hsa_code_object_reader_destroy` entry point.
  hsa_code_object_reader_destroy_fn_t hsa_code_object_reader_destroy;

  // `hsa_executable_create_alt` entry point.
  hsa_executable_create_alt_fn_t hsa_executable_create_alt;

  // `hsa_executable_destroy` entry point.
  hsa_executable_destroy_fn_t hsa_executable_destroy;

  // `hsa_executable_load_agent_code_object` entry point.
  hsa_executable_load_agent_code_object_fn_t
      hsa_executable_load_agent_code_object;

  // `hsa_executable_freeze` entry point.
  hsa_executable_freeze_fn_t hsa_executable_freeze;

  // `hsa_executable_get_symbol_by_name` entry point.
  hsa_executable_get_symbol_by_name_fn_t hsa_executable_get_symbol_by_name;

  // `hsa_executable_symbol_get_info` entry point.
  hsa_executable_symbol_get_info_fn_t hsa_executable_symbol_get_info;

  // `hsa_amd_agent_iterate_memory_pools` entry point.
  hsa_amd_agent_iterate_memory_pools_fn_t hsa_amd_agent_iterate_memory_pools;

  // `hsa_amd_memory_pool_get_info` entry point.
  hsa_amd_memory_pool_get_info_fn_t hsa_amd_memory_pool_get_info;

  // `hsa_amd_memory_pool_allocate` entry point.
  hsa_amd_memory_pool_allocate_fn_t hsa_amd_memory_pool_allocate;

  // `hsa_amd_memory_pool_free` entry point.
  hsa_amd_memory_pool_free_fn_t hsa_amd_memory_pool_free;

  // `hsa_amd_agents_allow_access` entry point.
  hsa_amd_agents_allow_access_fn_t hsa_amd_agents_allow_access;

  // `hsa_queue_create` entry point.
  hsa_queue_create_fn_t hsa_queue_create;

  // `hsa_queue_destroy` entry point.
  hsa_queue_destroy_fn_t hsa_queue_destroy;

  // `hsa_queue_load_read_index_scacquire` entry point.
  hsa_queue_load_read_index_scacquire_fn_t hsa_queue_load_read_index_scacquire;

  // `hsa_queue_load_write_index_relaxed` entry point.
  hsa_queue_load_write_index_relaxed_fn_t hsa_queue_load_write_index_relaxed;

  // `hsa_queue_store_write_index_screlease` entry point.
  hsa_queue_store_write_index_screlease_fn_t
      hsa_queue_store_write_index_screlease;

  // `hsa_amd_signal_create` entry point.
  hsa_amd_signal_create_fn_t hsa_amd_signal_create;

  // `hsa_signal_destroy` entry point.
  hsa_signal_destroy_fn_t hsa_signal_destroy;

  // `hsa_signal_store_screlease` entry point.
  hsa_signal_store_screlease_fn_t hsa_signal_store_screlease;

  // `hsa_signal_wait_scacquire` entry point.
  hsa_signal_wait_scacquire_fn_t hsa_signal_wait_scacquire;
} emit_amdgpu_hsa_api_t;

typedef struct emit_amdgpu_hsa_kernel_info_t {
  // Kernel object handle written into the AQL dispatch packet.
  uint64_t kernel_object;

  // Kernel kernarg segment byte length reported by the HSA loader.
  uint32_t kernarg_segment_size;

  // Kernel kernarg segment byte alignment reported by the HSA loader.
  uint32_t kernarg_segment_alignment;

  // Group segment size reported by the HSA loader.
  uint32_t group_segment_size;

  // Private segment size reported by the HSA loader.
  uint32_t private_segment_size;
} emit_amdgpu_hsa_kernel_info_t;

typedef struct emit_amdgpu_hsa_state_t {
  // True when the host has no usable HSA GPU target for this example.
  bool skipped;

  // Human-readable skip reason.
  char skip_message[2048];

  // Optional `.loom` or `.loombc` path used instead of the embedded source.
  const char* source_path;

  // Module name assigned during compilation.
  const char* module_name;

  // HSA kernel symbol expected in the emitted HSACO.
  const char* kernel_symbol_name;

  // Dynamically loaded raw HSA API table.
  emit_amdgpu_hsa_api_t api;

  // True once `hsa_init` has succeeded and must be balanced.
  bool hsa_initialized;

  // Selected HSA CPU agent used for host-visible allocations.
  hsa_agent_t cpu_agent;

  // HSA-reported CPU agent name.
  char cpu_agent_name[64];

  // Selected HSA GPU agent used for executable load and queue dispatch.
  hsa_agent_t gpu_agent;

  // HSA-reported GPU agent name.
  char gpu_agent_name[64];

  // HSA-reported ISA target id for the selected GPU agent.
  char isa_name[1024];

  // Loom AMDGPU processor derived from `isa_name`.
  loomc_string_view_t processor;

  // Queue used for one raw AQL dispatch.
  hsa_queue_t* queue;

  // HSA executable loaded from the Loom-produced HSACO bytes.
  hsa_executable_t executable;

  // Completion signal attached to the dispatch packet.
  hsa_signal_t completion_signal;

  // True once `completion_signal` has been created.
  bool completion_signal_created;

  // Host-visible GPU-accessible output allocation.
  void* output_data;

  // Kernarg allocation containing the output buffer pointer.
  void* kernarg_data;

  // Kernel metadata queried from the loaded executable.
  emit_amdgpu_hsa_kernel_info_t kernel_info;

  // AMDGPU target package linked into this embedding binary.
  loomc_target_environment_t* target_environment;

  // Shared API context with the AMDGPU target dialect registered.
  loomc_context_t* context;

  // Per-worker scratch storage used by deserialize, compile, and emit.
  loomc_workspace_t* workspace;

  // Immutable source containing the Loom kernel module.
  loomc_source_t* source;

  // Mutable module compiled and emitted by this invocation.
  loomc_module_t* module;

  // Live HSA-derived AMDGPU processor target profile.
  loomc_target_profile_t* target_profile;

  // Invocation-ready target selection derived from the profile.
  loomc_target_selection_t* target_selection;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Prepared target pipeline shared across invocations.
  loomc_pass_program_t* pass_program;

  // Last operation result, reset between phases.
  loomc_result_t* result;
} emit_amdgpu_hsa_state_t;

static void print_status(loomc_status_t status) {
  char buffer[1024] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
}

static void print_result_diagnostics(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == NULL) {
      continue;
    }
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
}

static loomc_status_t emit_amdgpu_hsa_make_status_format(
    loomc_status_code_t code, const char* file, uint32_t line,
    const char* format, ...) {
  char buffer[2048] = {0};
  va_list varargs;
  va_start(varargs, format);
  vsnprintf(buffer, sizeof(buffer), format, varargs);
  va_end(varargs);
  return loomc_status_allocate(code, file, line,
                               loomc_make_cstring_view(buffer));
}

#define EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(code, format, ...)              \
  emit_amdgpu_hsa_make_status_format((code), __FILE__, __LINE__, (format), \
                                     __VA_ARGS__)

static bool string_ends_with(const char* value, const char* suffix) {
  const size_t value_length = strlen(value);
  const size_t suffix_length = strlen(suffix);
  return value_length >= suffix_length &&
         memcmp(value + value_length - suffix_length, suffix, suffix_length) ==
             0;
}

static loomc_status_t source_format_from_path(
    const char* path, loomc_source_format_t* out_format) {
  if (string_ends_with(path, ".loombc")) {
    *out_format = LOOMC_SOURCE_FORMAT_BYTECODE;
    return loomc_ok_status();
  }
  if (string_ends_with(path, ".loom")) {
    *out_format = LOOMC_SOURCE_FORMAT_TEXT;
    return loomc_ok_status();
  }
  return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
      LOOMC_STATUS_INVALID_ARGUMENT,
      "source path must end in .loom or .loombc: %s", path);
}

static void emit_amdgpu_hsa_state_skip(emit_amdgpu_hsa_state_t* state,
                                       const char* message) {
  state->skipped = true;
  snprintf(state->skip_message, sizeof(state->skip_message), "%s", message);
}

static void emit_amdgpu_hsa_state_skip_from_loomc_status(
    emit_amdgpu_hsa_state_t* state, const char* prefix, loomc_status_t status) {
  char detail[1024] = {0};
  loomc_host_size_t detail_length = 0;
  loomc_status_format(status, sizeof(detail), detail, &detail_length);
  loomc_status_free(status);
  snprintf(state->skip_message, sizeof(state->skip_message), "%s: %s", prefix,
           detail);
  state->skipped = true;
}

static loomc_status_code_t hsa_status_code(hsa_status_t status) {
  switch (status) {
    case HSA_STATUS_SUCCESS:
      return LOOMC_STATUS_OK;
    case HSA_STATUS_INFO_BREAK:
      return LOOMC_STATUS_CANCELLED;
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
    case HSA_STATUS_ERROR_INVALID_AGENT:
    case HSA_STATUS_ERROR_INVALID_REGION:
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
    case HSA_STATUS_ERROR_INVALID_QUEUE:
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
    case HSA_STATUS_ERROR_INVALID_FILE:
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
    case HSA_STATUS_ERROR_INVALID_CACHE:
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return LOOMC_STATUS_INVALID_ARGUMENT;
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return LOOMC_STATUS_RESOURCE_EXHAUSTED;
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return LOOMC_STATUS_FAILED_PRECONDITION;
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
    case HSA_STATUS_ERROR_INVALID_ISA:
      return LOOMC_STATUS_UNIMPLEMENTED;
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return LOOMC_STATUS_OUT_OF_RANGE;
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return LOOMC_STATUS_NOT_FOUND;
    case HSA_STATUS_ERROR_EXCEPTION:
    case HSA_STATUS_ERROR_FATAL:
      return LOOMC_STATUS_DATA_LOSS;
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return LOOMC_STATUS_INTERNAL;
    default:
      return LOOMC_STATUS_UNKNOWN;
  }
}

static const char* hsa_fallback_status_name(hsa_status_t status) {
  switch (status) {
    case HSA_STATUS_SUCCESS:
      return "HSA_STATUS_SUCCESS";
    case HSA_STATUS_INFO_BREAK:
      return "HSA_STATUS_INFO_BREAK";
    case HSA_STATUS_ERROR:
      return "HSA_STATUS_ERROR";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
      return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case HSA_STATUS_ERROR_INVALID_AGENT:
      return "HSA_STATUS_ERROR_INVALID_AGENT";
    case HSA_STATUS_ERROR_INVALID_REGION:
      return "HSA_STATUS_ERROR_INVALID_REGION";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case HSA_STATUS_ERROR_INVALID_QUEUE:
      return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
      return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return "HSA_STATUS_ERROR_INVALID_INDEX";
    case HSA_STATUS_ERROR_INVALID_ISA:
      return "HSA_STATUS_ERROR_INVALID_ISA";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
      return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
      return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    case HSA_STATUS_ERROR_EXCEPTION:
      return "HSA_STATUS_ERROR_EXCEPTION";
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_FILE:
      return "HSA_STATUS_ERROR_INVALID_FILE";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER";
    case HSA_STATUS_ERROR_INVALID_CACHE:
      return "HSA_STATUS_ERROR_INVALID_CACHE";
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
      return "HSA_STATUS_ERROR_INVALID_WAVEFRONT";
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP";
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE";
    case HSA_STATUS_ERROR_FATAL:
      return "HSA_STATUS_ERROR_FATAL";
    default:
      return "HSA_STATUS_UNKNOWN";
  }
}

static const char* hsa_status_text(const emit_amdgpu_hsa_api_t* api,
                                   hsa_status_t status) {
  const char* status_string = NULL;
  if (api != NULL && api->hsa_status_string != NULL) {
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
    hsa_status_t string_status = api->hsa_status_string(status, &status_string);
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
    if (string_status == HSA_STATUS_SUCCESS && status_string != NULL) {
      return status_string;
    }
  }
  return hsa_fallback_status_name(status);
}

static loomc_status_t hsa_status_to_loomc_status(
    const emit_amdgpu_hsa_api_t* api, hsa_status_t status,
    const char* operation) {
  if (status == HSA_STATUS_SUCCESS) {
    return loomc_ok_status();
  }
  return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(hsa_status_code(status),
                                            "%s failed: %s", operation,
                                            hsa_status_text(api, status));
}

static void print_hsa_cleanup_status(const emit_amdgpu_hsa_api_t* api,
                                     hsa_status_t status,
                                     const char* operation) {
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "%s during cleanup failed: %s\n", operation,
            hsa_status_text(api, status));
  }
}

static void* hsa_library_lookup(hsa_library_t library, const char* symbol) {
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
#if defined(_WIN32)
  void* value = (void*)GetProcAddress(library, symbol);
#else
  void* value = dlsym(library, symbol);
#endif  // defined(_WIN32)
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  return value;
}

static void hsa_library_close(hsa_library_t library) {
  if (library == 0) {
    return;
  }
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
#if defined(_WIN32)
  FreeLibrary(library);
#else
  dlclose(library);
#endif  // defined(_WIN32)
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
}

static bool hsa_library_open_exact(const char* candidate,
                                   hsa_library_t* out_library, char* last_error,
                                   size_t last_error_capacity) {
  *out_library = 0;
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
#if defined(_WIN32)
  hsa_library_t library = LoadLibraryA(candidate);
  if (library == 0) {
    snprintf(last_error, last_error_capacity, "%s: Windows error %" PRIu32,
             candidate, (uint32_t)GetLastError());
  }
#else
  dlerror();
  hsa_library_t library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
  if (library == 0) {
    const char* error = dlerror();
    snprintf(last_error, last_error_capacity, "%s: %s", candidate,
             error != NULL ? error : "unknown dynamic loader error");
  }
#endif  // defined(_WIN32)
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  if (library == 0) {
    return false;
  }
  *out_library = library;
  return true;
}

static bool hsa_library_open(emit_amdgpu_hsa_state_t* state) {
  static const char* const kLibraryNames[] = {
#if defined(_WIN32)
      "hsa-runtime64.dll",
#else
      "libhsa-runtime64.so.1",
      "libhsa-runtime64.so",
#endif  // defined(_WIN32)
  };
  char last_error[1024] = {0};
  const char* env_path = getenv("LOOMC_HSA_RUNTIME_PATH");
  if (env_path != NULL && env_path[0] != '\0') {
    if (hsa_library_open_exact(env_path, &state->api.library, last_error,
                               sizeof(last_error))) {
      return true;
    }
    for (size_t i = 0; i < sizeof(kLibraryNames) / sizeof(kLibraryNames[0]);
         ++i) {
      char candidate[4096] = {0};
      const char separator =
#if defined(_WIN32)
          '\\';
#else
          '/';
#endif  // defined(_WIN32)
      snprintf(candidate, sizeof(candidate), "%s%c%s", env_path, separator,
               kLibraryNames[i]);
      if (hsa_library_open_exact(candidate, &state->api.library, last_error,
                                 sizeof(last_error))) {
        return true;
      }
    }
  }
  for (size_t i = 0; i < sizeof(kLibraryNames) / sizeof(kLibraryNames[0]);
       ++i) {
    if (hsa_library_open_exact(kLibraryNames[i], &state->api.library,
                               last_error, sizeof(last_error))) {
      return true;
    }
  }
  snprintf(state->skip_message, sizeof(state->skip_message),
           "HSA runtime library was not found; set LOOMC_HSA_RUNTIME_PATH to "
           "a library or containing directory. Last loader error: %s",
           last_error[0] != '\0' ? last_error : "no candidates were tried");
  state->skipped = true;
  return false;
}

static loomc_status_t hsa_api_load_symbol(void* symbol, const char* name) {
  if (symbol != NULL) {
    return loomc_ok_status();
  }
  return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
      LOOMC_STATUS_NOT_FOUND, "HSA runtime does not export %s", name);
}

#define EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, symbol)                               \
  do {                                                                         \
    (api)->symbol =                                                            \
        (symbol##_fn_t)hsa_library_lookup((api)->library, #symbol);            \
    LOOMC_RETURN_IF_ERROR(hsa_api_load_symbol((void*)(api)->symbol, #symbol)); \
  } while (0)

#define EMIT_AMDGPU_HSA_CALL_STATUS(result, api, symbol, ...) \
  do {                                                        \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();                  \
    (result) = (api)->symbol(__VA_ARGS__);                    \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();                   \
  } while (0)

#define EMIT_AMDGPU_HSA_CALL_VALUE(result, api, symbol, ...) \
  do {                                                       \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();                 \
    (result) = (api)->symbol(__VA_ARGS__);                   \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();                  \
  } while (0)

#define EMIT_AMDGPU_HSA_CALL_VOID(api, symbol, ...) \
  do {                                              \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();        \
    (api)->symbol(__VA_ARGS__);                     \
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();         \
  } while (0)

static loomc_status_t load_hsa_api(emit_amdgpu_hsa_state_t* state) {
  if (!hsa_library_open(state)) {
    return loomc_ok_status();
  }
  emit_amdgpu_hsa_api_t* api = &state->api;
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_init);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_shut_down);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_status_string);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_iterate_agents);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_agent_get_info);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_agent_iterate_isas);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_isa_get_info_alt);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_code_object_reader_create_from_memory);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_code_object_reader_destroy);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_create_alt);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_destroy);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_load_agent_code_object);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_freeze);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_get_symbol_by_name);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_executable_symbol_get_info);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_agent_iterate_memory_pools);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_memory_pool_get_info);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_memory_pool_allocate);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_memory_pool_free);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_agents_allow_access);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_queue_create);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_queue_destroy);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_queue_load_read_index_scacquire);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_queue_load_write_index_relaxed);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_queue_store_write_index_screlease);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_amd_signal_create);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_signal_destroy);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_signal_store_screlease);
  EMIT_AMDGPU_HSA_LOAD_SYMBOL(api, hsa_signal_wait_scacquire);
  return loomc_ok_status();
}

static void emit_amdgpu_hsa_state_initialize(emit_amdgpu_hsa_state_t* state,
                                             const char* source_path,
                                             const char* kernel_symbol_name,
                                             const char* module_name) {
  memset(state, 0, sizeof(*state));
  state->source_path = source_path;
  state->kernel_symbol_name = kernel_symbol_name;
  state->module_name = module_name;
}

static void emit_amdgpu_hsa_state_deinitialize(emit_amdgpu_hsa_state_t* state) {
  loomc_result_release(state->result);
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_target_selection_release(state->target_selection);
  loomc_target_profile_release(state->target_profile);
  loomc_module_release(state->module);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);

  emit_amdgpu_hsa_api_t* api = &state->api;
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  if (state->queue != NULL) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_queue_destroy,
                                state->queue);
    print_hsa_cleanup_status(api, hsa_status, "hsa_queue_destroy");
  }
  if (state->completion_signal_created) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_signal_destroy,
                                state->completion_signal);
    print_hsa_cleanup_status(api, hsa_status, "hsa_signal_destroy");
  }
  if (state->executable.handle != 0) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_executable_destroy,
                                state->executable);
    print_hsa_cleanup_status(api, hsa_status, "hsa_executable_destroy");
  }
  if (state->kernarg_data != NULL) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_amd_memory_pool_free,
                                state->kernarg_data);
    print_hsa_cleanup_status(api, hsa_status, "hsa_amd_memory_pool_free");
  }
  if (state->output_data != NULL) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_amd_memory_pool_free,
                                state->output_data);
    print_hsa_cleanup_status(api, hsa_status, "hsa_amd_memory_pool_free");
  }
  if (state->hsa_initialized) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_shut_down);
    print_hsa_cleanup_status(api, hsa_status, "hsa_shut_down");
  }
  hsa_library_close(state->api.library);
}

static void emit_amdgpu_hsa_state_reset_result(emit_amdgpu_hsa_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static loomc_status_t require_successful_result(const loomc_result_t* result,
                                                const char* failure_message) {
  if (loomc_result_succeeded(result)) {
    return loomc_ok_status();
  }
  print_result_diagnostics(result);
  return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION, failure_message);
}

static const loomc_artifact_t* find_result_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact == NULL) {
      continue;
    }
    if (artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return NULL;
}

typedef struct emit_amdgpu_hsa_agent_search_t {
  // HSA API table used by the agent callback.
  const emit_amdgpu_hsa_api_t* api;

  // First CPU agent found.
  hsa_agent_t cpu_agent;

  // HSA-reported CPU agent name.
  char cpu_agent_name[64];

  // True once a CPU agent has been selected.
  bool found_cpu;

  // First GPU agent found.
  hsa_agent_t gpu_agent;

  // HSA-reported GPU agent name.
  char gpu_agent_name[64];

  // True once a GPU agent has been selected.
  bool found_gpu;
} emit_amdgpu_hsa_agent_search_t;

static hsa_status_t emit_amdgpu_hsa_record_agent_name(
    const emit_amdgpu_hsa_api_t* api, hsa_agent_t agent, char* buffer,
    size_t buffer_capacity) {
  memset(buffer, 0, buffer_capacity);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(status, api, hsa_agent_get_info, agent,
                              HSA_AGENT_INFO_NAME, buffer);
  return status;
}

static hsa_status_t emit_amdgpu_hsa_find_agents(hsa_agent_t agent,
                                                void* user_data) {
  emit_amdgpu_hsa_agent_search_t* search =
      (emit_amdgpu_hsa_agent_search_t*)user_data;
  hsa_device_type_t device_type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(status, search->api, hsa_agent_get_info, agent,
                              HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (device_type == HSA_DEVICE_TYPE_CPU && !search->found_cpu) {
    status = emit_amdgpu_hsa_record_agent_name(search->api, agent,
                                               search->cpu_agent_name,
                                               sizeof(search->cpu_agent_name));
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    search->cpu_agent = agent;
    search->found_cpu = true;
  } else if (device_type == HSA_DEVICE_TYPE_GPU && !search->found_gpu) {
    status = emit_amdgpu_hsa_record_agent_name(search->api, agent,
                                               search->gpu_agent_name,
                                               sizeof(search->gpu_agent_name));
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    search->gpu_agent = agent;
    search->found_gpu = true;
  }
  return HSA_STATUS_SUCCESS;
}

typedef struct emit_amdgpu_hsa_isa_search_t {
  // HSA API table used by the ISA callback.
  const emit_amdgpu_hsa_api_t* api;

  // First ISA name found.
  char isa_name[1024];

  // True once an ISA name has been selected.
  bool found;
} emit_amdgpu_hsa_isa_search_t;

static hsa_status_t emit_amdgpu_hsa_find_first_isa(hsa_isa_t isa,
                                                   void* user_data) {
  emit_amdgpu_hsa_isa_search_t* search =
      (emit_amdgpu_hsa_isa_search_t*)user_data;
  uint32_t name_length = 0;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(status, search->api, hsa_isa_get_info_alt, isa,
                              HSA_ISA_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (name_length == 0 || name_length >= sizeof(search->isa_name)) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  memset(search->isa_name, 0, sizeof(search->isa_name));
  EMIT_AMDGPU_HSA_CALL_STATUS(status, search->api, hsa_isa_get_info_alt, isa,
                              HSA_ISA_INFO_NAME, search->isa_name);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  search->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static loomc_status_t discover_hsa_target(emit_amdgpu_hsa_state_t* state) {
  LOOMC_RETURN_IF_ERROR(load_hsa_api(state));
  if (state->skipped) {
    return loomc_ok_status();
  }

  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api, hsa_init);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    snprintf(state->skip_message, sizeof(state->skip_message),
             "hsa_init failed: %s", hsa_status_text(&state->api, hsa_status));
    state->skipped = true;
    return loomc_ok_status();
  }
  state->hsa_initialized = true;

  emit_amdgpu_hsa_agent_search_t agent_search = {
      .api = &state->api,
  };
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api, hsa_iterate_agents,
                              emit_amdgpu_hsa_find_agents, &agent_search);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(&state->api, hsa_status,
                                                   "hsa_iterate_agents"));
  if (!agent_search.found_gpu) {
    emit_amdgpu_hsa_state_skip(state, "no HSA GPU agent found");
    return loomc_ok_status();
  }
  if (!agent_search.found_cpu) {
    emit_amdgpu_hsa_state_skip(state, "no HSA CPU agent found");
    return loomc_ok_status();
  }
  state->cpu_agent = agent_search.cpu_agent;
  snprintf(state->cpu_agent_name, sizeof(state->cpu_agent_name), "%s",
           agent_search.cpu_agent_name);
  state->gpu_agent = agent_search.gpu_agent;
  snprintf(state->gpu_agent_name, sizeof(state->gpu_agent_name), "%s",
           agent_search.gpu_agent_name);

  emit_amdgpu_hsa_isa_search_t isa_search = {
      .api = &state->api,
  };
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api, hsa_agent_iterate_isas,
                              state->gpu_agent, emit_amdgpu_hsa_find_first_isa,
                              &isa_search);
  if (hsa_status != HSA_STATUS_SUCCESS && hsa_status != HSA_STATUS_INFO_BREAK) {
    return hsa_status_to_loomc_status(&state->api, hsa_status,
                                      "hsa_agent_iterate_isas");
  }
  if (!isa_search.found) {
    emit_amdgpu_hsa_state_skip(state,
                               "selected HSA GPU agent reports no ISA names");
    return loomc_ok_status();
  }
  snprintf(state->isa_name, sizeof(state->isa_name), "%s", isa_search.isa_name);

  loomc_status_t parse_status = loomc_amdgpu_processor_from_hsa_isa_name(
      loomc_make_cstring_view(state->isa_name), &state->processor);
  if (!loomc_status_is_ok(parse_status)) {
    emit_amdgpu_hsa_state_skip_from_loomc_status(
        state, "Loom does not support the HSA-reported AMDGPU ISA",
        parse_status);
  }
  return loomc_ok_status();
}

static loomc_status_t create_target_environment_and_context(
    emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &state->target_environment);
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = state->target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, loomc_allocator_system(),
                                  &state->context);
  }
  return status;
}

static loomc_status_t create_workspace_and_source(
    emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status =
      loomc_workspace_create(NULL, loomc_allocator_system(), &state->workspace);
  if (!loomc_status_is_ok(status)) {
    return status;
  }

  if (state->source_path != NULL) {
    loomc_source_format_t format = LOOMC_SOURCE_FORMAT_UNKNOWN;
    LOOMC_RETURN_IF_ERROR(source_format_from_path(state->source_path, &format));
    loomc_source_load_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = format,
    };
    return loomc_source_create_from_path(
        loomc_make_cstring_view(state->source_path), &source_options,
        loomc_allocator_system(), &state->source);
  }

  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("targetless_store_i32.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  return loomc_source_create(&source_options, loomc_allocator_system(),
                             &state->source);
}

static loomc_status_t create_target_profile_and_selection(
    emit_amdgpu_hsa_state_t* state) {
  loomc_amdgpu_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("hsa-current-amdgpu"),
      .processor = state->processor,
  };
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      state->target_environment, &profile_options, loomc_allocator_system(),
      &state->target_profile);
  if (!loomc_status_is_ok(status)) {
    emit_amdgpu_hsa_state_skip_from_loomc_status(
        state, "Loom cannot emit HSACO for the HSA-selected AMDGPU processor",
        status);
    return loomc_ok_status();
  }
  return loomc_target_selection_create_from_profile(state->target_profile,
                                                    loomc_allocator_system(),
                                                    &state->target_selection);
}

static loomc_status_t create_compiler_and_target_pipeline(
    emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = loomc_compiler_create(
      state->context, NULL, loomc_allocator_system(), &state->compiler);
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .next = &target_options,
      .identifier = loomc_make_cstring_view("hsa-amdgpu-prepared-low"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_from_target_pipeline(
        state->context, &pipeline_options, loomc_allocator_system(),
        &state->pass_program, &state->result);
  }
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "target pipeline preparation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_amdgpu_hsa_state_reset_result(state);
  }
  return status;
}

static loomc_status_t create_compiler_resources(
    emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = create_target_environment_and_context(state);
  if (loomc_status_is_ok(status)) {
    status = create_workspace_and_source(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_target_profile_and_selection(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = create_compiler_and_target_pipeline(state);
  }
  return status;
}

static loomc_status_t deserialize_source(emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_amdgpu_hsa_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_module_to_prepared_low(
    emit_amdgpu_hsa_state_t* state) {
  printf("hsa cpu_agent=%s gpu_agent=%s isa=%s processor=%.*s\n",
         state->cpu_agent_name, state->gpu_agent_name, state->isa_name,
         (int)state->processor.size, state->processor.data);
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view(state->module_name),
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &compile_options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_amdgpu_hsa_state_reset_result(state);
  }
  return status;
}

static loomc_status_t emit_amdgpu_artifact(emit_amdgpu_hsa_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
      .next = &target_options,
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &amdgpu_options,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("targetless_store_i32.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_status_t status = loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "AMDGPU emission failed");
  }
  return status;
}

static loomc_status_t compile_and_emit_hsaco(emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = create_compiler_resources(state);
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = deserialize_source(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = compile_module_to_prepared_low(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = emit_amdgpu_artifact(state);
  }
  return status;
}

static loomc_status_t load_hsaco_executable(emit_amdgpu_hsa_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "AMDGPU HSACO artifact was not produced");
  }

  emit_amdgpu_hsa_api_t* api = &state->api;
  hsa_code_object_reader_t reader = {0};
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_code_object_reader_create_from_memory,
      artifact->contents.data, artifact->contents.data_length, &reader);
  loomc_status_t status = hsa_status_to_loomc_status(
      api, hsa_status, "hsa_code_object_reader_create_from_memory");
  if (loomc_status_is_ok(status)) {
    EMIT_AMDGPU_HSA_CALL_STATUS(
        hsa_status, api, hsa_executable_create_alt, HSA_PROFILE_FULL,
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &state->executable);
    status = hsa_status_to_loomc_status(api, hsa_status,
                                        "hsa_executable_create_alt");
  }
  if (loomc_status_is_ok(status)) {
    EMIT_AMDGPU_HSA_CALL_STATUS(
        hsa_status, api, hsa_executable_load_agent_code_object,
        state->executable, state->gpu_agent, reader, NULL, NULL);
    status = hsa_status_to_loomc_status(
        api, hsa_status, "hsa_executable_load_agent_code_object");
  }
  if (reader.handle != 0) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_code_object_reader_destroy,
                                reader);
    if (loomc_status_is_ok(status)) {
      status = hsa_status_to_loomc_status(api, hsa_status,
                                          "hsa_code_object_reader_destroy");
    } else {
      print_hsa_cleanup_status(api, hsa_status,
                               "hsa_code_object_reader_destroy");
    }
  }
  if (loomc_status_is_ok(status)) {
    EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_executable_freeze,
                                state->executable, NULL);
    status =
        hsa_status_to_loomc_status(api, hsa_status, "hsa_executable_freeze");
  }
  return status;
}

static loomc_status_t query_kernel_symbol(emit_amdgpu_hsa_state_t* state) {
  emit_amdgpu_hsa_api_t* api = &state->api;
  hsa_executable_symbol_t symbol = {0};
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  char code_descriptor_symbol_name[1024] = {0};
  int print_result =
      snprintf(code_descriptor_symbol_name, sizeof(code_descriptor_symbol_name),
               "%s.kd", state->kernel_symbol_name);
  if (print_result < 0 ||
      (size_t)print_result >= sizeof(code_descriptor_symbol_name)) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "kernel symbol name is too long");
  }
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_executable_get_symbol_by_name, state->executable,
      code_descriptor_symbol_name, &state->gpu_agent, &symbol);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    EMIT_AMDGPU_HSA_CALL_STATUS(
        hsa_status, api, hsa_executable_get_symbol_by_name, state->executable,
        state->kernel_symbol_name, &state->gpu_agent, &symbol);
    LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
        api, hsa_status, "hsa_executable_get_symbol_by_name"));
  }

  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_executable_symbol_get_info,
                              symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                              &state->kernel_info.kernel_object);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_executable_symbol_get_info(KERNEL_OBJECT)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_executable_symbol_get_info, symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &state->kernel_info.kernarg_segment_size);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_executable_symbol_get_info(KERNARG_SEGMENT_SIZE)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_executable_symbol_get_info, symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &state->kernel_info.kernarg_segment_alignment);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status,
      "hsa_executable_symbol_get_info(KERNARG_SEGMENT_ALIGNMENT)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_executable_symbol_get_info, symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &state->kernel_info.group_segment_size);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_executable_symbol_get_info(GROUP_SEGMENT_SIZE)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_executable_symbol_get_info, symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &state->kernel_info.private_segment_size);
  return hsa_status_to_loomc_status(
      api, hsa_status, "hsa_executable_symbol_get_info(PRIVATE_SEGMENT_SIZE)");
}

typedef struct emit_amdgpu_hsa_pool_search_t {
  // HSA API table used by the pool callback.
  const emit_amdgpu_hsa_api_t* api;

  // Required global flag bits.
  hsa_amd_memory_pool_global_flag_t required_global_flags;

  // True when the memory pool must be accessible by all agents.
  bool require_accessible_by_all;

  // First matching memory pool.
  hsa_amd_memory_pool_t pool;

  // True once `pool` has been populated.
  bool found;
} emit_amdgpu_hsa_pool_search_t;

static hsa_status_t emit_amdgpu_hsa_find_memory_pool(hsa_amd_memory_pool_t pool,
                                                     void* user_data) {
  emit_amdgpu_hsa_pool_search_t* search =
      (emit_amdgpu_hsa_pool_search_t*)user_data;
  hsa_region_segment_t segment = 0;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(status, search->api, hsa_amd_memory_pool_get_info,
                              pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }
  bool alloc_allowed = false;
  EMIT_AMDGPU_HSA_CALL_STATUS(
      status, search->api, hsa_amd_memory_pool_get_info, pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }
  if (search->require_accessible_by_all) {
    bool accessible_by_all = false;
    EMIT_AMDGPU_HSA_CALL_STATUS(
        status, search->api, hsa_amd_memory_pool_get_info, pool,
        HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &accessible_by_all);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    if (!accessible_by_all) {
      return HSA_STATUS_SUCCESS;
    }
  }
  hsa_amd_memory_pool_global_flag_t global_flags = 0;
  EMIT_AMDGPU_HSA_CALL_STATUS(status, search->api, hsa_amd_memory_pool_get_info,
                              pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                              &global_flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if ((global_flags & search->required_global_flags) !=
      search->required_global_flags) {
    return HSA_STATUS_SUCCESS;
  }
  search->pool = pool;
  search->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static loomc_status_t find_global_memory_pool(
    emit_amdgpu_hsa_state_t* state, hsa_agent_t pool_agent,
    hsa_amd_memory_pool_global_flag_t required_global_flags,
    bool require_accessible_by_all, const char* description,
    hsa_amd_memory_pool_t* out_pool) {
  emit_amdgpu_hsa_pool_search_t search = {
      .api = &state->api,
      .required_global_flags = required_global_flags,
      .require_accessible_by_all = require_accessible_by_all,
  };
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api,
                              hsa_amd_agent_iterate_memory_pools, pool_agent,
                              emit_amdgpu_hsa_find_memory_pool, &search);
  if (hsa_status != HSA_STATUS_SUCCESS && hsa_status != HSA_STATUS_INFO_BREAK) {
    return hsa_status_to_loomc_status(&state->api, hsa_status,
                                      "hsa_amd_agent_iterate_memory_pools");
  }
  if (!search.found) {
    return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
        LOOMC_STATUS_NOT_FOUND, "no HSA global memory pool found for %s",
        description);
  }
  *out_pool = search.pool;
  return loomc_ok_status();
}

static loomc_status_t allocate_hsa_memory(emit_amdgpu_hsa_state_t* state) {
  if (state->kernel_info.kernarg_segment_size < sizeof(uint64_t)) {
    return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "kernel kernarg segment is only %" PRIu32 " bytes; expected at least 8",
        state->kernel_info.kernarg_segment_size);
  }

  hsa_amd_memory_pool_t output_pool = {0};
  LOOMC_RETURN_IF_ERROR(find_global_memory_pool(
      state, state->cpu_agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
      true, "host-visible output", &output_pool));
  hsa_amd_memory_pool_t kernarg_pool = {0};
  LOOMC_RETURN_IF_ERROR(find_global_memory_pool(
      state, state->cpu_agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
      true, "kernarg", &kernarg_pool));

  emit_amdgpu_hsa_api_t* api = &state->api;
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_amd_memory_pool_allocate,
                              output_pool, sizeof(uint32_t), 0,
                              &state->output_data);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_amd_memory_pool_allocate(output)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_amd_agents_allow_access, 1,
                              &state->gpu_agent, NULL, state->output_data);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_amd_agents_allow_access(output)"));
  *(uint32_t*)state->output_data = 0;

  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, api, hsa_amd_memory_pool_allocate, kernarg_pool,
      state->kernel_info.kernarg_segment_size, 0, &state->kernarg_data);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_amd_memory_pool_allocate(kernarg)"));
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, api, hsa_amd_agents_allow_access, 1,
                              &state->gpu_agent, NULL, state->kernarg_data);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      api, hsa_status, "hsa_amd_agents_allow_access(kernarg)"));
  if (state->kernel_info.kernarg_segment_alignment != 0 &&
      ((uintptr_t)state->kernarg_data %
       state->kernel_info.kernarg_segment_alignment) != 0) {
    return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "HSA kernarg allocation is not aligned to %" PRIu32 " bytes",
        state->kernel_info.kernarg_segment_alignment);
  }
  memset(state->kernarg_data, 0, state->kernel_info.kernarg_segment_size);
  uint64_t output_address = (uint64_t)(uintptr_t)state->output_data;
  memcpy(state->kernarg_data, &output_address, sizeof(output_address));
  return loomc_ok_status();
}

static void hsa_queue_error_callback(hsa_status_t status, hsa_queue_t* source,
                                     void* user_data) {
  (void)source;
  const emit_amdgpu_hsa_api_t* api = (const emit_amdgpu_hsa_api_t*)user_data;
  fprintf(stderr, "HSA queue callback status=%s\n",
          hsa_status_text(api, status));
}

static loomc_status_t create_hsa_queue_and_signal(
    emit_amdgpu_hsa_state_t* state) {
  uint32_t queue_max_size = 0;
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api, hsa_agent_get_info,
                              state->gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                              &queue_max_size);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(
      &state->api, hsa_status, "hsa_agent_get_info(QUEUE_MAX_SIZE)"));
  if (queue_max_size < 64) {
    return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "HSA agent queue max size %" PRIu32 " is too small", queue_max_size);
  }
  EMIT_AMDGPU_HSA_CALL_STATUS(
      hsa_status, &state->api, hsa_queue_create, state->gpu_agent, 64,
      HSA_QUEUE_TYPE_SINGLE, hsa_queue_error_callback, &state->api,
      state->kernel_info.private_segment_size,
      state->kernel_info.group_segment_size, &state->queue);
  LOOMC_RETURN_IF_ERROR(
      hsa_status_to_loomc_status(&state->api, hsa_status, "hsa_queue_create"));
  EMIT_AMDGPU_HSA_CALL_STATUS(hsa_status, &state->api, hsa_amd_signal_create, 1,
                              1, &state->gpu_agent, 0,
                              &state->completion_signal);
  LOOMC_RETURN_IF_ERROR(hsa_status_to_loomc_status(&state->api, hsa_status,
                                                   "hsa_amd_signal_create"));
  state->completion_signal_created = true;
  return loomc_ok_status();
}

static uint16_t make_kernel_dispatch_header(void) {
  return (
      uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                (HSA_FENCE_SCOPE_SYSTEM
                 << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                (HSA_FENCE_SCOPE_SYSTEM
                 << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

static loomc_status_t submit_and_wait(emit_amdgpu_hsa_state_t* state) {
  emit_amdgpu_hsa_api_t* api = &state->api;
  uint64_t write_index = 0;
  EMIT_AMDGPU_HSA_CALL_VALUE(write_index, api,
                             hsa_queue_load_write_index_relaxed, state->queue);
  uint64_t read_index = 0;
  EMIT_AMDGPU_HSA_CALL_VALUE(read_index, api,
                             hsa_queue_load_read_index_scacquire, state->queue);
  while (write_index - read_index >= state->queue->size) {
    EMIT_AMDGPU_HSA_CALL_VALUE(
        read_index, api, hsa_queue_load_read_index_scacquire, state->queue);
  }

  hsa_kernel_dispatch_packet_t* packets =
      (hsa_kernel_dispatch_packet_t*)state->queue->base_address;
  hsa_kernel_dispatch_packet_t* packet =
      &packets[write_index & (state->queue->size - 1)];
  memset(packet, 0, sizeof(*packet));
  packet->setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet->workgroup_size_x = 1;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->grid_size_x = 1;
  packet->grid_size_y = 1;
  packet->grid_size_z = 1;
  packet->private_segment_size = state->kernel_info.private_segment_size;
  packet->group_segment_size = state->kernel_info.group_segment_size;
  packet->kernel_object = state->kernel_info.kernel_object;
  packet->kernarg_address = state->kernarg_data;
  packet->completion_signal = state->completion_signal;

  const uint16_t header = make_kernel_dispatch_header();
  __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
  EMIT_AMDGPU_HSA_CALL_VOID(api, hsa_queue_store_write_index_screlease,
                            state->queue, write_index + 1);
  EMIT_AMDGPU_HSA_CALL_VOID(api, hsa_signal_store_screlease,
                            state->queue->doorbell_signal, write_index);

  hsa_signal_value_t wait_value = 0;
  EMIT_AMDGPU_HSA_CALL_VALUE(wait_value, api, hsa_signal_wait_scacquire,
                             state->completion_signal, HSA_SIGNAL_CONDITION_LT,
                             1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (wait_value != 0) {
    return loomc_make_status(LOOMC_STATUS_DEADLINE_EXCEEDED,
                             "HSA dispatch did not complete");
  }

  const uint32_t observed = *(const uint32_t*)state->output_data;
  if (observed != 42u) {
    return EMIT_AMDGPU_HSA_MAKE_STATUS_FORMAT(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "HSA dispatch wrote %" PRIu32 ", expected 42", observed);
  }
  printf("launched %s via raw HSA: output=%" PRIu32 "\n",
         state->kernel_symbol_name, observed);
  return loomc_ok_status();
}

static loomc_status_t run_hsa_launch(emit_amdgpu_hsa_state_t* state) {
  loomc_status_t status = load_hsaco_executable(state);
  if (loomc_status_is_ok(status)) {
    status = query_kernel_symbol(state);
  }
  if (loomc_status_is_ok(status)) {
    printf("loaded kernel object=0x%016" PRIx64 " kernarg_size=%" PRIu32
           " group_size=%" PRIu32 " private_size=%" PRIu32 "\n",
           state->kernel_info.kernel_object,
           state->kernel_info.kernarg_segment_size,
           state->kernel_info.group_segment_size,
           state->kernel_info.private_segment_size);
  }
  if (loomc_status_is_ok(status)) {
    status = allocate_hsa_memory(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_hsa_queue_and_signal(state);
  }
  if (loomc_status_is_ok(status)) {
    status = submit_and_wait(state);
  }
  return status;
}

static loomc_status_t run_emit_amdgpu_hsa_example(
    const char* source_path, const char* kernel_symbol_name,
    const char* module_name) {
  emit_amdgpu_hsa_state_t state;
  emit_amdgpu_hsa_state_initialize(&state, source_path, kernel_symbol_name,
                                   module_name);

  loomc_status_t status = discover_hsa_target(&state);
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = compile_and_emit_hsaco(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = run_hsa_launch(&state);
  }
  if (state.skipped) {
    printf("skipped: %s\n", state.skip_message);
  }

  emit_amdgpu_hsa_state_deinitialize(&state);
  return status;
}

static void print_usage(FILE* file) {
  fprintf(file,
          "Usage: emit_amdgpu_hsa [source.loom|source.loombc "
          "[kernel-symbol [module-name]]]\n");
  fprintf(file,
          "  source       Optional Loom source or bytecode module. Omit it to "
          "use the embedded targetless_store_i32 kernel.\n");
  fprintf(file,
          "  kernel-symbol HSA executable symbol to launch. The example also "
          "tries the .kd descriptor spelling when needed.\n");
  fprintf(file,
          "  module-name  Module name used for Loom compilation when source is "
          "provided.\n");
  fprintf(
      file,
      "The example discovers an HSA GPU agent and derives the AMDGPU target "
      "from the live ISA before emitting HSACO bytes.\n");
}

int main(int argc, char** argv) {
  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(stdout);
    return 0;
  }
  if (argc > 4) {
    print_usage(stderr);
    return 1;
  }

  const char* source_path = argc > 1 ? argv[1] : NULL;
  const char* kernel_symbol_name =
      argc > 2 ? argv[2] : kDefaultKernelSymbolName;
  const char* module_name = argc > 3 ? argv[3] : kDefaultModuleName;

  loomc_status_t status =
      run_emit_amdgpu_hsa_example(source_path, kernel_symbol_name, module_name);
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
