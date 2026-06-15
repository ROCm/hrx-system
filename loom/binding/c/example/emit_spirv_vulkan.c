// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif  // !VK_NO_PROTOTYPES

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
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

#include "loomc/loomc.h"
#include "loomc/target/spirv.h"
#include "loomc/target/spirv/vulkan.h"

static const char kSourceText[] =
    "spirv.target<vulkan1_3> @target {abi = hal_kernel}\n"
    "\n"
    "kernel.def target(@target) @double_i32_at_byte_offset() {\n"
    "  %unit = index.constant 1 : index\n"
    "  kernel.launch.config workgroups(%unit, %unit, %unit) "
    "workgroup_size(%unit, %unit, %unit) : index\n"
    "} launch(%input: buffer, %output: buffer, %byte_offset: offset) {\n"
    "  %byte_offset_aligned = index.assume %byte_offset [mul(%byte_offset, "
    "4)] : offset\n"
    "  %input_aligned = buffer.assume.alignment %input {minimum_alignment = "
    "4} : buffer\n"
    "  %output_aligned = buffer.assume.alignment %output {minimum_alignment = "
    "4} : buffer\n"
    "  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : "
    "buffer -> view<1xi32, #dense>\n"
    "  %loaded = view.load %input_view[0] : view<1xi32, #dense> -> i32\n"
    "  %doubled = scalar.addi %loaded, %loaded : i32\n"
    "  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : "
    "buffer -> view<1xi32, #dense>\n"
    "  view.store %doubled, %output_view[0] : i32, view<1xi32, #dense>\n"
    "  kernel.return\n"
    "}\n";

#if defined(_WIN32)
typedef HMODULE vulkan_library_t;
#else
typedef void* vulkan_library_t;
#endif  // defined(_WIN32)

typedef struct emit_spirv_vulkan_state_t {
  // Optional output path supplied by the caller.
  const char* output_path;

  // True when live Vulkan discovery found no usable physical device.
  bool skipped;

  // Human-readable reason printed to stderr when skipped is true.
  const char* skip_message;

  // Loaded Vulkan loader library.
  vulkan_library_t vulkan_library;

  // Vulkan loader entry point used to resolve all other functions.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;

  // Optional Vulkan instance-version query.
  PFN_vkEnumerateInstanceVersion enumerate_instance_version;

  // Vulkan instance creation function.
  PFN_vkCreateInstance create_instance;

  // Vulkan instance destruction function.
  PFN_vkDestroyInstance destroy_instance;

  // Vulkan physical-device enumeration function.
  PFN_vkEnumeratePhysicalDevices enumerate_physical_devices;

  // Vulkan physical-device properties query function.
  PFN_vkGetPhysicalDeviceProperties2 get_physical_device_properties2;

  // Vulkan physical-device features query function.
  PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2;

  // Vulkan physical-device extension enumeration function.
  PFN_vkEnumerateDeviceExtensionProperties
      enumerate_device_extension_properties;

  // Minimal Vulkan instance used only for physical-device queries.
  VkInstance instance;

  // Selected physical device queried by the Loom raw Vulkan adapter.
  VkPhysicalDevice physical_device;

  // Properties for the selected physical device, printed by the example.
  VkPhysicalDeviceProperties physical_device_properties;

  // SPIR-V target package linked into this embedding binary.
  loomc_target_environment_t* target_environment;

  // Shared API context with the SPIR-V target dialect registered.
  loomc_context_t* context;

  // Per-worker scratch storage used by deserialize, compile, and emit.
  loomc_workspace_t* workspace;

  // Immutable source containing the Loom kernel module.
  loomc_source_t* source;

  // Mutable module compiled and emitted by this invocation.
  loomc_module_t* module;

  // Live Vulkan-derived SPIR-V target profile.
  loomc_target_profile_t* target_profile;

  // Invocation-ready target selection derived from the profile.
  loomc_target_selection_t* target_selection;

  // Immutable prepared compiler handle.
  loomc_compiler_t* compiler;

  // Prepared target pipeline shared across invocations.
  loomc_pass_program_t* pass_program;

  // Last operation result, reset between phases.
  loomc_result_t* result;
} emit_spirv_vulkan_state_t;

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

static void emit_spirv_vulkan_state_initialize(emit_spirv_vulkan_state_t* state,
                                               const char* output_path) {
  memset(state, 0, sizeof(*state));
  state->output_path = output_path;
}

static void vulkan_library_close(vulkan_library_t library) {
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

static void emit_spirv_vulkan_state_deinitialize(
    emit_spirv_vulkan_state_t* state) {
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
  if (state->instance != VK_NULL_HANDLE && state->destroy_instance != NULL) {
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
    state->destroy_instance(state->instance, NULL);
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  }
  vulkan_library_close(state->vulkan_library);
}

static void emit_spirv_vulkan_state_reset_result(
    emit_spirv_vulkan_state_t* state) {
  loomc_result_release(state->result);
  state->result = NULL;
}

static void emit_spirv_vulkan_state_skip(emit_spirv_vulkan_state_t* state,
                                         const char* message) {
  state->skipped = true;
  state->skip_message = message;
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

static bool vulkan_result_is_success(VkResult result) {
  return result == VK_SUCCESS;
}

static bool vulkan_result_is_skip(VkResult result) {
  return result == VK_ERROR_INITIALIZATION_FAILED ||
         result == VK_ERROR_INCOMPATIBLE_DRIVER ||
         result == VK_ERROR_EXTENSION_NOT_PRESENT ||
         result == VK_ERROR_LAYER_NOT_PRESENT;
}

static loomc_status_t vulkan_result_to_status(VkResult result,
                                              const char* operation) {
  if (vulkan_result_is_success(result)) {
    return loomc_ok_status();
  }
  switch (result) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_TOO_MANY_OBJECTS:
      return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED, operation);
    case VK_ERROR_DEVICE_LOST:
      return loomc_make_status(LOOMC_STATUS_DATA_LOSS, operation);
    default:
      return loomc_make_status(LOOMC_STATUS_UNKNOWN, operation);
  }
}

static bool vulkan_library_open(vulkan_library_t* out_library) {
  static const char* const kVulkanLibraryNames[] = {
#if defined(_WIN32)
      "vulkan-1.dll",
#elif defined(__APPLE__)
      "libvulkan.1.dylib",
      "libvulkan.dylib",
      "libMoltenVK.dylib",
#else
      "libvulkan.so.1",
      "libvulkan.so",
#endif  // defined(_WIN32)
  };
  *out_library = 0;
  for (size_t i = 0;
       i < sizeof(kVulkanLibraryNames) / sizeof(kVulkanLibraryNames[0]); ++i) {
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
#if defined(_WIN32)
    vulkan_library_t library = LoadLibraryA(kVulkanLibraryNames[i]);
#else
    vulkan_library_t library = dlopen(kVulkanLibraryNames[i], RTLD_NOW);
#endif  // defined(_WIN32)
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
    if (library != 0) {
      *out_library = library;
      return true;
    }
  }
  return false;
}

static void* vulkan_library_lookup(vulkan_library_t library,
                                   const char* symbol) {
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
#if defined(_WIN32)
  void* proc = (void*)GetProcAddress(library, symbol);
#else
  void* proc = dlsym(library, symbol);
#endif  // defined(_WIN32)
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  return proc;
}

static PFN_vkVoidFunction vulkan_loader_proc(
    const emit_spirv_vulkan_state_t* state, const char* name) {
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
  PFN_vkVoidFunction proc = state->get_instance_proc_addr(VK_NULL_HANDLE, name);
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  return proc;
}

static PFN_vkVoidFunction vulkan_instance_proc(
    const emit_spirv_vulkan_state_t* state, const char* name) {
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
  PFN_vkVoidFunction proc =
      state->get_instance_proc_addr(state->instance, name);
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  return proc;
}

static loomc_status_t load_vulkan_loader(emit_spirv_vulkan_state_t* state) {
  if (!vulkan_library_open(&state->vulkan_library)) {
    emit_spirv_vulkan_state_skip(state, "Vulkan loader library was not found");
    return loomc_ok_status();
  }
  state->get_instance_proc_addr =
      (PFN_vkGetInstanceProcAddr)vulkan_library_lookup(state->vulkan_library,
                                                       "vkGetInstanceProcAddr");
  if (state->get_instance_proc_addr == NULL) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan loader does not export vkGetInstanceProcAddr");
    return loomc_ok_status();
  }

  state->create_instance =
      (PFN_vkCreateInstance)vulkan_loader_proc(state, "vkCreateInstance");
  state->enumerate_instance_version =
      (PFN_vkEnumerateInstanceVersion)vulkan_loader_proc(
          state, "vkEnumerateInstanceVersion");
  if (state->create_instance == NULL) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan loader does not expose vkCreateInstance");
  }
  return loomc_ok_status();
}

static loomc_status_t create_vulkan_instance(emit_spirv_vulkan_state_t* state) {
  uint32_t instance_api_version = VK_API_VERSION_1_0;
  if (state->enumerate_instance_version != NULL) {
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
    VkResult result = state->enumerate_instance_version(&instance_api_version);
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
    if (vulkan_result_is_skip(result)) {
      emit_spirv_vulkan_state_skip(
          state, "Vulkan instance version query is unavailable");
      return loomc_ok_status();
    }
    LOOMC_RETURN_IF_ERROR(vulkan_result_to_status(
        result, "Vulkan instance version query failed"));
  }
  if (instance_api_version < VK_API_VERSION_1_1) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan 1.1 instance support is required for live probing");
    return loomc_ok_status();
  }

  uint32_t requested_api_version = instance_api_version;
  if (requested_api_version > VK_API_VERSION_1_3) {
    requested_api_version = VK_API_VERSION_1_3;
  }
  VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "loomc live Vulkan SPIR-V example",
      .applicationVersion = 1,
      .pEngineName = "loomc",
      .engineVersion = 1,
      .apiVersion = requested_api_version,
  };
  VkInstanceCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
  };
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
  VkResult result =
      state->create_instance(&create_info, NULL, &state->instance);
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  if (vulkan_result_is_skip(result)) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan instance could not be created on this machine");
    return loomc_ok_status();
  }
  return vulkan_result_to_status(result, "Vulkan instance creation failed");
}

static loomc_status_t load_vulkan_instance_functions(
    emit_spirv_vulkan_state_t* state) {
  state->destroy_instance =
      (PFN_vkDestroyInstance)vulkan_instance_proc(state, "vkDestroyInstance");
  state->enumerate_physical_devices =
      (PFN_vkEnumeratePhysicalDevices)vulkan_instance_proc(
          state, "vkEnumeratePhysicalDevices");
  state->get_physical_device_properties2 =
      (PFN_vkGetPhysicalDeviceProperties2)vulkan_instance_proc(
          state, "vkGetPhysicalDeviceProperties2");
  state->get_physical_device_features2 =
      (PFN_vkGetPhysicalDeviceFeatures2)vulkan_instance_proc(
          state, "vkGetPhysicalDeviceFeatures2");
  state->enumerate_device_extension_properties =
      (PFN_vkEnumerateDeviceExtensionProperties)vulkan_instance_proc(
          state, "vkEnumerateDeviceExtensionProperties");
  if (state->destroy_instance == NULL ||
      state->enumerate_physical_devices == NULL ||
      state->get_physical_device_properties2 == NULL ||
      state->get_physical_device_features2 == NULL) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan instance does not expose required live-probe functions");
  }
  return loomc_ok_status();
}

static loomc_status_t select_first_vulkan_physical_device(
    emit_spirv_vulkan_state_t* state) {
  uint32_t physical_device_count = 0;
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
  VkResult result = state->enumerate_physical_devices(
      state->instance, &physical_device_count, NULL);
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  if (vulkan_result_is_skip(result)) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan physical-device enumeration is unavailable");
    return loomc_ok_status();
  }
  LOOMC_RETURN_IF_ERROR(vulkan_result_to_status(
      result, "Vulkan physical-device count query failed"));
  if (physical_device_count == 0) {
    emit_spirv_vulkan_state_skip(state,
                                 "no Vulkan physical devices are available");
    return loomc_ok_status();
  }

  VkPhysicalDevice* physical_devices = NULL;
  loomc_status_t status =
      loomc_allocator_malloc(loomc_allocator_system(),
                             sizeof(*physical_devices) * physical_device_count,
                             (void**)&physical_devices);
  if (!loomc_status_is_ok(status)) {
    return status;
  }

  uint32_t written_physical_device_count = physical_device_count;
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
  result = state->enumerate_physical_devices(
      state->instance, &written_physical_device_count, physical_devices);
  LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
  if (vulkan_result_is_skip(result)) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan physical-device enumeration is unavailable");
    result = VK_SUCCESS;
  }
  if (result == VK_INCOMPLETE) {
    emit_spirv_vulkan_state_skip(
        state, "Vulkan physical-device inventory changed during enumeration");
    result = VK_SUCCESS;
  }
  if (vulkan_result_is_success(result) && !state->skipped &&
      written_physical_device_count == 0) {
    emit_spirv_vulkan_state_skip(state,
                                 "no Vulkan physical devices are available");
  }
  if (vulkan_result_is_success(result) && !state->skipped) {
    state->physical_device = physical_devices[0];
    VkPhysicalDeviceProperties2 properties2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    };
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_PUSH();
    state->get_physical_device_properties2(state->physical_device,
                                           &properties2);
    LOOMC_EXAMPLE_LEAK_CHECK_DISABLE_POP();
    state->physical_device_properties = properties2.properties;
  } else if (!vulkan_result_is_success(result)) {
    status = vulkan_result_to_status(
        result, "Vulkan physical-device enumeration failed");
  }
  loomc_allocator_free(loomc_allocator_system(), physical_devices);
  return status;
}

static loomc_status_t initialize_live_vulkan(emit_spirv_vulkan_state_t* state) {
  loomc_status_t status = load_vulkan_loader(state);
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = create_vulkan_instance(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = load_vulkan_instance_functions(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = select_first_vulkan_physical_device(state);
  }
  return status;
}

static loomc_status_t create_target_environment_and_context(
    emit_spirv_vulkan_state_t* state) {
  loomc_status_t status = loomc_target_environment_create_spirv(
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
    emit_spirv_vulkan_state_t* state) {
  loomc_status_t status =
      loomc_workspace_create(NULL, loomc_allocator_system(), &state->workspace);
  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("double_i32_at_byte_offset.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create(&source_options, loomc_allocator_system(),
                                 &state->source);
  }
  return status;
}

static loomc_status_t create_target_profile_and_selection(
    emit_spirv_vulkan_state_t* state) {
  loomc_spirv_vulkan_function_table_t functions = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE,
      .structure_size = sizeof(functions),
      .get_physical_device_properties2 = state->get_physical_device_properties2,
      .get_physical_device_features2 = state->get_physical_device_features2,
      .enumerate_device_extension_properties =
          state->enumerate_device_extension_properties,
  };
  loomc_spirv_vulkan_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("live-vulkan-device"),
      .physical_device = state->physical_device,
      .functions = &functions,
  };
  loomc_status_t status = loomc_target_profile_create_spirv_vulkan(
      state->target_environment, &profile_options, loomc_allocator_system(),
      &state->target_profile, &state->result);
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(state->result)) {
    print_result_diagnostics(state->result);
    emit_spirv_vulkan_state_skip(
        state,
        "selected Vulkan device did not produce a usable SPIR-V "
        "target profile");
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    emit_spirv_vulkan_state_reset_result(state);
    status = loomc_target_selection_create_from_profile(
        state->target_profile, loomc_allocator_system(),
        &state->target_selection);
  }
  return status;
}

static loomc_status_t require_live_profile_feature(
    emit_spirv_vulkan_state_t* state, loomc_spirv_feature_t feature,
    const char* skip_message) {
  loomc_target_fact_state_t feature_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_status_t status = loomc_spirv_target_profile_query_feature(
      state->target_profile, feature, &feature_state);
  if (loomc_status_is_ok(status) &&
      feature_state != LOOMC_TARGET_FACT_STATE_TRUE) {
    emit_spirv_vulkan_state_skip(state, skip_message);
  }
  return status;
}

static loomc_status_t check_live_profile_support(
    emit_spirv_vulkan_state_t* state) {
  loomc_status_t status = require_live_profile_feature(
      state, LOOMC_SPIRV_FEATURE_VULKAN_SHADER,
      "selected Vulkan device cannot run Vulkan SPIR-V shaders");
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = require_live_profile_feature(
        state, LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
        "selected Vulkan device does not expose buffer device address");
  }
  return status;
}

static uint32_t spirv_version_major(uint64_t version) {
  return (uint32_t)((version >> 16) & UINT64_C(0xFF));
}

static uint32_t spirv_version_minor(uint64_t version) {
  return (uint32_t)((version >> 8) & UINT64_C(0xFF));
}

static loomc_status_t print_live_profile_summary(
    const emit_spirv_vulkan_state_t* state) {
  loomc_spirv_environment_value_t max_spirv = {0};
  loomc_spirv_limit_value_t flat_workgroup_size = {0};
  loomc_spirv_limit_value_t subgroup_size = {0};
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_environment(
      state->target_profile, LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
      &max_spirv));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      state->target_profile, LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
      &flat_workgroup_size));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      state->target_profile, LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, &subgroup_size));
  printf("device %s api=%u.%u.%u max_spirv=%" PRIu32 ".%" PRIu32
         " max_workgroup=%" PRIu64 " subgroup=%" PRIu64 "\n",
         state->physical_device_properties.deviceName,
         VK_VERSION_MAJOR(state->physical_device_properties.apiVersion),
         VK_VERSION_MINOR(state->physical_device_properties.apiVersion),
         VK_VERSION_PATCH(state->physical_device_properties.apiVersion),
         spirv_version_major(max_spirv.value),
         spirv_version_minor(max_spirv.value), flat_workgroup_size.value,
         subgroup_size.value);
  return loomc_ok_status();
}

static loomc_status_t create_compiler_and_target_pipeline(
    emit_spirv_vulkan_state_t* state) {
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
      .identifier = loomc_make_cstring_view("live-vulkan-spirv-prepared-low"),
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
    emit_spirv_vulkan_state_reset_result(state);
  }
  return status;
}

static loomc_status_t create_resources(emit_spirv_vulkan_state_t* state) {
  loomc_status_t status = create_target_environment_and_context(state);
  if (loomc_status_is_ok(status)) {
    status = create_workspace_and_source(state);
  }
  if (loomc_status_is_ok(status)) {
    status = create_target_profile_and_selection(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = check_live_profile_support(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = print_live_profile_summary(state);
  }
  if (loomc_status_is_ok(status) && !state->skipped) {
    status = create_compiler_and_target_pipeline(state);
  }
  return status;
}

static loomc_status_t deserialize_source(emit_spirv_vulkan_state_t* state) {
  loomc_status_t status = loomc_module_deserialize_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &state->module, &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result,
                                       "source deserialization failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_spirv_vulkan_state_reset_result(state);
  }
  return status;
}

static loomc_status_t compile_module_to_prepared_low(
    emit_spirv_vulkan_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("double_i32_at_byte_offset"),
  };
  loomc_status_t status = loomc_compile_module(
      state->compiler, state->workspace, state->pass_program, state->module,
      &compile_options, loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "compilation failed");
  }
  if (loomc_status_is_ok(status)) {
    emit_spirv_vulkan_state_reset_result(state);
  }
  return status;
}

static loomc_status_t emit_spirv_artifact(emit_spirv_vulkan_state_t* state) {
  loomc_target_selection_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_selection = state->target_selection,
  };
  loomc_spirv_emit_options_t spirv_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      .structure_size = sizeof(spirv_options),
      .next = &target_options,
  };
  const loomc_option_entry_t emit_entries[] = {
      {
          .key = loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          .value = loomc_make_cstring_view("double_i32_at_byte_offset.spv"),
      },
  };
  loomc_option_dict_t option_dict = {
      .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      .structure_size = sizeof(option_dict),
      .next = &spirv_options,
      .entries = emit_entries,
      .entry_count = 1,
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &option_dict,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_status_t status = loomc_emit_module(
      state->target_environment, state->workspace, state->module, &emit_options,
      loomc_allocator_system(), &state->result);
  if (loomc_status_is_ok(status)) {
    status = require_successful_result(state->result, "SPIR-V emission failed");
  }
  return status;
}

static loomc_status_t summarize_and_maybe_write_artifact(
    emit_spirv_vulkan_state_t* state) {
  const loomc_artifact_t* artifact = find_result_artifact(
      state->result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV));
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "SPIR-V executable artifact was not produced");
  }
  if (artifact->contents.data_length < sizeof(uint32_t)) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "SPIR-V artifact is too small");
  }

  uint32_t magic = 0;
  memcpy(&magic, artifact->contents.data, sizeof(magic));
  printf("artifact %.*s format=%.*s bytes=%zu magic=0x%08" PRIx32 "\n",
         (int)artifact->identifier.size, artifact->identifier.data,
         (int)artifact->format.size, artifact->format.data,
         (size_t)artifact->contents.data_length, magic);

  if (state->output_path == NULL) {
    return loomc_ok_status();
  }
  loomc_status_t status = loomc_artifact_write_to_path(
      artifact, loomc_make_cstring_view(state->output_path),
      loomc_allocator_system());
  if (loomc_status_is_ok(status)) {
    printf("wrote %s\n", state->output_path);
  }
  return status;
}

static loomc_status_t run_emit_spirv_vulkan_example(const char* output_path,
                                                    bool* out_skipped) {
  emit_spirv_vulkan_state_t state;
  emit_spirv_vulkan_state_initialize(&state, output_path);

  loomc_status_t status = initialize_live_vulkan(&state);
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = create_resources(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = deserialize_source(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = compile_module_to_prepared_low(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = emit_spirv_artifact(&state);
  }
  if (loomc_status_is_ok(status) && !state.skipped) {
    status = summarize_and_maybe_write_artifact(&state);
  }

  *out_skipped = state.skipped;
  if (state.skipped) {
    fprintf(stderr, "skipping live Vulkan SPIR-V example: %s\n",
            state.skip_message);
  }
  emit_spirv_vulkan_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  const char* output_path = NULL;
  loomc_status_t status = loomc_ok_status();
  if (argc > 2) {
    status = loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "usage: emit_spirv_vulkan [output.spv]");
  } else {
    output_path = argc > 1 ? argv[1] : NULL;
  }

  bool skipped = false;
  if (loomc_status_is_ok(status)) {
    status = run_emit_spirv_vulkan_example(output_path, &skipped);
  }
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return skipped ? 0 : 1;
}
