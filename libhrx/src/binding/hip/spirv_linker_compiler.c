// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/spirv_linker_compiler.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/internal/dynamic_library.h"

// Private declarations for the stable COMGR C ABI used by the HIP linker. The
// library is loaded dynamically so the runtime can still initialize on systems
// that do not install compiler components.
typedef int32_t iree_hip_comgr_status_t;
typedef int32_t iree_hip_comgr_data_kind_t;
typedef int32_t iree_hip_comgr_action_kind_t;

enum {
  IREE_HIP_COMGR_STATUS_SUCCESS = 0,
  IREE_HIP_COMGR_STATUS_OUT_OF_RESOURCES = 3,
};

enum {
  IREE_HIP_COMGR_DATA_KIND_BC = 0x6,
  IREE_HIP_COMGR_DATA_KIND_EXECUTABLE = 0x8,
  IREE_HIP_COMGR_DATA_KIND_BC_BUNDLE = 0x12,
  IREE_HIP_COMGR_DATA_KIND_SPIRV = 0x15,
};

enum {
  IREE_HIP_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 0x7,
  IREE_HIP_COMGR_ACTION_UNBUNDLE = 0xF,
  IREE_HIP_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE = 0x10,
};

typedef struct iree_hip_comgr_data_t {
  // Opaque COMGR data handle.
  uint64_t handle;
} iree_hip_comgr_data_t;

typedef struct iree_hip_comgr_data_set_t {
  // Opaque COMGR data-set handle.
  uint64_t handle;
} iree_hip_comgr_data_set_t;

typedef struct iree_hip_comgr_action_info_t {
  // Opaque COMGR action-information handle.
  uint64_t handle;
} iree_hip_comgr_action_info_t;

typedef struct iree_hip_comgr_library_t {
  // Loaded COMGR shared library.
  iree_dynamic_library_t* library;
  // Returns a diagnostic string for a COMGR status code.
  iree_hip_comgr_status_t (*status_string)(iree_hip_comgr_status_t status,
                                           const char** out_string);
  // Creates a compiler data object.
  iree_hip_comgr_status_t (*create_data)(iree_hip_comgr_data_kind_t kind,
                                         iree_hip_comgr_data_t* out_data);
  // Releases a compiler data object.
  iree_hip_comgr_status_t (*release_data)(iree_hip_comgr_data_t data);
  // Copies bytes into a compiler data object.
  iree_hip_comgr_status_t (*set_data)(iree_hip_comgr_data_t data, size_t size,
                                      const char* bytes);
  // Sets the diagnostic name of a compiler data object.
  iree_hip_comgr_status_t (*set_data_name)(iree_hip_comgr_data_t data,
                                           const char* name);
  // Copies bytes out of a compiler data object.
  iree_hip_comgr_status_t (*get_data)(iree_hip_comgr_data_t data,
                                      size_t* inout_size, char* bytes);
  // Creates an empty compiler data set.
  iree_hip_comgr_status_t (*create_data_set)(
      iree_hip_comgr_data_set_t* out_data_set);
  // Destroys a compiler data set.
  iree_hip_comgr_status_t (*destroy_data_set)(
      iree_hip_comgr_data_set_t data_set);
  // Adds a data object to a compiler data set.
  iree_hip_comgr_status_t (*data_set_add)(iree_hip_comgr_data_set_t data_set,
                                          iree_hip_comgr_data_t data);
  // Returns the number of data objects of |kind| in a data set.
  iree_hip_comgr_status_t (*action_data_count)(
      iree_hip_comgr_data_set_t data_set, iree_hip_comgr_data_kind_t kind,
      size_t* out_count);
  // Acquires one data object of |kind| from a data set.
  iree_hip_comgr_status_t (*action_data_get_data)(
      iree_hip_comgr_data_set_t data_set, iree_hip_comgr_data_kind_t kind,
      size_t index, iree_hip_comgr_data_t* out_data);
  // Creates compiler action information.
  iree_hip_comgr_status_t (*create_action_info)(
      iree_hip_comgr_action_info_t* out_action_info);
  // Destroys compiler action information.
  iree_hip_comgr_status_t (*destroy_action_info)(
      iree_hip_comgr_action_info_t action_info);
  // Sets the target ISA for a compiler action.
  iree_hip_comgr_status_t (*action_info_set_isa_name)(
      iree_hip_comgr_action_info_t action_info, const char* isa_name);
  // Sets command-line options for a compiler action.
  iree_hip_comgr_status_t (*action_info_set_option_list)(
      iree_hip_comgr_action_info_t action_info, const char* options[],
      size_t option_count);
  // Selects entries while unbundling compiler input.
  iree_hip_comgr_status_t (*action_info_set_bundle_entry_ids)(
      iree_hip_comgr_action_info_t action_info, const char* bundle_entry_ids[],
      size_t entry_count);
  // Enables device-library linking for a compiler action.
  iree_hip_comgr_status_t (*action_info_set_device_lib_linking)(
      iree_hip_comgr_action_info_t action_info, bool should_link);
  // Executes a compiler action.
  iree_hip_comgr_status_t (*do_action)(iree_hip_comgr_action_kind_t kind,
                                       iree_hip_comgr_action_info_t action_info,
                                       iree_hip_comgr_data_set_t input,
                                       iree_hip_comgr_data_set_t result);
} iree_hip_comgr_library_t;

static iree_status_t iree_hip_comgr_status_to_iree(
    const iree_hip_comgr_library_t* comgr, iree_hip_comgr_status_t status,
    const char* operation) {
  if (status == IREE_HIP_COMGR_STATUS_SUCCESS) return iree_ok_status();
  const char* status_string = NULL;
  if (comgr->status_string) {
    comgr->status_string(status, &status_string);
  }
  const iree_status_code_t code =
      status == IREE_HIP_COMGR_STATUS_OUT_OF_RESOURCES
          ? IREE_STATUS_RESOURCE_EXHAUSTED
          : IREE_STATUS_INVALID_ARGUMENT;
  return iree_make_status(code, "%s failed with COMGR status %d (%s)",
                          operation, status,
                          status_string ? status_string : "unknown");
}

static iree_status_t iree_hip_comgr_load(iree_allocator_t host_allocator,
                                         iree_hip_comgr_library_t* out_comgr) {
  memset(out_comgr, 0, sizeof(*out_comgr));
  static const char* kLibraryNames[] = {
      "libamd_comgr.so.3",
      "libamd_comgr.so",
  };
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kLibraryNames), kLibraryNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_comgr->library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AMD COMGR is required to link SPIR-V HIP modules");
  }

#define IREE_HIP_COMGR_LOOKUP(member, symbol)                                \
  if (iree_status_is_ok(status)) {                                           \
    status = iree_dynamic_library_lookup_symbol(out_comgr->library, symbol,  \
                                                (void**)&out_comgr->member); \
  }
  IREE_HIP_COMGR_LOOKUP(status_string, "amd_comgr_status_string");
  IREE_HIP_COMGR_LOOKUP(create_data, "amd_comgr_create_data");
  IREE_HIP_COMGR_LOOKUP(release_data, "amd_comgr_release_data");
  IREE_HIP_COMGR_LOOKUP(set_data, "amd_comgr_set_data");
  IREE_HIP_COMGR_LOOKUP(set_data_name, "amd_comgr_set_data_name");
  IREE_HIP_COMGR_LOOKUP(get_data, "amd_comgr_get_data");
  IREE_HIP_COMGR_LOOKUP(create_data_set, "amd_comgr_create_data_set");
  IREE_HIP_COMGR_LOOKUP(destroy_data_set, "amd_comgr_destroy_data_set");
  IREE_HIP_COMGR_LOOKUP(data_set_add, "amd_comgr_data_set_add");
  IREE_HIP_COMGR_LOOKUP(action_data_count, "amd_comgr_action_data_count");
  IREE_HIP_COMGR_LOOKUP(action_data_get_data, "amd_comgr_action_data_get_data");
  IREE_HIP_COMGR_LOOKUP(create_action_info, "amd_comgr_create_action_info");
  IREE_HIP_COMGR_LOOKUP(destroy_action_info, "amd_comgr_destroy_action_info");
  IREE_HIP_COMGR_LOOKUP(action_info_set_isa_name,
                        "amd_comgr_action_info_set_isa_name");
  IREE_HIP_COMGR_LOOKUP(action_info_set_option_list,
                        "amd_comgr_action_info_set_option_list");
  IREE_HIP_COMGR_LOOKUP(action_info_set_bundle_entry_ids,
                        "amd_comgr_action_info_set_bundle_entry_ids");
  IREE_HIP_COMGR_LOOKUP(action_info_set_device_lib_linking,
                        "amd_comgr_action_info_set_device_lib_linking");
  IREE_HIP_COMGR_LOOKUP(do_action, "amd_comgr_do_action");
#undef IREE_HIP_COMGR_LOOKUP

  if (!iree_status_is_ok(status)) {
    iree_dynamic_library_release(out_comgr->library);
    memset(out_comgr, 0, sizeof(*out_comgr));
  }
  return status;
}

static void iree_hip_comgr_unload(iree_hip_comgr_library_t* comgr) {
  iree_dynamic_library_release(comgr->library);
  memset(comgr, 0, sizeof(*comgr));
}

static iree_status_t iree_hip_comgr_create_action(
    const iree_hip_comgr_library_t* comgr, const char* target_isa,
    iree_host_size_t option_count, const char* const* options,
    bool link_device_libraries, iree_hip_comgr_action_info_t* out_action_info) {
  *out_action_info = (iree_hip_comgr_action_info_t){0};
  IREE_RETURN_IF_ERROR(iree_hip_comgr_status_to_iree(
      comgr, comgr->create_action_info(out_action_info),
      "amd_comgr_create_action_info"));
  iree_status_t status = iree_hip_comgr_status_to_iree(
      comgr, comgr->action_info_set_isa_name(*out_action_info, target_isa),
      "amd_comgr_action_info_set_isa_name");
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr,
        comgr->action_info_set_option_list(*out_action_info,
                                           (const char**)options, option_count),
        "amd_comgr_action_info_set_option_list");
  }
  if (iree_status_is_ok(status) && link_device_libraries) {
    status = iree_hip_comgr_status_to_iree(
        comgr,
        comgr->action_info_set_device_lib_linking(*out_action_info, true),
        "amd_comgr_action_info_set_device_lib_linking");
  }
  if (!iree_status_is_ok(status)) {
    comgr->destroy_action_info(*out_action_info);
    *out_action_info = (iree_hip_comgr_action_info_t){0};
  }
  return status;
}

static iree_status_t iree_hip_comgr_add_data(
    const iree_hip_comgr_library_t* comgr, iree_hip_comgr_data_set_t data_set,
    iree_hip_comgr_data_kind_t kind, iree_const_byte_span_t bytes,
    const char* name) {
  iree_hip_comgr_data_t data = {0};
  IREE_RETURN_IF_ERROR(iree_hip_comgr_status_to_iree(
      comgr, comgr->create_data(kind, &data), "amd_comgr_create_data"));
  iree_status_t status = iree_hip_comgr_status_to_iree(
      comgr, comgr->set_data(data, bytes.data_length, (const char*)bytes.data),
      "amd_comgr_set_data");
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr, comgr->set_data_name(data, name), "amd_comgr_set_data_name");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr, comgr->data_set_add(data_set, data), "amd_comgr_data_set_add");
  }
  comgr->release_data(data);
  return status;
}

static iree_status_t iree_hip_comgr_copy_first_data(
    const iree_hip_comgr_library_t* comgr, iree_hip_comgr_data_set_t data_set,
    iree_hip_comgr_data_kind_t kind, iree_allocator_t host_allocator,
    iree_byte_span_t* out_data) {
  *out_data = iree_byte_span_empty();
  size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_hip_comgr_status_to_iree(
      comgr, comgr->action_data_count(data_set, kind, &count),
      "amd_comgr_action_data_count"));
  if (count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "COMGR produced %zu outputs of kind %d; expected 1",
                            count, kind);
  }
  iree_hip_comgr_data_t data = {0};
  IREE_RETURN_IF_ERROR(iree_hip_comgr_status_to_iree(
      comgr, comgr->action_data_get_data(data_set, kind, 0, &data),
      "amd_comgr_action_data_get_data"));
  size_t data_length = 0;
  iree_status_t status = iree_hip_comgr_status_to_iree(
      comgr, comgr->get_data(data, &data_length, NULL), "amd_comgr_get_data");
  if (iree_status_is_ok(status) && data_length == 0) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "COMGR produced an empty output of kind %d", kind);
  }
  uint8_t* data_bytes = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, data_length, (void**)&data_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr, comgr->get_data(data, &data_length, (char*)data_bytes),
        "amd_comgr_get_data");
  }
  comgr->release_data(data);
  if (iree_status_is_ok(status)) {
    *out_data = iree_make_byte_span(data_bytes, data_length);
  } else {
    iree_allocator_free(host_allocator, data_bytes);
  }
  return status;
}

static bool iree_hip_spirv_input_is_bundled(iree_const_byte_span_t input) {
  static const char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  return input.data_length >= sizeof(kBundleMagic) - 1 &&
         memcmp(input.data, kBundleMagic, sizeof(kBundleMagic) - 1) == 0;
}

static iree_status_t iree_hip_comgr_add_spirv_input(
    const iree_hip_comgr_library_t* comgr, const char* target_isa,
    iree_host_size_t option_count, const char* const* options,
    const iree_hip_spirv_linker_input_t* input,
    iree_hip_comgr_data_set_t compile_inputs, iree_allocator_t host_allocator) {
  if (!iree_hip_spirv_input_is_bundled(input->data)) {
    return iree_hip_comgr_add_data(comgr, compile_inputs,
                                   IREE_HIP_COMGR_DATA_KIND_SPIRV, input->data,
                                   input->name);
  }

  iree_hip_comgr_data_set_t bundle_inputs = {0};
  iree_hip_comgr_data_set_t unbundled_outputs = {0};
  iree_hip_comgr_action_info_t action_info = {0};
  iree_byte_span_t unbundled_data = iree_byte_span_empty();
  iree_status_t status = iree_hip_comgr_status_to_iree(
      comgr, comgr->create_data_set(&bundle_inputs),
      "amd_comgr_create_data_set");
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_add_data(comgr, bundle_inputs,
                                     IREE_HIP_COMGR_DATA_KIND_BC_BUNDLE,
                                     input->data, input->name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr, comgr->create_data_set(&unbundled_outputs),
        "amd_comgr_create_data_set");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_create_action(
        comgr, target_isa, option_count, options,
        /*link_device_libraries=*/false, &action_info);
  }
  static const char* kSpirvBundleEntryIds[] = {
      "hip-spirv64-amd-amdhsa-unknown-amdgcnspirv",
  };
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr,
        comgr->action_info_set_bundle_entry_ids(
            action_info, kSpirvBundleEntryIds,
            IREE_ARRAYSIZE(kSpirvBundleEntryIds)),
        "amd_comgr_action_info_set_bundle_entry_ids");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        comgr,
        comgr->do_action(IREE_HIP_COMGR_ACTION_UNBUNDLE, action_info,
                         bundle_inputs, unbundled_outputs),
        "AMD_COMGR_ACTION_UNBUNDLE");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_copy_first_data(comgr, unbundled_outputs,
                                            IREE_HIP_COMGR_DATA_KIND_BC,
                                            host_allocator, &unbundled_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_add_data(
        comgr, compile_inputs, IREE_HIP_COMGR_DATA_KIND_SPIRV,
        iree_make_const_byte_span(unbundled_data.data,
                                  unbundled_data.data_length),
        input->name);
  }

  iree_allocator_free(host_allocator, unbundled_data.data);
  if (action_info.handle) comgr->destroy_action_info(action_info);
  if (unbundled_outputs.handle) comgr->destroy_data_set(unbundled_outputs);
  if (bundle_inputs.handle) comgr->destroy_data_set(bundle_inputs);
  return status;
}

iree_status_t iree_hip_spirv_linker_compile(
    iree_string_view_t target_isa, iree_host_size_t input_count,
    const iree_hip_spirv_linker_input_t* inputs, iree_host_size_t option_count,
    const char* const* options, iree_allocator_t host_allocator,
    iree_byte_span_t* out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = iree_byte_span_empty();
  if (input_count == 0 || !inputs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one SPIR-V input is required");
  }

  char target_isa_buffer[128];
  if (target_isa.size >= sizeof(target_isa_buffer)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "target ISA name exceeds %zu bytes",
                            sizeof(target_isa_buffer) - 1);
  }
  memcpy(target_isa_buffer, target_isa.data, target_isa.size);
  target_isa_buffer[target_isa.size] = '\0';

  iree_hip_comgr_library_t comgr;
  IREE_RETURN_IF_ERROR(iree_hip_comgr_load(host_allocator, &comgr));
  iree_hip_comgr_data_set_t compile_inputs = {0};
  iree_hip_comgr_data_set_t relocatables = {0};
  iree_hip_comgr_data_set_t executable_outputs = {0};
  iree_hip_comgr_action_info_t compile_action = {0};
  iree_hip_comgr_action_info_t link_action = {0};

  iree_status_t status = iree_hip_comgr_status_to_iree(
      &comgr, comgr.create_data_set(&compile_inputs),
      "amd_comgr_create_data_set");
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hip_comgr_add_spirv_input(&comgr, target_isa_buffer,
                                            option_count, options, &inputs[i],
                                            compile_inputs, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(&comgr,
                                           comgr.create_data_set(&relocatables),
                                           "amd_comgr_create_data_set");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_create_action(
        &comgr, target_isa_buffer, option_count, options,
        /*link_device_libraries=*/true, &compile_action);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        &comgr,
        comgr.do_action(IREE_HIP_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE,
                        compile_action, compile_inputs, relocatables),
        "AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        &comgr, comgr.create_data_set(&executable_outputs),
        "amd_comgr_create_data_set");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_create_action(
        &comgr, target_isa_buffer, /*option_count=*/0, /*options=*/NULL,
        /*link_device_libraries=*/false, &link_action);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_status_to_iree(
        &comgr,
        comgr.do_action(IREE_HIP_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                        link_action, relocatables, executable_outputs),
        "AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hip_comgr_copy_first_data(&comgr, executable_outputs,
                                            IREE_HIP_COMGR_DATA_KIND_EXECUTABLE,
                                            host_allocator, out_executable);
  }

  if (link_action.handle) comgr.destroy_action_info(link_action);
  if (compile_action.handle) comgr.destroy_action_info(compile_action);
  if (executable_outputs.handle) comgr.destroy_data_set(executable_outputs);
  if (relocatables.handle) comgr.destroy_data_set(relocatables);
  if (compile_inputs.handle) comgr.destroy_data_set(compile_inputs);
  iree_hip_comgr_unload(&comgr);
  return status;
}
