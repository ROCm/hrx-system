// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "binding/hip/api.h"
#include "binding/hip/spirv_linker_compiler.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"

typedef struct iree_hip_link_input_t {
  // Next input in insertion order.
  struct iree_hip_link_input_t* next;
  // Owned SPIR-V input bytes.
  uint8_t* data;
  // Number of bytes in |data|.
  size_t data_size;
  // Owned NUL-terminated diagnostic name.
  char* name;
} iree_hip_link_input_t;

struct ihipLinkState_t {
  // Retains the state while an API operation is in progress.
  iree_atomic_ref_count_t ref_count;
  // Next state in the live-handle registry. Guarded by the registry mutex.
  struct ihipLinkState_t* next;
  // Serializes inputs, options, and executable output for this link state.
  iree_slim_mutex_t mutex;
  // First owned linker input in insertion order.
  iree_hip_link_input_t* input_head;
  // Tail used to append inputs in constant time.
  iree_hip_link_input_t** input_tail;
  // Number of inputs in |input_head|.
  iree_host_size_t input_count;
  // Owned compiler option strings.
  char** options;
  // Number of entries in |options|.
  iree_host_size_t option_count;
  // Owned linked executable bytes, valid until mutation or destruction.
  uint8_t* executable_data;
  // Number of bytes in |executable_data|.
  size_t executable_size;
};

static iree_once_flag iree_hip_link_registry_once = IREE_ONCE_FLAG_INIT;
static iree_slim_mutex_t iree_hip_link_registry_mutex;
static hipLinkState_t iree_hip_link_registry_head;

static void iree_hip_link_registry_initialize(void) {
  iree_slim_mutex_initialize(&iree_hip_link_registry_mutex);
}

static void iree_hip_link_registry_ensure_initialized(void) {
  iree_call_once(&iree_hip_link_registry_once,
                 iree_hip_link_registry_initialize);
}

static void iree_hip_link_state_release(hipLinkState_t state) {
  if (state && iree_atomic_ref_count_dec(&state->ref_count) == 1) {
    iree_hip_link_input_t* input = state->input_head;
    while (input) {
      iree_hip_link_input_t* next = input->next;
      iree_allocator_free(iree_allocator_system(), input->name);
      iree_allocator_free(iree_allocator_system(), input->data);
      iree_allocator_free(iree_allocator_system(), input);
      input = next;
    }
    for (iree_host_size_t i = 0; i < state->option_count; ++i) {
      iree_allocator_free(iree_allocator_system(), state->options[i]);
    }
    iree_allocator_free(iree_allocator_system(), state->options);
    iree_allocator_free(iree_allocator_system(), state->executable_data);
    iree_slim_mutex_deinitialize(&state->mutex);
    iree_allocator_free(iree_allocator_system(), state);
  }
}

static hipError_t iree_hip_link_state_acquire(hipLinkState_t state,
                                              hipLinkState_t* out_state) {
  *out_state = NULL;
  if (!state) return hipErrorInvalidHandle;

  iree_hip_link_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_link_registry_mutex);
  for (hipLinkState_t current = iree_hip_link_registry_head; current;
       current = current->next) {
    if (current != state) continue;
    iree_atomic_ref_count_inc(&current->ref_count);
    *out_state = current;
    break;
  }
  iree_slim_mutex_unlock(&iree_hip_link_registry_mutex);
  return *out_state ? hipSuccess : hipErrorInvalidHandle;
}

static hipError_t iree_hip_link_validate_options(unsigned int count,
                                                 hipJitOption* options,
                                                 void** option_values) {
  if (count == 0) return hipSuccess;
  if (!options || !option_values) return hipErrorInvalidValue;
  for (unsigned int i = 0; i < count; ++i) {
    if (!option_values[i]) return hipErrorInvalidValue;
    switch (options[i]) {
      case hipJitOptionIRtoISAOptExt:
      case hipJitOptionIRtoISAOptCountExt:
        break;
      default:
        return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

static hipError_t iree_hip_link_copy_options(unsigned int count,
                                             hipJitOption* options,
                                             void** option_values,
                                             hipLinkState_t state) {
  const char* const* source_options = NULL;
  iree_host_size_t source_option_count = 0;
  bool found_options = false;
  bool found_count = false;
  for (unsigned int i = 0; i < count; ++i) {
    if (options[i] == hipJitOptionIRtoISAOptExt) {
      if (found_options) return hipErrorInvalidValue;
      source_options = (const char* const*)option_values[i];
      found_options = true;
    } else if (options[i] == hipJitOptionIRtoISAOptCountExt) {
      if (found_count) return hipErrorInvalidValue;
      source_option_count = (iree_host_size_t)(uintptr_t)option_values[i];
      found_count = true;
    }
  }
  if (found_options != found_count || source_option_count == 0) {
    return count == 0 ? hipSuccess : hipErrorInvalidValue;
  }
  if (source_option_count > SIZE_MAX / sizeof(*state->options)) {
    return hipErrorOutOfMemory;
  }

  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), source_option_count * sizeof(*state->options),
      (void**)&state->options);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(state->options, 0, source_option_count * sizeof(*state->options));
  for (iree_host_size_t i = 0; i < source_option_count; ++i) {
    if (!source_options[i]) {
      state->option_count = i;
      return hipErrorInvalidValue;
    }
    const size_t option_length = strlen(source_options[i]);
    if (option_length == SIZE_MAX) {
      state->option_count = i;
      return hipErrorOutOfMemory;
    }
    status = iree_allocator_malloc(iree_allocator_system(), option_length + 1,
                                   (void**)&state->options[i]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      state->option_count = i;
      return hipErrorOutOfMemory;
    }
    memcpy(state->options[i], source_options[i], option_length + 1);
  }
  state->option_count = source_option_count;
  return hipSuccess;
}

static hipError_t iree_hip_link_append_input(hipLinkState_t state,
                                             const void* data, size_t size,
                                             const char* name) {
  if (state->input_count == SIZE_MAX) return hipErrorOutOfMemory;
  static const char kDefaultName[] = "LinkerProgram.spv";
  if (!name) name = kDefaultName;
  const size_t name_length = strlen(name);
  if (name_length == SIZE_MAX) return hipErrorOutOfMemory;

  iree_hip_link_input_t* input = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*input), (void**)&input);
  if (iree_status_is_ok(status)) {
    memset(input, 0, sizeof(*input));
    status = iree_allocator_malloc(iree_allocator_system(), size,
                                   (void**)&input->data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(iree_allocator_system(), name_length + 1,
                                   (void**)&input->name);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (input) {
      iree_allocator_free(iree_allocator_system(), input->data);
      iree_allocator_free(iree_allocator_system(), input);
    }
    return hipErrorOutOfMemory;
  }
  memcpy(input->data, data, size);
  input->data_size = size;
  memcpy(input->name, name, name_length + 1);

  *state->input_tail = input;
  state->input_tail = &input->next;
  ++state->input_count;
  iree_allocator_free(iree_allocator_system(), state->executable_data);
  state->executable_data = NULL;
  state->executable_size = 0;
  return hipSuccess;
}

static hipError_t iree_hip_link_compiler_status_to_result(
    iree_status_t status) {
  if (iree_status_is_ok(status)) return hipSuccess;
  const iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);
  if (code == IREE_STATUS_RESOURCE_EXHAUSTED) return hipErrorOutOfMemory;
  if (code == IREE_STATUS_UNAVAILABLE) return hipErrorNotSupported;
  return hipErrorInvalidConfiguration;
}

HIPAPI hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options,
                                void** optionValues, hipLinkState_t* stateOut) {
  if (!stateOut) return hipErrorInvalidValue;
  *stateOut = NULL;
  hipError_t result =
      iree_hip_link_validate_options(numOptions, options, optionValues);
  if (result != hipSuccess) return result;

  hipLinkState_t state = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  memset(state, 0, sizeof(*state));
  iree_atomic_ref_count_init(&state->ref_count);
  iree_slim_mutex_initialize(&state->mutex);
  state->input_tail = &state->input_head;
  result = iree_hip_link_copy_options(numOptions, options, optionValues, state);
  if (result != hipSuccess) {
    iree_hip_link_state_release(state);
    return result;
  }

  iree_hip_link_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_link_registry_mutex);
  state->next = iree_hip_link_registry_head;
  iree_hip_link_registry_head = state;
  iree_slim_mutex_unlock(&iree_hip_link_registry_mutex);
  *stateOut = state;
  return hipSuccess;
}

HIPAPI hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type,
                                 void* data, size_t size, const char* name,
                                 unsigned int numOptions, hipJitOption* options,
                                 void** optionValues) {
  if (!data || size == 0) return hipErrorInvalidImage;
  if (type != hipJitInputSpirv) return hipErrorInvalidValue;
  hipError_t result =
      iree_hip_link_validate_options(numOptions, options, optionValues);
  if (result != hipSuccess) return result;

  hipLinkState_t retained_state = NULL;
  result = iree_hip_link_state_acquire(state, &retained_state);
  if (result != hipSuccess) return result;
  iree_slim_mutex_lock(&retained_state->mutex);
  result = iree_hip_link_append_input(retained_state, data, size, name);
  iree_slim_mutex_unlock(&retained_state->mutex);
  iree_hip_link_state_release(retained_state);
  return result;
}

HIPAPI hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type,
                                 const char* path, unsigned int numOptions,
                                 hipJitOption* options, void** optionValues) {
  if (!state) return hipErrorInvalidHandle;
  if (type != hipJitInputSpirv || !path) return hipErrorInvalidValue;
  hipError_t result =
      iree_hip_link_validate_options(numOptions, options, optionValues);
  if (result != hipSuccess) return result;

  hipLinkState_t retained_state = NULL;
  result = iree_hip_link_state_acquire(state, &retained_state);
  if (result != hipSuccess) return result;

  FILE* file = fopen(path, "rb");
  uint8_t* data = NULL;
  size_t data_size = 0;
  if (!file || fseek(file, 0, SEEK_END) != 0) {
    result = hipErrorInvalidConfiguration;
  } else {
    const long file_size = ftell(file);
    if (file_size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
      result = hipErrorInvalidImage;
    } else {
      data_size = (size_t)file_size;
      iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                                   data_size, (void**)&data);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        result = hipErrorOutOfMemory;
      } else if (fread(data, 1, data_size, file) != data_size) {
        result = hipErrorInvalidImage;
      } else {
        iree_slim_mutex_lock(&retained_state->mutex);
        result =
            iree_hip_link_append_input(retained_state, data, data_size, path);
        iree_slim_mutex_unlock(&retained_state->mutex);
      }
    }
  }
  if (file) fclose(file);
  iree_allocator_free(iree_allocator_system(), data);
  iree_hip_link_state_release(retained_state);
  return result;
}

HIPAPI hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut,
                                  size_t* sizeOut) {
  if (!hipBinOut || !sizeOut) return hipErrorInvalidValue;
  *hipBinOut = NULL;
  *sizeOut = 0;
  hipLinkState_t retained_state = NULL;
  hipError_t result = iree_hip_link_state_acquire(state, &retained_state);
  if (result != hipSuccess) return hipErrorInvalidValue;

  int device_ordinal = 0;
  result = hipGetDevice(&device_ordinal);
  hipDeviceProp_t properties;
  if (result == hipSuccess) {
    result = hipGetDeviceProperties(&properties, device_ordinal);
  }
  if (result != hipSuccess) {
    iree_hip_link_state_release(retained_state);
    return result;
  }
  static const char kTargetPrefix[] = "amdgcn-amd-amdhsa--";
  char target_isa[sizeof(kTargetPrefix) + sizeof(properties.gcnArchName)];
  const int target_length = snprintf(target_isa, sizeof(target_isa), "%s%s",
                                     kTargetPrefix, properties.gcnArchName);
  if (target_length < 0 || (size_t)target_length >= sizeof(target_isa)) {
    iree_hip_link_state_release(retained_state);
    return hipErrorInvalidConfiguration;
  }

  iree_slim_mutex_lock(&retained_state->mutex);
  iree_hip_spirv_linker_input_t* inputs = NULL;
  iree_status_t status = retained_state->input_count == 0
                             ? iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                                "link state has no inputs")
                             : iree_ok_status();
  if (iree_status_is_ok(status) &&
      retained_state->input_count > SIZE_MAX / sizeof(*inputs)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "linker input table size overflow");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        iree_allocator_system(), retained_state->input_count * sizeof(*inputs),
        (void**)&inputs);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t i = 0;
    for (iree_hip_link_input_t* input = retained_state->input_head; input;
         input = input->next) {
      inputs[i].data = iree_make_const_byte_span(input->data, input->data_size);
      inputs[i].name = input->name;
      ++i;
    }
    iree_byte_span_t executable = iree_byte_span_empty();
    status = iree_hip_spirv_linker_compile(
        iree_make_cstring_view(target_isa), retained_state->input_count, inputs,
        retained_state->option_count,
        (const char* const*)retained_state->options, iree_allocator_system(),
        &executable);
    if (iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(),
                          retained_state->executable_data);
      retained_state->executable_data = executable.data;
      retained_state->executable_size = executable.data_length;
      *hipBinOut = retained_state->executable_data;
      *sizeOut = retained_state->executable_size;
    }
  }
  iree_allocator_free(iree_allocator_system(), inputs);
  iree_slim_mutex_unlock(&retained_state->mutex);
  result = iree_hip_link_compiler_status_to_result(status);
  iree_hip_link_state_release(retained_state);
  return result;
}

HIPAPI hipError_t hipLinkDestroy(hipLinkState_t state) {
  if (!state) return hipErrorInvalidValue;
  iree_hip_link_registry_ensure_initialized();
  iree_slim_mutex_lock(&iree_hip_link_registry_mutex);
  hipLinkState_t* link = &iree_hip_link_registry_head;
  while (*link && *link != state) link = &(*link)->next;
  const bool found = *link != NULL;
  if (found) *link = state->next;
  iree_slim_mutex_unlock(&iree_hip_link_registry_mutex);
  if (!found) return hipErrorInvalidValue;
  iree_hip_link_state_release(state);
  return hipSuccess;
}
