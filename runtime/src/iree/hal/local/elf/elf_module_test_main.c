// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library.h"

// ELF module embedded in the binary. ELF-producing hosts exercise a native
// source-built fixture in addition to the checked-in compatibility artifacts
// used on hosts such as Windows that cannot emit ELF with the baseline
// toolchain.
#if defined(IREE_HAL_LOCAL_ELF_TEST_NATIVE_FIXTURE)
#include "iree/hal/local/elf/testdata/elementwise_mul_native.h"
#else
#include "iree/hal/local/elf/testdata/elementwise_mul_compatibility.h"
#endif  // IREE_HAL_LOCAL_ELF_TEST_NATIVE_FIXTURE

static iree_status_t query_test_file_data(
    iree_const_byte_span_t* out_file_data) {
  *out_file_data = iree_const_byte_span_empty();
#if defined(IREE_HAL_LOCAL_ELF_TEST_NATIVE_FIXTURE)
  if (elementwise_mul_native_size() != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected one native ELF fixture, got %zu",
                            elementwise_mul_native_size());
  }
  const struct iree_file_toc_t* file_toc = elementwise_mul_native_create();
#else
  if (elementwise_mul_compatibility_size() != 6) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected six compatibility ELF fixtures, got %zu",
                            elementwise_mul_compatibility_size());
  }

  iree_string_view_t pattern = iree_string_view_empty();
#if defined(IREE_ARCH_ARM_32)
  pattern = iree_make_cstring_view("*_arm_32.so");
#elif defined(IREE_ARCH_ARM_64)
  pattern = iree_make_cstring_view("*_arm_64.so");
#elif defined(IREE_ARCH_RISCV_32)
  pattern = iree_make_cstring_view("*_riscv_32.so");
#elif defined(IREE_ARCH_RISCV_64)
  pattern = iree_make_cstring_view("*_riscv_64.so");
#elif defined(IREE_ARCH_X86_32)
  pattern = iree_make_cstring_view("*_x86_32.so");
#elif defined(IREE_ARCH_X86_64)
  pattern = iree_make_cstring_view("*_x86_64.so");
#else
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no ELF compatibility fixture for this architecture");
#endif  // IREE_ARCH_*

  const struct iree_file_toc_t* file_toc = NULL;
  for (size_t i = 0; i < elementwise_mul_compatibility_size(); ++i) {
    const struct iree_file_toc_t* candidate =
        &elementwise_mul_compatibility_create()[i];
    if (iree_string_view_match_pattern(iree_make_cstring_view(candidate->name),
                                       pattern)) {
      if (file_toc) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "multiple ELF compatibility fixtures match '%s'", candidate->name);
      }
      file_toc = candidate;
    }
  }
  if (!file_toc) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no ELF compatibility fixture matches this architecture");
  }
#endif  // IREE_HAL_LOCAL_ELF_TEST_NATIVE_FIXTURE

  if (!file_toc->data || file_toc->size == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "embedded ELF module is empty");
  }
  *out_file_data = iree_make_const_byte_span(file_toc->data, file_toc->size);
  return iree_ok_status();
}

static iree_status_t run_module_test(iree_elf_module_t* module) {
  iree_hal_executable_environment_v0_t environment;
  iree_hal_executable_environment_initialize(iree_allocator_system(),
                                             &environment);

  void* query_fn_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_elf_module_lookup_export(
      module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME, &query_fn_ptr));

  const iree_hal_executable_library_header_t* const* query_result =
      (const iree_hal_executable_library_header_t* const*)iree_elf_call_p_ip(
          query_fn_ptr, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
          &environment);
  if (query_result == NULL || *query_result == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "library header is empty (version mismatch?)");
  }

  const iree_hal_executable_library_header_t* header = *query_result;
  if (header->version != IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "library version %u does not match expected %u",
                            header->version,
                            IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
  }

  if (strcmp(header->name, "elementwise_mul") != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "library name '%s' does not match expected name",
                            header->name);
  }

  const iree_hal_executable_library_v0_t* library =
      iree_hal_executable_library_v0_from_query_result(query_result);
  if (library->exports.count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "library exports %u entry points instead of one",
                            library->exports.count);
  }

  // ret0 = arg0 * arg1
  float arg0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float arg1[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  float ret0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[4] = {100.0f, 400.0f, 900.0f, 1600.0f};

  size_t binding_lengths[3] = {
      sizeof(arg0),
      sizeof(arg1),
      sizeof(ret0),
  };
  void* binding_ptrs[3] = {
      arg0,
      arg1,
      ret0,
  };
  const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = 1,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
      .workgroup_count_x = 1,
      .workgroup_count_y = 1,
      .workgroup_count_z = 1,
      .max_concurrency = 1,
      .binding_count = 3,
      .binding_lengths = binding_lengths,
      .binding_ptrs = binding_ptrs,
  };
  const iree_hal_executable_workgroup_state_v0_t workgroup_state = {
      .workgroup_id_x = 0,
      .workgroup_id_y = 0,
      .workgroup_id_z = 0,
      .processor_id = iree_cpu_query_processor_id(),
  };
  int ret = iree_elf_call_i_ppp((const void*)library->exports.ptrs[0],
                                (void*)&environment, (void*)&dispatch_state,
                                (void*)&workgroup_state);
  if (ret != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "dispatch function returned failure: %d", ret);
  }

  for (int i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    if (ret0[i] != expected[i]) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "output mismatch: ret[%d] = %.1f, expected %.1f",
                              i, ret0[i], expected[i]);
    }
  }

  return iree_ok_status();
}

static iree_status_t run_test() {
  iree_const_byte_span_t file_data;
  IREE_RETURN_IF_ERROR(query_test_file_data(&file_data));

  iree_elf_module_t module;
  IREE_RETURN_IF_ERROR(iree_elf_module_initialize_from_memory(
      file_data, iree_allocator_system(), &module));

  iree_status_t status = run_module_test(&module);
  iree_elf_module_deinitialize(&module);
  return status;
}

int main() {
  const iree_status_t result = run_test();
  int ret = (int)iree_status_code(result);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
  }
  return ret;
}
