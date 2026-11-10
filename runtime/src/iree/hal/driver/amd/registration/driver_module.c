// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/registration/driver_module.h"

#include "iree/base/tooling/flags.h"
#include "iree/hal/driver/amd/driver.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, amd_rocjitsu_library, "",
          "Optional rocjitsu library path or loader name. If empty, uses "
          "LIBAMDF_ROCJITSU_PATH, then librocjitsu.so through the system "
          "loader. Requires --amd_rocjitsu_config.");
IREE_FLAG_LIST(string, amd_rocjitsu_config,
               "Path to a rocjitsu JSON topology containing one or more GPUs. "
               "Configured topologies replace native AMD discovery.");

static iree_status_t iree_hal_amd_driver_factory_enumerate(
    void* self, iree_host_size_t* out_count,
    const iree_hal_driver_info_t** out_infos) {
  static const iree_hal_driver_info_t infos[] = {
#if IREE_HAL_AMD_XDNA_ENABLED
      {.driver_id = 0,
       .driver_name = IREE_SVL("xdna"),
       .full_name = IREE_SVL("AMD XDNA execution using libamdf")},
#endif
#if IREE_HAL_AMD_GPU_ENABLED
      {.driver_id = 0,
       .driver_name = IREE_SVL("rdna"),
       .full_name = IREE_SVL("AMD RDNA execution using libamdf")},
      {.driver_id = 0,
       .driver_name = IREE_SVL("cdna"),
       .full_name = IREE_SVL("AMD CDNA execution using libamdf")},
      {.driver_id = 0,
       .driver_name = IREE_SVL("amdgpu"),
       .full_name = IREE_SVL("AMD GPU execution using libamdf")},
#endif
  };
  *out_count = IREE_ARRAYSIZE(infos);
  *out_infos = infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  *out_driver = NULL;
  iree_hal_amd_driver_options_t options;
  iree_hal_amd_driver_options_initialize(&options);
  const iree_flag_string_list_t configurations =
      FLAG_amd_rocjitsu_config_list();
  const bool has_library = FLAG_amd_rocjitsu_library[0] != 0;
  if (has_library && configurations.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--amd_rocjitsu_library requires "
                            "--amd_rocjitsu_config");
  }
  if (configurations.count == 0) {
    return iree_hal_amd_driver_create(&options, host_allocator, out_driver);
  }
  if (configurations.count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many rocjitsu machine configurations");
  }
  iree_host_size_t total_size = 0;
  iree_host_size_t files_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_FAM(configurations.count,
                            amdf_rocjitsu_machine_create_info_t),
      IREE_STRUCT_FIELD(configurations.count, iree_io_file_contents_t*,
                        &files_offset)));
  amdf_rocjitsu_machine_create_info_t* machines = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&machines));
  iree_io_file_contents_t** files =
      (iree_io_file_contents_t**)((uint8_t*)machines + files_offset);
  iree_status_t status = iree_ok_status();
  iree_host_size_t loaded_count = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < configurations.count; ++i) {
    status = iree_io_file_contents_read(configurations.values[i],
                                        host_allocator, &files[i]);
    if (iree_status_is_ok(status)) {
      ++loaded_count;
      if (memchr(files[i]->buffer.data, 0, files[i]->buffer.data_length)) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "rocjitsu JSON configuration contains NUL bytes");
      }
    }
    if (iree_status_is_ok(status)) {
      machines[i] = (amdf_rocjitsu_machine_create_info_t){
          .type = AMDF_STRUCTURE_TYPE_ROCJITSU_MACHINE_CREATE_INFO,
          .structure_size = sizeof(machines[i]),
          .library_path = FLAG_amd_rocjitsu_library,
          .configuration_json = (const char*)files[i]->buffer.data,
      };
    }
  }
  if (iree_status_is_ok(status)) {
    options.rocjitsu_machine_count = (uint32_t)configurations.count;
    options.rocjitsu_machines = machines;
    status = iree_hal_amd_driver_create(&options, host_allocator, out_driver);
  }
  for (iree_host_size_t i = 0; i < loaded_count; ++i) {
    iree_io_file_contents_free(files[i]);
  }
  iree_allocator_free(host_allocator, machines);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_amd_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .enumerate = iree_hal_amd_driver_factory_enumerate,
      .try_create = iree_hal_amd_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
