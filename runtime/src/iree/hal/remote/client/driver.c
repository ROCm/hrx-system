// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/driver.h"

#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/transport_factory.h"

IREE_API_EXPORT void iree_hal_remote_client_driver_options_initialize(
    iree_hal_remote_client_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->numa_node_id = IREE_ASYNC_AFFINITY_NUMA_NODE_ANY;
  iree_hal_remote_client_device_options_initialize(
      &out_options->default_device_options);
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_options_parse(
    iree_hal_remote_client_driver_options_t* options,
    iree_string_pair_list_t params) {
  return iree_hal_remote_client_device_options_parse(
      &options->default_device_options, params);
}

typedef struct iree_hal_remote_client_driver_t {
  // Base HAL resource header.
  iree_hal_resource_t resource;
  // Host allocator used to allocate the driver.
  iree_allocator_t host_allocator;
  // Device identifier reported by devices this driver creates.
  iree_string_view_t identifier;
  // Driver defaults copied at creation time.
  iree_hal_remote_client_driver_options_t options;

  // Shared receive buffer pool. Created lazily on first device creation from
  // the proactor pool. All child devices retain this pool.
  iree_hal_remote_recv_pool_t* recv_pool;
} iree_hal_remote_client_driver_t;

static const iree_hal_driver_vtable_t iree_hal_remote_client_driver_vtable;

static iree_hal_remote_client_driver_t* iree_hal_remote_client_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_driver_vtable);
  return (iree_hal_remote_client_driver_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_driver = NULL;

  iree_host_size_t total_size = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t server_address_offset = 0;
  iree_hal_remote_client_driver_t* driver = NULL;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(*driver), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(identifier.size, char, 1, &identifier_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          options->default_device_options.server_address.size, char, 1,
          &server_address_offset));
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void**)&driver);
  }

  if (iree_status_is_ok(status)) {
    memset(driver, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_remote_client_driver_vtable,
                                 &driver->resource);
    driver->host_allocator = host_allocator;
    driver->options = *options;

    iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                      (char*)driver + identifier_offset);
    iree_net_transport_factory_retain(options->transport_factory);
    driver->options.default_device_options.transport_factory =
        options->transport_factory;
    iree_string_view_append_to_buffer(
        options->default_device_options.server_address,
        &driver->options.default_device_options.server_address,
        (char*)driver + server_address_offset);

    *out_driver = (iree_hal_driver_t*)driver;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_remote_client_driver_destroy(
    iree_hal_driver_t* base_driver) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_recv_pool_release(driver->recv_pool);
  iree_net_transport_factory_release(driver->options.transport_factory);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);
  *out_device_info_count = 0;
  *out_device_infos = NULL;

  iree_string_view_t server_address =
      driver->options.default_device_options.server_address;
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(server_address)) {
    static const iree_string_view_t name =
        iree_string_view_literal("remote endpoint");

    iree_host_size_t total_size = 0;
    iree_host_size_t name_offset = 0;
    iree_host_size_t path_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_device_info_t), &total_size,
        IREE_STRUCT_FIELD_ALIGNED(name.size, char, 1, &name_offset),
        IREE_STRUCT_FIELD_ALIGNED(server_address.size, char, 1, &path_offset));

    iree_hal_device_info_t* device_info = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(host_allocator, total_size,
                                     (void**)&device_info);
    }
    if (iree_status_is_ok(status)) {
      memset(device_info, 0, total_size);
      char* storage = (char*)device_info;
      device_info->device_id = IREE_HAL_DEVICE_ID_DEFAULT;
      iree_string_view_append_to_buffer(name, &device_info->name,
                                        storage + name_offset);
      iree_string_view_append_to_buffer(server_address, &device_info->path,
                                        storage + path_offset);

      *out_device_info_count = 1;
      *out_device_infos = device_info;
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_status_t status = iree_ok_status();
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    status = iree_string_builder_append_cstring(
        builder, "Remote HAL endpoint: client-supplied transport address.\n");
  }
  return status;
}

// Ensures the driver has a recv_pool, creating one lazily from the proactor
// pool if needed.
static iree_status_t iree_hal_remote_client_driver_ensure_recv_pool(
    iree_hal_remote_client_driver_t* driver,
    const iree_hal_device_create_params_t* create_params) {
  iree_status_t status = iree_ok_status();
  if (!driver->recv_pool) {
    if (!create_params || !create_params->proactor_pool) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "remote driver requires a proactor_pool in "
                                "create_params for recv_pool initialization");
    } else {
      status = iree_hal_remote_recv_pool_create(
          create_params->proactor_pool, driver->options.numa_node_id,
          driver->host_allocator, &driver->recv_pool);
    }
  }
  return status;
}

static iree_status_t iree_hal_remote_client_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);

  iree_hal_remote_client_device_options_t options =
      driver->options.default_device_options;
  iree_status_t status = iree_ok_status();
  if (device_id != IREE_HAL_DEVICE_ID_DEFAULT) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "remote drivers only expose the configured default endpoint by ID");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_options_parse(
        &options,
        (iree_string_pair_list_t){.count = param_count, .pairs = params});
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_client_driver_ensure_recv_pool(driver, create_params);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_create(
        driver->identifier, &options, create_params, driver->recv_pool,
        host_allocator, out_device);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_remote_client_driver_t* driver =
      iree_hal_remote_client_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_device_options_t options =
      driver->options.default_device_options;
  iree_status_t status = iree_hal_remote_client_device_options_parse(
      &options,
      (iree_string_pair_list_t){.count = param_count, .pairs = params});

  if (iree_status_is_ok(status) && !iree_string_view_is_empty(device_path)) {
    options.server_address = device_path;
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_hal_remote_client_driver_ensure_recv_pool(driver, create_params);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_create(
        driver->identifier, &options, create_params, driver->recv_pool,
        host_allocator, out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_driver_vtable_t iree_hal_remote_client_driver_vtable = {
    .destroy = iree_hal_remote_client_driver_destroy,
    .query_available_devices =
        iree_hal_remote_client_driver_query_available_devices,
    .dump_device_info = iree_hal_remote_client_driver_dump_device_info,
    .create_device_by_id = iree_hal_remote_client_driver_create_device_by_id,
    .create_device_by_path =
        iree_hal_remote_client_driver_create_device_by_path,
};
