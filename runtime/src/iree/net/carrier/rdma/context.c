// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/context.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"

#define IREE_NET_RDMA_CONTEXT_DEVICE_NAME_CAPACITY 64

struct iree_net_rdma_context_t {
  // Reference count controlling context lifetime.
  iree_atomic_ref_count_t ref_count;

  // Host allocator used for the context allocation.
  iree_allocator_t host_allocator;

  // Dynamically loaded libibverbs symbols.
  iree_net_libverbs_t libverbs;

  // Dynamically loaded librdmacm symbols.
  iree_net_librdmacm_t librdmacm;

  // True when libverbs was successfully initialized.
  bool libverbs_initialized;

  // True when librdmacm was successfully initialized.
  bool librdmacm_initialized;

  // Opened verbs device context.
  struct ibv_context* device_context;

  // Protection domain used for queue pairs and memory registrations.
  struct ibv_pd* protection_domain;

  // Queried verbs device attributes.
  struct ibv_device_attr device_attributes;

  // Queried attributes for the selected port.
  struct ibv_port_attr port_attributes;

  // Queried GID value for the selected port/GID index.
  union ibv_gid gid;

  // Selected one-based port number.
  uint8_t port_number;

  // Selected GID table index.
  uint8_t gid_index;

  // Length of the selected device name in device_name.
  iree_host_size_t device_name_length;

  // Selected verbs device name.
  char device_name[IREE_NET_RDMA_CONTEXT_DEVICE_NAME_CAPACITY];
};

static bool iree_net_rdma_context_device_name_matches(
    iree_string_view_t requested_name, const char* actual_name) {
  return iree_string_view_is_empty(requested_name) ||
         iree_string_view_equal(requested_name,
                                iree_make_cstring_view(actual_name));
}

static bool iree_net_rdma_context_port_is_active(
    const struct ibv_port_attr* port_attributes) {
  return port_attributes->state == IBV_PORT_ACTIVE;
}

static iree_status_t iree_net_rdma_context_query_port(
    iree_net_rdma_context_t* context, uint8_t port_number, uint8_t gid_index,
    struct ibv_port_attr* out_port_attributes, union ibv_gid* out_gid) {
  memset(out_port_attributes, 0, sizeof(*out_port_attributes));
  memset(out_gid, 0, sizeof(*out_gid));

  int result = context->libverbs.ibv_query_port(
      context->device_context, port_number, out_port_attributes);
  iree_status_t status =
      iree_status_from_errno(__FILE__, __LINE__, result, "ibv_query_port");

  if (iree_status_is_ok(status) &&
      !iree_net_rdma_context_port_is_active(out_port_attributes)) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "RDMA device port %u is not active",
                              (unsigned)port_number);
  }

  if (iree_status_is_ok(status)) {
    result = context->libverbs.ibv_query_gid(context->device_context,
                                             port_number, gid_index, out_gid);
    status =
        iree_status_from_errno(__FILE__, __LINE__, result, "ibv_query_gid");
  }

  return status;
}

static iree_status_t iree_net_rdma_context_select_port(
    iree_net_rdma_context_t* context, iree_net_rdma_context_options_t options) {
  iree_status_t status = iree_ok_status();

  if (options.port_number != 0) {
    status = iree_net_rdma_context_query_port(
        context, options.port_number, options.gid_index,
        &context->port_attributes, &context->gid);
    if (iree_status_is_ok(status)) {
      context->port_number = options.port_number;
      context->gid_index = options.gid_index;
    }
    return status;
  }

  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "RDMA device has no active ports");
  for (uint8_t port_number = 1;
       port_number <= context->device_attributes.phys_port_cnt; ++port_number) {
    struct ibv_port_attr port_attributes;
    union ibv_gid gid;
    iree_status_t port_status = iree_net_rdma_context_query_port(
        context, port_number, options.gid_index, &port_attributes, &gid);
    if (iree_status_is_ok(port_status)) {
      iree_status_ignore(status);
      context->port_attributes = port_attributes;
      context->gid = gid;
      context->port_number = port_number;
      context->gid_index = options.gid_index;
      status = iree_ok_status();
      break;
    }
    iree_status_ignore(port_status);
  }

  return status;
}

static iree_status_t iree_net_rdma_context_try_open_device(
    iree_net_rdma_context_t* context, struct ibv_device* device,
    iree_net_rdma_context_options_t options, iree_status_t* inout_first_failure,
    bool* out_selected) {
  *out_selected = false;

  const char* device_name = context->libverbs.ibv_get_device_name(device);
  bool is_requested_device = !iree_string_view_is_empty(options.device_name);
  if (!iree_net_rdma_context_device_name_matches(options.device_name,
                                                 device_name)) {
    return iree_ok_status();
  }

  context->device_context = context->libverbs.ibv_open_device(device);
  iree_status_t status = iree_ok_status();
  if (!context->device_context) {
    status =
        iree_status_from_errno(__FILE__, __LINE__, errno, "ibv_open_device");
  }

  if (iree_status_is_ok(status)) {
    int result = context->libverbs.ibv_query_device(
        context->device_context, &context->device_attributes);
    status =
        iree_status_from_errno(__FILE__, __LINE__, result, "ibv_query_device");
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_context_select_port(context, options);
  }

  if (iree_status_is_ok(status)) {
    context->protection_domain =
        context->libverbs.ibv_alloc_pd(context->device_context);
    if (!context->protection_domain) {
      status =
          iree_status_from_errno(__FILE__, __LINE__, errno, "ibv_alloc_pd");
    }
  }

  if (iree_status_is_ok(status)) {
    int device_name_length = snprintf(
        context->device_name, sizeof(context->device_name), "%s", device_name);
    if (device_name_length < 0) {
      context->device_name[0] = '\0';
      context->device_name_length = 0;
    } else {
      context->device_name_length = (iree_host_size_t)device_name_length;
    }
    if (context->device_name_length >= sizeof(context->device_name)) {
      context->device_name_length = sizeof(context->device_name) - 1;
    }
    *out_selected = true;
  } else {
    if (context->protection_domain) {
      int result = context->libverbs.ibv_dealloc_pd(context->protection_domain);
      IREE_ASSERT(result == 0, "ibv_dealloc_pd failed during cleanup: %d",
                  result);
      context->protection_domain = NULL;
    }
    if (context->device_context) {
      int result = context->libverbs.ibv_close_device(context->device_context);
      IREE_ASSERT(result == 0, "ibv_close_device failed during cleanup: %d",
                  result);
      context->device_context = NULL;
    }
  }

  if (is_requested_device || iree_status_is_ok(status)) {
    return status;
  }

  if (iree_status_is_ok(*inout_first_failure)) {
    *inout_first_failure = status;
  } else {
    iree_status_ignore(status);
  }
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_context_open_selected_device(
    iree_net_rdma_context_t* context, iree_net_rdma_context_options_t options) {
  int device_count = 0;
  struct ibv_device** device_list =
      context->libverbs.ibv_get_device_list(&device_count);

  iree_status_t status = iree_ok_status();
  if (!device_list || device_count == 0) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "no RDMA devices available");
  }

  iree_status_t first_failure = iree_ok_status();
  bool selected = false;
  for (int device_index = 0;
       iree_status_is_ok(status) && device_index < device_count && !selected;
       ++device_index) {
    status = iree_net_rdma_context_try_open_device(
        context, device_list[device_index], options, &first_failure, &selected);
  }

  if (iree_status_is_ok(status) && !selected &&
      iree_string_view_is_empty(options.device_name)) {
    if (!iree_status_is_ok(first_failure)) {
      status = first_failure;
      first_failure = iree_ok_status();
    } else {
      status =
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "no RDMA devices with an active port are available");
    }
  } else if (iree_status_is_ok(status) && !selected) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND, "RDMA device '%.*s' was not found",
        (int)options.device_name.size, options.device_name.data);
  }
  iree_status_ignore(first_failure);

  if (device_list) {
    context->libverbs.ibv_free_device_list(device_list);
  }

  return status;
}

static void iree_net_rdma_context_destroy(iree_net_rdma_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (context->protection_domain) {
    int result = context->libverbs.ibv_dealloc_pd(context->protection_domain);
    IREE_ASSERT(result == 0, "ibv_dealloc_pd failed during destroy: %d",
                result);
  }
  if (context->device_context) {
    int result = context->libverbs.ibv_close_device(context->device_context);
    IREE_ASSERT(result == 0, "ibv_close_device failed during destroy: %d",
                result);
  }
  if (context->librdmacm_initialized) {
    iree_net_librdmacm_deinitialize(&context->librdmacm);
  }
  if (context->libverbs_initialized) {
    iree_net_libverbs_deinitialize(&context->libverbs);
  }

  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_net_rdma_context_create(
    iree_net_rdma_context_options_t options, iree_allocator_t host_allocator,
    iree_net_rdma_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_rdma_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);

  if (iree_status_is_ok(status)) {
    memset(context, 0, sizeof(*context));
    iree_atomic_ref_count_init(&context->ref_count);
    context->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_libverbs_initialize(options.library_search_paths,
                                          host_allocator, &context->libverbs);
    context->libverbs_initialized = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_librdmacm_initialize(options.library_search_paths,
                                           host_allocator, &context->librdmacm);
    context->librdmacm_initialized = iree_status_is_ok(status);
  }

  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_context_open_selected_device(context, options);
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else if (context) {
    iree_net_rdma_context_destroy(context);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_net_rdma_context_retain(
    iree_net_rdma_context_t* context) {
  if (context) {
    iree_atomic_ref_count_inc(&context->ref_count);
  }
}

IREE_API_EXPORT void iree_net_rdma_context_release(
    iree_net_rdma_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_net_rdma_context_destroy(context);
  }
}

IREE_API_EXPORT const iree_net_libverbs_t* iree_net_rdma_context_libverbs(
    const iree_net_rdma_context_t* context) {
  return &context->libverbs;
}

IREE_API_EXPORT const iree_net_librdmacm_t* iree_net_rdma_context_librdmacm(
    const iree_net_rdma_context_t* context) {
  return &context->librdmacm;
}

IREE_API_EXPORT struct ibv_context* iree_net_rdma_context_device(
    const iree_net_rdma_context_t* context) {
  return context->device_context;
}

IREE_API_EXPORT struct ibv_pd* iree_net_rdma_context_protection_domain(
    const iree_net_rdma_context_t* context) {
  return context->protection_domain;
}

IREE_API_EXPORT iree_string_view_t
iree_net_rdma_context_device_name(const iree_net_rdma_context_t* context) {
  return iree_make_string_view(context->device_name,
                               context->device_name_length);
}

IREE_API_EXPORT uint8_t
iree_net_rdma_context_port_number(const iree_net_rdma_context_t* context) {
  return context->port_number;
}

IREE_API_EXPORT uint8_t
iree_net_rdma_context_gid_index(const iree_net_rdma_context_t* context) {
  return context->gid_index;
}

IREE_API_EXPORT const struct ibv_port_attr*
iree_net_rdma_context_port_attributes(const iree_net_rdma_context_t* context) {
  return &context->port_attributes;
}

IREE_API_EXPORT const union ibv_gid* iree_net_rdma_context_gid(
    const iree_net_rdma_context_t* context) {
  return &context->gid;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_context_register_host_memory(
    iree_net_rdma_context_t* context, void* base_ptr, iree_host_size_t length,
    int access_flags, struct ibv_mr** out_memory_region) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_memory_region);
  *out_memory_region = NULL;

  if (!base_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "base_ptr must be non-NULL");
  }
  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "length must be non-zero");
  }

  struct ibv_mr* memory_region = context->libverbs.ibv_reg_mr(
      context->protection_domain, base_ptr, length, access_flags);
  if (!memory_region) {
    return iree_status_from_errno(__FILE__, __LINE__, errno, "ibv_reg_mr");
  }

  *out_memory_region = memory_region;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_context_register_dmabuf(
    iree_net_rdma_context_t* context, uint64_t offset, iree_host_size_t length,
    uint64_t iova, int file_descriptor, int access_flags,
    struct ibv_mr** out_memory_region) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_memory_region);
  *out_memory_region = NULL;

  if (!iree_net_libverbs_has_dmabuf_mr(&context->libverbs)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "libibverbs does not provide ibv_reg_dmabuf_mr");
  }
  if (length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "length must be non-zero");
  }

  struct ibv_mr* memory_region = context->libverbs.ibv_reg_dmabuf_mr(
      context->protection_domain, offset, length, iova, file_descriptor,
      access_flags);
  if (!memory_region) {
    return iree_status_from_errno(__FILE__, __LINE__, errno,
                                  "ibv_reg_dmabuf_mr");
  }

  *out_memory_region = memory_region;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_context_deregister_memory(
    iree_net_rdma_context_t* context, struct ibv_mr* memory_region) {
  IREE_ASSERT_ARGUMENT(context);
  if (!memory_region) return iree_ok_status();

  int result = context->libverbs.ibv_dereg_mr(memory_region);
  return iree_status_from_errno(__FILE__, __LINE__, result, "ibv_dereg_mr");
}
