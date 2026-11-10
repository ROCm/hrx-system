// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "device_group.h"

#include "hrx_internal.h"
#include "iree/async/frontier_tracker.h"
#include "iree/base/internal/path.h"
#include "runtime.h"

static_assert((uint32_t)HRX_WAIT_FLAG_YIELD == IREE_ASYNC_WAIT_FLAG_YIELD,
              "wait flag mismatch");
static_assert((uint32_t)HRX_WAIT_FLAG_ACTIVE == IREE_ASYNC_WAIT_FLAG_ACTIVE,
              "wait flag mismatch");

IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_HOST_NUMA,
                           IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA,
                           "topology kinds must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE,
                           IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE,
                           "topology kinds must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE,
                           IREE_HAL_TOPOLOGY_NODE_KIND_LOGICAL_DEVICE,
                           "topology kinds must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_MEMORY_DOMAIN,
                           IREE_HAL_TOPOLOGY_NODE_KIND_MEMORY_DOMAIN,
                           "topology kinds must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_QUEUE_FAMILY,
                           IREE_HAL_TOPOLOGY_NODE_KIND_QUEUE_FAMILY,
                           "topology kinds must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_TOPOLOGY_NODE_KIND_FABRIC_ENDPOINT,
                           IREE_HAL_TOPOLOGY_NODE_KIND_FABRIC_ENDPOINT,
                           "topology kinds must match HAL");

iree_status_t hrx_device_group_create_with_extensions(
    hrx_runtime_t runtime, size_t device_count,
    const hrx_device_create_params_t* devices,
    const iree_hal_device_create_params_extension_t* extensions,
    iree_hal_device_runtime_feature_flags_t runtime_features,
    hrx_device_group_t* out_group) {
  if (!out_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_group = NULL;
  if (!runtime || !devices || !device_count ||
      device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid device group dimensions");
  }
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    if (!devices[i].uri || !devices[i].uri[0] ||
        (devices[i].completion_wait_flags &
         ~(HRX_WAIT_FLAG_YIELD | HRX_WAIT_FLAG_ACTIVE))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid device URI or completion wait flags");
    }
  }
  IREE_RETURN_IF_ERROR(status);

  // The topology capacity bounds both this stack scratch and the inline views.
  iree_hal_device_t* hal_devices[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  hrx_accelerator_type_t types[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  hrx_device_group_t group = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      runtime->host_allocator,
      sizeof(*group) + device_count * sizeof(group->devices[0]),
      (void**)&group));
  iree_atomic_ref_count_init(&group->ref_count);
  group->runtime = runtime;
  hrx_runtime_retain(runtime);

  size_t created_count = 0;
  for (size_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    const iree_string_view_t uri = iree_make_cstring_view(devices[i].uri);
    const iree_string_view_t scheme = iree_uri_schema(uri);
    types[i] =
        iree_string_view_equal(scheme, IREE_SV("task"))   ? HRX_ACCELERATOR_CPU
        : iree_string_view_equal(scheme, IREE_SV("xdna")) ? HRX_ACCELERATOR_NPU
                                                          : HRX_ACCELERATOR_GPU;
    iree_hal_driver_t* driver = NULL;
    status = hrx_runtime_acquire_driver(runtime, scheme, &driver);
    if (iree_status_is_ok(status)) {
      iree_hal_device_create_params_t params =
          iree_hal_device_create_params_default();
      params.next = extensions;
      params.runtime_features = runtime_features;
      params.proactor_pool = runtime->proactor_pool;
      params.event_sink = hrx_runtime_hal_event_sink(runtime);
      params.completion_wait_flags = devices[i].completion_wait_flags;
      status = iree_hal_driver_create_device_by_uri(
          driver, uri, &params, runtime->host_allocator,
          &hal_devices[created_count]);
      if (iree_status_is_ok(status)) {
        ++created_count;
      }
    }
    iree_hal_driver_release(driver);
  }

  iree_async_frontier_tracker_t* tracker = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), runtime->host_allocator,
        &tracker);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker);
    for (size_t i = 0; i < created_count && iree_status_is_ok(status); ++i) {
      status =
          iree_hal_device_group_builder_add_device(&builder, hal_devices[i]);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, runtime->host_allocator, &group->hal_group);
    } else {
      iree_hal_device_group_builder_deinitialize(&builder);
    }
  }
  iree_async_frontier_tracker_release(tracker);
  for (size_t i = 0; i < created_count; ++i) {
    iree_hal_device_release(hal_devices[i]);
  }

  // Only sealed devices expose queue coordinates and compatibility sources.
  for (size_t i = 0; i < device_count && iree_status_is_ok(status); ++i) {
    status = hrx_device_initialize(group, i, types[i], &group->devices[i]);
    ++group->device_count;
  }
  if (iree_status_is_ok(status)) {
    *out_group = group;
  } else {
    hrx_device_group_release(group);
  }
  return status;
}

hrx_status_t hrx_device_group_create(hrx_runtime_t runtime, size_t device_count,
                                     const hrx_device_create_params_t* devices,
                                     hrx_device_group_t* out_group) {
  return hrx_status_from_iree(hrx_device_group_create_with_extensions(
      runtime, device_count, devices, NULL, 0, out_group));
}

void hrx_device_group_retain(hrx_device_group_t group) {
  if (group) {
    iree_atomic_ref_count_inc(&group->ref_count);
  }
}

void hrx_device_group_release(hrx_device_group_t group) {
  if (!group) {
    return;
  }
  if (iree_atomic_ref_count_dec(&group->ref_count) != 1) {
    return;
  }
  hrx_runtime_t runtime = group->runtime;
  for (size_t i = 0; i < group->device_count; ++i) {
    hrx_device_deinitialize(&group->devices[i]);
  }
  iree_hal_device_group_release(group->hal_group);
  iree_allocator_free(runtime->host_allocator, group);
  hrx_runtime_release(runtime);
}

size_t hrx_device_group_device_count(hrx_device_group_t group) {
  return group ? group->device_count : 0;
}

hrx_device_t hrx_device_group_get_device(hrx_device_group_t group,
                                         size_t index) {
  return group && index < group->device_count ? &group->devices[index] : NULL;
}

size_t hrx_device_group_topology_node_count(hrx_device_group_t group) {
  return group ? iree_hal_topology_node_count(
                     iree_hal_device_group_topology(group->hal_group))
               : 0;
}

hrx_status_t hrx_device_group_query_topology_node(
    hrx_device_group_t group, size_t ordinal, hrx_topology_node_t* out_node) {
  if (!out_node) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "output is NULL");
  }
  *out_node = (hrx_topology_node_t){0};
  if (!group || ordinal >= hrx_device_group_topology_node_count(group)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE, "invalid topology node");
  }
  const iree_hal_topology_node_t* node = iree_hal_topology_node_at(
      iree_hal_device_group_topology(group->hal_group), ordinal);
  *out_node = (hrx_topology_node_t){
      .ordinal = node->ordinal,
      .parent_ordinal = node->parent_ordinal,
      .device_ordinal = node->device_ordinal,
      .kind = (hrx_topology_node_kind_t)node->kind,
      .local_ordinal = node->local_ordinal,
      .physical_device_affinity = node->physical_device_affinity,
  };
  return hrx_ok_status();
}
