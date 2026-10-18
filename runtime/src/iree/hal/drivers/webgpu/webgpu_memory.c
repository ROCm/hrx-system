// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_memory.h"

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/webgpu/webgpu_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tracing.h"

static const char* IREE_HAL_WEBGPU_SLAB_PROVIDER_TRACE_ID =
    "iree-hal-webgpu-slab-provider";

//===----------------------------------------------------------------------===//
// WebGPU slab provider
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_slab_provider_t {
  // Base slab provider for vtable dispatch and reference counting.
  iree_hal_slab_provider_t base;
  // Parent HAL device used for allocation placement metadata.
  iree_hal_device_t* parent_device;
  // Borrowed bridge handle for the native GPUDevice.
  iree_hal_webgpu_handle_t device_handle;
  // Queue families granted access to materialized views.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Public usage supported by buffers from this provider.
  iree_hal_buffer_usage_t supported_usage;
  // Native usage fixed for every GPUBuffer from this provider.
  iree_hal_webgpu_buffer_usage_t native_usage;
  // Named-memory trace stream for whole GPUBuffer allocations.
  iree_hal_memory_trace_t trace;
  // Cumulative GPUBuffer allocations.
  iree_atomic_int64_t total_acquired;
  // Cumulative GPUBuffer destructions.
  iree_atomic_int64_t total_released;
} iree_hal_webgpu_slab_provider_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_webgpu_slab_provider_vtable;

static iree_hal_webgpu_buffer_usage_t iree_hal_webgpu_native_buffer_usage(
    iree_hal_buffer_usage_t usage) {
  iree_hal_webgpu_buffer_usage_t native_usage =
      IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_SRC |
      IREE_HAL_WEBGPU_BUFFER_USAGE_COPY_DST;
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_STORAGE |
                                  IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE)) {
    native_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_STORAGE;
  }
  if (iree_any_bit_set(usage, IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ)) {
    native_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_UNIFORM;
  }
  if (iree_any_bit_set(usage,
                       IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS)) {
    native_usage |= IREE_HAL_WEBGPU_BUFFER_USAGE_INDIRECT;
  }
  return native_usage;
}

static iree_status_t iree_hal_webgpu_slab_provider_create(
    iree_hal_device_t* parent_device, iree_hal_webgpu_handle_t device_handle,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_buffer_usage_t supported_usage, iree_string_view_t trace_name,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;

  iree_hal_webgpu_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_webgpu_slab_provider_vtable,
                                    host_allocator, &provider->base);
  provider->parent_device = parent_device;
  provider->device_handle = device_handle;
  provider->queue_family_affinity = queue_family_affinity;
  provider->supported_usage = supported_usage;
  provider->native_usage = iree_hal_webgpu_native_buffer_usage(supported_usage);
  provider->total_acquired = IREE_ATOMIC_VAR_INIT(0);
  provider->total_released = IREE_ATOMIC_VAR_INIT(0);

  iree_status_t status = iree_hal_memory_trace_initialize(
      trace_name, IREE_HAL_WEBGPU_SLAB_PROVIDER_TRACE_ID, &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }
  return status;
}

static void iree_hal_webgpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_webgpu_slab_provider_t* provider =
      (iree_hal_webgpu_slab_provider_t*)base_provider;
  iree_allocator_t host_allocator = provider->base.host_allocator;
  iree_hal_memory_trace_deinitialize(&provider->trace);
  iree_allocator_free(host_allocator, provider);
}

static iree_status_t iree_hal_webgpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
  iree_hal_webgpu_slab_provider_t* provider =
      (iree_hal_webgpu_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  if (IREE_UNLIKELY(min_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "WebGPU slab allocations must be non-empty");
  }
  iree_device_size_t allocation_size = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_align(min_length, 4, &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "WebGPU slab length overflow aligning %" PRIu64
                            " bytes",
                            (uint64_t)min_length);
  }

  const iree_hal_webgpu_handle_t buffer_handle =
      iree_hal_webgpu_import_device_create_buffer(
          provider->device_handle, provider->native_usage,
          (uint64_t)allocation_size, /*mapped_at_creation=*/0);
  if (buffer_handle == IREE_HAL_WEBGPU_HANDLE_NULL) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "WebGPU device.createBuffer failed for %" PRIu64
                            " bytes",
                            (uint64_t)allocation_size);
  }

  const iree_hal_buffer_placement_t placement = {
      .device = provider->parent_device,
      .queue_family_affinity = provider->queue_family_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_webgpu_buffer_create(
      placement, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      iree_hal_buffer_usage_access(provider->supported_usage),
      provider->supported_usage, allocation_size, buffer_handle,
      /*mapped_at_creation=*/false, provider->base.host_allocator, &buffer);
  if (iree_status_is_ok(status)) {
    out_slab->base_ptr = NULL;
    out_slab->length = allocation_size;
    out_slab->provider_handle = (uint64_t)(uintptr_t)buffer;
    out_slab->native_allocation.byte_length = allocation_size;
    out_slab->native_allocation.granularity = 4;
    out_slab->maintenance_range =
        iree_hal_make_buffer_range(0, allocation_size);
    out_slab->allocation_alignment =
        iree_max(min_alignment, IREE_HAL_WEBGPU_SLAB_ALIGNMENT);
    iree_hal_memory_trace_alloc(
        &provider->trace, (void*)(uintptr_t)buffer_handle, allocation_size);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_hal_webgpu_import_buffer_destroy(buffer_handle);
  }
  return status;
}

static void iree_hal_webgpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_webgpu_slab_provider_t* provider =
      (iree_hal_webgpu_slab_provider_t*)base_provider;
  if (!slab->provider_handle) {
    return;
  }
  iree_hal_buffer_t* buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
  iree_hal_memory_trace_free(
      &provider->trace,
      (void*)(uintptr_t)iree_hal_webgpu_buffer_handle(buffer));
  iree_hal_buffer_release(buffer);
  iree_atomic_fetch_add(&provider->total_released, 1,
                        iree_memory_order_relaxed);
}

static iree_status_t iree_hal_webgpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_webgpu_slab_provider_t* provider =
      (iree_hal_webgpu_slab_provider_t*)base_provider;
  iree_hal_buffer_t* slab_buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
  iree_status_t status = iree_hal_subspan_buffer_create_with_callback(
      slab_buffer, slab_offset, allocation_size, release_callback,
      provider->base.host_allocator, out_buffer);
  if (iree_status_is_ok(status)) {
    (*out_buffer)->allowed_access = allowed_access;
    (*out_buffer)->allowed_usage = allowed_usage;
  }
  return status;
}

static iree_status_t iree_hal_webgpu_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "WebGPU slab provider does not support HAL ASAN range advice");
}

static void iree_hal_webgpu_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "WebGPU slab provider cannot advise ASAN ranges");
}

static void iree_hal_webgpu_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  (void)base_provider;
  (void)slab;
  (void)offset;
  (void)length;
}

static void iree_hal_webgpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_webgpu_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_webgpu_slab_provider_t* provider =
      (const iree_hal_webgpu_slab_provider_t*)base_provider;
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_webgpu_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_webgpu_slab_provider_t* provider =
      (const iree_hal_webgpu_slab_provider_t*)base_provider;
  *out_properties = (iree_hal_slab_provider_properties_t){
      .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .supported_usage = provider->supported_usage,
      .queue_family_affinity = provider->queue_family_affinity,
      .allocation_alignment = IREE_HAL_WEBGPU_SLAB_ALIGNMENT,
      .max_allocation_alignment = IREE_DEVICE_SIZE_MAX / 2 + 1,
      .maintenance_alignment = 1,
  };
}

static void iree_hal_webgpu_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  iree_hal_slab_provider_acquire_slabs_inline(
      provider, request, iree_hal_webgpu_slab_provider_acquire_slab,
      iree_hal_webgpu_slab_provider_release_slab);
}

static void iree_hal_webgpu_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  iree_hal_slab_provider_release_slabs_inline(
      provider, request, iree_hal_webgpu_slab_provider_release_slab);
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_webgpu_slab_provider_vtable = {
        .destroy = iree_hal_webgpu_slab_provider_destroy,
        .acquire_slabs = iree_hal_webgpu_slab_provider_acquire_slabs,
        .release_slabs = iree_hal_webgpu_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_webgpu_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_webgpu_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_webgpu_slab_provider_advise_asan_range,
        .prefault = iree_hal_webgpu_slab_provider_prefault,
        .trim = iree_hal_webgpu_slab_provider_trim,
        .query_stats = iree_hal_webgpu_slab_provider_query_stats,
        .query_properties = iree_hal_webgpu_slab_provider_query_properties,
};

//===----------------------------------------------------------------------===//
// WebGPU slab pool factory
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_slab_pool_plan_t {
  // Generic cold construction product.
  iree_hal_slab_pool_plan_t base;
  // Allocator for this temporary plan.
  iree_allocator_t host_allocator;
  // Borrowed native allocation and progress resources.
  const iree_hal_webgpu_memory_backend_t* backend;
  // Sealed group's completion tracker.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Owned qualified access facts retained by the created pool.
  iree_hal_memory_contract_t* contract;
  // Queue families granted access to materialized buffers.
  iree_hal_queue_family_affinity_t queue_family_affinity;
  // Native buffer capabilities needed by the qualified scope.
  iree_hal_buffer_usage_t usage;
  // Borrowed diagnostic name until synchronous creation returns.
  iree_string_view_t trace_name;
} iree_hal_webgpu_slab_pool_plan_t;

static void iree_hal_webgpu_slab_pool_plan_destroy(
    iree_hal_slab_pool_plan_t* base_plan) {
  iree_hal_webgpu_slab_pool_plan_t* plan =
      (iree_hal_webgpu_slab_pool_plan_t*)base_plan;
  iree_hal_memory_contract_release(plan->contract);
  iree_allocator_free(plan->host_allocator, plan);
}

static iree_status_t iree_hal_webgpu_slab_pool_plan_create(
    iree_hal_slab_pool_plan_t* base_plan, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  iree_hal_webgpu_slab_pool_plan_t* plan =
      (iree_hal_webgpu_slab_pool_plan_t*)base_plan;
  iree_hal_slab_provider_t* provider = NULL;
  iree_status_t status = iree_hal_webgpu_slab_provider_create(
      plan->backend->device, plan->backend->device_handle,
      plan->queue_family_affinity, plan->usage, plan->trace_name,
      iree_hal_device_host_allocator(plan->backend->device), &provider);
  if (iree_status_is_ok(status)) {
    const iree_hal_passthrough_pool_options_t options = {
        .memory_contract = plan->contract,
        .epoch_query = plan->backend->epoch_query,
        .owned_mode = IREE_HAL_SLAB_POOL_OWNED_ENABLED,
        .trace_name = plan->trace_name,
    };
    status = iree_hal_passthrough_pool_create(
        options, provider, plan->backend->notification, plan->frontier_tracker,
        host_allocator, out_pool);
  }
  iree_hal_slab_provider_release(provider);
  return status;
}

static const iree_hal_slab_pool_plan_vtable_t
    iree_hal_webgpu_slab_pool_plan_vtable = {
        .destroy = iree_hal_webgpu_slab_pool_plan_destroy,
        .create = iree_hal_webgpu_slab_pool_plan_create,
};

static iree_hal_pool_placement_t iree_hal_webgpu_memory_placement(
    iree_hal_device_group_t* group, uint32_t device_ordinal) {
  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const iree_hal_topology_node_t* node =
        iree_hal_topology_node_at(topology, i);
    if (node->device_ordinal == device_ordinal &&
        node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE) {
      return (iree_hal_pool_placement_t){
          .mode = IREE_HAL_POOL_PLACEMENT_REQUIRED,
          .node = node->ordinal,
      };
    }
  }
  return (iree_hal_pool_placement_t){0};
}

static iree_status_t iree_hal_webgpu_memory_query_pair(
    void* user_data, iree_hal_memory_scope_id_t producer,
    iree_hal_memory_scope_id_t consumer,
    iree_hal_memory_pair_info_t* out_info) {
  (void)user_data;
  (void)producer;
  (void)consumer;
  // WebGPU defines visibility within a GPUDevice queue. Explicit execution
  // dependencies still establish happens-before between operations.
  *out_info = (iree_hal_memory_pair_info_t){
      .flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
               IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN,
      .release = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
      .acquire = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_slab_pool_query_owner(
    const iree_hal_webgpu_memory_backend_t* backend,
    iree_hal_device_group_t* group, uint32_t device_ordinal,
    iree_hal_pool_scope_t scope, const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  if (!scope.family_count || scope.host.access ||
      options->preferences.host.access || options->imports.class_count ||
      options->owned.exports.class_count ||
      options->owned.mode != IREE_HAL_SLAB_POOL_OWNED_ENABLED) {
    return iree_ok_status();
  }

  const uint64_t interfaces = UINT64_C(1)
                              << IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER;
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  iree_hal_queue_family_affinity_t families = 0;
  const iree_hal_atomic_operation_capabilities_t atomic_operations = {0};
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    if (iree_hal_device_memory_backend(
            iree_hal_queue_family_device(access->family)) != &backend->base ||
        (access->interfaces & ~interfaces) || access->requirements ||
        !iree_hal_atomic_operation_capabilities_support(
            atomic_operations, access->atomic_operations)) {
      return iree_ok_status();
    }
    usage |= access->usage;
    families |= iree_hal_make_queue_family_affinity(access->family->ordinal);
  }

  const iree_hal_pool_placement_t placement =
      iree_hal_webgpu_memory_placement(group, device_ordinal);
  if (options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
      (placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
       placement.node != options->placement.node)) {
    return iree_ok_status();
  }

  iree_hal_webgpu_slab_pool_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->base.vtable = &iree_hal_webgpu_slab_pool_plan_vtable;
  plan->base.info.placement = placement;
  plan->host_allocator = host_allocator;
  plan->backend = backend;
  plan->frontier_tracker =
      iree_hal_device_topology_info(backend->device)->frontier.tracker;
  plan->queue_family_affinity = families;
  plan->usage = usage;
  plan->trace_name = options->trace_name;

  iree_status_t status = iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(group),
      iree_hal_device_group_memory_scope_count(group),
      iree_hal_webgpu_buffer_binding_layout(), host_allocator, &plan->contract);
  if (iree_status_is_ok(status)) {
    iree_hal_memory_contract_t* contract = plan->contract;
    contract->placement = placement;
    contract->allowed_access = iree_hal_buffer_usage_access(usage);
    contract->allowed_usage = usage;
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const uint32_t queue_scope_id =
          scope.families[i].family->memory.queue_scope_id;
      for (uint32_t id = queue_scope_id; id <= queue_scope_id + 1; ++id) {
        contract->scopes[id].usage = scope.families[i].usage;
        contract->scopes[id].interfaces = (uint32_t)interfaces;
        contract->scopes[id].bindings[IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER] =
            IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE;
      }
    }
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_webgpu_memory_query_pair, NULL);
  }
  if (iree_status_is_ok(status)) {
    status = callback.fn(callback.user_data, &plan->base);
  } else {
    iree_hal_webgpu_slab_pool_plan_destroy(&plan->base);
  }
  return status;
}

static iree_status_t iree_hal_webgpu_slab_pool_query(
    void* self, iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  (void)self;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group) &&
                               iree_status_is_ok(status);
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->type == IREE_HAL_MEMORY_BACKEND_WEBGPU) {
      status = iree_hal_webgpu_slab_pool_query_owner(
          (const iree_hal_webgpu_memory_backend_t*)backend, group, (uint32_t)i,
          scope, options, callback, host_allocator);
    }
  }
  return status;
}

static const iree_hal_slab_pool_factory_t iree_hal_webgpu_slab_pool_factory = {
    .query = iree_hal_webgpu_slab_pool_query,
};
static const iree_hal_slab_pool_factory_t* const
    iree_hal_webgpu_slab_pool_factories[] = {
        &iree_hal_webgpu_slab_pool_factory,
};

void iree_hal_webgpu_memory_backend_initialize(
    iree_hal_webgpu_memory_backend_t* backend) {
  backend->base = (iree_hal_memory_backend_t){
      .type = IREE_HAL_MEMORY_BACKEND_WEBGPU,
      .factory_count = IREE_ARRAYSIZE(iree_hal_webgpu_slab_pool_factories),
      .factories = iree_hal_webgpu_slab_pool_factories,
  };
}
