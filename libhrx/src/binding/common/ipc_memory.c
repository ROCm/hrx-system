// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/ipc_memory.h"

#include <string.h>

#include "common/internal.h"

typedef enum iree_hal_streaming_ipc_memory_import_state_e {
  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED = 0,
  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING,
  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY,
  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING,
} iree_hal_streaming_ipc_memory_import_state_t;

typedef struct iree_hal_streaming_ipc_memory_binding_t {
  // Next per-device binding sharing the registry entry's one attachment.
  struct iree_hal_streaming_ipc_memory_binding_t* next;
  // Exact importing context retained for the binding lifetime.
  iree_hal_streaming_context_t* context;
  // Context-local streaming wrapper over the anchor or an alias HAL buffer.
  iree_hal_streaming_buffer_t* buffer;
  // Exact HIP-visible base pointer registered in |context|.
  void* device_ptr;
  // Successful opens still attributed to this context. A close prefers its
  // current context but may consume another binding's reference because HIP
  // identifies the process-wide cached import only by its shared address.
  uint32_t open_count;
  // Current alias/wrapper transition; guarded by the registry mutex.
  iree_hal_streaming_ipc_memory_import_state_t state;
} iree_hal_streaming_ipc_memory_binding_t;

struct iree_hal_streaming_ipc_memory_import_t {
  // Next live import in the process-wide registry.
  iree_hal_streaming_ipc_memory_import_t* next;
  // HAL buffer owning the one process-wide ROCr attachment.
  iree_hal_buffer_t* anchor_buffer;
  // HAL device retaining the allocator and backend resources used by the
  // anchor without extending the lifetime of an importing primary context.
  iree_hal_device_t* anchor_device;
  // Per-device/context pointer-table bindings sharing |anchor_buffer|.
  iree_hal_streaming_ipc_memory_binding_t* bindings;
  // Exact base pointer of the attached import view.
  void* device_ptr;
  // Raw process-independent backend token used as the primary key.
  uint8_t token[IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE];
  // Requested shareable extent supplied by the importing handle.
  uint64_t allocation_size;
  // Byte offset paired with |token| to form the duplicate key.
  uint64_t byte_offset;
  // Byte length of the imported buffer view.
  uint64_t view_size;
  // Process that exported this allocation.
  int64_t exporter_process_id;
  // Exporter's device ordinal in the shared visible-device namespace.
  uint64_t exporter_device_ordinal;
  // Sum of binding open counts. Bindings remain alive until this reaches zero
  // because hipIpcCloseMemHandle identifies only the shared VA.
  uint32_t open_count;
  // Current anchor attach/detach transition; guarded by the registry mutex.
  iree_hal_streaming_ipc_memory_import_state_t state;
};

// One registry entry reserved by a context-retirement transaction. The target
// binding and import remain DETACHING from reservation until rollback or
// commit, keeping their pointers and binding lists stable without retaining
// any extra objects.
typedef struct iree_hal_streaming_ipc_memory_context_release_entry_t {
  // Registry import borrowed while its DETACHING reservation keeps it live.
  iree_hal_streaming_ipc_memory_import_t* import;
  // Exact context binding removed if the transaction commits.
  iree_hal_streaming_ipc_memory_binding_t* released_binding;
  // Populated during commit only when the target binding held the final opens.
  iree_hal_streaming_ipc_memory_binding_t* remaining_bindings;
  // Whether commit consumed the import's final process-wide open references.
  bool is_final_detach;
} iree_hal_streaming_ipc_memory_context_release_entry_t;

void iree_hal_streaming_ipc_memory_registry_initialize(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(registry);
  memset(registry, 0, sizeof(*registry));
  iree_slim_mutex_initialize(&registry->mutex);
  iree_notification_initialize(&registry->changed);
  registry->host_allocator = host_allocator;
}

void iree_hal_streaming_ipc_memory_registry_deinitialize(
    iree_hal_streaming_ipc_memory_registry_t* registry) {
  if (!registry) return;
  iree_slim_mutex_lock(&registry->mutex);
  iree_hal_streaming_ipc_memory_import_t* imports = registry->imports;
  registry->imports = NULL;
  iree_slim_mutex_unlock(&registry->mutex);

  while (imports) {
    iree_hal_streaming_ipc_memory_import_t* import = imports;
    imports = import->next;
    IREE_ASSERT(import->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY);
    import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING;
    iree_hal_streaming_ipc_memory_binding_t* bindings = import->bindings;
    while (bindings) {
      iree_hal_streaming_ipc_memory_binding_t* binding = bindings;
      bindings = binding->next;
      IREE_ASSERT(binding->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY);
      iree_hal_streaming_memory_release_wrapped_buffer(binding->buffer);
      iree_hal_streaming_context_release(binding->context);
      iree_allocator_free(registry->host_allocator, binding);
    }
    iree_hal_buffer_release(import->anchor_buffer);
    iree_hal_device_release(import->anchor_device);
    iree_allocator_free(registry->host_allocator, import);
  }
  iree_notification_deinitialize(&registry->changed);
  iree_slim_mutex_deinitialize(&registry->mutex);
}

typedef enum iree_hal_streaming_ipc_memory_export_snapshot_result_e {
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_NOT_FOUND = 0,
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INVALID_BASE,
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INELIGIBLE,
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_OUT_OF_RANGE,
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_REFERENCE_EXHAUSTED,
  IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_READY,
} iree_hal_streaming_ipc_memory_export_snapshot_result_t;

typedef struct iree_hal_streaming_ipc_memory_export_snapshot_t {
  uint64_t device_ptr;
  iree_hal_streaming_context_t* candidate_context;
  iree_hal_streaming_ipc_memory_export_snapshot_result_t result;
  iree_hal_streaming_context_t* owner_context;
  iree_hal_buffer_t* buffer;
  uint64_t logical_allocation_size;
  iree_host_size_t device_ordinal;
} iree_hal_streaming_ipc_memory_export_snapshot_t;

// Runs with the owning context's buffer-table mutex held. No wrapper field may
// be read after this returns because a concurrent free can then remove the
// entry and destroy the wrapper.
static void iree_hal_streaming_ipc_memory_snapshot_export(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data) {
  iree_hal_streaming_ipc_memory_export_snapshot_t* snapshot =
      (iree_hal_streaming_ipc_memory_export_snapshot_t*)user_data;
  iree_hal_streaming_buffer_t* buffer =
      (iree_hal_streaming_buffer_t*)entry->user_data;
  if (IREE_UNLIKELY(entry->device_ptr != snapshot->device_ptr || offset != 0 ||
                    !buffer || buffer->device_ptr != snapshot->device_ptr ||
                    buffer->context != snapshot->candidate_context)) {
    snapshot->result =
        IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INVALID_BASE;
    return;
  }
  if (IREE_UNLIKELY(
          !buffer->buffer ||
          !iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer->buffer),
                             IREE_HAL_BUFFER_USAGE_SHARING_EXPORT) ||
          buffer->allocation_pool || buffer->is_managed ||
          iree_hal_buffer_allocated_buffer(buffer->buffer) != buffer->buffer ||
          iree_hal_buffer_byte_offset(buffer->buffer) != 0 ||
          !buffer->is_device_freeable)) {
    snapshot->result = IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INELIGIBLE;
    return;
  }
  if (IREE_UNLIKELY(buffer->logical_size == 0 ||
                    buffer->logical_size > (iree_device_size_t)SIZE_MAX)) {
    snapshot->result =
        IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_OUT_OF_RANGE;
    return;
  }
  if (IREE_UNLIKELY(!iree_hal_streaming_context_try_retain(
          snapshot->candidate_context))) {
    snapshot->result =
        IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_REFERENCE_EXHAUSTED;
    return;
  }

  iree_hal_buffer_retain(buffer->buffer);
  snapshot->owner_context = snapshot->candidate_context;
  snapshot->buffer = buffer->buffer;
  snapshot->logical_allocation_size = (uint64_t)buffer->logical_size;
  snapshot->device_ordinal = buffer->context->device_ordinal;
  snapshot->result = IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_READY;
}

// Requires a retained context and takes its buffer-table mutex. Returns true
// whenever the pointer names an entry, including one that is not
// export-eligible.
static bool iree_hal_streaming_ipc_memory_try_snapshot_export(
    iree_hal_streaming_context_t* context, uint64_t device_ptr,
    iree_hal_streaming_ipc_memory_export_snapshot_t* snapshot) {
  snapshot->candidate_context = context;
  hrx_status_t find_status = hrx_buffer_table_visit(
      &context->buffer_table, device_ptr,
      iree_hal_streaming_ipc_memory_snapshot_export, snapshot);
  if (hrx_status_is_ok(find_status)) {
    hrx_status_ignore(find_status);
    IREE_ASSERT(snapshot->result !=
                IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_NOT_FOUND);
    return true;
  }
  IREE_ASSERT(hrx_status_code(find_status) == HRX_STATUS_NOT_FOUND);
  hrx_status_ignore(find_status);
  return false;
}

iree_status_t iree_hal_streaming_ipc_memory_export(
    iree_hal_streaming_device_registry_t* device_registry,
    iree_hal_streaming_context_t* preferred_context, uint64_t device_ptr,
    iree_hal_streaming_ipc_memory_export_fn_t export_fn,
    iree_hal_streaming_ipc_memory_descriptor_t* out_descriptor,
    iree_host_size_t* out_device_ordinal) {
  IREE_ASSERT_ARGUMENT(device_registry);
  IREE_ASSERT_ARGUMENT(preferred_context);
  IREE_ASSERT_ARGUMENT(export_fn);
  IREE_ASSERT_ARGUMENT(out_descriptor);
  IREE_ASSERT_ARGUMENT(out_device_ordinal);
  memset(out_descriptor, 0, sizeof(*out_descriptor));
  *out_device_ordinal = 0;
  if (IREE_UNLIKELY(!export_fn)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IPC memory exporter is required");
  }
  if (IREE_UNLIKELY(device_ptr == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IPC export requires a device pointer");
  }

  iree_hal_streaming_ipc_memory_export_snapshot_t snapshot = {
      .device_ptr = device_ptr,
  };
  bool found = false;
  iree_hal_streaming_context_t* retained_preferred_context =
      iree_hal_streaming_context_list_try_retain_registered(device_registry,
                                                            preferred_context);
  if (retained_preferred_context) {
    found = iree_hal_streaming_ipc_memory_try_snapshot_export(
        retained_preferred_context, device_ptr, &snapshot);
    iree_hal_streaming_context_release(retained_preferred_context);
  }
  iree_hal_streaming_context_t* context =
      found
          ? NULL
          : iree_hal_streaming_context_list_retain_next(device_registry, NULL);
  while (!found && context) {
    if (context != preferred_context) {
      found = iree_hal_streaming_ipc_memory_try_snapshot_export(
          context, device_ptr, &snapshot);
    }
    iree_hal_streaming_context_t* next_context =
        found ? NULL
              : iree_hal_streaming_context_list_retain_next(device_registry,
                                                            context);
    iree_hal_streaming_context_release(context);
    context = next_context;
  }

  iree_status_t status = iree_ok_status();
  switch (snapshot.result) {
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_NOT_FOUND:
      status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
      break;
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INVALID_BASE:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "IPC export requires an allocation base");
      break;
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_INELIGIBLE:
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "IPC export requires a live allocation returned by hipMalloc");
      break;
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_OUT_OF_RANGE:
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "IPC export logical allocation size exceeds the HIP size range");
      break;
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_REFERENCE_EXHAUSTED:
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "IPC export context reference exhausted");
      break;
    case IREE_HAL_STREAMING_IPC_MEMORY_EXPORT_SNAPSHOT_READY:
      status =
          export_fn(snapshot.owner_context, snapshot.buffer, out_descriptor);
      if (iree_status_is_ok(status) &&
          IREE_UNLIKELY(out_descriptor->allocation_size == 0 ||
                        out_descriptor->byte_offset >
                            out_descriptor->allocation_size ||
                        snapshot.logical_allocation_size >
                            out_descriptor->allocation_size -
                                out_descriptor->byte_offset)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "IPC exporter returned invalid allocation view metadata");
      }
      break;
  }
  if (iree_status_is_ok(status)) {
    *out_device_ordinal = snapshot.device_ordinal;
  }
  if (!iree_status_is_ok(status)) {
    memset(out_descriptor, 0, sizeof(*out_descriptor));
  }
  iree_hal_buffer_release(snapshot.buffer);
  iree_hal_streaming_context_release(snapshot.owner_context);
  return status;
}

static bool iree_hal_streaming_ipc_memory_key_matches(
    const iree_hal_streaming_ipc_memory_import_t* import,
    const uint8_t token[IREE_HAL_STREAMING_IPC_MEMORY_TOKEN_SIZE],
    uint64_t byte_offset) {
  return import->byte_offset == byte_offset &&
         memcmp(import->token, token, sizeof(import->token)) == 0;
}

// Finds the context owning imports from |descriptor|'s exporter on the local
// device. Returns NULL if no ready binding claims the ownership domain and
// reports whether an in-flight transition must complete before it is known.
static iree_hal_streaming_ipc_memory_binding_t*
iree_hal_streaming_ipc_memory_find_source_binding_locked(
    const iree_hal_streaming_ipc_memory_registry_t* registry,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_host_size_t importing_device_ordinal, bool* out_is_transitioning) {
  *out_is_transitioning = false;
  iree_hal_streaming_ipc_memory_binding_t* source_binding = NULL;
  for (iree_hal_streaming_ipc_memory_import_t* import = registry->imports;
       import; import = import->next) {
    if (import->exporter_process_id != descriptor->exporter_process_id ||
        import->exporter_device_ordinal !=
            descriptor->exporter_device_ordinal) {
      continue;
    }
    if (import->state != IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
      *out_is_transitioning = true;
      continue;
    }
    for (iree_hal_streaming_ipc_memory_binding_t* binding = import->bindings;
         binding; binding = binding->next) {
      if (binding->context->device_ordinal != importing_device_ordinal) {
        continue;
      }
      if (binding->state != IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
        *out_is_transitioning = true;
        continue;
      }
      if (!source_binding) source_binding = binding;
      IREE_ASSERT(source_binding->context == binding->context);
    }
  }
  return source_binding;
}

// Collapses driver attach, access, and imported-wrapper failures into one
// private distinction while preserving allocation failures for the HIP layer.
static iree_status_t iree_hal_streaming_ipc_memory_normalize_import_failure(
    iree_status_t status) {
  if (iree_status_is_ok(status) ||
      iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
    return status;
  }
  iree_status_free(status);
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "IPC memory handle could not be attached");
}

// Unlinks |import| while |registry->mutex| is held.
static void iree_hal_streaming_ipc_memory_unlink_import_locked(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_ipc_memory_import_t* import) {
  iree_hal_streaming_ipc_memory_import_t** current = &registry->imports;
  while (*current && *current != import) current = &(*current)->next;
  IREE_ASSERT(*current == import);
  if (*current == import) *current = import->next;
  import->next = NULL;
}

// Unlinks |binding| while |registry->mutex| is held.
static void iree_hal_streaming_ipc_memory_unlink_binding_locked(
    iree_hal_streaming_ipc_memory_import_t* import,
    iree_hal_streaming_ipc_memory_binding_t* binding) {
  iree_hal_streaming_ipc_memory_binding_t** current = &import->bindings;
  while (*current && *current != binding) current = &(*current)->next;
  IREE_ASSERT(*current == binding);
  if (*current == binding) *current = binding->next;
  binding->next = NULL;
}

// Selects the open reference consumed by a pointer-only HIP close. Prefer the
// current context so reset can revoke an exact context's remaining opens, but
// fall back to any live reference to preserve process-wide cached-object close
// behavior after the caller changes devices.
static iree_hal_streaming_ipc_memory_binding_t*
iree_hal_streaming_ipc_memory_select_close_binding_locked(
    iree_hal_streaming_ipc_memory_import_t* import,
    iree_hal_streaming_context_t* preferred_context) {
  iree_hal_streaming_ipc_memory_binding_t* fallback = NULL;
  for (iree_hal_streaming_ipc_memory_binding_t* binding = import->bindings;
       binding; binding = binding->next) {
    if (binding->open_count == 0) continue;
    if (!fallback) fallback = binding;
    if (binding->context == preferred_context) return binding;
  }
  return fallback;
}

// Waits for already-submitted work in every context owning |bindings|. The
// import remains in transition so registry mutations cannot add or release a
// wrapper during the unlocked walk. Ordinary pointer-table lookup is outside
// this gate; callers must not use an IPC pointer concurrently with close.
static iree_status_t iree_hal_streaming_ipc_memory_synchronize_bindings(
    iree_hal_streaming_ipc_memory_binding_t* bindings) {
  iree_status_t status = iree_ok_status();
  for (iree_hal_streaming_ipc_memory_binding_t* binding = bindings; binding;
       binding = binding->next) {
    status = iree_hal_streaming_context_synchronize(binding->context);
    if (!iree_status_is_ok(status)) break;
  }
  return status;
}

static void iree_hal_streaming_ipc_memory_release_binding(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_ipc_memory_binding_t* binding) {
  IREE_ASSERT(binding->open_count == 0);
  binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED;
  iree_hal_streaming_memory_release_wrapped_buffer(binding->buffer);
  iree_hal_streaming_context_release(binding->context);
  iree_allocator_free(registry->host_allocator, binding);
}

// Finishes a final detach while the DETACHING entry remains linked. No caller
// may hold |registry->mutex|: releasing the anchor invokes ROCr.
static void iree_hal_streaming_ipc_memory_complete_detach(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_ipc_memory_import_t* import) {
  iree_hal_buffer_release(import->anchor_buffer);
  iree_hal_device_release(import->anchor_device);

  iree_slim_mutex_lock(&registry->mutex);
  IREE_ASSERT(import->state ==
              IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
  IREE_ASSERT(!import->bindings);
  iree_hal_streaming_ipc_memory_unlink_import_locked(registry, import);
  import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED;
  iree_slim_mutex_unlock(&registry->mutex);
  iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
  iree_allocator_free(registry->host_allocator, import);
}

static iree_status_t iree_hal_streaming_ipc_memory_import_reserved(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size,
    iree_hal_streaming_ipc_memory_attach_fn_t attach_fn,
    iree_hal_streaming_ipc_memory_alias_fn_t alias_fn, void** out_device_ptr) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(attach_fn);
  IREE_ASSERT_ARGUMENT(alias_fn);
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = NULL;
  if (IREE_UNLIKELY(!attach_fn || !alias_fn)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IPC memory attach and alias callbacks are required");
  }
  if (IREE_UNLIKELY(descriptor->allocation_size == 0 || view_size == 0 ||
                    descriptor->byte_offset > descriptor->allocation_size ||
                    view_size > descriptor->allocation_size -
                                    descriptor->byte_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IPC memory descriptor has an invalid view range");
  }

  for (;;) {
    iree_slim_mutex_lock(&registry->mutex);
    bool source_is_transitioning = false;
    iree_hal_streaming_ipc_memory_binding_t* source_binding =
        iree_hal_streaming_ipc_memory_find_source_binding_locked(
            registry, descriptor, context->device_ordinal,
            &source_is_transitioning);
    if (source_binding && IREE_UNLIKELY(source_binding->context != context)) {
      iree_slim_mutex_unlock(&registry->mutex);
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "IPC memory from this exporter device is already open in another "
          "context for the importing device");
    }
    if (source_is_transitioning) {
      const iree_wait_token_t wait_token =
          iree_notification_prepare_wait(&registry->changed);
      iree_slim_mutex_unlock(&registry->mutex);
      (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                          IREE_DURATION_ZERO,
                                          IREE_TIME_INFINITE_FUTURE);
      continue;
    }

    iree_hal_streaming_ipc_memory_import_t* import = registry->imports;
    while (import && !iree_hal_streaming_ipc_memory_key_matches(
                         import, descriptor->token, descriptor->byte_offset)) {
      import = import->next;
    }
    if (import) {
      if (import->state != IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
        const iree_wait_token_t wait_token =
            iree_notification_prepare_wait(&registry->changed);
        iree_slim_mutex_unlock(&registry->mutex);
        (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                            IREE_DURATION_ZERO,
                                            IREE_TIME_INFINITE_FUTURE);
        continue;
      }
      if (IREE_UNLIKELY(import->allocation_size !=
                        descriptor->allocation_size)) {
        iree_slim_mutex_unlock(&registry->mutex);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate IPC memory handle has inconsistent allocation size");
      }
      if (IREE_UNLIKELY(import->view_size != view_size)) {
        iree_slim_mutex_unlock(&registry->mutex);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate IPC memory handle has inconsistent view size");
      }
      if (IREE_UNLIKELY(import->exporter_process_id !=
                            descriptor->exporter_process_id ||
                        import->exporter_device_ordinal !=
                            descriptor->exporter_device_ordinal)) {
        iree_slim_mutex_unlock(&registry->mutex);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate IPC memory handle has inconsistent exporter identity");
      }
      if (IREE_UNLIKELY(import->open_count == UINT32_MAX)) {
        iree_slim_mutex_unlock(&registry->mutex);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "IPC memory open count overflow");
      }

      iree_hal_streaming_ipc_memory_binding_t* binding = import->bindings;
      while (binding &&
             binding->context->device_ordinal != context->device_ordinal) {
        binding = binding->next;
      }
      if (binding) {
        if (binding->state !=
            IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
          const iree_wait_token_t wait_token =
              iree_notification_prepare_wait(&registry->changed);
          iree_slim_mutex_unlock(&registry->mutex);
          (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                              IREE_DURATION_ZERO,
                                              IREE_TIME_INFINITE_FUTURE);
          continue;
        }
        if (IREE_UNLIKELY(binding->context != context)) {
          iree_slim_mutex_unlock(&registry->mutex);
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "IPC memory handle is already open in another context for "
              "this device");
        }
        ++import->open_count;
        ++binding->open_count;
        *out_device_ptr = binding->device_ptr;
        iree_slim_mutex_unlock(&registry->mutex);
        return iree_ok_status();
      }

      iree_status_t status = iree_allocator_malloc(
          registry->host_allocator, sizeof(*binding), (void**)&binding);
      if (!iree_status_is_ok(status)) {
        iree_slim_mutex_unlock(&registry->mutex);
        return status;
      }
      memset(binding, 0, sizeof(*binding));
      binding->context = context;
      iree_hal_streaming_context_retain(context);
      binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING;
      binding->next = import->bindings;
      import->bindings = binding;
      import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING;
      iree_slim_mutex_unlock(&registry->mutex);

      // Create context-local placement and pointer-table state while the
      // ATTACHING entry keeps matching opens and closes serialized. Existing
      // wrappers remain usable by ordinary work throughout alias construction.
      iree_hal_buffer_t* hal_buffer = NULL;
      status = alias_fn(context, import->anchor_buffer, &hal_buffer);
      status = iree_hal_streaming_ipc_memory_normalize_import_failure(status);
      iree_hal_streaming_buffer_t* buffer = NULL;
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_memory_wrap_buffer(
            context, hal_buffer, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
            &buffer);
        status = iree_hal_streaming_ipc_memory_normalize_import_failure(status);
      }
      if (iree_status_is_ok(status) &&
          IREE_UNLIKELY((void*)(uintptr_t)buffer->device_ptr !=
                        import->device_ptr)) {
        iree_hal_streaming_memory_release_wrapped_buffer(buffer);
        buffer = NULL;
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "IPC memory alias did not preserve the process-wide address");
      }
      iree_hal_buffer_release(hal_buffer);
      iree_slim_mutex_lock(&registry->mutex);
      IREE_ASSERT(import->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING);
      IREE_ASSERT(binding->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING);
      if (iree_status_is_ok(status)) {
        binding->buffer = buffer;
        binding->device_ptr = (void*)(uintptr_t)buffer->device_ptr;
        binding->open_count = 1;
        binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
        ++import->open_count;
        import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
        *out_device_ptr = binding->device_ptr;
        iree_slim_mutex_unlock(&registry->mutex);
        iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
        return status;
      }

      iree_hal_streaming_ipc_memory_unlink_binding_locked(import, binding);
      binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED;
      import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      iree_slim_mutex_unlock(&registry->mutex);

      iree_hal_streaming_context_release(binding->context);
      iree_allocator_free(registry->host_allocator, binding);
      iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
      return status;
    }

    // Reserve the token before invoking ROCr. The binding retains its context
    // while the anchor retains only the generic HAL device required by the
    // process-wide attachment, keeping context reset able to retire the exact
    // importing context independently of global context-list teardown.
    iree_status_t status = iree_allocator_malloc(
        registry->host_allocator, sizeof(*import), (void**)&import);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&registry->mutex);
      return status;
    }
    memset(import, 0, sizeof(*import));
    iree_hal_streaming_ipc_memory_binding_t* binding = NULL;
    status = iree_allocator_malloc(registry->host_allocator, sizeof(*binding),
                                   (void**)&binding);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(registry->host_allocator, import);
      iree_slim_mutex_unlock(&registry->mutex);
      return status;
    }
    memset(binding, 0, sizeof(*binding));
    binding->context = context;
    iree_hal_streaming_context_retain(context);
    binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING;
    import->anchor_device = context->device;
    iree_hal_device_retain(import->anchor_device);
    import->bindings = binding;
    memcpy(import->token, descriptor->token, sizeof(import->token));
    import->allocation_size = descriptor->allocation_size;
    import->byte_offset = descriptor->byte_offset;
    import->view_size = view_size;
    import->exporter_process_id = descriptor->exporter_process_id;
    import->exporter_device_ordinal = descriptor->exporter_device_ordinal;
    import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING;
    import->next = registry->imports;
    registry->imports = import;
    iree_slim_mutex_unlock(&registry->mutex);

    // Keep the entry visible in ATTACHING while ROCr and wrapper construction
    // run without the process-wide lock. Matching opens wait instead of issuing
    // a second live attach for the same token.
    iree_hal_buffer_t* hal_buffer = NULL;
    status = attach_fn(context, descriptor, view_size, &hal_buffer);
    status = iree_hal_streaming_ipc_memory_normalize_import_failure(status);

    iree_hal_streaming_buffer_t* buffer = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_memory_wrap_buffer(
          context, hal_buffer, IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED,
          &buffer);
      status = iree_hal_streaming_ipc_memory_normalize_import_failure(status);
    }
    // If ROCr attached successfully but wrapper construction failed, finish
    // detaching while the ATTACHING entry is still visible to matching opens.
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_release(hal_buffer);
      hal_buffer = NULL;
    }

    iree_slim_mutex_lock(&registry->mutex);
    IREE_ASSERT(import->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING);
    IREE_ASSERT(binding->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_ATTACHING);
    if (iree_status_is_ok(status)) {
      import->anchor_buffer = hal_buffer;
      import->device_ptr = (void*)(uintptr_t)buffer->device_ptr;
      binding->buffer = buffer;
      binding->device_ptr = import->device_ptr;
      binding->open_count = 1;
      binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      import->open_count = 1;
      import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      *out_device_ptr = binding->device_ptr;
      iree_slim_mutex_unlock(&registry->mutex);
      iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
      return status;
    }

    iree_hal_streaming_ipc_memory_unlink_import_locked(registry, import);
    binding->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED;
    import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_UNMANAGED;
    iree_slim_mutex_unlock(&registry->mutex);
    iree_notification_post(&registry->changed, IREE_ALL_WAITERS);

    iree_hal_streaming_context_release(binding->context);
    iree_hal_device_release(import->anchor_device);
    iree_allocator_free(registry->host_allocator, binding);
    iree_allocator_free(registry->host_allocator, import);
    return status;
  }
}

iree_status_t iree_hal_streaming_ipc_memory_import(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_ipc_memory_descriptor_t* descriptor,
    iree_device_size_t view_size,
    iree_hal_streaming_ipc_memory_attach_fn_t attach_fn,
    iree_hal_streaming_ipc_memory_alias_fn_t alias_fn, void** out_device_ptr) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(descriptor);
  IREE_ASSERT_ARGUMENT(attach_fn);
  IREE_ASSERT_ARGUMENT(alias_fn);
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = NULL;

  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_context_try_begin_ipc_import(context));
  iree_status_t status = iree_hal_streaming_ipc_memory_import_reserved(
      registry, context, descriptor, view_size, attach_fn, alias_fn,
      out_device_ptr);
  iree_hal_streaming_context_end_ipc_import(context);
  return status;
}

iree_status_t iree_hal_streaming_ipc_memory_close(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context, void* device_ptr) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(context);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IPC close requires an imported device pointer");
  }

  for (;;) {
    iree_slim_mutex_lock(&registry->mutex);
    iree_hal_streaming_ipc_memory_import_t* import = registry->imports;
    while (import && import->device_ptr != device_ptr) import = import->next;
    if (IREE_UNLIKELY(!import)) {
      iree_slim_mutex_unlock(&registry->mutex);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pointer is not a live IPC memory import");
    }
    if (import->state != IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
      const iree_wait_token_t wait_token =
          iree_notification_prepare_wait(&registry->changed);
      iree_slim_mutex_unlock(&registry->mutex);
      (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                          IREE_DURATION_ZERO,
                                          IREE_TIME_INFINITE_FUTURE);
      continue;
    }
    IREE_ASSERT(import->open_count > 0);
    iree_hal_streaming_ipc_memory_binding_t* closed_binding =
        iree_hal_streaming_ipc_memory_select_close_binding_locked(import,
                                                                  context);
    IREE_ASSERT(closed_binding);
    import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING;
    iree_hal_streaming_ipc_memory_binding_t* bindings = import->bindings;
    iree_slim_mutex_unlock(&registry->mutex);

    // Wait for work submitted before this close in every binding context before
    // changing ownership. The HIP close contract requires the caller to exclude
    // concurrent pointer use and submission; the registry transition serializes
    // only open, close, and context-retirement mutations.
    iree_status_t status =
        iree_hal_streaming_ipc_memory_synchronize_bindings(bindings);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&registry->mutex);
      IREE_ASSERT(import->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
      import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      iree_slim_mutex_unlock(&registry->mutex);
      iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
      return status;
    }

    iree_slim_mutex_lock(&registry->mutex);
    IREE_ASSERT(import->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
    IREE_ASSERT(import->open_count > 0);
    IREE_ASSERT(closed_binding->open_count > 0);
    --closed_binding->open_count;
    --import->open_count;
    if (import->open_count > 0) {
      import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      iree_slim_mutex_unlock(&registry->mutex);
      iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
      return iree_ok_status();
    }
    import->bindings = NULL;
    iree_slim_mutex_unlock(&registry->mutex);

    // Keep every context-local wrapper until the process-wide reference count
    // reaches zero, then remove all aliases before detaching.
    while (bindings) {
      iree_hal_streaming_ipc_memory_binding_t* binding = bindings;
      bindings = binding->next;
      binding->next = NULL;
      iree_hal_streaming_ipc_memory_release_binding(registry, binding);
    }
    iree_hal_streaming_ipc_memory_complete_detach(registry, import);
    return iree_ok_status();
  }
}

iree_status_t iree_hal_streaming_ipc_memory_release_context(
    iree_hal_streaming_ipc_memory_registry_t* registry,
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(context);

  // Close admission before scanning the registry. Imports already past the
  // gate finish or roll back before the scan, and later attempts cannot add a
  // binding behind it.
  iree_hal_streaming_context_begin_ipc_import_retirement(context);

  // Wait for a stable registry snapshot and count every binding that this
  // context retirement must remove. Admission is closed and drained, so this
  // count can only decrease before reservation if a concurrent close wins.
  iree_host_size_t release_capacity = 0;
  for (;;) {
    iree_slim_mutex_lock(&registry->mutex);
    bool is_transitioning = false;
    release_capacity = 0;
    for (iree_hal_streaming_ipc_memory_import_t* candidate = registry->imports;
         candidate; candidate = candidate->next) {
      // A final close unlinks its bindings before releasing their wrappers and
      // the backend attachment. Wait for every in-flight registry transition
      // before declaring this context clean so reset cannot publish free-memory
      // accounting while a concurrent detach is still running.
      if (candidate->state !=
          IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
        is_transitioning = true;
        continue;
      }
      bool has_context_binding = false;
      for (iree_hal_streaming_ipc_memory_binding_t* binding =
               candidate->bindings;
           binding; binding = binding->next) {
        if (binding->state !=
            IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
          is_transitioning = true;
          break;
        }
        if (binding->context == context) {
          IREE_ASSERT(!has_context_binding);
          has_context_binding = true;
          ++release_capacity;
        }
      }
    }
    if (!is_transitioning) {
      iree_slim_mutex_unlock(&registry->mutex);
      break;
    }
    const iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&registry->changed);
    iree_slim_mutex_unlock(&registry->mutex);
    (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
  }
  if (release_capacity == 0) {
    iree_hal_streaming_context_commit_ipc_import_retirement(context);
    return iree_ok_status();
  }

  iree_host_size_t release_entries_size = 0;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          release_capacity,
          sizeof(iree_hal_streaming_ipc_memory_context_release_entry_t),
          &release_entries_size))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "IPC context release plan size overflow");
  }
  iree_hal_streaming_ipc_memory_context_release_entry_t* release_entries = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(registry->host_allocator, release_entries_size,
                              (void**)&release_entries);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_context_cancel_ipc_import_retirement(context);
    return status;
  }

  // Reserve every affected registry entry in one locked transaction. If a
  // close changed the snapshot while storage was allocated, wait for it and
  // rescan; closed admission guarantees the original capacity remains enough.
  iree_host_size_t release_count = 0;
  for (;;) {
    iree_slim_mutex_lock(&registry->mutex);
    bool is_transitioning = false;
    release_count = 0;
    for (iree_hal_streaming_ipc_memory_import_t* import = registry->imports;
         import; import = import->next) {
      if (import->state != IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
        is_transitioning = true;
        continue;
      }
      bool has_context_binding = false;
      for (iree_hal_streaming_ipc_memory_binding_t* binding = import->bindings;
           binding; binding = binding->next) {
        if (binding->state !=
            IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY) {
          is_transitioning = true;
          break;
        }
        if (binding->context != context) continue;
        IREE_ASSERT(!has_context_binding);
        has_context_binding = true;
        IREE_ASSERT_LT(release_count, release_capacity);
        release_entries[release_count++] =
            (iree_hal_streaming_ipc_memory_context_release_entry_t){
                .import = import,
                .released_binding = binding,
                .remaining_bindings = NULL,
                .is_final_detach = false,
            };
      }
    }
    if (is_transitioning) {
      const iree_wait_token_t wait_token =
          iree_notification_prepare_wait(&registry->changed);
      iree_slim_mutex_unlock(&registry->mutex);
      (void)iree_notification_commit_wait(&registry->changed, wait_token,
                                          IREE_DURATION_ZERO,
                                          IREE_TIME_INFINITE_FUTURE);
      continue;
    }
    for (iree_host_size_t i = 0; i < release_count; ++i) {
      release_entries[i].import->state =
          IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING;
      release_entries[i].released_binding->state =
          IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING;
    }
    iree_slim_mutex_unlock(&registry->mutex);
    break;
  }

  // Synchronization is the only fallible phase after reservation. No registry
  // counts, bindings, wrappers, or attachments change unless already-submitted
  // work has completed in every affected import's binding contexts.
  for (iree_host_size_t i = 0; i < release_count; ++i) {
    status = iree_hal_streaming_ipc_memory_synchronize_bindings(
        release_entries[i].import->bindings);
    if (!iree_status_is_ok(status)) break;
  }
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&registry->mutex);
    for (iree_host_size_t i = 0; i < release_count; ++i) {
      IREE_ASSERT(release_entries[i].import->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
      IREE_ASSERT(release_entries[i].released_binding->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
      release_entries[i].released_binding->state =
          IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      release_entries[i].import->state =
          IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
    }
    iree_slim_mutex_unlock(&registry->mutex);
    iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
    iree_allocator_free(registry->host_allocator, release_entries);
    iree_hal_streaming_context_cancel_ipc_import_retirement(context);
    return status;
  }

  // Commit all registry mutations together now that no operation can fail.
  iree_slim_mutex_lock(&registry->mutex);
  for (iree_host_size_t i = 0; i < release_count; ++i) {
    iree_hal_streaming_ipc_memory_context_release_entry_t* entry =
        &release_entries[i];
    IREE_ASSERT(entry->import->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
    IREE_ASSERT(entry->released_binding->state ==
                IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
    IREE_ASSERT(entry->import->open_count >=
                entry->released_binding->open_count);
    entry->import->open_count -= entry->released_binding->open_count;
    entry->released_binding->open_count = 0;
    iree_hal_streaming_ipc_memory_unlink_binding_locked(
        entry->import, entry->released_binding);
    entry->is_final_detach = entry->import->open_count == 0;
    if (entry->is_final_detach) {
      entry->remaining_bindings = entry->import->bindings;
      entry->import->bindings = NULL;
    }
  }
  iree_slim_mutex_unlock(&registry->mutex);

  // Retire wrappers and backend attachments outside the registry mutex. Each
  // non-final entry stays DETACHING until its removed wrapper is gone.
  for (iree_host_size_t i = 0; i < release_count; ++i) {
    iree_hal_streaming_ipc_memory_context_release_entry_t* entry =
        &release_entries[i];
    iree_hal_streaming_ipc_memory_release_binding(registry,
                                                  entry->released_binding);
    if (entry->is_final_detach) {
      while (entry->remaining_bindings) {
        iree_hal_streaming_ipc_memory_binding_t* binding =
            entry->remaining_bindings;
        entry->remaining_bindings = binding->next;
        binding->next = NULL;
        iree_hal_streaming_ipc_memory_release_binding(registry, binding);
      }
      iree_hal_streaming_ipc_memory_complete_detach(registry, entry->import);
    } else {
      iree_slim_mutex_lock(&registry->mutex);
      IREE_ASSERT(entry->import->state ==
                  IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_DETACHING);
      entry->import->state = IREE_HAL_STREAMING_IPC_MEMORY_IMPORT_STATE_READY;
      iree_slim_mutex_unlock(&registry->mutex);
      iree_notification_post(&registry->changed, IREE_ALL_WAITERS);
    }
  }
  iree_allocator_free(registry->host_allocator, release_entries);
  iree_hal_streaming_context_commit_ipc_import_retirement(context);
  return iree_ok_status();
}
