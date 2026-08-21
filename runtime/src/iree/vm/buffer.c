// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/buffer.h"

#include <stddef.h>
#include <string.h>

#include "iree/vm/buffer_provider.h"

// Private representation bits sharing the public access word.
enum iree_vm_buffer_flag_bits_e {
  // The buffer is a proper view retaining a flattened root.
  IREE_VM_BUFFER_FLAG_VIEW = 1u << 2,
  // The root wraps storage released by a callback.
  IREE_VM_BUFFER_FLAG_EXTERNAL = 1u << 3,
};
typedef uint32_t iree_vm_buffer_flags_t;

#define IREE_VM_BUFFER_ACCESS_MASK \
  (IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE)

// Uniform buffer object. The fields through |root| form the access-hot prefix;
// the allocator and release callback are touched only during construction and
// final destruction.
struct iree_vm_buffer_t {
  // Required offset-zero VM ref-count prefix.
  iree_vm_ref_object_t ref_object;
  // Public access bits and private representation bits.
  iree_vm_buffer_flags_t flags;
  // Immutable logical byte length.
  iree_host_size_t length;
  // Direct root or cached-view byte start.
  uint8_t* data;
  // Retained flattened root, or null when this object is a root.
  iree_vm_buffer_t* root;
  // Allocator owning this complete buffer object.
  iree_allocator_t host_allocator;
  // External root storage release callback, or null for heaps and views.
  iree_vm_buffer_release_callback_t release_callback;
};

#define IREE_VM_BUFFER_HOT_SIZE \
  (8 + sizeof(iree_host_size_t) + 2 * sizeof(void*))

static_assert(offsetof(iree_vm_buffer_t, ref_object) == 0,
              "VM buffers require an offset-zero ref object");
static_assert(offsetof(iree_vm_buffer_t, flags) == 4,
              "VM buffer flags must follow the ref object");
static_assert(offsetof(iree_vm_buffer_t, length) == 8,
              "VM buffer length must begin at byte eight");
static_assert(offsetof(iree_vm_buffer_t, host_allocator) ==
                  IREE_VM_BUFFER_HOT_SIZE,
              "VM buffer hot prefix layout changed");
static_assert(sizeof(void*) != 8 || IREE_VM_BUFFER_HOT_SIZE == 32,
              "64-bit VM buffer hot prefix must remain 32 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_buffer_t) == 64,
              "64-bit VM buffer object must remain 64 bytes");

static void iree_vm_buffer_destroy(void* object);

static const iree_vm_ref_type_table_t iree_vm_buffer_type_table_;
static const iree_vm_ref_type_descriptor_t iree_vm_buffer_type_ = {
    iree_vm_buffer_destroy,
    &iree_vm_buffer_type_table_,
    {"buffer", 6},
};
static const iree_vm_ref_types_t iree_vm_buffer_types_ = {
    &iree_vm_buffer_type_,
};
static const iree_vm_ref_type_table_t iree_vm_buffer_type_table_ = {
    sizeof(iree_vm_buffer_type_table_),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    {"vm", 2},
    {&iree_vm_buffer_types_, IREE_VM_REF_TYPE_COUNT},
};

const iree_vm_ref_type_table_t* iree_vm_buffer_provider_table(void) {
  return &iree_vm_buffer_type_table_;
}

static inline iree_vm_buffer_access_flags_t iree_vm_buffer_local_access(
    const iree_vm_buffer_t* buffer) {
  return buffer->flags & IREE_VM_BUFFER_ACCESS_MASK;
}

static inline iree_vm_buffer_access_flags_t iree_vm_buffer_effective_access(
    const iree_vm_buffer_t* buffer) {
  const iree_vm_buffer_access_flags_t access =
      iree_vm_buffer_local_access(buffer);
  if (access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) return access;
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return IREE_VM_BUFFER_ACCESS_FLAG_NONE;
  }
  return access;
}

static iree_status_t iree_vm_buffer_validate_access(
    iree_vm_buffer_access_flags_t access) {
  if (access == IREE_VM_BUFFER_ACCESS_FLAG_NONE ||
      (access & ~IREE_VM_BUFFER_ACCESS_MASK) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "buffer access must contain only READ and/or WRITE bits");
  }
  return iree_ok_status();
}

static void iree_vm_buffer_initialize(
    iree_vm_buffer_access_flags_t access, iree_vm_buffer_flags_t private_flags,
    iree_byte_span_t storage, iree_vm_buffer_t* root,
    iree_vm_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_vm_buffer_t* out_buffer) {
  iree_vm_ref_object_initialize(&out_buffer->ref_object);
  out_buffer->flags = access | private_flags;
  out_buffer->length = storage.data_length;
  out_buffer->data = storage.data;
  out_buffer->root = root;
  out_buffer->host_allocator = host_allocator;
  out_buffer->release_callback = release_callback;
}

static iree_status_t iree_vm_buffer_allocate_storage(
    iree_vm_buffer_access_flags_t access, iree_host_size_t length,
    iree_host_size_t minimum_alignment, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_buffer) {
  iree_host_size_t total_size = 0;
  iree_host_size_t data_offset = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_vm_buffer_t), &total_size,
                         IREE_STRUCT_FIELD(length, uint8_t, &data_offset)));

  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      host_allocator, total_size, minimum_alignment, data_offset,
      (void**)&buffer));

  iree_vm_buffer_initialize(
      access, /*private_flags=*/0,
      iree_make_byte_span((uint8_t*)buffer + data_offset, length), NULL,
      iree_vm_buffer_release_callback_null(), host_allocator, buffer);
  *out_buffer = buffer;
  return iree_ok_status();
}

static iree_status_t iree_vm_buffer_map_range(
    const iree_vm_buffer_t* buffer,
    iree_vm_buffer_access_flags_t required_access, iree_host_size_t offset,
    iree_host_size_t length, iree_byte_span_t* out_span) {
  const iree_vm_buffer_access_flags_t local_access =
      iree_vm_buffer_local_access(buffer);
  if ((local_access & required_access) != required_access) {
    if (local_access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "buffer is closed");
    }
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer does not permit the requested access");
  }
  if (offset > buffer->length || length > buffer->length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range offset %" PRIhsz " length %" PRIhsz
                            " exceeds buffer length %" PRIhsz,
                            offset, length, buffer->length);
  }
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer root is closed");
  }

  *out_span = length == 0 ? iree_byte_span_empty()
                          : iree_make_byte_span(buffer->data + offset, length);
  return iree_ok_status();
}

static void iree_vm_buffer_destroy(void* object) {
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)object;
  const iree_vm_buffer_flags_t flags = buffer->flags;
  iree_allocator_t host_allocator = buffer->host_allocator;

  if (flags & IREE_VM_BUFFER_FLAG_VIEW) {
    iree_vm_buffer_t* root = buffer->root;
    iree_allocator_free(host_allocator, buffer);
    iree_vm_buffer_release(root);
  } else if (flags & IREE_VM_BUFFER_FLAG_EXTERNAL) {
    iree_vm_buffer_release_callback_t release_callback =
        buffer->release_callback;
    iree_byte_span_t storage =
        iree_make_byte_span(buffer->data, buffer->length);
    iree_allocator_free(host_allocator, buffer);
    if (release_callback.fn) {
      release_callback.fn(release_callback.user_data, storage);
    }
  } else {
    iree_allocator_free_aligned(host_allocator, buffer);
  }
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_create(
    iree_host_size_t length, iree_host_size_t minimum_alignment,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer) {
  if (!out_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_buffer is required");
  }
  *out_buffer = NULL;
  return iree_vm_buffer_allocate_storage(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      length, minimum_alignment, host_allocator, out_buffer);
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_clone(
    iree_vm_buffer_access_flags_t access, iree_const_byte_span_t source_bytes,
    iree_host_size_t minimum_alignment, iree_allocator_t host_allocator,
    iree_vm_buffer_t** out_buffer) {
  if (!out_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_buffer is required");
  }
  *out_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_validate_access(access));
  if (source_bytes.data_length != 0 && !source_bytes.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "nonempty source bytes require a data pointer");
  }

  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_allocate_storage(
      access, source_bytes.data_length, minimum_alignment, host_allocator,
      &buffer));
  if (source_bytes.data_length != 0) {
    memcpy(buffer->data, source_bytes.data, source_bytes.data_length);
  }
  *out_buffer = buffer;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_wrap(
    iree_vm_buffer_access_flags_t access, iree_byte_span_t storage,
    iree_vm_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer) {
  if (!out_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_buffer is required");
  }
  *out_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_validate_access(access));
  if (storage.data_length != 0 && !storage.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "nonempty storage requires a data pointer");
  }

  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_vm_buffer_initialize(access, IREE_VM_BUFFER_FLAG_EXTERNAL, storage, NULL,
                            release_callback, host_allocator, buffer);
  *out_buffer = buffer;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_subspan(
    iree_vm_buffer_t* source_buffer, iree_host_size_t source_offset,
    iree_host_size_t length, iree_vm_buffer_access_flags_t access,
    iree_allocator_t host_allocator, iree_vm_buffer_t** out_buffer) {
  if (!source_buffer || !out_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source_buffer and out_buffer are required");
  }
  *out_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_validate_access(access));

  const iree_vm_buffer_access_flags_t source_access =
      iree_vm_buffer_effective_access(source_buffer);
  if (source_access == IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source buffer is closed");
  }
  if ((access & source_access) != access) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "subspan cannot widen source buffer access");
  }
  if (source_offset > source_buffer->length ||
      length > source_buffer->length - source_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "subspan offset %" PRIhsz " length %" PRIhsz
                            " exceeds source length %" PRIhsz,
                            source_offset, length, source_buffer->length);
  }

  if (source_offset == 0 && length == source_buffer->length &&
      access == source_access) {
    iree_vm_buffer_retain(source_buffer);
    *out_buffer = source_buffer;
    return iree_ok_status();
  }

  iree_vm_buffer_t* view = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*view), (void**)&view));
  iree_vm_buffer_t* root =
      source_buffer->root ? source_buffer->root : source_buffer;
  iree_vm_buffer_retain(root);
  uint8_t* data =
      source_buffer->data ? source_buffer->data + source_offset : NULL;
  iree_vm_buffer_initialize(
      access, IREE_VM_BUFFER_FLAG_VIEW, iree_make_byte_span(data, length), root,
      iree_vm_buffer_release_callback_null(), host_allocator, view);
  *out_buffer = view;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vm_buffer_retain(iree_vm_buffer_t* buffer) {
  iree_vm_ref_object_retain(buffer);
}

IREE_API_EXPORT void iree_vm_buffer_release(iree_vm_buffer_t* buffer) {
  iree_vm_ref_object_release(buffer, &iree_vm_buffer_type_);
}

IREE_API_EXPORT iree_vm_buffer_access_flags_t
iree_vm_buffer_access(const iree_vm_buffer_t* buffer) {
  return iree_vm_buffer_effective_access(buffer);
}

IREE_API_EXPORT iree_host_size_t
iree_vm_buffer_length(const iree_vm_buffer_t* buffer) {
  return buffer->length;
}

IREE_API_EXPORT iree_status_t iree_vm_buffer_map_read(
    const iree_vm_buffer_t* buffer, iree_host_size_t offset,
    iree_host_size_t length, iree_const_byte_span_t* out_span) {
  if (!buffer || !out_span) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer and out_span are required");
  }
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_range(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_READ, offset, length, &span));
  *out_span = iree_const_cast_byte_span(span);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_buffer_map_write(iree_vm_buffer_t* buffer, iree_host_size_t offset,
                         iree_host_size_t length, iree_byte_span_t* out_span) {
  if (!buffer || !out_span) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer and out_span are required");
  }
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_range(
      buffer, IREE_VM_BUFFER_ACCESS_FLAG_WRITE, offset, length, &span));
  *out_span = span;
  return iree_ok_status();
}

IREE_API_EXPORT void* iree_vm_buffer_data(iree_vm_buffer_t* buffer) {
  if (!(buffer->flags & IREE_VM_BUFFER_ACCESS_FLAG_WRITE)) {
    return NULL;
  }
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return NULL;
  }
  return buffer->data;
}

IREE_API_EXPORT const void* iree_vm_buffer_const_data(
    const iree_vm_buffer_t* buffer) {
  if (!(buffer->flags & IREE_VM_BUFFER_ACCESS_FLAG_READ)) {
    return NULL;
  }
  if ((buffer->flags & IREE_VM_BUFFER_FLAG_VIEW) &&
      iree_vm_buffer_local_access(buffer->root) ==
          IREE_VM_BUFFER_ACCESS_FLAG_NONE) {
    return NULL;
  }
  return buffer->data;
}

#undef IREE_VM_BUFFER_HOT_SIZE
#undef IREE_VM_BUFFER_ACCESS_MASK
