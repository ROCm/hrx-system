// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/environment.h"

#include "iree/base/threading/mutex.h"
#include "iree/vm/buffer.h"

// Returns the process-static provider table for the core "vm" family.
// This is a private seam consumed only by environment construction.
const iree_vm_ref_type_table_t* iree_vm_buffer_provider_table(void);

// Maximum provider families registered in one version-zero environment,
// including the automatically registered core "vm" family.
#define IREE_VM_ENVIRONMENT_TABLE_CAPACITY 16

struct iree_vm_environment_t {
  // Allocator owning this complete environment allocation.
  iree_allocator_t host_allocator;
  // Guards |table_count| and |tables|.
  iree_slim_mutex_t mutex;
  // Number of densely registered provider tables.
  iree_host_size_t table_count;
  // Borrowed provider tables in registration order.
  const iree_vm_ref_type_table_t* tables[IREE_VM_ENVIRONMENT_TABLE_CAPACITY];
};

static iree_status_t iree_vm_environment_validate_ref_type_table(
    const iree_vm_ref_type_table_t* table) {
  if (!table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type table is required");
  }
  if (table->structure_size < IREE_VM_REF_TYPE_TABLE_V0_REQUIRED_SIZE) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "ref-type table structure has %" PRIu32
                            " bytes but version zero requires at least %zu",
                            table->structure_size,
                            (size_t)IREE_VM_REF_TYPE_TABLE_V0_REQUIRED_SIZE);
  }
  if (table->flags != IREE_VM_REF_TYPE_TABLE_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "ref-type table flags 0x%08" PRIx32
                            " are not executable",
                            table->flags);
  }
  if (!table->namespace_name.data || table->namespace_name.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type namespace must be nonempty");
  }
  if (table->types.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type table must contain at least one type");
  }
  if (!table->types.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type storage is required");
  }
  if (!iree_host_ptr_has_alignment(table->types.data,
                                   iree_alignof(iree_vm_ref_type_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type storage is not naturally aligned");
  }
  if (table->types.count > IREE_HOST_SIZE_MAX / sizeof(iree_vm_ref_type_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref-type storage byte extent overflows");
  }

  for (iree_host_size_t i = 0; i < table->types.count; ++i) {
    iree_vm_ref_type_t type = iree_vm_ref_type_storage_at(table->types, i);
    if (!type || !iree_host_ptr_has_alignment(
                     type, iree_alignof(iree_vm_ref_type_descriptor_t))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ref type %" PRIhsz " has a null or misaligned descriptor", i);
    }
    if (type->table != table) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ref type %" PRIhsz " does not point back to its provider table", i);
    }
    if (!type->type_name.data || type->type_name.size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ref type %" PRIhsz " has an empty local name",
                              i);
    }
  }

  for (iree_host_size_t i = 0; i < table->types.count; ++i) {
    const iree_vm_ref_type_t lhs = iree_vm_ref_type_storage_at(table->types, i);
    for (iree_host_size_t j = i + 1; j < table->types.count; ++j) {
      const iree_vm_ref_type_t rhs =
          iree_vm_ref_type_storage_at(table->types, j);
      if (iree_string_view_equal(lhs->type_name, rhs->type_name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "ref types %" PRIhsz " and %" PRIhsz
                                " have duplicate local name '%.*s'",
                                i, j, (int)lhs->type_name.size,
                                lhs->type_name.data);
      }
    }
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_environment_allocate(
    iree_allocator_t host_allocator, iree_vm_environment_t** out_environment) {
  if (!out_environment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_environment is required");
  }
  *out_environment = NULL;

  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*environment), (void**)&environment));
  environment->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&environment->mutex);

  iree_status_t status = iree_vm_environment_register_ref_type_table(
      environment, iree_vm_buffer_provider_table());
  if (iree_status_is_ok(status)) {
    *out_environment = environment;
  } else {
    iree_slim_mutex_deinitialize(&environment->mutex);
    iree_allocator_free(host_allocator, environment);
  }
  return status;
}

IREE_API_EXPORT void iree_vm_environment_free(
    iree_vm_environment_t* environment) {
  if (!environment) return;
  iree_allocator_t host_allocator = environment->host_allocator;
  iree_slim_mutex_deinitialize(&environment->mutex);
  iree_allocator_free(host_allocator, environment);
}

IREE_API_EXPORT iree_status_t iree_vm_environment_register_ref_type_table(
    iree_vm_environment_t* environment, const iree_vm_ref_type_table_t* table) {
  if (!environment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "environment is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_environment_validate_ref_type_table(table));

  iree_slim_mutex_lock(&environment->mutex);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < environment->table_count && iree_status_is_ok(status); ++i) {
    if (iree_string_view_equal(environment->tables[i]->namespace_name,
                               table->namespace_name)) {
      status = iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "ref-type namespace '%.*s' is already registered",
          (int)table->namespace_name.size, table->namespace_name.data);
    }
  }
  if (iree_status_is_ok(status) &&
      environment->table_count == IREE_VM_ENVIRONMENT_TABLE_CAPACITY) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "ref-type environment capacity of %d provider families is exhausted",
        IREE_VM_ENVIRONMENT_TABLE_CAPACITY);
  }
  if (iree_status_is_ok(status)) {
    environment->tables[environment->table_count++] = table;
  }
  iree_slim_mutex_unlock(&environment->mutex);
  return status;
}

IREE_API_EXPORT const iree_vm_ref_type_table_t*
iree_vm_environment_lookup_ref_type_table(iree_vm_environment_t* environment,
                                          iree_string_view_t namespace_name) {
  if (!environment || !namespace_name.data || namespace_name.size == 0) {
    return NULL;
  }

  const iree_vm_ref_type_table_t* result = NULL;
  iree_slim_mutex_lock(&environment->mutex);
  for (iree_host_size_t i = 0; i < environment->table_count; ++i) {
    if (iree_string_view_equal(environment->tables[i]->namespace_name,
                               namespace_name)) {
      result = environment->tables[i];
      break;
    }
  }
  iree_slim_mutex_unlock(&environment->mutex);
  return result;
}

#undef IREE_VM_ENVIRONMENT_TABLE_CAPACITY
