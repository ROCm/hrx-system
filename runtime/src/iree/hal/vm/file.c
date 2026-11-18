// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vm/file.h"

#include "iree/hal/device.h"
#include "iree/vm/reflection.h"

typedef struct iree_hal_vm_file_module_t {
  // Native module prefix owning immutable callable declarations.
  iree_vm_module_t base;
  // Declaration borrowing this module's reference type array.
  iree_vm_module_descriptor_t descriptor;
  // Allocator for the module, completion fences, and large binding tables.
  iree_allocator_t host_allocator;
  // Canonical buffer, fence, and file identities used by callable signatures.
  iree_vm_ref_type_t types[3];
  // Retained device supplying completion semaphores.
  iree_hal_device_t* device;
  // Retained queue selected by the host for native file transfers.
  iree_hal_queue_t* queue;
} iree_hal_vm_file_module_t;

// Reference spans reach native calls without an element-type walk in the VM.
// Validate their actual cells here, before passing pointers to the native HAL.
static iree_status_t iree_hal_vm_file_bind_files(
    iree_vm_ref_type_t type, iree_vm_call_span_t objects,
    iree_vm_call_span_t windows, iree_hal_file_binding_t* bindings) {
  const iree_vm_ref_t* refs = objects.data;
  const uint8_t* bytes = windows.data;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < objects.count && iree_status_is_ok(status);
       ++i) {
    status =
        iree_vm_ptr_from_ref_borrowed(refs[i], type, (void**)&bindings[i].file);
    bindings[i].offset = iree_unaligned_load_le_u64(bytes + i * 16);
    bindings[i].length = iree_unaligned_load_le_u64(bytes + i * 16 + 8);
  }
  return status;
}

static iree_status_t iree_hal_vm_file_bind_buffers(
    iree_vm_ref_type_t type, iree_vm_call_span_t objects,
    iree_vm_call_span_t windows, iree_hal_buffer_binding_t* bindings) {
  const iree_vm_ref_t* refs = objects.data;
  const uint8_t* bytes = windows.data;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < objects.count && iree_status_is_ok(status);
       ++i) {
    const uint64_t offset = iree_unaligned_load_le_u64(bytes + i * 16);
    const uint64_t length = iree_unaligned_load_le_u64(bytes + i * 16 + 8);
    if (offset > IREE_DEVICE_SIZE_MAX ||
        (length != UINT64_MAX && length > IREE_DEVICE_SIZE_MAX)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "file buffer window exceeds device size");
    } else {
      bindings[i] = (iree_hal_buffer_binding_t){
          .offset = (iree_device_size_t)offset,
          .length = length == UINT64_MAX ? IREE_HAL_WHOLE_BUFFER
                                         : (iree_device_size_t)length,
      };
      status = iree_vm_ptr_from_ref_borrowed(refs[i], type,
                                             (void**)&bindings[i].buffer);
    }
  }
  return status;
}

static iree_status_t iree_hal_vm_file_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  iree_hal_vm_file_module_t* module = (iree_hal_vm_file_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  const iree_vm_call_span_t files = iree_vm_call_span_argument_load(call, 0);
  const iree_vm_call_span_t file_windows =
      iree_vm_call_span_argument_load(call, 2);
  const iree_vm_call_span_t buffers = iree_vm_call_span_argument_load(call, 4);
  const iree_vm_call_span_t buffer_windows =
      iree_vm_call_span_argument_load(call, 6);
  if (files.count > 65536 || buffers.count > 65536 ||
      file_windows.count != files.count * 2 ||
      buffer_windows.count != buffers.count * 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect file tables require at most 65536 entries and one "
        "offset/length pair per object");
  }
  const uint64_t records_offset = iree_vm_call_value_argument_load(call, 8);
  const uint64_t records_length = iree_vm_call_value_argument_load(call, 9);
  const uint64_t count_offset = iree_vm_call_value_argument_load(call, 10);
  if (records_offset > IREE_DEVICE_SIZE_MAX ||
      (records_length != UINT64_MAX && records_length > IREE_DEVICE_SIZE_MAX) ||
      count_offset > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file metadata window exceeds device size");
  }
  iree_vm_ref_t refs[3] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(refs); ++i) {
    iree_vm_call_ref_argument_load_borrow(call, i, &refs[i]);
  }
  if (!refs[1].object || !refs[2].object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indirect file records and count are required");
  }
  const iree_hal_file_transfer_list_t transfers = {
      .type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT,
      .indirect =
          {
              .records = {.buffer = (iree_hal_buffer_t*)refs[1].object,
                          .offset = (iree_device_size_t)records_offset,
                          .length = records_length == UINT64_MAX
                                        ? IREE_HAL_WHOLE_BUFFER
                                        : (iree_device_size_t)records_length},
              .count_buffer = (iree_hal_buffer_t*)refs[2].object,
              .count_offset = (iree_device_size_t)count_offset,
          },
  };

  // Common small batches allocate no adapter table storage. Larger tables use
  // one bounded temporary allocation, independent of the number of transfers.
  iree_hal_file_binding_t inline_files[4];
  iree_hal_buffer_binding_t inline_buffers[4];
  iree_hal_file_binding_t* file_bindings = inline_files;
  iree_hal_buffer_binding_t* buffer_bindings = inline_buffers;
  uint8_t* storage = NULL;
  if (files.count > IREE_ARRAYSIZE(inline_files) ||
      buffers.count > IREE_ARRAYSIZE(inline_buffers)) {
    iree_host_size_t allocation_size = 0;
    iree_host_size_t file_offset = 0;
    iree_host_size_t buffer_offset = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &allocation_size,
        IREE_STRUCT_FIELD(files.count, iree_hal_file_binding_t, &file_offset),
        IREE_STRUCT_FIELD(buffers.count, iree_hal_buffer_binding_t,
                          &buffer_offset)));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        module->host_allocator, allocation_size, (void**)&storage));
    file_bindings = (iree_hal_file_binding_t*)(storage + file_offset);
    buffer_bindings = (iree_hal_buffer_binding_t*)(storage + buffer_offset);
  }

  iree_status_t status = iree_hal_vm_file_bind_files(
      module->types[2], files, file_windows, file_bindings);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vm_file_bind_buffers(module->types[0], buffers,
                                           buffer_windows, buffer_bindings);
  }
  iree_hal_semaphore_t* semaphore = NULL;
  iree_hal_fence_t* completion = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        module->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_fence_create_at(semaphore, 1, module->host_allocator,
                                      &completion);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t waits =
        iree_hal_fence_semaphore_list((iree_hal_fence_t*)refs[0].object);
    const iree_hal_semaphore_list_t signals =
        iree_hal_fence_semaphore_list(completion);
    const iree_hal_file_binding_table_t file_table = {files.count,
                                                      file_bindings};
    const iree_hal_buffer_binding_table_t buffer_table = {buffers.count,
                                                          buffer_bindings};
    status = params->function_ordinal == 0
                 ? iree_hal_queue_read(module->queue, waits, signals,
                                       file_table, buffer_table, &transfers,
                                       NULL, IREE_HAL_READ_FLAG_NONE)
                 : iree_hal_queue_write(module->queue, waits, signals,
                                        buffer_table, file_table, &transfers,
                                        NULL, IREE_HAL_WRITE_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  iree_allocator_free(module->host_allocator, storage);
  if (iree_status_is_ok(status)) {
    void* owned = completion;
    iree_vm_ref_t result = iree_vm_ref_from_ptr_move(&owned, module->types[1]);
    iree_vm_call_ref_result_store_move(call, 0, &result);
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  } else {
    iree_hal_fence_release(completion);
  }
  return status;
}

static const iree_vm_module_signature_type_t arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN_READ_ONLY |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR,
     2},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN_READ_ONLY |
         (iree_vm_module_signature_type_kind_t)IREE_VM_SCALAR_TYPE_I64,
     0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN_READ_ONLY |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR,
     0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN |
         IREE_VM_MODULE_SIGNATURE_TYPE_KIND_SPAN_READ_ONLY |
         (iree_vm_module_signature_type_kind_t)IREE_VM_SCALAR_TYPE_I64,
     0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 1},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
};
static const iree_vm_module_signature_type_t result = {
    IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1};
static const iree_vm_module_callable_type_declaration_t callable = {
    .signature = {.arguments = {arguments, IREE_ARRAYSIZE(arguments), 11, 3, 0},
                  .results = {&result, 1, 0, 1, 0}},
};
static const iree_vm_module_export_declaration_t exports[] = {
    {.export_name = IREE_SVL("read_indirect"), .function_ordinal = 0},
    {.export_name = IREE_SVL("write_indirect"), .function_ordinal = 1},
};

static void iree_hal_vm_file_module_destroy(iree_vm_module_t* base) {
  iree_hal_vm_file_module_t* module = (iree_hal_vm_file_module_t*)base;
  iree_hal_queue_release(module->queue);
  iree_hal_device_release(module->device);
  iree_allocator_free(module->host_allocator, module);
}

static void iree_hal_vm_file_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = exports[ordinal];
}

static void iree_hal_vm_file_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = callable;
}

static void iree_hal_vm_file_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("file module has no imports");
}

static void iree_hal_vm_file_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("file module has no imports");
}

static const iree_vm_module_vtable_t module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = iree_hal_vm_file_module_destroy,
    .function_start = iree_hal_vm_file_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = iree_hal_vm_file_module_query_import_group,
    .query_import = iree_hal_vm_file_module_query_import,
    .query_export = iree_hal_vm_file_module_query_export,
    .query_callable_type = iree_hal_vm_file_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t iree_hal_vm_file_module_create(const iree_hal_vm_types_t* types,
                                             iree_hal_queue_t* queue,
                                             iree_allocator_t host_allocator,
                                             iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_hal_vm_file_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types[0] = types->buffer;
  module->types[1] = types->fence;
  module->types[2] = types->file;
  module->device = iree_hal_queue_family_device(iree_hal_queue_family(queue));
  iree_hal_device_retain(module->device);
  module->queue = queue;
  iree_hal_queue_retain(queue);
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("hal.file"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {module->types, IREE_ARRAYSIZE(module->types)},
      .counts = {.function_count = 2,
                 .callable_type_count = 1,
                 .export_count = IREE_ARRAYSIZE(exports),
                 .callable_fields = {.value_count = 11, .ref_count = 4}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_hal_vm_file_module_destroy(&module->base);
  }
  return status;
}
