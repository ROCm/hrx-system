// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/module.h"

#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"
#include "loom/serve/runtime/parameter/gather.h"

typedef struct loom_serve_parameters_module_t {
  // Native module prefix owning immutable callable declarations.
  iree_vm_module_t base;
  // Declaration borrowing this module's canonical reference identities.
  iree_vm_module_descriptor_t descriptor;
  // Allocator owning the capability and per-call descriptor snapshots.
  iree_allocator_t host_allocator;
  // Canonical VM buffer, HAL buffer, and HAL fence identities.
  iree_vm_ref_type_t types[3];
  // Retained physical device used by the provider to import checkpoint files.
  iree_hal_device_t* device;
  // Retained queue for descriptor downloads and deferred reads.
  iree_hal_queue_t* queue;
  // Retained immutable catalog selected at the invocation boundary.
  loom_serve_parameter_sources_t* sources;
} loom_serve_parameters_module_t;

void loom_serve_parameters_module_bind_sources(
    iree_vm_module_t* base, loom_serve_parameter_sources_t* sources) {
  loom_serve_parameters_module_t* module =
      (loom_serve_parameters_module_t*)base;
  loom_serve_parameter_sources_retain(sources);
  loom_serve_parameter_sources_release(module->sources);
  module->sources = sources;
}

static iree_status_t loom_serve_parameters_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_parameters_module_t* module =
      (loom_serve_parameters_module_t*)base;
  const iree_vm_call_packet_t* call = &params->call;
  const uint32_t source = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  const uint64_t offset = iree_vm_call_value_argument_load(call, 1);
  const uint64_t count = iree_vm_call_value_argument_load(call, 2);
  if (source >= loom_serve_parameter_sources_count(module->sources)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter source %u is not declared", source);
  }
  if (count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter gather count exceeds host capacity");
  }
  iree_vm_ref_t arguments[4] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(arguments); ++i) {
    iree_vm_call_ref_argument_load_borrow(call, i, &arguments[i]);
  }
  if (!arguments[0].object || !arguments[2].object || !arguments[3].object) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter gather requires key, descriptors, and target");
  }
  iree_vm_buffer_t* key_buffer = (iree_vm_buffer_t*)arguments[0].object;
  iree_const_byte_span_t key = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      key_buffer, 0, iree_vm_buffer_length(key_buffer), &key));
  iree_hal_fence_t* ready = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_parameter_gather(
      loom_serve_parameter_sources_provider(module->sources, source),
      module->device, module->queue,
      iree_hal_fence_semaphore_list((iree_hal_fence_t*)arguments[1].object),
      iree_make_string_view((const char*)key.data, key.data_length),
      (iree_hal_buffer_t*)arguments[2].object, offset, (iree_host_size_t)count,
      (iree_hal_buffer_t*)arguments[3].object, module->host_allocator, &ready));
  void* owned = ready;
  iree_vm_ref_t result = iree_vm_ref_from_ptr_move(&owned, module->types[2]);
  iree_vm_call_ref_result_store_move(call, 0, &result);
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t gather_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
};
static const iree_vm_module_signature_type_t gather_result = {
    IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2};
static const iree_vm_module_callable_type_declaration_t gather_callable = {
    .signature = {.arguments = {gather_arguments, 7, 3, 4, 0},
                  .results = {&gather_result, 1, 0, 1, 0}},
};
static const iree_vm_module_export_declaration_t gather_export = {
    .export_name = IREE_SVL("gather"),
    .callable_type_ordinal = 0,
    .function_ordinal = 0,
};

static void loom_serve_parameters_module_destroy(iree_vm_module_t* base) {
  loom_serve_parameters_module_t* module =
      (loom_serve_parameters_module_t*)base;
  loom_serve_parameter_sources_release(module->sources);
  iree_hal_queue_release(module->queue);
  iree_hal_device_release(module->device);
  iree_allocator_free(module->host_allocator, module);
}

static void loom_serve_parameters_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = gather_export;
}

static void loom_serve_parameters_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = gather_callable;
}

static void loom_serve_parameters_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("parameters module has no imports");
}

static void loom_serve_parameters_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("parameters module has no imports");
}

static const iree_vm_module_vtable_t module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_parameters_module_destroy,
    .function_start = loom_serve_parameters_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_parameters_module_query_import_group,
    .query_import = loom_serve_parameters_module_query_import,
    .query_export = loom_serve_parameters_module_query_export,
    .query_callable_type = loom_serve_parameters_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_parameters_module_create(
    iree_vm_environment_t* environment, const iree_hal_module_types_t* types,
    iree_hal_queue_t* queue, loom_serve_parameter_sources_t* sources,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t vm_types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &vm_types));
  loom_serve_parameters_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types[0] = vm_types.buffer;
  module->types[1] = types->buffer;
  module->types[2] = types->fence;
  module->device = iree_hal_queue_family_device(iree_hal_queue_family(queue));
  iree_hal_device_retain(module->device);
  module->queue = queue;
  iree_hal_queue_retain(queue);
  loom_serve_parameters_module_bind_sources(&module->base, sources);
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("parameters"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {module->types, IREE_ARRAYSIZE(module->types)},
      .counts = {.function_count = 1,
                 .callable_type_count = 1,
                 .export_count = 1,
                 .callable_fields = {.value_count = 3, .ref_count = 5}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    loom_serve_parameters_module_destroy(&module->base);
  }
  return status;
}
