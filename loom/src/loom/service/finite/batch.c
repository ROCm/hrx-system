// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/finite/batch.h"

#include "iree/hal/vm/types.h"
#include "iree/vm/reflection.h"
#include "loom/runtime/storage/ranges.h"

typedef struct loom_service_finite_batch_module_t {
  // Module and batch handles share the same intrusive owner-count prefix.
  iree_vm_module_t base;
  // Native exports borrowing the module-owned type array.
  iree_vm_module_descriptor_t descriptor;
  // Allocator owning this instance and explicitly requested packed uploads.
  iree_allocator_t host_allocator;
  // Canonical buffer and finite.batch identities.
  iree_vm_ref_type_t types[3];
  // Source invocation authority; device views are available only at submit.
  loom_service_finite_batch_phase_t phase;
  // Source-declared maximum additive work per epoch.
  uint64_t batch_limit;
  // Borrowed stable members, present only during one synchronous invocation.
  loom_service_prepared_request_t* requests;
  // Number of members in the active view.
  iree_host_size_t count;
  // Accepted output metadata; NULL while preparing, borrowed while collecting.
  const loom_service_finite_result_t* result;
} loom_service_finite_batch_module_t;

static void loom_service_finite_batch_destroy(void* pointer) {
  loom_service_finite_batch_module_t* module = pointer;
  iree_allocator_free(module->host_allocator, module);
}

static const iree_vm_ref_type_table_t batch_types;
static const iree_vm_ref_type_descriptor_t batch_type = {
    .destroy = loom_service_finite_batch_destroy,
    .table = &batch_types,
    .type_name = IREE_SVL("batch"),
};
static const iree_vm_ref_type_t batch_type_values[] = {&batch_type};
static const iree_vm_ref_type_table_t batch_types = {
    .structure_size = sizeof(iree_vm_ref_type_table_t),
    .namespace_name = IREE_SVL("finite"),
    .types = {batch_type_values, IREE_ARRAYSIZE(batch_type_values)},
};

iree_status_t loom_service_finite_batch_register_types(
    iree_vm_environment_t* environment) {
  return iree_vm_environment_register_ref_type_table(environment, &batch_types);
}

iree_vm_ref_type_t loom_service_finite_batch_type(void) { return &batch_type; }

iree_vm_variant_t loom_service_finite_batch_begin(
    iree_vm_module_t* base, loom_service_finite_batch_phase_t phase,
    iree_host_size_t count, loom_service_prepared_request_t* requests,
    const loom_service_finite_result_t* result) {
  loom_service_finite_batch_module_t* module =
      (loom_service_finite_batch_module_t*)base;
  module->phase = phase;
  module->requests = requests;
  module->count = count;
  module->result = result;
  return iree_vm_variant_from_ptr_borrowed(module, &batch_type);
}

iree_status_t loom_service_finite_batch_end(iree_vm_module_t* base,
                                            iree_status_t status) {
  loom_service_finite_batch_module_t* module =
      (loom_service_finite_batch_module_t*)base;
  if (module->result) {
    for (iree_host_size_t i = 0; i < module->count && iree_status_is_ok(status);
         ++i) {
      const bool published =
          loom_service_finite_result_row_count(module->result, i) != 0;
      if (published != (module->requests[i].units == 0)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "finite completion must agree with published result rows");
      }
    }
  }
  module->requests = NULL;
  module->count = 0;
  module->result = NULL;
  return status;
}

enum loom_service_finite_batch_function_e {
  BATCH_FINISH,
  BATCH_COUNT,
  BATCH_PACK,
  BATCH_RELEASE,
  BATCH_ORIGIN,
  BATCH_STATE,
  BATCH_BUFFER,
  BATCH_OFFER,
};

static iree_status_t loom_service_finite_batch_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  const iree_vm_call_packet_t* call = &params->call;
  iree_vm_ref_t argument = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &argument);
  loom_service_finite_batch_module_t* batch = argument.object;
  if (!batch || !batch->requests) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "finite batch is not active");
  }
  const iree_host_size_t function = params->function_ordinal;
  if ((function == BATCH_OFFER || function == BATCH_FINISH ||
       function == BATCH_RELEASE) &&
      batch->phase != LOOM_SERVICE_FINITE_BATCH_COLLECT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "finite offers require successful collection");
  }
  if (function == BATCH_BUFFER &&
      batch->phase != LOOM_SERVICE_FINITE_BATCH_SUBMIT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "finite device views require submission");
  }
  if (function == BATCH_COUNT) {
    iree_vm_call_value_result_store(call, 0, batch->count);
  } else if (function == BATCH_FINISH) {
    for (iree_host_size_t i = 0; i < batch->count; ++i) {
      batch->requests[i].units = 0;
    }
  } else if (function == BATCH_PACK) {
    iree_vm_buffer_t* ranges = NULL;
    iree_vm_buffer_t* packed = NULL;
    IREE_RETURN_IF_ERROR(loom_service_prepared_requests_pack(
        batch->count, batch->requests, batch->host_allocator, &ranges,
        &packed));
    iree_vm_ref_t results[] = {
        iree_vm_ref_from_ptr_move((void**)&ranges, batch->types[0]),
        iree_vm_ref_from_ptr_move((void**)&packed, batch->types[0]),
    };
    iree_vm_call_ref_result_store_move(call, 0, &results[0]);
    iree_vm_call_ref_result_store_move(call, 1, &results[1]);
  } else {
    const uint64_t ordinal = iree_vm_call_value_argument_load(call, 0);
    if (ordinal >= batch->count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "finite member ordinal exceeds batch count");
    }
    loom_service_prepared_request_t* request = &batch->requests[ordinal];
    if (function == BATCH_STATE) {
      iree_vm_ref_t state =
          iree_vm_ref_from_ptr_retained(request->bytes, batch->types[0]);
      iree_vm_call_ref_result_store_move(call, 0, &state);
    } else if (function == BATCH_RELEASE) {
      loom_runtime_range_reset(request->resource);
    } else if (function == BATCH_ORIGIN || function == BATCH_BUFFER) {
      if (!loom_runtime_range_is_active(request->resource)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "finite member has no retained interval");
      }
      if (function == BATCH_ORIGIN) {
        iree_vm_call_value_result_store(
            call, 0, loom_runtime_range_begin(request->resource));
      } else {
        const uint32_t root =
            (uint32_t)iree_vm_call_value_argument_load(call, 1);
        if (root >= loom_runtime_range_buffer_count(request->resource)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "finite root exceeds declared storage");
        }
        iree_vm_ref_t view = iree_vm_ref_from_ptr_retained(
            loom_runtime_range_buffer(request->resource, root),
            batch->types[2]);
        iree_vm_call_ref_result_store_move(call, 0, &view);
      }
    } else {
      const uint64_t units = iree_vm_call_value_argument_load(call, 1);
      if (units > batch->batch_limit) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "finite work exceeds declared batch capacity");
      }
      request->units = units;
      request->tag = iree_vm_call_value_argument_load(call, 2);
    }
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t batch_argument[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 1}};
static const iree_vm_module_signature_type_t member_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 1}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t root_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 1},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t root_result[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2}};
static const iree_vm_module_signature_type_t offer_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 1},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t count_result[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t buffer_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};

// Canonical signature order: argument count/types, then result count/types.
static const iree_vm_module_callable_type_declaration_t batch_callables[] = {
    {.signature = {.arguments = {batch_argument, 1, 0, 1, 0}}},
    {.signature = {.arguments = {batch_argument, 1, 0, 1, 0},
                   .results = {count_result, 1, 1, 0, 0}}},
    {.signature = {.arguments = {batch_argument, 1, 0, 1, 0},
                   .results = {buffer_results, 2, 0, 2, 0}}},
    {.signature = {.arguments = {member_arguments, 2, 1, 1, 0}}},
    {.signature = {.arguments = {member_arguments, 2, 1, 1, 0},
                   .results = {count_result, 1, 1, 0, 0}}},
    {.signature = {.arguments = {member_arguments, 2, 1, 1, 0},
                   .results = {buffer_results, 1, 0, 1, 0}}},
    {.signature = {.arguments = {root_arguments, 3, 2, 1, 0},
                   .results = {root_result, 1, 0, 1, 0}}},
    {.signature = {.arguments = {offer_arguments, 4, 3, 1, 0}}},
};

static const iree_vm_module_export_declaration_t batch_exports[] = {
    {.export_name = IREE_SVL("batch_buffer"),
     .callable_type_ordinal = BATCH_BUFFER,
     .function_ordinal = BATCH_BUFFER},
    {.export_name = IREE_SVL("batch_count"),
     .callable_type_ordinal = BATCH_COUNT,
     .function_ordinal = BATCH_COUNT},
    {.export_name = IREE_SVL("batch_finish"),
     .callable_type_ordinal = BATCH_FINISH,
     .function_ordinal = BATCH_FINISH},
    {.export_name = IREE_SVL("batch_offer"),
     .callable_type_ordinal = BATCH_OFFER,
     .function_ordinal = BATCH_OFFER},
    {.export_name = IREE_SVL("batch_origin"),
     .callable_type_ordinal = BATCH_ORIGIN,
     .function_ordinal = BATCH_ORIGIN},
    {.export_name = IREE_SVL("batch_pack"),
     .callable_type_ordinal = BATCH_PACK,
     .function_ordinal = BATCH_PACK},
    {.export_name = IREE_SVL("batch_release"),
     .callable_type_ordinal = BATCH_RELEASE,
     .function_ordinal = BATCH_RELEASE},
    {.export_name = IREE_SVL("batch_state"),
     .callable_type_ordinal = BATCH_STATE,
     .function_ordinal = BATCH_STATE},
};

static void loom_service_finite_batch_module_destroy(iree_vm_module_t* base) {
  loom_service_finite_batch_destroy(base);
}
static void loom_service_finite_batch_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = batch_exports[ordinal];
}
static void loom_service_finite_batch_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = batch_callables[ordinal];
}
static void loom_service_finite_batch_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("finite module has no imports");
}
static void loom_service_finite_batch_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("finite module has no imports");
}

static const iree_vm_module_vtable_t batch_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_service_finite_batch_module_destroy,
    .function_start = loom_service_finite_batch_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_service_finite_batch_query_import_group,
    .query_import = loom_service_finite_batch_query_import,
    .query_export = loom_service_finite_batch_query_export,
    .query_callable_type = loom_service_finite_batch_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_service_finite_batch_module_create(
    iree_vm_environment_t* environment, const iree_hal_vm_types_t* hal_types,
    uint64_t batch_limit, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_service_finite_batch_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->batch_limit = batch_limit;
  module->types[0] = types.buffer;
  module->types[1] = &batch_type;
  module->types[2] = hal_types->buffer;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("finite"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {module->types, IREE_ARRAYSIZE(module->types)},
      .counts = {.function_count = IREE_ARRAYSIZE(batch_exports),
                 .callable_type_count = IREE_ARRAYSIZE(batch_callables),
                 .export_count = IREE_ARRAYSIZE(batch_exports),
                 .callable_fields = {.value_count = 10, .ref_count = 12}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &batch_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
