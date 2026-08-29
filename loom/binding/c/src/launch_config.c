// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/environment.h"
#include "iree/vm/process.h"
#include "iree/vm/variant.h"
#include "loom/codegen/low/launch_config_abi.h"
#include "loomc/iree.h"

enum {
  // Reusable storage for synchronous launch configuration execution. The
  // compiled functions are deliberately small and may not yield.
  LOOMC_LAUNCH_CONFIG_INVOCATION_STORAGE_SIZE = 16 * 1024,
};

typedef struct loomc_launch_config_function_storage_t {
  // Process-bound function invoked without a lookup on the hot path.
  iree_vm_function_t function;

  // Stable public kernel export name borrowed from the loaded VM module.
  iree_string_view_t name;

  // Source-ordered scalar argument signature.
  struct {
    // Scalar types in program-owned slab storage.
    const iree_vm_scalar_type_t* types;

    // Number of entries in |types|.
    iree_host_size_t count;
  } arguments;
} loomc_launch_config_function_storage_t;

struct loomc_launch_config_program_t {
  // Atomic reference count for retained handle ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator owning the complete program slab and runtime objects.
  loomc_allocator_t allocator;

  // Mutable process instance exclusively driven by this handle.
  iree_vm_process_t* process;

  // Reusable fixed-capacity invocation embedded in this program slab.
  iree_vm_invocation_t* invocation;

  // Name-sorted export-ordinal functions.
  struct {
    // Function table in program-owned slab storage.
    loomc_launch_config_function_storage_t* data;

    // Number of entries in |data|.
    iree_host_size_t count;
  } functions;

  // Reusable source-ordered invocation arguments in slab storage.
  iree_vm_variant_t* argument_variants;
};

static bool loomc_launch_config_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_launch_config_validate_result(
    const loomc_launch_config_t* config) {
  if (config == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_config must not be NULL");
  }
  if (config->type != LOOMC_STRUCTURE_TYPE_NONE &&
      config->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config has an unknown structure type");
  }
  if (config->structure_size != 0 && config->structure_size < sizeof(*config)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config structure_size is too small");
  }
  if (config->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config result extensions are not supported");
  }
  return loomc_ok_status();
}

static bool loomc_launch_config_scalar_type_is_supported(uint16_t kind,
                                                         uint16_t ordinal) {
  if (ordinal != 0) return false;
  switch ((iree_vm_scalar_type_t)kind) {
    case IREE_VM_SCALAR_TYPE_I8:
    case IREE_VM_SCALAR_TYPE_I16:
    case IREE_VM_SCALAR_TYPE_I32:
    case IREE_VM_SCALAR_TYPE_I64:
    case IREE_VM_SCALAR_TYPE_F8E4M3FN:
    case IREE_VM_SCALAR_TYPE_F8E5M2:
    case IREE_VM_SCALAR_TYPE_F16:
    case IREE_VM_SCALAR_TYPE_BF16:
    case IREE_VM_SCALAR_TYPE_F32:
    case IREE_VM_SCALAR_TYPE_F64:
      return true;
    default:
      return false;
  }
}

static iree_status_t loomc_launch_config_query_function(
    const iree_vm_module_t* module, iree_host_size_t export_ordinal,
    iree_string_view_t* out_name,
    iree_vm_module_callable_type_declaration_t* out_callable) {
  *out_name = iree_string_view_empty();
  *out_callable = (iree_vm_module_callable_type_declaration_t){0};
  iree_vm_module_export_declaration_t export_declaration = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_module_query_export(module, export_ordinal, &export_declaration));
  const iree_string_view_t private_name = export_declaration.export_name;
  if (private_name.size <= LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH ||
      private_name.data[0] != LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config artifact export '%.*s' has no private name prefix",
        (int)private_name.size, private_name.data);
  }
  *out_name = iree_string_view_remove_prefix(
      private_name, LOOM_KERNEL_LAUNCH_CONFIG_EXPORT_PREFIX_LENGTH);
  IREE_RETURN_IF_ERROR(iree_vm_module_query_callable_type(
      module, export_declaration.callable_type_ordinal, out_callable));
  if (iree_any_bit_set(out_callable->flags,
                       IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "launch config export '%.*s' must not be yieldable",
                            (int)out_name->size, out_name->data);
  }

  const iree_vm_module_signature_type_span_t arguments =
      out_callable->signature.arguments;
  for (iree_host_size_t i = 0; i < arguments.count; ++i) {
    if (!loomc_launch_config_scalar_type_is_supported(
            arguments.data[i].kind, arguments.data[i].type_ordinal)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "launch config export '%.*s' argument %" PRIhsz
                              " is not a supported scalar",
                              (int)out_name->size, out_name->data, i);
    }
  }

  const iree_vm_module_signature_type_span_t results =
      out_callable->signature.results;
  if (results.count != LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "launch config export '%.*s' must return %u values",
                            (int)out_name->size, out_name->data,
                            (unsigned)LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT);
  }
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    if (results.data[i].kind != IREE_VM_SCALAR_TYPE_I64 ||
        results.data[i].type_ordinal != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "launch config export '%.*s' result %" PRIhsz
                              " must be i64",
                              (int)out_name->size, out_name->data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loomc_launch_config_measure_functions(
    const iree_vm_module_t* module, iree_host_size_t* out_argument_type_count,
    iree_host_size_t* out_max_argument_count) {
  *out_argument_type_count = 0;
  *out_max_argument_count = 0;
  const iree_host_size_t function_count = iree_vm_module_export_count(module);
  if (function_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config artifact contains no exported functions");
  }
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    iree_vm_module_callable_type_declaration_t callable = {0};
    IREE_RETURN_IF_ERROR(
        loomc_launch_config_query_function(module, i, &name, &callable));
    if (!iree_host_size_checked_add(*out_argument_type_count,
                                    callable.signature.arguments.count,
                                    out_argument_type_count)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "launch config argument type count exceeds the host size domain");
    }
    *out_max_argument_count =
        iree_max(*out_max_argument_count, callable.signature.arguments.count);
  }
  return iree_ok_status();
}

static iree_status_t loomc_launch_config_program_create_slab(
    iree_vm_module_t* module, iree_host_size_t argument_type_count,
    iree_host_size_t max_argument_count, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program) {
  *out_program = NULL;
  const iree_host_size_t function_count = iree_vm_module_export_count(module);
  iree_host_size_t total_size = 0;
  iree_host_size_t functions_offset = 0;
  iree_host_size_t argument_types_offset = 0;
  iree_host_size_t argument_variants_offset = 0;
  iree_host_size_t invocation_storage_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loomc_launch_config_program_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          function_count, loomc_launch_config_function_storage_t,
          iree_alignof(loomc_launch_config_function_storage_t),
          &functions_offset),
      IREE_STRUCT_FIELD_ALIGNED(argument_type_count, iree_vm_scalar_type_t,
                                iree_alignof(iree_vm_scalar_type_t),
                                &argument_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(max_argument_count, iree_vm_variant_t,
                                iree_alignof(iree_vm_variant_t),
                                &argument_variants_offset),
      IREE_STRUCT_FIELD_ALIGNED(LOOMC_LAUNCH_CONFIG_INVOCATION_STORAGE_SIZE,
                                uint8_t, iree_max_align_t,
                                &invocation_storage_offset)));

  loomc_launch_config_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_from_loomc(allocator), total_size, (void**)&program));
  memset(program, 0, total_size);
  iree_atomic_ref_count_init(&program->ref_count);
  program->allocator = allocator;
  program->functions.data =
      (loomc_launch_config_function_storage_t*)((uint8_t*)program +
                                                functions_offset);
  program->functions.count = function_count;
  program->argument_variants =
      max_argument_count != 0
          ? (iree_vm_variant_t*)((uint8_t*)program + argument_variants_offset)
          : NULL;

  iree_vm_scalar_type_t* argument_types =
      argument_type_count != 0
          ? (iree_vm_scalar_type_t*)((uint8_t*)program + argument_types_offset)
          : NULL;
  iree_host_size_t argument_type_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    iree_string_view_t name = iree_string_view_empty();
    iree_vm_module_callable_type_declaration_t callable = {0};
    iree_status_t status =
        loomc_launch_config_query_function(module, i, &name, &callable);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_from_loomc(allocator), program);
      return status;
    }
    loomc_launch_config_function_storage_t* function =
        &program->functions.data[i];
    function->name = name;
    function->arguments.types = callable.signature.arguments.count != 0
                                    ? argument_types + argument_type_offset
                                    : NULL;
    function->arguments.count = callable.signature.arguments.count;
    for (iree_host_size_t j = 0; j < function->arguments.count; ++j) {
      argument_types[argument_type_offset++] =
          (iree_vm_scalar_type_t)callable.signature.arguments.data[j].kind;
    }
  }
  iree_status_t status = iree_vm_invocation_initialize(
      iree_make_byte_span((uint8_t*)program + invocation_storage_offset,
                          LOOMC_LAUNCH_CONFIG_INVOCATION_STORAGE_SIZE),
      &program->invocation);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_from_loomc(allocator), program);
    return status;
  }
  *out_program = program;
  return iree_ok_status();
}

static void loomc_launch_config_program_destroy(
    loomc_launch_config_program_t* program) {
  if (program == NULL) return;
  loomc_allocator_t allocator = program->allocator;
  iree_vm_invocation_deinitialize(program->invocation);
  iree_vm_process_release(program->process);
  loomc_allocator_free(allocator, program);
}

static iree_status_t loomc_launch_config_bind_functions(
    const iree_vm_module_t* module, iree_vm_process_t* process,
    loomc_launch_config_program_t* program) {
  for (iree_host_size_t i = 0; i < program->functions.count; ++i) {
    iree_vm_export_t export_value = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_module_export_by_ordinal(module, i, &export_value));
    IREE_RETURN_IF_ERROR(iree_vm_function_from_export(
        process, export_value, &program->functions.data[i].function));
  }
  return iree_ok_status();
}

static loomc_status_t loomc_launch_config_program_load_impl(
    const loomc_artifact_t* artifact, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program) {
  if (out_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program must not be NULL");
  }
  *out_program = NULL;
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact must not be NULL");
  }
  if (artifact->kind != LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact kind is not LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG");
  }
  if (!loomc_launch_config_string_view_is_well_formed(artifact->format) ||
      !loomc_launch_config_string_view_is_well_formed(artifact->identifier)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact string view is malformed");
  }
  if (!loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE))) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch config artifact format '%.*s' is not supported",
        (int)iree_min(artifact->format.size, 128),
        artifact->format.data != NULL ? artifact->format.data : ""));
  }
  if (artifact->contents == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }

  const iree_allocator_t host_allocator = iree_allocator_from_loomc(allocator);
  iree_byte_span_t image_contents = iree_byte_span_empty();
  iree_vm_environment_t* environment = NULL;
  iree_vm_module_t* module = NULL;
  iree_vm_program_t* vm_program = NULL;
  loomc_launch_config_program_t* program = NULL;

  iree_status_t status = iree_byte_sequence_clone(
      iree_byte_sequence_from_loomc(artifact->contents), host_allocator,
      &image_contents);
  if (iree_status_is_ok(status)) {
    status = iree_vm_environment_allocate(host_allocator, &environment);
  }
  if (iree_status_is_ok(status)) {
    const iree_vm_bytecode_module_storage_t module_storage = {
        .contents = iree_make_const_byte_span(image_contents.data,
                                              image_contents.data_length),
        .deallocator = host_allocator,
    };
    status =
        iree_vm_bytecode_module_create(environment, IREE_SV("launch_config"),
                                       module_storage, host_allocator, &module);
    if (iree_status_is_ok(status)) {
      image_contents = iree_byte_span_empty();
    }
  }
  iree_vm_environment_free(environment);

  iree_host_size_t argument_type_count = 0;
  iree_host_size_t max_argument_count = 0;
  if (iree_status_is_ok(status)) {
    status = loomc_launch_config_measure_functions(module, &argument_type_count,
                                                   &max_argument_count);
  }
  if (iree_status_is_ok(status)) {
    status = loomc_launch_config_program_create_slab(
        module, argument_type_count, max_argument_count, allocator, &program);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_program_create(
        (iree_vm_program_modules_t){
            .executable = module,
            .libraries = iree_vm_module_span_empty(),
        },
        host_allocator, &vm_program);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(vm_program, program->invocation,
                                    iree_vm_variant_span_empty(),
                                    host_allocator, &program->process);
  }
  if (iree_status_is_ok(status)) {
    status =
        loomc_launch_config_bind_functions(module, program->process, program);
  }
  if (iree_status_is_ok(status)) {
    *out_program = program;
    program = NULL;
  }

  loomc_launch_config_program_destroy(program);
  iree_vm_program_release(vm_program);
  iree_vm_module_release(module);
  iree_allocator_free(host_allocator, image_contents.data);
  return loomc_status_from_iree(status);
}

loomc_status_t loomc_launch_config_program_load(
    const loomc_artifact_t* artifact, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program) {
  return loomc_launch_config_program_load_impl(artifact, allocator,
                                               out_program);
}

void loomc_launch_config_program_retain(
    loomc_launch_config_program_t* program) {
  if (program == NULL) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

void loomc_launch_config_program_release(
    loomc_launch_config_program_t* program) {
  if (program == NULL) return;
  if (iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    loomc_launch_config_program_destroy(program);
  }
}

loomc_status_t loomc_launch_config_program_lookup_function(
    const loomc_launch_config_program_t* program,
    loomc_string_view_t export_name,
    loomc_launch_config_function_t* out_function) {
  if (out_function == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_function must not be NULL");
  }
  *out_function = loomc_launch_config_function_invalid();
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config program must not be NULL");
  }
  if (!loomc_launch_config_string_view_is_well_formed(export_name) ||
      loomc_string_view_is_empty(export_name) || export_name.data[0] == '@') {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "export_name must be a non-empty name without a leading '@'");
  }

  const iree_string_view_t name = iree_string_view_from_loomc(export_name);
  iree_host_size_t low = 0;
  iree_host_size_t high = program->functions.count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    if (iree_string_view_compare(program->functions.data[middle].name, name) <
        0) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low < program->functions.count &&
      iree_string_view_equal(program->functions.data[low].name, name)) {
    *out_function = (loomc_launch_config_function_t){
        .value = low,
    };
    return loomc_ok_status();
  }
  return loomc_status_from_iree(iree_make_status(
      IREE_STATUS_NOT_FOUND, "launch config export '%.*s' was not found",
      (int)iree_min(name.size, 128), name.data));
}

static iree_status_t loomc_launch_config_extract_u32(
    const loomc_launch_config_function_storage_t* function,
    const iree_vm_variant_t* results, iree_host_size_t result_ordinal,
    const char* field_name, bool require_nonzero, uint32_t* out_value) {
  const uint64_t value = results[result_ordinal].payload;
  if (value > UINT32_MAX || (require_nonzero && value == 0)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config function '%.*s' %s value %" PRIu64
                            " is outside its u32 domain",
                            (int)function->name.size, function->name.data,
                            field_name, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loomc_launch_config_unpack_results(
    const loomc_launch_config_function_storage_t* function,
    const iree_vm_variant_t* results, loomc_launch_config_t* out_config) {
  loomc_launch_config_t config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(config),
  };
#define LOOMC_EXTRACT_U32(field, result, nonzero)       \
  IREE_RETURN_IF_ERROR(loomc_launch_config_extract_u32( \
      function, results, result, #field, nonzero, &config.field))
  LOOMC_EXTRACT_U32(workgroup_count.x,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X, false);
  LOOMC_EXTRACT_U32(workgroup_count.y,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y, false);
  LOOMC_EXTRACT_U32(workgroup_count.z,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z, false);
  LOOMC_EXTRACT_U32(workgroup_size.x,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X, true);
  LOOMC_EXTRACT_U32(workgroup_size.y,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y, true);
  LOOMC_EXTRACT_U32(workgroup_size.z,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z, true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.x,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X,
                    true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.y,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y,
                    true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.z,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z,
                    true);
  LOOMC_EXTRACT_U32(subgroup_size,
                    LOOM_KERNEL_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE, false);
#undef LOOMC_EXTRACT_U32
  const uint64_t workgroup_storage_bytes =
      results[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES].payload;
  if (workgroup_storage_bytes > INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "launch config function '%.*s' workgroup_storage_bytes value %" PRIu64
        " is outside its nonnegative i64 domain",
        (int)function->name.size, function->name.data, workgroup_storage_bytes);
  }
  config.workgroup_storage_bytes = workgroup_storage_bytes;
  *out_config = config;
  return iree_ok_status();
}

loomc_status_t loomc_launch_config_program_invoke(
    loomc_launch_config_program_t* program,
    loomc_launch_config_function_t function,
    const uint64_t* workload_argument_bits,
    loomc_host_size_t workload_argument_count,
    loomc_launch_config_t* out_config) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_result(out_config));
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config program must not be NULL");
  }
  if (function.value >= program->functions.count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config function token is out of range");
  }
  if (workload_argument_count != 0 && workload_argument_bits == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workload_argument_count is non-zero but workload_argument_bits is "
        "NULL");
  }

  const loomc_launch_config_function_storage_t* function_storage =
      &program->functions.data[function.value];
  if (workload_argument_count != function_storage->arguments.count) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '%.*s' expects %" PRIhsz
        " arguments but received %" PRIhsz,
        (int)function_storage->name.size, function_storage->name.data,
        function_storage->arguments.count, workload_argument_count));
  }
  for (iree_host_size_t i = 0; i < workload_argument_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_status_from_iree(iree_vm_variant_from_scalar_bits(
            function_storage->arguments.types[i], workload_argument_bits[i],
            &program->argument_variants[i])));
  }

  iree_vm_variant_t results[LOOM_KERNEL_LAUNCH_CONFIG_RESULT_COUNT];
  iree_status_t status =
      iree_vm_invoke(program->invocation, function_storage->function,
                     iree_vm_variant_span_from_ptr(program->argument_variants,
                                                   workload_argument_count),
                     iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    status = loomc_launch_config_unpack_results(function_storage, results,
                                                out_config);
  }
  return loomc_status_from_iree(status);
}
