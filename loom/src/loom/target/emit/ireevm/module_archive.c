// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/module_archive.h"

#include <inttypes.h>
#include <limits.h>
#include <string.h>

#include "iree/base/internal/flatcc/building.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/vm/bytecode/isa/isa.h"

typedef struct loom_ireevm_module_build_t {
  // FlatCC table references for each imported function.
  iree_vm_ImportFunctionDef_ref_t* import_references;
  // FlatCC table references for each local function signature.
  iree_vm_FunctionSignatureDef_ref_t* signature_references;
  // FlatCC table references for each exported function.
  iree_vm_ExportFunctionDef_ref_t* export_references;
  // FlatCC struct payloads for each local function descriptor.
  iree_vm_FunctionDescriptor_t* descriptors;
  // Concatenated padded VM function bytecode data.
  uint8_t* bytecode_data;
  // Number of bytes in |bytecode_data|.
  iree_host_size_t bytecode_data_length;
  // IREE VM FeatureBits required by all local function bodies.
  uint32_t feature_requirements;
} loom_ireevm_module_build_t;

static void loom_ireevm_module_build_deinitialize(
    loom_ireevm_module_build_t* build, iree_allocator_t allocator) {
  iree_allocator_free(allocator, build->import_references);
  iree_allocator_free(allocator, build->signature_references);
  iree_allocator_free(allocator, build->export_references);
  iree_allocator_free(allocator, build->descriptors);
  iree_allocator_free(allocator, build->bytecode_data);
  *build = (loom_ireevm_module_build_t){0};
}

static iree_status_t loom_ireevm_module_archive_validate_imports(
    const loom_ireevm_module_archive_import_t* imports,
    iree_host_size_t import_count) {
  if (import_count == 0) {
    return iree_ok_status();
  }
  if (!imports) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module import table is required");
  }
  if (import_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM module import count exceeds i32");
  }
  for (iree_host_size_t i = 0; i < import_count; ++i) {
    const loom_ireevm_module_archive_import_t* import = &imports[i];
    if (iree_string_view_is_empty(import->full_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM import full name is required");
    }
    if (iree_string_view_is_empty(import->calling_convention)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM import calling convention is required for '%.*s'",
          (int)import->full_name.size, import->full_name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_archive_validate_functions(
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, loom_ireevm_module_build_t* build) {
  if (!functions || function_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module requires at least one function");
  }
  if (function_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM module function count exceeds i32");
  }

  iree_host_size_t bytecode_data_length = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const loom_ireevm_module_archive_function_t* function = &functions[i];
    if (iree_string_view_is_empty(function->function_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM function name is required");
    }
    if (iree_string_view_is_empty(function->calling_convention)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM function calling convention is required for '%.*s'",
          (int)function->function_name.size, function->function_name.data);
    }
    if (!function->bytecode || !function->bytecode->data ||
        function->bytecode->data_length == 0 ||
        function->bytecode->bytecode_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM function bytecode is required for '%.*s'",
                              (int)function->function_name.size,
                              function->function_name.data);
    }
    if (function->bytecode->block_count > INT16_MAX ||
        function->bytecode->i32_register_count > INT16_MAX ||
        function->bytecode->ref_register_count > INT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM function descriptor counts exceed i16 for '%.*s'",
          (int)function->function_name.size, function->function_name.data);
    }
    if (function->bytecode->data_length > INT32_MAX ||
        function->bytecode->bytecode_length > INT32_MAX ||
        bytecode_data_length > INT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM function bytecode length exceeds i32 for '%.*s'",
          (int)function->function_name.size, function->function_name.data);
    }
    if (!iree_host_size_checked_add(bytecode_data_length,
                                    function->bytecode->data_length,
                                    &bytecode_data_length) ||
        bytecode_data_length > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM module bytecode data exceeds i32");
    }
    build->feature_requirements |= function->bytecode->feature_requirements;
  }
  build->bytecode_data_length = bytecode_data_length;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_archive_validate_exports(
    const loom_ireevm_module_archive_export_t* exports,
    iree_host_size_t export_count, iree_host_size_t function_count) {
  if (export_count == 0) {
    return iree_ok_status();
  }
  if (!exports) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module export table is required");
  }
  if (export_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM module export count exceeds i32");
  }
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const loom_ireevm_module_archive_export_t* export = &exports[i];
    if (iree_string_view_is_empty(export->local_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM export local name is required");
    }
    if (export->internal_ordinal >= function_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM export '%.*s' references local function ordinal %u of %" PRIhsz,
          (int)export->local_name.size, export->local_name.data,
          export->internal_ordinal, function_count);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_archive_validate_inputs(
    iree_string_view_t module_name,
    const loom_ireevm_module_archive_import_t* imports,
    iree_host_size_t import_count,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count,
    const loom_ireevm_module_archive_export_t* exports,
    iree_host_size_t export_count, loom_ireevm_module_build_t* build) {
  if (iree_string_view_is_empty(module_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module name is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_ireevm_module_archive_validate_imports(imports, import_count));
  IREE_RETURN_IF_ERROR(loom_ireevm_module_archive_validate_functions(
      functions, function_count, build));
  return loom_ireevm_module_archive_validate_exports(exports, export_count,
                                                     function_count);
}

static iree_status_t loom_ireevm_module_build_allocate(
    iree_host_size_t import_count, iree_host_size_t function_count,
    iree_host_size_t export_count, iree_allocator_t allocator,
    loom_ireevm_module_build_t* build) {
  iree_status_t status = iree_ok_status();
  if (import_count > 0) {
    status = iree_allocator_malloc_array(allocator, import_count,
                                         sizeof(*build->import_references),
                                         (void**)&build->import_references);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, function_count,
                                         sizeof(*build->signature_references),
                                         (void**)&build->signature_references);
  }
  if (iree_status_is_ok(status) && export_count > 0) {
    status = iree_allocator_malloc_array(allocator, export_count,
                                         sizeof(*build->export_references),
                                         (void**)&build->export_references);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, function_count,
                                         sizeof(*build->descriptors),
                                         (void**)&build->descriptors);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, build->bytecode_data_length,
                                   (void**)&build->bytecode_data);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_build_deinitialize(build, allocator);
  }
  return status;
}

static iree_status_t loom_ireevm_module_build_signature(
    flatcc_builder_t* builder, iree_string_view_t calling_convention,
    iree_vm_FunctionSignatureDef_ref_t* out_signature_reference) {
  *out_signature_reference = 0;
  flatbuffers_string_ref_t calling_convention_ref = flatbuffers_string_create(
      builder, calling_convention.data, calling_convention.size);
  if (!calling_convention_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM calling convention string");
  }
  if (iree_vm_FunctionSignatureDef_start(builder) ||
      iree_vm_FunctionSignatureDef_calling_convention_add(
          builder, calling_convention_ref)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to start VM function signature");
  }
  *out_signature_reference = iree_vm_FunctionSignatureDef_end(builder);
  if (!*out_signature_reference) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM function signature");
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_build_imports(
    flatcc_builder_t* builder,
    const loom_ireevm_module_archive_import_t* imports,
    iree_host_size_t import_count, loom_ireevm_module_build_t* build) {
  for (iree_host_size_t i = 0; i < import_count; ++i) {
    const loom_ireevm_module_archive_import_t* import = &imports[i];
    flatbuffers_string_ref_t full_name_ref = flatbuffers_string_create(
        builder, import->full_name.data, import->full_name.size);
    if (!full_name_ref) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM import name string");
    }
    iree_vm_FunctionSignatureDef_ref_t signature_ref = 0;
    IREE_RETURN_IF_ERROR(loom_ireevm_module_build_signature(
        builder, import->calling_convention, &signature_ref));
    if (iree_vm_ImportFunctionDef_start(builder) ||
        iree_vm_ImportFunctionDef_full_name_add(builder, full_name_ref) ||
        iree_vm_ImportFunctionDef_signature_add(builder, signature_ref)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start VM function import");
    }
    build->import_references[i] = iree_vm_ImportFunctionDef_end(builder);
    if (!build->import_references[i]) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function import");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_build_functions(
    flatcc_builder_t* builder,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, loom_ireevm_module_build_t* build) {
  iree_host_size_t bytecode_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const loom_ireevm_module_archive_function_t* function = &functions[i];
    const loom_ireevm_function_bytecode_t* bytecode = function->bytecode;

    IREE_RETURN_IF_ERROR(loom_ireevm_module_build_signature(
        builder, function->calling_convention,
        &build->signature_references[i]));
    iree_vm_FunctionDescriptor_assign(
        &build->descriptors[i], (int32_t)bytecode_offset,
        (int32_t)bytecode->bytecode_length,
        (iree_vm_FeatureBits_enum_t)bytecode->feature_requirements, 0,
        (int16_t)bytecode->block_count, (int16_t)bytecode->i32_register_count,
        (int16_t)bytecode->ref_register_count);
    memcpy(build->bytecode_data + bytecode_offset, bytecode->data,
           bytecode->data_length);
    bytecode_offset += bytecode->data_length;
  }
  IREE_ASSERT(bytecode_offset == build->bytecode_data_length);
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_build_exports(
    flatcc_builder_t* builder,
    const loom_ireevm_module_archive_export_t* exports,
    iree_host_size_t export_count, loom_ireevm_module_build_t* build) {
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const loom_ireevm_module_archive_export_t* export = &exports[i];
    flatbuffers_string_ref_t local_name_ref = flatbuffers_string_create(
        builder, export->local_name.data, export->local_name.size);
    if (!local_name_ref) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM export name string");
    }
    if (iree_vm_ExportFunctionDef_start(builder) ||
        iree_vm_ExportFunctionDef_local_name_add(builder, local_name_ref)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start VM function export");
    }
    if (export->internal_ordinal != 0 &&
        iree_vm_ExportFunctionDef_internal_ordinal_add(
            builder, (int32_t)export->internal_ordinal)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function export ordinal");
    }
    build->export_references[i] = iree_vm_ExportFunctionDef_end(builder);
    if (!build->export_references[i]) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function export");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_build_root(
    flatcc_builder_t* builder, iree_string_view_t module_name,
    iree_host_size_t import_count, iree_host_size_t function_count,
    iree_host_size_t export_count, const loom_ireevm_module_build_t* build,
    iree_allocator_t allocator, loom_ireevm_module_archive_t* out_archive) {
  flatbuffers_string_ref_t module_name_ref =
      flatbuffers_string_create(builder, module_name.data, module_name.size);
  if (!module_name_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module name string");
  }

  iree_vm_ImportFunctionDef_vec_ref_t imports_reference = 0;
  if (import_count > 0) {
    imports_reference = iree_vm_ImportFunctionDef_vec_create(
        builder, build->import_references, import_count);
    if (!imports_reference) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM import vector");
    }
  }
  iree_vm_ExportFunctionDef_vec_ref_t exports_reference = 0;
  if (export_count > 0) {
    exports_reference = iree_vm_ExportFunctionDef_vec_create(
        builder, build->export_references, export_count);
    if (!exports_reference) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM export vector");
    }
  }
  iree_vm_FunctionSignatureDef_vec_ref_t signatures_reference =
      iree_vm_FunctionSignatureDef_vec_create(
          builder, build->signature_references, function_count);
  iree_vm_FunctionDescriptor_vec_ref_t descriptors_reference =
      iree_vm_FunctionDescriptor_vec_create(builder, build->descriptors,
                                            function_count);
  flatbuffers_uint8_vec_ref_t bytecode_data_reference =
      flatbuffers_uint8_vec_create(builder, build->bytecode_data,
                                   build->bytecode_data_length);
  if (!signatures_reference || !descriptors_reference ||
      !bytecode_data_reference) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module vectors");
  }

  const uint32_t bytecode_version =
      ((uint32_t)IREE_VM_BYTECODE_VERSION_MAJOR << 16) |
      (uint32_t)IREE_VM_BYTECODE_VERSION_MINOR;
  if (iree_vm_BytecodeModuleDef_start_as_root_with_size(builder) ||
      iree_vm_BytecodeModuleDef_name_add(builder, module_name_ref) ||
      iree_vm_BytecodeModuleDef_function_signatures_add(builder,
                                                        signatures_reference) ||
      iree_vm_BytecodeModuleDef_function_descriptors_add(
          builder, descriptors_reference) ||
      iree_vm_BytecodeModuleDef_requirements_add(
          builder, (iree_vm_FeatureBits_enum_t)build->feature_requirements) ||
      iree_vm_BytecodeModuleDef_bytecode_version_add(builder,
                                                     bytecode_version) ||
      iree_vm_BytecodeModuleDef_bytecode_data_add(builder,
                                                  bytecode_data_reference)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module FlatBuffer fields");
  }
  if (imports_reference && iree_vm_BytecodeModuleDef_imported_functions_add(
                               builder, imports_reference)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module import table");
  }
  if (exports_reference && iree_vm_BytecodeModuleDef_exported_functions_add(
                               builder, exports_reference)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module export table");
  }
  flatbuffers_ref_t root_reference =
      iree_vm_BytecodeModuleDef_end_as_root(builder);
  if (!root_reference) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module FlatBuffer");
  }

  size_t flatbuffer_length = 0;
  void* flatbuffer_data =
      flatcc_builder_finalize_buffer(builder, &flatbuffer_length);
  if (!flatbuffer_data || flatbuffer_length > IREE_HOST_SIZE_MAX) {
    flatcc_builder_free(flatbuffer_data);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to finalize VM module FlatBuffer");
  }

  iree_status_t status =
      iree_allocator_malloc(allocator, (iree_host_size_t)flatbuffer_length,
                            (void**)&out_archive->data);
  if (iree_status_is_ok(status)) {
    memcpy(out_archive->data, flatbuffer_data, flatbuffer_length);
    out_archive->data_length = (iree_host_size_t)flatbuffer_length;
  }
  flatcc_builder_free(flatbuffer_data);
  return status;
}

static iree_status_t loom_ireevm_module_build_populate(
    flatcc_builder_t* builder, iree_string_view_t module_name,
    const loom_ireevm_module_archive_import_t* imports,
    iree_host_size_t import_count,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count,
    const loom_ireevm_module_archive_export_t* exports,
    iree_host_size_t export_count, loom_ireevm_module_build_t* build,
    iree_allocator_t allocator, loom_ireevm_module_archive_t* out_archive) {
  IREE_RETURN_IF_ERROR(
      loom_ireevm_module_build_imports(builder, imports, import_count, build));
  IREE_RETURN_IF_ERROR(loom_ireevm_module_build_functions(
      builder, functions, function_count, build));
  IREE_RETURN_IF_ERROR(
      loom_ireevm_module_build_exports(builder, exports, export_count, build));
  return loom_ireevm_module_build_root(builder, module_name, import_count,
                                       function_count, export_count, build,
                                       allocator, out_archive);
}

void loom_ireevm_module_archive_deinitialize(
    loom_ireevm_module_archive_t* archive, iree_allocator_t allocator) {
  if (!archive) {
    return;
  }
  iree_allocator_free(allocator, archive->data);
  *archive = (loom_ireevm_module_archive_t){0};
}

iree_status_t loom_ireevm_emit_module_archive(
    iree_string_view_t module_name,
    const loom_ireevm_module_archive_import_t* imports,
    iree_host_size_t import_count,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count,
    const loom_ireevm_module_archive_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive) {
  *out_archive = (loom_ireevm_module_archive_t){0};

  loom_ireevm_module_build_t build = {0};
  iree_status_t status = loom_ireevm_module_archive_validate_inputs(
      module_name, imports, import_count, functions, function_count, exports,
      export_count, &build);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_build_allocate(import_count, function_count,
                                               export_count, allocator, &build);
  }

  flatcc_builder_t builder;
  bool builder_initialized = false;
  if (iree_status_is_ok(status)) {
    if (flatcc_builder_init(&builder) != 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to initialize FlatBuffer builder");
    } else {
      builder_initialized = true;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_build_populate(
        &builder, module_name, imports, import_count, functions, function_count,
        exports, export_count, &build, allocator, out_archive);
  }

  if (builder_initialized) {
    flatcc_builder_clear(&builder);
  }
  loom_ireevm_module_build_deinitialize(&build, allocator);
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  return status;
}
