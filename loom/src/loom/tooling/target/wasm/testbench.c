// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/wasm/testbench.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/buffer.h"
#include "loom/tooling/target/wasm/host.h"
#include "loomc/artifact.h"
#include "loomc/iree.h"
#include "loomc/target/wasm.h"

enum {
  LOOM_WASM_TESTBENCH_ROOT_ALIGNMENT = 16,
  LOOM_WASM_TESTBENCH_NO_ROOT = UINT32_MAX,
};

typedef struct loom_wasm_testbench_product_t {
  // Host-owned ordinary Wasm module instance.
  loom_wasm_host_module_t host_module;
  // Copied physical parameter types in call order.
  loom_wasm_host_value_type_t* parameter_types;
  // Copied physical result types in return order.
  loom_wasm_host_value_type_t* result_types;
  // Source scalar type for each result, or NONE for buffer references.
  loom_scalar_type_t* result_scalar_types;
  // Reused raw argument payloads in physical signature order.
  uint64_t* argument_bits;
  // Reused raw result payloads in physical signature order.
  uint64_t* result_bits;
  // Root ordinal for each buffer argument, or NO_ROOT for scalars.
  uint32_t* argument_root_ordinals;
  // Per-call physical root selected by the first argument in each alias set.
  iree_hal_buffer_t** root_buffers;
  // Per-call scoped mapping for each complete physical root.
  iree_hal_buffer_mapping_t* root_mappings;
  // Per-call host transport descriptor for each complete root.
  loom_wasm_host_memory_region_t* root_regions;
  // Number of physical parameters.
  uint32_t parameter_count;
  // Number of physical results.
  uint32_t result_count;
  // Number of distinct logical allocation roots in the argument topology.
  uint32_t root_count;
  // Whether |argument_root_ordinals| and |root_count| are initialized.
  bool topology_initialized;
  // Allocator owning this product and its trailing storage.
  iree_allocator_t host_allocator;
} loom_wasm_testbench_product_t;

iree_status_t loom_wasm_testbench_initialize(
    loomc_target_environment_t* target_environment,
    iree_allocator_t host_allocator, loom_wasm_testbench_t* out_testbench) {
  *out_testbench = (loom_wasm_testbench_t){
      .host_allocator = host_allocator,
  };
  return iree_status_from_loomc(loomc_target_profile_select(
      target_environment, loomc_make_cstring_view("wasm:simd128"),
      loomc_allocator_from_iree(host_allocator),
      &out_testbench->target_profile));
}

void loom_wasm_testbench_deinitialize(loom_wasm_testbench_t* testbench) {
  loomc_target_profile_release(testbench->target_profile);
  memset(testbench, 0, sizeof(*testbench));
}

static iree_status_t loom_wasm_testbench_resolve_source_function(
    const loom_testbench_invocation_plan_t* invocation,
    iree_string_view_t* out_name, loom_func_like_t* out_function) {
  const loom_symbol_t* symbol =
      &invocation->module->symbols.entries[invocation->callee_ref.symbol_id];
  const loom_func_like_t function =
      loom_func_like_const_cast(invocation->module, symbol->defining_op);
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function, &parameter_count);
  IREE_ASSERT(parameter_count == invocation->input_count);
  IREE_ASSERT(function.op->result_count == invocation->result_count);
  for (uint16_t i = 0; i < parameter_count; ++i) {
    const loom_type_kind_t kind = loom_type_kind(
        loom_module_value_type(invocation->module, parameter_ids[i]));
    if (kind != LOOM_TYPE_SCALAR && kind != LOOM_TYPE_BUFFER) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm scenario parameter %u has unsupported source type kind %u",
          (unsigned)i, (unsigned)kind);
    }
  }
  const loom_value_id_t* result_ids = loom_op_results(function.op);
  for (uint16_t i = 0; i < function.op->result_count; ++i) {
    const loom_type_kind_t kind = loom_type_kind(
        loom_module_value_type(invocation->module, result_ids[i]));
    if (kind != LOOM_TYPE_SCALAR && kind != LOOM_TYPE_BUFFER) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm scenario result %u has unsupported source type kind %u",
          (unsigned)i, (unsigned)kind);
    }
  }
  *out_name =
      loom_string_table_get(&invocation->module->strings, symbol->name_id);
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_wasm_testbench_allocate_product(
    uint32_t parameter_count, uint32_t result_count,
    const loom_module_t* source_module, loom_func_like_t source_function,
    iree_allocator_t allocator, loom_wasm_testbench_product_t** out_product) {
  *out_product = NULL;
  iree_host_size_t total_size = 0;
  iree_host_size_t parameter_types_offset = 0;
  iree_host_size_t result_types_offset = 0;
  iree_host_size_t result_scalar_types_offset = 0;
  iree_host_size_t argument_bits_offset = 0;
  iree_host_size_t result_bits_offset = 0;
  iree_host_size_t argument_root_ordinals_offset = 0;
  iree_host_size_t root_buffers_offset = 0;
  iree_host_size_t root_mappings_offset = 0;
  iree_host_size_t root_regions_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_wasm_testbench_product_t), &total_size,
      IREE_STRUCT_FIELD(parameter_count, loom_wasm_host_value_type_t,
                        &parameter_types_offset),
      IREE_STRUCT_FIELD(result_count, loom_wasm_host_value_type_t,
                        &result_types_offset),
      IREE_STRUCT_FIELD(result_count, loom_scalar_type_t,
                        &result_scalar_types_offset),
      IREE_STRUCT_FIELD(parameter_count, uint64_t, &argument_bits_offset),
      IREE_STRUCT_FIELD(result_count, uint64_t, &result_bits_offset),
      IREE_STRUCT_FIELD(parameter_count, uint32_t,
                        &argument_root_ordinals_offset),
      IREE_STRUCT_FIELD(parameter_count, iree_hal_buffer_t*,
                        &root_buffers_offset),
      IREE_STRUCT_FIELD(parameter_count, iree_hal_buffer_mapping_t,
                        &root_mappings_offset),
      IREE_STRUCT_FIELD(parameter_count, loom_wasm_host_memory_region_t,
                        &root_regions_offset)));
  loom_wasm_testbench_product_t* product = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&product));
  uint8_t* storage = (uint8_t*)product;
  *product = (loom_wasm_testbench_product_t){
      .parameter_types =
          (loom_wasm_host_value_type_t*)(storage + parameter_types_offset),
      .result_types =
          (loom_wasm_host_value_type_t*)(storage + result_types_offset),
      .result_scalar_types =
          (loom_scalar_type_t*)(storage + result_scalar_types_offset),
      .argument_bits = (uint64_t*)(storage + argument_bits_offset),
      .result_bits = (uint64_t*)(storage + result_bits_offset),
      .argument_root_ordinals =
          (uint32_t*)(storage + argument_root_ordinals_offset),
      .root_buffers = (iree_hal_buffer_t**)(storage + root_buffers_offset),
      .root_mappings =
          (iree_hal_buffer_mapping_t*)(storage + root_mappings_offset),
      .root_regions =
          (loom_wasm_host_memory_region_t*)(storage + root_regions_offset),
      .parameter_count = parameter_count,
      .result_count = result_count,
      .host_allocator = allocator,
  };
  if (result_count != 0) {
    const loom_value_id_t* result_ids = loom_op_results(source_function.op);
    for (uint32_t i = 0; i < result_count; ++i) {
      const loom_type_t type =
          loom_module_value_type(source_module, result_ids[i]);
      product->result_scalar_types[i] = loom_type_kind(type) == LOOM_TYPE_SCALAR
                                            ? loom_type_element_type(type)
                                            : LOOM_SCALAR_TYPE_NONE;
    }
  }
  *out_product = product;
  return iree_ok_status();
}

static iree_status_t loom_wasm_testbench_resolve_export_name(
    const loom_wasm_testbench_t* testbench, iree_string_view_t function_name,
    iree_string_view_t* out_export_name) {
  loomc_module_function_t function;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_lookup_function(
      testbench->compilation.module, loomc_string_view_from_iree(function_name),
      &function)));
  *out_export_name = function_name;
  if (!iree_any_bit_set(function.flags,
                        LOOMC_MODULE_FUNCTION_FLAG_HAS_EXPORT_INFO)) {
    return iree_ok_status();
  }

  loomc_module_function_export_info_t export_info;
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_module_function_get_export_info(
          testbench->compilation.module, &function, &export_info)));
  if (iree_any_bit_set(export_info.flags,
                       LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL) &&
      !loomc_string_view_is_empty(export_info.export_symbol)) {
    *out_export_name = iree_string_view_from_loomc(export_info.export_symbol);
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_testbench_select_artifact(
    const loomc_result_t* result, const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE ||
        !loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_WASM_BINARY))) {
      continue;
    }
    if (*out_artifact != NULL) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "Wasm compilation returned more than one executable artifact");
    }
    *out_artifact = artifact;
  }
  return *out_artifact != NULL
             ? iree_ok_status()
             : iree_make_status(
                   IREE_STATUS_INTERNAL,
                   "Wasm compilation returned no executable artifact");
}

static iree_status_t loom_wasm_testbench_compile_product(
    loom_wasm_testbench_t* testbench,
    const loom_testbench_invocation_plan_t* invocation,
    iree_string_view_t function_name, loom_func_like_t source_function,
    iree_allocator_t product_allocator,
    loom_wasm_testbench_product_t** out_product) {
  *out_product = NULL;
  if (invocation->input_count > UINT32_MAX ||
      invocation->result_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm scenario signature exceeds u32");
  }

  iree_string_view_t export_name = iree_string_view_empty();
  iree_status_t status = loom_wasm_testbench_resolve_export_name(
      testbench, function_name, &export_name);
  loomc_module_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_clone(
        testbench->compilation.module, testbench->compilation.workspace,
        loomc_allocator_from_iree(testbench->host_allocator), &module));
  }

  const loomc_string_view_t root = loomc_string_view_from_iree(function_name);
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_WASM_BINARY),
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .roots = &root,
      .root_count = 1,
      .target_profile = testbench->target_profile,
      .config = testbench->compilation.config,
      .emit_options = &emit_options,
  };
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compile_artifact(
        testbench->compilation.compiler, testbench->compilation.workspace,
        /*pass_program=*/NULL, module, &compile_options,
        loomc_allocator_from_iree(testbench->host_allocator), &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_observe_compile_result(testbench->result_callback,
                                                   result);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(result)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Wasm scenario compilation rejected '%.*s'",
                              (int)function_name.size, function_name.data);
  }

  const loomc_artifact_t* artifact = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_wasm_testbench_select_artifact(result, &artifact);
  }
  loomc_byte_span_t cloned_contents = loomc_byte_span_empty();
  loomc_byte_span_t contents = loomc_byte_span_empty();
  if (iree_status_is_ok(status) && !loomc_byte_sequence_try_get_contiguous_span(
                                       artifact->contents, &contents)) {
    status = iree_status_from_loomc(loomc_byte_sequence_clone(
        artifact->contents,
        loomc_allocator_from_iree(testbench->host_allocator),
        &cloned_contents));
    contents = cloned_contents;
  }

  loom_wasm_testbench_product_t* product = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_wasm_testbench_allocate_product(
        (uint32_t)invocation->input_count, (uint32_t)invocation->result_count,
        invocation->module, source_function, product_allocator, &product);
  }
  if (iree_status_is_ok(status)) {
    status = loom_wasm_host_module_load(
        iree_make_const_byte_span(contents.data, contents.data_length),
        export_name, product->parameter_count, product->parameter_types,
        product->result_count, product->result_types, &product->host_module);
  }
  if (iree_status_is_ok(status)) {
    *out_product = product;
  } else if (product != NULL) {
    loom_wasm_host_module_release(&product->host_module);
    iree_allocator_free(product_allocator, product);
  }
  iree_allocator_free(testbench->host_allocator, (void*)cloned_contents.data);
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_workspace_trim(testbench->compilation.workspace);
  return status;
}

static void loom_wasm_testbench_initialize_topology(
    loom_wasm_testbench_product_t* product,
    const loom_testbench_value_t* arguments) {
  uint32_t root_count = 0;
  for (uint32_t i = 0; i < product->parameter_count; ++i) {
    const loom_testbench_value_t* argument = &arguments[i];
    if (argument->kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      product->argument_root_ordinals[i] = LOOM_WASM_TESTBENCH_NO_ROOT;
      continue;
    }
    IREE_ASSERT(argument->buffer_reference.is_traceable);
    uint32_t root_ordinal = LOOM_WASM_TESTBENCH_NO_ROOT;
    for (uint32_t j = 0; j < i; ++j) {
      if (arguments[j].kind == LOOM_TESTBENCH_VALUE_KIND_BUFFER &&
          arguments[j].buffer_reference.allocation_value_id ==
              argument->buffer_reference.allocation_value_id) {
        root_ordinal = product->argument_root_ordinals[j];
        break;
      }
    }
    if (root_ordinal == LOOM_WASM_TESTBENCH_NO_ROOT) {
      root_ordinal = root_count++;
    }
    product->argument_root_ordinals[i] = root_ordinal;
  }
  product->root_count = root_count;
  product->topology_initialized = true;
}

static iree_status_t loom_wasm_testbench_scalar_to_bits(
    const loom_testbench_value_t* value,
    loom_wasm_host_value_type_t physical_type, uint64_t* out_bits) {
  if (value->kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Wasm scalar parameter received a non-scalar value");
  }
  *out_bits = 0;
  switch (physical_type) {
    case LOOM_WASM_HOST_VALUE_TYPE_I32:
      switch (value->scalar.kind) {
        case IREE_TOOLING_VALUE_KIND_I32:
          *out_bits = (uint32_t)value->scalar.storage.i32;
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_U32:
        case IREE_TOOLING_VALUE_KIND_RAW_U32:
          *out_bits = value->scalar.storage.u32;
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_I64:
          *out_bits = (uint32_t)value->scalar.storage.i64;
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_U64:
          *out_bits = (uint32_t)value->scalar.storage.u64;
          return iree_ok_status();
        default:
          break;
      }
      break;
    case LOOM_WASM_HOST_VALUE_TYPE_I64:
      switch (value->scalar.kind) {
        case IREE_TOOLING_VALUE_KIND_I32:
          *out_bits = (uint64_t)(int64_t)value->scalar.storage.i32;
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_U32:
        case IREE_TOOLING_VALUE_KIND_RAW_U32:
          *out_bits = value->scalar.storage.u32;
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_I64:
          memcpy(out_bits, &value->scalar.storage.i64, sizeof(*out_bits));
          return iree_ok_status();
        case IREE_TOOLING_VALUE_KIND_U64:
          *out_bits = value->scalar.storage.u64;
          return iree_ok_status();
        default:
          break;
      }
      break;
    case LOOM_WASM_HOST_VALUE_TYPE_F32:
      if (value->scalar.kind == IREE_TOOLING_VALUE_KIND_F32) {
        uint32_t bits = 0;
        memcpy(&bits, &value->scalar.storage.f32, sizeof(bits));
        *out_bits = bits;
        return iree_ok_status();
      }
      break;
    case LOOM_WASM_HOST_VALUE_TYPE_F64:
      if (value->scalar.kind == IREE_TOOLING_VALUE_KIND_F64) {
        memcpy(out_bits, &value->scalar.storage.f64, sizeof(*out_bits));
        return iree_ok_status();
      }
      break;
    default:
      break;
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "testbench scalar kind %u cannot marshal to Wasm type 0x%02X",
      (unsigned)value->scalar.kind, (unsigned)physical_type);
}

static iree_status_t loom_wasm_testbench_scalar_from_bits(
    loom_scalar_type_t scalar_type, loom_wasm_host_value_type_t physical_type,
    uint64_t bits, loom_testbench_value_t* out_value) {
  *out_value = (loom_testbench_value_t){
      .kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR,
  };
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
      out_value->scalar.storage.i32 = (uint32_t)bits != 0;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I8:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
      out_value->scalar.storage.i32 = (int8_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I16:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
      out_value->scalar.storage.i32 = (int16_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I32:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
      out_value->scalar.storage.i32 = (int32_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_INDEX:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
      out_value->scalar.storage.i64 =
          physical_type == LOOM_WASM_HOST_VALUE_TYPE_I32 ? (int32_t)bits
                                                         : (int64_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_OFFSET:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
      out_value->scalar.storage.i64 =
          physical_type == LOOM_WASM_HOST_VALUE_TYPE_I32 ? (uint32_t)bits
                                                         : (int64_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I64:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_I64;
      memcpy(&out_value->scalar.storage.i64, &bits, sizeof(bits));
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_RAW_U32;
      out_value->scalar.storage.u32 = (uint32_t)bits;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F32: {
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_F32;
      const uint32_t raw_bits = (uint32_t)bits;
      memcpy(&out_value->scalar.storage.f32, &raw_bits, sizeof(raw_bits));
      return iree_ok_status();
    }
    case LOOM_SCALAR_TYPE_F64:
      out_value->scalar.kind = IREE_TOOLING_VALUE_KIND_F64;
      memcpy(&out_value->scalar.storage.f64, &bits, sizeof(bits));
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Wasm scenario scalar result type %u is unsupported",
          (unsigned)scalar_type);
  }
}

static bool loom_wasm_testbench_buffer_references_equal(
    const loom_testbench_buffer_reference_t* lhs,
    const loom_testbench_buffer_reference_t* rhs) {
  return lhs->is_traceable == rhs->is_traceable &&
         lhs->allocation_value_id == rhs->allocation_value_id &&
         lhs->byte_offset == rhs->byte_offset &&
         lhs->byte_length == rhs->byte_length;
}

static iree_status_t loom_wasm_testbench_buffer_from_bits(
    const loom_wasm_testbench_product_t* product,
    const loom_testbench_product_call_t* call,
    loom_wasm_host_value_type_t physical_type, uint64_t bits,
    loom_testbench_value_t* out_value) {
  *out_value = (loom_testbench_value_t){0};
  uint64_t address = 0;
  if (physical_type == LOOM_WASM_HOST_VALUE_TYPE_I32) {
    address = (uint32_t)bits;
  } else if (physical_type == LOOM_WASM_HOST_VALUE_TYPE_I64) {
    address = bits;
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Wasm buffer result has non-integer physical type 0x%02X",
        (unsigned)physical_type);
  }
  if (address > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Wasm buffer result address exceeds Wasm32");
  }

  const loom_testbench_value_t* match = NULL;
  for (uint32_t i = 0; i < product->parameter_count; ++i) {
    if (product->argument_root_ordinals[i] == LOOM_WASM_TESTBENCH_NO_ROOT ||
        product->argument_bits[i] != address) {
      continue;
    }
    const loom_testbench_value_t* argument = &call->arguments[i];
    if (match != NULL &&
        !loom_wasm_testbench_buffer_references_equal(
            &match->buffer_reference, &argument->buffer_reference)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Wasm buffer result address 0x%08" PRIX64
                              " names input bindings with different extents",
                              address);
    }
    match = argument;
  }
  if (match == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm buffer result address 0x%08" PRIX64
        " does not identify an input binding with a retained extent",
        address);
  }
  loom_testbench_value_retain(match, out_value);
  return iree_ok_status();
}

static iree_status_t loom_wasm_testbench_unmap_roots(
    loom_wasm_testbench_product_t* product, uint32_t mapped_count) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < mapped_count; ++i) {
    status = iree_status_join(
        status, iree_hal_buffer_unmap_range(&product->root_mappings[i]));
  }
  return status;
}

static iree_status_t loom_wasm_testbench_execute_call(
    loom_wasm_testbench_product_t* product,
    loom_testbench_product_call_t* call) {
  if (!product->topology_initialized) {
    loom_wasm_testbench_initialize_topology(product, call->arguments);
  }
  if (product->root_count != 0) {
    memset(product->root_buffers, 0,
           product->root_count * sizeof(*product->root_buffers));
  }
  for (uint32_t i = 0; i < product->parameter_count; ++i) {
    const uint32_t root_ordinal = product->argument_root_ordinals[i];
    if (root_ordinal == LOOM_WASM_TESTBENCH_NO_ROOT ||
        product->root_buffers[root_ordinal] != NULL) {
      continue;
    }
    product->root_buffers[root_ordinal] =
        iree_hal_buffer_allocated_buffer(call->arguments[i].buffer.buffer);
  }

  iree_status_t status = iree_ok_status();
  const bool has_memory = iree_any_bit_set(
      product->host_module.flags, LOOM_WASM_HOST_MODULE_FLAG_HAS_MEMORY);
  uint64_t next_address = 0;
  uint32_t mapped_count = 0;
  for (uint32_t i = 0; i < product->root_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_t* root = product->root_buffers[i];
    const iree_device_size_t byte_length = iree_hal_buffer_byte_length(root);
    next_address = (next_address + LOOM_WASM_TESTBENCH_ROOT_ALIGNMENT - 1) &
                   ~(uint64_t)(LOOM_WASM_TESTBENCH_ROOT_ALIGNMENT - 1);
    if (byte_length > UINT32_MAX || next_address > UINT32_MAX ||
        byte_length > UINT32_MAX - next_address) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Wasm scenario roots exceed Wasm32 memory");
      break;
    }
    product->root_regions[i] = (loom_wasm_host_memory_region_t){
        .address = (uint32_t)next_address,
        .data_length = (uint32_t)byte_length,
    };
    if (has_memory) {
      status = iree_hal_buffer_map_range(
          root, IREE_HAL_MAPPING_MODE_SCOPED,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
          IREE_HAL_BUFFER_MAP_FLAG_MAY_ALIAS, 0, IREE_HAL_WHOLE_BUFFER,
          &product->root_mappings[i]);
      if (iree_status_is_ok(status)) {
        ++mapped_count;
        product->root_regions[i].data = product->root_mappings[i].contents.data;
      }
    }
    next_address += byte_length != 0 ? byte_length : 1;
  }

  for (uint32_t i = 0;
       i < product->parameter_count && iree_status_is_ok(status); ++i) {
    const uint32_t root_ordinal = product->argument_root_ordinals[i];
    if (root_ordinal == LOOM_WASM_TESTBENCH_NO_ROOT) {
      status = loom_wasm_testbench_scalar_to_bits(&call->arguments[i],
                                                  product->parameter_types[i],
                                                  &product->argument_bits[i]);
      continue;
    }
    const iree_tooling_buffer_binding_t* binding = &call->arguments[i].buffer;
    iree_device_size_t byte_offset = 0;
    if (!iree_device_size_checked_add(
            iree_hal_buffer_byte_offset(binding->buffer), binding->byte_offset,
            &byte_offset)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Wasm buffer parameter %u physical offset overflows", i);
      continue;
    }
    const uint64_t address =
        product->root_regions[root_ordinal].address + byte_offset;
    if (address > UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Wasm buffer parameter %u address overflows Wasm32", i);
      continue;
    }
    product->argument_bits[i] = address;
  }

  if (iree_status_is_ok(status)) {
    status = loom_wasm_host_module_call(
        &product->host_module, product->argument_bits, product->result_bits,
        has_memory ? product->root_count : 0, product->root_regions);
  }
  for (uint32_t i = 0; i < product->result_count && iree_status_is_ok(status);
       ++i) {
    if (product->result_scalar_types[i] != LOOM_SCALAR_TYPE_NONE) {
      status = loom_wasm_testbench_scalar_from_bits(
          product->result_scalar_types[i], product->result_types[i],
          product->result_bits[i], &call->results[i]);
    } else {
      status = loom_wasm_testbench_buffer_from_bits(
          product, call, product->result_types[i], product->result_bits[i],
          &call->results[i]);
    }
  }
  return iree_status_join(
      status, loom_wasm_testbench_unmap_roots(product, mapped_count));
}

static iree_status_t loom_wasm_testbench_product_execute(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  loom_wasm_testbench_product_t* product = user_data;
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    iree_status_t status = loom_wasm_testbench_execute_call(product, &calls[i]);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(
          status,
          "executing Wasm scenario trial configuration %zu domain %zu "
          "ordinal %zu",
          calls[i].identity->configuration_ordinal,
          calls[i].identity->trial_index, calls[i].identity->trial_ordinal);
    }
  }
  return iree_ok_status();
}

static void loom_wasm_testbench_product_destroy(void* user_data) {
  loom_wasm_testbench_product_t* product = user_data;
  const iree_allocator_t host_allocator = product->host_allocator;
  loom_wasm_host_module_release(&product->host_module);
  iree_allocator_free(host_allocator, product);
}

static const loom_testbench_product_vtable_t
    loom_wasm_testbench_product_vtable = {
        .execute = loom_wasm_testbench_product_execute,
};

static iree_status_t loom_wasm_testbench_product_prepare(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL ||
      invocation->workload_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm scenario profile requires an ordinary function subject");
  }

  iree_string_view_t function_name = iree_string_view_empty();
  loom_func_like_t source_function = {0};
  IREE_RETURN_IF_ERROR(loom_wasm_testbench_resolve_source_function(
      invocation, &function_name, &source_function));
  loom_wasm_testbench_product_t* product = NULL;
  IREE_RETURN_IF_ERROR(loom_wasm_testbench_compile_product(
      user_data, invocation, function_name, source_function, host_allocator,
      &product));
  *out_product = (loom_testbench_prepared_product_t){
      .vtable = &loom_wasm_testbench_product_vtable,
      .destroy = loom_wasm_testbench_product_destroy,
      .user_data = product,
  };
  return iree_ok_status();
}

loom_testbench_execution_profile_t loom_wasm_testbench_execution_profile(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback) {
  (void)sources;
  (void)diagnostic_sink;
  loom_wasm_testbench_t* testbench = user_data;
  testbench->compilation = *compilation;
  testbench->result_callback = result_callback;
  return (loom_testbench_execution_profile_t){
      .name = IREE_SV("wasm:simd128"),
      .prepare = loom_wasm_testbench_product_prepare,
      .user_data = testbench,
  };
}
