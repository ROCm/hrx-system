// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/scenario_command_program.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/cmd/hal_command_buffer.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/testbench_staging.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/cmd/program.h"

typedef struct loom_run_hal_command_scenario_requirement_t {
  // Loaded executable and function satisfying this parent requirement.
  loom_cmd_program_hal_entry_t entry;
  // True after exactly one published request binds |entry|.
  bool assigned;
} loom_run_hal_command_scenario_requirement_t;

typedef struct loom_run_hal_command_scenario_product_t {
  // Host allocator owning every array and this product.
  iree_allocator_t host_allocator;
  // HAL profile and compiler services retained by the owning runner.
  loom_run_hal_testbench_scenario_profile_t* profile;
  // Authored command-program invocation accepted by this product.
  const loom_testbench_invocation_plan_t* subject;
  // Target-only clone containing the private frozen-specialization wrapper.
  loomc_module_t* adapted_module;
  // Scratch workspace used only by synchronous published-request compilation.
  loomc_workspace_t* request_workspace;
  // Device-selected target profile used for command and kernel products.
  loomc_target_profile_t* selected_target_profile;
  // Exact loader target borrowed from the live HAL device.
  const iree_hal_executable_target_t* executable_target;
  // Contiguous command artifact bytes owned by this product.
  iree_byte_span_t program_image;
  // Validated view borrowing |program_image|.
  loom_cmd_program_t program;
  // Program-local entries projected from product-wide requirements.
  loom_cmd_program_hal_entry_t* entries;
  // Number of entries in |entries|.
  iree_host_size_t entry_count;
  // Product-wide entries populated by published request bindings.
  loom_run_hal_command_scenario_requirement_t* requirements;
  // Allocated entry count in |requirements|.
  iree_host_size_t requirement_capacity;
  // Loaded executables retained for command-buffer recording.
  iree_hal_executable_t** executables;
  // Number of initialized entries in |executables|.
  iree_host_size_t executable_count;
  // Allocated entry count in |executables|.
  iree_host_size_t executable_capacity;
  // Number of physical kernel dispatches in one complete program execution.
  iree_host_size_t dispatch_count;
} loom_run_hal_command_scenario_product_t;

static iree_status_t loom_run_hal_command_scenario_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(allocator, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_reserve_requirements(
    loom_run_hal_command_scenario_product_t* product,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= product->requirement_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = iree_max(product->requirement_capacity, 8u);
  while (new_capacity < minimum_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = minimum_capacity;
      break;
    }
    new_capacity *= 2;
  }
  const iree_host_size_t previous_capacity = product->requirement_capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      product->host_allocator, new_capacity, sizeof(*product->requirements),
      (void**)&product->requirements));
  memset(product->requirements + previous_capacity, 0,
         (new_capacity - previous_capacity) * sizeof(*product->requirements));
  product->requirement_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_retain_executable(
    loom_run_hal_command_scenario_product_t* product,
    iree_hal_executable_t* executable) {
  if (product->executable_count == product->executable_capacity) {
    iree_host_size_t new_capacity =
        iree_max(product->executable_capacity * 2, 4u);
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        product->host_allocator, new_capacity, sizeof(*product->executables),
        (void**)&product->executables));
    product->executable_capacity = new_capacity;
  }
  product->executables[product->executable_count++] = executable;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command scenario subject has an invalid symbol");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario subject has an invalid symbol name");
  }
  *out_name = loom_string_table_get(&module->strings, symbol->name_id);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_find_program(
    loom_module_t* module, iree_string_view_t name, loom_symbol_ref_t* out_ref,
    loom_func_like_t* out_program) {
  *out_ref = loom_symbol_ref_null();
  *out_program = (loom_func_like_t){0};
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  const loom_symbol_id_t symbol_id =
      name_id == LOOM_STRING_ID_INVALID
          ? LOOM_SYMBOL_ID_INVALID
          : loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "command scenario program '@%.*s' was not cloned",
                            (int)name.size, name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!loom_command_def_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario subject '@%.*s' is not a command.def", (int)name.size,
        name.data);
  }
  *out_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  *out_program = loom_func_like_cast(module, symbol->defining_op);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_add_wrapper_symbol(
    loom_module_t* module, loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  for (iree_host_size_t ordinal = module->symbols.count;; ++ordinal) {
    char name_buffer[72];
    const int name_length =
        iree_snprintf(name_buffer, sizeof(name_buffer),
                      "__loom_command_scenario_%zu", ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command scenario symbol name is too long");
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module,
        iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
        &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }
}

static iree_status_t loom_run_hal_command_scenario_integer_attribute(
    loom_scalar_type_t type, const loom_testbench_value_t* value,
    loom_attribute_t* out_attribute) {
  if (value->kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command specialization is not a scalar");
  }
  switch (type) {
    case LOOM_SCALAR_TYPE_I1:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_I32 ||
          (value->scalar.storage.i32 != 0 && value->scalar.storage.i32 != 1)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "i1 specialization is not zero or one");
      }
      *out_attribute = loom_attr_bool(value->scalar.storage.i32 != 0);
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I8:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_I32 ||
          value->scalar.storage.i32 < INT8_MIN ||
          value->scalar.storage.i32 > INT8_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "i8 specialization is out of range");
      }
      *out_attribute = loom_attr_i64(value->scalar.storage.i32);
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I16:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_I32 ||
          value->scalar.storage.i32 < INT16_MIN ||
          value->scalar.storage.i32 > INT16_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "i16 specialization is out of range");
      }
      *out_attribute = loom_attr_i64(value->scalar.storage.i32);
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I32:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_I32) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "i32 specialization has the wrong value kind");
      }
      *out_attribute = loom_attr_i64(value->scalar.storage.i32);
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_I64 ||
          (type == LOOM_SCALAR_TYPE_OFFSET && value->scalar.storage.i64 < 0)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "64-bit command specialization has an invalid value");
      }
      *out_attribute = loom_attr_i64(value->scalar.storage.i64);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported integer specialization type %u",
                              (unsigned)type);
  }
}

static iree_status_t loom_run_hal_command_scenario_float_bits(
    loom_scalar_type_t type, const loom_testbench_value_t* value,
    loom_scalar_type_t* out_integer_type, int64_t* out_signed_bits) {
  if (value->kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command specialization is not a scalar");
  }

  uint64_t bits = 0;
  int32_t bit_count = 0;
  switch (type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_RAW_U32 ||
          value->scalar.storage.u32 > UINT8_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "f8 specialization has invalid bits");
      }
      bits = value->scalar.storage.u32;
      bit_count = 8;
      *out_integer_type = LOOM_SCALAR_TYPE_I8;
      break;
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_RAW_U32 ||
          value->scalar.storage.u32 > UINT16_MAX) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "16-bit float specialization has invalid bits");
      }
      bits = value->scalar.storage.u32;
      bit_count = 16;
      *out_integer_type = LOOM_SCALAR_TYPE_I16;
      break;
    case LOOM_SCALAR_TYPE_F32: {
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_F32) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "f32 specialization has the wrong value kind");
      }
      uint32_t raw_bits = 0;
      memcpy(&raw_bits, &value->scalar.storage.f32, sizeof(raw_bits));
      bits = raw_bits;
      bit_count = 32;
      *out_integer_type = LOOM_SCALAR_TYPE_I32;
      break;
    }
    case LOOM_SCALAR_TYPE_F64:
      if (value->scalar.kind != IREE_TOOLING_VALUE_KIND_F64) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "f64 specialization has the wrong value kind");
      }
      memcpy(&bits, &value->scalar.storage.f64, sizeof(bits));
      bit_count = 64;
      *out_integer_type = LOOM_SCALAR_TYPE_I64;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported float specialization type %u",
                              (unsigned)type);
  }

  if (bit_count == 64) {
    memcpy(out_signed_bits, &bits, sizeof(*out_signed_bits));
  } else {
    const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
    *out_signed_bits =
        (int64_t)bits - ((bits & sign_bit) ? (INT64_C(1) << bit_count) : 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_build_constant(
    loom_builder_t* builder, loom_type_t type,
    const loom_testbench_value_t* value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (loom_scalar_type_is_float(scalar_type)) {
    loom_scalar_type_t integer_scalar_type = LOOM_SCALAR_TYPE_NONE;
    int64_t signed_bits = 0;
    IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_float_bits(
        scalar_type, value, &integer_scalar_type, &signed_bits));
    const loom_type_t integer_type = loom_type_scalar(integer_scalar_type);
    loom_op_t* bits_constant = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_constant_build(builder, loom_attr_i64(signed_bits),
                                   integer_type, location, &bits_constant));
    loom_op_t* bitcast = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, loom_scalar_constant_result(bits_constant), integer_type, type,
        location, &bitcast));
    *out_value = loom_scalar_bitcast_result(bitcast);
    return iree_ok_status();
  }

  loom_attribute_t attribute = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_integer_attribute(
      scalar_type, value, &attribute));
  loom_op_t* constant = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attribute, type,
                                                   location, &constant));
    *out_value = loom_index_constant_result(constant);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attribute, type,
                                                    location, &constant));
    *out_value = loom_scalar_constant_result(constant);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_build_wrapper(
    loom_run_hal_command_scenario_product_t* product,
    const loom_testbench_value_table_t* configuration,
    loomc_string_view_t* out_root_symbol) {
  *out_root_symbol = loomc_string_view_empty();
  const loom_run_hal_testbench_actual_provider_options_t* provider_options =
      &product->profile->provider_options;
  const loom_testbench_invocation_plan_t* invocation = product->subject;
  iree_string_view_t program_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_symbol_name_from_ref(
      provider_options->native_module, invocation->callee_ref, &program_name));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_clone(
      provider_options->module, provider_options->compilation->workspace,
      loomc_allocator_from_iree(product->host_allocator),
      &product->adapted_module)));
  const loomc_module_mutable_interop_view_t mutable_view =
      loomc_module_get_mutable_interop_view(product->adapted_module);
  if (mutable_view.module == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command scenario module has no exact-version native IR");
  }
  loom_module_t* module = mutable_view.module;

  loom_symbol_ref_t program_ref = loom_symbol_ref_null();
  loom_func_like_t program = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_find_program(
      module, program_name, &program_ref, &program));
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(program);
  if (specialization_count_i64 < 0 ||
      specialization_count_i64 > argument_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command program '@%.*s' has an invalid specialization count",
        (int)program_name.size, program_name.data);
  }
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;
  const uint16_t binding_count = argument_count - specialization_count;
  if (invocation->workload_count != specialization_count ||
      invocation->input_count != binding_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command scenario plan has %" PRIhsz " specializations and %" PRIhsz
        " bindings for a signature with %u and %u",
        invocation->workload_count, invocation->input_count,
        (unsigned)specialization_count, (unsigned)binding_count);
  }

  loom_type_t* binding_types = NULL;
  if (binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, binding_count, sizeof(*binding_types),
        (void**)&binding_types));
  }
  for (uint16_t i = 0; i < specialization_count; ++i) {
    if (!loom_type_is_scalar(loom_module_value_type(module, arguments[i]))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command program specialization %u is not scalar",
                              (unsigned)i);
    }
  }
  for (uint16_t i = 0; i < binding_count; ++i) {
    binding_types[i] =
        loom_module_value_type(module, arguments[specialization_count + i]);
    if (!loom_type_is_buffer(binding_types[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command program binding %u is not a buffer",
                              (unsigned)i);
    }
  }

  loom_symbol_ref_t wrapper_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_run_hal_command_scenario_add_wrapper_symbol(module, &wrapper_ref));
  const loom_symbol_ref_t target = loom_func_like_target(program);
  loom_command_def_build_flags_t build_flags = 0;
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_COMMAND_DEF_BUILD_FLAG_HAS_TARGET;
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* wrapper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_command_def_build(
      &builder, build_flags, /*visibility=*/0, /*retain=*/0, target,
      wrapper_ref,
      /*specializations_types=*/NULL,
      /*specializations_types_count=*/0, binding_types, binding_count,
      /*predicates=*/NULL, /*predicates_count=*/0, program.op->location,
      &wrapper_op));
  const loom_func_like_t wrapper = loom_func_like_cast(module, wrapper_op);
  uint16_t wrapper_argument_count = 0;
  const loom_value_id_t* wrapper_arguments =
      loom_func_like_arg_ids(wrapper, &wrapper_argument_count);
  if (wrapper_argument_count != binding_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "generated command scenario wrapper has an inconsistent signature");
  }

  loom_value_id_t* specialization_values = NULL;
  if (specialization_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, specialization_count, sizeof(*specialization_values),
        (void**)&specialization_values));
  }
  const loom_builder_ip_t module_ip = loom_builder_enter_region(
      &builder, wrapper_op, loom_command_def_body(wrapper_op));
  // The wrapper has no specialization arguments and therefore owns no scalar
  // predicates. The launched program retains its predicate contract, which is
  // proved against the concrete constants materialized below.
  for (uint16_t i = 0; i < specialization_count; ++i) {
    const loom_testbench_value_t* value = NULL;
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_borrow(
        configuration, invocation->workload_value_ids[i], &value));
    IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_build_constant(
        &builder, loom_module_value_type(module, arguments[i]), value,
        program.op->location, &specialization_values[i]));
  }
  loom_op_t* launch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_command_launch_build(
      &builder, program_ref, specialization_values, specialization_count,
      wrapper_arguments, wrapper_argument_count, program.op->location,
      &launch_op));
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_command_return_build(&builder, program.op->location, &return_op));
  loom_builder_restore(&builder, module_ip);

  const loom_symbol_t* wrapper_symbol =
      &module->symbols.entries[wrapper_ref.symbol_id];
  *out_root_symbol = loomc_string_view_from_iree(
      loom_string_table_get(&module->strings, wrapper_symbol->name_id));
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_observe_result(
    loom_run_hal_command_scenario_product_t* product,
    const loomc_result_t* result, iree_string_view_t failure_message) {
  IREE_RETURN_IF_ERROR(loom_testbench_observe_compile_result(
      product->profile->provider_options.result_callback, result));
  return loom_run_hal_testbench_require_successful_result(result,
                                                          failure_message);
}

static iree_status_t loom_run_hal_command_scenario_select_executable_artifact(
    const loomc_result_t* result, const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      continue;
    }
    if (*out_artifact != NULL) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "kernel request returned multiple executables");
    }
    *out_artifact = artifact;
  }
  if (*out_artifact == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "kernel request returned no executable");
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_compile_request(
    loom_run_hal_command_scenario_product_t* product,
    loomc_request_t* request) {
  const loom_run_hal_testbench_actual_provider_options_t* provider_options =
      &product->profile->provider_options;
  const loom_testbench_compilation_t* compilation =
      provider_options->compilation;
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(product->host_allocator);
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  if (root_count == 0 || loomc_request_product_descriptor(request) !=
                             loomc_compiled_module_product_descriptor()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command product published an invalid kernel compilation request");
  }

  loomc_string_view_t* roots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      product->host_allocator, root_count, sizeof(*roots), (void**)&roots));
  memset(roots, 0, root_count * sizeof(*roots));
  loomc_string_view_t* owned_roots = NULL;
  iree_status_t status =
      iree_allocator_malloc_array(product->host_allocator, root_count,
                                  sizeof(*owned_roots), (void**)&owned_roots);
  if (iree_status_is_ok(status)) {
    memset(owned_roots, 0, root_count * sizeof(*owned_roots));
  }
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_request(
        compilation->context, product->request_workspace, request, allocator,
        &module, roots, &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_observe_result(
        product, result, IREE_SV("kernel request deserialization failed"));
  }
  loomc_result_release(result);
  result = NULL;

  loomc_host_size_t owned_root_count = 0;
  for (; iree_status_is_ok(status) && owned_root_count < root_count;
       ++owned_root_count) {
    status = iree_status_from_loomc(loomc_string_view_clone(
        roots[owned_root_count], allocator, &owned_roots[owned_root_count]));
  }
  iree_allocator_free(product->host_allocator, roots);
  roots = NULL;
  loomc_artifact_manifest_options_t artifact_manifest =
      provider_options->artifact_manifest;
  const void* emit_option_chain = NULL;
  if (artifact_manifest.mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE) {
    artifact_manifest.type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS;
    artifact_manifest.structure_size = sizeof(artifact_manifest);
    artifact_manifest.next = emit_option_chain;
    emit_option_chain = &artifact_manifest;
  }
  loomc_compile_report_options_t compile_report =
      provider_options->compile_report;
  if (compile_report.mode != LOOMC_COMPILE_REPORT_MODE_NONE) {
    compile_report.type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS;
    compile_report.structure_size = sizeof(compile_report);
    compile_report.next = emit_option_chain;
    emit_option_chain = &compile_report;
  }
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = emit_option_chain,
      .artifact_flags = provider_options->emit_artifact_flags |
                        LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = provider_options->sanitizer,
      .roots = owned_roots,
      .root_count = root_count,
      .target_profile = product->selected_target_profile,
      .config = compilation->config,
      .emit_options = &emit_options,
  };
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compile_artifact(
        compilation->compiler, product->request_workspace,
        provider_options->pass_program, module, &compile_options, allocator,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_observe_result(
        product, result, IREE_SV("kernel request compilation failed"));
  }

  const loomc_artifact_t* executable_artifact = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_select_executable_artifact(
        result, &executable_artifact);
  }
  iree_hal_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    const loom_device_artifact_t device_artifact = {
        .executable_target = product->executable_target,
        .contents =
            iree_byte_sequence_from_loomc(executable_artifact->contents),
    };
    status = loom_run_hal_artifact_prepare(
        &provider_options->context->runtime, &device_artifact,
        product->host_allocator, &executable);
  }

  iree_hal_executable_function_t* functions = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, root_count, sizeof(*functions),
        (void**)&functions);
  }
  for (loomc_host_size_t i = 0; iree_status_is_ok(status) && i < root_count;
       ++i) {
    status = iree_hal_executable_lookup_function_by_name(
        executable, iree_string_view_from_loomc(owned_roots[i]), &functions[i]);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_command_scenario_retain_executable(product, executable);
    if (iree_status_is_ok(status)) {
      executable = NULL;
    }
  }

  const loomc_host_size_t binding_count = loomc_request_binding_count(request);
  for (loomc_host_size_t i = 0; iree_status_is_ok(status) && i < binding_count;
       ++i) {
    loomc_request_binding_t binding = {0};
    if (!loomc_request_binding_at(request, i, &binding) ||
        binding.root_ordinal >= root_count ||
        binding.requirement_ordinal == LOOMC_REQUIREMENT_ORDINAL_INVALID) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "kernel request contains an invalid parent binding");
      break;
    }
    status = loom_run_hal_command_scenario_reserve_requirements(
        product, (iree_host_size_t)binding.requirement_ordinal + 1);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_run_hal_command_scenario_requirement_t* requirement =
        &product->requirements[binding.requirement_ordinal];
    if (requirement->assigned) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command entry requirement %u has multiple implementations",
          binding.requirement_ordinal);
      break;
    }
    requirement->entry = (loom_cmd_program_hal_entry_t){
        .executable = product->executables[product->executable_count - 1],
        .function = functions[binding.root_ordinal],
    };
    requirement->assigned = true;
  }

  iree_allocator_free(product->host_allocator, functions);
  iree_hal_executable_release(executable);
  loomc_result_release(result);
  loomc_module_release(module);
  for (loomc_host_size_t i = 0; i < owned_root_count; ++i) {
    loomc_allocator_free(allocator, (void*)owned_roots[i].data);
  }
  iree_allocator_free(product->host_allocator, owned_roots);
  return status;
}

static loomc_status_t LOOMC_API_PTR loom_run_hal_command_scenario_publish(
    void* user_data, loomc_request_t* request) {
  loom_run_hal_command_scenario_product_t* product =
      (loom_run_hal_command_scenario_product_t*)user_data;
  iree_status_t status =
      loom_run_hal_command_scenario_compile_request(product, request);
  loomc_request_release(request);
  return loomc_status_from_iree(status);
}

static void loom_run_hal_command_scenario_product_destroy(void* user_data) {
  loom_run_hal_command_scenario_product_t* product =
      (loom_run_hal_command_scenario_product_t*)user_data;
  if (product == NULL) {
    return;
  }
  const iree_allocator_t host_allocator = product->host_allocator;
  for (iree_host_size_t i = product->executable_count; i > 0; --i) {
    iree_hal_executable_release(product->executables[i - 1]);
  }
  iree_allocator_free(host_allocator, product->executables);
  iree_allocator_free(host_allocator, product->requirements);
  iree_allocator_free(host_allocator, product->entries);
  iree_allocator_free(host_allocator, product->program_image.data);
  loomc_target_profile_release(product->selected_target_profile);
  loomc_workspace_release(product->request_workspace);
  loomc_module_release(product->adapted_module);
  iree_allocator_free(host_allocator, product);
}

typedef struct loom_run_hal_command_scenario_calls_t {
  // Host allocator owning every scratch array.
  iree_allocator_t host_allocator;
  // Authored source-order bindings for every call, rewritten by staging.
  iree_hal_buffer_binding_t* source_bindings;
  // Fixed bindings projected for every recorded command buffer.
  iree_hal_buffer_binding_t* fixed_bindings;
  // Rebindable bindings projected for every queue submission.
  iree_hal_buffer_binding_t* rebindable_bindings;
  // Per-call buffers backing compiler-declared transient storage.
  iree_hal_buffer_t** transient_buffers;
  // Per-call command buffers recorded with concrete fixed roots.
  iree_hal_command_buffer_t** command_buffers;
  // Optional per-call command buffers retaining final-profile metadata.
  iree_hal_command_buffer_t** profile_command_buffers;
  // Per-call completion semaphores avoiding queue-order assumptions.
  iree_hal_semaphore_t** completion_semaphores;
  // Per-call flags identifying successfully accepted submissions.
  bool* submitted;
  // Alias-preserving host-to-device staging for all authored bindings.
  loom_run_hal_testbench_staging_t staging;
  // Number of physical calls represented by the arrays.
  iree_host_size_t call_count;
  // Number of authored bindings materialized for each physical call.
  iree_host_size_t binding_count_per_call;
  // Sum of authored binding byte lengths in the first physical call.
  uint64_t binding_set_bytes;
  // Sum of authored binding byte lengths across all physical calls.
  uint64_t binding_ring_bytes;
  // Number of disjoint reachable ranges in the first physical call.
  iree_host_size_t binding_range_count_per_call;
  // Sum of reachable range bytes in the first physical call.
  uint64_t binding_range_set_bytes;
  // Sum of reachable range bytes across all physical calls.
  uint64_t binding_range_ring_bytes;
  // Number of unique input allocations in the first physical call.
  iree_host_size_t materialized_storage_count_per_call;
  // Sum of unique input-allocation bytes in the first physical call.
  uint64_t materialized_storage_set_bytes;
  // Sum of unique input-allocation bytes across all physical calls.
  uint64_t materialized_storage_ring_bytes;
} loom_run_hal_command_scenario_calls_t;

typedef uint32_t loom_run_hal_command_scenario_call_flags_t;
enum loom_run_hal_command_scenario_call_flag_bits_e {
  LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_NONE = 0u,
  // Creates one correctness completion semaphore per physical call.
  LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_COMPLETION_SEMAPHORES = 1u << 0,
  // Records an equivalent metadata-retaining command buffer per call.
  LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_PROFILE_COMMAND_BUFFERS = 1u << 1,
  // Requires every physical call to have the first call's input footprint.
  LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_UNIFORM_FOOTPRINT = 1u << 2,
};

static iree_status_t loom_run_hal_command_scenario_calls_deinitialize(
    loom_run_hal_command_scenario_calls_t* calls) {
  for (iree_host_size_t i = calls->call_count; i > 0; --i) {
    iree_hal_semaphore_release(calls->completion_semaphores
                                   ? calls->completion_semaphores[i - 1]
                                   : NULL);
    iree_hal_command_buffer_release(
        calls->command_buffers ? calls->command_buffers[i - 1] : NULL);
    iree_hal_command_buffer_release(calls->profile_command_buffers
                                        ? calls->profile_command_buffers[i - 1]
                                        : NULL);
    iree_hal_buffer_release(
        calls->transient_buffers ? calls->transient_buffers[i - 1] : NULL);
  }
  iree_status_t status =
      loom_run_hal_testbench_staging_deinitialize(&calls->staging);
  iree_allocator_free(calls->host_allocator, calls->submitted);
  iree_allocator_free(calls->host_allocator, calls->completion_semaphores);
  iree_allocator_free(calls->host_allocator, calls->profile_command_buffers);
  iree_allocator_free(calls->host_allocator, calls->command_buffers);
  iree_allocator_free(calls->host_allocator, calls->transient_buffers);
  iree_allocator_free(calls->host_allocator, calls->rebindable_bindings);
  iree_allocator_free(calls->host_allocator, calls->fixed_bindings);
  iree_allocator_free(calls->host_allocator, calls->source_bindings);
  *calls = (loom_run_hal_command_scenario_calls_t){0};
  return status;
}

static iree_status_t loom_run_hal_command_scenario_validate_call(
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_product_call_t* call, uint64_t* out_binding_bytes) {
  *out_binding_bytes = 0;
  for (iree_host_size_t i = 0; i < invocation->input_count; ++i) {
    const loom_testbench_value_t* value = &call->arguments[i];
    if (value->kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER ||
        value->buffer.buffer == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command scenario input %" PRIhsz " is not a materialized buffer", i);
    }
    if (UINT64_MAX - *out_binding_bytes < value->buffer.byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command scenario binding bytes overflowed");
    }
    *out_binding_bytes += value->buffer.byte_length;
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_command_scenario_calls_initialize(
    loom_run_hal_command_scenario_product_t* product,
    iree_host_size_t call_count, loom_testbench_product_call_t* product_calls,
    iree_hal_command_buffer_mode_t command_buffer_mode,
    loom_run_hal_command_scenario_call_flags_t flags,
    loom_run_hal_command_scenario_calls_t* out_calls) {
  *out_calls = (loom_run_hal_command_scenario_calls_t){
      .host_allocator = product->host_allocator,
      .call_count = call_count,
      .binding_count_per_call = product->subject->input_count,
  };
  const loom_testbench_invocation_plan_t* invocation = product->subject;
  const loom_cmd_program_requirements_t* requirements =
      &product->program.requirements;
  if (product->program.launch_bindings.count != invocation->input_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command artifact has %u launch bindings for %" PRIhsz
        " scenario inputs",
        product->program.launch_bindings.count, invocation->input_count);
  }
  if (requirements->launch_counts.binding_index != UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command scenario cannot supply a host launch-count table");
  }

  iree_host_size_t source_binding_count = 0;
  iree_host_size_t fixed_binding_count = 0;
  iree_host_size_t rebindable_binding_count = 0;
  if (!iree_host_size_checked_mul(call_count, invocation->input_count,
                                  &source_binding_count) ||
      !iree_host_size_checked_mul(call_count, requirements->fixed_buffer_count,
                                  &fixed_binding_count) ||
      !iree_host_size_checked_mul(call_count,
                                  requirements->rebindable_binding_count,
                                  &rebindable_binding_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command scenario call storage overflowed");
  }
  iree_status_t status = loom_run_hal_command_scenario_allocate_array(
      product->host_allocator, source_binding_count,
      sizeof(*out_calls->source_bindings), (void**)&out_calls->source_bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, fixed_binding_count,
        sizeof(*out_calls->fixed_bindings), (void**)&out_calls->fixed_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, rebindable_binding_count,
        sizeof(*out_calls->rebindable_bindings),
        (void**)&out_calls->rebindable_bindings);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, call_count,
        sizeof(*out_calls->transient_buffers),
        (void**)&out_calls->transient_buffers);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, call_count,
        sizeof(*out_calls->command_buffers),
        (void**)&out_calls->command_buffers);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(
          flags,
          LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_PROFILE_COMMAND_BUFFERS)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, call_count,
        sizeof(*out_calls->profile_command_buffers),
        (void**)&out_calls->profile_command_buffers);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(
          flags,
          LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_COMPLETION_SEMAPHORES)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, call_count,
        sizeof(*out_calls->completion_semaphores),
        (void**)&out_calls->completion_semaphores);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(
          flags,
          LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_COMPLETION_SEMAPHORES)) {
    status = loom_run_hal_command_scenario_allocate_array(
        product->host_allocator, call_count, sizeof(*out_calls->submitted),
        (void**)&out_calls->submitted);
  }

  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < call_count; ++call_index) {
    uint64_t call_binding_bytes = 0;
    status = loom_run_hal_command_scenario_validate_call(
        invocation, &product_calls[call_index], &call_binding_bytes);
    const loom_testbench_buffer_footprint_t* footprint =
        &product_calls[call_index].input_footprint;
    if (iree_status_is_ok(status) && call_index == 0) {
      out_calls->binding_set_bytes = call_binding_bytes;
      out_calls->binding_range_count_per_call = footprint->binding_range_count;
      out_calls->binding_range_set_bytes = footprint->binding_range_byte_length;
      out_calls->materialized_storage_count_per_call =
          footprint->storage_allocation_count;
      out_calls->materialized_storage_set_bytes =
          footprint->storage_byte_length;
    } else if (iree_status_is_ok(status) &&
               iree_all_bits_set(
                   flags,
                   LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_UNIFORM_FOOTPRINT) &&
               !loom_testbench_buffer_footprint_equal(
                   footprint, &product_calls[0].input_footprint)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "command scenario call %" PRIhsz
                           " materialized a different input buffer footprint",
                           call_index);
    }
    if (iree_status_is_ok(status) &&
        (UINT64_MAX - out_calls->binding_ring_bytes < call_binding_bytes ||
         UINT64_MAX - out_calls->binding_range_ring_bytes <
             footprint->binding_range_byte_length ||
         UINT64_MAX - out_calls->materialized_storage_ring_bytes <
             footprint->storage_byte_length)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "command scenario input footprint byte accounting overflowed");
    }
    if (iree_status_is_ok(status)) {
      out_calls->binding_ring_bytes += call_binding_bytes;
      out_calls->binding_range_ring_bytes +=
          footprint->binding_range_byte_length;
      out_calls->materialized_storage_ring_bytes +=
          footprint->storage_byte_length;
    }
    for (iree_host_size_t input_index = 0;
         iree_status_is_ok(status) && input_index < invocation->input_count;
         ++input_index) {
      const iree_tooling_buffer_binding_t* source =
          &product_calls[call_index].arguments[input_index].buffer;
      out_calls->source_bindings[call_index * invocation->input_count +
                                 input_index] = (iree_hal_buffer_binding_t){
          .buffer = source->buffer,
          .offset = source->byte_offset,
          .length = source->byte_length,
      };
    }
  }
  loom_run_hal_testbench_context_t* context =
      product->profile->provider_options.context;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_initialize(
        &context->runtime, source_binding_count, out_calls->source_bindings,
        product->host_allocator, &out_calls->staging);
  }

  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < call_count; ++call_index) {
    iree_hal_buffer_binding_t* fixed_bindings =
        requirements->fixed_buffer_count == 0
            ? NULL
            : out_calls->fixed_bindings +
                  call_index * requirements->fixed_buffer_count;
    iree_hal_buffer_binding_t* rebindable_bindings =
        requirements->rebindable_binding_count == 0
            ? NULL
            : out_calls->rebindable_bindings +
                  call_index * requirements->rebindable_binding_count;
    const iree_hal_buffer_binding_t* source_bindings =
        invocation->input_count == 0
            ? NULL
            : out_calls->source_bindings + call_index * invocation->input_count;
    for (uint32_t i = 0; i < product->program.launch_bindings.count; ++i) {
      const loom_cmd_program_launch_binding_t placement =
          loom_cmd_program_launch_binding_at(&product->program, i);
      if (placement.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED) {
        fixed_bindings[placement.root_index] = source_bindings[i];
      } else {
        rebindable_bindings[placement.root_index] = source_bindings[i];
      }
    }
    if (requirements->transient.binding_index != UINT32_MAX) {
      status = iree_hal_pool_allocate_buffer(
          context->runtime.device_pool,
          iree_hal_make_pool_reservation_request(
              requirements->transient.required_byte_length,
              requirements->transient.minimum_alignment),
          iree_infinite_timeout(), &out_calls->transient_buffers[call_index]);
      if (iree_status_is_ok(status)) {
        rebindable_bindings[requirements->transient.binding_index] =
            (iree_hal_buffer_binding_t){
                .buffer = out_calls->transient_buffers[call_index],
                .offset = 0,
                .length = requirements->transient.required_byte_length,
            };
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_cmd_program_hal_command_buffer_create(
          iree_hal_queue_family(context->runtime.dispatch_queue),
          command_buffer_mode, &product->program,
          /*range=*/NULL, requirements->fixed_buffer_count, fixed_bindings,
          product->entry_count, product->entries, product->host_allocator,
          &out_calls->command_buffers[call_index]);
    }
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(
            flags,
            LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_PROFILE_COMMAND_BUFFERS)) {
      const iree_hal_command_buffer_mode_t profile_mode =
          (command_buffer_mode |
           IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA) &
          ~IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED;
      status = loom_cmd_program_hal_command_buffer_create(
          iree_hal_queue_family(context->runtime.dispatch_queue), profile_mode,
          &product->program,
          /*range=*/NULL, requirements->fixed_buffer_count, fixed_bindings,
          product->entry_count, product->entries, product->host_allocator,
          &out_calls->profile_command_buffers[call_index]);
    }
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(
            flags,
            LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_COMPLETION_SEMAPHORES)) {
      status = iree_hal_semaphore_create(
          context->runtime.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
          &out_calls->completion_semaphores[call_index]);
    }
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, loom_run_hal_command_scenario_calls_deinitialize(out_calls));
  }
  return status;
}

static iree_status_t loom_run_hal_command_scenario_calls_execute(
    loom_run_hal_command_scenario_product_t* product,
    loom_run_hal_command_scenario_calls_t* calls) {
  const loom_cmd_program_requirements_t* requirements =
      &product->program.requirements;
  loom_run_hal_testbench_context_t* context =
      product->profile->provider_options.context;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < calls->call_count;
       ++call_index) {
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signals = {
        .count = 1,
        .semaphores = &calls->completion_semaphores[call_index],
        .payload_values = &signal_value,
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        .count = requirements->rebindable_binding_count,
        .bindings =
            requirements->rebindable_binding_count == 0
                ? NULL
                : calls->rebindable_bindings +
                      call_index * requirements->rebindable_binding_count,
    };
    status = iree_hal_queue_execute(
        context->runtime.dispatch_queue, iree_hal_semaphore_list_empty(),
        signals, calls->command_buffers[call_index], binding_table,
        IREE_HAL_QUEUE_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME);
    calls->submitted[call_index] = iree_status_is_ok(status);
  }
  for (iree_host_size_t call_index = 0; call_index < calls->call_count;
       ++call_index) {
    if (!calls->submitted[call_index]) {
      continue;
    }
    status = iree_status_join(
        status, loom_run_hal_semaphore_wait(
                    calls->completion_semaphores[call_index], 1,
                    iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_readback(&context->runtime,
                                                     &calls->staging);
  }
  return status;
}

static iree_status_t loom_run_hal_command_scenario_product_execute(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  loom_run_hal_command_scenario_product_t* product =
      (loom_run_hal_command_scenario_product_t*)user_data;
  if (invocation != product->subject) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario invocation does not match its prepared product");
  }
  if (call_count == 0) {
    return iree_ok_status();
  }
  loom_run_hal_command_scenario_calls_t prepared_calls = {0};
  iree_status_t status = loom_run_hal_command_scenario_calls_initialize(
      product, call_count, calls, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_COMPLETION_SEMAPHORES,
      &prepared_calls);
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_command_scenario_calls_execute(product, &prepared_calls);
  }
  return iree_status_join(
      status,
      loom_run_hal_command_scenario_calls_deinitialize(&prepared_calls));
}

static iree_status_t loom_run_hal_command_scenario_benchmark_calls(
    loom_run_hal_command_scenario_product_t* product,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_hal_benchmark_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_scenario_benchmark_result_t* out_result) {
  *out_result = (loom_run_hal_testbench_scenario_benchmark_result_t){0};
  if (call_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL command scenario benchmark requires a physical call ring");
  }

  loom_run_hal_command_scenario_call_flags_t call_flags =
      LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_UNIFORM_FOOTPRINT;
  if (iree_all_bits_set(options->flags,
                        LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
    call_flags |=
        LOOM_RUN_HAL_COMMAND_SCENARIO_CALL_FLAG_PROFILE_COMMAND_BUFFERS;
  }
  loom_run_hal_command_scenario_calls_t prepared_calls = {0};
  iree_status_t status = loom_run_hal_command_scenario_calls_initialize(
      product, call_count, calls, options->dispatch_batch.command_buffer_mode,
      call_flags, &prepared_calls);
  if (iree_status_is_ok(status)) {
    const loom_run_hal_command_buffer_ring_t ring = {
        .command_buffers = prepared_calls.command_buffers,
        .profile_command_buffers = prepared_calls.profile_command_buffers,
        .binding_table = prepared_calls.rebindable_bindings,
        .ring_count = call_count,
        .binding_count = product->program.requirements.rebindable_binding_count,
        .execute_flags = options->dispatch_batch.execute_flags,
    };
    status = loom_run_hal_benchmark_command_buffer_ring(
        &product->profile->provider_options.context->runtime, &ring, options,
        host_allocator, &out_result->benchmark);
  }
  if (iree_status_is_ok(status)) {
    out_result->dispatch_count_per_call = product->dispatch_count;
    out_result->binding_count_per_call = prepared_calls.binding_count_per_call;
    out_result->binding_set_bytes = prepared_calls.binding_set_bytes;
    out_result->binding_ring_bytes = prepared_calls.binding_ring_bytes;
    out_result->binding_range_count_per_call =
        prepared_calls.binding_range_count_per_call;
    out_result->binding_range_set_bytes =
        prepared_calls.binding_range_set_bytes;
    out_result->binding_range_ring_bytes =
        prepared_calls.binding_range_ring_bytes;
    out_result->materialized_storage_count_per_call =
        prepared_calls.materialized_storage_count_per_call;
    out_result->materialized_storage_set_bytes =
        prepared_calls.materialized_storage_set_bytes;
    out_result->materialized_storage_ring_bytes =
        prepared_calls.materialized_storage_ring_bytes;
  }
  status = iree_status_join(
      status,
      loom_run_hal_command_scenario_calls_deinitialize(&prepared_calls));
  if (!iree_status_is_ok(status)) {
    *out_result = (loom_run_hal_testbench_scenario_benchmark_result_t){0};
  }
  return status;
}

static iree_status_t loom_run_hal_command_scenario_product_benchmark(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_benchmark_options_t* options,
    iree_allocator_t host_allocator, loom_run_benchmark_result_t* out_result) {
  loom_run_benchmark_result_initialize(out_result);
  loom_run_hal_command_scenario_product_t* product =
      (loom_run_hal_command_scenario_product_t*)user_data;
  if (invocation != product->subject) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario benchmark does not match its prepared product");
  }
  loom_run_hal_benchmark_options_t hal_options;
  loom_run_hal_benchmark_options_initialize(&hal_options);
  hal_options.timing = *options;
  hal_options.dispatch_batch.dispatch_count = options->batch_size;
  loom_run_hal_testbench_scenario_benchmark_result_t result;
  iree_status_t status = loom_run_hal_command_scenario_benchmark_calls(
      product, call_count, calls, &hal_options, host_allocator, &result);
  if (iree_status_is_ok(status)) {
    *out_result = result.benchmark.timing;
  }
  return status;
}

static const loom_testbench_product_vtable_t
    loom_run_hal_command_scenario_product_vtable = {
        .execute = loom_run_hal_command_scenario_product_execute,
        .benchmark = loom_run_hal_command_scenario_product_benchmark,
};

static iree_status_t loom_run_hal_command_scenario_select_target(
    loom_run_hal_command_scenario_product_t* product) {
  const loom_run_hal_testbench_actual_provider_options_t* provider_options =
      &product->profile->provider_options;
  loomc_iree_hal_target_selection_t selection = {0};
  loomc_result_t* result = NULL;
  iree_status_t status = loom_run_hal_testbench_context_select_target(
      provider_options->context,
      loomc_make_cstring_view("Loom command scenario live device"),
      provider_options->requested_target_profile, &selection, &result);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_observe_result(
        product, result,
        IREE_SV("command scenario target selection rejected the live device"));
  }
  if (iree_status_is_ok(status)) {
    product->selected_target_profile = selection.target_profile;
    selection.target_profile = NULL;
    product->executable_target = selection.executable_target;
  }
  loomc_result_release(result);
  loomc_target_profile_release(selection.target_profile);
  return status;
}

static iree_host_size_t loom_run_hal_command_scenario_dispatch_count(
    const loom_cmd_program_t* program) {
  iree_host_size_t dispatch_count = 0;
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const loom_cmd_program_command_kind_t kind =
        loom_cmd_program_command_kind_base(
            loom_cmd_program_command_at(program, i).kind);
    if (kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT ||
        kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
        kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC) {
      ++dispatch_count;
    }
  }
  return dispatch_count;
}

static iree_status_t loom_run_hal_command_scenario_resolve_program(
    loom_run_hal_command_scenario_product_t* product,
    const loomc_product_t* command_product) {
  if (loomc_cmd_program_product_program_count(command_product) != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "command scenario product did not contain exactly one program");
  }
  loomc_cmd_program_t portable_program = {0};
  if (!loomc_cmd_program_product_program_at(command_product, 0,
                                            &portable_program)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "command scenario product program is unavailable");
  }
  IREE_RETURN_IF_ERROR(iree_byte_sequence_clone(
      iree_byte_sequence_from_loomc(portable_program.artifact.contents),
      product->host_allocator, &product->program_image));
  IREE_RETURN_IF_ERROR(loom_cmd_program_parse(
      iree_const_cast_byte_span(product->program_image), &product->program));
  product->dispatch_count =
      loom_run_hal_command_scenario_dispatch_count(&product->program);

  const loomc_host_size_t requirement_count =
      loomc_cmd_program_product_entry_requirement_count(command_product);
  if (requirement_count > product->requirement_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command product published only a subset of its kernel requirements");
  }
  for (loomc_host_size_t i = 0; i < requirement_count; ++i) {
    if (!product->requirements[i].assigned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command entry requirement %" PRIhsz " has no implementation", i);
    }
  }
  for (iree_host_size_t i = requirement_count;
       i < product->requirement_capacity; ++i) {
    if (product->requirements[i].assigned) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel request bound an unknown command entry requirement");
    }
  }
  if (portable_program.entry_requirement_count !=
      product->program.requirements.entry_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "portable command entry projection does not match its artifact");
  }
  product->entry_count = portable_program.entry_requirement_count;
  IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_allocate_array(
      product->host_allocator, product->entry_count, sizeof(*product->entries),
      (void**)&product->entries));
  for (iree_host_size_t i = 0; i < product->entry_count; ++i) {
    const loomc_requirement_ordinal_t requirement_ordinal =
        portable_program.entry_requirement_ordinals[i];
    if (requirement_ordinal >= requirement_count) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "portable command entry refers to an invalid requirement");
    }
    product->entries[i] = product->requirements[requirement_ordinal].entry;
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_command_scenario_product_prepare(
    loom_run_hal_testbench_scenario_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  *out_product = (loom_testbench_prepared_product_t){0};
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_COMMAND_PROGRAM ||
      invocation->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario product requires a result-free command program");
  }
  const loom_run_hal_testbench_actual_provider_options_t* provider_options =
      &profile->provider_options;
  if (provider_options->compilation == NULL ||
      provider_options->compilation->context == NULL ||
      provider_options->compilation->compiler == NULL ||
      provider_options->compilation->workspace == NULL ||
      provider_options->module == NULL ||
      provider_options->native_module == NULL ||
      provider_options->native_module != invocation->module) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "command scenario profile is not bound to the invocation module");
  }

  loom_run_hal_command_scenario_product_t* product = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*product),
                                             (void**)&product));
  memset(product, 0, sizeof(*product));
  product->host_allocator = host_allocator;
  product->profile = profile;
  product->subject = invocation;

  loomc_string_view_t root_symbol = loomc_string_view_empty();
  iree_status_t status = loom_run_hal_command_scenario_build_wrapper(
      product, configuration, &root_symbol);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_select_target(product);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loomc_allocator_from_iree(host_allocator),
        &product->request_workspace));
  }

  loomc_product_t* command_product = NULL;
  loomc_result_t* result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_cmd_program_module_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS,
        .structure_size = sizeof(options),
        .root_symbols = &root_symbol,
        .root_symbol_count = 1,
        .artifact_flags = provider_options->emit_artifact_flags,
        .config = provider_options->compilation->config
                      ? *provider_options->compilation->config
                      : (loomc_config_options_t){0},
        .target_profile = product->selected_target_profile,
        .request_sink =
            {
                .publish = loom_run_hal_command_scenario_publish,
                .user_data = product,
            },
    };
    status = iree_status_from_loomc(loomc_cmd_program_product_build_module(
        provider_options->compilation->workspace, product->adapted_module,
        &options, loomc_allocator_from_iree(host_allocator), &command_product,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_command_scenario_observe_result(
        product, result,
        IREE_SV("command scenario product construction failed"));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_command_scenario_resolve_program(product, command_product);
  }
  loomc_result_release(result);
  loomc_product_release(command_product);

  if (iree_status_is_ok(status)) {
    loomc_target_profile_release(product->selected_target_profile);
    product->selected_target_profile = NULL;
    loomc_workspace_release(product->request_workspace);
    product->request_workspace = NULL;
    loomc_module_release(product->adapted_module);
    product->adapted_module = NULL;
    *out_product = (loom_testbench_prepared_product_t){
        .vtable = &loom_run_hal_command_scenario_product_vtable,
        .destroy = loom_run_hal_command_scenario_product_destroy,
        .user_data = product,
    };
  } else {
    loom_run_hal_command_scenario_product_destroy(product);
  }
  return status;
}

bool loom_run_hal_testbench_is_command_scenario_product(
    const loom_testbench_prepared_product_t* prepared_product) {
  return prepared_product != NULL &&
         prepared_product->vtable ==
             &loom_run_hal_command_scenario_product_vtable &&
         prepared_product->user_data != NULL;
}

iree_status_t loom_run_hal_testbench_benchmark_command_scenario_product(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_hal_benchmark_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_scenario_benchmark_result_t* out_result) {
  if (!loom_run_hal_testbench_is_command_scenario_product(prepared_product) ||
      invocation != prepared_product->invocation) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario benchmark does not match its prepared product");
  }
  loom_run_hal_command_scenario_product_t* product =
      (loom_run_hal_command_scenario_product_t*)prepared_product->user_data;
  return loom_run_hal_command_scenario_benchmark_calls(
      product, call_count, calls, options, host_allocator, out_result);
}

iree_status_t loom_run_hal_testbench_query_command_scenario_input_layout(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_product_call_t* call,
    loom_run_hal_testbench_scenario_input_layout_t* out_layout) {
  *out_layout = (loom_run_hal_testbench_scenario_input_layout_t){0};
  if (!loom_run_hal_testbench_is_command_scenario_product(prepared_product) ||
      invocation != prepared_product->invocation) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command scenario layout query does not match its prepared product");
  }
  uint64_t binding_bytes = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_command_scenario_validate_call(
      invocation, call, &binding_bytes));
  *out_layout = (loom_run_hal_testbench_scenario_input_layout_t){
      .binding_count = invocation->input_count,
      .binding_set_bytes = binding_bytes,
      .binding_range_count = call->input_footprint.binding_range_count,
      .binding_range_set_bytes =
          call->input_footprint.binding_range_byte_length,
      .materialized_storage_count =
          call->input_footprint.storage_allocation_count,
      .materialized_storage_set_bytes =
          call->input_footprint.storage_byte_length,
  };
  return iree_ok_status();
}
