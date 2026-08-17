// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/module.h"

#include <stdio.h>
#include <string.h>

#include "common/fat_binary.h"
#include "common/internal.h"
#include "iree/io/file_handle.h"

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_fat_binary_target_append_unique(
    iree_hal_streaming_fat_binary_target_t* targets,
    iree_host_size_t target_capacity, iree_host_size_t* target_count,
    const iree_hal_executable_target_t* executable_target) {
  if (executable_target == NULL ||
      iree_string_view_is_empty(executable_target->target_key)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fat-binary target is missing its target key");
  }
  for (iree_host_size_t i = 0; i < *target_count; ++i) {
    if (iree_string_view_equal(targets[i].executable_target->target_key,
                               executable_target->target_key)) {
      return iree_ok_status();
    }
  }
  if (*target_count >= target_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fat-binary target capacity exceeded");
  }
  targets[*target_count].executable_target = executable_target;
  *target_count += 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_fat_binary_targets_from_device(
    iree_hal_device_t* device, iree_host_size_t target_capacity,
    iree_hal_streaming_fat_binary_target_t* targets,
    iree_host_size_t* out_target_count) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(targets);
  IREE_ASSERT_ARGUMENT(out_target_count);

  iree_host_size_t target_count = 0;
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);

  iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
  };
  iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec does not report an exact executable target");
  }
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous exact executable targets");
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_target_append_unique(
      targets, target_capacity, &target_count, result.target));

  selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC;
  result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection);
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous generic executable targets");
  }
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_target_append_unique(
        targets, target_capacity, &target_count, result.target));
  }

  *out_target_count = target_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_extract_metadata(
    iree_hal_streaming_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query the number of exported functions.
  const iree_host_size_t executable_count =
      module->executable_count ? module->executable_count : 1;
  module->symbol_count = 0;
  for (iree_host_size_t executable_ordinal = 0;
       executable_ordinal < executable_count; ++executable_ordinal) {
    iree_hal_executable_t* executable =
        module->executables ? module->executables[executable_ordinal]
                            : module->executable;
    module->symbol_count += iree_hal_executable_export_count(executable);
  }
  if (module->symbol_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate storage for export infos and per-symbol op counts together.
  // We want to query the export info once and reuse it as we process. In order
  // to allocate the minimum amount of memory we need to precalculate the
  // required number of unpack operations. Once we do that we avoid
  // recalculating later by caching the results.
  typedef struct op_counts_t {
    uint16_t copy_count;
    uint16_t resolve_count;
  } op_counts_t;
  const iree_host_size_t export_infos_size =
      module->symbol_count * sizeof(iree_hal_executable_export_info_t);
  const iree_host_size_t export_executables_size =
      module->symbol_count * sizeof(iree_hal_executable_t*);
  const iree_host_size_t export_ordinals_size =
      module->symbol_count * sizeof(iree_hal_executable_export_ordinal_t);
  const iree_host_size_t op_counts_size =
      module->symbol_count * sizeof(op_counts_t);
  uint8_t* temp_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(module->host_allocator,
                                export_infos_size + export_executables_size +
                                    export_ordinals_size + op_counts_size,
                                (void**)&temp_buffer));
  memset(temp_buffer, 0,
         export_infos_size + export_executables_size + export_ordinals_size +
             op_counts_size);
  iree_hal_executable_export_info_t* export_infos =
      (iree_hal_executable_export_info_t*)temp_buffer;
  iree_hal_executable_t** export_executables =
      (iree_hal_executable_t**)(temp_buffer + export_infos_size);
  iree_hal_executable_export_ordinal_t* export_ordinals =
      (iree_hal_executable_export_ordinal_t*)(temp_buffer + export_infos_size +
                                              export_executables_size);
  op_counts_t* symbol_op_counts =
      (op_counts_t*)(temp_buffer + export_infos_size + export_executables_size +
                     export_ordinals_size);

  // Count all parameters in all exports so we can allocate one buffer to
  // fetch them all. This is somewhat wasteful as we'll be allocating quite a
  // bit but is easier to see in traces.
  iree_status_t status = iree_ok_status();
  iree_host_size_t total_parameter_count = 0;
  iree_host_size_t symbol_index = 0;
  for (iree_host_size_t executable_ordinal = 0;
       iree_status_is_ok(status) && executable_ordinal < executable_count;
       ++executable_ordinal) {
    iree_hal_executable_t* executable =
        module->executables ? module->executables[executable_ordinal]
                            : module->executable;
    const iree_host_size_t export_count =
        iree_hal_executable_export_count(executable);
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      export_executables[symbol_index] = executable;
      export_ordinals[symbol_index] = (iree_hal_executable_export_ordinal_t)i;
      status = iree_hal_executable_export_info(executable,
                                               export_ordinals[symbol_index],
                                               &export_infos[symbol_index]);
      if (!iree_status_is_ok(status)) break;
      total_parameter_count += export_infos[symbol_index].parameter_count;
      ++symbol_index;
    }
  }

  // Allocate the scratch space for querying parameter info.
  iree_hal_executable_export_parameter_t* parameters = NULL;
  if (iree_status_is_ok(status) && total_parameter_count > 0) {
    status = iree_allocator_malloc(module->host_allocator,
                                   total_parameter_count * sizeof(*parameters),
                                   (void**)&parameters);
  }

  // Analyze each export to determine operation counts.
  iree_host_size_t total_ops = 0;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    const iree_host_size_t parameter_count = export_infos[i].parameter_count;
    if (!parameter_count) continue;
    // Query parameters before allocating symbol-owned operation storage.
    status = iree_hal_executable_export_parameters(
        export_executables[i], export_ordinals[i], parameter_count,
        &parameters[parameter_base]);
    if (!iree_status_is_ok(status)) break;
    for (uint16_t j = 0; j < parameter_count; ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      const bool is_binding_parameter =
          parameter->type == IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING;
      if (is_binding_parameter) {
        ++symbol_op_counts[i].resolve_count;
        ++total_ops;
      } else {
        ++symbol_op_counts[i].copy_count;
        ++total_ops;
      }
    }
    parameter_base += parameter_count;
  }

  // Allocate all permanent storage in a single block.
  // Memory layout: [Symbol Array][Symbol0 ops][Symbol1 ops]...
  const iree_host_size_t symbols_size =
      module->symbol_count * sizeof(iree_hal_streaming_symbol_t);
  const iree_host_size_t ops_size =
      total_ops * sizeof(iree_hal_streaming_parameter_op_t);
  const iree_host_size_t total_size = symbols_size + ops_size;
  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(module->host_allocator, total_size,
                                   (void**)&buffer);
  }
  module->symbols = (iree_hal_streaming_symbol_t*)buffer;
  iree_hal_streaming_parameter_op_t* ops_base =
      buffer ? (iree_hal_streaming_parameter_op_t*)(buffer + symbols_size)
             : NULL;

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(module->context->device);
  iree_hal_streaming_parameter_op_t* current_ops = ops_base;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
    memset(symbol, 0, sizeof(*symbol));
    symbol->module = module;
    symbol->name = export_infos[i].name;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol->executable = export_executables[i];
    symbol->export_ordinal = export_ordinals[i];

    // Cache generic loaded-function facts for compatibility API queries and
    // launch validation.
    symbol->occupancy_info = export_infos[i].occupancy_info;
    status = iree_hal_streaming_function_attributes_initialize(
        device_spec, &export_infos[i], &symbol->function_attributes);
    if (!iree_status_is_ok(status)) break;
    symbol->preferred_shared_memory_carveout = -1;

    // Initialize parameter info.
    iree_hal_streaming_parameter_info_t* parameter_info = &symbol->parameters;
    if (export_infos[i].constant_byte_length > UINT16_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "function constant metadata exceeds supported parameter size");
      continue;
    }
    // Executable binding_count describes normal HAL dispatch bindings. Native
    // HIP packing uses the same reflected BINDING parameters and only consults
    // their optional target ABI offsets while constructing custom kernargs.
    parameter_info->buffer_size = 0;
    parameter_info->constant_bytes = 0;
    parameter_info->direct_arg_bytes = 0;
    parameter_info->binding_count = symbol_op_counts[i].resolve_count;
    parameter_info->copy_count = symbol_op_counts[i].copy_count;
    parameter_info->ops = current_ops;
    const uint16_t parameter_count = export_infos[i].parameter_count;
    if (parameter_count == 0) {
      // No parameters.
      continue;
    }

    // Build one operation per reflected parameter. Copy ops go first, then
    // resolve ops.
    uint16_t source_offset = 0;
    iree_host_size_t direct_arg_offset = 0;
    uint16_t buffer_size = 0;
    iree_host_size_t this_kernel_direct_arg_size = 0;
    iree_hal_streaming_parameter_op_t* copy_ops_start = current_ops;
    iree_hal_streaming_parameter_op_t* resolve_ops_start =
        current_ops + symbol_op_counts[i].copy_count;
    for (uint16_t j = 0; j < symbol_op_counts[i].resolve_count; ++j) {
      // This sentinel is replaced when its dense HAL binding ordinal is
      // reflected below. It detects duplicate metadata before the operation
      // table becomes visible to dispatch.
      resolve_ops_start[j].resolve.destination_ordinal = UINT16_MAX;
    }
    uint16_t copy_count = 0;
    uint16_t resolve_count = 0;
    for (uint16_t j = 0; iree_status_is_ok(status) && j < parameter_count;
         ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      const bool is_binding_parameter =
          parameter->type == IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING;
      iree_host_size_t native_abi_destination_offset = direct_arg_offset;
      if (parameter->flags &
          IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET) {
        native_abi_destination_offset = parameter->native_abi_offset;
      }
      iree_host_size_t source_extent = 0;
      iree_host_size_t native_extent = 0;
      iree_host_size_t next_direct_arg_offset = 0;
      if (IREE_UNLIKELY(
              native_abi_destination_offset > UINT16_MAX ||
              !iree_host_size_checked_add((iree_host_size_t)source_offset,
                                          parameter->size, &source_extent) ||
              source_extent > UINT16_MAX ||
              !iree_host_size_checked_add(native_abi_destination_offset,
                                          parameter->size, &native_extent) ||
              native_extent > UINT16_MAX ||
              !iree_host_size_checked_add(direct_arg_offset, parameter->size,
                                          &next_direct_arg_offset))) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "kernel parameter layout exceeds metadata "
                                  "field width");
        break;
      }
      if (!is_binding_parameter) {
        iree_host_size_t constant_destination_extent = 0;
        if (IREE_UNLIKELY(!iree_host_size_checked_add(
                              (iree_host_size_t)parameter->offset,
                              parameter->size, &constant_destination_extent) ||
                          constant_destination_extent >
                              export_infos[i].constant_byte_length)) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "kernel constant parameter exceeds the reflected constants "
              "range");
          break;
        }
      }
      const uint16_t next_source_offset = (uint16_t)source_extent;
      if (is_binding_parameter) {
        if (IREE_UNLIKELY(parameter->offset >=
                          symbol_op_counts[i].resolve_count)) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "kernel binding parameter ordinal %u exceeds binding count %u",
              parameter->offset, symbol_op_counts[i].resolve_count);
          break;
        }
        iree_hal_streaming_parameter_resolve_op_t* op =
            &resolve_ops_start[parameter->offset].resolve;
        if (IREE_UNLIKELY(op->destination_ordinal != UINT16_MAX)) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "kernel metadata assigns binding ordinal %u more than once",
              parameter->offset);
          break;
        }
        op->reserved = 0;
        op->source_offset = source_offset;
        op->destination_ordinal = parameter->offset;
        op->source_ordinal = j;
        // Native launches place raw device pointers at target ABI offsets.
        op->native_abi_destination_offset =
            (uint16_t)native_abi_destination_offset;
        source_offset = next_source_offset;
        buffer_size = source_offset;
        ++resolve_count;

        if (native_extent > this_kernel_direct_arg_size) {
          this_kernel_direct_arg_size = native_extent;
        }
        direct_arg_offset = iree_max(native_extent, next_direct_arg_offset);
      } else {
        // Constants use two layouts: a dense HAL constants buffer in source
        // order, and the target ABI byte image used by native HIP launches.
        iree_hal_streaming_parameter_copy_op_t* op =
            &copy_ops_start[copy_count].copy;
        op->size = parameter->size;
        op->source_offset = source_offset;
        op->source_ordinal = j;
        op->native_abi_destination_offset =
            (uint16_t)native_abi_destination_offset;
        op->constant_destination_offset = parameter->offset;
        ++copy_count;
        source_offset = next_source_offset;
        buffer_size = source_offset;

        if (native_extent > this_kernel_direct_arg_size) {
          this_kernel_direct_arg_size = native_extent;
        }
        direct_arg_offset = iree_max(native_extent, next_direct_arg_offset);
      }
    }
    if (!iree_status_is_ok(status)) break;
    parameter_info->buffer_size = buffer_size;
    if (IREE_UNLIKELY(export_infos[i].constant_byte_length > UINT16_MAX)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "kernel constant layout exceeds metadata "
                                "field width");
      break;
    }
    parameter_info->constant_bytes =
        (uint16_t)export_infos[i].constant_byte_length;
    if (buffer_size > this_kernel_direct_arg_size) {
      this_kernel_direct_arg_size = buffer_size;
    }
    if (IREE_UNLIKELY(this_kernel_direct_arg_size > UINT16_MAX)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "kernel direct argument layout exceeds "
                                "metadata field width");
      break;
    }
    parameter_info->direct_arg_bytes = (uint16_t)this_kernel_direct_arg_size;

    // Advance to next symbol's ops.
    parameter_base += parameter_count;
    current_ops += copy_count + resolve_count;
  }

  iree_allocator_free(module->host_allocator, parameters);
  iree_allocator_free(module->host_allocator, temp_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_module_destroy(
    iree_hal_streaming_module_t* module);
static iree_status_t iree_hal_streaming_module_initialize_managed_globals(
    iree_hal_streaming_module_t* module,
    const iree_hal_streaming_fat_binary_extract_t* fat_extract);

static iree_status_t iree_hal_streaming_module_load_executable(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_load_flags_t load_flags,
    const iree_hal_executable_target_t* executable_target,
    iree_const_byte_span_t executable_data,
    iree_hal_executable_t** out_executable) {
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.flags = load_flags;
  load_params.executable_data = executable_data;
  return iree_hal_device_load_executable(
      context->device, context->queue_affinity, executable_target, &load_params,
      out_executable);
}

iree_status_t iree_hal_streaming_module_create_from_memory(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_load_flags_t load_flags, iree_const_byte_span_t image,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(image.data);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the module structure up-front for terminal cleanup.
  iree_hal_streaming_module_t* module = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  iree_slim_mutex_initialize(&module->global_mutex);
  module->context = context;
  iree_hal_streaming_context_retain(context);
  module->host_allocator = host_allocator;

  // HIP toolchains hand us several container formats: raw AMDGPU ELFs,
  // __CLANG_OFFLOAD_BUNDLE__ archives, CCOB (zstd-compressed bundles), and
  // __hipFatBinaryWrapper-wrapped combinations of those. Unwrap everything here
  // and only forward raw ELF plus its selected target to the HAL device.
  iree_hal_streaming_fat_binary_extract_t fat_extract = {0};
  const bool try_fat_unwrap = context->device_entry != NULL &&
                              iree_hal_streaming_fat_binary_is_supported(image);
  iree_status_t status = iree_ok_status();
  if (try_fat_unwrap) {
    iree_hal_streaming_fat_binary_target_t targets[2] = {0};
    iree_host_size_t target_count = 0;
    status = iree_hal_streaming_fat_binary_targets_from_device(
        context->device_entry->hal_device, IREE_ARRAYSIZE(targets), targets,
        &target_count);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_fat_binary_extract_for_targets(
          image, target_count, targets, host_allocator, &fat_extract);
    }
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module binary is not a supported HRX AMDGPU "
                              "ELF, offload bundle, CCOB, or HIP fat binary");
  }

  // Create HAL executable from binary.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_load_executable(
        context, load_flags, fat_extract.matches[0].executable_target,
        fat_extract.matches[0].data, &module->executable);
  }

  // If the fat binary had multiple matching HSACO entries, prepare all of
  // them and expose their exports through the same hipModule_t. Native HIP lets
  // libraries such as hipBLAS/Tensile probe one module handle for a kernel that
  // may live in a later matching code object.
  if (iree_status_is_ok(status) && fat_extract.match_count > 1) {
    module->executable_count = fat_extract.match_count;
    status = iree_allocator_malloc(
        host_allocator, module->executable_count * sizeof(*module->executables),
        (void**)&module->executables);
    if (iree_status_is_ok(status)) {
      memset(module->executables, 0,
             module->executable_count * sizeof(*module->executables));
      module->executables[0] = module->executable;
    }
    for (iree_host_size_t i = 1;
         iree_status_is_ok(status) && i < module->executable_count; ++i) {
      status = iree_hal_streaming_module_load_executable(
          context, load_flags, fat_extract.matches[i].executable_target,
          fat_extract.matches[i].data, &module->executables[i]);
    }
  }

  // Extract kernel metadata.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_extract_metadata(module);
  }

  // Managed pointer slots must be valid before callers can launch a kernel.
  // Discover and publish their storage while the module is still private.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_initialize_managed_globals(module,
                                                                  &fat_extract);
  }

  iree_hal_streaming_fat_binary_extract_reset(&fat_extract);
  if (iree_status_is_ok(status)) {
    *out_module = module;
  } else {
    iree_hal_streaming_module_destroy(module);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_module_create_from_file(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_load_flags_t load_flags, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Preload the image into owned host memory. Module executables may alias this
  // data for their lifetime, but loading a module must not retain a descriptor
  // for the source file.
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_preload(IREE_IO_FILE_MODE_READ, path,
                                      host_allocator, &file_handle));

  // Map the entire file for read access.
  iree_io_file_mapping_t* file_mapping = NULL;
  iree_status_t status = iree_io_file_map_view(
      file_handle, IREE_IO_FILE_ACCESS_READ, 0, IREE_HOST_SIZE_MAX,
      IREE_IO_FILE_MAPPING_FLAG_NONE, host_allocator, &file_mapping);

  // Release the file handle; the mapping retains the preloaded allocation.
  iree_io_file_handle_release(file_handle);

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Get the read-only contents of the mapping.
  iree_const_byte_span_t image = iree_io_file_mapping_contents_ro(file_mapping);

  // Create the module from the mapped memory.
  iree_hal_streaming_module_t* module = NULL;
  status = iree_hal_streaming_module_create_from_memory(
      context, load_flags, image, host_allocator, &module);

  iree_io_file_mapping_release(file_mapping);
  if (iree_status_is_ok(status)) *out_module = module;

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_module_destroy(
    iree_hal_streaming_module_t* module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = module->host_allocator;

  // Release symbol metadata.
  iree_allocator_free(module->host_allocator, module->symbols);

  // Release cached executable globals while both the context pointer map and
  // executable-owned global buffers are still live.
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(
        module->globals[i]->managed_buffer);
    iree_hal_streaming_memory_release_wrapped_buffer(
        module->globals[i]->global_buffer);
    iree_allocator_free(host_allocator, module->globals[i]);
  }
  iree_allocator_free(host_allocator, module->globals);
  iree_slim_mutex_deinitialize(&module->global_mutex);

  // Release loaded executables.
  if (module->executables) {
    for (iree_host_size_t i = 0; i < module->executable_count; ++i) {
      iree_hal_executable_release(module->executables[i]);
    }
    iree_allocator_free(host_allocator, module->executables);
  } else {
    iree_hal_executable_release(module->executable);
  }

  // Release context.
  iree_hal_streaming_context_release(module->context);

  // Free module memory.
  iree_allocator_free(host_allocator, module);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_module_retain(iree_hal_streaming_module_t* module) {
  if (module) {
    iree_atomic_ref_count_inc(&module->ref_count);
  }
}

void iree_hal_streaming_module_release(iree_hal_streaming_module_t* module) {
  if (module && iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    iree_hal_streaming_module_destroy(module);
  }
}

static bool iree_hal_streaming_module_symbol_name_matches(
    iree_string_view_t symbol_name, iree_string_view_t name) {
  if (iree_string_view_equal(symbol_name, name)) return true;
  iree_string_view_t stripped_name =
      iree_string_view_strip_suffix(name, IREE_SV(".kd"));
  return stripped_name.size != name.size &&
         iree_string_view_equal(symbol_name, stripped_name);
}

iree_status_t iree_hal_streaming_module_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_type_t expected_type,
    iree_hal_streaming_symbol_t** out_symbol) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_symbol);
  *out_symbol = NULL;

  iree_string_view_t name_view =
      iree_string_view_trim(iree_make_cstring_view(name));
  for (uint32_t i = 0; i < module->symbol_count; ++i) {
    if (iree_hal_streaming_module_symbol_name_matches(module->symbols[i].name,
                                                      name_view)) {
      // Check if the symbol type matches expected type.
      if (module->symbols[i].type == expected_type) {
        // Return symbol info as pointer.
        *out_symbol = &module->symbols[i];
        return iree_ok_status();
      } else {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "symbol '%.*s' found but type mismatch (expected %d, got %d)",
            (int)name_view.size, name_view.data, expected_type,
            module->symbols[i].type);
      }
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "symbol '%.*s' not found in module",
                          (int)name_view.size, name_view.data);
}

iree_status_t iree_hal_streaming_module_function(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_function) {
  return iree_hal_streaming_module_symbol(
      module, name, IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION, out_function);
}

static iree_hal_streaming_symbol_t*
iree_hal_streaming_module_find_global_locked(
    iree_hal_streaming_module_t* module, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = module->globals[i];
    if (iree_hal_streaming_module_symbol_name_matches(symbol->name, name)) {
      return symbol;
    }
  }
  return NULL;
}

static iree_hal_streaming_symbol_t*
iree_hal_streaming_module_find_global_for_executable_locked(
    iree_hal_streaming_module_t* module, iree_hal_executable_t* executable,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = module->globals[i];
    if (symbol->executable == executable &&
        iree_hal_streaming_module_symbol_name_matches(symbol->name, name)) {
      return symbol;
    }
  }
  return NULL;
}

static iree_status_t iree_hal_streaming_module_grow_globals_locked(
    iree_hal_streaming_module_t* module, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= module->global_capacity) return iree_ok_status();

  const iree_host_size_t minimum_allocated_capacity =
      minimum_capacity < 4 ? 4 : minimum_capacity;
  return iree_allocator_grow_array(
      module->host_allocator, minimum_allocated_capacity,
      sizeof(*module->globals), &module->global_capacity,
      (void**)&module->globals);
}

static iree_status_t iree_hal_streaming_module_create_global_symbol_locked(
    iree_hal_streaming_module_t* module, iree_hal_executable_t* executable,
    iree_hal_executable_global_t global_handle,
    iree_hal_streaming_symbol_t** out_symbol) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_symbol);
  *out_symbol = NULL;

  iree_hal_executable_global_info_t global_info;
  IREE_RETURN_IF_ERROR(
      iree_hal_executable_global_info(executable, global_handle, &global_info));

  iree_hal_buffer_t* global_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_global_buffer(
      executable, global_handle, IREE_HAL_QUEUE_AFFINITY_ANY, &global_buffer));

  iree_hal_streaming_buffer_t* streaming_buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_wrap_buffer(
      module->context, global_buffer,
      IREE_HAL_STREAMING_BUFFER_CONTEXT_BORROWED, &streaming_buffer);

  iree_hal_streaming_symbol_t* symbol = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(module->host_allocator, sizeof(*symbol),
                                   (void**)&symbol);
  }
  if (iree_status_is_ok(status)) {
    memset(symbol, 0, sizeof(*symbol));
    symbol->module = module;
    symbol->name = global_info.name;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL;
    symbol->executable = executable;
    symbol->global_handle = global_handle;
    symbol->global_buffer = streaming_buffer;
    symbol->device_address =
        iree_hal_streaming_buffer_device_pointer(streaming_buffer);
    symbol->size_bytes = global_info.byte_length;
    status = iree_hal_streaming_module_grow_globals_locked(
        module, module->global_count + 1);
  }

  if (iree_status_is_ok(status)) {
    module->globals[module->global_count++] = symbol;
    *out_symbol = symbol;
  } else {
    iree_allocator_free(module->host_allocator, symbol);
    iree_hal_streaming_memory_release_wrapped_buffer(streaming_buffer);
  }
  return status;
}

static iree_status_t
iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
    iree_hal_streaming_module_t* module, iree_hal_executable_t* executable,
    iree_string_view_t name, bool* out_found,
    iree_hal_streaming_symbol_t** out_global) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_found);
  IREE_ASSERT_ARGUMENT(out_global);
  *out_found = false;
  *out_global = NULL;

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&module->global_mutex);

  iree_hal_streaming_symbol_t* cached_symbol =
      iree_hal_streaming_module_find_global_for_executable_locked(
          module, executable, name);
  if (cached_symbol) {
    *out_found = true;
    *out_global = cached_symbol;
  } else {
    iree_hal_executable_global_t global_handle =
        iree_hal_executable_global_invalid();
    bool found = false;
    status = iree_hal_executable_try_lookup_global_by_name(
        executable, name, &found, &global_handle);
    if (iree_status_is_ok(status) && found) {
      status = iree_hal_streaming_module_create_global_symbol_locked(
          module, executable, global_handle, out_global);
      if (iree_status_is_ok(status)) *out_found = true;
    }
  }

  iree_slim_mutex_unlock(&module->global_mutex);
  return status;
}

iree_status_t iree_hal_streaming_module_try_lookup_global_symbol(
    iree_hal_streaming_module_t* module, const char* name, bool* out_found,
    iree_hal_streaming_symbol_t** out_global) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_found);
  IREE_ASSERT_ARGUMENT(out_global);
  *out_found = false;
  *out_global = NULL;

  iree_string_view_t name_view =
      iree_string_view_trim(iree_make_cstring_view(name));

  for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
    if ((symbol->type == IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL ||
         symbol->type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) &&
        iree_hal_streaming_module_symbol_name_matches(symbol->name,
                                                      name_view)) {
      *out_found = true;
      *out_global = symbol;
      return iree_ok_status();
    }
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&module->global_mutex);

  iree_hal_streaming_symbol_t* cached_symbol =
      iree_hal_streaming_module_find_global_locked(module, name_view);
  if (cached_symbol) {
    *out_found = true;
    *out_global = cached_symbol;
  } else {
    const iree_host_size_t executable_count =
        module->executable_count ? module->executable_count : 1;
    for (iree_host_size_t executable_ordinal = 0;
         executable_ordinal < executable_count; ++executable_ordinal) {
      iree_hal_executable_t* executable =
          module->executables ? module->executables[executable_ordinal]
                              : module->executable;
      iree_hal_executable_global_t global_handle =
          iree_hal_executable_global_invalid();
      bool found = false;
      status = iree_hal_executable_try_lookup_global_by_name(
          executable, name_view, &found, &global_handle);
      if (!iree_status_is_ok(status)) break;
      if (!found) continue;
      status = iree_hal_streaming_module_create_global_symbol_locked(
          module, executable, global_handle, out_global);
      if (iree_status_is_ok(status)) *out_found = true;
      break;
    }
  }

  iree_slim_mutex_unlock(&module->global_mutex);
  return status;
}

iree_status_t iree_hal_streaming_module_global_symbol(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_symbol_t** out_global) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_global);
  *out_global = NULL;

  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_try_lookup_global_symbol(
      module, name, &found, out_global));
  if (found) return iree_ok_status();

  iree_string_view_t name_view =
      iree_string_view_trim(iree_make_cstring_view(name));
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "global '%.*s' not found in module",
                          (int)name_view.size, name_view.data);
}

iree_status_t iree_hal_streaming_module_global(
    iree_hal_streaming_module_t* module, const char* name,
    iree_hal_streaming_deviceptr_t* out_device_ptr,
    iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(name);
  IREE_ASSERT_ARGUMENT(out_device_ptr);
  *out_device_ptr = 0;
  if (out_size) *out_size = 0;

  iree_hal_streaming_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_global_symbol(module, name, &symbol));

  if (symbol->managed_buffer) {
    *out_device_ptr = symbol->managed_buffer->device_ptr;
    if (out_size) *out_size = symbol->managed_buffer->logical_size;
  } else {
    *out_device_ptr = symbol->device_address;
    if (out_size) *out_size = symbol->size_bytes;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_managed_symbol_pair(
    iree_hal_streaming_module_t* module,
    iree_hal_streaming_symbol_t* pointer_symbol,
    iree_hal_streaming_symbol_t* initializer_symbol, void** out_host_pointer,
    iree_device_size_t* out_size) {
  if (IREE_UNLIKELY(initializer_symbol->size_bytes == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "managed global initializer `%.*s` has zero byte length",
        (int)initializer_symbol->name.size, initializer_symbol->name.data);
  }
  if (IREE_UNLIKELY(pointer_symbol->size_bytes !=
                    sizeof(iree_hal_streaming_deviceptr_t))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "managed global pointer slot `%.*s` has length %" PRIu64
        "; expected %" PRIhsz,
        (int)pointer_symbol->name.size, pointer_symbol->name.data,
        (uint64_t)pointer_symbol->size_bytes,
        sizeof(iree_hal_streaming_deviceptr_t));
  }

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&module->global_mutex);
  if (!pointer_symbol->managed_buffer) {
    iree_hal_streaming_buffer_t* managed_buffer = NULL;
    status = iree_hal_streaming_memory_allocate_managed(
        module->context, initializer_symbol->size_bytes,
        /*allocation_flags=*/0, &managed_buffer);
    if (iree_status_is_ok(status)) {
      managed_buffer->logical_size = initializer_symbol->size_bytes;
    }
    if (iree_status_is_ok(status) && initializer_symbol->size_bytes > 0) {
      status = iree_hal_streaming_memcpy_device_to_host(
          module->context, managed_buffer->host_ptr,
          initializer_symbol->device_address, initializer_symbol->size_bytes,
          /*stream=*/NULL);
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_streaming_deviceptr_t managed_device_pointer =
          managed_buffer->device_ptr;
      status = iree_hal_streaming_memcpy_host_to_device(
          module->context, pointer_symbol->device_address,
          &managed_device_pointer, sizeof(managed_device_pointer),
          /*stream=*/NULL);
    }
    if (iree_status_is_ok(status)) {
      pointer_symbol->managed_buffer = managed_buffer;
      managed_buffer = NULL;
    }
    iree_hal_streaming_memory_release_wrapped_buffer(managed_buffer);
  }
  if (iree_status_is_ok(status)) {
    if (out_host_pointer) {
      *out_host_pointer = pointer_symbol->managed_buffer->host_ptr;
    }
    if (out_size) *out_size = initializer_symbol->size_bytes;
  }
  iree_slim_mutex_unlock(&module->global_mutex);
  return status;
}

static bool iree_hal_streaming_module_split_managed_name(
    iree_string_view_t name, iree_string_view_t* out_pointer_name) {
  static const iree_string_view_t suffix = IREE_SVL(".managed");
  *out_pointer_name = iree_string_view_empty();
  if (!iree_string_view_ends_with(name, suffix) || name.size == suffix.size) {
    return false;
  }
  *out_pointer_name = iree_make_string_view(name.data, name.size - suffix.size);
  return true;
}

typedef struct iree_hal_streaming_managed_global_visitor_t {
  // Module receiving initialized managed global storage.
  iree_hal_streaming_module_t* module;
  // Executable loaded from the ELF currently being visited.
  iree_hal_executable_t* executable;
} iree_hal_streaming_managed_global_visitor_t;

static iree_status_t iree_hal_streaming_module_visit_managed_global(
    void* user_data, iree_string_view_t initializer_name) {
  iree_hal_streaming_managed_global_visitor_t* visitor =
      (iree_hal_streaming_managed_global_visitor_t*)user_data;
  iree_string_view_t pointer_name = iree_string_view_empty();
  if (!iree_hal_streaming_module_split_managed_name(initializer_name,
                                                    &pointer_name)) {
    return iree_ok_status();
  }

  bool pointer_found = false;
  iree_hal_streaming_symbol_t* pointer_symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
          visitor->module, visitor->executable, pointer_name, &pointer_found,
          &pointer_symbol));
  if (!pointer_found) return iree_ok_status();

  bool initializer_found = false;
  iree_hal_streaming_symbol_t* initializer_symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
          visitor->module, visitor->executable, initializer_name,
          &initializer_found, &initializer_symbol));
  if (IREE_UNLIKELY(!initializer_found)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "managed initializer `%.*s` was declared but not resolved",
        (int)initializer_name.size, initializer_name.data);
  }

  return iree_hal_streaming_module_initialize_managed_symbol_pair(
      visitor->module, pointer_symbol, initializer_symbol,
      /*out_host_pointer=*/NULL, /*out_size=*/NULL);
}

static iree_status_t iree_hal_streaming_module_initialize_managed_globals(
    iree_hal_streaming_module_t* module,
    const iree_hal_streaming_fat_binary_extract_t* fat_extract) {
  IREE_ASSERT_ARGUMENT(fat_extract);
  const iree_host_size_t executable_count =
      module->executable_count ? module->executable_count : 1;
  if (IREE_UNLIKELY(fat_extract->match_count != executable_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "loaded executable and selected ELF counts do not match");
  }
  for (iree_host_size_t i = 0; i < executable_count; ++i) {
    iree_hal_executable_t* executable =
        module->executables ? module->executables[i] : module->executable;
    iree_hal_streaming_managed_global_visitor_t visitor = {
        .module = module,
        .executable = executable,
    };
    IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_visit_elf_global_objects(
        fat_extract->matches[i].data,
        iree_hal_streaming_module_visit_managed_global, &visitor));
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_module_try_initialize_managed_global(
    iree_hal_streaming_module_t* module, const char* pointer_name,
    const char* initializer_name, bool* out_found, void** out_host_pointer,
    iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(pointer_name);
  IREE_ASSERT_ARGUMENT(initializer_name);
  IREE_ASSERT_ARGUMENT(out_found);
  IREE_ASSERT_ARGUMENT(out_host_pointer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_found = false;
  *out_host_pointer = NULL;
  *out_size = 0;

  // The initializer companion identifies a managed global. A same-named
  // pointer slot without this companion is an ordinary device global.
  iree_hal_streaming_symbol_t* initializer_symbol = NULL;
  bool initializer_found = false;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_try_lookup_global_symbol(
      module, initializer_name, &initializer_found, &initializer_symbol));
  if (!initializer_found) return iree_ok_status();
  *out_found = true;
  if (IREE_UNLIKELY(!initializer_symbol->executable)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "managed initializer `%s` is not owned by an executable",
        initializer_name);
  }

  iree_hal_streaming_symbol_t* pointer_symbol = NULL;
  bool pointer_found = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
          module, initializer_symbol->executable,
          iree_make_cstring_view(pointer_name), &pointer_found,
          &pointer_symbol));
  if (!pointer_found) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "managed global pointer slot `%s` not found",
                            pointer_name);
  }
  return iree_hal_streaming_module_initialize_managed_symbol_pair(
      module, pointer_symbol, initializer_symbol, out_host_pointer, out_size);
}
