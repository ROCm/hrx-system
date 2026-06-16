// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "common/fat_binary.h"
#include "common/internal.h"
#include "iree/io/file_handle.h"

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

static iree_string_view_t iree_hal_streaming_executable_target_value(
    const iree_hal_executable_target_t* target) {
  if (!iree_string_view_is_empty(target->loader_target)) {
    return target->loader_target;
  }
  return target->processor;
}

static iree_status_t iree_hal_streaming_fat_binary_target_append_unique(
    iree_hal_streaming_fat_binary_target_t* targets,
    iree_host_size_t target_capacity, iree_host_size_t* target_count,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fat-binary target value is empty");
  }
  for (iree_host_size_t i = 0; i < *target_count; ++i) {
    if (iree_string_view_equal(targets[i].value, value)) {
      return iree_ok_status();
    }
  }
  if (*target_count >= target_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fat-binary target capacity exceeded");
  }
  targets[*target_count].value = value;
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
      .policy = IREE_HAL_EXECUTABLE_TARGET_SELECTION_POLICY_EXACT_DEVICE,
      .family = IREE_SV("amdgpu"),
  };
  const iree_hal_executable_target_t* target = NULL;
  iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, &selection,
                                                    &target);
  if (result == IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec does not report an exact executable target");
  }
  if (result == IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous exact executable targets");
  }
  IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_target_append_unique(
      targets, target_capacity, &target_count,
      iree_hal_streaming_executable_target_value(target)));

  selection.policy =
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_POLICY_COMPATIBLE_GENERIC;
  target = NULL;
  result = iree_hal_device_spec_select_executable_target(device_spec,
                                                         &selection, &target);
  if (result == IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL device spec reports ambiguous generic executable targets");
  }
  if (result == IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_SELECTED) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_target_append_unique(
        targets, target_capacity, &target_count,
        iree_hal_streaming_executable_target_value(target)));
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

  iree_host_size_t constants_size = 0;

  // Analyze each export to determine operation counts.
  // We count the total operations per symbol with copy coalescing.
  iree_host_size_t total_ops = 0;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    const iree_host_size_t parameter_count = export_infos[i].parameter_count;
    if (!parameter_count) continue;
    // Query parameters to analyze coalescing opportunities.
    status = iree_hal_executable_export_parameters(
        export_executables[i], export_ordinals[i], parameter_count,
        &parameters[parameter_base]);
    if (!iree_status_is_ok(status)) break;
    // TOOD re-enable coalescing, which doesn't work for
    //      args arrays
    // uint32_t src_offset = 0;
    //  int32_t last_constant_end = -1;
    for (uint16_t j = 0; j < parameter_count; ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      if (parameter->type ==
          IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING) {
        ++symbol_op_counts[i].resolve_count;
        ++total_ops;
        // src_offset += parameter->size;
        //  last_constant_end = -1;  // break contiguity
      } else {
        //// CONSTANT or BUFFER_PTR - check for contiguity.
        //// Calculate source offset based on parameter order and sizes.
        // if (src_offset != last_constant_end) {
        //  New copy operation needed.
        ++symbol_op_counts[i].copy_count;
        ++total_ops;

        if (parameters[parameter_base + j].offset +
                parameters[parameter_base + j].size >
            constants_size) {
          // Track the maximum extent needed for the constants buffer.
          // Constants are packed at their kernarg offsets within the buffer.
          constants_size = parameters[parameter_base + j].offset +
                           parameters[parameter_base + j].size;
        }
        //}
        // src_offset += parameter->size;
        // last_constant_end = src_offset;
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
      (iree_hal_streaming_parameter_op_t*)(buffer + symbols_size);

  iree_hal_streaming_parameter_op_t* current_ops = ops_base;
  for (iree_host_size_t i = 0, parameter_base = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
    symbol->module = module;
    symbol->name = export_infos[i].name;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol->executable = export_executables[i];
    symbol->export_ordinal = export_ordinals[i];

    // Function attributes - TODO: Query from export metadata when available.
    // TODO(benvanik): populate from occupancy_info when available.
    symbol->occupancy_info = export_infos[i].occupancy_info;
    symbol->max_threads_per_block = 1024;       // TODO: from metadata.
    symbol->shared_size_bytes = 0;              // TODO: from metadata.
    symbol->local_size_bytes = 0;               // TODO: from metadata.
    symbol->num_regs = 32;                      // TODO: from metadata.
    symbol->max_dynamic_shared_size_bytes = 0;  // TODO: from metadata.

    // Initialize parameter info.
    iree_hal_streaming_parameter_info_t* parameter_info = &symbol->parameters;
    // Executable binding_count describes normal HAL dispatch bindings. HRX's
    // unpacker needs the number of reflected BINDING parameters it will
    // resolve from the HIP launch ABI.
    parameter_info->binding_count = symbol_op_counts[i].resolve_count;
    parameter_info->copy_count = symbol_op_counts[i].copy_count;
    parameter_info->ops = current_ops;
    const uint16_t parameter_count = export_infos[i].parameter_count;
    if (parameter_count == 0) {
      // No parameters.
      parameter_info->constant_bytes = 0;
      parameter_info->buffer_size = 0;
      continue;
    }

    // Build operations with coalescing.
    // Copy ops go first, then resolve ops.
    uint16_t src_offset = 0;
    uint16_t buffer_size = 0;
    size_t this_kernel_constants_size = 0;  // Per-kernel constants size
    iree_hal_streaming_parameter_op_t* copy_ops_start = current_ops;
    iree_hal_streaming_parameter_op_t* resolve_ops_start =
        current_ops + symbol_op_counts[i].copy_count;
    uint16_t copy_count = 0;
    uint16_t resolve_count = 0;
    for (uint16_t j = 0; j < parameter_count; ++j) {
      const iree_hal_executable_export_parameter_t* parameter =
          &parameters[parameter_base + j];
      if (parameter->type ==
          IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING) {
        // Update offsets. Bindings are passed as pointers.
        // |parameter->offset| is the kernarg byte offset for all parameter
        // types when the backend populates it (e.g. AMDGPU HSACO). The
        // binding-list ordinal is recovered by iteration: |resolve_count|
        // is the running count of BINDING parameters seen so far, which is
        // exactly the index of this parameter in the bindings list.
        iree_hal_streaming_parameter_resolve_op_t* op =
            &resolve_ops_start[resolve_count].resolve;
        op->src_offset = src_offset;
        op->dst_ordinal = resolve_count;
        op->src_ordinal = j;
        op->dst_offset = parameter->offset;
        src_offset += parameter->size;
        buffer_size = src_offset;
        ++resolve_count;

        // For native kernels with CUSTOM_DIRECT_ARGUMENTS, bindings are also
        // part of the constants buffer. Track their extent as well.
        size_t param_extent = (size_t)parameter->offset + parameter->size;
        if (param_extent > this_kernel_constants_size) {
          this_kernel_constants_size = param_extent;
        }
      } else {
        // TODO: fix coalescing. It does not work when we have
        // parameter arrays because each constant comes in as a
        // separate parameter with its own offset.
        //// CONSTANT or BUFFER_PTR - try to coalesce and choose offsets.
        // if (active_copy &&
        //     active_copy->src_offset + active_copy->size == src_offset) {
        //   // Extend the current copy operation.
        //   active_copy->size += parameter->size;
        // } else {
        //  Start a new copy operation.
        iree_hal_streaming_parameter_copy_op_t* op =
            &copy_ops_start[copy_count].copy;
        op->size = parameter->size;
        op->src_offset = src_offset;
        op->src_ordinal = j;
        op->dst_offset = parameter->offset;  // offset in constants
        ++copy_count;
        // active_copy = op;
        // }
        src_offset += parameter->size;
        buffer_size = src_offset;

        // Track per-kernel constants size based on actual parameter extent
        size_t param_extent = parameter->offset + parameter->size;
        if (param_extent > this_kernel_constants_size) {
          this_kernel_constants_size = param_extent;
        }
      }
    }
    parameter_info->buffer_size = buffer_size;
    // The HAL expects the constants buffer to span the entire kernarg
    // segment reported by the export (includes padding between args and
    // trailing alignment). The per-parameter extent we tracked above
    // typically matches, but pad up to the export's declared size to
    // satisfy the strict length check in the dispatch code path.
    size_t export_constant_bytes = export_infos[i].constant_byte_length;
    if (export_constant_bytes > this_kernel_constants_size) {
      this_kernel_constants_size = export_constant_bytes;
    }
    parameter_info->constant_bytes = this_kernel_constants_size;

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

iree_status_t iree_hal_streaming_module_create_from_memory(
    iree_hal_streaming_context_t* context,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t image, iree_allocator_t host_allocator,
    iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(image.data);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate module structure up-front — we stash the fat-binary extract
  // directly on it so the (possibly decompressed) ELF backing store lives
  // as long as the HAL executable that may still alias it.
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
  module->cache = context->executable_cache;
  iree_hal_executable_cache_retain(module->cache);

  // HIP / CUDA hand us anything the toolchain emits — raw AMDGPU ELFs,
  // __CLANG_OFFLOAD_BUNDLE__ archives, CCOB (zstd-compressed bundles), and
  // __hipFatBinaryWrapper-wrapped combinations of all of the above. Unwrap
  // everything here and only forward raw ELF plus an explicit executable
  // format to the HAL executable cache.
  iree_const_byte_span_t executable_data = image;
  const char* executable_format = NULL;
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
          image, target_count, targets, host_allocator, &module->fat_extract);
    }
    if (iree_status_is_ok(status)) {
      // Multiple matches are possible (e.g. Tensile feature-specialized
      // kernels). Load all of them below and merge their exports into one HIP
      // module namespace.
      executable_data = module->fat_extract.matches[0].data;
      executable_format = module->fat_extract.matches[0].executable_format;
    }
  } else {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "module binary is not a supported HRX AMDGPU "
                              "ELF, offload bundle, CCOB, or HIP fat binary");
  }

  // Create HAL executable from binary.
  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t params;
    iree_hal_executable_params_initialize(&params);
    params.caching_mode = caching_mode;
    params.executable_format = iree_make_cstring_view(executable_format);
    params.executable_data = executable_data;
    status = iree_hal_executable_cache_prepare_executable(
        module->cache, &params, &module->executable);
  }

  // If the fat binary had multiple matching HSACO entries, prepare all of
  // them and expose their exports through the same hipModule_t. Native HIP lets
  // libraries such as hipBLAS/Tensile probe one module handle for a kernel that
  // may live in a later matching code object.
  if (iree_status_is_ok(status) && module->fat_extract.match_count > 1) {
    module->executable_count = module->fat_extract.match_count;
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
      iree_const_byte_span_t match_data = module->fat_extract.matches[i].data;

      iree_hal_executable_params_t params;
      iree_hal_executable_params_initialize(&params);
      params.caching_mode = caching_mode;
      params.executable_format = iree_make_cstring_view(
          module->fat_extract.matches[i].executable_format);
      params.executable_data = match_data;
      status = iree_hal_executable_cache_prepare_executable(
          module->cache, &params, &module->executables[i]);
    }
  }

  // Extract kernel metadata.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_extract_metadata(module);
  }

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
    iree_hal_executable_caching_mode_t caching_mode, iree_string_view_t path,
    iree_allocator_t host_allocator, iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Open the file for reading.
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, path, host_allocator,
                                   &file_handle));

  // Map the entire file for read access.
  iree_io_file_mapping_t* file_mapping = NULL;
  iree_status_t status = iree_io_file_map_view(
      file_handle, IREE_IO_FILE_ACCESS_READ, 0, IREE_HOST_SIZE_MAX,
      IREE_IO_FILE_MAPPING_FLAG_NONE, host_allocator, &file_mapping);

  // Release the file handle (mapping retains it).
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
      context,
      caching_mode | IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
      image, host_allocator, &module);

  if (iree_status_is_ok(status)) {
    module->file_mapping = file_mapping;
    *out_module = module;
  } else {
    iree_io_file_mapping_release(file_mapping);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_module_destroy(
    iree_hal_streaming_module_t* module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = module->host_allocator;

  // Release file mapping if present.
  iree_io_file_mapping_release(module->file_mapping);

  // Release symbol metadata.
  iree_allocator_free(module->host_allocator, module->symbols);

  // Release cached executable globals while both the context pointer map and
  // executable-owned global buffers are still live.
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
    iree_hal_streaming_memory_release_wrapped_buffer(
        module->globals[i]->global_buffer);
    iree_allocator_free(host_allocator, module->globals[i]);
  }
  iree_allocator_free(host_allocator, module->globals);
  iree_slim_mutex_deinitialize(&module->global_mutex);

  // Release executable before the fat-binary extract: the HAL's code
  // object reader may still alias pointers into the (possibly owned)
  // backing store held by the extract until the executable drops.
  if (module->executables) {
    for (iree_host_size_t i = 0; i < module->executable_count; ++i) {
      iree_hal_executable_release(module->executables[i]);
    }
    iree_allocator_free(host_allocator, module->executables);
  } else {
    iree_hal_executable_release(module->executable);
  }

  // Drop fat-binary / offload-bundle unpacking buffers.
  iree_hal_streaming_fat_binary_extract_reset(&module->fat_extract);

  // Release executable cache.
  iree_hal_executable_cache_release(module->cache);

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

  *out_device_ptr = symbol->device_address;
  if (out_size) *out_size = symbol->size_bytes;
  return iree_ok_status();
}
