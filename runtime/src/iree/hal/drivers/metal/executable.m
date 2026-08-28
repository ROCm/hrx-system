// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable.h"

#include <Metal/Metal.h>
#include <stddef.h>

#include "iree/base/api.h"

typedef struct iree_hal_metal_executable_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  NSArray<id<MTLLibrary>>* libraries;

  iree_host_size_t pipeline_count;
  iree_hal_metal_pipeline_t pipelines[];
} iree_hal_metal_executable_t;

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable;

static iree_hal_metal_executable_t* iree_hal_metal_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (iree_hal_metal_executable_t*)base_value;
}

static const iree_hal_metal_executable_t* iree_hal_metal_executable_const_cast(
    const iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_executable_vtable);
  return (const iree_hal_metal_executable_t*)base_value;
}

static iree_status_t iree_hal_metal_compile_source(
    id<MTLDevice> device, const iree_hal_metal_executable_library_t* library_def,
    id<MTLLibrary>* out_library) {
  *out_library = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(library_def->source_version !=
                        IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT &&
                    library_def->source_version > MTLLanguageVersion3_0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MSL language version %u is unsupported by the compiled runtime",
                            library_def->source_version);
  }

  iree_status_t status = iree_ok_status();
  id<MTLLibrary> library = nil;
  @autoreleasepool {
    MTLCompileOptions* compile_options = [[MTLCompileOptions new] autorelease];
    compile_options.languageVersion = MTLLanguageVersion3_0;
    if (library_def->source_version != IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT) {
      compile_options.languageVersion = (MTLLanguageVersion)library_def->source_version;
    }

    NSString* source = [[[NSString alloc] initWithBytes:library_def->source.data
                                                 length:library_def->source.size
                                               encoding:NSUTF8StringEncoding] autorelease];
    if (IREE_UNLIKELY(source == nil)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "MSL source is not valid UTF-8");
    } else {
      NSError* error = nil;
      library = [device newLibraryWithSource:source options:compile_options error:&error];  // +1
      if (IREE_UNLIKELY(library == nil)) {
        const char* error_message =
            [error.localizedDescription cStringUsingEncoding:NSUTF8StringEncoding];
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to create MTLLibrary: %s",
                                  error_message ? error_message : "unknown Metal error");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    [library release];
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_load_library(id<MTLDevice> device,
                                                 iree_const_byte_span_t metallib,
                                                 id<MTLLibrary>* out_library) {
  *out_library = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  id<MTLLibrary> library = nil;
  @autoreleasepool {
    dispatch_data_t data = dispatch_data_create(metallib.data, metallib.data_length,
                                                /*queue=*/NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (IREE_UNLIKELY(data == nil)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "failed to allocate metallib data");
    } else {
      NSError* error = nil;
      library = [device newLibraryWithData:data error:&error];  // +1
      dispatch_release(data);
      if (IREE_UNLIKELY(library == nil)) {
        const char* error_message =
            [error.localizedDescription cStringUsingEncoding:NSUTF8StringEncoding];
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "failed to create MTLLibrary: %s",
                                  error_message ? error_message : "unknown Metal error");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    [library release];
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Loads all MTLLibrary instances in the executable and returns an array with
// matching order.
static iree_status_t iree_hal_metal_load_libraries(id<MTLDevice> device,
                                                   const iree_hal_metal_executable_format_t* format,
                                                   NSArray<id<MTLLibrary>>** out_libraries) {
  *out_libraries = nil;
  IREE_TRACE_ZONE_BEGIN(z0);

  NSMutableArray<id<MTLLibrary>>* libraries =
      [[NSMutableArray alloc] initWithCapacity:format->library_count];  // +1

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < format->library_count && iree_status_is_ok(status); ++i) {
    iree_hal_metal_executable_library_t library_def;
    status = iree_hal_metal_executable_format_read_library(format, i, &library_def);
    if (!iree_status_is_ok(status)) break;
    id<MTLLibrary> library = nil;
    if (library_def.metallib.data_length > 0) {
      status = iree_hal_metal_load_library(device, library_def.metallib, &library);
    } else {
      status = iree_hal_metal_compile_source(device, &library_def, &library);
    }
    if (!iree_status_is_ok(status)) break;
    [libraries addObject:library];
    [library release];  // Ownership transferred to the array.
  }

  if (iree_status_is_ok(status)) {
    *out_libraries = libraries;
  } else {
    [libraries release];  // -1
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_metal_create_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library,
    const iree_hal_metal_executable_pipeline_t* pipeline_def,
    iree_hal_metal_pipeline_t* out_pipeline) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, pipeline_def->entry_point.data, pipeline_def->entry_point.size);

  iree_status_t status = iree_ok_status();
  const MTLSize device_maximum = [device maxThreadsPerThreadgroup];
  if (IREE_UNLIKELY(pipeline_def->threadgroup_size[0] > device_maximum.width ||
                    pipeline_def->threadgroup_size[1] > device_maximum.height ||
                    pipeline_def->threadgroup_size[2] > device_maximum.depth)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Metal pipeline `%.*s` threadgroup size %ux%ux%u exceeds the device "
                              "maximum %zux%zux%zu",
                              (int)pipeline_def->entry_point.size, pipeline_def->entry_point.data,
                              pipeline_def->threadgroup_size[0], pipeline_def->threadgroup_size[1],
                              pipeline_def->threadgroup_size[2], device_maximum.width,
                              device_maximum.height, device_maximum.depth);
  }

  @autoreleasepool {
    NSString* function_name = nil;
    if (iree_status_is_ok(status)) {
      function_name = [[[NSString alloc] initWithBytes:pipeline_def->entry_point.data
                                                length:pipeline_def->entry_point.size
                                              encoding:NSUTF8StringEncoding] autorelease];
      if (IREE_UNLIKELY(function_name == nil)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "Metal entry-point name is not valid UTF-8");
      }
    }
    if (iree_status_is_ok(status)) {
      out_pipeline->function = [library newFunctionWithName:function_name];  // +1
      if (IREE_UNLIKELY(out_pipeline->function == nil)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "function `%.*s` not found in MTLLibrary",
            (int)pipeline_def->entry_point.size, pipeline_def->entry_point.data);
      }
    }

    if (iree_status_is_ok(status)) {
      MTLComputePipelineDescriptor* descriptor =
          [[[MTLComputePipelineDescriptor alloc] init] autorelease];
      [descriptor setComputeFunction:out_pipeline->function];
      [descriptor setLabel:function_name];
      if (pipeline_def->max_threads_per_threadgroup != 0) {
        [descriptor setMaxTotalThreadsPerThreadgroup:pipeline_def->max_threads_per_threadgroup];
      }
      [descriptor
          setThreadGroupSizeIsMultipleOfThreadExecutionWidth:
              iree_any_bit_set(pipeline_def->flags,
                               IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAG_THREADGROUP_SIZE_ALIGNED)];
      [[[descriptor buffers] objectAtIndexedSubscript:0] setMutability:MTLMutabilityImmutable];
      [[[descriptor buffers] objectAtIndexedSubscript:IREE_HAL_METAL_PUSH_CONSTANT_BUFFER_INDEX]
          setMutability:MTLMutabilityImmutable];

      NSError* error = nil;
      out_pipeline->pipeline_state =
          [device newComputePipelineStateWithDescriptor:descriptor
                                                options:MTLPipelineOptionNone
                                             reflection:nil
                                                  error:&error];
      if (IREE_UNLIKELY(out_pipeline->pipeline_state == nil)) {
        const char* error_message =
            [error.localizedDescription cStringUsingEncoding:NSUTF8StringEncoding];
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "failed to create pipeline with function `%.*s`: %s",
            (int)pipeline_def->entry_point.size, pipeline_def->entry_point.data,
            error_message ? error_message : "unknown Metal error");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    const uint32_t threadgroup_size = pipeline_def->threadgroup_size[0] *
                                      pipeline_def->threadgroup_size[1] *
                                      pipeline_def->threadgroup_size[2];
    if (IREE_UNLIKELY(threadgroup_size >
                      out_pipeline->pipeline_state.maxTotalThreadsPerThreadgroup)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Metal pipeline `%.*s` requires %u threads per threadgroup, "
                                "exceeding the pipeline maximum of %zu",
                                (int)pipeline_def->entry_point.size, pipeline_def->entry_point.data,
                                threadgroup_size,
                                out_pipeline->pipeline_state.maxTotalThreadsPerThreadgroup);
    }
  }
  if (iree_status_is_ok(status)) {
    out_pipeline->threadgroup_size =
        MTLSizeMake(pipeline_def->threadgroup_size[0], pipeline_def->threadgroup_size[1],
                    pipeline_def->threadgroup_size[2]);
    out_pipeline->constant_count = pipeline_def->constant_count;
    out_pipeline->binding_count = pipeline_def->binding_count;
    out_pipeline->binding_read_only_bits = pipeline_def->binding_read_only_bits;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_metal_executable_create(id<MTLDevice> device,
                                               const iree_hal_executable_load_params_t* load_params,
                                               iree_allocator_t host_allocator,
                                               iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(load_params->constant_count != 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Metal executable specialization constants are not supported");
  }

  iree_hal_metal_executable_format_t executable_format;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_executable_format_parse(load_params->executable_data, &executable_format));
  const iree_host_size_t pipeline_count = executable_format.pipeline_count;

  // Calculate the total number of characters across all entry point names so
  // that the executable does not retain the encoded bundle.
  iree_host_size_t total_pipeline_name_length = 0;
  for (iree_host_size_t i = 0; i < pipeline_count; ++i) {
    iree_hal_metal_executable_pipeline_t pipeline_def;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_metal_executable_format_read_pipeline(&executable_format, i, &pipeline_def));
    if (IREE_UNLIKELY(!iree_host_size_checked_add(total_pipeline_name_length,
                                                  pipeline_def.entry_point.size,
                                                  &total_pipeline_name_length))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "Metal pipeline name storage exceeds host size "
                              "limits");
    }
  }

  // Allocate storage for the executable and its associated data structures.
  iree_hal_metal_executable_t* executable = NULL;
  iree_host_size_t pipeline_table_size = 0;
  iree_host_size_t total_size = sizeof(*executable);
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(pipeline_count, sizeof(executable->pipelines[0]),
                                      &pipeline_table_size) ||
          !iree_host_size_checked_add(total_size, pipeline_table_size, &total_size) ||
          !iree_host_size_checked_add(total_size, total_pipeline_name_length, &total_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Metal executable storage exceeds host size "
                            "limits");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  memset(executable, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_metal_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;
  executable->pipeline_count = pipeline_count;
  char* pipeline_name_ptr = (char*)executable->pipelines + pipeline_table_size;

  // Load all libraries that may be referenced by the pipelines.
  iree_status_t status =
      iree_hal_metal_load_libraries(device, &executable_format, &executable->libraries);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < pipeline_count && iree_status_is_ok(status); ++i) {
      iree_hal_metal_executable_pipeline_t pipeline_def;
      status = iree_hal_metal_executable_format_read_pipeline(&executable_format, i, &pipeline_def);
      if (!iree_status_is_ok(status)) break;
      id<MTLLibrary> library =
          [executable->libraries objectAtIndex:pipeline_def.library_ordinal];  // unretained

      iree_hal_metal_pipeline_t* pipeline = &executable->pipelines[i];
      status = iree_hal_metal_create_pipeline(device, library, &pipeline_def, pipeline);
      if (!iree_status_is_ok(status)) break;

      pipeline->name = iree_make_string_view(pipeline_name_ptr, pipeline_def.entry_point.size);
      memcpy(pipeline_name_ptr, pipeline_def.entry_point.data, pipeline_def.entry_point.size);
      pipeline_name_ptr += pipeline_def.entry_point.size;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_executable_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    iree_hal_metal_pipeline_t* entry_point = &executable->pipelines[i];
    [entry_point->pipeline_state release];  // -1
    [entry_point->function release];        // -1
  }

  [executable->libraries release];  // -1

  iree_allocator_free(executable->host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_metal_executable_lookup_pipeline(
    const iree_hal_executable_t* base_executable, iree_hal_executable_function_t function,
    const iree_hal_metal_pipeline_t** out_pipeline) {
  const iree_hal_metal_executable_t* executable =
      iree_hal_metal_executable_const_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(function, executable->pipeline_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64 " out of range (count: %" PRIhsz ")",
                            function.value, executable->pipeline_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  *out_pipeline = &executable->pipelines[export_ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_metal_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  return executable->pipeline_count;
}

static iree_status_t iree_hal_metal_executable_export_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_function_t export_ordinal,
    iree_hal_executable_function_info_t* out_info) {
  const iree_hal_metal_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_metal_executable_lookup_pipeline(base_executable, export_ordinal, &pipeline));
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = pipeline->name;
  out_info->flags = IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE;
  out_info->constant_byte_length = pipeline->constant_count * sizeof(uint32_t);
  out_info->binding_count = (uint16_t)pipeline->binding_count;
  out_info->workgroup_size[0] = (uint32_t)pipeline->threadgroup_size.width;
  out_info->workgroup_size[1] = (uint32_t)pipeline->threadgroup_size.height;
  out_info->workgroup_size[2] = (uint32_t)pipeline->threadgroup_size.depth;
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_executable_export_parameters(
    iree_hal_executable_t* base_executable, iree_hal_executable_function_t export_ordinal,
    iree_host_size_t capacity, iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  (void)executable;
  // TODO(metal): return export parameter information from kernel metadata.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "parameter reflection not implemented");
}

static iree_status_t iree_hal_metal_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    if (iree_string_view_equal(executable->pipelines[i].name, name)) {
      *out_export_ordinal = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "function '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_metal_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name, bool* out_found,
    iree_hal_executable_global_t* out_global) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  (void)executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  (void)executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid Metal executable global");
}

static iree_status_t iree_hal_metal_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_metal_executable_t* executable = iree_hal_metal_executable_cast(base_executable);
  (void)executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid Metal executable global");
}

static const iree_hal_executable_vtable_t iree_hal_metal_executable_vtable = {
    .destroy = iree_hal_metal_executable_destroy,
    .function_count = iree_hal_metal_executable_export_count,
    .function_info = iree_hal_metal_executable_export_info,
    .function_parameters = iree_hal_metal_executable_export_parameters,
    .lookup_function_by_name = iree_hal_metal_executable_lookup_export_by_name,
    .try_lookup_global_by_name = iree_hal_metal_executable_try_lookup_global_by_name,
    .global_info = iree_hal_metal_executable_global_info,
    .global_buffer = iree_hal_metal_executable_global_buffer,
};
