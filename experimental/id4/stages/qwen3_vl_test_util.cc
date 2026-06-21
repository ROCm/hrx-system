// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/qwen3_vl_test_util.h"

#include <cstring>

#include "experimental/id4/stages/qwen3_vl.h"
#include "iree/hal/local/local_executable.h"

namespace id4::test {

const char kQwen3VlConditionKernelSource[] = R"(
config.decl @id4.qwen3_vl.condition.element_count : %value: index where [range(%value, 1, 1048576)]
config.decl @id4.qwen3_vl.condition.workgroup_size_x : %value: index where [range(%value, 1, 1024)]

kernel.def export("id4_qwen3_vl_condition_forward_f32") @id4_qwen3_vl_condition_forward_f32() {
  %unit = index.constant 1 : index
  %element_count = config.get @id4.qwen3_vl.condition.element_count : index
  %workgroup_size_x = config.get @id4.qwen3_vl.condition.workgroup_size_x : index
  %rounding = index.sub %workgroup_size_x, %unit : index
  %rounded = index.add %element_count, %rounding : index
  %workgroups_x = index.div %rounded, %workgroup_size_x : index
  kernel.launch.config workgroups(%workgroups_x, %unit, %unit) workgroup_size(%workgroup_size_x, %unit, %unit) : index
} launch(%selected_hidden_states: buffer, %condition: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %element_count = config.get @id4.qwen3_vl.condition.element_count : index
  %workgroup_size_x = config.get @id4.qwen3_vl.condition.workgroup_size_x : index
  %element_index = index.madd %workgroup, %workgroup_size_x, %lane : index
  %in_bounds = index.cmp ult, %element_index, %element_count : index

  %selected_noalias, %condition_noalias = buffer.assume.noalias %selected_hidden_states, %condition : buffer, buffer
  %selected_global = buffer.assume.memory_space<global> %selected_noalias : buffer
  %condition_global = buffer.assume.memory_space<global> %condition_noalias : buffer
  %selected_view = buffer.view %selected_global[%base] : buffer -> view<[%element_count]xf32, #dense>
  %condition_view = buffer.view %condition_global[%base] : buffer -> view<[%element_count]xf32, #dense>

  scf.if %in_bounds {
    %value = view.load %selected_view[%element_index] : view<[%element_count]xf32, #dense> -> f32
    view.store %value, %condition_view[%element_index] : f32, view<[%element_count]xf32, #dense>
  }
  kernel.return
}
)";

typedef struct Qwen3VlExecutable {
  // Local HAL executable base.
  iree_hal_local_executable_t base;
} Qwen3VlExecutable;

static iree_status_t Qwen3VlExecutableIssueCall(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  (void)executable;
  (void)workgroup_state;
  (void)worker_id;
  if (ordinal != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL executable ordinal is out of range");
  }
  if (dispatch_state->binding_count != 2 ||
      dispatch_state->binding_lengths[0] !=
          dispatch_state->binding_lengths[1] ||
      (dispatch_state->binding_lengths[0] % sizeof(float)) != 0 ||
      !dispatch_state->binding_ptrs[0] || !dispatch_state->binding_ptrs[1]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL executable requires matching selected and condition "
        "bindings");
  }
  std::memcpy(dispatch_state->binding_ptrs[1], dispatch_state->binding_ptrs[0],
              dispatch_state->binding_lengths[0]);
  return iree_ok_status();
}

static const iree_hal_executable_dispatch_attrs_v0_t kQwen3VlDispatchAttrs[] = {
    {
        /*.flags=*/IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        /*.local_memory_pages=*/0,
        /*.binding_count=*/2,
        /*.reserved_0=*/0,
        /*.workgroup_size_x=*/256,
        /*.workgroup_size_y=*/1,
        /*.workgroup_size_z=*/1,
        /*.parameter_count=*/2,
        /*.constant_byte_length=*/0,
    }};

static const iree_hal_executable_dispatch_v0_t kQwen3VlDispatchPtrs[] = {
    nullptr,
};

static const char* const kQwen3VlExportNames[] = {
    "id4_qwen3_vl_condition_forward_f32",
};

static Qwen3VlExecutable* Qwen3VlExecutableCast(
    iree_hal_executable_t* base_executable) {
  return reinterpret_cast<Qwen3VlExecutable*>(base_executable);
}

static void Qwen3VlExecutableDestroy(iree_hal_executable_t* base_executable) {
  Qwen3VlExecutable* executable = Qwen3VlExecutableCast(base_executable);
  iree_allocator_t host_allocator = executable->base.host_allocator;
  iree_hal_local_executable_deinitialize(&executable->base);
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t Qwen3VlExecutableFunctionCount(
    iree_hal_executable_t* base_executable) {
  (void)base_executable;
  return 1;
}

static iree_status_t Qwen3VlExecutableFunctionInfo(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  (void)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(function, 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL executable function is out of range");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  out_info->name = IREE_SV("id4_qwen3_vl_condition_forward_f32");
  out_info->binding_count = 2;
  out_info->parameter_count = 2;
  out_info->workgroup_size[0] = 256;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t Qwen3VlExecutableFunctionParameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(function, 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen3-VL executable function is out of range");
  }
  if (capacity < 2 || !out_parameters) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen3-VL executable parameter storage is required");
  }
  out_parameters[0] = (iree_hal_executable_function_parameter_t){
      // Selected hidden states are passed as binding ordinal zero.
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
      // No parameter flags are required.
      /*.flags=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE,
      // Binding parameters have no scalar byte size.
      /*.size=*/0,
      // Binding ordinal for selected hidden states.
      /*.offset=*/0,
      // Human-readable parameter name.
      /*.name=*/IREE_SV("selected_hidden_states"),
  };
  out_parameters[1] = (iree_hal_executable_function_parameter_t){
      // Condition is passed as binding ordinal one.
      /*.type=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
      // No parameter flags are required.
      /*.flags=*/IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE,
      // Binding parameters have no scalar byte size.
      /*.size=*/0,
      // Binding ordinal for the condition tensor.
      /*.offset=*/1,
      // Human-readable parameter name.
      /*.name=*/IREE_SV("condition"),
  };
  return iree_ok_status();
}

static iree_status_t Qwen3VlExecutableLookupFunctionByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  (void)base_executable;
  if (iree_string_view_equal(name,
                             IREE_SV("id4_qwen3_vl_condition_forward_f32"))) {
    *out_function = iree_hal_executable_function_from_index(0);
    return iree_ok_status();
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "Qwen3-VL executable function was not found");
}

static iree_status_t Qwen3VlExecutableTryLookupGlobalByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t Qwen3VlExecutableGlobalInfo(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  (void)out_info;
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static iree_status_t Qwen3VlExecutableGlobalBuffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = nullptr;
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static const iree_hal_local_executable_vtable_t kQwen3VlExecutableVTable = {
    // Base HAL executable vtable.
    /*.base=*/
    {
        // Releases executable storage.
        /*.destroy=*/Qwen3VlExecutableDestroy,
        // Returns the single Qwen3-VL function.
        /*.function_count=*/Qwen3VlExecutableFunctionCount,
        // Returns Qwen3-VL function metadata.
        /*.function_info=*/Qwen3VlExecutableFunctionInfo,
        // Returns Qwen3-VL function parameters.
        /*.function_parameters=*/Qwen3VlExecutableFunctionParameters,
        // Resolves the Qwen3-VL function by name.
        /*.lookup_function_by_name=*/
        Qwen3VlExecutableLookupFunctionByName,
        // Reports no globals.
        /*.try_lookup_global_by_name=*/
        Qwen3VlExecutableTryLookupGlobalByName,
        // Reports no global metadata.
        /*.global_info=*/Qwen3VlExecutableGlobalInfo,
        // Reports no global buffers.
        /*.global_buffer=*/Qwen3VlExecutableGlobalBuffer,
    },
    // Issues the local Qwen3-VL dispatch.
    /*.issue_call=*/Qwen3VlExecutableIssueCall,
};

static iree_status_t Qwen3VlExecutableCreate(
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = nullptr;
  Qwen3VlExecutable* executable = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*executable),
                            reinterpret_cast<void**>(&executable)));
  std::memset(executable, 0, sizeof(*executable));
  iree_hal_local_executable_initialize(&kQwen3VlExecutableVTable,
                                       host_allocator, &executable->base);
  executable->base.dispatch_attrs = kQwen3VlDispatchAttrs;
  executable->base.dispatch_ptrs = kQwen3VlDispatchPtrs;
  executable->base.export_count = 1;
  executable->base.export_names = kQwen3VlExportNames;
  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);
  return iree_ok_status();
}

static Qwen3VlExecutableCache* Qwen3VlExecutableCacheCast(
    iree_hal_executable_cache_t* base_executable_cache) {
  return reinterpret_cast<Qwen3VlExecutableCache*>(base_executable_cache);
}

static void Qwen3VlExecutableCacheDestroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  Qwen3VlExecutableCache* cache =
      Qwen3VlExecutableCacheCast(base_executable_cache);
  iree_allocator_free(cache->host_allocator, cache);
}

static iree_status_t Qwen3VlExecutableCacheInferFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  (void)caching_mode;
  Qwen3VlExecutableCache* cache =
      Qwen3VlExecutableCacheCast(base_executable_cache);
  ++cache->infer_count;
  if (executable_data.data_length < 4 || executable_data.data[0] != 0x7f ||
      executable_data.data[1] != 'E' || executable_data.data[2] != 'L' ||
      executable_data.data[3] != 'F') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected an ELF executable");
  }
  const iree_string_view_t format = IREE_SV("gfx1100");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  std::memcpy(executable_format, format.data, format.size);
  executable_format[format.size] = 0;
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool Qwen3VlExecutableCacheCanPrepareFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)caching_mode;
  Qwen3VlExecutableCache* cache =
      Qwen3VlExecutableCacheCast(base_executable_cache);
  ++cache->can_prepare_count;
  return iree_string_view_equal(executable_format, IREE_SV("gfx1100"));
}

static iree_status_t Qwen3VlExecutableCachePrepareExecutable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  Qwen3VlExecutableCache* cache =
      Qwen3VlExecutableCacheCast(base_executable_cache);
  ++cache->prepare_count;
  cache->last_caching_mode = executable_params->caching_mode;
  if (executable_params->executable_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data is empty");
  }
  return Qwen3VlExecutableCreate(cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t kQwen3VlExecutableCacheVTable =
    {
        // Releases the fake executable cache.
        /*.destroy=*/Qwen3VlExecutableCacheDestroy,
        // Infers the fake gfx1100 executable format from HSACO bytes.
        /*.infer_format=*/Qwen3VlExecutableCacheInferFormat,
        // Accepts the fake gfx1100 executable format.
        /*.can_prepare_format=*/Qwen3VlExecutableCacheCanPrepareFormat,
        // Prepares a local executable for the Qwen3-VL dispatch.
        /*.prepare_executable=*/Qwen3VlExecutableCachePrepareExecutable,
};

iree_status_t CreateQwen3VlExecutableCache(iree_allocator_t host_allocator,
                                           Qwen3VlExecutableCache** out_cache) {
  *out_cache = nullptr;
  Qwen3VlExecutableCache* cache = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*cache),
                                             reinterpret_cast<void**>(&cache)));
  std::memset(cache, 0, sizeof(*cache));
  iree_hal_resource_initialize(&kQwen3VlExecutableCacheVTable,
                               &cache->resource);
  cache->host_allocator = host_allocator;
  *out_cache = cache;
  return iree_ok_status();
}

iree_status_t CreateQwen3VlStage(iree_hal_device_group_t* device_group,
                                 Qwen3VlExecutableCache* executable_cache,
                                 id4_pipeline_kernel_cache_t* kernel_cache,
                                 iree_allocator_t host_allocator,
                                 id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache);
  services.host_allocator = host_allocator;

  id4_qwen3_vl_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = kernel_cache;
  options.source_identifier = IREE_SV("qwen3_vl_condition_f32.loom");
  options.source_contents =
      iree_make_const_byte_span(kQwen3VlConditionKernelSource,
                                std::strlen(kQwen3VlConditionKernelSource));
  options.module_name = IREE_SV("id4_qwen3_vl_condition_f32");
  options.executable_identifier =
      IREE_SV("id4_qwen3_vl_condition_forward_f32.hsaco");
  options.forward_function_name = IREE_SV("id4_qwen3_vl_condition_forward_f32");
  options.condition_token_count = 32;
  options.hidden_size = 64;
  options.workgroup_size_x = 256;
  return id4_qwen3_vl_stage_create(&options, host_allocator, out_stage);
}

}  // namespace id4::test
