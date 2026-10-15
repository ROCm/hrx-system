// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/executable.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/maintenance.h"

//===----------------------------------------------------------------------===//
// Native image ownership
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_gpu_executable_t {
  // Exact-family HAL resource retained by captured dispatches.
  iree_hal_executable_t base;
  // Allocator for this object and its inline cold tables.
  iree_allocator_t host_allocator;
  // Borrowed native owner, dominated by the parent HAL device.
  const iree_hal_amd_gpu_context_t* context;
  // Owned parsed artifact, including reflection and retained source bytes.
  iree_hal_amd_gpu_image_t image;
  // Owned code/global backing and persistent upload/global mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Host and exact-device addresses of image-relative byte zero.
  iree_hal_buffer_native_binding_t bindings[2];
  // Captured native geometry shared by every global view.
  iree_hal_buffer_backing_facts_t backing;
  // Achieved host visibility and coherence of the backing.
  iree_hal_memory_type_t memory_type;
  // Immutable per-function state in metadata ordinal order.
  iree_hal_amd_gpu_pm4_function_t* functions;
  // Contracts indexed by READ/WRITE permissions; only used classes exist.
  iree_hal_memory_contract_t* contracts[4];
  // Cold global-view publication lock; never acquired during dispatch.
  iree_slim_mutex_t global_mutex;
  // Lazy executable-owned views in image symbol order.
  iree_hal_buffer_t** globals;
  // Native cleanup ownership after the last HAL reference is released.
  struct {
    // Borrowed shared device service, joined before native context teardown.
    iree_hal_memory_maintenance_t* maintenance;
    // Embedded one-shot work; final release performs no allocation.
    iree_hal_memory_maintenance_entry_t entry;
  } cleanup;
} iree_hal_amd_gpu_executable_t;

// Resource initialization precedes definitions of the reflection methods.
static const iree_hal_executable_vtable_t iree_hal_amd_gpu_executable_vtable;

static void iree_hal_amd_gpu_executable_free(void* user_data) {
  iree_hal_amd_gpu_executable_t* executable = user_data;
  for (iree_host_size_t i = 0; i < executable->image.symbol_count; ++i) {
    iree_hal_buffer_release(executable->globals[i]);
  }
  iree_slim_mutex_deinitialize(&executable->global_mutex);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(executable->contracts); ++i) {
    iree_hal_memory_contract_release(executable->contracts[i]);
  }
  iree_hal_amd_fabric_memory_deinitialize(executable->context->api,
                                          executable->context->event_sink,
                                          &executable->memory);
  for (iree_host_size_t i = 0; i < executable->image.metadata.kernel_count;
       ++i) {
    iree_hal_amd_gpu_kernarg_layout_free(executable->functions[i].kernarg);
  }
  iree_hal_amd_gpu_image_deinitialize(&executable->image);
  iree_allocator_free(executable->host_allocator, executable);
}

static void iree_hal_amd_gpu_executable_cleanup(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_amd_gpu_executable_free(
      (uint8_t*)entry - offsetof(iree_hal_amd_gpu_executable_t, cleanup.entry));
}

static void iree_hal_amd_gpu_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_amd_gpu_executable_t* executable =
      (iree_hal_amd_gpu_executable_t*)base_executable;
  iree_hal_memory_maintenance_enqueue(executable->cleanup.maintenance,
                                      &executable->cleanup.entry);
}

static iree_status_t iree_hal_amd_gpu_executable_parse(
    iree_const_byte_span_t data, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_image_t* out_image) {
  iree_byte_span_t copy = iree_make_byte_span(NULL, data.data_length);
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(host_allocator, data, (void**)&copy.data));
  iree_byte_sequence_t* source = NULL;
  iree_status_t status =
      iree_byte_sequence_create_from_span_move(&copy, host_allocator, &source);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_hsaco_image_initialize(source, host_allocator,
                                                     out_image);
  }
  iree_byte_sequence_release(source);
  iree_allocator_free(host_allocator, copy.data);
  return status;
}

// External artifact requirements are admitted before native allocation. Runtime
// services belonging to an individual dispatch remain recorded on its function.
static iree_status_t iree_hal_amd_gpu_executable_admit(
    const iree_hal_amd_gpu_image_t* image,
    const iree_hal_amd_gpu_target_identity_t* target) {
  if (image->metadata.kernel_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU function count exceeds HAL token indices");
  }
  if (image->dependency_count) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "GPU image requires external dependency `%.*s`",
                            (int)image->dependencies[0].size,
                            image->dependencies[0].data);
  }
  for (iree_host_size_t i = 0; i < image->import_count; ++i) {
    if (!image->imports[i].weak) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "GPU image requires external symbol `%.*s`",
                              (int)image->imports[i].name.size,
                              image->imports[i].name.data);
    }
  }
  // These globals request a host service even when the image supplies their
  // zero-initialized storage. Loading that storage without provisioning the
  // service would silently disable shader diagnostics and assertion reports.
  const iree_string_view_t service_globals[] = {
      IREE_SV("iree_asan_config"),
      IREE_SV("iree_tsan_config"),
      IREE_SV("iree_feedback_config"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(service_globals); ++i) {
    if (iree_hal_amd_gpu_image_find_symbol(image, service_globals[i])) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "GPU image requires runtime service global `%.*s`, which the native "
          "device has not provisioned",
          (int)service_globals[i].size, service_globals[i].data);
    }
  }
  iree_hal_amd_gpu_target_identity_t artifact = image->target;
  for (iree_host_size_t i = 0; i < image->metadata.kernel_count; ++i) {
    const iree_hal_amd_gpu_metadata_kernel_t* kernel =
        &image->metadata.kernels[i];
    if (kernel->kind != IREE_HAL_AMD_GPU_METADATA_KERNEL_KIND_NORMAL ||
        !iree_string_view_is_empty(kernel->device_enqueue_symbol)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "GPU image requires module initialization or device-enqueue linkage");
    }
    // LLVM emits this annotation for other processors too. It constrains only
    // GFX1250; interpreting it as an unconditional target would reject ordinary
    // GFX11 and GFX12.0 code objects produced by the same compiler.
    if (target->version.major == 12 && target->version.minor == 5 &&
        target->version.stepping == 0 &&
        !iree_string_view_is_empty(kernel->gfx1250_revision)) {
      iree_string_view_t required_target;
      if (iree_string_view_equal(kernel->gfx1250_revision, IREE_SV("A0"))) {
        required_target = IREE_SV("gfx1250-a0");
      } else if (iree_string_view_equal(kernel->gfx1250_revision,
                                        IREE_SV("B0"))) {
        required_target = IREE_SV("gfx1250");
      } else {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED, "unknown GPU code revision `%.*s`",
            (int)kernel->gfx1250_revision.size, kernel->gfx1250_revision.data);
      }
      if (!iree_string_view_equal(target->target, required_target)) {
        return iree_make_status(
            IREE_STATUS_INCOMPATIBLE, "GPU kernel requires target `%.*s`",
            (int)required_target.size, required_target.data);
      }
      if (artifact.kind == IREE_HAL_AMD_GPU_TARGET_KIND_EXACT &&
          artifact.version.major == 12 && artifact.version.minor == 5 &&
          artifact.version.stepping == 0) {
        artifact.target = required_target;
      }
    }
  }
  const iree_hal_amd_gpu_target_compatibility_t compatibility =
      iree_hal_amd_gpu_target_identity_check_compatible(&artifact, target);
  if (compatibility) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "GPU artifact `%.*s` is incompatible with `%.*s` (mismatch bits 0x%x)",
        (int)artifact.target.size, artifact.target.data,
        (int)target->target.size, target->target.data, compatibility);
  }
  return iree_ok_status();
}

static iree_hal_memory_access_t iree_hal_amd_gpu_executable_global_access(
    const iree_hal_amd_gpu_image_symbol_t* symbol) {
  if (symbol->kind != IREE_HAL_AMD_GPU_IMAGE_SYMBOL_DATA ||
      symbol->address.kind != IREE_HAL_AMD_GPU_IMAGE_ADDRESS_IMAGE) {
    return IREE_HAL_MEMORY_ACCESS_NONE;
  }
  return (iree_any_bit_set(symbol->access, IREE_HAL_AMD_GPU_IMAGE_ACCESS_READ)
              ? IREE_HAL_MEMORY_ACCESS_READ
              : 0) |
         (iree_any_bit_set(symbol->access, IREE_HAL_AMD_GPU_IMAGE_ACCESS_WRITE)
              ? IREE_HAL_MEMORY_ACCESS_WRITE
              : 0);
}

typedef struct iree_hal_amd_gpu_executable_transition_query_t {
  // Borrowed native API table.
  const amdf_api_t* api;
  // Exact sealed-group queue ID; the following ID is program execution.
  iree_hal_memory_scope_id_t queue_scope_id;
  // Established host view and exact native queue access, respectively.
  amdf_memory_site_t sites[2];
  // Largest independently maintained granule established by the native pairs.
  iree_device_size_t maintenance_alignment;
} iree_hal_amd_gpu_executable_transition_query_t;

static iree_status_t iree_hal_amd_gpu_executable_query_transition(
    void* user_data, iree_hal_memory_scope_id_t producer,
    iree_hal_memory_scope_id_t consumer,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_amd_gpu_executable_transition_query_t* query = user_data;
  // The native API qualifies queue cache operations, not shader instructions.
  if (producer == query->queue_scope_id + 1 ||
      consumer == query->queue_scope_id + 1) {
    return iree_ok_status();
  }
  amdf_memory_pair_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(native),
  };
  const amdf_status_t status = query->api->memory_query_pair_info(
      &query->sites[producer == 1 ? 0 : 1],
      &query->sites[consumer == 1 ? 0 : 1], &native);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "memory_query_pair_info"));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_fabric_memory_translate_pair(&native, out_info));
  query->maintenance_alignment =
      iree_max(query->maintenance_alignment,
               iree_max(out_info->release.range_granularity,
                        out_info->acquire.range_granularity));
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_initialize_contracts(
    iree_hal_amd_gpu_executable_t* executable,
    const iree_hal_amd_gpu_executable_options_t* options) {
  const iree_hal_queue_family_t* family = executable->base.queue_family;
  const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_HOST,
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(executable->bindings),
      .binding_count = IREE_ARRAYSIZE(types),
      .host_binding_index = 0,
      .types = types,
  };
  iree_hal_amd_gpu_executable_transition_query_t query = {
      .api = executable->context->api,
      .queue_scope_id = family->memory.queue_scope_id,
      .sites =
          {
              {
                  .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                  .structure_size = sizeof(amdf_memory_site_t),
                  .kind = AMDF_MEMORY_SITE_KIND_HOST,
                  .value.host.mapping = executable->memory.mapping,
              },
              {
                  .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
                  .structure_size = sizeof(amdf_memory_site_t),
                  .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
                  .value.device =
                      {
                          .memory = executable->memory.handle,
                          .access_ordinal = 0,
                          .queue_family_ordinal =
                              options->native_family_ordinal,
                      },
              },
          },
      .maintenance_alignment = 1,
  };
  uint32_t access_classes = 0;
  for (iree_host_size_t i = 0; i < executable->image.symbol_count; ++i) {
    access_classes |= 1u << iree_hal_amd_gpu_executable_global_access(
                          &executable->image.symbols[i]);
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t access = 1; access < IREE_ARRAYSIZE(executable->contracts) &&
                            iree_status_is_ok(status);
       ++access) {
    if (!(access_classes & (1u << access))) {
      continue;
    }
    iree_hal_memory_contract_t* contract = NULL;
    status = iree_hal_memory_contract_create(
        family->memory.domain, family->memory.queue_scope_id + 2, &layout,
        executable->host_allocator, &contract);
    if (!iree_status_is_ok(status)) {
      break;
    }
    executable->contracts[access] = contract;
    iree_hal_buffer_usage_t usage =
        (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)
             ? IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
                   IREE_HAL_BUFFER_USAGE_STORAGE_READ |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ
             : 0) |
        (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)
             ? IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_STORAGE_WRITE
             : 0);
    const iree_hal_memory_access_t host_access =
        access & ((iree_any_bit_set(options->memory->host_access,
                                    AMDF_MEMORY_MAP_FLAG_READ)
                       ? IREE_HAL_MEMORY_ACCESS_READ
                       : 0) |
                  (iree_any_bit_set(options->memory->host_access,
                                    AMDF_MEMORY_MAP_FLAG_WRITE)
                       ? IREE_HAL_MEMORY_ACCESS_WRITE
                       : 0));
    contract->allowed_access = access;
    contract->placement = options->placement;
    contract->allowed_usage =
        usage |
        (host_access ? IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                           IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
                     : 0) |
        (executable->memory.exports.class_count
             ? IREE_HAL_BUFFER_USAGE_SHARING_EXPORT
             : 0);
    contract->host = (iree_hal_pool_host_access_t){
        .access = host_access,
        .modes = host_access ? IREE_HAL_MAPPING_MODE_SCOPED |
                                   IREE_HAL_MAPPING_MODE_PERSISTENT
                             : 0,
        .cacheability = executable->memory.host_cacheability,
    };
    if (host_access) {
      contract->scopes[1].interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] = 0;
    }
    iree_hal_memory_scope_access_t* queue =
        &contract->scopes[family->memory.queue_scope_id];
    queue->usage = usage;
    queue->interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    queue->bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 1;
    contract->scopes[family->memory.queue_scope_id + 1] = *queue;
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_amd_gpu_executable_query_transition, &query);
  }
  executable->backing.maintenance_alignment = query.maintenance_alignment;
  return status;
}

static iree_status_t iree_hal_amd_gpu_executable_load_image(
    iree_hal_amd_gpu_executable_t* executable,
    const iree_hal_amd_gpu_executable_options_t* options) {
  const iree_hal_amd_gpu_image_t* image = &executable->image;
  // Three speculative 64-byte fetch lines beyond the complete linked image,
  // plus each descriptor's entry-relative 128-byte instruction prefetch range.
  uint64_t byte_length = 0;
  if (!iree_checked_add_u64(image->byte_length, 63 + 192, &byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU instruction-fetch extent overflows");
  }
  byte_length &= ~UINT64_C(63);
  const uint32_t prefetch_mask =
      options->target->version.major == 11 ? 0x3f : 0xff;
  for (iree_host_size_t i = 0; i < image->metadata.kernel_count; ++i) {
    uint64_t end = 0;
    if (!iree_checked_add_u64(
            image->kernels[i].entry_offset,
            ((image->kernels[i].compute_pgm_rsrc3 >> 4) & prefetch_mask) * 128u,
            &end)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "GPU instruction-prefetch extent overflows");
    }
    byte_length = iree_max(byte_length, end);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      executable->context->api, options->memory, byte_length, image->alignment,
      executable->context->event_sink, &executable->memory));
  executable->bindings[0].host_pointer = executable->memory.contents.data;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      executable->context->api->memory_query_address(
          executable->memory.handle, 0, AMDF_MEMORY_ADDRESS_GPU,
          &executable->bindings[1].device_address),
      "memory_query_address(code)"));
  amdf_memory_access_info_t access_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(access_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      executable->context->api->memory_query_access_info(
          executable->memory.handle, 0, &access_info),
      "memory_query_access_info(code)"));
  executable->memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
      (iree_any_bit_set(access_info.flags, AMDF_MEMORY_FLAG_HOST_COHERENT)
           ? IREE_HAL_MEMORY_TYPE_HOST_COHERENT
           : 0) |
      (iree_any_bit_set(executable->memory.info.flags,
                        AMDF_MEMORY_FLAG_DEVICE_LOCAL)
           ? IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
           : 0);
  executable->backing = (iree_hal_buffer_backing_facts_t){
      .maintenance_range = executable->memory.maintenance_range,
      .host_cacheability = executable->memory.host_cacheability,
      .physical_backing_id =
          {
              {
                  executable->memory.info.physical_backing_id.words[0],
                  executable->memory.info.physical_backing_id.words[1],
              },
          },
      .native_allocation =
          {
              .byte_length =
                  executable->memory.info.native_allocation_byte_length,
              .byte_offset = executable->memory.info.source_byte_offset,
              .granularity =
                  executable->memory.info.native_allocation_granularity,
          },
      .allocation_alignment = executable->memory.info.alignment,
  };
  iree_host_size_t storage_length = 0, mappings_offset = 0, imports_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_length,
      IREE_STRUCT_FIELD_ALIGNED(image->segment_count, iree_byte_span_t,
                                iree_alignof(iree_byte_span_t),
                                &mappings_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          image->import_count, iree_hal_amd_gpu_image_import_binding_t,
          iree_alignof(iree_hal_amd_gpu_image_import_binding_t),
          &imports_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
                                             storage_length, (void**)&storage));
  iree_byte_span_t* mappings = (iree_byte_span_t*)(storage + mappings_offset);
  for (iree_host_size_t i = 0; i < image->segment_count; ++i) {
    mappings[i] = iree_make_byte_span(
        executable->memory.contents.data + image->segments[i].offset,
        image->segments[i].length);
  }
  // Padding and linked gaps are owned instruction-fetch storage too. Segment
  // contents and zero-fill tails are initialized by image_load, once each.
  uint64_t cursor = 0;
  for (iree_host_size_t i = 0; i < image->segment_count; ++i) {
    memset(executable->memory.contents.data + cursor, 0,
           image->segments[i].offset - cursor);
    cursor = image->segments[i].offset + image->segments[i].length;
  }
  memset(executable->memory.contents.data + cursor, 0,
         executable->memory.contents.data_length - cursor);
  iree_status_t status = iree_hal_amd_gpu_image_load(
      image, executable->bindings[1].device_address, image->import_count,
      (const iree_hal_amd_gpu_image_import_binding_t*)(storage +
                                                       imports_offset),
      image->segment_count, mappings);
  iree_allocator_free(executable->host_allocator, storage);
  for (iree_host_size_t i = 0;
       i < image->metadata.kernel_count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_gpu_pm4_function_t* function = &executable->functions[i];
    function->metadata = &image->metadata.kernels[i];
    status = iree_hal_amd_gpu_kernarg_layout_allocate(
        function->metadata, executable->host_allocator, &function->kernarg);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_gpu_pm4_launch_initialize(
          options->target->version, &image->kernels[i],
          executable->bindings[1].device_address, &function->launch);
    }
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        executable->context->api->host_mapping_cache_control(
            executable->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
            executable->memory.contents.data_length),
        "host_mapping_cache_control(code upload)");
  }
  return status;
}

typedef struct iree_hal_amd_gpu_executable_prepare_t {
  // Unpublished owner borrowed until the synchronous service call joins.
  iree_hal_amd_gpu_executable_t* executable;
  // Caller-qualified construction facts, borrowed only for this call.
  const iree_hal_amd_gpu_executable_options_t* options;
  // Copied export policy inside executable-owned storage.
  const iree_hal_external_storage_class_t* classes;
  // Terminal construction result transferred back to the loading caller.
  iree_status_t status;
} iree_hal_amd_gpu_executable_prepare_t;

static void iree_hal_amd_gpu_executable_prepare(void* user_data) {
  iree_hal_amd_gpu_executable_prepare_t* prepare = user_data;
  prepare->status = iree_hal_amd_gpu_executable_load_image(prepare->executable,
                                                           prepare->options);
  if (iree_status_is_ok(prepare->status)) {
    prepare->executable->memory.exports.classes = prepare->classes;
    prepare->status = iree_hal_amd_gpu_executable_initialize_contracts(
        prepare->executable, prepare->options);
  }
}

iree_status_t iree_hal_amd_gpu_executable_create(
    const iree_hal_queue_family_t* family,
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_executable_options_t* options,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  IREE_ASSERT(family->memory.domain);
  IREE_ASSERT(context->families[options->native_family_ordinal].command_type ==
              AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  if (params->constant_count) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "HSACO does not define load-time constants");
  }
  iree_hal_amd_gpu_image_t image = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_parse(
      params->executable_data, host_allocator, &image));
  iree_status_t status =
      iree_hal_amd_gpu_executable_admit(&image, options->target);
  iree_host_size_t byte_length = 0, functions_offset = 0, globals_offset = 0;
  iree_host_size_t exports_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_amd_gpu_executable_t), &byte_length,
        IREE_STRUCT_FIELD_ALIGNED(
            image.metadata.kernel_count, iree_hal_amd_gpu_pm4_function_t,
            iree_alignof(iree_hal_amd_gpu_pm4_function_t), &functions_offset),
        IREE_STRUCT_FIELD_ALIGNED(image.symbol_count, iree_hal_buffer_t*,
                                  iree_alignof(iree_hal_buffer_t*),
                                  &globals_offset),
        IREE_STRUCT_FIELD_ALIGNED(
            options->memory->exports.class_count,
            iree_hal_external_storage_class_t,
            iree_alignof(iree_hal_external_storage_class_t), &exports_offset));
  }
  iree_hal_amd_gpu_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, byte_length, (void**)&executable);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_executable_initialize(family, &iree_hal_amd_gpu_executable_vtable,
                                   &executable->base);
    executable->host_allocator = host_allocator;
    executable->context = context;
    executable->cleanup.maintenance = options->maintenance;
    executable->cleanup.entry.fn = iree_hal_amd_gpu_executable_cleanup;
    executable->image = image;
    memset(&image, 0, sizeof(image));
    executable->functions =
        (iree_hal_amd_gpu_pm4_function_t*)((uint8_t*)executable +
                                           functions_offset);
    executable->globals =
        (iree_hal_buffer_t**)((uint8_t*)executable + globals_offset);
    iree_slim_mutex_initialize(&executable->global_mutex);
    iree_hal_external_storage_class_t* classes =
        (iree_hal_external_storage_class_t*)((uint8_t*)executable +
                                             exports_offset);
    if (options->memory->exports.class_count) {
      memcpy(classes, options->memory->exports.classes,
             options->memory->exports.class_count * sizeof(*classes));
    }
    iree_hal_amd_gpu_executable_prepare_t prepare = {
        .executable = executable,
        .options = options,
        .classes = classes,
    };
    iree_hal_memory_maintenance_call(
        options->maintenance, iree_hal_amd_gpu_executable_prepare, &prepare);
    status = prepare.status;
  }
  iree_hal_amd_gpu_image_deinitialize(&image);
  if (iree_status_is_ok(status)) {
    *out_executable = &executable->base;
  } else if (executable) {
    iree_hal_memory_maintenance_call(
        options->maintenance, iree_hal_amd_gpu_executable_free, executable);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Immutable function reflection
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amd_gpu_executable_lookup_pm4(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t token,
    const iree_hal_amd_gpu_pm4_function_t** out_function) {
  iree_hal_amd_gpu_executable_t* executable =
      (iree_hal_amd_gpu_executable_t*)base_executable;
  *out_function = NULL;
  if (!iree_hal_executable_function_is_index_in_range(
          token, executable->image.metadata.kernel_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid GPU executable function token");
  }
  *out_function = &executable->functions[token.value];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amd_gpu_executable_function_count(
    iree_hal_executable_t* executable) {
  return ((iree_hal_amd_gpu_executable_t*)executable)
      ->image.metadata.kernel_count;
}

static iree_status_t iree_hal_amd_gpu_executable_function_info(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_hal_executable_function_info_t* out_info) {
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_pm4(executable, token, &function));
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->metadata;
  *out_info = (iree_hal_executable_function_info_t){
      .name = metadata->reflection_name,
      .flags =
          (metadata->has_required_workgroup_size
               ? 0
               : IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC) |
          (metadata->uniform_workgroup_size
               ? IREE_HAL_EXECUTABLE_FUNCTION_FLAG_REQUIRES_UNIFORM_WORKGROUPS
               : 0),
      .constant_byte_length = function->kernarg->constant_byte_length,
      .binding_count = function->kernarg->binding_count,
      .parameter_count = function->kernarg->parameter_count,
      .maximum_workgroup_invocations = metadata->max_flat_workgroup_size,
      .workgroup_size = {1, 1, 1},
      .resource_usage =
          {
              .provided_flags = IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL,
              .fixed_workgroup_local_memory_size =
                  metadata->group_segment_fixed_size,
              .fixed_private_memory_size = metadata->private_segment_fixed_size,
              .invocation_register_count = metadata->vgpr_count,
          },
  };
  if (metadata->has_required_workgroup_size) {
    memcpy(out_info->workgroup_size, metadata->required_workgroup_size,
           sizeof(out_info->workgroup_size));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_function_parameters(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  const iree_hal_amd_gpu_pm4_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_pm4(executable, token, &function));
  if (capacity < function->kernarg->parameter_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  if (function->kernarg->parameter_count) {
    memcpy(out_parameters,
           iree_hal_amd_gpu_kernarg_layout_parameters(function->kernarg),
           function->kernarg->parameter_count * sizeof(*out_parameters));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  const iree_hal_amd_gpu_image_t* image =
      &((iree_hal_amd_gpu_executable_t*)base_executable)->image;
  for (iree_host_size_t i = 0; i < image->metadata.kernel_count; ++i) {
    if (iree_string_view_equal(name,
                               image->metadata.kernels[i].reflection_name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "GPU executable has no function `%.*s`",
                          (int)name.size, name.data);
}

//===----------------------------------------------------------------------===//
// Executable-owned global views
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amd_gpu_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  const iree_hal_amd_gpu_image_t* image =
      &((iree_hal_amd_gpu_executable_t*)base_executable)->image;
  const iree_hal_amd_gpu_image_symbol_t* symbol =
      iree_hal_amd_gpu_image_find_symbol(image, name);
  *out_found = symbol && iree_hal_amd_gpu_executable_global_access(symbol);
  *out_global =
      *out_found
          ? iree_hal_executable_global_from_value(symbol - image->symbols)
          : iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_lookup_global(
    iree_hal_amd_gpu_executable_t* executable,
    iree_hal_executable_global_t token,
    const iree_hal_amd_gpu_image_symbol_t** out_symbol) {
  if (token.value >= executable->image.symbol_count ||
      !iree_hal_amd_gpu_executable_global_access(
          &executable->image.symbols[token.value])) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid GPU executable global token");
  }
  *out_symbol = &executable->image.symbols[token.value];
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  const iree_hal_amd_gpu_image_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_executable_lookup_global(
      (iree_hal_amd_gpu_executable_t*)base_executable, global, &symbol));
  *out_info = (iree_hal_executable_global_info_t){
      .name = symbol->name,
      .byte_length = symbol->length,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_amd_gpu_executable_t* executable =
      (iree_hal_amd_gpu_executable_t*)base_executable;
  const iree_hal_amd_gpu_image_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_executable_lookup_global(executable, global, &symbol));
  const iree_hal_queue_family_t* family = executable->base.queue_family;
  const iree_hal_memory_access_t access =
      iree_hal_amd_gpu_executable_global_access(symbol);
  iree_hal_memory_contract_t* contract = executable->contracts[access];
  iree_slim_mutex_lock(&executable->global_mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_buffer_t* buffer = executable->globals[global.value];
  if (!buffer) {
    status = iree_hal_amd_fabric_buffer_wrap(
        (iree_hal_buffer_placement_t){
            .device = iree_hal_queue_family_device(family),
            .queue_family_affinity = iree_hal_make_queue_family_affinity(
                iree_hal_queue_family_ordinal(family)),
        },
        executable->context->api,
        (iree_hal_amd_fabric_buffer_storage_t){
            .memory = &executable->memory,
            .bindings = executable->bindings,
            .host_binding_index = 0,
            .offset = symbol->address.offset,
        },
        symbol->length, executable->memory_type, access,
        contract->allowed_usage, iree_hal_buffer_release_callback_null(),
        executable->host_allocator, &buffer);
    if (iree_status_is_ok(status)) {
      buffer->memory.contract = contract;
      buffer->memory.backing = &executable->backing;
      buffer->memory.offset = symbol->address.offset;
      executable->globals[global.value] = buffer;
    }
  }
  iree_slim_mutex_unlock(&executable->global_mutex);
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  }
  return status;
}

static const iree_hal_executable_vtable_t iree_hal_amd_gpu_executable_vtable = {
    .destroy = iree_hal_amd_gpu_executable_destroy,
    .function_count = iree_hal_amd_gpu_executable_function_count,
    .function_info = iree_hal_amd_gpu_executable_function_info,
    .function_parameters = iree_hal_amd_gpu_executable_function_parameters,
    .lookup_function_by_name =
        iree_hal_amd_gpu_executable_lookup_function_by_name,
    .try_lookup_global_by_name =
        iree_hal_amd_gpu_executable_try_lookup_global_by_name,
    .global_info = iree_hal_amd_gpu_executable_global_info,
    .global_buffer = iree_hal_amd_gpu_executable_global_buffer,
};
