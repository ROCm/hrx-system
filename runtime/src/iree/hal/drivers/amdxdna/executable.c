// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/drivers/amdxdna/allocator.h"
#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/ctrlcode_elf.h"
#include "iree/hal/drivers/amdxdna/device.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/drivers/amdxdna/xclbin_util.h"
#include "iree/schemas/amdxdna_elf_executable_def_reader.h"
#include "iree/schemas/amdxdna_elf_executable_def_verifier.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_reader.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_verifier.h"
#include "iree/schemas/pdi_executable_def_reader.h"
#include "iree/schemas/pdi_executable_def_verifier.h"

static const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable;
static iree_atomic_int64_t iree_hal_amdxdna_next_executable_cache_identity =
    IREE_ATOMIC_VAR_INIT(1);

static const iree_string_view_t kAmdxdnaPdiExecutableFormat = {"amdxdna-pdi-fb",
                                                               14};
static const iree_string_view_t kAmdxdnaXclbinExecutableFormat = {
    "amdxdna-xclbin-fb", 17};
static const iree_string_view_t kAmdxdnaXclbinExecutableCompatFormat = {
    "amdaie-amdxdna-xclbin-fb", 24};
static const iree_string_view_t kAmdxdnaElfExecutableFormat = {
    "amdxdna-elf-fb", 14};

iree_string_view_t iree_hal_amdxdna_executable_format(void) {
  return kAmdxdnaPdiExecutableFormat;
}

bool iree_hal_amdxdna_executable_format_supported(
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                kAmdxdnaPdiExecutableFormat) ||
         iree_string_view_equal(executable_format,
                                kAmdxdnaXclbinExecutableFormat) ||
         iree_string_view_equal(executable_format,
                                kAmdxdnaXclbinExecutableCompatFormat) ||
         iree_string_view_equal(executable_format, kAmdxdnaElfExecutableFormat);
}

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_amdxdna_executable_vtable,
      iree_hal_amdxdna_executable);
}

void iree_hal_amdxdna_executable_bind_device(
    iree_hal_executable_t* base_executable, iree_hal_amdxdna_device* device) {
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  if (!executable) return;
  executable->hal_device = device;
}

iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_executable_control_context_borrow(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  iree_slim_mutex_lock(&executable->context_mutex);
  iree_hal_amdxdna_native_context_t* context =
      iree_hal_amdxdna_native_context_ref_borrow(executable->context);
  iree_slim_mutex_unlock(&executable->context_mutex);
  return context;
}

iree_status_t iree_hal_amdxdna_executable_preload_contexts(
    iree_hal_amdxdna_device* device, iree_hal_executable_t* base_executable) {
  if (!device || !base_executable) return iree_ok_status();
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_amdxdna_kernel_params_t* params = &executable->entry_points[i];
    // Only preload self-contained entry points. Reconfiguration entries have
    // ordering semantics: a loader publishes executable->context for sibling
    // empty-image entries, so they remain lazy and execute-ordered.
    if (params->reconf_data_runlist_count != 0) continue;
    if (params->pdi.count == 0 && params->xclbin.count == 0) continue;

    iree_hal_amdxdna_context_cache_lease_t* context_lease = NULL;
    iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;
    iree_status_t status = iree_hal_amdxdna_device_pin_context(
        device, iree_make_const_byte_span(params->pdi.data, params->pdi.count),
        iree_make_const_byte_span(params->xclbin.data, params->xclbin.count),
        params->kernel_name, params->partition_cols, &context_ref,
        &context_lease);
    iree_hal_amdxdna_native_c_cu_index_t cu_idx;
    memset(&cu_idx, 0, sizeof(cu_idx));
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_context_ref_open_cu(
          context_ref, params->kernel_name, &cu_idx);
    }
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&executable->context_mutex);
      if (!params->cached_context_valid) {
        params->cached_context_lease = context_lease;
        context_lease = NULL;
        params->cached_cu_index = cu_idx;
        params->cached_context_valid = true;
      }
      iree_slim_mutex_unlock(&executable->context_mutex);
    }
    iree_hal_amdxdna_native_context_ref_release(context_ref);
    iree_hal_amdxdna_context_cache_lease_release(context_lease);
    if (!iree_status_is_ok(status)) return status;
  }
  return iree_ok_status();
}

static void iree_hal_amdxdna_u8_list_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_u8_list_t* list) {
  iree_allocator_free(host_allocator, list->data);
  list->data = NULL;
  list->count = 0;
}

static void iree_hal_amdxdna_u32_list_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_u32_list_t* list) {
  iree_allocator_free(host_allocator, list->data);
  list->data = NULL;
  list->count = 0;
}

static void iree_hal_amdxdna_pdi_relocation_list_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_pdi_relocation_list_t* list) {
  iree_allocator_free(host_allocator, list->data);
  list->data = NULL;
  list->count = 0;
}

static void iree_hal_amdxdna_control_parameter_relocation_list_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_control_parameter_relocation_list_t* list) {
  iree_allocator_free(host_allocator, list->data);
  list->data = NULL;
  list->count = 0;
}

static iree_status_t iree_hal_amdxdna_copy_u8_span(
    iree_allocator_t host_allocator, iree_const_byte_span_t source,
    iree_hal_amdxdna_u8_list_t* out_list) {
  out_list->data = NULL;
  out_list->count = 0;
  if (source.data_length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, source.data_length,
                                             (void**)&out_list->data));
  memcpy(out_list->data, source.data, source.data_length);
  out_list->count = source.data_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_u32_vec(
    iree_allocator_t host_allocator, flatbuffers_uint32_vec_t source,
    iree_hal_amdxdna_u32_list_t* out_list) {
  out_list->data = NULL;
  out_list->count = 0;
  if (!source) return iree_ok_status();
  const iree_host_size_t count = flatbuffers_uint32_vec_len(source);
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, count * sizeof(uint32_t), (void**)&out_list->data));
  memcpy(out_list->data, source, count * sizeof(uint32_t));
  out_list->count = count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_string_view(
    iree_allocator_t host_allocator, iree_string_view_t source,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (source.size == 0) return iree_ok_status();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, source.size, (void**)&storage));
  memcpy(storage, source.data, source.size);
  *out_string = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void iree_hal_amdxdna_kernel_params_deinitialize(
    iree_allocator_t host_allocator, iree_hal_amdxdna_kernel_params_t* params) {
  iree_hal_amdxdna_u8_list_deinitialize(host_allocator, &params->pdi);
  iree_hal_amdxdna_u8_list_deinitialize(host_allocator, &params->xclbin);
  for (iree_host_size_t i = 0; i < params->asm_inst_runlist_count; ++i) {
    iree_hal_amdxdna_u32_list_deinitialize(host_allocator,
                                           &params->asm_inst_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->asm_inst_runlist);
  params->asm_inst_runlist = NULL;
  params->asm_inst_runlist_count = 0;
  for (iree_host_size_t i = 0; i < params->reconf_data_runlist_count; ++i) {
    iree_hal_amdxdna_u32_list_deinitialize(host_allocator,
                                           &params->reconf_data_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->reconf_data_runlist);
  params->reconf_data_runlist = NULL;
  params->reconf_data_runlist_count = 0;
  for (iree_host_size_t i = 0; i < params->patch_runlist_count; ++i) {
    iree_hal_amdxdna_u32_list_deinitialize(host_allocator,
                                           &params->patch_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->patch_runlist);
  params->patch_runlist = NULL;
  params->patch_runlist_count = 0;
  for (iree_host_size_t i = 0; i < params->pdi_relocation_runlist_count; ++i) {
    iree_hal_amdxdna_pdi_relocation_list_deinitialize(
        host_allocator, &params->pdi_relocation_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->pdi_relocation_runlist);
  params->pdi_relocation_runlist = NULL;
  params->pdi_relocation_runlist_count = 0;
  for (iree_host_size_t i = 0;
       i < params->control_parameter_relocation_runlist_count; ++i) {
    iree_hal_amdxdna_control_parameter_relocation_list_deinitialize(
        host_allocator, &params->control_parameter_relocation_runlist[i]);
  }
  iree_allocator_free(host_allocator,
                      params->control_parameter_relocation_runlist);
  params->control_parameter_relocation_runlist = NULL;
  params->control_parameter_relocation_runlist_count = 0;
  for (iree_host_size_t i = 0; i < params->constant_patch_runlist_count; ++i) {
    iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
        host_allocator, &params->constant_patch_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->constant_patch_runlist);
  params->constant_patch_runlist = NULL;
  params->constant_patch_runlist_count = 0;
  iree_allocator_free(host_allocator, params->dpu_slices);
  params->dpu_slices = NULL;
  params->dpu_slice_count = 0;
  for (iree_host_size_t i = 0; i < params->parameter_count; ++i) {
    iree_allocator_free(host_allocator, (void*)params->parameters[i].name.data);
  }
  iree_allocator_free(host_allocator, params->parameters);
  params->parameters = NULL;
  params->parameter_count = 0;
  params->constant_byte_length = 0;
  iree_allocator_free(host_allocator, (void*)params->kernel_name.data);
  params->kernel_name = iree_string_view_empty();
  iree_hal_amdxdna_context_cache_lease_release(params->cached_context_lease);
  params->cached_context_lease = NULL;
  params->cached_context_valid = false;
  IREE_TRACE({
    iree_allocator_free(host_allocator, (void*)params->source_filename.data);
    params->source_filename = iree_string_view_empty();
  });
}

static void iree_hal_amdxdna_executable_deinitialize(
    iree_hal_amdxdna_executable* executable) {
  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_amdxdna_kernel_params_deinitialize(executable->host_allocator,
                                                &executable->entry_points[i]);
  }
  iree_allocator_free(executable->host_allocator, executable->entry_points);
  executable->entry_points = NULL;
  executable->entry_point_count = 0;
  for (iree_host_size_t i = 0; i < executable->pdi_count; ++i) {
    if (executable->pdi_buffers) {
      iree_hal_amdxdna_native_buffer_c_destroy(executable->pdi_buffers[i]);
    }
    iree_hal_amdxdna_u8_list_deinitialize(executable->host_allocator,
                                          &executable->pdis[i]);
  }
  iree_allocator_free(executable->host_allocator, executable->pdi_buffers);
  executable->pdi_buffers = NULL;
  iree_allocator_free(executable->host_allocator, executable->pdis);
  executable->pdis = NULL;
  executable->pdi_count = 0;
  iree_hal_amdxdna_native_context_ref_release(executable->context);
  executable->context = NULL;
  iree_slim_mutex_deinitialize(&executable->context_mutex);
}

iree_status_t iree_hal_amdxdna_executable_ensure_pdi_buffer(
    iree_hal_amdxdna_executable* executable, uint32_t pdi_ordinal) {
  if (pdi_ordinal >= executable->pdi_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna standalone PDI ordinal is out of range");
  }
  iree_slim_mutex_lock(&executable->context_mutex);
  if (executable->pdi_buffers && executable->pdi_buffers[pdi_ordinal]) {
    iree_slim_mutex_unlock(&executable->context_mutex);
    return iree_ok_status();
  }
  if (!executable->native_device) {
    iree_slim_mutex_unlock(&executable->context_mutex);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna standalone PDIs require a native device");
  }
  iree_status_t status = iree_ok_status();
  if (!executable->pdi_buffers) {
    status = iree_allocator_malloc(
        executable->host_allocator,
        executable->pdi_count * sizeof(*executable->pdi_buffers),
        (void**)&executable->pdi_buffers);
    if (iree_status_is_ok(status)) {
      memset(executable->pdi_buffers, 0,
             executable->pdi_count * sizeof(*executable->pdi_buffers));
    }
  }
  if (iree_status_is_ok(status) && !executable->pdi_buffers[pdi_ordinal]) {
    iree_hal_amdxdna_native_buffer_t* pdi_buffer = NULL;
    status = iree_hal_amdxdna_native_device_c_alloc_buffer(
        executable->native_device, executable->pdis[pdi_ordinal].count,
        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_PDI, &pdi_buffer);
    void* mapped_ptr = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdxdna_native_buffer_c_map(pdi_buffer, &mapped_ptr);
    }
    if (iree_status_is_ok(status)) {
      memcpy(mapped_ptr, executable->pdis[pdi_ordinal].data,
             executable->pdis[pdi_ordinal].count);
      status = iree_hal_amdxdna_native_buffer_c_sync_all(
          pdi_buffer, IREE_HAL_AMDXDNA_NATIVE_BUFFER_SYNC_HOST_TO_DEVICE);
    }
    if (iree_status_is_ok(status)) {
      executable->pdi_buffers[pdi_ordinal] = pdi_buffer;
    } else {
      iree_hal_amdxdna_native_buffer_c_destroy(pdi_buffer);
    }
  }
  iree_slim_mutex_unlock(&executable->context_mutex);
  return status;
}

static iree_status_t iree_hal_amdxdna_verify_run_list(
    const char* format_name, iree_host_size_t entry_index,
    iree_host_size_t run_count, iree_host_size_t payload_run_count,
    const iree_host_size_t* control_code_counts,
    const iree_host_size_t* patch_table_counts) {
  for (iree_host_size_t run_i = 0; run_i < run_count; ++run_i) {
    if (control_code_counts[run_i] == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s entry point %" PRIhsz " run %" PRIhsz
                              " has no control code",
                              format_name, entry_index, run_i);
    }
    if (patch_table_counts[run_i] % 3 != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%s entry point %" PRIhsz " run %" PRIhsz
          " patch_table length %" PRIhsz " is not a multiple of 3",
          format_name, entry_index, run_i, patch_table_counts[run_i]);
    }
  }
  if (payload_run_count != 0 && run_count != 2 * payload_run_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s entry point %" PRIhsz " has %" PRIhsz " runs but %" PRIhsz
        " reconfiguration payloads; expected paired reconfiguration/execution "
        "runs",
        format_name, entry_index, run_count, payload_run_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pdi_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_hal_amdxdna_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_amdxdna_PdiDef_vec_t pdis =
      iree_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  size_t pdi_count = iree_hal_amdxdna_PdiDef_vec_len(pdis);
  if (pdi_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no PDI present");
  }
  for (size_t i = 0; i < pdi_count; ++i) {
    iree_hal_amdxdna_PdiDef_table_t pdi =
        iree_hal_amdxdna_PdiDef_vec_at(pdis, i);
    flatbuffers_uint8_vec_t pdi_bytes = iree_hal_amdxdna_PdiDef_pdi_get(pdi);
    if (flatbuffers_uint8_vec_len(pdi_bytes) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable PDI %zu is empty", i);
    }
  }

  iree_hal_amdxdna_EntryPointDef_vec_t entry_points =
      iree_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count =
      iree_hal_amdxdna_EntryPointDef_vec_len(entry_points);
  if (entry_point_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }

  bool has_pdi_entry_point = false;
  for (size_t i = 0; i < entry_point_count; ++i) {
    iree_hal_amdxdna_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_EntryPointDef_vec_at(entry_points, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_EntryPointDef_name_get(entry_point);
    if (!name || flatbuffers_string_len(name) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
    int32_t pdi_index =
        iree_hal_amdxdna_EntryPointDef_pdi_index_get(entry_point);
    if (pdi_index >= 0 && (size_t)pdi_index >= pdi_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu pdi index %d out of range; "
                              "executable only contains %zu PDIs",
                              i, pdi_index, pdi_count);
    }
    has_pdi_entry_point |= pdi_index >= 0;

    iree_hal_amdxdna_RunDef_vec_t runs =
        iree_hal_amdxdna_EntryPointDef_runs_get(entry_point);
    size_t run_count = iree_hal_amdxdna_RunDef_vec_len(runs);
    if (run_count == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu has no runs", i);
    }
    iree_host_size_t control_code_counts[64];
    iree_host_size_t patch_table_counts[64];
    if (run_count > IREE_ARRAYSIZE(control_code_counts)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many amdxdna runs: %zu", run_count);
    }
    iree_host_size_t payload_run_count = 0;
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run =
          iree_hal_amdxdna_RunDef_vec_at(runs, run_i);
      control_code_counts[run_i] = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_RunDef_control_code_get(run));
      iree_host_size_t payload_count = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_RunDef_data_payload_get(run));
      patch_table_counts[run_i] = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_RunDef_patch_table_get(run));
      if (payload_count != 0) {
        if ((run_i & 1) != 0) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "%s entry point %zu reconfiguration payload run %zu is not in "
              "an even reconfiguration slot",
              "PDIX", i, run_i);
        }
        ++payload_run_count;
      }
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_verify_run_list(
                "PDIX", i, run_count, payload_run_count, control_code_counts,
                patch_table_counts));
  }
  if (!has_pdi_entry_point) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry point references a PDI");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static bool iree_hal_amdxdna_relocation_ranges_overlap(uint32_t lhs_offset,
                                                       uint32_t rhs_offset) {
  return lhs_offset < rhs_offset + sizeof(uint64_t) &&
         rhs_offset < lhs_offset + sizeof(uint64_t);
}

static iree_status_t iree_hal_amdxdna_xclbin_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_hal_amdxdna_xclbin_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdxdna_xclbin_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_xclbin_ExecutableDef_as_root(flatbuffer_data.data);

  iree_hal_amdxdna_xclbin_XclbinDef_vec_t xclbins =
      iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_get(executable_def);
  size_t xclbin_count = iree_hal_amdxdna_xclbin_XclbinDef_vec_len(xclbins);
  if (xclbin_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no xclbin present");
  }
  for (size_t i = 0; i < xclbin_count; ++i) {
    iree_hal_amdxdna_xclbin_XclbinDef_table_t xclbin =
        iree_hal_amdxdna_xclbin_XclbinDef_vec_at(xclbins, i);
    flatbuffers_string_t bytes =
        iree_hal_amdxdna_xclbin_XclbinDef_xclbin_get(xclbin);
    if (!bytes || flatbuffers_string_len(bytes) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable xclbin %zu is empty", i);
    }
  }

  iree_hal_amdxdna_xclbin_PdiDef_vec_t pdis =
      iree_hal_amdxdna_xclbin_ExecutableDef_pdis_get(executable_def);
  size_t pdi_count = iree_hal_amdxdna_xclbin_PdiDef_vec_len(pdis);
  for (size_t i = 0; i < pdi_count; ++i) {
    iree_hal_amdxdna_xclbin_PdiDef_table_t pdi =
        iree_hal_amdxdna_xclbin_PdiDef_vec_at(pdis, i);
    if (flatbuffers_uint8_vec_len(
            iree_hal_amdxdna_xclbin_PdiDef_pdi_get(pdi)) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "XADX standalone PDI %zu is empty", i);
    }
  }

  iree_hal_amdxdna_xclbin_EntryPointDef_vec_t entry_points =
      iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count =
      iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entry_points);
  if (entry_point_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }

  bool has_context_entry_point = false;
  for (size_t i = 0; i < entry_point_count; ++i) {
    iree_hal_amdxdna_xclbin_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entry_points, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_xclbin_EntryPointDef_name_get(entry_point);
    if (!name || flatbuffers_string_len(name) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }

    int32_t pdi_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(entry_point);
    int32_t xclbin_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(entry_point);
    if ((pdi_index < 0) != (xclbin_index < 0)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry point %zu must either reference an xclbin context and PDI "
          "index or neither",
          i);
    }
    if (xclbin_index >= 0 && (size_t)xclbin_index >= xclbin_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry point %zu xclbin index %d out of range; executable only "
          "contains %zu xclbins",
          i, xclbin_index, xclbin_count);
    }
    has_context_entry_point |= xclbin_index >= 0;

    iree_hal_amdxdna_xclbin_ControlParameterDef_vec_t control_parameters =
        iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_get(
            entry_point);
    const iree_host_size_t control_parameter_count =
        iree_hal_amdxdna_xclbin_ControlParameterDef_vec_len(control_parameters);
    if (control_parameter_count > 32) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "XADX entry point %zu has more than 32 control parameters", i);
    }
    uint32_t used_state_slots = 0;
    for (iree_host_size_t parameter_i = 0;
         parameter_i < control_parameter_count; ++parameter_i) {
      iree_hal_amdxdna_xclbin_ControlParameterDef_table_t parameter =
          iree_hal_amdxdna_xclbin_ControlParameterDef_vec_at(control_parameters,
                                                             parameter_i);
      flatbuffers_string_t parameter_name =
          iree_hal_amdxdna_xclbin_ControlParameterDef_name_get(parameter);
      flatbuffers_string_t scalar_type =
          iree_hal_amdxdna_xclbin_ControlParameterDef_scalar_type_get(
              parameter);
      const uint32_t constant_offset =
          iree_hal_amdxdna_xclbin_ControlParameterDef_constant_offset_get(
              parameter);
      const uint32_t state_table_index =
          iree_hal_amdxdna_xclbin_ControlParameterDef_state_table_index_get(
              parameter);
      const uint32_t byte_length =
          iree_hal_amdxdna_xclbin_ControlParameterDef_byte_length_get(
              parameter);
      const uint32_t kind =
          iree_hal_amdxdna_xclbin_ControlParameterDef_kind_get(parameter);
      if (!parameter_name || flatbuffers_string_len(parameter_name) == 0 ||
          !scalar_type || flatbuffers_string_len(scalar_type) == 0 ||
          constant_offset > UINT16_MAX || byte_length == 0 ||
          byte_length > sizeof(uint32_t) || state_table_index >= 32 ||
          kind > 1 || (used_state_slots & (1u << state_table_index)) != 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XADX entry point %zu control parameter %zu is invalid", i,
            parameter_i);
      }
      used_state_slots |= 1u << state_table_index;
      const uint32_t parameter_end = constant_offset + byte_length;
      for (iree_host_size_t other_i = 0; other_i < parameter_i; ++other_i) {
        iree_hal_amdxdna_xclbin_ControlParameterDef_table_t other =
            iree_hal_amdxdna_xclbin_ControlParameterDef_vec_at(
                control_parameters, other_i);
        const uint32_t other_offset =
            iree_hal_amdxdna_xclbin_ControlParameterDef_constant_offset_get(
                other);
        const uint32_t other_end =
            other_offset +
            iree_hal_amdxdna_xclbin_ControlParameterDef_byte_length_get(other);
        if (constant_offset < other_end && other_offset < parameter_end) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "XADX entry point %zu control parameter constant ranges overlap",
              i);
        }
        flatbuffers_string_t other_name =
            iree_hal_amdxdna_xclbin_ControlParameterDef_name_get(other);
        if (flatbuffers_string_len(parameter_name) ==
                flatbuffers_string_len(other_name) &&
            memcmp(parameter_name, other_name,
                   flatbuffers_string_len(parameter_name)) == 0) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "XADX entry point %zu control parameter name is reused", i);
        }
      }
    }

    iree_hal_amdxdna_xclbin_RunDef_vec_t runs =
        iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(entry_point);
    size_t run_count = iree_hal_amdxdna_xclbin_RunDef_vec_len(runs);
    if (run_count == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu has no runs", i);
    }
    iree_host_size_t control_code_counts[64];
    iree_host_size_t patch_table_counts[64];
    if (run_count > IREE_ARRAYSIZE(control_code_counts)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many amdxdna runs: %zu", run_count);
    }
    iree_host_size_t payload_run_count = 0;
    for (size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_xclbin_RunDef_table_t run =
          iree_hal_amdxdna_xclbin_RunDef_vec_at(runs, run_i);
      control_code_counts[run_i] = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_xclbin_RunDef_control_code_get(run));
      iree_host_size_t payload_count = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_xclbin_RunDef_data_payload_get(run));
      patch_table_counts[run_i] = flatbuffers_uint32_vec_len(
          iree_hal_amdxdna_xclbin_RunDef_patch_table_get(run));
      iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_t pdi_relocations =
          iree_hal_amdxdna_xclbin_RunDef_pdi_relocations_get(run);
      const iree_host_size_t pdi_relocation_count =
          iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_len(pdi_relocations);
      const iree_host_size_t control_code_bytes =
          control_code_counts[run_i] * sizeof(uint32_t);
      for (iree_host_size_t reloc_i = 0; reloc_i < pdi_relocation_count;
           ++reloc_i) {
        iree_hal_amdxdna_xclbin_PdiRelocationDef_table_t relocation =
            iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_at(pdi_relocations,
                                                            reloc_i);
        const uint32_t offset =
            iree_hal_amdxdna_xclbin_PdiRelocationDef_offset_get(relocation);
        const uint32_t pdi_index =
            iree_hal_amdxdna_xclbin_PdiRelocationDef_pdi_index_get(relocation);
        if ((offset % sizeof(uint32_t)) != 0 ||
            control_code_bytes < sizeof(uint64_t) ||
            offset > control_code_bytes - sizeof(uint64_t)) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "XADX entry point %zu run %zu PDI relocation %zu offset %u "
              "does not name a 64-bit address in the control code",
              i, run_i, reloc_i, offset);
        }
        if (pdi_index >= pdi_count) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "XADX entry point %zu run %zu PDI relocation %zu index %u "
              "out of range; executable only contains %zu standalone PDIs",
              i, run_i, reloc_i, pdi_index, pdi_count);
        }
        for (iree_host_size_t other_i = 0; other_i < reloc_i; ++other_i) {
          iree_hal_amdxdna_xclbin_PdiRelocationDef_table_t other =
              iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_at(pdi_relocations,
                                                              other_i);
          if (iree_hal_amdxdna_relocation_ranges_overlap(
                  offset,
                  iree_hal_amdxdna_xclbin_PdiRelocationDef_offset_get(other))) {
            IREE_TRACE_ZONE_END(z0);
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "XADX entry point %zu run %zu PDI relocation ranges overlap", i,
                run_i);
          }
        }
      }
      iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_t
          control_parameter_relocations =
              iree_hal_amdxdna_xclbin_RunDef_control_parameter_relocations_get(
                  run);
      const iree_host_size_t control_parameter_relocation_count =
          iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_len(
              control_parameter_relocations);
      if (control_parameter_relocation_count != 0 &&
          control_parameter_count == 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XADX entry point %zu has control parameter relocations but no "
            "parameter metadata",
            i);
      }
      for (iree_host_size_t reloc_i = 0;
           reloc_i < control_parameter_relocation_count; ++reloc_i) {
        iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_table_t
            relocation =
                iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_at(
                    control_parameter_relocations, reloc_i);
        const uint32_t offset =
            iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_offset_get(
                relocation);
        if ((offset % sizeof(uint32_t)) != 0 ||
            control_code_bytes < sizeof(uint64_t) ||
            offset > control_code_bytes - sizeof(uint64_t)) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "XADX entry point %zu run %zu control parameter relocation %zu "
              "offset %u does not name a 64-bit address in the control code",
              i, run_i, reloc_i, offset);
        }
        for (iree_host_size_t pdi_i = 0; pdi_i < pdi_relocation_count;
             ++pdi_i) {
          iree_hal_amdxdna_xclbin_PdiRelocationDef_table_t pdi_relocation =
              iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_at(pdi_relocations,
                                                              pdi_i);
          if (iree_hal_amdxdna_relocation_ranges_overlap(
                  offset, iree_hal_amdxdna_xclbin_PdiRelocationDef_offset_get(
                              pdi_relocation))) {
            IREE_TRACE_ZONE_END(z0);
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "XADX entry point %zu run %zu PDI and control parameter "
                "relocation ranges overlap",
                i, run_i);
          }
        }
        for (iree_host_size_t other_i = 0; other_i < reloc_i; ++other_i) {
          iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_table_t other =
              iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_at(
                  control_parameter_relocations, other_i);
          if (iree_hal_amdxdna_relocation_ranges_overlap(
                  offset,
                  iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_offset_get(
                      other))) {
            IREE_TRACE_ZONE_END(z0);
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "XADX entry point %zu run %zu control parameter relocation "
                "ranges overlap",
                i, run_i);
          }
        }
      }
      if (payload_count != 0) {
        if ((run_i & 1) != 0) {
          IREE_TRACE_ZONE_END(z0);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "%s entry point %zu reconfiguration payload run %zu is not in "
              "an even reconfiguration slot",
              "XADX", i, run_i);
        }
        ++payload_run_count;
      }
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_verify_run_list(
                "XADX", i, run_count, payload_run_count, control_code_counts,
                patch_table_counts));
  }
  if (!has_context_entry_point) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry point references an xclbin context");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_elf_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_hal_amdxdna_elf_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_hal_amdxdna_elf_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_elf_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_amdxdna_elf_ModuleDef_vec_t modules =
      iree_hal_amdxdna_elf_ExecutableDef_modules_get(executable_def);
  size_t module_count = iree_hal_amdxdna_elf_ModuleDef_vec_len(modules);
  if (module_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no ELF module present");
  }
  for (size_t i = 0; i < module_count; ++i) {
    iree_hal_amdxdna_elf_ModuleDef_table_t module =
        iree_hal_amdxdna_elf_ModuleDef_vec_at(modules, i);
    flatbuffers_uint8_vec_t bytes =
        iree_hal_amdxdna_elf_ModuleDef_elf_get(module);
    if (!bytes || flatbuffers_uint8_vec_len(bytes) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable ELF module %zu is empty", i);
    }
  }

  iree_hal_amdxdna_elf_ExportDef_vec_t exports =
      iree_hal_amdxdna_elf_ExecutableDef_exports_get(executable_def);
  size_t export_count = iree_hal_amdxdna_elf_ExportDef_vec_len(exports);
  if (export_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no exports found in the executable");
  }
  for (size_t i = 0; i < export_count; ++i) {
    iree_hal_amdxdna_elf_ExportDef_table_t export_def =
        iree_hal_amdxdna_elf_ExportDef_vec_at(exports, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_elf_ExportDef_name_get(export_def);
    if (!name || flatbuffers_string_len(name) == 0) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable export %zu has no name", i);
    }
    uint32_t module_ordinal =
        iree_hal_amdxdna_elf_ExportDef_module_ordinal_get(export_def);
    if ((size_t)module_ordinal >= module_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "export %zu module ordinal %u out of range; executable only contains "
          "%zu modules",
          i, module_ordinal, module_count);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  iree_string_view_t format = kAmdxdnaPdiExecutableFormat;
  iree_status_t status =
      iree_hal_amdxdna_pdi_flatbuffer_verify(executable_data);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    status = iree_hal_amdxdna_xclbin_flatbuffer_verify(executable_data);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      status = iree_hal_amdxdna_elf_flatbuffer_verify(executable_data);
      if (!iree_status_is_ok(status)) return status;
      format = kAmdxdnaElfExecutableFormat;
    } else {
      format = kAmdxdnaXclbinExecutableFormat;
    }
  }

  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + 1);
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_executable_allocate(
    iree_allocator_t host_allocator, iree_host_size_t entry_point_count,
    iree_hal_amdxdna_executable** out_executable) {
  iree_hal_amdxdna_executable* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  memset(executable, 0, sizeof(*executable));
  iree_hal_resource_initialize(&iree_hal_amdxdna_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->cache_identity = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_amdxdna_next_executable_cache_identity, 1,
      iree_memory_order_relaxed);
  executable->entry_point_count = entry_point_count;
  iree_slim_mutex_initialize(&executable->context_mutex);
  if (entry_point_count > 0) {
    iree_status_t status = iree_allocator_malloc(
        host_allocator, entry_point_count * sizeof(*executable->entry_points),
        (void**)&executable->entry_points);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_deinitialize(&executable->context_mutex);
      iree_allocator_free(host_allocator, executable);
      return status;
    }
    memset(executable->entry_points, 0,
           entry_point_count * sizeof(*executable->entry_points));
    for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
      executable->entry_points[i].n_reconfigure_runs = 1;
      executable->entry_points[i].n_pdi_loads = 1;
    }
  }
  *out_executable = executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_append_run_params(
    iree_allocator_t host_allocator, iree_hal_amdxdna_kernel_params_t* params,
    iree_host_size_t run_ordinal, flatbuffers_uint32_vec_t control_code,
    flatbuffers_uint32_vec_t data_payload,
    flatbuffers_uint32_vec_t patch_table) {
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_copy_u32_vec(
      host_allocator, control_code, &params->asm_inst_runlist[run_ordinal]));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_build_write32_constant_patch_list(
      host_allocator, params->asm_inst_runlist[run_ordinal].data,
      params->asm_inst_runlist[run_ordinal].count,
      &params->constant_patch_runlist[run_ordinal]));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_copy_u32_vec(
      host_allocator, patch_table, &params->patch_runlist[run_ordinal]));
  if (data_payload && flatbuffers_uint32_vec_len(data_payload) != 0) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_copy_u32_vec(
        host_allocator, data_payload,
        &params->reconf_data_runlist[params->reconf_data_runlist_count++]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_pdi_relocations(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_t source,
    iree_hal_amdxdna_pdi_relocation_list_t* out_list) {
  out_list->data = NULL;
  out_list->count = 0;
  if (!source) return iree_ok_status();
  const iree_host_size_t count =
      iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_len(source);
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator,
                                             count * sizeof(*out_list->data),
                                             (void**)&out_list->data));
  out_list->count = count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_amdxdna_xclbin_PdiRelocationDef_table_t relocation =
        iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_at(source, i);
    out_list->data[i].transaction_offset =
        iree_hal_amdxdna_xclbin_PdiRelocationDef_offset_get(relocation);
    out_list->data[i].pdi_ordinal =
        iree_hal_amdxdna_xclbin_PdiRelocationDef_pdi_index_get(relocation);
    out_list->data[i].addend =
        iree_hal_amdxdna_xclbin_PdiRelocationDef_addend_get(relocation);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_control_parameter_relocations(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_t source,
    iree_hal_amdxdna_control_parameter_relocation_list_t* out_list) {
  out_list->data = NULL;
  out_list->count = 0;
  if (!source) return iree_ok_status();
  const iree_host_size_t count =
      iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_len(source);
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator,
                                             count * sizeof(*out_list->data),
                                             (void**)&out_list->data));
  out_list->count = count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_table_t relocation =
        iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_at(source, i);
    out_list->data[i].transaction_offset =
        iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_offset_get(
            relocation);
    out_list->data[i].addend =
        iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_addend_get(
            relocation);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_copy_control_parameters(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_xclbin_ControlParameterDef_vec_t source,
    iree_hal_amdxdna_kernel_params_t* params) {
  if (!source) return iree_ok_status();
  const iree_host_size_t count =
      iree_hal_amdxdna_xclbin_ControlParameterDef_vec_len(source);
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, count * sizeof(*params->parameters),
                            (void**)&params->parameters));
  memset(params->parameters, 0, count * sizeof(*params->parameters));
  params->parameter_count = count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_amdxdna_xclbin_ControlParameterDef_table_t parameter =
        iree_hal_amdxdna_xclbin_ControlParameterDef_vec_at(source, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_xclbin_ControlParameterDef_name_get(parameter);
    iree_hal_executable_function_parameter_t* out_parameter =
        &params->parameters[i];
    out_parameter->type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
    out_parameter->flags =
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET;
    out_parameter->size =
        (uint16_t)iree_hal_amdxdna_xclbin_ControlParameterDef_byte_length_get(
            parameter);
    out_parameter->offset = (uint16_t)
        iree_hal_amdxdna_xclbin_ControlParameterDef_constant_offset_get(
            parameter);
    out_parameter->native_abi_offset =
        (uint16_t)(iree_hal_amdxdna_xclbin_ControlParameterDef_state_table_index_get(
                       parameter) *
                   sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_copy_string_view(
        host_allocator,
        iree_make_string_view(name, flatbuffers_string_len(name)),
        &out_parameter->name));
    const uint32_t parameter_end =
        (uint32_t)out_parameter->offset + out_parameter->size;
    if (parameter_end > params->constant_byte_length) {
      params->constant_byte_length = parameter_end;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_kernel_params_allocate_runlists(
    iree_allocator_t host_allocator, iree_hal_amdxdna_kernel_params_t* params,
    iree_host_size_t run_count) {
  if (run_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, run_count * sizeof(*params->asm_inst_runlist),
      (void**)&params->asm_inst_runlist));
  memset(params->asm_inst_runlist, 0,
         run_count * sizeof(*params->asm_inst_runlist));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, run_count * sizeof(*params->patch_runlist),
      (void**)&params->patch_runlist));
  memset(params->patch_runlist, 0, run_count * sizeof(*params->patch_runlist));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, run_count * sizeof(*params->pdi_relocation_runlist),
      (void**)&params->pdi_relocation_runlist));
  memset(params->pdi_relocation_runlist, 0,
         run_count * sizeof(*params->pdi_relocation_runlist));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      run_count * sizeof(*params->control_parameter_relocation_runlist),
      (void**)&params->control_parameter_relocation_runlist));
  memset(params->control_parameter_relocation_runlist, 0,
         run_count * sizeof(*params->control_parameter_relocation_runlist));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, run_count * sizeof(*params->constant_patch_runlist),
      (void**)&params->constant_patch_runlist));
  memset(params->constant_patch_runlist, 0,
         run_count * sizeof(*params->constant_patch_runlist));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, run_count * sizeof(*params->reconf_data_runlist),
      (void**)&params->reconf_data_runlist));
  memset(params->reconf_data_runlist, 0,
         run_count * sizeof(*params->reconf_data_runlist));
  params->asm_inst_runlist_count = run_count;
  params->patch_runlist_count = run_count;
  params->pdi_relocation_runlist_count = run_count;
  params->control_parameter_relocation_runlist_count = run_count;
  params->constant_patch_runlist_count = run_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_pdi_executable_create(
    iree_const_byte_span_t executable_data, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_pdi_flatbuffer_verify(executable_data));

  iree_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_ExecutableDef_as_root(executable_data.data);
  iree_hal_amdxdna_PdiDef_vec_t pdis_vec =
      iree_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  iree_hal_amdxdna_EntryPointDef_vec_t entry_points_vec =
      iree_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      iree_hal_amdxdna_EntryPointDef_vec_len(entry_points_vec);

  iree_hal_amdxdna_executable* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_executable_allocate(host_allocator,
                                               entry_point_count, &executable));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       ++entry_ordinal) {
    iree_hal_amdxdna_EntryPointDef_table_t entry_point_def =
        iree_hal_amdxdna_EntryPointDef_vec_at(entry_points_vec, entry_ordinal);
    iree_hal_amdxdna_kernel_params_t* params =
        &executable->entry_points[entry_ordinal];
    flatbuffers_string_t name =
        iree_hal_amdxdna_EntryPointDef_name_get(entry_point_def);
    status = iree_hal_amdxdna_copy_string_view(
        host_allocator,
        iree_make_string_view(name, flatbuffers_string_len(name)),
        &params->kernel_name);
    if (!iree_status_is_ok(status)) break;

    int32_t pdi_index =
        iree_hal_amdxdna_EntryPointDef_pdi_index_get(entry_point_def);
    if (pdi_index >= 0) {
      iree_hal_amdxdna_PdiDef_table_t pdi_def =
          iree_hal_amdxdna_PdiDef_vec_at(pdis_vec, pdi_index);
      flatbuffers_uint8_vec_t pdi_fb = iree_hal_amdxdna_PdiDef_pdi_get(pdi_def);
      status = iree_hal_amdxdna_copy_u8_span(
          host_allocator,
          iree_make_const_byte_span(pdi_fb, flatbuffers_uint8_vec_len(pdi_fb)),
          &params->pdi);
      if (!iree_status_is_ok(status)) break;
    }

    iree_hal_amdxdna_RunDef_vec_t runs_vec =
        iree_hal_amdxdna_EntryPointDef_runs_get(entry_point_def);
    iree_host_size_t run_count = iree_hal_amdxdna_RunDef_vec_len(runs_vec);
    status = iree_hal_amdxdna_kernel_params_allocate_runlists(
        host_allocator, params, run_count);
    if (!iree_status_is_ok(status)) break;
    for (iree_host_size_t run_i = 0; run_i < run_count; ++run_i) {
      iree_hal_amdxdna_RunDef_table_t run_def =
          iree_hal_amdxdna_RunDef_vec_at(runs_vec, run_i);
      status = iree_hal_amdxdna_append_run_params(
          host_allocator, params, run_i,
          iree_hal_amdxdna_RunDef_control_code_get(run_def),
          iree_hal_amdxdna_RunDef_data_payload_get(run_def),
          iree_hal_amdxdna_RunDef_patch_table_get(run_def));
      if (!iree_status_is_ok(status)) break;
    }
    if (!iree_status_is_ok(status)) break;

    IREE_TRACE({
      iree_hal_amdxdna_FileLineLocDef_table_t source_loc =
          iree_hal_amdxdna_EntryPointDef_source_location_get(entry_point_def);
      if (source_loc) {
        flatbuffers_string_t filename =
            iree_hal_amdxdna_FileLineLocDef_filename_get(source_loc);
        uint32_t line = iree_hal_amdxdna_FileLineLocDef_line_get(source_loc);
        if (filename) {
          status = iree_hal_amdxdna_copy_string_view(
              host_allocator,
              iree_make_string_view(filename, flatbuffers_string_len(filename)),
              &params->source_filename);
        }
        params->source_line = line;
      }
    });
    if (!iree_status_is_ok(status)) break;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_executable_deinitialize(executable);
    iree_allocator_free(host_allocator, executable);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_xclbin_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    iree_const_byte_span_t executable_data, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_xclbin_flatbuffer_verify(executable_data));

  iree_hal_amdxdna_xclbin_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_xclbin_ExecutableDef_as_root(executable_data.data);
  iree_hal_amdxdna_xclbin_XclbinDef_vec_t xclbins_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_get(executable_def);
  iree_hal_amdxdna_xclbin_PdiDef_vec_t pdis_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_pdis_get(executable_def);
  iree_hal_amdxdna_xclbin_EntryPointDef_vec_t entry_points_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entry_points_vec);

  iree_hal_amdxdna_executable* executable = NULL;
  iree_status_t status = iree_hal_amdxdna_executable_allocate(
      host_allocator, entry_point_count, &executable);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  executable->native_device = native_device;

  const iree_host_size_t pdi_count =
      iree_hal_amdxdna_xclbin_PdiDef_vec_len(pdis_vec);
  if (pdi_count != 0) {
    status = iree_allocator_malloc(host_allocator,
                                   pdi_count * sizeof(*executable->pdis),
                                   (void**)&executable->pdis);
    if (iree_status_is_ok(status)) {
      memset(executable->pdis, 0, pdi_count * sizeof(*executable->pdis));
      executable->pdi_count = pdi_count;
    }
    for (iree_host_size_t i = 0; i < pdi_count && iree_status_is_ok(status);
         ++i) {
      iree_hal_amdxdna_xclbin_PdiDef_table_t pdi =
          iree_hal_amdxdna_xclbin_PdiDef_vec_at(pdis_vec, i);
      flatbuffers_uint8_vec_t pdi_bytes =
          iree_hal_amdxdna_xclbin_PdiDef_pdi_get(pdi);
      status = iree_hal_amdxdna_copy_u8_span(
          host_allocator,
          iree_make_const_byte_span(pdi_bytes,
                                    flatbuffers_uint8_vec_len(pdi_bytes)),
          &executable->pdis[i]);
    }
  }

  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       ++entry_ordinal) {
    if (!iree_status_is_ok(status)) break;
    iree_hal_amdxdna_kernel_params_t* params =
        &executable->entry_points[entry_ordinal];
    iree_hal_amdxdna_xclbin_EntryPointDef_table_t entry_point =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entry_points_vec,
                                                     entry_ordinal);
    flatbuffers_string_t name =
        iree_hal_amdxdna_xclbin_EntryPointDef_name_get(entry_point);
    status = iree_hal_amdxdna_copy_string_view(
        host_allocator,
        iree_make_string_view(name, flatbuffers_string_len(name)),
        &params->kernel_name);
    if (!iree_status_is_ok(status)) break;
    status = iree_hal_amdxdna_copy_control_parameters(
        host_allocator,
        iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_get(
            entry_point),
        params);
    if (!iree_status_is_ok(status)) break;

    int32_t xclbin_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(entry_point);
    if (xclbin_index >= 0) {
      iree_hal_amdxdna_xclbin_XclbinDef_table_t xclbin_def =
          iree_hal_amdxdna_xclbin_XclbinDef_vec_at(xclbins_vec, xclbin_index);
      flatbuffers_string_t xclbin_fb =
          iree_hal_amdxdna_xclbin_XclbinDef_xclbin_get(xclbin_def);
      status = iree_hal_amdxdna_copy_u8_span(
          host_allocator,
          iree_make_const_byte_span((const uint8_t*)xclbin_fb,
                                    flatbuffers_string_len(xclbin_fb)),
          &params->xclbin);
      if (!iree_status_is_ok(status)) break;
    }
    int32_t pdi_index =
        iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(entry_point);
    if (pdi_index >= 0) {
      iree_byte_span_t pdi_span = iree_byte_span_empty();
      status = iree_hal_amdxdna_xclbin_extract_pdi(
          iree_make_const_byte_span(params->xclbin.data, params->xclbin.count),
          (uint32_t)pdi_index, host_allocator, &pdi_span);
      if (iree_status_is_ok(status)) {
        params->pdi.data = pdi_span.data;
        params->pdi.count = pdi_span.data_length;
      }
      if (!iree_status_is_ok(status)) break;
    }

    iree_hal_amdxdna_xclbin_RunDef_vec_t runs =
        iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(entry_point);
    iree_host_size_t run_count = iree_hal_amdxdna_xclbin_RunDef_vec_len(runs);
    status = iree_hal_amdxdna_kernel_params_allocate_runlists(
        host_allocator, params, run_count);
    if (!iree_status_is_ok(status)) break;
    for (iree_host_size_t run_ordinal = 0; run_ordinal < run_count;
         ++run_ordinal) {
      iree_hal_amdxdna_xclbin_RunDef_table_t run =
          iree_hal_amdxdna_xclbin_RunDef_vec_at(runs, run_ordinal);
      status = iree_hal_amdxdna_append_run_params(
          host_allocator, params, run_ordinal,
          iree_hal_amdxdna_xclbin_RunDef_control_code_get(run),
          iree_hal_amdxdna_xclbin_RunDef_data_payload_get(run),
          iree_hal_amdxdna_xclbin_RunDef_patch_table_get(run));
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_copy_pdi_relocations(
            host_allocator,
            iree_hal_amdxdna_xclbin_RunDef_pdi_relocations_get(run),
            &params->pdi_relocation_runlist[run_ordinal]);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amdxdna_copy_control_parameter_relocations(
            host_allocator,
            iree_hal_amdxdna_xclbin_RunDef_control_parameter_relocations_get(
                run),
            &params->control_parameter_relocation_runlist[run_ordinal]);
      }
      if (!iree_status_is_ok(status)) break;
    }
    if (!iree_status_is_ok(status)) break;

    IREE_TRACE({
      iree_hal_amdxdna_xclbin_FileLineLocDef_table_t source_loc =
          iree_hal_amdxdna_xclbin_EntryPointDef_source_location_get(
              entry_point);
      if (source_loc) {
        flatbuffers_string_t filename =
            iree_hal_amdxdna_xclbin_FileLineLocDef_filename_get(source_loc);
        uint32_t line =
            iree_hal_amdxdna_xclbin_FileLineLocDef_line_get(source_loc);
        if (filename) {
          status = iree_hal_amdxdna_copy_string_view(
              host_allocator,
              iree_make_string_view(filename, flatbuffers_string_len(filename)),
              &params->source_filename);
        }
        params->source_line = line;
      }
    });
    if (!iree_status_is_ok(status)) break;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_executable_deinitialize(executable);
    iree_allocator_free(host_allocator, executable);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_elf_executable_create(
    iree_const_byte_span_t executable_data, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_elf_flatbuffer_verify(executable_data));

  iree_hal_amdxdna_elf_ExecutableDef_table_t executable_def =
      iree_hal_amdxdna_elf_ExecutableDef_as_root(executable_data.data);
  iree_hal_amdxdna_elf_ModuleDef_vec_t modules_vec =
      iree_hal_amdxdna_elf_ExecutableDef_modules_get(executable_def);
  iree_hal_amdxdna_elf_ExportDef_vec_t exports_vec =
      iree_hal_amdxdna_elf_ExecutableDef_exports_get(executable_def);
  iree_host_size_t export_count =
      iree_hal_amdxdna_elf_ExportDef_vec_len(exports_vec);

  iree_hal_amdxdna_executable* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_executable_allocate(host_allocator, export_count,
                                               &executable));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t export_ordinal = 0; export_ordinal < export_count;
       ++export_ordinal) {
    iree_hal_amdxdna_elf_ExportDef_table_t export_def =
        iree_hal_amdxdna_elf_ExportDef_vec_at(exports_vec, export_ordinal);
    iree_hal_amdxdna_kernel_params_t* params =
        &executable->entry_points[export_ordinal];
    flatbuffers_string_t name =
        iree_hal_amdxdna_elf_ExportDef_name_get(export_def);
    status = iree_hal_amdxdna_copy_string_view(
        host_allocator,
        iree_make_string_view(name, flatbuffers_string_len(name)),
        &params->kernel_name);
    if (!iree_status_is_ok(status)) break;

    uint32_t module_ordinal =
        iree_hal_amdxdna_elf_ExportDef_module_ordinal_get(export_def);
    iree_hal_amdxdna_elf_ModuleDef_table_t module_def =
        iree_hal_amdxdna_elf_ModuleDef_vec_at(modules_vec, module_ordinal);
    flatbuffers_uint8_vec_t elf_fb =
        iree_hal_amdxdna_elf_ModuleDef_elf_get(module_def);
    iree_hal_amdxdna_ctrlcode_elf_t parsed_elf;
    memset(&parsed_elf, 0, sizeof(parsed_elf));
    status = iree_hal_amdxdna_ctrlcode_elf_parse(
        iree_make_const_byte_span(elf_fb, flatbuffers_uint8_vec_len(elf_fb)),
        host_allocator, &parsed_elf);
    if (!iree_status_is_ok(status)) break;
    if (parsed_elf.ctrl_word_count == 0) {
      iree_hal_amdxdna_ctrlcode_elf_deinitialize(host_allocator, &parsed_elf);
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AELF export %" PRIhsz " has no control code",
                                export_ordinal);
      break;
    }

    status = iree_hal_amdxdna_kernel_params_allocate_runlists(host_allocator,
                                                              params, 1);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_ctrlcode_elf_deinitialize(host_allocator, &parsed_elf);
      break;
    }
    params->asm_inst_runlist[0].data = parsed_elf.ctrl_words;
    params->asm_inst_runlist[0].count = parsed_elf.ctrl_word_count;
    parsed_elf.ctrl_words = NULL;
    parsed_elf.ctrl_word_count = 0;
    params->patch_runlist[0].data = parsed_elf.patches;
    params->patch_runlist[0].count = parsed_elf.patch_count;
    parsed_elf.patches = NULL;
    parsed_elf.patch_count = 0;
    params->n_pdi_loads = 0;
    params->n_reconfigure_runs = 0;
    params->partition_cols = parsed_elf.partition_cols;
    params->dpu_slices = parsed_elf.dpu_slices;
    params->dpu_slice_count = parsed_elf.dpu_slice_count;
    parsed_elf.dpu_slices = NULL;
    parsed_elf.dpu_slice_count = 0;
    iree_hal_amdxdna_ctrlcode_elf_deinitialize(host_allocator, &parsed_elf);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_executable_deinitialize(executable);
    iree_allocator_free(host_allocator, executable);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);

  iree_status_t pdi_status =
      iree_hal_amdxdna_pdi_flatbuffer_verify(load_params->executable_data);
  if (iree_status_is_ok(pdi_status)) {
    return iree_hal_amdxdna_pdi_executable_create(
        load_params->executable_data, host_allocator, out_executable);
  }
  iree_status_free(pdi_status);

  iree_status_t xclbin_status =
      iree_hal_amdxdna_xclbin_flatbuffer_verify(load_params->executable_data);
  if (iree_status_is_ok(xclbin_status)) {
    return iree_hal_amdxdna_xclbin_executable_create(
        native_device, load_params->executable_data, host_allocator,
        out_executable);
  }
  iree_status_free(xclbin_status);

  iree_status_t elf_status =
      iree_hal_amdxdna_elf_flatbuffer_verify(load_params->executable_data);
  if (iree_status_is_ok(elf_status)) {
    return iree_hal_amdxdna_elf_executable_create(
        load_params->executable_data, host_allocator, out_executable);
  }
  iree_status_free(elf_status);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unrecognized amdxdna executable image");
}

static void iree_hal_amdxdna_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_hal_amdxdna_device* hal_device = executable->hal_device;
  // Deinitialize first. That releases this executable's context leases and
  // destroys its PDI buffers, which frees code slots while the context cache
  // still holds the hardware context.
  iree_hal_amdxdna_executable_deinitialize(executable);
  if (hal_device) {
    iree_hal_amdxdna_device_drop_unleased_contexts(hal_device);
    iree_hal_amdxdna_allocator_set_model_reuse(hal_device->device_allocator,
                                               /*active=*/0);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_amdxdna_native_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  return executable->entry_point_count;
}

static iree_status_t iree_hal_amdxdna_native_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable function out of range");
  }
  const iree_hal_amdxdna_kernel_params_t* entry_point =
      &executable->entry_points[iree_hal_executable_function_index(function)];
  memset(out_info, 0, sizeof(*out_info));
  out_info->name = entry_point->kernel_name;
  out_info->constant_byte_length = entry_point->constant_byte_length;
  out_info->parameter_count = (uint16_t)entry_point->parameter_count;
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable function out of range");
  }
  const iree_hal_amdxdna_kernel_params_t* entry_point =
      &executable->entry_points[iree_hal_executable_function_index(function)];
  if (capacity < entry_point->parameter_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable parameter capacity %" PRIhsz
                            " is less than required count %" PRIhsz,
                            capacity, entry_point->parameter_count);
  }
  if (entry_point->parameter_count != 0 && !out_parameters) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna executable parameter output is NULL");
  }
  if (entry_point->parameter_count != 0) {
    memcpy(out_parameters, entry_point->parameters,
           entry_point->parameter_count * sizeof(*out_parameters));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    const iree_hal_amdxdna_kernel_params_t* entry_point =
        &executable->entry_points[i];
    if (iree_string_view_equal(name, entry_point->kernel_name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "amdxdna executable function '%.*s' not found",
                          (int)name.size, name.data);
}

static iree_status_t
iree_hal_amdxdna_native_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid amdxdna executable global");
}

static iree_status_t iree_hal_amdxdna_native_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid amdxdna executable global");
}

static const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable = {
    .destroy = iree_hal_amdxdna_native_executable_destroy,
    .function_count = iree_hal_amdxdna_native_executable_function_count,
    .function_info = iree_hal_amdxdna_native_executable_function_info,
    .function_parameters =
        iree_hal_amdxdna_native_executable_function_parameters,
    .lookup_function_by_name =
        iree_hal_amdxdna_native_executable_lookup_function_by_name,
    .try_lookup_global_by_name =
        iree_hal_amdxdna_native_executable_try_lookup_global_by_name,
    .global_info = iree_hal_amdxdna_native_executable_global_info,
    .global_buffer = iree_hal_amdxdna_native_executable_global_buffer,
};
