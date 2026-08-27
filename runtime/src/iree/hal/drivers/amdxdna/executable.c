// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/drivers/amdxdna/context_cache.h"
#include "iree/hal/drivers/amdxdna/executable_internal.h"
#include "iree/hal/drivers/amdxdna/util.h"
#include "iree/hal/drivers/amdxdna/xclbin_util.h"
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
                                kAmdxdnaXclbinExecutableCompatFormat);
}

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_amdxdna_executable_vtable,
      iree_hal_amdxdna_executable);
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
    iree_status_t status = iree_hal_amdxdna_device_pin_context(
        device,
        iree_make_const_byte_span(params->pdi.data, params->pdi.count),
        iree_make_const_byte_span(params->xclbin.data, params->xclbin.count),
        params->kernel_name, &context_lease);
    iree_hal_amdxdna_native_context_ref_t* context_ref = NULL;
    if (iree_status_is_ok(status)) {
      context_ref =
          iree_hal_amdxdna_context_cache_lease_retain_context(context_lease);
      if (!context_ref) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "amdxdna executable context lease could not retain context");
      }
    }
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
  for (iree_host_size_t i = 0; i < params->constant_patch_runlist_count; ++i) {
    iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
        host_allocator, &params->constant_patch_runlist[i]);
  }
  iree_allocator_free(host_allocator, params->constant_patch_runlist);
  params->constant_patch_runlist = NULL;
  params->constant_patch_runlist_count = 0;
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
  iree_hal_amdxdna_native_context_ref_release(executable->context);
  executable->context = NULL;
  iree_slim_mutex_deinitialize(&executable->context_mutex);
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
    if (!iree_status_is_ok(status)) return status;
    format = kAmdxdnaXclbinExecutableFormat;
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
  iree_hal_amdxdna_xclbin_EntryPointDef_vec_t entry_points_vec =
      iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(executable_def);
  iree_host_size_t entry_point_count =
      iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entry_points_vec);

  iree_hal_amdxdna_executable* executable = NULL;
  iree_status_t status = iree_hal_amdxdna_executable_allocate(
      host_allocator, entry_point_count, &executable);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       ++entry_ordinal) {
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

iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  (void)native_device;
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
        load_params->executable_data, host_allocator, out_executable);
  }
  iree_status_free(xclbin_status);
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
  iree_hal_amdxdna_executable_deinitialize(executable);
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
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_native_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  (void)capacity;
  (void)out_parameters;
  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->entry_point_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna executable function out of range");
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
