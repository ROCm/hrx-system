// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hrx_amdxdna.h"
#include "hrx_internal.h"

#if defined(HRX_HAS_IREE_AMDXDNA_DRIVER)
#include "iree/base/internal/flatcc/building.h"
#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"
#include "iree/hal/drivers/amdxdna/executable.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_builder.h"

static bool hrx_amdxdna_span_is_valid(hrx_const_byte_span_t span) {
  return span.data_length == 0 || span.data != NULL;
}

#define HRX_AMDXDNA_V0_RECORD_LENGTH(type, last_field) \
  (offsetof(type, last_field) + sizeof(((type*)0)->last_field))

static const size_t hrx_amdxdna_run_v0_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_run_t, data_payload);
static const size_t hrx_amdxdna_entry_point_v0_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_entry_point_t,
                                 run_count);
static const size_t hrx_amdxdna_entry_point_control_parameters_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_entry_point_t,
                                 control_parameter_count);
static const size_t hrx_amdxdna_control_parameter_v0_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_control_parameter_t, kind);
static const size_t hrx_amdxdna_create_params_v0_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_create_params_t,
                                 entry_point_count);
static const size_t hrx_amdxdna_run_pdi_relocations_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_run_t,
                                 pdi_relocation_count);
static const size_t
    hrx_amdxdna_run_control_parameter_relocations_record_length =
        HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_run_t,
                                     control_parameter_relocation_count);
static const size_t hrx_amdxdna_create_params_pdis_record_length =
    HRX_AMDXDNA_V0_RECORD_LENGTH(hrx_amdxdna_executable_create_params_t,
                                 pdi_count);
static const size_t hrx_amdxdna_abi_header_length =
    offsetof(hrx_amdxdna_executable_run_t, abi_version) + sizeof(uint32_t);

static bool hrx_amdxdna_record_has_field(uint32_t record_length,
                                         size_t field_end) {
  return record_length >= field_end;
}

static bool hrx_amdxdna_relocation_ranges_overlap(uint32_t lhs_offset,
                                                  uint32_t rhs_offset) {
  return lhs_offset < rhs_offset + sizeof(uint64_t) &&
         rhs_offset < lhs_offset + sizeof(uint64_t);
}

static bool hrx_amdxdna_record_stride_is_valid(const void* record,
                                               uint32_t record_length,
                                               size_t min_record_length,
                                               size_t record_alignment) {
  const uintptr_t address = (uintptr_t)record;
  return record && address % record_alignment == 0 &&
         record_length >= min_record_length &&
         record_length % record_alignment == 0 &&
         address <= UINTPTR_MAX - record_length;
}

static const hrx_amdxdna_executable_entry_point_t* hrx_amdxdna_next_entry(
    const hrx_amdxdna_executable_entry_point_t* entry) {
  return (const hrx_amdxdna_executable_entry_point_t*)((const uint8_t*)entry +
                                                       entry->record_length);
}

static const hrx_amdxdna_executable_run_t* hrx_amdxdna_next_run(
    const hrx_amdxdna_executable_run_t* run) {
  return (const hrx_amdxdna_executable_run_t*)((const uint8_t*)run +
                                               run->record_length);
}

static const hrx_amdxdna_control_parameter_t*
hrx_amdxdna_next_control_parameter(
    const hrx_amdxdna_control_parameter_t* parameter) {
  return (const hrx_amdxdna_control_parameter_t*)((const uint8_t*)parameter +
                                                  parameter->record_length);
}

static hrx_status_t hrx_amdxdna_validate_executable_create(
    const hrx_amdxdna_executable_create_params_t* params) {
  if (!params) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "params is NULL");
  }
  if ((uintptr_t)params % _Alignof(hrx_amdxdna_executable_create_params_t) !=
      0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "amdxdna executable parameter record is misaligned");
  }
  if (params->record_length < hrx_amdxdna_abi_header_length) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "amdxdna executable parameter ABI header is truncated");
  }
  if (params->abi_version >
      HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_1) {
    return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                           "unsupported amdxdna executable parameter ABI");
  }
  if (params->record_length < hrx_amdxdna_create_params_v0_record_length) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "amdxdna executable parameter record is truncated");
  }
  if (params->flags != 0 || params->reserved != 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "amdxdna executable reserved fields must be zero");
  }
  if (!params->xclbins || params->xclbin_count == 0 || !params->entry_points ||
      params->entry_point_count == 0) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "xclbins and entry points are required");
  }
  if (params->xclbin_count > INT32_MAX ||
      params->entry_point_count > UINT32_MAX) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "amdxdna executable record count is too large");
  }
  for (size_t i = 0; i < params->xclbin_count; ++i) {
    if (!params->xclbins[i].data || params->xclbins[i].data_length == 0) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "xclbin data is empty");
    }
    if (params->xclbins[i].data_length > UINT32_MAX) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "xclbin data is too large");
    }
  }
  const bool has_standalone_pdis = hrx_amdxdna_record_has_field(
      params->record_length, hrx_amdxdna_create_params_pdis_record_length);
  const size_t pdi_count = has_standalone_pdis ? params->pdi_count : 0;
  if (pdi_count != 0 &&
      params->abi_version <
          HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_1) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "standalone PDIs require amdxdna executable parameter ABI version 1");
  }
  if (pdi_count != 0 && !params->pdis) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "standalone PDI array is NULL");
  }
  if (pdi_count > UINT32_MAX) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "standalone PDI count is too large");
  }
  for (size_t i = 0; i < pdi_count; ++i) {
    if (!params->pdis[i].data || params->pdis[i].data_length == 0) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "standalone PDI data is empty");
    }
    if (params->pdis[i].data_length > UINT32_MAX) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "standalone PDI data is too large");
    }
  }

  const hrx_amdxdna_executable_entry_point_t* entry = params->entry_points;
  for (size_t i = 0; i < params->entry_point_count; ++i) {
    if ((uintptr_t)entry % _Alignof(hrx_amdxdna_executable_entry_point_t) !=
        0) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "amdxdna entry-point record is misaligned");
    }
    if (entry->record_length < hrx_amdxdna_abi_header_length) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "amdxdna entry-point ABI header is truncated");
    }
    if (entry->abi_version > HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_1) {
      return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                             "unsupported amdxdna entry-point record ABI");
    }
    if (!hrx_amdxdna_record_stride_is_valid(
            entry, entry->record_length,
            hrx_amdxdna_entry_point_v0_record_length,
            _Alignof(hrx_amdxdna_executable_entry_point_t))) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "amdxdna entry-point record stride is invalid");
    }
    if (!entry->name.data || entry->name.size == 0 || !entry->runs ||
        entry->run_count == 0 ||
        !hrx_amdxdna_span_is_valid(
            (hrx_const_byte_span_t){(const uint8_t*)entry->source_file.data,
                                    entry->source_file.size})) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "amdxdna entry-point description is invalid");
    }
    if (entry->name.size > UINT32_MAX || entry->source_file.size > UINT32_MAX ||
        entry->run_count > UINT32_MAX) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "amdxdna entry-point data is too large");
    }
    if (entry->source_line > INT32_MAX) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "source line is out of range");
    }
    if (entry->context_mode == HRX_AMDXDNA_CONTEXT_MODE_CREATE) {
      if (entry->xclbin_ordinal >= params->xclbin_count ||
          entry->xclbin_ordinal > INT32_MAX || entry->pdi_ordinal > INT32_MAX) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "entry-point context ordinal is out of range");
      }
    } else if (entry->context_mode == HRX_AMDXDNA_CONTEXT_MODE_REUSE) {
      if (entry->xclbin_ordinal != 0 || entry->pdi_ordinal != 0) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "reuse-context entry-point ordinals must be zero");
      }
    } else {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "unknown amdxdna context mode");
    }

    const bool has_control_parameters = hrx_amdxdna_record_has_field(
        entry->record_length,
        hrx_amdxdna_entry_point_control_parameters_record_length);
    const size_t control_parameter_count =
        has_control_parameters ? entry->control_parameter_count : 0;
    if (control_parameter_count != 0 &&
        entry->abi_version < HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_1) {
      return hrx_make_status(
          HRX_STATUS_INVALID_ARGUMENT,
          "control parameters require amdxdna entry-point ABI version 1");
    }
    if (control_parameter_count != 0 && !entry->control_parameters) {
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "control parameter array is NULL");
    }
    if (control_parameter_count > 32) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "control parameter count exceeds 32 slots");
    }
    uint32_t used_state_slots = 0;
    const hrx_amdxdna_control_parameter_t* parameter =
        entry->control_parameters;
    for (size_t j = 0; j < control_parameter_count; ++j) {
      if (!hrx_amdxdna_record_stride_is_valid(
              parameter, parameter->record_length,
              hrx_amdxdna_control_parameter_v0_record_length,
              _Alignof(hrx_amdxdna_control_parameter_t))) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "amdxdna control parameter record stride is invalid");
      }
      if (parameter->abi_version !=
          HRX_AMDXDNA_CONTROL_PARAMETER_ABI_VERSION_0) {
        return hrx_make_status(
            HRX_STATUS_UNIMPLEMENTED,
            "unsupported amdxdna control parameter record ABI");
      }
      if (!parameter->name.data || parameter->name.size == 0 ||
          parameter->name.size > UINT32_MAX || !parameter->scalar_type.data ||
          parameter->scalar_type.size == 0 ||
          parameter->scalar_type.size > UINT32_MAX ||
          parameter->byte_length == 0 || parameter->byte_length > 4 ||
          parameter->constant_offset > UINT16_MAX ||
          parameter->constant_offset > UINT32_MAX - parameter->byte_length) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "amdxdna control parameter is invalid");
      }
      if (parameter->state_table_index >= 32 ||
          (used_state_slots & (1u << parameter->state_table_index)) != 0) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "amdxdna control parameter state-table slot is invalid or reused");
      }
      used_state_slots |= 1u << parameter->state_table_index;
      if (parameter->kind != HRX_AMDXDNA_CONTROL_PARAMETER_KIND_CORE &&
          parameter->kind != HRX_AMDXDNA_CONTROL_PARAMETER_KIND_ADDRESS) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "unknown amdxdna control parameter kind");
      }
      const hrx_amdxdna_control_parameter_t* other = entry->control_parameters;
      for (size_t k = 0; k < j; ++k) {
        const uint32_t parameter_end =
            parameter->constant_offset + parameter->byte_length;
        const uint32_t other_end = other->constant_offset + other->byte_length;
        if (parameter->constant_offset < other_end &&
            other->constant_offset < parameter_end) {
          return hrx_make_status(
              HRX_STATUS_INVALID_ARGUMENT,
              "amdxdna control parameter constant ranges overlap");
        }
        if (parameter->name.size == other->name.size &&
            memcmp(parameter->name.data, other->name.data,
                   parameter->name.size) == 0) {
          return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                 "amdxdna control parameter name is reused");
        }
        other = hrx_amdxdna_next_control_parameter(other);
      }
      parameter = hrx_amdxdna_next_control_parameter(parameter);
    }

    const hrx_amdxdna_executable_run_t* run = entry->runs;
    for (size_t j = 0; j < entry->run_count; ++j) {
      if ((uintptr_t)run % _Alignof(hrx_amdxdna_executable_run_t) != 0) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "amdxdna run record is misaligned");
      }
      if (run->record_length < hrx_amdxdna_abi_header_length) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "amdxdna run ABI header is truncated");
      }
      if (run->abi_version > HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_1) {
        return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                               "unsupported amdxdna run record ABI");
      }
      if (!hrx_amdxdna_record_stride_is_valid(
              run, run->record_length, hrx_amdxdna_run_v0_record_length,
              _Alignof(hrx_amdxdna_executable_run_t))) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "amdxdna run record stride is invalid");
      }
      if (!run->transaction.data || run->transaction.data_length == 0 ||
          run->transaction.data_length % sizeof(uint32_t) != 0 ||
          !hrx_amdxdna_span_is_valid(run->data_payload) ||
          run->data_payload.data_length % sizeof(uint32_t) != 0) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "amdxdna run description is invalid");
      }
      if (run->transaction.data_length > UINT32_MAX ||
          run->data_payload.data_length > UINT32_MAX) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "amdxdna run data is too large");
      }
      const bool has_pdi_relocations = hrx_amdxdna_record_has_field(
          run->record_length, hrx_amdxdna_run_pdi_relocations_record_length);
      const size_t relocation_count =
          has_pdi_relocations ? run->pdi_relocation_count : 0;
      if (relocation_count != 0 &&
          run->abi_version < HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_1) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "PDI relocations require amdxdna run ABI version 1");
      }
      if (relocation_count != 0 && !run->pdi_relocations) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "PDI relocation array is NULL");
      }
      for (size_t k = 0; k < relocation_count; ++k) {
        const hrx_amdxdna_pdi_relocation_t* relocation =
            &run->pdi_relocations[k];
        if ((relocation->transaction_offset % sizeof(uint32_t)) != 0 ||
            run->transaction.data_length < sizeof(uint64_t) ||
            relocation->transaction_offset >
                run->transaction.data_length - sizeof(uint64_t)) {
          return hrx_make_status(
              HRX_STATUS_INVALID_ARGUMENT,
              "PDI relocation does not name a 64-bit transaction address");
        }
        if (relocation->pdi_ordinal >= pdi_count) {
          return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                 "PDI relocation ordinal is out of range");
        }
        for (size_t other_k = 0; other_k < k; ++other_k) {
          if (hrx_amdxdna_relocation_ranges_overlap(
                  relocation->transaction_offset,
                  run->pdi_relocations[other_k].transaction_offset)) {
            return hrx_make_status(
                HRX_STATUS_INVALID_ARGUMENT,
                "PDI relocation transaction address ranges overlap");
          }
        }
      }
      const bool has_control_parameter_relocations =
          hrx_amdxdna_record_has_field(
              run->record_length,
              hrx_amdxdna_run_control_parameter_relocations_record_length);
      const size_t control_parameter_relocation_count =
          has_control_parameter_relocations
              ? run->control_parameter_relocation_count
              : 0;
      if (control_parameter_relocation_count != 0 &&
          run->abi_version < HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_1) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "control parameter relocations require amdxdna run ABI version 1");
      }
      if (control_parameter_relocation_count != 0 &&
          !run->control_parameter_relocations) {
        return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                               "control parameter relocation array is NULL");
      }
      if (control_parameter_relocation_count != 0 &&
          control_parameter_count == 0) {
        return hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "control parameter relocation requires parameter metadata");
      }
      for (size_t k = 0; k < control_parameter_relocation_count; ++k) {
        const hrx_amdxdna_control_parameter_relocation_t* relocation =
            &run->control_parameter_relocations[k];
        if ((relocation->transaction_offset % sizeof(uint32_t)) != 0 ||
            run->transaction.data_length < sizeof(uint64_t) ||
            relocation->transaction_offset >
                run->transaction.data_length - sizeof(uint64_t)) {
          return hrx_make_status(
              HRX_STATUS_INVALID_ARGUMENT,
              "control parameter relocation does not name a 64-bit "
              "transaction address");
        }
        for (size_t pdi_k = 0; pdi_k < relocation_count; ++pdi_k) {
          if (hrx_amdxdna_relocation_ranges_overlap(
                  relocation->transaction_offset,
                  run->pdi_relocations[pdi_k].transaction_offset)) {
            return hrx_make_status(
                HRX_STATUS_INVALID_ARGUMENT,
                "PDI and control parameter relocation ranges overlap");
          }
        }
        for (size_t other_k = 0; other_k < k; ++other_k) {
          if (hrx_amdxdna_relocation_ranges_overlap(
                  relocation->transaction_offset,
                  run->control_parameter_relocations[other_k]
                      .transaction_offset)) {
            return hrx_make_status(
                HRX_STATUS_INVALID_ARGUMENT,
                "control parameter relocation transaction address ranges "
                "overlap");
          }
        }
      }
      run = hrx_amdxdna_next_run(run);
    }
    entry = hrx_amdxdna_next_entry(entry);
  }
  return hrx_ok_status();
}

static hrx_status_t hrx_amdxdna_builder_failure(const char* message) {
  return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY, message);
}

static hrx_status_t hrx_amdxdna_allocate_ref_array(size_t count,
                                                   size_t element_size,
                                                   void** out_ptr) {
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(count, element_size, &allocation_size)) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "amdxdna executable definition is too large");
  }
  return hrx_host_allocator_malloc_uninitialized(hrx_host_allocator_system(),
                                                 allocation_size, out_ptr);
}

static hrx_status_t hrx_amdxdna_create_u32_vec(
    flatbuffers_builder_t* builder, hrx_const_byte_span_t span,
    flatbuffers_uint32_vec_ref_t* out_ref) {
  *out_ref = 0;
  if (span.data_length == 0) {
    *out_ref = flatbuffers_uint32_vec_create(builder, NULL, 0);
    return *out_ref ? hrx_ok_status()
                    : hrx_amdxdna_builder_failure(
                          "failed to add empty uint32 vector to executable "
                          "package");
  }
  uint32_t* words = NULL;
  hrx_status_t status = hrx_host_allocator_malloc_uninitialized(
      hrx_host_allocator_system(), span.data_length, (void**)&words);
  if (!hrx_status_is_ok(status)) return status;
  memcpy(words, span.data, span.data_length);
  *out_ref = flatbuffers_uint32_vec_create(builder, words,
                                           span.data_length / sizeof(uint32_t));
  hrx_host_allocator_free(hrx_host_allocator_system(), words);
  return *out_ref ? hrx_ok_status()
                  : hrx_amdxdna_builder_failure(
                        "failed to add uint32 vector to executable package");
}

static iree_hal_amdxdna_xclbin_EntryPointDef_ref_t
hrx_amdxdna_create_entry_point(
    flatbuffers_builder_t* builder, flatbuffers_string_ref_t name_ref,
    int32_t pdi_index, int32_t xclbin_index,
    iree_hal_amdxdna_xclbin_RunDef_vec_ref_t runs_ref,
    iree_hal_amdxdna_xclbin_FileLineLocDef_ref_t source_ref,
    iree_hal_amdxdna_xclbin_ControlParameterDef_vec_ref_t parameters_ref) {
  if (iree_hal_amdxdna_xclbin_EntryPointDef_start(builder) ||
      iree_hal_amdxdna_xclbin_EntryPointDef_name_add(builder, name_ref) ||
      iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_add(builder, pdi_index) ||
      iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_add(builder,
                                                             xclbin_index) ||
      iree_hal_amdxdna_xclbin_EntryPointDef_runs_add(builder, runs_ref) ||
      (source_ref && iree_hal_amdxdna_xclbin_EntryPointDef_source_location_add(
                         builder, source_ref)) ||
      (parameters_ref &&
       iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_add(
           builder, parameters_ref))) {
    return 0;
  }
  return iree_hal_amdxdna_xclbin_EntryPointDef_end(builder);
}
#endif  // HRX_HAS_IREE_AMDXDNA_DRIVER

hrx_status_t hrx_amdxdna_xadx_serialize(
    const hrx_amdxdna_executable_create_params_t* params,
    hrx_host_allocator_t host_allocator, uint8_t** out_data,
    size_t* out_data_length) {
  if (!out_data || !out_data_length) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "out_data or out_data_length is NULL");
  }
  *out_data = NULL;
  *out_data_length = 0;
#if !defined(HRX_HAS_IREE_AMDXDNA_DRIVER)
  (void)params;
  (void)host_allocator;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "HRX was built without the amdxdna driver");
#else
  hrx_status_t status = hrx_amdxdna_validate_executable_create(params);
  if (!hrx_status_is_ok(status)) return status;

  flatbuffers_builder_t builder;
  if (flatcc_builder_init(&builder) != 0) {
    return hrx_amdxdna_builder_failure(
        "failed to initialize amdxdna executable builder");
  }

  iree_hal_amdxdna_xclbin_XclbinDef_ref_t* xclbin_refs = NULL;
  iree_hal_amdxdna_xclbin_PdiDef_ref_t* pdi_refs = NULL;
  iree_hal_amdxdna_xclbin_EntryPointDef_ref_t* entry_refs = NULL;
  void* executable_data = NULL;
  size_t executable_data_size = 0;
  if (flatbuffers_failed(
          iree_hal_amdxdna_xclbin_ExecutableDef_start_as_root(&builder))) {
    status = hrx_amdxdna_builder_failure(
        "failed to start amdxdna executable package");
    goto cleanup;
  }

  status = hrx_amdxdna_allocate_ref_array(
      params->xclbin_count, sizeof(*xclbin_refs), (void**)&xclbin_refs);
  if (!hrx_status_is_ok(status)) goto cleanup;
  for (size_t i = 0; i < params->xclbin_count; ++i) {
    flatbuffers_string_ref_t data_ref = flatbuffers_string_create(
        &builder, (const char*)params->xclbins[i].data,
        params->xclbins[i].data_length);
    xclbin_refs[i] =
        iree_hal_amdxdna_xclbin_XclbinDef_create(&builder, data_ref);
    if (!data_ref || !xclbin_refs[i]) {
      status = hrx_amdxdna_builder_failure(
          "failed to add xclbin to amdxdna executable package");
      goto cleanup;
    }
  }

  const bool has_standalone_pdis = hrx_amdxdna_record_has_field(
      params->record_length, hrx_amdxdna_create_params_pdis_record_length);
  const size_t pdi_count = has_standalone_pdis ? params->pdi_count : 0;
  if (pdi_count != 0) {
    status = hrx_amdxdna_allocate_ref_array(pdi_count, sizeof(*pdi_refs),
                                            (void**)&pdi_refs);
    if (!hrx_status_is_ok(status)) goto cleanup;
  }
  for (size_t i = 0; i < pdi_count; ++i) {
    flatbuffers_uint8_vec_ref_t data_ref = flatbuffers_uint8_vec_create(
        &builder, params->pdis[i].data, params->pdis[i].data_length);
    pdi_refs[i] = iree_hal_amdxdna_xclbin_PdiDef_create(&builder, data_ref);
    if (!data_ref || !pdi_refs[i]) {
      status = hrx_amdxdna_builder_failure(
          "failed to add standalone PDI to amdxdna executable package");
      goto cleanup;
    }
  }

  status = hrx_amdxdna_allocate_ref_array(
      params->entry_point_count, sizeof(*entry_refs), (void**)&entry_refs);
  if (!hrx_status_is_ok(status)) goto cleanup;
  const hrx_amdxdna_executable_entry_point_t* entry = params->entry_points;
  for (size_t i = 0; i < params->entry_point_count; ++i) {
    iree_hal_amdxdna_xclbin_RunDef_ref_t* run_refs = NULL;
    status = hrx_amdxdna_allocate_ref_array(entry->run_count, sizeof(*run_refs),
                                            (void**)&run_refs);
    if (!hrx_status_is_ok(status)) goto cleanup;

    const hrx_amdxdna_executable_run_t* run = entry->runs;
    for (size_t j = 0; j < entry->run_count; ++j) {
      iree_hal_amdxdna_host_patch_table_t patch_table = {0};
      flatbuffers_uint32_vec_ref_t transaction_ref = 0;
      flatbuffers_uint32_vec_ref_t payload_ref = 0;
      flatbuffers_uint32_vec_ref_t patch_ref = 0;
      iree_status_t patch_status = iree_hal_amdxdna_build_host_patch_table(
          iree_allocator_system(),
          iree_make_const_byte_span(run->transaction.data,
                                    run->transaction.data_length),
          &patch_table);
      status = hrx_status_from_iree(patch_status);
      if (hrx_status_is_ok(status)) {
        status = hrx_amdxdna_create_u32_vec(&builder, run->transaction,
                                            &transaction_ref);
      }
      if (hrx_status_is_ok(status)) {
        status = hrx_amdxdna_create_u32_vec(&builder, run->data_payload,
                                            &payload_ref);
      }
      if (hrx_status_is_ok(status)) {
        iree_host_size_t patch_byte_length = 0;
        if (!iree_host_size_checked_mul(patch_table.count, sizeof(uint32_t),
                                        &patch_byte_length)) {
          status = hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                                   "amdxdna host patch table is too large");
        } else {
          hrx_const_byte_span_t patch_span = {(const uint8_t*)patch_table.data,
                                              patch_byte_length};
          status = hrx_amdxdna_create_u32_vec(&builder, patch_span, &patch_ref);
        }
      }
      iree_hal_amdxdna_host_patch_table_deinitialize(iree_allocator_system(),
                                                     &patch_table);
      iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_ref_t relocations_ref = 0;
      iree_hal_amdxdna_xclbin_PdiRelocationDef_ref_t* relocation_refs = NULL;
      const bool has_pdi_relocations = hrx_amdxdna_record_has_field(
          run->record_length, hrx_amdxdna_run_pdi_relocations_record_length);
      const size_t relocation_count =
          has_pdi_relocations ? run->pdi_relocation_count : 0;
      if (hrx_status_is_ok(status) && relocation_count != 0) {
        status = hrx_amdxdna_allocate_ref_array(relocation_count,
                                                sizeof(*relocation_refs),
                                                (void**)&relocation_refs);
      }
      for (size_t k = 0; k < relocation_count && hrx_status_is_ok(status);
           ++k) {
        const hrx_amdxdna_pdi_relocation_t* relocation =
            &run->pdi_relocations[k];
        relocation_refs[k] = iree_hal_amdxdna_xclbin_PdiRelocationDef_create(
            &builder, relocation->transaction_offset, relocation->pdi_ordinal,
            relocation->addend);
        if (!relocation_refs[k]) {
          status = hrx_amdxdna_builder_failure(
              "failed to add PDI relocation to amdxdna executable package");
        }
      }
      if (hrx_status_is_ok(status)) {
        relocations_ref = iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_create(
            &builder, relocation_refs, relocation_count);
        if (!relocations_ref) {
          status = hrx_amdxdna_builder_failure(
              "failed to add PDI relocation vector to amdxdna executable "
              "package");
        }
      }
      hrx_host_allocator_free(hrx_host_allocator_system(), relocation_refs);
      if (!hrx_status_is_ok(status)) break;
      iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_ref_t
          control_parameter_relocations_ref = 0;
      iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_ref_t*
          control_parameter_relocation_refs = NULL;
      const bool has_control_parameter_relocations =
          hrx_amdxdna_record_has_field(
              run->record_length,
              hrx_amdxdna_run_control_parameter_relocations_record_length);
      const size_t control_parameter_relocation_count =
          has_control_parameter_relocations
              ? run->control_parameter_relocation_count
              : 0;
      if (control_parameter_relocation_count != 0) {
        status = hrx_amdxdna_allocate_ref_array(
            control_parameter_relocation_count,
            sizeof(*control_parameter_relocation_refs),
            (void**)&control_parameter_relocation_refs);
      }
      for (size_t k = 0;
           k < control_parameter_relocation_count && hrx_status_is_ok(status);
           ++k) {
        const hrx_amdxdna_control_parameter_relocation_t* relocation =
            &run->control_parameter_relocations[k];
        control_parameter_relocation_refs[k] =
            iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_create(
                &builder, relocation->transaction_offset, relocation->addend);
        if (!control_parameter_relocation_refs[k]) {
          status = hrx_amdxdna_builder_failure(
              "failed to add control parameter relocation to amdxdna "
              "executable package");
        }
      }
      if (hrx_status_is_ok(status)) {
        control_parameter_relocations_ref =
            iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_create(
                &builder, control_parameter_relocation_refs,
                control_parameter_relocation_count);
        if (!control_parameter_relocations_ref) {
          status = hrx_amdxdna_builder_failure(
              "failed to add control parameter relocation vector to amdxdna "
              "executable package");
        }
      }
      hrx_host_allocator_free(hrx_host_allocator_system(),
                              control_parameter_relocation_refs);
      if (!hrx_status_is_ok(status)) break;
      run_refs[j] = iree_hal_amdxdna_xclbin_RunDef_create(
          &builder, transaction_ref, payload_ref, patch_ref, relocations_ref,
          control_parameter_relocations_ref);
      if (!run_refs[j]) {
        status = hrx_amdxdna_builder_failure(
            "failed to add run to amdxdna executable package");
        break;
      }
      run = hrx_amdxdna_next_run(run);
    }

    iree_hal_amdxdna_xclbin_ControlParameterDef_ref_t* parameter_refs = NULL;
    if (hrx_status_is_ok(status)) {
      flatbuffers_string_ref_t name_ref = flatbuffers_string_create(
          &builder, entry->name.data, entry->name.size);
      iree_hal_amdxdna_xclbin_RunDef_vec_ref_t runs_ref =
          iree_hal_amdxdna_xclbin_RunDef_vec_create(&builder, run_refs,
                                                    entry->run_count);
      iree_hal_amdxdna_xclbin_ControlParameterDef_vec_ref_t parameters_ref = 0;
      const bool has_control_parameters = hrx_amdxdna_record_has_field(
          entry->record_length,
          hrx_amdxdna_entry_point_control_parameters_record_length);
      const size_t control_parameter_count =
          has_control_parameters ? entry->control_parameter_count : 0;
      if (control_parameter_count != 0) {
        status = hrx_amdxdna_allocate_ref_array(control_parameter_count,
                                                sizeof(*parameter_refs),
                                                (void**)&parameter_refs);
      }
      const hrx_amdxdna_control_parameter_t* parameter =
          entry->control_parameters;
      for (size_t j = 0;
           j < control_parameter_count && hrx_status_is_ok(status); ++j) {
        flatbuffers_string_ref_t parameter_name_ref = flatbuffers_string_create(
            &builder, parameter->name.data, parameter->name.size);
        flatbuffers_string_ref_t scalar_type_ref = flatbuffers_string_create(
            &builder, parameter->scalar_type.data, parameter->scalar_type.size);
        parameter_refs[j] = iree_hal_amdxdna_xclbin_ControlParameterDef_create(
            &builder, parameter_name_ref, scalar_type_ref,
            parameter->constant_offset, parameter->state_table_index,
            parameter->byte_length, parameter->kind);
        if (!parameter_name_ref || !scalar_type_ref || !parameter_refs[j]) {
          status = hrx_amdxdna_builder_failure(
              "failed to add control parameter to amdxdna executable package");
        }
        parameter = hrx_amdxdna_next_control_parameter(parameter);
      }
      if (hrx_status_is_ok(status) && control_parameter_count != 0) {
        parameters_ref = iree_hal_amdxdna_xclbin_ControlParameterDef_vec_create(
            &builder, parameter_refs, control_parameter_count);
        if (!parameters_ref) {
          status = hrx_amdxdna_builder_failure(
              "failed to add control parameter vector to amdxdna executable "
              "package");
        }
      }
      iree_hal_amdxdna_xclbin_FileLineLocDef_ref_t source_ref = 0;
      if (hrx_status_is_ok(status) && entry->source_file.size != 0) {
        flatbuffers_string_ref_t filename_ref = flatbuffers_string_create(
            &builder, entry->source_file.data, entry->source_file.size);
        source_ref = iree_hal_amdxdna_xclbin_FileLineLocDef_create(
            &builder, filename_ref, (int32_t)entry->source_line);
        if (!filename_ref || !source_ref) {
          status = hrx_amdxdna_builder_failure(
              "failed to add source location to amdxdna executable package");
        }
      }
      const int32_t xclbin_index =
          entry->context_mode == HRX_AMDXDNA_CONTEXT_MODE_CREATE
              ? (int32_t)entry->xclbin_ordinal
              : -1;
      const int32_t pdi_index =
          entry->context_mode == HRX_AMDXDNA_CONTEXT_MODE_CREATE
              ? (int32_t)entry->pdi_ordinal
              : -1;
      if (hrx_status_is_ok(status)) {
        entry_refs[i] = hrx_amdxdna_create_entry_point(
            &builder, name_ref, pdi_index, xclbin_index, runs_ref, source_ref,
            parameters_ref);
        if (!name_ref || !runs_ref || !entry_refs[i]) {
          status = hrx_amdxdna_builder_failure(
              "failed to add entry point to amdxdna executable package");
        }
      }
    }
    hrx_host_allocator_free(hrx_host_allocator_system(), parameter_refs);
    hrx_host_allocator_free(hrx_host_allocator_system(), run_refs);
    if (!hrx_status_is_ok(status)) goto cleanup;
    entry = hrx_amdxdna_next_entry(entry);
  }

  {
    iree_hal_amdxdna_xclbin_XclbinDef_vec_ref_t xclbins_ref =
        iree_hal_amdxdna_xclbin_XclbinDef_vec_create(&builder, xclbin_refs,
                                                     params->xclbin_count);
    iree_hal_amdxdna_xclbin_PdiDef_vec_ref_t pdis_ref =
        iree_hal_amdxdna_xclbin_PdiDef_vec_create(&builder, pdi_refs,
                                                  pdi_count);
    iree_hal_amdxdna_xclbin_EntryPointDef_vec_ref_t entries_ref =
        iree_hal_amdxdna_xclbin_EntryPointDef_vec_create(
            &builder, entry_refs, params->entry_point_count);
    if (!xclbins_ref || !pdis_ref || !entries_ref ||
        flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_add(
            &builder, xclbins_ref)) ||
        flatbuffers_failed(
            iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_add(
                &builder, entries_ref)) ||
        flatbuffers_failed(iree_hal_amdxdna_xclbin_ExecutableDef_pdis_add(
            &builder, pdis_ref)) ||
        !iree_hal_amdxdna_xclbin_ExecutableDef_end_as_root(&builder)) {
      status = hrx_amdxdna_builder_failure(
          "failed to finish amdxdna executable package");
      goto cleanup;
    }
  }

  executable_data =
      flatcc_builder_finalize_aligned_buffer(&builder, &executable_data_size);
  if (!executable_data || executable_data_size == 0) {
    status = hrx_amdxdna_builder_failure(
        "failed to finalize amdxdna executable package");
    goto cleanup;
  }
  status = hrx_host_allocator_clone(host_allocator, executable_data,
                                    executable_data_size, (void**)out_data);
  if (hrx_status_is_ok(status)) *out_data_length = executable_data_size;

cleanup:
  flatcc_builder_aligned_free(executable_data);
  hrx_host_allocator_free(hrx_host_allocator_system(), entry_refs);
  hrx_host_allocator_free(hrx_host_allocator_system(), pdi_refs);
  hrx_host_allocator_free(hrx_host_allocator_system(), xclbin_refs);
  flatcc_builder_clear(&builder);
  return status;
#endif  // HRX_HAS_IREE_AMDXDNA_DRIVER
}

hrx_status_t hrx_amdxdna_executable_create(
    hrx_device_t device, const hrx_amdxdna_executable_create_params_t* params,
    hrx_executable_t* executable) {
  if (!device || !executable) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or executable is NULL");
  }
  *executable = NULL;
  hrx_host_allocator_t host_allocator = hrx_host_allocator_system();
  uint8_t* executable_data = NULL;
  size_t executable_data_size = 0;
  hrx_status_t status = hrx_amdxdna_xadx_serialize(
      params, host_allocator, &executable_data, &executable_data_size);
  if (hrx_status_is_ok(status)) {
    status =
        hrx_executable_load_data(device, executable_data, executable_data_size,
                                 HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY,
                                 HRX_AMDXDNA_EXECUTABLE_TARGET_KEY, executable);
  }
  hrx_host_allocator_free(host_allocator, executable_data);
  return status;
}

hrx_status_t hrx_amdxdna_executable_cache_transaction(
    hrx_executable_t executable, uint32_t export_ordinal,
    hrx_const_byte_span_t transaction, uint64_t* out_variant) {
  if (!executable || !out_variant ||
      (transaction.data_length != 0 && !transaction.data)) {
    return hrx_make_status(
        HRX_STATUS_INVALID_ARGUMENT,
        "executable, transaction, or out_variant is invalid");
  }
  *out_variant = 0;
#if !defined(HRX_HAS_IREE_AMDXDNA_DRIVER)
  (void)export_ordinal;
  (void)transaction;
  return hrx_make_status(HRX_STATUS_UNIMPLEMENTED,
                         "amdxdna driver not enabled");
#else
  return hrx_status_from_iree(iree_hal_amdxdna_executable_cache_transaction(
      executable->hal_executable, export_ordinal,
      iree_make_const_byte_span(transaction.data, transaction.data_length),
      out_variant));
#endif  // HRX_HAS_IREE_AMDXDNA_DRIVER
}
