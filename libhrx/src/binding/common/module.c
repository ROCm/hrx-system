// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "common/fat_binary.h"
#include "common/internal.h"
#include "iree/hal/buffer_transfer.h"
#include "iree/io/file_handle.h"

//===----------------------------------------------------------------------===//
// AMDGPU ELF symbol scanning
//===----------------------------------------------------------------------===//

#define HRX_MODULE_ELF_MAGIC_INT 0x464c457fu  // 0x7f 'E' 'L' 'F'
#define HRX_MODULE_ELFCLASS64 2
#define HRX_MODULE_ELFDATA2LSB 1
#define HRX_MODULE_SHT_SYMTAB 2
#define HRX_MODULE_SHT_DYNSYM 11
#define HRX_MODULE_STB_GLOBAL 1
#define HRX_MODULE_STB_WEAK 2
#define HRX_MODULE_STT_OBJECT 1
#define HRX_MODULE_SHN_UNDEF 0

typedef struct hrx_module_elf64_header_t {
  // ELF ident bytes.
  uint8_t magic[4];
  // ELF class; HRX only accepts ELFCLASS64 here.
  uint8_t elf_class;
  // ELF data encoding; HRX only accepts little-endian here.
  uint8_t elf_data;
  // ELF ident version byte.
  uint8_t elf_version;
  // ELF OS ABI byte.
  uint8_t osabi;
  // ELF ABI version byte.
  uint8_t abiversion;
  // Remaining ELF ident padding.
  uint8_t padding[7];
  // ELF object type.
  uint16_t type;
  // ELF target machine.
  uint16_t machine;
  // ELF object version.
  uint32_t version;
  // Entry point address.
  uint64_t entry;
  // Program header table file offset.
  uint64_t phoff;
  // Section header table file offset.
  uint64_t shoff;
  // Target-specific ELF flags.
  uint32_t flags;
  // ELF header byte size.
  uint16_t ehsize;
  // Program header entry byte size.
  uint16_t phentsize;
  // Program header entry count.
  uint16_t phnum;
  // Section header entry byte size.
  uint16_t shentsize;
  // Section header entry count.
  uint16_t shnum;
  // Section-name string table section index.
  uint16_t shstrndx;
} hrx_module_elf64_header_t;
static_assert(sizeof(hrx_module_elf64_header_t) == 64,
              "ELF64 header must be 64 bytes");

typedef struct hrx_module_elf64_section_header_t {
  // Section name string-table offset.
  uint32_t name;
  // Section type.
  uint32_t type;
  // Section flags.
  uint64_t flags;
  // Section virtual address.
  uint64_t address;
  // Section file offset.
  uint64_t offset;
  // Section byte length.
  uint64_t size;
  // Section-specific linked section index.
  uint32_t link;
  // Section-specific extra info.
  uint32_t info;
  // Section alignment.
  uint64_t address_alignment;
  // Entry byte size for table sections.
  uint64_t entry_size;
} hrx_module_elf64_section_header_t;
static_assert(sizeof(hrx_module_elf64_section_header_t) == 64,
              "ELF64 section header must be 64 bytes");

typedef struct hrx_module_elf64_symbol_t {
  // Symbol name string-table offset.
  uint32_t name;
  // Symbol binding/type byte.
  uint8_t info;
  // Symbol visibility byte.
  uint8_t other;
  // Defining section index.
  uint16_t section_index;
  // Symbol value.
  uint64_t value;
  // Symbol byte size.
  uint64_t size;
} hrx_module_elf64_symbol_t;
static_assert(sizeof(hrx_module_elf64_symbol_t) == 24,
              "ELF64 symbol must be 24 bytes");

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
      (iree_hal_streaming_parameter_op_t*)(buffer + symbols_size);

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

  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_initialize_runtime_metadata(module);
  }

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
  iree_allocator_free(module->host_allocator, module->printf_formats);

  // Release cached executable globals while both the context pointer map and
  // executable-owned global buffers are still live.
  for (iree_host_size_t i = 0; i < module->global_count; ++i) {
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

static iree_status_t iree_hal_streaming_module_managed_device_name(
    iree_hal_streaming_module_t* module, const char* device_name,
    char** out_managed_device_name) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(device_name);
  IREE_ASSERT_ARGUMENT(out_managed_device_name);
  *out_managed_device_name = NULL;

  static const char suffix[] = ".managed";
  const iree_host_size_t name_length = (iree_host_size_t)strlen(device_name);
  iree_host_size_t name_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(name_length, sizeof(suffix),
                                                &name_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed symbol name size overflow");
  }

  char* managed_device_name = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(module->host_allocator, name_size,
                                             (void**)&managed_device_name));
  memcpy(managed_device_name, device_name, name_length);
  memcpy(managed_device_name + name_length, suffix, sizeof(suffix));
  *out_managed_device_name = managed_device_name;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_managed_pointer(
    iree_hal_streaming_module_t* module,
    iree_hal_streaming_symbol_t* pointer_symbol,
    iree_hal_streaming_symbol_t* storage_symbol, const char* device_name) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(pointer_symbol);
  IREE_ASSERT_ARGUMENT(storage_symbol);
  IREE_ASSERT_ARGUMENT(device_name);

  if (!pointer_symbol->global_buffer ||
      !pointer_symbol->global_buffer->buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "managed pointer `%s` has no HAL buffer",
                            device_name);
  }
  if (pointer_symbol->size_bytes < sizeof(storage_symbol->device_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "managed pointer `%s` is too small (%" PRIu64
                            " bytes)",
                            device_name, (uint64_t)pointer_symbol->size_bytes);
  }

  const iree_hal_streaming_deviceptr_t storage_device_address =
      storage_symbol->device_address;
  return iree_hal_device_transfer_h2d(
      module->context->device, &storage_device_address,
      pointer_symbol->global_buffer->buffer, /*target_offset=*/0,
      sizeof(storage_device_address), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout());
}

static iree_status_t iree_hal_streaming_module_elf_subspan(
    iree_const_byte_span_t elf, uint64_t offset64, uint64_t size64,
    const char* kind, iree_const_byte_span_t* out_span) {
  IREE_ASSERT_ARGUMENT(kind);
  IREE_ASSERT_ARGUMENT(out_span);
  memset(out_span, 0, sizeof(*out_span));

  if (offset64 > IREE_HOST_SIZE_MAX || size64 > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF %s range is too large", kind);
  }
  const iree_host_size_t offset = (iree_host_size_t)offset64;
  const iree_host_size_t size = (iree_host_size_t)size64;
  if (offset > elf.data_length || size > elf.data_length - offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF %s range is out of bounds", kind);
  }
  *out_span = iree_make_const_byte_span(elf.data + offset, size);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_elf_cstring_view(
    iree_const_byte_span_t string_table, uint32_t offset,
    iree_string_view_t* out_string) {
  IREE_ASSERT_ARGUMENT(out_string);
  *out_string = iree_string_view_empty();

  if (offset >= string_table.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF string-table offset is out of bounds");
  }
  const char* data = (const char*)string_table.data + offset;
  const iree_host_size_t capacity = string_table.data_length - offset;
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    if (data[i] == '\0') {
      *out_string = iree_make_string_view(data, i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "ELF string-table entry is unterminated");
}

static bool iree_hal_streaming_module_split_managed_name(
    iree_string_view_t name, iree_string_view_t* out_base_name) {
  IREE_ASSERT_ARGUMENT(out_base_name);
  *out_base_name = iree_string_view_empty();

  static const iree_string_view_t suffix = IREE_SVL(".managed");
  if (!iree_string_view_ends_with(name, suffix) || name.size == suffix.size) {
    return false;
  }
  *out_base_name = iree_make_string_view(name.data, name.size - suffix.size);
  return true;
}

static iree_status_t iree_hal_streaming_module_copy_cstring(
    iree_hal_streaming_module_t* module, iree_string_view_t value,
    char** out_string) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_string);
  *out_string = NULL;

  iree_host_size_t size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "managed symbol name size overflow");
  }

  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(module->host_allocator, size, (void**)&string));
  memcpy(string, value.data, value.size);
  string[value.size] = '\0';
  *out_string = string;
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_managed_symbol_pair(
    iree_hal_streaming_module_t* module, iree_hal_executable_t* executable,
    iree_string_view_t pointer_name, iree_string_view_t storage_name) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(executable);

  bool pointer_found = false;
  iree_hal_streaming_symbol_t* pointer_symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
          module, executable, pointer_name, &pointer_found, &pointer_symbol));
  if (!pointer_found || !pointer_symbol) return iree_ok_status();

  bool storage_found = false;
  iree_hal_streaming_symbol_t* storage_symbol = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_module_try_lookup_global_symbol_for_executable(
          module, executable, storage_name, &storage_found, &storage_symbol));
  if (!storage_found || !storage_symbol) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "managed storage `%.*s` was declared but not "
                            "resolved by the executable",
                            (int)storage_name.size, storage_name.data);
  }

  char* pointer_name_string = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_copy_cstring(
      module, pointer_name, &pointer_name_string));
  iree_status_t status = iree_hal_streaming_module_initialize_managed_pointer(
      module, pointer_symbol, storage_symbol, pointer_name_string);
  iree_allocator_free(module->host_allocator, pointer_name_string);
  return status;
}

static iree_status_t
iree_hal_streaming_module_initialize_managed_globals_from_elf(
    iree_hal_streaming_module_t* module, iree_hal_executable_t* executable,
    iree_const_byte_span_t elf) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(executable);

  if (elf.data_length < sizeof(hrx_module_elf64_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF data too small for managed-symbol scan");
  }

  hrx_module_elf64_header_t header;
  memcpy(&header, elf.data, sizeof(header));
  uint32_t magic = 0;
  memcpy(&magic, header.magic, sizeof(magic));
  if (magic != HRX_MODULE_ELF_MAGIC_INT ||
      header.elf_class != HRX_MODULE_ELFCLASS64 ||
      header.elf_data != HRX_MODULE_ELFDATA2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ELF for managed-symbol scan");
  }
  if (header.shentsize != sizeof(hrx_module_elf64_section_header_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unexpected ELF section-header size for managed-symbol scan");
  }

  iree_host_size_t section_headers_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)header.shentsize,
                                  (iree_host_size_t)header.shnum,
                                  &section_headers_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section-header table size overflow");
  }

  iree_const_byte_span_t section_headers_span;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_elf_subspan(
      elf, header.shoff, section_headers_size, "section-header table",
      &section_headers_span));
  const hrx_module_elf64_section_header_t* section_headers =
      (const hrx_module_elf64_section_header_t*)section_headers_span.data;

  for (uint16_t section_index = 0; section_index < header.shnum;
       ++section_index) {
    const hrx_module_elf64_section_header_t* symbol_table =
        &section_headers[section_index];
    if (symbol_table->type != HRX_MODULE_SHT_SYMTAB &&
        symbol_table->type != HRX_MODULE_SHT_DYNSYM) {
      continue;
    }
    if (symbol_table->entry_size != sizeof(hrx_module_elf64_symbol_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unexpected ELF symbol-table entry size for managed-symbol scan");
    }
    if (symbol_table->size % symbol_table->entry_size != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "ELF symbol-table size is not a multiple of entry size");
    }
    if (symbol_table->link >= header.shnum) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "ELF symbol-table string-table link is invalid");
    }

    iree_const_byte_span_t symbol_table_span;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_module_elf_subspan(
        elf, symbol_table->offset, symbol_table->size, "symbol table",
        &symbol_table_span));
    const hrx_module_elf64_symbol_t* symbols =
        (const hrx_module_elf64_symbol_t*)symbol_table_span.data;

    const hrx_module_elf64_section_header_t* string_table =
        &section_headers[symbol_table->link];
    iree_const_byte_span_t string_table_span;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_module_elf_subspan(
        elf, string_table->offset, string_table->size, "string table",
        &string_table_span));

    const iree_host_size_t symbol_count =
        (iree_host_size_t)(symbol_table->size / symbol_table->entry_size);
    for (iree_host_size_t i = 0; i < symbol_count; ++i) {
      const hrx_module_elf64_symbol_t* symbol = &symbols[i];
      const uint8_t binding = symbol->info >> 4;
      const uint8_t type = symbol->info & 0x0F;
      if ((binding != HRX_MODULE_STB_GLOBAL &&
           binding != HRX_MODULE_STB_WEAK) ||
          type != HRX_MODULE_STT_OBJECT ||
          symbol->section_index == HRX_MODULE_SHN_UNDEF) {
        continue;
      }

      iree_string_view_t storage_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(iree_hal_streaming_module_elf_cstring_view(
          string_table_span, symbol->name, &storage_name));
      iree_string_view_t pointer_name = iree_string_view_empty();
      if (!iree_hal_streaming_module_split_managed_name(storage_name,
                                                        &pointer_name)) {
        continue;
      }

      IREE_RETURN_IF_ERROR(
          iree_hal_streaming_module_initialize_managed_symbol_pair(
              module, executable, pointer_name, storage_name));
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_managed_globals(
    iree_hal_streaming_module_t* module,
    const iree_hal_streaming_fat_binary_extract_t* fat_extract) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fat_extract);

  if (fat_extract->match_count == 0) return iree_ok_status();
  if (module->executables &&
      module->executable_count != fat_extract->match_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "module executable count does not match extracted ELF count");
  }

  for (iree_host_size_t i = 0; i < fat_extract->match_count; ++i) {
    iree_hal_executable_t* executable =
        module->executables ? module->executables[i] : module->executable;
    IREE_RETURN_IF_ERROR(
        iree_hal_streaming_module_initialize_managed_globals_from_elf(
            module, executable, fat_extract->matches[i].data));
  }

  return iree_ok_status();
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

  bool found = false;
  iree_hal_streaming_symbol_t* symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_try_lookup_global_symbol(
      module, name, &found, &symbol));

  char* managed_device_name = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_module_managed_device_name(
      module, name, &managed_device_name));
  bool managed_found = false;
  iree_hal_streaming_symbol_t* managed_symbol = NULL;
  iree_status_t status = iree_hal_streaming_module_try_lookup_global_symbol(
      module, managed_device_name, &managed_found, &managed_symbol);
  iree_allocator_free(module->host_allocator, managed_device_name);
  IREE_RETURN_IF_ERROR(status);

  if (managed_found && managed_symbol) {
    if (found && symbol) {
      IREE_RETURN_IF_ERROR(iree_hal_streaming_module_initialize_managed_pointer(
          module, symbol, managed_symbol, name));
    }
    symbol = managed_symbol;
    found = true;
  }
  if (!found || !symbol) {
    iree_string_view_t name_view =
        iree_string_view_trim(iree_make_cstring_view(name));
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "global '%.*s' not found in module",
                            (int)name_view.size, name_view.data);
  }

  *out_device_ptr = symbol->device_address;
  if (out_size) *out_size = symbol->size_bytes;
  return iree_ok_status();
}
