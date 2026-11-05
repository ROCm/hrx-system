// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>

#include "libamdf/tool/amdf-info/report.h"

static const amdf_info_name_t amdf_info_engine_names[] = {
    {AMDF_ENGINE_KIND_UNKNOWN, "unknown"},
    {AMDF_ENGINE_KIND_GPU, "gpu"},
    {AMDF_ENGINE_KIND_XDNA, "xdna"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_endpoint_flags[] = {
    {AMDF_ENDPOINT_TYPE_FLAG_DISPLAY_SUPPORTED, "display"},
    {AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED, "render"},
    {AMDF_ENDPOINT_TYPE_FLAG_COMPUTE_ONLY, "compute_only"},
    {AMDF_ENDPOINT_TYPE_FLAG_SOFTWARE_DEVICE, "software"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_native_names[] = {
    {AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_NONE, "none"},
    {AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE, "linux_device"},
    {AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER, "windows_adapter"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_gpu_architectures[] = {
    {AMDF_GPU_ARCHITECTURE_UNKNOWN, "unknown"},
    {AMDF_GPU_ARCHITECTURE_GCN, "GCN"},
    {AMDF_GPU_ARCHITECTURE_RDNA, "RDNA"},
    {AMDF_GPU_ARCHITECTURE_CDNA, "CDNA"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_xdna_architectures[] = {
    {AMDF_XDNA_ARCHITECTURE_UNKNOWN, "unknown"},
    {AMDF_XDNA_ARCHITECTURE_AIE2, "AIE2"},
    {AMDF_XDNA_ARCHITECTURE_AIE2P, "AIE2P"},
    {AMDF_XDNA_ARCHITECTURE_AIE4, "AIE4"},
    {0, NULL},
};

void amdf_info_report_endpoints_begin(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "endpoints", '[');
  }
}

void amdf_info_report_endpoint_begin(amdf_info_output_t* output,
                                     const amdf_info_endpoint_t* endpoint) {
  char id[33];
  snprintf(id, sizeof(id), "%016" PRIx64 "%016" PRIx64,
           endpoint->summary.id.words[0], endpoint->summary.id.words[1]);
  char uri[48];
  amdf_info_selection_uri(endpoint, uri);
  char display_selector[48];
  if (endpoint->family_ordinal != UINT32_MAX) {
    snprintf(display_selector, sizeof(display_selector), "%s://%" PRIu32,
             amdf_info_endpoint_scheme(endpoint), endpoint->family_ordinal);
  } else {
    snprintf(display_selector, sizeof(display_selector), "%s",
             uri[0] ? uri : id);
  }
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, NULL, '{');
    amdf_info_json_string(output, "id", id);
    amdf_info_json_string(output, "uri", uri[0] ? uri : NULL);
    amdf_info_json_string(output, "name", endpoint->summary.name);
    amdf_info_json_u32(output, "enumeration_ordinal",
                       endpoint->enumeration_ordinal);
    if (endpoint->family_ordinal != UINT32_MAX) {
      amdf_info_json_u32(output, "family_ordinal", endpoint->family_ordinal);
    } else {
      amdf_info_json_string(output, "family_ordinal", NULL);
    }
    amdf_info_json_string(output, "family",
                          amdf_info_endpoint_scheme(endpoint));
    amdf_info_json_enum(output, "engine", endpoint->summary.engine_kind,
                        amdf_info_engine_names);
    amdf_info_json_flags(output, "type_flags", endpoint->summary.type_flags,
                         amdf_info_endpoint_flags);
  } else {
    amdf_info_output_print(output, "\n");
    amdf_info_output_heading(output, uri[0] ? display_selector : id);
    amdf_info_output_print(output, "  ");
    amdf_info_output_text(output, endpoint->summary.name);
    amdf_info_output_print(output, "\n");
    if (output->details && uri[0]) {
      amdf_info_output_print(output, "  Selector    %s\n", uri);
    }
  }
}

void amdf_info_report_endpoint_identity(amdf_info_output_t* output,
                                        const amdf_endpoint_info_t* info) {
  const amdf_endpoint_native_identity_t* native = &info->native_identity;
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "native_identity", '{');
    amdf_info_json_enum(output, "type", native->type, amdf_info_native_names);
    switch (native->type) {
      case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE:
        amdf_info_json_u32(output, "major", native->value.linux_device.major);
        amdf_info_json_u32(output, "minor", native->value.linux_device.minor);
        break;
      case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER:
        amdf_info_json_hex(output, "luid", native->value.windows_adapter.luid);
        amdf_info_json_u32(
            output, "physical_adapter_index",
            native->value.windows_adapter.physical_adapter_index);
        break;
      default:
        break;
    }
    amdf_info_json_end(output, '}');
    amdf_info_json_begin(output, "pci", '{');
    amdf_info_json_u32(output, "vendor_id", info->pci.vendor_id);
    amdf_info_json_u32(output, "device_id", info->pci.device_id);
    amdf_info_json_u32(output, "subsystem_vendor_id",
                       info->pci.subsystem_vendor_id);
    amdf_info_json_u32(output, "subsystem_device_id",
                       info->pci.subsystem_device_id);
    amdf_info_json_u32(output, "revision_id", info->pci.revision_id);
    amdf_info_json_end(output, '}');
  } else if (output->details) {
    switch (native->type) {
      case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE:
        amdf_info_output_print(
            output,
            "  Native      Linux character device %" PRIu32 ":%" PRIu32 "\n",
            native->value.linux_device.major, native->value.linux_device.minor);
        break;
      case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER:
        amdf_info_output_print(
            output,
            "  Native      Windows LUID %016" PRIx64
            ", physical adapter %" PRIu32 "\n",
            native->value.windows_adapter.luid,
            native->value.windows_adapter.physical_adapter_index);
        break;
      default:
        amdf_info_output_print(
            output, "  Native      unavailable (namespace %" PRIu32 ")\n",
            native->type);
        break;
    }
    amdf_info_output_print(
        output,
        "  PCI IDs     %04" PRIx32 ":%04" PRIx32 ", revision %02" PRIx32
        "; subsystem %04" PRIx32 ":%04" PRIx32 "\n",
        info->pci.vendor_id, info->pci.device_id, info->pci.revision_id,
        info->pci.subsystem_vendor_id, info->pci.subsystem_device_id);
  }
}

void amdf_info_report_gpu_target(amdf_info_output_t* output,
                                 const amdf_gpu_endpoint_info_t* info) {
  const amdf_gpu_architecture_t architecture =
      amdf_gpu_endpoint_architecture(info);
  const char* architecture_name =
      amdf_info_name(architecture, amdf_info_gpu_architectures);
  char target[AMDF_XDNA_TARGET_ID_CAPACITY];
  amdf_info_gpu_target(info, target);
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "gpu", '{');
    amdf_info_json_string(output, "target", target);
    amdf_info_json_enum(output, "architecture", architecture,
                        amdf_info_gpu_architectures);
    amdf_info_json_begin(output, "gfx_ip", '{');
    amdf_info_json_u32(output, "major", info->gfx_ip.major);
    amdf_info_json_u32(output, "minor", info->gfx_ip.minor);
    amdf_info_json_u32(output, "stepping", info->gfx_ip.stepping);
    amdf_info_json_end(output, '}');
    amdf_info_json_u32(output, "asic_revision", info->asic_revision);
    amdf_info_json_hex(output, "unique_id", info->unique_id);
    amdf_info_json_begin(output, "compute", '{');
    amdf_info_json_u32(output, "wavefront_size", info->compute.wavefront_size);
    amdf_info_json_u32(output, "compute_unit_count",
                       info->compute.compute_unit_count);
    amdf_info_json_u32(output, "maximum_wave_count_per_compute_unit",
                       info->compute.maximum_wave_count_per_compute_unit);
    amdf_info_json_u32(
        output, "maximum_scratch_wave_count_per_compute_unit",
        info->compute.maximum_scratch_wave_count_per_compute_unit);
    amdf_info_json_u64(output, "local_data_share_byte_length",
                       info->compute.local_data_share_byte_length);
    amdf_info_json_end(output, '}');
    amdf_info_json_begin(output, "topology", '{');
    amdf_info_json_u32(output, "xcc_count", info->topology.xcc_count);
    amdf_info_json_u32(output, "shader_engine_count_per_xcc",
                       info->topology.shader_engine_count_per_xcc);
    amdf_info_json_end(output, '}');
    amdf_info_json_begin(output, "execution_resources", '{');
    amdf_info_json_u32(output, "compute_units_per_resource",
                       info->execution_resources.compute_units_per_resource);
    amdf_info_json_u32(output, "group_count",
                       info->execution_resources.group_count);
    amdf_info_json_u32(
        output, "minimum_selected_resources_per_group",
        info->execution_resources.minimum_selected_resources_per_group);
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
  } else {
    amdf_info_output_print(
        output, "  Target      %s (%s), %" PRIu32 " compute units\n", target,
        architecture_name ? architecture_name : "unknown",
        info->compute.compute_unit_count);
    if (output->details) {
      amdf_info_output_print(
          output,
          "  Revision    %" PRIu32
          " (ASIC target)\n"
          "  Compute     %" PRIu32 " lanes/wave; %" PRIu32 " waves/CU, %" PRIu32
          " scratch waves/CU\n"
          "  LDS         %" PRIu64
          " bytes per compute unit\n"
          "  Topology    %" PRIu32 " XCCs, %" PRIu32
          " shader engines per XCC\n",
          info->asic_revision, info->compute.wavefront_size,
          info->compute.maximum_wave_count_per_compute_unit,
          info->compute.maximum_scratch_wave_count_per_compute_unit,
          info->compute.local_data_share_byte_length, info->topology.xcc_count,
          info->topology.shader_engine_count_per_xcc);
      if (info->execution_resources.group_count) {
        amdf_info_output_print(
            output,
            "  Selection   %" PRIu32 " CUs/resource; %" PRIu32
            " groups, minimum %" PRIu32 " resources/group\n",
            info->execution_resources.compute_units_per_resource,
            info->execution_resources.group_count,
            info->execution_resources.minimum_selected_resources_per_group);
      }
      if (info->unique_id) {
        amdf_info_output_print(output, "  GPU ID      GPU-%016" PRIx64 "\n",
                               info->unique_id);
      }
    }
  }
}

void amdf_info_report_xdna_target(amdf_info_output_t* output,
                                  const amdf_xdna_endpoint_info_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "xdna", '{');
    amdf_info_json_string(output, "target", info->target_id);
    amdf_info_json_enum(output, "architecture", info->architecture,
                        amdf_info_xdna_architectures);
    amdf_info_json_end(output, '}');
  } else {
    const char* architecture =
        amdf_info_name(info->architecture, amdf_info_xdna_architectures);
    amdf_info_output_print(output, "  Target      ");
    amdf_info_output_text(output, info->target_id);
    amdf_info_output_print(output, " (%s)\n",
                           architecture ? architecture : "unknown");
  }
}

void amdf_info_report_pci_address(amdf_info_output_t* output,
                                  const amdf_endpoint_pci_address_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    if (!info->available) {
      amdf_info_json_string(output, "pci_address", NULL);
    } else {
      amdf_info_json_begin(output, "pci_address", '{');
      amdf_info_json_u32(output, "domain", info->domain);
      amdf_info_json_u32(output, "bus", info->bus);
      amdf_info_json_u32(output, "device", info->device);
      amdf_info_json_u32(output, "function", info->function);
      amdf_info_json_end(output, '}');
    }
  } else if (info->available) {
    amdf_info_output_print(
        output,
        "  PCI         %04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIx32 "\n",
        info->domain, info->bus, info->device, info->function);
  } else if (output->details) {
    amdf_info_output_print(output, "  PCI         unavailable\n");
  }
}

void amdf_info_report_host_locality(amdf_info_output_t* output,
                                    const amdf_endpoint_host_locality_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    if (info->numa_node == UINT32_MAX) {
      amdf_info_json_string(output, "host_numa_node", NULL);
    } else {
      amdf_info_json_u32(output, "host_numa_node", info->numa_node);
    }
  } else if (info->numa_node != UINT32_MAX) {
    amdf_info_output_print(output, "  Locality    NUMA node %" PRIu32 "\n",
                           info->numa_node);
  } else if (output->details) {
    amdf_info_output_print(output, "  Locality    unavailable\n");
  }
}

void amdf_info_report_endpoint_end(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, '}');
  }
}

void amdf_info_report_endpoints_end(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, ']');
  }
}
