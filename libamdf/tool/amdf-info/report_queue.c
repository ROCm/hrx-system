// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "libamdf/tool/amdf-info/report.h"

static const amdf_info_name_t amdf_info_commands[] = {
    {AMDF_QUEUE_COMMAND_TYPE_UNKNOWN, "unknown"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, "PM4"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, "SDMA"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, "AQL"},
    {AMDF_QUEUE_COMMAND_TYPE_XDNA, "XDNA"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_AQL_METADATA, "AQL metadata"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_publication_modes[] = {
    {AMDF_QUEUE_PUBLICATION_MODE_USER, "user"},
    {AMDF_QUEUE_PUBLICATION_MODE_KERNEL, "kernel"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_roles[] = {
    {AMDF_QUEUE_ROLE_COMPUTE, "compute"},
    {AMDF_QUEUE_ROLE_TRANSFER, "transfer"},
    {AMDF_QUEUE_ROLE_ATOMIC, "atomic"},
    {AMDF_QUEUE_ROLE_CACHE_CONTROL, "cache_control"},
    {AMDF_QUEUE_ROLE_VIRTUAL_MEMORY, "virtual_memory"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_cache_operations[] = {
    {AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM, "release_to_system"},
    {AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM, "acquire_from_system"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_cache_kinds[] = {
    {AMDF_CACHE_TRANSITION_KINDS_RANGE, "range"},
    {AMDF_CACHE_TRANSITION_KINDS_GLOBAL, "global"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_user_capabilities[] = {
    {AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, "host_producer"},
    {AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER, "device_producer"},
    {AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH, "cooperative_dispatch"},
    {AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION,
     "execution_resource_selection"},
    {AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION,
     "host_non_temporal_publication"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_kernel_capabilities[] = {
    {AMDF_KERNEL_QUEUE_CAPABILITY_VECTOR_SUBMIT, "vector_submit"},
    {AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY, "virtual_memory"},
    {AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
     "external_synchronization"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_producers[] = {
    {AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE, "single"},
    {AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI, "multi"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_priorities[] = {
    {AMDF_QUEUE_PRIORITY_CAPABILITY_LOW, "low"},
    {AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL, "normal"},
    {AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH, "high"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_atomic_operations[] = {
    {AMDF_ATOMIC_OPERATION_WAIT, "wait"},
    {AMDF_ATOMIC_OPERATION_STORE, "store"},
    {AMDF_ATOMIC_OPERATION_ADD, "add"},
    {AMDF_ATOMIC_OPERATION_SUBTRACT, "subtract"},
    {AMDF_ATOMIC_OPERATION_AND, "and"},
    {AMDF_ATOMIC_OPERATION_OR, "or"},
    {AMDF_ATOMIC_OPERATION_XOR, "xor"},
    {AMDF_ATOMIC_OPERATION_LOAD, "load"},
    {AMDF_ATOMIC_OPERATION_COMPARE_EXCHANGE, "compare_exchange"},
    {AMDF_ATOMIC_OPERATION_EXCHANGE, "exchange"},
    {AMDF_ATOMIC_OPERATION_FETCH_ADD, "fetch_add"},
    {AMDF_ATOMIC_OPERATION_FETCH_SUBTRACT, "fetch_subtract"},
    {AMDF_ATOMIC_OPERATION_FETCH_AND, "fetch_and"},
    {AMDF_ATOMIC_OPERATION_FETCH_OR, "fetch_or"},
    {AMDF_ATOMIC_OPERATION_FETCH_XOR, "fetch_xor"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_atomic_conditions[] = {
    {AMDF_ATOMIC_WAIT_CONDITION_EQUAL, "equal"},
    {AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL, "not_equal"},
    {AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
     "unsigned_greater_equal"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_placements[] = {
    {AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL, "host_local"},
    {AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL, "device_local"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_host_publication[] = {
    {AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN, "known"},
    {AMDF_QUEUE_HOST_PUBLICATION_FLAG_DEVICE_READ, "device_read"},
    {AMDF_QUEUE_HOST_PUBLICATION_FLAG_KERNEL_CALL, "kernel_call"},
    {0, NULL},
};

static void amdf_info_report_queue_flags(amdf_info_output_t* output,
                                         const char* key, const char* label,
                                         uint64_t value,
                                         const amdf_info_name_t* names) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_flags(output, key, value, names);
  } else if (value) {
    amdf_info_output_print(output, "      %-14s  ", label);
    amdf_info_output_flags(output, value, names);
    amdf_info_output_print(output, "\n");
  }
}

static void amdf_info_report_atomics(amdf_info_output_t* output,
                                     const amdf_atomic_capabilities_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "atomic_capabilities", '{');
  }
  amdf_info_report_queue_flags(output, "operations_32", "Atomics32",
                               info->operations_32,
                               amdf_info_atomic_operations);
  amdf_info_report_queue_flags(output, "operations_64", "Atomics64",
                               info->operations_64,
                               amdf_info_atomic_operations);
  amdf_info_report_queue_flags(output, "wait_conditions_32", "Wait32",
                               info->wait_conditions_32,
                               amdf_info_atomic_conditions);
  amdf_info_report_queue_flags(output, "wait_conditions_64", "Wait64",
                               info->wait_conditions_64,
                               amdf_info_atomic_conditions);
  amdf_info_report_queue_flags(
      output, "operations_without_dispatch_32", "No-dispatch32",
      info->operations_without_dispatch_32, amdf_info_atomic_operations);
  amdf_info_report_queue_flags(
      output, "operations_without_dispatch_64", "No-dispatch64",
      info->operations_without_dispatch_64, amdf_info_atomic_operations);
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, '}');
  }
}

void amdf_info_report_queues_begin(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "queue_families", '[');
  }
}

void amdf_info_report_queue(amdf_info_output_t* output,
                            const amdf_queue_family_info_t* info) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, NULL, '{');
    amdf_info_json_u32(output, "ordinal", info->ordinal);
    amdf_info_json_enum(output, "command_type", info->command_type,
                        amdf_info_commands);
    amdf_info_json_u32(output, "format_version", info->format_version);
    amdf_info_json_hex(output, "format_features", info->format_features);
    amdf_info_json_flags(output, "publication_modes", info->publication_modes,
                         amdf_info_publication_modes);
  } else {
    const char* command =
        amdf_info_name(info->command_type, amdf_info_commands);
    amdf_info_output_print(output, "  Queue %-4" PRIu32 "  %s", info->ordinal,
                           command ? command : "unknown");
    if (!command) {
      amdf_info_output_print(output, " (%" PRIu32 ")", info->command_type);
    }
    amdf_info_output_print(output, "; ");
    amdf_info_output_flags(output, info->publication_modes,
                           amdf_info_publication_modes);
    amdf_info_output_print(output, " publication\n");
    if (!output->details) {
      return;
    }
    amdf_info_output_print(output,
                           "      Format          version %" PRIu32
                           ", native features 0x%" PRIx64 "\n",
                           info->format_version, info->format_features);
  }
  amdf_info_report_queue_flags(output, "roles", "Roles", info->roles,
                               amdf_info_roles);
  amdf_info_report_queue_flags(output, "cache_operations", "Cache",
                               info->cache_operations,
                               amdf_info_cache_operations);
  amdf_info_report_queue_flags(output, "cache_transition_kinds", "Cache range",
                               info->cache_transition_kinds,
                               amdf_info_cache_kinds);
  amdf_info_report_queue_flags(output, "user_queue_capabilities", "User queue",
                               info->user_queue_capabilities,
                               amdf_info_user_capabilities);
  amdf_info_report_queue_flags(output, "kernel_queue_capabilities",
                               "Kernel queue", info->kernel_queue_capabilities,
                               amdf_info_kernel_capabilities);
  amdf_info_report_queue_flags(output, "producer_modes", "Producers",
                               info->producer_modes, amdf_info_producers);
  amdf_info_report_queue_flags(output, "priority_capabilities", "Priorities",
                               info->priority_capabilities,
                               amdf_info_priorities);
  amdf_info_report_atomics(output, &info->atomic_capabilities);
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "metadata", '{');
    amdf_info_json_enum(output, "command_type", info->metadata.command_type,
                        amdf_info_commands);
    amdf_info_json_u32(output, "dispatch_version",
                       info->metadata.dispatch_version);
    amdf_info_json_u32(output, "barrier_version",
                       info->metadata.barrier_version);
    amdf_info_json_end(output, '}');
    amdf_info_json_u64(output, "minimum_ring_byte_length",
                       info->minimum_ring_byte_length);
    amdf_info_json_u64(output, "maximum_ring_byte_length",
                       info->maximum_ring_byte_length);
    amdf_info_json_u64(output, "ring_byte_length_alignment",
                       info->ring_byte_length_alignment);
    amdf_info_json_end(output, '}');
  } else {
    if (info->minimum_ring_byte_length || info->maximum_ring_byte_length) {
      amdf_info_output_print(output,
                             "      Ring            %" PRIu64 "..%" PRIu64
                             " bytes, alignment %" PRIu64 " bytes\n",
                             info->minimum_ring_byte_length,
                             info->maximum_ring_byte_length,
                             info->ring_byte_length_alignment);
    }
    if (info->metadata.command_type) {
      const char* metadata =
          amdf_info_name(info->metadata.command_type, amdf_info_commands);
      amdf_info_output_print(
          output,
          "      Metadata        %s (%" PRIu32 "), dispatch v%" PRIu32
          ", barrier v%" PRIu32 "\n",
          metadata ? metadata : "unknown", info->metadata.command_type,
          info->metadata.dispatch_version, info->metadata.barrier_version);
    }
  }
}

void amdf_info_report_queues_end(amdf_info_output_t* output) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_end(output, ']');
  }
}

void amdf_info_report_placement(
    amdf_info_output_t* output,
    const amdf_queue_family_placement_info_t* info) {
  static const char* const placement_names[AMDF_QUEUE_PLACEMENT_COUNT] = {
      "default",
      "host_local",
      "device_local",
  };
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, NULL, '{');
    amdf_info_json_u32(output, "ordinal", info->ordinal);
    amdf_info_json_flags(output, "placement_capabilities",
                         info->placement_capabilities, amdf_info_placements);
    amdf_info_json_begin(output, "host_publication", '{');
    for (uint32_t i = 0; i < AMDF_QUEUE_PLACEMENT_COUNT; ++i) {
      amdf_info_json_flags(output, placement_names[i],
                           info->host_publication[i],
                           amdf_info_host_publication);
    }
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
  } else if (output->details) {
    amdf_info_output_print(output, "  Placement   queue %" PRIu32 ": ",
                           info->ordinal);
    amdf_info_output_flags(output, info->placement_capabilities,
                           amdf_info_placements);
    amdf_info_output_print(output, "\n");
    for (uint32_t i = 0; i < AMDF_QUEUE_PLACEMENT_COUNT; ++i) {
      amdf_info_output_print(output, "      %-14s  ", placement_names[i]);
      if (!(info->host_publication[i] &
            AMDF_QUEUE_HOST_PUBLICATION_FLAG_KNOWN)) {
        amdf_info_output_print(output, "requirements not exposed");
        if (info->host_publication[i]) {
          amdf_info_output_print(output, " (0x%" PRIx32 ")",
                                 info->host_publication[i]);
        }
      } else {
        amdf_info_output_flags(output, info->host_publication[i],
                               amdf_info_host_publication);
      }
      amdf_info_output_print(output, "\n");
    }
  }
}
