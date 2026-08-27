// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/cpu_data.h"

typedef struct iree_cpu_feature_record_t {
  iree_cpu_architecture_t architecture;
  uint32_t field_index;
  uint64_t field_bit;
  iree_string_view_t name;
} iree_cpu_feature_record_t;

typedef struct iree_cpu_architecture_record_t {
  iree_cpu_architecture_t architecture;
  iree_string_view_t name;
} iree_cpu_architecture_record_t;

static const iree_cpu_architecture_record_t iree_cpu_architecture_records[] = {
    {IREE_CPU_ARCHITECTURE_ARM_32, IREE_SVL("arm_32")},
    {IREE_CPU_ARCHITECTURE_ARM_64, IREE_SVL("arm_64")},
    {IREE_CPU_ARCHITECTURE_RISCV_32, IREE_SVL("riscv_32")},
    {IREE_CPU_ARCHITECTURE_RISCV_64, IREE_SVL("riscv_64")},
    {IREE_CPU_ARCHITECTURE_WASM_32, IREE_SVL("wasm_32")},
    {IREE_CPU_ARCHITECTURE_WASM_64, IREE_SVL("wasm_64")},
    {IREE_CPU_ARCHITECTURE_X86_32, IREE_SVL("x86_32")},
    {IREE_CPU_ARCHITECTURE_X86_64, IREE_SVL("x86_64")},
};

static const iree_cpu_feature_record_t iree_cpu_feature_records[] = {
#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  {IREE_CPU_ARCHITECTURE_##arch, field_index, UINT64_C(1) << bit_pos,         \
   IREE_SVL(llvm_name)},
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
};

static const iree_cpu_feature_record_t* iree_cpu_feature_record_find(
    iree_cpu_architecture_t architecture, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_cpu_feature_records);
       ++i) {
    const iree_cpu_feature_record_t* record = &iree_cpu_feature_records[i];
    if (record->architecture == architecture &&
        iree_string_view_equal(record->name, name)) {
      return record;
    }
  }
  return NULL;
}

IREE_API_EXPORT iree_cpu_architecture_t iree_cpu_architecture_host(void) {
#if defined(IREE_ARCH_ARM_32)
  return IREE_CPU_ARCHITECTURE_ARM_32;
#elif defined(IREE_ARCH_ARM_64)
  return IREE_CPU_ARCHITECTURE_ARM_64;
#elif defined(IREE_ARCH_RISCV_32)
  return IREE_CPU_ARCHITECTURE_RISCV_32;
#elif defined(IREE_ARCH_RISCV_64)
  return IREE_CPU_ARCHITECTURE_RISCV_64;
#elif defined(IREE_ARCH_WASM_32)
  return IREE_CPU_ARCHITECTURE_WASM_32;
#elif defined(IREE_ARCH_WASM_64)
  return IREE_CPU_ARCHITECTURE_WASM_64;
#elif defined(IREE_ARCH_X86_32)
  return IREE_CPU_ARCHITECTURE_X86_32;
#elif defined(IREE_ARCH_X86_64)
  return IREE_CPU_ARCHITECTURE_X86_64;
#else
#error Unsupported CPU architecture.
#endif
}

IREE_API_EXPORT iree_string_view_t
iree_cpu_architecture_name(iree_cpu_architecture_t architecture) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_cpu_architecture_records); ++i) {
    if (iree_cpu_architecture_records[i].architecture == architecture) {
      return iree_cpu_architecture_records[i].name;
    }
  }
  return iree_string_view_empty();
}

IREE_API_EXPORT bool iree_cpu_architecture_parse(
    iree_string_view_t name, iree_cpu_architecture_t* out_architecture) {
  IREE_ASSERT_ARGUMENT(out_architecture);
  *out_architecture = IREE_CPU_ARCHITECTURE_UNKNOWN;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_cpu_architecture_records); ++i) {
    if (iree_string_view_equal(name, iree_cpu_architecture_records[i].name)) {
      *out_architecture = iree_cpu_architecture_records[i].architecture;
      return true;
    }
  }
  return false;
}

IREE_API_EXPORT iree_host_size_t
iree_cpu_feature_count(iree_cpu_architecture_t architecture) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_cpu_feature_records);
       ++i) {
    if (iree_cpu_feature_records[i].architecture == architecture) ++count;
  }
  return count;
}

IREE_API_EXPORT iree_string_view_t iree_cpu_feature_name(
    iree_cpu_architecture_t architecture, iree_host_size_t ordinal) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_cpu_feature_records);
       ++i) {
    const iree_cpu_feature_record_t* record = &iree_cpu_feature_records[i];
    if (record->architecture != architecture) continue;
    if (ordinal-- == 0) return record->name;
  }
  return iree_string_view_empty();
}

IREE_API_EXPORT iree_cpu_feature_availability_t iree_cpu_data_query_feature(
    const iree_cpu_data_t* cpu_data, iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(cpu_data);
  const iree_cpu_feature_record_t* record =
      iree_cpu_feature_record_find(cpu_data->architecture, name);
  if (!record) return IREE_CPU_FEATURE_AVAILABILITY_UNKNOWN;
  return iree_all_bits_set(cpu_data->fields[record->field_index],
                           record->field_bit)
             ? IREE_CPU_FEATURE_AVAILABILITY_AVAILABLE
             : IREE_CPU_FEATURE_AVAILABILITY_UNAVAILABLE;
}

IREE_API_EXPORT bool iree_cpu_data_satisfies_features(
    const iree_cpu_data_t* available, const iree_cpu_data_t* required) {
  IREE_ASSERT_ARGUMENT(available);
  IREE_ASSERT_ARGUMENT(required);
  if (available->architecture == IREE_CPU_ARCHITECTURE_UNKNOWN ||
      available->architecture != required->architecture) {
    return false;
  }
  uint64_t feature_masks[IREE_CPU_DATA_FIELD_COUNT] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_cpu_feature_records);
       ++i) {
    const iree_cpu_feature_record_t* record = &iree_cpu_feature_records[i];
    if (record->architecture == required->architecture) {
      feature_masks[record->field_index] |= record->field_bit;
    }
  }
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    if (iree_any_bit_set(required->fields[i], ~feature_masks[i]) ||
        iree_any_bit_set(required->fields[i], ~available->fields[i])) {
      return false;
    }
  }
  return true;
}

IREE_API_EXPORT iree_status_t iree_cpu_data_append_target_key(
    const iree_cpu_data_t* cpu_data, iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(cpu_data);
  IREE_ASSERT_ARGUMENT(builder);
  const iree_string_view_t architecture_name =
      iree_cpu_architecture_name(cpu_data->architecture);
  if (iree_string_view_is_empty(architecture_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU data has an invalid architecture %" PRIu32,
                            cpu_data->architecture);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, architecture_name));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_cpu_feature_records);
       ++i) {
    const iree_cpu_feature_record_t* record = &iree_cpu_feature_records[i];
    if (record->architecture != cpu_data->architecture ||
        !iree_all_bits_set(cpu_data->fields[record->field_index],
                           record->field_bit)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":+"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, record->name));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_cpu_data_parse_target_key(
    iree_string_view_t target_key, iree_cpu_data_t* out_cpu_data) {
  IREE_ASSERT_ARGUMENT(out_cpu_data);
  memset(out_cpu_data, 0, sizeof(*out_cpu_data));

  iree_string_view_t architecture_name = target_key;
  iree_string_view_t feature_list = iree_string_view_empty();
  if (iree_string_view_split(target_key, ':', &architecture_name,
                             &feature_list) != -1 &&
      (iree_string_view_is_empty(feature_list) ||
       target_key.data[target_key.size - 1] == ':')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CPU target key has an empty feature suffix");
  }
  if (!iree_cpu_architecture_parse(architecture_name,
                                   &out_cpu_data->architecture)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "unknown CPU target architecture '%.*s'",
        (int)architecture_name.size, architecture_name.data);
  }

  while (!iree_string_view_is_empty(feature_list)) {
    iree_string_view_t feature = feature_list;
    iree_string_view_t remaining_features = iree_string_view_empty();
    if (iree_string_view_split(feature_list, ':', &feature,
                               &remaining_features) == -1) {
      feature = feature_list;
    }
    const iree_string_view_t feature_suffix = feature;
    if (!iree_string_view_consume_prefix(&feature, IREE_SV("+")) ||
        iree_string_view_is_empty(feature)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "CPU target feature suffix must have the form '+feature': %.*s",
          (int)feature_suffix.size, feature_suffix.data);
    }
    const iree_cpu_feature_record_t* record =
        iree_cpu_feature_record_find(out_cpu_data->architecture, feature);
    if (!record) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown CPU target feature '%.*s' for %.*s", (int)feature.size,
          feature.data, (int)architecture_name.size, architecture_name.data);
    }
    if (iree_any_bit_set(out_cpu_data->fields[record->field_index],
                         record->field_bit)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate CPU target feature '%.*s'",
                              (int)feature.size, feature.data);
    }
    out_cpu_data->fields[record->field_index] |= record->field_bit;
    feature_list = remaining_features;
  }
  return iree_ok_status();
}
