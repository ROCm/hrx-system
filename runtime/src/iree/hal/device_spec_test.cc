// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal {
namespace {

static void ExpectStringViewEq(iree_string_view_t actual,
                               const char* expected) {
  EXPECT_TRUE(iree_string_view_equal(actual, iree_make_cstring_view(expected)));
}

static void StoreU32Le(std::vector<uint8_t>* bytes, size_t offset,
                       uint32_t value) {
  (*bytes)[offset + 0] = static_cast<uint8_t>(value >> 0);
  (*bytes)[offset + 1] = static_cast<uint8_t>(value >> 8);
  (*bytes)[offset + 2] = static_cast<uint8_t>(value >> 16);
  (*bytes)[offset + 3] = static_cast<uint8_t>(value >> 24);
}

static void StoreU64Le(std::vector<uint8_t>* bytes, size_t offset,
                       uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    (*bytes)[offset + i] = static_cast<uint8_t>(value >> (i * 8));
  }
}

static std::vector<uint8_t> SerializeEmptySpec() {
  iree_hal_device_spec_t* spec = nullptr;
  IREE_CHECK_OK(
      iree_hal_device_spec_create(nullptr, iree_allocator_system(), &spec));
  iree_byte_span_t bytes = iree_byte_span_empty();
  IREE_CHECK_OK(
      iree_hal_device_spec_serialize(spec, iree_allocator_system(), &bytes));
  std::vector<uint8_t> result(bytes.data, bytes.data + bytes.data_length);
  iree_allocator_free(iree_allocator_system(), bytes.data);
  iree_hal_device_spec_release(spec);
  return result;
}

static void ExpectParseFails(const std::vector<uint8_t>& bytes) {
  iree_hal_device_spec_t* spec = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_parse(
          iree_make_const_byte_span(bytes.data(), bytes.size()),
          iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);
}

static iree_hal_device_spec_params_t MakeTestSpecParams(
    iree_hal_device_identity_spec_t* out_identity,
    iree_hal_physical_device_spec_t* out_physical_devices,
    iree_hal_device_memory_spec_t* out_memory,
    iree_hal_memory_heap_spec_t* out_memory_heaps,
    iree_hal_memory_type_spec_t* out_memory_types,
    iree_hal_device_queue_spec_t* out_queues,
    iree_hal_queue_family_spec_t* out_queue_families,
    iree_hal_device_dispatch_spec_t* out_dispatch,
    iree_hal_device_timing_spec_t* out_timing,
    iree_hal_device_executable_spec_t* out_executables,
    iree_hal_executable_target_t* out_executable_targets,
    iree_hal_device_sanitizer_spec_t* out_sanitizer,
    iree_hal_device_spec_facet_t* out_facets,
    iree_const_byte_span_t facet_payload) {
  out_physical_devices[0] = {
      /*.identity=*/
      {
          /*.display_name=*/iree_make_cstring_view("Test GPU"),
          /*.backend_path=*/iree_make_cstring_view("pci:0000:01:00.0"),
          /*.vendor_id=*/0x1002,
          /*.device_id=*/0x744c,
          /*.revision_id=*/1,
          /*.uuid=*/{{0}},
          /*.pci=*/{0, 1, 0, 0},
          /*.numa=*/{0},
          /*.flags=*/IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS,
      },
      /*.physical_ordinal=*/0,
      /*.partition_ordinal=*/0,
      /*.partition_count=*/1,
      /*.physical_device_affinity=*/1,
  };
  *out_identity = {
      /*.logical_device_id=*/iree_make_cstring_view("test-device-0"),
      /*.display_name=*/iree_make_cstring_view("Test Device"),
      /*.driver_id=*/iree_make_cstring_view("test"),
      /*.driver_version=*/iree_make_cstring_view("1.0"),
      /*.backend_id=*/iree_make_cstring_view("test-backend"),
      /*.device_path=*/iree_make_cstring_view("test://0"),
      /*.vendor_name=*/iree_make_cstring_view("Example"),
      /*.vendor_id=*/0x1002,
      /*.device_id=*/0x744c,
      /*.revision_id=*/1,
      /*.logical_ordinal=*/0,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/out_physical_devices,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };

  out_memory_heaps[0] = {
      /*.name=*/iree_make_cstring_view("device-local"),
      /*.capacity_bytes=*/1024ull * 1024ull * 1024ull,
      /*.allocation_granularity=*/4096,
      /*.allocation_alignment=*/256,
      /*.maximum_allocation_size=*/512ull * 1024ull * 1024ull,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_MEMORY_HEAP_SPEC_FLAG_NONE,
  };
  out_memory_types[0] = {
      /*.heap_index=*/0,
      /*.memory_type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.minimum_alignment=*/256,
      /*.optimal_transfer_granularity=*/4096,
      /*.flags=*/IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
  };
  *out_memory = {
      /*.heap_count=*/1,
      /*.heaps=*/out_memory_heaps,
      /*.memory_type_count=*/1,
      /*.memory_types=*/out_memory_types,
      /*.external_buffer_handle_count=*/0,
      /*.external_buffer_handles=*/NULL,
      /*.flags=*/IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
  };

  out_queue_families[0] = {
      /*.name=*/iree_make_cstring_view("default"),
      /*.queue_count=*/1,
      /*.priority_count=*/1,
      /*.timestamp_valid_bits=*/64,
      /*.timestamp_frequency_hz=*/1000000000ull,
      /*.physical_device_affinity=*/1,
      /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
      /*.flags=*/IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE,
  };
  *out_queues = {
      /*.family_count=*/1,
      /*.families=*/out_queue_families,
      /*.external_timepoint_handle_count=*/0,
      /*.external_timepoint_handles=*/NULL,
      /*.flags=*/IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };

  *out_dispatch = {
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/1024,
          /*.maximum_workgroup_size=*/{1024, 1024, 64},
          /*.maximum_workgroup_count=*/{65535, 65535, 65535},
      },
      /*.subgroup=*/
      {
          /*.default_size=*/64,
          /*.minimum_size=*/32,
          /*.maximum_size=*/64,
          /*.supported_size_mask=*/1ull << 32,
      },
      /*.execution=*/
      {
          /*.unit_count=*/120,
          /*.group_count=*/1,
          /*.maximum_resident_workgroup_count=*/16,
          /*.maximum_resident_invocation_count=*/2048,
          /*.maximum_resident_subgroup_count=*/0,
          /*.maximum_register_count=*/65536,
          /*.maximum_workgroup_register_count=*/65536,
          /*.maximum_local_memory_size=*/64 * 1024,
          /*.maximum_workgroup_local_memory_size=*/64 * 1024,
          /*.maximum_workgroup_local_memory_size_optin=*/64 * 1024,
      },
      /*.addressing=*/
      {
          /*.pointer_size_bits=*/64,
          /*.address_space_bits=*/64,
          /*.minimum_buffer_device_address_alignment=*/0,
      },
      /*.flags=*/IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  *out_timing = {
      /*.timestamp_valid_bits=*/64,
      /*.timestamp_frequency_hz=*/1000000000ull,
      /*.flags=*/IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS,
  };

  out_executable_targets[0] = {
      /*.family=*/iree_make_cstring_view("amdgpu"),
      /*.target_key=*/iree_make_cstring_view("gfx1100:xnack-"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  out_executable_targets[1] = out_executable_targets[0];
  out_executable_targets[1].target_key =
      iree_make_cstring_view("gfx11-generic");
  out_executable_targets[1].kind = IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC;
  out_executable_targets[1].priority = 10;
  out_executable_targets[2] = out_executable_targets[1];
  out_executable_targets[2].target_key =
      iree_make_cstring_view("gfx11-generic-alt");
  *out_executables = {
      /*.target_count=*/3,
      /*.targets=*/out_executable_targets,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };

  *out_sanitizer = {
      /*.flags=*/IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN,
      /*.asan=*/
      {
          /*.pool_options=*/
          {
              /*.mode=*/IREE_HAL_ASAN_POOL_MODE_SHADOW,
              /*.shadow_granule_size=*/8,
              /*.redzone_size=*/64,
              /*.backing_alignment=*/4096,
              /*.quarantine_size=*/1024 * 1024,
          },
      },
  };

  out_facets[0] = {
      /*.schema_id=*/iree_make_cstring_view("test.facet"),
      /*.schema_version=*/1,
      /*.payload=*/facet_payload,
  };

  return {
      /*.identity=*/out_identity,
      /*.memory=*/out_memory,
      /*.virtual_memory=*/NULL,
      /*.queues=*/out_queues,
      /*.dispatch=*/out_dispatch,
      /*.timing=*/out_timing,
      /*.executables=*/out_executables,
      /*.sanitizer=*/out_sanitizer,
      /*.facet_count=*/1,
      /*.facets=*/out_facets,
  };
}

TEST(DeviceSpecTest, CreateSerializeParseAndSelect) {
  uint8_t facet_payload_storage[] = {0x01, 0x02, 0x03};
  iree_hal_device_identity_spec_t identity;
  iree_hal_physical_device_spec_t physical_devices[1];
  iree_hal_device_memory_spec_t memory;
  iree_hal_memory_heap_spec_t memory_heaps[1];
  iree_hal_memory_type_spec_t memory_types[1];
  iree_hal_device_queue_spec_t queues;
  iree_hal_queue_family_spec_t queue_families[1];
  iree_hal_device_dispatch_spec_t dispatch;
  iree_hal_device_timing_spec_t timing;
  iree_hal_device_executable_spec_t executables;
  iree_hal_executable_target_t executable_targets[3];
  iree_hal_device_sanitizer_spec_t sanitizer;
  iree_hal_device_spec_facet_t facets[1];
  iree_hal_device_spec_params_t params = MakeTestSpecParams(
      &identity, physical_devices, &memory, memory_heaps, memory_types, &queues,
      queue_families, &dispatch, &timing, &executables, executable_targets,
      &sanitizer, facets,
      iree_make_const_byte_span(facet_payload_storage,
                                sizeof(facet_payload_storage)));

  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));

  ExpectStringViewEq(iree_hal_device_spec_identity(spec)->display_name,
                     "Test Device");
  EXPECT_EQ(iree_hal_device_spec_memory(spec)->heap_count, 1);
  EXPECT_EQ(iree_hal_device_spec_queues(spec)->family_count, 1);
  const iree_hal_device_sanitizer_spec_t* sanitizer_spec =
      iree_hal_device_spec_sanitizer(spec);
  EXPECT_TRUE(iree_all_bits_set(sanitizer_spec->flags,
                                IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN));
  EXPECT_EQ(sanitizer_spec->asan.pool_options.mode,
            IREE_HAL_ASAN_POOL_MODE_SHADOW);
  EXPECT_EQ(sanitizer_spec->asan.pool_options.shadow_granule_size, 8u);
  EXPECT_EQ(sanitizer_spec->asan.pool_options.redzone_size, 64u);
  EXPECT_EQ(sanitizer_spec->asan.pool_options.backing_alignment, 4096u);
  EXPECT_EQ(sanitizer_spec->asan.pool_options.quarantine_size, 1024u * 1024u);
  EXPECT_EQ(iree_hal_device_spec_facet_count(spec), 1);
  const iree_hal_device_spec_facet_t* facet = iree_hal_device_spec_find_facet(
      spec, iree_make_cstring_view("test.facet"));
  ASSERT_NE(facet, nullptr);
  EXPECT_EQ(facet->schema_version, 1);
  ASSERT_EQ(facet->payload.data_length, 3);
  EXPECT_EQ(0, memcmp(facet->payload.data, facet_payload_storage,
                      sizeof(facet_payload_storage)));

  iree_hal_executable_target_selection_t exact_selection = {
      /*.family=*/iree_make_cstring_view("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  iree_hal_executable_target_selection_result_t selection_result =
      iree_hal_device_spec_select_executable_target(spec, &exact_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            selection_result.outcome);
  ASSERT_NE(selection_result.target, nullptr);
  EXPECT_EQ(selection_result.target_ordinal, 0u);
  ExpectStringViewEq(selection_result.target->target_key, "gfx1100:xnack-");

  iree_hal_executable_target_selection_t generic_selection = {
      /*.family=*/iree_make_cstring_view("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      /*.physical_device_affinity=*/0,
  };
  selection_result =
      iree_hal_device_spec_select_executable_target(spec, &generic_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS,
            selection_result.outcome);
  EXPECT_EQ(selection_result.target, nullptr);
  EXPECT_EQ(selection_result.target_ordinal, IREE_HOST_SIZE_MAX);

  iree_hal_executable_target_selection_t key_selection = {
      /*.family=*/iree_make_cstring_view("amdgpu"),
      /*.target_key=*/iree_make_cstring_view("gfx11-generic-alt"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      /*.physical_device_affinity=*/0,
  };
  selection_result =
      iree_hal_device_spec_select_executable_target(spec, &key_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            selection_result.outcome);
  ASSERT_NE(selection_result.target, nullptr);
  EXPECT_EQ(selection_result.target_ordinal, 2u);
  ExpectStringViewEq(selection_result.target->target_key, "gfx11-generic-alt");
  EXPECT_EQ(iree_hal_device_spec_executable_target_ordinal(
                spec, selection_result.target),
            2u);
  iree_hal_executable_target_t copied_target = *selection_result.target;
  EXPECT_EQ(
      iree_hal_device_spec_executable_target_ordinal(spec, &copied_target),
      IREE_HOST_SIZE_MAX);
  EXPECT_EQ(iree_hal_device_spec_executable_target_ordinal(spec, nullptr),
            IREE_HOST_SIZE_MAX);

  key_selection.target_key = iree_string_view_empty();
  key_selection.physical_device_affinity = 1u << 1;
  selection_result =
      iree_hal_device_spec_select_executable_target(spec, &key_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH,
            selection_result.outcome);
  EXPECT_EQ(selection_result.target, nullptr);
  EXPECT_EQ(selection_result.target_ordinal, IREE_HOST_SIZE_MAX);

  iree_byte_span_t serialized_bytes = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_device_spec_serialize(spec, iree_allocator_system(),
                                                &serialized_bytes));
  static const uint8_t kExpectedHeaderPrefix[] = {
      'D', 'S', 'P', 'C', 6, 0, 0, 0,
  };
  ASSERT_GE(serialized_bytes.data_length, sizeof(kExpectedHeaderPrefix));
  EXPECT_EQ(memcmp(serialized_bytes.data, kExpectedHeaderPrefix,
                   sizeof(kExpectedHeaderPrefix)),
            0);
  iree_hal_device_spec_t* parsed_spec = NULL;
  IREE_ASSERT_OK(
      iree_hal_device_spec_parse(iree_const_cast_byte_span(serialized_bytes),
                                 iree_allocator_system(), &parsed_spec));
  iree_byte_span_t reparsed_serialized_bytes = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_device_spec_serialize(
      parsed_spec, iree_allocator_system(), &reparsed_serialized_bytes));
  ASSERT_EQ(serialized_bytes.data_length,
            reparsed_serialized_bytes.data_length);
  EXPECT_EQ(memcmp(serialized_bytes.data, reparsed_serialized_bytes.data,
                   serialized_bytes.data_length),
            0);
  EXPECT_EQ(iree_hal_device_spec_fingerprint(spec),
            iree_hal_device_spec_fingerprint(parsed_spec));
  ExpectStringViewEq(iree_hal_device_spec_identity(parsed_spec)->display_name,
                     "Test Device");
  EXPECT_EQ(iree_hal_device_spec_sanitizer(parsed_spec)
                ->asan.pool_options.redzone_size,
            64u);

  iree_hal_device_spec_release(parsed_spec);
  iree_allocator_free(iree_allocator_system(), reparsed_serialized_bytes.data);
  iree_allocator_free(iree_allocator_system(), serialized_bytes.data);
  iree_hal_device_spec_release(spec);
}

TEST(DeviceSpecSerializationTest, RejectsMalformedImages) {
  {
    SCOPED_TRACE("non-empty NULL storage");
    iree_hal_device_spec_t* spec = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_device_spec_parse(iree_make_const_byte_span(nullptr, 1),
                                   iree_allocator_system(), &spec));
    EXPECT_EQ(spec, nullptr);
  }

  const std::vector<uint8_t> canonical_bytes = SerializeEmptySpec();
  ASSERT_EQ(canonical_bytes.size(), 356u);

  {
    SCOPED_TRACE("invalid magic");
    std::vector<uint8_t> bytes = canonical_bytes;
    StoreU32Le(&bytes, 0, 0);
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("unsupported version");
    std::vector<uint8_t> bytes = canonical_bytes;
    StoreU32Le(&bytes, 4, 7);
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("declared length mismatch");
    std::vector<uint8_t> bytes = canonical_bytes;
    StoreU64Le(&bytes, 8, bytes.size() + 1);
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("truncated body");
    std::vector<uint8_t> bytes = canonical_bytes;
    bytes.pop_back();
    StoreU64Le(&bytes, 8, bytes.size());
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("trailing data");
    std::vector<uint8_t> bytes = canonical_bytes;
    bytes.push_back(0);
    StoreU64Le(&bytes, 8, bytes.size());
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("impossible physical device count");
    std::vector<uint8_t> bytes = canonical_bytes;
    StoreU64Le(&bytes, 16, UINT64_MAX);
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("impossible string length");
    std::vector<uint8_t> bytes = canonical_bytes;
    StoreU64Le(&bytes, 88, UINT64_MAX);
    ExpectParseFails(bytes);
  }
  {
    SCOPED_TRACE("invalid ASAN mode");
    std::vector<uint8_t> bytes = canonical_bytes;
    // The empty v6 body places the sanitizer flags at byte 316 and its mode at
    // byte 320.
    StoreU32Le(&bytes, 320, UINT32_MAX);
    ExpectParseFails(bytes);
  }
}

TEST(DeviceSpecTest, RejectsInvalidExternalTimepointHandleTypes) {
  iree_hal_external_timepoint_handle_spec_t external_timepoint_handle = {
      /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_NONE,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
      /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT,
      /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE,
  };
  iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/0,
      /*.families=*/NULL,
      /*.external_timepoint_handle_count=*/1,
      /*.external_timepoint_handles=*/&external_timepoint_handle,
      /*.flags=*/IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/NULL,
      /*.memory=*/NULL,
      /*.virtual_memory=*/NULL,
      /*.queues=*/&queues,
      /*.dispatch=*/NULL,
      /*.timing=*/NULL,
      /*.executables=*/NULL,
      /*.sanitizer=*/NULL,
      /*.facet_count=*/0,
      /*.facets=*/NULL,
  };
  iree_hal_device_spec_t* spec = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);

  external_timepoint_handle.handle_type =
      (iree_hal_external_timepoint_type_t)UINT32_MAX;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);
}

TEST(DeviceSpecTest, RejectsInvalidPhysicalDeviceAffinityFacts) {
  iree_hal_physical_device_spec_t physical_device = {
      /*.identity=*/{},
      /*.physical_ordinal=*/0,
      /*.partition_ordinal=*/0,
      /*.partition_count=*/1,
      /*.physical_device_affinity=*/3,
  };
  iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/iree_make_cstring_view("test-device"),
      /*.display_name=*/iree_make_cstring_view("Test Device"),
      /*.driver_id=*/iree_make_cstring_view("test"),
      /*.driver_version=*/iree_string_view_empty(),
      /*.backend_id=*/iree_make_cstring_view("test"),
      /*.device_path=*/iree_string_view_empty(),
      /*.vendor_name=*/iree_string_view_empty(),
      /*.vendor_id=*/0,
      /*.device_id=*/0,
      /*.revision_id=*/0,
      /*.logical_ordinal=*/0,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_hal_executable_target_t target = {
      /*.family=*/iree_make_cstring_view("test"),
      /*.target_key=*/iree_make_cstring_view("test-target"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      /*.priority=*/0,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/1,
      /*.targets=*/&target,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/&identity,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/nullptr,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };

  iree_hal_device_spec_t* spec = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);

  physical_device.physical_device_affinity = 1;
  target.physical_device_affinity = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);

  target.physical_device_affinity = 1u << 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);
}

TEST(DeviceSpecTest,
     ExecutableTargetSelectionRequiresFullPhysicalDeviceCoverage) {
  iree_hal_physical_device_spec_t physical_devices[2] = {
      {
          /*.identity=*/{},
          /*.physical_ordinal=*/0,
          /*.partition_ordinal=*/0,
          /*.partition_count=*/1,
          /*.physical_device_affinity=*/1u << 0,
      },
      {
          /*.identity=*/{},
          /*.physical_ordinal=*/1,
          /*.partition_ordinal=*/0,
          /*.partition_count=*/1,
          /*.physical_device_affinity=*/1u << 1,
      },
  };
  iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/iree_make_cstring_view("test-device"),
      /*.display_name=*/iree_make_cstring_view("Test Device"),
      /*.driver_id=*/iree_make_cstring_view("test"),
      /*.driver_version=*/iree_string_view_empty(),
      /*.backend_id=*/iree_make_cstring_view("test"),
      /*.device_path=*/iree_string_view_empty(),
      /*.vendor_name=*/iree_string_view_empty(),
      /*.vendor_id=*/0,
      /*.device_id=*/0,
      /*.revision_id=*/0,
      /*.logical_ordinal=*/0,
      /*.physical_device_count=*/IREE_ARRAYSIZE(physical_devices),
      /*.physical_devices=*/physical_devices,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_hal_executable_target_t targets[2] = {
      {
          /*.family=*/iree_make_cstring_view("test"),
          /*.target_key=*/iree_make_cstring_view("device-0"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/100,
          /*.physical_device_affinity=*/1u << 0,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          /*.family=*/iree_make_cstring_view("test"),
          /*.target_key=*/iree_make_cstring_view("both-devices"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          /*.priority=*/10,
          /*.physical_device_affinity=*/(1u << 0) | (1u << 1),
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/IREE_ARRAYSIZE(targets),
      /*.targets=*/targets,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/&identity,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/nullptr,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };
  iree_hal_device_spec_t* spec = nullptr;
  IREE_ASSERT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));

  iree_hal_executable_target_selection_t selection = {
      /*.family=*/iree_make_cstring_view("test"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_NONE,
      /*.physical_device_affinity=*/(1u << 0) | (1u << 1),
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(spec, &selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            result.outcome);
  ASSERT_NE(nullptr, result.target);
  ExpectStringViewEq(result.target->target_key, "both-devices");

  iree_hal_device_spec_release(spec);
}

TEST(DeviceSpecTest, RejectsInvalidExtensionFacets) {
  iree_hal_device_spec_facet_t facets[2] = {
      {
          /*.schema_id=*/iree_string_view_empty(),
          /*.schema_version=*/1,
          /*.payload=*/iree_const_byte_span_empty(),
      },
      {
          /*.schema_id=*/iree_make_cstring_view("test.facet"),
          /*.schema_version=*/2,
          /*.payload=*/iree_const_byte_span_empty(),
      },
  };
  iree_hal_device_spec_params_t params = {
      /*.identity=*/NULL,
      /*.memory=*/NULL,
      /*.virtual_memory=*/NULL,
      /*.queues=*/NULL,
      /*.dispatch=*/NULL,
      /*.timing=*/NULL,
      /*.executables=*/NULL,
      /*.sanitizer=*/NULL,
      /*.facet_count=*/1,
      /*.facets=*/facets,
  };

  iree_hal_device_spec_t* spec = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);

  facets[0].schema_id = facets[1].schema_id;
  params.facet_count = IREE_ARRAYSIZE(facets);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));
  EXPECT_EQ(spec, nullptr);
}

TEST(DeviceObservationTest, MemoryTotalFromSpecSumsKnownHeaps) {
  uint8_t facet_payload_storage[] = {0x01, 0x02, 0x03};
  iree_hal_device_identity_spec_t identity;
  iree_hal_physical_device_spec_t physical_devices[1];
  iree_hal_device_memory_spec_t memory;
  iree_hal_memory_heap_spec_t memory_heaps[1];
  iree_hal_memory_type_spec_t memory_types[1];
  iree_hal_device_queue_spec_t queues;
  iree_hal_queue_family_spec_t queue_families[1];
  iree_hal_device_dispatch_spec_t dispatch;
  iree_hal_device_timing_spec_t timing;
  iree_hal_device_executable_spec_t executables;
  iree_hal_executable_target_t executable_targets[3];
  iree_hal_device_sanitizer_spec_t sanitizer;
  iree_hal_device_spec_facet_t facets[1];
  iree_hal_device_spec_params_t params = MakeTestSpecParams(
      &identity, physical_devices, &memory, memory_heaps, memory_types, &queues,
      queue_families, &dispatch, &timing, &executables, executable_targets,
      &sanitizer, facets,
      iree_make_const_byte_span(facet_payload_storage,
                                sizeof(facet_payload_storage)));

  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));

  iree_hal_device_observation_t observation = {};
  iree_hal_device_observation_initialize(
      IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY, &observation);
  IREE_ASSERT_OK(iree_hal_device_observation_populate_memory_total_from_spec(
      spec, &observation));
  EXPECT_EQ(IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY,
            observation.provided_flags);
  EXPECT_EQ(IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES,
            observation.memory.flags);
  EXPECT_EQ(1024ull * 1024ull * 1024ull, observation.memory.total_bytes);
  EXPECT_EQ(0, observation.memory.available_bytes);

  iree_hal_device_spec_release(spec);
}

TEST(DeviceObservationTest, MemoryTotalFromSpecSkipsUnknownCapacity) {
  uint8_t facet_payload_storage[] = {0x01, 0x02, 0x03};
  iree_hal_device_identity_spec_t identity;
  iree_hal_physical_device_spec_t physical_devices[1];
  iree_hal_device_memory_spec_t memory;
  iree_hal_memory_heap_spec_t memory_heaps[1];
  iree_hal_memory_type_spec_t memory_types[1];
  iree_hal_device_queue_spec_t queues;
  iree_hal_queue_family_spec_t queue_families[1];
  iree_hal_device_dispatch_spec_t dispatch;
  iree_hal_device_timing_spec_t timing;
  iree_hal_device_executable_spec_t executables;
  iree_hal_executable_target_t executable_targets[3];
  iree_hal_device_sanitizer_spec_t sanitizer;
  iree_hal_device_spec_facet_t facets[1];
  iree_hal_device_spec_params_t params = MakeTestSpecParams(
      &identity, physical_devices, &memory, memory_heaps, memory_types, &queues,
      queue_families, &dispatch, &timing, &executables, executable_targets,
      &sanitizer, facets,
      iree_make_const_byte_span(facet_payload_storage,
                                sizeof(facet_payload_storage)));
  memory_heaps[0].flags = IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN;

  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));

  iree_hal_device_observation_t observation = {};
  iree_hal_device_observation_initialize(
      IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY, &observation);
  IREE_ASSERT_OK(iree_hal_device_observation_populate_memory_total_from_spec(
      spec, &observation));
  EXPECT_EQ(IREE_HAL_DEVICE_OBSERVATION_FLAG_NONE, observation.provided_flags);
  EXPECT_EQ(IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_NONE,
            observation.memory.flags);

  iree_hal_device_spec_release(spec);
}

TEST(DeviceSpecTest, FindsVirtualMemoryAndExternalHandleRecords) {
  iree_hal_memory_heap_spec_t memory_heaps[1] = {
      {
          /*.name=*/iree_make_cstring_view("device-local"),
          /*.capacity_bytes=*/1024ull * 1024ull * 1024ull,
          /*.allocation_granularity=*/4096,
          /*.allocation_alignment=*/256,
          /*.maximum_allocation_size=*/512ull * 1024ull * 1024ull,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_MEMORY_HEAP_SPEC_FLAG_NONE,
      },
  };
  iree_hal_memory_type_spec_t memory_types[2] = {
      {
          /*.heap_index=*/0,
          /*.memory_type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_ALL,
          /*.minimum_alignment=*/256,
          /*.optimal_transfer_granularity=*/4096,
          /*.flags=*/IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
      },
      {
          /*.heap_index=*/0,
          /*.memory_type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
          /*.minimum_alignment=*/64,
          /*.optimal_transfer_granularity=*/4096,
          /*.flags=*/IREE_HAL_MEMORY_TYPE_SPEC_FLAG_NONE,
      },
  };
  iree_hal_external_buffer_handle_spec_t external_buffer_handles[2] = {
      {
          /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_OPAQUE_FD |
              IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
              IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER |
              IREE_HAL_BUFFER_USAGE_STORAGE_READ,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_READ |
              IREE_HAL_MEMORY_ACCESS_WRITE,
          /*.compatible_memory_type_mask=*/1u << 0,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS |
              IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_OWNING,
      },
      {
          /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
          /*.compatible_memory_type_mask=*/1u << 1,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_BORROWED,
      },
  };
  iree_hal_device_memory_spec_t memory = {
      /*.heap_count=*/1,
      /*.heaps=*/memory_heaps,
      /*.memory_type_count=*/2,
      /*.memory_types=*/memory_types,
      /*.external_buffer_handle_count=*/2,
      /*.external_buffer_handles=*/external_buffer_handles,
      /*.flags=*/IREE_HAL_DEVICE_MEMORY_SPEC_FLAG_NONE,
  };

  iree_hal_virtual_memory_class_spec_t virtual_memory_classes[2] = {
      {
          /*.compatible_memory_type_mask=*/1u << 0,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_ALL,
          /*.minimum_page_size=*/4096,
          /*.recommended_page_size=*/65536,
          /*.maximum_reservation_size=*/1ull << 32,
          /*.maximum_physical_allocation_size=*/1ull << 30,
          /*.operation_flags=*/IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RESERVE |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RELEASE |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_MAP |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_UNMAP |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PROTECT |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_ADVISE,
          /*.protection_flags=*/IREE_HAL_MEMORY_PROTECTION_READ_WRITE,
          /*.advice_flags=*/IREE_HAL_MEMORY_ADVICE_WILL_NEED,
          /*.flags=*/IREE_HAL_VIRTUAL_MEMORY_CLASS_SPEC_FLAG_NONE,
      },
      {
          /*.compatible_memory_type_mask=*/1u << 1,
          /*.allowed_buffer_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*.allowed_memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
          /*.minimum_page_size=*/4096,
          /*.recommended_page_size=*/4096,
          /*.maximum_reservation_size=*/1ull << 20,
          /*.maximum_physical_allocation_size=*/1ull << 20,
          /*.operation_flags=*/IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RESERVE |
              IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_RELEASE,
          /*.protection_flags=*/IREE_HAL_MEMORY_PROTECTION_READ,
          /*.advice_flags=*/IREE_HAL_MEMORY_ADVICE_NORMAL,
          /*.flags=*/IREE_HAL_VIRTUAL_MEMORY_CLASS_SPEC_FLAG_NONE,
      },
  };
  iree_hal_device_virtual_memory_spec_t virtual_memory = {
      /*.class_count=*/2,
      /*.classes=*/virtual_memory_classes,
      /*.flags=*/IREE_HAL_DEVICE_VIRTUAL_MEMORY_SPEC_FLAG_NONE,
  };

  iree_hal_external_timepoint_handle_spec_t external_timepoint_handles[2] = {
      {
          /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT |
              IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
          /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT |
              IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS,
      },
      {
          /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_ASYNC_PRIMITIVE,
          /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
          /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_WAIT,
          /*.flags=*/IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_NONE,
      },
  };
  iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/0,
      /*.families=*/NULL,
      /*.external_timepoint_handle_count=*/2,
      /*.external_timepoint_handles=*/external_timepoint_handles,
      /*.flags=*/IREE_HAL_DEVICE_QUEUE_SPEC_FLAG_NONE,
  };

  iree_hal_device_spec_params_t params = {
      /*.identity=*/NULL,
      /*.memory=*/&memory,
      /*.virtual_memory=*/&virtual_memory,
      /*.queues=*/&queues,
      /*.dispatch=*/NULL,
      /*.timing=*/NULL,
      /*.executables=*/NULL,
      /*.sanitizer=*/NULL,
      /*.facet_count=*/0,
      /*.facets=*/NULL,
  };
  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(
      iree_hal_device_spec_create(&params, iree_allocator_system(), &spec));

  iree_hal_virtual_memory_class_selection_t virtual_memory_selection = {
      /*.compatible_memory_type_mask=*/1u << 0,
      /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_STORAGE_READ,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
      /*.operation_flags=*/IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_MAP |
          IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_UNMAP,
      /*.protection_flags=*/IREE_HAL_MEMORY_PROTECTION_READ,
      /*.advice_flags=*/IREE_HAL_MEMORY_ADVICE_WILL_NEED,
  };
  const iree_hal_virtual_memory_class_spec_t* virtual_memory_class =
      iree_hal_device_spec_find_virtual_memory_class(spec,
                                                     &virtual_memory_selection);
  ASSERT_NE(virtual_memory_class, nullptr);
  EXPECT_EQ(virtual_memory_class->recommended_page_size, 65536);

  virtual_memory_selection.operation_flags =
      IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_PHYSICAL_ALLOCATE;
  EXPECT_EQ(iree_hal_device_spec_find_virtual_memory_class(
                spec, &virtual_memory_selection),
            nullptr);
  iree_hal_virtual_memory_class_selection_t wildcard_virtual_memory_selection =
      {};
  ASSERT_NE(iree_hal_device_spec_find_virtual_memory_class(
                spec, &wildcard_virtual_memory_selection),
            nullptr);

  iree_hal_external_buffer_handle_selection_t buffer_handle_selection = {
      /*.handle_type_mask=*/IREE_HAL_TOPOLOGY_HANDLE_TYPE_DMA_BUF,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_EXPORT,
      /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_READ,
      /*.compatible_memory_type_mask=*/1u << 0,
      /*.capability_flags=*/
      IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS,
  };
  const iree_hal_external_buffer_handle_spec_t* buffer_handle =
      iree_hal_device_spec_find_external_buffer_handle(
          spec, &buffer_handle_selection);
  ASSERT_NE(buffer_handle, nullptr);
  EXPECT_EQ(buffer_handle->compatible_memory_type_mask, 1u << 0);

  buffer_handle_selection.handle_type_mask = IREE_HAL_TOPOLOGY_HANDLE_TYPE_SHM;
  EXPECT_EQ(iree_hal_device_spec_find_external_buffer_handle(
                spec, &buffer_handle_selection),
            nullptr);
  iree_hal_external_buffer_handle_selection_t wildcard_buffer_selection = {};
  ASSERT_NE(iree_hal_device_spec_find_external_buffer_handle(
                spec, &wildcard_buffer_selection),
            nullptr);

  iree_hal_external_timepoint_handle_selection_t timepoint_selection = {
      /*.handle_type=*/IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT,
      /*.direction_flags=*/IREE_HAL_EXTERNAL_HANDLE_DIRECTION_FLAG_IMPORT,
      /*.compatibility=*/IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_WAIT,
      /*.capability_flags=*/
      IREE_HAL_EXTERNAL_HANDLE_CAPABILITY_FLAG_CROSS_PROCESS,
  };
  const iree_hal_external_timepoint_handle_spec_t* timepoint_handle =
      iree_hal_device_spec_find_external_timepoint_handle(spec,
                                                          &timepoint_selection);
  ASSERT_NE(timepoint_handle, nullptr);
  EXPECT_EQ(timepoint_handle->handle_type,
            IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_HIP_EVENT);

  timepoint_selection.compatibility =
      IREE_HAL_SEMAPHORE_COMPATIBILITY_DEVICE_SIGNAL;
  EXPECT_EQ(iree_hal_device_spec_find_external_timepoint_handle(
                spec, &timepoint_selection),
            nullptr);
  iree_hal_external_timepoint_handle_selection_t wildcard_timepoint_selection =
      {};
  ASSERT_NE(iree_hal_device_spec_find_external_timepoint_handle(
                spec, &wildcard_timepoint_selection),
            nullptr);

  iree_byte_span_t serialized_bytes = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_device_spec_serialize(spec, iree_allocator_system(),
                                                &serialized_bytes));
  iree_allocator_free(iree_allocator_system(), serialized_bytes.data);
  iree_hal_device_spec_release(spec);
}

TEST(DeviceSpecBuilderTest, CopiesInputsAndFinalizes) {
  iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/iree_make_cstring_view("builder-device"),
      /*.display_name=*/iree_make_cstring_view("Builder Device"),
      /*.driver_id=*/iree_make_cstring_view("test"),
      /*.driver_version=*/iree_make_cstring_view("1.0"),
      /*.backend_id=*/iree_make_cstring_view("test"),
      /*.device_path=*/iree_make_cstring_view("test://builder"),
      /*.vendor_name=*/iree_make_cstring_view("Example"),
      /*.vendor_id=*/1,
      /*.device_id=*/2,
      /*.revision_id=*/3,
      /*.logical_ordinal=*/0,
      /*.physical_device_count=*/0,
      /*.physical_devices=*/NULL,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_hal_device_dispatch_spec_t dispatch = {
      /*.launch=*/
      {
          /*.maximum_workgroup_invocations=*/256,
          /*.maximum_workgroup_size=*/{256, 1, 1},
          /*.maximum_workgroup_count=*/{1024, 1, 1},
      },
      /*.subgroup=*/
      {
          /*.default_size=*/32,
          /*.minimum_size=*/32,
          /*.maximum_size=*/32,
          /*.supported_size_mask=*/1ull << 32,
      },
      /*.execution=*/
      {
          /*.unit_count=*/1,
          /*.group_count=*/1,
          /*.maximum_resident_workgroup_count=*/1,
          /*.maximum_resident_invocation_count=*/256,
          /*.maximum_resident_subgroup_count=*/0,
          /*.maximum_register_count=*/65536,
          /*.maximum_workgroup_register_count=*/65536,
          /*.maximum_local_memory_size=*/32 * 1024,
          /*.maximum_workgroup_local_memory_size=*/32 * 1024,
          /*.maximum_workgroup_local_memory_size_optin=*/32 * 1024,
      },
      /*.addressing=*/
      {
          /*.pointer_size_bits=*/64,
          /*.address_space_bits=*/64,
          /*.minimum_buffer_device_address_alignment=*/0,
      },
      /*.flags=*/IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
  };
  uint8_t payload_storage[] = {0xaa, 0xbb};
  iree_hal_device_spec_facet_t facet = {
      /*.schema_id=*/iree_make_cstring_view("builder.facet"),
      /*.schema_version=*/7,
      /*.payload=*/
      iree_make_const_byte_span(payload_storage, sizeof(payload_storage)),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_hal_device_spec_builder_set_identity(&builder, &identity));
  IREE_ASSERT_OK(
      iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch));
  iree_hal_device_sanitizer_spec_t sanitizer = {
      /*.flags=*/IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN,
      /*.asan=*/
      {
          /*.pool_options=*/
          {
              /*.mode=*/IREE_HAL_ASAN_POOL_MODE_SHADOW,
              /*.shadow_granule_size=*/16,
              /*.redzone_size=*/128,
              /*.backing_alignment=*/4096,
              /*.quarantine_size=*/0,
          },
      },
  };
  IREE_ASSERT_OK(
      iree_hal_device_spec_builder_set_sanitizer(&builder, &sanitizer));
  IREE_ASSERT_OK(iree_hal_device_spec_builder_add_facet(&builder, &facet));

  iree_hal_device_spec_t* spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &spec));
  ExpectStringViewEq(iree_hal_device_spec_identity(spec)->logical_device_id,
                     "builder-device");
  EXPECT_EQ(
      iree_hal_device_spec_dispatch(spec)->launch.maximum_workgroup_invocations,
      256);
  EXPECT_EQ(
      iree_hal_device_spec_sanitizer(spec)->asan.pool_options.redzone_size,
      128u);
  const iree_hal_device_spec_facet_t* copied_facet =
      iree_hal_device_spec_facet_at(spec, 0);
  ASSERT_NE(copied_facet, nullptr);
  ExpectStringViewEq(copied_facet->schema_id, "builder.facet");
  EXPECT_EQ(copied_facet->schema_version, 7);

  iree_hal_device_spec_release(spec);
  iree_hal_device_spec_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace iree::hal
