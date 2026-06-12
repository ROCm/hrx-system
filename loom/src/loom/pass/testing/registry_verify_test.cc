// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/testing/registry_verify.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/pass/environment.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

static const loom_pass_descriptor_t* LookupTestPass(iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(
      loom_pass_registry_lookup(loom_test_pass_registry(), name, &descriptor));
  return descriptor;
}

static const loom_pass_info_t* BrokenStatisticsPassInfo() {
  static const loom_pass_statistic_layout_t kLayout = {
      /*.storage_size=*/sizeof(int64_t),
      /*.fields=*/nullptr,
      /*.field_count=*/1,
  };
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("test.broken-statistics"),
      /*.description=*/
      IREE_SVL("Synthetic pass with malformed statistics."),
      /*.kind=*/LOOM_PASS_MODULE,
      /*.option_defs=*/nullptr,
      /*.option_count=*/0,
      /*.statistic_layout=*/&kLayout,
  };
  return &kInfo;
}

static const loom_pass_info_t* DuplicateStatisticsPassInfo() {
  typedef struct duplicate_statistics_t {
    int64_t first;
    int64_t second;
  } duplicate_statistics_t;
  static const loom_pass_statistic_field_t kFields[] = {
      LOOM_PASS_STATISTIC_FIELD(duplicate_statistics_t, first, "duplicated",
                                "First field."),
      LOOM_PASS_STATISTIC_FIELD(duplicate_statistics_t, second, "duplicated",
                                "Second field."),
  };
  static const loom_pass_statistic_layout_t kLayout = {
      /*.storage_size=*/sizeof(duplicate_statistics_t),
      /*.fields=*/kFields,
      /*.field_count=*/IREE_ARRAYSIZE(kFields),
  };
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("test.duplicate-statistics"),
      /*.description=*/
      IREE_SVL("Synthetic pass with duplicate statistics."),
      /*.kind=*/LOOM_PASS_MODULE,
      /*.option_defs=*/nullptr,
      /*.option_count=*/0,
      /*.statistic_layout=*/&kLayout,
  };
  return &kInfo;
}

static const loom_pass_info_t* DuplicateStatisticOffsetsPassInfo() {
  static const loom_pass_statistic_field_t kFields[] = {
      {
          /*.name=*/IREE_SVL("first"),
          /*.description=*/IREE_SVL("First field."),
          /*.offset=*/0,
      },
      {
          /*.name=*/IREE_SVL("second"),
          /*.description=*/IREE_SVL("Second field."),
          /*.offset=*/0,
      },
  };
  static const loom_pass_statistic_layout_t kLayout = {
      /*.storage_size=*/sizeof(int64_t),
      /*.fields=*/kFields,
      /*.field_count=*/IREE_ARRAYSIZE(kFields),
  };
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("test.duplicate-statistic-offsets"),
      /*.description=*/
      IREE_SVL("Synthetic pass with duplicate statistic offsets."),
      /*.kind=*/LOOM_PASS_MODULE,
      /*.option_defs=*/nullptr,
      /*.option_count=*/0,
      /*.statistic_layout=*/&kLayout,
  };
  return &kInfo;
}

static const loom_pass_info_t* UnalignedStatisticsPassInfo() {
  static const loom_pass_statistic_field_t kFields[] = {
      {
          /*.name=*/IREE_SVL("unaligned"),
          /*.description=*/IREE_SVL("Unaligned synthetic statistic."),
          /*.offset=*/1,
      },
  };
  static const loom_pass_statistic_layout_t kLayout = {
      /*.storage_size=*/sizeof(int64_t) + 1,
      /*.fields=*/kFields,
      /*.field_count=*/IREE_ARRAYSIZE(kFields),
  };
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("test.unaligned-statistics"),
      /*.description=*/
      IREE_SVL("Synthetic pass with unaligned statistics."),
      /*.kind=*/LOOM_PASS_MODULE,
      /*.option_defs=*/nullptr,
      /*.option_count=*/0,
      /*.statistic_layout=*/&kLayout,
  };
  return &kInfo;
}

static const loom_pass_info_t* TooManyStatisticsPassInfo() {
  static const loom_pass_statistic_field_t kFields[] = {
      {/*.name=*/IREE_SVL("stat-00"),
       /*.description=*/IREE_SVL("Statistic 00."),
       /*.offset=*/sizeof(int64_t) * 0},
      {/*.name=*/IREE_SVL("stat-01"),
       /*.description=*/IREE_SVL("Statistic 01."),
       /*.offset=*/sizeof(int64_t) * 1},
      {/*.name=*/IREE_SVL("stat-02"),
       /*.description=*/IREE_SVL("Statistic 02."),
       /*.offset=*/sizeof(int64_t) * 2},
      {/*.name=*/IREE_SVL("stat-03"),
       /*.description=*/IREE_SVL("Statistic 03."),
       /*.offset=*/sizeof(int64_t) * 3},
      {/*.name=*/IREE_SVL("stat-04"),
       /*.description=*/IREE_SVL("Statistic 04."),
       /*.offset=*/sizeof(int64_t) * 4},
      {/*.name=*/IREE_SVL("stat-05"),
       /*.description=*/IREE_SVL("Statistic 05."),
       /*.offset=*/sizeof(int64_t) * 5},
      {/*.name=*/IREE_SVL("stat-06"),
       /*.description=*/IREE_SVL("Statistic 06."),
       /*.offset=*/sizeof(int64_t) * 6},
      {/*.name=*/IREE_SVL("stat-07"),
       /*.description=*/IREE_SVL("Statistic 07."),
       /*.offset=*/sizeof(int64_t) * 7},
      {/*.name=*/IREE_SVL("stat-08"),
       /*.description=*/IREE_SVL("Statistic 08."),
       /*.offset=*/sizeof(int64_t) * 8},
      {/*.name=*/IREE_SVL("stat-09"),
       /*.description=*/IREE_SVL("Statistic 09."),
       /*.offset=*/sizeof(int64_t) * 9},
      {/*.name=*/IREE_SVL("stat-10"),
       /*.description=*/IREE_SVL("Statistic 10."),
       /*.offset=*/sizeof(int64_t) * 10},
  };
  static const loom_pass_statistic_layout_t kLayout = {
      /*.storage_size=*/sizeof(int64_t) * IREE_ARRAYSIZE(kFields),
      /*.fields=*/kFields,
      /*.field_count=*/IREE_ARRAYSIZE(kFields),
  };
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("test.too-many-statistics"),
      /*.description=*/
      IREE_SVL("Synthetic pass with too many statistics."),
      /*.kind=*/LOOM_PASS_MODULE,
      /*.option_defs=*/nullptr,
      /*.option_count=*/0,
      /*.statistic_layout=*/&kLayout,
  };
  return &kInfo;
}

static bool SatisfyTestRequirement(
    const loom_pass_environment_capability_t* capability,
    iree_string_view_t requirement) {
  (void)capability;
  (void)requirement;
  return true;
}

static const loom_pass_environment_capability_type_t kTestRequirementType = {
    /*.name=*/IREE_SVL("test.requirements"),
    /*.satisfies_requirement=*/SatisfyTestRequirement,
};

TEST(PassRegistryCoreTest, SyntheticRegistryVerifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_test_pass_registry()));
}

TEST(PassRegistryCoreTest, FormatsRegistryMetadataJson) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);

  IREE_EXPECT_OK(loom_pass_report_format_registry_json(
      loom_test_pass_registry(), &stream));

  iree_string_view_t json = iree_string_builder_view(&builder);
  std::string text(json.data, json.size);
  EXPECT_NE(text.find("\"key\":\"test.options\""), std::string::npos);
  EXPECT_NE(text.find("\"kind\":\"uint32\""), std::string::npos);
  EXPECT_NE(text.find("\"minimum\":1"), std::string::npos);
  EXPECT_NE(text.find("\"values\":[\"alpha\",\"beta\"]"), std::string::npos);
  EXPECT_NE(text.find("\"required\":true"), std::string::npos);
  EXPECT_NE(text.find("\"key\":\"test.unavailable\""), std::string::npos);
  EXPECT_NE(text.find("\"available\":false"), std::string::npos);
  EXPECT_NE(text.find("\"unavailable_reason\":\"disabled for test\""),
            std::string::npos);
  EXPECT_NE(text.find("\"requirements\":[{\"key\":\"target.record\""),
            std::string::npos);
  EXPECT_NE(text.find("\"capability\":\"test.target-record\""),
            std::string::npos);

  iree_string_builder_deinitialize(&builder);
}

TEST(PassRegistryCoreTest, RejectsMissingStatisticMetadata) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.broken-statistics");
  descriptor.info = BrokenStatisticsPassInfo;
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsDuplicateStatisticNames) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.duplicate-statistics");
  descriptor.info = DuplicateStatisticsPassInfo;
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsDuplicateStatisticOffsets) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.duplicate-statistic-offsets");
  descriptor.info = DuplicateStatisticOffsetsPassInfo;
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsUnalignedStatisticFields) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.unaligned-statistics");
  descriptor.info = UnalignedStatisticsPassInfo;
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsTooManyStatisticFields) {
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.key = IREE_SVL("test.too-many-statistics");
  descriptor.info = TooManyStatisticsPassInfo;
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, VerifiesRequirementMetadata) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          /*.capability_type=*/&kTestRequirementType,
          /*.key=*/IREE_SVL("analysis.liveness"),
          /*.description=*/IREE_SVL("Requires precomputed liveness."),
      },
      {
          /*.capability_type=*/&kTestRequirementType,
          /*.key=*/IREE_SVL("target.low-descriptor-registry"),
          /*.description=*/
          IREE_SVL("Requires a target-low descriptor registry."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_ASSERT_OK(loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsRequirementWithoutCapabilityType) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          /*.capability_type=*/{},
          /*.key=*/IREE_SVL("analysis.liveness"),
          /*.description=*/IREE_SVL("Requires precomputed liveness."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, RejectsUnsortedRequirementMetadata) {
  const loom_pass_requirement_def_t requirements[] = {
      {
          /*.capability_type=*/&kTestRequirementType,
          /*.key=*/IREE_SVL("target.low-descriptor-registry"),
          /*.description=*/
          IREE_SVL("Requires a target-low descriptor registry."),
      },
      {
          /*.capability_type=*/&kTestRequirementType,
          /*.key=*/IREE_SVL("analysis.liveness"),
          /*.description=*/IREE_SVL("Requires precomputed liveness."),
      },
  };
  loom_pass_descriptor_t descriptor =
      *LookupTestPass(IREE_SV("test.module-noop"));
  descriptor.requirement_defs = requirements;
  descriptor.requirement_count = IREE_ARRAYSIZE(requirements);
  const loom_pass_registry_t registry = {
      /*.descriptors=*/&descriptor,
      /*.descriptor_count=*/1,
  };

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_registry_verify(&registry));
}

TEST(PassRegistryCoreTest, LookupKnownAndUnknownPasses) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      loom_test_pass_registry(), IREE_SV("test.options"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->key, IREE_SV("test.options")));
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
  EXPECT_NE(descriptor->function_run, nullptr);
  EXPECT_NE(descriptor->create, nullptr);

  descriptor = reinterpret_cast<const loom_pass_descriptor_t*>(0x1);
  IREE_ASSERT_OK(loom_pass_registry_lookup(loom_test_pass_registry(),
                                           IREE_SV("definitely-not-a-pass"),
                                           &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST(PassRegistryCoreTest, ValidatesUint32OptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=4")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=0")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("count=many")));
}

TEST(PassRegistryCoreTest, ValidatesEnumAndStringOptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("mode=alpha,string=payload")));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("mode=gamma")));
}

TEST(PassRegistryCoreTest, ValidatesRequiredOptionSchema) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.required"));
  ASSERT_NE(descriptor, nullptr);

  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      descriptor, IREE_SV("required=value")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, iree_string_view_empty()));
}

TEST(PassRegistryCoreTest, RejectsUnknownAndDuplicateOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("unknown=1")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            descriptor, IREE_SV("count=1,count=2")));
}

TEST(PassRegistryCoreTest, DecodesTypedOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, IREE_SV("count=7,mode=beta,string=payload"), &arena,
      &decoded_options));

  EXPECT_EQ(decoded_options.descriptor, descriptor);
  ASSERT_EQ(decoded_options.option_count, 3u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_TRUE(decoded_options.options[0].present);
  EXPECT_EQ(decoded_options.options[0].uint32_value, 7u);
  EXPECT_TRUE(decoded_options.options[1].present);
  EXPECT_EQ(decoded_options.options[1].enum_value_index, 1u);
  EXPECT_TRUE(decoded_options.options[2].present);
  EXPECT_TRUE(iree_string_view_equal(decoded_options.options[2].string_value,
                                     IREE_SV("payload")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, DecodedOptionalOptionsTrackAbsence) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_ASSERT_OK(loom_pass_descriptor_decode_options(
      descriptor, iree_string_view_empty(), &arena, &decoded_options));

  ASSERT_EQ(decoded_options.option_count, 3u);
  ASSERT_NE(decoded_options.options, nullptr);
  EXPECT_FALSE(decoded_options.options[0].present);
  EXPECT_FALSE(decoded_options.options[1].present);
  EXPECT_FALSE(decoded_options.options[2].present);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, DecodeRejectsInvalidOptions) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.options"));
  ASSERT_NE(descriptor, nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(/*block_size=*/4096, iree_allocator_system(),
                                   &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_pass_decoded_options_t decoded_options = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_decode_options(
          descriptor, IREE_SV("count=1,count=2"), &arena, &decoded_options));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(PassRegistryCoreTest, RejectsOptionsForDescriptorWithoutCreateCallback) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.module-noop"));
  ASSERT_NE(descriptor, nullptr);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_descriptor_validate_options(descriptor, IREE_SV("unknown=1")));
}

TEST(PassRegistryCoreTest, UnavailableDescriptorKeepsAvailabilityMetadata) {
  const loom_pass_descriptor_t* descriptor =
      LookupTestPass(IREE_SV("test.unavailable"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_FALSE(loom_pass_descriptor_is_available(descriptor));
  EXPECT_TRUE(iree_string_view_equal(descriptor->unavailable_reason,
                                     IREE_SV("disabled for test")));
}

}  // namespace
}  // namespace loom
