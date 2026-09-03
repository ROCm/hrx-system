// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/registry.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static void DestroyProduct(loom_product_t* product) { (void)product; }

static const loom_product_descriptor_t kKernelProductDescriptor = {
    /*.name=*/IREE_SV("kernel"),
    /*.destroy=*/DestroyProduct,
};

static const loom_product_descriptor_t kCommandProductDescriptor = {
    /*.name=*/IREE_SV("command"),
    /*.destroy=*/DestroyProduct,
};

static const iree_string_view_t kKernelRootOperationNames[] = {
    IREE_SV("kernel.def"),
    IREE_SV("low.kernel.def"),
};

static const loom_product_operation_t kKernelOperation = {
    /*.name=*/IREE_SV("kernel"),
    /*.product_descriptor=*/&kKernelProductDescriptor,
    /*.root_operation_names=*/kKernelRootOperationNames,
    /*.root_operation_name_count=*/IREE_ARRAYSIZE(kKernelRootOperationNames),
};

static const iree_string_view_t kCommandRootOperationNames[] = {
    IREE_SV("command.program.def"),
};

static const loom_product_operation_t kCommandOperation = {
    /*.name=*/IREE_SV("command"),
    /*.product_descriptor=*/&kCommandProductDescriptor,
    /*.root_operation_names=*/kCommandRootOperationNames,
    /*.root_operation_name_count=*/IREE_ARRAYSIZE(kCommandRootOperationNames),
};

static const loom_product_artifact_schema_t kAmdgpuHsacoArtifactSchemas[] = {
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
        /*.format=*/IREE_SV("amdgpu-hsaco"),
        /*.minimum_count=*/1,
        /*.maximum_count=*/1,
    },
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_LISTING),
        /*.format=*/IREE_SV("amdgpu-assembly"),
        /*.minimum_count=*/0,
        /*.maximum_count=*/1,
    },
};

static const loom_product_format_t kAmdgpuHsacoFormat = {
    /*.operation=*/&kKernelOperation,
    /*.name=*/IREE_SV("amdgpu-hsaco"),
    /*.persistence=*/LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    /*.single_file=*/
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
        /*.extension=*/IREE_SV(".hsaco"),
    },
    /*.artifact_schemas=*/kAmdgpuHsacoArtifactSchemas,
    /*.artifact_schema_count=*/IREE_ARRAYSIZE(kAmdgpuHsacoArtifactSchemas),
};

static const loom_product_artifact_schema_t kLlvmirTextArtifactSchemas[] = {
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
        /*.format=*/IREE_SV("llvmir-text"),
        /*.minimum_count=*/1,
        /*.maximum_count=*/1,
    },
};

static const loom_product_format_t kLlvmirTextFormat = {
    /*.operation=*/&kKernelOperation,
    /*.name=*/IREE_SV("llvmir-text"),
    /*.persistence=*/LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE,
    /*.single_file=*/
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
        /*.extension=*/IREE_SV(".ll"),
    },
    /*.artifact_schemas=*/kLlvmirTextArtifactSchemas,
    /*.artifact_schema_count=*/IREE_ARRAYSIZE(kLlvmirTextArtifactSchemas),
};

static const loom_product_artifact_schema_t kCommandArtifactSchemas[] = {
    {
        /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_COMMAND_PROGRAM),
        /*.format=*/IREE_SV("loom-command"),
        /*.minimum_count=*/1,
        /*.maximum_count=*/IREE_HOST_SIZE_MAX,
    },
};

static const loom_product_format_t kCommandFormat = {
    /*.operation=*/&kCommandOperation,
    /*.name=*/IREE_SV("loom-command"),
    /*.persistence=*/LOOM_PRODUCT_PERSISTENCE_ARTIFACT_SET,
    /*.single_file=*/{},
    /*.artifact_schemas=*/kCommandArtifactSchemas,
    /*.artifact_schema_count=*/IREE_ARRAYSIZE(kCommandArtifactSchemas),
};

static const loom_target_profile_type_t kAmdgpuProfileType = {
    /*.name=*/IREE_SV("amdgpu"),
};

static const loom_target_profile_type_t kSpirvProfileType = {
    /*.name=*/IREE_SV("spirv"),
};

static const loom_target_profile_type_t kX86ProfileType = {
    /*.name=*/IREE_SV("x86"),
};

static const loom_target_product_contract_t kAmdgpuHsacoContract = {
    /*.name=*/IREE_SV("amdgpu-hsaco"),
    /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
    /*.linkage=*/LOOM_TARGET_LINKAGE_DEFAULT,
};

static const loom_target_product_contract_t kLlvmirTextContract = {
    /*.name=*/IREE_SV("llvmir-text"),
    /*.codegen_format=*/LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    /*.artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_LLVMIR_TEXT,
    /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
    /*.linkage=*/LOOM_TARGET_LINKAGE_DEFAULT,
};

static iree_status_t BuildProduct(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  (void)provider;
  (void)request;
  *out_product = nullptr;
  return iree_ok_status();
}

static bool AcceptAllTargets(const loom_product_format_provider_t* provider,
                             const loom_target_profile_t* profile) {
  (void)provider;
  (void)profile;
  return true;
}

static const loom_product_format_provider_t kAmdgpuHsacoProvider = {
    /*.name=*/IREE_SV("amdgpu-native-hsaco"),
    /*.operation=*/&kKernelOperation,
    /*.format=*/&kAmdgpuHsacoFormat,
    /*.target_profile_type=*/&kAmdgpuProfileType,
    /*.accepts_target=*/nullptr,
    /*.flags=*/LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    /*.target_product_contract=*/&kAmdgpuHsacoContract,
    /*.default_pipeline_options=*/nullptr,
    /*.build=*/BuildProduct,
};

static const loom_product_format_provider_t kAmdgpuLlvmirTextProvider = {
    /*.name=*/IREE_SV("amdgpu-llvmir-text"),
    /*.operation=*/&kKernelOperation,
    /*.format=*/&kLlvmirTextFormat,
    /*.target_profile_type=*/&kAmdgpuProfileType,
    /*.accepts_target=*/nullptr,
    /*.flags=*/0,
    /*.target_product_contract=*/&kLlvmirTextContract,
    /*.default_pipeline_options=*/nullptr,
    /*.build=*/BuildProduct,
};

static const loom_product_format_provider_t kX86LlvmirTextProvider = {
    /*.name=*/IREE_SV("x86-llvmir-text"),
    /*.operation=*/&kKernelOperation,
    /*.format=*/&kLlvmirTextFormat,
    /*.target_profile_type=*/&kX86ProfileType,
    /*.accepts_target=*/nullptr,
    /*.flags=*/LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    /*.target_product_contract=*/&kLlvmirTextContract,
    /*.default_pipeline_options=*/nullptr,
    /*.build=*/BuildProduct,
};

static const loom_product_format_provider_t kCommandProvider = {
    /*.name=*/IREE_SV("portable-command"),
    /*.operation=*/&kCommandOperation,
    /*.format=*/&kCommandFormat,
    /*.target_profile_type=*/nullptr,
    /*.accepts_target=*/nullptr,
    /*.flags=*/LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL,
    /*.target_product_contract=*/nullptr,
    /*.default_pipeline_options=*/nullptr,
    /*.build=*/BuildProduct,
};

static const loom_product_operation_t* const kOperations[] = {
    &kKernelOperation,
    &kCommandOperation,
};

static const loom_product_format_t* const kFormats[] = {
    &kAmdgpuHsacoFormat,
    &kLlvmirTextFormat,
    &kCommandFormat,
};

static const loom_product_format_provider_t* const kProviders[] = {
    &kAmdgpuHsacoProvider,
    &kAmdgpuLlvmirTextProvider,
    &kX86LlvmirTextProvider,
    &kCommandProvider,
};

static const loom_product_registry_t kRegistry = {
    /*.operations=*/
    {
        /*.values=*/kOperations,
        /*.count=*/IREE_ARRAYSIZE(kOperations),
    },
    /*.formats=*/
    {
        /*.values=*/kFormats,
        /*.count=*/IREE_ARRAYSIZE(kFormats),
    },
    /*.providers=*/
    {
        /*.values=*/kProviders,
        /*.count=*/IREE_ARRAYSIZE(kProviders),
    },
};

TEST(ProductRegistryTest, ValidatesAndLooksUpOpenIdentities) {
  IREE_EXPECT_OK(loom_product_registry_validate(&kRegistry));
  EXPECT_EQ(
      loom_product_registry_lookup_operation(&kRegistry, IREE_SV("kernel")),
      &kKernelOperation);
  EXPECT_EQ(
      loom_product_registry_lookup_operation(&kRegistry, IREE_SV("missing")),
      nullptr);
  EXPECT_EQ(loom_product_registry_lookup_root_operation(&kRegistry,
                                                        IREE_SV("kernel.def")),
            &kKernelOperation);
  EXPECT_EQ(loom_product_registry_lookup_root_operation(
                &kRegistry, IREE_SV("low.kernel.def")),
            &kKernelOperation);
  EXPECT_EQ(loom_product_registry_lookup_root_operation(
                &kRegistry, IREE_SV("command.program.def")),
            &kCommandOperation);
  EXPECT_EQ(loom_product_registry_lookup_format(&kRegistry, &kKernelOperation,
                                                IREE_SV("llvmir-text")),
            &kLlvmirTextFormat);
  EXPECT_EQ(loom_product_registry_lookup_format(&kRegistry, &kCommandOperation,
                                                IREE_SV("llvmir-text")),
            nullptr);
}

TEST(ProductRegistryTest, SelectsCanonicalAndExplicitFormats) {
  const loom_target_profile_t amdgpu_profile = {
      /*.type=*/&kAmdgpuProfileType,
  };
  const loom_product_format_provider_t* provider = nullptr;
  IREE_ASSERT_OK(loom_product_registry_select_provider(
      &kRegistry, &kKernelOperation, iree_string_view_empty(), &amdgpu_profile,
      &provider));
  EXPECT_EQ(provider, &kAmdgpuHsacoProvider);

  IREE_ASSERT_OK(loom_product_registry_select_provider(
      &kRegistry, &kKernelOperation, IREE_SV("llvmir-text"), &amdgpu_profile,
      &provider));
  EXPECT_EQ(provider, &kAmdgpuLlvmirTextProvider);

  IREE_ASSERT_OK(loom_product_registry_select_provider(
      &kRegistry, &kCommandOperation, iree_string_view_empty(), nullptr,
      &provider));
  EXPECT_EQ(provider, &kCommandProvider);

  const loom_target_profile_t x86_profile = {
      /*.type=*/&kX86ProfileType,
  };
  IREE_ASSERT_OK(loom_product_registry_select_provider(
      &kRegistry, &kKernelOperation, iree_string_view_empty(), &x86_profile,
      &provider));
  EXPECT_EQ(provider, &kX86LlvmirTextProvider);
}

TEST(ProductRegistryTest, RejectsUnsupportedTargetAndFormatTuples) {
  const loom_target_profile_t spirv_profile = {
      /*.type=*/&kSpirvProfileType,
  };
  const loom_product_format_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_product_registry_select_provider(&kRegistry, &kKernelOperation,
                                            IREE_SV("amdgpu-hsaco"),
                                            &spirv_profile, &provider));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_product_registry_select_provider(
                            &kRegistry, &kKernelOperation, IREE_SV("missing"),
                            &spirv_profile, &provider));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_product_registry_select_provider(&kRegistry, &kCommandOperation,
                                            iree_string_view_empty(),
                                            &spirv_profile, &provider));
}

TEST(ProductRegistryTest, RejectsAmbiguousCanonicalFormats) {
  loom_product_format_provider_t first_canonical = kAmdgpuHsacoProvider;
  first_canonical.accepts_target = AcceptAllTargets;
  loom_product_format_provider_t second_canonical = kAmdgpuLlvmirTextProvider;
  second_canonical.name = IREE_SV("amdgpu-llvmir-text-canonical");
  second_canonical.accepts_target = AcceptAllTargets;
  second_canonical.flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL;
  const loom_product_format_provider_t* providers[] = {
      &first_canonical,
      &second_canonical,
      &kCommandProvider,
  };
  loom_product_registry_t registry = kRegistry;
  registry.providers.values = providers;
  registry.providers.count = IREE_ARRAYSIZE(providers);
  IREE_ASSERT_OK(loom_product_registry_validate(&registry));

  const loom_target_profile_t amdgpu_profile = {
      /*.type=*/&kAmdgpuProfileType,
  };
  const loom_product_format_provider_t* provider = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_product_registry_select_provider(&registry, &kKernelOperation,
                                            iree_string_view_empty(),
                                            &amdgpu_profile, &provider));
}

TEST(ProductRegistryTest, RejectsUnconditionalCanonicalOverlap) {
  loom_product_format_provider_t second_canonical = kAmdgpuLlvmirTextProvider;
  second_canonical.name = IREE_SV("amdgpu-llvmir-text-canonical");
  second_canonical.flags = LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL;
  const loom_product_format_provider_t* providers[] = {
      &kAmdgpuHsacoProvider,
      &second_canonical,
      &kCommandProvider,
  };
  loom_product_registry_t registry = kRegistry;
  registry.providers.values = providers;
  registry.providers.count = IREE_ARRAYSIZE(providers);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_product_registry_validate(&registry));
}

TEST(ProductRegistryTest, RejectsDuplicateRootOwnership) {
  const iree_string_view_t duplicate_root_names[] = {
      IREE_SV("kernel.def"),
  };
  loom_product_operation_t duplicate_operation = kCommandOperation;
  duplicate_operation.root_operation_names = duplicate_root_names;
  duplicate_operation.root_operation_name_count =
      IREE_ARRAYSIZE(duplicate_root_names);
  const loom_product_operation_t* operations[] = {
      &kKernelOperation,
      &duplicate_operation,
  };
  loom_product_registry_t registry = kRegistry;
  registry.operations.values = operations;
  registry.operations.count = IREE_ARRAYSIZE(operations);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_product_registry_validate(&registry));
}

TEST(ProductRegistryTest, RejectsMalformedSingleFileSchema) {
  loom_product_format_t format = kAmdgpuHsacoFormat;
  format.single_file.role = IREE_SV("missing");
  const loom_product_format_t* formats[] = {
      &format,
      &kLlvmirTextFormat,
      &kCommandFormat,
  };
  loom_product_format_provider_t provider = kAmdgpuHsacoProvider;
  provider.format = &format;
  const loom_product_format_provider_t* providers[] = {
      &provider,
      &kAmdgpuLlvmirTextProvider,
      &kCommandProvider,
  };
  loom_product_registry_t registry = kRegistry;
  registry.formats.values = formats;
  registry.formats.count = IREE_ARRAYSIZE(formats);
  registry.providers.values = providers;
  registry.providers.count = IREE_ARRAYSIZE(providers);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_product_registry_validate(&registry));
}

TEST(ProductRegistryTest, ValidatesProductArtifactSchema) {
  static const uint8_t kData[] = {1, 2, 3, 4};
  iree_byte_span_t data = iree_byte_span_empty();
  data.data_length = sizeof(kData);
  IREE_ASSERT_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), data.data_length, (void**)&data.data));
  memcpy(data.data, kData, sizeof(kData));
  iree_byte_sequence_t* contents = nullptr;
  IREE_ASSERT_OK(iree_byte_sequence_create_from_span_move(
      &data, iree_allocator_system(), &contents));
  loom_product_artifact_t artifacts[] = {
      {
          /*.role=*/IREE_SV(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
          /*.format=*/IREE_SV("amdgpu-hsaco"),
          /*.identifier=*/IREE_SV("kernel.hsaco"),
          /*.contents=*/contents,
      },
  };
  loom_product_t product = {0};
  loom_product_initialize(&kKernelProductDescriptor, artifacts,
                          IREE_ARRAYSIZE(artifacts), /*export_count=*/1,
                          /*requirement_count=*/0, &product);
  IREE_EXPECT_OK(
      loom_product_format_validate_product(&kAmdgpuHsacoFormat, &product));

  artifacts[0].format = IREE_SV("elf");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_product_format_validate_product(&kAmdgpuHsacoFormat, &product));
  artifacts[0].format = IREE_SV("amdgpu-hsaco");
  artifacts[0].role = IREE_SV("undeclared");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_product_format_validate_product(&kAmdgpuHsacoFormat, &product));

  iree_byte_sequence_release(contents);
}

}  // namespace
}  // namespace loom
