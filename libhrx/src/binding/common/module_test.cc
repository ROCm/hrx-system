// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/module.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FakeFunction {
  std::string name;
  iree_hal_executable_function_info_t info = {};
  std::vector<iree_hal_executable_function_parameter_t> parameters;
};

// Metadata-only executable whose storage follows the real HAL resource and
// vtable contracts. Tests vary reflected records while the production module
// extractor owns all parsing, validation, and allocation behavior.
class FakeExecutable {
 public:
  FakeExecutable() { iree_hal_resource_initialize(&kVtable, &resource_); }

  FakeExecutable(const FakeExecutable&) = delete;
  FakeExecutable& operator=(const FakeExecutable&) = delete;

  iree_hal_executable_t* base() {
    return reinterpret_cast<iree_hal_executable_t*>(this);
  }

  void AddFunction(
      std::string name, uint32_t constant_byte_length, uint16_t binding_count,
      std::vector<iree_hal_executable_function_parameter_t> parameters) {
    ASSERT_LE(parameters.size(), UINT16_MAX);
    FakeFunction function;
    function.name = std::move(name);
    function.info.constant_byte_length = constant_byte_length;
    function.info.binding_count = binding_count;
    function.info.parameter_count = static_cast<uint16_t>(parameters.size());
    function.parameters = std::move(parameters);
    functions_.push_back(std::move(function));
  }

  void OverrideFunctionCount(iree_host_size_t function_count) {
    has_function_count_override_ = true;
    function_count_override_ = function_count;
  }

 private:
  static FakeExecutable* Cast(iree_hal_executable_t* base_executable) {
    IREE_HAL_ASSERT_TYPE(base_executable, &kVtable);
    return reinterpret_cast<FakeExecutable*>(base_executable);
  }

  static void Destroy(iree_hal_executable_t* base_executable) {
    (void)base_executable;
  }

  static iree_host_size_t FunctionCount(
      iree_hal_executable_t* base_executable) {
    FakeExecutable* executable = Cast(base_executable);
    return executable->has_function_count_override_
               ? executable->function_count_override_
               : executable->functions_.size();
  }

  static iree_status_t FunctionInfo(
      iree_hal_executable_t* base_executable,
      iree_hal_executable_function_t function,
      iree_hal_executable_function_info_t* out_info) {
    FakeExecutable* executable = Cast(base_executable);
    if (!iree_hal_executable_function_is_index_in_range(
            function, executable->functions_.size())) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
    }
    FakeFunction& fake_function =
        executable->functions_[iree_hal_executable_function_index(function)];
    *out_info = fake_function.info;
    out_info->name = iree_make_string_view(fake_function.name.data(),
                                           fake_function.name.size());
    return iree_ok_status();
  }

  static iree_status_t FunctionParameters(
      iree_hal_executable_t* base_executable,
      iree_hal_executable_function_t function, iree_host_size_t capacity,
      iree_hal_executable_function_parameter_t* out_parameters) {
    FakeExecutable* executable = Cast(base_executable);
    if (!iree_hal_executable_function_is_index_in_range(
            function, executable->functions_.size())) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
    }
    const FakeFunction& fake_function =
        executable->functions_[iree_hal_executable_function_index(function)];
    if (capacity < fake_function.parameters.size() ||
        (!out_parameters && !fake_function.parameters.empty())) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
    }
    if (!fake_function.parameters.empty()) {
      memcpy(out_parameters, fake_function.parameters.data(),
             fake_function.parameters.size() * sizeof(*out_parameters));
    }
    return iree_ok_status();
  }

  static iree_status_t LookupFunctionByName(
      iree_hal_executable_t* base_executable, iree_string_view_t name,
      iree_hal_executable_function_t* out_function) {
    FakeExecutable* executable = Cast(base_executable);
    for (iree_host_size_t i = 0; i < executable->functions_.size(); ++i) {
      const FakeFunction& function = executable->functions_[i];
      if (iree_string_view_equal(
              iree_make_string_view(function.name.data(), function.name.size()),
              name)) {
        *out_function =
            iree_hal_executable_function_from_index(static_cast<uint32_t>(i));
        return iree_ok_status();
      }
    }
    *out_function = iree_hal_executable_function_invalid();
    return iree_make_status(IREE_STATUS_NOT_FOUND);
  }

  static iree_status_t TryLookupGlobalByName(
      iree_hal_executable_t* base_executable, iree_string_view_t name,
      bool* out_found, iree_hal_executable_global_t* out_global) {
    (void)base_executable;
    (void)name;
    *out_found = false;
    *out_global = iree_hal_executable_global_invalid();
    return iree_ok_status();
  }

  static iree_status_t GlobalInfo(iree_hal_executable_t* base_executable,
                                  iree_hal_executable_global_t global,
                                  iree_hal_executable_global_info_t* out_info) {
    (void)base_executable;
    (void)global;
    memset(out_info, 0, sizeof(*out_info));
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  static iree_status_t GlobalBuffer(iree_hal_executable_t* base_executable,
                                    iree_hal_executable_global_t global,
                                    iree_hal_queue_affinity_t queue_affinity,
                                    iree_hal_buffer_t** out_buffer) {
    (void)base_executable;
    (void)global;
    (void)queue_affinity;
    *out_buffer = nullptr;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  static const iree_hal_executable_vtable_t kVtable;

  // Must be first so the HAL may cast between the resource and executable.
  iree_hal_resource_t resource_;
  std::vector<FakeFunction> functions_;
  bool has_function_count_override_ = false;
  iree_host_size_t function_count_override_ = 0;
};

const iree_hal_executable_vtable_t FakeExecutable::kVtable = {
    /*.destroy=*/FakeExecutable::Destroy,
    /*.function_count=*/FakeExecutable::FunctionCount,
    /*.function_info=*/FakeExecutable::FunctionInfo,
    /*.function_parameters=*/FakeExecutable::FunctionParameters,
    /*.lookup_function_by_name=*/FakeExecutable::LookupFunctionByName,
    /*.try_lookup_global_by_name=*/FakeExecutable::TryLookupGlobalByName,
    /*.global_info=*/FakeExecutable::GlobalInfo,
    /*.global_buffer=*/FakeExecutable::GlobalBuffer,
};

iree_hal_executable_function_parameter_t MakeParameter(
    iree_hal_executable_function_parameter_type_t type, uint16_t size,
    uint16_t offset, uint16_t native_abi_offset) {
  return iree_hal_executable_function_parameter_t{
      /*.type=*/type,
      /*.flags=*/
      IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET,
      /*.size=*/size,
      /*.offset=*/offset,
      /*.native_abi_offset=*/native_abi_offset,
      /*.name=*/iree_string_view_empty(),
  };
}

class ModuleMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    options.identifier = IREE_SV("module-metadata-test");
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &options, iree_allocator_system(), &device_));
    context_.device = device_;
    module_.context = &context_;
    module_.executable = executable_.base();
    module_.host_allocator = iree_allocator_system();
  }

  void TearDown() override {
    iree_allocator_free(module_.host_allocator, module_.symbols);
    iree_hal_device_release(device_);
  }

  FakeExecutable executable_;
  FakeExecutable second_executable_;
  iree_hal_device_t* device_ = nullptr;
  iree_hal_streaming_context_t context_ = {};
  iree_hal_streaming_module_t module_ = {};
};

TEST_F(ModuleMetadataTest, AcceptsZeroSymbolsWithoutAllocatingStorage) {
  IREE_ASSERT_OK(iree_hal_streaming_module_extract_metadata(&module_));
  EXPECT_EQ(0u, module_.symbol_count);
  EXPECT_EQ(nullptr, module_.symbols);
}

TEST_F(ModuleMetadataTest, MergesSymbolsFromMultipleOwnedExecutables) {
  executable_.AddFunction(
      "first_kernel", /*constant_byte_length=*/1, /*binding_count=*/0,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                     /*size=*/1, /*offset=*/0, /*native_abi_offset=*/0)});
  second_executable_.AddFunction(
      "second_kernel", /*constant_byte_length=*/1, /*binding_count=*/0,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                     /*size=*/1, /*offset=*/0, /*native_abi_offset=*/0)});
  std::array<iree_hal_executable_t*, 2> executables = {
      executable_.base(),
      second_executable_.base(),
  };
  module_.executables = executables.data();
  module_.executable_count = executables.size();

  IREE_ASSERT_OK(iree_hal_streaming_module_extract_metadata(&module_));
  ASSERT_EQ(2u, module_.symbol_count);
  EXPECT_EQ(executable_.base(), module_.symbols[0].executable);
  EXPECT_EQ(second_executable_.base(), module_.symbols[1].executable);
  EXPECT_EQ(module_.symbols[0].parameters.ops + 1,
            module_.symbols[1].parameters.ops);
}

TEST_F(ModuleMetadataTest, OwnsStableContiguousSymbolsAndOperations) {
  executable_.AddFunction(
      "mixed_kernel", /*constant_byte_length=*/6, /*binding_count=*/2,
      {
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/4, /*offset=*/0, /*native_abi_offset=*/0),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/1, /*native_abi_offset=*/8),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/2, /*offset=*/4, /*native_abi_offset=*/16),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/0, /*native_abi_offset=*/24),
      });
  executable_.AddFunction(
      "tail_kernel", /*constant_byte_length=*/1, /*binding_count=*/0,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                     /*size=*/1, /*offset=*/0, /*native_abi_offset=*/0)});

  IREE_ASSERT_OK(iree_hal_streaming_module_extract_metadata(&module_));
  ASSERT_EQ(2u, module_.symbol_count);
  ASSERT_NE(nullptr, module_.symbols);

  const iree_hal_streaming_symbol_t& mixed = module_.symbols[0];
  EXPECT_EQ(22u, mixed.parameters.buffer_size);
  EXPECT_EQ(6u, mixed.parameters.constant_bytes);
  EXPECT_EQ(32u, mixed.parameters.direct_arg_bytes);
  EXPECT_EQ(2u, mixed.parameters.copy_count);
  EXPECT_EQ(2u, mixed.parameters.binding_count);

  const iree_hal_streaming_parameter_op_t* copy_ops = mixed.parameters.ops;
  const iree_hal_streaming_parameter_op_t* resolve_ops = copy_ops + 2;
  EXPECT_EQ(0u, copy_ops[0].copy.source_ordinal);
  EXPECT_EQ(2u, copy_ops[1].copy.source_ordinal);
  EXPECT_EQ(1u, resolve_ops[0].resolve.source_ordinal);
  EXPECT_EQ(1u, resolve_ops[0].resolve.destination_ordinal);
  EXPECT_EQ(3u, resolve_ops[1].resolve.source_ordinal);
  EXPECT_EQ(0u, resolve_ops[1].resolve.destination_ordinal);

  iree_host_size_t expected_ops_offset = 0;
  ASSERT_TRUE(iree_host_size_checked_align(
      module_.symbol_count * sizeof(*module_.symbols),
      alignof(iree_hal_streaming_parameter_op_t), &expected_ops_offset));
  EXPECT_EQ(reinterpret_cast<uint8_t*>(module_.symbols) + expected_ops_offset,
            reinterpret_cast<const uint8_t*>(mixed.parameters.ops));
  EXPECT_EQ(mixed.parameters.ops + 4, module_.symbols[1].parameters.ops);

  iree_hal_streaming_symbol_t* found_symbol = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_module_function(
      &module_, "  mixed_kernel.kd  ", &found_symbol));
  EXPECT_EQ(&module_.symbols[0], found_symbol);
}

TEST_F(ModuleMetadataTest, RejectsDuplicateBindingOrdinals) {
  executable_.AddFunction(
      "duplicate_binding", /*constant_byte_length=*/0, /*binding_count=*/2,
      {
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/0, /*native_abi_offset=*/0),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/0, /*native_abi_offset=*/8),
      });

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
  // Permanent storage is module-owned as soon as allocation succeeds, even
  // when later metadata validation rejects the module.
  EXPECT_NE(nullptr, module_.symbols);
}

TEST_F(ModuleMetadataTest, RejectsOutOfRangeBindingOrdinal) {
  executable_.AddFunction(
      "out_of_range_binding", /*constant_byte_length=*/0,
      /*binding_count=*/1,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                     /*size=*/8, /*offset=*/1, /*native_abi_offset=*/0)});

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, RejectsBindingCountMismatch) {
  executable_.AddFunction(
      "binding_count_mismatch", /*constant_byte_length=*/0,
      /*binding_count=*/2,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                     /*size=*/8, /*offset=*/0, /*native_abi_offset=*/0)});

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, RejectsNonDevicePointerBindingWidth) {
  executable_.AddFunction(
      "short_binding", /*constant_byte_length=*/0, /*binding_count=*/1,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                     /*size=*/4, /*offset=*/0, /*native_abi_offset=*/0)});

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, NormalizesResolveOnlyOperationsBySourceOrdinal) {
  executable_.AddFunction(
      "resolve_only", /*constant_byte_length=*/0, /*binding_count=*/2,
      {
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/1, /*native_abi_offset=*/0),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
                        /*size=*/8, /*offset=*/0, /*native_abi_offset=*/8),
      });

  IREE_ASSERT_OK(iree_hal_streaming_module_extract_metadata(&module_));
  const iree_hal_streaming_parameter_info_t& parameters =
      module_.symbols[0].parameters;
  ASSERT_EQ(0u, parameters.copy_count);
  ASSERT_EQ(2u, parameters.binding_count);
  EXPECT_EQ(0u, parameters.ops[0].resolve.source_ordinal);
  EXPECT_EQ(1u, parameters.ops[0].resolve.destination_ordinal);
  EXPECT_EQ(1u, parameters.ops[1].resolve.source_ordinal);
  EXPECT_EQ(0u, parameters.ops[1].resolve.destination_ordinal);
}

TEST_F(ModuleMetadataTest, RejectsConstantCountBeforeNarrowing) {
  executable_.AddFunction("wide_constants",
                          /*constant_byte_length=*/UINT16_MAX + 1u,
                          /*binding_count=*/0, {});

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, RejectsOverlappingNativeAbiExtents) {
  executable_.AddFunction(
      "overlapping_native_layout", /*constant_byte_length=*/8,
      /*binding_count=*/0,
      {
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/4, /*offset=*/0, /*native_abi_offset=*/8),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/4, /*offset=*/4, /*native_abi_offset=*/10),
      });

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, RejectsConstantDestinationOutsideReflectedRange) {
  executable_.AddFunction(
      "constant_range", /*constant_byte_length=*/4, /*binding_count=*/0,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                     /*size=*/4, /*offset=*/2, /*native_abi_offset=*/0)});

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, RejectsSourceAndNativeLayoutOverflow) {
  executable_.AddFunction(
      "overflowing_layout", /*constant_byte_length=*/UINT16_MAX,
      /*binding_count=*/0,
      {
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/UINT16_MAX, /*offset=*/0,
                        /*native_abi_offset=*/0),
          MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                        /*size=*/1, /*offset=*/0,
                        /*native_abi_offset=*/UINT16_MAX),
      });

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_streaming_module_extract_metadata(&module_));
}

TEST_F(ModuleMetadataTest, AcceptsMaximumRepresentableParameterCount) {
  std::vector<iree_hal_executable_function_parameter_t> parameters;
  parameters.reserve(UINT16_MAX);
  for (uint32_t i = 0; i < UINT16_MAX; ++i) {
    parameters.push_back(
        MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                      /*size=*/1, /*offset=*/static_cast<uint16_t>(i),
                      /*native_abi_offset=*/static_cast<uint16_t>(i)));
  }
  executable_.AddFunction("maximum_parameter_count",
                          /*constant_byte_length=*/UINT16_MAX,
                          /*binding_count=*/0, std::move(parameters));

  IREE_ASSERT_OK(iree_hal_streaming_module_extract_metadata(&module_));
  ASSERT_EQ(1u, module_.symbol_count);
  EXPECT_EQ(UINT16_MAX, module_.symbols[0].parameters.buffer_size);
  EXPECT_EQ(UINT16_MAX, module_.symbols[0].parameters.constant_bytes);
  EXPECT_EQ(UINT16_MAX, module_.symbols[0].parameters.direct_arg_bytes);
  EXPECT_EQ(UINT16_MAX, module_.symbols[0].parameters.copy_count);
  EXPECT_EQ(0u, module_.symbols[0].parameters.binding_count);
}

TEST_F(ModuleMetadataTest, RejectsFunctionCountBeforeOrdinalNarrowing) {
  if (sizeof(iree_host_size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "host-size function counts cannot exceed uint32_t";
  }
  executable_.OverrideFunctionCount(static_cast<iree_host_size_t>(UINT32_MAX) +
                                    1);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_streaming_module_extract_metadata(&module_));
  EXPECT_EQ(nullptr, module_.symbols);
}

struct FailingAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  int fail_on_allocation = 0;
  int allocation_attempt_count = 0;
  int successful_allocation_count = 0;
  int free_count = 0;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<FailingAllocator*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
        command == IREE_ALLOCATOR_COMMAND_CALLOC ||
        command == IREE_ALLOCATOR_COMMAND_REALLOC) {
      ++allocator->allocation_attempt_count;
      if (allocator->allocation_attempt_count ==
          allocator->fail_on_allocation) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
      ++allocator->successful_allocation_count;
    } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->free_count;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t AsAllocator() {
    return iree_allocator_t{this, &FailingAllocator::Control};
  }
};

TEST_F(ModuleMetadataTest, FreesScratchWhenPermanentStorageAllocationFails) {
  executable_.AddFunction(
      "allocation_failure", /*constant_byte_length=*/4, /*binding_count=*/0,
      {MakeParameter(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
                     /*size=*/4, /*offset=*/0, /*native_abi_offset=*/0)});
  FailingAllocator allocator;
  allocator.fail_on_allocation = 3;
  module_.host_allocator = allocator.AsAllocator();

  iree_status_t status = iree_hal_streaming_module_extract_metadata(&module_);
  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(3, allocator.allocation_attempt_count);
  EXPECT_EQ(2, allocator.successful_allocation_count);
  EXPECT_EQ(2, allocator.free_count);
  EXPECT_EQ(nullptr, module_.symbols);

  // The fixture teardown must never outlive allocator callback state.
  module_.host_allocator = iree_allocator_system();
}

}  // namespace
