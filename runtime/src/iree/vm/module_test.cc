// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module_test_provider.h"

namespace {

std::string_view ToStringView(iree_string_view_t value) {
  return std::string_view(value.data ? value.data : "", value.size);
}

template <iree_host_size_t Capacity>
struct alignas(iree_max_align_t) AlignedStorage {
  // Max-aligned caller-owned description storage.
  std::array<uint8_t, Capacity> bytes;

  iree_byte_span_t span(iree_host_size_t length) {
    return length == 0 ? iree_byte_span_empty()
                       : iree_make_byte_span(bytes.data(), length);
  }
};

struct NativeModule {
  // Environment used only while resolving the provider's ref types.
  iree_vm_environment_t* environment = nullptr;
  // Published immutable native C module.
  iree_vm_module_t* module = nullptr;
  // Observable native provider lifecycle counters.
  iree_vm_test_module_counters_t counters = {};

  void Create() {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    IREE_ASSERT_OK(iree_vm_test_module_create(
        environment, &counters, iree_allocator_system(), &module));
  }

  void Reset() {
    iree_vm_module_release(module);
    module = nullptr;
    iree_vm_environment_free(environment);
    environment = nullptr;
  }
};

TEST(VMModuleTest, NativeCProviderPublishesImmutableDefinition) {
  NativeModule native_module;
  native_module.Create();

  EXPECT_EQ(ToStringView(iree_vm_module_name(native_module.module)),
            "test.module");
  EXPECT_EQ(iree_vm_module_ref_type_count(native_module.module), 1u);
  EXPECT_EQ(iree_vm_module_function_count(native_module.module), 1u);
  EXPECT_EQ(iree_vm_module_import_count(native_module.module), 1u);
  EXPECT_EQ(iree_vm_module_export_count(native_module.module), 2u);

  iree_vm_ref_type_t ref_type = nullptr;
  IREE_ASSERT_OK(
      iree_vm_module_ref_type_by_ordinal(native_module.module, 0, &ref_type));
  ASSERT_NE(ref_type, nullptr);
  EXPECT_EQ(ToStringView(ref_type->table->namespace_name), "vm");
  EXPECT_EQ(ToStringView(ref_type->type_name), "buffer");

  iree_vm_module_import_group_t import_group = {};
  IREE_ASSERT_OK(iree_vm_module_query_import_group(native_module.module, 0,
                                                   &import_group));
  EXPECT_EQ(ToStringView(import_group.target_module_name), "dependency.module");
  EXPECT_EQ(import_group.first_import_ordinal, 0u);
  EXPECT_EQ(import_group.import_count, 1u);

  iree_vm_module_import_declaration_t import_declaration = {};
  IREE_ASSERT_OK(iree_vm_module_query_import(native_module.module, 0,
                                             &import_declaration));
  EXPECT_EQ(ToStringView(import_declaration.target_export_name),
            "optional_callback");
  EXPECT_EQ(import_declaration.callable_type_ordinal, 0u);
  EXPECT_EQ(import_declaration.flags, IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL);

  iree_vm_module_export_declaration_t alias_declaration = {};
  iree_vm_module_export_declaration_t run_declaration = {};
  IREE_ASSERT_OK(
      iree_vm_module_query_export(native_module.module, 0, &alias_declaration));
  IREE_ASSERT_OK(
      iree_vm_module_query_export(native_module.module, 1, &run_declaration));
  EXPECT_EQ(ToStringView(alias_declaration.export_name), "alias");
  EXPECT_EQ(ToStringView(run_declaration.export_name), "run");
  EXPECT_EQ(alias_declaration.function_ordinal,
            run_declaration.function_ordinal);
  EXPECT_EQ(alias_declaration.callable_type_ordinal,
            run_declaration.callable_type_ordinal);

  iree_vm_module_retain(native_module.module);
  iree_vm_module_release(native_module.module);
  EXPECT_EQ(native_module.counters.destroy_count, 0);

  // A module borrows resolved provider descriptors rather than retaining the
  // environment used during construction.
  iree_vm_environment_free(native_module.environment);
  native_module.environment = nullptr;
  EXPECT_EQ(ToStringView(iree_vm_module_name(native_module.module)),
            "test.module");

  native_module.Reset();
  EXPECT_EQ(native_module.counters.destroy_count, 1);
}

TEST(VMModuleTest, DescriptionsResolveExactTypesAndPresentation) {
  NativeModule native_module;
  native_module.Create();

  iree_vm_export_t run_export = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(native_module.module,
                                              IREE_SV("run"), &run_export));
  EXPECT_EQ(ToStringView(iree_vm_export_name(run_export)), "run");

  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      run_export, iree_byte_span_empty(), &required_size, nullptr));
  ASSERT_GT(required_size, 8u);

  AlignedStorage<1024> storage;
  ASSERT_LE(required_size, storage.bytes.size());
  storage.bytes.fill(0xCD);
  iree_vm_export_description_t description;
  std::memset(&description, 0xA5, sizeof(description));
  const iree_vm_export_description_t original_description = description;

  const iree_host_size_t exact_required_size = required_size;
  required_size = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_export_query_description(
          run_export,
          iree_make_byte_span(storage.bytes.data() + 1, exact_required_size),
          &required_size, &description));
  EXPECT_EQ(required_size, 123u);
  EXPECT_EQ(
      std::memcmp(&description, &original_description, sizeof(description)), 0);

  required_size = exact_required_size;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      run_export, storage.span(required_size - 1), &required_size,
      &description));
  EXPECT_EQ(
      std::memcmp(&description, &original_description, sizeof(description)), 0);
  for (iree_host_size_t i = 0; i < required_size - 1; ++i) {
    EXPECT_EQ(storage.bytes[i], 0xCD);
  }

  IREE_ASSERT_OK(iree_vm_export_query_description(
      run_export, storage.span(required_size), &required_size, &description));
  EXPECT_EQ(ToStringView(description.name), "run");
  EXPECT_EQ(description.callable_flags, IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD);
  ASSERT_EQ(description.arguments.count, 3u);
  ASSERT_EQ(description.results.count, 3u);
  ASSERT_NE(description.arguments.data, nullptr);
  ASSERT_NE(description.results.data, nullptr);
  EXPECT_EQ(ToStringView(description.documentation), "dynamic");
  EXPECT_EQ(ToStringView(description.authored_type),
            "(i32, buffer, (i32) -> i32) -> (i64, buffer, (i32) -> i32)");

  EXPECT_EQ(description.arguments.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(description.arguments.data[0].type.value.scalar,
            IREE_VM_SCALAR_TYPE_I32);
  EXPECT_EQ(ToStringView(description.arguments.data[0].name), "value");
  EXPECT_EQ(description.arguments.data[1].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(
      ToStringView(description.arguments.data[1].type.value.ref->type_name),
      "buffer");
  EXPECT_EQ(description.arguments.data[2].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION);
  EXPECT_EQ(ToStringView(description.arguments.data[2].name), "callback");
  EXPECT_EQ(description.results.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(description.results.data[0].type.value.scalar,
            IREE_VM_SCALAR_TYPE_I64);
  EXPECT_EQ(description.results.data[1].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(description.results.data[2].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION);

  const iree_vm_callable_type_t callback_type =
      description.arguments.data[2].type.value.callable;
  iree_host_size_t callable_size = 0;
  IREE_ASSERT_OK(iree_vm_callable_type_query_description(
      callback_type, iree_byte_span_empty(), &callable_size, nullptr));
  AlignedStorage<256> callable_storage;
  ASSERT_LE(callable_size, callable_storage.bytes.size());
  iree_vm_callable_type_description_t callable_description = {};
  IREE_ASSERT_OK(iree_vm_callable_type_query_description(
      callback_type, callable_storage.span(callable_size), &callable_size,
      &callable_description));
  EXPECT_EQ(callable_description.flags, IREE_VM_CALLABLE_TYPE_FLAG_NONE);
  ASSERT_EQ(callable_description.arguments.count, 1u);
  ASSERT_EQ(callable_description.results.count, 1u);
  EXPECT_EQ(callable_description.arguments.data[0].kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(callable_description.arguments.data[0].value.scalar,
            IREE_VM_SCALAR_TYPE_I32);
  EXPECT_EQ(callable_description.results.data[0].value.scalar,
            IREE_VM_SCALAR_TYPE_I32);

  iree_vm_import_t import_value = {};
  IREE_ASSERT_OK(
      iree_vm_module_import_by_ordinal(native_module.module, 0, &import_value));
  const iree_vm_import_target_t target = iree_vm_import_target(import_value);
  EXPECT_EQ(ToStringView(target.module_name), "dependency.module");
  EXPECT_EQ(ToStringView(target.export_name), "optional_callback");
  iree_host_size_t import_size = 0;
  IREE_ASSERT_OK(iree_vm_import_query_description(
      import_value, iree_byte_span_empty(), &import_size, nullptr));
  AlignedStorage<256> import_storage;
  ASSERT_LE(import_size, import_storage.bytes.size());
  iree_vm_import_description_t import_description = {};
  IREE_ASSERT_OK(iree_vm_import_query_description(
      import_value, import_storage.span(import_size), &import_size,
      &import_description));
  EXPECT_EQ(import_description.flags, IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL);
  EXPECT_EQ(import_description.callable_flags, IREE_VM_CALLABLE_TYPE_FLAG_NONE);
  EXPECT_EQ(ToStringView(import_description.documentation),
            "Optional dependency callback.");

  native_module.Reset();
}

TEST(VMModuleTest, MetadataRemainsDeclarationLocalAndTyped) {
  NativeModule native_module;
  native_module.Create();

  EXPECT_EQ(iree_vm_module_metadata_count(native_module.module), 2u);
  iree_vm_metadata_entry_t entry = {};
  IREE_ASSERT_OK(
      iree_vm_module_metadata_by_ordinal(native_module.module, 0, &entry));
  EXPECT_EQ(ToStringView(entry.key), "build");
  iree_string_view_t build = iree_string_view_empty();
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(entry.value, &build));
  EXPECT_EQ(ToStringView(build), "native-c");

  IREE_ASSERT_OK(
      iree_vm_module_metadata_by_ordinal(native_module.module, 1, &entry));
  uint64_t revision = 0;
  IREE_ASSERT_OK(iree_vm_u64_from_metadata_value(entry.value, &revision));
  EXPECT_EQ(revision, 42u);

  iree_vm_import_t import_value = {};
  IREE_ASSERT_OK(
      iree_vm_module_import_by_ordinal(native_module.module, 0, &import_value));
  EXPECT_EQ(iree_vm_import_metadata_count(import_value), 1u);
  IREE_ASSERT_OK(iree_vm_import_metadata_by_ordinal(import_value, 0, &entry));
  bool is_optional = false;
  IREE_ASSERT_OK(iree_vm_bool_from_metadata_value(entry.value, &is_optional));
  EXPECT_TRUE(is_optional);

  iree_vm_export_t alias_export = {};
  iree_vm_export_t run_export = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_export(native_module.module,
                                              IREE_SV("alias"), &alias_export));
  IREE_ASSERT_OK(iree_vm_module_lookup_export(native_module.module,
                                              IREE_SV("run"), &run_export));
  EXPECT_EQ(iree_vm_export_metadata_count(alias_export), 1u);
  EXPECT_EQ(iree_vm_export_metadata_count(run_export), 2u);

  bool found = false;
  iree_vm_metadata_value_t value = {};
  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      alias_export, IREE_SV("route"), &found, &value));
  ASSERT_TRUE(found);
  iree_string_view_t route = iree_string_view_empty();
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(value, &route));
  EXPECT_EQ(ToStringView(route), "alias");

  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      run_export, IREE_SV("route"), &found, &value));
  ASSERT_TRUE(found);
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(value, &route));
  EXPECT_EQ(ToStringView(route), "primary");

  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      run_export, IREE_SV("weight"), &found, &value));
  ASSERT_TRUE(found);
  double weight = 0.0;
  IREE_ASSERT_OK(iree_vm_f64_from_metadata_value(value, &weight));
  EXPECT_DOUBLE_EQ(weight, 2.5);

  const iree_vm_metadata_value_t original_value = value;
  found = true;
  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      run_export, IREE_SV("missing"), &found, &value));
  EXPECT_FALSE(found);
  EXPECT_EQ(std::memcmp(&value, &original_value, sizeof(value)), 0);

  uint64_t untouched = 1234;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_u64_from_metadata_value({IREE_VM_METADATA_VALUE_TYPE_BOOL,
                                       iree_make_const_byte_span("\x01", 1)},
                                      &untouched));
  EXPECT_EQ(untouched, 1234u);

  const std::array<uint8_t, 3> opaque_bytes = {0x00, 0x80, 0xFF};
  iree_const_byte_span_t byte_span = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_const_byte_span_from_metadata_value(
      {IREE_VM_METADATA_VALUE_TYPE_BYTES,
       iree_make_const_byte_span(opaque_bytes.data(), opaque_bytes.size())},
      &byte_span));
  EXPECT_EQ(byte_span.data, opaque_bytes.data());
  EXPECT_EQ(byte_span.data_length, opaque_bytes.size());

  native_module.Reset();
}

TEST(VMModuleTest, PhysicalCallPacketAndOpaqueStateUseProviderABI) {
  NativeModule native_module;
  native_module.Create();

  iree_vm_ref_types_t vm_types = {};
  IREE_ASSERT_OK(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    native_module.environment, IREE_SV("vm")),
                                &vm_types));
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_vm_buffer_create(16, 0, iree_allocator_system(), &buffer));

  uint64_t value_arguments[] = {35};
  uint64_t value_results[] = {0};
  iree_vm_ref_t ref_arguments[] = {
      iree_vm_buffer_ref_from_ptr_borrowed(&vm_types, buffer),
  };
  iree_vm_ref_t ref_results[] = {iree_vm_ref_null()};
  const iree_vm_function_ref_t function_arguments[] = {
      {UINT64_C(0x1000), UINT64_C(0x2000)},
  };
  iree_vm_function_ref_t function_results[] = {
      iree_vm_function_ref_null(),
  };
  const iree_vm_call_packet_t call = {
      {value_arguments, nullptr},    {ref_arguments, nullptr},
      {value_results, nullptr},      {ref_results, nullptr},
      {function_arguments, nullptr}, {function_results, nullptr},
  };
  const iree_vm_module_function_start_params_t params = {
      {nullptr, nullptr, nullptr},
      0,
      call,
  };
  iree_vm_execution_outcome_t outcome =
      static_cast<iree_vm_execution_outcome_t>(99);
  IREE_ASSERT_OK(native_module.module->vtable->function_start(
      native_module.module, &params, &outcome));
  EXPECT_EQ(outcome, IREE_VM_EXECUTION_OUTCOME_COMPLETED);
  EXPECT_EQ(value_results[0], 42u);
  EXPECT_EQ(ref_arguments[0].object, buffer);
  EXPECT_EQ(ref_results[0].object, buffer);
  EXPECT_EQ(iree_vm_ref_type(ref_results[0]), vm_types.buffer);
  EXPECT_EQ(ref_results[0].type_and_state & IREE_VM_REF_STATE_MASK,
            IREE_VM_REF_STATE_OWNED);
  EXPECT_EQ(function_results[0].program_bits,
            function_arguments[0].program_bits);
  EXPECT_EQ(function_results[0].target_bits, function_arguments[0].target_bits);

  iree_vm_buffer_release(buffer);
  iree_vm_ref_reset(&ref_arguments[0]);
  iree_vm_ref_reset(&ref_results[0]);

  alignas(iree_max_align_t) std::array<uint8_t, 8> process_storage = {};
  ASSERT_EQ(native_module.module->descriptor->process_storage_size,
            process_storage.size());
  const iree_byte_span_t process_span =
      iree_make_byte_span(process_storage.data(), process_storage.size());
  IREE_ASSERT_OK(native_module.module->vtable->attach_state(
      native_module.module, process_span, iree_allocator_system()));
  IREE_ASSERT_OK(native_module.module->vtable->seal_state(native_module.module,
                                                          process_span));
  native_module.module->vtable->detach_state(native_module.module,
                                             process_span);
  EXPECT_EQ(native_module.counters.attach_count, 1);
  EXPECT_EQ(native_module.counters.seal_count, 1);
  EXPECT_EQ(native_module.counters.detach_count, 1);
  for (uint8_t byte : process_storage) EXPECT_EQ(byte, 0u);

  native_module.Reset();
}

TEST(VMModuleTest, PhysicalCallPacketAddressesDirectAndOverflowBanks) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_ref_types_t vm_types = {};
  IREE_ASSERT_OK(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &vm_types));

  std::array<uint64_t, 16> direct_value_arguments = {};
  std::array<uint64_t, 1> overflow_value_arguments = {};
  std::array<uint64_t, 16> direct_value_results = {};
  std::array<uint64_t, 1> overflow_value_results = {};
  direct_value_arguments[15] = 15;
  overflow_value_arguments[0] = 16;

  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(0, 0, iree_allocator_system(), &buffer));
  std::array<iree_vm_ref_t, 16> direct_ref_arguments = {};
  std::array<iree_vm_ref_t, 1> overflow_ref_arguments = {
      iree_vm_buffer_ref_from_ptr_move(&vm_types, &buffer),
  };
  std::array<iree_vm_ref_t, 16> direct_ref_results = {};
  std::array<iree_vm_ref_t, 1> overflow_ref_results = {};
  ASSERT_EQ(buffer, nullptr);

  std::array<iree_vm_function_ref_t, 16> direct_function_arguments = {};
  std::array<iree_vm_function_ref_t, 1> overflow_function_arguments = {
      iree_vm_function_ref_t{UINT64_C(0x3000), UINT64_C(0x4000)},
  };
  std::array<iree_vm_function_ref_t, 16> direct_function_results = {};
  std::array<iree_vm_function_ref_t, 1> overflow_function_results = {};

  const iree_vm_call_packet_t call = {
      {direct_value_arguments.data(), overflow_value_arguments.data()},
      {direct_ref_arguments.data(), overflow_ref_arguments.data()},
      {direct_value_results.data(), overflow_value_results.data()},
      {direct_ref_results.data(), overflow_ref_results.data()},
      {direct_function_arguments.data(), overflow_function_arguments.data()},
      {direct_function_results.data(), overflow_function_results.data()},
  };
  EXPECT_EQ(iree_vm_call_value_argument_load(&call, 15), 15u);
  EXPECT_EQ(iree_vm_call_value_argument_load(&call, 16), 16u);
  iree_vm_call_value_result_store(&call, 15, 115);
  iree_vm_call_value_result_store(&call, 16, 116);
  EXPECT_EQ(direct_value_results[15], 115u);
  EXPECT_EQ(overflow_value_results[0], 116u);

  iree_vm_ref_t moved_ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_move(&call, 16, &moved_ref);
  EXPECT_TRUE(iree_vm_ref_is_null(overflow_ref_arguments[0]));
  EXPECT_FALSE(iree_vm_ref_is_null(moved_ref));
  iree_vm_call_ref_result_store_move(&call, 16, &moved_ref);
  EXPECT_TRUE(iree_vm_ref_is_null(moved_ref));
  EXPECT_FALSE(iree_vm_ref_is_null(overflow_ref_results[0]));

  const iree_vm_function_ref_t function_ref =
      iree_vm_call_function_argument_load(&call, 16);
  iree_vm_call_function_result_store(&call, 16, function_ref);
  EXPECT_EQ(overflow_function_results[0].program_bits,
            overflow_function_arguments[0].program_bits);
  EXPECT_EQ(overflow_function_results[0].target_bits,
            overflow_function_arguments[0].target_bits);

  iree_vm_ref_reset(&overflow_ref_results[0]);
  iree_vm_environment_free(environment);
}

TEST(VMModuleTest, ImmutableProviderQueriesAreConcurrent) {
  NativeModule native_module;
  native_module.Create();

  std::atomic<bool> observed_mismatch = false;
  std::vector<std::thread> threads;
  for (int thread_i = 0; thread_i < 4; ++thread_i) {
    threads.emplace_back([&]() {
      for (int i = 0; i < 1000; ++i) {
        iree_vm_module_export_declaration_t export_declaration = {};
        native_module.module->vtable->query_export(native_module.module, i % 2,
                                                   &export_declaration);
        const std::string_view expected = i % 2 == 0 ? "alias" : "run";
        if (ToStringView(export_declaration.export_name) != expected) {
          observed_mismatch.store(true, std::memory_order_relaxed);
        }
        iree_vm_module_callable_type_declaration_t callable_type = {};
        native_module.module->vtable->query_callable_type(native_module.module,
                                                          1, &callable_type);
        if (callable_type.signature.arguments.count != 3 ||
            callable_type.signature.results.count != 3) {
          observed_mismatch.store(true, std::memory_order_relaxed);
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_FALSE(observed_mismatch.load(std::memory_order_relaxed));

  native_module.Reset();
}

}  // namespace
