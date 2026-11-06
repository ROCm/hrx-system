// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <type_traits>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/reference_transport_testdata.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/sync.h"

namespace {

struct Object {
  // Intrusive runtime prefix at offset zero.
  iree_vm_ref_object_t header;
  // Finalizer counter that outlives the invocation.
  int* destroyed;
  // Payload observed by native methods to detect stale access under ASAN.
  int32_t identity;
};

void DestroyObject(void* pointer) {
  auto* object = static_cast<Object*>(pointer);
  ++*object->destroyed;
  delete object;
}

extern const iree_vm_ref_type_table_t kObjectTypes;
const iree_vm_ref_type_descriptor_t kObjectType = {DestroyObject, &kObjectTypes,
                                                   IREE_SV("object")};
const iree_vm_ref_type_t kRefTypes[] = {&kObjectType};
const iree_vm_ref_type_table_t kObjectTypes = {
    sizeof(kObjectTypes),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("refcopy"),
    {kRefTypes, IREE_ARRAYSIZE(kRefTypes)}};
const iree_vm_module_signature_type_t kArguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_PTR, 0}};
const iree_vm_module_signature_type_t kResults[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0}};
const iree_vm_module_callable_type_declaration_t kCallableTypes[] = {
    {{{kArguments, 1, 0, 1, 0}, {kResults, 3, 3, 0, 0}}, 0, 0, 0}};
const iree_vm_module_export_declaration_t kExports[] = {
    {IREE_SV("inspect"), 0, 0, 0}};

struct ProbeModule {
  // Offset-zero table provider used by the real linked native call.
  iree::vm::testing::TableModule table;
  // Observes destination replacement before the bytecode frame unwinds.
  int* destroyed = nullptr;

  static iree_status_t Start(
      iree_vm_module_t* module,
      const iree_vm_module_function_start_params_t* params,
      iree_vm_execution_outcome_t* outcome) {
    auto* self = reinterpret_cast<ProbeModule*>(module);
    iree_vm_ref_t value = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&params->call, 0, &value);
    void* pointer = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_vm_ptr_from_ref_borrowed(value, &kObjectType, &pointer));
    const auto* object = static_cast<const Object*>(pointer);
    const int32_t results[] = {
        object ? iree_atomic_ref_count_load(&object->header.ref_count) : 0,
        object ? object->identity : -1,
        *self->destroyed,
    };
    for (size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
      iree_vm_call_value_result_store(&params->call, i, results[i]);
    }
    *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }
};
static_assert(std::is_standard_layout_v<ProbeModule>);
static_assert(offsetof(ProbeModule, table) == 0);

class BytecodeReferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    probe_.destroyed = &destroyed_;
    probe_.table.vtable().function_start = ProbeModule::Start;
    IREE_ASSERT_OK(probe_.table.Publish(
        iree::vm::testing::TableModuleDefinition(IREE_SV("probe"), 1)
            .WithRefTypes(kRefTypes)
            .WithCallableTypes(kCallableTypes)
            .WithExports(kExports)));
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(environment_,
                                                               &kObjectTypes));
    const iree_file_toc_t* files =
        iree_vm_bytecode_reference_transport_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_reference_transport_testdata_size(), 1u);
    IREE_ASSERT_OK(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                              files[0].size),
        iree_allocator_system(), &fixture_));
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment_, IREE_SV("copies"),
        {iree_const_cast_byte_span(fixture_.contents), iree_allocator_null()},
        iree_allocator_system(), &module_));
    iree_vm_module_t* libraries[] = {probe_.table.module()};
    IREE_ASSERT_OK(iree_vm_program_create(
        {module_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_bytecode_module_fixture_deinitialize(&fixture_);
    iree_vm_environment_free(environment_);
  }

  iree_vm_ref_t MakeObject(int32_t identity) {
    auto* object = new Object;
    iree_vm_ref_object_initialize(&object->header);
    object->destroyed = &destroyed_;
    object->identity = identity;
    void* pointer = object;
    return iree_vm_ref_from_ptr_move(&pointer, &kObjectType);
  }

  iree_vm_function_t Lookup(const char* name) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_EXPECT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("copies"), iree_make_cstring_view(name), &function));
    return function;
  }

  // Observed native finalizers, reset only after all owners from a case retire.
  int destroyed_ = 0;
  // Native observations use the production invocation boundary.
  ProbeModule probe_;
  // Canonical descriptors used to link both native and bytecode modules.
  iree_vm_environment_t* environment_ = nullptr;
  // Assembled storage borrowed by the module for its complete lifetime.
  iree_vm_bytecode_module_fixture_t fixture_ = {};
  // Executable module and program ownership.
  iree_vm_module_t* module_ = nullptr;
  // Linked native and bytecode providers.
  iree_vm_program_t* program_ = nullptr;
  // Process storage retired before its invocation.
  iree_vm_process_t* process_ = nullptr;
  // Frame backing for the synchronous call, including local reference slots.
  alignas(iree_max_align_t) std::array<uint8_t, 16 * 1024> storage_ = {};
  // Initialized invocation within the caller's storage.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(BytecodeReferenceTest, CopiesPreserveStateAndReplaceDestination) {
  struct Case {
    // Export selecting a physical transport operation.
    const char* name;
    // Live VM owners observed when the input already owns its carrier.
    int32_t owned_count;
    // Live VM owners acquired when the input is borrowed.
    int32_t borrowed_count;
  };
  const Case cases[] = {{"copy_register", 2, 0},
                        {"copy_load", 3, 0},
                        {"copy_store", 2, 0},
                        {"retain", 2, 1}};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.name);
    for (bool owned : {false, true}) {
      SCOPED_TRACE(owned);
      for (bool null_source : {false, true}) {
        SCOPED_TRACE(null_source);
        for (bool replace : {false, true}) {
          SCOPED_TRACE(replace);
          destroyed_ = 0;
          iree_vm_ref_t source =
              null_source ? iree_vm_ref_null() : MakeObject(42);
          iree_vm_ref_t destination =
              replace ? MakeObject(7) : iree_vm_ref_null();
          iree_vm_variant_t arguments[] = {
              owned ? iree_vm_variant_from_ref_move(&source)
                    : iree_vm_variant_from_ref_borrowed(source),
              iree_vm_variant_from_ref_move(&destination),
          };
          iree_vm_variant_t results[4] = {};
          IREE_EXPECT_OK(
              iree_vm_invoke(invocation_, Lookup(test.name),
                             iree_vm_variant_span_from_array(arguments),
                             iree_vm_variant_span_from_array(results)));
          const int32_t expected[] = {null_source ? 0
                                      : owned     ? test.owned_count
                                                  : 1 + test.borrowed_count,
                                      null_source ? -1 : 42, replace ? 1 : 0,
                                      1};
          for (size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
            int32_t actual = 0;
            IREE_EXPECT_OK(iree_vm_i32_from_variant(results[i], &actual));
            EXPECT_EQ(actual, expected[i]) << "result " << i;
          }
          EXPECT_EQ(destroyed_, int(replace) + int(owned && !null_source));
          iree_vm_variant_span_reset(
              iree_vm_variant_span_from_array(arguments));
          iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
          iree_vm_ref_reset(&source);
          EXPECT_EQ(destroyed_, int(replace) + int(!null_source));
        }
      }
    }
  }
}

TEST_F(BytecodeReferenceTest, FailureRetiresCopiesWithoutConsumingHostAnchor) {
  for (bool owned : {false, true}) {
    for (bool null_source : {false, true}) {
      destroyed_ = 0;
      iree_vm_ref_t source = null_source ? iree_vm_ref_null() : MakeObject(42);
      iree_vm_variant_t arguments[] = {
          owned ? iree_vm_variant_from_ref_move(&source)
                : iree_vm_variant_from_ref_borrowed(source)};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_ABORTED,
          iree_vm_invoke(invocation_, Lookup("fail"),
                         iree_vm_variant_span_from_array(arguments),
                         iree_vm_variant_span_empty()));
      EXPECT_EQ(destroyed_, int(owned && !null_source));
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
      iree_vm_ref_reset(&source);
      EXPECT_EQ(destroyed_, int(!null_source));
    }
  }
}

TEST_F(BytecodeReferenceTest, ReturnedCopyProtectsItsEscapingLifetime) {
  for (bool owned : {false, true}) {
    for (bool null_source : {false, true}) {
      destroyed_ = 0;
      iree_vm_ref_t source = null_source ? iree_vm_ref_null() : MakeObject(42);
      iree_vm_variant_t arguments[] = {
          owned ? iree_vm_variant_from_ref_move(&source)
                : iree_vm_variant_from_ref_borrowed(source)};
      iree_vm_variant_t results[1] = {};
      IREE_EXPECT_OK(iree_vm_invoke(invocation_, Lookup("copy_return"),
                                    iree_vm_variant_span_from_array(arguments),
                                    iree_vm_variant_span_from_array(results)));
      iree_vm_ref_reset(&source);
      EXPECT_EQ(destroyed_, 0);
      void* pointer = nullptr;
      IREE_EXPECT_OK(iree_vm_ptr_from_variant_borrowed(results[0], &kObjectType,
                                                       &pointer));
      if (null_source) {
        EXPECT_EQ(pointer, nullptr);
      } else {
        ASSERT_NE(pointer, nullptr);
        EXPECT_EQ(static_cast<Object*>(pointer)->identity, 42);
      }
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
      EXPECT_EQ(destroyed_, int(!null_source));
    }
  }
}

}  // namespace
