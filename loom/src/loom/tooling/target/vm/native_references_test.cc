// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <utility>

#include "iree/hal/api.h"
#include "iree/hal/vm/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/format/location.h"
#include "loom/tooling/target/vm/native_references_bytecode.h"

namespace {

constexpr iree_vm_module_signature_type_t kStorage[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr iree_vm_module_signature_type_t kFail[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2}};
constexpr iree_vm_module_signature_type_t kRelay[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 2}};
constexpr auto kWide = [] {
  std::array<iree_vm_module_signature_type_t, 17> fields = {};
  for (auto& field : fields) {
    field = {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1};
  }
  return fields;
}();

const iree_vm_module_callable_type_declaration_t kCallables[] = {
    {{{kStorage, 1, 0, 1, 0}, {kStorage, 1, 0, 1, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kFail, 3, 0, 3, 0}, {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kRelay, 4, 1, 3, 0}, {kRelay, 4, 1, 3, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kWide.data(), 17, 0, 17, 0}, {kWide.data(), 17, 0, 17, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
};
const iree_vm_module_export_declaration_t kExports[] = {
    {IREE_SVL("fail"), 1, 0, 0},
    {IREE_SVL("relay"), 2, 1, 0},
    {IREE_SVL("storage"), 0, 2, 0},
    {IREE_SVL("wide"), 3, 3, 0},
};

struct ResourceModule {
  // Native module prefix, borrowing all following immutable declaration data.
  iree_vm_module_t base = {};
  // Canonical HAL buffer, HAL view, and Core buffer identities.
  std::array<iree_vm_ref_type_t, 3> types = {};
  // Fixed native module description.
  iree_vm_module_descriptor_t descriptor = {};
  // Number of relay calls that actually ran inside the compiled program.
  int relay_count = 0;
  // Source retained by the native error observer until module destruction.
  iree_vm_ref_t failure_site = {};
};

iree_status_t Start(iree_vm_module_t* base,
                    const iree_vm_module_function_start_params_t* params,
                    iree_vm_execution_outcome_t* outcome) {
  auto* module = reinterpret_cast<ResourceModule*>(base);
  if (params->function_ordinal == 0) {
    iree_vm_ref_reset(&module->failure_site);
    iree_vm_ref_t site = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_borrow(&params->call, 2, &site);
    module->failure_site = iree_vm_ref_retain(site);
    return iree_make_status(IREE_STATUS_ABORTED, "resource service failed");
  }
  if (params->function_ordinal == 1) {
    ++module->relay_count;
    iree_vm_call_value_result_store(
        &params->call, 0,
        iree_vm_call_value_argument_load(&params->call, 0) + 7);
  }
  const uint16_t count = params->function_ordinal == 3   ? 17
                         : params->function_ordinal == 1 ? 3
                                                         : 1;
  // Direct argument/result banks may alias. Consume the arguments before
  // rotating results into those same slots.
  std::array<iree_vm_ref_t, 17> refs = {};
  for (uint16_t i = 0; i < count; ++i) {
    iree_vm_call_ref_argument_load_move(&params->call, i, &refs[i]);
    const uint16_t type_ordinal = params->function_ordinal == 3   ? 1
                                  : params->function_ordinal == 2 ? 0
                                  : i == 0                        ? 1
                                  : i == 1                        ? 0
                                                                  : 2;
    if (!iree_vm_ref_is_null(refs[i])) {
      EXPECT_EQ(iree_vm_ref_type(refs[i]), module->types[type_ordinal]);
    }
  }
  for (uint16_t i = 0; i < count; ++i) {
    // Rotate the native overflow results so a dropped or misindexed overflow
    // slot cannot pass by returning the unchanged direct register bank.
    const uint16_t result = params->function_ordinal == 3 ? (i + 1) % 17 : i;
    iree_vm_call_ref_result_store_move(&params->call, result, &refs[i]);
  }
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

void QueryExport(const iree_vm_module_t*, iree_host_size_t ordinal,
                 iree_vm_module_export_declaration_t* value) {
  *value = kExports[ordinal];
}
void QueryCallable(const iree_vm_module_t*, iree_host_size_t ordinal,
                   iree_vm_module_callable_type_declaration_t* value) {
  *value = kCallables[ordinal];
}
void QueryImportGroup(const iree_vm_module_t*, iree_host_size_t,
                      iree_vm_module_import_group_t*) {
  IREE_CHECK_UNREACHABLE("resource module has no imports");
}
void QueryImport(const iree_vm_module_t*, iree_host_size_t,
                 iree_vm_module_import_declaration_t*) {
  IREE_CHECK_UNREACHABLE("resource module has no imports");
}
void DestroyModule(iree_vm_module_t* base) {
  iree_vm_ref_reset(&reinterpret_cast<ResourceModule*>(base)->failure_site);
}
const iree_vm_module_vtable_t kVtable = {
    sizeof(kVtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    DestroyModule,
    Start,
    iree_vm_module_function_resume_unreachable,
    nullptr,
    nullptr,
    nullptr,
    QueryImportGroup,
    QueryImport,
    QueryExport,
    QueryCallable,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none};

iree_status_t InitializeResourceModule(iree_vm_ref_type_t buffer,
                                       iree_vm_ref_type_t view,
                                       iree_vm_ref_type_t site,
                                       ResourceModule* module) {
  module->types = {buffer, view, site};
  module->descriptor = {IREE_SVL("resources"),
                        IREE_VM_MODULE_FLAG_LINKABLE,
                        {module->types.data(), module->types.size()},
                        {4, 4, 0, 0, 4, 0, {2, 45, 0}},
                        0};
  return iree_vm_module_initialize(&kVtable, &module->descriptor,
                                   &module->base);
}

iree_status_t LoadImage(iree_vm_environment_t* environment,
                        iree_vm_module_t** out_module) {
  const auto* file = loom_vm_native_references_bytecode_create();
  uint8_t* image = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), file[0].size, reinterpret_cast<void**>(&image)));
  std::memcpy(image, file[0].data, file[0].size);
  iree_status_t status = iree_vm_bytecode_module_create(
      environment, IREE_SV("compiled"),
      {iree_make_const_byte_span(image, file[0].size), iree_allocator_system()},
      iree_allocator_system(), out_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), image);
  }
  return status;
}

void CountRelease(void* user_data, iree_hal_buffer_t*) {
  ++*static_cast<int*>(user_data);
}

struct HalAllocation {
  // Host backing storage retained until every HAL and VM owner is released.
  alignas(64) std::array<int32_t, 16> data = {};
  // Number of final releases of the wrapped allocation.
  int releases = 0;
  // Owned root buffer, or null after transfer to the VM.
  iree_hal_buffer_t* buffer = nullptr;
  // Owned view of a nonzero-offset subspan, or null after transfer to the VM.
  iree_hal_buffer_view_t* view = nullptr;

  HalAllocation() {
    data[2] = 37;
    data[5] = 91;
    IREE_CHECK_OK(iree_hal_heap_buffer_wrap(
        {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_MAPPING, sizeof(data),
        iree_make_byte_span(data.data(), sizeof(data)),
        {CountRelease, &releases}, iree_allocator_system(), &buffer));
    iree_hal_buffer_t* subspan = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_subspan(buffer, 2 * sizeof(int32_t),
                                          4 * sizeof(int32_t),
                                          iree_allocator_system(), &subspan));
    const iree_hal_dim_t shape[] = {2, 2};
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        subspan, 2, shape, IREE_HAL_ELEMENT_TYPE_INT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(),
        &view));
    iree_hal_buffer_release(subspan);
  }
  ~HalAllocation() {
    iree_hal_buffer_view_release(view);
    iree_hal_buffer_release(buffer);
    EXPECT_EQ(releases, 1);
  }

  void ReleaseOwners() {
    iree_hal_buffer_view_release(std::exchange(view, nullptr));
    iree_hal_buffer_release(std::exchange(buffer, nullptr));
  }

  void ExpectView(iree_hal_buffer_view_t* returned) {
    ASSERT_NE(returned, nullptr);
    EXPECT_EQ(iree_hal_buffer_view_byte_length(returned), 16u);
    ASSERT_EQ(iree_hal_buffer_view_shape_rank(returned), 2u);
    EXPECT_EQ(iree_hal_buffer_view_shape_dims(returned)[0], 2u);
    EXPECT_EQ(iree_hal_buffer_view_shape_dims(returned)[1], 2u);
    EXPECT_EQ(iree_hal_buffer_view_element_type(returned),
              IREE_HAL_ELEMENT_TYPE_INT_32);
    EXPECT_EQ(iree_hal_buffer_view_encoding_type(returned),
              IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR);
    iree_hal_buffer_mapping_t mapping = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(returned), IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
        IREE_HAL_WHOLE_BUFFER, &mapping));
    EXPECT_EQ(mapping.contents.data, reinterpret_cast<uint8_t*>(&data[2]));
    EXPECT_EQ(mapping.contents.data_length, 16u);
    EXPECT_EQ(reinterpret_cast<const int32_t*>(mapping.contents.data)[0], 37);
    EXPECT_EQ(reinterpret_cast<const int32_t*>(mapping.contents.data)[3], 91);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  }
};

class NativeReferencesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_vm_register_types(environment, &table));
    IREE_ASSERT_OK(iree_hal_vm_types_resolve(table, &hal_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &core_));
    IREE_ASSERT_OK(LoadImage(environment, &bytecode_));
    iree_vm_environment_free(environment);
    IREE_ASSERT_OK(InitializeResourceModule(hal_.buffer, hal_.buffer_view,
                                            core_.buffer, &native_));
    iree_vm_module_t* libraries[] = {&native_.base};
    IREE_ASSERT_OK(iree_vm_program_create(
        {bytecode_, iree_vm_module_span_from_array(libraries)},
        iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_.data(), storage_.size()), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
  }

  void ReleaseVM() {
    iree_vm_process_release(std::exchange(process_, nullptr));
    if (invocation_) {
      iree_vm_invocation_deinitialize(std::exchange(invocation_, nullptr));
    }
    iree_vm_program_release(std::exchange(program_, nullptr));
    iree_vm_module_release(std::exchange(bytecode_, nullptr));
  }

  void TearDown() override {
    ReleaseVM();
    if (native_.base.vtable) {
      iree_vm_module_release(&native_.base);
    }
  }

  iree_status_t Invoke(iree_string_view_t name,
                       iree_vm_variant_span_t arguments,
                       iree_vm_variant_span_t results) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("compiled"), name, &function));
    return iree_vm_invoke(invocation_, function, arguments, results);
  }

  iree_hal_buffer_view_t* View(iree_vm_variant_t variant) {
    iree_hal_buffer_view_t* view = nullptr;
    IREE_CHECK_OK(
        iree_hal_buffer_view_ptr_from_variant_borrowed(&hal_, variant, &view));
    return view;
  }

  void ExpectLocation(iree_vm_variant_t variant, uint32_t line) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        iree_vm_buffer_ptr_from_variant_borrowed(&core_, variant, &buffer));
    iree_const_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    loom_location_value_t location;
    IREE_ASSERT_OK(loom_location_value_parse(bytes, &location));
    ASSERT_EQ(location.node_count, 1u);
    const auto file = loom_location_value_file(location, 0);
    EXPECT_TRUE(iree_string_view_equal(file.source, IREE_SV("resource.cc")));
    EXPECT_EQ(file.range.start_line, line);
  }

  // Resolved HAL handles borrowed from the linked static provider.
  iree_hal_vm_types_t hal_ = {};
  // Resolved Core buffer handle for captured source values.
  iree_vm_ref_types_t core_ = {};
  // Native service implementation and declaration storage.
  ResourceModule native_;
  // Module owning the separately compiled fixture image.
  iree_vm_module_t* bytecode_ = nullptr;
  // Linked compilation/native-service composition.
  iree_vm_program_t* program_ = nullptr;
  // Process executing ordinary compiled calls and control flow.
  iree_vm_process_t* process_ = nullptr;
  // Host-owned invocation storage for nested full reference banks and their
  // argument/result overflow packets (about 26 KiB at the deepest call).
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage_ = {};
  // Invocation borrowing storage_ while the VM is live.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(NativeReferencesTest, ExecutesMixedReferencesThroughLoopAndLocalCalls) {
  for (int32_t count : {0, 1, 3}) {
    SCOPED_TRACE(count);
    HalAllocation left;
    HalAllocation right;
    auto* expected_left = left.view;
    auto* expected_view = count < 2 ? left.view : right.view;
    auto* expected_buffer = left.buffer;
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(count), iree_vm_variant_from_i64(42),
        iree_hal_buffer_view_variant_from_ptr_move(&hal_, &left.view),
        iree_hal_buffer_view_variant_from_ptr_move(&hal_, &right.view),
        iree_hal_buffer_variant_from_ptr_move(&hal_, &left.buffer)};
    right.ReleaseOwners();
    iree_vm_variant_t results[5] = {};
    const int previous_calls = native_.relay_count;
    IREE_ASSERT_OK(Invoke(IREE_SV("roundtrip"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    EXPECT_EQ(native_.relay_count - previous_calls, count);
    EXPECT_EQ(View(results[0]), expected_left);
    EXPECT_EQ(View(results[1]), expected_view);
    left.ExpectView(View(results[0]));
    if (count < 2) {
      left.ExpectView(View(results[1]));
    } else {
      right.ExpectView(View(results[1]));
    }
    int64_t tag = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[2], &tag));
    EXPECT_EQ(tag, 42 + 7 * count);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        iree_hal_buffer_ptr_from_variant_borrowed(&hal_, results[3], &buffer));
    EXPECT_EQ(buffer, expected_buffer);
    ExpectLocation(results[4], 41);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    EXPECT_EQ(left.releases, 1);
    EXPECT_EQ(right.releases, 1);
  }
}

TEST_F(NativeReferencesTest, EscapedOwnersSurviveVMTeardown) {
  HalAllocation allocation;
  auto* expected = allocation.view;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(1), iree_vm_variant_from_i64(9),
      iree_hal_buffer_view_variant_from_ptr_retained(&hal_, allocation.view),
      iree_hal_buffer_view_variant_from_ptr_move(&hal_, &allocation.view),
      iree_hal_buffer_variant_from_ptr_move(&hal_, &allocation.buffer)};
  iree_vm_variant_t results[5] = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("roundtrip"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  ReleaseVM();
  EXPECT_EQ(allocation.releases, 0);
  EXPECT_EQ(View(results[0]), expected);
  EXPECT_EQ(View(results[1]), expected);
  allocation.ExpectView(View(results[1]));
  ExpectLocation(results[4], 41);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  EXPECT_EQ(allocation.releases, 1);
}

TEST_F(NativeReferencesTest, FinalHALOwnerOutlivesVMOwners) {
  HalAllocation allocation;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(1),
      iree_hal_buffer_view_variant_from_ptr_borrowed(&hal_, allocation.view),
      iree_hal_buffer_view_variant_from_ptr_borrowed(&hal_, allocation.view)};
  iree_vm_variant_t result = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("select"),
                        iree_vm_variant_span_from_array(arguments),
                        {&result, 1}));
  ReleaseVM();
  iree_vm_variant_reset(&result);
  EXPECT_EQ(allocation.releases, 0);
  allocation.ExpectView(allocation.view);
  allocation.ReleaseOwners();
  EXPECT_EQ(allocation.releases, 1);
}

TEST_F(NativeReferencesTest, TransportsNullThroughNativeCallsAndControlFlow) {
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(3), iree_vm_variant_from_i64(1),
      iree_hal_buffer_view_variant_from_ptr_borrowed(&hal_, nullptr),
      iree_hal_buffer_view_variant_from_ptr_borrowed(&hal_, nullptr),
      iree_hal_buffer_variant_from_ptr_borrowed(&hal_, nullptr)};
  iree_vm_variant_t results[5] = {};
  IREE_ASSERT_OK(Invoke(IREE_SV("roundtrip"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  EXPECT_EQ(View(results[0]), nullptr);
  EXPECT_EQ(View(results[1]), nullptr);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_buffer_ptr_from_variant_borrowed(&hal_, results[3], &buffer));
  EXPECT_EQ(buffer, nullptr);
  ExpectLocation(results[4], 41);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
}

TEST_F(NativeReferencesTest, PreservesNativeReferenceOverflowOwners) {
  std::array<HalAllocation, 17> allocations;
  std::array<iree_hal_buffer_view_t*, 17> expected;
  iree_vm_variant_t arguments[17] = {};
  iree_vm_variant_t results[17] = {};
  for (size_t i = 0; i < 17; ++i) {
    expected[i] = allocations[i].view;
    arguments[i] =
        iree_hal_buffer_view_variant_from_ptr_move(&hal_, &allocations[i].view);
    allocations[i].ReleaseOwners();
  }
  IREE_ASSERT_OK(Invoke(IREE_SV("overflow"),
                        iree_vm_variant_span_from_array(arguments),
                        iree_vm_variant_span_from_array(results)));
  for (size_t i = 0; i < 17; ++i) {
    EXPECT_TRUE(iree_vm_variant_is_empty(arguments[i]));
    EXPECT_EQ(View(results[i]), expected[(i + 16) % 17]);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  for (const auto& allocation : allocations) {
    EXPECT_EQ(allocation.releases, 1);
  }
}

TEST_F(NativeReferencesTest, NativeFailureUnwindsAliasesAndKeepsCallerResults) {
  HalAllocation input;
  HalAllocation sentinel;
  auto* sentinel_view = sentinel.view;
  iree_vm_variant_t result =
      iree_hal_buffer_view_variant_from_ptr_retained(&hal_, sentinel.view);
  iree_vm_variant_t arguments[] = {
      iree_hal_buffer_view_variant_from_ptr_move(&hal_, &input.view),
      iree_hal_buffer_variant_from_ptr_move(&hal_, &input.buffer)};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      Invoke(IREE_SV("failure"), iree_vm_variant_span_from_array(arguments),
             {&result, 1}));
  EXPECT_EQ(input.releases, 1);
  EXPECT_EQ(View(result), sentinel_view);
  auto site = iree_vm_variant_from_ref_retained(native_.failure_site);
  ReleaseVM();
  iree_vm_ref_reset(&native_.failure_site);
  ExpectLocation(site, 73);
  iree_vm_variant_reset(&site);
  iree_vm_variant_reset(&result);
}

TEST_F(NativeReferencesTest, PrivateStoragePreservesReplacedAndRepeatedOwners) {
  for (bool use_null : {false, true}) {
    SCOPED_TRACE(use_null);
    HalAllocation left;
    HalAllocation right;
    auto* original = left.view;
    auto* replacement = use_null ? nullptr : right.view;
    iree_vm_variant_t arguments[] = {
        iree_hal_buffer_view_variant_from_ptr_move(&hal_, &left.view),
        use_null
            ? iree_hal_buffer_view_variant_from_ptr_borrowed(&hal_, nullptr)
            : iree_hal_buffer_view_variant_from_ptr_move(&hal_, &right.view)};
    left.ReleaseOwners();
    right.ReleaseOwners();
    iree_vm_variant_t results[3] = {};
    IREE_ASSERT_OK(Invoke(IREE_SV("reference_storage"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    EXPECT_EQ(View(results[0]), original);
    EXPECT_EQ(View(results[1]), replacement);
    EXPECT_EQ(View(results[2]), replacement);
    left.ExpectView(View(results[0]));
    if (!use_null) {
      right.ExpectView(View(results[1]));
    }
    iree_vm_variant_reset(&results[1]);
    EXPECT_EQ(right.releases, use_null ? 1 : 0);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    EXPECT_EQ(left.releases, 1);
    EXPECT_EQ(right.releases, 1);
  }
}

TEST_F(NativeReferencesTest, NativeFailureReleasesPrivateStorageOwners) {
  HalAllocation input;
  iree_vm_variant_t arguments[] = {
      iree_hal_buffer_view_variant_from_ptr_move(&hal_, &input.view),
      iree_vm_buffer_variant_from_ptr_borrowed(&core_, nullptr)};
  input.ReleaseOwners();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        Invoke(IREE_SV("reference_storage_failure"),
                               iree_vm_variant_span_from_array(arguments), {}));
  EXPECT_EQ(input.releases, 1);
}

TEST_F(NativeReferencesTest, SaturatedPermutationRetainsAndReleasesOwners) {
  for (int32_t count : {0, 1, 2, 3}) {
    for (bool use_null : {false, true}) {
      SCOPED_TRACE(count);
      SCOPED_TRACE(use_null);
      HalAllocation left;
      HalAllocation right;
      auto* first = left.view;
      auto* second = use_null ? nullptr : right.view;
      iree_vm_variant_t arguments[257] = {};
      for (size_t i = 0; i < 256; ++i) {
        arguments[i] = iree_hal_buffer_view_variant_from_ptr_retained(
            &hal_, i % 2 ? second : first);
      }
      arguments[256] = iree_vm_variant_from_i32(count);
      left.ReleaseOwners();
      right.ReleaseOwners();
      iree_vm_variant_t results[2] = {};
      IREE_ASSERT_OK(Invoke(IREE_SV("reference_permutation"),
                            iree_vm_variant_span_from_array(arguments),
                            iree_vm_variant_span_from_array(results)));
      EXPECT_EQ(View(results[0]), count % 2 ? second : first);
      EXPECT_EQ(View(results[1]), count % 2 ? first : second);
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
      EXPECT_EQ(left.releases, 1);
      EXPECT_EQ(right.releases, 1);
    }
  }
}

TEST_F(NativeReferencesTest, StoredCallBoundariesPreserveAndUnwindOwners) {
  constexpr size_t kCount = 257;
  for (int32_t count : {0, 1, 3}) {
    SCOPED_TRACE(count);
    for (bool should_fail : {false, true}) {
      SCOPED_TRACE(should_fail);
      std::array<HalAllocation, 17> allocations;
      std::array<iree_hal_buffer_view_t*, kCount> expected;
      iree_vm_variant_t arguments[kCount + 2] = {};
      for (size_t i = 0; i < kCount; ++i) {
        expected[i] =
            i % 11 ? allocations[i % allocations.size()].view : nullptr;
        arguments[i] =
            iree_hal_buffer_view_variant_from_ptr_retained(&hal_, expected[i]);
      }
      arguments[kCount] = iree_vm_variant_from_i32(count);
      arguments[kCount + 1] = iree_vm_variant_from_i32(should_fail);
      for (auto& allocation : allocations) {
        allocation.ReleaseOwners();
      }
      iree_vm_variant_t results[kCount] = {};
      if (should_fail) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                              Invoke(IREE_SV("many_owners"),
                                     iree_vm_variant_span_from_array(arguments),
                                     iree_vm_variant_span_from_array(results)));
      } else {
        IREE_ASSERT_OK(Invoke(IREE_SV("many_owners"),
                              iree_vm_variant_span_from_array(arguments),
                              iree_vm_variant_span_from_array(results)));
        for (size_t i = 0; i < kCount; ++i) {
          EXPECT_EQ(View(results[i]),
                    expected[(kCount - 1 - i + 19 * count) % kCount]);
        }
      }
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
      for (const auto& allocation : allocations) {
        EXPECT_EQ(allocation.releases, 1);
      }
    }
  }
}

TEST_F(NativeReferencesTest, RejectsWrongReferenceAtPublicInvocation) {
  HalAllocation input;
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(1),
      iree_hal_buffer_variant_from_ptr_move(&hal_, &input.buffer),
      iree_hal_buffer_view_variant_from_ptr_move(&hal_, &input.view)};
  iree_vm_variant_t result = iree_vm_variant_from_i64(123);
  const auto before = result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Invoke(IREE_SV("select"), iree_vm_variant_span_from_array(arguments),
             {&result, 1}));
  EXPECT_EQ(result.payload, before.payload);
  EXPECT_EQ(result.metadata, before.metadata);
  EXPECT_EQ(input.releases, 1);
}

TEST_F(NativeReferencesTest, RejectsIncompatibleNativeSignatureAtLink) {
  ResourceModule wrong;
  IREE_ASSERT_OK(InitializeResourceModule(hal_.buffer, core_.buffer,
                                          core_.buffer, &wrong));
  iree_vm_module_t* libraries[] = {&wrong.base};
  iree_vm_program_t* program = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_program_create(
          {bytecode_, iree_vm_module_span_from_array(libraries)},
          iree_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
  iree_vm_module_release(&wrong.base);
}

TEST(NativeReferenceBindingTest, MissingNamespaceAndTypeFailImageConstruction) {
  iree_vm_ref_type_table_t incomplete = {};
  const iree_vm_ref_type_descriptor_t buffer = {nullptr, &incomplete,
                                                IREE_SVL("buffer")};
  const iree_vm_ref_type_t types[] = {&buffer};
  incomplete = {sizeof(incomplete),
                IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
                IREE_SV("hal"),
                {types, 1}};
  for (auto* table :
       {static_cast<const iree_vm_ref_type_table_t*>(nullptr),
        static_cast<const iree_vm_ref_type_table_t*>(&incomplete)}) {
    iree_vm_environment_t* environment = nullptr;
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment));
    if (table) {
      IREE_ASSERT_OK(
          iree_vm_environment_register_ref_type_table(environment, table));
    }
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          LoadImage(environment, &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_environment_free(environment);
  }
}

}  // namespace
