// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

iree_vm_instance_t* hal_testbench_actual_test_vm_instance = nullptr;
static int hal_testbench_actual_test_resolver_call_count = 0;

static iree_status_t hal_testbench_actual_test_resolve_target(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_symbol_ref_t* out_target_ref) {
  (void)provider;
  (void)target;
  ++hal_testbench_actual_test_resolver_call_count;
  const loom_string_id_t target_name_id =
      loom_module_lookup_string(module, IREE_SV("target"));
  if (target_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "test target symbol name is not interned");
  }
  const uint16_t target_symbol_id =
      loom_module_find_symbol(module, target_name_id);
  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "test target symbol does not exist");
  }
  *out_target_ref = (loom_symbol_ref_t){
      /*.module_id=*/0,
      /*.symbol_id=*/target_symbol_id,
  };
  return iree_ok_status();
}

static const loom_run_hal_artifact_provider_t
    hal_testbench_actual_test_artifact_provider = {
        /*.name=*/IREE_SVL("test-hal"),
        /*.hal_driver_name=*/{},
        /*.target_family_name=*/IREE_SVL("test"),
        /*.default_pipeline_options=*/{},
        /*.select_device_target=*/{},
        /*.select_target_key=*/{},
        /*.deinitialize_device_target=*/{},
        /*.resolve_device_target_ref=*/hal_testbench_actual_test_resolve_target,
};

static bool hal_testbench_actual_test_symbol_refs_equal(loom_symbol_ref_t lhs,
                                                        loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

class HalTestbenchActualTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(),
        &hal_testbench_actual_test_vm_instance));
  }

  static void TearDownTestSuite() {
    iree_vm_instance_release(hal_testbench_actual_test_vm_instance);
    hal_testbench_actual_test_vm_instance = nullptr;
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    hal_testbench_actual_test_resolver_call_count = 0;
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("hal_testbench_actual_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_symbol_ref_t FindSymbolRef(const loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_op_t* FindKernel(loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_ref_t symbol_ref = FindSymbolRef(module, name);
    loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
    IREE_ASSERT(symbol->defining_op != nullptr);
    IREE_ASSERT(loom_kernel_def_isa(symbol->defining_op));
    return symbol->defining_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
};

TEST_F(HalTestbenchActualTest, ScalarInputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_i32(0x12345678)),
      iree_vm_make_variant_value(
          iree_vm_value_make_i64(static_cast<int64_t>(0x1122334455667788ull))),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 3u);
  EXPECT_EQ(options.constants[0], 0x12345678u);
  EXPECT_EQ(options.constants[1], 0x55667788u);
  EXPECT_EQ(options.constants[2], 0x11223344u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, F64InputsPackDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_f64(1.0)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F64),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x00000000u);
  EXPECT_EQ(options.constants[1], 0x3ff00000u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, IndexInputPacksAsOneDispatchConstantWord) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(iree_vm_value_make_i64(3584)),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 1u);
  EXPECT_EQ(options.constants[0], 3584u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest, OffsetInputPacksAsTwoDispatchConstantWords) {
  iree_vm_variant_t inputs[] = {
      iree_vm_make_variant_value(
          iree_vm_value_make_i64(static_cast<int64_t>(0x1122334455667788ull))),
  };
  loom_type_t input_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
  };
  loom_run_hal_invocation_options_t options = {};
  loom_run_hal_invocation_options_initialize(&options);
  iree_vm_list_t* bindings = nullptr;

  IREE_ASSERT_OK(loom_run_hal_testbench_invocation_inputs_from_variants(
      inputs, input_types, IREE_ARRAYSIZE(inputs), &options,
      iree_allocator_system(), &bindings));

  EXPECT_EQ(iree_vm_list_size(bindings), 0u);
  EXPECT_EQ(options.constant_count, 2u);
  EXPECT_EQ(options.constants[0], 0x55667788u);
  EXPECT_EQ(options.constants[1], 0x11223344u);

  iree_vm_list_release(bindings);
}

TEST_F(HalTestbenchActualTest,
       TargetlessAssignmentIsLazyWhenAllKernelsTargeted) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target

kernel.def target(@target) @entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}
)");
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("target"));

  loom_run_hal_targetless_kernel_assignment_result_t result = {};
  loom_run_hal_device_target_t device_target = {};
  IREE_ASSERT_OK(loom_run_hal_assign_targetless_kernel_targets(
      &hal_testbench_actual_test_artifact_provider, &device_target,
      module.get(), &result));

  EXPECT_EQ(hal_testbench_actual_test_resolver_call_count, 0);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(result.targetless_kernel_count, 0u);
  EXPECT_EQ(result.assigned_kernel_count, 0u);
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(
      loom_kernel_def_target(FindKernel(module.get(), IREE_SV("entry"))),
      target_ref));
}

TEST_F(HalTestbenchActualTest, TargetlessAssignmentUsesProviderOnce) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target

kernel.def @first() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}

kernel.def @second() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}
)");
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("target"));

  loom_run_hal_targetless_kernel_assignment_result_t result = {};
  loom_run_hal_device_target_t device_target = {};
  IREE_ASSERT_OK(loom_run_hal_assign_targetless_kernel_targets(
      &hal_testbench_actual_test_artifact_provider, &device_target,
      module.get(), &result));

  EXPECT_EQ(hal_testbench_actual_test_resolver_call_count, 1);
  EXPECT_TRUE(result.changed);
  EXPECT_EQ(result.targetless_kernel_count, 2u);
  EXPECT_EQ(result.assigned_kernel_count, 2u);
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(result.target_ref,
                                                          target_ref));
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(
      loom_kernel_def_target(FindKernel(module.get(), IREE_SV("first"))),
      target_ref));
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(
      loom_kernel_def_target(FindKernel(module.get(), IREE_SV("second"))),
      target_ref));
}

TEST_F(HalTestbenchActualTest, TargetlessAssignmentPreservesExplicitTargets) {
  ModulePtr module = ParseModule(R"(
test.target<low_core> @target
test.target<low_core> @explicit_target

kernel.def target(@explicit_target) @explicit_entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}

kernel.def @targetless_entry() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch() {
  kernel.return
}
)");
  const loom_symbol_ref_t target_ref =
      FindSymbolRef(module.get(), IREE_SV("target"));
  const loom_symbol_ref_t explicit_target_ref =
      FindSymbolRef(module.get(), IREE_SV("explicit_target"));

  loom_run_hal_targetless_kernel_assignment_result_t result = {};
  loom_run_hal_device_target_t device_target = {};
  IREE_ASSERT_OK(loom_run_hal_assign_targetless_kernel_targets(
      &hal_testbench_actual_test_artifact_provider, &device_target,
      module.get(), &result));

  EXPECT_EQ(hal_testbench_actual_test_resolver_call_count, 1);
  EXPECT_TRUE(result.changed);
  EXPECT_EQ(result.targetless_kernel_count, 1u);
  EXPECT_EQ(result.assigned_kernel_count, 1u);
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(
      loom_kernel_def_target(
          FindKernel(module.get(), IREE_SV("explicit_entry"))),
      explicit_target_ref));
  EXPECT_TRUE(hal_testbench_actual_test_symbol_refs_equal(
      loom_kernel_def_target(
          FindKernel(module.get(), IREE_SV("targetless_entry"))),
      target_ref));
}

}  // namespace
}  // namespace loom
