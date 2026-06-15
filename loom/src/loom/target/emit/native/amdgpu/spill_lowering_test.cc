// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_lowering.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace loom {
namespace {

struct LoweredSpillTraffic {
  // Number of low.storage.address ops materialized by spill lowering.
  int storage_address_count = 0;
  // Number of offset-only scratch packet ops materialized by spill lowering.
  int offset_only_packet_count = 0;
  // Number of vaddr scratch packet ops materialized by spill lowering.
  int vaddr_packet_count = 0;
  // Number of SGPR-to-VGPR copy ops materialized before scratch stores.
  int sgpr_to_vgpr_copy_count = 0;
  // Number of VGPR-to-SGPR readfirstlane ops materialized after scratch loads.
  int vgpr_to_sgpr_read_count = 0;
  // Number of EXEC read ops materialized around SGPR scratch traffic.
  int exec_read_count = 0;
  // Number of full-EXEC write ops materialized around SGPR scratch traffic.
  int exec_full_count = 0;
  // Number of EXEC restore ops materialized around SGPR scratch traffic.
  int exec_restore_count = 0;
  // Offset carried by the materialized low.storage.address op.
  int64_t storage_address_offset = 0;
  // Offset immediate carried by the materialized scratch packet.
  int64_t packet_offset = 0;
  // Explicit operand count on the materialized scratch packet.
  uint16_t packet_operand_count = 0;
};

enum class SpillTrafficOp {
  kSpill,
  kReload,
};

enum class SpillRegisterClass {
  kSgpr,
  kVgpr,
};

class AmdgpuSpillLoweringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry, IREE_SV("amdgpu.rdna3.core"));
    ASSERT_NE(descriptor_set_, nullptr);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return loom_symbol_ref_t{
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  loom_type_t RegisterType(SpillRegisterClass register_class,
                           uint32_t unit_count) {
    loom_type_t type = loom_type_none();
    const uint16_t register_class_id =
        register_class == SpillRegisterClass::kSgpr
            ? LOOM_AMDGPU_REG_CLASS_ID_SGPR
            : LOOM_AMDGPU_REG_CLASS_ID_VGPR;
    IREE_CHECK_OK(loom_low_build_register_type(
        descriptor_set_, register_class_id, unit_count, &type));
    loom_type_t interned_type = loom_type_none();
    IREE_CHECK_OK(loom_module_intern_type(module_, type, &interned_type));
    return interned_type;
  }

  loom_op_t* BuildFunction(
      iree_string_view_t name, int64_t byte_offset, SpillTrafficOp traffic_op,
      SpillRegisterClass register_class = SpillRegisterClass::kVgpr,
      uint32_t unit_count = 1) {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    const loom_symbol_ref_t target_ref = AddSymbol(IREE_SV("target"));
    const loom_symbol_ref_t function_ref = AddSymbol(name);
    const loom_type_t arg_type = RegisterType(register_class, unit_count);
    loom_op_t* function_op = nullptr;
    IREE_CHECK_OK(loom_low_func_def_build(
        &module_builder, 0, /*visibility=*/0, /*cc=*/0, /*purity=*/0,
        /*allocation=*/0, /*schedule=*/0, target_ref, /*abi=*/0,
        loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), function_ref,
        &arg_type,
        /*arg_types_count=*/1, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    loom_region_t* body = loom_low_func_def_body(function_op);
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body), &body_builder);
    loom_op_t* storage_op = nullptr;
    IREE_CHECK_OK(loom_low_storage_reserve_build(
        &body_builder, /*byte_length=*/8192, /*byte_alignment=*/4,
        loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), LOOM_LOCATION_UNKNOWN,
        &storage_op));
    const loom_value_id_t storage =
        loom_low_storage_reserve_storage(storage_op);
    if (traffic_op == SpillTrafficOp::kSpill) {
      loom_op_t* spill_op = nullptr;
      IREE_CHECK_OK(loom_low_spill_build(
          &body_builder, loom_region_entry_arg_id(body, 0), storage,
          byte_offset, LOOM_LOCATION_UNKNOWN, &spill_op));
    } else {
      loom_op_t* reload_op = nullptr;
      IREE_CHECK_OK(loom_low_reload_build(&body_builder, storage, byte_offset,
                                          arg_type, LOOM_LOCATION_UNKNOWN,
                                          &reload_op));
    }
    return function_op;
  }

  static int64_t PacketOffset(const loom_op_t* op) {
    const loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    if (attrs.count != 1 || attrs.entries[0].value.kind != LOOM_ATTR_I64) {
      return INT64_MIN;
    }
    return attrs.entries[0].value.i64;
  }

  LoweredSpillTraffic SummarizeLoweredTraffic(
      const loom_op_t* function_op, loom_amdgpu_descriptor_ref_t offset_ref,
      loom_amdgpu_descriptor_ref_t vaddr_ref) {
    LoweredSpillTraffic traffic = {};
    const uint32_t offset_only_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, offset_ref);
    const uint32_t vaddr_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, vaddr_ref);
    const uint32_t sgpr_to_vgpr_copy_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(
            descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY);
    const uint32_t vgpr_to_sgpr_read_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(
            descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32);
    const uint32_t exec_read_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ);
    const uint32_t exec_full_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_FULL);
    const uint32_t exec_restore_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set_, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC);
    const loom_region_t* body = loom_low_func_def_body(function_op);
    const loom_block_t* block = loom_region_const_entry_block(body);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
        ADD_FAILURE() << "storage traffic op survived lowering";
      } else if (loom_low_storage_address_isa(op)) {
        ++traffic.storage_address_count;
        traffic.storage_address_offset = loom_low_storage_address_offset(op);
      } else if (loom_low_op_isa(op)) {
        const int64_t descriptor_ordinal = loom_low_op_descriptor_ordinal(op);
        if (descriptor_ordinal == offset_only_ordinal) {
          ++traffic.offset_only_packet_count;
          traffic.packet_offset = PacketOffset(op);
          traffic.packet_operand_count = op->operand_count;
        } else if (descriptor_ordinal == vaddr_ordinal) {
          ++traffic.vaddr_packet_count;
          traffic.packet_offset = PacketOffset(op);
          traffic.packet_operand_count = op->operand_count;
        } else if (descriptor_ordinal == sgpr_to_vgpr_copy_ordinal) {
          ++traffic.sgpr_to_vgpr_copy_count;
        } else if (descriptor_ordinal == vgpr_to_sgpr_read_ordinal) {
          ++traffic.vgpr_to_sgpr_read_count;
        } else if (descriptor_ordinal == exec_read_ordinal) {
          ++traffic.exec_read_count;
        } else if (descriptor_ordinal == exec_full_ordinal) {
          ++traffic.exec_full_count;
        } else if (descriptor_ordinal == exec_restore_ordinal) {
          ++traffic.exec_restore_count;
        }
      }
    }
    return traffic;
  }

  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(AmdgpuSpillLoweringTest, KeepsOffsetOnlyStoreWhenOffsetFits) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("small_spill"), 4092, SpillTrafficOp::kSpill);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 0);
  EXPECT_EQ(traffic.offset_only_packet_count, 1);
  EXPECT_EQ(traffic.vaddr_packet_count, 0);
  EXPECT_EQ(traffic.exec_read_count, 0);
  EXPECT_EQ(traffic.exec_full_count, 0);
  EXPECT_EQ(traffic.exec_restore_count, 0);
  EXPECT_EQ(traffic.packet_offset, 4092);
  EXPECT_EQ(traffic.packet_operand_count, 1);
}

TEST_F(AmdgpuSpillLoweringTest, UsesVaddrStoreWhenOffsetExceedsImmediate) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("large_spill"), 4104, SpillTrafficOp::kSpill);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 1);
  EXPECT_EQ(traffic.storage_address_offset, 4104);
  EXPECT_EQ(traffic.offset_only_packet_count, 0);
  EXPECT_EQ(traffic.vaddr_packet_count, 1);
  EXPECT_EQ(traffic.packet_offset, 0);
  EXPECT_EQ(traffic.packet_operand_count, 2);
}

TEST_F(AmdgpuSpillLoweringTest, KeepsOffsetOnlyReloadWhenOffsetFits) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("small_reload"), 4092, SpillTrafficOp::kReload);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 0);
  EXPECT_EQ(traffic.offset_only_packet_count, 1);
  EXPECT_EQ(traffic.vaddr_packet_count, 0);
  EXPECT_EQ(traffic.packet_offset, 4092);
  EXPECT_EQ(traffic.packet_operand_count, 0);
}

TEST_F(AmdgpuSpillLoweringTest, UsesVaddrReloadWhenOffsetExceedsImmediate) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("large_reload"), 4104, SpillTrafficOp::kReload);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 1);
  EXPECT_EQ(traffic.storage_address_offset, 4104);
  EXPECT_EQ(traffic.offset_only_packet_count, 0);
  EXPECT_EQ(traffic.vaddr_packet_count, 1);
  EXPECT_EQ(traffic.packet_offset, 0);
  EXPECT_EQ(traffic.packet_operand_count, 1);
}

TEST_F(AmdgpuSpillLoweringTest, LowersSgprSpillThroughVgprScratchValue) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("sgpr_spill"), 0, SpillTrafficOp::kSpill,
                    SpillRegisterClass::kSgpr, 2);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_STORE_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 0);
  EXPECT_EQ(traffic.offset_only_packet_count, 2);
  EXPECT_EQ(traffic.vaddr_packet_count, 0);
  EXPECT_EQ(traffic.sgpr_to_vgpr_copy_count, 2);
  EXPECT_EQ(traffic.vgpr_to_sgpr_read_count, 0);
  EXPECT_EQ(traffic.exec_read_count, 1);
  EXPECT_EQ(traffic.exec_full_count, 1);
  EXPECT_EQ(traffic.exec_restore_count, 1);
  EXPECT_EQ(traffic.packet_offset, 4);
  EXPECT_EQ(traffic.packet_operand_count, 1);
}

TEST_F(AmdgpuSpillLoweringTest, LowersSgprReloadThroughReadfirstlane) {
  loom_op_t* function_op =
      BuildFunction(IREE_SV("sgpr_reload"), 0, SpillTrafficOp::kReload,
                    SpillRegisterClass::kSgpr, 2);

  IREE_ASSERT_OK(loom_amdgpu_lower_spill_traffic(
      module_, function_op, descriptor_set_, &scratch_arena_));

  LoweredSpillTraffic traffic = SummarizeLoweredTraffic(
      function_op, LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_SCRATCH_LOAD_B32_VADDR);
  EXPECT_EQ(traffic.storage_address_count, 0);
  EXPECT_EQ(traffic.offset_only_packet_count, 2);
  EXPECT_EQ(traffic.vaddr_packet_count, 0);
  EXPECT_EQ(traffic.sgpr_to_vgpr_copy_count, 0);
  EXPECT_EQ(traffic.vgpr_to_sgpr_read_count, 2);
  EXPECT_EQ(traffic.exec_read_count, 1);
  EXPECT_EQ(traffic.exec_full_count, 1);
  EXPECT_EQ(traffic.exec_restore_count, 1);
  EXPECT_EQ(traffic.packet_offset, 4);
  EXPECT_EQ(traffic.packet_operand_count, 0);
}

}  // namespace
}  // namespace loom
