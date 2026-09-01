// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer_provider.h"
#include "iree/vm/bytecode/interpreter_atomic.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/process.h"

namespace iree::vm::bytecode::testing {
namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

struct BufferTestObject {
  // Required offset-zero VM-visible ownership prefix.
  iree_vm_ref_object_t ref_object;
};

struct BufferTestTypes {
  // Type deliberately incompatible with vm.buffer.
  iree_vm_ref_type_t object;
};

extern const iree_vm_ref_type_table_t kBufferTestTypeTable;

const iree_vm_ref_type_descriptor_t kBufferTestObjectType = {
    /*destroy=*/nullptr,
    &kBufferTestTypeTable,
    IREE_SV("object"),
};
const BufferTestTypes kBufferTestTypes = {
    &kBufferTestObjectType,
};
const iree_vm_ref_type_table_t kBufferTestTypeTable = {
    sizeof(kBufferTestTypeTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("zz_test"),
    {&kBufferTestTypes, 1},
};

iree_vm_variant_t VariantFromI64Bits(uint64_t bits) {
  int64_t value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return iree_vm_variant_from_i64(value);
}

void ExpectVariantEqual(iree_vm_variant_t actual, iree_vm_variant_t expected) {
  EXPECT_EQ(actual.payload, expected.payload);
  EXPECT_EQ(actual.metadata, expected.metadata);
}

struct BufferExecutionHarness {
  // Immutable storage backing the loaded bytecode module.
  std::vector<uint8_t> image;
  // Owned bytecode module.
  iree_vm_module_t* module = nullptr;
  // Owned immutable linked program.
  iree_vm_program_t* program = nullptr;
  // Owned initialized process.
  iree_vm_process_t* process = nullptr;
  // Owned reusable invocation storage.
  iree_vm_invocation_t* invocation = nullptr;
  // Module-resolved core type prefix.
  iree_vm_ref_types_t vm_types = {};

  BufferExecutionHarness() : image(BuildBufferModuleImage()) {}
  BufferExecutionHarness(const BufferExecutionHarness&) = delete;
  BufferExecutionHarness& operator=(const BufferExecutionHarness&) = delete;

  ~BufferExecutionHarness() {
    iree_vm_process_release(process);
    iree_vm_invocation_free(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(module);
  }

  iree_status_t Initialize() {
    iree_vm_environment_t* environment = nullptr;
    iree_status_t status =
        iree_vm_environment_allocate(iree_allocator_system(), &environment);
    if (iree_status_is_ok(status)) {
      status = iree_vm_environment_register_ref_type_table(
          environment, &kBufferTestTypeTable);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_bytecode_module_create(
          environment, IREE_SV("buffer_ops"),
          {iree_make_const_byte_span(image.data(), image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &module);
    }
    iree_vm_environment_free(environment);
    if (iree_status_is_ok(status)) {
      status = iree_vm_module_ref_type_by_ordinal(module, 0, &vm_types.buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_program_create({module, iree_vm_module_span_empty()},
                                      iree_allocator_system(), &program);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_invocation_allocate(
          kInvocationStorageSize, iree_allocator_system(), &invocation);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_process_create(program, invocation,
                                      iree_vm_variant_span_empty(),
                                      iree_allocator_system(), &process);
    }
    return status;
  }

  iree_status_t Invoke(iree_string_view_t name,
                       iree_vm_variant_span_t arguments,
                       iree_vm_variant_span_t results) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process, IREE_SV("buffer_ops"), name, &function));
    return iree_vm_invoke(invocation, function, arguments, results);
  }
};

void RewriteI64RoundtripAsAtomicAdd(BufferExecutionHarness* harness) {
  MutableFunctionImage function =
      FindFunctionImage(&harness->image, kBufferRoundtripFunctionBase +
                                             IREE_VM_ISA_MEMORY_FORMAT_I64_X1);
  ASSERT_NE(function.row, nullptr);
  ASSERT_EQ(function.row->bytecode_length_u32, 32u);

  std::memset(function.bytecode, 0, function.row->bytecode_length_u32);
  const iree_vm_isa_control_block_record_t block = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}};
  std::memcpy(function.bytecode, &block, sizeof(block));
  const iree_vm_isa_constant_zero_record_t offset = {
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 1, 0};
  std::memcpy(function.bytecode + 4, &offset, sizeof(offset));
  const iree_vm_isa_buffer_atomic_rmw_record_t atomic = {
      IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_RMW,
      0,
      0,
      1,
      0,
      (uint8_t)(IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER |
                (IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I64 << 7)),
      (uint8_t)(IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM << 3),
      0,
  };
  std::memcpy(function.bytecode + 8, &atomic, sizeof(atomic));
  const iree_vm_isa_constant_zero_record_t zero = {
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 2, 0};
  std::memcpy(function.bytecode + 16, &zero, sizeof(zero));
  std::memcpy(function.bytecode + 20, &zero, sizeof(zero));
  std::memcpy(function.bytecode + 24, &zero, sizeof(zero));
  const iree_vm_isa_control_return_record_t return_record = {
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}};
  std::memcpy(function.bytecode + 28, &return_record, sizeof(return_record));
}

TEST(VMBytecodeModuleBufferTest, VerifiesBufferInstructionRecords) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  IREE_ASSERT_OK(iree_vm_environment_register_ref_type_table(
      environment, &kBufferTestTypeTable));

  std::vector<uint8_t> valid_image = BuildBufferModuleImage();
  iree_vm_module_t* valid_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("buffer_ops"),
      {iree_make_const_byte_span(valid_image.data(), valid_image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &valid_module));
  iree_vm_module_release(valid_module);

  const auto expect_rejected = [&](const char* label, uint32_t ordinal,
                                   const auto& mutate) {
    SCOPED_TRACE(label);
    std::vector<uint8_t> image = BuildBufferModuleImage();
    MutableFunctionImage function = FindFunctionImage(&image, ordinal);
    ASSERT_NE(function.row, nullptr);
    mutate(function);
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("malformed_buffer"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };
  const auto write_record = [](MutableFunctionImage function,
                               const auto& record) {
    function.row->value_register_count_u16 = 6;
    function.row->ref_register_count_u16 = 2;
    uint8_t* cursor = function.bytecode;
    const uint8_t* const end =
        function.bytecode + function.row->bytecode_length_u32;
    const iree_vm_isa_control_block_record_t block = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}};
    std::memcpy(cursor, &block, sizeof(block));
    cursor += sizeof(block);
    ASSERT_LE(sizeof(record), static_cast<size_t>(end - cursor));
    std::memcpy(cursor, &record, sizeof(record));
    cursor += sizeof(record);
    const iree_vm_isa_constant_zero_record_t zero = {
        IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0};
    while (cursor < end - sizeof(iree_vm_isa_control_return_record_t)) {
      std::memcpy(cursor, &zero, sizeof(zero));
      cursor += sizeof(zero);
    }
    const iree_vm_isa_control_return_record_t return_record = {
        IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}};
    std::memcpy(cursor, &return_record, sizeof(return_record));
    cursor += sizeof(return_record);
    ASSERT_EQ(cursor, end);
  };
  const auto expect_record_status = [&](const char* label, const auto& record,
                                        iree_status_code_t expected_code) {
    SCOPED_TRACE(label);
    std::vector<uint8_t> image = BuildBufferModuleImage();
    MutableFunctionImage function =
        FindFunctionImage(&image, kBufferRoundtripFunctionBase);
    ASSERT_NE(function.row, nullptr);
    write_record(function, record);
    iree_vm_module_t* module = nullptr;
    IREE_EXPECT_STATUS_IS(
        expected_code,
        iree_vm_bytecode_module_create(
            environment, IREE_SV("buffer_record"),
            {iree_make_const_byte_span(image.data(), image.size()),
             iree_allocator_null()},
            iree_allocator_system(), &module));
    EXPECT_EQ(module, nullptr);
    iree_vm_module_release(module);
  };
  const auto expect_record_rejected = [&](const char* label,
                                          const auto& record) {
    expect_record_status(label, record, IREE_STATUS_INVALID_ARGUMENT);
  };
  const auto expect_record_accepted = [&](const char* label,
                                          const auto& record) {
    SCOPED_TRACE(label);
    std::vector<uint8_t> image = BuildBufferModuleImage();
    MutableFunctionImage function =
        FindFunctionImage(&image, kBufferRoundtripFunctionBase);
    ASSERT_NE(function.row, nullptr);
    write_record(function, record);
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        environment, IREE_SV("valid_buffer_record"),
        {iree_make_const_byte_span(image.data(), image.size()),
         iree_allocator_null()},
        iree_allocator_system(), &module));
    iree_vm_module_release(module);
  };

  expect_rejected("allocate ref register", kBufferAllocateFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->ref_register_count_u16 = 0;
                  });
  expect_rejected("allocate value register", kBufferAllocateFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->value_register_count_u16 = 0;
                  });
  expect_rejected("allocate alignment exponent", kBufferAllocateFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_allocate_record_t*>(
                            function.bytecode + 4);
                    record->minimum_alignment_log2_u8 = 64;
                  });
  expect_rejected("length value register", kBufferLengthFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->value_register_count_u16 = 0;
                  });
  expect_rejected("length ref register", kBufferLengthFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->ref_register_count_u16 = 0;
                  });
  expect_rejected("length padding", kBufferLengthFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_length_record_t*>(
                            function.bytecode + 4);
                    record->zero_padding_u8 = 1;
                  });
  expect_rejected("subspan value register", kBufferSubspanFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->value_register_count_u16 = 1;
                  });
  expect_rejected("subspan ref register", kBufferSubspanFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->ref_register_count_u16 = 0;
                  });
  expect_rejected("subspan padding", kBufferSubspanFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_subspan_record_t*>(
                            function.bytecode + 4);
                    record->zero_padding_u8[1] = 1;
                  });
  expect_rejected("load address register", kBufferLoadFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_load_record_t*>(
                            function.bytecode + 4);
                    record->base_v8 = 2;
                  });
  expect_rejected("load ref register", kBufferLoadFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->ref_register_count_u16 = 0;
                  });
  expect_rejected("load lane range", kBufferLoadFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_load_record_t*>(
                            function.bytecode + 4);
                    record->format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X8;
                  });
  expect_rejected("load format", kBufferLoadFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_load_record_t*>(
                            function.bytecode + 4);
                    record->format_u8 = UINT8_MAX;
                  });
  expect_rejected("load padding", kBufferLoadFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_load_record_t*>(
                            function.bytecode + 4);
                    record->zero_padding_u8 = 1;
                  });
  expect_rejected("store address register", kBufferStoreFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_store_record_t*>(
                            function.bytecode + 4);
                    record->base_v8 = 3;
                  });
  expect_rejected("store ref register", kBufferStoreFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    function.row->ref_register_count_u16 = 0;
                  });
  expect_rejected("store lane range", kBufferStoreFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_store_record_t*>(
                            function.bytecode + 4);
                    record->format_u8 = IREE_VM_ISA_MEMORY_FORMAT_I64_X8;
                  });
  expect_rejected("store format", kBufferStoreFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_store_record_t*>(
                            function.bytecode + 4);
                    record->format_u8 = UINT8_MAX;
                  });
  expect_rejected("store padding", kBufferStoreFunctionOrdinal,
                  [](MutableFunctionImage function) {
                    auto* record =
                        reinterpret_cast<iree_vm_isa_buffer_store_record_t*>(
                            function.bytecode + 4);
                    record->zero_padding_u8 = 1;
                  });

  iree_vm_isa_buffer_atomic_reduce_record_t atomic_reduce = {};
  atomic_reduce.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_REDUCE;
  atomic_reduce.operand_v8 = 1;
  atomic_reduce.selector0_u8 = IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER;
  atomic_reduce.zero_padding_u16 = 1;
  expect_record_rejected("atomic reduce padding", atomic_reduce);
  atomic_reduce.zero_padding_u16 = 0;
  atomic_reduce.selector0_u8 = IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER;
  expect_record_rejected("atomic reduce exchange integer", atomic_reduce);
  atomic_reduce.selector0_u8 = IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT;
  expect_record_rejected("atomic reduce exchange float", atomic_reduce);
  atomic_reduce.selector0_u8 =
      IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER | 0x10u;
  expect_record_rejected("atomic reduce selector0 reserved bits",
                         atomic_reduce);
  atomic_reduce.selector0_u8 = IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER;
  atomic_reduce.selector1_u8 = 0x05u;
  expect_record_rejected("atomic reduce ordering", atomic_reduce);
  atomic_reduce.selector1_u8 = 0x28u;
  expect_record_rejected("atomic reduce scope", atomic_reduce);
  atomic_reduce.selector1_u8 = 0;
  atomic_reduce.buffer_r8 = 2;
  expect_record_rejected("atomic reduce ref register", atomic_reduce);
  atomic_reduce.buffer_r8 = 0;
  atomic_reduce.offset_v8 = 6;
  expect_record_rejected("atomic reduce offset register", atomic_reduce);
  atomic_reduce.offset_v8 = 0;
  atomic_reduce.operand_v8 = 6;
  expect_record_rejected("atomic reduce operand register", atomic_reduce);
  atomic_reduce.operand_v8 = 1;

  iree_vm_isa_buffer_atomic_rmw_record_t atomic_rmw = {};
  atomic_rmw.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_RMW;
  atomic_rmw.operand_v8 = 1;
  atomic_rmw.zero_padding_u8 = 1;
  expect_record_rejected("atomic rmw padding", atomic_rmw);
  atomic_rmw.zero_padding_u8 = 0;
  atomic_rmw.selector0_u8 = 0x10u;
  expect_record_rejected("atomic rmw selector0 reserved bits", atomic_rmw);
  atomic_rmw.selector0_u8 = 0;
  atomic_rmw.selector1_u8 = 0x40u;
  expect_record_rejected("atomic rmw selector1 reserved bits", atomic_rmw);
  atomic_rmw.selector1_u8 = 0;
  atomic_rmw.old_v8 = 6;
  expect_record_rejected("atomic rmw result register", atomic_rmw);
  atomic_rmw.old_v8 = 0;
  atomic_rmw.buffer_r8 = 2;
  expect_record_rejected("atomic rmw ref register", atomic_rmw);
  atomic_rmw.buffer_r8 = 0;
  atomic_rmw.offset_v8 = 6;
  expect_record_rejected("atomic rmw offset register", atomic_rmw);
  atomic_rmw.offset_v8 = 0;
  atomic_rmw.operand_v8 = 6;
  expect_record_rejected("atomic rmw operand register", atomic_rmw);
  atomic_rmw.operand_v8 = 1;

  iree_vm_isa_buffer_atomic_cmpxchg_record_t atomic_cmpxchg = {};
  atomic_cmpxchg.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_ATOMIC_CMPXCHG;
  atomic_cmpxchg.expected_v8 = 1;
  atomic_cmpxchg.replacement_v8 = 2;
  atomic_cmpxchg.selector0_u8 = 0x40u;
  expect_record_rejected("atomic cmpxchg selector0 reserved bits",
                         atomic_cmpxchg);
  atomic_cmpxchg.selector0_u8 = 0;
  atomic_cmpxchg.selector1_u8 = 0x08u;
  expect_record_rejected("atomic cmpxchg selector1 reserved bits",
                         atomic_cmpxchg);
  atomic_cmpxchg.selector1_u8 = 0;
  atomic_cmpxchg.old_v8 = 6;
  expect_record_rejected("atomic cmpxchg result register", atomic_cmpxchg);
  atomic_cmpxchg.old_v8 = 0;
  atomic_cmpxchg.buffer_r8 = 2;
  expect_record_rejected("atomic cmpxchg ref register", atomic_cmpxchg);
  atomic_cmpxchg.buffer_r8 = 0;
  atomic_cmpxchg.offset_v8 = 6;
  expect_record_rejected("atomic cmpxchg offset register", atomic_cmpxchg);
  atomic_cmpxchg.offset_v8 = 0;
  atomic_cmpxchg.expected_v8 = 6;
  expect_record_rejected("atomic cmpxchg expected register", atomic_cmpxchg);
  atomic_cmpxchg.expected_v8 = 1;
  atomic_cmpxchg.replacement_v8 = 6;
  expect_record_rejected("atomic cmpxchg replacement register", atomic_cmpxchg);
  atomic_cmpxchg.replacement_v8 = 2;
  atomic_cmpxchg.selector0_u8 = 0x05u;
  expect_record_rejected("atomic cmpxchg success ordering", atomic_cmpxchg);
  atomic_cmpxchg.selector0_u8 = 0x28u;
  expect_record_rejected("atomic cmpxchg failure ordering", atomic_cmpxchg);
  atomic_cmpxchg.selector0_u8 = 0;
  atomic_cmpxchg.selector1_u8 = 0x05u;
  expect_record_rejected("atomic cmpxchg scope", atomic_cmpxchg);
  atomic_cmpxchg.selector1_u8 = 0;

  const bool legal_ordering_pairs[5][5] = {
      {true, false, false, false, false}, {true, true, false, false, false},
      {true, false, false, false, false}, {true, true, false, false, false},
      {true, true, false, false, true},
  };
  for (uint8_t carrier = IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I32;
       carrier <= IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I64; ++carrier) {
    SCOPED_TRACE(static_cast<int>(carrier));
    if (!iree_vm_bytecode_atomic_carrier_is_supported(carrier)) {
      atomic_rmw.selector0_u8 = (uint8_t)(carrier << 7);
      atomic_rmw.selector1_u8 = 0;
      expect_record_status("unsupported atomic carrier", atomic_rmw,
                           IREE_STATUS_INCOMPATIBLE);
      continue;
    }

    for (uint8_t kind = IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER;
         kind <= IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT; ++kind) {
      atomic_reduce.selector0_u8 = (uint8_t)(kind | (carrier << 7));
      atomic_reduce.selector1_u8 = 0;
      expect_record_accepted("atomic reduce kind", atomic_reduce);
    }
    for (uint8_t kind = IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER;
         kind <= IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT; ++kind) {
      atomic_rmw.selector0_u8 = (uint8_t)(kind | (carrier << 7));
      atomic_rmw.selector1_u8 = 0;
      expect_record_accepted("atomic rmw kind", atomic_rmw);
    }
    for (uint8_t ordering = IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED;
         ordering <= IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST; ++ordering) {
      atomic_rmw.selector0_u8 = (uint8_t)(carrier << 7);
      atomic_rmw.selector1_u8 = ordering;
      expect_record_accepted("atomic rmw ordering", atomic_rmw);
    }
    for (uint8_t scope = IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_THREAD;
         scope <= IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM; ++scope) {
      atomic_rmw.selector0_u8 = (uint8_t)(carrier << 7);
      atomic_rmw.selector1_u8 = (uint8_t)(scope << 3);
      expect_record_accepted("atomic rmw scope", atomic_rmw);
      atomic_cmpxchg.selector0_u8 = (uint8_t)(carrier << 7);
      atomic_cmpxchg.selector1_u8 = scope;
      expect_record_accepted("atomic cmpxchg scope", atomic_cmpxchg);
    }
    for (uint8_t success_ordering = 0; success_ordering < 5;
         ++success_ordering) {
      for (uint8_t failure_ordering = 0; failure_ordering < 5;
           ++failure_ordering) {
        atomic_cmpxchg.selector0_u8 =
            (uint8_t)(success_ordering | (failure_ordering << 3) |
                      (carrier << 7));
        atomic_cmpxchg.selector1_u8 = 0;
        if (legal_ordering_pairs[success_ordering][failure_ordering]) {
          expect_record_accepted("atomic cmpxchg legal ordering pair",
                                 atomic_cmpxchg);
        } else {
          expect_record_rejected("atomic cmpxchg illegal ordering pair",
                                 atomic_cmpxchg);
        }
      }
    }
  }

  iree_vm_isa_buffer_fill_record_t fill = {};
  fill.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_FILL;
  fill.length_v8 = 1;
  fill.pattern_v8 = 2;
  fill.pattern_width_u8 = 1;
  fill.zero_padding_u16 = 1;
  expect_record_rejected("fill padding", fill);
  fill.zero_padding_u16 = 0;
  fill.pattern_width_u8 = 3;
  expect_record_rejected("fill pattern width", fill);
  fill.pattern_width_u8 = 1;
  fill.buffer_r8 = 2;
  expect_record_rejected("fill ref register", fill);
  fill.buffer_r8 = 0;
  fill.offset_v8 = 6;
  expect_record_rejected("fill value register", fill);

  iree_vm_isa_buffer_copy_record_t copy = {};
  copy.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_COPY;
  copy.source_r8 = 1;
  copy.source_offset_v8 = 1;
  copy.length_v8 = 2;
  copy.zero_padding_u16 = 1;
  expect_record_rejected("copy padding", copy);
  copy.zero_padding_u16 = 0;
  copy.source_r8 = 2;
  expect_record_rejected("copy ref register", copy);
  copy.source_r8 = 1;
  copy.length_v8 = 6;
  expect_record_rejected("copy value register", copy);

  iree_vm_isa_buffer_compare_record_t compare = {};
  compare.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_COMPARE;
  compare.rhs_r8 = 1;
  compare.rhs_offset_v8 = 1;
  compare.length_v8 = 2;
  compare.zero_padding_u8 = 1;
  expect_record_rejected("compare padding", compare);
  compare.zero_padding_u8 = 0;
  compare.dst_v8 = 6;
  expect_record_rejected("compare result register", compare);
  compare.dst_v8 = 0;
  compare.rhs_r8 = 2;
  expect_record_rejected("compare ref register", compare);

  iree_vm_isa_buffer_copy_rodata_record_t copy_rodata = {};
  copy_rodata.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_COPY_RODATA;
  copy_rodata.length_v8 = 1;
  copy_rodata.zero_padding0_u8 = 1;
  expect_record_rejected("copy rodata leading padding", copy_rodata);
  copy_rodata.zero_padding0_u8 = 0;
  copy_rodata.zero_padding1_u8 = 1;
  expect_record_rejected("copy rodata trailing padding", copy_rodata);
  copy_rodata.zero_padding1_u8 = 0;
  copy_rodata.rodata_u16 = 1;
  expect_record_rejected("copy rodata ordinal", copy_rodata);
  copy_rodata.rodata_u16 = 0;
  copy_rodata.source_offset_u32 = UINT32_MAX;
  expect_record_rejected("copy rodata source offset", copy_rodata);
  copy_rodata.source_offset_u32 = 0;
  copy_rodata.target_r8 = 2;
  expect_record_rejected("copy rodata ref register", copy_rodata);
  copy_rodata.target_r8 = 0;
  copy_rodata.length_v8 = 6;
  expect_record_rejected("copy rodata value register", copy_rodata);

  iree_vm_isa_buffer_rodata_load_record_t rodata_load = {};
  rodata_load.opcode_u8 = IREE_VM_ISA_CORE_OPCODE_BUFFER_RODATA_LOAD;
  rodata_load.dst_r8 = 2;
  expect_record_rejected("rodata load ref register", rodata_load);
  rodata_load.dst_r8 = 0;
  rodata_load.rodata_u16 = 1;
  expect_record_rejected("rodata load ordinal", rodata_load);

  std::vector<uint8_t> arbitrary_scale_image = BuildBufferModuleImage();
  MutableFunctionImage store =
      FindFunctionImage(&arbitrary_scale_image, kBufferStoreFunctionOrdinal);
  ASSERT_NE(store.row, nullptr);
  reinterpret_cast<iree_vm_isa_buffer_store_record_t*>(store.bytecode + 4)
      ->scale_u8 = UINT8_MAX;
  iree_vm_module_t* arbitrary_scale_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      environment, IREE_SV("arbitrary_scale"),
      {iree_make_const_byte_span(arbitrary_scale_image.data(),
                                 arbitrary_scale_image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &arbitrary_scale_module));
  iree_vm_module_release(arbitrary_scale_module);
  iree_vm_environment_free(environment);
}

TEST(VMBytecodeModuleBufferTest, AllocatesViewsAndTransfersEveryLaneFormat) {
  BufferExecutionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());

  iree_vm_variant_t allocate_arguments[] = {iree_vm_variant_from_i64(256)};
  iree_vm_variant_t allocate_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("allocate"), iree_vm_variant_span_from_array(allocate_arguments),
      iree_vm_variant_span_from_array(allocate_results)));
  iree_vm_buffer_t* root = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(
      &harness.vm_types, &allocate_results[0], &root));
  ASSERT_NE(root, nullptr);
  ASSERT_EQ(iree_vm_buffer_length(root), 256u);
  const auto* root_data =
      static_cast<const uint8_t*>(iree_vm_buffer_const_data(root));
  ASSERT_NE(root_data, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(root_data) & 63u, 0u);
  for (iree_host_size_t i = 0; i < 256; ++i) EXPECT_EQ(root_data[i], 0u);

  iree_vm_variant_t length_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, root)};
  iree_vm_variant_t length_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("length"), iree_vm_variant_span_from_array(length_arguments),
      iree_vm_variant_span_from_array(length_results)));
  int64_t length = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(length_results[0], &length));
  EXPECT_EQ(length, 256);

  iree_vm_variant_t null_length_arguments[] = {iree_vm_variant_null()};
  iree_vm_variant_t null_length_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("length"), iree_vm_variant_span_from_array(null_length_arguments),
      iree_vm_variant_span_from_array(null_length_results)));
  int64_t null_length = -1;
  IREE_ASSERT_OK(
      iree_vm_i64_from_variant(null_length_results[0], &null_length));
  EXPECT_EQ(null_length, 0);

  iree_vm_variant_t subspan_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, root),
      iree_vm_variant_from_i64(33),
      iree_vm_variant_from_i64(128),
  };
  iree_vm_variant_t subspan_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("subspan"), iree_vm_variant_span_from_array(subspan_arguments),
      iree_vm_variant_span_from_array(subspan_results)));
  iree_vm_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(
      &harness.vm_types, &subspan_results[0], &view));
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(iree_vm_buffer_length(view), 128u);
  EXPECT_EQ(iree_vm_buffer_const_data(view), root_data + 33);
  iree_vm_buffer_release(root);

  iree_vm_variant_t whole_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(128),
  };
  iree_vm_variant_t whole_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("subspan"), iree_vm_variant_span_from_array(whole_arguments),
      iree_vm_variant_span_from_array(whole_results)));
  iree_vm_buffer_t* whole = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(
      &harness.vm_types, &whole_results[0], &whole));
  EXPECT_EQ(whole, view);

  iree_vm_variant_t nested_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(16),
      iree_vm_variant_from_i64(64),
  };
  iree_vm_variant_t nested_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("subspan"), iree_vm_variant_span_from_array(nested_arguments),
      iree_vm_variant_span_from_array(nested_results)));
  iree_vm_buffer_t* nested = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(
      &harness.vm_types, &nested_results[0], &nested));
  EXPECT_EQ(iree_vm_buffer_const_data(nested), root_data + 49);

  iree_vm_variant_t empty_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, nested),
      iree_vm_variant_from_i64(4),
      iree_vm_variant_from_i64(0),
  };
  iree_vm_variant_t empty_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("subspan"), iree_vm_variant_span_from_array(empty_arguments),
      iree_vm_variant_span_from_array(empty_results)));
  iree_vm_buffer_t* empty = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_move(
      &harness.vm_types, &empty_results[0], &empty));
  EXPECT_EQ(iree_vm_buffer_length(empty), 0u);

  for (uint8_t format = IREE_VM_ISA_MEMORY_FORMAT_I8_X1;
       format <= IREE_VM_ISA_MEMORY_FORMAT_I64_X8; ++format) {
    SCOPED_TRACE(static_cast<int>(format));
    const uint8_t lane_count = (uint8_t)(1u << (format & 3u));
    const uint8_t lane_bit_count = (uint8_t)(8u << (format >> 2));
    const uint64_t lane_mask =
        lane_bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << lane_bit_count) - 1;
    std::vector<iree_vm_variant_t> arguments;
    arguments.push_back(
        iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view));
    std::vector<uint64_t> expected;
    for (uint8_t i = 0; i < lane_count; ++i) {
      const uint64_t bits =
          UINT64_C(0xFEDCBA9876543210) + UINT64_C(0x0101010101010101) * i;
      arguments.push_back(VariantFromI64Bits(bits));
      expected.push_back(bits & lane_mask);
    }
    std::vector<iree_vm_variant_t> results(lane_count);
    char function_name[32] = {};
    const int function_name_length =
        std::snprintf(function_name, sizeof(function_name), "roundtrip.%02u",
                      static_cast<unsigned>(format));
    ASSERT_GT(function_name_length, 0);
    IREE_ASSERT_OK(harness.Invoke(
        iree_make_string_view(function_name, function_name_length),
        iree_vm_variant_span_from_ptr(arguments.data(), arguments.size()),
        iree_vm_variant_span_from_ptr(results.data(), results.size())));
    for (uint8_t i = 0; i < lane_count; ++i) {
      uint64_t actual = 0;
      IREE_ASSERT_OK(iree_vm_scalar_bits_from_variant(
          results[i], IREE_VM_SCALAR_TYPE_I64, &actual));
      EXPECT_EQ(actual, expected[i]);
    }
    iree_vm_variant_span_reset(
        iree_vm_variant_span_from_ptr(results.data(), results.size()));
  }

  iree_vm_variant_t store_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(8),
      iree_vm_variant_from_i64(2),
      VariantFromI64Bits(UINT64_C(0x0123456789ABCDEF)),
  };
  IREE_ASSERT_OK(harness.Invoke(
      IREE_SV("store"), iree_vm_variant_span_from_array(store_arguments),
      iree_vm_variant_span_empty()));
  iree_vm_variant_t load_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(8),
      iree_vm_variant_from_i64(2),
  };
  iree_vm_variant_t load_results[1] = {};
  IREE_ASSERT_OK(harness.Invoke(IREE_SV("load"),
                                iree_vm_variant_span_from_array(load_arguments),
                                iree_vm_variant_span_from_array(load_results)));
  uint64_t loaded_bits = 0;
  IREE_ASSERT_OK(iree_vm_scalar_bits_from_variant(
      load_results[0], IREE_VM_SCALAR_TYPE_I64, &loaded_bits));
  EXPECT_EQ(loaded_bits, UINT64_C(0x0123456789ABCDEF));

  iree_vm_buffer_release(empty);
  iree_vm_buffer_release(nested);
  iree_vm_buffer_release(whole);
  iree_vm_buffer_release(view);
}

TEST(VMBytecodeModuleBufferTest, AtomicRmwFailsBeforePublishingOrMutating) {
  if (!iree_vm_bytecode_atomic_carrier_is_supported(
          IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I64)) {
    GTEST_SKIP() << "i64 atomics are unsupported by this target";
  }

  BufferExecutionHarness harness;
  RewriteI64RoundtripAsAtomicAdd(&harness);
  IREE_ASSERT_OK(harness.Initialize());
  char function_name[32] = {};
  const int function_name_length =
      std::snprintf(function_name, sizeof(function_name), "roundtrip.%02u",
                    static_cast<unsigned>(IREE_VM_ISA_MEMORY_FORMAT_I64_X1));
  ASSERT_GT(function_name_length, 0);
  const iree_string_view_t function_name_view =
      iree_make_string_view(function_name, function_name_length);
  const iree_vm_variant_t sentinel = VariantFromI64Bits(UINT64_C(0x1234));

  const auto invoke = [&](iree_vm_buffer_t* buffer, iree_vm_variant_t* result) {
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, buffer),
        iree_vm_variant_from_i64(5),
    };
    return harness.Invoke(function_name_view,
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_ptr(result, 1));
  };

  alignas(8) std::array<uint8_t, 8> aligned_storage = {};
  const uint64_t initial_bits = 7;
  std::memcpy(aligned_storage.data(), &initial_bits, sizeof(initial_bits));
  iree_vm_buffer_t* aligned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(aligned_storage.data(), aligned_storage.size()),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &aligned_buffer));
  iree_vm_variant_t result = sentinel;
  IREE_ASSERT_OK(invoke(aligned_buffer, &result));
  int64_t old_value = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(result, &old_value));
  EXPECT_EQ(old_value, 7);
  uint64_t updated_bits = 0;
  std::memcpy(&updated_bits, aligned_storage.data(), sizeof(updated_bits));
  EXPECT_EQ(updated_bits, 12u);
  iree_vm_buffer_release(aligned_buffer);

  alignas(8) std::array<uint8_t, 8> read_only_storage = {};
  std::memcpy(read_only_storage.data(), &initial_bits, sizeof(initial_bits));
  iree_vm_buffer_t* read_only_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span(read_only_storage.data(), read_only_storage.size()),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &read_only_buffer));
  result = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        invoke(read_only_buffer, &result));
  ExpectVariantEqual(result, sentinel);
  EXPECT_EQ(std::memcmp(read_only_storage.data(), &initial_bits,
                        sizeof(initial_bits)),
            0);
  iree_vm_buffer_release(read_only_buffer);

  alignas(8) std::array<uint8_t, 4> short_storage = {};
  iree_vm_buffer_t* short_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(short_storage.data(), short_storage.size()),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &short_buffer));
  result = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        invoke(short_buffer, &result));
  ExpectVariantEqual(result, sentinel);
  EXPECT_EQ(short_storage, (std::array<uint8_t, 4>{}));
  iree_vm_buffer_release(short_buffer);

  alignas(8) std::array<uint8_t, 9> misaligned_storage = {};
  std::memcpy(misaligned_storage.data() + 1, &initial_bits,
              sizeof(initial_bits));
  iree_vm_buffer_t* misaligned_buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_byte_span(misaligned_storage.data() + 1, sizeof(initial_bits)),
      iree_vm_buffer_release_callback_null(), iree_allocator_system(),
      &misaligned_buffer));
  result = sentinel;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        invoke(misaligned_buffer, &result));
  ExpectVariantEqual(result, sentinel);
  EXPECT_EQ(std::memcmp(misaligned_storage.data() + 1, &initial_bits,
                        sizeof(initial_bits)),
            0);
  iree_vm_buffer_release(misaligned_buffer);
}

TEST(VMBytecodeModuleBufferTest, CopiesBetweenBufferAndLocalStorage) {
  BufferExecutionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());

  const std::array<uint8_t, 16> initial_bytes = {0, 1, 2,  3,  4,  5,  6,  7,
                                                 8, 9, 10, 11, 12, 13, 14, 15};
  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(initial_bytes.data(), initial_bytes.size()), 0,
      iree_allocator_system(), &buffer));

  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, buffer),
      iree_vm_variant_from_i64(1),
      iree_vm_variant_from_i64(8),
  };
  IREE_ASSERT_OK(harness.Invoke(IREE_SV("stack_copy"),
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_empty()));
  const auto* data =
      static_cast<const uint8_t*>(iree_vm_buffer_const_data(buffer));
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(std::memcmp(data + 8, initial_bytes.data() + 1, 4), 0);

  iree_vm_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      buffer, 2, 10,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_allocator_system(), &view));
  iree_vm_variant_t view_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(1),
      iree_vm_variant_from_i64(6),
  };
  IREE_ASSERT_OK(harness.Invoke(IREE_SV("stack_copy"),
                                iree_vm_variant_span_from_array(view_arguments),
                                iree_vm_variant_span_empty()));
  EXPECT_EQ(std::memcmp(data + 8, initial_bytes.data() + 3, 4), 0);

  iree_vm_buffer_release(view);
  iree_vm_buffer_release(buffer);
}

TEST(VMBytecodeModuleBufferTest, FailsBeforePublishingOrMutating) {
  BufferExecutionHarness harness;
  IREE_ASSERT_OK(harness.Initialize());

  const iree_vm_variant_t sentinel = VariantFromI64Bits(UINT64_C(0x1234));
  iree_vm_variant_t excessive_arguments[] = {VariantFromI64Bits(UINT64_MAX)};
  iree_vm_variant_t excessive_results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      harness.Invoke(IREE_SV("allocate"),
                     iree_vm_variant_span_from_array(excessive_arguments),
                     iree_vm_variant_span_from_array(excessive_results)));
  ExpectVariantEqual(excessive_results[0], sentinel);

  BufferTestObject wrong_object = {};
  iree_vm_ref_object_initialize(&wrong_object.ref_object);
  iree_vm_variant_t wrong_arguments[] = {
      iree_vm_variant_from_ptr_borrowed(&wrong_object, &kBufferTestObjectType)};
  iree_vm_variant_t wrong_results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      harness.Invoke(IREE_SV("wrong_length"),
                     iree_vm_variant_span_from_array(wrong_arguments),
                     iree_vm_variant_span_from_array(wrong_results)));
  ExpectVariantEqual(wrong_results[0], sentinel);

  const std::array<uint8_t, 16> initial_bytes = {0, 1, 2,  3,  4,  5,  6,  7,
                                                 8, 9, 10, 11, 12, 13, 14, 15};
  iree_vm_buffer_t* read_only = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_const_byte_span(initial_bytes.data(), initial_bytes.size()), 0,
      iree_allocator_system(), &read_only));
  iree_vm_variant_t read_only_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, read_only),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(0),
      VariantFromI64Bits(UINT64_MAX),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      harness.Invoke(IREE_SV("store"),
                     iree_vm_variant_span_from_array(read_only_arguments),
                     iree_vm_variant_span_empty()));
  EXPECT_EQ(std::memcmp(iree_vm_buffer_const_data(read_only),
                        initial_bytes.data(), initial_bytes.size()),
            0);
  iree_vm_buffer_release(read_only);

  iree_vm_buffer_t* write_only = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(initial_bytes.data(), initial_bytes.size()), 0,
      iree_allocator_system(), &write_only));
  iree_vm_variant_t write_only_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, write_only),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(0),
  };
  iree_vm_variant_t write_only_results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      harness.Invoke(IREE_SV("load"),
                     iree_vm_variant_span_from_array(write_only_arguments),
                     iree_vm_variant_span_from_array(write_only_results)));
  ExpectVariantEqual(write_only_results[0], sentinel);
  iree_vm_buffer_release(write_only);

  iree_vm_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_make_const_byte_span(initial_bytes.data(), initial_bytes.size()), 0,
      iree_allocator_system(), &buffer));
  const auto expect_store_out_of_range = [&](uint64_t base, uint64_t index) {
    iree_vm_variant_t arguments[] = {
        iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, buffer),
        VariantFromI64Bits(base),
        VariantFromI64Bits(index),
        VariantFromI64Bits(UINT64_MAX),
    };
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        harness.Invoke(IREE_SV("store"),
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_empty()));
    EXPECT_EQ(std::memcmp(iree_vm_buffer_const_data(buffer),
                          initial_bytes.data(), initial_bytes.size()),
              0);
  };
  expect_store_out_of_range(UINT64_MAX, 1);
  expect_store_out_of_range(12, 0);

  iree_vm_buffer_t* view = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_subspan(
      buffer, 0, 8,
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE,
      iree_allocator_system(), &view));
  buffer->flags &= ~IREE_VM_BUFFER_ACCESS_MASK;
  buffer->data = nullptr;

  iree_vm_variant_t closed_length_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, buffer)};
  iree_vm_variant_t closed_length_results[1] = {};
  IREE_ASSERT_OK(
      harness.Invoke(IREE_SV("length"),
                     iree_vm_variant_span_from_array(closed_length_arguments),
                     iree_vm_variant_span_from_array(closed_length_results)));
  int64_t closed_length = 0;
  IREE_ASSERT_OK(
      iree_vm_i64_from_variant(closed_length_results[0], &closed_length));
  EXPECT_EQ(closed_length, 16);

  iree_vm_variant_t closed_load_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, view),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(0),
  };
  iree_vm_variant_t closed_load_results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      harness.Invoke(IREE_SV("load"),
                     iree_vm_variant_span_from_array(closed_load_arguments),
                     iree_vm_variant_span_from_array(closed_load_results)));
  ExpectVariantEqual(closed_load_results[0], sentinel);

  iree_vm_variant_t closed_subspan_arguments[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&harness.vm_types, buffer),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(0),
  };
  iree_vm_variant_t closed_subspan_results[] = {sentinel};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      harness.Invoke(IREE_SV("subspan"),
                     iree_vm_variant_span_from_array(closed_subspan_arguments),
                     iree_vm_variant_span_from_array(closed_subspan_results)));
  ExpectVariantEqual(closed_subspan_results[0], sentinel);

  iree_vm_buffer_release(view);
  iree_vm_buffer_release(buffer);
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
