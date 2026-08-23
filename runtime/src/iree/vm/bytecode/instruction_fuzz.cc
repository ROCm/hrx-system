// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured fuzzing for executable VM instruction records.
//
// Each input selects one executable Core opcode and supplies its record fields.
// The record is framed by canonical block/return records inside an otherwise
// valid module. This keeps malformed fields deep enough to exercise
// disassembly, inspection, executable verification, linking, and invocation
// instead of being rejected by the image envelope.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/vm/bytecode/inspection.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/tooling/dump.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/process.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;
constexpr iree_host_size_t kMaximumRecordLength = 36;

#define IREE_VM_BYTECODE_EXECUTION_INFO_ROW(record_length, form) record_length,
constexpr uint8_t kRecordLengths[256] = {
#define IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS
#include "iree/vm/bytecode/execution_tables.inl"
#undef IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS
};
#undef IREE_VM_BYTECODE_EXECUTION_INFO_ROW

constexpr bool RecordLengthsFitFixture() {
  for (uint8_t record_length : kRecordLengths) {
    if (record_length > kMaximumRecordLength || record_length % 4 != 0) {
      return false;
    }
  }
  return true;
}

static_assert(IREE_ARRAYSIZE(kRecordLengths) == 256,
              "execution table must cover every Core opcode");
static_assert(RecordLengthsFitFixture(),
              "instruction fuzz fixture must fit every executable record");

enum class FixtureKind {
  kScalarState,
  kOwnership,
  kValueOverflow,
};

void RequireOk(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

// Returns true for success and false for an explicitly modeled failure caused
// by malformed fuzz input. Any other failure is an infrastructure defect and
// remains visible to the fuzzing engine.
bool IsSuccessfulOrExpectedFailure(
    iree_status_t status,
    std::initializer_list<iree_status_code_t> expected_codes) {
  iree::Status owned_status(std::move(status));
  if (owned_status.ok()) return true;
  const iree_status_code_t actual_code =
      static_cast<iree_status_code_t>(owned_status.code());
  for (const iree_status_code_t expected_code : expected_codes) {
    if (actual_code == expected_code) return false;
  }
  iree_status_abort(owned_status.release());
}

bool IsFuzzableOpcode(uint8_t opcode) {
  return kRecordLengths[opcode] != 0 &&
         opcode != IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK &&
         opcode != IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN;
}

uint8_t SelectOpcode(uint8_t selector) {
  for (uint16_t offset = 0; offset < 256; ++offset) {
    const uint8_t opcode = static_cast<uint8_t>(selector + offset);
    if (IsFuzzableOpcode(opcode)) return opcode;
  }
  std::abort();
}

FixtureKind SelectFixture(uint8_t opcode) {
  switch (opcode) {
    case IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_ARGUMENT_LOAD:
    case IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_RESULT_STORE:
      return FixtureKind::kValueOverflow;
    case IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_LOAD:
    case IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_STORE:
    case IREE_VM_ISA_CORE_OPCODE_BUFFER_RODATA_LOAD:
      return FixtureKind::kOwnership;
    default:
      return FixtureKind::kScalarState;
  }
}

iree_string_view_t FixtureExportName(FixtureKind fixture_kind) {
  return fixture_kind == FixtureKind::kValueOverflow ? IREE_SV("identity")
                                                     : IREE_SV("run");
}

template <typename T>
void AppendRecord(const T& record, uint8_t** cursor) {
  std::memcpy(*cursor, &record, sizeof(record));
  *cursor += sizeof(record);
}

void FillScalarFunction(const uint8_t* record, uint8_t record_length,
                        std::vector<uint8_t>* image) {
  using iree::vm::bytecode::testing::FindFunctionImage;
  using iree::vm::bytecode::testing::MutableFunctionImage;
  const MutableFunctionImage function = FindFunctionImage(image, 1);
  if (function.row == nullptr || function.row->bytecode_length_u32 != 68 ||
      function.row->value_register_count_u16 != 6) {
    std::abort();
  }
  function.row->local_byte_length_u16 = 512;

  uint8_t* cursor = function.bytecode;
  const uint8_t* const end =
      function.bytecode + function.row->bytecode_length_u32;
  AppendRecord(
      iree_vm_isa_control_block_record_t{IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK,
                                         {0, 0, 0}},
      &cursor);
  for (uint8_t ordinal = 0; ordinal < function.row->value_register_count_u16;
       ++ordinal) {
    AppendRecord(
        iree_vm_isa_constant_zero_record_t{
            IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, ordinal, 0},
        &cursor);
  }
  std::memcpy(cursor, record, record_length);
  cursor += record_length;
  while (cursor < end - sizeof(iree_vm_isa_control_return_record_t)) {
    AppendRecord(
        iree_vm_isa_constant_zero_record_t{
            IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0},
        &cursor);
  }
  AppendRecord(
      iree_vm_isa_control_return_record_t{
          IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}},
      &cursor);
  if (cursor != end) std::abort();
}

void FillOwnershipFunction(const uint8_t* record, uint8_t record_length,
                           std::vector<uint8_t>* image) {
  using iree::vm::bytecode::testing::FindFunctionImage;
  using iree::vm::bytecode::testing::MutableFunctionImage;
  const MutableFunctionImage function = FindFunctionImage(image, 1);
  if (function.row == nullptr || function.row->bytecode_length_u32 != 20 ||
      record_length != 4) {
    std::abort();
  }

  uint8_t* cursor = function.bytecode;
  AppendRecord(
      iree_vm_isa_control_block_record_t{IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK,
                                         {0, 0, 0}},
      &cursor);
  AppendRecord(
      iree_vm_isa_global_value_immutable_load_record_t{
          IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_LOAD, 1, 0},
      &cursor);
  std::memcpy(cursor, record, record_length);
  cursor += record_length;
  AppendRecord(
      iree_vm_isa_constant_zero_record_t{IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO,
                                         0, 0},
      &cursor);
  AppendRecord(
      iree_vm_isa_control_return_record_t{
          IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}},
      &cursor);
  if (cursor != function.bytecode + function.row->bytecode_length_u32) {
    std::abort();
  }
}

void FillValueOverflowFunction(const uint8_t* record, uint8_t record_length,
                               std::vector<uint8_t>* image) {
  using iree::vm::bytecode::testing::FindFunctionImage;
  using iree::vm::bytecode::testing::MutableFunctionImage;
  const MutableFunctionImage function = FindFunctionImage(image, 0);
  if (function.row == nullptr || function.row->bytecode_length_u32 != 24 ||
      function.row->value_register_count_u16 != 18 || record_length != 4) {
    std::abort();
  }
  std::memcpy(function.bytecode + sizeof(iree_vm_isa_control_block_record_t),
              record, record_length);
}

std::vector<uint8_t> BuildRecordImage(const uint8_t* data, size_t size,
                                      FixtureKind* out_fixture_kind,
                                      uint8_t* out_opcode) {
  const uint8_t opcode = SelectOpcode(size > 0 ? data[0] : 0);
  const uint8_t record_length = kRecordLengths[opcode];
  std::array<uint8_t, kMaximumRecordLength> record = {};
  record[0] = opcode;
  for (iree_host_size_t i = 1; i < record_length && i < size; ++i) {
    record[i] = data[i];
  }

  const FixtureKind fixture_kind = SelectFixture(opcode);
  std::vector<uint8_t> image;
  switch (fixture_kind) {
    case FixtureKind::kScalarState:
      image = iree::vm::bytecode::testing::BuildScalarStateModuleImage();
      FillScalarFunction(record.data(), record_length, &image);
      break;
    case FixtureKind::kOwnership:
      image = iree::vm::bytecode::testing::BuildOwnershipModuleImage();
      FillOwnershipFunction(record.data(), record_length, &image);
      break;
    case FixtureKind::kValueOverflow:
      image = iree::vm::bytecode::testing::BuildValueOverflowModuleImage();
      FillValueOverflowFunction(record.data(), record_length, &image);
      break;
  }
  *out_fixture_kind = fixture_kind;
  *out_opcode = opcode;
  return image;
}

struct CountingDumpSink {
  // Total bytes accepted from the dumper.
  iree_host_size_t byte_count;
};

iree_status_t CountDumpBytes(void* user_data, iree_string_view_t text) {
  auto* sink = static_cast<CountingDumpSink*>(user_data);
  if (text.size > IREE_HOST_SIZE_MAX - sink->byte_count) std::abort();
  sink->byte_count += text.size;
  return iree_ok_status();
}

void ExerciseDescription(iree_vm_export_t export_value,
                         iree_string_view_t expected_name) {
  iree_host_size_t required_size = 0;
  RequireOk(iree_vm_export_query_description(
      export_value, iree_byte_span_empty(), &required_size, nullptr));
  alignas(max_align_t) std::array<uint8_t, 4096> storage = {};
  if (required_size > storage.size()) std::abort();
  iree_vm_export_description_t description = {};
  RequireOk(iree_vm_export_query_description(
      export_value, iree_make_byte_span(storage.data(), required_size),
      &required_size, &description));
  if (!iree_string_view_equal(description.name, expected_name)) {
    std::abort();
  }
}

void ExerciseInspection(const std::vector<uint8_t>& image,
                        FixtureKind fixture_kind) {
  const iree_const_byte_span_t image_span =
      iree_make_const_byte_span(image.data(), image.size());
  CountingDumpSink sink = {};
  iree_status_t dump_status = iree_vm_bytecode_module_dump(
      IREE_SV("instruction_fuzz"), image_span, {CountDumpBytes, &sink},
      iree_allocator_system());
  if (IsSuccessfulOrExpectedFailure(dump_status,
                                    {IREE_STATUS_INVALID_ARGUMENT}) &&
      sink.byte_count == 0) {
    std::abort();
  }

  iree_vm_module_t* module = nullptr;
  RequireOk(iree_vm_bytecode_module_create_for_inspection(
      IREE_SV("instruction_fuzz"), {image_span, iree_allocator_null()},
      iree_allocator_system(), &module));
  if (iree_vm_module_export_count(module) != 2) std::abort();
  iree_vm_export_t run_export = {};
  const iree_string_view_t export_name = FixtureExportName(fixture_kind);
  RequireOk(iree_vm_module_lookup_export(module, export_name, &run_export));
  ExerciseDescription(run_export, export_name);
  iree_vm_module_release(module);
}

template <size_t ArgumentCount, size_t ResultCount>
void InvokeRecord(iree_vm_invocation_t* invocation, iree_vm_function_t function,
                  std::array<iree_vm_variant_t, ArgumentCount>* arguments,
                  std::array<iree_vm_variant_t, ResultCount>* results) {
  const std::array<iree_vm_variant_t, ResultCount> untouched_results = *results;
  iree_status_t status = iree_vm_invoke(
      invocation, function,
      iree_vm_variant_span_from_ptr(arguments->data(), arguments->size()),
      iree_vm_variant_span_from_ptr(results->data(), results->size()));
  for (const iree_vm_variant_t argument : *arguments) {
    if (!iree_vm_variant_is_empty(argument)) std::abort();
  }
  if (IsSuccessfulOrExpectedFailure(
          status, {IREE_STATUS_INVALID_ARGUMENT, IREE_STATUS_OUT_OF_RANGE,
                   IREE_STATUS_FAILED_PRECONDITION})) {
    iree_vm_variant_span_reset(
        iree_vm_variant_span_from_ptr(results->data(), results->size()));
  } else if (std::memcmp(results->data(), untouched_results.data(),
                         sizeof(*results)) != 0) {
    std::abort();
  }
}

bool AllowsUnimplementedSelector(uint8_t opcode) {
  return opcode == IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER ||
         opcode == IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_EXTEND ||
         opcode == IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER;
}

void ExerciseExecutable(const std::vector<uint8_t>& image,
                        FixtureKind fixture_kind, uint8_t opcode) {
  iree_vm_environment_t* environment = nullptr;
  RequireOk(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* module = nullptr;
  iree_status_t module_status = iree_vm_bytecode_module_create(
      environment, IREE_SV("instruction_fuzz"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module);
  iree_vm_environment_free(environment);
  const bool module_created =
      AllowsUnimplementedSelector(opcode)
          ? IsSuccessfulOrExpectedFailure(
                module_status,
                {IREE_STATUS_INVALID_ARGUMENT, IREE_STATUS_UNIMPLEMENTED})
          : IsSuccessfulOrExpectedFailure(module_status,
                                          {IREE_STATUS_INVALID_ARGUMENT});
  if (!module_created) {
    if (module != nullptr) std::abort();
    return;
  }
  if (module == nullptr) std::abort();

  iree_vm_export_t run_export = {};
  const iree_string_view_t export_name = FixtureExportName(fixture_kind);
  RequireOk(iree_vm_module_lookup_export(module, export_name, &run_export));
  ExerciseDescription(run_export, export_name);

  iree_vm_program_t* program = nullptr;
  RequireOk(iree_vm_program_create({module, iree_vm_module_span_empty()},
                                   iree_allocator_system(), &program));
  alignas(max_align_t) std::array<uint8_t, kInvocationStorageSize>
      invocation_storage = {};
  iree_vm_invocation_t* invocation = nullptr;
  RequireOk(iree_vm_invocation_initialize(
      iree_make_byte_span(invocation_storage.data(), invocation_storage.size()),
      &invocation));
  iree_vm_process_t* process = nullptr;
  RequireOk(iree_vm_process_create(program, invocation,
                                   iree_vm_variant_span_empty(),
                                   iree_allocator_system(), &process));
  iree_vm_function_t run = iree_vm_function_null();
  RequireOk(iree_vm_function_from_export(process, run_export, &run));

  switch (fixture_kind) {
    case FixtureKind::kScalarState: {
      std::array<iree_vm_variant_t, 0> arguments = {};
      std::array<iree_vm_variant_t, 5> results = {};
      InvokeRecord(invocation, run, &arguments, &results);
      break;
    }
    case FixtureKind::kOwnership: {
      std::array<iree_vm_variant_t, 1> arguments = {
          iree_vm_variant_from_i32(35)};
      std::array<iree_vm_variant_t, 2> results = {};
      InvokeRecord(invocation, run, &arguments, &results);
      break;
    }
    case FixtureKind::kValueOverflow: {
      std::array<iree_vm_variant_t, 18> arguments = {};
      for (iree_host_size_t i = 0; i < arguments.size(); ++i) {
        arguments[i] = iree_vm_variant_from_i64(static_cast<int64_t>(i));
      }
      std::array<iree_vm_variant_t, 18> results = {};
      InvokeRecord(invocation, run, &arguments, &results);
      break;
    }
  }

  iree_vm_invocation_deinitialize(invocation);
  iree_vm_process_release(process);
  iree_vm_program_release(program);
  iree_vm_module_release(module);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FixtureKind fixture_kind = FixtureKind::kScalarState;
  uint8_t opcode = 0;
  const std::vector<uint8_t> image =
      BuildRecordImage(data, size, &fixture_kind, &opcode);
  ExerciseInspection(image, fixture_kind);
  ExerciseExecutable(image, fixture_kind, opcode);
  return 0;
}
