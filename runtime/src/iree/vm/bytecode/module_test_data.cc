// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/module_test_data.h"

#include <cstring>
#include <initializer_list>
#include <string_view>
#include <utility>

#include "iree/vm/bytecode/wire/core/abi.h"
#include "iree/vm/bytecode/wire/core/buffer.h"
#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/conversion.h"
#include "iree/vm/bytecode/wire/core/function.h"
#include "iree/vm/bytecode/wire/core/global.h"
#include "iree/vm/bytecode/wire/core/integer.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/ref.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/core/value.h"
#include "iree/vm/bytecode/wire/hal/device.h"
#include "iree/vm/bytecode/wire/hal/opcodes.h"
#include "iree/vm/bytecode/wire/module_format.h"

namespace iree::vm::bytecode::testing {
namespace {

class ByteBuffer {
 public:
  template <typename T>
  void Append(const T& value) {
    const size_t offset = data_.size();
    data_.resize(offset + sizeof(value));
    std::memcpy(data_.data() + offset, &value, sizeof(value));
  }

  void AppendBytes(const void* data, size_t length) {
    if (length == 0) return;
    const size_t offset = data_.size();
    data_.resize(offset + length);
    std::memcpy(data_.data() + offset, data, length);
  }

  void Align(size_t alignment) {
    data_.resize((data_.size() + alignment - 1) & ~(alignment - 1), 0);
  }

  std::vector<uint8_t> Take() { return std::move(data_); }

 private:
  std::vector<uint8_t> data_;
};

struct Section {
  // Architectural section identifier.
  uint16_t type;
  // Exact architectural section flags.
  uint16_t flags;
  // Complete section payload.
  std::vector<uint8_t> payload;
};

std::vector<uint8_t> BuildImage(std::initializer_list<Section> sections) {
  ByteBuffer image;
  iree_vm_bytecode_v0_image_header_t header = {};
  std::memcpy(header.magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
              IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH);
  header.core_major_u16 = IREE_VM_BYTECODE_CORE_MAJOR;
  header.core_required_minor_u16 = IREE_VM_BYTECODE_CORE_MINOR;
  header.section_count_u16 = static_cast<uint16_t>(sections.size());
  image.Append(header);
  for (const Section& section : sections) {
    const iree_vm_bytecode_v0_section_directory_row_t row = {
        section.type,
        section.flags,
        0,
        static_cast<uint64_t>(section.payload.size()),
    };
    image.Append(row);
  }
  for (const Section& section : sections) {
    image.Align(IREE_VM_BYTECODE_SECTION_ALIGNMENT);
    image.AppendBytes(section.payload.data(), section.payload.size());
  }
  return image.Take();
}

std::vector<uint8_t> BuildStrings(
    std::initializer_list<std::string_view> strings) {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_strings_header_t{
      static_cast<uint32_t>(strings.size()),
  });
  uint32_t offset = 0;
  section.Append(offset);
  for (std::string_view value : strings) {
    offset += static_cast<uint32_t>(value.size());
    section.Append(offset);
  }
  for (std::string_view value : strings) {
    section.AppendBytes(value.data(), value.size());
  }
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipRefTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_ref_types_header_t{1});
  section.Append(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 1});
  section.Append(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefStateRefTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_ref_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 1});
  section.Append(iree_vm_bytecode_v0_ref_type_group_row_t{6, 0, 1});
  section.Append(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0});
  section.Append(iree_vm_bytecode_v0_ref_type_entry_row_t{7, 0});
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      1,
      1,
      0,
      1,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{2});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{3, 1, 1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_constant_s16_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16, 0, 7});
  bytecode.Append(iree_vm_isa_global_value_immutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_STORE, 0, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_value_immutable_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_IMMUTABLE_LOAD, 1, 0});
  bytecode.Append(iree_vm_isa_integer_add_i32_record_t{
      IREE_VM_ISA_CORE_OPCODE_INTEGER_ADD_I32, 0, 0, 1});
  bytecode.Append(iree_vm_isa_buffer_rodata_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_BUFFER_RODATA_LOAD, 0, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{2});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 16, 20, 0, 0, 0, 2, 1, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipGlobals() {
  ByteBuffer section;
  section.Append(
      iree_vm_bytecode_v0_globals_header_t{1, 1, 0, 0, 0, 0, {0, 0}});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{});
  section.Append(iree_vm_bytecode_v0_signature_row_t{0, 0, 1, 0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I8, 0});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionImports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_imports_header_t{1});
  section.Append(iree_vm_bytecode_v0_import_group_row_t{0, 0, 1});
  section.Append(iree_vm_bytecode_v0_import_entry_row_t{
      1, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{1});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(
      iree_vm_isa_func_null_record_t{IREE_VM_ISA_CORE_OPCODE_FUNC_NULL, 0, 0});
  bytecode.Append(iree_vm_isa_func_compare_null_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_COMPARE_NULL, 0, 0, 0});
  bytecode.Append(iree_vm_isa_func_copy_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_COPY, 1, 0, 0});
  bytecode.Append(iree_vm_isa_func_address_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ADDRESS, 2,
      IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 0, 1, 0});
  bytecode.Append(iree_vm_isa_func_address_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ADDRESS, 3,
      IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT, 0, 0, 0});
  bytecode.Append(iree_vm_isa_func_import_resolved_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_IMPORT_RESOLVED, 1, 0});
  bytecode.Append(iree_vm_isa_func_stack_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_STORE, 2, 0});
  bytecode.Append(iree_vm_isa_func_stack_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_STACK_LOAD, 4, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{2});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 48, 0, 0, 0, 2, 0, 5, 0, 1, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 48, 8, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildFunctionStateSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{7});
  section.Append(iree_vm_bytecode_v0_signature_row_t{0, 1, 1, 0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_signature_row_t{2, 0, 0, 0, 0, 2, 0});
  section.Append(iree_vm_bytecode_v0_signature_row_t{4, 0, 0, 0, 0, 0, 2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{6, 0, 0, 0, 0, 1, 0});
  section.Append(iree_vm_bytecode_v0_signature_row_t{7, 0, 0, 0, 0, 18, 18});
  section.Append(iree_vm_bytecode_v0_signature_row_t{43, 0, 0, 0, 0, 0, 1});
  section.Append(iree_vm_bytecode_v0_signature_row_t{44, 0, 0, 0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  for (int i = 0; i < 42; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_FUNCTION, 0});
  }
  return section.Take();
}

std::vector<uint8_t> BuildFunctionStateCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{7});
  for (uint16_t i = 0; i < 7; ++i) {
    section.Append(iree_vm_bytecode_v0_callable_type_row_t{i, 0});
  }
  return section.Take();
}

std::vector<uint8_t> BuildFunctionStateExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{7});
  section.Append(iree_vm_bytecode_v0_export_row_t{0, 5, 5, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{1, 6, 6, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{3, 1, 1, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{4, 4, 4, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{5, 2, 2, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{6, 3, 3, 0});
  return section.Take();
}

std::vector<uint8_t> BuildFunctionStateFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_func_immutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_IMMUTABLE_STORE, 0, 0});
  bytecode.Append(iree_vm_isa_global_func_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_MUTABLE_STORE, 1, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_func_immutable_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_IMMUTABLE_LOAD, 0, 0});
  bytecode.Append(iree_vm_isa_global_func_mutable_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_MUTABLE_LOAD, 1, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_func_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_MUTABLE_STORE, 0, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_func_abi_argument_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ABI_ARGUMENT_LOAD, 16, 0});
  bytecode.Append(iree_vm_isa_func_abi_argument_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ABI_ARGUMENT_LOAD, 17, 1});
  bytecode.Append(iree_vm_isa_func_abi_result_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ABI_RESULT_STORE, 16, 0});
  bytecode.Append(iree_vm_isa_func_abi_result_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ABI_RESULT_STORE, 17, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_func_address_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ADDRESS, 0,
      IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 0, 5, 5});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_func_address_record_t{
      IREE_VM_ISA_CORE_OPCODE_FUNC_ADDRESS, 0,
      IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 0, 6, 6});
  bytecode.Append(iree_vm_isa_global_func_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_FUNC_MUTABLE_STORE, 0, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{7});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 8, 16, 0, 0, 0, 0, 0, 2, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      2, 0, 24, 16, 0, 0, 0, 0, 0, 2, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      3, 0, 40, 12, 0, 0, 0, 0, 0, 1, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      4, 0, 52, 24, 0, 0, 0, 0, 0, 18, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      5, 0, 76, 16, 0, 0, 0, 0, 0, 1, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      6, 0, 92, 20, 0, 0, 0, 0, 0, 1, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildFunctionStateGlobals() {
  ByteBuffer section;
  section.Append(
      iree_vm_bytecode_v0_globals_header_t{0, 0, 0, 0, 2, 1, {0, 0}});
  section.Append(iree_vm_bytecode_v0_global_function_descriptor_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_global_function_descriptor_row_t{0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      0,
      3,
      1,
      1,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      5,
      0,
      0,
      1,
      0,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  for (int i = 0; i < 3; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  }
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{2});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{3, 1, 1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(
      iree_vm_isa_ref_move_record_t{IREE_VM_ISA_CORE_OPCODE_REF_MOVE, 2, 0, 0});
  bytecode.Append(
      iree_vm_isa_ref_move_record_t{IREE_VM_ISA_CORE_OPCODE_REF_MOVE, 0, 2, 0});
  bytecode.Append(iree_vm_isa_ref_compare_null_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_COMPARE_NULL, 0, 0, 0});
  bytecode.Append(
      iree_vm_isa_ref_null_record_t{IREE_VM_ISA_CORE_OPCODE_REF_NULL, 1, 0});
  bytecode.Append(iree_vm_isa_ref_compare_eq_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_COMPARE_EQ, 1, 0, 1});
  bytecode.Append(iree_vm_isa_ref_stack_store_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_RETAIN, 0, 0});
  bytecode.Append(iree_vm_isa_ref_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_RETAIN, 1, 0, 0});
  bytecode.Append(iree_vm_isa_ref_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_RETAIN, 1, 1, 0});
  bytecode.Append(iree_vm_isa_ref_stack_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_MOVE, 1, 1});
  bytecode.Append(iree_vm_isa_ref_stack_load_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_RETAIN, 2, 1});
  bytecode.Append(
      iree_vm_isa_ref_move_record_t{IREE_VM_ISA_CORE_OPCODE_REF_MOVE, 1, 2, 0});
  bytecode.Append(iree_vm_isa_ref_stack_load_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_LOAD_MOVE, 2, 1});
  bytecode.Append(iree_vm_isa_ref_compare_eq_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_COMPARE_EQ, 2, 1, 2});
  bytecode.Append(iree_vm_isa_ref_stack_discard_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_DISCARD, 0, 0});
  bytecode.Append(iree_vm_isa_ref_discard_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_DISCARD, 1, 0});
  bytecode.Append(
      iree_vm_isa_ref_null_record_t{IREE_VM_ISA_CORE_OPCODE_REF_NULL, 2, 0});
  bytecode.Append(iree_vm_isa_ref_stack_store_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_RETAIN, 0, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_ref_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_RETAIN, 1, 0, 0});
  bytecode.Append(iree_vm_isa_ref_stack_store_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_STACK_STORE_RETAIN, 0, 0});
  bytecode.Append(iree_vm_isa_constant_i32_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_I32, 0, 0, UINT32_C(0x7FC00000)});
  bytecode.Append(iree_vm_isa_conversion_float_to_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER, 1, 0,
      IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{2});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 76, 0, 0, 0, 3, 3, 0, 2, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 76, 28, 0, 0, 0, 2, 2, 0, 1, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildRefStateSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{4});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      0,
      0,
      2,
      0,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      2,
      0,
      0,
      0,
      2,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      4,
      0,
      0,
      17,
      17,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      38,
      0,
      0,
      1,
      0,
      0,
      0,
  });
  for (int i = 0; i < 38; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  }
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 1});
  return section.Take();
}

std::vector<uint8_t> BuildRefStateCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{4});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{2, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{3, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefStateExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{4});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{3, 2, 2, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{4, 1, 1, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{5, 3, 3, 0});
  return section.Take();
}

std::vector<uint8_t> BuildRefStateFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_ref_immutable_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_REF_IMMUTABLE_STORE_MOVE, 0, 0});
  bytecode.Append(iree_vm_isa_global_ref_mutable_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_REF_MUTABLE_STORE_MOVE, 1, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_ref_immutable_load_borrow_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_REF_IMMUTABLE_LOAD_BORROW, 0, 0});
  bytecode.Append(iree_vm_isa_global_ref_mutable_load_retain_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_REF_MUTABLE_LOAD_RETAIN, 1, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_ref_abi_argument_load_borrow_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_ABI_ARGUMENT_LOAD_BORROW, 16, 0});
  bytecode.Append(iree_vm_isa_ref_abi_argument_load_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_ABI_ARGUMENT_LOAD_MOVE, 17, 0});
  bytecode.Append(iree_vm_isa_ref_abi_result_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_REF_ABI_RESULT_STORE_MOVE, 16, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_ref_mutable_store_move_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_REF_MUTABLE_STORE_MOVE, 0, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{4});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 16, 0, 0, 0, 0, 2, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 16, 16, 0, 0, 0, 0, 2, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      2, 0, 32, 20, 0, 0, 0, 0, 18, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      3, 0, 52, 12, 0, 0, 0, 0, 1, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildRefStateGlobals() {
  ByteBuffer section;
  section.Append(
      iree_vm_bytecode_v0_globals_header_t{0, 0, 2, 1, 0, 0, {0, 0}});
  section.Append(iree_vm_bytecode_v0_global_ref_descriptor_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_global_ref_descriptor_row_t{0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildTestRodata() {
  constexpr std::string_view kPayload = "loom-vm-v1";
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_rodata_header_t{1, 0});
  section.Append(
      static_cast<iree_vm_bytecode_v0_rodata_block_length_t>(kPayload.size()));
  section.AppendBytes(kPayload.data(), kPayload.size());
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipPresentation() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_presentation_header_t{1});
  section.Append(iree_vm_bytecode_v0_presentation_entry_row_t{
      1, IREE_VM_BYTECODE_PRESENTATION_DECLARATION_KIND_EXPORT, 4, 5, 0});
  section.Append(iree_vm_bytecode_v0_presentation_field_row_t{6, 7});
  section.Append(iree_vm_bytecode_v0_presentation_field_row_t{8, 7});
  section.Append(iree_vm_bytecode_v0_presentation_field_row_t{9, 10});
  return section.Take();
}

std::vector<uint8_t> BuildOwnershipMetadata() {
  constexpr std::string_view kModuleValue = "ownership";
  constexpr std::string_view kExportValue = "stable";
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_metadata_header_t{1, 0, 1, 2});
  section.Append(iree_vm_bytecode_v0_metadata_scope_row_t{1, 1, 1});
  section.Append(iree_vm_bytecode_v0_metadata_entry_row_t{
      11, IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8});
  section.Append(iree_vm_bytecode_v0_metadata_entry_row_t{
      13, IREE_VM_BYTECODE_METADATA_VALUE_TYPE_UTF8});
  section.Align(8);
  section.Append(static_cast<iree_vm_bytecode_v0_metadata_value_offset_t>(0));
  section.Append(static_cast<iree_vm_bytecode_v0_metadata_value_offset_t>(
      kModuleValue.size()));
  section.Append(static_cast<iree_vm_bytecode_v0_metadata_value_offset_t>(
      kModuleValue.size() + kExportValue.size()));
  section.AppendBytes(kModuleValue.data(), kModuleValue.size());
  section.AppendBytes(kExportValue.data(), kExportValue.size());
  return section.Take();
}

std::vector<uint8_t> BuildLaunchConfigSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      2,
      11,
      0,
      0,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      13,
      0,
      0,
      0,
      0,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_BF16, 0});
  for (int i = 0; i < 11; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_I64, 0});
  }
  return section.Take();
}

std::vector<uint8_t> BuildLaunchConfigCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildLaunchConfigExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{3});
  section.Append(iree_vm_bytecode_v0_export_row_t{0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{1, 1, 1, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 2, 0});
  return section.Take();
}

std::vector<uint8_t> BuildLaunchConfigFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_conversion_float_extend_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_EXTEND, 2, 1,
      IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32});
  bytecode.Append(iree_vm_isa_conversion_float_to_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER, 2, 2,
      IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32});
  bytecode.Append(iree_vm_isa_conversion_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER, 0, 0,
      IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64});
  bytecode.Append(iree_vm_isa_conversion_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_INTEGER, 2, 2,
      IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64});
  bytecode.Append(iree_vm_isa_integer_mul_i64_record_t{
      IREE_VM_ISA_CORE_OPCODE_INTEGER_MUL_I64, 0, 0, 2});
  bytecode.Append(iree_vm_isa_constant_s16_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16, 1, 1});
  for (uint8_t dst = 2; dst <= 8; ++dst) {
    bytecode.Append(iree_vm_isa_value_copy_record_t{
        IREE_VM_ISA_CORE_OPCODE_VALUE_COPY, dst, 1, 0});
  }
  bytecode.Append(iree_vm_isa_constant_s16_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16, 9, 32});
  bytecode.Append(iree_vm_isa_constant_s16_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16, 10, 256});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{3});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 68, 0, 0, 0, 11, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 68, 8, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 76, 8, 0, 0, 0, 11, 0, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildValueOverflowSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{1});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      18,
      18,
      0,
      0,
      0,
      0,
  });
  for (int i = 0; i < 36; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_I64, 0});
  }
  return section.Take();
}

std::vector<uint8_t> BuildValueOverflowCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{1});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildValueOverflowExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{2});
  section.Append(iree_vm_bytecode_v0_export_row_t{1, 0, 1, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{0, 0, 0, 0});
  return section.Take();
}

void AppendValueOverflowTransfers(ByteBuffer* bytecode) {
  bytecode->Append(iree_vm_isa_value_abi_argument_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_ARGUMENT_LOAD, 16, 0});
  bytecode->Append(iree_vm_isa_value_abi_argument_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_ARGUMENT_LOAD, 17, 1});
  bytecode->Append(iree_vm_isa_value_abi_result_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_RESULT_STORE, 16, 0});
  bytecode->Append(iree_vm_isa_value_abi_result_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_ABI_RESULT_STORE, 17, 1});
}

std::vector<uint8_t> BuildValueOverflowFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  AppendValueOverflowTransfers(&bytecode);
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  AppendValueOverflowTransfers(&bytecode);
  bytecode.Append(iree_vm_isa_conversion_float_to_integer_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONVERSION_FLOAT_TO_INTEGER, 0, 0,
      IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{2});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 24, 0, 0, 0, 18, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 24, 28, 0, 0, 0, 18, 0, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{2});
  section.Append(iree_vm_bytecode_v0_signature_row_t{});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      0,
      5,
      0,
      0,
      0,
      0,
  });
  for (int i = 0; i < 5; ++i) {
    section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
        IREE_VM_BYTECODE_SIGNATURE_KIND_I64, 0});
  }
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{2});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{2});
  section.Append(iree_vm_bytecode_v0_export_row_t{0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_export_row_t{1, 1, 1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_constant_zero_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0});
  bytecode.Append(iree_vm_isa_global_value_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_MUTABLE_STORE, 0, 0});
  bytecode.Append(iree_vm_isa_constant_s16_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_S16, 1, 1});
  bytecode.Append(iree_vm_isa_global_value_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_MUTABLE_STORE, 1, 1});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_constant_zero_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_ZERO, 0, 0});
  bytecode.Append(iree_vm_isa_constant_i32_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_I32, 1, 0, UINT32_C(0x89ABCDEF)});
  bytecode.Append(iree_vm_isa_constant_i64_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_I64, 2, 0, UINT32_C(0x89ABCDEF),
      UINT32_C(0x01234567)});
  bytecode.Append(iree_vm_isa_constant_pool_load_i32_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_POOL_LOAD_I32, 4, 0});
  bytecode.Append(iree_vm_isa_constant_pool_load_i64_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONSTANT_POOL_LOAD_I64, 3, 1});
  bytecode.Append(iree_vm_isa_global_value_mutable_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_MUTABLE_LOAD, 5, 1});
  bytecode.Append(iree_vm_isa_value_select_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_SELECT, 2, 5, 2, 0, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_value_select_record_t{
      IREE_VM_ISA_CORE_OPCODE_VALUE_SELECT, 4, 0, 1, 4, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_global_value_mutable_store_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_MUTABLE_STORE, 2, 0});
  bytecode.Append(iree_vm_isa_global_value_mutable_load_record_t{
      IREE_VM_ISA_CORE_OPCODE_GLOBAL_VALUE_MUTABLE_LOAD, 2, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{2});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 24, 0, 0, 0, 2, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      1, 0, 24, 68, 0, 0, 0, 6, 0, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateConstants() {
  ByteBuffer section;
  section.Append(static_cast<iree_vm_bytecode_v0_constant_cell_t>(
      UINT64_C(0x00000000DEADBEEF)));
  section.Append(static_cast<iree_vm_bytecode_v0_constant_cell_t>(
      UINT64_C(0xFEDCBA9876543210)));
  return section.Take();
}

std::vector<uint8_t> BuildScalarStateGlobals() {
  ByteBuffer section;
  section.Append(
      iree_vm_bytecode_v0_globals_header_t{2, 0, 0, 0, 0, 0, {0, 0}});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionRefTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_ref_types_header_t{1});
  section.Append(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 1});
  section.Append(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{1});
  section.Append(iree_vm_bytecode_v0_signature_row_t{
      0,
      0,
      1,
      1,
      0,
      0,
      0,
  });
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_REF, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I64, 0});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{1});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionImports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_imports_header_t{1});
  section.Append(iree_vm_bytecode_v0_import_group_row_t{3, 0, 1});
  section.Append(iree_vm_bytecode_v0_import_entry_row_t{
      4, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{1});
  section.Append(iree_vm_bytecode_v0_export_row_t{2, 0, 0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildHALInspectionFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_hal_device_group_count_record_t{
      IREE_VM_ISA_PAGE_HAL, IREE_VM_ISA_HAL_OPCODE_DEVICE_GROUP_COUNT, 0, 0});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});

  std::vector<uint8_t> bytecode_data = bytecode.Take();
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{1});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 12, 0, 0, 0, 1, 1, 0, 0, 0, {0, 0, 0}});
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

std::vector<uint8_t> BuildSwitchInspectionSignatures() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_signatures_header_t{1});
  section.Append(iree_vm_bytecode_v0_signature_row_t{0, 1, 0, 0, 0, 0, 0});
  section.Append(iree_vm_bytecode_v0_signature_descriptor_row_t{
      IREE_VM_BYTECODE_SIGNATURE_KIND_I32, 0});
  return section.Take();
}

std::vector<uint8_t> BuildSwitchInspectionCallableTypes() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_callable_types_header_t{1});
  section.Append(iree_vm_bytecode_v0_callable_type_row_t{0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildSwitchInspectionExports() {
  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_exports_header_t{1});
  section.Append(iree_vm_bytecode_v0_export_row_t{0, 0, 0, 0});
  return section.Take();
}

std::vector<uint8_t> BuildSwitchInspectionFunctions() {
  ByteBuffer bytecode;
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_switch_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_SWITCH, 0, 1, 0});
  bytecode.Append(iree_vm_isa_control_block_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_BLOCK, {0, 0, 0}});
  bytecode.Append(iree_vm_isa_control_return_record_t{
      IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN, {0, 0, 0}});
  std::vector<uint8_t> bytecode_data = bytecode.Take();

  ByteBuffer section;
  section.Append(iree_vm_bytecode_v0_functions_header_t{1});
  section.Append(iree_vm_bytecode_v0_function_row_t{
      0, 0, 0, 20, 0, 1, 0, 1, 0, 0, 0, 0, {0, 0, 0}});
  section.Append(static_cast<iree_vm_bytecode_v0_switch_target_entry_t>(3));
  section.AppendBytes(bytecode_data.data(), bytecode_data.size());
  return section.Take();
}

}  // namespace

uint8_t* FindSectionPayload(std::vector<uint8_t>* image,
                            uint16_t section_type) {
  auto* header =
      reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image->data());
  auto* rows = reinterpret_cast<iree_vm_bytecode_v0_section_directory_row_t*>(
      header + 1);
  size_t offset = sizeof(*header) + header->section_count_u16 * sizeof(*rows);
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    offset = (offset + IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1) &
             ~(IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1);
    if (rows[i].section_type_u16 == section_type) {
      return image->data() + offset;
    }
    offset += static_cast<size_t>(rows[i].byte_length_u64);
  }
  return nullptr;
}

MutableFunctionImage FindFunctionImage(std::vector<uint8_t>* image,
                                       uint32_t ordinal) {
  uint8_t* section =
      FindSectionPayload(image, IREE_VM_BYTECODE_SECTION_FUNCTIONS);
  if (section == nullptr) return {};
  auto* header =
      reinterpret_cast<iree_vm_bytecode_v0_functions_header_t*>(section);
  if (ordinal >= header->function_count_u32) return {};
  auto* rows = reinterpret_cast<iree_vm_bytecode_v0_function_row_t*>(
      section + sizeof(*header));
  uint8_t* bytecode_data =
      reinterpret_cast<uint8_t*>(rows + header->function_count_u32);
  bytecode_data +=
      header->function_count_u32 == 0
          ? 0
          : rows[header->function_count_u32 - 1].switch_target_base_u32 *
                sizeof(iree_vm_bytecode_v0_switch_target_entry_t);
  bytecode_data +=
      header->function_count_u32 == 0
          ? 0
          : rows[header->function_count_u32 - 1].switch_target_entry_count_u32 *
                sizeof(iree_vm_bytecode_v0_switch_target_entry_t);
  return {&rows[ordinal], bytecode_data + rows[ordinal].bytecode_offset_u32};
}

std::vector<uint8_t> BuildOwnershipModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"vm", "buffer", "initialize", "run",
                     "Adds the process seed and returns image rodata.",
                     "(i32) -> (i32, !vm.ref<vm, buffer>)", "value", "i32",
                     "sum", "payload", "!vm.ref<vm, buffer>", "model.kind",
                     "ownership", "result.note", "stable"})},
      {IREE_VM_BYTECODE_SECTION_REF_TYPES, 0, BuildOwnershipRefTypes()},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildOwnershipSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildOwnershipCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildOwnershipExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildOwnershipFunctions()},
      {IREE_VM_BYTECODE_SECTION_GLOBALS, 0, BuildOwnershipGlobals()},
      {IREE_VM_BYTECODE_SECTION_RODATA, 0, BuildTestRodata()},
      {IREE_VM_BYTECODE_SECTION_PRESENTATION,
       IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE, BuildOwnershipPresentation()},
      {IREE_VM_BYTECODE_SECTION_METADATA,
       IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE, BuildOwnershipMetadata()},
  });
}

std::vector<uint8_t> BuildLaunchConfigModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"decode", "empty", "noop"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildLaunchConfigSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildLaunchConfigCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildLaunchConfigExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildLaunchConfigFunctions()},
  });
}

std::vector<uint8_t> BuildRefModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"vm", "buffer", "exercise", "fail"})},
      {IREE_VM_BYTECODE_SECTION_REF_TYPES, 0, BuildOwnershipRefTypes()},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildRefSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0, BuildRefCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildRefExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildRefFunctions()},
  });
}

std::vector<uint8_t> BuildRefStateModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"vm", "buffer", "initialize", "overflow", "read",
                     "wrong_store", "zz_test", "object"})},
      {IREE_VM_BYTECODE_SECTION_REF_TYPES, 0, BuildRefStateRefTypes()},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildRefStateSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildRefStateCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildRefStateExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildRefStateFunctions()},
      {IREE_VM_BYTECODE_SECTION_GLOBALS, 0, BuildRefStateGlobals()},
  });
}

std::vector<uint8_t> BuildFunctionModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"missing", "callback", "run"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildFunctionSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildFunctionCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_IMPORTS, 0, BuildFunctionImports()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildFunctionExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildFunctionFunctions()},
  });
}

std::vector<uint8_t> BuildFunctionStateModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"bad_result", "bad_store", "callback", "initialize",
                     "overflow", "read", "replace"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildFunctionStateSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildFunctionStateCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildFunctionStateExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildFunctionStateFunctions()},
      {IREE_VM_BYTECODE_SECTION_GLOBALS, 0, BuildFunctionStateGlobals()},
  });
}

std::vector<uint8_t> BuildValueOverflowModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"identity", "fail_after_store"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildValueOverflowSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildValueOverflowCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildValueOverflowExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildValueOverflowFunctions()},
  });
}

std::vector<uint8_t> BuildScalarStateModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"initialize", "run"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildScalarStateSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildScalarStateCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildScalarStateExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildScalarStateFunctions()},
      {IREE_VM_BYTECODE_SECTION_CONSTANTS, 0, BuildScalarStateConstants()},
      {IREE_VM_BYTECODE_SECTION_GLOBALS, 0, BuildScalarStateGlobals()},
      {IREE_VM_BYTECODE_SECTION_RODATA, 0, BuildTestRodata()},
  });
}

std::vector<uint8_t> BuildHALInspectionModuleImage() {
  ByteBuffer requirements;
  requirements.Append(iree_vm_bytecode_v0_requirement_row_t{
      IREE_VM_ISA_PAGE_HAL, IREE_VM_ISA_HAL_MAJOR, IREE_VM_ISA_HAL_MINOR});
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_REQUIREMENTS, 0, requirements.Take()},
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0,
       BuildStrings({"hal", "device_group", "device_count", "runtime.support",
                     "query_device_count"})},
      {IREE_VM_BYTECODE_SECTION_REF_TYPES, 0, BuildHALInspectionRefTypes()},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0, BuildHALInspectionSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildHALInspectionCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_IMPORTS, 0, BuildHALInspectionImports()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildHALInspectionExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildHALInspectionFunctions()},
  });
}

std::vector<uint8_t> BuildSwitchInspectionModuleImage() {
  return BuildImage({
      {IREE_VM_BYTECODE_SECTION_STRINGS, 0, BuildStrings({"select"})},
      {IREE_VM_BYTECODE_SECTION_SIGNATURES, 0,
       BuildSwitchInspectionSignatures()},
      {IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES, 0,
       BuildSwitchInspectionCallableTypes()},
      {IREE_VM_BYTECODE_SECTION_EXPORTS, 0, BuildSwitchInspectionExports()},
      {IREE_VM_BYTECODE_SECTION_FUNCTIONS, 0, BuildSwitchInspectionFunctions()},
  });
}

}  // namespace iree::vm::bytecode::testing
