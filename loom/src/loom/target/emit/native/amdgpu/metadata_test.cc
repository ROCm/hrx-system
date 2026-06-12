// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/metadata.h"

#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/emit/native/elf.h"

namespace loom {
namespace {

std::string BuilderString(const iree_string_builder_t& builder) {
  iree_string_view_t view = iree_string_builder_view(&builder);
  return std::string(view.data, view.size);
}

bool Contains(const std::string& value, const char* substring) {
  return value.find(substring) != std::string::npos;
}

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient ELF writer storage.
  iree_arena_allocator_t arena_ = {0};
};

StreamPtr CreateStream() {
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  return bytes;
}

uint16_t LoadLeU16(const std::string& bytes, size_t offset) {
  return (uint16_t)(uint8_t)bytes[offset + 0] |
         ((uint16_t)(uint8_t)bytes[offset + 1] << 8);
}

uint32_t LoadLeU32(const std::string& bytes, size_t offset) {
  return ((uint32_t)(uint8_t)bytes[offset + 0]) |
         ((uint32_t)(uint8_t)bytes[offset + 1] << 8) |
         ((uint32_t)(uint8_t)bytes[offset + 2] << 16) |
         ((uint32_t)(uint8_t)bytes[offset + 3] << 24);
}

uint64_t LoadLeU64(const std::string& bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= (uint64_t)(uint8_t)bytes[offset + i] << (8 * i);
  }
  return value;
}

size_t Align4(size_t value) { return (value + 3u) & ~3u; }

loom_amdgpu_metadata_kernel_t MinimalKernel() {
  return {
      /*.name=*/IREE_SV("loom_kernel"),
      /*.descriptor_symbol=*/IREE_SV("loom_kernel.kd"),
      /*.kernarg_segment_size=*/0,
      /*.kernarg_segment_alignment=*/8,
      /*.wavefront_size=*/32,
      /*.group_segment_fixed_size=*/0,
      /*.private_segment_fixed_size=*/0,
      /*.sgpr_count=*/3,
      /*.vgpr_count=*/0,
      /*.max_flat_workgroup_size=*/64,
      /*.required_workgroup_size=*/{/*.x=*/64, /*.y=*/1, /*.z=*/1},
      /*.has_required_workgroup_size=*/true,
      /*.arguments=*/nullptr,
      /*.argument_count=*/0,
  };
}

loom_amdgpu_code_object_metadata_t MetadataForKernel(
    const loom_amdgpu_metadata_kernel_t* kernel) {
  return {
      /*.target=*/IREE_SV("amdgcn-amd-amdhsa--gfx1100"),
      /*.kernels=*/kernel,
      /*.kernel_count=*/1,
  };
}

TEST(AmdgpuMetadataTest, AppendsAssemblyMetadataForNoArgumentKernel) {
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_metadata_append_assembly(&metadata, &builder));
  std::string text = BuilderString(builder);
  iree_string_builder_deinitialize(&builder);

  EXPECT_TRUE(Contains(text, ".amdgpu_metadata\n")) << text;
  EXPECT_TRUE(Contains(text, "  amdhsa.version:\n")) << text;
  EXPECT_TRUE(Contains(text, "    - 1\n    - 2\n")) << text;
  EXPECT_TRUE(Contains(text, "  amdhsa.target: 'amdgcn-amd-amdhsa--gfx1100'\n"))
      << text;
  EXPECT_TRUE(Contains(text, "  amdhsa.kernels:\n")) << text;
  EXPECT_TRUE(Contains(text, "    - .name: 'loom_kernel'\n")) << text;
  EXPECT_TRUE(Contains(text, "      .symbol: 'loom_kernel.kd'\n")) << text;
  EXPECT_TRUE(Contains(text, "      .kernarg_segment_size: 0\n")) << text;
  EXPECT_TRUE(Contains(text, "      .kernarg_segment_align: 8\n")) << text;
  EXPECT_TRUE(Contains(text, "      .wavefront_size: 32\n")) << text;
  EXPECT_TRUE(Contains(text, "      .sgpr_count: 3\n")) << text;
  EXPECT_TRUE(Contains(text, "      .vgpr_count: 0\n")) << text;
  EXPECT_TRUE(Contains(text, "      .max_flat_workgroup_size: 64\n")) << text;
  EXPECT_TRUE(Contains(text, "      .reqd_workgroup_size:\n")) << text;
  EXPECT_TRUE(Contains(text, "      .args: []\n")) << text;
  EXPECT_TRUE(Contains(text, ".end_amdgpu_metadata\n")) << text;
}

TEST(AmdgpuMetadataTest, AppendsMsgpackMetadataForNoArgumentKernel) {
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_metadata_append_msgpack(&metadata, &builder));
  std::string bytes = BuilderString(builder);
  iree_string_builder_deinitialize(&builder);

  ASSERT_GT(bytes.size(), 0u);
  EXPECT_TRUE(Contains(bytes, "amdhsa.version"));
  EXPECT_TRUE(Contains(bytes, "amdhsa.target"));
  EXPECT_TRUE(Contains(bytes, "amdhsa.kernels"));
  EXPECT_TRUE(Contains(bytes, "amdgcn-amd-amdhsa--gfx1100"));
  EXPECT_TRUE(Contains(bytes, "loom_kernel"));
  EXPECT_TRUE(Contains(bytes, "loom_kernel.kd"));
  EXPECT_TRUE(Contains(bytes, ".wavefront_size"));
  EXPECT_TRUE(Contains(bytes, ".args"));
  EXPECT_FALSE(Contains(bytes, ".amdgpu_metadata"));
}

TEST(AmdgpuMetadataTest, AppendsElfNoteMetadata) {
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t payload_builder;
  iree_string_builder_initialize(iree_allocator_system(), &payload_builder);
  IREE_ASSERT_OK(
      loom_amdgpu_metadata_append_msgpack(&metadata, &payload_builder));
  std::string payload = BuilderString(payload_builder);
  iree_string_builder_deinitialize(&payload_builder);

  iree_string_builder_t note_builder;
  iree_string_builder_initialize(iree_allocator_system(), &note_builder);
  IREE_ASSERT_OK(
      loom_amdgpu_metadata_append_elf_note(&metadata, &note_builder));
  std::string note = BuilderString(note_builder);
  iree_string_builder_deinitialize(&note_builder);

  const char note_name[] = {'A', 'M', 'D', 'G', 'P', 'U', '\0'};
  constexpr size_t kHeaderSize = 12;
  const size_t name_size = sizeof(note_name);
  const size_t desc_offset = Align4(kHeaderSize + name_size);
  const size_t note_size = Align4(desc_offset + payload.size());
  ASSERT_EQ(note.size(), note_size);
  ASSERT_GE(note.size(), desc_offset + payload.size());

  EXPECT_EQ(LoadLeU32(note, 0), name_size);
  EXPECT_EQ(LoadLeU32(note, 4), payload.size());
  EXPECT_EQ(LoadLeU32(note, 8), 32u);
  EXPECT_EQ(note.substr(kHeaderSize, name_size),
            std::string(note_name, name_size));
  for (size_t i = kHeaderSize + name_size; i < desc_offset; ++i) {
    EXPECT_EQ((uint8_t)note[i], 0u);
  }
  EXPECT_EQ(note.substr(desc_offset, payload.size()), payload);
  for (size_t i = desc_offset + payload.size(); i < note_size; ++i) {
    EXPECT_EQ((uint8_t)note[i], 0u);
  }
}

TEST(AmdgpuMetadataTest, WritesElfEnvelopeContainingMetadataNote) {
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t note_builder;
  iree_string_builder_initialize(iree_allocator_system(), &note_builder);
  IREE_ASSERT_OK(
      loom_amdgpu_metadata_append_elf_note(&metadata, &note_builder));
  std::string note = BuilderString(note_builder);
  iree_string_builder_deinitialize(&note_builder);

  const loom_native_elf64le_section_t sections[] = {{
      /*.name=*/IREE_SV(".note"),
      /*.type=*/LOOM_NATIVE_ELF_SECTION_TYPE_NOTE,
      /*.flags=*/LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC,
      /*.address=*/0,
      /*.alignment=*/4,
      /*.entry_size=*/0,
      /*.link=*/0,
      /*.info=*/0,
      /*.contents=*/iree_make_const_byte_span(note.data(), note.size()),
  }};
  const loom_native_elf64le_segment_t segments[] = {{
      /*.type=*/LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE,
      /*.flags=*/LOOM_NATIVE_ELF_PROGRAM_FLAG_READ,
      /*.file_offset=*/{},
      /*.file_size=*/{},
      /*.memory_size=*/{},
      /*.first_section=*/0,
      /*.section_count=*/1,
      /*.virtual_address=*/0,
      /*.physical_address=*/0,
      /*.alignment=*/4,
  }};
  const loom_native_elf64le_file_t file = {
      /*.type=*/LOOM_NATIVE_ELF_FILE_TYPE_DYN,
      /*.machine=*/LOOM_NATIVE_ELF_MACHINE_AMDGPU,
      /*.os_abi=*/LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA,
      /*.abi_version=*/LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5,
      /*.flags=*/LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100,
      /*.entry=*/0,
      /*.sections=*/sections,
      /*.section_count=*/IREE_ARRAYSIZE(sections),
      /*.segments=*/segments,
      /*.segment_count=*/IREE_ARRAYSIZE(segments),
  };

  StreamPtr stream = CreateStream();
  TestArena arena;
  IREE_ASSERT_OK(
      loom_native_elf64le_write_file(&file, stream.get(), arena.arena()));
  std::string bytes = StreamBytes(stream.get());

  EXPECT_EQ(bytes.substr(0, 4), std::string("\x7f"
                                            "ELF",
                                            4));
  EXPECT_EQ((uint8_t)bytes[7], LOOM_NATIVE_ELF_OS_ABI_AMDGPU_HSA);
  EXPECT_EQ((uint8_t)bytes[8], LOOM_NATIVE_ELF_ABI_VERSION_AMDGPU_HSA_V5);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AMDGPU);
  EXPECT_EQ(LoadLeU32(bytes, 48), LOOM_NATIVE_ELF_AMDGPU_FLAG_MACH_GFX1100);
  ASSERT_EQ(LoadLeU16(bytes, 56), 1u);
  const size_t program_header_offset = (size_t)LoadLeU64(bytes, 32);
  EXPECT_EQ(LoadLeU32(bytes, program_header_offset),
            LOOM_NATIVE_ELF_PROGRAM_TYPE_NOTE);
  const size_t note_offset =
      (size_t)LoadLeU64(bytes, program_header_offset + 8);
  const size_t note_size = (size_t)LoadLeU64(bytes, program_header_offset + 32);
  ASSERT_LE(note_offset + note_size, bytes.size());
  EXPECT_EQ(bytes.substr(note_offset, note_size), note);
}

TEST(AmdgpuMetadataTest, AppendsArgumentMetadata) {
  const loom_amdgpu_metadata_argument_t arguments[] = {
      {
          /*.name=*/IREE_SV("lhs"),
          /*.offset=*/0,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
          /*.address_space=*/IREE_SV("global"),
          /*.access=*/IREE_SV("read_write"),
          /*.actual_access=*/IREE_SV("read_only"),
      },
      {
          /*.name=*/IREE_SV("scale"),
          /*.offset=*/8,
          /*.size=*/4,
          /*.alignment=*/4,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
  };
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  kernel.kernarg_segment_size = 12;
  kernel.arguments = arguments;
  kernel.argument_count = IREE_ARRAYSIZE(arguments);
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t text_builder;
  iree_string_builder_initialize(iree_allocator_system(), &text_builder);
  IREE_ASSERT_OK(
      loom_amdgpu_metadata_append_assembly(&metadata, &text_builder));
  std::string text = BuilderString(text_builder);
  iree_string_builder_deinitialize(&text_builder);

  EXPECT_TRUE(Contains(text, "      .args:\n")) << text;
  EXPECT_TRUE(Contains(text, "        - .name: 'lhs'\n")) << text;
  EXPECT_TRUE(Contains(text, "          .value_kind: global_buffer\n")) << text;
  EXPECT_TRUE(Contains(text, "          .address_space: global\n")) << text;
  EXPECT_TRUE(Contains(text, "          .access: read_write\n")) << text;
  EXPECT_TRUE(Contains(text, "          .actual_access: read_only\n")) << text;
  EXPECT_TRUE(Contains(text, "        - .name: 'scale'\n")) << text;
  EXPECT_TRUE(Contains(text, "          .value_kind: by_value\n")) << text;

  iree_string_builder_t msgpack_builder;
  iree_string_builder_initialize(iree_allocator_system(), &msgpack_builder);
  IREE_ASSERT_OK(
      loom_amdgpu_metadata_append_msgpack(&metadata, &msgpack_builder));
  std::string bytes = BuilderString(msgpack_builder);
  iree_string_builder_deinitialize(&msgpack_builder);

  EXPECT_TRUE(Contains(bytes, "lhs"));
  EXPECT_TRUE(Contains(bytes, "global_buffer"));
  EXPECT_TRUE(Contains(bytes, "read_write"));
  EXPECT_TRUE(Contains(bytes, "read_only"));
  EXPECT_TRUE(Contains(bytes, "scale"));
  EXPECT_TRUE(Contains(bytes, "by_value"));
}

TEST(AmdgpuMetadataTest, RejectsInvalidArgumentRange) {
  const loom_amdgpu_metadata_argument_t arguments[] = {
      {
          /*.name=*/{},
          /*.offset=*/0,
          /*.size=*/8,
          /*.alignment=*/8,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
      },
      {
          /*.name=*/{},
          /*.offset=*/4,
          /*.size=*/4,
          /*.alignment=*/4,
          /*.kind=*/LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
      },
  };
  loom_amdgpu_metadata_kernel_t kernel = MinimalKernel();
  kernel.kernarg_segment_size = 8;
  kernel.arguments = arguments;
  kernel.argument_count = IREE_ARRAYSIZE(arguments);
  loom_amdgpu_code_object_metadata_t metadata = MetadataForKernel(&kernel);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_metadata_append_msgpack(&metadata, &builder));
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
