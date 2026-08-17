// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "context_blob.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr size_t kAxlfBaseOffset = 0xE8;
constexpr size_t kLegacyContextTailSize = 0x370;
constexpr size_t kLegacyV0AxlfBaseOffset = 0xC0;
constexpr size_t kLegacyV0ContextTailSize = 0x530;
constexpr size_t kLegacyV0MlirAieContextTailSize = 0x370;
constexpr size_t kLegacyV2AxlfBaseOffset = 0xE0;
constexpr size_t kLegacyV2ContextTailSize = 0x3C8;
constexpr size_t kLegacyV2MlirAieContextTailSize = 0x370;
constexpr uint64_t kCommandApertureBase = 0x04000000;
constexpr uint64_t kContextCommandBoSize = 0x1000;
constexpr size_t kCompactContextPrivateDataSize = 0xA0;
constexpr uint32_t kCompactContextInstructionOffset = 0x800;
constexpr uint32_t kBuildMetadataSection = 14;
constexpr uint32_t kAiePartitionSection = 32;
constexpr uint32_t kIpLayoutSection = 8;
constexpr uint32_t kMaxAxlfSections = 4096;
constexpr uint32_t kMaxIpLayoutRecords = 4096;
constexpr uint32_t kMaxAiePartitionPdis = 4096;
constexpr size_t kMaxContextBlobSize = 512ull * 1024ull * 1024ull;
constexpr size_t kIpLayoutAieHeaderSize = 8;
constexpr size_t kIpLayoutLegacyHeaderSize = 4;
constexpr size_t kIpDataRecordSize = 80;
constexpr size_t kIpDataNameOffset = 16;
constexpr size_t kIpDataNameSize = 64;

struct AxlfSection {
  uint32_t kind = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct AxlfSectionList {
  AxlfSection* values = nullptr;
  uint32_t count = 0;
  iree_allocator_t allocator = iree_allocator_null();
};

bool Fail(const char* message, Error* out_error) {
  if (out_error) {
    std::snprintf(out_error->message, sizeof(out_error->message), "%s",
                  message ? message : "unknown context blob error");
  }
  return false;
}

bool FailStatus(iree_status_t status, const char* message, Error* out_error) {
  if (iree_status_is_ok(status)) return true;
  iree_status_ignore(status);
  return Fail(message, out_error);
}

bool CheckRange(size_t data_size, uint64_t offset, uint64_t size,
                const char* what, Error* out_error) {
  if (offset > data_size || size > data_size - offset) {
    if (out_error) {
      std::snprintf(out_error->message, sizeof(out_error->message),
                    "%s is out of bounds", what);
    }
    return false;
  }
  return true;
}

void AxlfSectionListDeinitialize(AxlfSectionList* sections) {
  if (!sections) return;
  if (!iree_allocator_is_null(sections->allocator)) {
    iree_allocator_free(sections->allocator, sections->values);
  }
  *sections = AxlfSectionList();
}

uint16_t ReadU16(const uint8_t* data, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint32_t ReadU32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint64_t ReadU64(const uint8_t* data, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

void WriteU32(uint8_t* data, size_t offset, uint32_t value) {
  std::memcpy(data + offset, &value, sizeof(value));
}

void WriteU64(uint8_t* data, size_t offset, uint64_t value) {
  std::memcpy(data + offset, &value, sizeof(value));
}

size_t FixedStringLength(const char* value, size_t capacity) {
  size_t length = 0;
  while (length < capacity && value[length] != 0) ++length;
  return length;
}

bool CopyFixedName(const uint8_t* data, size_t size, char* out,
                   const char* what, Error* out_error) {
  size_t length = 0;
  while (length < size && data[length] != 0) ++length;
  if (length >= kContextBlobNameCapacity) {
    if (out_error) {
      std::snprintf(out_error->message, sizeof(out_error->message),
                    "%s does not fit in fixed metadata storage", what);
    }
    return false;
  }
  std::memset(out, 0, kContextBlobNameCapacity);
  if (length > 0) std::memcpy(out, data, length);
  return true;
}

bool CopyLiteralName(const char* value, char* out, Error* out_error) {
  size_t length = FixedStringLength(value, kContextBlobNameCapacity);
  if (length >= kContextBlobNameCapacity) {
    return Fail("literal name does not fit in fixed metadata storage",
                out_error);
  }
  std::memset(out, 0, kContextBlobNameCapacity);
  std::memcpy(out, value, length);
  return true;
}

bool WriteCString(uint8_t* data, size_t offset, size_t size, const char* value,
                  Error* out_error) {
  size_t length = FixedStringLength(value, size);
  if (length >= size) {
    return Fail("string does not fit in context blob field", out_error);
  }
  std::memset(data + offset, 0, size);
  std::memcpy(data + offset, value, length);
  return true;
}

bool ParseSections(const uint8_t* xclbin, size_t xclbin_size,
                   iree_allocator_t allocator, AxlfSectionList* out_sections,
                   Error* out_error) {
  if (xclbin_size < 0x1C8 || std::memcmp(xclbin, "xclbin2\0", 8) != 0) {
    return Fail("input is not an AXLF/xclbin2 file", out_error);
  }
  uint32_t section_count = ReadU32(xclbin, 0x1C0);
  if (section_count > kMaxAxlfSections) {
    return Fail("AXLF section count exceeds supported limit", out_error);
  }
  if (!CheckRange(xclbin_size, 0x1C8, uint64_t{40} * section_count,
                  "AXLF section table", out_error)) {
    return false;
  }

  out_sections->allocator = allocator;
  out_sections->count = section_count;
  if (section_count > 0) {
    iree_status_t status = iree_allocator_malloc_array(
        allocator, section_count, sizeof(AxlfSection),
        reinterpret_cast<void**>(&out_sections->values));
    if (!FailStatus(status, "AXLF section table allocation failed",
                    out_error)) {
      return false;
    }
  }

  for (uint32_t i = 0; i < section_count; ++i) {
    size_t record = 0x1C8 + size_t{i} * 40;
    AxlfSection* section = &out_sections->values[i];
    section->kind = ReadU32(xclbin, record);
    section->offset = ReadU64(xclbin, record + 24);
    section->size = ReadU64(xclbin, record + 32);
    if (!CheckRange(xclbin_size, section->offset, section->size, "AXLF section",
                    out_error)) {
      return false;
    }
  }
  return true;
}

const AxlfSection* FindFirstSection(const AxlfSectionList& sections,
                                    uint32_t kind) {
  for (uint32_t i = 0; i < sections.count; ++i) {
    if (sections.values[i].kind == kind) return &sections.values[i];
  }
  return nullptr;
}

size_t FindBytes(const uint8_t* data, size_t size, const char* needle,
                 size_t start) {
  const size_t needle_size = std::strlen(needle);
  if (needle_size == 0 || start > size || needle_size > size - start) {
    return size;
  }
  for (size_t i = start; i <= size - needle_size; ++i) {
    if (std::memcmp(data + i, needle, needle_size) == 0) return i;
  }
  return size;
}

bool ExtractJsonStringAfter(const uint8_t* json, size_t json_size, size_t start,
                            const char* key, char* out_value) {
  size_t key_pos = FindBytes(json, json_size, key, start);
  if (key_pos == json_size) return false;
  size_t colon = FindBytes(json, json_size, ":", key_pos + std::strlen(key));
  if (colon == json_size) return false;
  size_t quote = FindBytes(json, json_size, "\"", colon + 1);
  if (quote == json_size) return false;

  size_t out_length = 0;
  for (size_t i = quote + 1; i < json_size; ++i) {
    char c = static_cast<char>(json[i]);
    if (c == '"') {
      out_value[out_length] = 0;
      return true;
    }
    if (c == '\\' && i + 1 < json_size) {
      c = static_cast<char>(json[++i]);
    }
    if (out_length + 1 >= kContextBlobNameCapacity) return false;
    out_value[out_length++] = c;
  }
  return false;
}

void StripLinkSuffix(char* name) {
  constexpr char kLinkSuffix[] = ".link";
  constexpr size_t kLinkSuffixLength = sizeof(kLinkSuffix) - 1;
  size_t length = FixedStringLength(name, kContextBlobNameCapacity);
  if (length > kLinkSuffixLength &&
      std::memcmp(name + length - kLinkSuffixLength, kLinkSuffix,
                  kLinkSuffixLength) == 0) {
    name[length - kLinkSuffixLength] = 0;
  }
}

void NormalizeIpName(char* name) {
  for (size_t i = 0; i < kContextBlobNameCapacity && name[i] != 0; ++i) {
    if (name[i] == ':') {
      name[i] = 0;
      return;
    }
  }
}

bool DeriveKernelNameFromMetadata(const uint8_t* xclbin,
                                  const AxlfSectionList& sections,
                                  char* out_name, Error* out_error) {
  const AxlfSection* section =
      FindFirstSection(sections, kBuildMetadataSection);
  if (!section) return CopyLiteralName("kernel", out_name, out_error);
  const uint8_t* json = xclbin + section->offset;
  const size_t json_size = static_cast<size_t>(section->size);

  size_t kernels_pos = FindBytes(json, json_size, "\"kernels\"", 0);
  if (kernels_pos != json_size &&
      ExtractJsonStringAfter(json, json_size, kernels_pos, "\"name\"",
                             out_name) &&
      out_name[0] != 0) {
    return true;
  }

  if (ExtractJsonStringAfter(json, json_size, 0, "\"xclbin_name\"", out_name) &&
      out_name[0] != 0) {
    StripLinkSuffix(out_name);
    return true;
  }

  return CopyLiteralName("kernel", out_name, out_error);
}

bool AllocateNameTable(iree_allocator_t allocator, uint32_t count,
                       char** out_names, const char* what, Error* out_error) {
  if (count == 0) return true;
  iree_status_t status =
      iree_allocator_malloc_array(allocator, count, kContextBlobNameCapacity,
                                  reinterpret_cast<void**>(out_names));
  return FailStatus(status, what, out_error);
}

char* MutableKernelName(ContextBlobInfo* info, uint32_t index) {
  return info->kernel_names + size_t{index} * kContextBlobNameCapacity;
}

char* MutablePdiName(ContextBlobInfo* info, uint32_t index) {
  return info->pdi_names + size_t{index} * kContextBlobNameCapacity;
}

bool ParseIpLayout(const uint8_t* xclbin, size_t xclbin_size,
                   const AxlfSectionList& sections, ContextBlobInfo* info,
                   Error* out_error) {
  const AxlfSection* section = FindFirstSection(sections, kIpLayoutSection);
  if (!section) return true;
  if (!CheckRange(xclbin_size, section->offset, section->size,
                  "IP_LAYOUT section", out_error)) {
    return false;
  }
  const uint8_t* data = xclbin + section->offset;
  size_t size = static_cast<size_t>(section->size);
  if (size < kIpLayoutLegacyHeaderSize) {
    return Fail("IP_LAYOUT section is too small", out_error);
  }

  uint32_t count = ReadU32(data, 0);
  if (count > kMaxIpLayoutRecords) {
    return Fail("IP_LAYOUT record count exceeds supported limit", out_error);
  }
  size_t records_offset = kIpLayoutAieHeaderSize;
  if (size < records_offset + uint64_t{kIpDataRecordSize} * count) {
    records_offset = kIpLayoutLegacyHeaderSize;
  }
  if (size < records_offset + uint64_t{kIpDataRecordSize} * count) {
    return Fail("IP_LAYOUT record table is out of bounds", out_error);
  }
  if (!AllocateNameTable(info->allocator, count, &info->kernel_names,
                         "IP_LAYOUT name allocation failed", out_error)) {
    return false;
  }

  for (uint32_t i = 0; i < count; ++i) {
    size_t record = records_offset + size_t{i} * kIpDataRecordSize;
    char name[kContextBlobNameCapacity] = {};
    if (!CopyFixedName(data + record + kIpDataNameOffset, kIpDataNameSize, name,
                       "IP_LAYOUT name", out_error)) {
      return false;
    }
    NormalizeIpName(name);
    if (name[0] == 0) continue;
    std::memcpy(MutableKernelName(info, info->kernel_name_count), name,
                kContextBlobNameCapacity);
    ++info->kernel_name_count;
  }
  return true;
}

bool ParseAiePartition(const uint8_t* xclbin, size_t xclbin_size,
                       const AxlfSectionList& sections, ContextBlobInfo* info,
                       Error* out_error) {
  const AxlfSection* section = FindFirstSection(sections, kAiePartitionSection);
  if (!section) return Fail("AIE_PARTITION AXLF section is missing", out_error);
  if (!CheckRange(xclbin_size, section->offset, section->size,
                  "AIE_PARTITION section", out_error)) {
    return false;
  }
  const uint8_t* data = xclbin + section->offset;
  size_t size = static_cast<size_t>(section->size);
  if (size < 0xC8) {
    return Fail("AIE_PARTITION section is too small", out_error);
  }

  info->column_width = ReadU16(data, 32);
  uint32_t start_columns_count = ReadU32(data, 40);
  uint32_t start_columns_offset = ReadU32(data, 44);
  if (start_columns_count == 0) {
    info->start_column = 0;
  } else {
    if (!CheckRange(size, start_columns_offset, 2, "start_columns",
                    out_error)) {
      return false;
    }
    info->start_column = ReadU16(data, start_columns_offset);
  }

  uint32_t pdi_count = ReadU32(data, 120);
  uint32_t pdi_offset = ReadU32(data, 124);
  if (pdi_count == 0) {
    return Fail("AIE_PARTITION contains no PDI records", out_error);
  }
  if (pdi_count > kMaxAiePartitionPdis) {
    return Fail("AIE_PARTITION PDI count exceeds supported limit", out_error);
  }
  if (!CheckRange(size, pdi_offset, uint64_t{0x60} * pdi_count,
                  "AIE_PARTITION PDI table", out_error)) {
    return false;
  }

  if (!AllocateNameTable(info->allocator, pdi_count, &info->pdi_names,
                         "AIE_PARTITION PDI name allocation failed",
                         out_error)) {
    return false;
  }
  iree_status_t status = iree_allocator_malloc_array(
      info->allocator, pdi_count, sizeof(uint64_t),
      reinterpret_cast<void**>(&info->dpu_kernel_ids));
  if (!FailStatus(status, "AIE_PARTITION kernel id allocation failed",
                  out_error)) {
    return false;
  }

  info->pdi_count = pdi_count;
  info->pdi_name_count = pdi_count;
  for (uint32_t i = 0; i < pdi_count; ++i) {
    size_t pdi_record = pdi_offset + size_t{i} * 0x60;
    uint32_t cdo_count = ReadU32(data, pdi_record + 24);
    uint32_t cdo_offset = ReadU32(data, pdi_record + 28);
    if (cdo_count == 0 ||
        !CheckRange(size, cdo_offset, uint64_t{0x70} * cdo_count,
                    "AIE_PARTITION CDO table", out_error)) {
      return Fail("AIE_PARTITION CDO table is missing", out_error);
    }

    uint32_t pdi_name_offset = ReadU32(data, cdo_offset);
    if (!CheckRange(size, pdi_name_offset, 64, "AIE_PARTITION CDO name",
                    out_error)) {
      return false;
    }
    if (!CopyFixedName(data + pdi_name_offset, 64, MutablePdiName(info, i),
                       "AIE_PARTITION CDO name", out_error)) {
      return false;
    }

    uint32_t kernel_count = ReadU32(data, cdo_offset + 16);
    uint32_t kernel_offset = ReadU32(data, cdo_offset + 20);
    if (kernel_count == 0 ||
        !CheckRange(size, kernel_offset, uint64_t{8} * kernel_count,
                    "AIE_PARTITION kernel id table", out_error)) {
      return Fail("AIE_PARTITION kernel id table is missing", out_error);
    }
    info->dpu_kernel_ids[i] = ReadU64(data, kernel_offset);
  }

  std::memcpy(info->pdi_name, ContextBlobInfoPdiName(info, 0),
              kContextBlobNameCapacity);
  info->dpu_kernel_id = info->dpu_kernel_ids[0];
  return true;
}

bool ParseContextBlobInfo(const uint8_t* xclbin, size_t xclbin_size,
                          iree_allocator_t allocator,
                          ContextBlobInfo* out_info, Error* out_error) {
  AxlfSectionList sections;
  ContextBlobInfo info;
  info.allocator = allocator;

  if (!ParseSections(xclbin, xclbin_size, allocator, &sections, out_error)) {
    goto fail;
  }
  if (!ParseIpLayout(xclbin, xclbin_size, sections, &info, out_error)) {
    goto fail;
  }
  if (info.kernel_name_count == 0) {
    if (!DeriveKernelNameFromMetadata(xclbin, sections, info.kernel_name,
                                      out_error)) {
      goto fail;
    }
  } else {
    std::memcpy(info.kernel_name, ContextBlobInfoKernelName(&info, 0),
                kContextBlobNameCapacity);
  }
  if (!ParseAiePartition(xclbin, xclbin_size, sections, &info, out_error)) {
    goto fail;
  }

  *out_info = info;
  AxlfSectionListDeinitialize(&sections);
  return true;

fail:
  ContextBlobInfoDeinitialize(&info);
  AxlfSectionListDeinitialize(&sections);
  return false;
}

}  // namespace

const char* ContextBlobInfoKernelName(const ContextBlobInfo* info,
                                      uint32_t index) {
  if (!info || !info->kernel_names || index >= info->kernel_name_count) {
    return "";
  }
  return info->kernel_names + size_t{index} * kContextBlobNameCapacity;
}

const char* ContextBlobInfoPdiName(const ContextBlobInfo* info,
                                   uint32_t index) {
  if (!info || !info->pdi_names || index >= info->pdi_name_count) return "";
  return info->pdi_names + size_t{index} * kContextBlobNameCapacity;
}

void ContextBlobInfoDeinitialize(ContextBlobInfo* info) {
  if (!info) return;
  if (!iree_allocator_is_null(info->allocator)) {
    iree_allocator_free(info->allocator, info->kernel_names);
    iree_allocator_free(info->allocator, info->pdi_names);
    iree_allocator_free(info->allocator, info->dpu_kernel_ids);
  }
  *info = ContextBlobInfo();
}

bool BuildContextPrivateDataFromXclbin(const uint8_t* xclbin,
                                       size_t xclbin_size, uint32_t process_id,
                                       iree_allocator_t allocator,
                                       iree_byte_span_t* out_blob,
                                       ContextBlobInfo* out_info,
                                       Error* out_error) {
  if (!xclbin || !out_blob || iree_allocator_is_null(allocator)) {
    return Fail("invalid output/context arguments", out_error);
  }
  *out_blob = iree_byte_span_empty();
  if (xclbin_size < 0x1B0) return Fail("xclbin is too small", out_error);

  ContextBlobInfo info;
  uint8_t* blob = nullptr;

  if (!ParseContextBlobInfo(xclbin, xclbin_size, allocator, &info,
                            out_error)) {
    return false;
  }

  if (xclbin_size > std::numeric_limits<size_t>::max() - kAxlfBaseOffset -
                        kLegacyContextTailSize) {
    Fail("context blob size overflows size_t", out_error);
    goto fail;
  }
  {
    size_t total_size =
        kAxlfBaseOffset + xclbin_size + kLegacyContextTailSize;
    if (total_size > kMaxContextBlobSize || total_size > IREE_HOST_SIZE_MAX) {
      Fail("context blob size exceeds supported limit", out_error);
      goto fail;
    }
    iree_status_t status = iree_allocator_malloc(
        allocator, static_cast<iree_host_size_t>(total_size),
        reinterpret_cast<void**>(&blob));
    if (!FailStatus(status, "context blob allocation failed", out_error)) {
      goto fail;
    }

    std::memcpy(blob, xclbin + 0x1A0, 16);
    WriteU64(blob, 0x48, kCommandApertureBase);
    WriteU64(blob, 0x50, 0x48);
    WriteU64(blob, 0x58, total_size - 0x80);
    WriteU64(blob, 0x60, process_id);
    WriteU64(blob, 0x80, 1);
    WriteU64(blob, 0xC8, kContextCommandBoSize);
    WriteU64(blob, 0xD0, xclbin_size);
    WriteU64(blob, 0xD8, total_size - 0x138);
    WriteU64(blob, 0xE0, total_size - 0xE8);
    std::memcpy(blob + kAxlfBaseOffset, xclbin, xclbin_size);

    size_t tail = kAxlfBaseOffset + xclbin_size;
    if (!WriteCString(blob, tail + 0x00, 64, info.kernel_name, out_error)) {
      goto fail;
    }
    blob[tail + 0x3F] = '0';
    WriteU64(blob, tail + 0x40, 0x10000);
    WriteU64(blob, tail + 0x48, 8);
    WriteU64(blob, tail + 0x58, 0x901);
    WriteU32(blob, tail + 0x360, 0x800);
    WriteU32(blob, tail + 0x364, 1);
    WriteU32(blob, tail + 0x368, info.column_width);
    WriteU32(blob, tail + 0x36C, info.start_column);

    *out_blob = iree_make_byte_span(blob, total_size);
  }
  if (out_info) {
    *out_info = info;
    info = ContextBlobInfo();
  } else {
    ContextBlobInfoDeinitialize(&info);
  }
  return true;

fail:
  iree_allocator_free(allocator, blob);
  ContextBlobInfoDeinitialize(&info);
  return false;
}

bool BuildCompactContextPrivateDataFromXclbin(
    const uint8_t* xclbin, size_t xclbin_size, uint32_t process_id,
    const Buffer& context_private_buffer, iree_allocator_t allocator,
    iree_byte_span_t* out_blob, ContextBlobInfo* out_info, Error* out_error) {
  if (!xclbin || !out_blob || iree_allocator_is_null(allocator) ||
      !context_private_buffer.allocation ||
      context_private_buffer.size != kContextCommandBoSize ||
      !context_private_buffer.cpu_ptr) {
    return Fail("invalid compact context arguments", out_error);
  }
  *out_blob = iree_byte_span_empty();
  if (xclbin_size < 0x1B0) return Fail("xclbin is too small", out_error);

  ContextBlobInfo info;
  if (!ParseContextBlobInfo(xclbin, xclbin_size, allocator, &info,
                            out_error)) {
    return false;
  }

  uint8_t* blob = nullptr;
  iree_status_t status = iree_allocator_malloc(
      allocator, kCompactContextPrivateDataSize,
      reinterpret_cast<void**>(&blob));
  if (!FailStatus(status, "compact context allocation failed", out_error)) {
    ContextBlobInfoDeinitialize(&info);
    return false;
  }
  std::memset(blob, 0, kCompactContextPrivateDataSize);
  std::memcpy(blob, xclbin + 0x1A0, 16);
  WriteU64(blob, 0x48, kCommandApertureBase);
  WriteU32(blob, 0x50, process_id);
  WriteU32(blob, 0x54, info.column_width);
  WriteU32(blob, 0x5C, kCompactContextInstructionOffset);
  WriteU32(blob, 0x68, context_private_buffer.allocation);
  WriteU32(blob, 0x74,
           static_cast<uint32_t>(context_private_buffer.size));
  WriteU64(blob, 0x78,
           reinterpret_cast<uintptr_t>(context_private_buffer.cpu_ptr));

  *out_blob =
      iree_make_byte_span(blob, kCompactContextPrivateDataSize);
  if (out_info) {
    *out_info = info;
  } else {
    ContextBlobInfoDeinitialize(&info);
  }
  return true;
}

bool BuildLegacyV0ContextPrivateDataFromXclbin(
    const uint8_t* xclbin, size_t xclbin_size, uint32_t process_id,
    iree_allocator_t allocator, iree_byte_span_t* out_blob,
    ContextBlobInfo* out_info, Error* out_error) {
  if (!xclbin || !out_blob || iree_allocator_is_null(allocator)) {
    return Fail("invalid v0 context arguments", out_error);
  }
  *out_blob = iree_byte_span_empty();
  if (out_info) *out_info = ContextBlobInfo();
  if (xclbin_size < 0x1B0) return Fail("xclbin is too small", out_error);

  ContextBlobInfo info;
  if (!ParseContextBlobInfo(xclbin, xclbin_size, allocator, &info,
                            out_error)) {
    return false;
  }
  const bool is_mlir_aie =
      std::strcmp(info.kernel_name, "MLIR_AIE") == 0 ||
      std::strcmp(info.pdi_name, "MLIR_AIE") == 0;
  const size_t tail_size =
      is_mlir_aie ? kLegacyV0MlirAieContextTailSize : kLegacyV0ContextTailSize;
  if (xclbin_size > std::numeric_limits<size_t>::max() -
                        kLegacyV0AxlfBaseOffset - tail_size) {
    ContextBlobInfoDeinitialize(&info);
    return Fail("v0 context blob size overflows size_t", out_error);
  }
  size_t total_size =
      kLegacyV0AxlfBaseOffset + xclbin_size + tail_size;
  if (total_size > kMaxContextBlobSize || total_size > IREE_HOST_SIZE_MAX) {
    ContextBlobInfoDeinitialize(&info);
    return Fail("v0 context blob size exceeds supported limit", out_error);
  }

  uint8_t* blob = nullptr;
  iree_status_t status =
      iree_allocator_malloc(allocator, static_cast<iree_host_size_t>(total_size),
                            reinterpret_cast<void**>(&blob));
  if (!FailStatus(status, "v0 context blob allocation failed", out_error)) {
    ContextBlobInfoDeinitialize(&info);
    return false;
  }
  std::memset(blob, 0, total_size);
  std::memcpy(blob, xclbin + 0x1A0, 16);
  WriteU64(blob, 0x38, kCommandApertureBase);
  WriteU64(blob, 0x40, 0x48);
  WriteU64(blob, 0x48, total_size - 0x58);
  WriteU64(blob, 0x50, process_id);
  WriteU64(blob, 0x58, 1);
  WriteU64(blob, 0xA0, kContextCommandBoSize);
  WriteU64(blob, 0xA8, xclbin_size);
  WriteU64(blob, 0xB0, total_size - 0x110);
  WriteU64(blob, 0xB8, total_size - 0xC0);
  std::memcpy(blob + kLegacyV0AxlfBaseOffset, xclbin, xclbin_size);

  size_t tail = kLegacyV0AxlfBaseOffset + xclbin_size;
  if (!WriteCString(blob, tail + 0x00, 64, info.kernel_name, out_error)) {
    iree_allocator_free(allocator, blob);
    ContextBlobInfoDeinitialize(&info);
    return false;
  }
  blob[tail + 0x3F] = '0';
  if (is_mlir_aie) {
    WriteU64(blob, tail + 0x40, 0x10000);
    WriteU64(blob, tail + 0x48, 8);
    WriteU32(blob, tail + 0x58, 0x901);
    WriteU32(blob, tail + 0x360, 0x800);
    WriteU32(blob, tail + 0x364, 1);
    WriteU32(blob, tail + 0x368, info.column_width);
  } else {
    WriteU64(blob, tail + 0x40, kContextCommandBoSize);
    WriteU64(blob, tail + 0x48, info.column_width);
    if (!WriteCString(blob, tail + 0x1C0, 64, info.pdi_name, out_error)) {
      iree_allocator_free(allocator, blob);
      ContextBlobInfoDeinitialize(&info);
      return false;
    }
    blob[tail + 0x1FF] = '0';
    WriteU64(blob, tail + 0x200, 0x10000);
    WriteU64(blob, tail + 0x208, 8);
    WriteU32(blob, tail + 0x218, 0x100);
  }

  *out_blob = iree_make_byte_span(blob, total_size);
  if (out_info) {
    *out_info = info;
    info = ContextBlobInfo();
  }
  ContextBlobInfoDeinitialize(&info);
  return true;
}

bool BuildLegacyV2ContextPrivateDataFromXclbin(
    const uint8_t* xclbin, size_t xclbin_size, uint32_t process_id,
    iree_allocator_t allocator, iree_byte_span_t* out_blob,
    ContextBlobInfo* out_info, Error* out_error) {
  if (!xclbin || !out_blob || iree_allocator_is_null(allocator)) {
    return Fail("invalid legacy_v2 context arguments", out_error);
  }
  *out_blob = iree_byte_span_empty();
  if (out_info) *out_info = ContextBlobInfo();
  if (xclbin_size < 0x1B0) return Fail("xclbin is too small", out_error);

  ContextBlobInfo info;
  if (!ParseContextBlobInfo(xclbin, xclbin_size, allocator, &info,
                            out_error)) {
    return false;
  }
  const bool is_mlir_aie =
      std::strcmp(info.kernel_name, "MLIR_AIE") == 0 ||
      std::strcmp(info.pdi_name, "MLIR_AIE") == 0;
  const size_t tail_size = is_mlir_aie
                               ? kLegacyV2MlirAieContextTailSize
                               : kLegacyV2ContextTailSize;
  if (xclbin_size > std::numeric_limits<size_t>::max() -
                        kLegacyV2AxlfBaseOffset - tail_size) {
    ContextBlobInfoDeinitialize(&info);
    return Fail("legacy_v2 context blob size overflows size_t", out_error);
  }
  size_t total_size = kLegacyV2AxlfBaseOffset + xclbin_size + tail_size;
  if (total_size > kMaxContextBlobSize || total_size > IREE_HOST_SIZE_MAX) {
    ContextBlobInfoDeinitialize(&info);
    return Fail("legacy_v2 context blob size exceeds supported limit",
                out_error);
  }

  uint8_t* blob = nullptr;
  iree_status_t status =
      iree_allocator_malloc(allocator, static_cast<iree_host_size_t>(total_size),
                            reinterpret_cast<void**>(&blob));
  if (!FailStatus(status, "legacy_v2 context blob allocation failed",
                  out_error)) {
    ContextBlobInfoDeinitialize(&info);
    return false;
  }
  std::memset(blob, 0, total_size);
  std::memcpy(blob, xclbin + 0x1A0, 16);
  WriteU64(blob, 0x40, kCommandApertureBase);
  WriteU64(blob, 0x48, 0x48);
  WriteU64(blob, 0x50, total_size - 0x78);
  WriteU64(blob, 0x58, process_id);
  WriteU64(blob, 0x78, 1);
  WriteU64(blob, 0xC0, kContextCommandBoSize);
  WriteU64(blob, 0xC8, xclbin_size);
  WriteU64(blob, 0xD0, total_size - 0x130);
  WriteU64(blob, 0xD8, total_size - 0xE0);
  std::memcpy(blob + kLegacyV2AxlfBaseOffset, xclbin, xclbin_size);

  size_t tail = kLegacyV2AxlfBaseOffset + xclbin_size;
  if (!WriteCString(blob, tail + 0x00, 64, info.kernel_name, out_error)) {
    iree_allocator_free(allocator, blob);
    ContextBlobInfoDeinitialize(&info);
    return false;
  }
  blob[tail + 0x3F] = '0';
  if (is_mlir_aie) {
    WriteU64(blob, tail + 0x40, 0x10000);
    WriteU64(blob, tail + 0x48, 8);
    WriteU32(blob, tail + 0x58, 0x901);
    WriteU32(blob, tail + 0x360, 0x800);
    WriteU32(blob, tail + 0x364, 1);
    WriteU32(blob, tail + 0x368, info.column_width);
  } else {
    WriteU64(blob, tail + 0x40, 0x10000);
    WriteU64(blob, tail + 0x48, 9);
    WriteU32(blob, tail + 0x3B8, 0x800);
    WriteU32(blob, tail + 0x3BC, 1);
    WriteU32(blob, tail + 0x3C0, info.column_width);
    WriteU32(blob, tail + 0x3C4, info.start_column);
  }

  *out_blob = iree_make_byte_span(blob, total_size);
  if (out_info) {
    *out_info = info;
    info = ContextBlobInfo();
  }
  ContextBlobInfoDeinitialize(&info);
  return true;
}

bool BuildContextPrivateDataForDevice(
    const KmtApi& api, const Device& device, const uint8_t* xclbin,
    size_t xclbin_size, uint32_t process_id, iree_allocator_t allocator,
    iree_byte_span_t* out_blob, ContextBlobInfo* out_info,
    Buffer* out_context_private_buffer, Error* out_error) {
  if (!out_context_private_buffer) {
    return Fail("invalid context-private buffer output", out_error);
  }
  *out_context_private_buffer = {};
  if (device.mcdm_abi == McdmAbi::legacy_v0) {
    return BuildLegacyV0ContextPrivateDataFromXclbin(
        xclbin, xclbin_size, process_id, allocator, out_blob, out_info,
        out_error);
  }
  if (device.mcdm_abi == McdmAbi::legacy_v2) {
    return BuildLegacyV2ContextPrivateDataFromXclbin(
        xclbin, xclbin_size, process_id, allocator, out_blob, out_info,
        out_error);
  }
  if (device.mcdm_abi == McdmAbi::legacy) {
    return BuildContextPrivateDataFromXclbin(
        xclbin, xclbin_size, process_id, allocator, out_blob, out_info,
        out_error);
  }

  if (!CreateBuffer(api, device, BufferKind::context_private,
                    kContextCommandBoSize, out_context_private_buffer,
                    out_error)) {
    return false;
  }
  if (BuildCompactContextPrivateDataFromXclbin(
          xclbin, xclbin_size, process_id, *out_context_private_buffer,
          allocator, out_blob, out_info, out_error)) {
    return true;
  }
  DestroyBuffer(api, device, out_context_private_buffer);
  return false;
}

}  // namespace iree::hal::amdxdna::mcdm
