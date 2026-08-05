// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_parser.h"

#include "iree/schemas/parameter_archive.h"

typedef struct iree_io_irpa_resolved_range_t {
  // Absolute byte offset of the range in the archive file.
  iree_io_physical_offset_t offset;
  // Host-accessible contents of the range in the mapped archive file.
  iree_const_byte_span_t contents;
} iree_io_irpa_resolved_range_t;

// Resolves a header-relative range into the mapped archive file.
static iree_status_t iree_io_resolve_irpa_file_range(
    iree_const_byte_span_t file_contents, iree_io_physical_offset_t base_offset,
    iree_io_parameter_archive_range_t range,
    iree_io_irpa_resolved_range_t* out_range) {
  out_range->offset = 0;
  out_range->contents = iree_const_byte_span_empty();
  if (range.length == 0) return iree_ok_status();

  iree_io_physical_offset_t absolute_offset = 0;
  if (!iree_checked_add_u64(base_offset, range.offset, &absolute_offset) ||
      absolute_offset > file_contents.data_length ||
      range.length >
          file_contents.data_length - (iree_host_size_t)absolute_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "file range out of bounds (base=%" PRIu64 ", offset=%" PRIu64
        ", length=%" PRIu64 ", file_size=%" PRIhsz ")",
        base_offset, range.offset, range.length, file_contents.data_length);
  }

  out_range->offset = absolute_offset;
  out_range->contents = iree_make_const_byte_span(
      file_contents.data + (iree_host_size_t)absolute_offset,
      (iree_host_size_t)range.length);
  return iree_ok_status();
}

// Resolves a range relative to an already resolved parent range.
static iree_status_t iree_io_resolve_irpa_subrange(
    const iree_io_irpa_resolved_range_t* parent_range,
    iree_io_parameter_archive_range_t range,
    iree_io_irpa_resolved_range_t* out_range) {
  out_range->offset = 0;
  out_range->contents = iree_const_byte_span_empty();
  if (range.length == 0) return iree_ok_status();

  if (range.offset > parent_range->contents.data_length ||
      range.length >
          parent_range->contents.data_length - (iree_host_size_t)range.offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "segment range out of bounds (offset=%" PRIu64
                            ", length=%" PRIu64 ", segment_size=%" PRIhsz ")",
                            range.offset, range.length,
                            parent_range->contents.data_length);
  }

  iree_io_physical_offset_t absolute_offset = 0;
  if (!iree_checked_add_u64(parent_range->offset, range.offset,
                            &absolute_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "absolute segment offset overflow (base=%" PRIu64
                            ", offset=%" PRIu64 ")",
                            parent_range->offset, range.offset);
  }

  out_range->offset = absolute_offset;
  out_range->contents = iree_make_const_byte_span(
      parent_range->contents.data + (iree_host_size_t)range.offset,
      (iree_host_size_t)range.length);
  return iree_ok_status();
}

static iree_status_t iree_io_resolve_irpa_v0_storage(
    const iree_io_irpa_resolved_range_t* storage_segment,
    iree_io_parameter_archive_storage_ref_t range,
    iree_io_physical_offset_t* out_offset) {
  *out_offset = 0;
  iree_io_irpa_resolved_range_t resolved_range;
  IREE_RETURN_IF_ERROR(
      iree_io_resolve_irpa_subrange(storage_segment, range, &resolved_range));
  *out_offset = resolved_range.offset;
  return iree_ok_status();
}

static iree_status_t iree_io_parse_irpa_v0_splat_entry(
    const iree_io_parameter_archive_splat_entry_t* splat_entry,
    iree_string_view_t name, iree_const_byte_span_t metadata,
    iree_io_parameter_index_t* index) {
  if (splat_entry->header.entry_size < sizeof(*splat_entry)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "splat entry length underflow");
  }
  if (splat_entry->pattern_length == 0 ||
      splat_entry->pattern_length >
          IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH ||
      !iree_is_power_of_two_uint64(splat_entry->pattern_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "splat pattern length %u invalid; must be 1, 2, 4, 8, or 16 bytes",
        splat_entry->pattern_length);
  }
  if (splat_entry->length % splat_entry->pattern_length != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat data length %" PRIu64
                            " is not evenly divisible by pattern length %u",
                            splat_entry->length, splat_entry->pattern_length);
  }
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = splat_entry->length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage =
          {
              .splat =
                  {
                      .pattern = {0},  // set below
                      .pattern_length = splat_entry->pattern_length,
                  },
          },
  };
  memcpy(entry.storage.splat.pattern, splat_entry->pattern,
         entry.storage.splat.pattern_length);
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t iree_io_parse_irpa_v0_data_entry(
    iree_io_file_handle_t* file_handle,
    const iree_io_irpa_resolved_range_t* storage_segment,
    const iree_io_parameter_archive_data_entry_t* data_entry,
    iree_string_view_t name, iree_const_byte_span_t metadata,
    iree_io_parameter_index_t* index) {
  if (data_entry->header.entry_size < sizeof(*data_entry)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "data entry length underflow");
  }
  const iree_io_physical_size_t minimum_alignment =
      data_entry->header.minimum_alignment;
  if (minimum_alignment != 0 &&
      !iree_is_power_of_two_uint64(minimum_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "data entry minimum alignment %" PRIu64
                            " is not zero or a power of two",
                            minimum_alignment);
  }
  iree_io_physical_offset_t storage_offset = 0;
  IREE_RETURN_IF_ERROR(iree_io_resolve_irpa_v0_storage(
      storage_segment, data_entry->storage, &storage_offset));
  if (minimum_alignment != 0 && storage_offset % minimum_alignment != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "data entry storage offset %" PRIu64
                            " is not aligned to %" PRIu64 " bytes",
                            storage_offset, minimum_alignment);
  }
  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_entry->storage.length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
      .storage =
          {
              .file =
                  {
                      .handle = file_handle,
                      .offset = storage_offset,
                      .minimum_alignment = minimum_alignment,
                  },
          },
  };
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t iree_io_parse_irpa_v0_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_physical_offset_t base_offset,
    const iree_io_parameter_archive_header_prefix_t* header_prefix,
    iree_io_parameter_index_t* index) {
  // Get the full header struct.
  if (header_prefix->version_minor > 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "IRPA version %u.%u not supported (major supported "
        "but minor is newer than the runtime trying to parse it)",
        header_prefix->version_major, header_prefix->version_minor);
  }
  if (header_prefix->header_size !=
      sizeof(iree_io_parameter_archive_header_v0_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IRPA v0 header expected to be exactly %" PRIhsz
                            " bytes but was reported as %" PRIu64,
                            sizeof(iree_io_parameter_archive_header_v0_t),
                            header_prefix->header_size);
  }
  const iree_io_parameter_archive_header_v0_t* header =
      (const iree_io_parameter_archive_header_v0_t*)header_prefix;

  // Resolve the base data ranges once so all entry references are checked
  // against spans that have already been proven to reside in the mapped file.
  iree_io_irpa_resolved_range_t entry_segment;
  IREE_RETURN_IF_ERROR(
      iree_io_resolve_irpa_file_range(file_contents, base_offset,
                                      header->entry_segment, &entry_segment),
      "resolving entry table");
  iree_io_irpa_resolved_range_t metadata_segment;
  IREE_RETURN_IF_ERROR(iree_io_resolve_irpa_file_range(
                           file_contents, base_offset, header->metadata_segment,
                           &metadata_segment),
                       "resolving metadata segment");
  iree_io_irpa_resolved_range_t storage_segment;
  IREE_RETURN_IF_ERROR(iree_io_resolve_irpa_file_range(
                           file_contents, base_offset, header->storage_segment,
                           &storage_segment),
                       "resolving storage segment");

  // Walk the entry table, which has variable-length entries.
  iree_host_size_t entry_offset = 0;
  for (iree_io_physical_size_t i = 0; i < header->entry_count; ++i) {
    const iree_host_size_t entry_size_remaining =
        entry_segment.contents.data_length - entry_offset;

    // Ensure there's enough space in the table for the base entry header.
    if (entry_size_remaining <
        sizeof(iree_io_parameter_archive_entry_header_t)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "entry table truncated; insufficient bytes for base entry header");
    }

    // Ensure there's enough space for the declared entry size (if any larger).
    const iree_io_parameter_archive_entry_header_t* entry_header =
        (const iree_io_parameter_archive_entry_header_t*)(entry_segment.contents
                                                              .data +
                                                          entry_offset);
    if (entry_header->entry_size < sizeof(*entry_header) ||
        entry_size_remaining < entry_header->entry_size) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "entry table truncated; insufficient bytes for sized header");
    }

    // Resolve entry metadata from the archive metadata segment.
    iree_io_irpa_resolved_range_t name_range;
    IREE_RETURN_IF_ERROR(
        iree_io_resolve_irpa_subrange(&metadata_segment, entry_header->name,
                                      &name_range),
        "resolving entry name");
    iree_string_view_t name =
        name_range.contents.data_length == 0
            ? iree_string_view_empty()
            : iree_make_string_view((const char*)name_range.contents.data,
                                    name_range.contents.data_length);
    iree_io_irpa_resolved_range_t metadata_range;
    IREE_RETURN_IF_ERROR(
        iree_io_resolve_irpa_subrange(&metadata_segment, entry_header->metadata,
                                      &metadata_range),
        "resolving entry metadata");

    // Handle each entry type.
    switch (entry_header->type) {
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SKIP:
        break;
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT: {
        IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_splat_entry(
            (const iree_io_parameter_archive_splat_entry_t*)entry_header, name,
            metadata_range.contents, index));
        break;
      }
      case IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA: {
        IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_data_entry(
            file_handle, &storage_segment,
            (const iree_io_parameter_archive_data_entry_t*)entry_header, name,
            metadata_range.contents, index));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "parser does not support entry type %d",
                                (int)entry_header->type);
    }

    // Padding is required only between entries; the final entry may end at the
    // end of the segment without trailing alignment bytes.
    if (i + 1 < header->entry_count) {
      iree_io_physical_size_t aligned_entry_size = 0;
      if (!iree_checked_align_u64(entry_header->entry_size,
                                  IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT,
                                  &aligned_entry_size) ||
          aligned_entry_size > entry_size_remaining) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "entry table truncated; insufficient bytes for entry alignment");
      }
      entry_offset += (iree_host_size_t)aligned_entry_size;
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_io_parse_irpa_index_from_memory(
    iree_io_file_handle_t* file_handle, iree_const_byte_span_t file_contents,
    iree_io_physical_offset_t base_offset, iree_io_parameter_index_t* index) {
  while (true) {
    // Check the basic header information before forming a pointer into the
    // mapped file. Once this succeeds base_offset is host-addressable.
    if (base_offset > file_contents.data_length ||
        sizeof(iree_io_parameter_archive_header_prefix_t) >
            file_contents.data_length - (iree_host_size_t)base_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "not enough bytes for a valid IRPA header at "
                              "offset %" PRIu64 "; file may be truncated",
                              base_offset);
    }
    const iree_io_parameter_archive_header_prefix_t* header_prefix =
        (const iree_io_parameter_archive_header_prefix_t*)(file_contents.data +
                                                           (iree_host_size_t)
                                                               base_offset);
    if (header_prefix->magic != IREE_IO_PARAMETER_ARCHIVE_MAGIC) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "IRPA file magic missing or invalid %08X; expected %08X",
          header_prefix->magic, IREE_IO_PARAMETER_ARCHIVE_MAGIC);
    }
    if (header_prefix->header_size >
        file_contents.data_length - (iree_host_size_t)base_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "file buffer underrun parsing header of reported size %" PRIu64
          " at offset %" PRIu64 " (only %" PRIhsz " bytes available)",
          header_prefix->header_size, base_offset,
          file_contents.data_length - (iree_host_size_t)base_offset);
    }

    // Route major versions to their parsers, allowing us to change everything
    // but the prefix without breaking compatibility.
    switch (header_prefix->version_major) {
      case 0: {
        IREE_RETURN_IF_ERROR(iree_io_parse_irpa_v0_index_from_memory(
            file_handle, file_contents, base_offset, header_prefix, index));
        break;
      }
      default: {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "IRPA major version %u.%u not supported by this runtime",
            header_prefix->version_major, header_prefix->version_minor);
      }
    }

    // Advance through the linked list without carrying user-controlled depth on
    // the native stack. A nonzero relative offset makes forward progress; the
    // checked addition rejects wraparound before the next pointer is formed.
    if (header_prefix->next_header_offset == 0) return iree_ok_status();
    if (header_prefix->next_header_offset %
            IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT !=
        0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "linked IRPA header offset %" PRIu64 " is not aligned to %u bytes",
          header_prefix->next_header_offset,
          (unsigned int)IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT);
    }
    iree_io_physical_offset_t next_base_offset = 0;
    if (!iree_checked_add_u64(base_offset, header_prefix->next_header_offset,
                              &next_base_offset)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "linked IRPA header offset overflow (base=%" PRIu64
          ", relative_offset=%" PRIu64 ")",
          base_offset, header_prefix->next_header_offset);
    }
    base_offset = next_base_offset;
  }
}

IREE_API_EXPORT iree_status_t iree_io_parse_irpa_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // The parser requires a host pointer but will only reference the file handle
  // in the index.
  iree_io_file_mapping_t* file_mapping = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_file_map_view(file_handle, IREE_IO_FILE_ACCESS_READ, 0,
                                IREE_HOST_SIZE_MAX,
                                IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS,
                                host_allocator, &file_mapping));

  iree_status_t status = iree_io_parse_irpa_index_from_memory(
      file_handle, iree_io_file_mapping_contents_ro(file_mapping),
      /*base_offset=*/0, index);

  iree_io_file_mapping_release(file_mapping);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
