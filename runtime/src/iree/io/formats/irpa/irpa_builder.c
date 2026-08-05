// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_builder.h"

// Fully resolved relative layout of an IRPA archive.
typedef struct iree_io_parameter_archive_layout_t {
  // Serialized entry table range.
  iree_io_parameter_archive_range_t entry_segment;
  // Serialized names and metadata range.
  iree_io_parameter_archive_range_t metadata_segment;
  // File-backed parameter data range.
  iree_io_parameter_archive_range_t storage_segment;
  // Required alignment of the archive header in the target file.
  iree_io_physical_size_t archive_alignment;
  // Final archive size including trailing file-alignment padding.
  iree_io_physical_size_t total_size;
} iree_io_parameter_archive_layout_t;

// Normalizes an optional alignment and verifies that it can be used by the
// checked alignment helpers.
static iree_status_t iree_io_parameter_archive_normalize_alignment(
    iree_io_physical_size_t alignment, iree_io_physical_size_t* out_alignment) {
  *out_alignment = alignment ? alignment : 1;
  if (!iree_is_power_of_two_uint64(*out_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IRPA alignment %" PRIu64 " is not zero or a power of two", alignment);
  }
  return iree_ok_status();
}

// Resolves all archive ranges from the builder segment extents. The builder's
// public add APIs establish this layout before mutating the entry index; this
// function also validates callers that customize exposed builder alignment
// fields directly.
static iree_status_t iree_io_parameter_archive_builder_calculate_layout(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_parameter_archive_layout_t* out_layout) {
  iree_io_physical_size_t file_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_io_parameter_archive_normalize_alignment(
      builder->file_alignment, &file_alignment));
  iree_io_physical_size_t storage_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_io_parameter_archive_normalize_alignment(
      builder->segments.storage_alignment, &storage_alignment));

  iree_io_parameter_archive_layout_t layout;
  memset(&layout, 0, sizeof(layout));
  layout.archive_alignment =
      iree_max(IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT, storage_alignment);
  if (!iree_checked_align_u64(sizeof(iree_io_parameter_archive_header_v0_t),
                              IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT,
                              &layout.entry_segment.offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA entry segment offset overflow");
  }
  layout.entry_segment.length = builder->segments.entry;
  if (!iree_checked_add_u64(layout.entry_segment.offset,
                            layout.entry_segment.length,
                            &layout.metadata_segment.offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA metadata segment offset overflow");
  }
  layout.metadata_segment.length = builder->segments.metadata;
  iree_io_physical_offset_t metadata_end = 0;
  if (!iree_checked_add_u64(layout.metadata_segment.offset,
                            layout.metadata_segment.length, &metadata_end) ||
      !iree_checked_align_u64(metadata_end, storage_alignment,
                              &layout.storage_segment.offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA storage segment offset overflow");
  }
  layout.storage_segment.length = builder->segments.storage;
  iree_io_physical_size_t unaligned_total_size = 0;
  if (!iree_checked_add_u64(layout.storage_segment.offset,
                            layout.storage_segment.length,
                            &unaligned_total_size) ||
      !iree_checked_align_u64(unaligned_total_size, file_alignment,
                              &layout.total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA total size overflow");
  }

  *out_layout = layout;
  return iree_ok_status();
}

// Verifies that a resolved archive layout can be written at an exact file
// offset using the signed stream position type.
static iree_status_t iree_io_parameter_archive_validate_file_range(
    const iree_io_parameter_archive_layout_t* layout,
    iree_io_physical_offset_t file_offset) {
  if (file_offset % layout->archive_alignment != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "IRPA file offset %" PRIu64
                            " is not aligned to %" PRIu64 " bytes",
                            file_offset, layout->archive_alignment);
  }
  iree_io_physical_offset_t file_end = 0;
  if (!iree_checked_add_u64(file_offset, layout->total_size, &file_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA file range overflow (offset=%" PRIu64
                            ", length=%" PRIu64 ")",
                            file_offset, layout->total_size);
  }
  if (file_end > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA file range end %" PRIu64
                            " exceeds stream position range",
                            file_end);
  }
  return iree_ok_status();
}

// Calculates the common entry and metadata extents for one prospective add.
// No builder-owned state is modified until the caller validates the complete
// layout and inserts the entry template into the index.
static iree_status_t iree_io_parameter_archive_builder_prepare_add(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_physical_size_t archive_entry_size, iree_string_view_t name,
    iree_const_byte_span_t metadata,
    iree_io_parameter_archive_builder_t* out_prospective_builder) {
  iree_io_parameter_archive_builder_t prospective_builder = *builder;
  iree_io_physical_size_t entry_offset = 0;
  if (!iree_checked_align_u64(builder->segments.entry,
                              IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT,
                              &entry_offset) ||
      !iree_checked_add_u64(entry_offset, archive_entry_size,
                            &prospective_builder.segments.entry)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA entry segment size overflow");
  }
  iree_io_physical_size_t metadata_size = 0;
  if (!iree_checked_add_u64(name.size, metadata.data_length, &metadata_size) ||
      !iree_checked_add_u64(builder->segments.metadata, metadata_size,
                            &prospective_builder.segments.metadata)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA metadata segment size overflow");
  }
  *out_prospective_builder = prospective_builder;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_initialize(
    iree_allocator_t host_allocator,
    iree_io_parameter_archive_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
  out_builder->file_alignment =
      IREE_IO_PARAMETER_ARCHIVE_DEFAULT_FILE_ALIGNMENT;
  return iree_io_parameter_index_create(host_allocator, &out_builder->index);
}

IREE_API_EXPORT void iree_io_parameter_archive_builder_deinitialize(
    iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_io_parameter_index_release(builder->index);
  memset(builder, 0, sizeof(*builder));
}

IREE_API_EXPORT bool iree_io_parameter_archive_builder_is_empty(
    const iree_io_parameter_archive_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  return iree_io_parameter_index_count(builder->index) == 0;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_header_size(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_physical_size_t* out_header_size) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_header_size);
  *out_header_size = 0;
  iree_io_parameter_archive_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_archive_builder_calculate_layout(builder, &layout));
  *out_header_size = layout.storage_segment.offset;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_total_size(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_physical_size_t* out_total_size) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_total_size);
  *out_total_size = 0;
  iree_io_parameter_archive_layout_t layout;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_archive_builder_calculate_layout(builder, &layout));
  *out_total_size = layout.total_size;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_write(
    const iree_io_parameter_archive_builder_t* builder,
    iree_io_file_handle_t* file_handle, iree_io_physical_offset_t file_offset,
    iree_io_stream_t* stream, iree_io_parameter_index_t* target_index) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(target_index);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve the relative segment ranges once for both the serialized header
  // and target index entries.
  iree_io_parameter_archive_layout_t layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_builder_calculate_layout(builder, &layout));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_validate_file_range(&layout, file_offset));

  // The complete file range and relative storage segment have been checked,
  // so this absolute segment base is representable for every builder entry.
  const iree_io_physical_offset_t storage_base_offset =
      file_offset + layout.storage_segment.offset;

  // Write the archive header referencing the other segments in the file.
  iree_io_parameter_archive_header_v0_t header = {
      .prefix =
          {
              .magic = IREE_IO_PARAMETER_ARCHIVE_MAGIC,
              .version_major = 0,
              .version_minor = 0,
              .header_size = sizeof(header),
              .next_header_offset = 0,
              .flags = 0,
          },
      .entry_count = iree_io_parameter_index_count(builder->index),
      .entry_segment = layout.entry_segment,
      .metadata_segment = layout.metadata_segment,
      .storage_segment = layout.storage_segment,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_stream_write(stream, sizeof(header), &header));

  // Write entry table following the header.
  // This references ranges in the metadata and storage segment but to preserve
  // forward-only writes we populate those after writing the table.
  const uint8_t zero = 0;
  iree_io_physical_offset_t entry_offset = sizeof(header);
  iree_io_physical_offset_t metadata_offset = 0;
  for (iree_host_size_t i = 0;
       i < iree_io_parameter_index_count(builder->index); ++i) {
    // Explicitly zero alignment padding so writing into reused storage produces
    // the deterministic bytes required by the archive format.
    const iree_io_physical_offset_t aligned_entry_offset = iree_align_uint64(
        entry_offset, IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_io_stream_fill(
            stream, (iree_io_stream_pos_t)(aligned_entry_offset - entry_offset),
            &zero, sizeof(zero)));
    entry_offset = aligned_entry_offset;

    // Query the source entry template.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_get(builder->index, i, &source_entry));

    // Reserve space in the metadata segment.
    const iree_io_parameter_archive_metadata_ref_t name_ref = {
        .offset = metadata_offset,
        .length = source_entry->key.size,
    };
    metadata_offset += name_ref.length;
    const iree_io_parameter_archive_metadata_ref_t metadata_ref = {
        .offset = source_entry->metadata.data_length ? metadata_offset : 0,
        .length = source_entry->metadata.data_length,
    };
    metadata_offset += metadata_ref.length;

    // Produce the target archive entry based on the template.
    iree_io_parameter_index_entry_t target_entry = {
        .key = source_entry->key,
        .metadata = source_entry->metadata,
        .length = source_entry->length,
        .type = source_entry->type,
        .storage = source_entry->storage,
    };
    switch (source_entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
        iree_io_parameter_archive_splat_entry_t splat_entry = {
            .header =
                {
                    .entry_size = sizeof(splat_entry),
                    .type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT,
                    .flags = 0,
                    .name = name_ref,
                    .metadata = metadata_ref,
                    .minimum_alignment = 0,
                },
            .length = target_entry.length,
            .pattern_length = target_entry.storage.splat.pattern_length,
        };
        memcpy(splat_entry.pattern, target_entry.storage.splat.pattern,
               sizeof(splat_entry.pattern));
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            iree_io_stream_write(stream, sizeof(splat_entry), &splat_entry));
        entry_offset += sizeof(splat_entry);
        break;
      }
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
        iree_io_parameter_archive_data_entry_t data_entry = {
            .header =
                {
                    .entry_size = sizeof(data_entry),
                    .type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA,
                    .flags = 0,
                    .name = name_ref,
                    .metadata = metadata_ref,
                    .minimum_alignment =
                        target_entry.storage.file.minimum_alignment,
                },
            .storage =
                {
                    .offset = target_entry.storage.file.offset,
                    .length = target_entry.length,
                },
        };
        target_entry.storage.file.handle = file_handle;
        target_entry.storage.file.offset += storage_base_offset;
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, iree_io_stream_write(stream, sizeof(data_entry), &data_entry));
        entry_offset += sizeof(data_entry);
        break;
      }
      default: {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unhandled entry type %d",
                                (int)source_entry->type);
      }
    }

    // Add the entry to the target_index referencing the location in the file
    // reserved for the entry storage.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_add(target_index, &target_entry));
  }

  // Write out the metadata table.
  for (iree_host_size_t i = 0;
       i < iree_io_parameter_index_count(builder->index); ++i) {
    // Query the source entry template.
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_parameter_index_get(builder->index, i, &source_entry));

    // Write header metadata.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_write(stream, source_entry->key.size,
                                 source_entry->key.data));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_io_stream_write(stream, source_entry->metadata.data_length,
                                 source_entry->metadata.data));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_splat_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, const void* pattern,
    uint8_t pattern_length, iree_io_physical_size_t data_length) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(pattern);
  if (pattern_length == 0 ||
      pattern_length > IREE_IO_PARAMETER_MAX_SPLAT_PATTERN_LENGTH ||
      !iree_is_power_of_two_uint64(pattern_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "splat pattern length %u invalid; must be 1, 2, 4, 8, or 16 bytes",
        pattern_length);
  }
  if (data_length % pattern_length != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "splat data length %" PRIu64
                            " is not evenly divisible by pattern length %u",
                            data_length, pattern_length);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);

  iree_io_parameter_archive_builder_t prospective_builder;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_builder_prepare_add(
              builder, sizeof(iree_io_parameter_archive_splat_entry_t), name,
              metadata, &prospective_builder));
  iree_io_parameter_archive_layout_t prospective_layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_builder_calculate_layout(
              &prospective_builder, &prospective_layout));

  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage =
          {
              .splat =
                  {
                      .pattern = {0},  // set below
                      .pattern_length = pattern_length,
                  },
          },
  };
  memcpy(entry.storage.splat.pattern, pattern, pattern_length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_add(builder->index, &entry));
  builder->segments = prospective_builder.segments;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_archive_builder_add_data_entry(
    iree_io_parameter_archive_builder_t* builder, iree_string_view_t name,
    iree_const_byte_span_t metadata, iree_io_physical_size_t minimum_alignment,
    iree_io_physical_size_t data_length) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_io_physical_size_t normalized_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_io_parameter_archive_normalize_alignment(
      minimum_alignment, &normalized_alignment));
  iree_io_physical_size_t current_storage_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_io_parameter_archive_normalize_alignment(
      builder->segments.storage_alignment, &current_storage_alignment));
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);

  iree_io_parameter_archive_builder_t prospective_builder;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_builder_prepare_add(
              builder, sizeof(iree_io_parameter_archive_data_entry_t), name,
              metadata, &prospective_builder));
  iree_io_physical_offset_t storage_offset = 0;
  if (!iree_checked_align_u64(builder->segments.storage, normalized_alignment,
                              &storage_offset) ||
      !iree_checked_add_u64(storage_offset, data_length,
                            &prospective_builder.segments.storage)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "IRPA storage segment size overflow");
  }
  prospective_builder.segments.storage_alignment =
      iree_max(current_storage_alignment, normalized_alignment);
  iree_io_parameter_archive_layout_t prospective_layout;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_archive_builder_calculate_layout(
              &prospective_builder, &prospective_layout));

  iree_io_parameter_index_entry_t entry = {
      .key = name,
      .metadata = metadata,
      .length = data_length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
      .storage =
          {
              .file =
                  {
                      .handle = NULL,  // set on commit
                      .offset = storage_offset,
                      .minimum_alignment = minimum_alignment,
                  },
          },
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_add(builder->index, &entry));
  builder->segments = prospective_builder.segments;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_build_parameter_archive(
    iree_io_parameter_index_t* source_index,
    iree_io_parameter_index_t* target_index,
    iree_io_parameter_archive_file_open_callback_t target_file_open,
    iree_io_physical_offset_t target_file_offset,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(source_index);
  IREE_ASSERT_ARGUMENT(target_index);
  IREE_ASSERT_ARGUMENT(target_file_open.fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_archive_builder_t builder;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_io_parameter_archive_builder_initialize(host_allocator, &builder));

  // Declare a parameter for each entry in the index.
  // This lets us calculate the size we require to store the entry metadata and
  // its contents (if any). No data is accessed yet.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_io_parameter_index_count(source_index);
       ++i) {
    const iree_io_parameter_index_entry_t* source_entry = NULL;
    status = iree_io_parameter_index_get(source_index, i, &source_entry);
    if (!iree_status_is_ok(status)) break;
    switch (source_entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
        status = iree_io_parameter_archive_builder_add_splat_entry(
            &builder, source_entry->key, source_entry->metadata,
            source_entry->storage.splat.pattern,
            source_entry->storage.splat.pattern_length, source_entry->length);
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        status = iree_io_parameter_archive_builder_add_data_entry(
            &builder, source_entry->key, source_entry->metadata,
            iree_max(IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT,
                     source_entry->storage.file.minimum_alignment),
            source_entry->length);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unhandled index entry storage type %d",
                                  (int)source_entry->type);
        break;
    }
    if (!iree_status_is_ok(status)) break;
  }

  // Resolve the complete archive range before opening or mutating the target.
  iree_io_parameter_archive_layout_t layout = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_io_parameter_archive_builder_calculate_layout(&builder, &layout);
  }
  iree_io_physical_offset_t archive_offset = 0;
  if (iree_status_is_ok(status) &&
      !iree_checked_align_u64(target_file_offset, layout.archive_alignment,
                              &archive_offset)) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "IRPA file offset alignment overflow (offset=%" PRIu64
                         ", alignment=%" PRIu64 ")",
                         target_file_offset, layout.archive_alignment);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_parameter_archive_validate_file_range(&layout, archive_offset);
  }
  const iree_io_physical_size_t archive_length =
      iree_status_is_ok(status) ? layout.total_size : 0;
  iree_io_file_handle_t* target_file_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = target_file_open.fn(target_file_open.user_data, archive_offset,
                                 archive_length, &target_file_handle);
  }

  // Wrap the target file in a stream.
  iree_io_stream_t* target_stream = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, target_file_handle,
                            archive_offset, host_allocator, &target_stream);
  }

  // Commit the archive header to the file and produce an index referencing it.
  // This will allow us to know where to copy file contents.
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_archive_builder_write(
        &builder, target_file_handle, archive_offset, target_stream,
        target_index);
  }

  // Copy over parameter entry file contents (if any).
  // This is a slow operation and something we could optimize with lower-level
  // platform primitives.
  if (iree_status_is_ok(status)) {
    // The writer leaves the stream immediately after the metadata segment.
    // Track that archive-relative position independently of the stream's
    // implementation-specific base coordinate.
    iree_io_physical_offset_t archive_stream_offset =
        layout.metadata_segment.offset + layout.metadata_segment.length;
    for (iree_host_size_t i = 0;
         i < iree_io_parameter_index_count(source_index); ++i) {
      const iree_io_parameter_index_entry_t* source_entry = NULL;
      status = iree_io_parameter_index_get(source_index, i, &source_entry);
      if (!iree_status_is_ok(status)) break;
      const iree_io_parameter_index_entry_t* builder_entry = NULL;
      status = iree_io_parameter_index_get(builder.index, i, &builder_entry);
      if (!iree_status_is_ok(status)) break;
      switch (source_entry->type) {
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
          // No work to do.
          break;
        case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
          // Builder storage offsets are monotonically allocated, allowing all
          // target implementations to use the same forward-only positioning.
          const iree_io_physical_offset_t target_offset =
              layout.storage_segment.offset +
              builder_entry->storage.file.offset;
          status = iree_io_stream_seek(
              target_stream, IREE_IO_STREAM_SEEK_FROM_CURRENT,
              (iree_io_stream_pos_t)(target_offset - archive_stream_offset));
          if (!iree_status_is_ok(status)) break;
          status = iree_io_stream_write_file(
              target_stream, source_entry->storage.file.handle,
              source_entry->storage.file.offset,
              (iree_io_stream_pos_t)source_entry->length, host_allocator);
          archive_stream_offset = target_offset + source_entry->length;
          break;
        }
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "unhandled index entry storage type %d",
                                    (int)source_entry->type);
          break;
      }
      if (!iree_status_is_ok(status)) break;
    }
  }

  iree_io_stream_release(target_stream);

  // Flush file contents before returning to the caller (in case they open the
  // file via a different handle).
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_flush(target_file_handle);
  }

  iree_io_file_handle_release(target_file_handle);
  iree_io_parameter_archive_builder_deinitialize(&builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
