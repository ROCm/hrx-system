// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_IMAGE_BUILDER_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_IMAGE_BUILDER_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/directory.h"

namespace iree::hal::amd::xdna::testing {

// Options controlling the synthetic ELF envelope built for image tests.
struct ImageBuilderOptions {
  // Target-specific ELF flags written to the image header.
  uint32_t target_flags = IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS;
  // Optional diagnostic section-header directory byte offset.
  uint32_t section_header_offset = 0;
  // Optional diagnostic section-header count.
  uint16_t section_header_count = 0;
  // Minimum complete source byte length.
  iree_host_size_t minimum_source_length = 512;
};

// Returns a read-only program header with the given source range.
iree_hal_amd_xdna_image_program_header_t MakeProgramHeader(uint32_t type,
                                                           uint32_t file_offset,
                                                           uint32_t file_size);

// Builds synthetic canonical ELF images from explicitly placed programs.
//
// The builder is intentionally container-only: it writes the ELF envelope and
// program headers but does not interpret program payloads. Programs without an
// explicit payload receive an ordinal-specific fill byte. Explicit payloads
// must exactly match the program header's file range length.
class ImageBuilder {
 public:
  explicit ImageBuilder(ImageBuilderOptions options = {});

  // Adds |program_header| with an ordinal-specific fill payload.
  ImageBuilder& AddProgram(
      iree_hal_amd_xdna_image_program_header_t program_header);

  // Adds |program_header| with exactly |payload| as its serialized bytes.
  ImageBuilder& AddProgram(
      iree_hal_amd_xdna_image_program_header_t program_header,
      std::vector<uint8_t> payload);

  // Serializes the complete image.
  std::vector<uint8_t> Build() const;

 private:
  struct Program {
    // Program header written to the ELF directory.
    iree_hal_amd_xdna_image_program_header_t header;
    // Optional exact payload bytes.
    std::vector<uint8_t> payload;
    // Whether |payload| replaces the ordinal-specific fill.
    bool has_payload;
  };

  // ELF envelope options captured at construction.
  ImageBuilderOptions options_;
  // Programs in stable ELF directory order.
  std::vector<Program> programs_;
};

}  // namespace iree::hal::amd::xdna::testing

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_IMAGE_BUILDER_H_
