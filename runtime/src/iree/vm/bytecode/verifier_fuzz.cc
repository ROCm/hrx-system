// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/vm/bytecode/verifier.h"
#include "iree/vm/bytecode/verifier_testdata.h"

namespace {

constexpr size_t kMaximumInputSize = 1024 * 1024;

void VerifyImage(iree_const_byte_span_t contents) {
  iree_vm_bytecode_module_plan_t plan = {};
  iree_status_t status =
      iree_vm_bytecode_verify_module_structure(contents, &plan);
  if (!iree_status_is_ok(status)) {
    // Rejection is expected for arbitrary input and is fully handled at this
    // terminal fuzz boundary.
    iree_status_free(status);
    return;
  }

  std::vector<uint32_t> block_offsets(
      plan.layout.functions.maximum_block_count);
  status =
      iree_vm_bytecode_verify_module_instructions(&plan, block_offsets.data());
  iree_status_free(status);
}

void VerifyRawInput(const uint8_t* data, size_t size) {
  // The production boundary requires aligned image storage. Copying also
  // makes the fuzz engine input lifetime independent from mapped table views.
  std::vector<uint8_t> aligned_bytes(size);
  if (size != 0) std::memcpy(aligned_bytes.data(), data, size);
  VerifyImage(
      iree_make_const_byte_span(aligned_bytes.data(), aligned_bytes.size()));
}

void VerifyFixtureMutation(const uint8_t* data, size_t size) {
  const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
  std::vector<uint8_t> mutated(files[0].data, files[0].data + files[0].size);

  // A valid comprehensive seed reaches every record verifier. Interpret the
  // input as offset/value triples so coverage-guided mutation can perturb deep
  // fields without first rediscovering the complete module envelope.
  const size_t program_size =
      iree_min(size, static_cast<size_t>(mutated.size() * 3));
  for (size_t i = 0; i + 2 < program_size; i += 3) {
    const size_t offset = (static_cast<size_t>(data[i]) |
                           (static_cast<size_t>(data[i + 1]) << 8)) %
                          mutated.size();
    mutated[offset] ^= data[i + 2];
  }
  VerifyImage(iree_make_const_byte_span(mutated.data(), mutated.size()));
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > kMaximumInputSize) return 0;
  VerifyRawInput(data, size);
  VerifyFixtureMutation(data, size);
  return 0;
}
