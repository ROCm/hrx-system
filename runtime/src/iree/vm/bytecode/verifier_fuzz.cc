// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/verifier_testdata.h"
#include "iree/vm/environment.h"

namespace {

constexpr size_t kMaximumInputSize = 1024 * 1024;

iree_vm_bytecode_module_fixture_t g_fixture;
iree_vm_environment_t* g_environment = nullptr;

void ExerciseModuleLifecycle(iree_const_byte_span_t contents) {
  iree_vm_module_t* module = nullptr;
  iree_status_t status = iree_vm_bytecode_module_create(
      g_environment, IREE_SV("fuzz"), {contents, iree_allocator_null()},
      iree_allocator_system(), &module);
  // Rejection is expected for arbitrary input and is fully handled at this
  // terminal fuzz boundary.
  iree_status_free(status);
  iree_vm_module_release(module);
}

void ExerciseRawInput(const uint8_t* data, size_t size) {
  // The production boundary requires aligned image storage. Copying also
  // makes the fuzz engine input lifetime independent from mapped table views.
  std::vector<uint8_t> aligned_bytes(size);
  if (size != 0) {
    std::memcpy(aligned_bytes.data(), data, size);
  }
  ExerciseModuleLifecycle(
      iree_make_const_byte_span(aligned_bytes.data(), aligned_bytes.size()));
}

void ExerciseFixtureMutation(const uint8_t* data, size_t size) {
  std::vector<uint8_t> mutated(
      g_fixture.contents.data,
      g_fixture.contents.data + g_fixture.contents.data_length);

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
  ExerciseModuleLifecycle(
      iree_make_const_byte_span(mutated.data(), mutated.size()));
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;
  iree_status_t status =
      iree_vm_environment_allocate(iree_allocator_system(), &g_environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    std::abort();
  }
  const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
  status = iree_vm_bytecode_module_fixture_initialize(
      iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                            files[0].size),
      iree_allocator_system(), &g_fixture);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    std::abort();
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > kMaximumInputSize) {
    return 0;
  }
  ExerciseRawInput(data, size);
  ExerciseFixtureMutation(data, size);
  return 0;
}
