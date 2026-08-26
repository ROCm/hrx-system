// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Raw-byte fuzzing for the inspection and executable module factories.
//
// Inputs either pass bytes directly to the factories or blindly mutate a
// known-good source fixture. The mutation modes intentionally know nothing
// about the module schema so that malformed envelope, section, declaration,
// and instruction bytes reach the production parsers without a second decoder
// sharing their assumptions.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/vm/bytecode/inspection.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/tooling/dump.h"

namespace {

struct FuzzRefTypes {
  // Type used by the ref-state and buffer fixtures.
  iree_vm_ref_type_t object;
};

extern const iree_vm_ref_type_table_t kFuzzRefTypeTable;

void UnexpectedObjectDestroy(void* object) {
  (void)object;
  std::abort();
}

const iree_vm_ref_type_descriptor_t kFuzzObjectType = {
    UnexpectedObjectDestroy,
    &kFuzzRefTypeTable,
    IREE_SV("object"),
};

const FuzzRefTypes kFuzzRefTypes = {
    &kFuzzObjectType,
};

const iree_vm_ref_type_table_t kFuzzRefTypeTable = {
    sizeof(kFuzzRefTypeTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("zz_test"),
    {&kFuzzRefTypes, 1},
};

void RequireOk(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

std::vector<uint8_t> BuildSeed(uint8_t selector) {
  using namespace iree::vm::bytecode::testing;
  switch (selector % 8) {
    case 0: {
      static const std::vector<uint8_t> seed = BuildOwnershipModuleImage();
      return seed;
    }
    case 1: {
      static const std::vector<uint8_t> seed = BuildLaunchConfigModuleImage();
      return seed;
    }
    case 2: {
      static const std::vector<uint8_t> seed = BuildRefStateModuleImage();
      return seed;
    }
    case 3: {
      static const std::vector<uint8_t> seed = BuildFunctionStateModuleImage();
      return seed;
    }
    case 4: {
      static const std::vector<uint8_t> seed = BuildValueOverflowModuleImage();
      return seed;
    }
    case 5: {
      static const std::vector<uint8_t> seed = BuildScalarStateModuleImage();
      return seed;
    }
    case 6: {
      static const std::vector<uint8_t> seed = BuildBufferModuleImage();
      return seed;
    }
    default: {
      static const std::vector<uint8_t> seed = BuildHALInspectionModuleImage();
      return seed;
    }
  }
}

std::vector<uint8_t> BuildInputImage(const uint8_t* data, size_t size) {
  if (size == 0) return {};

  const uint8_t control = data[0];
  const uint8_t mutation_mode = control & 3u;
  if (mutation_mode == 0) {
    return std::vector<uint8_t>(data + 1, data + size);
  }

  std::vector<uint8_t> image = BuildSeed(control >> 2);
  switch (mutation_mode) {
    case 1:
      // Blind random-access replacement keeps the overall image extent while
      // allowing mutations to reach sections far beyond the header.
      for (size_t i = 1; i + 2 < size; i += 3) {
        const size_t offset =
            (static_cast<size_t>(data[i]) << 8 | data[i + 1]) % image.size();
        image[offset] = data[i + 2];
      }
      break;
    case 2:
      // Appended bytes exercise exact-packing and trailing-data checks.
      image.insert(image.end(), data + 1, data + size);
      break;
    case 3:
      // Every prefix, including the empty image, is reachable without knowing
      // where any schema boundary lies.
      if (size > 1) {
        const size_t prefix_length =
            (static_cast<size_t>(data[1]) << 8 | data[size > 2 ? 2 : 1]) %
            (image.size() + 1);
        image.resize(prefix_length);
      }
      break;
  }
  return image;
}

struct ModuleSummary {
  // Number of public import declarations.
  iree_host_size_t import_count;
  // Number of public export declarations.
  iree_host_size_t export_count;
  // Number of private function declarations.
  iree_host_size_t function_count;
  // Number of module-local ref types.
  iree_host_size_t ref_type_count;
  // Number of module-level metadata entries.
  iree_host_size_t metadata_count;
};

ModuleSummary SummarizeModule(const iree_vm_module_t* module) {
  return {
      iree_vm_module_import_count(module),
      iree_vm_module_export_count(module),
      iree_vm_module_function_count(module),
      iree_vm_module_ref_type_count(module),
      iree_vm_module_metadata_count(module),
  };
}

bool SummariesMatch(const ModuleSummary& lhs, const ModuleSummary& rhs) {
  return lhs.import_count == rhs.import_count &&
         lhs.export_count == rhs.export_count &&
         lhs.function_count == rhs.function_count &&
         lhs.ref_type_count == rhs.ref_type_count &&
         lhs.metadata_count == rhs.metadata_count;
}

bool TryLoadInspection(iree_const_byte_span_t image,
                       ModuleSummary* out_summary) {
  iree_vm_module_t* module = reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree::Status status(iree_vm_bytecode_module_create_for_inspection(
      IREE_SV("module_factory_fuzz"), {image, iree_allocator_null()},
      iree_allocator_system(), &module));
  if (!status.ok()) {
    if (module != nullptr) std::abort();
    return false;
  }
  if (module == nullptr) std::abort();
  *out_summary = SummarizeModule(module);
  iree_vm_module_release(module);
  return true;
}

bool TryLoadExecutable(iree_const_byte_span_t image,
                       ModuleSummary* out_summary) {
  struct Environment {
    Environment() {
      RequireOk(iree_vm_environment_allocate(iree_allocator_system(), &value));
      RequireOk(iree_vm_environment_register_ref_type_table(
          value, &kFuzzRefTypeTable));
    }
    ~Environment() { iree_vm_environment_free(value); }

    // Shared immutable factory environment.
    iree_vm_environment_t* value = nullptr;
  };
  static Environment environment;

  iree_vm_module_t* module = reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree::Status status(iree_vm_bytecode_module_create(
      environment.value, IREE_SV("module_factory_fuzz"),
      {image, iree_allocator_null()}, iree_allocator_system(), &module));
  if (!status.ok()) {
    if (module != nullptr) std::abort();
    return false;
  }
  if (module == nullptr) std::abort();
  *out_summary = SummarizeModule(module);
  iree_vm_module_release(module);
  return true;
}

struct DumpSink {
  // Number of bytes emitted by the dumper.
  iree_host_size_t byte_count;
  // FNV-1a hash of the emitted byte stream.
  uint64_t hash;
};

iree_status_t HashDumpBytes(void* user_data, iree_string_view_t text) {
  auto* sink = static_cast<DumpSink*>(user_data);
  if (text.size > IREE_HOST_SIZE_MAX - sink->byte_count) std::abort();
  sink->byte_count += text.size;
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    sink->hash ^= static_cast<uint8_t>(text.data[i]);
    sink->hash *= UINT64_C(1099511628211);
  }
  return iree_ok_status();
}

bool TryDump(iree_const_byte_span_t image, DumpSink* out_sink) {
  *out_sink = {0, UINT64_C(14695981039346656037)};
  iree::Status status(iree_vm_bytecode_module_dump(
      IREE_SV("module_factory_fuzz"), image, {HashDumpBytes, out_sink},
      iree_allocator_system()));
  return status.ok();
}

void ExerciseModuleFactories(iree_const_byte_span_t image) {
  ModuleSummary inspection_summary = {};
  const bool inspection_loaded = TryLoadInspection(image, &inspection_summary);

  ModuleSummary executable_summary = {};
  const bool executable_loaded = TryLoadExecutable(image, &executable_summary);
  if (executable_loaded &&
      (!inspection_loaded ||
       !SummariesMatch(inspection_summary, executable_summary))) {
    std::abort();
  }

  DumpSink dump_sink = {};
  const bool dumped = TryDump(image, &dump_sink);
  if (executable_loaded && (!dumped || dump_sink.byte_count == 0)) {
    std::abort();
  }

  if (executable_loaded) {
    DumpSink repeated_dump_sink = {};
    if (!TryDump(image, &repeated_dump_sink) ||
        dump_sink.byte_count != repeated_dump_sink.byte_count ||
        dump_sink.hash != repeated_dump_sink.hash) {
      std::abort();
    }

    ModuleSummary reloaded_summary = {};
    if (!TryLoadExecutable(image, &reloaded_summary) ||
        !SummariesMatch(executable_summary, reloaded_summary)) {
      std::abort();
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  const std::vector<uint8_t> image = BuildInputImage(data, size);
  std::vector<uint64_t> aligned_storage(image.size() / sizeof(uint64_t) +
                                        (image.size() % sizeof(uint64_t) != 0));
  if (!image.empty()) {
    std::memcpy(aligned_storage.data(), image.data(), image.size());
  }
  ExerciseModuleFactories(iree_make_const_byte_span(
      reinterpret_cast<const uint8_t*>(aligned_storage.data()), image.size()));
  return 0;
}
