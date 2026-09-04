// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_TEST_UTIL_H_
#define IREE_VM_MODULE_TEST_UTIL_H_

#include <type_traits>

#include "iree/vm/reflection.h"

namespace iree::vm::testing {

// Immutable tables published by a TableModule. Table storage is borrowed for
// the complete published module lifetime.
struct TableModuleDefinition {
  // Complete generic module descriptor.
  iree_vm_module_descriptor_t descriptor = {};
  // Import groups matching the descriptor count.
  const iree_vm_module_import_group_t* import_groups = nullptr;
  // Imports matching the descriptor count.
  const iree_vm_module_import_declaration_t* imports = nullptr;
  // Exports matching the descriptor count.
  const iree_vm_module_export_declaration_t* exports = nullptr;
  // Callable types matching the descriptor count.
  const iree_vm_module_callable_type_declaration_t* callable_types = nullptr;
  // Metadata entries matching the descriptor count.
  const iree_vm_metadata_entry_t* metadata = nullptr;

  TableModuleDefinition()
      : TableModuleDefinition(IREE_SV("test.module"), 0, 0,
                              IREE_VM_MODULE_FLAG_NONE) {}

  explicit TableModuleDefinition(
      iree_string_view_t name, iree_host_size_t function_count = 0,
      iree_host_size_t process_storage_size = 0,
      iree_vm_module_flags_t flags = IREE_VM_MODULE_FLAG_LINKABLE) {
    descriptor.name = name;
    descriptor.flags = flags;
    descriptor.counts.function_count = function_count;
    descriptor.process_storage_size = process_storage_size;
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithRefTypes(const iree_vm_ref_type_t (&values)[N]) {
    descriptor.ref_types = {values, N};
    return *this;
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithCallableTypes(
      const iree_vm_module_callable_type_declaration_t (&values)[N]) {
    callable_types = values;
    descriptor.counts.callable_type_count = N;
    descriptor.counts.callable_fields = {};
    for (const auto& value : values) {
      descriptor.counts.callable_fields.value_count +=
          value.signature.arguments.value_count +
          value.signature.results.value_count;
      descriptor.counts.callable_fields.ref_count +=
          value.signature.arguments.ref_count +
          value.signature.results.ref_count;
      descriptor.counts.callable_fields.function_count +=
          value.signature.arguments.function_count +
          value.signature.results.function_count;
    }
    return *this;
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithImportGroups(
      const iree_vm_module_import_group_t (&values)[N]) {
    import_groups = values;
    descriptor.counts.import_group_count = N;
    return *this;
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithImports(
      const iree_vm_module_import_declaration_t (&values)[N]) {
    imports = values;
    descriptor.counts.import_count = N;
    return *this;
  }

  template <iree_host_size_t GroupCount, iree_host_size_t ImportCount>
  TableModuleDefinition& WithImports(
      const iree_vm_module_import_group_t (&groups)[GroupCount],
      const iree_vm_module_import_declaration_t (&declarations)[ImportCount]) {
    return WithImportGroups(groups).WithImports(declarations);
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithExports(
      const iree_vm_module_export_declaration_t (&values)[N]) {
    exports = values;
    descriptor.counts.export_count = N;
    return *this;
  }

  template <iree_host_size_t N>
  TableModuleDefinition& WithMetadata(
      const iree_vm_metadata_entry_t (&values)[N]) {
    metadata = values;
    descriptor.counts.metadata_count = N;
    return *this;
  }
};

// Minimal table-backed native module for generic module and program tests.
// Publication copies the definition while borrowing its table storage.
class TableModule {
 public:
  TableModule()
      : vtable_{
            sizeof(vtable_),
            IREE_VM_MODULE_ABI_VERSION_0,
            Destroy,
            FunctionStart,
            iree_vm_module_function_resume_unreachable,
            nullptr,
            nullptr,
            nullptr,
            QueryImportGroup,
            QueryImport,
            QueryExport,
            QueryCallableType,
            iree_vm_module_query_presentation_none,
            MetadataByOrdinal,
        } {}
  TableModule(const TableModule&) = delete;
  TableModule& operator=(const TableModule&) = delete;
  ~TableModule() { ReleaseOwner(); }

  // Publishes the first module owner from |definition|.
  iree_status_t Publish(const TableModuleDefinition& definition) {
    definition_ = definition;
    return Publish();
  }

  // Publishes the first module owner from the mutable local definition.
  iree_status_t Publish() {
    iree_status_t status =
        iree_vm_module_initialize(&vtable_, &definition_.descriptor, &storage_);
    if (iree_status_is_ok(status)) module_ = &storage_;
    return status;
  }

  // Releases the test's owner while leaving externally retained owners live.
  void ReleaseOwner() {
    iree_vm_module_t* module = module_;
    module_ = nullptr;
    iree_vm_module_release(module);
  }

  iree_vm_module_t* module() { return module_; }
  iree_vm_module_t* storage() { return &storage_; }
  iree_vm_module_vtable_t& vtable() { return vtable_; }
  TableModuleDefinition& definition() { return definition_; }
  iree_vm_module_descriptor_t& descriptor() { return definition_.descriptor; }
  int destruction_count() const { return destruction_count_; }

 private:
  static TableModule* Cast(iree_vm_module_t* module) {
    return reinterpret_cast<TableModule*>(module);
  }

  static const TableModule* Cast(const iree_vm_module_t* module) {
    return reinterpret_cast<const TableModule*>(module);
  }

  static void Destroy(iree_vm_module_t* module) {
    ++Cast(module)->destruction_count_;
  }

  static iree_status_t FunctionStart(
      iree_vm_module_t* module,
      const iree_vm_module_function_start_params_t* params,
      iree_vm_execution_outcome_t* out_outcome) {
    (void)module;
    (void)params;
    (void)out_outcome;
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "table test module is not executable");
  }

  static void QueryImportGroup(const iree_vm_module_t* module,
                               iree_host_size_t ordinal,
                               iree_vm_module_import_group_t* out_group) {
    *out_group = Cast(module)->definition_.import_groups[ordinal];
  }

  static void QueryImport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_import_declaration_t* out_import) {
    *out_import = Cast(module)->definition_.imports[ordinal];
  }

  static void QueryExport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_export_declaration_t* out_export) {
    *out_export = Cast(module)->definition_.exports[ordinal];
  }

  static void QueryCallableType(
      const iree_vm_module_t* module, iree_host_size_t ordinal,
      iree_vm_module_callable_type_declaration_t* out_callable_type) {
    *out_callable_type = Cast(module)->definition_.callable_types[ordinal];
  }

  static void MetadataByOrdinal(const iree_vm_module_t* module,
                                const iree_vm_module_metadata_query_t* query,
                                iree_vm_metadata_entry_t* out_entry) {
    *out_entry = Cast(module)->definition_.metadata[query->ordinal];
  }

  // Generic module storage published at offset zero.
  iree_vm_module_t storage_ = {};
  // Mutable vtable used by validation tests to inject ABI defects.
  iree_vm_module_vtable_t vtable_ = {};
  // Owned descriptor and borrowed immutable semantic tables.
  TableModuleDefinition definition_;
  // Live owner pointer, or null outside the published owner lifetime.
  iree_vm_module_t* module_ = nullptr;
  // Number of final provider destruction callbacks.
  int destruction_count_ = 0;
};

static_assert(std::is_standard_layout_v<TableModule>,
              "table module storage must remain standard layout");

}  // namespace iree::vm::testing

#endif  // IREE_VM_MODULE_TEST_UTIL_H_
