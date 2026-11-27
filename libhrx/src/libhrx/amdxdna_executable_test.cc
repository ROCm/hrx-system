// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "amdxdna_executable_reuse_key.h"
#include "hrx_amdxdna.h"
#include "iree/schemas/amdxdna_elf_executable_def_reader.h"
#include "iree/schemas/amdxdna_elf_executable_def_verifier.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_reader.h"
#include "iree/schemas/amdxdna_xclbin_executable_def_verifier.h"

namespace {

int failure_count = 0;

#define CHECK(expression)                                                   \
  do {                                                                      \
    if (!(expression)) {                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                   #expression);                                            \
      ++failure_count;                                                      \
    }                                                                       \
  } while (0)

hrx_const_byte_span_t ByteSpan(const void* data, size_t size) {
  return {static_cast<const uint8_t*>(data), size};
}

hrx_string_view_t StringView(const char* value) {
  return {value, std::strlen(value)};
}

struct ExecutableDescription {
  std::array<uint8_t, 4> xclbin0 = {0x01, 0x02, 0x03, 0x04};
  std::array<uint8_t, 3> xclbin1 = {0xA0, 0xA1, 0xA2};
  std::array<hrx_const_byte_span_t, 2> xclbins;
  std::array<uint8_t, 3> pdi0 = {0xB0, 0xB1, 0xB2};
  std::array<hrx_const_byte_span_t, 1> pdis;
  std::array<uint32_t, 4> transaction0 = {0, 0, 0, 0};
  std::array<uint32_t, 4> transaction1 = {0, 0, 0, 0};
  std::array<uint32_t, 2> payload = {0x11223344, 0x55667788};
  std::array<hrx_amdxdna_pdi_relocation_t, 1> pdi_relocations = {
      hrx_amdxdna_pdi_relocation_t{/*transaction_offset=*/0,
                                   /*pdi_ordinal=*/0,
                                   /*addend=*/-16}};
  std::array<hrx_amdxdna_control_parameter_relocation_t, 1>
      control_parameter_relocations = {
          hrx_amdxdna_control_parameter_relocation_t{/*transaction_offset=*/8,
                                                     /*addend=*/0}};
  std::array<hrx_amdxdna_control_parameter_t, 1> control_parameters;
  std::array<hrx_amdxdna_executable_run_t, 2> runs;
  std::array<hrx_amdxdna_executable_entry_point_t, 2> entry_points;
  hrx_amdxdna_executable_create_params_t params;

  ExecutableDescription() {
    xclbins = {ByteSpan(xclbin0.data(), xclbin0.size()),
               ByteSpan(xclbin1.data(), xclbin1.size())};
    pdis = {ByteSpan(pdi0.data(), pdi0.size())};
    runs[0] = hrx_amdxdna_executable_run_default();
    runs[0].abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_1;
    runs[0].transaction =
        ByteSpan(transaction0.data(), transaction0.size() * sizeof(uint32_t));
    runs[0].data_payload =
        ByteSpan(payload.data(), payload.size() * sizeof(uint32_t));
    runs[0].pdi_relocations = pdi_relocations.data();
    runs[0].pdi_relocation_count = pdi_relocations.size();
    runs[0].control_parameter_relocations =
        control_parameter_relocations.data();
    runs[0].control_parameter_relocation_count =
        control_parameter_relocations.size();
    runs[1] = hrx_amdxdna_executable_run_default();
    runs[1].transaction =
        ByteSpan(transaction1.data(), transaction1.size() * sizeof(uint32_t));

    entry_points[0] = hrx_amdxdna_executable_entry_point_default();
    entry_points[0].abi_version =
        HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_1;
    entry_points[0].name = StringView("create");
    entry_points[0].xclbin_ordinal = 1;
    entry_points[0].pdi_ordinal = 2;
    entry_points[0].source_line = 42;
    entry_points[0].source_file = StringView("model.mlir");
    entry_points[0].runs = runs.data();
    entry_points[0].run_count = runs.size();
    control_parameters[0] = hrx_amdxdna_control_parameter_default();
    control_parameters[0].name = StringView("input_offset");
    control_parameters[0].scalar_type = StringView("i32");
    control_parameters[0].constant_offset = 0;
    control_parameters[0].state_table_index = 3;
    control_parameters[0].byte_length = 4;
    control_parameters[0].kind = HRX_AMDXDNA_CONTROL_PARAMETER_KIND_ADDRESS;
    entry_points[0].control_parameters = control_parameters.data();
    entry_points[0].control_parameter_count = control_parameters.size();

    entry_points[1] = hrx_amdxdna_executable_entry_point_default();
    entry_points[1].name = StringView("reuse");
    entry_points[1].context_mode = HRX_AMDXDNA_CONTEXT_MODE_REUSE;
    entry_points[1].runs = &runs[1];
    entry_points[1].run_count = 1;

    params = hrx_amdxdna_executable_create_params_default();
    params.abi_version = HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_1;
    params.xclbins = xclbins.data();
    params.xclbin_count = xclbins.size();
    params.entry_points = entry_points.data();
    params.entry_point_count = entry_points.size();
    params.pdis = pdis.data();
    params.pdi_count = pdis.size();
  }
};

bool CheckStatus(hrx_status_t status, hrx_status_code_t expected) {
  const hrx_status_code_t actual = hrx_status_code(status);
  if (actual != expected) {
    char* message = nullptr;
    size_t message_length = 0;
    hrx_status_t format_status =
        hrx_status_to_string(status, &message, &message_length);
    std::fprintf(stderr, "expected HRX status %d, got %d: %.*s\n", expected,
                 actual, static_cast<int>(message_length),
                 message ? message : "");
    hrx_status_ignore(format_status);
    hrx_status_free_message(message);
    ++failure_count;
  }
  hrx_status_ignore(status);
  return actual == expected;
}

void TestDefaults() {
  auto run = hrx_amdxdna_executable_run_default();
  CHECK(run.record_length == sizeof(run));
  CHECK(run.abi_version == HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0);

  auto entry = hrx_amdxdna_executable_entry_point_default();
  CHECK(entry.record_length == sizeof(entry));
  CHECK(entry.abi_version == HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0);
  CHECK(entry.context_mode == HRX_AMDXDNA_CONTEXT_MODE_CREATE);

  auto params = hrx_amdxdna_executable_create_params_default();
  CHECK(params.record_length == sizeof(params));
  CHECK(params.abi_version ==
        HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0);
  auto parameter = hrx_amdxdna_control_parameter_default();
  CHECK(parameter.record_length == sizeof(parameter));
  CHECK(parameter.abi_version == HRX_AMDXDNA_CONTROL_PARAMETER_ABI_VERSION_0);
}

void TestCompleteDescription() {
  ExecutableDescription description;
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  hrx_status_t status = hrx_amdxdna_xadx_serialize(
      &description.params, allocator, &data, &data_length);
  if (hrx_status_code(status) == HRX_STATUS_UNIMPLEMENTED) {
    hrx_status_ignore(status);
    std::printf("SKIP: HRX was built without the amdxdna driver\n");
    return;
  }
  if (!CheckStatus(status, HRX_STATUS_OK)) return;
  CHECK(data != nullptr);
  CHECK(data_length > 0);
  if (!data || data_length == 0) return;
  CHECK(iree_hal_amdxdna_xclbin_ExecutableDef_verify_as_root(data,
                                                             data_length) == 0);

  auto root = iree_hal_amdxdna_xclbin_ExecutableDef_as_root(data);
  auto xclbins = iree_hal_amdxdna_xclbin_ExecutableDef_xclbins_get(root);
  CHECK(iree_hal_amdxdna_xclbin_XclbinDef_vec_len(xclbins) == 2);
  auto entries = iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(root);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_vec_len(entries) == 2);
  auto pdis = iree_hal_amdxdna_xclbin_ExecutableDef_pdis_get(root);
  CHECK(iree_hal_amdxdna_xclbin_PdiDef_vec_len(pdis) == 1);

  auto create = iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entries, 0);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(create) == 1);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(create) == 2);
  auto create_runs = iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(create);
  CHECK(iree_hal_amdxdna_xclbin_RunDef_vec_len(create_runs) == 2);
  auto first_run = iree_hal_amdxdna_xclbin_RunDef_vec_at(create_runs, 0);
  CHECK(flatbuffers_uint32_vec_len(
            iree_hal_amdxdna_xclbin_RunDef_data_payload_get(first_run)) == 2);
  auto relocations =
      iree_hal_amdxdna_xclbin_RunDef_pdi_relocations_get(first_run);
  CHECK(iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_len(relocations) == 1);
  auto relocation =
      iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_at(relocations, 0);
  CHECK(iree_hal_amdxdna_xclbin_PdiRelocationDef_offset_get(relocation) == 0);
  CHECK(iree_hal_amdxdna_xclbin_PdiRelocationDef_pdi_index_get(relocation) ==
        0);
  CHECK(iree_hal_amdxdna_xclbin_PdiRelocationDef_addend_get(relocation) == -16);
  auto parameter_relocations =
      iree_hal_amdxdna_xclbin_RunDef_control_parameter_relocations_get(
          first_run);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_len(
            parameter_relocations) == 1);
  auto parameters =
      iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_get(create);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterDef_vec_len(parameters) == 1);
  auto parameter =
      iree_hal_amdxdna_xclbin_ControlParameterDef_vec_at(parameters, 0);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterDef_state_table_index_get(
            parameter) == 3);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterDef_byte_length_get(
            parameter) == 4);
  auto source =
      iree_hal_amdxdna_xclbin_EntryPointDef_source_location_get(create);
  CHECK(source != nullptr);
  if (source) {
    CHECK(iree_hal_amdxdna_xclbin_FileLineLocDef_line_get(source) == 42);
  }

  auto reuse = iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entries, 1);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(reuse) == -1);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(reuse) == -1);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_source_location_get(reuse) ==
        nullptr);
  hrx_host_allocator_free(allocator, data);
}

void TestPriorEntryPointRecordOmitsControlParameters() {
  ExecutableDescription description;
  description.params.entry_point_count = 1;
  description.entry_points[0].run_count = 1;
  description.entry_points[0].record_length =
      offsetof(hrx_amdxdna_executable_entry_point_t, control_parameters);
  description.entry_points[0].abi_version =
      HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0;
  description.runs[0].control_parameter_relocation_count = 0;

  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  CHECK(CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                               &data, &data_length),
                    HRX_STATUS_OK));
  CHECK(data != nullptr);
  auto root = iree_hal_amdxdna_xclbin_ExecutableDef_as_root(data);
  auto entries = iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(root);
  auto entry = iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entries, 0);
  auto parameters =
      iree_hal_amdxdna_xclbin_EntryPointDef_control_parameters_get(entry);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterDef_vec_len(parameters) == 0);
  hrx_host_allocator_free(allocator, data);
}

void TestPriorRunRecordOmitsRelocations() {
  ExecutableDescription description;
  description.params.entry_point_count = 1;
  description.entry_points[0].run_count = 1;
  description.runs[0].record_length =
      offsetof(hrx_amdxdna_executable_run_t, pdi_relocations);
  description.runs[0].abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0;

  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  CHECK(CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                               &data, &data_length),
                    HRX_STATUS_OK));
  CHECK(data != nullptr);
  auto root = iree_hal_amdxdna_xclbin_ExecutableDef_as_root(data);
  auto entries = iree_hal_amdxdna_xclbin_ExecutableDef_entry_points_get(root);
  auto entry = iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entries, 0);
  auto runs = iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(entry);
  auto run = iree_hal_amdxdna_xclbin_RunDef_vec_at(runs, 0);
  CHECK(iree_hal_amdxdna_xclbin_PdiRelocationDef_vec_len(
            iree_hal_amdxdna_xclbin_RunDef_pdi_relocations_get(run)) == 0);
  CHECK(iree_hal_amdxdna_xclbin_ControlParameterRelocationDef_vec_len(
            iree_hal_amdxdna_xclbin_RunDef_control_parameter_relocations_get(
                run)) == 0);
  hrx_host_allocator_free(allocator, data);
}

void TestPriorCreateRecordOmitsStandalonePdis() {
  ExecutableDescription description;
  description.params.entry_point_count = 1;
  description.entry_points[0].run_count = 1;
  description.runs[0].pdi_relocation_count = 0;
  description.params.record_length =
      offsetof(hrx_amdxdna_executable_create_params_t, pdis);
  description.params.abi_version =
      HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0;

  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  CHECK(CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                               &data, &data_length),
                    HRX_STATUS_OK));
  CHECK(data != nullptr);
  auto root = iree_hal_amdxdna_xclbin_ExecutableDef_as_root(data);
  auto pdis = iree_hal_amdxdna_xclbin_ExecutableDef_pdis_get(root);
  CHECK(iree_hal_amdxdna_xclbin_PdiDef_vec_len(pdis) == 0);
  hrx_host_allocator_free(allocator, data);
}

void TestInvalidRecords() {
  ExecutableDescription description;
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = reinterpret_cast<uint8_t*>(uintptr_t{1});
  size_t data_length = 1;

  description.params.abi_version = UINT32_MAX;
  CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator, &data,
                                         &data_length),
              HRX_STATUS_UNIMPLEMENTED);
  CHECK(data == nullptr);
  CHECK(data_length == 0);

  description.params = hrx_amdxdna_executable_create_params_default();
  description.params.record_length =
      offsetof(hrx_amdxdna_executable_create_params_t, entry_point_count);
  CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator, &data,
                                         &data_length),
              HRX_STATUS_INVALID_ARGUMENT);

  {
    ExecutableDescription invalid_stride;
    invalid_stride.runs[0].record_length = sizeof(invalid_stride.runs[0]) + 1;
    CheckStatus(hrx_amdxdna_xadx_serialize(&invalid_stride.params, allocator,
                                           &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }

  {
    ExecutableDescription malformed_transaction;
    malformed_transaction.transaction0[2] = 1;
    CheckStatus(hrx_amdxdna_xadx_serialize(&malformed_transaction.params,
                                           allocator, &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }

  {
    ExecutableDescription overlapping_relocations;
    overlapping_relocations.control_parameter_relocations[0]
        .transaction_offset = 4;
    CheckStatus(hrx_amdxdna_xadx_serialize(&overlapping_relocations.params,
                                           allocator, &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }
}

void TestVersion1RequiredForExtensions() {
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;

  {
    ExecutableDescription description;
    description.params.abi_version =
        HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0;
    CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                           &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }
  {
    ExecutableDescription description;
    description.params.pdis = nullptr;
    description.params.pdi_count = 0;
    description.entry_points[0].abi_version =
        HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0;
    CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                           &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }
  {
    ExecutableDescription description;
    description.params.pdis = nullptr;
    description.params.pdi_count = 0;
    description.runs[0].pdi_relocations = nullptr;
    description.runs[0].pdi_relocation_count = 0;
    description.runs[0].abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0;
    CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                           &data, &data_length),
                HRX_STATUS_INVALID_ARGUMENT);
  }
}

void TestVersion2XclbinIdentities() {
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;

  ExecutableDescription description;
  std::array<hrx_amdxdna_xclbin_identity_t, 2> identities = {
      hrx_amdxdna_xclbin_identity_t{1, 2, description.xclbin0.size()},
      hrx_amdxdna_xclbin_identity_t{3, 4, description.xclbin1.size()}};
  description.params.abi_version =
      HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_2;
  description.params.xclbin_identities = identities.data();
  description.params.xclbin_identity_count = identities.size();
  CHECK(CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator,
                                               &data, &data_length),
                    HRX_STATUS_OK));
  CHECK(data != nullptr);
  hrx_host_allocator_free(allocator, data);

  description.params.abi_version =
      HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_1;
  CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator, &data,
                                         &data_length),
              HRX_STATUS_INVALID_ARGUMENT);

  description.params.abi_version =
      HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_2;
  description.params.xclbin_identity_count = 1;
  CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator, &data,
                                         &data_length),
              HRX_STATUS_INVALID_ARGUMENT);

  description.params.xclbin_identity_count = identities.size();
  identities[0].data_length = description.xclbin0.size() + 1;
  CheckStatus(hrx_amdxdna_xadx_serialize(&description.params, allocator, &data,
                                         &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
}

struct ReuseKeyInputs {
  hrx_amdxdna_xclbin_identity_t identity = {1, 2, 4};
  std::array<uint32_t, 4> transaction = {1, 2, 3, 4};
  std::string name = "MLIR_AIE";
  hrx_amdxdna_executable_run_t run = hrx_amdxdna_executable_run_default();
  hrx_amdxdna_executable_entry_point_t entry_point =
      hrx_amdxdna_executable_entry_point_default();

  // Builds |key| from the current inputs.
  bool Build(hrx_amdxdna_executable_reuse_key_t* key) {
    run.transaction =
        ByteSpan(transaction.data(), transaction.size() * sizeof(uint32_t));
    entry_point.name = {name.data(), name.size()};
    return CheckStatus(hrx_amdxdna_executable_reuse_key_initialize(
                           &identity, &entry_point, &run, key),
                       HRX_STATUS_OK);
  }
};

std::vector<uint8_t> Descriptor(const hrx_amdxdna_executable_reuse_key_t& key) {
  return std::vector<uint8_t>(
      key.hal_key.descriptor.data,
      key.hal_key.descriptor.data + key.hal_key.descriptor.data_length);
}

// Builds a key after |mutate| changes the default inputs and returns its
// descriptor.
template <typename Mutate>
std::vector<uint8_t> DescriptorAfter(Mutate mutate) {
  ReuseKeyInputs inputs;
  mutate(inputs);
  hrx_amdxdna_executable_reuse_key_t key;
  CHECK(inputs.Build(&key));
  std::vector<uint8_t> descriptor = Descriptor(key);
  hrx_amdxdna_executable_reuse_key_deinitialize(&key);
  return descriptor;
}

void TestReuseKeyDescribesEveryInput() {
  ReuseKeyInputs inputs;
  hrx_amdxdna_executable_reuse_key_t key;
  CHECK(inputs.Build(&key));
  // The control code aliases the transaction rather than copying it.
  CHECK(key.hal_key.control_code.data ==
        reinterpret_cast<const uint8_t*>(inputs.transaction.data()));
  CHECK(key.hal_key.control_code.data_length ==
        inputs.transaction.size() * sizeof(uint32_t));
  CHECK(key.heap_descriptor == nullptr);
  const std::vector<uint8_t> base = Descriptor(key);
  hrx_amdxdna_executable_reuse_key_deinitialize(&key);

  // The transaction is the control code, not part of the descriptor.
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) { i.transaction[0] = 9; }) ==
        base);
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) { i.identity.hash_1 = 7; }) !=
        base);
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) { i.identity.hash_2 = 7; }) !=
        base);
  CHECK(DescriptorAfter(
            [](ReuseKeyInputs& i) { i.identity.data_length = 5; }) != base);
  CHECK(DescriptorAfter(
            [](ReuseKeyInputs& i) { i.entry_point.pdi_ordinal = 1; }) != base);
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) {
          i.entry_point.context_mode = HRX_AMDXDNA_CONTEXT_MODE_REUSE;
        }) != base);
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) { i.name = "MLIR_AIF"; }) !=
        base);
  CHECK(DescriptorAfter([](ReuseKeyInputs& i) { i.name = "MLIR_AI"; }) != base);
}

void TestReuseKeyLongNameUsesHeapStorage() {
  ReuseKeyInputs inputs;
  inputs.name.assign(HRX_AMDXDNA_EXECUTABLE_REUSE_KEY_INLINE_CAPACITY, 'n');
  hrx_amdxdna_executable_reuse_key_t key;
  CHECK(inputs.Build(&key));
  CHECK(key.heap_descriptor != nullptr);
  CHECK(key.hal_key.descriptor.data == key.heap_descriptor);
  CHECK(key.hal_key.descriptor.data_length > inputs.name.size());
  CHECK(std::memcmp(key.hal_key.descriptor.data +
                        key.hal_key.descriptor.data_length - inputs.name.size(),
                    inputs.name.data(), inputs.name.size()) == 0);
  hrx_amdxdna_executable_reuse_key_deinitialize(&key);
  CHECK(key.heap_descriptor == nullptr);
}

hrx_amdxdna_elf_export_t ElfExport(const char* kernel_name,
                                   uint32_t binding_count) {
  hrx_amdxdna_elf_export_t elf_export = hrx_amdxdna_elf_export_default();
  elf_export.kernel_name = StringView(kernel_name);
  elf_export.binding_count = binding_count;
  return elf_export;
}

hrx_amdxdna_elf_executable_create_params_t ElfParams(
    hrx_const_byte_span_t elf, const hrx_amdxdna_elf_export_t* exports,
    size_t export_count) {
  hrx_amdxdna_elf_executable_create_params_t params =
      hrx_amdxdna_elf_executable_create_params_default();
  params.elf = elf;
  params.exports = exports;
  params.export_count = export_count;
  return params;
}

hrx_status_t SerializeElf(hrx_const_byte_span_t elf,
                          const hrx_amdxdna_elf_export_t* exports,
                          size_t export_count, uint8_t** out_data,
                          size_t* out_data_length) {
  const hrx_amdxdna_elf_executable_create_params_t params =
      ElfParams(elf, exports, export_count);
  return hrx_amdxdna_elf_serialize(&params, hrx_host_allocator_system(),
                                   out_data, out_data_length);
}

void TestElfDefaults() {
  const hrx_amdxdna_elf_export_t elf_export = hrx_amdxdna_elf_export_default();
  CHECK(elf_export.record_length == sizeof(hrx_amdxdna_elf_export_t));
  CHECK(elf_export.abi_version == HRX_AMDXDNA_ELF_EXPORT_ABI_VERSION_0);
  CHECK(elf_export.kernel_name.data == nullptr);
  CHECK(elf_export.kernel_name.size == 0);
  CHECK(elf_export.binding_count == 0);
  CHECK(elf_export.reserved == 0);
  CHECK(elf_export.control_parameters == nullptr);
  CHECK(elf_export.control_parameter_count == 0);

  const hrx_amdxdna_elf_executable_create_params_t params =
      hrx_amdxdna_elf_executable_create_params_default();
  CHECK(params.record_length ==
        sizeof(hrx_amdxdna_elf_executable_create_params_t));
  CHECK(params.abi_version ==
        HRX_AMDXDNA_ELF_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0);
  CHECK(params.flags == 0);
  CHECK(params.reserved == 0);
  CHECK(params.elf.data == nullptr);
  CHECK(params.elf.data_length == 0);
  CHECK(params.exports == nullptr);
  CHECK(params.export_count == 0);
}

void TestElfSerialize() {
  // The serializer does not parse the ELF; any bytes stand in for one.
  const std::array<uint8_t, 6> elf = {0x7F, 'E', 'L', 'F', 0x02, 0x01};
  const std::array<hrx_amdxdna_elf_export_t, 2> elf_exports = {
      ElfExport("main:gemm", 66), ElfExport("main:norm", 3)};
  uint8_t* data = nullptr;
  size_t data_length = 0;
  hrx_status_t status =
      SerializeElf(ByteSpan(elf.data(), elf.size()), elf_exports.data(),
                   elf_exports.size(), &data, &data_length);
  if (hrx_status_code(status) == HRX_STATUS_UNIMPLEMENTED) {
    hrx_status_ignore(status);
    std::printf("SKIP: HRX was built without the amdxdna driver\n");
    return;
  }
  if (!CheckStatus(status, HRX_STATUS_OK)) return;
  CHECK(data != nullptr);
  if (!data) return;
  CHECK(iree_hal_amdxdna_elf_ExecutableDef_verify_as_root(data, data_length) ==
        0);
  auto root = iree_hal_amdxdna_elf_ExecutableDef_as_root(data);
  auto modules = iree_hal_amdxdna_elf_ExecutableDef_modules_get(root);
  CHECK(iree_hal_amdxdna_elf_ModuleDef_vec_len(modules) == 1);
  auto module_elf = iree_hal_amdxdna_elf_ModuleDef_elf_get(
      iree_hal_amdxdna_elf_ModuleDef_vec_at(modules, 0));
  CHECK(flatbuffers_uint8_vec_len(module_elf) == elf.size());
  CHECK(std::memcmp(module_elf, elf.data(), elf.size()) == 0);
  auto exports = iree_hal_amdxdna_elf_ExecutableDef_exports_get(root);
  CHECK(iree_hal_amdxdna_elf_ExportDef_vec_len(exports) == elf_exports.size());
  for (size_t i = 0; i < elf_exports.size(); ++i) {
    auto export_def = iree_hal_amdxdna_elf_ExportDef_vec_at(exports, i);
    flatbuffers_string_t name =
        iree_hal_amdxdna_elf_ExportDef_name_get(export_def);
    CHECK(std::string(name, flatbuffers_string_len(name)) ==
          std::string(elf_exports[i].kernel_name.data,
                      elf_exports[i].kernel_name.size));
    CHECK(iree_hal_amdxdna_elf_ExportDef_module_ordinal_get(export_def) == 0);
    CHECK(iree_hal_amdxdna_elf_ExportDef_binding_count_get(export_def) ==
          elf_exports[i].binding_count);
  }
  hrx_host_allocator_free(hrx_host_allocator_system(), data);
}

void TestElfSerializeAcceptsLongerRecords() {
  // A newer producer's records are longer; HRX steps exports by record_length
  // and ignores trailing parameter fields it does not know.
  struct ExtendedElfExport {
    hrx_amdxdna_elf_export_t elf_export;
    uint64_t future_field;
  };
  struct ExtendedElfParams {
    hrx_amdxdna_elf_executable_create_params_t params;
    uint64_t future_field;
  };
  const std::array<uint8_t, 4> elf = {0x7F, 'E', 'L', 'F'};
  std::array<ExtendedElfExport, 2> records = {};
  records[0].elf_export = ElfExport("first", 2);
  records[1].elf_export = ElfExport("second", 5);
  for (ExtendedElfExport& record : records) {
    record.elf_export.record_length = sizeof(ExtendedElfExport);
    record.future_field = UINT64_MAX;
  }
  ExtendedElfParams extended = {};
  extended.params = ElfParams(ByteSpan(elf.data(), elf.size()),
                              &records[0].elf_export, records.size());
  extended.params.record_length = sizeof(ExtendedElfParams);
  extended.future_field = UINT64_MAX;
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  hrx_status_t status = hrx_amdxdna_elf_serialize(&extended.params, allocator,
                                                  &data, &data_length);
  if (hrx_status_code(status) == HRX_STATUS_UNIMPLEMENTED) {
    hrx_status_ignore(status);
    std::printf("SKIP: HRX was built without the amdxdna driver\n");
    return;
  }
  if (!CheckStatus(status, HRX_STATUS_OK)) return;
  auto root = iree_hal_amdxdna_elf_ExecutableDef_as_root(data);
  auto exports = iree_hal_amdxdna_elf_ExecutableDef_exports_get(root);
  CHECK(iree_hal_amdxdna_elf_ExportDef_vec_len(exports) == records.size());
  auto second = iree_hal_amdxdna_elf_ExportDef_vec_at(exports, 1);
  flatbuffers_string_t name = iree_hal_amdxdna_elf_ExportDef_name_get(second);
  CHECK(std::string(name, flatbuffers_string_len(name)) == "second");
  CHECK(iree_hal_amdxdna_elf_ExportDef_binding_count_get(second) == 5);
  hrx_host_allocator_free(allocator, data);
}

void TestElfSerializeRejectsInvalidParams() {
  const std::array<uint8_t, 4> elf = {0x7F, 'E', 'L', 'F'};
  const hrx_amdxdna_elf_export_t elf_export = ElfExport("main", 1);
  const hrx_amdxdna_elf_executable_create_params_t valid =
      ElfParams(ByteSpan(elf.data(), elf.size()), &elf_export, 1);
  hrx_host_allocator_t allocator = hrx_host_allocator_system();
  uint8_t* data = nullptr;
  size_t data_length = 0;
  hrx_status_t probe =
      hrx_amdxdna_elf_serialize(&valid, allocator, &data, &data_length);
  if (hrx_status_code(probe) == HRX_STATUS_UNIMPLEMENTED) {
    hrx_status_ignore(probe);
    std::printf("SKIP: HRX was built without the amdxdna driver\n");
    return;
  }
  CheckStatus(probe, HRX_STATUS_OK);
  hrx_host_allocator_free(allocator, data);

  CheckStatus(
      hrx_amdxdna_elf_serialize(nullptr, allocator, &data, &data_length),
      HRX_STATUS_INVALID_ARGUMENT);
  CHECK(data == nullptr && data_length == 0);
  CheckStatus(
      hrx_amdxdna_elf_serialize(&valid, allocator, nullptr, &data_length),
      HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_executable_create_params_t header_only = valid;
  header_only.record_length = 4;
  CheckStatus(
      hrx_amdxdna_elf_serialize(&header_only, allocator, &data, &data_length),
      HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_executable_create_params_t truncated = valid;
  truncated.record_length =
      offsetof(hrx_amdxdna_elf_executable_create_params_t, export_count);
  CheckStatus(
      hrx_amdxdna_elf_serialize(&truncated, allocator, &data, &data_length),
      HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_executable_create_params_t future_abi = valid;
  future_abi.abi_version =
      HRX_AMDXDNA_ELF_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0 + 1;
  CheckStatus(
      hrx_amdxdna_elf_serialize(&future_abi, allocator, &data, &data_length),
      HRX_STATUS_UNIMPLEMENTED);
  hrx_amdxdna_elf_executable_create_params_t flags = valid;
  flags.flags = 1;
  CheckStatus(hrx_amdxdna_elf_serialize(&flags, allocator, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_executable_create_params_t reserved = valid;
  reserved.reserved = 1;
  CheckStatus(
      hrx_amdxdna_elf_serialize(&reserved, allocator, &data, &data_length),
      HRX_STATUS_INVALID_ARGUMENT);
  CHECK(data == nullptr && data_length == 0);

  hrx_executable_t executable = nullptr;
  CheckStatus(hrx_amdxdna_elf_executable_create(nullptr, &valid, &executable),
              HRX_STATUS_INVALID_ARGUMENT);
  CHECK(executable == nullptr);
}

void TestElfSerializeRejectsInvalidExports() {
  const std::array<uint8_t, 4> elf = {0x7F, 'E', 'L', 'F'};
  const hrx_const_byte_span_t elf_span = ByteSpan(elf.data(), elf.size());
  const hrx_amdxdna_elf_export_t name = ElfExport("main", 1);
  uint8_t* data = nullptr;
  size_t data_length = 0;
  hrx_status_t probe = SerializeElf(elf_span, &name, 1, &data, &data_length);
  if (hrx_status_code(probe) == HRX_STATUS_UNIMPLEMENTED) {
    hrx_status_ignore(probe);
    std::printf("SKIP: HRX was built without the amdxdna driver\n");
    return;
  }
  CheckStatus(probe, HRX_STATUS_OK);
  hrx_host_allocator_free(hrx_host_allocator_system(), data);

  CheckStatus(SerializeElf(ByteSpan(nullptr, 0), &name, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  CHECK(data == nullptr && data_length == 0);
  CheckStatus(SerializeElf(ByteSpan(nullptr, 4), &name, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  CheckStatus(SerializeElf(elf_span, nullptr, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  CheckStatus(SerializeElf(elf_span, &name, 0, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_export_t empty_name = name;
  empty_name.kernel_name = {"", 0};
  CheckStatus(SerializeElf(elf_span, &empty_name, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_export_t truncated = name;
  truncated.record_length =
      offsetof(hrx_amdxdna_elf_export_t, control_parameter_count);
  CheckStatus(SerializeElf(elf_span, &truncated, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_export_t misaligned = name;
  misaligned.record_length = sizeof(hrx_amdxdna_elf_export_t) + 1;
  CheckStatus(SerializeElf(elf_span, &misaligned, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  hrx_amdxdna_elf_export_t future_abi = name;
  future_abi.abi_version = HRX_AMDXDNA_ELF_EXPORT_ABI_VERSION_0 + 1;
  CheckStatus(SerializeElf(elf_span, &future_abi, 1, &data, &data_length),
              HRX_STATUS_UNIMPLEMENTED);
  hrx_amdxdna_elf_export_t reserved = name;
  reserved.reserved = 1;
  CheckStatus(SerializeElf(elf_span, &reserved, 1, &data, &data_length),
              HRX_STATUS_INVALID_ARGUMENT);
  const hrx_amdxdna_control_parameter_t parameter =
      hrx_amdxdna_control_parameter_default();
  hrx_amdxdna_elf_export_t with_parameters = name;
  with_parameters.control_parameters = &parameter;
  with_parameters.control_parameter_count = 1;
  CheckStatus(SerializeElf(elf_span, &with_parameters, 1, &data, &data_length),
              HRX_STATUS_UNIMPLEMENTED);
  CHECK(data == nullptr && data_length == 0);
}

}  // namespace

int main() {
  TestDefaults();
  TestCompleteDescription();
  TestElfDefaults();
  TestElfSerialize();
  TestElfSerializeAcceptsLongerRecords();
  TestElfSerializeRejectsInvalidParams();
  TestElfSerializeRejectsInvalidExports();
  TestPriorEntryPointRecordOmitsControlParameters();
  TestPriorRunRecordOmitsRelocations();
  TestPriorCreateRecordOmitsStandalonePdis();
  TestInvalidRecords();
  TestVersion1RequiredForExtensions();
  TestVersion2XclbinIdentities();
  TestReuseKeyDescribesEveryInput();
  TestReuseKeyLongNameUsesHeapStorage();
  if (failure_count != 0) {
    std::fprintf(stderr, "amdxdna executable API test: %d failure(s)\n",
                 failure_count);
    return 1;
  }
  std::printf("amdxdna executable API test: PASS\n");
  return 0;
}
