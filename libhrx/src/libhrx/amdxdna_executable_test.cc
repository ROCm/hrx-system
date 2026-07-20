// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "hrx_amdxdna.h"
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
  std::array<uint32_t, 4> transaction0 = {0, 0, 0, 0};
  std::array<uint32_t, 4> transaction1 = {0, 0, 0, 0};
  std::array<uint32_t, 2> payload = {0x11223344, 0x55667788};
  std::array<hrx_amdxdna_executable_run_t, 2> runs;
  std::array<hrx_amdxdna_executable_entry_point_t, 2> entry_points;
  hrx_amdxdna_executable_create_params_t params;

  ExecutableDescription() {
    xclbins = {ByteSpan(xclbin0.data(), xclbin0.size()),
               ByteSpan(xclbin1.data(), xclbin1.size())};
    runs[0] = hrx_amdxdna_executable_run_default();
    runs[0].transaction =
        ByteSpan(transaction0.data(), transaction0.size() * sizeof(uint32_t));
    runs[0].data_payload =
        ByteSpan(payload.data(), payload.size() * sizeof(uint32_t));
    runs[1] = hrx_amdxdna_executable_run_default();
    runs[1].transaction =
        ByteSpan(transaction1.data(), transaction1.size() * sizeof(uint32_t));

    entry_points[0] = hrx_amdxdna_executable_entry_point_default();
    entry_points[0].name = StringView("create");
    entry_points[0].xclbin_ordinal = 1;
    entry_points[0].pdi_ordinal = 2;
    entry_points[0].source_line = 42;
    entry_points[0].source_file = StringView("model.mlir");
    entry_points[0].runs = runs.data();
    entry_points[0].run_count = runs.size();

    entry_points[1] = hrx_amdxdna_executable_entry_point_default();
    entry_points[1].name = StringView("reuse");
    entry_points[1].context_mode = HRX_AMDXDNA_CONTEXT_MODE_REUSE;
    entry_points[1].runs = &runs[1];
    entry_points[1].run_count = 1;

    params = hrx_amdxdna_executable_create_params_default();
    params.xclbins = xclbins.data();
    params.xclbin_count = xclbins.size();
    params.entry_points = entry_points.data();
    params.entry_point_count = entry_points.size();
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

  auto create = iree_hal_amdxdna_xclbin_EntryPointDef_vec_at(entries, 0);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_xclbin_index_get(create) == 1);
  CHECK(iree_hal_amdxdna_xclbin_EntryPointDef_pdi_index_get(create) == 2);
  auto create_runs = iree_hal_amdxdna_xclbin_EntryPointDef_runs_get(create);
  CHECK(iree_hal_amdxdna_xclbin_RunDef_vec_len(create_runs) == 2);
  auto first_run = iree_hal_amdxdna_xclbin_RunDef_vec_at(create_runs, 0);
  CHECK(flatbuffers_uint32_vec_len(
            iree_hal_amdxdna_xclbin_RunDef_data_payload_get(first_run)) == 2);
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
}

}  // namespace

int main() {
  TestDefaults();
  TestCompleteDescription();
  TestInvalidRecords();
  if (failure_count != 0) {
    std::fprintf(stderr, "amdxdna executable API test: %d failure(s)\n",
                 failure_count);
    return 1;
  }
  std::printf("amdxdna executable API test: PASS\n");
  return 0;
}
