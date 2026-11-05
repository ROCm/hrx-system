// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/dxg/kmt.h"

namespace {

using InvokeHandler = uint32_t (*)(void* context, uint64_t request,
                                   void* arguments);

struct Invocation {
  // Explicit connection forwarded as the transport context.
  const amdf_linux_dxg_connection_t* connection = nullptr;
  // Encoded Linux ioctl request observed by the transport.
  uint64_t request = 0;
  // Operation argument storage forwarded to the transport.
  void* arguments = nullptr;
  // Optional typed observer controlling the native result.
  InvokeHandler handler = nullptr;
  // Observer-owned state forwarded unchanged to the handler.
  void* handler_context = nullptr;
  // Number of native invocations observed.
  uint32_t count = 0;
  // Native result returned when no handler is installed.
  uint32_t result = 0;
};

Invocation invocation;

using Utf16 = uint16_t;

struct UmdFilenameInfoWire {
  // Requested user-mode driver interface version.
  KMTUMDVERSION version;
  // Returned Windows UTF-16 driver path.
  Utf16 file_name[MAX_PATH];
};

struct OpenGlInfoWire {
  // Returned Windows UTF-16 OpenGL driver path.
  Utf16 file_name[MAX_PATH];
  // OpenGL driver interface version.
  ULONG version;
  // OpenGL driver property bits.
  ULONG flags;
};

struct AdapterRegistryInfoWire {
  // Display adapter description.
  Utf16 adapter_string[MAX_PATH];
  // Display adapter BIOS description.
  Utf16 bios_string[MAX_PATH];
  // Display adapter DAC description.
  Utf16 dac_type[MAX_PATH];
  // Display adapter chip description.
  Utf16 chip_type[MAX_PATH];
};

struct DriverDescriptionWire {
  // Returned Windows UTF-16 driver description.
  Utf16 description[4096];
};

struct AdapterGuidWire {
  // Returned Windows UTF-16 adapter GUID.
  Utf16 guid[40];
};

#pragma pack(push, 1)
struct NodeMetadataWire {
  // Engine class exposed by the node.
  DXGK_ENGINE_TYPE engine_type;
  // Returned Windows UTF-16 node name.
  Utf16 friendly_name[DXGK_MAX_METADATA_NAME_LENGTH];
  // Node scheduling and queue property bits.
  DXGK_NODEMETADATA_FLAGS flags;
  // Nonzero when the node supports GPU page tables.
  BOOLEAN gpu_mmu_supported;
  // Nonzero when the node supports an IO memory management unit.
  BOOLEAN io_mmu_supported;
};

struct QueryNodeMetadataWire {
  // Packed physical-adapter and node ordinal supplied by the caller.
  UINT node_ordinal_and_adapter_index;
  // Returned node metadata.
  NodeMetadataWire node_data;
};
#pragma pack(pop)

struct QueryRegistryInfoWire {
  // Registry namespace selected by the query.
  D3DDDI_QUERYREGISTRY_TYPE query_type;
  // Registry query behavior bits.
  D3DDDI_QUERYREGISTRY_FLAGS query_flags;
  // Windows UTF-16 value name supplied by the caller.
  Utf16 value_name[MAX_PATH];
  // Windows registry value type.
  ULONG value_type;
  // Physical adapter selected within a linked adapter.
  ULONG physical_adapter_index;
  // Returned or required value length in wire bytes.
  ULONG output_value_size;
  // Result of the registry lookup itself.
  D3DDDI_QUERYREGISTRY_STATUS status;
  // Returned registry value storage.
  union {
    // 32-bit integer result.
    DWORD output_dword;
    // 64-bit integer result.
    uint64_t output_qword;
    // Windows UTF-16 string result.
    Utf16 output_string[1];
    // Uninterpreted binary result.
    BYTE output_binary[1];
  } value;
};

static_assert(sizeof(UmdFilenameInfoWire) == 524);
static_assert(sizeof(OpenGlInfoWire) == 528);
static_assert(sizeof(AdapterRegistryInfoWire) == 2080);
static_assert(sizeof(DriverDescriptionWire) == 8192);
static_assert(sizeof(AdapterGuidWire) == 80);
static_assert(sizeof(QueryNodeMetadataWire) == 78);
static_assert(sizeof(QueryRegistryInfoWire) == 552);

template <size_t Capacity>
void CopyUtf16(std::u16string_view source, Utf16 (&target)[Capacity]) {
  ASSERT_LT(source.size(), Capacity);
  for (size_t i = 0; i < source.size(); ++i) {
    target[i] = static_cast<Utf16>(source[i]);
  }
  target[source.size()] = 0;
}

std::wstring ReadWideString(const void* source, size_t capacity) {
  std::wstring value;
  for (size_t i = 0; i < capacity; ++i) {
    wchar_t character = 0;
    std::memcpy(&character,
                static_cast<const std::byte*>(source) + i * sizeof(character),
                sizeof(character));
    if (character == 0) {
      break;
    }
    value.push_back(character);
  }
  return value;
}

template <typename T>
void StoreNativeField(void* storage, size_t offset, T value) {
  std::memcpy(static_cast<std::byte*>(storage) + offset, &value, sizeof(value));
}

template <typename T>
T LoadNativeField(const void* storage, size_t offset) {
  T value = {};
  std::memcpy(&value, static_cast<const std::byte*>(storage) + offset,
              sizeof(value));
  return value;
}

struct ExpectedLayout {
  // Original caller storage that must not be passed to the kernel.
  void* native_data;
  // Expected UTF-16 wire payload size.
  uint32_t wire_size;
};

uint32_t CheckLayout(void* context, uint64_t request, void* arguments) {
  auto* expected = static_cast<ExpectedLayout*>(context);
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  EXPECT_EQ(request,
            static_cast<uint64_t>(_IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO)));
  EXPECT_NE(query->pPrivateDriverData, expected->native_data);
  EXPECT_EQ(query->PrivateDriverDataSize, expected->wire_size);
  return static_cast<uint32_t>(STATUS_UNSUCCESSFUL);
}

class DxgKmtQueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    connection_.descriptor = 17;
    host_allocator_ = amdf_allocator_system();
    kmt_.connection = &connection_;
    kmt_.host_allocator = &host_allocator_;
    invocation = {};
  }

  void TearDown() override { invocation = {}; }

  // Synthetic connection observed by the wrapped transport.
  amdf_linux_dxg_connection_t connection_ = {};
  // Allocator borrowed by the typed KMT view.
  amdf_allocator_t host_allocator_ = {};
  // Typed KMT view under test.
  amdf_linux_dxg_kmt_t kmt_ = {};
};

TEST_F(DxgKmtQueryTest, ConvertsEveryStringBearingFixedLayout) {
  struct QueryCase {
    // Query type selecting the row.
    KMTQUERYADAPTERINFOTYPE type;
    // Linux wchar_t-bearing structure size.
    uint32_t native_size;
    // Windows UTF-16 wire structure size.
    uint32_t wire_size;
  };
  const QueryCase cases[] = {
      {KMTQAITYPE_UMDRIVERNAME, sizeof(D3DKMT_UMDFILENAMEINFO), 524},
      {KMTQAITYPE_UMOPENGLINFO, sizeof(D3DKMT_OPENGLINFO), 528},
      {KMTQAITYPE_ADAPTERREGISTRYINFO, sizeof(D3DKMT_ADAPTERREGISTRYINFO),
       2080},
      {KMTQAITYPE_ADAPTERREGISTRYINFO_RENDER,
       sizeof(D3DKMT_ADAPTERREGISTRYINFO), 2080},
      {KMTQAITYPE_DRIVER_DESCRIPTION, sizeof(D3DKMT_DRIVER_DESCRIPTION), 8192},
      {KMTQAITYPE_DRIVER_DESCRIPTION_RENDER, sizeof(D3DKMT_DRIVER_DESCRIPTION),
       8192},
      {KMTQAITYPE_QUERY_ADAPTER_UNIQUE_GUID,
       sizeof(D3DKMT_QUERY_ADAPTER_UNIQUE_GUID), 80},
      {KMTQAITYPE_NODEMETADATA, sizeof(D3DKMT_NODEMETADATA), 78},
  };
  alignas(std::max_align_t)
      std::array<std::byte, sizeof(D3DKMT_DRIVER_DESCRIPTION)>
          native_data;
  for (const QueryCase& test_case : cases) {
    SCOPED_TRACE(test_case.type);
    std::memset(native_data.data(), 0, native_data.size());
    D3DKMT_QUERYADAPTERINFO query = {
        .hAdapter = 3,
        .Type = test_case.type,
        .pPrivateDriverData = native_data.data(),
        .PrivateDriverDataSize = test_case.native_size,
    };
    ExpectedLayout expected = {
        .native_data = native_data.data(),
        .wire_size = test_case.wire_size,
    };
    invocation.handler = CheckLayout;
    invocation.handler_context = &expected;
    EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query),
              STATUS_UNSUCCESSFUL);
    EXPECT_EQ(invocation.count, 1u);
    invocation.count = 0;
  }
}

uint32_t PopulateUmdFilename(void*, uint64_t, void* arguments) {
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  auto* wire = static_cast<UmdFilenameInfoWire*>(query->pPrivateDriverData);
  EXPECT_EQ(wire->version, KMTUMDVERSION_DX12);
  CopyUtf16(uR"(C:\Windows\System32\DriverStore\FileRepository\amd\driver.dll)",
            wire->file_name);
  return 0;
}

uint32_t PopulateNodeMetadata(void*, uint64_t, void* arguments) {
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  auto* wire = static_cast<QueryNodeMetadataWire*>(query->pPrivateDriverData);
  EXPECT_EQ(wire->node_ordinal_and_adapter_index, 0x12340007u);
  wire->node_data.engine_type = DXGK_ENGINE_TYPE_3D;
  CopyUtf16(u"Compute \u03A9", wire->node_data.friendly_name);
  wire->node_data.flags.Value = 0x12345678u;
  wire->node_data.gpu_mmu_supported = 1;
  wire->node_data.io_mmu_supported = 1;
  return 0;
}

TEST_F(DxgKmtQueryTest, PreservesMixedFixedInputsAndOutputs) {
  D3DKMT_UMDFILENAMEINFO filename = {.Version = KMTUMDVERSION_DX12};
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_UMDRIVERNAME,
      .pPrivateDriverData = &filename,
      .PrivateDriverDataSize = sizeof(filename),
  };
  invocation.handler = PopulateUmdFilename;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(std::wstring(filename.UmdFileName),
            L"/usr/lib/wsl/drivers/amd/driver.dll");

  alignas(wchar_t) std::array<std::byte, sizeof(D3DKMT_NODEMETADATA) + 2>
      metadata_storage = {};
  void* metadata = metadata_storage.data() + 2;
  constexpr size_t kNodeDataOffset = offsetof(D3DKMT_NODEMETADATA, NodeData);
  constexpr size_t kFriendlyNameOffset =
      kNodeDataOffset + offsetof(DXGK_NODEMETADATA, FriendlyName);
  StoreNativeField<UINT>(
      metadata, offsetof(D3DKMT_NODEMETADATA, NodeOrdinalAndAdapterIndex),
      0x12340007u);
  EXPECT_NE((reinterpret_cast<uintptr_t>(metadata) + kFriendlyNameOffset) %
                alignof(wchar_t),
            0u);
  query.Type = KMTQAITYPE_NODEMETADATA;
  query.pPrivateDriverData = metadata;
  query.PrivateDriverDataSize = sizeof(D3DKMT_NODEMETADATA);
  invocation.handler = PopulateNodeMetadata;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(
      LoadNativeField<DXGK_ENGINE_TYPE>(
          metadata, kNodeDataOffset + offsetof(DXGK_NODEMETADATA, EngineType)),
      DXGK_ENGINE_TYPE_3D);
  EXPECT_EQ(ReadWideString(
                static_cast<const std::byte*>(metadata) + kFriendlyNameOffset,
                DXGK_MAX_METADATA_NAME_LENGTH),
            L"Compute \u03A9");
  EXPECT_EQ(LoadNativeField<DXGK_NODEMETADATA_FLAGS>(
                metadata, kNodeDataOffset + offsetof(DXGK_NODEMETADATA, Flags))
                .Value,
            0x12345678u);
  EXPECT_EQ(LoadNativeField<BOOLEAN>(
                metadata,
                kNodeDataOffset + offsetof(DXGK_NODEMETADATA, GpuMmuSupported)),
            1u);
  EXPECT_EQ(LoadNativeField<BOOLEAN>(
                metadata,
                kNodeDataOffset + offsetof(DXGK_NODEMETADATA, IoMmuSupported)),
            1u);
}

struct RegistryStringState {
  // Original native allocation size used to derive the wire capacity.
  uint32_t native_size;
};

uint32_t PopulateRegistryString(void* context, uint64_t, void* arguments) {
  auto* state = static_cast<RegistryStringState*>(context);
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  auto* wire = static_cast<QueryRegistryInfoWire*>(query->pPrivateDriverData);
  const uint32_t expected_size =
      sizeof(QueryRegistryInfoWire) +
      (state->native_size - sizeof(D3DDDI_QUERYREGISTRY_INFO)) /
          sizeof(wchar_t) * sizeof(Utf16);
  EXPECT_EQ(query->PrivateDriverDataSize, expected_size);
  EXPECT_EQ(wire->query_type, D3DDDI_QUERYREGISTRY_SERVICEKEY);
  EXPECT_EQ(std::u16string(reinterpret_cast<const char16_t*>(wire->value_name)),
            u"ReleaseVersion");
  static constexpr char16_t value[] = u"version \U0001F642";
  std::memcpy(wire->value.output_string, value, sizeof(value));
  wire->output_value_size = sizeof(value);
  wire->status = D3DDDI_QUERYREGISTRY_STATUS_SUCCESS;
  return 0;
}

TEST_F(DxgKmtQueryTest, ConvertsDynamicRegistryStringsAndSurrogatePairs) {
  alignas(std::max_align_t)
      std::array<std::byte, sizeof(D3DDDI_QUERYREGISTRY_INFO) + 256>
          native_storage = {};
  auto* registry =
      reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(native_storage.data());
  registry->QueryType = D3DDDI_QUERYREGISTRY_SERVICEKEY;
  registry->ValueType = 1;  // REG_SZ
  std::wmemcpy(registry->ValueName, L"ReleaseVersion", 15);
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_QUERYREGISTRY,
      .pPrivateDriverData = registry,
      .PrivateDriverDataSize = native_storage.size(),
  };
  RegistryStringState state = {
      .native_size = query.PrivateDriverDataSize,
  };
  invocation.handler = PopulateRegistryString;
  invocation.handler_context = &state;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(registry->Status, D3DDDI_QUERYREGISTRY_STATUS_SUCCESS);
  EXPECT_EQ(registry->OutputValueSize, sizeof(L"version \U0001F642"));
  EXPECT_EQ(ReadWideString(registry->OutputString,
                           registry->OutputValueSize / sizeof(wchar_t)),
            L"version \U0001F642");
}

uint32_t PopulateRegistryBinary(void*, uint64_t, void* arguments) {
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  auto* wire = static_cast<QueryRegistryInfoWire*>(query->pPrivateDriverData);
  EXPECT_EQ(query->PrivateDriverDataSize, sizeof(QueryRegistryInfoWire) + 32u);
  wire->value.output_dword = 0x12345678u;
  wire->output_value_size = sizeof(wire->value.output_dword);
  wire->status = D3DDDI_QUERYREGISTRY_STATUS_SUCCESS;
  return 0;
}

TEST_F(DxgKmtQueryTest, PreservesDynamicRegistryBinaryValues) {
  alignas(std::max_align_t)
      std::array<std::byte, sizeof(D3DDDI_QUERYREGISTRY_INFO) + 32>
          native_storage = {};
  auto* registry =
      reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(native_storage.data());
  registry->QueryType = D3DDDI_QUERYREGISTRY_SERVICEKEY;
  registry->ValueType = 4;  // REG_DWORD
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_QUERYREGISTRY,
      .pPrivateDriverData = registry,
      .PrivateDriverDataSize = native_storage.size(),
  };
  invocation.handler = PopulateRegistryBinary;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(registry->Status, D3DDDI_QUERYREGISTRY_STATUS_SUCCESS);
  EXPECT_EQ(registry->OutputValueSize, sizeof(uint32_t));
  EXPECT_EQ(registry->OutputDword, 0x12345678u);
}

struct RegistrySequenceState {
  // Query result selected for this invocation.
  D3DDDI_QUERYREGISTRY_STATUS status;
};

uint32_t PopulateRegistrySequence(void* context, uint64_t, void* arguments) {
  auto* state = static_cast<RegistrySequenceState*>(context);
  auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
  auto* wire = static_cast<QueryRegistryInfoWire*>(query->pPrivateDriverData);
  wire->status = state->status;
  if (state->status == D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW) {
    wire->output_value_size = 40;
    return 0;
  }
  static constexpr char16_t values[] = u"one\0two\0";
  std::memcpy(wire->value.output_string, values, sizeof(values));
  wire->output_value_size = sizeof(values);
  return 0;
}

TEST_F(DxgKmtQueryTest, ConvertsRegistrySequencesAndRequiredSizes) {
  alignas(std::max_align_t)
      std::array<std::byte, sizeof(D3DDDI_QUERYREGISTRY_INFO) + 128>
          native_storage = {};
  auto* registry =
      reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(native_storage.data());
  registry->QueryType = D3DDDI_QUERYREGISTRY_SERVICEKEY;
  registry->ValueType = 7;  // REG_MULTI_SZ
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_QUERYREGISTRY,
      .pPrivateDriverData = registry,
      .PrivateDriverDataSize = native_storage.size(),
  };
  RegistrySequenceState state = {
      .status = D3DDDI_QUERYREGISTRY_STATUS_SUCCESS,
  };
  invocation.handler = PopulateRegistrySequence;
  invocation.handler_context = &state;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(registry->OutputValueSize, sizeof(L"one\0two\0"));
  EXPECT_EQ(ReadWideString(registry->OutputString, 9), L"one");
  EXPECT_EQ(ReadWideString(
                reinterpret_cast<const std::byte*>(registry->OutputString) +
                    4 * sizeof(wchar_t),
                5),
            L"two");
  wchar_t terminator = 1;
  std::memcpy(&terminator,
              reinterpret_cast<const std::byte*>(registry->OutputString) +
                  8 * sizeof(wchar_t),
              sizeof(terminator));
  EXPECT_EQ(terminator, L'\0');

  state.status = D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW;
  registry->OutputValueSize = 0;
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query), 0);
  EXPECT_EQ(registry->Status, D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW);
  EXPECT_EQ(registry->OutputValueSize, 80u);
}

void* AMDF_CALL RejectAllocation(void*, uint64_t, uint64_t) { return nullptr; }

void AMDF_CALL RejectAllocationFree(void*, void*) {
  ADD_FAILURE() << "failed allocation unexpectedly published storage";
}

TEST_F(DxgKmtQueryTest, RejectsInvalidAndUnallocatableConversionsBeforeIoctl) {
  D3DKMT_ADAPTERREGISTRYINFO registry_info = {};
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_ADAPTERREGISTRYINFO,
      .pPrivateDriverData = &registry_info,
      .PrivateDriverDataSize = sizeof(registry_info) - 1,
  };
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query),
            STATUS_INVALID_PARAMETER);
  EXPECT_EQ(invocation.count, 0u);

  alignas(std::max_align_t)
      std::array<std::byte, sizeof(D3DDDI_QUERYREGISTRY_INFO) + 32>
          native_storage = {};
  auto* registry =
      reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(native_storage.data());
  registry->QueryType = D3DDDI_QUERYREGISTRY_SERVICEKEY;
  registry->ValueType = 1;
  registry->ValueName[0] = static_cast<wchar_t>(0x1F642u);
  query.Type = KMTQAITYPE_QUERYREGISTRY;
  query.pPrivateDriverData = registry;
  query.PrivateDriverDataSize = native_storage.size();
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query),
            STATUS_INVALID_PARAMETER);
  EXPECT_EQ(invocation.count, 0u);

  const amdf_allocator_t rejecting_allocator = {
      .user_data = nullptr,
      .allocate = RejectAllocation,
      .resize = nullptr,
      .free = RejectAllocationFree,
  };
  kmt_.host_allocator = &rejecting_allocator;
  query.Type = KMTQAITYPE_ADAPTERREGISTRYINFO;
  query.pPrivateDriverData = &registry_info;
  query.PrivateDriverDataSize = sizeof(registry_info);
  EXPECT_EQ(amdf_linux_dxg_kmt_query_adapter_info(&kmt_, &query),
            STATUS_NO_MEMORY);
  EXPECT_EQ(invocation.count, 0u);
}

TEST_F(DxgKmtQueryTest, BridgeDispatchRoutesOnlyQueryAdapterConversion) {
  D3DKMT_ADAPTERREGISTRYINFO registry_info = {};
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = 3,
      .Type = KMTQAITYPE_ADAPTERREGISTRYINFO,
      .pPrivateDriverData = &registry_info,
      .PrivateDriverDataSize = sizeof(registry_info),
  };
  ExpectedLayout expected = {
      .native_data = &registry_info,
      .wire_size = sizeof(AdapterRegistryInfoWire),
  };
  invocation.handler = CheckLayout;
  invocation.handler_context = &expected;
  EXPECT_EQ(amdf_linux_dxg_kmt_invoke(
                &kmt_, _IOWR('G', 0x09, D3DKMT_QUERYADAPTERINFO), &query),
            static_cast<uint32_t>(STATUS_UNSUCCESSFUL));

  invocation.handler = nullptr;
  invocation.result = 0xC0000001u;
  uint32_t arguments = 0;
  EXPECT_EQ(amdf_linux_dxg_kmt_invoke(&kmt_, 0x12345678u, &arguments),
            invocation.result);
  EXPECT_EQ(invocation.connection, &connection_);
  EXPECT_EQ(invocation.request, 0x12345678u);
  EXPECT_EQ(invocation.arguments, &arguments);
}

}  // namespace

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  invocation.connection =
      static_cast<const amdf_linux_dxg_connection_t*>(context);
  invocation.request = request;
  invocation.arguments = arguments;
  ++invocation.count;
  return invocation.handler == nullptr
             ? invocation.result
             : invocation.handler(invocation.handler_context, request,
                                  arguments);
}
