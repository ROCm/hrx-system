// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/dynamic_library.h"

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/testing/dynamic_library_test_library_embed.h"
#include "iree/io/file_contents.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <psapi.h>

#include "iree/base/internal/symbol_resolver_win32.h"
#endif  // IREE_PLATFORM_WINDOWS
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE)
#include <signal.h>
#include <sys/resource.h>
#endif

#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_LINUX) || \
    defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE)
#include <filesystem>
#endif

namespace iree {
namespace {

using iree::testing::status::StatusIs;

static const char* kUnknownName = "library_that_does_not_exist.so";

class DynamicLibraryTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    // Making files available to tests, particularly across operating systems
    // and build tools (Bazel/CMake) is complicated. Rather than include a test
    // dynamic library as a "testdata" file, we use c_embed_data to package
    // the file so it's embedded in a C module, then write that embedded file
    // to a platform/test-environment specific temp file for loading.

    // System APIs for loading dynamic libraries typically require an extension.
#if defined(IREE_PLATFORM_WINDOWS)
    static constexpr const char* ext = ".dll";
#else
    static constexpr const char* ext = ".so";
#endif
    library_temp_path_ =
        iree::testing::TempFilePath("iree_dynamic_library_test", ext);

    const struct iree_file_toc_t* file_toc =
        dynamic_library_test_library_create();
    IREE_ASSERT_OK(iree_io_file_contents_write(
        library_temp_path_.path_view(),
        iree_make_const_byte_span(file_toc->data, file_toc->size),
        iree_allocator_system()));

    std::cout << "Embedded test library written to temp path: "
              << library_temp_path_.path() << "\n";
  }

  static iree::testing::TempFilePath library_temp_path_;
};

iree::testing::TempFilePath DynamicLibraryTest::library_temp_path_;

TEST_F(DynamicLibraryTest, LoadLibrarySuccess) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));
  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, LoadLibraryFailure) {
  iree_dynamic_library_t* library = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_dynamic_library_load_from_file(
                            kUnknownName, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                            iree_allocator_system(), &library));
}

TEST_F(DynamicLibraryTest, LoadLibraryFromMemory) {
  const auto* file = dynamic_library_test_library_create();
  iree_dynamic_library_t* library = nullptr;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_memory(
      IREE_SV("embedded_test_library"),
      iree_make_const_byte_span(file->data, file->size),
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(), &library));
  std::unique_ptr<iree_dynamic_library_t,
                  decltype(&iree_dynamic_library_release)>
      owned_library(library, iree_dynamic_library_release);
  int (*function)(int) = nullptr;
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
      library, "times_two", reinterpret_cast<void**>(&function)));
  EXPECT_EQ(246, function(123));
}

#if defined(IREE_PLATFORM_WINDOWS)
TEST_F(DynamicLibraryTest, AttachSymbolsFromMemory) {
  const auto* files = dynamic_library_test_library_create();
  ASSERT_EQ(2u, dynamic_library_test_library_size());
  iree_dynamic_library_t* library = nullptr;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_memory(
      IREE_SV("embedded_symbols"),
      iree_make_const_byte_span(files[0].data, files[0].size),
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(), &library));
  std::unique_ptr<iree_dynamic_library_t,
                  decltype(&iree_dynamic_library_release)>
      owned_library(library, iree_dynamic_library_release);
  using PrivateFunction = int (*)(int);
  PrivateFunction (*get_private_function)() = nullptr;
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
      library, "get_private_function",
      reinterpret_cast<void**>(&get_private_function)));
  PrivateFunction private_function = get_private_function();
  EXPECT_EQ(246, private_function(123));

  HMODULE module = nullptr;
  ASSERT_TRUE(
      GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(private_function), &module));
  MODULEINFO mapping = {};
  ASSERT_TRUE(GetModuleInformation(GetCurrentProcess(), module, &mapping,
                                   sizeof(mapping)));
  char module_path[MAX_PATH];
  DWORD path_length =
      GetModuleFileNameA(module, module_path, sizeof(module_path));
  ASSERT_GT(path_length, 0u);
  ASSERT_LT(path_length, sizeof(module_path));
  const iree_symbol_resolver_t* resolver = nullptr;
  ASSERT_EQ(ERROR_SUCCESS, iree_symbol_resolver_get(&resolver));
  const DWORD64 address = reinterpret_cast<DWORD64>(private_function);
  alignas(SYMBOL_INFO) unsigned char
      symbol_storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
  auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_storage);
  symbol->SizeOfStruct = sizeof(*symbol);
  symbol->MaxNameLen = MAX_SYM_NAME;
  DWORD64 symbol_displacement = 0;

  // A profiler scan can register the image before its debug bytes arrive.
  iree_symbol_resolver_lock();
  DWORD64 registered = resolver->SymLoadModuleEx(
      resolver->process, nullptr, module_path, nullptr,
      reinterpret_cast<DWORD64>(mapping.lpBaseOfDll), mapping.SizeOfImage,
      nullptr, 0);
  DWORD register_error = registered ? ERROR_SUCCESS : GetLastError();
  IMAGEHLP_LINE64 line = {};
  line.SizeOfStruct = sizeof(line);
  DWORD line_displacement = 0;
  BOOL found_line_before = resolver->SymGetLineFromAddr64(
      resolver->process, address, &line_displacement, &line);
  BOOL found_symbol_before = resolver->SymFromAddr(
      resolver->process, address, &symbol_displacement, symbol);
  iree_symbol_resolver_unlock();
  ASSERT_EQ(ERROR_SUCCESS, register_error);
  ASSERT_FALSE(found_line_before);
  if (found_symbol_before) {
    ASSERT_STRNE("private_times_two", symbol->Name);
  }

  IREE_ASSERT_OK(iree_dynamic_library_attach_symbols_from_memory(
      library, iree_make_const_byte_span(files[1].data, files[1].size)));

  iree_symbol_resolver_lock();
  BOOL found_symbol = resolver->SymFromAddr(resolver->process, address,
                                            &symbol_displacement, symbol);
  BOOL found_line = resolver->SymGetLineFromAddr64(resolver->process, address,
                                                   &line_displacement, &line);
  std::string filename = found_line ? line.FileName : "";
  iree_symbol_resolver_unlock();
  ASSERT_TRUE(found_symbol);
  EXPECT_STREQ("private_times_two", symbol->Name);
  EXPECT_TRUE(found_line);
  EXPECT_GT(line.LineNumber, 0u);
  EXPECT_THAT(filename,
              ::testing::HasSubstr("dynamic_library_test_library.cc"));

  // Qualify the native PDB ownership/registration mechanism with the real
  // image.
  auto pdb_path = iree::testing::TempFilePath("owned_debug_database", ".pdb");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      pdb_path.path_view(),
      iree_make_const_byte_span(files[1].data, files[1].size),
      iree_allocator_system()));
  HANDLE pdb_handle =
      CreateFileA(pdb_path.path().c_str(), GENERIC_READ | DELETE,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  ASSERT_NE(INVALID_HANDLE_VALUE, pdb_handle) << GetLastError();
  std::unique_ptr<void, decltype(&CloseHandle)> pdb(pdb_handle, CloseHandle);
  HMODULE dbghelp = GetModuleHandleA("dbghelp.dll");
  ASSERT_NE(nullptr, dbghelp);
  auto get_file_index =
      reinterpret_cast<BOOL(WINAPI*)(PCSTR, PSYMSRV_INDEX_INFO, DWORD)>(
          GetProcAddress(dbghelp, "SymSrvGetFileIndexInfo"));
  auto unload_module = reinterpret_cast<BOOL(WINAPI*)(HANDLE, DWORD64)>(
      GetProcAddress(dbghelp, "SymUnloadModule64"));
  auto get_options = reinterpret_cast<DWORD(WINAPI*)(void)>(
      GetProcAddress(dbghelp, "SymGetOptions"));
  auto set_options = reinterpret_cast<DWORD(WINAPI*)(DWORD)>(
      GetProcAddress(dbghelp, "SymSetOptions"));
  ASSERT_NE(nullptr, get_file_index);
  ASSERT_NE(nullptr, unload_module);
  ASSERT_NE(nullptr, get_options);
  ASSERT_NE(nullptr, set_options);
  SYMSRV_INDEX_INFO image_index = {};
  image_index.sizeofstruct = sizeof(image_index);
  SYMSRV_INDEX_INFO pdb_index = {};
  pdb_index.sizeofstruct = sizeof(pdb_index);
  IMAGEHLP_MODULE64 registered_info = {};
  registered_info.SizeOfStruct = sizeof(registered_info);
  iree_symbol_resolver_lock();
  BOOL indexed_image = get_file_index(module_path, &image_index, 0);
  DWORD image_index_error = indexed_image ? ERROR_SUCCESS : GetLastError();
  BOOL indexed_pdb = get_file_index(pdb_path.path().c_str(), &pdb_index, 0);
  DWORD pdb_index_error = indexed_pdb ? ERROR_SUCCESS : GetLastError();
  BOOL unloaded = unload_module(resolver->process,
                                reinterpret_cast<DWORD64>(mapping.lpBaseOfDll));
  DWORD unload_error = unloaded ? ERROR_SUCCESS : GetLastError();
  DWORD options = get_options();
  set_options(options & ~SYMOPT_DEFERRED_LOADS);
  DWORD64 loaded = resolver->SymLoadModuleEx(
      resolver->process, pdb.get(), pdb_path.path().c_str(), nullptr,
      reinterpret_cast<DWORD64>(mapping.lpBaseOfDll), mapping.SizeOfImage,
      nullptr, 0);
  DWORD load_error = loaded ? ERROR_SUCCESS : GetLastError();
  set_options(options);
  BOOL found_module = resolver->SymGetModuleInfo64(resolver->process, address,
                                                   &registered_info);
  found_symbol = resolver->SymFromAddr(resolver->process, address,
                                       &symbol_displacement, symbol);
  found_line = resolver->SymGetLineFromAddr64(resolver->process, address,
                                              &line_displacement, &line);
  filename = found_line ? line.FileName : "";
  iree_symbol_resolver_unlock();
  std::cout << "native PDB: image_index_error=" << image_index_error
            << " pdb_index_error=" << pdb_index_error
            << " unload_error=" << unload_error << " load_error=" << load_error
            << " symbol_type=" << registered_info.SymType << "\n";
  EXPECT_TRUE(indexed_image);
  EXPECT_TRUE(indexed_pdb);
  EXPECT_EQ(0, memcmp(&image_index.guid, &pdb_index.guid, sizeof(GUID)));
  EXPECT_EQ(image_index.age, pdb_index.age);
  EXPECT_TRUE(unloaded);
  EXPECT_NE(0u, loaded);
  EXPECT_TRUE(found_module);
  EXPECT_EQ(SymPdb, registered_info.SymType);
  EXPECT_TRUE(found_symbol);
  EXPECT_STREQ("private_times_two", symbol->Name);
  EXPECT_TRUE(found_line);
  EXPECT_THAT(filename,
              ::testing::HasSubstr("dynamic_library_test_library.cc"));
}
#endif  // IREE_PLATFORM_WINDOWS

TEST_F(DynamicLibraryTest, LoadLibraryTwice) {
  iree_dynamic_library_t* library1 = NULL;
  iree_dynamic_library_t* library2 = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library1));
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library2));
  iree_dynamic_library_release(library1);
  iree_dynamic_library_release(library2);
}

TEST_F(DynamicLibraryTest, NodeletePreservesModuleState) {
  int (*next_value)() = nullptr;
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NODELETE,
      iree_allocator_system(), &library));
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
      library, "next_value", reinterpret_cast<void**>(&next_value)));
  EXPECT_EQ(1, next_value());
  iree_dynamic_library_release(library);

  library = NULL;
  next_value = nullptr;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NODELETE,
      iree_allocator_system(), &library));
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
      library, "next_value", reinterpret_cast<void**>(&next_value)));
  EXPECT_EQ(2, next_value());
  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, GetSymbolSuccess) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  int (*fn_ptr)(int);
  IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(library, "times_two",
                                                    (void**)&fn_ptr));
  ASSERT_NE(nullptr, fn_ptr);
  EXPECT_EQ(246, fn_ptr(123));

  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, GetSymbolFailure) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  int (*fn_ptr)(int);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_dynamic_library_lookup_symbol(library, "unknown", (void**)&fn_ptr));
  EXPECT_EQ(nullptr, fn_ptr);

  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, TryLookupSymbolSuccess) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  void* symbol = iree_dynamic_library_try_lookup_symbol(library, "times_two");
  EXPECT_NE(nullptr, symbol);

  // Verify the symbol works.
  auto fn_ptr = reinterpret_cast<int (*)(int)>(symbol);
  EXPECT_EQ(246, fn_ptr(123));

  iree_dynamic_library_release(library);
}

TEST_F(DynamicLibraryTest, TryLookupSymbolNotFound) {
  iree_dynamic_library_t* library = NULL;
  IREE_ASSERT_OK(iree_dynamic_library_load_from_file(
      library_temp_path_.path().c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      iree_allocator_system(), &library));

  void* symbol = iree_dynamic_library_try_lookup_symbol(library, "unknown");
  EXPECT_EQ(nullptr, symbol);

  iree_dynamic_library_release(library);
}

#if defined(IREE_PLATFORM_WINDOWS)

static void CheckMemoryExtraction(const char* temp_directory,
                                  const char* mode) {
  ASSERT_TRUE(SetEnvironmentVariableA("TMP", temp_directory));
  const auto* file = dynamic_library_test_library_create();
  if (!strcmp(mode, "native-posix-close")) {
    auto path = std::filesystem::path(temp_directory) / "native_image.dll";
    HANDLE writer =
        CreateFileA(path.string().c_str(), GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    ASSERT_NE(INVALID_HANDLE_VALUE, writer) << GetLastError();
    DWORD written = 0;
    BOOL wrote = WriteFile(writer, file->data, static_cast<DWORD>(file->size),
                           &written, nullptr);
    DWORD write_error = wrote ? ERROR_SUCCESS : GetLastError();
    HANDLE image =
        ReOpenFile(writer, GENERIC_READ | DELETE,
                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                   FILE_FLAG_DELETE_ON_CLOSE);
    DWORD reopen_error =
        image != INVALID_HANDLE_VALUE ? ERROR_SUCCESS : GetLastError();
    CloseHandle(writer);
    ASSERT_TRUE(wrote) << write_error;
    ASSERT_EQ(file->size, written);
    ASSERT_NE(INVALID_HANDLE_VALUE, image) << reopen_error;
    std::unique_ptr<void, decltype(&CloseHandle)> image_owner(image,
                                                              CloseHandle);
    HMODULE module = LoadLibraryA(path.string().c_str());
    ASSERT_NE(nullptr, module) << GetLastError();
    auto function =
        reinterpret_cast<int (*)(int)>(GetProcAddress(module, "times_two"));
    ASSERT_NE(nullptr, function);
    EXPECT_EQ(246, function(123));
    FILE_DISPOSITION_INFO_EX disposition = {};
    disposition.Flags = FILE_DISPOSITION_FLAG_DELETE |
                        FILE_DISPOSITION_FLAG_POSIX_SEMANTICS |
                        FILE_DISPOSITION_FLAG_ON_CLOSE;
    BOOL marked = SetFileInformationByHandle(image, FileDispositionInfoEx,
                                             &disposition, sizeof(disposition));
    DWORD mark_error = marked ? ERROR_SUCCESS : GetLastError();
    std::cout << "native retained image: disposition_error=" << mark_error
              << "\n";
    EXPECT_TRUE(marked);
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_EQ(246, function(123));
    // Exercise OS process-exit cleanup with the image still mapped.
    image_owner.release();
  } else if (!strcmp(mode, "release") || !strcmp(mode, "nodelete") ||
             !strcmp(mode, "nodelete-os-temp")) {
    iree_dynamic_library_flags_t flags =
        strcmp(mode, "release") ? IREE_DYNAMIC_LIBRARY_FLAG_NODELETE
                                : IREE_DYNAMIC_LIBRARY_FLAG_NONE;
    iree_dynamic_library_t* library = nullptr;
    IREE_ASSERT_OK(iree_dynamic_library_load_from_memory(
        IREE_SV("extraction_lifetime"),
        iree_make_const_byte_span(file->data, file->size), flags,
        iree_allocator_system(), &library));
    std::unique_ptr<iree_dynamic_library_t,
                    decltype(&iree_dynamic_library_release)>
        owned_library(library, iree_dynamic_library_release);
    int (*function)(int) = nullptr;
    IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
        library, "times_two", reinterpret_cast<void**>(&function)));
    EXPECT_EQ(246, function(123));
    std::vector<std::filesystem::path> paths;
    for (const auto& entry :
         std::filesystem::directory_iterator(temp_directory)) {
      paths.push_back(entry.path());
    }
    ASSERT_EQ(1u, paths.size());
    owned_library.reset();
    bool retained = flags == IREE_DYNAMIC_LIBRARY_FLAG_NODELETE;
#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION) || \
    defined(IREE_SANITIZER_ADDRESS)
    retained = true;
#endif
    EXPECT_EQ(retained, std::filesystem::exists(paths[0]));
    if (retained) {
      EXPECT_EQ(246, function(123));
    }
  } else {
    const char invalid_image[] = "not a shared library";
    iree_const_byte_span_t data =
        iree_make_const_byte_span(invalid_image, sizeof(invalid_image));
    iree_allocator_t allocator = iree_allocator_system();
    iree_status_code_t expected_status = IREE_STATUS_NOT_FOUND;
    if (!strcmp(mode, "create-failure")) {
      std::string missing_directory = std::string(temp_directory) + "/missing";
      ASSERT_TRUE(SetEnvironmentVariableA("TMP", missing_directory.c_str()));
      data = iree_make_const_byte_span(file->data, file->size);
    } else if (!strcmp(mode, "allocation-failure")) {
      allocator = iree_allocator_null();
      data = iree_make_const_byte_span(file->data, file->size);
      expected_status = IREE_STATUS_INVALID_ARGUMENT;
    } else if (!strcmp(mode, "empty-image")) {
      data = iree_const_byte_span_empty();
    }
    iree_dynamic_library_t* library = nullptr;
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_dynamic_library_load_from_memory(
            IREE_SV("extraction_failure"), data, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
            allocator, &library));
    EXPECT_EQ(nullptr, library);
    EXPECT_TRUE(std::filesystem::is_empty(temp_directory));
  }
}

#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE)

// Each child initializes the loader's environment policy in its own process.
static void CheckMemoryExtraction(const char* temp_directory,
                                  const char* mode) {
  ASSERT_EQ(0, setenv("TMPDIR", temp_directory, 1));
  const bool preserve = !strcmp(mode, "preserve") || !strcmp(mode, "directory");
  if (!strcmp(mode, "discard") || !strcmp(mode, "write-failure")) {
    ASSERT_EQ(0, unsetenv("IREE_PRESERVE_DYLIB_TEMP_FILES"));
  } else {
    const char* value = !strcmp(mode, "directory") ? temp_directory : "1";
    ASSERT_EQ(
        0, setenv("IREE_PRESERVE_DYLIB_TEMP_FILES", preserve ? value : "", 1));
  }

  if (!strcmp(mode, "write-failure")) {
    // A real short write followed by EFBIG must retire the partial image.
    struct rlimit limit;
    ASSERT_EQ(0, getrlimit(RLIMIT_FSIZE, &limit));
    limit.rlim_cur = 1;
    ASSERT_EQ(0, setrlimit(RLIMIT_FSIZE, &limit));
    ASSERT_NE(SIG_ERR, signal(SIGXFSZ, SIG_IGN));
  }

  const auto* file = dynamic_library_test_library_create();
  iree_dynamic_library_t* library = nullptr;
  iree_status_t status = iree_dynamic_library_load_from_memory(
      IREE_SV("memory_extraction"),
      iree_make_const_byte_span(file->data, file->size),
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_system(), &library);
  if (!strcmp(mode, "write-failure")) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
    EXPECT_EQ(nullptr, library);
  } else {
    IREE_ASSERT_OK(status);
    int (*function)(int) = nullptr;
    IREE_ASSERT_OK(iree_dynamic_library_lookup_symbol(
        library, "times_two", reinterpret_cast<void**>(&function)));
    EXPECT_EQ(246, function(123));
    iree_dynamic_library_release(library);
  }

  std::vector<std::filesystem::path> paths;
  for (const auto& entry :
       std::filesystem::directory_iterator(temp_directory)) {
    paths.push_back(entry.path());
  }
  if (preserve) {
    ASSERT_EQ(1u, paths.size());
    iree_io_file_contents_t* contents = nullptr;
    IREE_ASSERT_OK(
        iree_io_file_contents_read(iree_make_cstring_view(paths[0].c_str()),
                                   iree_allocator_system(), &contents));
    EXPECT_EQ(file->size, contents->buffer.data_length);
    if (file->size == contents->buffer.data_length) {
      EXPECT_EQ(0, memcmp(file->data, contents->buffer.data, file->size));
    }
    iree_io_file_contents_free(contents);
  } else {
    EXPECT_TRUE(paths.empty());
    if (!strcmp(mode, "write-failure")) {
      return;
    }

    // Rejected images and an unavailable allocator leave no extraction behind.
    const char invalid_image[] = "not a shared library";
    library = nullptr;
    for (auto image :
         {iree_make_const_byte_span(invalid_image, sizeof(invalid_image)),
          iree_const_byte_span_empty()}) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_NOT_FOUND,
          iree_dynamic_library_load_from_memory(
              IREE_SV("invalid_image"), image, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
              iree_allocator_system(), &library));
      EXPECT_EQ(nullptr, library);
    }
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_dynamic_library_load_from_memory(
            IREE_SV("unavailable_allocator"),
            iree_make_const_byte_span(file->data, file->size),
            IREE_DYNAMIC_LIBRARY_FLAG_NONE, iree_allocator_null(), &library));
    EXPECT_EQ(nullptr, library);
    EXPECT_TRUE(std::filesystem::is_empty(temp_directory));
  }
}

#endif  // IREE_PLATFORM_WINDOWS / POSIX

#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_LINUX) || \
    defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE)

static int MemoryExtractionRole(int argc, char** argv,
                                const char* temp_directory) {
#if defined(IREE_PLATFORM_WINDOWS)
  CheckMemoryExtraction(argv[argc - 2], argv[argc - 1]);
#else
  CheckMemoryExtraction(temp_directory, argv[argc - 1]);
#endif
  return ::testing::Test::HasFailure() ? 1 : 0;
}

static const iree_test_role_t kMemoryExtractionRoles[] = {
    {"memory-extraction", MemoryExtractionRole, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kMemoryExtractionConfig = {
    kMemoryExtractionRoles, IREE_ARRAYSIZE(kMemoryExtractionRoles)};
IREE_COORDINATED_TEST_REGISTER(kMemoryExtractionConfig);

class DynamicLibraryMemoryTest : public ::testing::TestWithParam<const char*> {
};

TEST_P(DynamicLibraryMemoryTest, OwnsExtractedFile) {
  std::string mode = GetParam();
  int argc = iree_coordinated_test_argc();
  char** argv = iree_coordinated_test_argv();
  std::vector<char*> arguments(argv, argv + argc);
#if defined(IREE_PLATFORM_WINDOWS)
  // This directory belongs to the parent, outside the child harness's cleanup.
  // Observing it after process exit proves that Windows retired retained files.
  auto directory = iree::testing::MakeTempFilePath("memory_extraction");
  if (mode == "nodelete-os-temp") {
    directory = (std::filesystem::temp_directory_path() /
                 std::filesystem::path(directory).filename())
                    .string();
  }
  std::cout << "extraction parent directory=" << directory << "\n";
  ASSERT_TRUE(std::filesystem::create_directory(directory));
  arguments.push_back(directory.data());
#endif
  arguments.push_back(mode.data());
  arguments.push_back(nullptr);
  EXPECT_EQ(
      0, iree_coordinated_test_run(static_cast<int>(arguments.size() - 1),
                                   arguments.data(), &kMemoryExtractionConfig));
#if defined(IREE_PLATFORM_WINDOWS)
  bool empty = std::filesystem::is_empty(directory);
  EXPECT_TRUE(empty);
  if (empty) {
    EXPECT_TRUE(std::filesystem::remove(directory));
  }
#endif
}

#if defined(IREE_PLATFORM_WINDOWS)
INSTANTIATE_TEST_SUITE_P(Policies, DynamicLibraryMemoryTest,
                         ::testing::Values("create-failure", "load-failure",
                                           "allocation-failure", "empty-image",
                                           "release", "nodelete",
                                           "nodelete-os-temp",
                                           "native-posix-close"));
#else
INSTANTIATE_TEST_SUITE_P(Policies, DynamicLibraryMemoryTest,
                         ::testing::Values("discard", "empty", "preserve",
                                           "directory", "write-failure"));
#endif

#endif  // IREE_PLATFORM_WINDOWS || POSIX

}  // namespace
}  // namespace iree
