// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/path.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

static bool operator==(const iree_string_pair_t& lhs,
                       const iree_string_pair_t& rhs) noexcept {
  return iree_string_view_equal(lhs.key, rhs.key) &&
         iree_string_view_equal(lhs.value, rhs.value);
}

static std::ostream& operator<<(std::ostream& os,
                                const iree_string_pair_t& pair) {
  return os << std::string(pair.key.data, pair.key.size) << "="
            << std::string(pair.value.data, pair.value.size);
}

namespace {

using ::iree::StatusCode;
using ::iree::testing::status::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::IsEmpty;

#define _SV(str) iree_make_cstring_view(str)

#define EXPECT_SV_EQ(actual, expected) \
  EXPECT_TRUE(iree_string_view_equal(actual, expected))

TEST(FilePathTest, Canonicalize) {
  auto canonicalize = [](std::string value) {
    value.resize(
        iree_file_path_canonicalize((char*)value.data(), value.size()));
    return value;
  };
  EXPECT_EQ(canonicalize(""), "");
  EXPECT_EQ(canonicalize("a"), "a");
  EXPECT_EQ(canonicalize("ab"), "ab");

#if defined(IREE_PLATFORM_WINDOWS)
  EXPECT_EQ(canonicalize("/"), "\\");
  EXPECT_EQ(canonicalize("\\"), "\\");
  EXPECT_EQ(canonicalize("a/b"), "a\\b");
  EXPECT_EQ(canonicalize("a//b"), "a\\b");
  EXPECT_EQ(canonicalize("a////b"), "a\\b");
  EXPECT_EQ(canonicalize("a\\//b"), "a\\b");
  EXPECT_EQ(canonicalize("a\\\\b"), "a\\b");
  EXPECT_EQ(canonicalize("\\a"), "\\a");
  EXPECT_EQ(canonicalize("/a"), "\\a");
  EXPECT_EQ(canonicalize("//a"), "\\a");
  EXPECT_EQ(canonicalize("a/"), "a\\");
  EXPECT_EQ(canonicalize("a//"), "a\\");
#else
  EXPECT_EQ(canonicalize("/"), "/");
  EXPECT_EQ(canonicalize("a/b"), "a/b");
  EXPECT_EQ(canonicalize("a//b"), "a/b");
  EXPECT_EQ(canonicalize("a////b"), "a/b");
  EXPECT_EQ(canonicalize("/a"), "/a");
  EXPECT_EQ(canonicalize("//a"), "/a");
  EXPECT_EQ(canonicalize("a/"), "a/");
  EXPECT_EQ(canonicalize("a//"), "a/");
#endif  // IREE_PLATFORM_WINDOWS
}

static std::string JoinPaths(std::string lhs, std::string rhs) {
  char* result_str = NULL;
  IREE_IGNORE_ERROR(
      iree_file_path_join(iree_make_string_view(lhs.data(), lhs.size()),
                          iree_make_string_view(rhs.data(), rhs.size()),
                          iree_allocator_system(), &result_str));
  std::string result;
  result.resize(strlen(result_str));
  memcpy((char*)result.data(), result_str, result.size());
  iree_allocator_free(iree_allocator_system(), result_str);
  return result;
}

TEST(FilePathTest, JoinPathsEmpty) {
  EXPECT_EQ(JoinPaths("", ""), "");
  EXPECT_EQ(JoinPaths("", "bar"), "bar");
  EXPECT_EQ(JoinPaths("foo", ""), "foo");
}

TEST(FilePathTest, JoinPathsSlash) {
  EXPECT_EQ(JoinPaths("foo", "bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo", "bar/"), "foo/bar/");
  EXPECT_EQ(JoinPaths("foo", "/bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo", "/bar/"), "foo/bar/");

  EXPECT_EQ(JoinPaths("foo/", "bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo/", "bar/"), "foo/bar/");
  EXPECT_EQ(JoinPaths("foo/", "/bar"), "foo/bar");
  EXPECT_EQ(JoinPaths("foo/", "/bar/"), "foo/bar/");

  EXPECT_EQ(JoinPaths("/foo", "bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo", "bar/"), "/foo/bar/");
  EXPECT_EQ(JoinPaths("/foo", "/bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo", "/bar/"), "/foo/bar/");

  EXPECT_EQ(JoinPaths("/foo/", "bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo/", "bar/"), "/foo/bar/");
  EXPECT_EQ(JoinPaths("/foo/", "/bar"), "/foo/bar");
  EXPECT_EQ(JoinPaths("/foo/", "/bar/"), "/foo/bar/");
}

TEST(FilePathTest, JoinPathsDoubleSlash) {
  EXPECT_EQ(JoinPaths("foo//", "bar"), "foo//bar");
  EXPECT_EQ(JoinPaths("foo", "//bar"), "foo//bar");
}

TEST(FilePathTest, DirnameEmpty) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("")), _SV(""));
}

TEST(FilePathTest, DirnameAbsolute) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/")), _SV("/"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo")), _SV("/"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/")), _SV("/foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/bar")), _SV("/foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("/foo/bar/")), _SV("/foo/bar"));
}

TEST(FilePathTest, DirnameRelative) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/bar")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo/bar/")), _SV("foo/bar"));
}

TEST(FilePathTest, DirnameDoubleSlash) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo//")), _SV("foo/"));
}

#if defined(IREE_PLATFORM_WINDOWS)
TEST(FilePathTest, DirnameWindowsNative) {
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("\\")), _SV("\\"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("\\foo")), _SV("\\"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("foo\\bar")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("C:\\foo")), _SV("C:\\"));
  EXPECT_SV_EQ(iree_file_path_dirname(_SV("C:\\foo\\bar")), _SV("C:\\foo"));
}
#endif  // IREE_PLATFORM_WINDOWS

TEST(FilePathTest, BasenameEmpty) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("")), _SV(""));
}

TEST(FilePathTest, BasenameAbsolute) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("/foo/bar/")), _SV(""));
}

TEST(FilePathTest, BasenameRelative) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo/bar/")), _SV(""));
}

TEST(FilePathTest, BasenameDoubleSlash) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo//")), _SV(""));
}

#if defined(IREE_PLATFORM_WINDOWS)
TEST(FilePathTest, BasenameWindowsNative) {
  EXPECT_SV_EQ(iree_file_path_basename(_SV("\\")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("\\foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("foo\\bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("C:\\foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_basename(_SV("C:\\foo\\bar")), _SV("bar"));
}
#endif  // IREE_PLATFORM_WINDOWS

TEST(FilePathTest, Stem) {
  EXPECT_SV_EQ(iree_file_path_stem(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo.")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo.bar")), _SV("foo"));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo..")), _SV("foo."));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("foo..bar")), _SV("foo."));
  EXPECT_SV_EQ(iree_file_path_stem(_SV(".bar")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_stem(_SV("..bar")), _SV("."));
}

TEST(FilePathTest, Extension) {
  EXPECT_SV_EQ(iree_file_path_extension(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo.")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo.bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo..")), _SV(""));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("foo..bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV(".bar")), _SV("bar"));
  EXPECT_SV_EQ(iree_file_path_extension(_SV("..bar")), _SV("bar"));
}

TEST(FilePathTest, IsDynamicLibrary) {
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("/opt/rocm/lib")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("foo")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("foo.so.")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("foo.so..1")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("foo.so.1a")));
  EXPECT_FALSE(iree_file_path_is_dynamic_library(_SV("foo.so.backup")));

  EXPECT_TRUE(iree_file_path_is_dynamic_library(_SV("foo.dll")));
  EXPECT_TRUE(iree_file_path_is_dynamic_library(_SV("foo.dylib")));
  EXPECT_TRUE(iree_file_path_is_dynamic_library(_SV("foo.so")));
  EXPECT_TRUE(iree_file_path_is_dynamic_library(_SV("foo.so.1")));
  EXPECT_TRUE(iree_file_path_is_dynamic_library(_SV("foo.so.1.2.3")));
  EXPECT_TRUE(iree_file_path_is_dynamic_library(
      _SV("/opt/rocm/lib/libhsa-runtime64.so.1")));
}

#if defined(IREE_PLATFORM_WINDOWS)

typedef struct counting_allocator_t {
  // Allocator receiving forwarded commands.
  iree_allocator_t delegate;
  // Number of allocation or reallocation commands received.
  int allocation_count;
  // Number of free commands received.
  int free_count;
} counting_allocator_t;

static iree_status_t counting_allocator_ctl(void* self,
                                            iree_allocator_command_t command,
                                            const void* params,
                                            void** inout_ptr) {
  counting_allocator_t* allocator = (counting_allocator_t*)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_count;
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_count;
      break;
    default:
      break;
  }
  return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                 inout_ptr);
}

TEST(FilePathTest, ToWin32ConvertsRelativeUtf8Path) {
  wchar_t* converted_path = NULL;
  IREE_ASSERT_OK(
      iree_file_path_to_win32(IREE_SV("relative/\xE8\xB7\xAF\xE5\xBE\x84"),
                              iree_allocator_system(), &converted_path));
  ASSERT_NE(converted_path, nullptr);

  std::wstring path(converted_path);
  iree_allocator_free(iree_allocator_system(), converted_path);
  EXPECT_EQ(path.substr(0, 4), L"\\\\?\\");
  const std::wstring expected_suffix = L"\\relative\\\u8DEF\u5F84";
  ASSERT_GE(path.size(), expected_suffix.size());
  EXPECT_EQ(path.compare(path.size() - expected_suffix.size(),
                         expected_suffix.size(), expected_suffix),
            0);
}

TEST(FilePathTest, ToWin32ConvertsUncPath) {
  wchar_t* converted_path = NULL;
  IREE_ASSERT_OK(iree_file_path_to_win32(IREE_SV("\\\\server\\share\\path"),
                                         iree_allocator_system(),
                                         &converted_path));
  ASSERT_NE(converted_path, nullptr);

  EXPECT_EQ(std::wstring(converted_path), L"\\\\?\\UNC\\server\\share\\path");
  iree_allocator_free(iree_allocator_system(), converted_path);
}

TEST(FilePathTest, ToWin32PreservesNamespacePaths) {
  auto expect_preserved = [](iree_string_view_t input,
                             const wchar_t* expected) {
    wchar_t* converted_path = NULL;
    IREE_ASSERT_OK(iree_file_path_to_win32(input, iree_allocator_system(),
                                           &converted_path));
    ASSERT_NE(converted_path, nullptr);
    EXPECT_EQ(std::wstring(converted_path), expected);
    iree_allocator_free(iree_allocator_system(), converted_path);
  };
  expect_preserved(IREE_SV("\\\\?\\C:\\extended\\path"),
                   L"\\\\?\\C:\\extended\\path");
  expect_preserved(IREE_SV("\\\\.\\pipe\\device"), L"\\\\.\\pipe\\device");
}

TEST(FilePathTest, ToWin32ConvertsPathBeyondCommonLimit) {
  std::string input = "C:\\";
  while (input.size() <= IREE_MAX_PATH + 64) {
    input.append("segment\\");
  }
  input.append("file.bin");
  ASSERT_GT(input.size(), IREE_MAX_PATH);

  counting_allocator_t allocator_state = {
      /*.delegate=*/iree_allocator_system(),
      /*.allocation_count=*/0,
      /*.free_count=*/0,
  };
  iree_allocator_t allocator = {
      /*.self=*/&allocator_state,
      /*.ctl=*/counting_allocator_ctl,
  };
  wchar_t* converted_path = NULL;
  IREE_ASSERT_OK(
      iree_file_path_to_win32(iree_make_string_view(input.data(), input.size()),
                              allocator, &converted_path));
  ASSERT_NE(converted_path, nullptr);
  EXPECT_EQ(allocator_state.allocation_count, 1);

  std::wstring path(converted_path);
  iree_allocator_free(allocator, converted_path);
  EXPECT_EQ(allocator_state.free_count, 1);
  EXPECT_EQ(path.substr(0, 7), L"\\\\?\\C:\\");
  EXPECT_EQ(path.size(), input.size() + 4);
}

TEST(FilePathTest, ToWin32RejectsInvalidUtf8AndEmbeddedNul) {
  wchar_t* converted_path = NULL;
  EXPECT_THAT(iree_file_path_to_win32(IREE_SV(""), iree_allocator_system(),
                                      &converted_path),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(converted_path, nullptr);

  EXPECT_THAT(iree_file_path_to_win32(iree_make_string_view(nullptr, 1),
                                      iree_allocator_system(), &converted_path),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(converted_path, nullptr);

  const char invalid_utf8[] = {'p', 'a', 't', 'h', (char)0xFF};
  EXPECT_THAT(iree_file_path_to_win32(
                  iree_make_string_view(invalid_utf8, sizeof(invalid_utf8)),
                  iree_allocator_system(), &converted_path),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(converted_path, nullptr);

  const char embedded_nul[] = {'a', '\0', 'b'};
  EXPECT_THAT(iree_file_path_to_win32(
                  iree_make_string_view(embedded_nul, sizeof(embedded_nul)),
                  iree_allocator_system(), &converted_path),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(converted_path, nullptr);
}

#endif  // IREE_PLATFORM_WINDOWS

// NOTE: these URI methods are all implemented using the same iree_uri_split and
// we test each independently because it's easier.

TEST(URITest, Schema) {
  EXPECT_SV_EQ(iree_uri_schema(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_uri_schema(_SV("s")), _SV("s"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema:")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema:path")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema:/")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema:///")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema:///path")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path/")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path/sub")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path?")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path?p")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path?params")), _SV("schema"));
  EXPECT_SV_EQ(iree_uri_schema(_SV("schema://path?params??")), _SV("schema"));
}

TEST(URITest, Path) {
  EXPECT_SV_EQ(iree_uri_path(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("s")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema:")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema:path")), _SV("path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema:/")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://")), _SV(""));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema:///")), _SV("/"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema:///path")), _SV("/path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path")), _SV("path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path/")), _SV("path/"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path/sub")), _SV("path/sub"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path?")), _SV("path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path?p")), _SV("path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path?params")), _SV("path"));
  EXPECT_SV_EQ(iree_uri_path(_SV("schema://path?params??")), _SV("path"));
}

TEST(URITest, Params) {
  EXPECT_SV_EQ(iree_uri_params(_SV("s")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema:")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema:path")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema:/")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema:///")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema:///path")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path/")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path/sub")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path?")), _SV(""));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path?p")), _SV("p"));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path?params")), _SV("params"));
  EXPECT_SV_EQ(iree_uri_params(_SV("schema://path?params??")), _SV("params??"));
}

using StringPairs = std::vector<iree_string_pair_t>;
static StringPairs SplitParams(iree_string_view_t params) {
  StringPairs storage;
  iree_host_size_t count = 0;
  while (
      !iree_uri_split_params(params, storage.size(), &count, storage.data())) {
    storage.resize(count);
  }
  return storage;
}

#define StringPair iree_make_cstring_pair

TEST(URITest, SplitParams) {
  EXPECT_THAT(SplitParams(_SV("")), IsEmpty());
  EXPECT_THAT(SplitParams(_SV("&")), IsEmpty());
  EXPECT_THAT(SplitParams(_SV("a")), ElementsAreArray({StringPair("a", "")}));
  EXPECT_THAT(SplitParams(_SV("&a")), ElementsAreArray({StringPair("a", "")}));
  EXPECT_THAT(SplitParams(_SV("a&")), ElementsAreArray({StringPair("a", "")}));
  EXPECT_THAT(SplitParams(_SV("&a&")), ElementsAreArray({StringPair("a", "")}));
  EXPECT_THAT(SplitParams(_SV("a=")), ElementsAreArray({StringPair("a", "")}));
  EXPECT_THAT(SplitParams(_SV("a=b")),
              ElementsAreArray({StringPair("a", "b")}));
  EXPECT_THAT(SplitParams(_SV("a=b&c")), ElementsAreArray({
                                             StringPair("a", "b"),
                                             StringPair("c", ""),
                                         }));
  EXPECT_THAT(SplitParams(_SV("a=b&c=")), ElementsAreArray({
                                              StringPair("a", "b"),
                                              StringPair("c", ""),
                                          }));
  EXPECT_THAT(SplitParams(_SV("a=b&c=d")), ElementsAreArray({
                                               StringPair("a", "b"),
                                               StringPair("c", "d"),
                                           }));
  EXPECT_THAT(SplitParams(_SV("a=b&c=d&e&f&g=h")), ElementsAreArray({
                                                       StringPair("a", "b"),
                                                       StringPair("c", "d"),
                                                       StringPair("e", ""),
                                                       StringPair("f", ""),
                                                       StringPair("g", "h"),
                                                   }));
}

}  // namespace
