// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::streamoff kMaxEmbeddedFileSize = 100000000;

struct Options {
  // C identifier used as the generated function prefix.
  std::string identifier = "resources";
  // Path to the generated C header.
  std::string output_header;
  // Path to the generated C implementation.
  std::string output_impl;
  // Verbatim path prefix removed from table-of-contents names.
  std::string strip_prefix;
  // Whether table-of-contents names should drop directory components.
  bool flatten = false;
  // Input file paths to embed, in table-of-contents order.
  std::vector<std::string> input_files;
};

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

bool IsIdentifierStart(char value) {
  return std::isalpha(static_cast<unsigned char>(value)) || value == '_';
}

bool IsIdentifierContinue(char value) {
  return std::isalnum(static_cast<unsigned char>(value)) || value == '_';
}

bool IsValidIdentifier(const std::string& identifier) {
  if (identifier.empty() || !IsIdentifierStart(identifier[0])) return false;
  for (char value : identifier) {
    if (!IsIdentifierContinue(value)) return false;
  }
  return true;
}

std::string BaseName(std::string value) {
  const size_t slash_pos = value.find_last_of("/\\");
  if (slash_pos != std::string::npos) {
    value.erase(0, slash_pos + 1);
  }
  return value;
}

std::string TocNameForInput(const Options& options,
                            const std::string& input_file) {
  std::string toc_name = input_file;
  if (!options.strip_prefix.empty() &&
      StartsWith(toc_name, options.strip_prefix)) {
    toc_name.erase(0, options.strip_prefix.size());
  }
  if (options.flatten) {
    toc_name = BaseName(toc_name);
  }
  return toc_name;
}

std::string EscapeCString(const std::string& source) {
  static const char kHexCharacters[] = "0123456789ABCDEF";
  std::string escaped;
  bool last_was_hex_escape = false;
  for (unsigned char value : source) {
    bool is_hex_escape = false;
    switch (value) {
      case '\n':
        escaped.append("\\n");
        break;
      case '\r':
        escaped.append("\\r");
        break;
      case '\t':
        escaped.append("\\t");
        break;
      case '"':
        escaped.append("\\\"");
        break;
      case '\'':
        escaped.append("\\'");
        break;
      case '\\':
        escaped.append("\\\\");
        break;
      default:
        if (!std::isprint(value) ||
            (last_was_hex_escape && std::isxdigit(value))) {
          escaped.append("\\x");
          escaped.push_back(kHexCharacters[value / 16]);
          escaped.push_back(kHexCharacters[value % 16]);
          is_hex_escape = true;
        } else {
          escaped.push_back(static_cast<char>(value));
        }
        break;
    }
    last_was_hex_escape = is_hex_escape;
  }
  return escaped;
}

void WriteExternCOpen(std::ostream& stream) {
  stream << "\n#if defined(__cplusplus)\n";
  stream << "extern \"C\" {\n";
  stream << "#endif  // defined(__cplusplus)\n";
}

void WriteExternCClose(std::ostream& stream) {
  stream << "#if defined(__cplusplus)\n";
  stream << "}  // extern \"C\"\n";
  stream << "#endif  // defined(__cplusplus)\n\n";
}

void WriteTocStruct(std::ostream& stream) {
  stream << "#ifndef IREE_FILE_TOC\n";
  stream << "#define IREE_FILE_TOC\n";
  WriteExternCOpen(stream);
  stream << "typedef struct iree_file_toc_t {\n";
  stream << "  const char* name;\n";
  stream << "  const char* data;\n";
  stream << "  size_t size;\n";
  stream << "} iree_file_toc_t;\n";
  WriteExternCClose(stream);
  stream << "#endif  // IREE_FILE_TOC\n";
}

bool ReadFile(const std::string& file_name, std::vector<uint8_t>* contents) {
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "failed to open input file '" << file_name << "'\n";
    return false;
  }
  file.seekg(0, file.end);
  const std::streamoff length = file.tellg();
  file.seekg(0, file.beg);
  if (!file.good() || length < 0) {
    std::cerr << "failed to stat input file '" << file_name << "'\n";
    return false;
  }
  if (length > kMaxEmbeddedFileSize) {
    std::cerr << "input file '" << file_name << "' is too large to embed ("
              << length << " bytes > " << kMaxEmbeddedFileSize << " bytes)\n";
    return false;
  }

  contents->resize(static_cast<size_t>(length));
  if (!contents->empty()) {
    file.read(reinterpret_cast<char*>(contents->data()),
              static_cast<std::streamsize>(contents->size()));
  }
  if (!file.good()) {
    std::cerr << "failed to read input file '" << file_name << "'\n";
    return false;
  }
  return true;
}

bool WriteHeader(const Options& options,
                 const std::vector<std::string>& toc_files) {
  std::ofstream stream(options.output_header, std::ios::out | std::ios::trunc);
  if (!stream) {
    std::cerr << "failed to open output header '" << options.output_header
              << "'\n";
    return false;
  }

  stream << "#pragma once\n";
  stream << "#include <stddef.h>\n";
  WriteTocStruct(stream);
  WriteExternCOpen(stream);
  stream << "const iree_file_toc_t* " << options.identifier
         << "_create(void);\n";
  stream << "static inline size_t " << options.identifier << "_size(void) {\n";
  stream << "  return " << toc_files.size() << ";\n";
  stream << "}\n";
  WriteExternCClose(stream);
  return stream.good();
}

bool WriteImpl(const Options& options,
               const std::vector<std::string>& toc_files) {
  std::ofstream stream(options.output_impl, std::ios::out | std::ios::trunc);
  if (!stream) {
    std::cerr << "failed to open output implementation '" << options.output_impl
              << "'\n";
    return false;
  }

  stream << "#include <stddef.h>\n";
  stream << "#include <stdint.h>\n";
  stream << R"(
#if !defined(IREE_DATA_ALIGNAS_PTR)
#if defined(_MSC_VER)
#define IREE_DATA_ALIGNAS_PTR __declspec(align(64))
#else
#define IREE_DATA_ALIGNAS_PTR _Alignas(64)
#endif  // defined(_MSC_VER)
#endif  // !defined(IREE_DATA_ALIGNAS_PTR)
)";
  WriteTocStruct(stream);

  for (size_t i = 0; i < options.input_files.size(); ++i) {
    std::vector<uint8_t> contents;
    if (!ReadFile(options.input_files[i], &contents)) return false;

    stream << "IREE_DATA_ALIGNAS_PTR static const uint8_t file_" << i
           << "[] = {\n";
    constexpr size_t kMaxBytesPerLine = 1024;
    for (size_t offset = 0; offset < contents.size();
         offset += kMaxBytesPerLine) {
      const size_t line_length =
          std::min(kMaxBytesPerLine, contents.size() - offset);
      for (size_t j = 0; j < line_length; ++j) {
        stream << static_cast<unsigned int>(contents[offset + j]) << ",";
      }
      stream << "\n";
    }
    stream << "0,\n";
    stream << "};\n";
  }

  stream << "static const iree_file_toc_t toc[] = {\n";
  for (size_t i = 0; i < options.input_files.size(); ++i) {
    stream << "  {\n";
    stream << "    \"" << EscapeCString(toc_files[i]) << "\",\n";
    stream << "    (const char*)file_" << i << ",\n";
    stream << "    sizeof(file_" << i << ") - 1,\n";
    stream << "  },\n";
  }
  stream << "  {NULL, NULL, 0},\n";
  stream << "};\n";
  WriteExternCOpen(stream);
  stream << "const iree_file_toc_t* " << options.identifier
         << "_create(void) {\n";
  stream << "  return &toc[0];\n";
  stream << "}\n";
  WriteExternCClose(stream);
  return stream.good();
}

bool ParseOptionValue(const char* arg, std::string* key, std::string* value) {
  if (arg[0] != '-' || arg[1] != '-') return false;
  const std::string option(arg + 2);
  const size_t separator_pos = option.find('=');
  if (separator_pos == std::string::npos) {
    *key = option;
    value->clear();
    return true;
  }
  *key = option.substr(0, separator_pos);
  *value = option.substr(separator_pos + 1);
  return true;
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    std::string key;
    std::string value;
    if (!ParseOptionValue(argv[i], &key, &value)) {
      options->input_files.push_back(argv[i]);
      continue;
    }

    if (key == "flatten") {
      if (!value.empty()) {
        std::cerr << "--flatten does not accept a value\n";
        return false;
      }
      options->flatten = true;
    } else if (key == "identifier") {
      options->identifier = value;
    } else if (key == "output_header") {
      options->output_header = value;
    } else if (key == "output_impl") {
      options->output_impl = value;
    } else if (key == "strip_prefix") {
      options->strip_prefix = value;
    } else {
      std::cerr << "unrecognized option '--" << key << "'\n";
      return false;
    }
  }

  if (options->output_header.empty() || options->output_impl.empty()) {
    std::cerr << "--output_header and --output_impl are required\n";
    return false;
  }
  if (!IsValidIdentifier(options->identifier)) {
    std::cerr << "invalid C identifier '" << options->identifier << "'\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return EXIT_FAILURE;

  std::vector<std::string> toc_files;
  toc_files.reserve(options.input_files.size());
  for (const std::string& input_file : options.input_files) {
    toc_files.push_back(TocNameForInput(options, input_file));
  }

  if (!WriteHeader(options, toc_files)) return EXIT_FAILURE;
  if (!WriteImpl(options, toc_files)) return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
