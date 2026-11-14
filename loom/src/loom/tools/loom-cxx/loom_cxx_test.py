# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiles tiny single-source translation units with loom-cxx and checks the
objects: host code, the embedded device code object with its load record, and
the kernel descriptors, all from one parse of the source."""

import argparse
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ARGS = None

EM_X86_64, EM_AARCH64 = 62, 183
HOST_MACHINES = (EM_X86_64, EM_AARCH64)
STB_LOCAL, STB_GLOBAL, STB_WEAK = 0, 1, 2
STT_OBJECT, STT_FUNC, STT_FILE = 1, 2, 4
SHN_UNDEF = 0
SHT_RELA, SHT_NOTE = 4, 7
KERNEL_REF_PREFIX = "_ZN4loom6kernel10kernel_ref"
DESCRIPTOR_VERSION = 3

KERNEL_PRELUDE = r"""
#include <loomcxx/kernel.h>
static loom::kernel::configuration launch_config(unsigned groups) {
  return {{groups, 1, 1}, {64, 1, 1}};
}
template <class T>
[[loom::kernel(launch_config)]]
void run(T tag) { tag(); }
struct Body {
  float a;
  float* y;
  [[loom::inline]] void operator()() const { y[0] = a; }
};
"""

# Range-based for statements over an array, a class range and, in a lambda,
# nested arrays: in a kernel body and in a function host code also calls.
RANGE_FOR_SOURCE = (
    KERNEL_PRELUDE
    + r"""
struct Counter {
  struct Iterator {
    unsigned index;
    void operator++() { ++index; }
    bool operator!=(const Iterator& other) const { return index != other.index; }
    unsigned operator*() const { return index; }
  };
  unsigned count;
  Iterator begin() const { return {0}; }
  Iterator end() const { return {count}; }
};
float total(const float* y) {
  float values[3] = {y[0], y[1], y[2]};
  float sum = 0.0f;
  for (float value : values) sum += value;
  for (unsigned index : Counter{3}) sum += values[index] * y[index];
  return sum;
}
struct Ranges {
  float* y;
  [[loom::inline]] void operator()() const {
    float values[2] = {y[0], y[1]};
    auto product = [&] {
      float p = 0.0f;
      for (float v : values)
        for (float w : values) p += v * w;
      return p;
    };
    y[0] = total(y) + product();
  }
};
float host_total(const float* y) {
  float pair[2] = {y[0], y[1]};
  float sum = 0.0f;
  for (float value : pair) sum += value;
  return total(y) + sum;
}
const loom::kernel::descriptor* ranges() {
  return &loom::kernel::kernel_ref<&run<Ranges>>;
}
"""
)
RANGES_REF = "_ZN4loom6kernel10kernel_refIXadL_Z3runI6RangesEvT_EEEE"

BODY_REF = "_ZN4loom6kernel10kernel_refIXadL_Z3runI4BodyEvT_EEEE"


class Elf:
    """Minimal little-endian ELF64 relocatable object reader."""

    def __init__(self, data):
        assert data[:4] == b"\x7fELF" and data[4] == 2, "not an ELF64 object"
        self.data = data
        (shoff,) = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
        self.sections = []
        for i in range(shnum):
            fields = struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
            self.sections.append(fields)
        names = self.sections[shstrndx]
        self.names = [self._string(names, s[0]) for s in self.sections]

    def _string(self, table, offset):
        start = table[4] + offset
        return self.data[start : self.data.index(b"\0", start)].decode()

    def section(self, name):
        for section_name, section in zip(self.names, self.sections):
            if section_name == name:
                return section
        return None

    def contents(self, section):
        return self.data[section[4] : section[4] + section[5]]

    def symbol_table(self):
        """(name, binding, type, section index, value) per symbol, by index."""
        symtab = self.section(".symtab")
        strtab = self.sections[symtab[6]]
        result = []
        for offset in range(0, symtab[5], 24):
            name, info, _, shndx, value, _ = struct.unpack_from(
                "<IBBHQQ", self.data, symtab[4] + offset
            )
            name = self._string(strtab, name) if name else ""
            result.append((name, info >> 4, info & 15, shndx, value))
        return result

    def symbols(self):
        return {
            name: (binding, kind, shndx)
            for name, binding, kind, shndx, _ in self.symbol_table()
            if name
        }

    def kernel_ref_symbols(self):
        """(binding, section index) of every kernel_ref symbol."""
        return [
            (name, binding, shndx)
            for name, binding, _, shndx, _ in self.symbol_table()
            if name.startswith(KERNEL_REF_PREFIX)
        ]

    def relocations(self, section_index):
        """Pointer relocations applying to a section: offset -> (section, offset)."""
        table = self.symbol_table()
        result = {}
        for section in self.sections:
            if section[1] != SHT_RELA or section[7] != section_index:
                continue
            for offset in range(0, section[5], 24):
                r_offset, r_info, r_addend = struct.unpack_from(
                    "<QQq", self.data, section[4] + offset
                )
                _, _, _, shndx, value = table[r_info >> 32]
                result[r_offset] = (shndx, value + r_addend)
        return result

    def read(self, section_index, offset, size):
        start = self.sections[section_index][4] + offset
        return self.data[start : start + size]

    def c_string(self, section_index, offset):
        start = self.sections[section_index][4] + offset
        return self.data[start : self.data.index(b"\0", start)].decode()

    def descriptor(self, symbol):
        """The loom::kernel::descriptor (loomcxx/kernel_ref.h) at a symbol."""
        for name, _, _, shndx, value in self.symbol_table():
            if name == symbol and shndx != SHN_UNDEF:
                break
        else:
            return None
        relocations = self.relocations(shndx)
        (version, flags) = struct.unpack("<II", self.read(shndx, value, 8))
        wg = struct.unpack("<III", self.read(shndx, value + 16, 12))
        kernarg_size, parameter_count, export_ordinal = struct.unpack(
            "<III", self.read(shndx, value + 28, 12)
        )
        name = relocations.get(value + 8)
        return {
            "version": version,
            "flags": flags,
            "name": self.c_string(*name) if name else None,
            "workgroup_size": wg,
            "kernarg_size": kernarg_size,
            "parameter_count": parameter_count,
            "export_ordinal": export_ordinal,
            "code_object": relocations.get(value + 40),
            "pack": relocations.get(value + 48),
        }

    def function_at(self, location):
        """(name, binding) of the function symbol defined at (section,
        offset), or None."""
        for name, binding, kind, shndx, value in self.symbol_table():
            if kind == STT_FUNC and (shndx, value) == tuple(location):
                return name, binding
        return None

    def code_object_record(self, location):
        """The loom::kernel::embedded_code_object at (section, offset)."""
        relocations = self.relocations(location[0])
        family, key, data = (relocations.get(location[1] + i) for i in (0, 8, 16))
        (size,) = struct.unpack("<Q", self.read(location[0], location[1] + 24, 8))
        return {
            "target_family": self.c_string(*family) if family else None,
            "target_key": self.c_string(*key) if key else None,
            "data": data,
            "size": size,
        }

    def code_object(self, descriptor):
        """The ELF code object embedded at the descriptor's code object
        record's data address, of the record's size."""
        record = self.code_object_record(descriptor["code_object"])
        location = record["data"]
        assert location, "the code object record has no data"
        start = self.sections[location[0]][4] + location[1]
        data = self.data[start : start + record["size"]]
        assert data[:4] == b"\x7fELF", "no code object at the record's address"
        assert len(data) == record["size"], "the code object exceeds its section"
        return Elf(data)

    def amdgpu_kernel_arguments(self):
        """Kernel name -> (kernarg segment size, [(offset, size, value kind)])
        from the AMDGPU code object metadata note."""
        return {
            kernel[".name"]: (
                kernel[".kernarg_segment_size"],
                [
                    (a[".offset"], a[".size"], a[".value_kind"])
                    for a in kernel.get(".args", [])
                ],
            )
            for kernel in self.amdgpu_kernels()
        }

    def amdgpu_export_names(self):
        """Kernel names in export ordinal order: the order of the AMDGPU code
        object metadata, in which HRX's loader numbers the exports."""
        return [kernel[".name"] for kernel in self.amdgpu_kernels()]

    def amdgpu_kernels(self):
        """The amdhsa.kernels entries of the AMDGPU code object metadata note."""
        result = []
        for section in self.sections:
            if section[1] != SHT_NOTE:
                continue
            notes = self.data[section[4] : section[4] + section[5]]
            position = 0
            while position + 12 <= len(notes):
                namesz, descsz, kind = struct.unpack_from("<III", notes, position)
                position += 12
                name = notes[position : position + namesz]
                position += (namesz + 3) & ~3
                desc = notes[position : position + descsz]
                position += (descsz + 3) & ~3
                if name.rstrip(b"\0") != b"AMDGPU" or kind != 32:
                    continue
                metadata, _ = msgpack_decode(desc, 0)
                result.extend(metadata["amdhsa.kernels"])
        return result


def msgpack_decode(data, position):
    """Minimal MessagePack decoder for AMDGPU code object metadata."""
    tag = data[position]
    position += 1

    def take(fmt):
        nonlocal position
        (value,) = struct.unpack_from(">" + fmt, data, position)
        position += struct.calcsize(fmt)
        return value

    def items(count, mapping):
        nonlocal position
        if mapping:
            result = {}
            for _ in range(count):
                key, position = msgpack_decode(data, position)
                result[key], position = msgpack_decode(data, position)
            return result
        result = []
        for _ in range(count):
            value, position = msgpack_decode(data, position)
            result.append(value)
        return result

    def raw(count, text):
        nonlocal position
        value = data[position : position + count]
        position += count
        return value.decode() if text else bytes(value)

    if tag <= 0x7F:
        return tag, position
    if tag >= 0xE0:
        return tag - 0x100, position
    if 0x80 <= tag <= 0x8F:
        return items(tag & 0x0F, True), position
    if 0x90 <= tag <= 0x9F:
        return items(tag & 0x0F, False), position
    if 0xA0 <= tag <= 0xBF:
        return raw(tag & 0x1F, True), position
    simple = {0xC0: None, 0xC2: False, 0xC3: True}
    if tag in simple:
        return simple[tag], position
    numbers = {
        0xCA: "f", 0xCB: "d", 0xCC: "B", 0xCD: "H", 0xCE: "I", 0xCF: "Q",
        0xD0: "b", 0xD1: "h", 0xD2: "i", 0xD3: "q",
    }
    if tag in numbers:
        return take(numbers[tag]), position
    if tag in (0xD9, 0xDA, 0xDB):
        return raw(take({0xD9: "B", 0xDA: "H", 0xDB: "I"}[tag]), True), position
    if tag in (0xC4, 0xC5, 0xC6):
        return raw(take({0xC4: "B", 0xC5: "H", 0xC6: "I"}[tag]), False), position
    if tag in (0xDC, 0xDD):
        return items(take("H" if tag == 0xDC else "I"), False), position
    if tag in (0xDE, 0xDF):
        return items(take("H" if tag == 0xDE else "I"), True), position
    raise ValueError(f"unsupported MessagePack tag {tag:#x}")


class LoomCxxTest(unittest.TestCase):
    def compile(self, source, *extra, expect_success=True, name="unit"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = pathlib.Path(temporary.name)
        source_path = directory / f"{name}.cpp"
        source_path.write_text(source)
        object_path = directory / f"{name}.o"
        self.object_path = object_path
        command = [
            ARGS.loom_cxx,
            "-c",
            str(source_path),
            "-o",
            str(object_path),
            "-std=gnu++17",
            "-O2",
            "-Wall",
            "-fPIC",
            "-MD",
            "-MT",
            str(object_path),
            "-MF",
            str(directory / f"{name}.d"),
            f"--loom-target={ARGS.target}",
            "--loom-stats",
            *extra,
        ]
        run = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if expect_success:
            self.assertEqual(run.returncode, 0, run.stderr)
            elf = Elf(object_path.read_bytes())
            depfile = (directory / f"{name}.d").read_text()
            return elf, run.stderr, depfile
        self.assertNotEqual(run.returncode, 0, run.stderr)
        self.assertFalse(object_path.exists())
        return None, run.stderr, None

    def device_code_objects(self, elf):
        """Machines of the ELF code objects embedded in read-only data."""
        found = []
        for name, section in zip(elf.names, elf.sections):
            if not name.startswith(".rodata") and not name.startswith(".lrodata"):
                continue
            contents = elf.contents(section)
            start = contents.find(b"\x7fELF")
            while start >= 0:
                (machine,) = struct.unpack_from("<H", contents, start + 18)
                if machine not in HOST_MACHINES:
                    found.append(machine)
                start = contents.find(b"\x7fELF", start + 1)
        return found

    def test_kernel_unit_is_parsed_once_into_one_object(self):
        elf, stderr, depfile = self.compile(
            KERNEL_PRELUDE
            + """
const loom::kernel::descriptor* host_entry() {
  return &loom::kernel::kernel_ref<&run<Body>>;
}
"""
        )
        # One preprocess and one parse fed host code generation and import.
        self.assertIn("preprocess=1 parse=1", stderr)
        self.assertIn("host_codegen=1 imports=1 device_compiles=1", stderr)
        symbols = elf.symbols()
        # Host code.
        binding, kind, shndx = symbols["_Z10host_entryv"]
        self.assertEqual((binding, kind), (STB_GLOBAL, STT_FUNC))
        self.assertNotEqual(shndx, SHN_UNDEF)
        # The kernel descriptor: a local definition, available, naming the
        # object's own code object.
        binding, kind, shndx = symbols[BODY_REF]
        self.assertEqual((binding, kind), (STB_LOCAL, STT_OBJECT))
        self.assertNotEqual(shndx, SHN_UNDEF)
        self.assertNoExportedKernelRefs(elf)
        descriptor = elf.descriptor(BODY_REF)
        self.assertEqual((descriptor["version"], descriptor["flags"]), (3, 1))
        self.assertEqual(descriptor["name"], "_Z3runI4BodyEvT_")
        self.assertIsNotNone(descriptor["code_object"])
        # The export ordinal numbers the kernel in that code object.
        exports = elf.code_object(descriptor).amdgpu_export_names()
        self.assertEqual(exports[descriptor["export_ordinal"]], "_Z3runI4BodyEvT_")
        # The embedded device code object and its record: what a runtime
        # needs to load it, at the first launch of one of its kernels.
        self.assertTrue(self.device_code_objects(elf))
        record = elf.code_object_record(descriptor["code_object"])
        family, key = ARGS.target.split(":", 1)
        self.assertEqual((record["target_family"], record["target_key"]), (family, key))
        self.assertGreater(record["size"], 0)
        self.assertNoStartupCode(elf)
        # No device-only symbols reach the host object.
        self.assertFalse(
            [s for s in symbols if s.startswith("_ZN4loom") and s != BODY_REF]
        )
        # loom-cxx found the loomcxx/ headers next to itself, without -isystem.
        headers = pathlib.Path(ARGS.loom_cxx).resolve().parent / "include"
        self.assertIn(str(headers / "loomcxx" / "kernel.h"), depfile)

    def test_host_only_unit_has_no_device_pass(self):
        # -v, which CMake passes while checking the compiler, is accepted.
        elf, stderr, _ = self.compile(
            "int host_only(int x) { return x + 1; }\n", "-v"
        )
        self.assertIn("preprocess=1 parse=1", stderr)
        self.assertIn("imports=0 device_compiles=0", stderr)
        self.assertIsNone(elf.section(".init_array"))
        self.assertEqual(self.device_code_objects(elf), [])
        self.assertIn("_Z9host_onlyi", elf.symbols())

    def test_undefines_follow_every_definition(self):
        # -U removes a -D definition whatever the order, as in GCC, and a
        # predefined macro of the hosted profile.
        source = """
#if defined(FOO) || defined(__linux__)
#error macro still defined
#endif
int host_only(int value) { return value; }
"""
        self.compile(source, "-UFOO", "-DFOO=1", "-U__linux__")
        self.compile(source, "-DFOO=1", expect_success=False)

    def test_rejected_kernels_are_compile_errors(self):
        # The importer rejects Bad, so the unit does not compile.
        source = (
            KERNEL_PRELUDE
            + """
struct Bad {
  float* y;
  [[loom::inline]] void operator()() const { if (y < y + 1) y[0] = 2.0f; }
};
const loom::kernel::descriptor* good() {
  return &loom::kernel::kernel_ref<&run<Body>>;
}
const loom::kernel::descriptor* bad() {
  return &loom::kernel::kernel_ref<&run<Bad>>;
}
"""
        )
        _, stderr, _ = self.compile(source, expect_success=False)
        self.assertIn("device import failed", stderr)

    def test_export_ordinals_number_the_code_object_exports(self):
        # Several kernels in one code object: each descriptor names its own
        # kernel by the ordinal HRX's loader gives the export, the kernel's
        # position in the code object's metadata.
        elf, _, _ = self.compile(
            KERNEL_PRELUDE
            + """
struct Twice {
  float* y;
  [[loom::inline]] void operator()() const { y[0] = 2.0f * y[1]; }
};
struct Thrice {
  float* y;
  int n;
  [[loom::inline]] void operator()() const { y[n] = 3.0f; }
};
const loom::kernel::descriptor* entries(int i) {
  if (i == 0) return &loom::kernel::kernel_ref<&run<Thrice>>;
  if (i == 1) return &loom::kernel::kernel_ref<&run<Body>>;
  return &loom::kernel::kernel_ref<&run<Twice>>;
}
"""
        )
        descriptors = [elf.descriptor(name) for name, _, _ in elf.kernel_ref_symbols()]
        self.assertEqual(len(descriptors), 3)
        self.assertEqual(len({d["code_object"] for d in descriptors}), 1)
        exports = elf.code_object(descriptors[0]).amdgpu_export_names()
        self.assertEqual(
            sorted(d["export_ordinal"] for d in descriptors), list(range(len(exports)))
        )
        for descriptor in descriptors:
            self.assertEqual(exports[descriptor["export_ordinal"]], descriptor["name"])

    def assertNoStartupCode(self, elf):
        """Nothing runs before main: no constructors register or load the
        code object, and the object needs no runtime symbol."""
        for name in (".init_array", ".preinit_array", ".ctors", ".init"):
            self.assertIsNone(elf.section(name), name)
        undefined = [
            name
            for name, _, _, shndx, _ in elf.symbol_table()
            if name and shndx == SHN_UNDEF
        ]
        self.assertEqual(undefined, [])

    def assertNoExportedKernelRefs(self, elf):
        """Descriptors are local definitions: none exported, none undefined."""
        references = elf.kernel_ref_symbols()
        self.assertTrue(references)
        for name, binding, shndx in references:
            self.assertEqual(binding, STB_LOCAL, name)
            self.assertNotEqual(shndx, SHN_UNDEF, name)

    PACK_MAIN = r"""
#include <stdio.h>
#include <string.h>
struct descriptor {
  unsigned version, flags;
  const char* name;
  unsigned workgroup_size[3], kernarg_size, parameter_count, export_ordinal;
  const void* code_object;
  void (*pack)(void* kernarg, const void* const* parameters);
};
const struct descriptor* mixed_entry(void);
int main(void) {
  const struct descriptor* d = mixed_entry();
  unsigned char parameter[48];
  for (int i = 0; i < 48; ++i) parameter[i] = (unsigned char)(i + 1);
  unsigned char kernarg[80];
  memset(kernarg, 0xaa, sizeof kernarg);
  const void* parameters[1] = {parameter};
  d->pack(kernarg, parameters);
  for (unsigned i = 0; i < sizeof kernarg; ++i) printf("%02x", kernarg[i]);
  printf("\n");
  return 0;
}
"""

    def test_packing_function_follows_the_compiled_kernel(self):
        # Sub-word scalars, mixed with pointers and wider scalars. The
        # descriptor's packing function writes the kernel arguments where the
        # compiled kernel reads them, as its code object metadata declares.
        elf, _, _ = self.compile(
            KERNEL_PRELUDE
            + """
struct Mixed {
  char c;
  short s;
  bool b;
  float* y;
  char c2;
  double d;
  int* z;
  bool b2;
  unsigned char u;
  int i;
  [[loom::inline]] void operator()() const {
    y[0] = static_cast<float>(c) + static_cast<float>(s) +
           static_cast<float>(c2) + static_cast<float>(d) + (b ? 1.0f : 0.0f);
    z[0] = static_cast<int>(u) + i + (b2 ? 1 : 0);
  }
};
extern "C" const loom::kernel::descriptor* mixed_entry() {
  return &loom::kernel::kernel_ref<&run<Mixed>>;
}
"""
        )
        symbol = "_ZN4loom6kernel10kernel_refIXadL_Z3runI5MixedEvT_EEEE"
        descriptor = elf.descriptor(symbol)
        self.assertEqual((descriptor["version"], descriptor["flags"]), (3, 1))
        self.assertEqual(descriptor["parameter_count"], 1)
        self.assertEqual(descriptor["kernarg_size"], 64)
        # The compiled kernel's layout: each by-value argument starts on a
        # 4-byte word of the kernarg segment and sub-word scalars do not pack
        # together.
        kernels = elf.code_object(descriptor).amdgpu_kernel_arguments()
        segment_size, arguments = kernels[descriptor["name"]]
        self.assertEqual(segment_size, descriptor["kernarg_size"])
        self.assertEqual(
            [(offset, size, kind) for offset, size, kind in arguments],
            [
                (0, 1, "by_value"),
                (4, 2, "by_value"),
                (8, 1, "by_value"),
                (16, 8, "global_buffer"),
                (24, 1, "by_value"),
                (28, 8, "by_value"),
                (40, 8, "global_buffer"),
                (48, 1, "by_value"),
                (52, 1, "by_value"),
                (56, 4, "by_value"),
            ],
        )
        # The packing function is host code local to the object.
        self.assertEqual(elf.function_at(descriptor["pack"])[1], STB_LOCAL)
        # Run it on a host object (LP64 layout of Mixed) whose bytes are 1, 2,
        # 3, ..., into a buffer of 0xaa bytes: each argument gets its payload
        # from its field, every other byte of the segment is zero (the rest
        # of a sub-word scalar's slot included), and nothing past it changes.
        output = self.link_and_run([self.object_path], self.PACK_MAIN)
        if output is None:
            self.skipTest("no C compiler to link with")
        host_offsets = [0, 2, 4, 8, 16, 24, 32, 40, 41, 44]
        parameter = bytes(range(1, 49))
        expected = bytearray(segment_size)
        for (offset, size, _), source in zip(arguments, host_offsets):
            expected[offset : offset + size] = parameter[source : source + size]
        self.assertEqual(output[0], (bytes(expected) + b"\xaa" * 16).hex())

    def link_and_run(self, objects, main_source):
        """Links host objects with a C main and runs it; None without a C
        compiler."""
        compiler = next(
            (c for c in ("cc", "gcc", "clang") if shutil.which(c)), None
        )
        if compiler is None:
            return None
        directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        main_path = directory / "main.c"
        main_path.write_text(main_source)
        program = directory / "program"
        link = subprocess.run(
            [compiler, "-no-pie", "-o", str(program), str(main_path)]
            + [str(o) for o in objects],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(link.returncode, 0, link.stderr)
        run = subprocess.run(
            [str(program)], stdout=subprocess.PIPE, universal_newlines=True
        )
        self.assertEqual(run.returncode, 0)
        return run.stdout.split("\n")

    # Code objects as the descriptors name them: -1 for none, -2 for a record
    # that does not hold an amdgpu code object, else the index of the record
    # in order of first appearance.
    CODE_OBJECTS_C = r"""
#include <stdio.h>
#include <string.h>
struct code_object {
  const char* target_family;
  const char* target_key;
  const void* data;
  unsigned long size;
};
static const struct code_object* seen[8];
static int seen_count;
static int code_object(const struct code_object* c) {
  if (!c) return -1;
  for (int i = 0; i < seen_count; ++i) if (seen[i] == c) return i;
  if (c->size < 4 || memcmp(c->data, "\177ELF", 4) != 0 ||
      strcmp(c->target_family, "amdgpu") != 0) return -2;
  seen[seen_count] = c;
  return seen_count++;
}
"""

    DESCRIPTOR_MAIN = CODE_OBJECTS_C + r"""
struct descriptor {
  unsigned version, flags;
  const char* name;
  unsigned workgroup_size[3], kernarg_size, parameter_count, export_ordinal;
  const struct code_object* code_object;
  void (*pack)(void* kernarg, const void* const* parameters);
};
const struct descriptor* entry_a(void);
const struct descriptor* entry_b(void);
static void print(const struct descriptor* d) {
  printf("%s %u %u %d\n", d->name, d->flags, d->kernarg_size, code_object(d->code_object));
}
int main(void) {
  const struct descriptor* a = entry_a();
  const struct descriptor* b = entry_b();
  print(a);
  print(b);
  printf("%d %d\n", seen_count, a == b);
  return 0;
}
"""

    def test_internal_lambda_kernels_keep_their_own_descriptors(self):
        # Two objects launch lambdas declared in static functions of the same
        # name. The lambdas have internal linkage and mangle alike, so both
        # objects reference the same kernel_ref symbol and export name for
        # different kernels. Each object defines its descriptor locally and
        # names its own code object, so after linking each launches its own.
        objects = []
        refs = []
        elves = []
        for name, body in (
            ("a", "y[0] = 1.0f;"),
            ("b", "y[0] = static_cast<float>(n);"),
        ):
            elf, _, _ = self.compile(
                KERNEL_PRELUDE
                + f"""
static const loom::kernel::descriptor* entry(float* y, int n) {{
  auto body = [=]() {{ {body} }};
  return &loom::kernel::kernel_ref<&run<decltype(body)>>;
}}
extern "C" const loom::kernel::descriptor* entry_{name}() {{
  return entry(nullptr, 0);
}}
""",
                name=name,
            )
            self.assertNoExportedKernelRefs(elf)
            (ref,) = [r for r, _, _ in elf.kernel_ref_symbols()]
            refs.append((ref, elf.descriptor(ref)))
            elves.append(elf)
            objects.append(self.object_path)
        # The same descriptor symbol and export name, different kernels.
        self.assertEqual(refs[0][0], refs[1][0])
        self.assertEqual(refs[0][1]["name"], refs[1][1]["name"])
        arguments = [
            len(elf.code_object(d).amdgpu_kernel_arguments()[d["name"]][1])
            for elf, d in ((elves[0], refs[0][1]), (elves[1], refs[1][1]))
        ]
        self.assertEqual(arguments, [1, 2])
        output = self.link_and_run(objects, self.DESCRIPTOR_MAIN)
        if output is None:
            self.skipTest("no C compiler to link with")
        name = refs[0][1]["name"]
        # Two code objects, two descriptors, each naming its own object's.
        sizes = [str(d["kernarg_size"]) for _, d in refs]
        self.assertNotEqual(sizes[0], sizes[1])
        a, b = output[0].split(), output[1].split()
        self.assertEqual(a, [name, "1", sizes[0], "0"])
        self.assertEqual(b, [name, "1", sizes[1], "1"])
        self.assertEqual(output[2], "2 0")

    def test_range_for_unit_compiles_in_one_pass(self):
        # The importer translates range-based for statements without changing
        # the unit's AST, which host code generation has already read. -O0
        # keeps host code close to the AST it was generated from.
        elf, stderr, _ = self.compile(RANGE_FOR_SOURCE, "-O0")
        self.assertIn("preprocess=1 parse=1", stderr)
        self.assertIn("host_codegen=1 imports=1 device_compiles=1", stderr)
        symbols = elf.symbols()
        for symbol in ("_Z5totalPKf", "_Z10host_totalPKf"):
            self.assertEqual(symbols[symbol][:2], (STB_GLOBAL, STT_FUNC))
            self.assertNotEqual(symbols[symbol][2], SHN_UNDEF)
        # Descriptors are local to the object that defines them.
        self.assertEqual(symbols[RANGES_REF][:2], (STB_LOCAL, STT_OBJECT))
        self.assertTrue(self.device_code_objects(elf))

    def test_host_errors_are_compile_errors(self):
        _, stderr, _ = self.compile(
            "int broken() { return undeclared_name; }\n", expect_success=False
        )
        self.assertIn("undeclared_name", stderr)
        self.assertIn("error", stderr)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--loom-cxx", required=True)
    parser.add_argument("--target", required=True)
    ARGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *rest])


if __name__ == "__main__":
    main()
