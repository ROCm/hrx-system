# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolves relocatable ELF dynamic-library closures for local artifacts."""

# The execution platform owns its glibc ABI. Putting a second glibc beside an
# artifact can make the worker's dynamic loader bind an incompatible process
# runtime. All other resolved libraries travel with the artifact so selected
# local SDK content does not leave undeclared dependencies on its source host.
_GLIBC_DYNAMIC_LIBRARY_PREFIXES = [
    "libanl.so",
    "libBrokenLocale.so",
    "libc.so",
    "libdl.so",
    "libm.so",
    "libmvec.so",
    "libnss_compat.so",
    "libnss_dns.so",
    "libnss_files.so",
    "libnss_hesiod.so",
    "libpthread.so",
    "libresolv.so",
    "librt.so",
    "libthread_db.so",
    "libutil.so",
]

_LOADER_ENVIRONMENT_DEFAULTS = {
    "LANG": "C",
    "LC_ALL": "C",
    "LD_AUDIT": "",
    "LD_PRELOAD": "",
}

def elf_loader_environment(library_search_path = ""):
    """Returns a controlled environment for inspecting or invoking ELF files.

    Args:
      library_search_path: Explicit colon-separated dynamic-library search path.

    Returns:
      An environment dictionary that replaces ambient loader injection and
      locale state while preserving the caller-selected library search path.
    """
    environment = dict(_LOADER_ENVIRONMENT_DEFAULTS)
    environment["LD_LIBRARY_PATH"] = library_search_path
    return environment

def _is_glibc_dynamic_library(name):
    for prefix in _GLIBC_DYNAMIC_LIBRARY_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return True
    return False

def _canonical_watched_path(repository_ctx, path, description):
    path = repository_ctx.path(path)
    if not path.exists:
        return struct(
            error = "{} does not exist: {}".format(description, path),
            path = "",
        )
    repository_ctx.watch(path)
    canonical_path = repository_ctx.path(path.realpath)
    if not canonical_path.exists:
        return struct(
            error = "{} resolves to a missing path: {}".format(
                description,
                canonical_path,
            ),
            path = "",
        )
    repository_ctx.watch(canonical_path)
    return struct(error = "", path = str(canonical_path.realpath))

def try_resolve_elf_dynamic_library_closure(
        repository_ctx,
        artifacts,
        library_search_path = ""):
    """Tries to resolve non-glibc ELF libraries required by selected artifacts.

    Each selected entry artifact is inspected once. The loader reports its
    transitive closure; this function does not scan SDK trees or reinvoke the
    loader for every resolved library.

    The selected files and every resolved symlink and canonical library are
    watched repository inputs. Replacing local SDK content therefore
    invalidates the repository even when its configured path is unchanged.

    Args:
      repository_ctx: Repository context used to execute ldd and inspect paths.
      artifacts: Dictionary mapping stable logical names to artifact paths.
      library_search_path: Explicit colon-separated loader search path. The
        ambient LD_LIBRARY_PATH, LD_PRELOAD, and LD_AUDIT are always replaced so
        dependency selection cannot drift with the invoking shell.

    Returns:
      A struct containing an error string and a libraries dictionary mapping
      loader-visible library names to canonical absolute paths. The error is
      empty when resolution succeeds.
    """
    ldd = repository_ctx.which("ldd")
    if not ldd:
        return struct(
            error = "Could not inspect ELF dynamic libraries: ldd was not found on PATH.",
            libraries = {},
        )
    ldd_result = _canonical_watched_path(repository_ctx, ldd, "ldd")
    if ldd_result.error:
        return struct(error = ldd_result.error, libraries = {})
    ldd = ldd_result.path

    loader_environment = elf_loader_environment(library_search_path)

    libraries = {}
    for artifact_name in sorted(artifacts.keys()):
        artifact_result = _canonical_watched_path(
            repository_ctx,
            artifacts[artifact_name],
            "Artifact {}".format(artifact_name),
        )
        if artifact_result.error:
            return struct(error = artifact_result.error, libraries = {})
        artifact = artifact_result.path
        result = repository_ctx.execute(
            [ldd, artifact],
            environment = loader_environment,
            quiet = True,
        )
        output = result.stdout + "\n" + result.stderr
        if "not a dynamic executable" in output or "statically linked" in output:
            continue
        if result.return_code != 0:
            return struct(
                error = "Failed to inspect ELF dynamic libraries for {}:\n{}".format(
                    artifact,
                    output,
                ),
                libraries = {},
            )

        for line in output.splitlines():
            parts = line.strip().split(" => ")
            if len(parts) != 2:
                continue
            library_name = parts[0]
            library_path = parts[1].split(" (")[0].strip()
            if library_path == "not found":
                return struct(
                    error = "Artifact {} ({}) cannot load required dynamic library {}".format(
                        artifact_name,
                        artifact,
                        library_name,
                    ),
                    libraries = {},
                )
            if _is_glibc_dynamic_library(library_name):
                continue
            if not library_path.startswith("/"):
                return struct(
                    error = "ldd returned a non-absolute path for {}: {}".format(
                        library_name,
                        library_path,
                    ),
                    libraries = {},
                )
            library_result = _canonical_watched_path(
                repository_ctx,
                library_path,
                "Dynamic library {}".format(library_name),
            )
            if library_result.error:
                return struct(error = library_result.error, libraries = {})
            library_path = library_result.path
            existing_path = libraries.get(library_name)
            if existing_path and existing_path != library_path:
                return struct(
                    error = "Selected artifacts resolve {} inconsistently: {} and {}".format(
                        library_name,
                        existing_path,
                        library_path,
                    ),
                    libraries = {},
                )
            libraries[library_name] = library_path
    return struct(error = "", libraries = libraries)

def resolve_elf_dynamic_library_closure(
        repository_ctx,
        artifacts,
        library_search_path = ""):
    """Returns the non-glibc ELF libraries required by selected artifacts.

    Args:
      repository_ctx: Repository context used to execute ldd and inspect paths.
      artifacts: Dictionary mapping stable logical names to artifact paths.
      library_search_path: Explicit colon-separated loader search path.

    Returns:
      A dictionary mapping loader-visible library names to canonical absolute
      paths.
    """
    result = try_resolve_elf_dynamic_library_closure(
        repository_ctx,
        artifacts,
        library_search_path,
    )
    if result.error:
        fail(result.error)
    return result.libraries
