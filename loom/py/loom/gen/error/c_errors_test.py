# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.error.type import ERR_TYPE_001
from loom.gen.error.c_errors import (
    generate_error_catalog_c,
    generate_error_catalog_h,
    generate_error_runtime_tables_inl,
)


def test_generate_error_catalog_exports_canonical_definitions() -> None:
    catalog_c = generate_error_catalog_c(
        [ERR_TYPE_001],
        catalog_symbol="loom_error_catalog_test",
        public_header="loom/error/error_catalog.h",
    )

    assert "const loom_error_def_t loom_err_type_001" in catalog_c
    assert "static const loom_error_def_t loom_err_type_001" not in catalog_c
    assert "const loom_error_catalog_t loom_error_catalog_test" in catalog_c
    assert "loom_error_catalog_lookup_ref" not in catalog_c
    assert "loom_diagnostic_severity_name" not in catalog_c


def test_generate_error_catalog_composes_fallback_catalog() -> None:
    catalog_c = generate_error_catalog_c(
        [ERR_TYPE_001],
        catalog_symbol="loom_error_catalog_test",
        public_header="loom/error/error_catalog.h",
        fallback_catalog_symbol="loom_error_catalog_core",
        fallback_public_header="loom/error/error_catalog.h",
    )

    assert '#include "loom/error/error_catalog.h"' in catalog_c
    assert ".fallback_catalog = &loom_error_catalog_core" in catalog_c


def test_generate_error_catalog_header_uses_canonical_names() -> None:
    catalog_h = generate_error_catalog_h(
        [ERR_TYPE_001],
        catalog_symbol="loom_error_catalog_test",
        public_header="loom/error/error_catalog.h",
    )

    assert "extern const loom_error_def_t loom_err_type_001" in catalog_h
    assert "#define LOOM_ERR_TYPE_001 (&loom_err_type_001)" in catalog_h
    assert "#define LOOM_ERR_TYPE_001_REF" in catalog_h
    assert "LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TYPE, 1)" in catalog_h


def test_generate_error_runtime_tables_uses_enum_source_of_truth() -> None:
    tables_inl = generate_error_runtime_tables_inl()

    assert '[LOOM_DIAGNOSTIC_ERROR] = "error"' in tables_inl
    assert '[LOOM_ERROR_DOMAIN_AMDGPU] = "AMDGPU"' in tables_inl
    assert '[LOOM_ERROR_DOMAIN_SPIRV] = "SPIRV"' in tables_inl
    assert '[LOOM_EMITTER_BYTECODE_READER] = "bytecode_reader"' in tables_inl
    assert "Python ErrorDomain enum must match loom_error_domain_t" in tables_inl
    assert "loom_error_catalog_lookup" not in tables_inl
