# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for loom structured error catalog."""

import pytest

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity


class TestErrorDefValidation:
    """Tests for ErrorDef.__post_init__ validation."""

    def test_valid_error(self) -> None:
        """A well-formed ErrorDef should construct without error."""
        error = ErrorDef(
            domain=ErrorDomain.TYPE,
            code=1,
            severity=Severity.ERROR,
            summary="Test error.",
            message="'{name}' has type {actual_type}",
            params=(
                ErrorParam("name", ParamKind.STRING),
                ErrorParam("actual_type", ParamKind.TYPE),
            ),
        )
        assert error.error_id == "ERR_TYPE_001"
        assert error.domain == ErrorDomain.TYPE
        assert error.code == 1

    def test_zero_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="code must be >= 1"):
            ErrorDef(
                domain=ErrorDomain.TYPE,
                code=0,
                severity=Severity.ERROR,
                summary="Bad.",
                message="bad",
                params=(),
            )

    def test_negative_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="code must be >= 1"):
            ErrorDef(
                domain=ErrorDomain.TYPE,
                code=-1,
                severity=Severity.ERROR,
                summary="Bad.",
                message="bad",
                params=(),
            )

    def test_duplicate_param_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate param name 'x'"):
            ErrorDef(
                domain=ErrorDomain.TYPE,
                code=1,
                severity=Severity.ERROR,
                summary="Bad.",
                message="{x}",
                params=(
                    ErrorParam("x", ParamKind.STRING),
                    ErrorParam("x", ParamKind.I64),
                ),
            )

    def test_undefined_message_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="message references undefined"):
            ErrorDef(
                domain=ErrorDomain.TYPE,
                code=1,
                severity=Severity.ERROR,
                summary="Bad.",
                message="{missing_param}",
                params=(),
            )

    def test_undefined_fix_hint_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="fix_hint references undefined"):
            ErrorDef(
                domain=ErrorDomain.TYPE,
                code=1,
                severity=Severity.ERROR,
                summary="Bad.",
                message="ok",
                params=(),
                fix_hint="{missing_param}",
            )

    def test_no_params_no_placeholders_ok(self) -> None:
        """An error with no params and no placeholders is valid."""
        error = ErrorDef(
            domain=ErrorDomain.PARSE,
            code=1,
            severity=Severity.ERROR,
            summary="Simple error.",
            message="something went wrong",
            params=(),
        )
        assert error.params == ()

    def test_error_id_formatting(self) -> None:
        """error_id should be zero-padded to 3 digits."""
        error = ErrorDef(
            domain=ErrorDomain.SHAPE,
            code=12,
            severity=Severity.WARNING,
            summary="Test.",
            message="test",
            params=(),
        )
        assert error.error_id == "ERR_SHAPE_012"

    def test_repr_matches_error_id(self) -> None:
        error = ErrorDef(
            domain=ErrorDomain.FOLD,
            code=3,
            severity=Severity.REMARK,
            summary="Test.",
            message="test",
            params=(),
        )
        assert repr(error) == "ERR_FOLD_003"

    def test_frozen(self) -> None:
        """ErrorDef instances should be immutable."""
        error = ErrorDef(
            domain=ErrorDomain.TYPE,
            code=1,
            severity=Severity.ERROR,
            summary="Test.",
            message="test",
            params=(),
        )
        with pytest.raises(AttributeError):
            error.code = 2  # type: ignore[misc]


class TestCatalogIntegrity:
    """Tests for the full error catalog."""

    def test_all_errors_imports_clean(self) -> None:
        """ALL_ERRORS should import without error."""
        from loom.error import ALL_ERRORS

        assert len(ALL_ERRORS) > 0

    def test_no_duplicate_domain_code_pairs(self) -> None:
        """Every (domain, code) pair must be unique across all modules."""
        from loom.error import ALL_ERRORS

        seen: set[tuple[int, int]] = set()
        for error in ALL_ERRORS:
            key = (error.domain, error.code)
            assert key not in seen, (
                f"duplicate (domain, code): {error.error_id} "
                f"collides with existing {error.domain.name}/{error.code}"
            )
            seen.add(key)

    def test_all_severities_used(self) -> None:
        """At least ERROR and REMARK should be present."""
        from loom.error import ALL_ERRORS

        severities = {e.severity for e in ALL_ERRORS}
        assert Severity.ERROR in severities
        assert Severity.REMARK in severities

    def test_param_kinds_used(self) -> None:
        """STRING, I64, U32, U64, TYPE, and STRING_LIST should appear.

        BOOL is defined for future use (e.g., verifier warning flags)
        but no current error naturally needs a boolean parameter.
        """
        from loom.error import ALL_ERRORS

        kinds: set[ParamKind] = set()
        for error in ALL_ERRORS:
            for param in error.params:
                kinds.add(param.kind)
        for kind in (
            ParamKind.STRING,
            ParamKind.I64,
            ParamKind.U32,
            ParamKind.U64,
            ParamKind.TYPE,
            ParamKind.STRING_LIST,
        ):
            assert kind in kinds, f"ParamKind.{kind.name} is not used by any error"

    def test_catalog_fits_compact_error_refs(self) -> None:
        """C LOOM_ERROR_REF packs a domain and code into 16 bits."""
        from loom.error import ALL_ERRORS

        max_code = (1 << 10) - 1
        max_domain_value = (1 << 6) - 2
        for error in ALL_ERRORS:
            assert error.code <= max_code, (
                f"{error.error_id}: code {error.code} exceeds LOOM_ERROR_REF max"
            )
            assert error.domain.value <= max_domain_value, (
                f"{error.error_id}: domain {error.domain.name} exceeds "
                "LOOM_ERROR_REF max"
            )


class TestOldErrorsPorted:
    """Spot-check that key errors from the old system were ported."""

    def test_broadcast_ported(self) -> None:
        """Old LOOM_BROADCAST_0001 should map to ERR_SHAPE_004."""
        from loom.error.shape import ERR_SHAPE_004

        assert "broadcast" in ERR_SHAPE_004.summary.lower()
        assert ERR_SHAPE_004.domain == ErrorDomain.SHAPE
        assert ERR_SHAPE_004.code == 4

    def test_encoding_ported(self) -> None:
        """Old LOOM_ENCODING_0001 should map to ERR_ENCODING_001."""
        from loom.error.encoding import ERR_ENCODING_001

        assert "encoding" in ERR_ENCODING_001.summary.lower()

    def test_fill_ported(self) -> None:
        """Old LOOM_FILL_0001 should map to ERR_TYPE_006."""
        from loom.error.type import ERR_TYPE_006

        assert "fill" in ERR_TYPE_006.summary.lower()

    def test_constant_ported(self) -> None:
        """Old LOOM_CONSTANT_0001/0002 should map to ERR_TYPE_007."""
        from loom.error.type import ERR_TYPE_007

        assert "constant" in ERR_TYPE_007.summary.lower()

    def test_subrange_offset_ported(self) -> None:
        """Old LOOM_SUBRANGE_0001 should map to ERR_SUBRANGE_001."""
        from loom.error.subrange import ERR_SUBRANGE_001

        assert "offset" in ERR_SUBRANGE_001.summary.lower()

    def test_fold_poison_ported(self) -> None:
        """Old LOOM_FOLD_0001 should map to ERR_FOLD_001."""
        from loom.error.fold import ERR_FOLD_001

        assert ERR_FOLD_001.severity == Severity.REMARK
        assert "poison" in ERR_FOLD_001.summary.lower()

    def test_fold_negative_offset_ported(self) -> None:
        """Old LOOM_FOLD_0005 should map to ERR_FOLD_005."""
        from loom.error.fold import ERR_FOLD_005

        assert "negative" in ERR_FOLD_005.summary.lower()

    def test_elementwise_block_arg_count_ported(self) -> None:
        """Old LOOM_ELEMENTWISE_0004 should map to ERR_STRUCTURE_007."""
        from loom.error.structure import ERR_STRUCTURE_007

        assert "block argument" in ERR_STRUCTURE_007.summary.lower()

    def test_dim_index_ported(self) -> None:
        """Old LOOM_DIM_0001 should map to ERR_SUBRANGE_002."""
        from loom.error.subrange import ERR_SUBRANGE_002

        assert "dimension" in ERR_SUBRANGE_002.summary.lower()


class TestIndividualAccess:
    """Test that individual errors are accessible from the package."""

    def test_type_error_from_package(self) -> None:
        from loom.error import ERR_TYPE_001

        assert ERR_TYPE_001.error_id == "ERR_TYPE_001"

    def test_fold_error_from_package(self) -> None:
        from loom.error import ERR_FOLD_005

        assert ERR_FOLD_005.error_id == "ERR_FOLD_005"

    def test_parse_error_from_package(self) -> None:
        from loom.error import ERR_PARSE_003

        assert ERR_PARSE_003.error_id == "ERR_PARSE_003"
