# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json

import pytest

from loom.launch.errors import ProfileError
from loom.launch.hub import Hub, HubAuthRequired, HubError


def transport_for(routes, calls=None):
    """A transport answering by URL fragment."""

    def transport(url, headers):
        if calls is not None:
            calls.append((url, dict(headers)))
        for fragment, (status, body) in routes.items():
            if fragment in url:
                if isinstance(body, (dict, list)):
                    return status, json.dumps(body).encode("utf-8")
                return status, body
        return 404, b""

    return transport


SEARCH = [
    {
        "id": "loom-fun/a-loom",
        "sha": "a" * 40,
        "pipeline_tag": "text-generation",
        "tags": ["loom", "loom-serve", "base_model:org/a-GGUF"],
        "downloads": 3,
        "likes": 1,
        "gated": False,
        "cardData": {"base_model": "org/a-GGUF"},
        "siblings": [
            {"rfilename": "model.json"},
            {"rfilename": "info.json"},
            {"rfilename": "README.md"},
        ],
    },
    {
        "id": "somebody/catalog",
        "sha": "b" * 40,
        "tags": ["loom-serve"],
        "siblings": [{"rfilename": "catalog.json"}],
    },
]


def test_search_returns_models_with_files_and_base_model():
    hub = Hub(transport=transport_for({"api/models": (200, SEARCH)}))
    models = hub.search("a-loom", tags=("loom-serve",))
    assert [model.repo for model in models] == ["loom-fun/a-loom", "somebody/catalog"]
    first = models[0]
    assert first.has_file("model.json", "info.json")
    assert not first.has_file("catalog.json")
    assert first.base_model == "org/a-GGUF"
    assert first.revision == "a" * 40


def test_owner_prefix_becomes_an_author_filter():
    calls = []
    hub = Hub(transport=transport_for({"api/models": (200, SEARCH)}, calls=calls))
    hub.search("loom-fun/")
    url = calls[0][0]
    assert "author=loom-fun" in url
    assert "search=" not in url


def test_profiles_require_a_root_model_card():
    hub = Hub(transport=transport_for({"api/models": (200, SEARCH)}))
    profiles = hub.profiles("loom")
    assert [model.repo for model in profiles] == ["loom-fun/a-loom"]


def test_raw_reads_a_pinned_revision_and_caches_it():
    calls = []
    hub = Hub(transport=transport_for({"model.json": (200, b"{}")}, calls=calls))
    assert hub.raw("loom-fun/a-loom", "model.json", "c" * 40) == b"{}"
    assert hub.raw("loom-fun/a-loom", "model.json", "c" * 40) == b"{}"
    assert len(calls) == 1
    assert "/raw/" + "c" * 40 + "/model.json" in calls[0][0]


def test_missing_files_read_as_empty():
    hub = Hub(transport=transport_for({}))
    assert hub.raw_optional("loom-fun/a-loom", "info.json") == b""
    assert hub.json_optional("loom-fun/a-loom", "info.json") == {}


def test_invalid_json_reads_as_empty():
    hub = Hub(transport=transport_for({"info.json": (200, b"not json")}))
    assert hub.json_optional("loom-fun/a-loom", "info.json") == {}


def test_credentials_failure_explains_what_to_do():
    hub = Hub(transport=transport_for({"api/models": (401, b"unauthorized")}))
    with pytest.raises(HubAuthRequired) as error:
        hub.search("absent")
    assert "hf auth login" in str(error.value)


def test_a_token_is_sent_only_when_configured():
    calls = []
    hub = Hub(
        token="secret",
        transport=transport_for({"api/models": (200, SEARCH)}, calls=calls),
    )
    hub.search("a-loom")
    assert calls[0][1]["Authorization"] == "Bearer secret"
    anonymous = []
    hub = Hub(
        token="",
        transport=transport_for({"api/models": (200, SEARCH)}, calls=anonymous),
    )
    hub.search("a-loom")
    assert "Authorization" not in anonymous[0][1]


def test_server_failure_is_reported_with_its_status():
    hub = Hub(transport=transport_for({"api/models": (500, b"boom")}))
    with pytest.raises(HubError) as error:
        hub.search("absent")
    assert "HTTP 500" in str(error.value)


def test_model_detail_requires_a_repository_object():
    hub = Hub(transport=transport_for({"api/models/": (200, ["not", "an", "object"])}))
    with pytest.raises(HubError) as error:
        hub.model("loom-fun/a-loom")
    assert "does not describe a repository" in str(error.value)


def test_profile_requires_a_root_model_card():
    from loom.launch.profile import Profile

    repository = {
        "id": "loom-fun/catalog-only",
        "siblings": [{"rfilename": "catalog.json"}],
    }
    hub = Hub(transport=transport_for({"api/models/": (200, repository)}))
    with pytest.raises(ProfileError) as error:
        Profile.from_hub(hub, "loom-fun/catalog-only")
    assert "does not publish a root" in str(error.value)
