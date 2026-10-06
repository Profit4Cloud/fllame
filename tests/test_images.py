import urllib.error

import pytest

from fllame.compose.images import ImageResolveError, is_floating, pin_image

_HUB = "https://hub.docker.com/v2/repositories"


def _fake_hub(latest_digest: str, pages: list[list[dict]], repository="vllm/vllm-openai"):
    tags_url = f"{_HUB}/{repository}/tags?page_size=100&ordering=last_updated&name=v"
    responses = {f"{_HUB}/{repository}/tags/latest": {"digest": latest_digest}}
    for i, results in enumerate(pages):
        next_url = f"page-{i + 1}" if i + 1 < len(pages) else None
        responses[tags_url if i == 0 else f"page-{i}"] = {"results": results, "next": next_url}
    return responses.__getitem__


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("vllm/vllm-openai:latest", True),
        ("vllm/vllm-openai", True),
        ("localhost:5000/vllm", True),
        ("vllm/vllm-openai:v0.30.0", False),
        ("vllm/vllm-openai:nightly", False),
        ("vllm/vllm-openai@sha256:abc", False),
    ],
)
def test_is_floating(image, expected):
    assert is_floating(image) is expected


def test_pinned_image_is_returned_without_network():
    def fetch(url):
        raise AssertionError("no network expected")

    assert pin_image("vllm/vllm-openai:v0.30.0", fetch) == "vllm/vllm-openai:v0.30.0"


def test_latest_resolves_to_release_tag_with_same_digest():
    fetch = _fake_hub(
        "sha256:aaa",
        [
            [
                {"name": "v0.31.0-cu129", "digest": "sha256:bbb"},
                {"name": "v0.31.0-aarch64", "digest": "sha256:ccc"},
                {"name": "v0.31.0", "digest": "sha256:aaa"},
                {"name": "v0.30.0", "digest": "sha256:ddd"},
            ]
        ],
    )

    assert pin_image("vllm/vllm-openai:latest", fetch) == "vllm/vllm-openai:v0.31.0"


def test_untagged_image_is_treated_as_latest():
    fetch = _fake_hub("sha256:aaa", [[{"name": "v0.31.0", "digest": "sha256:aaa"}]])
    assert pin_image("vllm/vllm-openai", fetch) == "vllm/vllm-openai:v0.31.0"


def test_release_tag_found_on_a_later_page():
    fetch = _fake_hub(
        "sha256:aaa",
        [
            [{"name": "v0.32.0rc1", "digest": "sha256:zzz"}],
            [{"name": "v0.31.0", "digest": "sha256:aaa"}],
        ],
    )

    assert pin_image("vllm/vllm-openai:latest", fetch) == "vllm/vllm-openai:v0.31.0"


def test_falls_back_to_digest_when_no_release_tag_matches():
    fetch = _fake_hub("sha256:aaa", [[{"name": "v0.31.0-cu129", "digest": "sha256:aaa"}]])
    assert pin_image("vllm/vllm-openai:latest", fetch) == "vllm/vllm-openai@sha256:aaa"


def test_official_image_uses_library_namespace():
    fetch = _fake_hub("sha256:aaa", [[]], repository="library/ubuntu")
    assert pin_image("ubuntu", fetch) == "ubuntu@sha256:aaa"


def test_docker_io_prefix_is_docker_hub():
    fetch = _fake_hub("sha256:aaa", [[{"name": "v1.0.0", "digest": "sha256:aaa"}]])
    pinned = pin_image("docker.io/vllm/vllm-openai:latest", fetch)
    assert pinned == "docker.io/vllm/vllm-openai:v1.0.0"


def test_other_registry_cannot_be_resolved():
    with pytest.raises(ImageResolveError, match="Docker Hub"):
        pin_image("ghcr.io/org/image:latest", lambda url: {})


def test_network_error_is_reported():
    def fetch(url):
        raise urllib.error.URLError("offline")

    with pytest.raises(ImageResolveError, match="offline"):
        pin_image("vllm/vllm-openai:latest", fetch)
