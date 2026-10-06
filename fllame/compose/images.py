"""Pins a floating `latest` image to the release it currently points at,
so a built compose.yaml or Dockerfile never drifts to another vLLM version.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable

_HUB_API = "https://hub.docker.com/v2/repositories"
_RELEASE_TAG = re.compile(r"^v?\d+\.\d+\.\d+$")
# vllm/vllm-openai pushes many nightly and per-arch tags a day; a few
# pages of `v`-prefixed tags reach well past the latest release.
_MAX_PAGES = 5


class ImageResolveError(RuntimeError):
    pass


class HubUnreachableError(ImageResolveError):
    pass


def is_floating(image: str) -> bool:
    if "@" in image:
        return False
    _, tag = _split_tag(image)
    return tag in (None, "latest")


def pin_image(image: str, fetch_json: Callable[[str], dict] | None = None) -> str:
    """`image` itself when it isn't floating. Otherwise the release tag
    sharing `latest`'s digest."""
    if not is_floating(image):
        return image
    fetch = fetch_json or _fetch_json
    name, _ = _split_tag(image)
    repository = _hub_repository(name)
    if repository is None:
        raise ImageResolveError(
            f"can't resolve '{image}' to a fixed version: only Docker Hub images "
            "are supported. Use an image with a version tag."
        )

    try:
        digest = fetch(f"{_HUB_API}/{repository}/tags/latest")["digest"]
        url = f"{_HUB_API}/{repository}/tags?page_size=100&ordering=last_updated&name=v"
        for _ in range(_MAX_PAGES):
            page = fetch(url)
            for tag in page.get("results", []):
                if tag.get("digest") == digest and _RELEASE_TAG.match(tag["name"]):
                    return f"{name}:{tag['name']}"
            url = page.get("next")
            if not url:
                break
    except OSError as e:
        raise HubUnreachableError(
            f"can't resolve '{image}' to a fixed version: Docker Hub unreachable ({e})"
        ) from e
    except (ValueError, KeyError, TypeError) as e:
        raise ImageResolveError(
            f"can't resolve '{image}' to a fixed version via Docker Hub: {e}"
        ) from e
    raise ImageResolveError(
        f"can't resolve '{image}' to a fixed version: no release tag (vX.Y.Z) on "
        "Docker Hub matches it. Set an image with a version tag via "
        "`fllame config set-default-image` or in the recipe."
    )


def image_name(image: str) -> str:
    return _split_tag(image)[0]


def newest_release(name: str, tags: list[str]) -> str | None:
    releases = [tag for tag in tags if _RELEASE_TAG.match(tag)]
    if not releases:
        return None
    newest = max(releases, key=lambda tag: tuple(int(part) for part in tag.lstrip("v").split(".")))
    return f"{name}:{newest}"


def _split_tag(image: str) -> tuple[str, str | None]:
    last_part = image.rsplit("/", 1)[-1]
    if ":" not in last_part:
        return image, None
    name, tag = image.rsplit(":", 1)
    return name, tag


def _hub_repository(name: str) -> str | None:
    """`namespace/repo` on Docker Hub, or `None` for another registry."""
    parts = name.split("/")
    first = parts[0]
    if len(parts) > 1 and ("." in first or ":" in first or first == "localhost"):
        if first not in ("docker.io", "index.docker.io", "registry-1.docker.io"):
            return None
        parts = parts[1:]
    if len(parts) == 1:
        parts = ["library", *parts]
    return "/".join(parts)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)
