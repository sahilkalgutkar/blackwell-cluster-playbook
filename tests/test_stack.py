from bcp.stack import (
    check,
    is_stable,
    latest_dockerhub_tag,
    latest_github_release,
    latest_github_tag,
    render_markdown,
    rewrite_pinned,
    version_key,
)


class FakeHttp:
    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def get_json(self, url):
        self.calls.append(url)
        for prefix, payload in self.responses.items():
            if url.startswith(prefix):
                return payload
        raise RuntimeError(f"unexpected url {url}")


def release(tag, published="2026-01-01T00:00:00Z", prerelease=False, draft=False):
    return {"tag_name": tag, "published_at": published, "prerelease": prerelease, "draft": draft}


def test_version_helpers():
    assert version_key("v2.32.6-1") > version_key("v2.9.9-1")
    assert not is_stable("v0.33.0rc3")
    assert not is_stable("v1.0.0-beta.2")
    assert is_stable("v26.7.2")


def test_release_skips_rc_and_other_projects():
    http = FakeHttp(
        {
            "https://api.github.com/repos/NVIDIA/nccl/releases": [
                release("nccl4py-v0.6.0"),
                release("v2.33.0rc1"),
                release("v2.32.6-1"),
                release("v2.32.5-1"),
            ]
        }
    )
    assert latest_github_release(http, "NVIDIA/nccl") == "nccl4py-v0.6.0"  # why patterns exist
    assert latest_github_release(http, "NVIDIA/nccl", r"^v\d+\.\d+\.\d+-\d+$") == "v2.32.6-1"


def test_release_order_by_date_handles_renumbering():
    http = FakeHttp(
        {
            "https://api.github.com/repos/x/y/releases": [
                release("v25.12.0", "2025-12-01T00:00:00Z"),
                release("v0.5.1", "2026-10-12T00:00:00Z"),
            ]
        }
    )
    assert latest_github_release(http, "x/y") == "v25.12.0"
    assert latest_github_release(http, "x/y", order="date") == "v0.5.1"


def test_tags_and_dockerhub():
    http = FakeHttp(
        {
            "https://api.github.com/repos/NVIDIA/nccl-tests/tags": [
                {"name": "v2.21.1"},
                {"name": "v2.3.0"},
                {"name": "v2.22.0-rc1"},
            ],
            "https://hub.docker.com/v2/repositories/nvidia/cuda/tags": {
                "results": [
                    {"name": "13.5.0-devel-ubuntu24.04"},
                    {"name": "13.4.3-devel-ubuntu24.04"},
                    {"name": "13.5.0-runtime-ubuntu24.04"},
                ]
            },
        }
    )
    assert latest_github_tag(http, "NVIDIA/nccl-tests") == "v2.21.1"
    assert (
        latest_dockerhub_tag(http, "nvidia/cuda", r"(\d+\.\d+\.\d+)-devel-ubuntu24\.04") == "13.5.0"
    )


def test_check_states():
    http = FakeHttp(
        {
            "https://api.github.com/repos/a/a/releases": [release("v1.2.0")],
            "https://api.github.com/repos/b/b/releases": [release("v1.0.0")],
            "https://api.github.com/repos/c/c/releases": [],
        }
    )
    components = [
        {"name": "a", "source": "github-release", "repo": "a/a", "pinned": "v1.1.0"},
        {"name": "b", "source": "github-release", "repo": "b/b", "pinned": "v1.0.0"},
        {"name": "c", "source": "github-release", "repo": "c/c", "pinned": "v1.0.0"},
        {"name": "d", "source": "manual", "pinned": None},
        {"name": "e", "source": "github-release", "repo": "missing/repo", "pinned": "v1"},
    ]
    states = {s.name: s.state for s in check(components, http)}
    assert states == {"a": "update", "b": "current", "c": "error", "d": "manual", "e": "error"}
    md = render_markdown(check(components, http))
    assert "| a | v1.1.0 | **v1.2.0** |" in md


def test_rewrite_pinned_keeps_comments_and_other_entries():
    text = (
        "components:\n"
        "  # comment survives\n"
        "  - name: nccl\n"
        "    repo: NVIDIA/nccl\n"
        '    pinned: "v2.32.6-1"   # note\n'
        "  - name: vllm\n"
        "    pinned: null\n"
        "  - name: dynamo\n"
        '    pinned: "v1.6.2"\n'
    )
    out = rewrite_pinned(text, {"nccl": "v2.33.0-1", "vllm": "v0.32.0"})
    assert '    pinned: "v2.33.0-1"   # note\n' in out
    assert '    pinned: "v0.32.0"\n' in out
    assert '    pinned: "v1.6.2"\n' in out
    assert "# comment survives" in out


def test_ngc_source_uses_an_anonymous_pull_token():
    from bcp import stack

    calls = []

    class FakeNgc:
        def get_json(self, url, headers=None):
            calls.append((url, headers))
            if "proxy_auth" in url:
                return {"token": "anon"}
            return {
                "tags": [
                    "26.09-py3",
                    "26.10-py3",
                    "26.11-py3-igpu",
                    "26.12rc1-py3",
                    "sha256-abc.sig",
                ]
            }

    comp = {
        "name": "ngc-pytorch",
        "source": "ngc",
        "repo": "nvidia/pytorch",
        "tag_pattern": r"(\d\d\.\d\d)-py3",
        "pinned": "26.09",
    }
    [status] = stack.check([comp], FakeNgc())
    assert (status.state, status.latest) == ("update", "26.10")
    assert calls[1][1] == {"Authorization": "Bearer anon"}
