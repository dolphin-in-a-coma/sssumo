"""Exercise the patched proxy-token refresh against a stubbed client.

No VM and no network: `state._client` is replaced with a stub, so this only
checks the logic 0001-refresh-proxy-token.patch adds. Run it with the CLI's
own interpreter after regenerating the patch against a new colab_cli --
`apply_cli_patch.py --apply` runs it for you.

    <cli-python> scripts/colab/patches/test_refresh.py
"""
import base64, json, os, tempfile, time, types

from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.commands.session import proxy_token_expiry, refresh_proxy_token

tmp = tempfile.mkdtemp()
state.config_path = os.path.join(tmp, "sessions.json")

def tok(seconds):
    body = base64.urlsafe_b64encode(json.dumps(
        {"aud": "ep-1", "exp": int(time.time() + seconds), "port": 1}).encode()
    ).rstrip(b"=").decode()
    return "h." + body + ".s"

assert proxy_token_expiry("not-a-jwt") is None
assert abs(proxy_token_expiry(tok(100)) - (time.time() + 100)) < 2
print("PASS  expiry decoding")

class FakeClient:
    calls = 0
    def list_assignments(self):
        FakeClient.calls += 1
        info = types.SimpleNamespace(token=tok(3600), url="https://new",
                                     token_expires_in_seconds=3600)
        return [types.SimpleNamespace(endpoint="ep-1", runtime_proxy_info=info)]
state._client = FakeClient()

# 1. a fresh token is left alone, and costs no API call
s = SessionState(name="j", token=tok(3000), url="https://old", endpoint="ep-1",
                 kernel_id="k0")
state.store.add(s)
assert refresh_proxy_token(s) is False
assert FakeClient.calls == 0
assert state.store.get("j").url == "https://old"
print("PASS  fresh token untouched, no API call")

# 2. a token inside the margin is re-minted
s = SessionState(name="j", token=tok(300), url="https://old", endpoint="ep-1",
                 kernel_id="k0")
state.store.add(s)
# something else records a kernel id in between, as an exec would
state.store.update("j", kernel_id="k1", session_id="sess-9")
assert refresh_proxy_token(s) is True
got = state.store.get("j")
assert got.url == "https://new", got.url
assert proxy_token_expiry(got.token) - time.time() > 3000
assert got.kernel_id == "k1" and got.session_id == "sess-9", "update clobbered a field"
print("PASS  stale token re-minted; concurrent kernel_id survived")

# 3. an unreadable token forces a refresh rather than being trusted
s = SessionState(name="j", token="garbage", url="https://old", endpoint="ep-1")
state.store.add(s)
assert refresh_proxy_token(s) is True
print("PASS  unreadable token refreshed")

# 4. a vanished assignment is not an error here
s = SessionState(name="j", token=tok(10), url="https://old", endpoint="ep-gone")
state.store.add(s)
assert refresh_proxy_token(s) is False
print("PASS  missing assignment returns False, does not raise")

# 5. update on an absent session is a no-op
assert state.store.update("nope", token="x") is None
print("PASS  update of an unknown session returns None")
print("\nall patch unit checks passed")
