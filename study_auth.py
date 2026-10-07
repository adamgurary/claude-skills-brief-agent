"""SDK OAuth login with a reserved localhost callback and reusable SDK cache."""

from __future__ import annotations

import argparse
import errno
import functools
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.credentials_provider import credentials_strategy
from databricks.sdk.oauth import OAuthClient, TokenCache

CALLBACK_PORTS = range(8020, 8041)
LOGIN_TIMEOUT_SECONDS = 300


@credentials_strategy("study-oidc-discovery", [])
def _discovery_credentials(_):
    return lambda: {}


class _Callback(BaseHTTPRequestHandler):
    def __init__(self, *args, state: str, feedback: list, **kwargs):
        self.expected_state = state
        self.feedback = feedback
        super().__init__(*args, **kwargs)

    def log_message(self, *_):
        # Callback URLs contain authorization codes.
        pass

    def do_GET(self):
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        if (parts.path != "/" or query.get("state") != [self.expected_state]
                or not (query.get("code") or query.get("error"))):
            self.send_error(400, "Invalid login callback")
            return
        self.feedback.append({key: values[0] for key, values in query.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b"Login received. Return to your terminal. You can close this tab.")


def _reserve_callback() -> HTTPServer:
    # Match the official CLI's callback range and keep the socket until login ends.
    for port in CALLBACK_PORTS:
        try:
            return HTTPServer(("localhost", port), BaseHTTPRequestHandler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    raise RuntimeError(
        "Login callback ports 8020-8040 are all busy. Finish other Databricks "
        "login attempts, then rerun bash setup-v2.sh. Your files are preserved."
    )


def _browser_login(endpoints, cache: TokenCache, scopes: list[str]) -> None:
    with _reserve_callback() as server:
        redirect = f"http://localhost:{server.server_port}"
        consent = OAuthClient(endpoints, redirect, "databricks-cli", scopes=scopes).initiate_consent()
        feedback = []
        server.RequestHandlerClass = functools.partial(
            _Callback, state=consent.as_dict()["state"], feedback=feedback,
        )
        print(f"Sign in to Databricks in your browser (callback port {server.server_port}).", flush=True)
        if not webbrowser.open(consent.authorization_url, new=2):
            print(f"Open this login link yourself:\n{consent.authorization_url}", flush=True)
        deadline = time.monotonic() + LOGIN_TIMEOUT_SECONDS
        server.timeout = 1
        while not feedback:
            if time.monotonic() >= deadline:
                raise RuntimeError("Browser login timed out. Rerun setup to try again. Your files are preserved.")
            server.handle_request()
        credentials = consent.exchange_callback_parameters(feedback[0])
        cache.save(credentials)


def browser_workspace_client(host: str) -> WorkspaceClient:
    config = Config(host=host, credentials_strategy=_discovery_credentials)
    endpoints = config.databricks_oidc_endpoints
    if endpoints is None:
        raise RuntimeError("This workspace did not provide Databricks OAuth endpoints.")
    scopes = ["all-apis", "offline_access"]
    cache = TokenCache(
        host=config.host, oidc_endpoints=endpoints, client_id="databricks-cli",
        redirect_url="http://localhost:8020", scopes=scopes,
    )
    credentials = cache.load()
    if credentials is not None:
        try:
            credentials.token()
        except Exception:
            credentials = None
    if credentials is None:
        _browser_login(endpoints, cache, scopes)
    return WorkspaceClient(host=config.host, auth_type="external-browser")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["auth_mode"] == "external-browser":
        client = browser_workspace_client(manifest["workspace_host"])
        if client.current_user.me().user_name != manifest["principal"]:
            raise RuntimeError("Login does not match this study attempt. Your files are preserved.")


if __name__ == "__main__":
    main()
