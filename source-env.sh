#!/usr/bin/env bash

if [[ -z "${REQUESTS_CA_BUNDLE:-}" && -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

# Reuse an existing login to avoid the shared anonymous GitHub rate limit.
if [[ -z "${GITHUB_TOKEN:-}" ]] && command -v gh >/dev/null 2>&1; then
  GITHUB_TOKEN="$(gh auth token --hostname github.com 2>/dev/null || true)"
  if [[ -n "$GITHUB_TOKEN" ]]; then
    export GITHUB_TOKEN
  fi
fi
