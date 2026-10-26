#!/usr/bin/env bash
# usage: scripts/ci-apt-install.sh <package>...
# Installs Ubuntu packages on a hosted CI runner. The runners' package mirror is usually fast
# (about 20 s) but sometimes stalls for 10-30 min, so each attempt (index update and install)
# gets APT_ATTEMPT_SECONDS (default 240) and fails over to the next of three attempts.
set -u
set -o pipefail

[ $# -gt 0 ] || {
    sed -n '2p' "$0" | sed 's/^# //' >&2
    exit 2
}
limit=${APT_ATTEMPT_SECONDS:-240}
for attempt in 1 2 3; do
    if sudo timeout "$limit" sh -c '
        apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update &&
            apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
                install -y --no-install-recommends "$@"
    ' sh "$@"; then
        exit 0
    fi
    echo "ci-apt-install: attempt $attempt failed or took over $limit s" >&2
    # An interrupted install can leave dpkg half-configured.
    sudo dpkg --configure -a || true
done
echo "ci-apt-install: giving up on $*" >&2
exit 1
