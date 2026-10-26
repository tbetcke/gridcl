#!/usr/bin/env bash
# usage: scripts/ci-apt-install.sh <package>...
# Installs Ubuntu packages on a hosted CI runner. The runners' mirror is usually fast but
# sometimes serves large packages (the ROCm libraries that libfabric pulls in) at a crawl.
# The first two attempts get APT_ATTEMPT_SECONDS (default 240) each; the third has no limit,
# so a slow mirror delays the job but does not fail it. Partial downloads carry over.
# With APT_ARCHIVES set, the .deb files are kept there (and owned by the runner's user), so
# that the CI cache can restore them and apt downloads only what changed.
set -u
set -o pipefail

[ $# -gt 0 ] || {
    sed -n '2p' "$0" | sed 's/^# //' >&2
    exit 2
}
limit=${APT_ATTEMPT_SECONDS:-240}
opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30)
if [ -n "${APT_ARCHIVES:-}" ]; then
    mkdir -p "$APT_ARCHIVES/partial"
    opts+=(-o "Dir::Cache::Archives=$APT_ARCHIVES")
fi
apt_install() { # apt_install <seconds or 0 for no limit> <package>...
    local seconds=$1
    shift
    local run=(sudo)
    [ "$seconds" = 0 ] || run+=(timeout "$seconds")
    "${run[@]}" apt-get "${opts[@]}" update &&
        "${run[@]}" apt-get "${opts[@]}" install -y --no-install-recommends "$@"
}
status=1
for attempt in 1 2 3; do
    seconds=$limit
    [ "$attempt" -lt 3 ] || seconds=0
    if apt_install "$seconds" "$@"; then
        status=0
        break
    fi
    echo "ci-apt-install: attempt $attempt failed or took over $limit s" >&2
    # An interrupted install can leave dpkg half-configured.
    sudo dpkg --configure -a || true
done
if [ -n "${APT_ARCHIVES:-}" ]; then
    sudo rm -rf "$APT_ARCHIVES/partial" "$APT_ARCHIVES/lock"
    sudo chown -R "$(id -u):$(id -g)" "$APT_ARCHIVES"
fi
[ "$status" = 0 ] || echo "ci-apt-install: giving up on $*" >&2
exit "$status"
