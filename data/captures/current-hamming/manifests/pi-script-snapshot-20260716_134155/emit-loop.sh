#!/bin/sh
# Repeated coil beacon without microphone input.
# Usage: ./emit-loop.sh EVENT [PAUSE_SECONDS]
# Example: ./emit-loop.sh injured 30

set -u
TX=/data/home/qnxuser/transmitter/transmitter.py
EVENT=${1:-}
PAUSE=${2:-30}
CHILD=""

usage() {
    echo "Usage: $0 {fire|trapped|lost|injured|sos|heartbeat|4-bit-code} [pause-seconds]" >&2
    exit 2
}

[ -n "$EVENT" ] || usage
case "$EVENT" in
    fire|trapped|lost|injured|sos|heartbeat|[01][01][01][01]) ;;
    *) usage ;;
esac
case "$PAUSE" in
    ""|*[!0-9]*) usage ;;
esac
[ -f "$TX" ] || { echo "ERROR: missing $TX" >&2; exit 1; }

safe_off() {
    # Defense in depth: transmitter.py also forces these safe values itself.
    printf off > /dev/gpio/27 2>/dev/null || true
    printf off > /dev/gpio/22 2>/dev/null || true
    printf off > /dev/gpio/17 2>/dev/null || true
}

cleanup() {
    trap - INT TERM EXIT
    if [ -n "$CHILD" ]; then
        kill -TERM "$CHILD" 2>/dev/null || true
        wait "$CHILD" 2>/dev/null || true
    fi
    safe_off
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] stopped; coil forced off"
}
trap cleanup INT TERM EXIT

echo "Rocko direct emission loop"
echo "event=$EVENT; pause=${PAUSE}s; microphone bypassed"
echo "Each emission is one complete 12-bit frame. Ctrl+C stops safely."

COUNT=0
while :; do
    COUNT=$((COUNT + 1))
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] emission #$COUNT starting: $EVENT"
    python3 "$TX" --send "$EVENT" &
    CHILD=$!
    wait "$CHILD"
    RC=$?
    CHILD=""
    if [ "$RC" -ne 0 ]; then
        echo "ERROR: transmitter exited with status $RC; stopping" >&2
        exit "$RC"
    fi
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] emission #$COUNT complete; pausing ${PAUSE}s"
    sleep "$PAUSE"
done
