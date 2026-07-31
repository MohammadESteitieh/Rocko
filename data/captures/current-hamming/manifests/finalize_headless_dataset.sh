#!/bin/bash
set -u
BASE=/Users/msteitieh/Desktop/CU-hakcing-captures/current-hamming
IP=172.17.235.186
meta=$(cat "$BASE/manifests/dataset_receiver.metapath")
raw=$(cat "$BASE/manifests/dataset_receiver.path")
cpid=$(cat "$BASE/manifests/dataset_capture.pid")
remote_pid=$(awk -F= '$1=="remote_pid"{print $2}' "$meta")
remote_manifest=$(awk -F= '$1=="remote_manifest"{print $2}' "$meta")
remote_log=$(awk -F= '$1=="remote_log"{print $2}' "$meta")
state="$BASE/manifests/dataset_finalizer.state"
SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 qnxuser@$IP)
SCP=(sshpass -e scp -O -q -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8)
echo "WAITING $(date -u +%FT%TZ)" > "$state"
sleep 2220
status=$("${SSH[@]}" "cd /data/home/qnxuser/transmitter && if grep -q 'DIAGNOSTIC COMPLETE' '$remote_log' 2>/dev/null; then echo COMPLETE; elif kill -0 '$remote_pid' 2>/dev/null; then echo STILL_RUNNING; else echo FAILED; fi" 2>/dev/null) || { echo "NETWORK_ISSUE $(date -u +%FT%TZ)" > "$state"; exit 2; }
if [ "$status" != COMPLETE ]; then echo "$status $(date -u +%FT%TZ)" > "$state"; exit 3; fi
# capture.py should stop itself after 2200 seconds; terminate only if still lingering.
if kill -0 "$cpid" 2>/dev/null; then kill -TERM "$cpid" 2>/dev/null || true; sleep 2; fi
mkdir -p "$BASE/manifests/pi"
"${SCP[@]}" "qnxuser@$IP:/data/home/qnxuser/transmitter/$remote_manifest" "$BASE/manifests/pi/"
"${SCP[@]}" "qnxuser@$IP:/data/home/qnxuser/transmitter/$remote_log" "$BASE/manifests/pi/"
"${SSH[@]}" 'for p in 27 18 22 17; do printf out > /dev/gpio/$p; printf off > /dev/gpio/$p; done'
shasum -a 256 "$raw" > "${raw%.csv}.sha256"
printf 'dataset_outcome=COMPLETE\narchived_utc=%s\nraw_sha256=%s\n' "$(date -u +%FT%TZ)" "$(awk '{print $1}' "${raw%.csv}.sha256")" >> "$meta"
echo "ARCHIVED $(date -u +%FT%TZ)" > "$state"
