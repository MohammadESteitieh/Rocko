#!/bin/bash
set -u

BASE=/Users/msteitieh/Desktop/CU-hakcing-captures/current-hamming
MPATH="$BASE/manifests/dataset_receiver.metapath"
PPATH="$BASE/manifests/dataset_receiver.pid"
RPATH="$BASE/manifests/dataset_receiver.path"
STATE="$BASE/manifests/dataset_supervisor.state"
SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=5 qnxuser@172.20.10.2)
SCP=(sshpass -e scp -O -q -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=5)

meta=$(cat "$MPATH")
receiver_pid=$(cat "$PPATH")
raw=$(cat "$RPATH")
run=$(awk -F= '$1=="run"{print $2}' "$meta")
remote_manifest="${run}.transmitter.csv"
remote_log="${run}.transmitter.log"

printf 'supervisor_started_utc=%s\n' "$(date -u +%FT%TZ)" >> "$meta"
echo "WAITING_FOR_PI $(date -u +%FT%TZ)" > "$STATE"

# Never launch the transmitter if the intended continuous receiver has died.
while kill -0 "$receiver_pid" 2>/dev/null; do
    if "${SSH[@]}" 'echo READY' 2>/dev/null | grep -q READY; then
        break
    fi
    sleep 5
done
if ! kill -0 "$receiver_pid" 2>/dev/null; then
    echo "ABORTED_RECEIVER_NOT_RUNNING $(date -u +%FT%TZ)" > "$STATE"
    exit 1
fi

start_output=$("${SSH[@]}" "cd /data/home/qnxuser/transmitter || exit 1
for p in 27 18 22 17; do printf out > /dev/gpio/\$p; printf off > /dev/gpio/\$p; done
rm -f /tmp/beacon.pid 2>/dev/null || true
echo qnxuser | sudo -S rm -f /tmp/alphabet_beacon.pid >/dev/null 2>&1 || true
nohup ./duty_pair_test.py --dataset --manifest '$remote_manifest' > '$remote_log' 2>&1 </dev/null &
pid=\$!
echo \$pid > /tmp/descending_dataset.pid
echo \$pid" 2>/dev/null)
remote_pid=$(printf '%s\n' "$start_output" | tail -1)
case "$remote_pid" in ''|*[!0-9]*)
    echo "FAILED_TO_START $(date -u +%FT%TZ) output=$start_output" > "$STATE"
    exit 1;;
esac
printf 'transmitter_started_utc=%s\nremote_pid=%s\nremote_manifest=%s\nremote_log=%s\n' \
    "$(date -u +%FT%TZ)" "$remote_pid" "$remote_manifest" "$remote_log" >> "$meta"
echo "RUNNING pid=$remote_pid $(date -u +%FT%TZ)" > "$STATE"

while :; do
    status=$("${SSH[@]}" "cd /data/home/qnxuser/transmitter || exit 1
if grep -q 'DIAGNOSTIC COMPLETE' '$remote_log' 2>/dev/null; then echo COMPLETE
elif kill -0 '$remote_pid' 2>/dev/null; then echo RUNNING
else echo FAILED; fi" 2>/dev/null || echo UNREACHABLE)
    case "$status" in
        *COMPLETE*) outcome=COMPLETE; break;;
        *FAILED*) outcome=FAILED; break;;
        *) sleep 20;;
    esac
done

echo "$outcome $(date -u +%FT%TZ)" > "$STATE"
# Preserve a clean post-run H0 interval before closing the one continuous CSV.
sleep 30
kill -TERM "$receiver_pid" 2>/dev/null || true
sleep 2
kill -KILL "$receiver_pid" 2>/dev/null || true
mkdir -p "$BASE/manifests/pi"
"${SCP[@]}" "qnxuser@172.20.10.2:/data/home/qnxuser/transmitter/$remote_manifest" "$BASE/manifests/pi/" 2>/dev/null || true
"${SCP[@]}" "qnxuser@172.20.10.2:/data/home/qnxuser/transmitter/$remote_log" "$BASE/manifests/pi/" 2>/dev/null || true
"${SSH[@]}" 'for p in 27 18 22 17; do printf out > /dev/gpio/$p 2>/dev/null || true; printf off > /dev/gpio/$p 2>/dev/null || true; done' 2>/dev/null || true
shasum -a 256 "$raw" > "${raw%.csv}.sha256"
printf 'dataset_outcome=%s\nreceiver_stopped_utc=%s\nraw_sha256=%s\n' \
    "$outcome" "$(date -u +%FT%TZ)" "$(awk '{print $1}' "${raw%.csv}.sha256")" >> "$meta"
echo "ARCHIVED outcome=$outcome $(date -u +%FT%TZ)" > "$STATE"
