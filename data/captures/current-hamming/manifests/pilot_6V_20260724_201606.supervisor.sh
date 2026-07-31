#!/bin/bash
set -u
BASE=/Users/msteitieh/Desktop/CU-hakcing-captures/current-hamming
REPO=/Users/msteitieh/Desktop/CU-hakcing-2026
RUN="pilot_6V_20260724_201606"
RAW="$BASE/raw/$RUN.csv"
META="$BASE/manifests/$RUN.metadata.txt"
CAPLOG="$BASE/manifests/$RUN.capture.log"
TXMAN="$BASE/manifests/pi/$RUN.transmitter.csv"
TXLOG="$BASE/manifests/pi/$RUN.transmitter.log"
REMOTE=/data/home/qnxuser/calibration-runner
SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8 qnxuser@172.17.235.186)
SCP=(sshpass -e scp -O -q -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8)
SAFE='safe_status=0; for p in 27 18 22 17; do printf out > /dev/gpio/$p 2>/dev/null || safe_status=1; printf off > /dev/gpio/$p 2>/dev/null || safe_status=1; done; [ $safe_status -eq 0 ] || exit 1'
CPID=''
utc() { /usr/bin/python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"))'; }
cleanup() {
  "${SSH[@]}" "$SAFE" >/dev/null 2>&1 || true
  if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then kill -INT "$CPID" 2>/dev/null || true; sleep 2; kill -KILL "$CPID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
mkdir -p "$BASE/raw" "$BASE/manifests/pi"
printf 'run=%s\nvoltage_v=6\ndistance_m=0.35\nhardware_note=l298n-onboard-jumpers-observed-exact-module-pending\nplanned_utc=%s\nraw_path=%s\n' "$RUN" "$(utc)" "$RAW" > "$META"
"${SSH[@]}" "$SAFE" || { echo preflight_gpio_safe_failed; exit 1; }
/Users/msteitieh/Desktop/CU-hakcing-2026/receiver/.venv/bin/python "$REPO/receiver/rocko_receiver.py" --port /dev/cu.usbmodem1201 --baud 115200 --output "$RAW" --plot-seconds 90 > "$CAPLOG" 2>&1 &
CPID=$!
printf 'capture_started_utc=%s\ncapture_pid=%s\n' "$(utc)" "$CPID" >> "$META"
sleep 3
kill -0 "$CPID" 2>/dev/null || { echo live_receiver_failed; exit 1; }
printf 'transmitter_requested_utc=%s\n' "$(utc)" >> "$META"
"${SSH[@]}" "cd '$REMOTE' || exit 1; $SAFE; ./duty_pair_test.py --single-duty 100 --letter A --manifest '$RUN.transmitter.csv' > '$RUN.transmitter.log' 2>&1; status=\$?; $SAFE; exit \$status" || { echo transmitter_failed; exit 1; }
echo 'Pilot frame complete; collecting 15-second H0 gap.'
sleep 15
kill -INT "$CPID" 2>/dev/null || true
wait "$CPID" 2>/dev/null || true
CPID=''
"${SCP[@]}" "qnxuser@172.17.235.186:$REMOTE/$RUN.transmitter.csv" "$TXMAN" || exit 1
"${SCP[@]}" "qnxuser@172.17.235.186:$REMOTE/$RUN.transmitter.log" "$TXLOG" || exit 1
"${SSH[@]}" "$SAFE" || { echo final_gpio_safe_failed; exit 1; }
printf 'outcome=COMPLETE\nsafe_shutdown_outcome=VERIFIED_WRITES\nfinished_utc=%s\n' "$(utc)" >> "$META"
for f in "$RAW" "$META" "$CAPLOG" "$TXMAN" "$TXLOG"; do shasum -a 256 "$f" > "$f.sha256"; done
trap - EXIT INT TERM
echo "PILOT COMPLETE raw=$RAW"
