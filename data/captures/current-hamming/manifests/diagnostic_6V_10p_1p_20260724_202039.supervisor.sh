#!/bin/bash
set -u
BASE=/Users/msteitieh/Desktop/CU-hakcing-captures/current-hamming
REPO=/Users/msteitieh/Desktop/CU-hakcing-2026
RUN="diagnostic_6V_10p_1p_20260724_202039"
RAW="$BASE/raw/$RUN.csv"; META="$BASE/manifests/$RUN.metadata.txt"; CAPLOG="$BASE/manifests/$RUN.capture.log"
REMOTE=/data/home/qnxuser/calibration-runner
SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8 qnxuser@172.17.235.186)
SCP=(sshpass -e scp -O -q -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8)
SAFE='safe_status=0; for p in 27 18 22 17; do printf out > /dev/gpio/$p 2>/dev/null || safe_status=1; printf off > /dev/gpio/$p 2>/dev/null || safe_status=1; done; [ $safe_status -eq 0 ] || exit 1'
CPID=''
utc(){ /usr/bin/python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"))'; }
cleanup(){ "${SSH[@]}" "$SAFE" >/dev/null 2>&1 || true; if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then kill -INT "$CPID" 2>/dev/null || true; sleep 2; kill -KILL "$CPID" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
mkdir -p "$BASE/raw" "$BASE/manifests/pi"
printf 'run=%s\nvoltage_v=6\ndistance_m=0.35\nduties_percent=10,1\nanalysis_scope=exploratory_1_percent_timing_unreliable\nplanned_utc=%s\nraw_path=%s\n' "$RUN" "$(utc)" "$RAW" > "$META"
"${SSH[@]}" "$SAFE" || exit 1
"$REPO/receiver/.venv/bin/python" "$REPO/receiver/rocko_receiver.py" --port /dev/cu.usbmodem1201 --baud 115200 --output "$RAW" --plot-seconds 90 > "$CAPLOG" 2>&1 &
CPID=$!; printf 'capture_started_utc=%s\ncapture_pid=%s\n' "$(utc)" "$CPID" >> "$META"; sleep 3; kill -0 "$CPID" 2>/dev/null || exit 1
printf 'transmitter_requested_utc=%s\n' "$(utc)" >> "$META"
"${SSH[@]}" "cd '$REMOTE' || exit 1; $SAFE; echo 'INITIAL OFF 15s' > '$RUN.transmitter.log'; sleep 15; ./duty_pair_test.py --single-duty 10 --letter A --manifest '$RUN.10.csv' >> '$RUN.transmitter.log' 2>&1 || exit 1; $SAFE; echo 'GAP OFF 15s' >> '$RUN.transmitter.log'; sleep 15; ./duty_pair_test.py --single-duty 1 --letter A --manifest '$RUN.1.csv' >> '$RUN.transmitter.log' 2>&1 || exit 1; $SAFE" || exit 1
sleep 15
kill -INT "$CPID" 2>/dev/null || true; wait "$CPID" 2>/dev/null || true; CPID=''
for f in "$RUN.10.csv" "$RUN.1.csv" "$RUN.transmitter.log"; do "${SCP[@]}" "qnxuser@172.17.235.186:$REMOTE/$f" "$BASE/manifests/pi/$f" || exit 1; done
"${SSH[@]}" "$SAFE" || exit 1
printf 'outcome=COMPLETE\nsafe_shutdown_outcome=VERIFIED_WRITES\nfinished_utc=%s\n' "$(utc)" >> "$META"
for f in "$RAW" "$META" "$CAPLOG" "$BASE/manifests/pi/$RUN.10.csv" "$BASE/manifests/pi/$RUN.1.csv" "$BASE/manifests/pi/$RUN.transmitter.log"; do shasum -a 256 "$f" > "$f.sha256"; done
trap - EXIT INT TERM
echo "DIAGNOSTIC COMPLETE raw=$RAW"
