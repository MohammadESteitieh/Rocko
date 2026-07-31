#!/bin/bash
set -u
umask 077

AUTHENTICATED=0
for ATTEMPT in 1 2 3; do
    PASSWORD=$(/usr/bin/osascript <<APPLESCRIPT
tell application "System Events"
    activate
    set response to display dialog "Enter the QNX password for the Rocko transmitter (attempt ${ATTEMPT} of 3)." default answer "" with hidden answer buttons {"Cancel", "Authenticate"} default button "Authenticate" cancel button "Cancel" with title "Rocko transmitter authentication"
    return text returned of response
end tell
APPLESCRIPT
    ) || { echo "Password prompt cancelled" >&2; exit 2; }
    if [ -z "$PASSWORD" ]; then
        /usr/bin/osascript -e 'display alert "Password must not be empty." as warning' >/dev/null 2>&1 || true
        continue
    fi
    export SSHPASS="$PASSWORD"
    unset PASSWORD
    if sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8 qnxuser@172.17.235.186 'echo AUTH_OK' 2>/dev/null | grep -q AUTH_OK; then
        AUTHENTICATED=1
        break
    fi
    unset SSHPASS
    /usr/bin/osascript -e 'display alert "Authentication failed. Please try again." as warning' >/dev/null 2>&1 || true
done
if [ "$AUTHENTICATED" -ne 1 ]; then
    echo "QNX authentication failed three times" >&2
    exit 1
fi
unset AUTHENTICATED ATTEMPT

BASE=/Users/msteitieh/Desktop/CU-hakcing-captures/current-hamming
REPO=/Users/msteitieh/Desktop/CU-hakcing-2026
RUN=pilot_11V_3m_100p_20260727_190519
RAW="$BASE/raw/$RUN.csv"
META="$BASE/manifests/$RUN.metadata.txt"
CAPLOG="$BASE/manifests/$RUN.capture.log"
REMOTE=/data/home/qnxuser/calibration-runner
PORT=/dev/cu.usbmodem1201
SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8 qnxuser@172.17.235.186)
SCP=(sshpass -e scp -O -q -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=8)
SAFE='safe_status=0; for p in 27 18 22 17; do printf out > /dev/gpio/$p 2>/dev/null || safe_status=1; printf off > /dev/gpio/$p 2>/dev/null || safe_status=1; done; [ $safe_status -eq 0 ] || exit 1'
CPID=''
utc(){ /usr/bin/python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"))'; }
cleanup(){
    "${SSH[@]}" "$SAFE" >/dev/null 2>&1 || true
    if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then
        kill -INT "$CPID" 2>/dev/null || true
        sleep 2
        kill -KILL "$CPID" 2>/dev/null || true
        wait "$CPID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$BASE/raw" "$BASE/manifests/pi" "$BASE/derived"
for path in "$RAW" "$META" "$CAPLOG"; do
    [ ! -e "$path" ] || { echo "Refusing to overwrite $path" >&2; exit 2; }
done
{
    printf 'run=%s\nvoltage_v=11\ndistance_m=3\nduties_percent=100\n' "$RUN"
    printf 'frame_protocol=legacy_hamming_~A\ncoded_bit_rate_bps=0.5\n'
    printf 'analysis_scope=11V_3m_100_percent_detection_pilot_not_matrix\n'
    printf 'planned_utc=%s\nraw_path=%s\n' "$(utc)" "$RAW"
} > "$META"

"${SSH[@]}" "$SAFE"
"${SCP[@]}" "$REPO/transmitter/transmitter.py" "$REPO/transmitter/alphabet_transmitter.py" "$REPO/transmitter/duty_pair_test.py" "qnxuser@172.17.235.186:$REMOTE/"
"${SSH[@]}" "cd '$REMOTE' && chmod +x transmitter.py alphabet_transmitter.py duty_pair_test.py && $SAFE"

"$REPO/receiver/.venv/bin/python" "$REPO/receiver/rocko_receiver.py" --port "$PORT" --baud 115200 --output "$RAW" --plot-seconds 90 > "$CAPLOG" 2>&1 &
CPID=$!
printf 'capture_owner=rocko-live-decoder\ncapture_started_utc=%s\ncapture_pid=%s\n' "$(utc)" "$CPID" >> "$META"
sleep 3
kill -0 "$CPID" 2>/dev/null || { echo "Live decoder exited early; inspect $CAPLOG" >&2; exit 1; }

/usr/bin/osascript <<'APPLESCRIPT' >/dev/null
tell application "System Events"
    activate
    display dialog "Confirm the supply is measured at 11 V, the receiver is 3 m away, and the lab is clear. Click Start, then leave the transmitter area for about 90 seconds. A notification will say when it is safe to return." buttons {"Cancel", "Start pilot"} default button "Start pilot" cancel button "Cancel" with title "Rocko 11 V pilot"
end tell
APPLESCRIPT
printf 'operator_confirmed_11V_3m_lab_clear_utc=%s\ntransmitter_requested_utc=%s\n' "$(utc)" "$(utc)" >> "$META"
/usr/bin/osascript -e 'display notification "Leave now. Signal begins after 15 seconds; safe-return notice in about 90 seconds." with title "Rocko 11 V pilot"' >/dev/null 2>&1 || true

"${SSH[@]}" "cd '$REMOTE' || exit 1; $SAFE; echo 'INITIAL OFF 15s' > '$RUN.transmitter.log'; sleep 15; echo '100 PERCENT START' >> '$RUN.transmitter.log'; ./duty_pair_test.py --single-duty 100 --letter A --manifest '$RUN.100.csv' >> '$RUN.transmitter.log' 2>&1 || exit 1; $SAFE" || exit 1
sleep 15
"${SSH[@]}" "$SAFE" || exit 1
printf 'safe_return_utc=%s\n' "$(utc)" >> "$META"
/usr/bin/osascript -e 'display notification "Transmission is complete and all four GPIOs were forced low. It is safe to return." with title "Rocko 11 V pilot complete" sound name "Glass"' >/dev/null 2>&1 || true

kill -INT "$CPID" 2>/dev/null || true
wait "$CPID" 2>/dev/null || true
CPID=''
"${SCP[@]}" "qnxuser@172.17.235.186:$REMOTE/$RUN.100.csv" "$BASE/manifests/pi/$RUN.100.csv"
"${SCP[@]}" "qnxuser@172.17.235.186:$REMOTE/$RUN.transmitter.log" "$BASE/manifests/pi/$RUN.transmitter.log"
"${SSH[@]}" "$SAFE"
printf 'outcome=COMPLETE\nsafe_shutdown_outcome=VERIFIED_WRITES\nfinished_utc=%s\n' "$(utc)" >> "$META"
for file in "$RAW" "$META" "$CAPLOG" "$BASE/manifests/pi/$RUN.100.csv" "$BASE/manifests/pi/$RUN.transmitter.log"; do shasum -a 256 "$file" > "$file.sha256"; done
trap - EXIT INT TERM
unset SSHPASS
echo "11V PILOT COMPLETE raw=$RAW"
