#!/usr/bin/env python3
"""Capture and persist one approved final-experiment session from macOS.

Starts the sole serial capture owner, deploys the reviewed QNX experiment,
archives the transmitter manifest/log atomically, validates every declared
frame, hashes all artifacts, and independently forces GPIO27/18/22/17 low on
every exit path. Export SSHPASS; the password is never a CLI argument.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import glob
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_DIR = ROOT / "transmitter"
sys.path.insert(0, str(TRANSMITTER_DIR))
import final_experiment as experiment  # noqa: E402
import final_experiment_protocol as protocol  # noqa: E402

DEFAULT_DATA_ROOT = Path("~/Desktop/CU-hakcing-captures/final-experiment").expanduser()
DEFAULT_HOST = "172.17.235.186"
DEFAULT_USER = "qnxuser"
DEFAULT_REMOTE_DIR = "/data/home/qnxuser/final-experiment-runner"
CAPTURE_LEAD_SECONDS = 3.0
SESSION_GRACE_SECONDS = 120.0
REMOTE_CONTACT_GRACE_SECONDS = 60.0
SUBPROCESS_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 10.0
DEPLOY_NAMES = (
    "hardware.py",
    "alphabet_transmitter.py",
    "duty_pair_test.py",
    "final_experiment_protocol.py",
    "final_experiment.py",
)
SAFE_GPIO_COMMAND = (
    "safe_status=0; for p in 27 18 22 17; do "
    "printf out > /dev/gpio/$p 2>/dev/null || safe_status=1; "
    "printf off > /dev/gpio/$p 2>/dev/null || safe_status=1; done; "
    "[ $safe_status -eq 0 ] || exit 1"
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def run_name(phase: str, when: datetime | None = None) -> str:
    if phase not in ("short", "final", "all"):
        raise ValueError("phase must be short, final, or all")
    when = when or datetime.now()
    return f"final_experiment_{phase}_11V_{when.strftime('%Y%m%d_%H%M%S')}"


def validate_note(value: str) -> str:
    note = value.strip()
    if not note or "\n" in note or "\r" in note:
        raise ValueError("hardware note must be non-empty and single-line")
    return note


def validate_remote_dir(value: str) -> str:
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", value) or ".." in Path(value).parts:
        raise ValueError("remote directory must be an absolute conservative QNX path")
    return value


def select_serial_port(requested: str | None, candidates: list[str] | None = None) -> str:
    if requested:
        if not Path(requested).exists():
            raise ValueError(f"serial port does not exist: {requested}")
        return requested
    found = sorted(candidates if candidates is not None else glob.glob("/dev/cu.usbmodem*"))
    if len(found) != 1:
        raise ValueError(
            "expected exactly one /dev/cu.usbmodem* port; pass --port explicitly "
            f"(found {found})"
        )
    return found[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_metadata(path: Path, **values) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}\n")
        output.flush()
        os.fsync(output.fileno())


def checked_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("timeout", SUBPROCESS_TIMEOUT_SECONDS)
    return subprocess.run(command, check=True, text=True, **kwargs)


def unchecked_run(command: list[str], **kwargs) -> subprocess.CompletedProcess | None:
    kwargs.setdefault("timeout", SUBPROCESS_TIMEOUT_SECONDS)
    try:
        return subprocess.run(command, check=False, text=True, **kwargs)
    except (OSError, subprocess.TimeoutExpired):
        return None


def ssh_base(user: str, host: str) -> list[str]:
    return [
        "sshpass", "-e", "ssh",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-o", "ConnectTimeout=8",
        f"{user}@{host}",
    ]


def scp_base() -> list[str]:
    return [
        "sshpass", "-e", "scp", "-O", "-q",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-o", "ConnectTimeout=8",
    ]


def transmitter_command(remote_dir: str, phase: str, manifest: str, log: str,
                        hardware_note: str) -> str:
    args = [
        "./final_experiment.py", "--phase", phase, "--manifest", manifest,
        "--hardware-note", hardware_note, "--voltage", "11", "--distance-m", "3",
        "--execute",
    ]
    invocation = " ".join(shlex.quote(item) for item in args)
    return (
        f"cd {shlex.quote(remote_dir)} || exit 1; {SAFE_GPIO_COMMAND}; "
        f"nohup {invocation} > {shlex.quote(log)} 2>&1 </dev/null & "
        "pid=$!; echo $pid"
    )


def expected_trials(phase: str) -> tuple[experiment.Trial, ...]:
    return experiment.experiment_schedule(phase)


def expected_seconds(phase: str) -> float:
    return experiment.estimated_seconds(expected_trials(phase))


def validate_manifest(path: Path, phase: str) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    expected = expected_trials(phase)
    if len(rows) != len(expected):
        raise ValueError(f"manifest must contain exactly {len(expected)} frames")
    required = {
        "sequence", "phase", "scheme", "repetition", "duty_percent",
        "voltage_v", "distance_m", "payload_bits", "coded_body_bits",
        "frame_bits", "bit_seconds", "started_utc", "finished_utc", "duration_s",
    }
    for sequence, (row, trial) in enumerate(zip(rows, expected), 1):
        if not required.issubset(row) or any(row.get(key, "") == "" for key in required):
            raise ValueError(f"manifest frame {sequence} has missing required values")
        frame = protocol.FRAMES[trial.scheme]
        payload = (protocol.FINAL_PAYLOAD_TEXT if trial.phase == "final"
                   else protocol.SHORT_PAYLOAD_TEXT)
        actual = (
            int(row["sequence"]), row["phase"], row["scheme"],
            int(row["repetition"]), float(row["duty_percent"]),
        )
        wanted = (
            sequence, trial.phase, trial.scheme, trial.repetition, trial.duty_percent,
        )
        if actual != wanted:
            raise ValueError(f"manifest frame {sequence} violates frozen schedule")
        if (row["frame_bits"] != frame
                or row["coded_body_bits"] != frame[len(protocol.SYNC_TEXT):]
                or row["payload_bits"] != payload):
            raise ValueError(f"manifest frame {sequence} violates golden frame")
        if (float(row["voltage_v"]), float(row["distance_m"]),
                float(row["bit_seconds"])) != (11.0, 3.0, 0.5):
            raise ValueError(f"manifest frame {sequence} violates physical contract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("short", "final", "all"), default="all")
    parser.add_argument("--hardware-note", required=True)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--poll", type=float, default=POLL_SECONDS)
    parser.add_argument("--execute", action="store_true",
                        help="required to start capture and physical transmission")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        note = validate_note(args.hardware_note)
        port = select_serial_port(args.port)
        remote_dir = validate_remote_dir(args.remote_dir)
        if args.poll <= 0:
            raise ValueError("poll interval must be positive")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    duration = expected_seconds(args.phase)
    name = run_name(args.phase)
    print(f"Run: {name}")
    print(f"Phase: {args.phase}; frames: {len(expected_trials(args.phase))}")
    print(f"Frozen condition: 11 V, 3 m, 2 coded bits/s")
    print(f"Expected session: {duration / 60:.2f} minutes")
    print(f"Serial: {port}")
    if not args.execute:
        print("DRY RUN: add --execute after physical preflight")
        return 0
    if shutil.which("sshpass") is None or "SSHPASS" not in os.environ:
        print("ERROR: install sshpass and export SSHPASS", file=sys.stderr)
        return 2

    root = args.data_root.expanduser().resolve()
    raw_dir, manifests_dir = root / "raw", root / "manifests"
    pi_dir = manifests_dir / "qnx"
    for directory in (raw_dir, manifests_dir, pi_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{name}.csv"
    capture_log = manifests_dir / f"{name}.capture.log"
    metadata_path = manifests_dir / f"{name}.metadata.txt"
    remote_manifest = f"{name}.transmitter.csv"
    remote_log = f"{name}.transmitter.log"
    local_manifest = pi_dir / remote_manifest
    local_log = pi_dir / remote_log
    artifacts = (raw_path, capture_log, metadata_path, local_manifest, local_log)
    collisions = [path for path in artifacts if path.exists()]
    if collisions:
        print(f"ERROR: refusing to overwrite {collisions}", file=sys.stderr)
        return 2

    ssh, scp = ssh_base(args.user, args.host), scp_base()
    capture = None
    capture_output = None
    remote_pid = None
    outcome = "FAILED"
    transfer_outcome = "NOT_RUN"
    manifest_outcome = "NOT_RUN"
    safe_outcome = "NOT_ATTEMPTED"
    interrupted = False
    metadata_path.write_text("", encoding="utf-8")
    append_metadata(
        metadata_path, run=name, planned_utc=utc_stamp(), phase=args.phase,
        expected_frames=len(expected_trials(args.phase)), expected_session_s=duration,
        voltage_v=11, distance_m=3, coded_bits_per_second=2,
        hardware_note=note, serial_port=port, baud=args.baud, raw_path=raw_path,
        host=args.host, remote_dir=remote_dir,
    )

    try:
        checked_run(
            ssh + [f"mkdir -p {shlex.quote(remote_dir)}; {SAFE_GPIO_COMMAND}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        deploy = [TRANSMITTER_DIR / name for name in DEPLOY_NAMES]
        checked_run(
            scp + [*(str(path) for path in deploy), f"{args.user}@{args.host}:{remote_dir}/"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        checked_run(
            ssh + [f"cd {shlex.quote(remote_dir)} && chmod +x *.py && {SAFE_GPIO_COMMAND}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        checked_run(
            ssh + [
                f"cd {shlex.quote(remote_dir)} && "
                "./final_experiment.py --phase all "
                "--hardware-note deployment-preflight >/dev/null && "
                f"{SAFE_GPIO_COMMAND}"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        capture_output = capture_log.open("x", encoding="utf-8")
        capture = subprocess.Popen(
            [sys.executable, str(ROOT / "receiver" / "capture.py"),
             "--port", port, "--baud", str(args.baud), "--out", str(raw_path)],
            stdout=capture_output, stderr=subprocess.STDOUT, text=True,
        )
        append_metadata(metadata_path, capture_started_utc=utc_stamp(), capture_pid=capture.pid)
        time.sleep(CAPTURE_LEAD_SECONDS)
        if capture.poll() is not None:
            raise RuntimeError(f"capture exited early; inspect {capture_log}")
        started = checked_run(
            ssh + [transmitter_command(remote_dir, args.phase, remote_manifest,
                                       remote_log, note)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        candidate = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else ""
        if not candidate.isdigit():
            raise RuntimeError(f"QNX did not return transmitter PID: {candidate!r}")
        remote_pid = int(candidate)
        began = time.monotonic()
        last_contact = began
        append_metadata(metadata_path, transmitter_pid_ack_utc=utc_stamp(), remote_pid=remote_pid)
        deadline = began + duration + SESSION_GRACE_SECONDS
        while True:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError("QNX experiment exceeded hard deadline")
            if capture.poll() is not None:
                raise RuntimeError("capture exited before transmitter")
            command = (
                f"cd {shlex.quote(remote_dir)} || exit 1; "
                f"if grep -q 'FINAL EXPERIMENT COMPLETE' {shlex.quote(remote_log)} 2>/dev/null; "
                f"then echo COMPLETE; elif kill -0 {remote_pid} 2>/dev/null; "
                "then echo RUNNING; else echo FAILED; fi"
            )
            try:
                status = checked_run(
                    ssh + [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).stdout.strip()
                last_contact = time.monotonic()
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                if time.monotonic() - last_contact > REMOTE_CONTACT_GRACE_SECONDS:
                    raise RuntimeError("QNX unreachable beyond contact grace") from exc
                print(f"{utc_stamp()} WARNING: transient QNX status failure", flush=True)
                time.sleep(args.poll)
                continue
            if status == "COMPLETE":
                outcome = "COMPLETE"
                break
            if status == "FAILED":
                raise RuntimeError("QNX experiment exited without completion marker")
            print(f"{utc_stamp()} experiment running; capture PID {capture.pid}", flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        interrupted, outcome = True, "INTERRUPTED"
        print("Interrupted; forcing safe state and preserving partial data", file=sys.stderr)
    except (OSError, RuntimeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        shutdown_command = SAFE_GPIO_COMMAND
        if remote_pid is not None and outcome != "COMPLETE":
            shutdown_command = (
                f"kill -TERM {remote_pid} 2>/dev/null || true; sleep 2; "
                f"kill -0 {remote_pid} 2>/dev/null && kill -KILL {remote_pid} "
                f"2>/dev/null || true; {SAFE_GPIO_COMMAND}"
            )
        shutdown = unchecked_run(
            ssh + [shutdown_command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        safe_outcome = ("VERIFIED_WRITES" if shutdown is not None
                        and shutdown.returncode == 0 else "FAILED_OR_UNREACHABLE")
        if capture is not None and capture.poll() is None:
            try:
                capture.send_signal(signal.SIGINT)
                capture.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                capture.kill()
                capture.wait(timeout=5)
        if capture_output is not None:
            capture_output.close()

        if remote_pid is not None:
            transfer_outcome = "COMPLETE"
            for remote_name, local_path in (
                (remote_manifest, local_manifest), (remote_log, local_log)
            ):
                partial = local_path.with_suffix(local_path.suffix + ".partial")
                try:
                    partial.unlink(missing_ok=True)
                    checked_run(
                        scp + [f"{args.user}@{args.host}:{remote_dir}/{remote_name}",
                               str(partial)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    partial.replace(local_path)
                except (OSError, subprocess.CalledProcessError,
                        subprocess.TimeoutExpired) as exc:
                    transfer_outcome = "FAILED"
                    partial.unlink(missing_ok=True)
                    print(f"ERROR archiving {remote_name}: {exc}", file=sys.stderr)
        if outcome == "COMPLETE" and transfer_outcome == "COMPLETE":
            try:
                validate_manifest(local_manifest, args.phase)
                manifest_outcome = "VALID"
            except (OSError, ValueError) as exc:
                manifest_outcome = "INVALID"
                print(f"ERROR validating transmitter manifest: {exc}", file=sys.stderr)

        hashes = {}
        for artifact in (raw_path, capture_log, local_manifest, local_log):
            if artifact.exists():
                digest = sha256_file(artifact)
                artifact.with_suffix(artifact.suffix + ".sha256").write_text(
                    f"{digest}  {artifact.name}\n", encoding="ascii"
                )
                hashes[f"sha256_{artifact.name}"] = digest
        append_metadata(
            metadata_path, outcome=outcome, transfer_outcome=transfer_outcome,
            manifest_outcome=manifest_outcome, safe_shutdown_outcome=safe_outcome,
            finished_utc=utc_stamp(), **hashes,
        )
        digest = sha256_file(metadata_path)
        metadata_path.with_suffix(metadata_path.suffix + ".sha256").write_text(
            f"{digest}  {metadata_path.name}\n", encoding="ascii"
        )

    success = (outcome == "COMPLETE" and transfer_outcome == "COMPLETE"
               and manifest_outcome == "VALID" and safe_outcome == "VERIFIED_WRITES"
               and raw_path.exists())
    if success:
        print(f"Experiment capture complete: {raw_path}")
        print(f"Validated QNX manifest: {local_manifest}")
        return 0
    if safe_outcome != "VERIFIED_WRITES":
        print("GPIO-low writes were not independently verified; inspect hardware now",
              file=sys.stderr)
    return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
