#!/usr/bin/env python3
"""Orchestrate one persistent calibration capture from macOS.

This is the operator-facing companion to ``transmitter/calibration_sweep.py``.
It creates the persistent run layout, starts the authoritative serial capture,
deploys and launches the QNX transmitter in an isolated remote directory,
archives the transmitter manifest/log, records metadata, computes SHA-256
checksums, and makes a best-effort four-pin safe shutdown on every exit path.

The QNX password is never accepted on the command line. Export ``SSHPASS`` in
the invoking shell; ``sshpass -e`` reads it without storing it in artifacts.
"""

from __future__ import annotations

import argparse
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

SESSION_SECONDS = 872.0
FRAME_SECONDS = 56.0
INITIAL_WAIT_SECONDS = 15.0
GAP_SECONDS = 15.0
FINAL_WAIT_SECONDS = 5.0
SESSION_GRACE_SECONDS = 120.0
REMOTE_CONTACT_GRACE_SECONDS = 60.0
SUBPROCESS_TIMEOUT_SECONDS = 30.0
CAPTURE_LEAD_SECONDS = 3.0
POLL_SECONDS = 10.0
DEFAULT_DATA_ROOT = Path("~/Desktop/CU-hakcing-captures/current-hamming").expanduser()
DEFAULT_HOST = "172.17.235.186"
DEFAULT_USER = "qnxuser"
DEFAULT_REMOTE_DIR = "/data/home/qnxuser/calibration-runner"
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


def run_name(
    voltage: float,
    when: datetime | None = None,
    *,
    prefix: str = "calibration",
) -> str:
    when = when or datetime.now()
    voltage_text = f"{float(voltage):g}".replace(".", "p")
    return f"{prefix}_{voltage_text}V_{when.strftime('%Y%m%d_%H%M%S')}"


def parse_duty_sequence(value: str) -> tuple[float, ...]:
    """Parse an explicit one-frame-per-duty schedule for the existing runner."""
    try:
        duties = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("duty sequence must be comma-separated numbers") from exc
    if not duties or any(not 0 < duty <= 100 for duty in duties):
        raise ValueError("every duty in the sequence must be in (0, 100]")
    if len(set(duties)) != len(duties):
        raise ValueError("duty sequence must not contain duplicates")
    return duties


def expected_session_seconds(
    duties: tuple[float, ...] | None = None,
    *,
    bit_seconds: float = 2.0,
) -> float:
    if bit_seconds <= 0:
        raise ValueError("bit duration must be positive")
    frames = 12 if duties is None else len(duties)
    frame_seconds = 28 * bit_seconds
    return (
        INITIAL_WAIT_SECONDS
        + frames * (frame_seconds + GAP_SECONDS)
        + FINAL_WAIT_SECONDS
    )


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
        path = Path(requested)
        if not path.exists():
            raise ValueError(f"serial port does not exist: {requested}")
        return str(path)
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


def transmitter_command(
    *,
    remote_dir: str,
    manifest_name: str,
    log_name: str,
    voltage: float,
    distance_m: float,
    hardware_note: str,
    allow_six_volts: bool,
    duty_sequence: tuple[float, ...] | None = None,
    bit_seconds: float = 2.0,
) -> str:
    args = [
        "./calibration_sweep.py",
        "--voltage", f"{voltage:g}",
        "--distance-m", f"{distance_m:g}",
        "--hardware-note", hardware_note,
        "--manifest", manifest_name,
    ]
    if allow_six_volts:
        args.append("--allow-six-volts")
    if duty_sequence is not None:
        args.extend(("--duty-sequence", ",".join(f"{duty:g}" for duty in duty_sequence)))
    if bit_seconds != 2.0:
        args.extend(("--bit-seconds", f"{bit_seconds:g}"))
    invocation = " ".join(shlex.quote(item) for item in args)
    return (
        f"cd {shlex.quote(remote_dir)} || exit 1; "
        f"{SAFE_GPIO_COMMAND}; "
        f"nohup {invocation} > {shlex.quote(log_name)} 2>&1 </dev/null & "
        "pid=$!; echo $pid"
    )


def append_metadata(path: Path, **values) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            text = str(value).replace("\n", " ").replace("\r", " ")
            output.write(f"{key}={text}\n")
        output.flush()
        os.fsync(output.fileno())


def capture_command(
    repo: Path,
    python: str,
    port: str,
    baud: int,
    raw_path: Path,
    *,
    live_dashboard: bool,
    bit_seconds: float = 2.0,
) -> list[str]:
    if live_dashboard:
        return [
            python, str(repo / "receiver" / "legacy_decoder" / "rocko_receiver.py"),
            "--port", port, "--baud", str(baud), "--output", str(raw_path),
            "--plot-seconds", "90", "--bit-seconds", f"{bit_seconds:g}",
        ]
    return [
        python, str(repo / "receiver" / "capture.py"),
        "--port", port, "--baud", str(baud), "--out", str(raw_path),
    ]


def checked_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("timeout", SUBPROCESS_TIMEOUT_SECONDS)
    return subprocess.run(command, check=True, text=True, **kwargs)


def unchecked_run(command: list[str], **kwargs) -> subprocess.CompletedProcess | None:
    kwargs.setdefault("timeout", SUBPROCESS_TIMEOUT_SECONDS)
    try:
        return subprocess.run(command, check=False, text=True, **kwargs)
    except (OSError, subprocess.TimeoutExpired):
        return None


def remote_contact_expired(last_contact: float, now: float) -> bool:
    """Allow transient SSH loss while retaining a bounded safety deadline."""
    return now - last_contact > REMOTE_CONTACT_GRACE_SECONDS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voltage", type=float, required=True)
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--hardware-note", required=True)
    parser.add_argument("--allow-six-volts", action="store_true")
    parser.add_argument("--port", help="receiver serial port; auto-detected if unique")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--poll", type=float, default=POLL_SECONDS)
    parser.add_argument(
        "--duty-sequence",
        help="exploratory comma-separated duties, one ~A frame each; default stays frozen",
    )
    parser.add_argument(
        "--bit-seconds", type=float, choices=(0.5, 1.0, 2.0), default=2.0,
        help="coded-bit duration; 2.0 is frozen, 1.0/0.5 are pilot-only",
    )
    parser.add_argument(
        "--live-dashboard", action="store_true",
        help="use the visible Rocko live decoder as the sole serial/capture owner",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    transmitter_dir = repo / "transmitter"
    try:
        note = validate_note(args.hardware_note)
        port = select_serial_port(args.port)
        args.remote_dir = validate_remote_dir(args.remote_dir)
        duty_sequence = (
            parse_duty_sequence(args.duty_sequence)
            if args.duty_sequence is not None
            else None
        )
        if args.distance_m <= 0:
            raise ValueError("distance must be positive")
        if args.voltage == 6 and not args.allow_six_volts:
            raise ValueError("6 V requires --allow-six-volts")
        if args.voltage not in (6, 7, 8, 9, 10, 11):
            raise ValueError("voltage must be 7-11 V, or conditional 6 V")
        if args.poll <= 0:
            raise ValueError("poll interval must be positive")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    session_seconds = expected_session_seconds(
        duty_sequence,
        bit_seconds=args.bit_seconds,
    )
    name = run_name(
        args.voltage,
        prefix="duty_sweep" if duty_sequence is not None else "calibration",
    )
    root = args.data_root.expanduser().resolve()
    raw_dir = root / "raw"
    manifests_dir = root / "manifests"
    pi_dir = manifests_dir / "pi"
    derived_dir = root / "derived"
    for directory in (raw_dir, manifests_dir, pi_dir, derived_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{name}.csv"
    capture_log = manifests_dir / f"{name}.capture.log"
    metadata_path = manifests_dir / f"{name}.metadata.txt"
    remote_manifest = f"{name}.transmitter.csv"
    remote_log = f"{name}.transmitter.log"
    local_manifest = pi_dir / remote_manifest
    local_tx_log = pi_dir / remote_log
    analysis_csv = derived_dir / f"{name}.calibration.csv"
    analysis_json = derived_dir / f"{name}.calibration-summary.json"
    reserved_paths = (
        raw_path, capture_log, metadata_path, local_manifest, local_tx_log,
        analysis_csv, analysis_json,
    )
    collisions = [path for path in reserved_paths if path.exists()]
    if collisions:
        print(f"ERROR: refusing to overwrite existing run artifacts: {collisions}", file=sys.stderr)
        return 2

    print(f"Run: {name}")
    print(f"Raw capture: {raw_path}")
    print(f"Serial: {port}")
    print(f"Voltage: {args.voltage:g} V; distance: {args.distance_m:g} m")
    if duty_sequence is not None:
        print("Duty sequence: " + ", ".join(f"{duty:g}%" for duty in duty_sequence))
    print(f"Coded-bit duration: {args.bit_seconds:g} seconds")
    print(f"Expected transmitter session: {session_seconds / 60:.2f} minutes")
    if args.dry_run:
        print("Dry run: no serial, SSH, or GPIO activity.")
        return 0

    if shutil.which("sshpass") is None:
        print("ERROR: sshpass is required", file=sys.stderr)
        return 2
    if "SSHPASS" not in os.environ:
        print("ERROR: export SSHPASS before starting the calibration", file=sys.stderr)
        return 2

    ssh = ssh_base(args.user, args.host)
    scp = scp_base()
    capture = None
    capture_output = None
    remote_pid = None
    outcome = "FAILED"
    analysis_outcome = "NOT_RUN"
    transfer_outcome = "NOT_RUN"
    safe_shutdown_outcome = "NOT_ATTEMPTED"
    interrupted = False

    metadata_path.write_text("", encoding="utf-8")
    append_metadata(
        metadata_path,
        run=name,
        planned_utc=utc_stamp(),
        voltage_v=f"{args.voltage:g}",
        distance_m=f"{args.distance_m:g}",
        hardware_note=note,
        serial_port=port,
        baud=args.baud,
        bit_seconds=f"{args.bit_seconds:g}",
        host=args.host,
        remote_dir=args.remote_dir,
        expected_session_s=session_seconds,
        duty_sequence=(
            ",".join(f"{duty:g}" for duty in duty_sequence)
            if duty_sequence is not None else "frozen-default"
        ),
        raw_path=raw_path,
        capture_owner="rocko-live-decoder" if args.live_dashboard else "capture.py",
    )

    try:
        checked_run(ssh + [f"mkdir -p {shlex.quote(args.remote_dir)}; {SAFE_GPIO_COMMAND}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deploy = [
            transmitter_dir / "hardware.py",
            transmitter_dir / "alphabet_transmitter.py",
            transmitter_dir / "duty_pair_test.py",
            transmitter_dir / "calibration_sweep.py",
        ]
        checked_run(
            scp + [*(str(path) for path in deploy),
                   f"{args.user}@{args.host}:{args.remote_dir}/"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        checked_run(
            ssh + [f"cd {shlex.quote(args.remote_dir)} && chmod +x *.py && {SAFE_GPIO_COMMAND}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        capture_output = capture_log.open("w", encoding="utf-8")
        capture = subprocess.Popen(
            capture_command(
                repo, sys.executable, port, args.baud, raw_path,
                live_dashboard=args.live_dashboard,
                bit_seconds=args.bit_seconds,
            ),
            cwd=repo,
            stdout=capture_output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        append_metadata(metadata_path, capture_started_utc=utc_stamp(), capture_pid=capture.pid)
        time.sleep(CAPTURE_LEAD_SECONDS)
        if capture.poll() is not None:
            raise RuntimeError(f"capture exited early; inspect {capture_log}")

        remote_command = transmitter_command(
            remote_dir=args.remote_dir,
            manifest_name=remote_manifest,
            log_name=remote_log,
            voltage=args.voltage,
            distance_m=args.distance_m,
            hardware_note=note,
            allow_six_volts=args.allow_six_volts,
            duty_sequence=duty_sequence,
            bit_seconds=args.bit_seconds,
        )
        started = checked_run(
            ssh + [remote_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        candidate = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else ""
        if not candidate.isdigit():
            raise RuntimeError(f"remote transmitter did not return a PID: {candidate!r}")
        remote_pid = int(candidate)
        remote_started_monotonic = time.monotonic()
        append_metadata(
            metadata_path,
            transmitter_pid_ack_utc=utc_stamp(),
            remote_pid=remote_pid,
            remote_manifest=remote_manifest,
            remote_log=remote_log,
        )
        print(f"Calibration running on QNX as PID {remote_pid}", flush=True)

        deadline = remote_started_monotonic + session_seconds + SESSION_GRACE_SECONDS
        last_remote_contact = remote_started_monotonic
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "remote calibration exceeded its hard session deadline"
                )
            if capture.poll() is not None:
                raise RuntimeError(f"capture exited before transmitter; inspect {capture_log}")
            status_command = (
                f"cd {shlex.quote(args.remote_dir)} || exit 1; "
                f"if grep -q 'CALIBRATION COMPLETE' {shlex.quote(remote_log)} 2>/dev/null; "
                "then echo COMPLETE; "
                f"elif kill -0 {remote_pid} 2>/dev/null; then echo RUNNING; "
                "else echo FAILED; fi"
            )
            try:
                status = checked_run(
                    ssh + [status_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).stdout.strip()
                last_remote_contact = time.monotonic()
            except (
                OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired
            ) as exc:
                now = time.monotonic()
                if remote_contact_expired(last_remote_contact, now):
                    raise RuntimeError(
                        "QNX status remained unreachable beyond the contact grace period"
                    ) from exc
                print(
                    f"{utc_stamp()} WARNING transient QNX status failure; retrying",
                    flush=True,
                )
                time.sleep(args.poll)
                continue
            if status == "COMPLETE":
                outcome = "COMPLETE"
                break
            if status == "FAILED":
                raise RuntimeError("remote calibration exited without completion marker")
            print(f"{utc_stamp()} calibration running; capture PID {capture.pid}", flush=True)
            time.sleep(args.poll)
    except KeyboardInterrupt:
        interrupted = True
        outcome = "INTERRUPTED"
        print("Interrupted; forcing transmitter safe and preserving partial artifacts.",
              file=sys.stderr)
    except (
        OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        shutdown_command = SAFE_GPIO_COMMAND
        if remote_pid is not None and outcome != "COMPLETE":
            shutdown_command = (
                f"kill -TERM {remote_pid} 2>/dev/null || true; sleep 2; "
                f"kill -0 {remote_pid} 2>/dev/null && "
                f"kill -KILL {remote_pid} 2>/dev/null || true; {SAFE_GPIO_COMMAND}"
            )
        shutdown = unchecked_run(
            ssh + [shutdown_command], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        safe_shutdown_outcome = (
            "VERIFIED_WRITES"
            if shutdown is not None and shutdown.returncode == 0
            else "FAILED_OR_UNREACHABLE"
        )
        if capture is not None and capture.poll() is None:
            capture.send_signal(signal.SIGINT)
            try:
                capture.wait(timeout=5)
            except subprocess.TimeoutExpired:
                capture.kill()
                capture.wait()
        if capture_output is not None:
            capture_output.close()

        if remote_pid is not None:
            transfer_outcome = "COMPLETE"
            for remote_name, local_path in (
                (remote_manifest, local_manifest), (remote_log, local_tx_log)
            ):
                partial = local_path.with_suffix(local_path.suffix + ".partial")
                try:
                    partial.unlink(missing_ok=True)
                    checked_run(
                        scp + [
                            f"{args.user}@{args.host}:{args.remote_dir}/{remote_name}",
                            str(partial),
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    partial.replace(local_path)
                except (
                    OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired
                ) as exc:
                    transfer_outcome = "FAILED"
                    partial.unlink(missing_ok=True)
                    print(f"ERROR archiving {remote_name}: {exc}", file=sys.stderr)

        if (outcome == "COMPLETE" and transfer_outcome == "COMPLETE"
                and raw_path.exists() and local_manifest.exists()):
            try:
                analysis_command = [
                    sys.executable,
                    str(repo / "receiver" / "legacy_decoder" / "analyze_calibration.py"),
                    str(raw_path), str(local_manifest), str(metadata_path),
                    "--output", str(analysis_csv),
                    "--summary", str(analysis_json),
                ]
                if duty_sequence is not None:
                    analysis_command.extend((
                        "--duty-sequence",
                        ",".join(f"{duty:g}" for duty in duty_sequence),
                    ))
                analysis_command.extend(("--bit-seconds", f"{args.bit_seconds:g}"))
                if args.bit_seconds != 2.0:
                    analysis_command.append("--manifest-boundaries")
                checked_run(
                    analysis_command,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    timeout=600,
                )
                analysis_outcome = "COMPLETE"
            except (
                OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired
            ) as exc:
                analysis_outcome = "FAILED"
                print(f"ERROR: automatic calibration analysis failed: {exc}", file=sys.stderr)

        artifacts = [
            raw_path, capture_log, local_manifest, local_tx_log,
            analysis_csv, analysis_json,
        ]
        hashes = {}
        for artifact in artifacts:
            if artifact.exists():
                digest = sha256_file(artifact)
                artifact.with_suffix(artifact.suffix + ".sha256").write_text(
                    f"{digest}  {artifact.name}\n", encoding="ascii"
                )
                hashes[f"sha256_{artifact.name}"] = digest
        append_metadata(
            metadata_path,
            outcome=outcome,
            transfer_outcome=transfer_outcome,
            analysis_outcome=analysis_outcome,
            safe_shutdown_outcome=safe_shutdown_outcome,
            analysis_csv=analysis_csv if analysis_csv.exists() else "",
            analysis_summary=analysis_json if analysis_json.exists() else "",
            finished_utc=utc_stamp(),
            **hashes,
        )
        metadata_digest = sha256_file(metadata_path)
        metadata_path.with_suffix(metadata_path.suffix + ".sha256").write_text(
            f"{metadata_digest}  {metadata_path.name}\n", encoding="ascii"
        )

    if (outcome == "COMPLETE" and transfer_outcome == "COMPLETE"
            and analysis_outcome == "COMPLETE"
            and safe_shutdown_outcome == "VERIFIED_WRITES"):
        print(f"Calibration complete: {raw_path}")
        print(f"Analysis complete: {analysis_csv}")
        return 0
    if safe_shutdown_outcome != "VERIFIED_WRITES":
        print(
            "Calibration did not finish with independently verified GPIO-low writes; "
            "check the QNX transmitter physically before continuing.",
            file=sys.stderr,
        )
        return 1
    if outcome == "COMPLETE":
        print(f"Calibration capture completed but analysis failed: {raw_path}", file=sys.stderr)
        return 1
    return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
