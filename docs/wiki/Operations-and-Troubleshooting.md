# Operations and Troubleshooting

## Before a capture

1. Confirm the intended voltage, distance, duty schedule, bit rate, payload,
   sensor geometry, and operator-presence condition.
2. Verify GPIO22→IN3, GPIO17→IN4, GPIO27→ENB, and shared ground.
3. Confirm exactly one `/dev/cu.usbmodem*` device or pass `--port` explicitly.
4. Ensure no process owns the serial port.
5. Run the transmitter and macOS runner without `--execute` first.
6. Confirm the output directory is persistent and collision-free.
7. Allow the declared person-absent settling period.

## During a physical run

Use the macOS `receiver/run_*.py` orchestrator. It should be the only serial
owner and should run under `caffeinate` for long sessions. Do not close a
foreground harness that owns capture or transmission.

Stop and return control to the operator on DNS, routing, SSH, relay, or remote
host failure. Do not automatically retry a physical transmission.

## After a run

Verify all of the following:

- raw capture exists and is nonempty;
- transmitter manifest and log were transferred;
- manifest schedule validates exactly;
- metadata says `outcome=COMPLETE` where applicable;
- checksums pass;
- GPIO27, GPIO18, GPIO22, and GPIO17 were independently forced low;
- metadata says `safe_shutdown_outcome=VERIFIED_WRITES`.

## Common failures

### Serial device busy

Use `lsof /dev/cu.usbmodem...` and stop the competing capture/viewer. Never run
two serial readers.

### Capture exists but no useful signal

Check desired carrier power and synchronization before changing the decoder.
Low background noise is not proof of sensor coupling. Verify sensor sensitive
axis, signal wiring, cable motion, transmitter current, coil orientation, and
physical distance.

### One sensor is much noisier

Swap complete sensor connections while holding geometry fixed. If the problem
moves, inspect the sensor and sensor-end wiring. If it stays on one ADC channel,
inspect its cable, ground, frontend, and Pico input. A miswired diagnostic must
be marked invalid rather than interpreted.

### Transmitter import failure

Deploy the complete transitive file list. Preserve a failed pre-transmission run
and perform independent GPIO-low cleanup.

### No positive SNR estimate

This is a valid result. Do not add a numerical floor. Record that active carrier
power did not exceed transmitter-off power.

## Credentials

Never store or relay the QNX password. Use a hidden local prompt and ephemeral
process environment only.
