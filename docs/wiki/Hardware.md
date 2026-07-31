# Hardware

## Signal path

```text
QNX Raspberry Pi → GPIO → L298N H-bridge → transmit coil
                                              ↓ magnetic field
Mac ← USB serial ← Raspberry Pi Pico ← two magnetic sensors
```

## Transmitter wiring

| Raspberry Pi GPIO | L298N function |
|---|---|
| GPIO22 | IN3 |
| GPIO17 | IN4 |
| GPIO27 | ENB |
| GPIO18 | Defensively forced low |
| GND | Shared ground |

The coil is connected to OUT3/OUT4. The motor supply used in the accepted
experiments was commanded at 11 V. Never connect the motor supply to the Pi or
Pico logic rails.

The exact L298N board revision, ENB jumper state, and logic-power arrangement
must be recorded before another physical campaign.

## Receiver wiring

| Pico input | Captured column |
|---|---|
| GP26 / ADC0 | `x` |
| GP27 / ADC1 | `y` |
| GND | Shared sensor ground |

`receiver/pico_main.py` uses MicroPython `read_u16()`. Some accepted captures
were delivered in a 12-bit numerical range by the deployed acquisition path,
so every analyzer records the inferred ADC full scale and clipping percentage.
Do not assume a fixed numerical range without checking the capture metadata.

## Sensor geometry

The accepted RS18 collection used pinned, co-located, parallel, level,
same-polarity sensors with the operator absent. Human proximity, breathing,
cable motion, loose Dupont contacts, sensor stacking pressure, and power-source
changes all produced measurable disturbances during diagnostics.

The final corrected-wiring background run was
`corrected_wiring_final_placement_background_20260728_154237`. It showed no
clipping and a particularly quiet sensor-y carrier band. The subsequent active
quick screen nevertheless had inadequate desired signal, so quiet background
alone is not proof of useful orientation or coupling.

## Safety

After every transmission attempt, independently configure GPIO27, GPIO18,
GPIO22, and GPIO17 as outputs and force all four low. A successful process exit
is not sufficient evidence; accepted metadata must say
`safe_shutdown_outcome=VERIFIED_WRITES`.
