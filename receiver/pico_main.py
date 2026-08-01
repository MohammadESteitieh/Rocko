# Rocko receiver — Pico ADC streamer (MicroPython). Copy onto the Pico as main.py.
#
# Emits the exact "t,x,y" line format the surface receiver expects:
#
#     t,x,y
#     0.000000,812,1571
#
#   t : receiver time in seconds (sample_index / RATE, evenly spaced)
#   x : sensor 1 ADC (GP26 / ADC0), 0..65535
#   y : sensor 2 ADC (GP27 / ADC1), 0..65535
#
# Wiring (MDT magnetic sensor front-end -> Pico, 0-3.3 V ONLY on the ADC pins):
#   sensor 1 out -> GP26 (ADC0)        sensor 2 out -> GP27 (ADC1)
#   sensor ground -> GND               Pico -> laptop over USB
#
# The legacy live receiver reads this over USB serial at
# 115200 baud, ~200 Hz. A one-channel front-end can tie GP27 to GND; the
# receiver still decodes from the combined amplitude.
from machine import ADC
import time

adc_x = ADC(26)                     # GP26 = ADC0  (sensor 1 / X)
adc_y = ADC(27)                     # GP27 = ADC1  (sensor 2 / Y)
RATE = 200                          # samples/sec — must match the receiver
PERIOD_US = 1_000_000 // RATE

print("t,x,y")                      # header line (receiver skips non-numeric rows)

sample = 0
next_t = time.ticks_us()
while True:
    x = adc_x.read_u16()            # 0..65535
    y = adc_y.read_u16()
    # sample_index / RATE gives an evenly spaced receiver clock, so the decoder
    # derives fs = 200 Hz exactly from the timestamp column.
    print("%.6f,%d,%d" % (sample / RATE, x, y))
    sample += 1
    next_t = time.ticks_add(next_t, PERIOD_US)
    dt = time.ticks_diff(next_t, time.ticks_us())
    if dt > 0:
        time.sleep_us(dt)
