# GOOFER (Work in progress)
___
# Flag List (SillySampler)

| Flag | Range    | Default | Description |
|------|----------|---------|-------------|
| `g`  | `-100`-`100`  | `0` | Global formant shift  |
| `fa` | `-100`-`100`  | `0` | Scale Formant 1 (F1)  |
| `fb` | `-100`-`100`  | `0` | Scale Formant 2 (F2)  |
| `fc` | `-100`-`100`  | `0` | Scale Formant 3 (F3)  |
| `fd` | `-100`-`100`  | `0` | Scale Formant 4 (F4)  |
| `V`  | `0`-`100`     | `100` | Harmonic (voiced) level. `0` = no harmonic, `100` = full voice |
| `B`  | `-100`-`100`  | `0`   | Breathiness level. `-100` = none, `0` = raw, `100` = 2x louder |
| `U`  | `-100`-`100`  | `0`   | Unvoiced (fricative) level. `-100` = none, `0` = raw, `100` = 2x louder |
| `sh` | `0`-`100`     | `0`   | F0 jitter (pitch instability). Harsh vocal effect |
| `sr` | `0`-`100`     | `0`   | Volume jitter (voiced only). Rough vocal effect |
| `st` | `0`-`100`     | `0`   | Tension effect (attempt to) |
| `sg` | `0`-`100`     | `0`   | growl effect (attempt to) |
| `L`  | `0`, `1`, `2` | `0`   | Sustain behavior: <br> `L0` = concat loop <br> `L1` = averaged mirror loop <br> `L2` = stretch |
| `t`  | `-100`-`100`  | `0`   | Pitch offset flag |
