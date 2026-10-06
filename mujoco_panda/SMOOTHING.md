# Sliding-contact retuning

The original Panda setting used one spherical contact point, a 20 ms contact
time constant with high boundary impedance, a 50 ms normal-force filter, and
an instantaneous 0.03 m/s slide reference. The revised setting uses:

- a finite pad with four soft contact elements (two carry load during the
  nominal slide),
- contact `solref = 0.10 1` and `solimp = 0.05 0.95 0.005`, which makes the
  activation boundary soft while retaining higher impedance deeper inside,
- a 70 ms causal normal-force filter,
- normal-force gains $k_f=0.30$, $k_i=3.0$, and damping $d_f=24$,
- a 600 ms cosine ramp into the sliding trajectory,
- the unchanged 1 ms physics/control step, friction, and torque/rate limits.

A 5.2 s C0 straight-slide comparison produced:

| metric | original | revised | change |
|---|---:|---:|---:|
| force RMSE | 0.267 N | 0.263 N | -2% |
| planar RMSE | 3.66 mm | 3.40 mm | -7% |
| slide-onset acceleration p99 | 1.42 m/s2 | 0.108 m/s2 | -92% |
| established-slide acceleration p99 | 1.40 m/s2 | 0.0055 m/s2 | >99% lower |
| established-slide jerk p99 | 2415 m/s3 | 0.108 m/s3 | >99% lower |
| torque-rate norm p99 | 58.5 N m/s | 3.90 N m/s | -93% |
| normal velocity standard deviation | 1.49 mm/s | 0.175 mm/s | -88% |
| raw contact-constraint occupancy | 48.3% | 100% | +51.7 points |
| mean active contact points | 0.48 | 2.00 | finite patch |
| raw force standard deviation | 5.31 N | 0.101 N | -98% |
| raw force peak | 12.64 N | 5.54 N | -56% |
| sample-to-sample raw force-jump RMS | 5.26 N | 0.00034 N | >99% lower |

The unfiltered 1 kHz signal is shown in
`results_panda/smoothing/raw_contact.png`. The revised raw force is already
smooth and continuous before filtering; the 70 ms filter only attenuates the
remaining low-amplitude variation. The accepted setting places the 5 N
equilibrium about 0.14--0.33 mm inside the 1 mm soft-contact activation band,
so small normal motion no longer switches the entire contact constraint off.

An independently sprung four-element trial was rejected: it removed the
solver-scale pulse but introduced an approximately 8 Hz pad resonance, raised
force RMSE to 0.92 N, and increased normal-velocity variation to 14.8 mm/s.
This failed candidate was not used in the final evidence.

Acceleration and jerk use the logged tool twist and finite differences at
1 kHz. The original/revised layout and parameter snapshots are saved with the
logs. After the new model was frozen, the complete A--E experiment suite was
regenerated. All focused C3 cases had zero QP fallback, protected-floor, and
prediction-bound violations.
