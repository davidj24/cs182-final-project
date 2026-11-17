# T01 repair 2 candidate measurements (translation-only scene)

Scripts: docs/evidence/T01_candidates.py, docs/evidence/T01_armature_sweep.py (outputs below are real).

## Corner cases: N/mg min and max over push steps (min uses only steps with contact), exc_mm = centre-height excursion
```
== i_xy_only
(0.5, 0.6, 26, 200) {'minnc': 4.0, 'N': 0.0, 'Nmax': 0.0, 'ratio': nan, 'Rdev': nan, 'exc_mm': 0.0, 'zrest': 0.025}
(0.5, 0.3, 26, 200) {'minnc': 4.0, 'N': 0.0, 'Nmax': 0.0, 'ratio': nan, 'Rdev': nan, 'exc_mm': 0.0, 'zrest': 0.025}
(1.0, 0.45, 20, 150) {'minnc': 4.0, 'N': 0.0, 'Nmax': 0.0, 'ratio': nan, 'Rdev': nan, 'exc_mm': 0.0, 'zrest': 0.025}
(1.5, 0.6, 26, 200) {'minnc': 4.0, 'N': 0.0, 'Nmax': 0.0, 'ratio': nan, 'Rdev': nan, 'exc_mm': 0.0, 'zrest': 0.025}
(0.5, 0.6, 13, 100) {'minnc': 4.0, 'N': 0.0, 'Nmax': 0.0, 'ratio': nan, 'Rdev': nan, 'exc_mm': 0.0, 'zrest': 0.025}
== ii_xyz_free_z
(0.5, 0.6, 26, 200) {'minnc': 0.0, 'N': 3.0698, 'Nmax': 44.0592, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 2.21098, 'zrest': 1e-05}
(0.5, 0.3, 26, 200) {'minnc': 0.0, 'N': 2.37311, 'Nmax': 29.08574, 'ratio': 0.3, 'Rdev': 0.0, 'exc_mm': 1.05218, 'zrest': 0.0}
(1.0, 0.45, 20, 150) {'minnc': 0.0, 'N': 1.59213, 'Nmax': 11.08536, 'ratio': 0.45, 'Rdev': 0.0, 'exc_mm': 0.5251, 'zrest': -0.0}
(1.5, 0.6, 26, 200) {'minnc': 0.0, 'N': 1.51293, 'Nmax': 12.04563, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 1.39312, 'zrest': -0.0}
(0.5, 0.6, 13, 100) {'minnc': 0.0, 'N': 1.9013, 'Nmax': 10.89948, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 0.57959, 'zrest': -0.0}
== ii_xyz_zrange
(0.5, 0.6, 26, 200) {'minnc': 0.0, 'N': 3.0698, 'Nmax': 43.07035, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 2.05689, 'zrest': 1e-05}
(0.5, 0.3, 26, 200) {'minnc': 0.0, 'N': 2.37311, 'Nmax': 29.08574, 'ratio': 0.3, 'Rdev': 0.0, 'exc_mm': 1.05218, 'zrest': 0.0}
(1.0, 0.45, 20, 150) {'minnc': 0.0, 'N': 1.59213, 'Nmax': 11.08536, 'ratio': 0.45, 'Rdev': 0.0, 'exc_mm': 0.5251, 'zrest': -0.0}
(1.5, 0.6, 26, 200) {'minnc': 0.0, 'N': 1.51293, 'Nmax': 12.05055, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 1.30836, 'zrest': -0.0}
(0.5, 0.6, 13, 100) {'minnc': 0.0, 'N': 1.9013, 'Nmax': 10.89948, 'ratio': 0.6, 'Rdev': 0.0, 'exc_mm': 0.57959, 'zrest': -0.0}
```

## Armature 1000 kg on z slide, random pushes
```
train-range n=300 dt=0.001: |d/an-1| median 3.71e-03 p95 2.06e-02 max 6.04e-02; frac of push steps in contact: min 0.09 median 0.17; N/mg: worst-min 0.00 worst-max 105.9 median-of-means 0.93
OOD heavy n=50 dt=0.001: |d/an-1| median 9.93e-03 p95 3.80e-02 max 5.16e-02; frac of push steps in contact: min 0.12 median 0.21; N/mg: worst-min 0.00 worst-max 58.2 median-of-means 1.05
dt=0.001 (same 50) n=50 dt=0.001: |d/an-1| median 3.04e-03 p95 8.01e-03 max 2.58e-02; frac of push steps in contact: min 0.09 median 0.15; N/mg: worst-min 0.00 worst-max 83.4 median-of-means 1.02
dt=0.0005 (same 50) n=50 dt=0.0005: |d/an-1| median 2.93e-03 p95 8.24e-03 max 2.30e-02; frac of push steps in contact: min 0.09 median 0.15; N/mg: worst-min 0.00 worst-max 85.4 median-of-means 1.00
```

## Why N != mg: first-step normal force, 1 kg, mu 0.45, F 20 N (z slide free)
Measured first-step N/mg = 1.593 for impratio 1/10/100/1000, elliptic and pyramidal (1.594-1.595), solref timeconst 0.002-0.5,
solimp d 0.9-0.9999 (1.58-1.595). The prediction (mg + mu*F)/(1+mu^2)/mg = 1.5945 (projection of the unconstrained force
onto the friction cone) matches. At F=0 N/mg = 1.000; at F=5 N/mg = 1.022. For m=0.5, mu=0.6, F=26 first-step N/mg = 3.07.
So MuJoCo's convex contact relaxation lifts N above mg whenever the push force exceeds mu*N; no solver parameter tested removes it.
Candidate (i) (x,y slides only): contact present (ncon=4) but N = 0 exactly on all 5 corners (zero z Jacobian; gravity never reaches the contact).
Candidate "pad": floor on a z-spring body with a dof-less cube: no force (gravity on a dof-less direction is absorbed).
