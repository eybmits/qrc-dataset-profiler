# Santa Fe laser real-data bridge

The protocol includes `santa_fe_laser` as a real-data bridge. The committed data files are
public Santa Fe laser benchmark series downloaded from the source below, and the current
full catalog and 1000-row sweep include this row/family.

The loader in `src/qrc_dataset_profiler/generators.py` will use either:

- `data/santa_fe_laser.*` with at least the requested series length, or
- the canonical Santa Fe laser files `data/SantaFeA.dat` and `data/SantaFeA2.dat`.

Downloaded/public lookup result on 2026-06-28:

- `https://web.cecs.pdx.edu/mcnames/DataSets/SantaFeA.dat` contains 1,000 values.
- `https://web.cecs.pdx.edu/mcnames/DataSets/SantaFeA2.dat` contains 9,093 continuation values.

These two files together provide 10,093 values, sufficient for the default 4,000-point
full-catalog row and 40 deterministic sweep windows.
