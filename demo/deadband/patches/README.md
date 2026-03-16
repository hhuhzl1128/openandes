# Patch Notes

This folder stores the local compatibility patches needed to reproduce the
`deadband` workflow against upstream ANDES/AMS sources without maintaining
personal fork repositories.

- `andes_deadband.patch`
  Adds `fdbdu` support to `PVD1` and documents matching `ESD1` upper deadband
  behavior used by this study.
- `ams_runtime_compat.patch`
  Carries the runtime compatibility fixes used for the dispatch-to-TDS path,
  including ANDES 2.0-style setters and replacement of legacy `cache.df_in`
  access.

Apply from an upstream checkout if needed:

```bash
git -C /path/to/andes apply /path/to/openandes/demo/deadband/patches/andes_deadband.patch
git -C /path/to/ams apply /path/to/openandes/demo/deadband/patches/ams_runtime_compat.patch
```
