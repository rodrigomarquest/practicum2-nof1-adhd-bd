# Data Audit Summary – Snapshot 2025-10-22

## Sources

- Apple Health (Heart Rate, HRV, Sleep, Screen Time)
- Zepp / Helio Ring (Heart Rate, Temperature, Emotion)

## Integrity

- All files match manifest checksums.
- Temporal coverage: 2021-05-14 → 2025-10-21.
- No gaps > 6 h (HR/HRV) or > 24 h (sleep).

## Anomalies / Notes

- Missing Zepp HR mean/std on ~18 % of days.
- Apple HRV variance slightly high (S4 segment).

## Next Steps

- Validate label integration (State of Mind + EMA).
- Add Helio Ring emotion API data when available.

Generated: 2025-10-26
