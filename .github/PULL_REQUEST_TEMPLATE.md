## Summary

<!-- What changed and why (not what — the diff shows that). -->

## Checklist

- [ ] Builds clean with `WERROR=ON ./build.sh`
- [ ] mAP unchanged (if this touches export/build/decode) — `profile.py --backends trt cpp --subset 2000` (or `--full`) holds the reference AP; before/after table included below if it applies
- [ ] Sanitizers clean on exercised paths (`BUILD_TYPE=UBSAN`/`ASAN`, `compute-sanitizer` for new kernels)
- [ ] References the relevant [HANDOFF](../docs/HANDOFF.md) section (or updates it, if this changes a documented decision)

## mAP before/after (if applicable)

```
paste profile.py table here, or delete this section
```
