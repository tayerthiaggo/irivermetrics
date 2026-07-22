# Legacy notebooks (quarantined)

These are the pre-v1.2 `iRivermetrics` example notebooks. Neither is wired
to the current `hydrofragments` API (`open_water_cube`/`analyze`) -- for
runnable, up-to-date examples see `../01_quickstart.ipynb`,
`../02_dea_via_tsfill.ipynb`, and `../03_metrics_walkthrough.ipynb` instead.

- `STAC_query.ipynb` -- valid JSON (a leading UTF-8 BOM makes a strict
  `utf-8`-mode `json.load` reject it, but it parses cleanly with
  `encoding="utf-8-sig"`); references the legacy `irivermetrics` package
  and STAC-querying code that predates this package's rewrite.
- `irm_example.ipynb` -- **genuinely malformed JSON**, not just a BOM
  issue: after stripping the BOM it still fails to parse (`Expecting
  value: line 26 column 26`) because a literal, unescaped `\r\n` sequence
  sits directly in the JSON structure between two cell-source string
  entries, where a complete JSON token is expected. This is real
  corruption (likely from a bad save/merge), not an artifact of decoding.
  Kept here rather than deleted so the historical example is not silently
  lost, but it will not open correctly in Jupyter or any strict JSON
  parser without manual repair.
