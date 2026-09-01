# Reproducibility and integrity

## What is included

The repository includes the two best scored CSV files and a machine-readable
manifest containing their public scores, row counts, and SHA-256 hashes. Raw
competition data, third-party source files, credentials, downloaded model
weights, and intermediate predictions are intentionally excluded.

## Verify archived submissions

From the repository root:

```bash
python scripts/verify_final_submissions.py
```

The verifier checks:

1. exact `id,tuketim` schema;
2. 714,688 rows and 714,688 unique IDs;
3. finite and non-negative predictions;
4. exact SHA-256 digest from the frozen manifest.

If the official competition ZIP is available, also verify sample order:

```bash
python scripts/verify_final_submissions.py \
  --competition-zip /path/to/grid-up-datathon.zip
```

On Windows PowerShell:

```powershell
python .\scripts\verify_final_submissions.py `
  --competition-zip C:\path\to\grid-up-datathon.zip
```

## Rebuild scope

The original starter pipeline in `src/` and the experiment utilities remain in
the repository. The final V108 CSV is archived as an immutable output rather
than advertised as a clean one-command rebuild: its endpoint ensemble depends
on numerous intermediate predictions and external datasets that are not
redistributed.

This distinction is intentional. File integrity is fully reproducible from the
manifest; end-to-end model training additionally requires the competition data,
the declared external sources, and the archived experiment lineage.
