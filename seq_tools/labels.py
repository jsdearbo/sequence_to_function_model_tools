"""
Splicing label generation from rMATS output and StringTie annotations.

Generates both hard (binary exon/intron) and soft (continuous PSI) labels
by integrating rMATS event calls with transcript structure, using a
priority-based disjoint segmentation approach.

Typical workflow::

    gtf = load_stringtie_gtf("stringtie.gtf")
    se  = load_rmats_se("SE.MATS.JCEC.txt")
    ri  = load_rmats_ri("RI.MATS.JCEC.txt")

    # Binary labels
    hard = generate_hard_labels(gtf, se, ri)

    # Continuous PSI labels
    soft = generate_soft_labels(gtf, se, ri)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pyranges as pr
    _HAS_PYRANGES = True
except ImportError:
    _HAS_PYRANGES = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_strand(df_like):
    """Standardize Strand to a strict Category (+, -, .)."""
    if not _HAS_PYRANGES:
        raise ImportError("pyranges is required for label generation.")

    if df_like is None:
        return pr.PyRanges()

    if isinstance(df_like, pr.PyRanges):
        df = df_like.df.copy()
    else:
        df = df_like.copy()

    if "Strand" not in df.columns:
        df["Strand"] = "."
    df["Strand"] = pd.Categorical(
        df["Strand"].fillna("."), categories=["+", "-", "."]
    )
    return pr.PyRanges(df)


def _ensure_pr(obj):
    """Guarantee input is a valid, normalized PyRanges object."""
    if not _HAS_PYRANGES:
        raise ImportError("pyranges is required for label generation.")

    if obj is None:
        return pr.PyRanges()
    if hasattr(obj, "__len__") and len(obj) == 0:
        return pr.PyRanges()
    if isinstance(obj, pr.PyRanges):
        return _normalize_strand(obj)
    if isinstance(obj, pd.DataFrame):
        required = {"Chromosome", "Start", "End"}
        missing = required - set(obj.columns)
        if missing:
            raise ValueError(f"Missing columns for PyRanges conversion: {missing}")
        obj = obj.copy()
        obj["Start"] = obj["Start"].astype(int)
        obj["End"] = obj["End"].astype(int)
        return _normalize_strand(obj)
    raise TypeError(f"Unsupported type {type(obj)} for PyRanges conversion")


def _aggregate_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Average PSI and sum coverage for exact coordinate duplicates."""
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    agg_rules = {"psi": "mean", "coverage": "sum"}
    return df.groupby(
        ["Chromosome", "Start", "End", "Strand"], observed=True, as_index=False
    ).agg(agg_rules)


def collapse_exon_coords_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate exon coordinates by summing junction counts,
    then recomputing PSI from aggregated counts.

    This gives proper read-weighted aggregation rather than naively
    averaging PSI across contexts with different read support.
    """
    if df.empty:
        return df
    df = df.reset_index(drop=True)

    ijc_cols = [c for c in df.columns if c.startswith("ijc_")]
    sjc_cols = [c for c in df.columns if c.startswith("sjc_")]

    if not ijc_cols or not sjc_cols:
        return _aggregate_exact_duplicates(df)

    gcols = ["Chromosome", "Start", "End", "Strand"]
    agg_dict = {c: "sum" for c in ijc_cols + sjc_cols}
    if "coverage" in df.columns:
        agg_dict["coverage"] = "sum"

    summed = df.groupby(gcols, observed=True, as_index=False).agg(agg_dict)

    # Recompute per-sample PSI from summed counts
    n_samples = len(ijc_cols)
    for i in range(n_samples):
        inc = summed[f"ijc_{i}"].astype(float)
        skp = summed[f"sjc_{i}"].astype(float)
        denom = inc + skp
        summed[f"psi_{i}"] = np.where(denom > 0, inc / denom, np.nan)

    # Pooled PSI across all samples
    total_ijc = summed[ijc_cols].sum(axis=1).astype(float)
    total_sjc = summed[sjc_cols].sum(axis=1).astype(float)
    pooled_denom = total_ijc + total_sjc
    summed["psi_pooled"] = np.where(
        pooled_denom > 0, total_ijc / pooled_denom, np.nan
    )

    psi_cols = [c for c in summed.columns if c.startswith("psi_")]
    summed["psi_mean"] = summed[psi_cols].mean(axis=1, skipna=True)
    summed["psi"] = summed["psi_pooled"].where(
        summed["psi_pooled"].notna(), summed["psi_mean"]
    )
    summed["confidence"] = np.log1p(total_ijc + total_sjc)

    return summed


# ---------------------------------------------------------------------------
# rMATS loaders
# ---------------------------------------------------------------------------

def _load_rmats_common(
    rmats_file: str,
    min_coverage: int = 20,
    mode: str = "per-sample",
) -> pd.DataFrame:
    """
    Load rMATS JCEC output with per-sample PSI and junction counts.

    Args:
        rmats_file: Path to rMATS output (e.g. SE.MATS.JCEC.txt).
        min_coverage: Minimum total junction count to keep an event.
        mode: 'per-sample' keeps individual PSI columns; 'mean' adds psi_mean.

    Returns:
        DataFrame with columns psi_0..psi_N, ijc_0..ijc_N, sjc_0..sjc_N,
        and coverage.
    """
    try:
        df = pd.read_csv(rmats_file, sep="\t")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning("Could not load %s: %s", rmats_file, exc)
        return pd.DataFrame()

    def _parse_float_list(values):
        if pd.isna(values):
            return []
        out = []
        for t in str(values).split(","):
            if t in {"", "NA"}:
                continue
            try:
                out.append(float(t))
            except ValueError:
                continue
        return out

    def _parse_int_list(values):
        if pd.isna(values):
            return []
        out = []
        for t in str(values).split(","):
            if t in {"", "NA"}:
                out.append(0)
            else:
                try:
                    out.append(int(float(t)))
                except ValueError:
                    out.append(0)
        return out

    # Per-sample PSI
    psi_lists = df["IncLevel1"].apply(_parse_float_list)
    max_n = psi_lists.map(len).max()
    if pd.isna(max_n) or max_n == 0:
        return pd.DataFrame()

    psi_cols = [f"psi_{i}" for i in range(max_n)]
    psi_df = pd.DataFrame(psi_lists.tolist(), columns=psi_cols, index=df.index)
    df = pd.concat([df, psi_df], axis=1)

    if mode == "mean":
        df["psi_mean"] = psi_df.mean(axis=1, skipna=True)

    # Per-sample junction counts
    ijc_lists = df["IJC_SAMPLE_1"].apply(_parse_int_list)
    sjc_lists = df["SJC_SAMPLE_1"].apply(_parse_int_list)

    def _pad(xs, n, fill=0):
        xs = list(xs)
        return (xs + [fill] * max(0, n - len(xs)))[:n]

    ijc_mat = np.vstack([_pad(x, max_n) for x in ijc_lists.tolist()])
    sjc_mat = np.vstack([_pad(x, max_n) for x in sjc_lists.tolist()])

    for i in range(max_n):
        df[f"ijc_{i}"] = ijc_mat[:, i]
        df[f"sjc_{i}"] = sjc_mat[:, i]

    # Total coverage
    count_cols = [c for c in df.columns if c.startswith(("IJC_SAMPLE", "SJC_SAMPLE"))]

    def _parse_int_sum(values):
        if pd.isna(values):
            return 0
        total = 0
        for token in str(values).split(","):
            try:
                total += int(float(token))
            except ValueError:
                continue
        return total

    for col in count_cols:
        df[col] = df[col].apply(_parse_int_sum)

    df["coverage"] = df[count_cols].sum(axis=1)
    df = df[df["coverage"] >= min_coverage]

    psi_non_na = psi_df.notna().any(axis=1)
    df = df.loc[psi_non_na].reset_index(drop=True)
    return df


def load_rmats_se(
    rmats_file: str, min_coverage: int = 20, mode: str = "per-sample"
):
    """Load rMATS skipped exon (SE) events as PyRanges."""
    df = _load_rmats_common(rmats_file, min_coverage, mode=mode)
    if df.empty:
        return _ensure_pr(None)

    df = df.rename(columns={
        "chr": "Chromosome",
        "exonStart_0base": "Start",
        "exonEnd": "End",
        "strand": "Strand",
    })
    df[["Start", "End"]] = df[["Start", "End"]].astype(int)
    df = df[df["Start"] < df["End"]]
    return _normalize_strand(df)


def load_rmats_ri(
    rmats_file: str, min_coverage: int = 20, mode: str = "per-sample"
):
    """Load rMATS retained intron (RI) events as PyRanges."""
    df = _load_rmats_common(rmats_file, min_coverage, mode=mode)
    if df.empty:
        return _ensure_pr(None)

    df["Start"] = df["upstreamEE"]
    df["End"] = df["downstreamES"]
    df = df.rename(columns={"chr": "Chromosome", "strand": "Strand"})
    df[["Start", "End"]] = df[["Start", "End"]].astype(int)
    df = df[df["Start"] < df["End"]]
    return _normalize_strand(df)


def load_rmats_alt_site(
    rmats_file: str, min_coverage: int = 20, mode: str = "per-sample"
):
    """Load rMATS A3SS/A5SS events as PyRanges (extension-only regions)."""
    df = _load_rmats_common(rmats_file, min_coverage, mode=mode)
    if df.empty:
        return _ensure_pr(None)

    payload_cols = [c for c in df.columns if c.startswith(("psi_", "ijc_", "sjc_"))]
    if "psi_mean" in df.columns:
        payload_cols.append("psi_mean")
    if "coverage" in df.columns:
        payload_cols.append("coverage")

    long_df = df.rename(columns={
        "chr": "Chromosome",
        "longExonStart_0base": "Start",
        "longExonEnd": "End",
        "strand": "Strand",
    })[["Chromosome", "Start", "End", "Strand"] + payload_cols]

    short_df = df.rename(columns={
        "chr": "Chromosome",
        "shortES": "Start",
        "shortEE": "End",
        "strand": "Strand",
    })[["Chromosome", "Start", "End", "Strand"]]

    pr_long = _normalize_strand(long_df)
    pr_short = _normalize_strand(short_df)

    extension = pr_long.subtract(pr_short)
    if len(extension) == 0:
        return _ensure_pr(None)
    return _normalize_strand(extension.df)


def load_stringtie_gtf(gtf_file: str):
    """Load a StringTie GTF as PyRanges."""
    if not _HAS_PYRANGES:
        raise ImportError("pyranges is required.")
    return pr.read_gtf(gtf_file)


# ---------------------------------------------------------------------------
# Disjoint labeling with priority
# ---------------------------------------------------------------------------

def _disjoint_labels_with_priority(
    transcripts_df: pd.DataFrame,
    exons_df: pd.DataFrame,
    psi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build disjoint genomic segments and assign p(exon) with priority:
    PSI > annotated exon > intron.

    Uses a sweep-line algorithm over sorted breakpoints to create
    non-overlapping segments, then assigns labels based on which
    annotation layers are active at each segment.

    Returns:
        DataFrame with columns [Chromosome, Start, End, p_exon, confidence, Strand].
    """
    def _prep(df, need_psi=False, need_conf=False):
        if df is None or (hasattr(df, "empty") and df.empty):
            cols = ["Chromosome", "Start", "End", "Strand"]
            if need_psi:
                cols += ["psi"]
            if need_conf:
                cols += ["confidence"]
            return pd.DataFrame(columns=cols)
        df = df.copy()
        if "Strand" not in df.columns:
            df["Strand"] = "."
        df["Start"] = df["Start"].astype(int)
        df["End"] = df["End"].astype(int)
        keep = ["Chromosome", "Start", "End", "Strand"]
        if need_psi:
            keep += ["psi"]
        if need_conf:
            if "confidence" not in df.columns:
                df["confidence"] = 0.0
            keep += ["confidence"]
        return df[keep]

    T = _prep(transcripts_df)
    E = _prep(exons_df)
    P = _prep(psi_df, need_psi=True, need_conf=True)

    if T.empty:
        return pd.DataFrame(
            columns=["Chromosome", "Start", "End", "p_exon", "confidence", "Strand"]
        )

    out_rows = []
    keys = T[["Chromosome", "Strand"]].drop_duplicates().itertuples(
        index=False, name=None
    )

    for chrom, strand in keys:
        t = T[(T.Chromosome == chrom) & (T.Strand == strand)]
        e = E[(E.Chromosome == chrom) & (E.Strand == strand)]
        p = P[(P.Chromosome == chrom) & (P.Strand == strand)]

        # Collect all breakpoints
        boundaries = np.concatenate([
            t["Start"].to_numpy(), t["End"].to_numpy(),
            e["Start"].to_numpy() if not e.empty else np.array([], dtype=int),
            e["End"].to_numpy() if not e.empty else np.array([], dtype=int),
            p["Start"].to_numpy() if not p.empty else np.array([], dtype=int),
            p["End"].to_numpy() if not p.empty else np.array([], dtype=int),
        ])
        if boundaries.size == 0:
            continue
        boundaries = np.unique(boundaries)
        boundaries.sort()

        # Build add/remove event maps
        t_add, t_rem = {}, {}
        for s, en in zip(t["Start"].to_numpy(), t["End"].to_numpy()):
            t_add.setdefault(int(s), 0)
            t_add[int(s)] += 1
            t_rem.setdefault(int(en), 0)
            t_rem[int(en)] += 1

        e_add, e_rem = {}, {}
        if not e.empty:
            for s, en in zip(e["Start"].to_numpy(), e["End"].to_numpy()):
                e_add.setdefault(int(s), 0)
                e_add[int(s)] += 1
                e_rem.setdefault(int(en), 0)
                e_rem[int(en)] += 1

        p_add, p_rem = {}, {}
        if not p.empty:
            p = p.reset_index(drop=True)
            has_conf = "confidence" in p.columns
            for idx, row in p.iterrows():
                s_val = int(row["Start"])
                e_val = int(row["End"])
                psi_val = float(row["psi"])
                conf_val = float(row["confidence"]) if has_conf else 0.0
                p_add.setdefault(s_val, []).append((idx, psi_val, conf_val))
                p_rem.setdefault(e_val, []).append((idx, psi_val, conf_val))

        # Sweep line
        active_t = 0
        active_e = 0
        active_p_ids: dict = {}
        active_p_sum = 0.0
        active_c_sum = 0.0

        def _p_mean():
            n = len(active_p_ids)
            return (active_p_sum / n) if n else None

        def _c_mean():
            n = len(active_p_ids)
            return (active_c_sum / n) if n else 0.0

        prev_label = None
        prev_conf = None
        prev_start = None

        for i in range(len(boundaries) - 1):
            x = int(boundaries[i])
            y = int(boundaries[i + 1])
            if x == y:
                continue

            if x in t_add:
                active_t += t_add[x]
            if x in t_rem:
                active_t -= t_rem[x]
            if x in e_add:
                active_e += e_add[x]
            if x in e_rem:
                active_e -= e_rem[x]
            if x in p_add:
                for pid, psi, conf in p_add[x]:
                    if pid not in active_p_ids:
                        active_p_ids[pid] = (psi, conf)
                        active_p_sum += psi
                        active_c_sum += conf
            if x in p_rem:
                for pid, _psi, _conf in p_rem[x]:
                    if pid in active_p_ids:
                        old_psi, old_conf = active_p_ids.pop(pid)
                        active_p_sum -= old_psi
                        active_c_sum -= old_conf

            if active_t <= 0:
                if prev_start is not None:
                    out_rows.append(
                        (chrom, prev_start, x, prev_label, prev_conf, strand)
                    )
                    prev_start, prev_label, prev_conf = None, None, None
                continue

            pm = _p_mean()
            cm = _c_mean()
            label = pm if pm is not None else (1.0 if active_e > 0 else 0.0)

            if prev_start is None:
                prev_start = x
                prev_label = label
                prev_conf = cm
            elif label != prev_label:
                out_rows.append(
                    (chrom, prev_start, x, prev_label, prev_conf, strand)
                )
                prev_start = x
                prev_label = label
                prev_conf = cm

        last_boundary = int(boundaries[-1])
        if prev_start is not None and prev_start < last_boundary:
            out_rows.append(
                (chrom, prev_start, last_boundary, prev_label, prev_conf, strand)
            )

    return pd.DataFrame(
        out_rows,
        columns=["Chromosome", "Start", "End", "p_exon", "confidence", "Strand"],
    )


# ---------------------------------------------------------------------------
# Public label generators
# ---------------------------------------------------------------------------

def generate_hard_labels(
    stringtie_gr,
    rmats_se,
    rmats_ri=None,
    rmats_a3ss=None,
    rmats_a5ss=None,
    psi_low: float = 0.2,
    psi_high: float = 0.8,
) -> pd.DataFrame:
    """
    Generate binary exon/intron labels corrected by rMATS events.

    Events with PSI < ``psi_low`` are treated as introns; events with
    PSI > ``psi_high`` are treated as exons.

    Returns:
        DataFrame with a 'label' column (1 = intron, 2 = exon).
    """
    stringtie_gr = _ensure_pr(stringtie_gr)
    rmats_se = _ensure_pr(rmats_se)
    rmats_ri = _ensure_pr(rmats_ri)
    rmats_a3ss = _ensure_pr(rmats_a3ss)
    rmats_a5ss = _ensure_pr(rmats_a5ss)

    st_exons = stringtie_gr[stringtie_gr.Feature == "exon"]
    st_transcripts = stringtie_gr[stringtie_gr.Feature == "transcript"]
    if len(st_transcripts) == 0:
        st_transcripts = st_exons.merge(by="transcript_id")

    # Correct exons using SE events
    false_exons = rmats_se[rmats_se.psi < psi_low]
    corrected = st_exons.subtract(false_exons) if len(false_exons) else st_exons

    missed_exons = rmats_se[rmats_se.psi > psi_high]
    if len(missed_exons):
        corrected = pr.concat([corrected, missed_exons]).merge()

    # Retained introns
    if len(rmats_ri):
        ri_high = rmats_ri[rmats_ri.psi > psi_high]
        if len(ri_high):
            ri_high = _normalize_strand(ri_high)
            corrected = pr.concat([corrected, ri_high]).merge()

    # Alternative splice sites (A3SS/A5SS)
    extensions_list = []
    if len(rmats_a3ss):
        extensions_list.append(rmats_a3ss)
    if len(rmats_a5ss):
        extensions_list.append(rmats_a5ss)
    extensions = pr.concat(extensions_list) if extensions_list else pr.PyRanges()

    if len(extensions):
        ext_high = extensions[extensions.psi > psi_high]
        if len(ext_high):
            corrected = pr.concat([corrected, _normalize_strand(ext_high)]).merge()
        ext_low = extensions[extensions.psi < psi_low]
        if len(ext_low):
            corrected = corrected.subtract(_normalize_strand(ext_low))

    introns = st_transcripts.subtract(corrected)
    corrected = _normalize_strand(corrected)
    introns = _normalize_strand(introns)

    corrected.label = 2
    introns.label = 1

    return pr.concat([corrected, introns]).sort().df


def generate_soft_labels(
    stringtie_gr,
    rmats_se,
    rmats_ri=None,
    rmats_a3ss=None,
    rmats_a5ss=None,
) -> pd.DataFrame:
    """
    Generate continuous PSI-based labels using a priority segmentation approach.

    Segments where rMATS provides PSI estimates use those values directly.
    Other exonic segments get p_exon=1.0; intronic segments get p_exon=0.0.

    Returns:
        DataFrame with columns [Chromosome, Start, End, p_exon, confidence, Strand].
    """
    stringtie_gr = _ensure_pr(stringtie_gr)
    rmats_se = _ensure_pr(rmats_se)
    rmats_ri = _ensure_pr(rmats_ri)
    rmats_a3ss = _ensure_pr(rmats_a3ss)
    rmats_a5ss = _ensure_pr(rmats_a5ss)

    # Collect all PSI regions and weighted-collapse duplicates
    vars_list = [
        gr for gr in (rmats_se, rmats_ri, rmats_a3ss, rmats_a5ss) if len(gr) > 0
    ]
    psi_df = pd.DataFrame(
        columns=["Chromosome", "Start", "End", "Strand", "psi", "confidence"]
    )
    if vars_list:
        vars_pr = pr.concat(vars_list)
        vars_collapsed = collapse_exon_coords_weighted(vars_pr.df)
        psi_df = vars_collapsed[
            ["Chromosome", "Start", "End", "Strand", "psi", "confidence"]
        ].copy()

    # Transcript and exon structure from StringTie
    gdf = stringtie_gr.df
    exons_df = gdf[gdf.Feature == "exon"][
        ["Chromosome", "Start", "End", "Strand"]
    ].copy()
    transcripts_df = gdf[gdf.Feature == "transcript"][
        ["Chromosome", "Start", "End", "Strand"]
    ].copy()

    if transcripts_df.empty and not exons_df.empty:
        if "transcript_id" in gdf.columns:
            spans = (
                gdf[gdf.Feature == "exon"]
                .groupby(["Chromosome", "Strand", "transcript_id"], as_index=False)
                .agg(Start=("Start", "min"), End=("End", "max"))
            )
            transcripts_df = spans[["Chromosome", "Start", "End", "Strand"]].copy()
        else:
            transcripts_df = exons_df.copy()

    out = _disjoint_labels_with_priority(transcripts_df, exons_df, psi_df)

    if "confidence" not in out.columns:
        out["confidence"] = 0.0

    return out
