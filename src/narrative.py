from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from . import config, model as model_mod


@dataclass
class Narrative:
    employee_id: int | None
    proba_exit: float
    headline: str
    exit_story: str
    stay_story: str
    top_pressures: List[Tuple[str, float]]
    top_stabilizers: List[Tuple[str, float]]
    peer_context: Dict[str, Any]
    counterfactuals: List[Dict[str, Any]]


def _get_feature_names(pre) -> List[str]:
    # Works for sklearn ColumnTransformer + OneHotEncoder
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        # fallback: build manually
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    tn = list(trans.get_feature_names_out(cols))
                    names.extend(tn)
                    continue
                except Exception:
                    pass
            # last resort
            names.extend([f"{name}__{c}" for c in cols])
        return names


def _friendly_feature_name(raw: str) -> str:
    # Convert sklearn names like "cat__OverTime_Yes" or "num__Age"
    s = raw.replace("num__", "").replace("cat__", "")
    s = s.replace("__", ": ")
    return s


def _template_for_feature(f: str, value: Any, direction: str) -> str:
    # direction: "pressure" or "stabilizer"
    # Very small template layer: readable but not too rigid.
    base = f
    if "OverTime" in f:
        return "Persistent overtime increases exhaustion and reduces recovery capacity."
    if "YearsSinceLastPromotion" in f:
        return "Promotion stagnation suggests a blocked growth path."
    if "JobSatisfaction" in f:
        return "Low job satisfaction erodes the psychological reason to stay."
    if "WorkLifeBalance" in f:
        return "Work–life imbalance accumulates as hidden stress."
    if "MonthlyIncome" in f:
        return "Compensation acts as a retention buffer; weak buffers make exit easier."
    if "DistanceFromHome" in f:
        return "Commute friction increases long-term attrition pressure."
    if "YearsWithCurrManager" in f:
        return "Manager relationship stability can anchor retention."
    if "TrainingTimesLastYear" in f:
        return "Learning velocity is a retention buffer; low training reduces future options."
    if "JobInvolvement" in f:
        return "Low involvement often precedes disengagement."
    if "EnvironmentSatisfaction" in f:
        return "Poor environment satisfaction makes staying costly."
    if "RelationshipSatisfaction" in f:
        return "Weak social ties reduce resistance to leaving."
    return f"{base} is a key driver in this profile."


def _peer_context(df: pd.DataFrame, row: pd.Series) -> Dict[str, Any]:
    ctx = {"group": {}, "percentiles": {}}
    # Build peer group
    group_cols = [c for c in config.PEER_GROUP_COLS if c in df.columns and c in row.index]
    if not group_cols:
        return ctx
    mask = pd.Series(True, index=df.index)
    for c in group_cols:
        mask &= (df[c].astype(str) == str(row[c]))
        ctx["group"][c] = str(row[c])
    peers = df.loc[mask].copy()
    ctx["group_size"] = int(len(peers))

    def pct(col: str) -> float | None:
        if col not in peers.columns:
            return None
        s = pd.to_numeric(peers[col], errors="coerce")
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if s.notna().sum() < 5 or pd.isna(v):
            return None
        return float((s <= v).mean())

    # Pick a few meaningful context metrics
    for col in ["MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", "YearsSinceLastPromotion", "YearsAtCompany"]:
        p = pct(col)
        if p is not None:
            ctx["percentiles"][col] = p

    return ctx


def _contributions_for_row(pipe, row_df: pd.DataFrame) -> List[Tuple[str, float]]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X = row_df.copy()
    Xt = pre.transform(X)
    # Xt may be sparse depending on sklearn; force array
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    coefs = clf.coef_.reshape(-1)  # (n_features,)
    contrib = (Xt.reshape(-1) * coefs).tolist()

    names = _get_feature_names(pre)
    items = list(zip(names, contrib))
    # Sort by absolute magnitude
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    return items


def _headline_from_proba(p: float) -> str:
    if p >= 0.75:
        return "🔴 Exit narrative is structurally strong (high exit pressure)."
    if p >= 0.55:
        return "🟠 Exit narrative is forming (moderate exit pressure)."
    if p >= 0.40:
        return "🟡 Borderline: retention and exit forces are competing."
    return "🟢 Staying is structurally supported (low exit pressure)."


def build_narrative(
    df_all: pd.DataFrame,
    pipe,
    row: pd.Series,
    top_k: int = 6,
) -> Narrative:
    # Predict probability
    Xrow = pd.DataFrame([row.drop(labels=[config.TARGET_COL], errors="ignore")])
    p = float(model_mod.predict_proba(pipe, Xrow)[0])

    contribs = _contributions_for_row(pipe, Xrow)

    pressures = []
    stabilizers = []
    for name, val in contribs:
        if len(pressures) >= top_k and len(stabilizers) >= top_k:
            break
        if val > 0 and len(pressures) < top_k:
            pressures.append((_friendly_feature_name(name), float(val)))
        if val < 0 and len(stabilizers) < top_k:
            stabilizers.append((_friendly_feature_name(name), float(val)))

    # Build narrative text
    pressure_sentences = []
    for feat, val in pressures[:min(4, len(pressures))]:
        pressure_sentences.append(_template_for_feature(feat, None, "pressure"))

    stabilizer_sentences = []
    for feat, val in stabilizers[:min(3, len(stabilizers))]:
        stabilizer_sentences.append(_template_for_feature(feat, None, "stabilizer"))

    exit_story = (
        "This profile does not look like a sudden decision. It looks like **accumulated pressure**. "
        + " ".join(pressure_sentences)
        + " Together these forces reduce recovery and make exit increasingly rational."
    )

    stay_story = (
        "At the same time, there are **retention anchors**. "
        + (" ".join(stabilizer_sentences) if stabilizer_sentences else "Some stabilizing factors are present, but weaker than the exit pressures.")
        + " These anchors can delay exit, but may not fully counter the dominant pressures."
    )

    ctx = _peer_context(df_all, row)

    return Narrative(
        employee_id=int(row.get(config.ID_COL)) if pd.notna(row.get(config.ID_COL, np.nan)) else None,
        proba_exit=p,
        headline=_headline_from_proba(p),
        exit_story=exit_story,
        stay_story=stay_story,
        top_pressures=pressures,
        top_stabilizers=stabilizers,
        peer_context=ctx,
        counterfactuals=[],
    )


def suggest_counterfactuals(
    df_all: pd.DataFrame,
    pipe,
    row: pd.Series,
    max_suggestions: int = 5,
) -> List[Dict[str, Any]]:
    # Greedy single-step interventions on controllable levers
    base = pd.DataFrame([row.drop(labels=[config.TARGET_COL], errors="ignore")])
    base_p = float(model_mod.predict_proba(pipe, base)[0])

    suggestions: List[Dict[str, Any]] = []

    for feat, choices in config.CONTROLLABLE.items():
        if feat not in base.columns:
            continue
        cur = base.iloc[0][feat]
        for new_val in choices:
            if str(new_val) == str(cur):
                continue
            trial = base.copy()
            trial.loc[trial.index[0], feat] = new_val
            p = float(model_mod.predict_proba(pipe, trial)[0])
            delta = base_p - p
            suggestions.append(
                {
                    "feature": feat,
                    "from": cur,
                    "to": new_val,
                    "proba_before": base_p,
                    "proba_after": p,
                    "delta": delta,
                }
            )

    suggestions.sort(key=lambda d: d["delta"], reverse=True)
    out = []
    for s in suggestions:
        if len(out) >= max_suggestions:
            break
        # Keep only meaningful improvements
        if s["delta"] > 0.01:
            out.append(s)
    return out
