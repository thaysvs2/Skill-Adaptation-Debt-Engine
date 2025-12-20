import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import config, data_prep, model, narrative  # noqa: E402


@st.cache_data
def load_data() -> pd.DataFrame:
    return data_prep.load_processed()


@st.cache_resource
def load_pipe():
    try:
        return model.load_model()
    except Exception:
        return None


def _pick_employee(df: pd.DataFrame, employee_number: int) -> pd.Series:
    if config.ID_COL in df.columns:
        m = df[config.ID_COL] == employee_number
        sub = df.loc[m]
        if not sub.empty:
            return sub.iloc[0]
    # fallback: by index
    return df.iloc[0]


def _contrib_chart(pairs, title: str):
    if not pairs:
        st.info("No contributions available.")
        return
    d = pd.DataFrame(pairs, columns=["Feature", "Contribution"])
    d["Contribution"] = pd.to_numeric(d["Contribution"], errors="coerce").fillna(0.0)
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Contribution to exit pressure (log-odds)"),
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature", alt.Tooltip("Contribution:Q", format=".4f")],
        )
        .properties(title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def main():
    st.set_page_config(page_title="Exit Narrative Generator", layout="wide")
    st.title("Exit Narrative Generator")
    st.caption("Turn attrition data into human-readable exit stories + counterfactual interventions.")

    df = load_data()
    if df is None or df.empty:
        st.error("Dataset is empty. Run `python -m src.cli prepare-data`.")
        return

    pipe = load_pipe()

    # Sidebar selection
    with st.sidebar:
        st.header("Select Employee")
        if config.ID_COL in df.columns and df[config.ID_COL].notna().any():
            ids = df[config.ID_COL].dropna().astype(int).unique().tolist()
            ids.sort()
            emp_id = st.selectbox("EmployeeNumber", ids, index=0)
        else:
            emp_id = None
            st.warning("EmployeeNumber not found. Falling back to first row.")

        st.header("Model")
        if pipe is None:
            st.warning("Model not trained yet. Run `python -m src.cli train`.")
        else:
            st.success("Model loaded.")

    if emp_id is not None:
        row = _pick_employee(df, int(emp_id))
    else:
        row = df.iloc[0]

    tabs = st.tabs(["Narrative Report", "Peer Context", "Intervention Builder"])

    # ----------------------------
    # Tab 1: Narrative Report
    # ----------------------------
    with tabs[0]:
        st.subheader("Narrative Report")

        if pipe is None:
            st.info("Train the model to enable narratives. Command: `python -m src.cli train`.")
            st.dataframe(pd.DataFrame(row).T.astype(str), width="stretch")
        else:
            nar = narrative.build_narrative(df, pipe, row)
            nar.counterfactuals = narrative.suggest_counterfactuals(df, pipe, row)

            col1, col2, col3 = st.columns(3)
            col1.metric("Exit Probability (model)", f"{nar.proba_exit:.3f}")
            col2.metric("Headline", nar.headline)
            col3.metric("Peer Group Size", str(nar.peer_context.get("group_size", "—")))

            st.markdown("### Exit Narrative")
            st.write(nar.exit_story)

            st.markdown("### Staying Narrative")
            st.write(nar.stay_story)

            st.markdown("### Dominant Pressures")
            _contrib_chart(nar.top_pressures[:8], "Top exit pressures (positive contributions)")

            st.markdown("### Retention Anchors")
            _contrib_chart(nar.top_stabilizers[:8], "Top stabilizers (negative contributions)")

            with st.expander("Raw record (debug)"):
                st.dataframe(pd.DataFrame(row).T.astype(str), width="stretch")

    # ----------------------------
    # Tab 2: Peer Context
    # ----------------------------
    with tabs[1]:
        st.subheader("Peer Context")
        st.write(
            "This view compares the selected employee against peers with the same "
            "**Department + JobRole + JobLevel** (when available)."
        )

        if pipe is None:
            st.info("Train the model first to enable peer summaries.")
        else:
            nar = narrative.build_narrative(df, pipe, row)
            ctx = nar.peer_context
            group = ctx.get("group", {})
            if not group:
                st.warning("Peer grouping columns not found in this dataset.")
            else:
                st.markdown("#### Peer Group Definition")
                st.json(group)

                st.markdown("#### Percentile Context (within peer group)")
                pct = ctx.get("percentiles", {})
                if not pct:
                    st.info("Not enough data to compute percentiles.")
                else:
                    pct_df = pd.DataFrame(
                        [{"Metric": k, "Percentile": v} for k, v in pct.items()]
                    )
                    pct_df["Percentile"] = (pct_df["Percentile"] * 100).round(1)
                    st.dataframe(pct_df.astype(str), width="stretch")

    # ----------------------------
    # Tab 3: Intervention Builder
    # ----------------------------
    with tabs[2]:
        st.subheader("Intervention Builder")
        st.write(
            "This tool searches for **single-step** counterfactual changes on a small set of "
            "controllable levers (overtime, work life balance, satisfaction, training)."
        )

        if pipe is None:
            st.info("Train the model first to enable interventions.")
        else:
            base = pd.DataFrame([row.drop(labels=[config.TARGET_COL], errors="ignore")])
            base_p = float(model.predict_proba(pipe, base)[0])

            st.markdown("#### Current Exit Probability")
            st.metric("P(exit)", f"{base_p:.3f}")

            st.markdown("#### Suggested Interventions (best first)")
            sug = narrative.suggest_counterfactuals(df, pipe, row, max_suggestions=8)

            if not sug:
                st.info("No strong interventions found (or dataset missing controllable columns).")
            else:
                sug_df = pd.DataFrame(sug)
                sug_df["delta"] = sug_df["delta"].map(lambda x: f"{x:+.3f}")
                sug_df["proba_before"] = sug_df["proba_before"].map(lambda x: f"{x:.3f}")
                sug_df["proba_after"] = sug_df["proba_after"].map(lambda x: f"{x:.3f}")
                st.dataframe(sug_df.astype(str), width="stretch")

            st.markdown("#### Manual What-if")
            st.caption("Change a few levers manually to see how the probability moves (counterfactual, not a promise).")

            editable = {}
            for feat, choices in config.CONTROLLABLE.items():
                if feat not in base.columns:
                    continue
                cur = base.iloc[0][feat]
                # Determine a safe default index
                default_idx = 0
                try:
                    if isinstance(choices[0], int):
                        cur_int = int(float(cur)) if cur is not None and str(cur) != "nan" else None
                        if cur_int in choices:
                            default_idx = choices.index(cur_int)
                    else:
                        cur_str = str(cur)
                        choice_strs = [str(x) for x in choices]
                        if cur_str in choice_strs:
                            default_idx = choice_strs.index(cur_str)
                except Exception:
                    default_idx = 0

                editable[feat] = st.selectbox(feat, choices, index=default_idx)

            if st.button("Compute manual scenario"):
                trial = base.copy()
                for k, v in editable.items():
                    trial.loc[trial.index[0], k] = v
                p2 = float(model.predict_proba(pipe, trial)[0])
                st.metric("P(exit) after scenario", f"{p2:.3f}")
                st.write(f"ΔP = {p2 - base_p:+.3f}")


if __name__ == "__main__":
    main()
