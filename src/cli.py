from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data_prep, model, narrative


def cmd_prepare(args: argparse.Namespace) -> None:
    df = data_prep.prepare_processed()
    print(f"Prepared processed data: {config.DATA_PROCESSED}")
    print(f"Rows: {len(df)} | Cols: {len(df.columns)}")


def cmd_train(args: argparse.Namespace) -> None:
    res = model.train(random_state=args.seed, test_size=args.test_size)
    print(f"Saved model: {res['model_path']}")
    print("Metrics:")
    for k, v in res["metrics"].items():
        print(f"- {k}: {v}")


def _pick_row(df: pd.DataFrame, employee_id: int | None, index: int | None) -> pd.Series:
    if employee_id is not None and config.ID_COL in df.columns:
        m = df[config.ID_COL] == employee_id
        sub = df.loc[m]
        if sub.empty:
            raise SystemExit(f"No row found with {config.ID_COL}={employee_id}")
        return sub.iloc[0]
    idx = index if index is not None else 0
    if idx < 0 or idx >= len(df):
        raise SystemExit(f"Index {idx} out of range.")
    return df.iloc[idx]


def cmd_explain(args: argparse.Namespace) -> None:
    df = data_prep.load_processed()
    pipe = model.load_model()
    row = _pick_row(df, args.id, args.index)
    nar = narrative.build_narrative(df, pipe, row)
    nar.counterfactuals = narrative.suggest_counterfactuals(df, pipe, row)

    print(nar.headline)
    print(f"Exit probability (model): {nar.proba_exit:.3f}")
    print("\nExit Narrative:")
    print(nar.exit_story)
    print("\nStaying Narrative:")
    print(nar.stay_story)

    print("\nTop Pressures:")
    for f, v in nar.top_pressures:
        print(f"- {f}: +{v:.4f}")

    print("\nTop Stabilizers:")
    for f, v in nar.top_stabilizers:
        print(f"- {f}: {v:.4f}")

    if nar.peer_context.get("group"):
        print("\nPeer Context:")
        print(nar.peer_context)

    if nar.counterfactuals:
        print("\nCounterfactual suggestions (single-step):")
        for s in nar.counterfactuals:
            print(f"- {s['feature']}: {s['from']} -> {s['to']} | Δp={s['delta']:.3f} (to {s['proba_after']:.3f})")


def cmd_export(args: argparse.Namespace) -> None:
    df = data_prep.load_processed()
    pipe = model.load_model()

    rows = []
    for _, r in df.iterrows():
        nar = narrative.build_narrative(df, pipe, r)
        rows.append(
            {
                config.ID_COL: nar.employee_id,
                "proba_exit": nar.proba_exit,
                "headline": nar.headline,
                "exit_story": nar.exit_story,
                "stay_story": nar.stay_story,
                "peer_group_size": nar.peer_context.get("group_size", None),
            }
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved narratives to: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exit Narrative Generator CLI")
    sp = p.add_subparsers(dest="command", required=True)

    p1 = sp.add_parser("prepare-data", help="Clean and cache dataset to parquet.")
    p1.set_defaults(func=cmd_prepare)

    p2 = sp.add_parser("train", help="Train interpretable attrition model.")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--test-size", type=float, default=0.25)
    p2.set_defaults(func=cmd_train)

    p3 = sp.add_parser("explain", help="Generate narrative for one employee.")
    p3.add_argument("--id", type=int, default=None, help="EmployeeNumber")
    p3.add_argument("--index", type=int, default=None, help="Row index fallback")
    p3.set_defaults(func=cmd_explain)

    p4 = sp.add_parser("export-narratives", help="Export short narratives for all employees.")
    p4.add_argument("--out", type=Path, default=config.REPORTS_NARRATIVES_DIR / "all_narratives.csv")
    p4.set_defaults(func=cmd_export)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
