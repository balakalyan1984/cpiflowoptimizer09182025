import os, json, pathlib, pandas as pd, joblib
pipe.fit(X_tr, y_tr)


# --- Eval ---
y_pred = pipe.predict(X_te)
acc = float(accuracy_score(y_te, y_pred))
print("Accuracy:", acc)
print(classification_report(y_te, y_pred, zero_division=0))


# --- Insights: per-artifact KPIs ---
def build_artifact_kpis(frame: pd.DataFrame) -> pd.DataFrame:
total = frame.groupby(ART_COL).size().rename("events")
status = frame.pivot_table(index=ART_COL, columns=TARGET_COLUMN, values=LVL_COL,
aggfunc="count", fill_value=0)
success = status.get("SUCCESS", pd.Series(0, index=total.index)).rename("success")
errors = (status.sum(axis=1) - success).rename("errors")
out = pd.concat([total, success, errors], axis=1)
out["success_rate"] = (out["success"] / out["events"]).round(3)
out["error_rate"] = (out["errors"] / out["events"]).round(3)
err_cols = [c for c in status.columns if c != "SUCCESS"]
out["top_error_type"] = status[err_cols].idxmax(axis=1) if err_cols else ""
out["top_error_share"] = ((status[err_cols].max(axis=1) / out["events"]).fillna(0).round(3)
if err_cols else 0)


errs = frame[frame[TARGET_COLUMN] != "SUCCESS"]
if not errs.empty:
comp_lvl = (errs.groupby([ART_COL, COMP_COL, LVL_COL]).size()
.rename("count").reset_index())
top_pairs = (comp_lvl.sort_values([ART_COL, "count"], ascending=[True, False])
.groupby(ART_COL).head(2))
pair_str = (top_pairs
.assign(pair=lambda x: x[COMP_COL] + "|" + x[LVL_COL] + " (n=" + x["count"].astype(str) + ")")
.groupby(ART_COL)["pair"].apply(lambda s: "; ".join(s.tolist())))
out["top_err_components"] = out.index.map(pair_str.to_dict()).fillna("")
else:
out["top_err_components"] = ""
return out.sort_values(["error_rate","events"], ascending=[False, False]).reset_index()


artifact_kpis = build_artifact_kpis(df)


# --- Insights: global error hotspots ---
hotspots = (df[df[TARGET_COLUMN]!="SUCCESS"]
.groupby([COMP_COL, LVL_COL]).size()
.sort_values(ascending=False).reset_index(name="count"))


# --- Attach metadata into the pipeline (so serving can expose insights) ---
pipe.meta = {
"artifact_kpis": artifact_kpis.set_index("ARTIFACT_NAME")[
["events","success_rate","error_rate","top_error_type","top_err_components"]
].to_dict(orient="index"),
"hotspots_top10": hotspots.head(10).to_dict(orient="records"),
}


# --- Save model + side products ---
joblib.dump(pipe, MODEL_PATH)
with open(METRICS_JSON, "w") as f:
json.dump({"accuracy": acc, "features": feat_cols, "target": TARGET_COLUMN}, f)
artifact_kpis.to_csv(INSIGHTS_CSV, index=False)
hotspots.to_csv(HOTSPOTS_CSV, index=False)


# Tiny Markdown
lines = [
"# CPI Log Insights",
f"- Source: {os.path.basename(DATA_PATH)}",
f"- Total rows: {len(df)}",
"
## Top 3 artifacts by error rate:",
]
for _, r in artifact_kpis[["ARTIFACT_NAME","events","error_rate","top_error_type","top_err_components"]].head(3).iterrows():
lines.append(f"- {r.ARTIFACT_NAME}: error_rate {r.error_rate:.3f}, top_error_type {r.top_error_type}, hotspots {r.top_err_components}")
with open(REPORT_MD, "w") as f:
f.write("
".join(lines))


print(f"[info] Saved model → {MODEL_PATH}")
print(f"[info] Saved metrics → {METRICS_JSON}")
print(f"[info] Saved insights → {INSIGHTS_CSV}")
print(f"[info] Saved hotspots → {HOTSPOTS_CSV}")
print(f"[info] Saved report → {REPORT_MD}")