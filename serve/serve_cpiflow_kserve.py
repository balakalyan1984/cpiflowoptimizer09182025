import os, tarfile, tempfile, logging
out = {
"artifact": name,
"known_artifact": _is_known_artifact(name),
"prediction": top[0]["label"],
"top_probs": top
}
if ART_KPIS.get(name):
out["artifact_kpis"] = ART_KPIS[name]
return out


# ---------- Endpoints ----------
@app.get("/healthz")
def healthz(): return {"status":"ok"}


@app.get("/v2/greet")
def greet():
return {
"status": "ok",
"mount_dir": MOUNT_DIR,
"env_model_path": ENV_MODEL_PATH or None,
"dir_listing": _listdir(MOUNT_DIR),
"has_meta": bool(ART_KPIS) or bool(GLOBAL_HOTSPOTS)
}


@app.post("/v2/predict")
def predict(req: PredictRequest):
try:
records = [{
"ARTIFACT_NAME": (x.ARTIFACT_NAME or ""),
"ORIGIN_COMPONENT_NAME": (x.ORIGIN_COMPONENT_NAME or ""),
"LOG_LEVEL": (x.LOG_LEVEL or ""),
} for x in req.instances]
preds = pipe.predict(records)
return {"predictions": [str(p) for p in preds]}
except Exception as e:
log.exception("Prediction failed")
raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.post("/v2/analyze")
def analyze(req: AnalyzeRequest):
try:
name = (req.artifact_name or "").strip()
if not name:
raise HTTPException(status_code=422, detail="artifact_name must be non-empty")
resp = _analyze_one(name, top_k=req.top_k or 4)
if GLOBAL_HOTSPOTS:
resp["global_hotspots"] = GLOBAL_HOTSPOTS
return resp
except HTTPException:
raise
except Exception as e:
log.exception("Analyze failed")
raise HTTPException(status_code=400, detail=f"Analyze failed: {e}")


@app.get("/v2/analyze_all")
def analyze_all(
top_k: int = Query(4, ge=1, le=10),
sort: str = Query("alpha", pattern="^(alpha|error_rate)$")
):
try:
names = _known_artifacts()
results = [_analyze_one(n, top_k=top_k) for n in names]
if sort == "error_rate" and ART_KPIS:
results.sort(key=lambda r: ART_KPIS.get(r["artifact"], {}).get("error_rate", 0), reverse=True)
else:
results.sort(key=lambda r: r["artifact"])
resp = {"count": len(results), "results": results}
if GLOBAL_HOTSPOTS:
resp["global_hotspots"] = GLOBAL_HOTSPOTS
return resp
except Exception as e:
log.exception("Analyze_all failed")
raise HTTPException(status_code=400, detail=f"Analyze_all failed: {e}")


@app.post("/v2/analyze_many")
def analyze_many(req: AnalyzeManyRequest):
try:
names = [n.strip() for n in (req.artifact_names or []) if n and n.strip()]
if not names:
raise HTTPException(status_code=422, detail="artifact_names must be a non-empty list")
results = [_analyze_one(n, top_k=req.top_k or 4) for n in names]
return { "count": len(results), "results": results }
except HTTPException:
raise
except Exception as e:
log.exception("Analyze_many failed")
raise HTTPException(status_code=400, detail=f"Analyze_many failed: {e}")