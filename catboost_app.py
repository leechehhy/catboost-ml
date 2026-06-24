"""
CatBoost ML System — Flask 기반 단일 파일 앱
실행: python catboost_app.py
접속: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import io, json, os, tempfile, traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier, Pool

app = Flask(__name__)

# ── 전역 상태 ──────────────────────────────────────────────────────
_state = {
    "model": None,
    "le": None,           # target LabelEncoder
    "cat_features": [],   # 범주형 컬럼 인덱스
    "feature_names": [],
    "cat_col_names": [],
    "classes": [],
    "train_df": None,
    "pred_result_csv": None,
}

# ═══════════════════════════════════════════════════════════════════
# 전처리
# ═══════════════════════════════════════════════════════════════════
def auto_preprocess(df, target_col, feature_names=None):
    """결측값 처리 + 범주형 컬럼 탐지. catboost는 범주형을 직접 처리 가능."""
    df = df.copy()
    cols = feature_names if feature_names else [c for c in df.columns if c != target_col]

    # 결측값 처리
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].fillna("missing").astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    cat_col_names = [c for c in cols if df[c].dtype == object or df[c].nunique() <= 20 and df[c].dtype != float]
    # 수치형처럼 보여도 object면 cat으로
    cat_col_names = [c for c in cols if df[c].dtype == object]
    return df, cat_col_names

# ═══════════════════════════════════════════════════════════════════
# API 엔드포인트
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        f = request.files["file"]
        name = f.filename.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
        _state["train_df"] = df
        return jsonify({
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records"),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def train():
    try:
        body = request.json
        target = body["target"]
        features = body.get("features") or [c for c in _state["train_df"].columns if c != target]
        test_size = float(body.get("test_size", 0.2))
        params = body.get("params", {})

        df = _state["train_df"].copy()
        df, cat_col_names = auto_preprocess(df, target, features)

        X = df[features]
        y_raw = df[target].astype(str)

        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        cat_indices = [features.index(c) for c in cat_col_names if c in features]

        cb_params = {
            "iterations": int(params.get("iterations", 500)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "depth": int(params.get("depth", 6)),
            "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
            "random_seed": 42,
            "verbose": False,
            "eval_metric": "F1",
            "early_stopping_rounds": 50,
        }
        if len(le.classes_) > 2:
            cb_params["loss_function"] = "MultiClass"
            cb_params["eval_metric"] = "Accuracy"
        else:
            cb_params["loss_function"] = "Logloss"

        model = CatBoostClassifier(**cb_params)
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        eval_pool  = Pool(X_test,  y_test,  cat_features=cat_indices)
        model.fit(train_pool, eval_set=eval_pool, verbose=False)

        y_pred = model.predict(eval_pool).flatten()
        is_binary = len(le.classes_) == 2

        acc  = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec  = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1   = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        auc = None
        if is_binary and hasattr(model, "predict_proba"):
            proba = model.predict_proba(eval_pool)[:, 1]
            try: auc = float(roc_auc_score(y_test, proba))
            except: pass

        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importance
        fi_vals = model.get_feature_importance()
        fi = sorted(zip(features, fi_vals.tolist()), key=lambda x: -x[1])[:15]

        # Per-class (binary)
        per_class = {}
        if is_binary:
            for i, cls in enumerate(le.classes_):
                per_class[str(cls)] = {
                    "precision": float(precision_score(y_test, y_pred, pos_label=i, average="binary", zero_division=0)),
                    "recall":    float(recall_score(y_test, y_pred, pos_label=i, average="binary", zero_division=0)),
                    "f1":        float(f1_score(y_test, y_pred, pos_label=i, average="binary", zero_division=0)),
                }

        # 저장
        _state.update({
            "model": model,
            "le": le,
            "cat_features": cat_indices,
            "feature_names": features,
            "cat_col_names": cat_col_names,
            "classes": le.classes_.tolist(),
        })

        return jsonify({
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
            "classes": le.classes_.tolist(),
            "feature_importance": fi,
            "per_class": per_class,
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ else cb_params["iterations"],
            "n_train": len(X_train), "n_test": len(X_test),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if _state["model"] is None:
            return jsonify({"error": "먼저 모델을 학습해주세요."}), 400

        f = request.files["file"]
        name = f.filename.lower()
        if name.endswith(".csv"):
            df_new = pd.read_csv(f)
        else:
            df_new = pd.read_excel(f)

        model    = _state["model"]
        le       = _state["le"]
        features = _state["feature_names"]
        cat_idx  = _state["cat_features"]

        df_p, _ = auto_preprocess(df_new, None, features)

        # 누락 컬럼 체크
        missing = [c for c in features if c not in df_p.columns]
        if missing:
            return jsonify({"error": f"누락 컬럼: {missing}"}), 400

        X_new = df_p[features]
        pool_new = Pool(X_new, cat_features=cat_idx)

        preds_enc = model.predict(pool_new).flatten()
        preds = le.inverse_transform(preds_enc.astype(int))

        result = df_new.copy()
        result["예측결과"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(pool_new)
            classes = le.classes_.tolist()
            for i, cls in enumerate(classes):
                result[f"P({cls})"] = np.round(proba[:, i], 4)
            if len(classes) == 2:
                result["확률(양성)"] = [f"{v:.2f}%" for v in proba[:, 1] * 100]

        csv_str = result.to_csv(index=False, encoding="utf-8-sig")
        _state["pred_result_csv"] = csv_str

        # Summary
        counts = result["예측결과"].value_counts().to_dict()
        preview = result.head(100).to_dict(orient="records")
        headers = result.columns.tolist()

        return jsonify({
            "summary": {str(k): int(v) for k, v in counts.items()},
            "total": len(result),
            "headers": headers,
            "preview": preview,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/download")
def download():
    if not _state["pred_result_csv"]:
        return "결과 없음", 404
    buf = io.BytesIO(_state["pred_result_csv"].encode("utf-8-sig"))
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="catboost_predictions.csv")


# ═══════════════════════════════════════════════════════════════════
# HTML 프론트엔드
# ═══════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CatBoost ML Studio</title>
<style>
:root{--primary:#7B2FBE;--secondary:#5A1E9A;--success:#16A34A;--warning:#D97706;--danger:#DC2626;--bg:#F9F5FF;--surface:#FFF;--surface2:#F3EEFF;--border:#DDD0F5;--text:#1A0A2E;--muted:#6B5A8A;}
*{margin:0;padding:0;box-sizing:border-box;}
html.dark{--primary:#A855F7;--secondary:#C084FC;--success:#34D399;--bg:#0D0618;--surface:#180A28;--surface2:#220F38;--border:#3D2060;--text:#F3E8FF;--muted:#9D7FC5;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;}
header{background:linear-gradient(135deg,#4A0E8F,#7B2FBE);padding:14px 32px;display:flex;align-items:center;gap:14px;box-shadow:0 2px 12px rgba(123,47,190,.25);}
header h1{font-size:20px;font-weight:800;color:#fff;letter-spacing:-.3px;}
header .sub{color:rgba(255,255,255,.7);font-size:12px;margin-top:2px;}
.badge-cat{background:rgba(255,255,255,.15);color:#fff;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;border:1px solid rgba(255,255,255,.3);}
main{max-width:1100px;margin:0 auto;padding:28px 20px;display:flex;flex-direction:column;gap:22px;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px;box-shadow:0 2px 12px rgba(123,47,190,.07);}
.card-title{font-size:15px;font-weight:700;margin-bottom:18px;display:flex;align-items:center;gap:8px;}
.tag{font-size:11px;background:rgba(123,47,190,.12);color:var(--primary);padding:2px 9px;border-radius:20px;font-weight:600;}
.steps{display:flex;align-items:center;gap:0;margin-bottom:6px;flex-wrap:wrap;}
.step-item{display:flex;flex-direction:column;align-items:center;gap:4px;}
.step-circle{width:32px;height:32px;border-radius:50%;border:2px solid var(--border);background:var(--surface);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:var(--muted);transition:.3s;}
.step-circle.active{border-color:var(--primary);background:var(--primary);color:#fff;}
.step-circle.done{border-color:var(--success);background:var(--success);color:#fff;}
.step-lbl{font-size:11px;font-weight:600;color:var(--muted);white-space:nowrap;}
.step-lbl.active{color:var(--primary);}
.step-lbl.done{color:var(--success);}
.step-line{flex:1;height:2px;background:var(--border);min-width:30px;transition:.3s;}
.step-line.done{background:var(--success);}
.upload-zone{border:2px dashed var(--border);border-radius:12px;padding:36px;text-align:center;cursor:pointer;transition:.2s;background:var(--surface2);}
.upload-zone:hover{border-color:var(--primary);background:rgba(123,47,190,.04);}
.upload-zone .ico{font-size:40px;margin-bottom:8px;}
.upload-zone .ttl{font-size:14px;font-weight:600;margin-bottom:4px;}
.upload-zone .sub{font-size:12px;color:var(--muted);}
.hidden{display:none!important;}
.btn{padding:9px 20px;border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:.2s;}
.btn-primary{background:var(--primary);color:#fff;}
.btn-primary:hover{opacity:.88;}
.btn-primary:disabled{opacity:.45;cursor:not-allowed;}
.btn-success{background:var(--success);color:#fff;}
.btn-success:hover{opacity:.88;}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text);}
.btn-outline:hover{border-color:var(--primary);color:var(--primary);}
.btn-danger{background:var(--danger);color:#fff;padding:6px 11px;font-size:14px;font-weight:700;border-radius:8px;}
.alert{padding:10px 14px;border-radius:10px;font-size:13px;margin-bottom:0;}
.alert-success{background:rgba(22,163,74,.1);border:1px solid rgba(22,163,74,.25);color:#15803d;}
.alert-warning{background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.25);color:#92400e;}
.alert-danger{background:rgba(220,38,38,.1);border:1px solid rgba(220,38,38,.25);color:#b91c1c;}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px;margin-bottom:18px;}
.metric-card{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:14px;text-align:center;}
.metric-val{font-size:22px;font-weight:800;margin-bottom:4px;}
.metric-lbl{font-size:11px;color:var(--muted);font-weight:600;}
.c-good{color:var(--success);}
.c-mid{color:var(--warning);}
.c-bad{color:var(--danger);}
.tbl-wrap{overflow-x:auto;border-radius:10px;border:1px solid var(--border);}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{padding:8px 12px;background:var(--surface2);border-bottom:2px solid var(--border);text-align:left;white-space:nowrap;font-size:12px;}
td{padding:6px 12px;border-bottom:1px solid var(--border);}
tr:hover td{background:rgba(123,47,190,.03);}
.slider{width:100%;accent-color:var(--primary);}
.param-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px;}
@media(max-width:600px){.param-row{grid-template-columns:1fr;}}
.param-lbl{font-size:12px;font-weight:600;margin-bottom:4px;color:var(--text);}
.fi-bar-wrap{display:flex;align-items:center;gap:10px;margin-bottom:7px;}
.fi-name{width:130px;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600;}
.fi-bar{flex:1;background:var(--surface2);border-radius:4px;height:20px;overflow:hidden;border:1px solid var(--border);}
.fi-fill{height:100%;background:linear-gradient(90deg,#7B2FBE,#A855F7);border-radius:3px;transition:width .4s;}
.fi-val{min-width:44px;font-size:12px;font-weight:600;color:var(--primary);text-align:right;}
.spin{display:inline-block;width:16px;height:16px;border:2px solid rgba(123,47,190,.3);border-top-color:var(--primary);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;}
@keyframes spin{to{transform:rotate(360deg)}}
.dark-toggle{margin-left:auto;background:rgba(255,255,255,.15);border:none;color:#fff;padding:6px 12px;border-radius:8px;cursor:pointer;font-size:12px;}
.feat-chip{display:inline-flex;align-items:center;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--surface);color:var(--muted);transition:.15s;user-select:none;}
.feat-chip.on{background:rgba(123,47,190,.12);color:var(--primary);border-color:var(--primary);}
.feat-chip:hover{border-color:var(--primary);}
select{padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px;}
.chip{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(123,47,190,.1);color:var(--primary);border:1px solid rgba(123,47,190,.2);}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
@media(max-width:700px){.two-col{grid-template-columns:1fr;}}
.per-class-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:12px;}
.pcchip{padding:2px 8px;border-radius:6px;font-weight:600;}
</style>
</head>
<body>
<header>
  <div>
    <h1>🐱 CatBoost ML Studio</h1>
    <div class="sub">CatBoost 전용 머신러닝 분류 시스템</div>
  </div>
  <span class="badge-cat">CatBoost</span>
  <button class="dark-toggle" onclick="toggleDark()">🌙 다크모드</button>
</header>

<div style="max-width:1100px;margin:20px auto 0;padding:0 20px;">
  <div class="steps">
    <div class="step-item"><div class="step-circle active" id="sc1">1</div><div class="step-lbl active" id="sl1">데이터 업로드</div></div>
    <div class="step-line" id="ln1"></div>
    <div class="step-item"><div class="step-circle" id="sc2">2</div><div class="step-lbl" id="sl2">파라미터 설정</div></div>
    <div class="step-line" id="ln2"></div>
    <div class="step-item"><div class="step-circle" id="sc3">3</div><div class="step-lbl" id="sl3">학습 & 평가</div></div>
    <div class="step-line" id="ln3"></div>
    <div class="step-item"><div class="step-circle" id="sc4">4</div><div class="step-lbl" id="sl4">예측 실행</div></div>
  </div>
</div>

<main>

<!-- STEP 1 -->
<div id="step1">
  <div class="card">
    <div class="card-title">📂 학습 데이터 업로드 <span class="tag">STEP 1</span></div>
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="ico">📊</div>
      <div class="ttl">파일을 클릭하여 업로드</div>
      <div class="sub" style="margin-bottom:10px;">라벨(정답 컬럼) 포함 학습 데이터셋</div>
      <span style="font-size:12px;background:rgba(123,47,190,.12);color:var(--primary);padding:3px 10px;border-radius:8px;font-weight:600;">.csv &nbsp;/&nbsp; .xlsx</span>
    </div>
    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" class="hidden">
    <div id="fileInfo" class="hidden" style="margin-top:14px;"></div>
    <div id="previewSection" class="hidden" style="margin-top:18px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
        <div style="font-size:13px;font-weight:600;">데이터 미리보기 (상위 5행)</div>
        <div style="display:flex;align-items:center;gap:10px;">
          <label style="font-size:12px;color:var(--muted);">예측 대상 (Target):</label>
          <select id="targetSelect" onchange="onTargetChange()"></select>
        </div>
      </div>
      <div class="tbl-wrap" id="previewTable"></div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;" id="infoChips"></div>

      <!-- Feature 선택 -->
      <div style="margin-top:18px;padding:14px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
          <div style="font-size:13px;font-weight:700;">🎛️ 사용할 Feature 선택</div>
          <div style="display:flex;gap:6px;">
            <button onclick="featSelectAll()" style="font-size:11px;padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;">전체 선택</button>
            <button onclick="featDeselectAll()" style="font-size:11px;padding:4px 10px;border-radius:7px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;">전체 해제</button>
          </div>
        </div>
        <div id="featChips" style="display:flex;flex-wrap:wrap;gap:7px;"></div>
        <div id="featWarn" class="hidden" style="margin-top:8px;font-size:12px;color:var(--danger);">⚠ 최소 1개 이상 선택해야 합니다.</div>
        <div style="margin-top:10px;font-size:11px;color:var(--muted);" id="featCount"></div>
      </div>

      <div style="margin-top:18px;">
        <button class="btn btn-primary" onclick="goStep2()">파라미터 설정하기 →</button>
      </div>
    </div>
  </div>
</div>

<!-- STEP 2 -->
<div id="step2" class="hidden">
  <div class="card">
    <div class="card-title">⚙️ CatBoost 파라미터 설정 <span class="tag">STEP 2</span></div>
    <div style="margin-bottom:14px;padding:12px 16px;background:rgba(123,47,190,.06);border-radius:10px;border:1px solid rgba(123,47,190,.15);font-size:12px;color:var(--muted);line-height:1.8;">
      <strong style="color:var(--primary);">CatBoost 특징</strong> — 범주형 데이터를 별도 인코딩 없이 자동 처리합니다. 과적합에 강하며 하이퍼파라미터 민감도가 낮아 기본값으로도 좋은 성능을 냅니다.
    </div>
    <div class="param-row">
      <div>
        <div class="param-lbl">🔄 Iterations (트리 개수): <span id="pv-iterations">500</span></div>
        <input type="range" class="slider" id="ps-iterations" min="100" max="2000" step="50" value="500"
          oninput="document.getElementById('pv-iterations').textContent=this.value">
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:2px;"><span>100</span><span>2000</span></div>
      </div>
      <div>
        <div class="param-lbl">📐 Learning Rate: <span id="pv-lr">0.05</span></div>
        <input type="range" class="slider" id="ps-lr" min="1" max="30" step="1" value="5"
          oninput="document.getElementById('pv-lr').textContent=(this.value/100).toFixed(2)">
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:2px;"><span>0.01</span><span>0.30</span></div>
      </div>
      <div>
        <div class="param-lbl">🌳 Depth (트리 깊이): <span id="pv-depth">6</span></div>
        <input type="range" class="slider" id="ps-depth" min="2" max="12" step="1" value="6"
          oninput="document.getElementById('pv-depth').textContent=this.value">
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:2px;"><span>2</span><span>12</span></div>
      </div>
      <div>
        <div class="param-lbl">🛡️ L2 Regularization: <span id="pv-l2">3.0</span></div>
        <input type="range" class="slider" id="ps-l2" min="1" max="20" step="1" value="3"
          oninput="document.getElementById('pv-l2').textContent=this.value+'.0'">
        <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:2px;"><span>1</span><span>20</span></div>
      </div>
    </div>
    <div style="margin-top:16px;padding:12px 16px;background:rgba(123,47,190,.06);border-radius:10px;border:1px solid rgba(123,47,190,.15);">
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:12px;font-weight:600;white-space:nowrap;">✂️ 훈련/테스트 분할</span>
        <input type="range" class="slider" id="splitSlider" min="60" max="90" step="5" value="80"
          oninput="updateSplitLabel(this.value)" style="flex:1;">
        <div id="splitLabel" style="font-size:12px;font-weight:700;color:var(--secondary);min-width:140px;text-align:right;">훈련 80% / 테스트 20%</div>
      </div>
    </div>
    <div style="margin-top:18px;display:flex;gap:10px;align-items:center;">
      <button class="btn btn-outline" onclick="setStep(1)">← 데이터 다시 선택</button>
      <button class="btn btn-primary" onclick="runTrain()">🚀 CatBoost 학습 시작</button>
      <div id="trainSpinner" class="hidden" style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);"><div class="spin"></div> 학습 중... (Early Stopping 포함)</div>
    </div>
  </div>
</div>

<!-- STEP 3 -->
<div id="step3" class="hidden">
  <div class="card">
    <div class="card-title">📊 학습 결과 & 성능 평가 <span class="tag">STEP 3</span></div>
    <div id="trainInfo" style="margin-bottom:16px;font-size:12px;color:var(--muted);"></div>

    <div class="metric-grid" id="metricsGrid"></div>

    <div class="two-col">
      <!-- 혼동 행렬 -->
      <div>
        <div style="font-size:13px;font-weight:600;margin-bottom:10px;">🔲 혼동 행렬 (Confusion Matrix)</div>
        <div id="cmWrap" style="overflow-x:auto;"></div>
        <div id="perClassSection" style="margin-top:12px;"></div>
      </div>
      <!-- Feature importance -->
      <div>
        <div style="font-size:13px;font-weight:600;margin-bottom:10px;">📊 Feature Importance (상위 15개)</div>
        <div id="fiChart"></div>
      </div>
    </div>

    <div style="margin-top:20px;display:flex;gap:10px;flex-wrap:wrap;">
      <button class="btn btn-outline" onclick="setStep(2)">← 파라미터 수정</button>
      <button class="btn btn-primary" onclick="setStep(4)">예측 실행하기 →</button>
    </div>
  </div>
</div>

<!-- STEP 4 -->
<div id="step4" class="hidden">
  <div class="card">
    <div class="card-title">🔮 새 데이터 예측 <span class="tag">STEP 4</span></div>
    <div class="alert alert-warning" style="margin-bottom:16px;">
      ⚠️ 예측 데이터는 학습 데이터와 동일한 컬럼 구조여야 합니다 (Target 컬럼 제외 가능)
    </div>
    <div class="two-col">
      <div>
        <div class="upload-zone" id="predZone" onclick="document.getElementById('predInput').click()" style="padding:28px;">
          <div class="ico" style="font-size:36px;">📤</div>
          <div class="ttl" style="font-size:13px;">예측할 데이터 업로드</div>
          <div class="sub">.csv / .xlsx</div>
        </div>
        <input type="file" id="predInput" accept=".csv,.xlsx,.xls" class="hidden">
        <div id="predFileInfo" class="hidden" style="margin-top:11px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <div class="alert alert-success" id="predFileMsg" style="flex:1;margin:0;"></div>
            <button class="btn-danger" onclick="clearPredFile()" title="파일 제거">✕</button>
          </div>
        </div>
        <div style="margin-top:14px;display:flex;align-items:center;gap:10px;">
          <button class="btn btn-primary" id="predictBtn" disabled onclick="runPredict()">🔮 예측 실행</button>
          <div id="predictSpinner" class="hidden" style="display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px;"><div class="spin"></div> 예측 중...</div>
        </div>
      </div>
      <div id="predSummary" class="hidden">
        <div style="font-size:13px;font-weight:600;margin-bottom:10px;">예측 요약</div>
        <div id="predSummaryContent"></div>
      </div>
    </div>
    <div id="predResults" class="hidden" style="margin-top:22px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:11px;">
        <div style="font-size:13px;font-weight:600;">예측 결과 미리보기</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button class="btn btn-success" onclick="downloadResults()">⬇️ CSV 다운로드</button>
          <button class="btn btn-outline" onclick="resetPredict()">🔄 다른 데이터 예측하기</button>
        </div>
      </div>
      <div class="tbl-wrap" id="predTable"></div>
    </div>
    <div style="margin-top:18px;">
      <button class="btn btn-outline" onclick="setStep(3)">← 학습 결과 보기</button>
    </div>
  </div>
</div>

</main>

<script>
// ── 상태 ──────────────────────────────────────────────────────────
let allHeaders=[], targetCol='', trainDone=false, predFile=null;
let selectedFeatures=new Set();

// ── 다크모드 ─────────────────────────────────────────────────────
function toggleDark(){document.documentElement.classList.toggle('dark');}

// ── Step 네비 ────────────────────────────────────────────────────
function setStep(n){
  [1,2,3,4].forEach(i=>{
    document.getElementById('step'+i).classList.toggle('hidden',i!==n);
    const sc=document.getElementById('sc'+i), sl=document.getElementById('sl'+i);
    sc.className='step-circle'+(i<n?' done':i===n?' active':'');
    sl.className='step-lbl'+(i<n?' done':i===n?' active':'');
    sc.textContent=i<n?'✓':i;
  });
  [1,2,3].forEach(i=>document.getElementById('ln'+i).className='step-line'+(i<n?' done':''));
}

function updateSplitLabel(v){
  document.getElementById('splitLabel').textContent=`훈련 ${v}% / 테스트 ${100-v}%`;
}

// ── STEP 1: 파일 업로드 ──────────────────────────────────────────
document.getElementById('fileInput').addEventListener('change',e=>{
  if(e.target.files[0]) handleUpload(e.target.files[0]);
});
const uz=document.getElementById('uploadZone');
uz.addEventListener('dragover',e=>{e.preventDefault();uz.style.borderColor='var(--primary)';});
uz.addEventListener('dragleave',()=>uz.style.borderColor='');
uz.addEventListener('drop',e=>{e.preventDefault();uz.style.borderColor='';if(e.dataTransfer.files[0])handleUpload(e.dataTransfer.files[0]);});

async function handleUpload(file){
  const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){alert('오류: '+d.error);return;}
    allHeaders=d.columns;
    // target select
    const sel=document.getElementById('targetSelect');
    sel.innerHTML=d.columns.map(c=>`<option value="${c}">${c}</option>`).join('');
    // 마지막 컬럼을 기본 target으로
    sel.value=d.columns[d.columns.length-1];
    targetCol=sel.value;
    // preview
    buildPreviewTable(d.columns, d.preview);
    buildFeatChips();
    // info
    const fi=document.getElementById('fileInfo');
    fi.classList.remove('hidden');
    fi.innerHTML=`<div class="alert alert-success">✅ <strong>${file.name}</strong> — ${d.rows.toLocaleString()}행, ${d.columns.length}컬럼 로드 완료</div>`;
    document.getElementById('infoChips').innerHTML=
      `<span class="chip">📊 ${d.rows.toLocaleString()} 행</span>`+
      `<span class="chip">📋 ${d.columns.length} 컬럼</span>`+
      `<span class="chip">🎯 Target: ${targetCol}</span>`;
    document.getElementById('previewSection').classList.remove('hidden');
  }catch(e){alert('업로드 실패: '+e.message);}
}

function buildPreviewTable(cols, rows){
  let tbl=`<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(row=>{
    tbl+=`<tr>${cols.map(c=>`<td>${row[c]??''}</td>`).join('')}</tr>`;
  });
  tbl+=`</tbody></table>`;
  document.getElementById('previewTable').innerHTML=tbl;
}

function onTargetChange(){
  targetCol=document.getElementById('targetSelect').value;
  document.getElementById('infoChips').querySelector('.chip:last-child').textContent='🎯 Target: '+targetCol;
  buildFeatChips();
}

function buildFeatChips(){
  const feats=allHeaders.filter(h=>h!==targetCol);
  selectedFeatures=new Set(feats);
  const area=document.getElementById('featChips');
  area.innerHTML=feats.map(f=>
    `<span class="feat-chip on" data-feat="${f}" onclick="toggleFeat(this)">${f}</span>`
  ).join('');
  updateFeatCount();
}

function toggleFeat(el){
  const f=el.dataset.feat;
  if(selectedFeatures.has(f)){
    selectedFeatures.delete(f);
    el.classList.remove('on');
  } else {
    selectedFeatures.add(f);
    el.classList.add('on');
  }
  updateFeatCount();
}

function featSelectAll(){
  allHeaders.filter(h=>h!==targetCol).forEach(f=>selectedFeatures.add(f));
  document.querySelectorAll('.feat-chip').forEach(el=>el.classList.add('on'));
  updateFeatCount();
}

function featDeselectAll(){
  selectedFeatures.clear();
  document.querySelectorAll('.feat-chip').forEach(el=>el.classList.remove('on'));
  updateFeatCount();
}

function updateFeatCount(){
  const n=selectedFeatures.size;
  document.getElementById('featCount').textContent=`선택된 feature: ${n}개`;
  document.getElementById('featWarn').classList.toggle('hidden', n>0);
}

function goStep2(){
  if(!allHeaders.length){alert('데이터를 먼저 업로드해주세요.');return;}
  if(selectedFeatures.size===0){document.getElementById('featWarn').classList.remove('hidden');return;}
  setStep(2);
}

// ── STEP 2: 학습 ─────────────────────────────────────────────────
async function runTrain(){
  const iterations=parseInt(document.getElementById('ps-iterations').value);
  const lr=parseInt(document.getElementById('ps-lr').value)/100;
  const depth=parseInt(document.getElementById('ps-depth').value);
  const l2=parseInt(document.getElementById('ps-l2').value);
  const testSize=(100-parseInt(document.getElementById('splitSlider').value))/100;
  const features=Array.from(selectedFeatures);

  const btn=document.querySelector('#step2 .btn-primary:last-of-type');
  const spin=document.getElementById('trainSpinner');
  btn.disabled=true; spin.classList.remove('hidden'); spin.style.display='flex';

  try{
    const r=await fetch('/api/train',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({target:targetCol,features,test_size:testSize,
        params:{iterations,learning_rate:lr,depth,l2_leaf_reg:l2}})
    });
    const d=await r.json();
    if(d.error){alert('학습 오류: '+d.error);return;}
    renderResults(d);
    trainDone=true;
    setStep(3);
  }catch(e){alert('오류: '+e.message);}
  finally{btn.disabled=false;spin.classList.add('hidden');}
}

// ── STEP 3: 결과 렌더링 ──────────────────────────────────────────
function cClass(v){return v>=0.8?'c-good':v>=0.6?'c-mid':'c-bad';}

function renderResults(d){
  // 학습 정보
  document.getElementById('trainInfo').innerHTML=
    `✅ 학습 완료 &nbsp;|&nbsp; 훈련 <strong>${d.n_train.toLocaleString()}</strong>건 / 테스트 <strong>${d.n_test.toLocaleString()}</strong>건 &nbsp;|&nbsp; 최적 반복 횟수: <strong>${d.best_iteration}</strong>`;

  // 메트릭
  const ms=[{k:'accuracy',l:'Accuracy'},{k:'precision',l:'Precision'},{k:'recall',l:'Recall'},{k:'f1',l:'F1 Score'}];
  if(d.auc!=null) ms.push({k:'auc',l:'AUC-ROC'});
  document.getElementById('metricsGrid').innerHTML=ms.map(m=>{
    const v=d[m.k]; const pct=v!=null?(v*100).toFixed(1)+'%':'—';
    return `<div class="metric-card"><div class="metric-val ${v!=null?cClass(v):''}">${pct}</div><div class="metric-lbl">${m.l}</div></div>`;
  }).join('');

  // 혼동 행렬
  const cm=d.confusion_matrix, labels=d.classes;
  let tbl=`<table style="border-collapse:collapse;font-size:13px;margin:0 auto;"><tr><td></td>${labels.map(l=>`<td style="padding:6px 12px;font-weight:700;text-align:center;color:var(--muted);font-size:11px;">예측: ${l}</td>`).join('')}</tr>`;
  cm.forEach((row,i)=>{
    tbl+=`<tr><td style="padding:6px 10px;font-weight:700;color:var(--muted);font-size:11px;white-space:nowrap;">실제: ${labels[i]}</td>`;
    row.forEach((v,j)=>{
      const bg=i===j?'rgba(22,163,74,.15)':'rgba(220,38,38,.07)';
      tbl+=`<td style="padding:10px 18px;text-align:center;background:${bg};border:1px solid var(--border);font-weight:${i===j?'700':'400'};font-size:16px;">${v}</td>`;
    });
    tbl+=`</tr>`;
  });
  document.getElementById('cmWrap').innerHTML=tbl+'</table>';

  // 클래스별 성능
  if(d.per_class){
    document.getElementById('perClassSection').innerHTML=
      `<div style="font-size:12px;font-weight:600;margin-bottom:8px;">클래스별 상세 성능</div>`+
      Object.entries(d.per_class).map(([cls,m])=>
        `<div class="per-class-row">
          <span style="min-width:50px;font-weight:700;">${cls}</span>
          <span class="pcchip" style="background:rgba(123,47,190,.1);color:var(--primary);">P: ${(m.precision*100).toFixed(1)}%</span>
          <span class="pcchip" style="background:rgba(22,163,74,.1);color:var(--success);">R: ${(m.recall*100).toFixed(1)}%</span>
          <span class="pcchip" style="background:rgba(74,108,247,.1);color:#4a6cf7;">F1: ${(m.f1*100).toFixed(1)}%</span>
        </div>`
      ).join('');
  }

  // Feature importance
  const fi=d.feature_importance;
  if(fi&&fi.length){
    const maxV=fi[0][1]||1;
    document.getElementById('fiChart').innerHTML=fi.map(([name,val])=>{
      const barPct=(val/maxV*100).toFixed(1);
      return `<div class="fi-bar-wrap">
        <div class="fi-name" title="${name}">${name}</div>
        <div class="fi-bar"><div class="fi-fill" style="width:${barPct}%"></div></div>
        <div class="fi-val">${(val).toFixed(2)}%</div>
      </div>`;
    }).join('');
  }
}

// ── STEP 4: 예측 ─────────────────────────────────────────────────
document.getElementById('predInput').addEventListener('change',e=>{
  if(e.target.files[0]) handlePredFile(e.target.files[0]);
});

function handlePredFile(file){
  predFile=file;
  document.getElementById('predFileInfo').classList.remove('hidden');
  document.getElementById('predFileMsg').innerHTML=`✅ <strong>${file.name}</strong> 업로드됨`;
  document.getElementById('predictBtn').disabled=false;
}

function clearPredFile(){
  predFile=null;
  document.getElementById('predInput').value='';
  document.getElementById('predFileInfo').classList.add('hidden');
  document.getElementById('predictBtn').disabled=true;
  document.getElementById('predResults').classList.add('hidden');
  document.getElementById('predSummary').classList.add('hidden');
}

function resetPredict(){
  clearPredFile();
  const z=document.getElementById('predZone');
  z.style.borderColor='var(--primary)';
  setTimeout(()=>z.style.borderColor='',1200);
}

async function runPredict(){
  if(!predFile){alert('예측 파일을 업로드해주세요.');return;}
  const btn=document.getElementById('predictBtn');
  const spin=document.getElementById('predictSpinner');
  btn.disabled=true; spin.classList.remove('hidden'); spin.style.display='flex';
  try{
    const fd=new FormData(); fd.append('file',predFile);
    const r=await fetch('/api/predict',{method:'POST',body:fd});
    const d=await r.json();
    if(d.error){alert('예측 오류: '+d.error);return;}
    // Summary
    document.getElementById('predSummary').classList.remove('hidden');
    document.getElementById('predSummaryContent').innerHTML=
      Object.entries(d.summary).map(([k,v])=>
        `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;">
          <span style="font-weight:600;">${k}</span>
          <span style="color:var(--primary);font-weight:700;">${v.toLocaleString()}명 (${(v/d.total*100).toFixed(1)}%)</span>
        </div>`
      ).join('')+
      `<div style="margin-top:8px;font-size:11px;color:var(--muted);">총 ${d.total.toLocaleString()}명 예측 완료</div>`;
    // Table
    const hdrs=d.headers, rows=d.preview;
    const predCol=hdrs.indexOf('예측결과');
    let tbl=`<table><thead><tr>${hdrs.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>`;
    rows.forEach(row=>{
      const pv=predCol>=0?row['예측결과']:'';
      const bg=pv==='이직'?'rgba(220,38,38,.04)':pv==='재직'?'rgba(22,163,74,.03)':'';
      tbl+=`<tr style="background:${bg}">${hdrs.map(h=>{
        const v=row[h]??'';
        let style='';
        if(h==='예측결과') style=`font-weight:700;color:${pv==='이직'?'var(--danger)':pv==='재직'?'var(--success)':'var(--text)'}`;
        return `<td style="${style}">${v}</td>`;
      }).join('')}</tr>`;
    });
    tbl+=`</tbody></table>`;
    if(d.total>100) tbl+=`<div style="text-align:center;padding:8px;font-size:12px;color:var(--muted);">… 상위 100행만 표시 (전체 ${d.total.toLocaleString()}행)</div>`;
    document.getElementById('predTable').innerHTML=tbl;
    document.getElementById('predResults').classList.remove('hidden');
  }catch(e){alert('예측 오류: '+e.message);}
  finally{btn.disabled=false;spin.classList.add('hidden');}
}

function downloadResults(){
  window.location.href='/api/download';
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML

if __name__ == "__main__":
    print("=" * 55)
    print("  🐱 CatBoost ML Studio 시작")
    print("  브라우저에서 열기: http://localhost:5000")
    print("  종료: Ctrl+C")
    print("=" * 55)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
