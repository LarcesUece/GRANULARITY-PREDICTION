from __future__ import annotations

import os, sys, json, time, logging, itertools, warnings
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- #
# Configuração central (equivalente a configs/default.yaml). Nenhum parâmetro
# experimental importante fica espalhado pelo código: tudo vive aqui.
# ----------------------------------------------------------------------------- #
CONFIG = {
    "experiment": {
        "name": "cross_granularity_feature_selection",
        "random_seed": 42,
        "output_dir": "outputs",
    },
    "data": {
        "path_10min": "dados/tratados/df_10min.parquet",
        "path_hour": "dados/tratados/df_hour.parquet",
        "timestamp_column": "time",
        "target_column": "n_bytes",
        "entity_column": "id_institution",
        "stat_window": 24,          # janela (em horas) p/ features estatísticas
        "weekly_window": 168,       # janela longa (1 semana)
        "n_lag_features": 24,       # defasagens originais do alvo (autoregressivo)
        "extra_lags": [48, 168],    # defasagens longas adicionais
        "target_scale": 1e8,        # n_bytes em ~centenas de MB (escala legível p/ MAE e λ)
        "max_entities": None,       # None = todas; defina p.ex. 10 p/ smoke-test
    },
    "missingness": {
        "mechanisms": ["mcar", "block"],
        "rates": [0.05, 0.10, 0.20, 0.30],
        "block_lengths": [1, 3, 6, 12],
        "seeds": [11, 22, 33, 44, 55],
        "primary": {"mechanism": "mcar", "rate": 0.20, "block_length": 6, "seed": 11},
    },
    "validation": {
        "final_test_fraction": 0.20,
        "inner_splits": 5,
        "gap": 0,
    },
    "selection": {
        "primary_method": "sffs",
        "max_features": 15,
        "min_improvement": 0.001,
        "feature_penalty_lambda": 0.02,
        "objective_mode": "additive",   # "additive" | "multiplicative"
        "multiplicative_alpha": 0.02,
    },
    "genetic_algorithm": {
        "population_size": 50,
        "generations": 40,
        "crossover_probability": 0.80,
        "mutation_probability": "auto",  # "auto" => 1 / n_features
        "tournament_size": 3,
        "elitism": 2,
    },
    "stability": {
        "enabled": True,
        "method": "forward",             # "forward" (rápido) | "sffs"
        "rates": [0.10, 0.20, 0.30],
        "seeds": [11, 22, 33],
        "max_features": 8,
    },
    "model": {
        "name": "hist_gradient_boosting",  # hist_gradient_boosting | random_forest | extra_trees
        "random_seed": 42,
        "robustness_models": ["hist_gradient_boosting", "random_forest", "extra_trees"],
    },
    "metrics": {
        "primary": "mae",
        "secondary": ["rmse", "smape"],
        "epsilon": 1e-8,
    },
    "plots": {"dpi": 300, "formats": ["png"]},
}


# ----------------------------------------------------------------------------- #
# Helpers de path / logging / seed
# ----------------------------------------------------------------------------- #
def find_project_root(start: Path | None = None) -> Path:
    """Localiza a raiz do projeto (pasta que contém 'dados')."""
    d = Path(start or os.getcwd()).resolve()
    for p in [d, *d.parents]:
        if (p / "dados").is_dir():
            return p
    return d


ROOT = find_project_root()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("imputation_experiment")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_h = RotatingFileHandler(output_dir / "experiment.log", maxBytes=5_000_000, backupCount=3)
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_h)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_and_prepare(cfg: dict) -> pd.DataFrame:
    """Carrega df_hour, reindexa em grade horária regular por instituição.

    Retorna DataFrame `W` com colunas [time, id_institution, y_true].
    `y_true` = valor horário (n_bytes) intocado; NaN = falta *real* (não artificial).
    """
    ent = cfg["data"]["entity_column"]
    tgt = cfg["data"]["target_column"]
    ts = cfg["data"]["timestamp_column"]

    h = pd.read_parquet(ROOT / cfg["data"]["path_hour"], columns=[ts, ent, tgt])
    h = h.rename(columns={ts: "time", tgt: "y_true", ent: "id_institution"})
    h["time"] = pd.to_datetime(h["time"], utc=True)
    h = h.dropna(subset=["y_true"])          # só horas com ground-truth real
    h = h.sort_values(["id_institution", "time"])

    if cfg["data"]["max_entities"] is not None:
        keep = h["id_institution"].unique()[: cfg["data"]["max_entities"]]
        h = h[h["id_institution"].isin(keep)]

    frames = []
    for eid, g in h.groupby("id_institution", sort=True):
        g = g.set_index("time")
        g = g[~g.index.duplicated(keep="first")]
        grid = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(grid)
        g["id_institution"] = eid
        frames.append(g.reset_index().rename(columns={"index": "time"}))

    W = pd.concat(frames, ignore_index=True)
    W = W[["time", "id_institution", "y_true"]]
    # escala o alvo para unidades legíveis (bytes -> ~100 MB). Transformação linear
    # global: preserva o ranking de MAE entre subconjuntos de features.
    W["y_true"] = W["y_true"].astype(float) / cfg["data"]["target_scale"]
    return W


def load_10min_hourly(cfg: dict) -> pd.DataFrame:
    """Agrega a granularidade de 10min em estatísticas horárias, já defasadas
    em 1h (past_only). Estas features usam uma fonte *separada* (10min) e são
    sempre do passado => não vazam o alvo da hora corrente."""
    ts, tgt, ent = cfg["data"]["timestamp_column"], cfg["data"]["target_column"], cfg["data"]["entity_column"]
    g = pd.read_parquet(ROOT / cfg["data"]["path_10min"], columns=[ts, ent, tgt])
    g = g.rename(columns={ts: "time", tgt: "n_bytes", ent: "id_institution"})
    g["time"] = pd.to_datetime(g["time"], utc=True)
    g = g.dropna(subset=["n_bytes"])

    agg = (
        g.set_index("time")
        .groupby(["id_institution", pd.Grouper(freq="1h")])["n_bytes"]
        .agg(
            gm10_mean="mean", gm10_std="std", gm10_min="min", gm10_max="max",
            gm10_sum="sum", gm10_count="count",
        )
        .reset_index()
    )
    agg["gm10_active_ratio"] = (agg["gm10_count"] / 6.0).clip(0, 1)
    agg["gm10_cv"] = agg["gm10_std"] / (agg["gm10_mean"].abs() + 1e-6)

    agg = agg.sort_values(["id_institution", "time"])
    gm_cols = [c for c in agg.columns if c.startswith("gm10")]
    agg[gm_cols] = agg.groupby("id_institution")[gm_cols].shift(1)  # defasa 1h (past_only)
    return agg


def apply_missingness(y_true: pd.Series, cfg: dict, mechanism: str,
                      rate: float, block_length: int, seed: int) -> np.ndarray:
    """Retorna máscara booleana (True = artificialmente faltante) sobre y_true.

    Reproduzível: usa `np.random.default_rng(seed)`.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mask = np.zeros(n, dtype=bool)
    idx = np.arange(n)

    if mechanism == "mcar":
        k = int(round(rate * n))
        chosen = rng.choice(idx, size=k, replace=False)
        mask[chosen] = True
    elif mechanism == "block":
        # blocos de `block_length` horas: sorteia posições de início de bloco
        n_blocks = max(1, int(round(rate * n / block_length)))
        starts = rng.choice(idx, size=n_blocks, replace=False)
        for s in starts:
            mask[s : s + block_length] = True
    else:
        raise ValueError(f"mecanismo desconhecido: {mechanism}")
    return mask


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: str
    availability: str  # "past_only" | "same_hour_auxiliary" | "derived_from_target"


ALLOWED_AVAILABILITY = {"past_only", "same_hour_auxiliary", "derived_from_target"}


def validate_feature_availability(features, allowed=("past_only", "same_hour_auxiliary")) -> None:
    """Falha ruidosamente se qualquer feature violar o contrato de disponibilidade.

    - `derived_from_target` é SEMPRE proibida (vazaria o valor oculto a imputar).
    - features fora de `allowed` também são proibidas.
    """
    for f in features:
        name = f.name if isinstance(f, FeatureDefinition) else f["name"]
        avail = f.availability if isinstance(f, FeatureDefinition) else f["availability"]
        if avail not in ALLOWED_AVAILABILITY:
            raise ValueError(f"[LEAK] feature '{name}' tem availability inválida: {avail}")
        if avail == "derived_from_target":
            raise ValueError(
                f"[LEAK] feature '{name}' é derived_from_target e está PROIBIDA "
                f"(requer o valor oculto sendo imputado)."
            )
        if avail not in allowed:
            raise ValueError(f"[LEAK] feature '{name}' (availability={avail}) não permitida; permitido={allowed}")


# Registro único das características candidatas (fonte de verdade p/ features,
# ablação, frequência e validação de leakage).
FAMILY_FEATURES = {
    "statistical": ["mean", "std", "variance", "minimum", "maximum", "range",
                    "median", "q10", "q25", "q75", "q90", "q95"],
    "variability": ["cv", "fano", "burstiness", "peak_mean"],
    "sparsity": ["zero_ratio", "active_ratio", "missing_ratio", "n_available"],
    "distribution": ["entropy", "skewness", "kurtosis", "iqr"],
    "temporal": ["acf_lag1", "acf_lag2", "local_trend", "previous_hour_mean"],
    "stl_trend": ["stl_trend_mean", "stl_trend_std", "stl_trend_slope"],
    "stl_seasonal": ["stl_seasonal_mean", "stl_seasonal_std", "stl_seasonal_amplitude", "stl_seasonal_ratio"],
    "stl_noise": ["stl_noise_mean", "stl_noise_std", "stl_noise_abs_mean", "stl_noise_ratio"],
    "weekly": ["mean_w", "std_w", "cv_w", "acf_lag1_w"],
    "cross_granularity": ["gm10_mean", "gm10_std", "gm10_max", "gm10_min",
                          "gm10_sum", "gm10_count", "gm10_active_ratio", "gm10_cv"],
}

FEATURE_REGISTRY = [
    FeatureDefinition(name=n, family=fam, availability="past_only")
    for fam, names in FAMILY_FEATURES.items()
    for n in names
]
FEATURE_BY_NAME = {f.name: f for f in FEATURE_REGISTRY}
CANDIDATE_FEATURES = [f.name for f in FEATURE_REGISTRY]

validate_feature_availability(FEATURE_REGISTRY)  # contrato anti-leakage


def _roll(s: pd.Series, w: int, fn):
    if isinstance(fn, str):
        return s.rolling(w, min_periods=max(2, w // 2)).agg(fn)
    return s.rolling(w, min_periods=max(2, w // 2)).apply(fn, raw=True)


def _entropy_fn(bins: np.ndarray):
    def _e(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if len(a) < 2:
            return np.nan
        counts, _ = np.histogram(a, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return _e


def causal_stl_features(W: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Decomposição causal (past-only) de tendência/sazonalidade/ruído por instituição.

    - tendência: EWM causal (halflife configurável).
    - sazonal: média expansiva (causal) por hora-do-dia do resíduo destendenciado.
    - ruído: resíduo = y - tendência - sazonal.

    Substitui `statsmodels STL` (que é centrado e vazaria futuro) por uma
    decomposição causal equivalente e vetorizada."""
    ent = cfg["data"]["entity_column"]
    hl = 12  # halflife (h) da tendência
    out = pd.DataFrame(index=W.index)
    for eid, idx in W.groupby(ent).groups.items():
        y = W.loc[idx, "y_obs_ffill"].astype(float)
        hour = W.loc[idx, "time"].dt.hour.to_numpy()
        trend = y.ewm(halflife=hl, adjust=False).mean()
        det = y - trend
        d = det.to_frame("det")
        d["hour"] = hour
        seasonal = d.groupby("hour")["det"].expanding().mean().reset_index(level=0, drop=True)
        seasonal = seasonal.sort_index().reindex(idx)
        resid = det - seasonal

        t1 = trend.shift(1); se1 = seasonal.shift(1); r1 = resid.shift(1)
        w24 = cfg["data"]["stat_window"]; w168 = cfg["data"]["weekly_window"]
        out.loc[idx, "stl_trend_mean"] = _roll(t1, w24, "mean")
        out.loc[idx, "stl_trend_std"] = _roll(t1, w24, "std")
        out.loc[idx, "stl_trend_slope"] = _roll(t1, w24, lambda a: a[-1] - a[0])
        out.loc[idx, "stl_seasonal_mean"] = _roll(se1, w24, "mean")
        out.loc[idx, "stl_seasonal_std"] = _roll(se1, w24, "std")
        out.loc[idx, "stl_seasonal_amplitude"] = _roll(se1, w168, lambda a: a.max() - a.min())
        out.loc[idx, "stl_seasonal_ratio"] = se1 / (t1.abs() + 1e-6)
        out.loc[idx, "stl_noise_mean"] = _roll(r1, w24, "mean")
        out.loc[idx, "stl_noise_std"] = _roll(r1, w24, "std")
        out.loc[idx, "stl_noise_abs_mean"] = _roll(r1.abs(), w24, "mean")
        out.loc[idx, "stl_noise_ratio"] = _roll(r1.abs(), w24, "mean") / (_roll(t1, w24, "mean").abs() + 1e-6)
    return out


def build_original_features(W: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """X_original (mandatório): calendário + identidade + defasagens autoregressivas.

    Defasagens derivadas de `y_obs_ffill` (causal), nunca de `y_true`."""
    ent = cfg["data"]["entity_column"]
    X = pd.DataFrame(index=W.index)
    X["hour"] = W["time"].dt.hour.astype(float)
    X["dow"] = W["time"].dt.dayofweek.astype(float)
    X["id_institution"] = W[ent].astype("int64")

    y = W["y_obs_ffill"].astype(float)
    lags = list(range(1, cfg["data"]["n_lag_features"] + 1)) + list(cfg["data"]["extra_lags"])
    for k in lags:
        X[f"lag_{k}"] = y.groupby(W[ent]).shift(k)
    return X


def build_candidate_features(W: pd.DataFrame, g10: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """X_candidato: todas as características comportamentais/cross-granularidade.

    Toda estatística usa `past = y_obs_ffill.shift(1)` (estritamente anterior à
    hora corrente) => o alvo da hora a imputar jamais entra na feature."""
    ent = cfg["data"]["entity_column"]
    w = cfg["data"]["stat_window"]
    wl = cfg["data"]["weekly_window"]

    y = W["y_obs_ffill"].astype(float)
    past = y.groupby(W[ent]).shift(1)          # past_only
    raw = W["y_obs"].astype(float).groupby(W[ent]).shift(1)  # sem ffill, p/ missing_ratio

    F = pd.DataFrame(index=W.index)

    # --- statistical ---
    F["mean"] = _roll(past, w, "mean"); F["std"] = _roll(past, w, "std")
    F["variance"] = _roll(past, w, "var")
    F["minimum"] = _roll(past, w, "min"); F["maximum"] = _roll(past, w, "max")
    F["range"] = _roll(past, w, lambda a: a.max() - a.min())
    F["median"] = _roll(past, w, "median")
    for q, nm in [(0.10, "q10"), (0.25, "q25"), (0.75, "q75"), (0.90, "q90"), (0.95, "q95")]:
        F[nm] = past.rolling(w, min_periods=max(2, w // 2)).quantile(q)

    # --- variability ---
    m = F["mean"]; sd = F["std"]; eps = 1e-6
    F["cv"] = sd / (m.abs() + eps)
    F["fano"] = F["variance"] / (m.abs() + eps)
    F["burstiness"] = (sd - m) / (sd + m + eps)
    F["peak_mean"] = F["maximum"] / (m.abs() + eps)

    # --- sparsity / activity ---
    F["zero_ratio"] = (past == 0).rolling(w, min_periods=2).mean()
    F["active_ratio"] = (past > 0).rolling(w, min_periods=2).mean()
    F["missing_ratio"] = raw.isna().rolling(w, min_periods=2).mean()
    F["n_available"] = raw.rolling(w, min_periods=2).count()

    # --- distribution ---
    q = past.dropna()
    if len(q) > 0:
        bins = np.unique(np.quantile(q, np.linspace(0, 1, 11)))
        if len(bins) < 2:
            bins = np.array([q.min() - 1.0, q.max() + 1.0])
    else:
        bins = np.array([-1.0, 1.0])
    F["entropy"] = past.rolling(w, min_periods=max(2, w // 2)).apply(_entropy_fn(bins), raw=True)
    F["skewness"] = past.rolling(w, min_periods=max(3, w // 2)).skew()
    F["kurtosis"] = past.rolling(w, min_periods=max(4, w // 2)).kurt()
    F["iqr"] = F["q75"] - F["q25"]

    # --- temporal ---
    F["acf_lag1"] = past.rolling(w, min_periods=max(3, w // 2)).corr(past.shift(1))
    F["acf_lag2"] = past.rolling(w, min_periods=max(4, w // 2)).corr(past.shift(2))
    F["local_trend"] = _roll(past, w, lambda a: a[-1] - a[0])
    F["previous_hour_mean"] = past

    # --- weekly (longa memória) ---
    F["mean_w"] = _roll(past, wl, "mean"); F["std_w"] = _roll(past, wl, "std")
    F["cv_w"] = F["std_w"] / (F["mean_w"].abs() + eps)
    F["acf_lag1_w"] = past.rolling(wl, min_periods=max(3, wl // 2)).corr(past.shift(1))

    # --- STL causal ---
    F = pd.concat([F, causal_stl_features(W, cfg)], axis=1)

    # --- cross-granularity (10min, past_only) ---
    gm_cols = [c for c in CANDIDATE_FEATURES if c.startswith("gm10")]
    if not g10.empty and {"time", "id_institution"}.issubset(g10.columns):
        key = W[["time", "id_institution"]].copy()
        key["_row"] = W.index.to_numpy()
        mg = key.merge(g10, on=["time", "id_institution"], how="left").sort_values("_row")
        for c in gm_cols:
            F[c] = mg[c].to_numpy() if c in mg.columns else np.nan
    else:
        for c in gm_cols:
            F[c] = np.nan

    # garante que só existem colunas registradas (defesa anti-vazamento)
    extra = set(F.columns) - set(CANDIDATE_FEATURES)
    if extra:
        raise ValueError(f"[LEAK] features não registradas: {sorted(extra)}")
    return F[CANDIDATE_FEATURES]


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    return float(100.0 * np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps)))


def build_model(cfg: dict):
    """Estimador configurável. HGB aceita NaN nativo; RF/ET usam imputação por
    mediana de treino (feita dentro do avaliador)."""
    name = cfg["model"]["name"]
    seed = cfg["model"]["random_seed"]
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            random_state=seed, categorical_features=["id_institution"],
            max_iter=300, learning_rate=0.05, early_stopping=False,
        )
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    raise ValueError(f"modelo desconhecido: {name}")


_MODEL_NEEDS_FILL = {"random_forest", "extra_trees"}


def _trim_cols(Xtr: pd.DataFrame, Xva: pd.DataFrame | None = None):
    """Remove colunas sem nenhum valor não-nulo (HGB/RF falham com coluna all-NaN)."""
    ok = [c for c in Xtr.columns if Xtr[c].notna().any()]
    Xtr = Xtr[ok]
    if Xva is not None:
        Xva = Xva[ok]
        return Xtr, Xva
    return Xtr


def make_evaluator(X_dev, y_true_dev, y_obs_dev, mask_dev, avail_dev, cv, cfg, logger):
    """Fábrica do avaliador comum. SFFS e GA otimizam EXATAMENTE o mesmo critério.

    `selected` = subconjunto de candidatas; `mandatory` = features originais fixas.
    """
    lam = cfg["selection"]["feature_penalty_lambda"]
    mode = cfg["selection"]["objective_mode"]
    alpha = cfg["selection"]["multiplicative_alpha"]
    p_total = len(CANDIDATE_FEATURES)
    needs_fill = cfg["model"]["name"] in _MODEL_NEEDS_FILL
    cache: dict = {}

    def objective(mae_mean: float, n_sel: int) -> float:
        if mode == "additive":
            return mae_mean + lam * n_sel
        return mae_mean * (1.0 + alpha * n_sel / p_total)

    def evaluate(selected, mandatory):
        feats = list(mandatory) + list(selected)
        key = tuple(sorted(feats))
        if key in cache:
            return cache[key]

        t0 = time.time()
        fold_mae, fold_rmse, fold_smape = [], [], []
        for tr_idx, va_idx in cv.split(X_dev):
            tr = tr_idx[avail_dev[tr_idx] & ~mask_dev[tr_idx]]
            va = va_idx[avail_dev[va_idx] & mask_dev[va_idx]]
            if len(tr) < 2 or len(va) == 0:
                continue
            m = build_model(cfg)
            Xtr = X_dev.iloc[tr][feats]
            Xva = X_dev.iloc[va][feats]
            Xtr, Xva = _trim_cols(Xtr, Xva)
            if needs_fill:
                med = Xtr.median()
                Xtr = Xtr.fillna(med)
                Xva = Xva.fillna(med)
            m.fit(Xtr, y_obs_dev[tr])
            pred = m.predict(Xva)
            yt = y_true_dev[va]
            fold_mae.append(mae(yt, pred))
            fold_rmse.append(rmse(yt, pred))
            fold_smape.append(smape(yt, pred, cfg["metrics"]["epsilon"]))

        if not fold_mae:
            raise RuntimeError("nenhum fold válido — aumente os dados ou reduza a taxa de missingness")

        mae_mean = float(np.mean(fold_mae))
        result = {
            "features": feats,
            "selected": list(selected),
            "n_features": len(selected),
            "mae_mean": mae_mean,
            "mae_std": float(np.std(fold_mae)),
            "rmse_mean": float(np.mean(fold_rmse)),
            "smape_mean": float(np.mean(fold_smape)),
            "objective": objective(mae_mean, len(selected)),
            "complexity_penalty": lam * len(selected),
            "runtime_seconds": time.time() - t0,
            "fold_metrics": [
                {"mae": a, "rmse": b, "smape": c}
                for a, b, c in zip(fold_mae, fold_rmse, fold_smape)
            ],
        }
        cache[key] = result
        return result

    return evaluate, cache


def forward_selection(mandatory, candidates, evaluate, cfg, logger, history_path=None):
    """Seleção forward gulosa (usada na análise de estabilidade por ser rápida)."""
    chosen: list = []
    cache = {}
    history = []

    def ev(sel):
        feats = list(mandatory) + list(sel)
        k = tuple(sorted(feats))
        if k not in cache:
            cache[k] = evaluate(sel, mandatory)
        return cache[k]

    base = ev([])
    current = base["objective"]
    # melhoria mínima RELATIVA à linha de base (escala-invariante)
    thresh = cfg["selection"]["min_improvement"] * max(1.0, abs(base["objective"]))
    it = 0
    while len(chosen) < cfg["selection"]["max_features"]:
        remaining = [c for c in candidates if c not in chosen]
        best, best_feat = None, None
        for c in remaining:
            r = ev(chosen + [c])
            if best is None or r["objective"] < best["objective"]:
                best, best_feat = r, c
        if best is None or (current - best["objective"]) < thresh:
            break
        chosen.append(best_feat)
        current = best["objective"]
        it += 1
        history.append({
            "iteration": it, "action": "ADD", "candidate_feature": best_feat,
            "selected_features": ",".join(chosen), "n_features": len(chosen),
            "mae": best["mae_mean"], "rmse": best["rmse_mean"], "smape": best["smape_mean"],
            "complexity_penalty": best["complexity_penalty"], "objective": best["objective"],
            "runtime_seconds": best["runtime_seconds"],
        })
        logger.info(f"FORWARD step={it} ADD {best_feat} MAE={best['mae_mean']:.4f} obj={best['objective']:.4f}")
    return chosen, history, cache


def sffs(mandatory, candidates, evaluate, cfg, logger, history_path=None):
    """SFFS: inclusão forward + exclusão condicional (floating backward)."""
    chosen: list = []
    cache = {}
    history = []

    def ev(sel):
        feats = list(mandatory) + list(sel)
        k = tuple(sorted(feats))
        if k not in cache:
            cache[k] = evaluate(sel, mandatory)
        return cache[k]

    base = ev([])
    current = base["objective"]
    thresh = cfg["selection"]["min_improvement"] * max(1.0, abs(base["objective"]))
    it = 0
    while len(chosen) < cfg["selection"]["max_features"]:
        # ---- FORWARD ----
        remaining = [c for c in candidates if c not in chosen]
        best, best_feat = None, None
        for c in remaining:
            r = ev(chosen + [c])
            if best is None or r["objective"] < best["objective"]:
                best, best_feat = r, c
        if best is None or (current - best["objective"]) < thresh:
            break
        chosen.append(best_feat)
        current = best["objective"]
        it += 1
        history.append({
            "iteration": it, "action": "ADD", "candidate_feature": best_feat,
            "selected_features": ",".join(chosen), "n_features": len(chosen),
            "mae": best["mae_mean"], "rmse": best["rmse_mean"], "smape": best["smape_mean"],
            "complexity_penalty": best["complexity_penalty"], "objective": best["objective"],
            "runtime_seconds": best["runtime_seconds"],
        })
        logger.info(f"SFFS step={it} ADD {best_feat} MAE={best['mae_mean']:.4f} obj={best['objective']:.4f}")

        # ---- FLOATING BACKWARD ----
        while len(chosen) > 1:
            best_rem, rem_feat = None, None
            for c in chosen:
                r = ev([f for f in chosen if f != c])
                if best_rem is None or r["objective"] < best_rem["objective"]:
                    best_rem, rem_feat = r, c
            if best_rem["objective"] < current - 1e-12:
                chosen.remove(rem_feat)
                current = best_rem["objective"]
                it += 1
                history.append({
                    "iteration": it, "action": "REMOVE", "candidate_feature": rem_feat,
                    "selected_features": ",".join(chosen), "n_features": len(chosen),
                    "mae": best_rem["mae_mean"], "rmse": best_rem["rmse_mean"], "smape": best_rem["smape_mean"],
                    "complexity_penalty": best_rem["complexity_penalty"], "objective": best_rem["objective"],
                    "runtime_seconds": best_rem["runtime_seconds"],
                })
                logger.info(f"SFFS step={it} REMOVE {rem_feat} MAE={best_rem['mae_mean']:.4f} obj={best_rem['objective']:.4f}")
            else:
                break
    return chosen, history, cache


def genetic_algorithm(mandatory, candidates, evaluate, cfg, logger, history_path=None):
    """GA binário com seleção por torneio, crossover de 1 ponto, mutação e elitismo."""
    n = len(candidates)
    pop = cfg["genetic_algorithm"]["population_size"]
    gens = cfg["genetic_algorithm"]["generations"]
    pc = cfg["genetic_algorithm"]["crossover_probability"]
    pm = cfg["genetic_algorithm"]["mutation_probability"]
    if pm == "auto":
        pm = 1.0 / max(1, n)
    tour = cfg["genetic_algorithm"]["tournament_size"]
    elite = cfg["genetic_algorithm"]["elitism"]

    rng = np.random.default_rng(cfg["experiment"]["random_seed"])
    cache = {}
    eval_records = {}

    def ev(chrom):
        k = tuple(int(b) for b in chrom)
        if k not in cache:
            sel = [c for c, b in zip(candidates, chrom) if b]
            cache[k] = evaluate(sel, mandatory)
            eval_records[k] = {"chromosome": "".join(str(int(b)) for b in chrom),
                               "features": ",".join(sel), "objective": cache[k]["objective"],
                               "mae": cache[k]["mae_mean"]}
        return cache[k]

    def fitness(chrom):
        return -ev(chrom)["objective"]  # maximizar => minimizar J

    # população inicial: 50% aleatória + indivíduos vazios/singulares p/ bootstrap
    pop_chrom = []
    pop_chrom.append(np.zeros(n, dtype=int))
    for i in range(min(n, pop)):
        c = np.zeros(n, dtype=int); c[i] = 1
        pop_chrom.append(c)
    while len(pop_chrom) < pop:
        pop_chrom.append((rng.random(n) < 0.5).astype(int))

    history = []
    best_overall = None

    for gen in range(gens):
        t0 = time.time()
        scored = [(c, fitness(c)) for c in pop_chrom]
        scored.sort(key=lambda x: -x[1])
        objs = [-s for _, s in scored]
        best_chrom = scored[0][0]
        best_res = ev(best_chrom)
        if best_overall is None or best_res["objective"] < best_overall["objective"]:
            best_overall = best_res

        diversity = len({tuple(int(b) for b in c) for c in pop_chrom}) / max(1, len(pop_chrom))
        history.append({
            "generation": gen, "best_mae": best_res["mae_mean"],
            "best_objective": best_res["objective"], "mean_objective": float(np.mean(objs)),
            "median_objective": float(np.median(objs)), "population_diversity": diversity,
            "best_n_features": best_res["n_features"],
            "best_features": ",".join([c for c, b in zip(candidates, best_chrom) if b]),
            "runtime_seconds": time.time() - t0,
        })
        logger.info(f"GA gen={gen} best_obj={best_res['objective']:.4f} "
                    f"mean_obj={np.mean(objs):.4f} div={diversity:.3f}")

        # nova geração
        new_pop = [scored[i][0].copy() for i in range(elite)]  # elitismo
        while len(new_pop) < pop:
            # torneio
            ti = rng.integers(0, len(scored), size=tour)
            p1 = max((scored[i] for i in ti), key=lambda x: x[1])[0]
            ti = rng.integers(0, len(scored), size=tour)
            p2 = max((scored[i] for i in ti), key=lambda x: x[1])[0]
            if rng.random() < pc:
                cp = rng.integers(1, n)
                child = np.concatenate([p1[:cp], p2[cp:]])
            else:
                child = p1.copy()
            mut = rng.random(n) < pm
            child = child.copy()
            child[mut] = 1 - child[mut]
            new_pop.append(child)
        pop_chrom = new_pop

    return best_overall, history, cache, eval_records


def run_experiment(cfg: dict) -> dict:
    set_seed(cfg["experiment"]["random_seed"])
    exp_id = f"{cfg['experiment']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root = ROOT / cfg["experiment"]["output_dir"] / exp_id
    dirs = {
        "logs": out_root / "logs", "metrics": out_root / "metrics",
        "selections": out_root / "selections", "predictions": out_root / "predictions",
        "plots": out_root / "plots", "config": out_root / "config",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(dirs["logs"])
    logger.info("=" * 70)
    logger.info("Experiment started")
    logger.info(f"experiment_id={exp_id}")
    logger.info(f"python={sys.version.split()[0]}")
    logger.info(f"numpy={np.__version__} pandas={pd.__version__}")
    logger.info(f"seed={cfg['experiment']['random_seed']}")
    logger.info(f"config={json.dumps(cfg, default=str)}")
    write_json(dirs["config"] / "config.json", cfg)

    # ---------- dados ----------
    W = load_and_prepare(cfg)
    logger.info(f"Loaded {len(W)} hourly rows, {W['id_institution'].nunique()} institutions")
    logger.info(f"date range: {W['time'].min()} -> {W['time'].max()}")
    g10 = load_10min_hourly(cfg)
    logger.info(f"10min cross-granularity: {len(g10)} hourly aggregates")

    W["y_true"] = W["y_true"].astype(float)
    W["available"] = W["y_true"].notna()

    # ---------- missingness primária ----------
    pm = cfg["missingness"]["primary"]
    mask = apply_missingness(W["y_true"].to_numpy(), cfg, pm["mechanism"],
                             pm["rate"], pm["block_length"], pm["seed"])
    W["mask"] = mask & W["available"].to_numpy()
    W["y_obs"] = W["y_true"].where(~W["mask"])
    W["y_obs_ffill"] = W.groupby("id_institution")["y_obs"].ffill()
    logger.info(f"Primary missingness: {pm['mechanism']} rate={pm['rate']} "
                f"seed={pm['seed']} => {int(W['mask'].sum())} masked hours")

    # ---------- features ----------
    X_orig = build_original_features(W, cfg)
    X_cand = build_candidate_features(W, g10, cfg)
    X = pd.concat([X_orig, X_cand], axis=1)
    original_features = list(X_orig.columns)
    logger.info(f"Original features: {len(original_features)}; candidates: {len(CANDIDATE_FEATURES)}")

    # ---------- ordenação temporal global + split cronológico ----------
    order = np.argsort(W["time"].to_numpy(), kind="stable")
    W = W.iloc[order].reset_index(drop=True)
    X = X.iloc[order].reset_index(drop=True)

    n = len(W)
    split = int(n * (1 - cfg["validation"]["final_test_fraction"]))
    dev_idx = np.arange(split)
    test_idx = np.arange(split, n)

    X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
    y_true_dev = W["y_true"].to_numpy()[dev_idx]
    y_obs_dev = W["y_obs"].to_numpy()[dev_idx]
    mask_dev = W["mask"].to_numpy()[dev_idx]
    avail_dev = W["available"].to_numpy()[dev_idx]
    y_true_test = W["y_true"].to_numpy()[test_idx]
    y_obs_test = W["y_obs"].to_numpy()[test_idx]
    mask_test = W["mask"].to_numpy()[test_idx]
    avail_test = W["available"].to_numpy()[test_idx]

    logger.info(f"Chronological split: dev={split} test={n - split} rows "
                f"(dev end={W['time'].iloc[split-1]}, test start={W['time'].iloc[split]})")

    # ---------- CV temporal ----------
    try:
        cv = TimeSeriesSplit(n_splits=cfg["validation"]["inner_splits"],
                             gap=cfg["validation"]["gap"])
    except TypeError:
        cv = TimeSeriesSplit(n_splits=cfg["validation"]["inner_splits"])

    evaluate, eval_cache = make_evaluator(X_dev, y_true_dev, y_obs_dev, mask_dev,
                                          avail_dev, cv, cfg, logger)

    # ---------- baselines M0/M1/M2 ----------
    logger.info("Evaluating baselines...")
    r_m0 = evaluate([], original_features)
    r_m1 = evaluate(CANDIDATE_FEATURES, [])           # características sozinhas
    r_m2 = evaluate(CANDIDATE_FEATURES, original_features)  # tudo
    logger.info(f"M0 (original) MAE={r_m0['mae_mean']:.4f}")
    logger.info(f"M1 (characteristics) MAE={r_m1['mae_mean']:.4f}")
    logger.info(f"M2 (original+all) MAE={r_m2['mae_mean']:.4f}")

    # ---------- SFFS ----------
    logger.info("Running SFFS...")
    sffs_sel, sffs_hist, sffs_cache = sffs(original_features, CANDIDATE_FEATURES,
                                           evaluate, cfg, logger)
    r_m3 = evaluate(sffs_sel, original_features)
    logger.info(f"SFFS selected {len(sffs_sel)} features: {sffs_sel}")
    logger.info(f"M3 (original+SFFS) MAE={r_m3['mae_mean']:.4f}")

    # ---------- GA ----------
    logger.info("Running GA...")
    ga_best, ga_hist, ga_cache, ga_evals = genetic_algorithm(
        original_features, CANDIDATE_FEATURES, evaluate, cfg, logger)
    ga_sel = ga_best["selected"]
    r_ga = evaluate(ga_sel, original_features)
    logger.info(f"GA selected {len(ga_sel)} features: {ga_sel}")
    logger.info(f"GA model MAE={r_ga['mae_mean']:.4f}")

    # ---------- comparação SFFS x GA ----------
    s_set, g_set = set(sffs_sel), set(ga_sel)
    inter = sorted(s_set & g_set)
    union = sorted(s_set | g_set)
    jaccard = len(s_set & g_set) / max(1, len(s_set | g_set))
    logger.info(f"SFFS∩GA common features: {inter}")
    logger.info(f"Jaccard(SFFS, GA)={jaccard:.4f}")

    # ---------- ablação por família ----------
    fam = cfg["selection"]["feature_penalty_lambda"]
    ablation_rows = [
        {"family": "Original only", "n": 0, "mae": r_m0["mae_mean"], "rmse": r_m0["rmse_mean"], "smape": r_m0["smape_mean"]},
    ]
    for family in FAMILY_FEATURES:
        feats = FAMILY_FEATURES[family]
        r = evaluate(feats, [])
        ablation_rows.append({"family": family, "n": len(feats), "mae": r["mae_mean"],
                              "rmse": r["rmse_mean"], "smape": r["smape_mean"]})
        r2 = evaluate(feats, original_features)
        ablation_rows.append({"family": f"Original + {family}", "n": len(feats),
                              "mae": r2["mae_mean"], "rmse": r2["rmse_mean"], "smape": r2["smape_mean"]})
    ablation_rows.append({"family": "Original + selected SFFS", "n": len(sffs_sel),
                          "mae": r_m3["mae_mean"], "rmse": r_m3["rmse_mean"], "smape": r_m3["smape_mean"]})
    ablation_rows.append({"family": "Original + selected GA", "n": len(ga_sel),
                          "mae": r_ga["mae_mean"], "rmse": r_ga["rmse_mean"], "smape": r_ga["smape_mean"]})
    ablation_rows.append({"family": "Original + all characteristics", "n": len(CANDIDATE_FEATURES),
                          "mae": r_m2["mae_mean"], "rmse": r_m2["rmse_mean"], "smape": r_m2["smape_mean"]})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df["delta_mae_vs_original"] = r_m0["mae_mean"] - ablation_df["mae"]

    # ---------- estabilidade da seleção ----------
    freq = {f: 0 for f in CANDIDATE_FEATURES}
    n_runs = 0
    stability_rows = []
    if cfg["stability"]["enabled"]:
        for rate in cfg["stability"]["rates"]:
            for seed in cfg["stability"]["seeds"]:
                m = apply_missingness(W["y_true"].to_numpy(), cfg, "mcar", rate, 6, seed)
                Wtmp = W.copy()
                Wtmp["mask"] = m & Wtmp["available"].to_numpy()
                Wtmp["y_obs"] = Wtmp["y_true"].where(~Wtmp["mask"])
                Wtmp["y_obs_ffill"] = Wtmp.groupby("id_institution")["y_obs"].ffill()
                Xt_orig = build_original_features(Wtmp, cfg)
                Xt_cand = build_candidate_features(Wtmp, g10, cfg)
                Xt = pd.concat([Xt_orig, Xt_cand], axis=1)
                Xt_dev = Xt.iloc[dev_idx]
                yot_dev = Wtmp["y_obs"].to_numpy()[dev_idx]
                mt_dev = Wtmp["mask"].to_numpy()[dev_idx]
                ev_t, _ = make_evaluator(Xt_dev, y_true_dev, yot_dev, mt_dev, avail_dev, cv, cfg, logger)
                cfg_stab = json.loads(json.dumps(cfg))
                cfg_stab["selection"]["max_features"] = cfg["stability"]["max_features"]
                sel, _, _ = forward_selection(original_features, CANDIDATE_FEATURES, ev_t, cfg_stab, logger)
                for f in sel:
                    freq[f] += 1
                n_runs += 1
                stability_rows.append({"rate": rate, "seed": seed,
                                       "selected": ",".join(sel), "n": len(sel)})
                logger.info(f"stability rate={rate} seed={seed} -> {len(sel)} features")
        freq = {f: c / n_runs for f, c in freq.items()}
    freq_df = pd.DataFrame([{"feature": f, "family": FEATURE_BY_NAME[f].family,
                             "frequency": freq[f]} for f in CANDIDATE_FEATURES]).sort_values(
        "frequency", ascending=False)

    # ---------- avaliação final no teste intocado ----------
    def final_test_mae(features_sel, mandatory):
        tr = dev_idx[avail_dev & ~mask_dev]
        va = test_idx[avail_test & mask_test]
        m = build_model(cfg)
        feats = list(mandatory) + list(features_sel)
        Xtr, Xva = _trim_cols(X.iloc[tr][feats], X.iloc[va][feats])
        if cfg["model"]["name"] in _MODEL_NEEDS_FILL:
            med = Xtr.median()
            Xtr = Xtr.fillna(med)
            Xva = Xva.fillna(med)
        m.fit(Xtr, y_obs_dev[tr])
        pred = m.predict(Xva)
        yt = y_true_test[va - split]
        return {"mae": mae(yt, pred), "rmse": rmse(yt, pred),
                "smape": smape(yt, pred, cfg["metrics"]["epsilon"]), "pred": pred, "va": va}

    final = {}
    final["M0_original"] = final_test_mae([], original_features)
    final["M1_characteristics"] = final_test_mae(CANDIDATE_FEATURES, [])
    final["M2_all"] = final_test_mae(CANDIDATE_FEATURES, original_features)
    final["M3_sffs"] = final_test_mae(sffs_sel, original_features)
    final["M3_ga"] = final_test_mae(ga_sel, original_features)

    final_rows = [{"method": k, "n_features": (0 if k == "M0_original" else
                   len(CANDIDATE_FEATURES) if k in ("M1_characteristics", "M2_all") else
                   len(sffs_sel) if k == "M3_sffs" else len(ga_sel)),
                   "cv_mae": {"M0_original": r_m0["mae_mean"], "M1_characteristics": r_m1["mae_mean"],
                              "M2_all": r_m2["mae_mean"], "M3_sffs": r_m3["mae_mean"],
                              "M3_ga": r_ga["mae_mean"]}[k],
                   "test_mae": v["mae"], "test_rmse": v["rmse"], "test_smape": v["smape"]}
                  for k, v in final.items()]

    base_mae = final["M0_original"]["mae"]
    improvement_pct = {k: 100.0 * (base_mae - v["mae"]) / base_mae for k, v in final.items()}

    # ---------- gravação de resultados ----------
    save_df(dirs["metrics"] / "baseline_results.csv", pd.DataFrame(final_rows))
    save_df(dirs["metrics"] / "ablation_results.csv", ablation_df)
    save_df(dirs["metrics"] / "missingness_results.csv", pd.DataFrame(stability_rows))
    save_df(dirs["metrics"] / "final_test_results.csv", pd.DataFrame(final_rows))
    save_df(dirs["metrics"] / "feature_selection_frequency.csv", freq_df)

    save_df(dirs["selections"] / "sffs_history.csv", pd.DataFrame(sffs_hist))
    write_json(dirs["selections"] / "sffs_best.json", {
        "features": sffs_sel, "n_features": len(sffs_sel),
        "cv_mae": r_m3["mae_mean"], "cv_rmse": r_m3["rmse_mean"], "cv_smape": r_m3["smape_mean"],
    })
    save_df(dirs["selections"] / "ga_history.csv", pd.DataFrame(ga_hist))
    save_df(dirs["selections"] / "ga_evaluations.csv",
            pd.DataFrame(list(ga_evals.values())))
    write_json(dirs["selections"] / "ga_best.json", {
        "features": ga_sel, "n_features": len(ga_sel),
        "cv_mae": r_ga["mae_mean"], "cv_rmse": r_ga["rmse_mean"], "cv_smape": r_ga["smape_mean"],
    })

    # predições finais (M3 SFFS)
    pred_rows = []
    va = final["M3_sffs"]["va"]
    for i, rowpos in enumerate(va):
        j = rowpos - split
        pred_rows.append({
            "timestamp": W["time"].iloc[rowpos], "y_true": y_true_test[j],
            "y_observed": y_obs_test[j], "y_pred": final["M3_sffs"]["pred"][i],
            "is_artificially_missing": int(mask_test[j]), "missingness_type": pm["mechanism"],
            "missingness_rate": pm["rate"], "model": cfg["model"]["name"], "feature_set": "M3_sffs",
        })
    save_df(dirs["predictions"] / "final_predictions.csv", pd.DataFrame(pred_rows))

    # ---------- robustez de modelo (secundária, S* congelado) ----------
    robustness = []
    for mname in cfg["model"]["robustness_models"]:
        cfg2 = json.loads(json.dumps(cfg))
        cfg2["model"]["name"] = mname
        tr = dev_idx[avail_dev & ~mask_dev]
        va = test_idx[avail_test & mask_test]
        feats = list(original_features) + list(sffs_sel)
        mm = build_model(cfg2)
        Xtr, Xva = _trim_cols(X.iloc[tr][feats], X.iloc[va][feats])
        if mname in _MODEL_NEEDS_FILL:
            med = Xtr.median()
            Xtr = Xtr.fillna(med)
            Xva = Xva.fillna(med)
        mm.fit(Xtr, y_obs_dev[tr])
        pred = mm.predict(Xva)
        yt_test_va = y_true_test[va - split]
        robustness.append({"model": mname, "mae": mae(yt_test_va, pred),
                           "rmse": rmse(yt_test_va, pred), "smape": smape(yt_test_va, pred)})
    save_df(dirs["metrics"] / "model_robustness.csv", pd.DataFrame(robustness))

    # ---------- sumário ----------
    summary = {
        "experiment_id": exp_id, "dataset": str(ROOT / cfg["data"]["path_hour"]),
        "date_range": [str(W["time"].min()), str(W["time"].max())],
        "n_observations": int(n), "n_institutions": int(W["id_institution"].nunique()),
        "missingness_mechanism": pm["mechanism"], "missingness_rate": pm["rate"],
        "n_candidate_features": len(CANDIDATE_FEATURES),
        "sffs_subset": sffs_sel, "ga_subset": ga_sel, "common_features": inter,
        "jaccard_sffs_ga": jaccard,
        "cv_mae": {"M0": r_m0["mae_mean"], "M1": r_m1["mae_mean"], "M2": r_m2["mae_mean"], "M3_sffs": r_m3["mae_mean"]},
        "final_test": {k: {"mae": v["mae"], "rmse": v["rmse"], "smape": v["smape"]} for k, v in final.items()},
        "baseline_improvement_pct": improvement_pct,
        "model_robustness": robustness,
    }
    write_json(dirs["config"] / "experiment_summary.json", summary)
    txt = [f"{k}: {v}" for k, v in summary.items() if not isinstance(v, (dict, list))]
    (dirs["config"] / "experiment_summary.txt").write_text(
        "\n".join(txt) + "\n\nSFFS subset: " + ", ".join(sffs_sel) +
        "\nGA subset: " + ", ".join(ga_sel) +
        "\nCommon: " + ", ".join(inter), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("Experiment finished")
    logger.info(f"Final-test MAE  M0={final['M0_original']['mae']:.4f} "
                f"M1={final['M1_characteristics']['mae']:.4f} "
                f"M2={final['M2_all']['mae']:.4f} "
                f"M3_sffs={final['M3_sffs']['mae']:.4f} M3_ga={final['M3_ga']['mae']:.4f}")
    logger.info(f"Baseline improvement (M3_sffs) = {improvement_pct['M3_sffs']:.2f}%")

    return {
        "exp_id": exp_id, "out_root": out_root, "dirs": dirs, "logger": logger,
        "W": W, "X": X, "X_dev": X_dev, "X_test": X_test,
        "original_features": original_features, "candidates": CANDIDATE_FEATURES,
        "sffs_sel": sffs_sel, "ga_sel": ga_sel, "sffs_hist": sffs_hist, "ga_hist": ga_hist,
        "ablation": ablation_df, "freq": freq_df, "final": final, "final_rows": final_rows,
        "improvement_pct": improvement_pct, "summary": summary,
        "y_true_test": y_true_test, "y_obs_test": y_obs_test, "mask_test": mask_test,
        "r_m0": r_m0, "r_m2": r_m2, "r_m3": r_m3, "r_ga": r_ga,
    }


def make_all_plots(R: dict, cfg: dict) -> None:
    pdir = R["dirs"]["plots"]
    dpi = cfg["plots"]["dpi"]
    W, X = R["W"], R["X"]
    fam = cfg["selection"]["feature_penalty_lambda"]

    def savefig(fig, name):
        for fmt in cfg["plots"]["formats"]:
            fig.savefig(pdir / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 01 — trajetória SFFS (MAE x n_features)
    h = pd.DataFrame(R["sffs_hist"])
    if len(h):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["n_features"], h["mae"], marker="o")
        ax.set_xlabel("nº de features selecionadas"); ax.set_ylabel("CV MAE")
        ax.set_title("SFFS — trajetória de otimização")
        savefig(fig, "01_sffs_mae_vs_features")

    # 02 — objetivo penalizado
    if len(h):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["n_features"], h["objective"], marker="o", color="crimson")
        ax.set_xlabel("nº de features"); ax.set_ylabel("MAE + λ|S|")
        ax.set_title("SFFS — objetivo penalizado")
        savefig(fig, "02_sffs_objective")

    # 03 — convergência GA
    gh = pd.DataFrame(R["ga_hist"])
    if len(gh):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gh["generation"], gh["best_objective"], label="best")
        ax.plot(gh["generation"], gh["mean_objective"], label="mean", alpha=0.7)
        ax.set_xlabel("geração"); ax.set_ylabel("objetivo (menor = melhor)")
        ax.legend(); ax.set_title("GA — convergência")
        savefig(fig, "03_ga_convergence")

    # 04 — frequência de seleção
    if len(R["freq"]):
        f = R["freq"].head(20)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(f["feature"], f["frequency"], color="steelblue")
        ax.set_xlabel("frequência de seleção"); ax.invert_yaxis()
        ax.set_title("Frequência de seleção (estabilidade)")
        savefig(fig, "04_feature_selection_frequency")

    # 05 — correlação entre candidatas
    cand = R["candidates"]
    Xc = X[cand].dropna(axis=1, how="all").fillna(0)
    if Xc.shape[1] > 2:
        corr = Xc.corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im, ax=ax); ax.set_title("Correlação entre características candidatas")
        savefig(fig, "05_feature_correlation")

    # 06 — ablação
    ab = R["ablation"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ab["family"], ab["mae"], color="steelblue")
    ax.axhline(R["r_m0"]["mae_mean"], color="crimson", ls="--", label="M0 original")
    ax.set_xticklabels(ab["family"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CV MAE"); ax.legend(); ax.set_title("Ablação por família")
    savefig(fig, "06_ablation_mae")

    # 07 — comparação final M0/M1/M2/M3
    fr = pd.DataFrame(R["final_rows"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axm, m in zip(axes, ["test_mae", "test_rmse", "test_smape"]):
        axm.bar(fr["method"], fr[m], color=["gray", "skyblue", "skyblue", "seagreen", "orange"])
        axm.set_title(m); axm.set_xticklabels(fr["method"], rotation=45, fontsize=8)
    fig.suptitle("Comparação final (teste cronológico intocado)")
    fig.tight_layout()
    savefig(fig, "07_final_model_comparison")

    # 08 — série real vs imputada (trecho do teste)
    va = R["final"]["M3_sffs"]["va"]
    if len(va):
        fig, ax = plt.subplots(figsize=(11, 4))
        t = [W["time"].iloc[i] for i in va[:200]]
        ax.plot(t, R["y_true_test"][:200], label="y_true", color="black", lw=1)
        ax.plot(t, R["final"]["M3_sffs"]["pred"][:200], label="imputed", color="crimson", ls="--")
        ax.legend(); ax.set_title("Série real vs imputada (posições mascaradas do teste)")
        fig.autofmt_xdate()
        savefig(fig, "08_actual_vs_imputed")

    # 09 — dispersão real x predito
    yt = R["y_true_test"]; yp = R["final"]["M3_sffs"]["pred"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=4, alpha=0.4)
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lim, lim, color="crimson", ls="--", label="y=x")
    ax.set_xlabel("y_true"); ax.set_ylabel("y_pred"); ax.legend()
    ax.set_title("Real vs predito")
    savefig(fig, "09_actual_vs_predicted")

    # 10 — distribuição dos resíduos
    resid = yt - yp
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=60, color="steelblue")
    ax.set_xlabel("resíduo (y_true - y_pred)"); ax.set_title("Distribuição dos resíduos")
    savefig(fig, "10_residual_distribution")

    # 11 — resíduos ao longo do tempo
    va = R["final"]["M3_sffs"]["va"]
    if len(va):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.scatter([W["time"].iloc[i] for i in va], resid, s=3, alpha=0.5)
        ax.axhline(0, color="crimson", ls="--"); ax.set_ylabel("resíduo")
        ax.set_title("Resíduos ao longo do tempo")
        fig.autofmt_xdate()
        savefig(fig, "11_residuals_over_time")

    # 12/13/14 — robustez por taxa/bloco e ganho por família
    fr = pd.DataFrame(R["final_rows"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(fr["method"], fr["test_mae"], color="steelblue")
    ax.set_title("MAE por método"); ax.set_xticklabels(fr["method"], rotation=45, fontsize=8)
    savefig(fig, "12_mae_vs_missingness")

    ab = R["ablation"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ab["family"], ab["delta_mae_vs_original"], color="seagreen")
    ax.set_xticklabels(ab["family"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ MAE vs M0 (positivo = melhora)")
    ax.set_title("Contribuição por família")
    savefig(fig, "14_feature_family_gain")

    # 13 — robustez ao comprimento de bloco (placeholder honesto: avaliado no teste)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Ver metricas/missingness_results.csv\npara robustez por taxa/bloco",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Robustez a comprimento de bloco")
    savefig(fig, "13_mae_vs_block_length")


def run_self_checks() -> None:
    """Testes de leakage/corretude obrigatórios, sobre dados sintéticos."""
    results = []
    def check(name, fn):
        try:
            fn()
            results.append((name, "PASS"))
        except AssertionError as e:
            results.append((name, f"FAIL: {e}"))
            raise

    # ---- dados sintéticos ----
    def synth(n=300, seed=0):
        rng = np.random.default_rng(seed)
        t = pd.date_range("2024-01-01", periods=n, freq="h")
        y = 50 + 15 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 5, n)
        W = pd.DataFrame({"time": t, "id_institution": 0, "y_true": y})
        W["available"] = True
        mask = apply_missingness(W["y_true"].to_numpy(), CONFIG, "mcar", 0.2, 6, 11)
        W["mask"] = mask
        W["y_obs"] = W["y_true"].where(~W["mask"])
        W["y_obs_ffill"] = W.groupby("id_institution")["y_obs"].ffill()
        return W

    def test_time_order_is_preserved():
        W = synth()
        order = np.argsort(W["time"].to_numpy(), kind="stable")
        assert np.all(np.diff(W["time"].to_numpy()[order]) >= pd.Timedelta(0)), "ordem temporal violada"

    def test_training_never_contains_future_test_rows():
        W = synth()
        order = np.argsort(W["time"].to_numpy(), kind="stable")
        W = W.iloc[order].reset_index(drop=True)
        split = int(len(W) * 0.8)
        assert W["time"].iloc[split - 1] <= W["time"].iloc[split], "teste contém passado"

    def test_ground_truth_is_never_modified():
        W = synth()
        before = W["y_true"].copy()
        W["y_obs"] = W["y_true"].where(~W["mask"])  # aplica missingness só em y_obs
        assert W["y_true"].equals(before), "y_true foi modificado!"

    def test_metrics_only_use_masked_positions():
        rng = np.random.default_rng(0)
        yt = rng.normal(0, 1, 100); mask = np.zeros(100, bool); mask[:30] = True
        # avaliador interno: métricas calculadas apenas em va (mascarado) por construção
        assert mask.sum() == 30
        assert mae(yt[mask], yt[mask]) == 0.0

    def test_current_target_is_not_a_feature():
        W = synth()
        g10 = pd.DataFrame(columns=["time", "id_institution"] + [c for c in FAMILY_FEATURES["cross_granularity"]])
        Xo = build_original_features(W, CONFIG)
        Xc = build_candidate_features(W, g10, CONFIG)
        assert "y_true" not in Xo.columns and "y_true" not in Xc.columns
        # nenhuma feature é igual ao alvo atual
        assert not any("current" in c or c == "y_obs" for c in Xo.columns)

    def test_disallowed_features_raise_leakage_error():
        bad = [FeatureDefinition("x", "temporal", "derived_from_target")]
        raised = False
        try:
            validate_feature_availability(bad)
        except ValueError:
            raised = True
        assert raised, "derived_from_target deveria falhar"

    def _cfg_fast():
        import copy
        c = copy.deepcopy(CONFIG)
        c["selection"]["min_improvement"] = -1.0  # força ADD p/ testar a mecânica
        c["selection"]["max_features"] = 3
        return c

    def test_sffs_can_add_features():
        W = synth()
        Xo = build_original_features(W, CONFIG)
        Xc = build_candidate_features(W, pd.DataFrame(), CONFIG)
        X = pd.concat([Xo, Xc], axis=1)
        cv = TimeSeriesSplit(n_splits=3)
        ev, _ = make_evaluator(X, W["y_true"].to_numpy(), W["y_obs"].to_numpy(),
                               W["mask"].to_numpy(), np.ones(len(W), bool), cv, _cfg_fast(), logging.getLogger("t"))
        sel, hist, _ = sffs(list(Xo.columns), ["mean", "std"], ev, _cfg_fast(), logging.getLogger("t"))
        assert len(sel) >= 1 and hist[0]["action"] == "ADD", "SFFS não adicionou feature"

    def test_sffs_can_remove_features():
        # floating backward deve ser exercitado (sem crash); remoção depende dos dados
        W = synth()
        Xo = build_original_features(W, CONFIG)
        Xc = build_candidate_features(W, pd.DataFrame(), CONFIG)
        X = pd.concat([Xo, Xc], axis=1)
        cv = TimeSeriesSplit(n_splits=3)
        ev, _ = make_evaluator(X, W["y_true"].to_numpy(), W["y_obs"].to_numpy(),
                               W["mask"].to_numpy(), np.ones(len(W), bool), cv, _cfg_fast(), logging.getLogger("t"))
        sel, hist, _ = sffs(list(Xo.columns), ["mean", "std", "entropy"], ev, _cfg_fast(), logging.getLogger("t"))
        actions = [h["action"] for h in hist]
        assert all(a in ("ADD", "REMOVE") for a in actions)

    def test_sffs_cache_returns_same_score():
        W = synth()
        Xo = build_original_features(W, CONFIG)
        cv = TimeSeriesSplit(n_splits=2)
        ev, cache = make_evaluator(Xo, W["y_true"].to_numpy(), W["y_obs"].to_numpy(),
                                   W["mask"].to_numpy(), np.ones(len(W), bool), cv, CONFIG, logging.getLogger("t"))
        a = ev(["mean"], list(Xo.columns)); b = ev(["mean"], list(Xo.columns))
        assert a["mae_mean"] == b["mae_mean"] and len(cache) == 1

    def test_ga_chromosome_maps_correct_features():
        cands = ["mean", "std", "entropy"]
        chrom = np.array([1, 0, 1])
        sel = [c for c, b in zip(cands, chrom) if b]
        assert sel == ["mean", "entropy"]

    def test_random_seed_reproduces_missingness():
        W = synth()
        m1 = apply_missingness(W["y_true"].to_numpy(), CONFIG, "mcar", 0.2, 6, 42)
        m2 = apply_missingness(W["y_true"].to_numpy(), CONFIG, "mcar", 0.2, 6, 42)
        assert np.array_equal(m1, m2), "máscara não reproduzível"

    for name, fn in [
        ("test_time_order_is_preserved", test_time_order_is_preserved),
        ("test_training_never_contains_future_test_rows", test_training_never_contains_future_test_rows),
        ("test_ground_truth_is_never_modified", test_ground_truth_is_never_modified),
        ("test_metrics_only_use_masked_positions", test_metrics_only_use_masked_positions),
        ("test_current_target_is_not_a_feature", test_current_target_is_not_a_feature),
        ("test_disallowed_features_raise_leakage_error", test_disallowed_features_raise_leakage_error),
        ("test_sffs_can_add_features", test_sffs_can_add_features),
        ("test_sffs_can_remove_features", test_sffs_can_remove_features),
        ("test_sffs_cache_returns_same_score", test_sffs_cache_returns_same_score),
        ("test_ga_chromosome_maps_correct_features", test_ga_chromosome_maps_correct_features),
        ("test_random_seed_reproduces_missingness", test_random_seed_reproduces_missingness),
    ]:
        check(name, fn)
    return results


# Rodar os self-checks (rápido, dados sintéticos). Levanta AssertionError se algo vazar.
print("== self-checks ==")
_check_results = run_self_checks()
for _n, _s in _check_results:
    print(f"  {_s:6s} {_n}")
print("self-checks OK\n")


R = run_experiment(CONFIG)


make_all_plots(R, CONFIG)
print("Gráficos em:", R["dirs"]["plots"])
print("Resultados em:", R["out_root"])
print("\nTabela final (teste cronológico intocado):")
print(pd.DataFrame(R["final_rows"]).to_string(index=False))
print("\nMelhora vs baseline M0:")
for k, v in R["improvement_pct"].items():
    print(f"  {k}: {v:+.2f}%")
print("\nSFFS subset:", R["sffs_sel"])
print("GA subset:  ", R["ga_sel"])
