#!/usr/bin/env python3
"""
Analise de Machine Learning da producao energetica maritima.

Modelos do trabalho:
- KNN Regressor
- Regressao Linear Simples
- Regressao Linear Multipla
- Regressao Logistica
"""

from __future__ import annotations

from pathlib import Path
import os
import warnings

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "resultados_ml"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_theme(style="whitegrid")

DATA_PATH = ROOT_DIR / "producao_maritima_tratada.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.30
SAMPLE_SIZE = None
MAX_SCATTER_POINTS = 3000
KNN_CV_SAMPLE_SIZE = 6000

TARGET = "producao_oleo_m3"
SIMPLE_FEATURE = "producao_gas_associado_mm3"
FEATURES = [
    "ano",
    "producao_condensado_m3",
    "producao_gas_associado_mm3",
    "producao_gas_nao_associado_mm3",
    "producao_agua_m3",
    "injecao_gas_mm3",
    "injecao_agua_recuperacao_secundaria_m3",
    "injecao_agua_descarte_m3",
    "injecao_gas_carbonico_mm3",
    "injecao_nitrogenio_mm3",
    "injecao_vapor_agua_t",
    "injecao_polimeros_m3",
    "injecao_outros_fluidos_m3",
]


def salvar_grafico(nome_arquivo: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / nome_arquivo, dpi=160, bbox_inches="tight")
    plt.close()


def carregar_dados() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    colunas = [TARGET] + FEATURES
    faltantes = [col for col in colunas if col not in df.columns]
    if faltantes:
        raise KeyError(f"Colunas ausentes no CSV: {faltantes}")

    df = df[colunas].copy()
    for col in colunas:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)


def remover_colunas_constantes(df: pd.DataFrame, features: list[str]) -> list[str]:
    features_validas = [col for col in features if df[col].nunique() > 1]
    removidas = sorted(set(features) - set(features_validas))
    if removidas:
        print(f"Colunas removidas por nao terem variacao: {removidas}")
    return features_validas


def metricas_regressao(y_real, y_previsto) -> dict[str, float]:
    return {
        "MSE": mean_squared_error(y_real, y_previsto),
        "MAE": mean_absolute_error(y_real, y_previsto),
        "R2": r2_score(y_real, y_previsto),
    }


def r2_ajustado(r2: float, n_amostras: int, n_features: int) -> float:
    return 1 - (1 - r2) * (n_amostras - 1) / (n_amostras - n_features - 1)


def imprimir_metricas_regressao(nome: str, metricas: dict[str, float]) -> None:
    print(f"\n{nome}")
    print(f"MSE: {metricas['MSE']:.2f}")
    print(f"MAE: {metricas['MAE']:.2f}")
    print(f"R2:  {metricas['R2']:.4f}")
    if "R2_ajustado" in metricas:
        print(f"R2 ajustado: {metricas['R2_ajustado']:.4f}")


def amostra_para_grafico(x, y) -> pd.DataFrame:
    pontos = pd.DataFrame({"real": np.asarray(x), "previsto": np.asarray(y)})
    if len(pontos) > MAX_SCATTER_POINTS:
        pontos = pontos.sample(MAX_SCATTER_POINTS, random_state=RANDOM_STATE)
    return pontos


def grafico_real_previsto(y_real, y_previsto, titulo: str, arquivo: str) -> None:
    pontos = amostra_para_grafico(y_real, y_previsto)
    limite_min = min(pontos["real"].min(), pontos["previsto"].min())
    limite_max = max(pontos["real"].max(), pontos["previsto"].max())

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pontos, x="real", y="previsto", alpha=0.45, s=18)
    plt.plot([limite_min, limite_max], [limite_min, limite_max], "r--", linewidth=2)
    plt.xlabel("Valor real")
    plt.ylabel("Valor previsto")
    plt.title(titulo)
    salvar_grafico(arquivo)


def grafico_residuos(y_real, y_previsto, titulo: str, arquivo: str) -> None:
    pontos = amostra_para_grafico(y_previsto, np.asarray(y_real) - np.asarray(y_previsto))
    pontos.columns = ["previsto", "residuo"]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pontos, x="previsto", y="residuo", alpha=0.45, s=18)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Valor previsto")
    plt.ylabel("Residuo")
    plt.title(titulo)
    salvar_grafico(arquivo)


def grafico_exploratorio(df: pd.DataFrame, features: list[str]) -> None:
    cols_corr = [TARGET] + features

    plt.figure(figsize=(11, 8))
    corr = df[cols_corr].corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.4)
    plt.title("Correlacao entre as variaveis")
    salvar_grafico("heatmap_correlacao.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=40, kde=True)
    plt.title("Distribuicao da producao de oleo")
    plt.xlabel("Producao de oleo (m3)")
    salvar_grafico("histograma_producao_oleo.png")


def regressao_linear_simples(X_train, X_test, y_train, y_test) -> dict[str, float]:
    modelo = LinearRegression()
    modelo.fit(X_train[[SIMPLE_FEATURE]], y_train)
    y_previsto = modelo.predict(X_test[[SIMPLE_FEATURE]])
    metricas = metricas_regressao(y_test, y_previsto)

    imprimir_metricas_regressao("2. Regressao Linear Simples", metricas)
    print(f"Feature usada: {SIMPLE_FEATURE}")
    print(f"Coeficiente: {modelo.coef_[0]:.6f}")
    print(f"Intercepto:  {modelo.intercept_:.6f}")

    pontos = pd.DataFrame(
        {SIMPLE_FEATURE: X_test[SIMPLE_FEATURE], TARGET: y_test, "previsto": y_previsto}
    )
    if len(pontos) > MAX_SCATTER_POINTS:
        pontos = pontos.sample(MAX_SCATTER_POINTS, random_state=RANDOM_STATE)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=pontos, x=SIMPLE_FEATURE, y=TARGET, alpha=0.35, s=18)
    ordem = np.argsort(pontos[SIMPLE_FEATURE].to_numpy())
    plt.plot(
        pontos[SIMPLE_FEATURE].to_numpy()[ordem],
        pontos["previsto"].to_numpy()[ordem],
        color="red",
        linewidth=2,
    )
    plt.title("Regressao Linear Simples")
    salvar_grafico("regressao_simples_reta.png")

    grafico_residuos(
        y_test,
        y_previsto,
        "Regressao Linear Simples - Residuos",
        "regressao_simples_residuos.png",
    )
    return metricas


def regressao_linear_multipla(X_train, X_test, y_train, y_test) -> dict[str, float]:
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_previsto = modelo.predict(X_test)
    metricas = metricas_regressao(y_test, y_previsto)
    metricas["R2_ajustado"] = r2_ajustado(metricas["R2"], len(y_test), X_test.shape[1])

    imprimir_metricas_regressao("3. Regressao Linear Multipla", metricas)

    X_train_ols = sm.add_constant(X_train, has_constant="add")
    ols = sm.OLS(y_train, X_train_ols).fit()
    coeficientes = pd.DataFrame(
        {
            "feature": ols.params.index,
            "coeficiente": ols.params.values,
            "erro_padrao": ols.bse.values,
            "p_valor": ols.pvalues.values,
        }
    ).sort_values("coeficiente", key=lambda s: s.abs(), ascending=False)
    coeficientes.to_csv(OUTPUT_DIR / "coeficientes_regressao_multipla.csv", index=False)
    print("\nPrincipais coeficientes:")
    print(coeficientes.head(8).to_string(index=False))

    grafico_real_previsto(
        y_test,
        y_previsto,
        "Regressao Linear Multipla - Real vs. Previsto",
        "regressao_multipla_real_vs_previsto.png",
    )
    grafico_residuos(
        y_test,
        y_previsto,
        "Regressao Linear Multipla - Residuos",
        "regressao_multipla_residuos.png",
    )
    return metricas


def knn_regressor(X_train, X_test, y_train, y_test) -> dict[str, float]:
    X_cv = X_train
    y_cv = y_train
    if len(X_cv) > KNN_CV_SAMPLE_SIZE:
        X_cv = X_cv.sample(KNN_CV_SAMPLE_SIZE, random_state=RANDOM_STATE)
        y_cv = y_train.loc[X_cv.index]

    k_values = list(range(1, 16, 2))
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    mse_por_k = []
    for k in k_values:
        modelo_cv = Pipeline(
            [
                ("padronizacao", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=k)),
            ]
        )
        scores = cross_val_score(
            modelo_cv, X_cv, y_cv, cv=cv, scoring="neg_mean_squared_error"
        )
        mse_por_k.append(-scores.mean())

    melhor_k = k_values[int(np.argmin(mse_por_k))]
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, mse_por_k, marker="o", linewidth=2)
    plt.xlabel("Valor de k")
    plt.ylabel("MSE medio na validacao cruzada")
    plt.title("KNN - Escolha de k")
    salvar_grafico("knn_mse_por_k.png")

    modelo = Pipeline(
        [
            ("padronizacao", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=melhor_k)),
        ]
    )
    modelo.fit(X_train, y_train)
    y_previsto = modelo.predict(X_test)
    metricas = metricas_regressao(y_test, y_previsto)

    imprimir_metricas_regressao("1. KNN Regressor", metricas)
    print(f"Melhor k pela validacao cruzada: {melhor_k}")
    grafico_real_previsto(
        y_test,
        y_previsto,
        "KNN - Real vs. Previsto",
        "knn_real_vs_previsto.png",
    )
    grafico_residuos(y_test, y_previsto, "KNN - Residuos", "knn_residuos.png")
    return metricas


def regressao_logistica(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    df_log = df.copy()
    df_log["produziu_oleo"] = (df_log[TARGET] > 0).astype(int)

    X = df_log[features]
    y = df_log["produziu_oleo"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelo = Pipeline(
        [
            ("padronizacao", StandardScaler()),
            (
                "logistica",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    modelo.fit(X_train, y_train)
    y_previsto = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "Acuracia": accuracy_score(y_test, y_previsto),
        "Precisao": precision_score(y_test, y_previsto, zero_division=0),
        "Recall": recall_score(y_test, y_previsto, zero_division=0),
        "F1": f1_score(y_test, y_previsto, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba),
    }

    print("\n4. Regressao Logistica")
    print("Target binario: produziu_oleo = 1 quando producao_oleo_m3 > 0")
    print(f"Distribuicao das classes: {y.value_counts().to_dict()}")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")
    print("\nRelatorio de classificacao:")
    print(classification_report(y_test, y_previsto, zero_division=0))

    cm = confusion_matrix(y_test, y_previsto)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Nao produziu", "Produziu"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Regressao Logistica - Matriz de Confusao")
    salvar_grafico("logistica_matriz_confusao.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metricas['AUC']:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("Taxa de falso positivo")
    plt.ylabel("Taxa de verdadeiro positivo")
    plt.title("Regressao Logistica - Curva ROC")
    plt.legend()
    salvar_grafico("logistica_roc.png")

    return metricas


def main() -> None:
    print("=" * 70)
    print("ANALISE DE MACHINE LEARNING - PRODUCAO MARITIMA")
    print("=" * 70)

    df = carregar_dados()
    features = remover_colunas_constantes(df, FEATURES)

    print(f"Base usada: {len(df)} linhas")
    print(f"Target de regressao: {TARGET}")
    print(f"Features usadas: {len(features)}")

    grafico_exploratorio(df, features)

    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    resultados = []

    knn = knn_regressor(X_train, X_test, y_train, y_test)
    resultados.append({"Modelo": "KNN Regressor", **knn})

    simples = regressao_linear_simples(X_train, X_test, y_train, y_test)
    resultados.append({"Modelo": "Regressao Linear Simples", **simples})

    multipla = regressao_linear_multipla(X_train, X_test, y_train, y_test)
    resultados.append({"Modelo": "Regressao Linear Multipla", **multipla})

    logistica = regressao_logistica(df, features)

    resumo_regressao = pd.DataFrame(resultados)
    resumo_classificacao = pd.DataFrame([{"Modelo": "Regressao Logistica", **logistica}])

    resumo_regressao.to_csv(OUTPUT_DIR / "resumo_modelos_regressao.csv", index=False)
    resumo_classificacao.to_csv(
        OUTPUT_DIR / "resumo_modelo_logistico.csv", index=False
    )

    print("\n" + "=" * 70)
    print("RESUMO - MODELOS DE REGRESSAO")
    print("=" * 70)
    print(resumo_regressao.to_string(index=False))

    print("\n" + "=" * 70)
    print("RESUMO - MODELO DE CLASSIFICACAO")
    print("=" * 70)
    print(resumo_classificacao.to_string(index=False))

    print(f"\nGraficos e tabelas salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
