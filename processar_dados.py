from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "dados"
OUTPUT_DIR = ROOT_DIR / "saida"
OUTPUT_FILE = OUTPUT_DIR / "producao_maritima_tratada.csv"

ENCODINGS = ("utf-8", "utf-8-sig", "iso-8859-1")


def normalize_column_name(name: str) -> str:
    #remove acentuações, caracteres especiais e normaliza para snake_case
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    #remove letras maiúsculas
    text = text.strip().lower()
    #troca / ou () por _
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    #aliases para nomes com colunas muito grandes, remove junções desnecessárias (de, para, da, etc.)
    aliases = {
        "mes_ano": "mes_ano",
        "producaooleometroscubicos": "producao_oleo_m3",
        "producao_de_condensado_m3": "producao_condensado_m3",
        "producao_de_gas_associado_mm3": "producao_gas_associado_mm3",
        "producao_de_gas_nao_associado_mm3": "producao_gas_nao_associado_mm3",
        "producao_de_agua_m3": "producao_agua_m3",
        "injecao_de_gas_mm3": "injecao_gas_mm3",
        "injecao_de_agua_para_recuperacao_secundaria_m3": "injecao_agua_recuperacao_secundaria_m3",
        "injecao_de_agua_para_descarte_m3": "injecao_agua_descarte_m3",
        "injecao_de_gas_carbonico_mm3": "injecao_gas_carbonico_mm3",
        "injecao_de_nitrogenio_mm3": "injecao_nitrogenio_mm3",
        "injecao_de_vapor_de_agua_t": "injecao_vapor_agua_t",
        "injecao_de_polimeros_m3": "injecao_polimeros_m3",
        "injecao_de_outros_fluidos_m3": "injecao_outros_fluidos_m3",
    }

    return aliases.get(text, text)


def load_csv(path: Path) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=str)
            df.columns = [normalize_column_name(col) for col in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Nao foi possivel decodificar {path.name}")


def parse_br_number(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip()
    clean = clean.replace({"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA})

    virgula = clean.str.contains(",", na=False)
    ponto = clean.str.contains(r"\.", na=False)

    # Quando ha ',' e '.', assume formato brasileiro com '.' de milhar.
    clean = clean.where(~(virgula & ponto), clean.str.replace(".", "", regex=False))
    clean = clean.str.replace(",", ".", regex=False)

    return pd.to_numeric(clean, errors="coerce")


def transform(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    required_columns = [
        "ano",
        "mes_ano",
        "estado",
        "bacia",
        "campo",
        "poco",
        "ambiente",
        "instalacao",
        "producao_oleo_m3",
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

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[required_columns].copy()

    text_columns = ["estado", "bacia", "campo", "poco", "ambiente", "instalacao"]
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA})

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["competencia"] = pd.to_datetime(df["mes_ano"], format="%m/%Y", errors="coerce")

    numeric_columns = [
        "producao_oleo_m3",
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
    for col in numeric_columns:
        df[col] = parse_br_number(df[col])

    df["fonte_arquivo"] = source_file
    return df


def main() -> None:
    input_files = sorted(INPUT_DIR.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {INPUT_DIR}")

    transformed_frames = []
    for file_path in input_files:
        raw = load_csv(file_path)
        transformed = transform(raw, file_path.name)
        transformed_frames.append(transformed)

    final_df = pd.concat(transformed_frames, ignore_index=True)
    final_df = final_df.drop_duplicates()
    final_df = final_df.sort_values(["competencia", "estado", "bacia", "campo", "poco"], na_position="last")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Arquivos processados: {len(input_files)}")
    print(f"Registros finais: {len(final_df)}")
    print(f"Saida gerada em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
