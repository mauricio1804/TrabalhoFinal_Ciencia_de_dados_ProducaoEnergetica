import os
import logging

import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from mysql.connector import errorcode
import numpy as np

CSV_PATH = "./producao_maritima_tratada.csv"

DB_CONFIG = {
    "host": "localhost",
    "user": "mauricio",
    "password": "1234",
    "database": "producao_energetica",
}

PASTA_SAIDA = "resultados_mysql"
os.makedirs(PASTA_SAIDA, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(PASTA_SAIDA, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def carregar_dados():
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")

        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df["competencia"] = pd.to_datetime(df["competencia"], errors="coerce")

        colunas_numericas = [
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

        for coluna in colunas_numericas:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        df = df.dropna(subset=["ano", "estado", "bacia", "campo", "poco", "competencia", "fonte_arquivo"])
        df["ano"] = df["ano"].astype(int)

        logging.info("Dados carregados com sucesso.")
        return df

    except Exception as e:
        logging.error(f"Erro ao carregar dados: {e}")
        print("Erro ao carregar dados:", e)
        return None


def criar_estrutura():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS producao_energetica")
        cursor.execute("USE producao_energetica")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS producao_maritima (
                ano INT,
                mes_ano CHAR(7),
                estado VARCHAR(100),
                bacia VARCHAR(100),
                campo VARCHAR(120),
                poco VARCHAR(120),
                ambiente VARCHAR(40),
                instalacao VARCHAR(180),
                producao_oleo_m3 DECIMAL(18,5),
                producao_condensado_m3 DECIMAL(18,5),
                producao_gas_associado_mm3 DECIMAL(18,5),
                producao_gas_nao_associado_mm3 DECIMAL(18,5),
                producao_agua_m3 DECIMAL(18,5),
                injecao_gas_mm3 DECIMAL(18,5),
                injecao_agua_recuperacao_secundaria_m3 DECIMAL(18,5),
                injecao_agua_descarte_m3 DECIMAL(18,5),
                injecao_gas_carbonico_mm3 DECIMAL(18,5),
                injecao_nitrogenio_mm3 DECIMAL(18,5),
                injecao_vapor_agua_t DECIMAL(18,5),
                injecao_polimeros_m3 DECIMAL(18,5),
                injecao_outros_fluidos_m3 DECIMAL(18,5),
                competencia DATE,
                fonte_arquivo VARCHAR(255),
                PRIMARY KEY (competencia, estado, bacia, campo, poco, fonte_arquivo)
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except mysql.connector.Error as err:
        print("Erro no MySQL:", err)
        logging.error(f"Erro ao criar estrutura: {err}")
        return False


def salvar_no_mysql(df):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for _, linha in df.iterrows():
            # Converte NaN/NaT para None para o MySQL aceitar NULL.
            linha = linha.where(pd.notnull(linha), None)

            competencia = linha["competencia"]
            competencia = competencia.date() if pd.notnull(competencia) else None

            cursor.execute("""
                SELECT COUNT(*) FROM producao_maritima
                WHERE competencia = %s AND estado = %s AND bacia = %s AND campo = %s AND poco = %s AND fonte_arquivo = %s
            """, (competencia, linha["estado"], linha["bacia"], linha["campo"], linha["poco"], linha["fonte_arquivo"]))

            existe = cursor.fetchone()[0]

            dados = (
                linha["ano"],
                linha["mes_ano"],
                linha["estado"],
                linha["bacia"],
                linha["campo"],
                linha["poco"],
                linha["ambiente"],
                linha["instalacao"],
                linha["producao_oleo_m3"],
                linha["producao_condensado_m3"],
                linha["producao_gas_associado_mm3"],
                linha["producao_gas_nao_associado_mm3"],
                linha["producao_agua_m3"],
                linha["injecao_gas_mm3"],
                linha["injecao_agua_recuperacao_secundaria_m3"],
                linha["injecao_agua_descarte_m3"],
                linha["injecao_gas_carbonico_mm3"],
                linha["injecao_nitrogenio_mm3"],
                linha["injecao_vapor_agua_t"],
                linha["injecao_polimeros_m3"],
                linha["injecao_outros_fluidos_m3"],
                competencia,
                linha["fonte_arquivo"]
            )

            if existe:
                    cursor.execute("""
                    UPDATE producao_maritima SET
                        ano=%s, mes_ano=%s, ambiente=%s, instalacao=%s,
                        producao_oleo_m3=%s, producao_condensado_m3=%s,
                        producao_gas_associado_mm3=%s, producao_gas_nao_associado_mm3=%s,
                        producao_agua_m3=%s, injecao_gas_mm3=%s,
                        injecao_agua_recuperacao_secundaria_m3=%s, injecao_agua_descarte_m3=%s,
                        injecao_gas_carbonico_mm3=%s, injecao_nitrogenio_mm3=%s,
                        injecao_vapor_agua_t=%s, injecao_polimeros_m3=%s,
                        injecao_outros_fluidos_m3=%s
                    WHERE competencia=%s AND estado=%s AND bacia=%s AND campo=%s AND poco=%s AND fonte_arquivo=%s
                """, (
                    linha["ano"],
                    linha["mes_ano"],
                    linha["ambiente"],
                    linha["instalacao"],
                    linha["producao_oleo_m3"],
                    linha["producao_condensado_m3"],
                    linha["producao_gas_associado_mm3"],
                    linha["producao_gas_nao_associado_mm3"],
                    linha["producao_agua_m3"],
                    linha["injecao_gas_mm3"],
                    linha["injecao_agua_recuperacao_secundaria_m3"],
                    linha["injecao_agua_descarte_m3"],
                    linha["injecao_gas_carbonico_mm3"],
                    linha["injecao_nitrogenio_mm3"],
                    linha["injecao_vapor_agua_t"],
                    linha["injecao_polimeros_m3"],
                    linha["injecao_outros_fluidos_m3"],
                    competencia,
                    linha["estado"],
                    linha["bacia"],
                    linha["campo"],
                    linha["poco"],
                    linha["fonte_arquivo"],
                ))
            else:
                cursor.execute("""
                    INSERT INTO producao_maritima VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, dados)

        conn.commit()
        cursor.close()
        conn.close()

        return True

    except mysql.connector.Error as err:
        print("Erro no MySQL:", err)
        return False


def consultar_dados():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        df = pd.read_sql("SELECT * FROM producao_maritima", conn)
        conn.close()
        return df

    except Exception as e:
        print("Erro ao consultar dados:", e)
        return None


def calcular_estatisticas(df):
    if "producao_oleo_m3" not in df.columns:
        print("\nESTATISTICAS: coluna producao_oleo_m3 nao encontrada.")
        return

    df["producao_oleo_m3"] = pd.to_numeric(df["producao_oleo_m3"], errors="coerce")
    serie = df["producao_oleo_m3"].dropna()

    if serie.empty:
        print("\nESTATISTICAS: sem dados validos de producao_oleo_m3.")
        return

    stats = serie.describe().round(2)
    total = serie.sum()
    print("\nESTATISTICAS DE PRODUCAO DE OLEO (m3):")
    print(stats.to_string())
    print(f"Total acumulado (m3): {total:.2f}")


def grafico_barras(df):
    if df.empty:
        return

    df["producao_oleo_m3"] = pd.to_numeric(df["producao_oleo_m3"], errors="coerce")
    producao_ano = df.groupby("ano")["producao_oleo_m3"].sum()

    producao_ano = producao_ano[producao_ano > 0]
    if producao_ano.empty:
        return

    plt.figure(figsize=(9, 7))

    cores = plt.cm.tab20(np.linspace(0, 1, len(producao_ano)))

    plt.pie(
        producao_ano.values,
        labels=producao_ano.index.astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=cores
    )
    plt.title("\n Distribuicao da Producao de Oleo por Ano \n")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "grafico_pizza.png"))
    plt.close()


def grafico_area_empilhada(df):
    if df.empty:
        return

    df["producao_oleo_m3"] = pd.to_numeric(df["producao_oleo_m3"], errors="coerce")
    tabela = df.pivot_table(
        index="ano",
        columns="estado",
        values="producao_oleo_m3",
        aggfunc="sum",
        fill_value=0
    )

    if tabela.empty or tabela.shape[1] == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.stackplot(tabela.index, tabela.T.values, labels=tabela.columns)
    plt.title("Producao de Oleo por Ano e Estado")
    plt.xlabel("Ano")
    plt.ylabel("Producao de oleo (m3)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Estado")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_SAIDA, "grafico_area.png"))
    plt.close()


def main():
    df = carregar_dados()
    if df is None:
        return

    if not criar_estrutura():
        print("Erro ao conectar ao MySQL.")
        return

    salvar_no_mysql(df)

    df_db = consultar_dados()
    if df_db is None or df_db.empty:
        df_db = df

    calcular_estatisticas(df_db)
    grafico_barras(df_db)
    grafico_area_empilhada(df_db)

    print("Pipeline executado com sucesso!")


if __name__ == "__main__":
    main()