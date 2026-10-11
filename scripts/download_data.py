"""
Baixa a base "Default of Credit Card Clients" do UCI Machine Learning Repository
e salva em data/raw/ como CSV limpo, pronto para a análise exploratória.

Fonte: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Licença: CC BY 4.0 (Yeh, I. C. & Lien, C. H., 2009)

Uso (a partir da raiz do repositório):
    python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_CSV = RAW_DIR / "credit_default.csv"

TARGET = "default"
EXPECTED_ROWS = 30_000

# Nomes originais da UCI -> nomes usados no projeto.
# Os nomes originais confundem: PAY_0 é um código de situação (não um valor pago),
# a sequência pula de PAY_0 para PAY_2 e o alvo tem espaços no nome.
# O sufixo numérico indica o mês: 1 = setembro/2005 (mais recente) ... 6 = abril/2005.
COLUMN_NAMES = {
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_status_1",
    **{f"PAY_{i}": f"pay_status_{i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
    "default payment next month": TARGET,
}


def download_zip(url: str) -> bytes:
    """Faz o download do zip e devolve os bytes."""
    print(f"Baixando base de {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def read_excel_from_zip(content: bytes) -> pd.DataFrame:
    """Abre o .xls que vem dentro do zip.

    A primeira linha do arquivo é um cabeçalho genérico (X1, X2, ...),
    por isso usamos header=1 para pegar os nomes reais das colunas.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xls_name = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        with zf.open(xls_name) as f:
            return pd.read_excel(io.BytesIO(f.read()), header=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove o ID (não é variável preditiva) e aplica nomes de colunas claros."""
    df = df.rename(columns=lambda c: c.strip())
    return df.drop(columns=["ID"]).rename(columns=COLUMN_NAMES)


def validate(df: pd.DataFrame) -> None:
    """Checagens mínimas para garantir que a base veio íntegra."""
    assert list(df.columns) == list(COLUMN_NAMES.values()), (
        f"Colunas inesperadas: {sorted(set(df.columns) ^ set(COLUMN_NAMES.values()))}"
    )
    assert len(df) == EXPECTED_ROWS, f"Esperava {EXPECTED_ROWS} linhas, veio {len(df)}"
    assert TARGET in df.columns, "Coluna alvo não encontrada"
    assert set(df[TARGET].unique()) == {0, 1}, "Alvo deveria ser binário (0/1)"
    assert df.isna().sum().sum() == 0, "Há valores ausentes inesperados"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists():
        print(f"Base já existe em {OUTPUT_CSV}. Apague o arquivo para baixar de novo.")
        return 0

    df = clean(read_excel_from_zip(download_zip(URL)))
    validate(df)
    df.to_csv(OUTPUT_CSV, index=False)

    taxa = df[TARGET].mean()
    print(f"Base salva em {OUTPUT_CSV}")
    print(f"Linhas: {len(df):,} | Colunas: {df.shape[1]} | Taxa de inadimplência: {taxa:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
