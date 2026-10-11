"""
Preparação da base: limpeza de categorias e separação treino/teste.

Este módulo é a única fonte da verdade sobre como os dados são tratados.
O treino do modelo (dia 3) e a API (dias 4 e 5) importam daqui, para que
a mesma limpeza seja aplicada em todo lugar.

Uso (a partir da raiz do repositório):
    python -m src.data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "credit_default.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sufixo numérico = mês: 1 é setembro/2005 (mais recente), 6 é abril/2005.
CATEGORICAL = ["sex", "education", "marriage"]
PAY_STATUS = [f"pay_status_{i}" for i in range(1, 7)]  # código de situação do pagamento
BILL_AMT = [f"bill_amt_{i}" for i in range(1, 7)]      # valor da fatura (NT$)
PAY_AMT = [f"pay_amt_{i}" for i in range(1, 7)]        # valor efetivamente pago (NT$)
NUMERIC = ["credit_limit", "age"] + PAY_STATUS + BILL_AMT + PAY_AMT
FEATURES = CATEGORICAL + NUMERIC


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode antes: python scripts/download_data.py"
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige códigos não documentados nas variáveis categóricas.

    Segundo a documentação da UCI:
      education: 1 pós-graduação, 2 universidade, 3 ensino médio, 4 outros
      marriage:  1 casado, 2 solteiro, 3 outros
    Mas a base traz education em {0, 5, 6} e marriage = 0, que não existem
    na documentação. Como são poucos casos, agrupamos em "outros".
    """
    df = df.copy()
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste estratificando pelo alvo.

    A estratificação garante a mesma taxa de inadimplência nas duas partes.
    O random_state fixo garante que a separação seja sempre a mesma.
    """
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )


def main() -> None:
    df = clean(load_raw())
    train, test = split(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    print(f"Treino: {len(train):,} linhas | inadimplência {train[TARGET].mean():.2%}")
    print(f"Teste:  {len(test):,} linhas | inadimplência {test[TARGET].mean():.2%}")
    print(f"Arquivos salvos em {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
