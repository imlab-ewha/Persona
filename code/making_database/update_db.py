from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# DB 접속 정보
DB_USER = "pdp"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "persona"

# 입력 변수
CSV_FILE_PATH = "/home/imlab/Persona/data/processed2/persona_profile_filled.csv"
TABLE_NAME = "persona_profile"

# PostgreSQL 엔진 생성
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def load_csv_to_postgres(csv_file_path: str, table_name: str) -> None:
    csv_path = Path(csv_file_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_file_path}")

    if not csv_path.is_file():
        raise ValueError(f"지정한 경로가 파일이 아닙니다: {csv_file_path}")

    print(f"CSV 읽는 중: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"DB 적재 중: {csv_path.name} -> {table_name}")
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",   # 기존 테이블 있으면 삭제 후 다시 생성
        index=False
    )

    print(f"완료: {table_name} ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    load_csv_to_postgres(CSV_FILE_PATH, TABLE_NAME)