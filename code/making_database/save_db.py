import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# DB 접속 정보
DB_USER = "pdp"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "persona"

CSV_DIR = Path("/home/imlab/Persona/data/processed2")

# PostgreSQL 엔진 생성
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# CSV 전체 적재
# - 테이블 이름: csv 파일명 그대로
# - 기존 테이블 있으면 덮어쓰기
csv_files = sorted(CSV_DIR.glob("*.csv"))

if not csv_files:
    print("CSV 파일이 없습니다.")
else:
    print(f"총 {len(csv_files)}개 CSV 파일 발견")

for csv_path in csv_files:
    table_name = csv_path.stem  # 파일명 그대로, .csv 제외
    print(f"\n적재 중: {csv_path.name} -> {table_name}")

    df = pd.read_csv(csv_path, low_memory=False)

    # PostgreSQL에 적재
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",   # 기존 테이블 있으면 교체
        index=False
    )

    print(f"완료: {table_name} ({df.shape[0]} rows, {df.shape[1]} cols)")

print("\n모든 CSV 적재 완료")