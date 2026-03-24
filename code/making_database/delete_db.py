import os
from sqlalchemy import create_engine, text

# DB 접속 정보
DB_USER = "pdp"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "persona"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

drop_all_tables_sql = """
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
"""

with engine.begin() as conn:
    conn.execute(text(drop_all_tables_sql))

print("public 스키마의 모든 테이블 삭제 완료")