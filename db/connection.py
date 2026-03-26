import psycopg2
from config import PG_DSN

def get_pg_conn():
    return psycopg2.connect(PG_DSN)
