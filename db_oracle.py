import oracledb, json

oracle_user = ""                  
oracle_pw   = ""               
oracle_dsn  = "oracle.fiap.com.br:1521/orcl"       

pool = oracledb.create_pool(
    user=oracle_user,
    password=oracle_pw,
    dsn=oracle_dsn,
    min=1, max=4, increment=1
)

def log_event(moto_id, ev_type, payload=None):
    data = json.dumps(payload or {})
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "insert into elysia_events (moto_id, type, payload) values (:m, :t, :p)",
                m=str(moto_id), t=ev_type, p=data
            )
        conn.commit()
