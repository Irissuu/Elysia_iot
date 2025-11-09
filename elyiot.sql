create table elysia_events (
    id         number generated always as identity primary key,
    ts         timestamp default current_timestamp,
    moto_id    varchar2(50),
    type       varchar2(20),   
    payload    clob
);

create index idx_elysia_events_moto on elysia_events (moto_id);

select
    id,
    to_char(ts, 'yyyy-mm-dd hh24:mi:ss') as data_hora,
    moto_id,
    type,
    payload
from elysia_events
order by id desc
fetch first 20 rows only;

select
    moto_id,
    type,
    count(*) as total
from elysia_events
group by moto_id, type
order by moto_id, type;

select *
from (
    select 
        id, moto_id, type, payload,
        row_number() over (partition by moto_id order by id desc) as rn
    from elysia_events
)
where rn = 1
order by moto_id;
