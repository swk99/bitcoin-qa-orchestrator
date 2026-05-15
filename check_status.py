from btc_live_db import LiveSnapshotDB
db = LiveSnapshotDB.from_env()
for src in ['block_event', 'live', 'historical', 'historical_reconstructed']:
    n = db.count(source=src)
    if n > 0:
        print(f'{src:30s}: {n:,}개')
print(f'{"전체":30s}: {db.count():,}개')
db.close()
