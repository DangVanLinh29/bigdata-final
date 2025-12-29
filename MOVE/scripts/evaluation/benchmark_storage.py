import time
import sqlite3
import json
import os
import pandas as pd
import numpy as np

# Cấu hình giả lập
NUM_RECORDS = 50000  # Số lượng bản ghi metadata video
DB_FILE = "test_sql.db"
NOSQL_FILE = "test_nosql.jsonl"

def benchmark_storage():
    # Dữ liệu mẫu (Metadata video)
    sample_data = {
        "video_id": "vid_001",
        "path": "/hdfs/data/videos/vid_001.mp4",
        "duration": 120,
        "obj": "Person",
        "action": "Running",
        "confidence": 0.98,
        "timestamp": time.time()
    }

    print(f"🚀 Đang Benchmark Lưu trữ với {NUM_RECORDS} bản ghi...")

    # ==========================================
    # 1. TEST SQL (Relational Database)
    # ==========================================
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Tạo bảng có Index (Mô phỏng môi trường thật)
    c.execute('''CREATE TABLE videos 
                 (id INTEGER PRIMARY KEY, video_id TEXT, path TEXT, 
                  duration REAL, obj TEXT, action TEXT, conf REAL)''')
    c.execute("CREATE INDEX idx_video_id ON videos (video_id)")
    
    # Đo tốc độ GHI (SQL)
    start_time = time.time()
    for i in range(NUM_RECORDS):
        c.execute("INSERT INTO videos (video_id, path, duration, obj, action, conf) VALUES (?, ?, ?, ?, ?, ?)",
                  (f"vid_{i}", sample_data['path'], sample_data['duration'], 
                   sample_data['obj'], sample_data['action'], sample_data['confidence']))
    conn.commit()
    sql_write_time = time.time() - start_time

    # Đo tốc độ ĐỌC/TRUY VẤN (SQL)
    start_time = time.time()
    c.execute("SELECT * FROM videos WHERE action = 'Running'")
    _ = c.fetchall()
    sql_read_time = time.time() - start_time
    conn.close()

    # ==========================================
    # 2. TEST NoSQL/Big Data (File System Append)
    # ==========================================
    if os.path.exists(NOSQL_FILE): os.remove(NOSQL_FILE)
    
    # Đo tốc độ GHI (NoSQL - Append Only)
    start_time = time.time()
    with open(NOSQL_FILE, 'w') as f:
        for i in range(NUM_RECORDS):
            record = sample_data.copy()
            record['video_id'] = f"vid_{i}"
            # Cơ chế Big Data thường ghi dạng dòng (Line-based) hoặc Batch
            f.write(json.dumps(record) + "\n")
    nosql_write_time = time.time() - start_time

    # Đo tốc độ ĐỌC/SCAN (NoSQL - MapReduce style)
    start_time = time.time()
    count = 0
    with open(NOSQL_FILE, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec['action'] == 'Running':
                count += 1
    nosql_read_time = time.time() - start_time

    # ==========================================
    # 3. TỔNG HỢP KẾT QUẢ
    # ==========================================
    # Giả lập Complexity (Độ phức tạp mở rộng)
    # SQL: O(N) hoặc O(logN) khi insert do phải update B-Tree Index
    # NoSQL: O(1) khi insert (Append only)
    
    results = {
        "Tiêu chí": ["Thời gian Ghi (Write Time)", "Tốc độ Ghi (Ops/sec)", "Thời gian Truy vấn (Query)", "Độ phức tạp Mở rộng"],
        "SQL (Truyền thống)": [
            f"{sql_write_time:.2f}s", 
            f"{int(NUM_RECORDS/sql_write_time)} rec/s",
            f"{sql_read_time:.4f}s",
            "Cao (Vertical Scaling)"
        ],
        "NoSQL (Big Data)": [
            f"{nosql_write_time:.2f}s", 
            f"{int(NUM_RECORDS/nosql_write_time)} rec/s",
            f"{nosql_read_time:.4f}s",
            "Thấp (Horizontal Scaling)"
        ]
    }
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f" BẢNG 4. SO SÁNH HIỆU NĂNG LƯU TRỮ (Dữ liệu thử nghiệm: {NUM_RECORDS} bản ghi)")
    print("="*80)
    print(df.to_string(index=False, justify='center'))
    print("-" * 80)
    
    # Dọn dẹp file rác
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    if os.path.exists(NOSQL_FILE): os.remove(NOSQL_FILE)

if __name__ == "__main__":
    benchmark_storage()