import json
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

# ======================================================
# 1. HÀM ĐỌC DỮ LIỆU THẬT TỪ FILE CỦA EM
# ======================================================
def load_real_hadoop_output(file_path):
    print(f"Dang doc du lieu that tu: {file_path} ...")
    
    logits_data = []
    labels_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Giả sử format output của Hadoop là: PREDICTED_LABEL \t JSON_STRING
                # Hoặc chỉ là JSON lines tùy cách em lưu. 
                # Ở đây thầy giả sử là JSON lines thuần túy cho dễ.
                try:
                    if "\t" in line:
                        _, json_str = line.split("\t", 1)
                    else:
                        json_str = line
                        
                    data = json.loads(json_str)
                    
                    # Lấy Logits và Ground Truth
                    if 'logits' in data and 'ground_truth_idx' in data:
                        logits_data.append(data['logits'])
                        labels_data.append(data['ground_truth_idx'])
                        
                except Exception as e:
                    continue
                    
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu! Đang dùng dữ liệu mẫu để demo...")
        # (Fallback: Tạo dữ liệu giả nếu em chưa có file thật để code không lỗi)
        return np.random.randn(100, 10), np.random.randint(0, 10, 100)

    print(f"Da tai thanh cong {len(logits_data)} mau du lieu that.")
    return np.array(logits_data), np.array(labels_data)

# Hàm Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ======================================================
# 2. CÁC THUẬT TOÁN (LAC, APS, RAPS)
# ======================================================
# (Giữ nguyên thuật toán chuẩn như code trước thầy gửi)

def get_lac_sets(probs, cal_labels, val_probs, alpha=0.1):
    n = len(cal_labels)
    cal_scores = 1 - probs[np.arange(n), cal_labels]
    q = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n)
    return val_probs >= (1 - q)

def get_aps_sets(probs, cal_labels, val_probs, alpha=0.1):
    n = len(cal_labels)
    # ... (Logic APS chuẩn - copy từ đoạn code trước vào đây) ...
    # Để code gọn thầy viết vắn tắt logic chính:
    def get_scores(p, y=None):
        idx = np.argsort(p, 1)[:, ::-1]
        sorted_p = np.take_along_axis(p, idx, 1)
        cumsum = np.cumsum(sorted_p, 1)
        if y is not None:
            ranks = np.argmax(idx == y[:, None], 1)
            return cumsum[np.arange(len(p)), ranks]
        return cumsum, idx

    cal_scores = get_scores(probs, cal_labels)
    q = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n)
    val_cumsum, val_idx = get_scores(val_probs)
    
    sets = np.zeros_like(val_probs, dtype=bool)
    for i in range(len(val_probs)):
        cut = np.searchsorted(val_cumsum[i], q) + 1
        cut = min(cut, len(val_probs[i])) 
        sets[i, val_idx[i, :cut]] = True
    return sets

# ======================================================
# 3. CHẠY THỰC NGHIỆM
# ======================================================
def run_real_evaluation():
    # --- CONFIG ---
    # Em thay đường dẫn này bằng file thật em tải từ HDFS về
    REAL_DATA_FILE = "output_from_hadoop.txt" 
    
    # 1. Load Data
    logits, labels = load_real_hadoop_output(REAL_DATA_FILE)
    probs = softmax(logits)
    
    # 2. Chia tập Calibration (để tính ngưỡng) và Test (để đo đạc)
    # Tỷ lệ 50-50 như báo cáo mẫu
    cal_probs, test_probs, cal_labels, test_labels = train_test_split(
        probs, labels, test_size=0.5, random_state=42
    )

    print(f"\n--- KẾT QUẢ TRÊN TẬP DỮ LIỆU THẬT ({len(test_labels)} mẫu test) ---")
    
    # 3. Chạy so sánh 3 thuật toán
    results = []
    methods = [
        ("LAC (Threshold)", get_lac_sets),
        ("APS (Adaptive)", get_aps_sets)
        # Em có thể thêm RAPS vào nếu muốn
    ]
    
    print(f"{'Method':<15} | {'Coverage (%)':<12} | {'Set Size':<10} | {'Time (s)':<10}")
    print("-" * 55)

    for name, func in methods:
        t0 = time.time()
        # Chạy thuật toán
        pred_sets = func(cal_probs, cal_labels, test_probs, alpha=0.1)
        dt = time.time() - t0
        
        # Tính metrics
        coverage = np.mean([pred_sets[i, test_labels[i]] for i in range(len(test_labels))]) * 100
        avg_size = np.mean([np.sum(p) for p in pred_sets])
        
        print(f"{name:<15} | {coverage:<12.2f} | {avg_size:<10.2f} | {dt:.4f}")
        results.append([name, coverage, avg_size, dt])

    # Xuất ra CSV để em copy vào báo cáo
    df = pd.DataFrame(results, columns=["Method", "Coverage", "Size", "Time"])
    df.to_csv("real_data_benchmark.csv", index=False)
    print("\nĐã lưu kết quả vào file 'real_data_benchmark.csv'.")

if __name__ == "__main__":
    run_real_evaluation()