#!/usr/bin/env python3
import sys
import json

current_label = None
total_count = 0
correct_count = 0
total_frames = 0

print("---------- FINAL REPORT (DMA MAPREDUCE) ----------")
print(f"{'CATEGORY':<20} | {'VIDEOS':<10} | {'FRAMES':<10} | {'ACCURACY':<10}")
print("-" * 60)

def extract_ground_truth(filename):
    try:
        # Lấy phần đầu trước dấu gạch dưới làm nhãn
        return filename.split('_')[0].lower()
    except:
        return "unknown"

for line in sys.stdin:
    line = line.strip()
    if not line: continue

    try:
        # Input từ Mapper: PREDICTED_LABEL [TAB] JSON_DATA
        # VD: Run    {"filename": "Run_001.mp4", "label": "Run", "conf": 0.98, ...}
        pred_label, json_str = line.split('\t', 1)
        data = json.loads(json_str)
        
        # --- LOGIC GOM NHÓM (SHUFFLE & SORT) ---
        if current_label != pred_label:
            if current_label:
                # --- TÍNH TOÁN ĐỘ CHÍNH XÁC (ACCURACY CALCULATION) ---
                # Khớp với khối R3 trong hình vẽ
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
                
                # Xuất báo cáo (Khớp khối R4)
                print(f"{current_label:<20} | {total_count:<10} | {total_frames:<10} | {accuracy:.2f}%")

            # Reset cho nhóm mới
            current_label = pred_label
            total_count = 0
            correct_count = 0
            total_frames = 0
        
        # --- TỔNG HỢP THỐNG KÊ (AGGREGATION) ---
        # Khớp với khối R2 trong hình vẽ
        total_count += 1
        
        # Lấy metadata nếu có
        if 'meta' in data and 'frames' in data['meta']:
            total_frames += data['meta']['frames']
        elif 'frames' in data: # Fallback
            total_frames += data['frames']

        # Kiểm tra đúng sai (Prediction vs Ground Truth)
        ground_truth = extract_ground_truth(data.get('filename', ''))
        predicted = data.get('label', '').lower()
        
        if predicted in ground_truth or ground_truth in predicted:
            correct_count += 1

    except ValueError:
        continue

# In nhóm cuối cùng
if current_label:
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
    print(f"{current_label:<20} | {total_count:<10} | {total_frames:<10} | {accuracy:.2f}%")

print("-" * 60)
print("---------- END OF REPORT ----------")