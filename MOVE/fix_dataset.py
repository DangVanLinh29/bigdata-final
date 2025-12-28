import os
import json
import glob

# Cấu hình đường dẫn (Sửa lại nếu khác)
DATA_ROOT = "data/MOVE_release"
ANNO_DIR = os.path.join(DATA_ROOT, "annotations")
FRAME_DIR = os.path.join(DATA_ROOT, "frames")

def fix_dataset():
    print("🚑 ĐANG KIỂM TRA VÀ SỬA LỖI DATASET...")
    
    # Lấy danh sách file json
    json_files = glob.glob(os.path.join(ANNO_DIR, "*.json"))
    if not json_files:
        print("❌ Không tìm thấy file annotation nào!")
        return

    fixed_count = 0
    for j_path in json_files:
        try:
            with open(j_path, 'r') as f:
                data = json.load(f)
            
            # Lấy tên video từ tên file json
            video_name = os.path.splitext(os.path.basename(j_path))[0]
            video_frame_dir = os.path.join(FRAME_DIR, video_name)
            
            if not os.path.exists(video_frame_dir):
                print(f"⚠️ Cảnh báo: Không thấy thư mục ảnh cho video {video_name}")
                continue
                
            # Đếm số lượng file ảnh thực tế (.jpg)
            real_frames = len(glob.glob(os.path.join(video_frame_dir, "*.jpg")))
            
            # So sánh với độ dài trong JSON
            json_length = data.get('length', 0)
            
            if real_frames != json_length:
                print(f"🔧 Sửa video '{video_name}': JSON báo {json_length} -> Thực tế có {real_frames}")
                
                # Cập nhật lại length trong JSON
                data['length'] = real_frames
                
                # Cập nhật lại end_frame của các hành động (để không trỏ ra ngoài)
                for obj in data.get('objects', []):
                    for action in obj.get('actions', []):
                        # Nếu end_frame vượt quá số lượng ảnh thực, kéo nó về cuối video
                        if action['end_frame'] >= real_frames:
                            action['end_frame'] = real_frames - 1
                            
                # Lưu đè lại file JSON
                with open(j_path, 'w') as f:
                    json.dump(data, f, indent=4)
                fixed_count += 1
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {j_path}: {e}")

    print("-" * 50)
    print(f"✅ HOÀN TẤT! Đã sửa lỗi cho {fixed_count} video.")
    print("Bây giờ em có thể chạy Train lại bình thường!")

if __name__ == "__main__":
    fix_dataset()