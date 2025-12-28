import matplotlib.pyplot as plt
import re
import numpy as np
import os

# ==========================================
# CẤU HÌNH TÊN FILE LOG
# ==========================================
LOG_FILE_PATH = "training_log.txt"  # Đảm bảo file này chứa nội dung bạn vừa copy

def parse_log_file(file_path):
    episodes = []
    total_losses = []
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        print("👉 Hãy tạo file 'training_log.txt' và paste nội dung log vào đó nhé!")
        return [], []

    print("📂 Đang đọc file log...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Regex để tìm dòng chứa Episode và Total Loss
            # Mẫu: Episode [10/5000], ..., Total Loss: 12.3007, ...
            match_ep = re.search(r'Episode \[(\d+)/', line)
            match_loss = re.search(r'Total Loss: ([\d\.]+)', line)
            
            if match_ep and match_loss:
                episodes.append(int(match_ep.group(1)))
                total_losses.append(float(match_loss.group(1)))
                
    return episodes, total_losses

def draw_chart(episodes, losses):
    if not episodes: return

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(10, 6))

    # 1. Vẽ đường Loss gốc (Mờ hơn chút để làm nền)
    plt.plot(episodes, losses, color='#ffcccc', linewidth=1, label='Raw Loss')

    # 2. Vẽ đường Loss được làm mượt (Moving Average) để biểu đồ đẹp và chuyên nghiệp hơn
    # Giúp loại bỏ nhiễu (noise) dao động lên xuống
    window_size = 50 # Độ mượt (tùy chỉnh)
    if len(losses) > window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        # Điều chỉnh trục x cho khớp với dữ liệu đã làm mượt
        smooth_x = episodes[len(episodes)-len(smoothed_losses):]
        plt.plot(smooth_x, smoothed_losses, color='#d32f2f', linewidth=2, label='Smoothed Loss (Trend)')
    else:
        plt.plot(episodes, losses, color='#d32f2f', linewidth=2)

    # Trang trí biểu đồ
    plt.title('Training Loss Convergence (Episode 10 - 5000)', fontweight='bold', fontsize=12)
    plt.xlabel('Episodes', fontweight='bold')
    plt.ylabel('Total Loss', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Chú thích điểm đầu và cuối
    plt.scatter([episodes[0], episodes[-1]], [losses[0], losses[-1]], color='blue', zorder=5)
    plt.annotate(f'Start: {losses[0]:.2f}', (episodes[0], losses[0]), xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'End: {losses[-1]:.2f}', (episodes[-1], losses[-1]), xytext=(-10, 10), textcoords='offset points', ha='right')

    plt.tight_layout()
    plt.savefig('hinh_7_full_training.png', dpi=300)
    print(f"✅ Đã xử lý {len(episodes)} dòng log.")
    print("✅ Đã lưu biểu đồ: hinh_7_full_training.png")
    plt.show()

if __name__ == "__main__":
    eps, losses = parse_log_file(LOG_FILE_PATH)
    if eps:
        draw_chart(eps, losses)