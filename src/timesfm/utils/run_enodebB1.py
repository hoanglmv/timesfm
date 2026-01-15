import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path("/home/myvh07/hoanglmv/Project/timesfm")
DATA_FILE = BASE_DIR / "datasets/viettel/enodebB1.csv"
FIGURE_DIR = BASE_DIR / "figures"
NODE_NAME = "enodebB1"

# Các cột cần dự báo
TARGET_COLS = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used']

# Cấu hình thời gian
FREQ = '15min'
POINTS_PER_DAY = 24 * 4  # 96 điểm dữ liệu một ngày
CONTEXT_DAYS = 7        # Nhìn lại 14 ngày (Input)
HORIZON_DAYS = 1         # Dự báo 1 ngày tiếp theo (Output)

CONTEXT_LEN = CONTEXT_DAYS * POINTS_PER_DAY
HORIZON_LEN = HORIZON_DAYS * POINTS_PER_DAY

# --- PHẦN 1: HÀM XỬ LÝ DỮ LIỆU ---

def load_and_preprocess_data(file_path):
    """
    Đọc CSV, xử lý trùng lặp timestamp, chuẩn hóa tần suất 15p và nội suy dữ liệu thiếu.
    """
    print(f"Đang đọc dữ liệu từ: {file_path}")
    df = pd.read_csv(file_path)
    
    # Chuyển đổi cột timestamp sang dạng datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Xóa các dòng timestamp bị trùng lặp (giữ lại dòng đầu tiên)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Đặt timestamp làm index
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 2. Thực hiện resample để đảm bảo dữ liệu liên tục mỗi 15 phút
    df_resampled = df.resample(FREQ).interpolate(method='linear')
    
    # Loại bỏ các hàng vẫn còn NaN ở đầu/cuối
    df_final = df_resampled.dropna()
    
    print(f"Dữ liệu sau khi xử lý: {len(df_final)} điểm dữ liệu (từ {df_final.index.min()} đến {df_final.index.max()})")
    return df_final

def smooth_data(df, cols, window_length=11, polyorder=3):
    """
    Làm mượt dữ liệu sử dụng Savitzky-Golay filter.
    """
    print("Đang thực hiện làm mượt dữ liệu (Smoothing)...")
    df_smoothed = df.copy()
    for col in cols:
        smoothed_values = savgol_filter(df[col].values, window_length=window_length, polyorder=polyorder)
        df_smoothed[col] = np.clip(smoothed_values, a_min=0, a_max=None)
    return df_smoothed

# --- PHẦN 2: HÀM DỰ BÁO VÀ VẼ HÌNH ---

def run_forecasting_and_plot(df_smoothed, df_raw):
    """
    Thực hiện dự báo trên dữ liệu đã làm mượt, nhưng vẽ cả dữ liệu gốc để đối chiếu.
    """
    min_required_len = CONTEXT_LEN + HORIZON_LEN
    if len(df_smoothed) < min_required_len:
        print(f"LỖI: Dữ liệu không đủ dài. Cần tối thiểu {min_required_len} điểm. Hiện có: {len(df_smoothed)}")
        return

    # --- CẬP NHẬT: TẠO TÊN THƯ MỤC THEO FORMAT YÊU CẦU ---
    # Format: {Tên Node}_{Số ngày Input}In_{Số ngày Output}Out
    folder_name = f"{NODE_NAME}_{CONTEXT_DAYS}In_{HORIZON_DAYS}Out"
    save_dir = FIGURE_DIR / folder_name
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Kết quả sẽ được lưu tại: {save_dir}")

    try:
        import timesfm
    except ImportError:
        print("LỖI: Chưa cài đặt thư viện timesfm.")
        return

    print("Đang tải mô hình TimesFM (PyTorch)...")
    
    # Kiểm tra GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # Load model
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        device=device 
    )
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LEN,
            max_horizon=HORIZON_LEN,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
        )
    )

    # --- Chuẩn bị dữ liệu ---
    # Cắt lấy đoạn dữ liệu cuối cùng cho cả dữ liệu gốc và dữ liệu mượt
    data_slice_smoothed = df_smoothed.tail(min_required_len)
    data_slice_raw = df_raw.tail(min_required_len) # Cắt tương ứng để vẽ đối chiếu
    
    # Tách tập train (lịch sử) và test (tương lai thực tế)
    # Input cho model là dữ liệu đã làm mượt
    train_data_smoothed = data_slice_smoothed.iloc[:CONTEXT_LEN]
    
    # Dữ liệu để vẽ đối chiếu (Raw History)
    train_data_raw = data_slice_raw.iloc[:CONTEXT_LEN]
    
    # Ground Truth (Thực tế tương lai - nên so sánh với Raw thực tế để xem độ chính xác thực)
    actual_future_data = data_slice_raw.iloc[CONTEXT_LEN:]
    
    timestamps_history = train_data_smoothed.index
    timestamps_future = actual_future_data.index

    # --- Thực hiện vòng lặp dự báo ---
    for target_col in TARGET_COLS:
        print(f"\n--- Đang xử lý KPI: {target_col} ---")
        
        input_signal = train_data_smoothed[target_col].values
        
        # Thực hiện dự báo
        point_forecast, _ = model.forecast(
            inputs=[input_signal],
            horizon=HORIZON_LEN
        )
        
        forecast_values = point_forecast[0]

        # --- Vẽ hình ---
        plt.figure(figsize=(15, 7))
        
        # 1. Vẽ dữ liệu GỐC (Lịch sử) - Màu xám nhạt, nằm dưới
        plt.plot(timestamps_history, train_data_raw[target_col], label='Dữ liệu Gốc (Nhiễu)', color='gray', alpha=0.3, linewidth=1)
        
        # 2. Vẽ dữ liệu ĐÃ LÀM MƯỢT (Lịch sử) - Màu xanh dương đậm
        plt.plot(timestamps_history, train_data_smoothed[target_col], label='Đã làm mượt (Input Model)', color='blue', alpha=0.8, linewidth=1.5)
        
        # 3. Vẽ dữ liệu THỰC TẾ (Tương lai) - Màu xanh lá
        plt.plot(timestamps_future, actual_future_data[target_col], label='Thực tế (Raw)', color='green', linewidth=2)
        
        # 4. Vẽ dữ liệu DỰ BÁO - Màu đỏ nét đứt
        plt.plot(timestamps_future, forecast_values, label='Dự báo TimesFM', color='red', linestyle='--', linewidth=2.5)
        
        plt.title(f'Dự báo KPI {target_col} - {NODE_NAME}\n(Input: {CONTEXT_DAYS} ngày | Output: {HORIZON_DAYS} ngày)')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        file_name = f"{NODE_NAME}_forecast_{target_col}.png"
        save_path = save_dir / file_name
        plt.savefig(save_path)
        print(f"Đã lưu biểu đồ: {save_path}")
        plt.close()

# --- MAIN ---
if __name__ == "__main__":
    if not DATA_FILE.exists():
        print(f"LỖI: Không tìm thấy file dữ liệu tại {DATA_FILE}")
        exit(1)
        
    # 1. Load dữ liệu gốc (đã resample/interpolate nhưng chưa smooth)
    df_raw = load_and_preprocess_data(DATA_FILE)
    
    # 2. Tạo bản sao làm mượt
    if len(df_raw) > 100:
         df_smoothed = smooth_data(df_raw, TARGET_COLS)
    else:
         print("CẢNH BÁO: Dữ liệu quá ngắn, bỏ qua bước làm mượt.")
         df_smoothed = df_raw.copy()

    # 3. Chạy dự báo (truyền cả 2 bản raw và smoothed)
    run_forecasting_and_plot(df_smoothed, df_raw)