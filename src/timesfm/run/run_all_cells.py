import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path("/home/myvh07/hoanglmv/Project/timesfm")
INPUT_DATA_DIR = BASE_DIR / "datasets/viettel/processed_cells"
FIGURE_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"

# Các cột cần dự báo
TARGET_COLS = ['ps_traffic_mb', 'avg_rrc_connected_user', 'prb_dl_used']

# Cấu hình thời gian
FREQ = '15min'
POINTS_PER_DAY = 24 * 4
CONTEXT_DAYS = 21
HORIZON_DAYS = 1

CONTEXT_LEN = CONTEXT_DAYS * POINTS_PER_DAY
HORIZON_LEN = HORIZON_DAYS * POINTS_PER_DAY

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- CÁC HÀM TIỆN ÍCH ---

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.set_index('timestamp').sort_index()
        # Resample và điền 0 cho các khoảng trống
        df_resampled = df.resample(FREQ).asfreq().fillna(0)
        return df_resampled
    except Exception as e:
        print(f"❌ Lỗi đọc file {file_path}: {e}")
        return None

def smooth_data(df, cols, window_length=11, polyorder=3):
    df_smoothed = df.copy()
    try:
        for col in cols:
            if col in df.columns:
                if df[col].sum() > 10: 
                    smoothed_values = savgol_filter(df[col].values, window_length=window_length, polyorder=polyorder)
                    df_smoothed[col] = np.clip(smoothed_values, a_min=0, a_max=None)
    except Exception:
        pass
    return df_smoothed

# --- CẬP NHẬT: HÀM TÍNH ACCURACY MỚI ---
def calculate_metrics(y_true, y_pred):
    """
    Tính toán metrics với công thức Accuracy mới:
    Accuracy = 1 - (MAE / Mean_Absolute_Actual)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Tính trung bình trị tuyệt đối thực tế
    mean_abs_true = np.mean(np.abs(y_true))
    epsilon = 1e-10 

    # Nếu dữ liệu thực tế toàn 0 (hoặc rất nhỏ), accuracy = 0
    if mean_abs_true < 0.1:
        return mae, rmse, 0.0, 0.0

    # 1. Tính MAPE cũ (để tham khảo)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # 2. Tính Accuracy theo công thức MỚI: 1 - (MAE / Mean_True)
    # Kết quả nhân 100 để ra phần trăm
    accuracy = (1 - (mae / mean_abs_true)) * 100
    
    # Giới hạn min là 0 (trường hợp sai số lớn hơn cả giá trị thực)
    accuracy = max(0, accuracy)
    
    return mae, rmse, mape, accuracy

# --- HÀM XỬ LÝ CHÍNH ---

def process_single_cell(file_path, model):
    cell_name = Path(file_path).stem
    
    df_raw = load_and_preprocess_data(file_path)
    if df_raw is None: return []
    
    min_len = CONTEXT_LEN + HORIZON_LEN
    if len(df_raw) < min_len:
        return []

    df_smoothed = smooth_data(df_raw, TARGET_COLS)

    data_slice_smoothed = df_smoothed.tail(min_len)
    data_slice_raw = df_raw.tail(min_len)
    
    train_data_smoothed = data_slice_smoothed.iloc[:CONTEXT_LEN]
    actual_future_data = data_slice_raw.iloc[CONTEXT_LEN:]
    
    if train_data_smoothed[TARGET_COLS].sum().sum() < 10:
        return [] 

    timestamps_future = actual_future_data.index
    cell_results = [] 

    for target_col in TARGET_COLS:
        if target_col not in df_raw.columns: continue

        y_true = actual_future_data[target_col].values
        
        if np.mean(y_true) < 0.5:
            continue

        input_signal = train_data_smoothed[target_col].values
        
        point_forecast, _ = model.forecast(inputs=[input_signal], horizon=HORIZON_LEN)
        forecast_values = point_forecast[0]

        # Gọi hàm tính metrics mới
        mae, rmse, mape, acc = calculate_metrics(y_true, forecast_values)
        
        result_row = {
            'Cell_Name': cell_name,
            'KPI': target_col,
            'Accuracy (%)': round(acc, 2),  # Đưa lên đầu cho dễ nhìn
            'MAE': round(mae, 2),
            'Data_Mean': round(np.mean(y_true), 2),
            'RMSE': round(rmse, 2),
            'MAPE_Ref (%)': round(mape, 2)  # Đổi tên để phân biệt
        }
        cell_results.append(result_row)
        
        if acc > 0:
            folder_name = f"{CONTEXT_DAYS}In_{HORIZON_DAYS}Out"
            save_dir = FIGURE_DIR / folder_name / cell_name
            os.makedirs(save_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps_future, y_true, label='Thực tế', color='green')
            plt.plot(timestamps_future, forecast_values, label='Dự báo', color='red', linestyle='--')
            plt.title(f"{cell_name} - {target_col}\nNew Accuracy: {acc:.1f}% (MAE: {mae:.2f})")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f"{target_col}.png")
            plt.close()

    return cell_results

# --- MAIN ---
if __name__ == "__main__":
    all_files = glob.glob(str(INPUT_DATA_DIR / "*.csv"))
    print(f"📂 Tìm thấy {len(all_files)} trạm.")

    if not all_files:
        print("❌ Không có file dữ liệu.")
        exit()

    try:
        import timesfm
        print("🚀 Đang tải Model TimesFM...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", device=device
        )
        model.compile(timesfm.ForecastConfig(
            max_context=CONTEXT_LEN, max_horizon=HORIZON_LEN,
            normalize_inputs=True, use_continuous_quantile_head=False
        ))
    except ImportError:
        print("❌ Lỗi thư viện.")
        exit()

    all_results = []
    print("⏳ Đang xử lý...")
    for file_path in tqdm(all_files):
        results = process_single_cell(file_path, model)
        all_results.extend(results)

    if all_results:
        df_report = pd.DataFrame(all_results)
        df_report = df_report.sort_values(by=['Accuracy (%)'], ascending=False)
        
        report_path = REPORT_DIR / "final_report_new_accuracy.csv"
        df_report.to_csv(report_path, index=False)
        
        print("\n" + "="*50)
        print(f"📊 Báo cáo mới đã lưu tại: {report_path}")
        print("="*50)
        
        # Hiển thị kết quả với cột Accuracy mới
        cols_to_show = ['Cell_Name', 'KPI', 'Accuracy (%)', 'MAE', 'Data_Mean']
        print(df_report[cols_to_show].head(10))
    else:
        print("\n⚠️ Không có kết quả.")