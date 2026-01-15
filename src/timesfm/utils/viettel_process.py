import pandas as pd
import os
import sys

# --- CẤU HÌNH NGƯỜI DÙNG ---
TARGET_CELL = 'enodebB8'           # Tên trạm cần xuất dữ liệu
INPUT_FILE = 'viettel.csv'         # Tên file gốc

# Định nghĩa đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SỬA LẠI Ở ĐÂY:
# Từ 'utils' lùi ra 3 cấp: utils -> timesfm -> src -> (Project Root)
# Sau đó mới đi vào datasets/viettel
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../datasets/viettel'))

# Hoặc nếu bạn muốn dùng đường dẫn tuyệt đối cho chắc chắn (Hardcode), hãy dùng dòng dưới đây (bỏ comment):
# DATA_DIR = '/home/myvh07/hoanglmv/Project/timesfm/datasets/viettel'

# Đảm bảo folder tồn tại
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------------------------------
# ... (Phần còn lại của code giữ nguyên)
# -------------------------------------------------------------------------------------

def extract_minutes(time_str):
    """Hàm xử lý chuỗi phút giây: '45:00.0' -> lấy 45"""
    try:
        return int(str(time_str).split(':')[0])
    except:
        return 0

def load_and_process_data():
    """Đọc file gốc và xử lý cột timestamp chuẩn"""
    file_path = os.path.join(DATA_DIR, INPUT_FILE)
    
    print(f"[1/3] Đang đọc dữ liệu từ: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path)
    print(f"   -> Đã load {len(df)} dòng.")

    print("[2/3] Đang xử lý cột thời gian (Time Mapping)...")
    # 1. Chuyển đổi Date Hour
    df['base_time'] = pd.to_datetime(df['date_hour'], format='%Y-%m-%d-%H')
    
    # 2. Xử lý phút
    df['minute_offset'] = df['update_time'].apply(extract_minutes)
    
    # 3. Tạo timestamp hoàn chỉnh
    df['timestamp'] = df['base_time'] + pd.to_timedelta(df['minute_offset'], unit='m')
    
    # 4. Sắp xếp và dọn dẹp
    df_clean = df.drop(columns=['base_time', 'minute_offset'])
    df_clean = df_clean.sort_values(by=['cell_name', 'timestamp'])
    
    return df_clean

def export_cell_data(df_clean, target_cell_name):
    """Lọc trạm và lưu ra file CSV riêng"""
    print(f"[3/3] Đang xuất dữ liệu trạm {target_cell_name} ra CSV...")

    output_filename = f'{target_cell_name}.csv'
    
    # Lọc dữ liệu trạm đích
    df_export = df_clean[df_clean['cell_name'] == target_cell_name].copy()

    if df_export.empty:
        print(f"❌ Lỗi: Không tìm thấy dữ liệu cho trạm {target_cell_name}.")
        return None
    
    # Chọn các cột quan trọng
    cols_to_keep = [
        'timestamp',                # Thời gian
        'ps_traffic_mb',            # Traffic
        'avg_rrc_connected_user',   # User
        'prb_dl_used',              # Tài nguyên mạng
        'prb_dl_available_total'    # Tài nguyên tổng
    ]
    
    # Chỉ giữ lại các cột có thực trong file
    existing_cols = [col for col in cols_to_keep if col in df_export.columns]
    df_export = df_export[existing_cols]

    # Lưu file
    full_path = os.path.join(DATA_DIR, output_filename)
    df_export.to_csv(full_path, index=False)

    print(f"✅ ĐÃ LƯU THÀNH CÔNG!")
    print(f"📂 File: {output_filename}")
    print(f"📍 Đường dẫn: {full_path}")
    print(f"📊 Kích thước: {df_export.shape[0]} dòng, {df_export.shape[1]} cột")
    
    print("\nXem trước dữ liệu:")
    print(df_export.head())

# --- MAIN ---
if __name__ == "__main__":
    df_clean = load_and_process_data()
    export_cell_data(df_clean, TARGET_CELL)
    print("\n=== HOÀN TẤT ===")