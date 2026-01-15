import pandas as pd
import os
import sys

# --- CẤU HÌNH ---
INPUT_FILE = 'viettel.csv'         # Tên file gốc

# Định nghĩa đường dẫn
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến folder datasets/viettel
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../datasets/viettel'))

# Đường dẫn folder Output (để chứa hàng loạt file CSV sau khi tách)
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed_cells')

# Đảm bảo folder tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    print(f"[1/3] Đang đọc dữ liệu tổng từ: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {file_path}")
        sys.exit(1)

    # Đọc file CSV
    df = pd.read_csv(file_path)
    print(f"   -> Đã load {len(df)} dòng dữ liệu thô.")

    print("[2/3] Đang xử lý cột thời gian (Time Mapping)...")
    
    # 1. Chuyển đổi Date Hour
    df['base_time'] = pd.to_datetime(df['date_hour'], format='%Y-%m-%d-%H')
    
    # 2. Xử lý phút
    df['minute_offset'] = df['update_time'].apply(extract_minutes)
    
    # 3. Tạo timestamp hoàn chỉnh
    df['timestamp'] = df['base_time'] + pd.to_timedelta(df['minute_offset'], unit='m')
    
    # 4. Sắp xếp và dọn dẹp cột thừa
    df_clean = df.drop(columns=['base_time', 'minute_offset'])
    
    # Sắp xếp theo tên trạm và thời gian
    df_clean = df_clean.sort_values(by=['cell_name', 'timestamp'])
    
    return df_clean

def export_all_cells(df_clean):
    """Tách và lưu dữ liệu của TẤT CẢ các trạm ra từng file CSV riêng biệt"""
    
    # Lấy danh sách các trạm duy nhất
    unique_cells = df_clean['cell_name'].unique()
    total_cells = len(unique_cells)
    
    print(f"[3/3] Tìm thấy {total_cells} trạm. Đang tiến hành xuất file...")
    print(f"📂 Thư mục lưu trữ: {OUTPUT_DIR}")

    # Các cột cần giữ lại
    cols_to_keep = [
        'timestamp',                
        'ps_traffic_mb',            
        'avg_rrc_connected_user',   
        'prb_dl_used',              
        'prb_dl_available_total'    
    ]
    
    # Kiểm tra cột nào thực sự tồn tại trong file
    existing_cols = [col for col in cols_to_keep if col in df_clean.columns]

    # Sử dụng GroupBy để xử lý nhanh hơn thay vì lọc từng lần
    grouped = df_clean.groupby('cell_name')
    
    count = 0
    for cell_name, df_cell in grouped:
        count += 1
        
        # Tên file sạch (tránh lỗi ký tự đặc biệt nếu có)
        safe_name = str(cell_name).replace('/', '_').replace('\\', '_')
        output_filename = f'{safe_name}.csv'
        full_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Chỉ lấy các cột cần thiết
        df_export = df_cell[existing_cols]
        
        # Lưu file
        df_export.to_csv(full_path, index=False)
        
        # In tiến trình (ví dụ: cứ mỗi 10 trạm thì in 1 lần cho đỡ rối màn hình)
        if count % 10 == 0 or count == total_cells:
            print(f"   Processed {count}/{total_cells}: {output_filename} ({len(df_export)} rows)")

    print(f"\n✅ ĐÃ HOÀN TẤT! Xuất thành công {count} file.")

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load và xử lý dữ liệu chung
    df_main = load_and_process_data()
    
    # 2. Xuất tất cả các trạm
    export_all_cells(df_main)
    
    print("\n=== KẾT THÚC ===")