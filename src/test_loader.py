from data_loader import load_datasets

# Đường dẫn tính từ thư mục gốc của dự án
PATH = "E:/fruit-recognition-system/dataset"

try:
    print("--- Đang kiểm tra Data Loader ---")
    train_ds, val_ds, class_names = load_datasets(PATH)
    
    print("\n--- KẾT QUẢ ---")
    print(f"Số lượng lớp tìm thấy: {len(class_names)}")
    print(f"Danh sách các loại quả: {class_names}")
    print("\n✅ Chúc mừng! Data loader đã chạy hoàn hảo.")
except Exception as e:
    print(f"\n❌ Lỗi rồi: {e}")