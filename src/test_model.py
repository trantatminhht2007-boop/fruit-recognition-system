from model import build_model
import tensorflow as tf

# 1. Khởi tạo model
print("--- Đang khởi tạo model MobileNetV2... ---")
model = build_model()

# 2. In tóm tắt cấu trúc
model.summary()

# 3. Kiểm tra các thông số quan trọng (Cầm tay chỉ việc)
trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

print(f"\n[CHECK] Trainable params: {trainable_count:,}")
print(f"\n[CHECK] Non-trainable params: {non_trainable_count:,}")

if trainable_count < 1000000:
    print("\n✅ KẾT QUẢ: Freeze base model THÀNH CÔNG. Bạn có thể yên tâm train!")
else:
    print("\n❌ CẢNH BÁO: Base model chưa được freeze hoàn toàn. Hãy kiểm tra lại dòng base_model.trainable = False")