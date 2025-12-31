import cv2
import numpy as np
fruit_colors = {
    "Tao Do": [([0, 150, 50], [10, 255, 255]), ([170, 150, 50], [180, 255, 255])],
    "Chuoi Vang": [([20, 100, 100], [35, 255, 255])],
    "Cam": [([10, 160, 150], [25, 255, 255])],
    "Oi": [([35, 50, 50], [85, 255, 255])],
    "Nho Tim": [([130, 50, 50], [165, 255, 255])]
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for name, ranges in fruit_colors.items():
        # Quét dải màu để tạo Mask
        if name == "Tao Do":
            mask = cv2.bitwise_or(cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1])),
                                  cv2.inRange(hsv, np.array(ranges[1][0]), np.array(ranges[1][1])))
        else:
            mask = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1]))

        # Khử nhiễu để khung bám sát hơn
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Tìm các đường bao (Contours) xung quanh vùng màu
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Chỉ lấy đường bao lớn nhất (chính là quả)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:  # Ngưỡng để tránh nhiễu li ti
                # Lấy tọa độ khung hình chữ nhật vừa khít nhất
                x, y, w, h = cv2.boundingRect(c)

                # Vẽ khung lable vừa khít lên vật thể
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Hiển thị tên ngay sát mép trên của khung
                cv2.putText(frame, f"{name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Quet mau vua khit hoa qua", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()