-	Tiền xử lý dữ liệu: Loại bỏ các cột không cần thiết, ánh xạ các giá trị tấn công thành các nhóm chính (DoS, Probe, U2R, R2L, và Normal), chuyển đổi các thuộc tính dạng chuỗi thành số.
-	Chia tách dữ liệu: Sử dụng train_test_split để chia dữ liệu thành tập huấn luyện và tập kiểm thử.
-	 Huấn luyện mô hình: Huấn luyện các mô hình Naive Bayes, Decision Tree, và Random Forest.
-	Đánh giá mô hình: Sử dụng các thước đo như độ chính xác (Accuracy), và báo cáo phân loại (Classification Report) để đánh giá hiệu suất của mô hình trên cả tập huấn luyện và tập kiểm thử.
Từ đó thấy dược rằng:
-	Random Forest có hiệu suất cao nhất với độ chính xác tốt trên cả tập huấn luyện và tập kiểm thử, nhờ khả năng tổng hợp dự đoán từ nhiều cây quyết định.
-	Decision Tree có độ chính xác thấp hơn một chút so với Random Forest, nhưng vẫn cho kết quả tốt, đặc biệt là với việc sử dụng độ sâu hợp lý.
-	Naive Bayes có tốc độ huấn luyện nhanh nhưng độ chính xác thấp hơn so với các mô hình khác, do sự giả định đơn giản của nó về tính độc lập của các thuộc tính.
 

