**Yêu cầu kiểm tra và hoàn thiện mã nguồn + test**

Tôi đã chạy được task `test train`, nhưng chưa chắc chắn tính đúng đắn của toàn bộ mã nguồn và hệ thống kiểm thử hiện tại. Nhiệm vụ của bạn:

### 1. **Thông tin và chuẩn bị**

* Đọc nội dung ở #file:info.md  để nắm thông tin tổng quát về dự án.
* Đọc toàn bộ mã nguồn trong #file:src  (mã mới) và #file:old  (mã cũ).

### 2. **Yêu cầu công việc**

Đối với mỗi file trong `src`, thực hiện các bước sau:

#### a. **So sánh & kiểm tra chức năng**

* Đối chiếu với file tương ứng trong `old` để hiểu thay đổi.
* Kiểm tra xem chức năng hiện tại có đúng với yêu cầu không.
* Phát hiện lỗi hoặc đoạn code sai logic nếu có.

#### b. **Rà soát test**

* Kiểm tra các test hiện có liên quan đến file đang xét:

  * Test nào **lỗi thời** do code thay đổi → **Cập nhật lại**.
  * Test nào **dùng mock** → **Thay bằng đối tượng thực tế**.
  * Test nào **thiếu hoặc không hợp lý** → **Viết lại hoặc bổ sung mới**.

* **Thêm test mới** nếu file chưa có test.

* Test cần **ngắn gọn, rõ ràng, sát thực tế** và tập trung vào **chức năng chính**.

* **Không dùng hay tạo class** trong test, chỉ dùng hàm đơn giản, nếu thấy class thì bỏ đi. 

* Sử dụng **pytest.mark.parametrize** để giúp test ngắn gọn và dễ bảo trì.

* Tham khảo các file `.json` trong **vitabset** để biết dạng input cụ thể của dataset dùng cho test.

* Khi file test quá dài, cần xem lại các test có bị trùng lặp không, nếu có thì gộp lại cho ngắn gọn hơn.

#### c. **Báo cáo**

* Với mỗi nhóm file đã xử lý, tạo báo cáo trong thư mục **docs/report/**, nội dung viết bằng Tiếng Việt.

### 3. **Nguyên tắc**

* Quan trọng nhất là đảm bảo chức năng của code mới sẽ hoạt động như code cũ.
* Không được thay đổi logic của code, test sai thì sửa test, chỉ sửa code khi nó không đúng logic với code cũ.
* Không test lan man, dư thừa hoặc quá cứng nhắc.
* Chỉ test đúng các tính năng mà file đó chịu trách nhiệm, khi có sự kế thừa của các class thì chỉ cần test các chức năng mới hoặc khác biệt, không cần test lại toàn bộ.
* Ưu tiên tính rõ ràng, dễ hiểu, và dễ bảo trì.
* Toàn bộ code phải có test đầy đủ.

**Hãy bắt đầu thực hiện theo hướng dẫn trên, tuần tự từng file trong thư mục `src`.**
