# Đồ Án: Nghiên cứu khoa học
## Giới Thiệu
> - Tên đề tài: Hệ thống phát hiện tấn công DDoS dựa trên AI
> - Tập dữ liệu: CIC-DDoS2019
> - Mục tiêu: Xây dựng được một hệ thống phát hiện DDoS và có khả năng tự động chặn ip tấn công
> - Hệ thống được triển khai trên hệ điều hành linux

## Tải và sử dụng hệ thống

- Hệ thống sẽ chạy trên ip local thông qua port 5000
- Bắt buộc phải chạy hệ thống với người dùng root
- Nên sử dụng venv 
- Bạn có thể thử mô phỏng tấn công DDoS để kiểm thử hệ thống (sử dụng hping3 hoặc cách khác)

```
$ git clone https://github.com/Yairoo04/CodeAIDDoS_CNPM.git
$ cd NCKH_Code_AIOT/NCKH_PhatHien_PhanLoai_CICIoT2023
$ python3 -m venv myvenv
$ sudo su
$ source myvenv/bin/active
$ pip install -r requirements.txt
$ python3 app/app.py
```
## Mô phỏng tấn công
Ở đây tôi sử dụng hping3 để mô phỏng tấn công từ một máy bên ngoài
```
hping3 -d 10000 --flood 192.168.91.136 -p 5000
hping3 --icmp --flood 192.168.91.136
```