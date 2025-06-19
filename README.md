- Tải và cài đặt Miniconda trên Ubuntu 24.04:
  
```bash
# download
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# run
bash Miniconda3-latest-Linux-x86_64.sh
# delete
rm Miniconda3-latest-Linux-x86_64.sh
```

Làm theo hướng dẫn trên màn hình để hoàn tất cài đặt. Sau khi cài đặt xong, khởi động lại terminal hoặc chạy:

- Cài đặt GCC (GNU Compiler Collection) để biên dịch mã nguồn C/C++, (còn được gọi là g++): 

```bash
sudo apt update
sudo apt install build-essential -y
```
Kiểm tra
```bash
gcc --version
```
- Cài đặt NCCL (NVIDIA Collective Communications Library) để hỗ trợ giao tiếp giữa các GPU, 

```bash
sudo apt install libnccl2 libnccl-dev -y
```
kiểm tra

```bash
dpkg -l | grep nccl
```

3. **Xóa môi trường Conda cũ (nếu có):**

```bash
conda deactivate
conda env remove -n myenv
```

4. **Tạo môi trường Conda mới:**

```bash
conda create -n myenv python=3.8 -y
```
5. **Kích hoạt môi trường Conda:**

```bash
conda activate myenv
```

6. **Cài PyTorch hỗ trợ CUDA 11.1:**

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install cudnn if necessary.
conda install cudnn -c conda-forge

```

Sử dụng phiên bản 2.1.0 để phù hợp với mmcv 2.1.0 chứ không nó phải lên 2.2.0 mới phù hợp và như vậy thì lại không cài được mmDetection.
xem https://pytorch.org/get-started/locally/ để biết thêm chi tiết.

phiên bản torchvision phù hợp với pytorch 2.1.0 xem tại https://pypi.org/project/torchvision/

7. **Kiểm tra cài đặt PyTorch:**

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())

```
Nếu thấy phiên bản PyTorch và thông tin GPU, thì cài đặt đã thành công Pytorch và CUDA, nếu cài đặt cuDNN thì cũng sẽ có thông tin về cuDNN (tang cường hiệu suất cho các mô hình học sâu).

9. **Cài đặt openMim để quản lý các mô hình và công cụ của MMDetection:**

```bash
pip install -U openmim
```

10. **Cài đặt mmEngine, một thư viện cơ sở cho các dự án của OpenMMLab:**

```bash
mim install mmengine
```

11. **Cài đặt mmCV (OpenMMLab Computer Vision Foundation):**

```bash
mim install mmcv==1.2.4
```
Khi cài mmDetection 2.11.0 thì nó yêu cầu mmcv-full>= 1.2.4, <1.4.0 nếu không là nó báo lỗi.

12. **Cài đặt cpython để hỗ trợ biên dịch các gói Python:**

```bash
conda install -c conda-forge cython
```

13. Tạo thư mục dự án ETV (End to End Table Vision)

```bash
mkdir ~/ETV
cd ~/ETV
```

14. Clone dự án `mmDetection 2.11.0` và dự án `mmOCR 0.2.0` về và cài đặt

```bash
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmocr.git
cd mmdetection
pip install -v -e .
cd ../mmocr
pip install -v -e .
```
