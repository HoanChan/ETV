<details>
<summary>1. Hướng dẫn cài đặt môi trường</summary>

1. Tải và cài đặt Miniconda trên Ubuntu 24.04:

```bash
# download
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# run
bash Miniconda3-latest-Linux-x86_64.sh
# delete
rm Miniconda3-latest-Linux-x86_64.sh
```

Làm theo hướng dẫn trên màn hình để hoàn tất cài đặt. Sau khi cài đặt xong, khởi động lại terminal hoặc chạy:

2. Cài đặt GCC (GNU Compiler Collection) để biên dịch mã nguồn C/C++, (còn được gọi là g++): 

```bash
sudo apt update
sudo apt install build-essential -y
```
Kiểm tra
```bash
gcc --version
```
3. Cài đặt NCCL (NVIDIA Collective Communications Library) để hỗ trợ giao tiếp giữa các GPU, 

```bash
sudo apt install libnccl2 libnccl-dev -y
```
kiểm tra

```bash
dpkg -l | grep nccl
```

4. **Xóa môi trường Conda cũ (nếu có):**

```bash
conda deactivate
conda env remove -n myenv
```

5. **Tạo môi trường Conda mới:**

```bash
conda create -n myenv python=3.8 -y
```
6. **Kích hoạt môi trường Conda:**

```bash
conda activate myenv
```
</details>
<details>
<summary>2. Cài đặt PyTorch và các thư viện cần thiết</summary>
<details>
<summary>Phiên bản cũ</summary>

1. **Cài PyTorch hỗ trợ CUDA 11.1:**

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install cudnn if necessary.
conda install cudnn -c conda-forge

```

Sử dụng phiên bản 2.1.0 để phù hợp với mmcv 2.1.0 chứ không nó phải lên 2.2.0 mới phù hợp và như vậy thì lại không cài được mmDetection.
xem https://pytorch.org/get-started/locally/ để biết thêm chi tiết.

phiên bản torchvision phù hợp với pytorch 2.1.0 xem tại https://pypi.org/project/torchvision/

2. **Kiểm tra cài đặt PyTorch:**

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())

```
Nếu thấy phiên bản PyTorch và thông tin GPU, thì cài đặt đã thành công Pytorch và CUDA, nếu cài đặt cuDNN thì cũng sẽ có thông tin về cuDNN (tang cường hiệu suất cho các mô hình học sâu).

3. **Cài đặt openMim để quản lý các mô hình và công cụ của MMDetection:**

```bash
pip install -U openmim
```

4. **Cài đặt mmEngine, một thư viện cơ sở cho các dự án của OpenMMLab:**

```bash
mim install mmengine
```

5. **Cài đặt mmCV (OpenMMLab Computer Vision Foundation):**

```bash
mim install mmcv==1.2.4
```
Khi cài mmDetection 2.11.0 thì nó yêu cầu mmcv-full>= 1.2.4, <1.4.0 nếu không là nó báo lỗi.

6. **Cài đặt cpython để hỗ trợ biên dịch các gói Python:**

```bash
conda install -c conda-forge cython
```

7. Tạo thư mục dự án ETV (End to End Table Vision)

```bash
mkdir ~/ETV
cd ~/ETV
```
8. Cài cpython để hỗ trợ biên dịch các gói Python:

```bash
pip install cython==0.29.33
```
dùng đúng phiên bản 0.29.33 để tương thích với mmcv 1.2.4 và mmDetection 2.11.0.

9. Clone dự án `mmDetection 2.11.0` và dự án `mmOCR 0.2.0` về và cài đặt

```bash
git clone --branch v2.11.0 https://github.com/open-mmlab/mmdetection.git
git clone --branch v0.2.0 https://github.com/open-mmlab/mmocr.git
cd mmdetection
pip install -v -e .
cd ../mmocr
pip install -v -e .
```
</details>
<details>
<summary>Phiên bản mới</summary>

1. **Cài PyTorch hỗ trợ CUDA 11.8:**

```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

Sử dụng phiên bản 2.1.0 để phù hợp với mmcv 2.1.0 chứ không nó phải lên 2.2.0 mới phù hợp và như vậy thì lại không cài được mmDetection.
xem https://pytorch.org/get-started/locally/ để biết thêm chi tiết.

phiên bản torchvision phù hợp với pytorch 2.1.0 xem tại https://pypi.org/project/torchvision/

2. **Kiểm tra cài đặt PyTorch:**

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```
Nếu bạn thấy phiên bản PyTorch và thông tin GPU, thì cài đặt đã thành công.

3. **Cài đặt cuDNN (NVIDIA CUDA Deep Neural Network library): (Option)**

```bash
conda install cudnn -c conda-forge
```

4. **Cài đặt openMim để quản lý các mô hình và công cụ của MMDetection:**

```bash
pip install -U openmim
```

5. **Cài đặt mmEngine, một thư viện cơ sở cho các dự án của OpenMMLab:**

```bash
mim install mmengine
```

6. **Cài đặt mmCV (OpenMMLab Computer Vision Foundation):**

```bash
mim install mmcv==2.1.0
```
Khi cài mmDetection thì nó yêu cầu mmcv<2.2.0,>=2.0.0rc4 nếu không là nó báo lỗi

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mmdet 3.3.0 requires mmcv<2.2.0,>=2.0.0rc4; extra == "mim", but you have mmcv 2.2.0 which is incompatible.

7. Tạo thư mục dự án ETV (End to End Table Vision)

```bash
mkdir ~/ETV
cd ~/ETV
```

8. Clone dự án mmDetection và dự án mmOCR về và cài đặt

```bash
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmocr.git
cd mmdetection
pip install -v -e .
cd ../mmocr
pip install -v -e .
```

</details>
</details>
<details>
<summary>3. Chuẩn bị dataset</summary>

# Tải dataset

Cài đặt gdown để tải file từ Google Drive
```bash
pip install gdown
```

## Tải Dataset ViTabNet

Chạy lệnh sau để tải dataset ViTabNet (đã preprocess) về:
```bash
curl -L -o vitabset.zip 'https://docs.google.com/uc?export=download&id=1dwbYq5nbUj_0rqiGeuCjGRGwHKiKs2VR'
```
hoặc
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dwbYq5nbUj_0rqiGeuCjGRGwHKiKs2VR' -O vitabset.zip
```

Sau đó giải nén bằng cách chạy lệnh:

```bash
unzip vitabset.zip
```

## Tải Dataset ViTabNet đã Preprecessing

Chạy lệnh sau để tải dataset ViTabNet (đã preprocess) về:
```bash
curl -L -o vitabset_preprocess.zip 'https://docs.google.com/uc?export=download&id=1o_fCCeYqv3_j2ccbP4oEpC4kVZr480PS'
```
hoặc
```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o_fCCeYqv3_j2ccbP4oEpC4kVZr480PS' -O vitabset_preprocess.zip
```

Sau đó giải nén bằng cách chạy lệnh:

```bash
unzip vitabset_preprocess.zip -d /preprocess/
```

</details>