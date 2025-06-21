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
conda create -n myenv python=3.11.0 -y
```
6. **Kích hoạt môi trường Conda:**

```bash
conda activate myenv
```
</details>

<details>
<summary>2. Cài đặt PyTorch và các thư viện cần thiết</summary>

1. **Cài PyTorch hỗ trợ CUDA 11.8:**

```bash
pip install "numpy<2"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# Install cudnn if necessary.
conda install cudnn -c conda-forge
```
Cần cài đặt numpy phiên bản <2 để tương thích với pytorch 2.0.1. Nếu không thì nó lại tự cài numpy mới nhất và không tương thích với pytorch 2.0.1.

Xem chi tiết về các phiên bản PyTorch tại: https://pytorch.org/get-started/previous-versions/

Chọn Python 3.11.0, pyTorch 2.0.1 với CUDA 11.8, vì nó tương thích với mmEngine và mmCV.

| MMEngine Version         | PyTorch Version    | Python Version      |
|-------------------------|--------------------|---------------------|
| main                    | >=1.6, <=2.1       | >=3.8, <=3.11       |
| >=0.9.0, <=0.10.4       | >=1.6, <=2.1       | >=3.8, <=3.11       |

Xem thêm tại: https://github.com/open-mmlab/mmengine?tab=readme-ov-file#installation

2. **Kiểm tra cài đặt PyTorch:**

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```
Nếu thấy phiên bản PyTorch và thông tin GPU, thì cài đặt đã thành công.

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
mim install mmcv==2.0.1
```
Khi cài mmDetection mmOCR và mmCV cần đúng phiên bản tương thích với nhau, nếu không sẽ báo lỗi.
Xem thêm tại https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html

7. Tạo thư mục dự án ETV (End to End Table Vision)

```bash
mkdir ~/ETV
cd ~/ETV
```

8. Clone dự án mmDetection và dự án mmOCR về và cài đặt

```bash
git clone --brand 3.1.0  https://github.com/open-mmlab/mmdetection.git
git clone --brand 1.0.1  https://github.com/open-mmlab/mmocr.git
cd mmdetection
pip install -v -e .
cd ../mmocr
pip install -v -e .
```

9. Xoá thư mục .github trong 2 dự án để khỏi bị lỗi khi sync repo ở nhiều máy khác nhau:

```bash
rm -rf mmdetection/.github
rm -rf mmocr/.github
```

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

</details>