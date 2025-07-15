### Hướng dẫn cài đặt môi trường phát triển ViTable

<details>
<summary><b>1. Hướng dẫn cài đặt môi trường</b></summary>

1. **Tải và cài đặt Miniconda:**

    <details>
    <summary>Dành cho Unbuntu</summary>

    ```bash
    # download
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # run
    bash Miniconda3-latest-Linux-x86_64.sh
    # delete
    rm Miniconda3-latest-Linux-x86_64.sh
    ```
    </details>
    <details>
    <summary>Dành cho Windows</summary>

    Tải Miniconda từ trang chính thức: [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)
    Chọn phiên bản phù hợp với hệ điều hành của bạn (Windows 64-bit) và tải xuống tệp cài đặt. Sau đó, chạy tệp cài đặt và làm theo hướng dẫn hoặc chạy lệnh sau trong Command Prompt hoặc PowerShell:
    ```bash
    # download
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    # run
    start /wait Miniconda3-latest-Windows-x86_64.exe /S /D=C:\Miniconda3
    # delete
    del Miniconda3-latest-Windows-x86_64.exe
    ```

    Nếu không sử dụng được lệnh `conda`, cần thêm đường dẫn của Miniconda vào biến môi trường PATH.
    
    - Bước 1: Nhấn Windows + S, tìm: "Environment Variables" → Chọn Edit the system environment variables → Nhấn nút Environment Variables...
    - Bước 2: Trong phần User variables, chọn Path → nhấn Edit → Nhấn New, thêm các dòng sau: `D:\SOFTS\Miniconda`, `D:\SOFTS\Miniconda\Scripts`
    - Bước 3: Xác nhận Nhấn OK → OK → OK → Tắt CMD hoặc PowerShell cũ, mở mới → chạy: `conda --version` để kiểm tra.
    </details>

2. **Cài đặt một số thư viện cần thiết:**

    <details>
    <summary>Dành cho Ubuntu</summary>
    1. Cài đặt GCC (GNU Compiler Collection) để biên dịch mã nguồn C/C++, (còn được gọi là g++): 

    ```bash
    sudo apt update
    sudo apt install build-essential -y
    ```
    Kiểm tra
    ```bash
    gcc --version
    ```
    2. Cài đặt NCCL (NVIDIA Collective Communications Library) để hỗ trợ giao tiếp giữa các GPU, 

    ```bash
    sudo apt install libnccl2 libnccl-dev -y
    ```
    kiểm tra

    ```bash
    dpkg -l | grep nccl
    ```
    </details>
    <details>
    <summary>Dành cho Windows</summary>
    Cài đặt đủ driver NVIDIA mới nhất từ trang chủ của NVIDIA: [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx)
    Cài đặt Microsoft Visual C++ Redistributable để đảm bảo các thư viện cần thiết đã được cài đặt.
    Cài đặt Visual Studio Build Tools từ trang chính thức: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
        1. Chạy file cài đặt, chọn: "Desktop development with C++"
        2. Tick thêm "MSVC v14.x" (C++ build tools)
        3. Tick "Windows 10 SDK" (hoặc Windows 11 SDK)
        4. Nhấn Install/Modify để cài đặt.
        5. Sau khi cài xong, mở "x86_x64 Cross Tools Command Prompt for VS 2022".
        6. Kích hoạt môi trường conda, cài các package Python cần build (mmocr, lanms-neo, ...).
    </details>

3. **Thiết lập môi trường conda:**

    Xoá môi trường Conda cũ nếu có:

    ```bash
    conda deactivate
    conda env remove -n viTable
    ```

    Tạo môi trường Conda mới:

    ```bash
    conda create -n viTable python=3.9.23 -y
    ```
    Kích hoạt môi trường Conda:

    ```bash
    conda activate viTable
    ```

    Kiểm tra phiên bản Python:

    ```bash
    python --version
    ```
    Đảm bảo phiên bản `Python` là `3.9.x`, vì `viTable` yêu cầu `Python 3.9`.
</details>

<details>
<summary><b>2. Cài đặt PyTorch và các thư viện cần thiết</b></summary>

1. **Cài PyTorch hỗ trợ CUDA 11.8:**

    ```bash
    pip install "numpy>=1.21.0,<1.24.0"
    pip install "opencv-python<4.12.0"
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
    Cần cài đặt `numpy` phiên bản < 1.24 để tương thích vì `pytorch` phiên bản này ra đời trước `numpy 1.24`. `opencv-python < 4.12.0` để tránh việc nó tự cài `numpy 2.x` gây xung đột.

    Xem chi tiết về các phiên bản PyTorch tại: https://pytorch.org/get-started/previous-versions/

2. **Kiểm tra cài đặt PyTorch:**

    ```python
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    ```
    Nếu thấy phiên bản `PyTorch` và thông tin `GPU`, thì cài đặt đã thành công.
    Lưu ý là không cài `cudnn`, vì nó đã được cài sẵn trong `PyTorch` nhờ `pip` rồi.

3. **Cài đặt openMim để quản lý các mô hình và công cụ của MMDetection:**

    ```bash
    pip install -U openmim
    ```

5. **Cài đặt mmCV (OpenMMLab Computer Vision Foundation):**

    ```bash
    mim install "mmcv<1.8.0"
    ```
    Khi cài mmDetection mmOCR và mmCV cần đúng phiên bản tương thích với nhau, nếu không sẽ báo lỗi.
    Xem thêm tại https://github.com/open-mmlab/mmocr/blob/v0.6.3/docs/en/install.md

6. **Cài đặt mmDetection (OpenMMLab Object Detection Toolbox):**

    ```bash
    mim install "mmdet<3.0.0"
    ```
    Xem thêm tại https://mmdetection.readthedocs.io/en/latest/get_started.html

7. **Cài đặt mmOCR (OpenMMLab Optical Character Recognition Toolbox):**

    ```bash
    mim install "mmocr==0.6.3"
    ```
    Xem thêm tại https://github.com/open-mmlab/mmocr/blob/v0.6.3/docs/en/install.md

9. **Cài đặt các thư viện bổ sung cho việc tính TEDS:**

    ```bash
    pip install apted
    pip install distance
    ```

    Hai thư viện này sẽ được sử dụng để tính toán TEDS (Tree Edit Distance) trong quá trình đánh giá mô hình. Cụ thể là `apted` sẽ được sử dụng để tính toán TEDS giữa các cây cấu trúc của văn bản, trong khi `distance` sẽ hỗ trợ tính toán khoảng cách giữa các chuỗi ký tự.

10. **Xoá cache của pip, conda và openmim:**

    ```bash
    pip cache purge
    conda clean --all
    # poweshell
    Remove-Item -Path "$env:USERPROFILE\.cache\mim" -Recurse -Force
    # cmd
    rmdir /s /q %USERPROFILE%\.cache\mim
    # bash
    rm -rf ~/.cache/mim
    ```

</details>