# TableMaster - Checklist 14 Ngày Đầu
## Cài đặt & Huấn luyện TableMaster

*Thời gian: 14 ngày (2 tuần)*  
*Ngày bắt đầu: 30/06/2025*  
*Ngày kết thúc dự kiến: 13/07/2025*

---

## 📅 **TUẦN 1: CÀI ĐẶT & TRIỂN KHAI CƠ BẢN (7 NGÀY)**

### **NGÀY 1: Cài đặt môi trường & Phân tích code**
**Ngày**: 30/06/2025  
**Mục tiêu**: Thiết lập môi trường và phân tích code có sẵn

#### ✅ **Cài đặt môi trường mmOCR 1.x**
- [x] Clone repository mmOCR
- [x] Checkout version v1.0.1
- [x] Cài đặt dependencies
- [x] Verify installation

#### ✅ **Phân tích code 0.x có sẵn**
- [x] Liệt kê tất cả các tệp trong repo MTL-TabNet
- [x] Xác định các thành phần cốt lõi: models, losses, datasets
- [x] Ghi lại các dependencies và requirements
- [x] Phân tích cấu trúc code hiện tại

#### ✅ **Thiết lập cấu trúc dự án**
- [x] Tạo thư mục `tablemaster_thesis/`
- [x] Tạo thư mục con: `configs/`, `mmocr_custom/`, `data/`, `tools/`, `experiments/`, `docs/`
- [x] Khởi tạo git repository
- [x] Tạo file README.md cơ bản

**Sản phẩm ngày 1**: ✅ Môi trường sẵn sàng + Cấu trúc dự án

---

### **NGÀY 2: Xác thực và chuẩn bị Dataset**
**Ngày**: 01/07/2025  
**Mục tiêu**: Kiểm tra và chuẩn bị dataset cho quá trình huấn luyện

#### ✅ **Xác thực dataset**
- [x] Kiểm tra định dạng dataset nhất quán với PubTabNet
- [x] Xác minh đường dẫn hình ảnh và tệp annotation
- [x] Kiểm tra tính toàn vẹn của dữ liệu
- [x] Validate annotation format

#### ⏳ **Chuẩn bị dataset**
- [ ] Tạo script phân chia dataset (train/val/test)
- [ ] Thực hiện phân chia dataset theo tỷ lệ 80/10/10
- [ ] Tạo báo cáo thống kê dataset (số lượng images, tables, cells)
- [ ] Viết script kiểm tra dataset integrity
- [ ] Tạo sample visualization script

#### ✅ **Thiết lập pipeline dữ liệu ban đầu**
- [x] Tạo data loader cơ bản
- [x] Kiểm tra định dạng input/output
- [x] Test data loading với sample nhỏ

**Sản phẩm ngày 2**: Dataset được xác thực + Pipeline dữ liệu cơ bản

---

### **NGÀY 3: Chuyển đổi Model TableMaster**
**Ngày**: 02/07/2025  
**Mục tiêu**: Triển khai model chính TableMaster sang mmOCR 1.x

#### ⏳ **Chuyển đổi model TableMaster**
- [ ] Tạo file `mmocr_custom/models/textrecog/recognizers/tablemaster.py`
- [ ] Implement class `TABLEMASTER(BaseRecognizer)`
- [ ] Chuyển đổi cài đặt từ 0.x sang cấu trúc 1.x
- [ ] Register model với decorator `@MODELS.register_module()`

#### ⏳ **Cài đặt architecture cơ bản**
- [ ] Định nghĩa `__init__()` method
- [ ] Implement `forward()` function
- [ ] Thiết lập input/output interfaces
- [ ] Kiểm tra compatibility với mmOCR 1.x framework

#### ⏳ **Tạo test cases cơ bản**
- [ ] Tạo file test cho model initialization
- [ ] Test model instantiation
- [ ] Test forward pass với dummy data
- [ ] Validate output shapes và types

**Sản phẩm ngày 3**: Model TableMaster cơ bản hoạt động

---

### **NGÀY 4: Chuyển đổi Backbone và Decoder**
**Ngày**: 03/07/2025  
**Mục tiêu**: Triển khai các thành phần backbone và decoder

#### ⏳ **Chuyển đổi backbone (TableResNetExtra)**
- [ ] Tạo file `mmocr_custom/models/textrecog/backbones/table_resnet_extra.py`
- [ ] Implement class `TableResNetExtra(BaseBackbone)`
- [ ] Triển khai logic Global Context Block
- [ ] Implement feature extraction layers
- [ ] Register backbone với `@MODELS.register_module()`

#### ⏳ **Chuyển đổi decoder**
- [ ] Tạo file `mmocr_custom/models/textrecog/decoders/tablemaster_decoder.py`
- [ ] Implement class `TableMasterDecoder(BaseDecoder)`
- [ ] Cài đặt Transformer decoder architecture
- [ ] Implement attention mechanisms
- [ ] Register decoder với `@MODELS.register_module()`

#### ⏳ **Tích hợp và test thành phần**
- [ ] Kiểm tra kết nối giữa backbone và decoder
- [ ] Test tensor shapes và data flow
- [ ] Validate forward pass của integrated model
- [ ] Debug integration issues

**Sản phẩm ngày 4**: Backbone và Decoder hoạt động + Integration tests

---

### **NGÀY 5: Cài đặt Loss Functions**
**Ngày**: 04/07/2025  
**Mục tiêu**: Triển khai các hàm loss đa nhiệm

#### ⏳ **Chuyển đổi các hàm loss sang ModuleLoss**
- [ ] Tạo thư mục `mmocr_custom/models/textrecog/module_losses/`
- [ ] Implement class `TableMasterModuleLoss(BaseModuleLoss)`
- [ ] Thiết kế multi-task loss architecture
- [ ] Register loss với `@MODELS.register_module()`

#### ⏳ **Cài đặt từng loại loss**
- [ ] Implement Structure loss (cho nhận dạng cấu trúc bảng)
- [ ] Implement Bbox regression loss (L1/L2 loss)
- [ ] Implement Text recognition loss (CrossEntropy)
- [ ] Implement loss weighting mechanism

#### ⏳ **Test và debug loss functions**
- [ ] Kiểm tra gradient flow
- [ ] Test với dữ liệu synthetic
- [ ] Validate loss computation
- [ ] Debug backward pass

**Sản phẩm ngày 5**: Các hàm Loss hoạt động + Test cases

---

### **NGÀY 6: Dataset và Data Transforms**
**Ngày**: 05/07/2025  
**Mục tiêu**: Cài đặt bộ tải dataset và các phép biến đổi

#### ⏳ **Cài đặt Dataset**
- [ ] Tạo file `mmocr_custom/datasets/table_dataset.py`
- [ ] Implement class `OCRTableDataset(BaseDataset)`
- [ ] Tải dữ liệu định dạng PubTabNet
- [ ] Chuyển đổi sang TextRecogDataSample format
- [ ] Register dataset với `@DATASETS.register_module()`

#### ⏳ **Các phép biến đổi dữ liệu**
- [ ] Implement TableResize transform
- [ ] Implement TablePad transform
- [ ] Implement TableBboxEncode transform
- [ ] Chuyển đổi sang định dạng transform của mmOCR 1.x
- [ ] Register transforms với registry

#### ⏳ **Test data pipeline**
- [ ] Kiểm tra tải dữ liệu
- [ ] Validate data augmentation
- [ ] Test data transformation pipeline
- [ ] Verify data sample format

**Sản phẩm ngày 6**: Bộ tải Dataset hoạt động + Data transforms

---

### **NGÀY 7: Cấu hình & Đăng ký**
**Ngày**: 06/07/2025  
**Mục tiêu**: Cài đặt các tệp cấu hình và đăng ký component

#### ⏳ **Tạo cấu hình cơ bản**
- [ ] Tạo file `configs/tablemaster/tablemaster_base.py`
- [ ] Migrate từ `table_master_ResnetExtract_Ranger_0705.py`
- [ ] Configure model architecture
- [ ] Set up training parameters
- [ ] Configure dataset paths

#### ⏳ **Cài đặt Registry**
- [ ] Tạo file `mmocr_custom/__init__.py`
- [ ] Import và register tất cả các component tùy chỉnh
- [ ] Ensure proper module discovery
- [ ] Test registry functionality

#### ⏳ **Kiểm tra tích hợp nhanh**
- [ ] Test model loading từ config
- [ ] Verify all components are registered
- [ ] Run integration test
- [ ] Debug any loading issues

**Command để test**:
```bash
python -c "
from mmocr_custom.models import TABLEMASTER
model = TABLEMASTER.from_file('configs/tablemaster/tablemaster_base.py')
print('✅ Model tải thành công')
"
```

**Sản phẩm ngày 7**: Cấu hình hoạt động + Tất cả component đã được đăng ký

---

## 📅 **TUẦN 2: HUẤN LUYỆN & ĐÁNH GIÁ (7 NGÀY)**

### **NGÀY 8: Kiểm tra Pipeline cơ bản**
**Ngày**: 07/07/2025  
**Mục tiêu**: Thiết lập và test pipeline huấn luyện với dữ liệu nhỏ

#### ⏳ **Kiểm tra huấn luyện quy mô nhỏ**
- [ ] Tạo subset dataset nhỏ (100 mẫu)
- [ ] Configure training với small dataset
- [ ] Run training command
- [ ] Monitor training progress

**Command để test**:
```bash
python tools/train.py configs/tablemaster/tablemaster_base.py \
    --work-dir work_dirs/test_small
```

#### ⏳ **Debug cơ bản**
- [ ] Kiểm tra model initialization
- [ ] Test forward pass
- [ ] Validate loss computation
- [ ] Debug common errors

#### ⏳ **Setup monitoring**
- [ ] Tích hợp logging cơ bản
- [ ] Theo dõi memory usage
- [ ] Monitor GPU utilization
- [ ] Set up progress tracking

**Sản phẩm ngày 8**: Pipeline cơ bản hoạt động với dữ liệu nhỏ

---

### **NGÀY 9: Debug và Tối ưu hóa Pipeline**
**Ngày**: 08/07/2025  
**Mục tiêu**: Gỡ lỗi và tối ưu hóa các vấn đề thường gặp

#### ⏳ **Gỡ lỗi các vấn đề thường gặp**
- [ ] Fix CUDA memory issues
- [ ] Resolve data loading errors
- [ ] Debug model forward/backward errors
- [ ] Fix loss computation issues
- [ ] Handle tensor shape mismatches

#### ⏳ **Cài đặt ghi log & giám sát nâng cao**
- [ ] Tích hợp TensorBoard
- [ ] Theo dõi Loss chi tiết (structure, bbox, text)
- [ ] Giám sát GPU utilization
- [ ] Set up checkpoint saving
- [ ] Configure logging levels

#### ⏳ **Tối ưu hóa performance**
- [ ] Optimize data loading (num_workers, pin_memory)
- [ ] Memory optimization (gradient accumulation)
- [ ] Speed profiling
- [ ] Batch size optimization

**Sản phẩm ngày 9**: Pipeline ổn định + Monitoring system

---

### **NGÀY 10: Tinh chỉnh siêu tham số**
**Ngày**: 09/07/2025  
**Mục tiêu**: Tối ưu hóa các siêu tham số cho huấn luyện

#### ⏳ **Tinh chỉnh siêu tham số**
- [ ] Optimize learning rate schedule
- [ ] Tối ưu hóa batch size
- [ ] Cân bằng trọng số loss (structure vs bbox vs text)
- [ ] Configure optimizer parameters
- [ ] Set up warmup strategy

#### ⏳ **Chuẩn bị config cuối cùng**
- [ ] Tạo config cho full training
- [ ] Setup checkpointing strategy
- [ ] Configure validation schedule
- [ ] Set up early stopping

#### ⏳ **Pre-training validation**
- [ ] Test với subset lớn hơn (1000 mẫu)
- [ ] Validate convergence behavior
- [ ] Check training stability
- [ ] Verify gradient flow

**Sản phẩm ngày 10**: Siêu tham số tối ưu + Config sẵn sàng cho full training

---

### **NGÀY 11: Bắt đầu huấn luyện đầy đủ**
**Ngày**: 10/07/2025  
**Mục tiêu**: Khởi động quá trình huấn luyện trên toàn bộ dataset

#### ⏳ **Bắt đầu huấn luyện đầy đủ**
- [ ] Prepare full dataset
- [ ] Launch full training

**Command để training**:
```bash
python tools/train.py configs/tablemaster/tablemaster_full.py \
    --work-dir work_dirs/tablemaster_full \
    --gpu-ids 0
```

#### ⏳ **Setup monitoring dashboard**
- [ ] Real-time loss tracking
- [ ] Training progress visualization
- [ ] Resource utilization monitoring
- [ ] Set up alerts for issues

#### ⏳ **Initial training supervision**
- [ ] Monitor first few epochs closely
- [ ] Check for any immediate issues
- [ ] Validate training progression
- [ ] Ensure checkpoints are saved

**Sản phẩm ngày 11**: Quá trình huấn luyện đầy đủ đã bắt đầu

---

### **NGÀY 12: Đánh giá Model**
**Ngày**: 11/07/2025  
**Mục tiêu**: Đánh giá hiệu suất model đã huấn luyện

#### ⏳ **Đánh giá model**
- [ ] Run evaluation on test set

**Command để evaluation**:
```bash
python tools/test.py configs/tablemaster/tablemaster_full.py \
    work_dirs/tablemaster_full/latest.pth
```

#### ⏳ **Thu thập các chỉ số hiệu suất**
- [ ] Đo độ chính xác nhận dạng cấu trúc
- [ ] Đo độ chính xác nhận dạng văn bản ô
- [ ] Đo các chỉ số hồi quy bbox (IoU, mAP)
- [ ] Đo độ chính xác tái tạo bảng tổng thể (TEDS)
- [ ] Calculate inference speed

#### ⏳ **Tạo báo cáo đánh giá**
- [ ] Create performance metrics table
- [ ] Generate visualization results
- [ ] Compare with baseline (if available)
- [ ] Document evaluation methodology

**Sản phẩm ngày 12**: Báo cáo đánh giá model + Performance metrics

---

### **NGÀY 13: Phân tích lỗi và Tối ưu hóa**
**Ngày**: 12/07/2025  
**Mục tiêu**: Phân tích chi tiết lỗi và tối ưu hóa

#### ⏳ **Phân tích lỗi chi tiết**
- [ ] Phân tích các trường hợp thất bại
- [ ] Xác định các lĩnh vực cần cải thiện
- [ ] Categorize error types (structure, bbox, text)
- [ ] Identify common failure patterns
- [ ] Create failure case gallery

#### ⏳ **Tối ưu hóa (nếu cần)**
- [ ] Fine-tune based on error analysis
- [ ] Adjust hyperparameters if needed
- [ ] Implement quick fixes
- [ ] Test improvements

#### ⏳ **Ghi lại các hạn chế**
- [ ] Document current limitations
- [ ] Identify future improvements
- [ ] Create failure case analysis report
- [ ] Suggest research directions

**Sản phẩm ngày 13**: Phân tích lỗi chi tiết + Model tối ưu hóa (nếu có)

---

### **NGÀY 14: Tài liệu & Dọn dẹp code**
**Ngày**: 13/07/2025  
**Mục tiêu**: Hoàn thiện cài đặt và tài liệu

#### ⏳ **Tài liệu hóa code**
- [ ] Thêm docstrings cho tất cả các class/hàm
- [ ] Tạo tài liệu API
- [ ] Viết ví dụ sử dụng
- [ ] Create README cho từng module

#### ⏳ **Ghi log thí nghiệm**
- [ ] Ghi lại tất cả các thí nghiệm
- [ ] Lưu lại các trọng số model tốt nhất
- [ ] Tạo các script inference
- [ ] Create experiment log

#### ⏳ **Chuẩn bị cho việc viết luận văn**
- [ ] Xuất training logs
- [ ] Tạo bảng/hình ảnh kết quả
- [ ] Tổ chức dữ liệu thực nghiệm
- [ ] Prepare visual materials

#### ⏳ **Code cleanup**
- [ ] Remove unused code
- [ ] Organize file structure
- [ ] Update requirements.txt
- [ ] Create final git commit

**Sản phẩm ngày 14**: Hoàn thành cài đặt + Tài liệu đầy đủ

---

## 📊 **TIẾN ĐỘ TỔNG QUAN**

### **Tuần 1 - Cài đặt & Triển khai**
- **Ngày 1**: ✅ Môi trường & Cấu trúc
- **Ngày 2**: ⏳ Dataset & Pipeline
- **Ngày 3**: ⏳ Model TableMaster
- **Ngày 4**: ⏳ Backbone & Decoder
- **Ngày 5**: ⏳ Loss Functions
- **Ngày 6**: ⏳ Dataset & Transforms
- **Ngày 7**: ⏳ Config & Registry

### **Tuần 2 - Huấn luyện & Đánh giá**
- **Ngày 8**: ⏳ Pipeline Testing
- **Ngày 9**: ⏳ Debug & Optimization
- **Ngày 10**: ⏳ Hyperparameter Tuning
- **Ngày 11**: ⏳ Full Training
- **Ngày 12**: ⏳ Model Evaluation
- **Ngày 13**: ⏳ Error Analysis
- **Ngày 14**: ⏳ Documentation

---

## 🎯 **MILESTONE CHÍNH**

### **Milestone 1** (Ngày 3): Model cơ bản hoạt động
- [ ] TableMaster model có thể khởi tạo
- [ ] Forward pass không lỗi
- [ ] Basic integration test pass

### **Milestone 2** (Ngày 7): Pipeline hoàn chỉnh
- [ ] Tất cả components đã được implement
- [ ] Config loading thành công
- [ ] Ready for training

### **Milestone 3** (Ngày 11): Training bắt đầu
- [ ] Full training pipeline hoạt động
- [ ] Monitoring system active
- [ ] No critical errors

### **Milestone 4** (Ngày 14): Hoàn thành giai đoạn 1
- [ ] Model đã được train
- [ ] Evaluation results available
- [ ] Code và documentation hoàn chỉnh

---

## 🚨 **CRITICAL PATH & DEPENDENCIES**

### **Critical Path**:
1. **Ngày 1-2**: Môi trường + Dataset → **BLOCKING** cho tất cả công việc sau
2. **Ngày 3-4**: Model + Components → **BLOCKING** cho training
3. **Ngày 5-6**: Loss + Dataset → **BLOCKING** cho training
4. **Ngày 7**: Config → **BLOCKING** cho training
5. **Ngày 8-11**: Training pipeline → **BLOCKING** cho evaluation

### **Dependencies**:
- **Dataset preparation** (Ngày 2) → Dataset class (Ngày 6)
- **Model TableMaster** (Ngày 3) → Integration (Ngày 7)
- **Backbone & Decoder** (Ngày 4) → Model complete (Ngày 7)
- **Loss Functions** (Ngày 5) → Training (Ngày 8)
- **Config & Registry** (Ngày 7) → All subsequent work