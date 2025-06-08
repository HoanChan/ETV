import sys
import os

# Thêm đường dẫn tới thư mục chứa mmocr vào sys.path
mmocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mmocr'))
if mmocr_path not in sys.path:
    sys.path.append(mmocr_path)

from mmocr.apis import MMOCRInferencer

ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
ocr(os.path.join(mmocr_path, 'demo/demo_text_ocr.jpg'), show=True, print_result=True)