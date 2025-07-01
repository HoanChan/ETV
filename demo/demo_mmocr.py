from mmocr.apis import MMOCRInferencer

# Tạo một demo OCR với hình ảnh từ internet hoặc hình ảnh local
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')

# Sử dụng URL hình ảnh demo từ internet hoặc có thể sử dụng hình ảnh local nếu có
demo_img = 'https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_text_ocr.jpg'

try:
    # Thử với URL hình ảnh demo
    ocr(demo_img, show=True, print_result=True)
    print("OCR completed successfully!")
except Exception as e:
    print(f"Error occurred: {e}")
    print("Please make sure you have internet connection or provide a local image path.")