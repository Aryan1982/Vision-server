import easyocr
import cv2
from paddleocr import PaddleOCR


class OcrService:
    def __init__(self):
        # Initialize OCR pipelines
        self.paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True)
        self.easy_reader = easyocr.Reader(["en", "th"])

    def perform_ocr(self, method, img):
        if method == "PaddleOCR":
            return self.ocr_with_paddle(img)
        elif method == "EasyOCR":
            return self.ocr_with_easy(img)
        else:
            return "Invalid OCR method"

    def ocr_with_paddle(self, img):
        result = self.paddle_ocr.ocr(img)
        return " ".join([line[1][0] for line in result[0]])

    def ocr_with_easy(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("image_temp.png", gray_img)
        result = self.easy_reader.readtext("image_temp.png", detail=0)
        return " ".join(result)
