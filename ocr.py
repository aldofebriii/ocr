from fastapi import UploadFile
from exception import BadRequestException, InternalServerException
from typing import Optional
from functools import lru_cache
from paddleocr import PaddleOCR
from PIL import Image
from urllib.request import Request, urlopen
from io import BytesIO
import os
import time
import io
from dataclasses import dataclass
import pytesseract

@dataclass
class OcrResult:
    words: list[str]
    boxes: list[list[int]]

class Ocr:
    @lru_cache
    def __init__(self):
        self.model = self.__load_ocr_model()
        self.tesseract = self.__load_tesseract()
        
    def __normalize_bbox(self, bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]
    
    def __load_ocr_model(self):
        return PaddleOCR(use_angle_cls=True, lang='id', image_orientation=True, max_text_length=50)
    
    def __load_tesseract(self):
        return pytesseract

    def __merge_data(self, values, width, height):
        word = []
        boxes  = []
        for idx in range(len(values)):
            box = values[idx][0]
            left_corner = box[0]
            right_corner = box[2]
            x_left_corner = left_corner[0]
            y_left_corner = left_corner[1]
            x_right_corner = right_corner[0]
            y_right_corner = right_corner[1]
            text = values[idx][1][0];
            word.append(text)
            boxes.append(self.__normalize_bbox([x_left_corner, y_left_corner, x_right_corner, y_right_corner], width, height))
        return word, boxes

    def __invoke_ocr(self, doc, content_type):
        worker_pid = os.getpid()
        print(f"Handling OCR request with worker PID: {worker_pid}")
        start_time = time.time()

        model = self.model

        bytes_img = io.BytesIO()

        format_img = "JPEG"
        if content_type == "image/png":
            format_img = "PNG"
        #get the image width and height
        image_width = doc.width
        image_height = doc.height
        doc.save(bytes_img, format=format_img)
        bytes_data = bytes_img.getvalue()
        bytes_img.close()

        result = model.ocr(bytes_data, cls=True)
        values = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                values.append(line)

        words, boxes = self.__merge_data(values, image_width, image_height)
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"OCR done, worker PID: {worker_pid}")
        result = OcrResult(words, boxes)
        return result, processing_time

    async def inference(self, file: UploadFile,
                        image_url: Optional[str] = None):
        result = None
        if file:
            self.img = Image.open(BytesIO(await file.read()))
        elif image_url:
            headers = {"User-Agent": "Mozilla/5.0"} # to avoid 403 error
            req = Request(image_url, headers=headers)
            with urlopen(req) as response:
                content_type = response.info().get_content_type()
                if content_type in ["image/jpeg", "image/jpg", "image/png"]:
                    self.img = Image.open(BytesIO(response.read()))
                else:
                    raise BadRequestException(msg="invalid image url")
        result, processing_time = self.__invoke_ocr(self.img, file.content_type)
        print(f"Processing time OCR: {processing_time:.2f} seconds")
        
        if result is None:
            raise InternalServerException(msg="parsing error > result is empty ")

        return result