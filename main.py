from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from exception import BadRequestException, InternalServerException
from ocr import Ocr, OcrResult
from post_ocr import PostOcr

#Simple FAST API Just to READ IMAGE.
app = FastAPI()
ocr = Ocr()
post_ocr = PostOcr()

@app.get("/")
def read_root():
    return "Server is alived"

@app.post('/word-ocr')
async def wordOcrParsing(img: UploadFile):
    try:
        if not img.content_type.startswith("image/"):
            raise BadRequestException("image harus merupakan sebuah gambar")
        try:
            result: OcrResult = await ocr.inference(img, None)
            words = result.words
            return JSONResponse(status_code=200, content=words)
        except Exception as e:
            raise InternalServerException('parsing error > ' + str(e))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

@app.post("/post-ocr")
async def postOcrPasring(img: UploadFile):
    try:
        if not img.content_type.startswith("image/"):
            raise BadRequestException("image harus merupakan sebuah gambar")
        try:
            result: OcrResult = await ocr.inference(img, None)
            words = result.words
            boxes = result.boxes
            items = post_ocr.process(image=ocr.img, bbox=boxes, words=words)
            return JSONResponse(status_code=200, content=items)
        except Exception as e:
            raise InternalServerException('parsing error > ' + str(e))
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)
