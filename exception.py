from fastapi import HTTPException
class BadRequestException(HTTPException):
    def __init__(self, msg: str):
        super().__init__(status_code=400, detail=msg)

class InternalServerException(HTTPException):
    def __init__(self, msg: str):
        super().__init__(status_code=500, detail=msg)