from pydantic import BaseModel

class Image(BaseModel):
    title: str
    size: float
    extension: str
    caption: str