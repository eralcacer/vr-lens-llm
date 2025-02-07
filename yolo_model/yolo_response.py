from pydantic import BaseModel

class YoloResponse(BaseModel):
    name: str
    description: str | None = None