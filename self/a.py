from pydantic import Field, BaseModel, RootModel
from typing import List

class SubStruct(BaseModel):
    name : str = Field(..., description='用户名')

class Struct(RootModel):
    root : List[SubStruct] =  Field(..., description='用户列表')


if __name__ == '__main__':
    print(Struct.model_json_schema())