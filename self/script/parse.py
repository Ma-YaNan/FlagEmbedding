
from dataclasses import dataclass, field
import os
from transformers import HfArgumentParser


@dataclass
class TrainParse:
    # train_data: str = field(
    #     default=None, 
    #     metadata={
    #         "help": "One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data.",
    #         "nargs": "?"
    #     }
    # )

    train_data: list[str] = field(default_factory=list)

    def __post_init__(self):

        for train_dir in self.train_data:
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"cannot find file: {train_dir}, please set a true path")

def example():
    parser = HfArgumentParser((TrainParse))
    training_args = parser.parse_args_into_dataclasses()
    print(training_args)
            

if __name__ == '__main__':
    example()
    # parser = HfArgumentParser((TrainParse))
    # # model_args, data_args, training_args = parser.parse_json_file(json_file='./self/base_encode_emb/config.json')
    # training_args = parser.parse_args_into_dataclasses()
    # print(training_args)