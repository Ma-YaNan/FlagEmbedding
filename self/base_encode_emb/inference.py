from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset


def test_base_single_device():
    model = AutoModel.from_pretrained('BAAI/bge-multilingual-gemma2')
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-multilingual-gemma2')
    ds = load_dataset('json', data_dir="examples/finetune/embedder/example_data/classification-no_in_batch_neg", split='train')
    sentences = ds[0]['neg']
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512)
    for item in inputs['input_ids']:
        print(len(item))
        print(tokenizer.convert_ids_to_tokens(item))
    pass
    # inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
    # r = model(**inputs, return_dict=True)
    # print(r)

def load():
    AutoModel.from_pretrained("BAAI/bge-m3")


if __name__  == "__main__":
    # test_base_single_device()
    # load()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
