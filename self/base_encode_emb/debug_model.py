from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import BiEncoderOnlyEmbedderModel


def loss_compute():
    base_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    model = BiEncoderOnlyEmbedderModel(base_model=base_model, tokenizer=tokenizer)
    ds = load_dataset("json", data_dir="examples/finetune/embedder/example_data/classification-no_in_batch_neg", split='train')
    p_sentences = tokenizer(ds[0]['neg'], return_tensors='pt', padding=True, truncation=True)
    q_sentences = tokenizer(ds[0]['query'], return_tensors='pt', padding=True, truncation=True)
    p_reps = model.encode(p_sentences)
    q_reps = model.encode(q_sentences)
    r = model(queries=q_sentences,passages=p_sentences)
    # r = model.compute_score(p_reps=p_reps,q_reps=q_reps)
    pass

if __name__ == "__main__":
    loss_compute()
