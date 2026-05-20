"""Data loading and preprocessing for IMDB sentiment classification."""

from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenize(config: dict):
    """Load IMDB dataset, subsample, and tokenize.

    Args:
        config: dict loaded from train_config.yaml

    Returns:
        tokenized_train, tokenized_eval, tokenizer
    """
    # Load tokenizer matching the model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Load IMDB; it has 'train' and 'test' splits
    dataset = load_dataset(config["dataset"])

    # Subsample for fast iteration (shuffle first to avoid label imbalance)
    train_ds = dataset["train"].shuffle(seed=config["seed"]).select(range(config["train_size"]))
    eval_ds = dataset["test"].shuffle(seed=config["seed"]).select(range(config["eval_size"]))

    # Tokenize: turn text into model-ready input_ids and attention_mask
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True)

    return tokenized_train, tokenized_eval, tokenizer
