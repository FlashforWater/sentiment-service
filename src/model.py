from transformers import AutoModelForSequenceClassification


def load_model(config: dict):
    """Load a pre-trained model with a classification head."""
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
    )
    return model
