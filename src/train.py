"""Training entry point for IMDB sentiment classification."""

import argparse
import yaml
import numpy as np
import mlflow
from transformers import Trainer, TrainingArguments, TrainerCallback

from src.data import load_and_tokenize
from src.model import load_model



class MLflowMetricsCallback(TrainerCallback):
    """Forward Trainer logs to MLflow at every logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=state.global_step)


def compute_metrics(eval_pred):
    """Compute accuracy from logits and labels."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}

def main(config_path: str):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")

    # 2. Setup MLflow
    mlflow.set_experiment("sentiment-service")
    run_name = config["output_dir"].split("/")[-1]

    with mlflow.start_run(run_name=run_name):
        # Log all hyperparameters
        mlflow.log_params(config)

        # 3. Load data and model
        train_ds, eval_ds, tokenizer = load_and_tokenize(config)
        model = load_model(config)

        # 4. Setup HuggingFace Trainer
        training_args = TrainingArguments(
            output_dir=config["output_dir"],
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=float(config["learning_rate"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            seed=config["seed"],
            report_to="none",  # we use mlflow manually
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            callbacks=[MLflowMetricsCallback()],
        )

        # 5. Train
        trainer.train()

        # 6. Final evaluation
        eval_results = trainer.evaluate()
        print(f"Final eval results: {eval_results}")


        # 7. Save model and tokenizer (so we can use it later for serving)
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        print(f"Model saved to {config['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
