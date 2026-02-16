"""
ML Classifier Training Script
PHASE 5: Train BERT-based classifier for query complexity and domain
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.data_collection import data_collection_service
from src.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# ============================================================================
# PHASE 5: ML Classifier Training
# ============================================================================


class QueryDataset(Dataset):
    """PyTorch dataset for query classification"""

    def __init__(
        self,
        queries: List[str],
        complexity_labels: List[int],
        domain_labels: List[int],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ):
        self.queries = queries
        self.complexity_labels = complexity_labels
        self.domain_labels = domain_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.queries[idx]
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "complexity_label": torch.tensor(self.complexity_labels[idx], dtype=torch.long),
            "domain_label": torch.tensor(self.domain_labels[idx], dtype=torch.long),
        }


class QueryClassifier:
    """
    Multi-task BERT classifier for query routing
    
    Tasks:
    1. Complexity classification (simple/medium/complex)
    2. Domain classification (code/analysis/creative/chat)
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.complexity_model = None
        self.domain_model = None

        # Label mappings
        self.complexity_labels = {0: "simple", 1: "medium", 2: "complex"}
        self.domain_labels = {0: "code", 1: "analysis", 2: "creative", 3: "chat"}

    def prepare_data(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Prepare training data from routing logs
        
        This is a heuristic labeling strategy based on historical routing decisions
        In production, you would use actual user feedback and performance data
        """
        queries = []
        complexity_labels = []
        domain_labels = []

        for entry in training_data:
            query = entry["query"]
            features = entry["features"]
            model_used = entry["model_used"]

            queries.append(query)

            # Derive complexity label from model used and features
            if model_used == "llama-7b":
                complexity = 0  # simple
            elif model_used == "claude-sonnet":
                complexity = 1  # medium
            else:  # gpt-4
                complexity = 2  # complex

            complexity_labels.append(complexity)

            # Derive domain label from features
            if features.get("is_coding", False):
                domain = 0  # code
            elif features.get("is_analytical", False):
                domain = 1  # analysis
            elif features.get("is_creative", False):
                domain = 2  # creative
            else:
                domain = 3  # chat

            domain_labels.append(domain)

        return queries, complexity_labels, domain_labels

    def train_complexity_classifier(
        self,
        train_dataset: QueryDataset,
        eval_dataset: QueryDataset,
        output_dir: str = "./models/complexity",
        epochs: int = 3,
    ) -> None:
        """Train complexity classification model"""
        logger.info("training_complexity_classifier_started")

        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,  # simple, medium, complex
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)

        self.complexity_model = model
        logger.info("complexity_classifier_trained", output_dir=output_dir)

    def train_domain_classifier(
        self,
        train_dataset: QueryDataset,
        eval_dataset: QueryDataset,
        output_dir: str = "./models/domain",
        epochs: int = 3,
    ) -> None:
        """Train domain classification model"""
        logger.info("training_domain_classifier_started")

        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4,  # code, analysis, creative, chat
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)

        self.domain_model = model
        logger.info("domain_classifier_trained", output_dir=output_dir)

    def predict(self, query: str) -> Dict[str, Any]:
        """
        Predict complexity and domain for a query
        
        Returns:
            Dict with complexity and domain predictions
        """
        if self.complexity_model is None or self.domain_model is None:
            raise ValueError("Models not trained yet")

        encoding = self.tokenizer(
            query,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Complexity prediction
        with torch.no_grad():
            complexity_output = self.complexity_model(**encoding)
            complexity_pred = torch.argmax(complexity_output.logits, dim=1).item()
            complexity_confidence = torch.softmax(complexity_output.logits, dim=1)[0][
                complexity_pred
            ].item()

        # Domain prediction
        with torch.no_grad():
            domain_output = self.domain_model(**encoding)
            domain_pred = torch.argmax(domain_output.logits, dim=1).item()
            domain_confidence = torch.softmax(domain_output.logits, dim=1)[0][domain_pred].item()

        return {
            "complexity": self.complexity_labels[complexity_pred],
            "complexity_confidence": complexity_confidence,
            "domain": self.domain_labels[domain_pred],
            "domain_confidence": domain_confidence,
        }


async def main():
    """Main training pipeline"""
    logger.info("ml_training_started")

    # Fetch training data from database
    logger.info("fetching_training_data")
    training_data = await data_collection_service.get_training_data(limit=10000)

    if len(training_data) < 100:
        logger.error("insufficient_training_data", count=len(training_data))
        print("Error: Need at least 100 samples. Generate more routing logs first.")
        return

    logger.info("training_data_fetched", count=len(training_data))

    # Initialize classifier
    classifier = QueryClassifier()

    # Prepare data
    queries, complexity_labels, domain_labels = classifier.prepare_data(training_data)

    # Split data
    (
        train_queries,
        eval_queries,
        train_complexity,
        eval_complexity,
        train_domain,
        eval_domain,
    ) = train_test_split(
        queries,
        complexity_labels,
        domain_labels,
        test_size=0.2,
        random_state=42,
    )

    logger.info(
        "data_split_complete",
        train_size=len(train_queries),
        eval_size=len(eval_queries),
    )

    # Create datasets
    train_dataset = QueryDataset(
        train_queries,
        train_complexity,
        train_domain,
        classifier.tokenizer,
    )

    eval_dataset = QueryDataset(
        eval_queries,
        eval_complexity,
        eval_domain,
        classifier.tokenizer,
    )

    # Train complexity classifier
    classifier.train_complexity_classifier(
        train_dataset,
        eval_dataset,
        output_dir="./training/models/complexity",
    )

    # Train domain classifier
    classifier.train_domain_classifier(
        train_dataset,
        eval_dataset,
        output_dir="./training/models/domain",
    )

    logger.info("ml_training_complete")
    print("\n✅ Training complete!")
    print("Models saved to:")
    print("  - ./training/models/complexity")
    print("  - ./training/models/domain")


if __name__ == "__main__":
    asyncio.run(main())
