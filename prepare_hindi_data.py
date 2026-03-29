"""
Prepare Hindi Question Generation Data

This script prepares training data for Hindi question generation using:
1. XQuAD Hindi dataset (high quality human translated)
2. Or translates English SQuAD to Hindi using translation model

Usage:
    python prepare_hindi_data.py --method xquad --model_type mt5
    python prepare_hindi_data.py --method translate --model_type mt5
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    method: str = field(
        default="xquad",
        metadata={"help": "Method to get Hindi data: 'xquad' (use XQuAD Hindi) or 'translate' (translate SQuAD)"}
    )
    model_type: str = field(
        default="mt5",
        metadata={"help": "Model type: 'mt5' for multilingual T5"}
    )
    max_source_length: int = field(default=512)
    max_target_length: int = field(default=64)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit training samples (for faster testing)"}
    )


def load_xquad_hindi():
    """Load XQuAD Hindi dataset - human translated subset of SQuAD"""
    logger.info("Loading XQuAD Hindi dataset...")
    dataset = load_dataset("xquad", "xquad.hi")
    return dataset["validation"]  # XQuAD only has validation split


def load_mlqa_hindi():
    """Load MLQA Hindi dataset"""
    logger.info("Loading MLQA Hindi dataset...")
    dataset = load_dataset("mlqa", "mlqa-translate-train.hi")
    return dataset["train"]


def translate_squad_to_hindi(max_samples=None):
    """Translate English SQuAD to Hindi using translation model"""
    logger.info("Loading English SQuAD and translating to Hindi...")
    logger.info("This may take a while...")
    
    # Load English SQuAD
    squad = load_dataset("squad", split="train")
    if max_samples:
        squad = squad.select(range(min(max_samples, len(squad))))
    
    # Load translator
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=-1)
    
    translated_data = []
    for example in tqdm(squad, desc="Translating"):
        try:
            # Translate context and question
            context_hi = translator(example['context'][:1000])[0]['translation_text']  # Limit context length
            question_hi = translator(example['question'])[0]['translation_text']
            
            translated_data.append({
                'context': context_hi,
                'question': question_hi,
                'answers': example['answers']
            })
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            continue
    
    return Dataset.from_list(translated_data)


def process_to_qg_format(dataset, language="hi"):
    """Convert dataset to question generation format"""
    processed = []
    
    # Group by context for e2e_qg
    context_to_questions = {}
    
    for example in dataset:
        context = example['context'].strip()
        question = example['question'].strip()
        
        if context not in context_to_questions:
            context_to_questions[context] = []
        context_to_questions[context].append(question)
    
    for context, questions in context_to_questions.items():
        # Hindi prompt
        if language == "hi":
            source_text = f"प्रश्न उत्पन्न करें: {context}"
        else:
            source_text = f"generate questions: {context}"
            
        target_text = " <sep> ".join(questions) + " <sep>"
        
        processed.append({
            "source_text": source_text,
            "target_text": target_text,
            "task": "e2e_qg"
        })
    
    return Dataset.from_list(processed)


class DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=64):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def process(self, dataset):
        dataset = dataset.map(self._add_eos)
        dataset = dataset.map(self._tokenize, batched=True)
        return dataset

    def _add_eos(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example

    def _tokenize(self, batch):
        source = self.tokenizer(
            batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True
        )
        target = self.tokenizer(
            batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True
        )
        return {
            'source_ids': source['input_ids'],
            'target_ids': target['input_ids'],
            'attention_mask': source['attention_mask']
        }


def main():
    parser = HfArgumentParser((DataArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # Load tokenizer
    logger.info("Loading mT5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    tokenizer.add_tokens(['<sep>', '<hl>'])

    # Get Hindi data based on method
    if args.method == "xquad":
        raw_data = load_xquad_hindi()
    elif args.method == "mlqa":
        raw_data = load_mlqa_hindi()
    elif args.method == "translate":
        raw_data = translate_squad_to_hindi(args.max_train_samples)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Loaded {len(raw_data)} examples")

    # Process to QG format
    logger.info("Processing to question generation format...")
    qg_data = process_to_qg_format(raw_data, language="hi")
    logger.info(f"Created {len(qg_data)} QG examples")

    # Split into train/valid (80/20)
    split = qg_data.train_test_split(test_size=0.2, seed=42)
    train_data = split['train']
    valid_data = split['test']

    # Process and tokenize
    processor = DataProcessor(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )

    train_processed = processor.process(train_data)
    valid_processed = processor.process(valid_data)

    # Convert to torch format
    def to_torch_list(dataset):
        result = []
        for i in range(len(dataset)):
            item = dataset[i]
            result.append({
                'source_ids': torch.tensor(item['source_ids']),
                'target_ids': torch.tensor(item['target_ids']),
                'attention_mask': torch.tensor(item['attention_mask'])
            })
        return result

    logger.info("Converting to torch format...")
    train_torch = to_torch_list(train_processed)
    valid_torch = to_torch_list(valid_processed)

    # Save
    os.makedirs("data", exist_ok=True)
    
    train_path = f"data/train_data_hindi_e2e_qg_{args.method}_mt5.pt"
    valid_path = f"data/valid_data_hindi_e2e_qg_{args.method}_mt5.pt"
    
    torch.save(train_torch, train_path)
    torch.save(valid_torch, valid_path)
    
    logger.info(f"Saved train data: {train_path} ({len(train_torch)} examples)")
    logger.info(f"Saved valid data: {valid_path} ({len(valid_torch)} examples)")

    # Save tokenizer
    tokenizer_path = "mt5_hindi_qg_tokenizer"
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Saved tokenizer: {tokenizer_path}")


if __name__ == "__main__":
    main()
