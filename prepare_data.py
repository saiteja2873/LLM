import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import nltk
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, BartTokenizer, HfArgumentParser

# Download punkt for sentence tokenization
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: str = field(
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    dataset_path: Optional[str] = field(
        default="squad",
        metadata={"help": "Hugging Face dataset name (default: squad)"}, 
    )
    train_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    valid_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    valid_for_qg_only: bool = field(
        default=False,
        metadata={"help": "For multitask dataset valid split should contain only qg task or all tasks."}
    )
    qg_format: Optional[str] = field(
        default='highlight',
        metadata={"help": "How to format inputs for que generation, 'highlight' or 'prepend'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


def _get_correct_alignment(context, answer):
    """Some original examples in SQuAD have indices wrong by 1 or 2 characters. We test and fix this here."""
    gold_text = answer['text'][0] if isinstance(answer['text'], list) else answer['text']
    start_idx = answer['answer_start'][0] if isinstance(answer['answer_start'], list) else answer['answer_start']
    end_idx = start_idx + len(gold_text)
    
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2
    else:
        return start_idx, end_idx  # Return original if no fix found


def process_squad_to_qg_format(dataset, qg_format, task):
    """Process SQuAD dataset into question generation format."""
    processed_examples = []
    
    # Group examples by context for e2e_qg task
    if task == 'e2e_qg':
        context_to_questions = {}
        for example in dataset:
            context = example['context'].strip()
            question = example['question'].strip()
            if context not in context_to_questions:
                context_to_questions[context] = []
            context_to_questions[context].append(question)
        
        for context, questions in context_to_questions.items():
            source_text = f"generate questions: {context}"
            target_text = " {{sep_token}} ".join(questions) + " {{sep_token}}"
            processed_examples.append({
                "source_text": source_text,
                "target_text": target_text,
                "task": "e2e_qg"
            })
    else:
        for example in dataset:
            context = example['context'].strip()
            question = example['question'].strip()
            answers = example['answers']
            
            if len(answers['text']) == 0:
                continue
                
            answer_text = answers['text'][0].strip()
            answer = {'text': answer_text, 'answer_start': answers['answer_start'][0]}
            
            # QA task
            if task in ['qa', 'multi']:
                qa_source = f"question: {question}  context: {context}"
                qa_target = answer_text
                processed_examples.append({
                    "source_text": qa_source,
                    "target_text": qa_target,
                    "task": "qa"
                })
            
            # QG task
            if task in ['qg', 'multi']:
                if qg_format == "prepend":
                    qg_source = f"answer: {answer_text}  context: {context}"
                else:  # highlight format
                    start_pos, end_pos = _get_correct_alignment(context, answer)
                    qg_source = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
                
                processed_examples.append({
                    "source_text": qg_source,
                    "target_text": question,
                    "task": "qg"
                })
            
            # Answer extraction task
            if task in ['ans_ext', 'multi']:
                sents = nltk.sent_tokenize(context)
                positions = []
                prev_end = 0
                for i, sent in enumerate(sents):
                    if i == 0:
                        start, end = 0, len(sent)
                    else:
                        start, end = prev_end + 1, prev_end + len(sent) + 1
                    prev_end = end
                    positions.append({'start': start, 'end': end})
                
                ans_start = answer['answer_start']
                for i, pos in enumerate(positions):
                    if ans_start >= pos['start'] and ans_start < pos['end']:
                        # Build input with highlighted sentence
                        input_parts = ["extract answers:"]
                        for j, sent in enumerate(sents):
                            if i == j:
                                input_parts.append(f"{{hl_token}} {sent} {{hl_token}}")
                            else:
                                input_parts.append(sent)
                        
                        processed_examples.append({
                            "source_text": " ".join(input_parts),
                            "target_text": f"{answer_text} {{sep_token}}",
                            "task": "ans_ext"
                        })
                        break
    
    return Dataset.from_list(processed_examples)

class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"
        
        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"
  
    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True, 
        )
        target_encoding = self.tokenizer(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings


def filter_qa(example):
    return example['task'] == 'qa'

def filter_qg(example):
    return example['task'] == 'qg'

def filter_e2e_qg(example):
    return example['task'] == 'e2e_qg'

def filter_ans_ext(example):
    return example['task'] == 'ans_ext'

def filter_multi(example):
    return example['task'] != 'e2e_qg'


TASK_TO_FILTER_FN = {
    'qa': filter_qa,
    'qg': filter_qg,
    'e2e_qg': filter_e2e_qg,
    'ans_ext': filter_ans_ext,
    'multi': filter_multi
}


def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if data_args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    # Load SQuAD dataset directly from Hugging Face
    logger.info("Loading SQuAD dataset from Hugging Face...")
    raw_train = load_dataset("squad", split="train")
    raw_valid = load_dataset("squad", split="validation")
    
    # Process into QG format  
    logger.info(f"Processing data for task: {data_args.task}, format: {data_args.qg_format}")
    train_dataset = process_squad_to_qg_format(raw_train, data_args.qg_format, data_args.task)
    valid_dataset = process_squad_to_qg_format(raw_valid, data_args.qg_format, data_args.task)
    
    logger.info(f"Train examples: {len(train_dataset)}, Valid examples: {len(valid_dataset)}")

    processor = DataProcessor(
        tokenizer,
        model_type=data_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    if data_args.task == 'multi' and data_args.valid_for_qg_only:
        logger.info("processing valid data only for qg task")
        valid_dataset = valid_dataset.filter(filter_qg)

    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    # Convert to list of dicts with torch tensors for proper saving
    def dataset_to_torch_list(dataset):
        torch_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            torch_data.append({
                'source_ids': torch.tensor(item['source_ids']),
                'target_ids': torch.tensor(item['target_ids']),
                'attention_mask': torch.tensor(item['attention_mask']),
            })
        return torch_data

    logger.info("Converting train dataset to torch format...")
    train_torch = dataset_to_torch_list(train_dataset)
    logger.info("Converting valid dataset to torch format...")
    valid_torch = dataset_to_torch_list(valid_dataset)

    if data_args.train_file_name is None:
        train_file_name = f"train_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        train_path = os.path.join("data", train_file_name)

        valid_file_name = f"valid_data_{data_args.task}_{data_args.qg_format}_{data_args.model_type}.pt"
        valid_path = os.path.join("data", valid_file_name)
    else:
        train_path = os.path.join("data", data_args.train_file_name)
        valid_path = os.path.join("data", data_args.valid_file_name)
    
    torch.save(train_torch, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_torch, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    tokenizer_path = f"{data_args.model_type}_qg_tokenizer"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"saved tokenizer at {tokenizer_path}")


if __name__ == "__main__":
    main()
