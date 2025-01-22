import pandas as pd
import torch
from typing import List, Tuple, Optional
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, BatchEncoding
from dygpt.tokenizer import Tokenizer, initial_tokenizer_with_vocabulary

class MyDataset(Dataset):
    """
    Custom PyTorch Dataset for Causal Language Modeling (CLM), regression, and conditional generation tasks.

    Args:
        task_type (str): Task type ('causallm', 'downstream', or 'conditional').
        input_df (pd.DataFrame): DataFrame containing input strings and optionally target values.
        vocab_file (str): Path to the vocabulary file for tokenizing input strings.
        tokenizer (GPT2Tokenizer): Tokenizer for encoding input strings.
        max_len (int): Maximum sequence length for tokenization.
        target_name (List[str]): List of target column names for regression or conditional tasks.
    """
    def __init__(self,
                 task_type: str,
                 input_df: pd.DataFrame,
                 vocab_file: str,
                 tokenizer: GPT2Tokenizer,
                 max_len: int,
                 target_name: List[str]):
        self.task_type = task_type
        self.input_df = input_df
        self.vocab_file = vocab_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_name = target_name

        # Validate arguments based on task type
        if self.task_type == 'downstream' and self.target_name is None:
            raise ValueError("target_name must be provided for regression task.")

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.input_df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a data sample at the given index.

        Args:
            idx (int): Index of the input string to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', 'labels', and 'conditions' (if applicable).
        """
        # Get the input SMILES string
        text = self.input_df['input'].iloc[idx]

        # Tokenize and encode the input string
        inputs: BatchEncoding = Tokenizer(smiles=text,
                                          vocab_file=self.vocab_file,
                                          tokenizer=self.tokenizer,
                                          max_len=self.max_len)

        # Convert input_ids and attention_mask to tensors
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).squeeze(0)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).squeeze(0)

        if self.task_type == 'causallm':
            # For Causal Language Modeling, labels are identical to input_ids
            labels = input_ids.clone()

        elif self.task_type == 'downstream':
            # Retrieve regression target values            
            target_values = self.input_df[self.target_name].iloc[idx].values
            labels = torch.tensor(target_values, dtype=torch.float32)

        elif self.task_type == 'conditional':
            # Retrieve condition values (e.g., HOMO, LUMO)
            target_values = self.input_df[self.target_name].iloc[idx].values.astype(float)

            # Prepare condition tokens based on target_name
            condition_tokens = [f"[{name}]" for name in self.target_name]

            # Initialize condition token IDs
            condition_ids = [
                self.tokenizer.convert_tokens_to_ids(token)
                for token in condition_tokens
                if token in self.tokenizer.all_special_tokens
                ]
            # Append the [SEP] token
            sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
            if sep_id is None:
                raise ValueError("[SEP] token not found in tokenizer.")
            condition_ids.append(sep_id)
            
            condition_ids = torch.tensor(condition_ids, dtype=torch.long)
            condition_mask = torch.ones_like(condition_ids, dtype=torch.long)
            
            # Combine condition tokens and input_ids
            input_ids = torch.cat([condition_ids, input_ids], dim=0)
            attention_mask = torch.cat([condition_mask, attention_mask], dim=0)

            padding_length = max(0, self.max_len - len(input_ids))
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)[:self.max_len]
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)], dim=0)[:self.max_len]

            # Set labels to input_ids (auto-regressive generation)
            labels = input_ids.clone()

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Default conditions to zeros if not applicable
        conditions = torch.tensor(target_values, dtype=torch.float32) if self.task_type == 'conditional' else torch.tensor([])

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'conditions': conditions
        }

def MyDataLoader(task_type: str,
                 vocab_file: str,
                 tokenizer: GPT2Tokenizer,
                 max_len: int,
                 batch_size: int,
                 target_name: List[str],
                 df_train: pd.DataFrame,
                 df_val: pd.DataFrame,
                 df_test: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for training, validation, and testing datasets for Causal Language Modeling (CLM),
    regression, or conditional generation tasks.

    Args:
        task_type (str): Type of task ('causallm', 'regression', or 'conditional').
        vocab_file (str): Path to the vocabulary file used for tokenizing string.
        tokenizer (GPT2Tokenizer): Tokenizer used for text tokenization.
        max_len (int): Maximum length for input sequences.
        batch_size (int): Batch size for the DataLoader.
        target_name (List[str]): Target names for regression tasks.
        df_train (pd.DataFrame): DataFrame for training data.
        df_val (pd.DataFrame): DataFrame for validation data.
        df_test (pd.DataFrame): DataFrame for test data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and testing datasets.
    """
    # Create datasets for training, validation, and optionally testing
    train_dataloader = None

    if df_train is not None:
        train_dataset = MyDataset(task_type=task_type,
                                  input_df=df_train, 
                                  vocab_file=vocab_file,
                                  tokenizer=tokenizer,
                                  max_len=max_len,
                                  target_name=target_name)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = None

    if df_val is not None:
        val_dataset = MyDataset(task_type=task_type,
                                input_df=df_val,
                                vocab_file=vocab_file,
                                tokenizer=tokenizer,
                                max_len=max_len,
                                target_name=target_name)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = None

    if df_test is not None:
        test_dataset = MyDataset(task_type=task_type,
                                 input_df=df_test,
                                 vocab_file=vocab_file,
                                 tokenizer=tokenizer,
                                 max_len=max_len,
                                 target_name=target_name)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
