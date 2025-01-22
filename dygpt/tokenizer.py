import os
import re
import json
from typing import Union, List
from transformers import GPT2Tokenizer, BatchEncoding

def initial_tokenizer_with_vocabulary(path: str, target_name: List[str]) -> GPT2Tokenizer:
    """
    Creates a vocabulary file and initializes a GPT2Tokenizer using the custom vocabulary.

    Args:
        vocab_path (str): Path where the vocabulary file will be saved.
        target_name (List[str]): List of target column names

    Returns:
        GPT2Tokenizer: A GPT2 tokenizer instance initialized with the custom vocabulary.
    """
    # List of element symbols commonly used in SMILES strings
    element_symbols: List[str] = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                                  'K', 'Sc', 'Zn', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Ag', 'In', 'Sn', 'Te', 'I', 'Xe',
                                  'Cs', 'Ba', 'At', 'Ra', 'Np']

    # List of special symbols used in SMILES notation
    special_symbols: List[str] = ['@', '@@', '/', '\\', '-', '=', '#', '(', ')', '[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '%', '+', '.']

    # List of aromatic symbols used in SMILES notation
    aromatic_symbols: List[str] = ['c', 'n', 'o', 'p', 's']

    # List of vocabulary tokens, including special BERT-like tokens
    vocab_list: List[str] = ['[EOS]', '[PAD]', '[SEP]'] + element_symbols + aromatic_symbols + special_symbols

    # Add target names as custom tokens if provided
    if target_name is not None:
        target_tokens = [f"[{name}]" for name in target_name]
        vocab_list += target_tokens

    # Save vocabulary to the specified path
    with open(path, 'w') as f:
        f.write('\n'.join(vocab_list))

    # Prepare dictionary and merge rules for GPT2Tokenizer initialization
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    merges = [""]  # GPT2 merge rules are not used in this case

    vocab_json_path = "vocab.json"
    merges_path = "merges.txt"

    with open(vocab_json_path, 'w') as f:
        json.dump(vocab, f)

    with open(merges_path, 'w') as f:
        f.write("\n".join(merges))

    # Initialize GPT2TokenizerFast with the custom files
    tokenizer = GPT2Tokenizer(vocab_file=vocab_json_path, merges_file=merges_path)

    # Add special tokens if necessary
    tokenizer.add_special_tokens({
        "eos_token": "[EOS]",
        "pad_token": "[PAD]",
        "additional_special_tokens": ["[SEP]"] + (target_tokens if target_name else [])
    })

    return tokenizer

def TokenGenerator(smiles: str, vocab_file: str) -> List[str]:
    """
    Tokenizes a SMILES string based on a custom vocabulary.

    Args:
        smiles (str): The SMILES string to tokenize.
        vocab_file (str): Path to the vocabulary file used for tokenizing SMILES.

    Returns:
        List[str]: A list of tokens representing the SMILES string.
    """
    # Load vocabulary symbols from file
    with open(vocab_file, 'r') as f:
        sorted_symbols = sorted([line.strip() for line in f.readlines()], key=len, reverse=True)

    # Tokenize the SMILES string using the generated pattern
    token_pattern = '(' + '|'.join(map(re.escape, sorted_symbols)) + '|.)'
    tokens = re.findall(token_pattern, smiles)

    return tokens

def Tokenizer(smiles: Union[str, List[str]],
              vocab_file: str,
              tokenizer: GPT2Tokenizer,
              max_len: int) -> BatchEncoding:
    """
    Encodes SMILES strings into tokenized input suitable for GPT models.

    Args:
        smiles (str or List[str]): The SMILES string(s) to tokenize.
        vocab_file (str): Path to the vocabulary file used for tokenizing SMILES.
        tokenizer: GPT2Tokenizer initialized with the custom vocabulary.
        max_len (int): Maximum length for sequences.

    Returns:
        BatchEncoding: Encoded SMILES input with input_ids and attention_mask.
    """
    # Ensure smiles is a list
    smiles_list = smiles if isinstance(smiles, list) else [smiles]

    # Initialize lists to store tokenized and encoded information
    all_input_ids, all_attention_mask = [], []

    for smiles in smiles_list:
        # Tokenize the SMILES string
        tokens = TokenGenerator(smiles, vocab_file)
        tokens.append('[EOS]')

        # Encode the tokens using the provided tokenizer
        output = tokenizer.encode_plus(
            tokens,
            is_split_into_words=False,      
            add_special_tokens=False,       
            return_attention_mask=True,     
            padding='max_length',           
            max_length=max_len,     
            truncation=True     
        )

        # Append the encoded components to their respective lists
        all_input_ids.append(output['input_ids'])
        all_attention_mask.append(output['attention_mask'])

    # Create a BatchEncoding object containing the encoded inputs
    batch_encoding = BatchEncoding(data={
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask
    })

    return batch_encoding
