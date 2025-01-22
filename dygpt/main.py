import os
import torch
import argparse
import pandas as pd
from torch.utils.data import random_split
from transformers import BertTokenizer, GPT2Tokenizer
from dygpt import GPTForCausalLM, GPTForDownstream 
from dygpt import train
from dygpt import set_seed
from dygpt.tokenizer import initial_tokenizer_with_vocabulary

# Set the random seed for reproducibility
SEED = 42
set_seed(SEED)

def main():
    # Parse command line arguments for model training configuration
    parser = argparse.ArgumentParser(description="Unified GPT Training for CausalLM and Downstream Tasks")
    parser.add_argument('--task', type=str, required=True, choices=['causallm', 'downstream', 'conditional'],
                        help="Task type: 'causallm' for Causal Language Model, 'downstream' for target prediction.")
    parser.add_argument('--pretrained', type=str, default=None, help='Path of pre-trained Model')
    parser.add_argument('--dataset', type=str, help='Path to the dataset with CSV format')
    parser.add_argument('--vocabfile', type=str, default='./vocab.txt', help='Vocabulary file for tokenizing')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum length for input sequences')
    parser.add_argument('--batchsize', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--modelsavepath', type=str, default='./results', help='Save path of finetuned model')
    parser.add_argument('--target', type=str, nargs='+', default=None,
                        help='List of target names for downstream prediction (e.g., --target_list HOMO LUMO)')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Check if the vocabulary file exists; if not, create it and initialize tokenizer
    if not os.path.exists(args.vocabfile):
        print(f"Vocabulary file '{args.vocabfile}' not found. Creating vocabulary...")
        tokenizer = initial_tokenizer_with_vocabulary(path=args.vocabfile, target_name=args.target)
    else:
        tokenizer = initial_tokenizer_with_vocabulary(path=args.vocabfile, target_name=args.target)

    # Load dataset CSV file
    df = pd.read_csv(args.dataset)

    # Task type: Masked Language Modeling (MLM)
    if args.task in ['causallm', 'conditional']:
        # Split data into training and validation sets
        train_size = int(0.8 * len(df))
        val_size = len(df) - train_size

        train_subset, val_subset = random_split(df, [train_size, val_size])
        
        # Convert Subsets back to DataFrame
        df_train = df.iloc[train_subset.indices].reset_index(drop=True)
        df_val = df.iloc[val_subset.indices].reset_index(drop=True)
        df_test = None

        # Initialize GPTForCausalLM model
        if args.pretrained is None:
            model = GPTForCausalLM(tokenizer=tokenizer, target_name=args.target)
        else:
            model = GPTForCausalLM.from_pretrained_model(tokenizer=tokenizer,
                                                         pretrained_path=args.pretrained,
                                                         target_name=args.target)

    # Task type: Downstream task for target prediction
    elif args.task == 'downstream':
        # Load training, validation, and test datasets using pandas
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        test_size = len(df) - train_size - val_size

        train_subset, val_subset, test_subset = random_split(df, [train_size, val_size, test_size])

        # Convert Subsets back to DataFrame
        df_train = df.iloc[train_subset.indices].reset_index(drop=True)
        df_val = df.iloc[val_subset.indices].reset_index(drop=True)
        df_test = df.iloc[test_subset.indices].reset_index(drop=True)

        # Initialize GPTForDownstream model
        if args.pretrained is None:
            model = GPTForDownstream(tokenizer=tokenizer, target_name=args.target)
        else:
            model = GPTForDownstream.from_pretrained_model(tokenizer=tokenizer,
                                                           target_name=args.target,
                                                           pretrained_path=args.pretrained)

    # Start training the model
    train(task_type=args.task,
            model=model,
            vocab_file=args.vocabfile,
            tokenizer=tokenizer,
            max_len=args.max_len,
            batch_size=args.batchsize,
            epochs=args.epochs,
            target_name=args.target,
            model_save_path=args.modelsavepath,
            df_train=df_train,
            df_val=df_val,
            df_test=df_test)

if __name__ == '__main__':
    main()
