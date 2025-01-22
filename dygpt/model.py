import torch
import torch.nn as nn
from typing import List, Optional
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model, GPT2Tokenizer

class NumericalEmbedding(nn.Module):
    """
    Converts numerical values (e.g., HOMO, LUMO) into learnable embeddings.
    """
    def __init__(self, embedding_dim):
        super(NumericalEmbedding, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)  # Map float to embedding vector
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for numerical embedding.

        Args:
            x (Tensor): Numerical input of shape [batch_size, 1].

        Returns:
            Tensor: Embedding of shape [batch_size, embedding_dim].
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x  # Ensure input has shape [batch_size, 1]
        return self.activation(self.linear(x))

class GPTForCausalLM(nn.Module):
    """
    Causal Language Model (CausalLM) using a GPT-2 architecture.

    Args:
        tokenizer (GPT2Tokenizer): Tokenizer used for text tokenization.
        target_name (List[str]): List of target names for conditional generation.
        hidden_size (int): Dimensionality of the encoder layers.
        num_attention_heads (int): Number of attention heads in each attention layer.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
    """
    def __init__(self,
                 tokenizer: GPT2Tokenizer,
                 target_name: List[str],
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12):
        super(GPTForCausalLM, self).__init__()

        # Configure GPT model for Causal Language Modeling
        self.tokenizer = tokenizer
        self.target_name = target_name  # Target names for conditional generation
        self.config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            pad_token_id=tokenizer.pad_token_id,  # Use [PAD] token
            eos_token_id=tokenizer.eos_token_id   # Use [EOS] token
        )

        # Initialize the GPT-2 model for Causal Language Modeling
        self.model = GPT2LMHeadModel(self.config)
        
        # Initialize Numerical Embedding layers dynamically based on target_name
        if self.target_name:
            self.target_embeddings = nn.ModuleDict({
                name: NumericalEmbedding(hidden_size) for name in self.target_name
            })
            self.condition_projector = nn.Linear(hidden_size, self.config.vocab_size)  # Map hidden_size to vocab_size

    def forward(self,
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
                conditions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Conditional Generation model.

        Args:
            input_ids (Tensor): Token IDs of shape [batch_size, sequence_length].
            attention_mask (Tensor): Attention mask of shape [batch_size, sequence_length].
            labels (Tensor): Labels for auto-regressive generation.
            conditions (Tensor): Numerical conditions of shape [batch_size, num_conditions].

        Returns:
            Tensor: Output logits of shape [batch_size, sequence_length, vocab_size].
        """
        # Initialize condition embedding
        condition_embedding = 0

        # Check if conditions are provided
        if self.target_name and len(conditions) != 0:
            for i, target in enumerate(self.target_name):  # Iterate over target names dynamically
                if conditions.size(1) > i:  # Ensure condition exists for the target
                    condition_value = conditions[:, i:i+1]  # Extract the i-th condition
                    if target in self.target_embeddings:  # Check if embedding exists
                        condition_embedding += self.target_embeddings[target](condition_value)

            # Expand the embedding and project it to vocab_size
            condition_embedding = condition_embedding.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
            condition_embedding = self.condition_projector(condition_embedding)  # Shape: (batch_size, 1, vocab_size)
            condition_embedding = condition_embedding.expand(-1, input_ids.size(1), -1)  # Shape: (batch_size, seq_len, vocab_size)
        else:
            condition_embedding = 0

        # Forward pass through the GPT model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.logits += condition_embedding  # Adjust logits with condition embedding

        return outputs

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: GPT2Tokenizer, target_name: Optional[List[str]] = None):
        """
        Load a pre-trained CausalLM model from a given path.

        Args:
            pretrained_path (str): Path to the pre-trained model directory.
            tokenizer (GPT2Tokenizer): Tokenizer used for text tokenization.
            target_name (List[str], optional): List of target names for conditional generation.

        Returns:
            GPTForCausalLM: An instance of GPTForCausalLM with the pre-trained weights loaded.
        """
        model = cls(tokenizer=tokenizer, target_name=target_name)
        model.model = GPT2LMHeadModel.from_pretrained(pretrained_path)
        return model

class GPTForDownstream(nn.Module):
    """
    A custom model for regression tasks using a GPT-2 model.

    Args:
        tokenizer (GPT2Tokenizer): Tokenizer used for text tokenization.
        target_name (List[str]): List of target columns.
        hidden_size (int): Dimensionality of the encoder layers.
        num_attention_heads (int): Number of attention heads in each attention layer.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
    """
    def __init__(self,
                 tokenizer: GPT2Tokenizer,
                 target_name: List[str],
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12):
        super(GPTForDownstream, self).__init__()

        # Configure GPT-2 model from scratch (random initialization)
        self.tokenizer = tokenizer
        self.config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            pad_token_id=tokenizer.pad_token_id,  # Use [PAD] token
            eos_token_id=tokenizer.eos_token_id   # Use [EOS] token
        )
        
        # Initialize GPT-2 model
        self.gpt = GPT2Model(self.config)
        self.target_name = target_name

        # Fully connected layer for regression
        self.fc = nn.Linear(self.gpt.config.n_embd, len(self.target_name))  # Fully connected layer to predict target values
        self.dropout = nn.Dropout(0.3)  # Dropout layer for regularization

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the model to predict target values.

        Args:
            input_ids (Tensor): Tensor containing token IDs for input sequences.
            attention_mask (Tensor, optional): Tensor indicating which tokens should be attended to.

        Returns:
            Tensor: A tensor containing the predicted target values for each input sequence.
        """
        # Forward pass through GPT-2 model
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last token's output from the last hidden state
        last_token_output = outputs.last_hidden_state[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout for regularization
        last_token_output = self.dropout(last_token_output)

        # Pass through the fully connected layer to predict target values
        prediction = self.fc(last_token_output)  # Shape: (batch_size, len(target_name))

        return prediction

    @classmethod
    def from_pretrained_model(cls, pretrained_path: str, tokenizer: GPT2Tokenizer, target_name: List[str]):
        """
        Load a pre-trained GPT model for regression tasks from a given path.

        Args:
            pretrained_path (str): Path to the pre-trained model directory.
            tokenizer (GPT2Tokenizer): Tokenizer used for text tokenization.
            target_list (List[str]): List of target columns.

        Returns:
            GPTForRegression: An instance of GPTForRegression with the pre-trained weights loaded.
        """
        model = cls(tokenizer=tokenizer, target_name=target_name)
        model.gpt = GPT2Model.from_pretrained(pretrained_path)
        return model
