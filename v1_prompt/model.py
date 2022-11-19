import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from softEmbedding import SoftEmbedding

class PromptModel(nn.Module):
    def __init__(self, model_name_or_path, n_tokens, initialize_from_vocab=True):
        super(PromptModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.s_wte = SoftEmbedding(self.model.get_input_embeddings(), 
                n_tokens=n_tokens, 
                initialize_from_vocab=initialize_from_vocab)
        self.model.set_input_embeddings(self.s_wte)
    
