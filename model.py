import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from softEmbedding import SoftEmbedding, MultiSoftEmbedding

class PromptModel(nn.Module):
    def __init__(self, model_name_or_path, n_tokens, num_tasks=4, initialize_from_vocab=True, prompt_paradigm='single'):
        super(PromptModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        if prompt_paradigm == 'single':
            self.s_wte = SoftEmbedding(self.model.get_input_embeddings(), 
                n_tokens=n_tokens, 
                initialize_from_vocab=initialize_from_vocab)
        elif prompt_paradigm == 'multi':
            self.s_wte = MultiSoftEmbedding(self.model.get_input_embeddings(), 
                n_tokens=n_tokens, 
                initialize_from_vocab=initialize_from_vocab,
                num_tasks=num_tasks)
        self.model.set_input_embeddings(self.s_wte)
    
