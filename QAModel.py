from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, BertConfig, BertTokenizer, pipeline, AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
import nltk

# import transformers
# from datasets import load_dataset
# from transformers import AutoModelForQuestionAnswering, BertModel, BertConfig, BertTokenizer, pipeline, AutoTokenizer
from transformers import AutoModelForQuestionAnswering, BertConfig, BertTokenizer, pipeline, AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import math


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=1024, pad_token_id=0, max_position_embeddings=512, type_vocab_size=2):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # Make position_ids a nn.Parameter
        self.position_ids = nn.Parameter(torch.arange(max_position_embeddings).unsqueeze(0), requires_grad=False)

        # LayerNorm and dropout Module
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, :input_ids.size(1)]  # use pre-computed position_ids

        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if position_embeddings.size(1) < word_embeddings.size(1):       # to handle size mismatch by padding
            padding = torch.zeros((position_embeddings.size(0), word_embeddings.size(1) - position_embeddings.size(1), position_embeddings.size(2)), device=position_embeddings.device)
            position_embeddings = torch.cat([position_embeddings, padding], dim=1)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=1024, dropout_prob=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Implement the forward pass
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention(nn.Module):
    def __init__(self, hidden_size=1024, num_attention_heads=16, attention_probs_dropout_prob=0.1):
        super(BertAttention, self).__init__()

        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        # Implement the forward pass
        self_output = self.self(input_tensor, attention_mask)
        if isinstance(self_output, tuple):
            self_output = self_output[0]
        attention_output = self.output(self_output, input_tensor)
        return attention_output
    
class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
    
class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, intermediate_size=4096, hidden_size=1024, dropout_prob=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Implement the forward pass
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096, num_attention_heads=16, attention_probs_dropout_prob=0.1):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size, attention_probs_dropout_prob)


    def forward(self, hidden_states, attention_mask):
        # Implement the forward pass
        attention_output = self.attention(hidden_states, attention_mask)
        if isinstance(attention_output, tuple):
                attention_output = attention_output[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=24, hidden_size=1024, intermediate_size=4096, num_attention_heads=16, attention_probs_dropout_prob=0.1):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob) for _ in range(num_hidden_layers)])
        

    def forward(self, hidden_states, attention_mask):
        # Implement the forward pass
        for layer in self.layer:
            # check type of hidden_states
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
    
class BertModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=1024, num_hidden_layers=24, intermediate_size=4096, num_attention_heads=16, attention_probs_dropout_prob=0.1, pad_token_id = 0, max_position_embeddings=512, type_vocab_size=2):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, pad_token_id, max_position_embeddings, type_vocab_size)
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob)
        

        # self.pooler = BertPooler(hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Implement the forward pass
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        # pooled_output = self.pooler(encoder_output)

        return encoder_output
        # return pooled_output  # or return pooled_output lw hnst3ml el pooler


class CustomBertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super(CustomBertForQuestionAnswering, self).__init__()
        self.config = config
        self.bert = BertModel(vocab_size=config.vocab_size, hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers, intermediate_size=config.intermediate_size, num_attention_heads=config.num_attention_heads, attention_probs_dropout_prob=config.attention_probs_dropout_prob, pad_token_id=config.pad_token_id ,max_position_embeddings=config.max_position_embeddings, type_vocab_size=config.type_vocab_size)
        
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if isinstance(sequence_output, tuple):
            sequence_output = sequence_output[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


def QA(model, tokenizer, question, context):
    
    # Process the inputs
    inputs = tokenizer(question, context, return_tensors='pt')

    # Pass the inputs through the model and get the start and end scores
    start_scores, end_scores = model(**inputs)

    # Get the start and end positions
    start_position = torch.argmax(start_scores)
    end_position = torch.argmax(end_scores)

    # Get the answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position+1]))

    # x = question_answerer(question="What is the capital of India?", context="India is a country in South Asia. It is the seventh-largest country by land area, the second-most populous country, and the most populous democracy in the world. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar and Indonesia. Delhi is the capital of India.", truncation=True, padding=True, return_tensors='pt')

    return answer


def QAs(model, tokenizer, questions, contexts):
    answers = []
    for question, context in zip(questions, contexts):
        answer = QA(model, tokenizer, question, context)
        answers.append(answer)
    return answers
