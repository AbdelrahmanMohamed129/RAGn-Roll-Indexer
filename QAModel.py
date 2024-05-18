from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, BertModel, BertConfig, BertTokenizer, pipeline, AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
import nltk

class CustomBertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super(CustomBertForQuestionAnswering, self).__init__()
        self.config = config  # Attach the config object as an attribute
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
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

    return answer

def QAs(model, tokenizer, questions, contexts):
    answers = []
    for question, context in zip(questions, contexts):
        answer = QA(model, tokenizer, question, context)
        answers.append(answer)
    return answers

def Evaluation(model, tokenizer, validation):
    correct = 0
    EM = 0
    total = 0
    errors = []
    for record in tqdm(validation):
        try:
            total += 1
            if (total % 500 == 0):
                print(f"\nAccuracy: {100*correct/total}")
                print(f"Correct: {correct}, out of {total}")
                print(f"EM: {100*EM/total}")
                print(f"EM Correct: {EM}, out of {total}\n")

            predicted_answer = QA(model, tokenizer, record['question'], record['context'])
            if predicted_answer.lower() in record['answers']['text'][0].lower() or record['answers']['text'][0].lower() in predicted_answer.lower():
                correct += 1
            if predicted_answer.lower() == record['answers']['text'][0].lower():
                EM += 1
        except Exception as e:
            errors.append(total)
            print(f"Error at {total}: {e} ")
            continue
    return correct, EM, total

# Instantiate the model with the provided configuration
config = BertConfig.from_dict({
    "_name_or_path": "ourModel",
    "architectures": [
        "BertForQuestionAnswering"
    ],
    "attention_probs_dropout_prob": 0.1,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.17.0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522
})

# model = CustomBertForQuestionAnswering(config)
# model.load_state_dict(torch.load('model.pth'), strict=False)
# model_checkpoint = "atharvamundada99/bert-large-question-answering-finetuned-legal"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# x = QA(model, tokenizer, "What is the capital of India?", "India is a country in South Asia. It is the seventh-largest country by land area, the second-most populous country, and the most populous democracy in the world. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar and Indonesia. Delhi is the capital of India.")
# print(x)