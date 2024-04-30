from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import numpy 

# tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# #save tokenizer to file
# tokenizer.save_pretrained('./Data/Model')

# #save model to file
# model.save_pretrained('./Data/Model')

# read model from file
model = DPRQuestionEncoder.from_pretrained('./Data/Model')

# read tokenizer from file
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('./Data/Model')

input_ids = tokenizer("When was the first time auto racing started?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
torch.set_printoptions(precision=10)
with open('./Data/query.txt', 'w') as f:
    f.write(str(embeddings.detach().numpy()).replace('[','').replace(']','').replace(',','').replace('\n',''))
    f.close()