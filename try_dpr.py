from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
input_ids = tokenizer("Who is Aaron?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
print(embeddings[0][:5]) # Let's check embeddings for the first text