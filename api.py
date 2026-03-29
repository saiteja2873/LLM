from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-e2e-qg")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-e2e-qg")

class InputText(BaseModel):
    text: str

@app.post("/generate")
def generate_questions(data: InputText):
    prompt = "generate questions: " + data.text

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=64)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"questions": result}