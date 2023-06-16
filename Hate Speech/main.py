import os
import numpy as np
import uvicorn
import traceback
import tensorflow as tf

from pydantic import BaseModel
from fastapi import FastAPI, Response
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('hate_speech_model.h5')
tokenizer = Tokenizer()
maxlen = 50  # Set the maximum sequence length according to your training data

app = FastAPI()

@app.get("/")
def index():
    return "Hate Speech"

# If your model needs text input, use this endpoint!
class RequestText(BaseModel):
    text: str

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        text = req.text

        # Step 1: (Optional) Do your text preprocessing

        # Step 2: Prepare your data for the model
        text = np.array(tokenizer.texts_to_sequences([text]))  # Tokenize the text
        text = pad_sequences(text, padding='post', maxlen=maxlen)  # Pad the sequence

        print("Uploaded text:", text)

        # Step 3: Predict the data
        result = model.predict(text)

        cutoff = 0.86
        # Apply the cutoff to determine the class label
        predicted_label = 1 if result >= cutoff else 0

        # Step 4: Return the predicted label
        return {"predicted_label": predicted_label}

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

# port = os.environ.get("PORT", 8080)
# print(f"Listening to http://0.0.0.0:{port}")
# uvicorn.run(app, host='0.0.0.0', port=port)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
