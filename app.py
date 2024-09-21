import gradio as gr
from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request
import uvicorn
import threading
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the environment variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")

client = InferenceClient("microsoft/Phi-3-mini-4k-instruct", token=hf_token)

app = FastAPI()

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    # Collect tokens from the model
    for msg in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        # Only append new tokens if they exist
        token = msg.choices[0].delta.content
        if token:
            response += token

    return response  # Return the complete response directly

# Gradio Interface
def launch_gradio():
    demo = gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
            gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
            gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        ],
    )
    demo.launch(server_name="localhost", server_port=7860)

# Endpoint to access the API via FastAPI
@app.post("/generate-response/")
async def generate_response(request: Request):
    body = await request.json()
    message = body["message"]
    history = body.get("history", [])
    system_message = body.get("system_message", "You are a friendly Chatbot.")
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.95)

    # Call respond and return the complete response
    response = respond(message, history, system_message, max_tokens, temperature, top_p)
    print(response)
    return {"response": response}

# Launch Gradio in a separate thread so it doesn't block the API
if __name__ == "__main__":
    threading.Thread(target=launch_gradio, daemon=True).start()
    uvicorn.run(app, host="localhost", port=8000)
