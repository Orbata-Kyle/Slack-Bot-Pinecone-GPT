import os
import numpy as np
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pinecone
import openai
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import torch
import clip
from PIL import Image
import requests
from io import BytesIO

load_dotenv(dotenv_path="./my_secrets/DOT_ENV")  # Update the path to your DOT_ENV file

slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
slack_app_token = os.environ["SLACK_APP_TOKEN"]
openai_api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_index = os.environ["PINECONE_INDEX"]
pinecone_url = os.environ["PINECONE_URL"]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

def pinecone_fetch(index_name, ids):
    index = pinecone.Index(index_name)
    fetched_vectors = index.fetch(ids=ids)
    return fetched_vectors

def pinecone_upsert(index_name, items):
    with pinecone.connection(api_key=pinecone_api_key) as pinecone_client:
        return pinecone_client.upsert(index_name=index_name, items=items)

def response_to_vector(response_text):
    text_inputs = torch.tensor([clip.tokenize(response_text)]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0]

def image_to_vector(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    image_input = transform(img).unsqueeze(0).to(device)
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0]

def generate_response(prompt, image_url=None):
    openai.api_key = openai_api_key

    memory_vector, = pinecone_fetch(index_name=pinecone_index, ids=[prompt])

    if memory_vector is None:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message["content"]
        if image_url is None:
            memory_vector = response_to_vector(response_text)
        else:
            memory_vector = image_to_vector(image_url)
        
        pinecone_upsert(index_name=pinecone_index, items={prompt: memory_vector})

    else:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            memory=memory_vector.tolist(),
        )
        response_text = response.choices[0].message["content"]

    return response_text

client = WebClient(slack_bot_token)
bot_user_id = client.auth_test()["user_id"]

app = App(token=slack_bot_token)

@app.event("app_mention")
def handle_app_mention_events(body, client, logger):
    logger.info(body)
    event = body['event']
    bot_mention = f"<@{bot_user_id}>"

    if "files" in event:
        for file in event["files"]:
            if file["mimetype"].startswith("image/"):
                image_url = file["url_private_download"]
                logger.info(f"Received image input: {image_url}")
                prompt = f"Generate a caption for this image: {image_url}"
                response = generate_response(prompt, image_url)
                post_response(event["channel"], body, response, client)
    elif "text" in event:
        text = event["text"]
        if bot_mention in text:
            text_input = text.replace(bot_mention, "").strip()
            logger.info(f"Received text input: {text_input}")
            response = generate_response(text_input)
            post_response(event["channel"], body, response, client)
            
def post_response(channel_id, body, response, client):
    client.chat_postMessage(
        channel=body["event"]["channel"], 
        thread_ts=body["event"]["event_ts"],
        text=f"\n{response}"
    )

if __name__ == "__main__":
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_url)
        handler = SocketModeHandler(app, slack_app_token)
        handler.start()
    finally:
        pinecone.deinit()

