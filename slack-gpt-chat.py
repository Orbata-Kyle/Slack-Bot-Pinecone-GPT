import pinecone
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path="./DOT_ENV") # Load environment variables from .env file

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

pinecone_index = os.getenv("PINECONE_INDEX", "your_default_index")
pinecone_vector_length = int(os.getenv("PINECONE_VECTOR_LENGTH", 768))

print("Pinecone index:", pinecone_index)
print("Pinecone vector length:", pinecone_vector_length)

# Function to fetch memory vectors from Pinecone
def pinecone_fetch(index_name, ids):
    index = pinecone.Index(index_name)
    fetched_vectors = index.fetch(ids=ids)
    return fetched_vectors

# Function to upsert memory vectors to Pinecone
def pinecone_upsert(index_name, items):
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-central1-gcp")
    with pinecone.connection(api_key=PINECONE_API_KEY) as pinecone_client:
        return pinecone_client.upsert(index_name=index_name, items=items)

from slack_sdk import WebClient

client = WebClient(SLACK_BOT_TOKEN)

import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai

app = App(token=SLACK_BOT_TOKEN)

# Handler for app_mention events
@app.event("app_mention")
def handle_app_mention_events(body, client, logger):
    logger.info(body)
    event = body['event']
    if "files" in event:
        # Image input
        for file in event["files"]:
            if file["mimetype"].startswith("image/"):
                image_url = file["url_private_download"]
                logger.info(f"Received image input: {image_url}")
                prompt = f"Generate a caption for this image: {image_url}"
                response = generate_response(prompt)
                post_response(event["channel"], body, response, client)
    elif "text" in event:
        # Text input
        text_input = event["text"]
        logger.info(f"Received text input: {text_input}")
        response = generate_response(text_input)
        post_response(event["channel"], body, response, client)

def generate_response(prompt):
    openai.api_key = OPENAI_API_KEY

    # Fetch the memory vector from Pinecone
    memory_vector, = pinecone_fetch(index_name=pinecone_index, ids=[prompt])

    if memory_vector is None:
        # No memory vector found, generate a new response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message["content"]

        # Generate a memory vector for the new response
        memory_vector = np.random.rand(pinecone_vector_length)

        # Store the memory vector in Pinecone
        pinecone_upsert(index_name=pinecone_index, items={prompt: memory_vector})

    else:
        # Memory vector found, generate a response using the memory vector
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


# Function to post response to Slack channel
def post_response(channel_id, body, response, client):
    # Reply to thread 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"\n{response}")

if __name__ == "__main__":
    try:
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        handler.start()
    finally:
        pinecone.deinit()
