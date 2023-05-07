SLACK_BOT_TOKEN = "xoxb-5182055281588-5214269744978-LV2BqdwVnaLCOdC1UCRgw02u"
SLACK_APP_TOKEN = "xapp-1-A055VL43YJK-5216729310196-4f3b165c7a269dfd99d79aa46a264bef09c0767656dcbed5dcc67adfa75087fc"
OPENAI_API_KEY  = "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"

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

# Function to generate response using OpenAI's GPT-4 model
def generate_response(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"]

# Function to post response to Slack channel
def post_response(channel_id, body, response, client):
    # Reply to thread 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"\n{response}")

if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()