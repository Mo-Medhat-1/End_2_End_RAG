"""
https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits

10 requests per minute
50 requests per day
8,000 input tokens per request
4,000 output tokens per request
2 concurrent requests

"""
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)

import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(GITHUB_TOKEN),
)


def call_openai_images(image_file, model_name="openai/gpt-4o"):

    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant that describes images in details."),
            UserMessage(
                content=[
                    TextContentItem(text="What's in this image?"),
                    ImageContentItem(
                        image_url=ImageUrl.load(
                            image_file=image_file,
                            image_format="jfif",
                            detail=ImageDetailLevel.LOW)
                    ),
                ],
            ),
        ],
        model=model_name,
    )

    return str(response.choices[0].message.content)

def call_openai_chat(messages, model="openai/gpt-4o", temperature=0.1):
    sdk_messages = []
    for m in messages:
        if m["role"] == "system":
            sdk_messages.append(SystemMessage(m["content"]))
        elif m["role"] == "user":
            sdk_messages.append(UserMessage(m["content"]))
        elif m["role"] == "assistant":
            sdk_messages.append(AssistantMessage(m["content"]))

    response = client.complete(
        messages=sdk_messages,
        model=model,
        temperature=temperature,
        max_tokens=300,
    )
    return response.choices[0].message.content

print("gpt called")
# print(call_openai_chat([
#                 {"role": "system", "content": "You are a helpful instructor."},
#                 {"role": "user", "content": "What is Graph DB?"}
#             ]))

print(call_openai_images("..\data\raw\family.jfif"))