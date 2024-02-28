import json
import os
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

GPT_MODEL = "gpt-3.5-turbo-0613"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    },
]

messages = []
messages.append(
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)
messages.append({"role": "user", "content": "What's the weather like today"})
messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(messages, tools=tools)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
fn_name=assistant_message.tool_calls[0].function.name
args=json.loads(assistant_message.tool_calls[0].function.arguments)

def get_current_weather(location, format):
    return location + format

function = globals()[fn_name]
print(args)
result = function(**args)
print(result)
