import time
import os
import json
import aiohttp
import asyncio
import replicate
from dotenv import load_dotenv
from gradio_client import Client
import openai

# Load environment variables from the .env file
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")


class ImageToTextModel:
    # See API to set temperature and top_p
    def __init__(
        self,
        api_url_replicate="naklecha/cogvlm:ec3886f9ea85dd0aee216585be5e6d07b04c9650f7b8b08363a14eb89e207eb2",
    ):
        self.model = "REPLICATE"
        self.api_url = api_url_replicate
        self.api = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"),)
        self.prompt = """Describe what you see here? """

    @classmethod
    def write_to_json(cls, data, json_file_path):
        # Check if the JSON file already exists
        if os.path.exists(json_file_path):
            # If the file exists, load its content and update it with new data
            with open(json_file_path, "r") as json_file:
                existing_data = json.load(json_file)
                existing_data.update(data)

            # Write the updated data back to the JSON file
            with open(json_file_path, "w") as json_file:
                json.dump(existing_data, json_file, indent=2)
        else:
            # If the file doesn't exist, create a new one with the current data
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)

    def get_description(self, image_path):
        output = self.api.run(
            self.api_url,
            input={
                "image": open(image_path, "rb"),
                "prompt": self.prompt,
            },
        )
        return output

    def run(self, image_path, json_file_path="./json/description.json"):
        # Store in json
        image_filename = os.path.basename(image_path)
        description = self.get_description(image_path)
        data = {image_filename: description}
        ImageToTextModel.write_to_json(data, json_file_path)
        return description


if __name__ == "__main__":
    start = time.time()
    model = ImageToTextModel()
    model.run(f"./youtube_video/best_images/img-10.jpg")
    print(time.time() - start)
