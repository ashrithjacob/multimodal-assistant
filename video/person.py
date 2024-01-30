"""
Before running ensure you download the weights in a `weights` folder:
%cd {HOME}
!mkdir {HOME}/weights
%cd {HOME}/weights
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

"""

import os
import groundingdino
import os
import supervision as sv
import torch
from torchvision.ops import box_convert
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from deepface import DeepFace
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class FaceRecognition:
    def __init__(self):
        self.HOME = os.getcwd()
        self.backends = [
            "opencv",
            "ssd",
            "dlib",
            "mtcnn",
            "retinaface",
            "mediapipe",
            "yolov8",
            "yunet",
            "fastmtcnn",
        ]
        self.models = [
            "VGG-Face",
            "Facenet",
            "Facenet512",
            "OpenFace",
            "DeepFace",
            "DeepID",
            "ArcFace",
            "Dlib",
            "SFace",
        ]
        self.CONFIG_PATH = os.path.join(
            self.HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )

        if (self.CONFIG_PATH, "; exist:", os.path.isfile(self.CONFIG_PATH)):
            self.WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
            self.WEIGHTS_PATH = os.path.join(self.HOME, "weights", self.WEIGHTS_NAME)
            self.MODEL = load_model(self.CONFIG_PATH, self.WEIGHTS_PATH)

    """
    uses grounding dinos to box faces
    """
    def _box_face(
        self,
        image_path,
        text_prompt="human face",
        box_threshold=0.3,
        text_threshold=0.25,
    ):
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.MODEL,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return {
            "image": image_source,
            "boxes": boxes,
            "logits": logits,
            "phrases": phrases,
        }

    @classmethod
    def inflate_box(cls, boxes):
        boxes[:, 2:] *= 1.7  # scaling factor(fix box overlap issues)
        return boxes

    def display(self, image_path):
        properties = self._box_face(image_path=image_path)
        annotated_frame = annotate(
            image_source=properties["image"],
            boxes=properties["boxes"],
            logits=properties["logits"],
            phrases=properties["phrases"],
        )
        sv.plot_image(annotated_frame, (16, 16))

    def get_xyxy_mask(self, image_path):
        properties = self._box_face(image_path=image_path)
        img = properties["image"]
        boxes = properties["boxes"]
        boxes = FaceRecognition.inflate_box(boxes)
        h, w, _ = img.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xyxy = [tuple(map(int, map(round, mask))) for mask in xyxy]
        return img, xyxy

    def crop(self, image_path):
        img_list = []
        img, masks = self.get_xyxy_mask(image_path)
        for mask in masks:
            x1, y1, x2, y2 = map(int, map(round, mask))
            img_list.append(img[y1:y2, x1:x2])
        return img_list

    """
    use deepface to box faces
    """
    def deepface_box_face(self, image_path):
        boxes = []
        phrase = []
        confidence = []
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name=self.models[2],
            detector_backend=self.backends[4],
            enforce_detection=True,
        )
        image_source, _ = load_image(image_path)
        h, w, _ = image_source.shape
        for e in embedding_obj:
            box = list(e["facial_area"].values())
            box = [x / w if i % 2 == 0 else x / h for i, x in enumerate(box)]
            box[0] += box[2] / 2
            box[1] += box[3] / 2
            boxes.append(box)
            confidence.append(e["face_confidence"])
            phrase.append("face_confidence")
        return {
            "image": image_source,
            "boxes": torch.Tensor(boxes),
            "logits": torch.Tensor(confidence),
            "phrases": phrase,
        }

    def display_deepface(self, image_path):
        properties = self.deepface_box_face(image_path=image_path)
        annotated_frame = annotate(
            image_source=properties["image"],
            boxes=properties["boxes"],
            logits=properties["logits"],
            phrases=properties["phrases"],
        )
        sv.plot_image(annotated_frame, (16, 16))

    def create_face_embedding(self, image_path):
        embeddings = []
        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name=self.models[2],
            detector_backend=self.backends[4],
            enforce_detection=False,
        )
        embeddings = [e["embedding"] for e in embedding_obj]
        return embeddings
    
    def get_embedding(self, image_path):
        emb = {}
        embeddings=model.create_face_embedding(image_path)
        emb[image_path]=embeddings


load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PersonaRecognition:
    def __init__(self, path_to_db="./"):
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
        )
        # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
        self.client = chromadb.Client(Settings(persist_directory="./"))
        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        self.collection = self.client.get_or_create_collection(
            "json", embedding_function=openai_ef
        )
        

    def get_entry_numbers(self):
        number_of_entries = self.collection.count()
        return number_of_entries

    def add_features(self, feature_list):
        number_of_entries = self.get_entry_numbers()
        self.collection.add(
            documents=[
                str(feature_list)
            ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            # metadatas=[{"key":key_value}], # filter on these!
            ids=[str(number_of_entries)],  # unique for each document
        )

    def retrieve(self, query):
        results = self.collection.query(
            query_texts=query,
            n_results=2,
            # where={"metadata_field": "is_equal_to_this"}, # optional filter
            # where_document={"$contains":"search_string"}  # optional filter
        )
        print(results)

    def remove_none_values(input_dict):
        # Use dictionary comprehension to filter out key-value pairs with values equal to "none"
        filtered_dict = {key: value for key, value in input_dict.items() if value.lower() != "none"}
        return filtered_dict

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

def main():
    image_folder = "./demo/"
    #img_list = ["img-2.jpg", "img-73.jpg", "img-90.jpg", "c-0-img-95.jpg", "img-71.jpg", "img-69.jpg"]
    img_list = ["img-2.jpg"]
    face_recognition = FaceRecognition()

    for img in img_list:
        face_recognition.display_deepface(f'{image_folder}{img}')
        embedding = face_recognition.get_embedding(img)
        print(embedding)

main()





"""
dictionary1 = {
    "gender": "male",
    "ethnicity": "African descent",
    "physique": "medium to large build, noticeable beard and mustache",
}

dictionary2 = {
    "gender": "female",
    "ethnicity": "African descent",
    "physique": "medium build, no facial hair",
}

dictionary3 = {
    "gender": "male",
    "ethnicity": "latino",
    "physique": "medium build, no facial hair",
}

model = PersonaRecognition()
# model.add_features(dictionary1)
model.add_features(dictionary2)
model.add_features(dictionary3)
print(model.get_entry_numbers())
model.retrieve(str(dictionary1))
"""

"""
model = FaceRecognition()
img_path = "./demo/img-90.jpg"
#model.display_deepface(img_path)
img_list = model.crop(img_path)
plt.imshow(img_list[0])
plt.show()
"""
"""
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
query_embedding = model.encode('noticeable beard and mustache')
passage_embedding = model.encode([
    "very light beard",
    "mustache",
    "clean shaven"
])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

from openai import OpenAI
client = OpenAI(api_key="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")

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
query_embedding = get_embedding('noticeable beard and mustache')
t1 = get_embedding("very light beard")
t2 = get_embedding("mustache")
t3 = get_embedding("clean shaven")
print("Similarity:", util.dot_score(query_embedding, [t1,t2,t3]))

#try fuzzy match
"""
