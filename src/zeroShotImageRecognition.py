from transformers import pipeline
from PIL import Image
import requests

filepath = "src/resources/zeroShotImage.model"

checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
detector.save_pretrained(filepath)

url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)

predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl"])
