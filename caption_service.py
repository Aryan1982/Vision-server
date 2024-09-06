import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pickle


class CaptionService:
    def __init__(self, model_path, processor_path):
        # Load the processor
        with open(processor_path, "rb") as f:
            self.processor = pickle.load(f)

        # Load the BLIP model and weights
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def generate_caption(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path)
        inputs = self.processor(images=img, return_tensors="pt")

        # Generate the caption
        with torch.no_grad():
            out = self.model.generate(**inputs)

        # Decode the generated caption
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
