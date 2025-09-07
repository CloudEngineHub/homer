import os
import json
import re
from PIL import Image
import numpy as np
import google.generativeai as genai

class GeminiWrapper:
    def __init__(self, api_key, model="gemini-2.5-pro", verbose=True):
        self.api_key = api_key
        self.model_name = model
        self.verbose = verbose
        self._configure_model()

    def _configure_model(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _clean_response(self, text):
        text = text.strip()
        return re.sub(r"^```(?:json)?\n|\n```$", "", text)

    def point_to_object(self, image, prompt="object"):
        """
        Args:
            image: Can be a PIL.Image.Image, np.ndarray, or path (str).
        """
        if isinstance(image, str):
            # path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            pil_img = Image.open(image)
        elif isinstance(image, np.ndarray):
            # numpy (BGR or RGB)
            if image.ndim == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise TypeError("Unsupported image type. Must be str, np.ndarray, or PIL.Image")

        if self.verbose:
            print(f"Sending prompt to Gemini: {prompt}")

        full_prompt = (
            f"Point to the {prompt}. "
            "The answer should follow the json format: "
            '[{\"point\": <point>, \"label\": <label1>}, ...]. '
            "The points are in [y, x] format normalized to 0-1000."
        )

        print("Waiting for response")
        response = self.model.generate_content([full_prompt, pil_img])
        raw_text = response.text.strip()
        print("Response received")

        if self.verbose:
            print("Gemini response:")
            print(raw_text)

        cleaned_json = self._clean_response(raw_text)

        try:
            data = json.loads(cleaned_json)
            if not data or "point" not in data[0]:
                raise ValueError("Response JSON missing 'point' data.")
        except Exception as e:
            print("Error parsing Gemini response:", e)
            return None

        # Convert normalized [y, x] to absolute [cx, cy]
        point_norm = data[0]["point"]
        height, width = np.array(pil_img).shape[:2]

        cy = int((point_norm[0] / 1000.0) * height)
        cx = int((point_norm[1] / 1000.0) * width)

        if self.verbose:
            print(f"Success! Circle coordinates: cx = {cx}, cy = {cy}")

        return {"cx": cx, "cy": cy}
