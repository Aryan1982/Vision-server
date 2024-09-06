from flask import Flask, jsonify, request
import os
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import requests
from flask_cors import CORS
from caption_service import CaptionService

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Set up your Hugging Face API details
API_URL = (
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
)
headers = {
    "Authorization": "Bearer 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"
    # "Authorization": "Bearer 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"
}


def query_image(image_data):
    """Send image to Hugging Face API for captioning."""
    response = requests.post(API_URL, headers=headers, data=image_data)
    return response.json()


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on_error_default
def default_error_handler(e):
    print(f"An error occurred: {str(e)}")


@socketio.on("image")
def handle_image(data):
    # Decode the base64 image
    image_data = base64.b64decode(data.split(",")[1])

    # Convert to numpy array
    # nparr = np.frombuffer(image_data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save the frame as a temporary image file
    # temp_image_path = "temp.jpg"
    # cv2.imwrite(temp_image_path, img)

    # Query the image captioning API
    try:
        result = query_image(image_data)
        caption = result
        # if "generated_text" in result:
        #     caption = result["generated_text"]
        # else:
        #     caption = "No caption generated or incorrect response format."
    except Exception as e:
        caption = f"Error: {str(e)}"

    # Emit the caption back to the client
    emit("caption", caption)


caption_service = CaptionService(
    model_path="/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/blip_model.pth",
    processor_path="/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/blip_processor.pkl",
)


# Define the API route to upload images and get captions
@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    image_path = os.path.join("/tmp", image.filename)
    image.save(image_path)

    # Generate the caption
    caption = caption_service.generate_caption(image_path)

    # Clean up the saved image
    os.remove(image_path)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=8080)
