from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import requests
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Set up your Hugging Face API details
# API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large/"
API_URL = "https://72c1-34-169-38-85.ngrok-free.app/"
# headers = {
#     "Authorization": "Bearer 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"
# }  # Replace with your API token


def query_image(image_data):
    """Send image to Hugging Face API for captioning."""
    # response = requests.post(API_URL, headers=headers, data=image_data)
    files = {"file": image_data}
    response = requests.post(f"{API_URL}/caption", files=files)
    print(response)
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
    files = {"file": data}
    response = requests.post(f"{API_URL}/caption", files=files)
    print(response)
    # image_data = base64.b64decode(data.split(",")[1])

    # # Convert to numpy array
    # nparr = np.frombuffer(image_data, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # # Save the frame as a temporary image file
    # temp_image_path = "temp.jpg"
    # cv2.imwrite(temp_image_path, img)

    # # Query the image captioning API
    # try:
    #     result = query_image(image_data)
    #     caption = result
    #     # if "generated_text" in result:
    #     #     caption = result["generated_text"]
    #     # else:
    #     #     caption = "No caption generated or incorrect response format."
    # except Exception as e:
    #     caption = f"Error: {str(e)}"

    # Emit the caption back to the client
    emit("caption", response)


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=8080)
