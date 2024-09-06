from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO
import cv2
from caption_service import CaptionService
import os
import numpy as np
import base64
from PIL import Image
import face_recognition
from flask_cors import CORS


def load_known_faces(dataset_path):
    """
    Load known face encodings and corresponding names from a dataset directory.

    Args:
        dataset_path (str): Path to the dataset containing directories of people.

    Returns:
        (list, list): A tuple of two lists - known_face_encodings and known_face_names.
    """
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            try:
                # Use PIL to open the image
                with Image.open(image_path) as img:
                    print(f"Loading image: {image_path}")

                    # Convert to RGB mode
                    img = img.convert("RGB")

                    # Convert to numpy array
                    rgb_image = np.array(img)

                # Detect faces in the image
                face_locations = face_recognition.face_locations(rgb_image)

                if not face_locations:
                    print(f"No face found in image: {image_path}")
                    continue

                # Compute face encodings
                face_encodings = face_recognition.face_encodings(
                    rgb_image, face_locations
                )

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
                    print(f"Successfully encoded face for: {person_name}")
                else:
                    print(f"Could not compute face encoding for image: {image_path}")

            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                import traceback

                traceback.print_exc()

    return known_face_encodings, known_face_names


def recognize_faces(image_path, known_face_encodings, known_face_names):
    """
    Recognize faces in an image given known face encodings.

    Args:
        image_path (str): Path to the image to recognize faces from.
        known_face_encodings (list): List of known face encodings.
        known_face_names (list): List of names corresponding to known face encodings.

    Returns:
        list: A list of recognized face names in the image.
    """
    recognized_faces = []

    try:
        # Load the image
        with Image.open(image_path) as img:
            # Convert to RGB mode
            img = img.convert("RGB")

            # Convert to numpy array
            rgb_image = np.array(img)

        # Find all faces in the image
        face_locations = face_recognition.face_locations(rgb_image)

        if not face_locations:
            print("No faces found in the image.")
            return recognized_faces

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Loop through each face found in the unknown image
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            recognized_faces.append(name)

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        import traceback

        traceback.print_exc()

    return recognized_faces


app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

DATASET_PATH = (
    "/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/dataset"
)

known_face_encodings, known_face_names = load_known_faces(DATASET_PATH)

# Initialize models
caption_service = CaptionService(
    model_path="/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/blip_model.pth",
    processor_path="/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/blip_processor.pkl",
)


def process_image(image_data):
    # Decode the base64 image string into image bytes
    image_bytes = base64.b64decode(image_data.split(",")[1])

    # Convert the image bytes to a numpy array and decode it as an image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


@socketio.on("request_caption")
def handle_caption_request(data):
    """Handles the base64 image sent via SocketIO and generates a caption."""
    try:
        img = process_image(data["image"])

        # Save the image as a temporary file for processing
        temp_image_path = "/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/temp.jpg"
        cv2.imwrite(temp_image_path, img)

        # Generate a caption using the local BLIP model
        caption = caption_service.generate_caption(temp_image_path)

        # Clean up the temporary image file
        os.remove(temp_image_path)

    except Exception as e:
        print(f"Error processing caption: {str(e)}")
        # caption = ""

    # Emit the caption back to the client
    emit("caption_result", {"caption": caption})


ocr = PaddleOCR(lang="en", use_angle_cls=True)


@socketio.on("request_ocr")
def handle_ocr_request(data):
    try:
        print("Debug: Starting OCR process")

        # Decode base64 image
        base64_image = data["image"]
        image_data = base64.b64decode(base64_image.split(",")[1])

        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))

        # Convert to OpenCV format (numpy array)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print("Debug: Image processed")
        print(f"Debug: Image shape: {img.shape}")

        # Perform OCR using PaddleOCR
        print("Debug: Beginning OCR")
        result = ocr.ocr(img)  # Pass the numpy array directly
        print("Debug: OCR completed")

        # Extract the OCR text from the results
        finaltext = ""
        for line in result[0]:
            finaltext += " " + line[1][0]

        finaltext = finaltext.strip()
        print(f"Debug: Extracted text: {finaltext[:100]}...")  # Print first 100 chars
        socketio.emit("ocr_result", {"text": finaltext})
    except Exception as e:
        print(f"Error processing OCR: {str(e)}")
        finaltext = "Error during OCR"

    print("Debug: Emitting OCR result")

    print("Debug: OCR result emitted")


def recognize_face(image_path):
    print(image_path)
    recognized_faces = recognize_faces(
        image_path, known_face_encodings, known_face_names
    )

    return jsonify({"recognized_faces": recognized_faces})


def caption():
    print("captioning...")


@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join("/tmp", image.filename)
    image.save(image_path)

    # Recognize faces in the uploaded image
    recognized_faces = recognize_faces(
        image_path, known_face_encodings, known_face_names
    )

    return jsonify({"recognized_faces": recognized_faces})


@app.route("/speech_command", methods=["POST"])
def speech_command():
    try:

        image = request.files["image"]
        image_path = os.path.join("/tmp", image.filename)
        image.save(image_path)
        print(image.save(image_path))
        command = request.form.get("command")
        print(command)
        commands = {
            "recognize": recognize_face,
            "caption": caption,
        }
        handler = commands.get(command)
        if handler:
            return handler(image_path)
        else:
            print(f"Unknown command: {commands}")

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]
        image_path = os.path.join(
            "/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/tmp",
            image.filename,
        )
        image.save(image_path)
        print(image.save(image_path))
        recognized_faces = recognize_faces(
            image_path, known_face_encodings, known_face_names
        )

        return jsonify({"recognized_faces": recognized_faces})

    except Exception as e:
        print("speech_command", e)
        return jsonify({"works": "no"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)
