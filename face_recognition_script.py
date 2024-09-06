import face_recognition
from PIL import Image
import numpy as np
import os


def load_known_faces(dataset_path):
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
                    print(f"Image info: {img}")

                    # Convert to RGB mode
                    img = img.convert("RGB")

                    # Convert to numpy array
                    rgb_image = np.array(img)

                    print(f"Numpy array shape: {rgb_image.shape}")
                    print(f"Numpy array dtype: {rgb_image.dtype}")

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
    try:
        # Load the image
        with Image.open(image_path) as img:
            print(f"Loading image to recognize: {image_path}")
            print(f"Image info: {img}")

            # Convert to RGB mode
            img = img.convert("RGB")

            # Convert to numpy array
            rgb_image = np.array(img)

            print(f"Numpy array shape: {rgb_image.shape}")
            print(f"Numpy array dtype: {rgb_image.dtype}")
            print(rgb_image)
            if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
                raise ValueError("Image must be RGB (3 channels)")

        # Find all faces in the image
        face_locations = face_recognition.face_locations(rgb_image)

        if not face_locations:
            print("No faces found in the image.")
            return

        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
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

            print(
                f"Found {name} at location (top: {top}, right: {right}, bottom: {bottom}, left: {left})"
            )

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        print("Error details:")
        import traceback

        traceback.print_exc()


# Define the path to your dataset
dataset_path = (
    "/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/dataset"
)

# Load known faces
print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces(dataset_path)
print(f"Loaded {len(known_face_encodings)} known faces")

# Path to the image you want to recognize faces in
image_to_recognize = (
    "/Users/aryanthakor/projects/practice-projects/team-troubleshoot/vision/biden.jpeg"
)

# Perform face recognition
print(f"Recognizing faces in {image_to_recognize}")
recognize_faces(image_to_recognize, known_face_encodings, known_face_names)
