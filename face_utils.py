import face_recognition
from PIL import Image
import numpy as np
import os


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
