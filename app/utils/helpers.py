import cv2


def draw_annotations(image, annotations):
    """
    Draws bounding boxes and class IDs on an image based on the provided annotations.

    Parameters:
    - image: The image on which to draw the annotations (numpy array).
    - annotations: A list of dictionaries, where each dictionary contains 'class_id', 'confidence', and 'bbox' keys.

    The function modifies the input image in-place.
    """

    for annotation in annotations:
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max, _ = annotation["bbox"]
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))

        # Extract class ID and confidence
        class_id = annotation["class_id"]
        confidence = annotation["confidence"]

        # Set rectangle color and thickness
        color = (0, 255, 0)  # Green
        thickness = 2

        # Draw the bounding box
        cv2.rectangle(image, start_point, end_point, color, thickness)

        # Set the text to display class ID and confidence
        text = f"{class_id}: {confidence:.2f}"

        # Set text font, scale, and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        # Put the text on the image
        cv2.putText(image, text, (int(x_min), int(y_min) - 10), font, font_scale, color, text_thickness)
    # This function modifies the image in-place, so there's no need to return the image
