import base64
import cv2

def reshape_frame_to_512(image):
    resized_image = cv2.resize(image, (512, 512))
    return resized_image

def image_to_frame(image_path):
    frame = cv2.imread(image_path)
    return frame

def frame_to_image(frame, image_path):
    cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return image_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_frame_cv(frame):
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')