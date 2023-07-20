import cv2
import tensorflow as tf
import numpy as np
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_width = 10  # Set the desired width of the image
    aspect_ratio = image_rgb.shape[1] / image_rgb.shape[0]
    target_height = int(target_width / aspect_ratio)
    target_shape = (target_width, target_height)
    image_resized = cv2.resize(image_rgb, target_shape)
    image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension (1, target_width, target_height, 3)
    return image_tensor / 255.0


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def detect_bibit_udang(image_path, model):
    image_tensor = preprocess_image(image_path)
    input_name = list(model.signatures['serving_default'].inputs.keys())[0]
    detections = model(image_tensor)

    threshold = 0.5
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int64)
    detection_boxes = detections['detection_boxes'][0].numpy()

    detected_objects = []
    for i in range(len(detection_scores)):
        if detection_scores[i] >= threshold:
            bbox = detection_boxes[i]
            class_id = detection_classes[i]
            detected_objects.append((bbox, class_id))

    detections = {
        'detection_scores': detection_scores,
        'detection_classes': detection_classes,
        'detection_boxes': detection_boxes
    }

    return detected_objects, detections


if __name__ == "__main__":
    # Ganti 'D:/BISA!/OpenCvUdang/model/' dengan jalur menuju model yang telah dilatih pada bibit udang.
    model_path = 'D:/BISA!/OpenCvUdang/model/'
    print("Files in model directory:", os.listdir(model_path))

    # Load the model
    model = load_model(model_path)

    # Ganti 'path/to/your/image.jpg' dengan jalur menuju gambar yang ingin Anda deteksi.
    image_path = 'test.png'
    detected_objects, detections = detect_bibit_udang(image_path, model)

    image = cv2.imread(image_path)
    for bbox, class_id in detected_objects:
        y_min, x_min, y_max, x_max = bbox
        x_min, y_min, x_max, y_max = int(x_min * image.shape[1]), int(y_min * image.shape[0]), int(x_max * image.shape[1]), int(y_max * image.shape[0])
        class_name = f"Class {class_id}"  # Ganti ini dengan label kelas yang sesuai
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Simpan gambar yang telah dideteksi dengan kotak bingkai
    output_image_path = 'detected_image.png'
    cv2.imwrite(output_image_path, image)
