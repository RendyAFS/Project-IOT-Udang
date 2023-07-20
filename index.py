import cv2
import numpy as np

def count_detected_black_shrimp_seed(image_path, sample_paths):
    # Load the test image
    image = cv2.imread(image_path)

    # Convert the test image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize a variable to count the total detected black shrimp seeds
    total_detected_shrimp_seeds = 0

    for sample_path in sample_paths:
        # Load the sample image
        sample = cv2.imread(sample_path)

        # Convert the sample image to grayscale
        grayscale_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        # Get the dimensions of the sample image
        sample_height, sample_width = grayscale_sample.shape[::-1]

        # Perform template matching
        result = cv2.matchTemplate(grayscale_image, grayscale_sample, cv2.TM_CCOEFF_NORMED)

        # Set a threshold for considering a match
        threshold = 0.8

        # Find locations where the result is above the threshold
        loc = np.where(result >= threshold)

        # Draw rectangles around the detected regions
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + sample_width, pt[1] + sample_height), (0, 0, 255), 2)
            total_detected_shrimp_seeds += 1

    # Display the result image with detected black shrimp seeds
    cv2.imshow("Detected Black Shrimp Seeds", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return total_detected_shrimp_seeds

if __name__ == "__main__":
    image_path = "test.png"  # Ganti dengan nama file gambar yang ingin dideteksi

    sample_paths = [
        "sample1.png",
        "sample2.png",
        "sample3.png",
        "sample4.png"
    ]  # Ganti dengan daftar nama file gambar sampel

    detected_count = count_detected_black_shrimp_seed(image_path, sample_paths)
    print(f"Jumlah bibit udang yang terdeteksi: {detected_count}")
