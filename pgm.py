import cv2
import matplotlib.pyplot as plt

PATH = r"C:\Users\ve00yn139\Downloads\Database_134_Angiograms\48.pgm"
# Load the PGM image
image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if image is None:
    raise FileNotFoundError("The image file was not found or could not be opened.")

# Display the image
plt.imshow(image, cmap='gray')
plt.title('PGM Image')
plt.axis('off')  # Turn off axis numbers
plt.show()
