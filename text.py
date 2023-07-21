import cv2
import matplotlib.pyplot as plt
# Path to the Haar Cascade XML file for face detection
xml_file_path = 'C:/Users/TEJASWINI/Desktop/t/haarcascade_frontalface_default.xml'

# Check if the XML file exists and is accessible
try:
    with open(xml_file_path):
        print("Haar Cascade XML file found and accessible.")
except FileNotFoundError:
    print("Error: Haar Cascade XML file not found or inaccessible.")
    exit()

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(xml_file_path)

# Check if the Haar Cascade classifier is loaded properly
if face_cascade.empty():
    print("Error: Failed to load the Haar Cascade classifier.")
    exit()
else:
    print("Haar Cascade classifier loaded successfully.")

# Read and display an image
image_path = 'C:/Users/TEJASWINI/Desktop/t/image.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected faces
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
