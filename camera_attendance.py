import cv2
import os
import psycopg2
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import faiss
from PIL import Image
from align_trans import get_reference_facial_points, warp_and_crop_face
import time
import logging
import matplotlib.pyplot as plt
logging.getLogger().handlers = []

class FaceRecognition:
   def __init__(self, host, database, user, password):
      self.host = host
      self.database = database
      self.user = user
      self.password = password
      self.conn = None
      self.cursor = None
      self.model = None
      self.index = None
      self.mtcnn = None

      # Configure logging
      logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

      # Create a file handler and set the logging level
      file_handler = logging.FileHandler("face_recognition.log")
      file_handler.setLevel(logging.INFO)

      # Create a formatter and add it to the file handler
      formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
      file_handler.setFormatter(formatter)

      # Add the file handler to the logger
      logging.getLogger().addHandler(file_handler)

      self.logger = logging.getLogger(__name__)
      
   def connect_to_database(self):
      # Connect to PostgreSQL database
      self.logger.info("Connecting to the database...")
      self.conn = psycopg2.connect(
         host=self.host,
         database=self.database,
         user=self.user,
         password=self.password
      )
      self.cursor = self.conn.cursor()
      self.logger.info("Connected to the database.")

   def close_database_connection(self):
      # Close the database connection
      self.logger.info("Closing the database connection...")
      # Close the database connection
      self.cursor.close()
      self.conn.close()
      self.logger.info("Database connection closed.")

   def load_facenet_model(self):
      # Load FaceNet model
      self.logger.info("Loading FaceNet model...")
      self.model = InceptionResnetV1(pretrained='vggface2').eval()
      self.logger.info("FaceNet model loaded.")

   def load_mtcnn_model(self):
      # Load MTCNN model for face detection
      self.logger.info("Loading MTCNN model...")
      self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=10,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
      self.logger.info("MTCNN model loaded.")

   def preprocess_image(self, image):
      # Image preprocessing
      transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize((160, 160)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      ])
      preprocessed_image = transform(image).unsqueeze(0)
      return preprocessed_image

   def detect_face(self, image):
      # Perform face detection using MTCNN
      #self.logger.info("Performing face detection using MTCNN...")
      bounding_boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
      try:
         bounding_boxes = bounding_boxes[probs > 0.9]
         
         aligned_faces = []
         
         for box, landmark in zip(bounding_boxes, landmarks):
            x, y, w, h = box.astype(int)
            
            # Perform face alignment
            crop_size = 112
            scale = crop_size / 112.
            reference = get_reference_facial_points(default_square=True) * scale
            facial5points = [[landmark[j][0], landmark[j][1]] for j in range(5)]
            
            # Warp and crop the face
            warped_face = warp_and_crop_face(np.array(image), facial5points, reference, crop_size=(crop_size, crop_size))
            aligned_faces.append(warped_face)
         #self.logger.info("Face detection and alignment completed.")
         return bounding_boxes, landmarks, aligned_faces
      except:
         #self.logger.info("No face detected.")
         return [],[],[]

   def create_embeddings_table(self):
      # Create table to store embeddings if it doesn't exist
      create_table_query = """
         CREATE TABLE IF NOT EXISTS embeddings (
               id SERIAL PRIMARY KEY,
               person_name TEXT,
               image_path TEXT,
               embedding BYTEA
         )
      """
      self.cursor.execute(create_table_query)
      self.conn.commit()
   def delete_all_embeddings(self):
      self.connect_to_database()
      # Delete all existing embeddings
      delete_query = "DELETE FROM embeddings"
      self.cursor.execute(delete_query)
      self.conn.commit()
        
   def insert_embeddings_to_database(self, base_dir):
      self.connect_to_database()
      self.create_embeddings_table()
      self.load_facenet_model()
      self.load_mtcnn_model()

      start_time = time.time()
      # Iterate over the directory structure
      for root, dirs, files in os.walk(base_dir):
         for directory in dirs:
            person_name = directory
            self.logger.info(f"Inserting images for person: {person_name}")

            # Process each image in the directory
            for filename in os.listdir(os.path.join(root, directory)):
                  image_path = os.path.join(root, directory, filename)
                  img = cv2.imread(image_path)

                  # Detect face and get aligned faces
                  boxes, landmarks, aligned_faces = self.detect_face(img)

                  if len(boxes) > 0:
                     # Crop and preprocess face images
                     for face_img in aligned_faces:
                        # Preprocess image
                        preprocessed_image = self.preprocess_image(face_img)

                        # Extract embeddings using FaceNet
                        with torch.no_grad():
                              embedding = self.model(preprocessed_image)

                        # Convert embedding to bytes
                        serialized_data = pickle.dumps(embedding.numpy())

                        # Store the embedding in the PostgreSQL database
                        self.cursor.execute("INSERT INTO embeddings (person_name, image_path, embedding) VALUES (%s, %s, %s)",
                                             (person_name, image_path, psycopg2.Binary(serialized_data)))
                        self.conn.commit()
         elapsed_time = time.time() - start_time
         self.logger.info(f"Insertion completed in {elapsed_time:.2f} seconds.")
         self.logger.info("Finished inserting images for all people.")

      self.close_database_connection()
            
   def build_faiss_index(self):
      # Retrieve person names, embeddings, and image paths from the database
      self.cursor.execute("SELECT person_name, embedding, image_path FROM embeddings")
      rows = self.cursor.fetchall()

      embeddings = []
      person_names = []
      image_paths = []

      # Iterate over the rows and append embeddings, person names, and image paths
      for row in rows:
         embedding = pickle.loads(row[1])
         embeddings.append(embedding)
         person_names.append(row[0])
         image_paths.append(row[2])


      # Convert embeddings to a numpy array
      embeddings = np.stack(embeddings)
      embeddings = embeddings.reshape(embeddings.shape[0], -1)
      # Build the Faiss index
      self.index = faiss.IndexFlatL2(embeddings.shape[1])
      self.index.add(embeddings)

      self.person_names = person_names
      self.image_paths = image_paths

      return person_names


   def insert_new_person(self, image_path, person_name):
      self.connect_to_database()
      self.load_facenet_model()
      self.load_mtcnn_model()

      # Read image
      img = cv2.imread(image_path)

      # Detect face and get aligned faces
      boxes, landmarks, aligned_faces = self.detect_face(img)

      if len(boxes) > 0:
         # Crop and preprocess face images
         for face_img in aligned_faces:
               # Preprocess the face image
               preprocessed_image = self.preprocess_image(face_img)

               with torch.no_grad():
                  new_embedding = self.model(preprocessed_image).numpy()

               # Store the new person's name and embedding in the database
               serialized_data = pickle.dumps(new_embedding)

               self.cursor.execute("INSERT INTO embeddings (person_name, embedding) VALUES (%s, %s)",
                                    (person_name, psycopg2.Binary(serialized_data)))
               self.conn.commit()
         # Rebuild the Faiss index
         self.build_faiss_index()
         self.logger.info("New person inserted successfully.")
      else:
         self.logger.info("No face detected in the image.")

      self.close_database_connection()

   
   def get_top_matching_people(self, image_path, k=5):
      self.connect_to_database()
      self.load_facenet_model()
      self.load_mtcnn_model()

      # Build Faiss index if it doesn't exist
      if self.index is None:
         self.logger.info("Building Faiss index...")
         person_names = self.build_faiss_index()
         self.logger.info("Faiss index built.")
      else:
         self.logger.info("Using existing Faiss index.")
         person_names = self.person_names
         image_paths = self.image_paths

      # Read image
      img = cv2.imread(image_path)
      self.logger.info("Detecting face and aligning faces...")
      # Detect face and get aligned faces
      boxes, landmarks, aligned_faces = self.detect_face(img)

      if len(boxes) > 0:
         self.logger.info("Face detected.")
         # Crop and preprocess face images
         preprocessed_images = []
         for face_img in aligned_faces:
            # Preprocess the face image
            preprocessed_image = self.preprocess_image(face_img)
            preprocessed_images.append(preprocessed_image)

         preprocessed_images = torch.cat(preprocessed_images)

         self.logger.info("Computing embeddings using FaceNet...")
         with torch.no_grad():
            new_embeddings = self.model(preprocessed_images).numpy()

         # Perform a search in the Faiss index
         D, I = self.index.search(new_embeddings, k)
         # Display the top-k similar images
         fig, axs = plt.subplots(1, k, figsize=(20, 10))
         for i in range(len(I)):
            self.logger.info(f"Query image: {image_path}")
            self.logger.info("Top matching people:")
            for j in range(len(I[i])):
                  person_index = I[i][j]
                  distance = D[i][j]
                  person_name = person_names[person_index]
                  person_image_path = self.image_paths[person_index]
                  self.logger.info(f"Rank {j + 1}: {person_name} (Distance: {distance}) - Image Path: {person_image_path}")
                  
                  # Load and display the top k images
                  top_k_img = cv2.imread(person_image_path)
                  axs[j].imshow(cv2.cvtColor(top_k_img, cv2.COLOR_BGR2RGB))
                  axs[j].set_title(f"Rank {j + 1}: {person_name}")

            plt.show()
            self.logger.info("---------")
      else:
         self.logger.info("No face detected in the image.")

      self.close_database_connection()

      
if __name__ == "__main__":
   # Create an instance of FaceRecognition class
   fr = FaceRecognition(host="192.168.0.188", database="test", user="jaylong", password="Triet1712002")
   #fr.delete_all_embeddings()
   # Set the base directory for embeddings insertion
   base_dir = 'AFDB_dataset_160_part1'

   # Insert embeddings to the database
   #fr.insert_embeddings_to_database(base_dir)

   # Path and name of the new person's image
   image_path = "AFDB_dataset_160_part1/dongyijin/dongyijin_0061.jpg"
   person_name = "Triet Le"

   # Insert a new person to the database
   #fr.insert_new_person(image_path, person_name)

   # Get the top matching people for a given image
   fr.get_top_matching_people(image_path)
