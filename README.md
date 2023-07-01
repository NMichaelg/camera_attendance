# Face Recognition

Face Recognition is a Python library that provides functionalities for performing face recognition tasks using pre-trained models. It allows you to detect faces in images, extract face embeddings, store them in a database, and perform face search to identify similar faces.

## Features

- Face detection using the MTCNN model
- Face embedding extraction using the FaceNet model
- Database integration for storing face embeddings
- Faiss index for fast face search
- Insert new faces into the database
- Retrieve top matching people based on an input image

## Installation

1. Clone the repository:

```

git clone https://github.com/LeTriet17/camera_attendance.git

```

2. Change to the project directory:

```

cd face-recognition

```

3. Install the required dependencies:

```

pip install -r requirements.txt

```

## Usage

1. Connect to the database by modifying the connection details in the `FaceRecognition` class constructor (`face_recognition.py` file).

2. Load the pre-trained FaceNet model and MTCNN model by calling the `load_facenet_model()` and `load_mtcnn_model()` methods, respectively.

3. Create a table in the database to store face embeddings by calling the `create_embeddings_table()` method.

4. Insert face embeddings from a directory structure into the database by calling the `insert_embeddings_to_database(base_dir)` method, where `base_dir` is the base directory containing the images for different people.

5. Build the Faiss index for fast face search by calling the `build_faiss_index()` method.

6. Insert a new person's face embedding into the database by calling the `insert_new_person(image_path, person_name)` method, where `image_path` is the path to the image of the new person and `person_name` is the name of the new person.

7. Retrieve the top matching people based on an input image by calling the `get_top_matching_people(image_path, k)` method, where `image_path` is the path to the input image and `k` is the number of top matching people to retrieve.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MTCNN face detection model: Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10), 1499-1503.
- The FaceNet face embedding model: Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 815-823).
- The Faiss library for efficient similarity search and clustering of dense vectors: Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(1), 95-105.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
