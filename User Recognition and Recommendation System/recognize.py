import os
import cv2
import pickle

from imutils import paths
import face_recognition


def train_model(model_name="face_enc_live",
                training_dataset="face_data/training"):
  # get paths of each file in folder named Images
  # Images here contains my data(folders of various persons)

  imagePaths = list(paths.list_images(training_dataset))
  knownEncodings = []
  knownNames = []

  # loop over the image paths
  for (i, imagePath) in enumerate(imagePaths):
    print("Working on image: ", i + 1, " of ", len(imagePaths))
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb, model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    if len(encodings) > 1:
      print("More than one face found. Skipping the image...")
      continue
    # loop over the encodings
    for encoding in encodings:
      knownEncodings.append(encoding)
      knownNames.append(name)

  print(f"\nTotal faces saved: {len(list(set(knownNames)))}")
  # save emcodings along with their names in dictionary data
  data = {"encodings": knownEncodings, "names": knownNames}
  # use pickle to save data into a file for later use
  f = open(model_name, "wb")
  f.write(pickle.dumps(data))
  f.close()


def identify_face(image, faceCascade, model):
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # convert image to Greyscale for haarcascade
  # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # faces = faceCascade.detectMultiScale(gray,
  #                                      scaleFactor=1.1,
  #                                      minNeighbors=5,
  #                                      minSize=(60, 60),
  #                                      flags=cv2.CASCADE_SCALE_IMAGE)

  encodings = face_recognition.face_encodings(rgb)

  if len(encodings) > 1:
    print("Found more than one encoding. Most likely, the image has multiple faces")
    return None, None
  elif len(encodings) == 0:
    # print("No encoding match found for the given face.")
    return None, None

  names = []
  matches = face_recognition.compare_faces(model["encodings"], encodings[0])

  if True in matches:
    # Find positions at which we get True and store them
    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    counts = {}
    # loop over the matched indexes and maintain a count for
    # each recognized face face
    for i in matchedIdxs:
      # Check the names at respective indexes we stored in matchedIdxs
      name = model["names"][i]
      # increase count for the name we got
      counts[name] = counts.get(name, 0) + 1
      # set name which has highest count
      name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)
    if len(names) > 1:
      # print("Found more than one match for facial encoding")
      return None, None

  else:
    # print("No encoding matches found. Possibly a new face")
    return None, None
  if len(names) == 0:
    return None
  return names[0], image


def validate_model(model_name, testing_dataset="face_data/testing"):
  # find path of xml file containing haarcascade file
  cascPathface = os.path.dirname(
      cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
  # load the harcaascade in the cascade classifier
  faceCascade = cv2.CascadeClassifier(cascPathface)
  # load the known faces and embeddings saved earlier
  data = pickle.loads(open(model_name, "rb").read())

  prediction_status = {'correct': 0, 'incorrect': 0, 'unidentified': 0}

  # testing on validation dataset
  imagePaths = list(paths.list_images(testing_dataset))

  # loop over the image paths
  for (i, imagePath) in enumerate(imagePaths):
    try:
      name = imagePath.split(os.path.sep)[-2]
    except ValueError:
      name = "Unknown"
    image = cv2.imread(imagePath)
    pred_name, pred_image = identify_face(image, faceCascade, data)
    if pred_name is None:
      print(
          f"Face could not be recognized for {imagePath.split(os.path.sep)[-1]}")
    else:
      print(f"Original face: {name}; Identified face: {pred_name}")
      if pred_name != name:
        print("Wrong face match found.")
    if pred_name == name:
      prediction_status['correct'] += 1
    elif pred_name == None:
      prediction_status['unidentified'] += 1
    elif pred_name != name:
      prediction_status['incorrect'] += 1
    print()


def make_prediction_image(model_name, image):
  # find path of xml file containing haarcascade file
  cascPathface = os.path.dirname(
      cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
  # load the harcaascade in the cascade classifier
  faceCascade = cv2.CascadeClassifier(cascPathface)
  # load the known faces and embeddings saved earlier
  data = pickle.loads(open(model_name, "rb").read())

  image = cv2.imread(image)
  pred_name, pred_image = identify_face(image, faceCascade, data)
  print(f"Identified face: {pred_name}")
  return pred_name


def make_prediction(model_name, display=False):
  # find path of xml file containing haarcascade file
  cascPathface = os.path.dirname(
      cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
  # load the harcaascade in the cascade classifier
  faceCascade = cv2.CascadeClassifier(cascPathface)
  # load the known faces and embeddings saved earlier
  data = pickle.loads(open(model_name, "rb").read())

  try:
    print("Streaming started")
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # loop over frames from the video file stream
    while True:
      pred_name, pred_image = None, None
      # grab the frame from the threaded video stream
      ret, frame = video_capture.read()
      pred_name, pred_image = identify_face(frame, faceCascade, data)
      if pred_name is not None:
        print(f"Identified face: {pred_name}")
        video_capture.release()
        cv2.destroyAllWindows()
        return pred_name
      if display:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  finally:
    video_capture.release()
    cv2.destroyAllWindows()
