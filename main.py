from src.classifier import classifier
from src.loader import loader
from src.preprocess import preprocess
from datetime import datetime
from PIL import Image
import numpy as np

def main():

    paths = [
        "/opt/project/batch-serving/misc/samples/brothy.jpg",
        "/opt/project/batch-serving/misc/samples/rice.jpg",
        "/opt/project/batch-serving/misc/samples/noodle.jpg",
        "/opt/project/batch-serving/misc/samples/brothy2.jpg",
        "/opt/project/batch-serving/misc/samples/rice2.jpg",
        "/opt/project/batch-serving/misc/samples/noodle2.jpg"
    ]
    test_model_path = "/opt/project/batch-serving/misc/model_repository/noodle_model.pt"
    # test_brothy_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/brothy.jpg"))
    # test_rice_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/rice.jpg"))
    # test_noodle_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/noodle.jpg"))
    # test_brothy_image2 = np.array(Image.open("/opt/project/batch-serving/misc/samples/brothy2.jpg"))
    # test_rice_image2 = np.array(Image.open("/opt/project/batch-serving/misc/samples/rice2.jpg"))
    # test_noodle_image2 = np.array(Image.open("/opt/project/batch-serving/misc/samples/noodle2.jpg"))
    
    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)
    test_loader = loader.JitScriptLoader(model_path=test_model_path)
    test_classifier = classifier.ImageClassifier(image_preprocess=test_preprocess, loader=test_loader)

    # output
    start_time = datetime.now()

    for path in paths:
        with Image.open(path) as f:

            sample_image = np.array(f)
            output = test_classifier(sample_image)
            print(output, "\n")

    # print(brothy_output, rice_output, noodle_output)
    print("end time: ", datetime.now() - start_time)

if __name__=="__main__":
    main()