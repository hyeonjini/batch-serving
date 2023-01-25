
def test_opencv_image_preprocess():
    import cv2
    from torch import Tensor
    from src.preprocess import preprocess
    
    brothy_image_path = "/opt/project/batch-serving/misc/samples/brothy.jpg"
    rice_image_path = "/opt/project/batch-serving/misc/samples/rice.jpg"
    noodle_image_path = "/opt/project/batch-serving/misc/samples/noodle.jpg"

    test_image1 = cv2.imread(brothy_image_path)
    test_image1 = cv2.cvtColor(test_image1, cv2.COLOR_BGR2RGB)

    test_image2 = cv2.imread(rice_image_path)
    test_image2 = cv2.cvtColor(test_image2, cv2.COLOR_BGR2RGB)

    test_image3 = cv2.imread(noodle_image_path)
    test_image3 = cv2.cvtColor(test_image3, cv2.COLOR_BGR2RGB)

    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)

    test_image1 = test_preprocess(test_image1)
    test_image2 = test_preprocess(test_image2)
    test_image3 = test_preprocess(test_image3)

    assert type(test_image1) == Tensor
    assert type(test_image2) == Tensor
    assert type(test_image3) == Tensor

def test_pillow_image_preprocess():
    import numpy as np
    from PIL import Image
    from torch import Tensor
    from src.preprocess import preprocess

    brothy_image_path = "/opt/project/batch-serving/misc/samples/brothy.jpg"
    rice_image_path = "/opt/project/batch-serving/misc/samples/rice.jpg"
    noodle_image_path = "/opt/project/batch-serving/misc/samples/noodle.jpg"

    brothy_image = np.array(Image.open(brothy_image_path))
    rice_image = np.array(Image.open(rice_image_path))
    noodle_image = np.array(Image.open(noodle_image_path))

    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)

    brothy_image = test_preprocess(brothy_image)
    rice_image = test_preprocess(rice_image)
    noodle_image = test_preprocess(noodle_image)

    assert type(brothy_image) == Tensor
    assert type(rice_image) == Tensor
    assert type(noodle_image) == Tensor

def test_text_preprocess():
    pass

def test_jit_script_model_load():

    from src.loader import loader

    test_model_path = "/opt/project/batch-serving/misc/model_repository/brothy_model.pt"
    test_model = loader.JitScriptLoader(test_model_path)
    test_model = test_model.get_model()

    assert test_model != None

def test_state_dict_densenet_model_load():
    from src.loader import loader
    from src.model import ResNet, DenseNet

    test_model_path = "/opt/project/batch-serving/misc/model_repository/brothy_model_state_dict.pt"
    
    test_loader = loader.StateDictLoader(test_model_path, "DenseNet", "densenet161", 2)
    test_model = test_loader.get_model()
    assert type(test_model) == DenseNet

def test_image_classification_torch_model_inference():

    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from src.utils import load_image
    from datetime import datetime
    import numpy as np


    paths = [
        "/opt/project/batch-serving/misc/samples/brothy.jpg",
        "/opt/project/batch-serving/misc/samples/rice.jpg",
        "/opt/project/batch-serving/misc/samples/noodle.jpg",
        "/opt/project/batch-serving/misc/samples/brothy2.jpg",
        "/opt/project/batch-serving/misc/samples/rice2.jpg",
        "/opt/project/batch-serving/misc/samples/noodle2.jpg"
    ]
    test_model_path = "/opt/project/batch-serving/misc/model_repository/brothy_model_state_dict.pt"
    
    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    start_time = datetime.now()

    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=test_model_path,
        model_name="DenseNet",
        model_arc="densenet161",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(image_preprocess=test_preprocess, loader=test_loader)

    # output
    for path in paths:
        image = load_image(path)
        sample_image = np.array(image)
        output = test_classifier(sample_image)
        
    print("end time: ", datetime.now() - start_time)

def test_image_classification_inference():

    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from src.utils import load_image
    from datetime import datetime
    from PIL import Image
    import numpy as np


    paths = [
        "/opt/project/batch-serving/misc/samples/brothy.jpg",
        "/opt/project/batch-serving/misc/samples/rice.jpg",
        "/opt/project/batch-serving/misc/samples/noodle.jpg",
        "/opt/project/batch-serving/misc/samples/brothy2.jpg",
        "/opt/project/batch-serving/misc/samples/rice2.jpg",
        "/opt/project/batch-serving/misc/samples/noodle2.jpg"
    ]
    test_model_path = "/opt/project/batch-serving/misc/model_repository/brothy_model_trace.pt"
    
    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)
    # PyTorch Jit Script Loader
    test_loader = loader.JitScriptLoader(model_path=test_model_path)
    test_classifier = classifier.ImageClassifier(image_preprocess=test_preprocess, loader=test_loader)

    # output
    start_time = datetime.now()

    for path in paths:
        image = load_image(path)
        sample_image = np.array(image)
        output = test_classifier(sample_image)

    # print(brothy_output, rice_output, noodle_output)
    print("end time: ", datetime.now() - start_time)

def test_image_classification_batch_inference():

    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from src.utils import load_image
    from datetime import datetime
    import numpy as np

    image_paths = [
        "/opt/project/batch-serving/misc/samples/brothy.jpg",
        "/opt/project/batch-serving/misc/samples/rice.jpg",
        "/opt/project/batch-serving/misc/samples/noodle.jpg",
        "/opt/project/batch-serving/misc/samples/brothy2.jpg",
        "/opt/project/batch-serving/misc/samples/rice2.jpg",
        "/opt/project/batch-serving/misc/samples/noodle2.jpg"
    ]
    test_model_path = "/opt/project/batch-serving/misc/model_repository/brothy_model_state_dict.pt"
    
    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    start_time = datetime.now()

    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)
    test_loader = loader.StateDictLoader(
        model_path=test_model_path,
        model_name="DenseNet",
        model_arc="densenet161",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(image_preprocess=test_preprocess, loader=test_loader)

    images = [load_image(image_path) for image_path in image_paths]


    outputs = test_classifier(images)
    print(outputs)

    print("end time: ", datetime.now() - start_time)
    
