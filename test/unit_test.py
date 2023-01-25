import cProfile

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

def test_state_dict_model_load():
    pass

def test_image_classification_inference():

    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from datetime import datetime
    from PIL import Image
    import numpy as np
    from cProfile import Profile
    from pstats import Stats

    profiler = Profile()


    test_model_path = "/opt/project/batch-serving/misc/model_repository/noodle_model.pt"
    test_brothy_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/brothy.jpg"))
    test_rice_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/rice.jpg"))
    test_noodle_image = np.array(Image.open("/opt/project/batch-serving/misc/samples/noodle.jpg"))
    
    resize = (256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # data preprocess, model loader, classifier


    test_preprocess = preprocess.ImagePreprocess(resize=resize, std=std, mean=mean)
    test_loader = loader.JitScriptLoader(model_path=test_model_path)
    test_classifier = classifier.ImageClassifier(image_preprocess=test_preprocess, loader=test_loader)

    # output
    start_time = datetime.now()
    
    brothy_output = test_classifier(test_brothy_image)
    rice_output = test_classifier(test_rice_image)
    noodle_output = test_classifier(test_noodle_image)

    # profiler.runcall(test_classifier, test_brothy_image)

    # stats = Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('cumulative')
    # stats.print_stats()

    print(brothy_output, rice_output, noodle_output)
    print("end time: ", datetime.now() - start_time)

    assert brothy_output != None

def test_image_classification_batch_inference():
    ...

    
