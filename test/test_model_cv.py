paths = [
    "/opt/project/batch-serving/misc/samples/brothy.jpg",
    "/opt/project/batch-serving/misc/samples/rice.jpg",
    "/opt/project/batch-serving/misc/samples/noodle.jpg",
    "/opt/project/batch-serving/misc/samples/brothy2.jpg",
    "/opt/project/batch-serving/misc/samples/rice2.jpg",
    "/opt/project/batch-serving/misc/samples/noodle2.jpg"
]

def test_brothy_inference():

    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from src.utils import load_image, load_yaml
    import numpy as np

    config = load_yaml("./config.yaml").image_classification

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.brothy.aug.resize,
        std=config.brothy.aug.std,
        mean=config.brothy.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.brothy.model.path,
        model_name=config.brothy.model.name,
        model_arc=config.brothy.model.arch,
        num_classes=config.brothy.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        }
    )

    # output
    for path in paths:
        image = load_image(path)
        sample_image = np.array(image)
        output = test_classifier(sample_image)

        assert isinstance(output, list)
        assert isinstance(output[0], dict)


def test_rice_inference():
    from src.classifier import classifier
    from src.loader import loader
    from src.preprocess import preprocess
    from src.utils import load_image, load_yaml
    import numpy as np

    config = load_yaml("./config.yaml").image_classification

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.rice.aug.resize,
        std=config.rice.aug.std,
        mean=config.rice.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.rice.model.path,
        model_name=config.rice.model.name,
        model_arc=config.rice.model.arch,
        num_classes=config.rice.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        }
    )

    # output
    for path in paths:
        image = load_image(path)
        sample_image = np.array(image)
        output = test_classifier(sample_image)

        assert isinstance(output, list)
        assert isinstance(output[0], dict)