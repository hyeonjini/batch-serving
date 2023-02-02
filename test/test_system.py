import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.preprocess import preprocess
from src.loader import loader
from src.classifier import classifier
from src.utils import load_image, load_yaml
import random
import string

ROW_LIMIT = 7

def test_text_classification_cpu():

    from transformers import AutoTokenizer

    config = load_yaml("./config.yaml").text_classification

    test_loader = loader.HuggingfacePreTrainedModelLoader(
        model_name=config.model.name,
        model_path=config.model.path
    )

    test_preprocess = preprocess.TextPreprocess(
        tokenizer=AutoTokenizer.from_pretrained(
            config.tokenizer.path
        ),
        tokenizer_config=config.tokenizer.config
    )

    test_classifier = classifier.TextClassifier(
        preprocessor=test_preprocess,
        id2label=test_loader.get_id2label(),
        loader=test_loader,
        device="cpu"
    )

    inputs = "".join(random.choices(string.ascii_lowercase, k=100))
    test_classifier(inputs)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))


def test_text_classification_gpu():
    from transformers import AutoTokenizer

    config = load_yaml("./config.yaml").text_classification

    test_loader = loader.HuggingfacePreTrainedModelLoader(
        model_name=config.model.name,
        model_path=config.model.path
    )

    test_preprocess = preprocess.TextPreprocess(
        tokenizer=AutoTokenizer.from_pretrained(
            config.tokenizer.path
        ),
        tokenizer_config=config.tokenizer.config
    )

    test_classifier = classifier.TextClassifier(
        preprocessor=test_preprocess,
        id2label=test_loader.get_id2label(),
        loader=test_loader,
    )

    inputs = "".join(random.choices(string.ascii_lowercase, k=100))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=ROW_LIMIT))


def test_brothy_classification_cpu():

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
        },
        device="cpu"
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))


def test_brothy_classification_gpu():
    config = load_yaml("./config.yaml").image_classification.brothy

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.aug.resize,
        std=config.aug.std,
        mean=config.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.model.path,
        model_name=config.model.name,
        model_arc=config.model.arch,
        num_classes=config.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        }
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=ROW_LIMIT))


def test_rice_classification_cpu():

    config = load_yaml("./config.yaml").image_classification.rice

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.aug.resize,
        std=config.aug.std,
        mean=config.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.model.path,
        model_name=config.model.name,
        model_arc=config.model.arch,
        num_classes=config.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        },
        device="cpu"
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))


def test_rice_classification_gpu():
    config = load_yaml("./config.yaml").image_classification.rice

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.aug.resize,
        std=config.aug.std,
        mean=config.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.model.path,
        model_name=config.model.name,
        model_arc=config.model.arch,
        num_classes=config.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        }
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))



def test_noodle_classification_cpu():
    config = load_yaml("./config.yaml").image_classification.noodle

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.aug.resize,
        std=config.aug.std,
        mean=config.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.model.path,
        model_name=config.model.name,
        model_arc=config.model.arch,
        num_classes=config.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        },
        device="cpu"
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))


def test_noodle_classification_gpu():
    config = load_yaml("./config.yaml").image_classification.noodle

    test_preprocess = preprocess.ImagePreprocess(
        resize=config.aug.resize,
        std=config.aug.std,
        mean=config.aug.mean
    )

    # PyTorch State Dictionary Loader
    test_loader = loader.StateDictLoader(
        model_path=config.model.path,
        model_name=config.model.name,
        model_arc=config.model.arch,
        num_classes=config.model.num_classes
    )
    test_classifier = classifier.ImageClassifier(
        preprocessor=test_preprocess,
        loader=test_loader,
        id2label={
            0: False,
            1: True
        }
    )

    inputs = torch.rand(3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=ROW_LIMIT))