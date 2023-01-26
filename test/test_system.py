import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.preprocess import preprocess
from src.loader import loader
from src.classifier import classifier
import random
import string


def test_text_classification_cpu():

    from transformers import AutoTokenizer

    pretrained_model_path = "./misc/model_repository/food_name_classification_pretrained_model"
    model_name = "ElectraForSequenceClassification"
    tokenizer = AutoTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    tokenizer_config = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 256,
        "padding": 'max_length', 
        "add_special_tokens": True,
    }

    test_loader = loader.HuggingfacePreTrainedModelLoader(
        model_name=model_name,
        model_path=pretrained_model_path
    )

    test_preprocess = preprocess.TextPreprocess(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config
    )

    test_classifier = classifier.TextClassifier(
        text_preprocess=test_preprocess,
        id2label=test_loader.get_id2label(),
        loader=test_loader,
        device="cpu"
    )

    inputs = "".join(random.choices(string.ascii_lowercase, k=100))
    test_classifier(inputs)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


def test_text_classification_gpu():
    from transformers import AutoTokenizer

    pretrained_model_path = "./misc/model_repository/food_name_classification_pretrained_model"
    model_name = "ElectraForSequenceClassification"
    tokenizer = AutoTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    tokenizer_config = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": 256,
        "padding": 'max_length', 
        "add_special_tokens": True,
    }

    test_loader = loader.HuggingfacePreTrainedModelLoader(
        model_name=model_name,
        model_path=pretrained_model_path
    )

    test_preprocess = preprocess.TextPreprocess(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config
    )

    test_classifier = classifier.TextClassifier(
        text_preprocess=test_preprocess,
        id2label=test_loader.get_id2label(),
        loader=test_loader,
    )

    inputs = "".join(random.choices(string.ascii_lowercase, k=100))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


def test_brothy_classification_cpu():
    model_path = "./misc/model_repository/brothy_model_state_dict.pt"

    test_preprocess = preprocess.ImagePreprocess()
    test_loader = loader.StateDictLoader(
        model_path=model_path,
        model_name="DenseNet",
        model_arc="densenet161",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(
        image_preprocess=test_preprocess,
        id2label={
            0:False,
            1:True,
        },
        loader=test_loader,
        device="cpu"
    )

    inputs = torch.rand(3, 256, 256)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


def test_brothy_classification_gpu():
    model_path = "./misc/model_repository/brothy_model_state_dict.pt"

    test_preprocess = preprocess.ImagePreprocess()
    test_loader = loader.StateDictLoader(
        model_path=model_path,
        model_name="DenseNet",
        model_arc="densenet161",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(
        image_preprocess=test_preprocess,
        id2label={
            0:False,
            1:True,
        },
        loader=test_loader,
    )

    inputs = torch.rand(3, 256, 256)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


def test_rice_classification_cpu():
    model_path = "./misc/model_repository/rice_model_state_dict.pt"

    test_preprocess = preprocess.ImagePreprocess()
    test_loader = loader.StateDictLoader(
        model_path=model_path,
        model_name="DenseNet",
        model_arc="densenet121",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(
        image_preprocess=test_preprocess,
        id2label={
            0:False,
            1:True,
        },
        loader=test_loader,
        device="cpu"
    )

    # inputs = [torch.randn(3, 256, 256) for _ in range(100)]
    inputs = torch.rand(3, 256, 256)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))


def test_rice_classification_gpu():
    model_path = "./misc/model_repository/rice_model_state_dict.pt"

    test_preprocess = preprocess.ImagePreprocess()
    test_loader = loader.StateDictLoader(
        model_path=model_path,
        model_name="DenseNet",
        model_arc="densenet121",
        num_classes=2,
    )
    test_classifier = classifier.ImageClassifier(
        image_preprocess=test_preprocess,
        id2label={
            0:False,
            1:True,
        },
        loader=test_loader,
    )

    # inputs = [torch.randn(3, 256, 256) for _ in range(100)]
    inputs = torch.rand(3, 256, 256)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            test_classifier(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

def test_noodle_classification():
    pass
