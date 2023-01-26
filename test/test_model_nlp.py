def test_preprocess():
    import torch
    from src.preprocess import preprocess
    from transformers import AutoTokenizer

    # 테스트 음식명
    # 토크나이저 로드, 주입
    # 토크나이저 파라미터 정의
    text = "부대찌개"
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
    test_preprocess = preprocess.TextPreprocess(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config
    )

    # 전처리 수행
    input_ids, attention_mask = test_preprocess(text)


    # 전처리 타입은 'preproces.Preprocess'
    # input_ids 타입 -> torch.Tensor
    # attention_mask 타입 -> torch.Tensor

    assert isinstance(test_preprocess, preprocess.Preprocess)
    assert type(input_ids) == torch.Tensor
    assert type(attention_mask) == torch.Tensor


def test_loader():
    """
        Huggingface pretrained model 로드 테스트
    """
    import torch.nn as nn
    from src.loader import loader

    pretrained_model_path = "./misc/model_repository/food_name_classification_pretrained_model"
    model_name = "ElectraForSequenceClassification"

    test_loader = loader.HuggingfacePreTrainedModelLoader(
        model_name=model_name,
        model_path=pretrained_model_path
    )

    test_model = test_loader.get_model()

    assert isinstance(test_loader, loader.ModelLoader), "test_loader의 타입은 loader.ModelLoader 타입이여야 한다."
    assert isinstance(test_model, nn.Module), "test_loader:get_model의 반환 타입은 pytorch의 nn.Module 타입이여야 한다."


def test_classifier():
    from src.loader import loader
    from src.preprocess import preprocess
    from src.classifier import classifier
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

    assert isinstance(test_classifier, classifier.Classifier)
    

def test_str_input_inference():
    from src.loader import loader
    from src.preprocess import preprocess
    from src.classifier import classifier
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
    topk = 3

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

    output = test_classifier("부대찌개")

    assert len(output) == topk
    assert isinstance(output, list)


def test_list_input_inference():
    from src.loader import loader
    from src.preprocess import preprocess
    from src.classifier import classifier
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
    topk = 3

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

    inputs = [
        "부대찌개",
        "냉면",
        "짜장면",
        "자장면",
        "불닭볶음면",
        "삼계탕",
        "삼치구이"
    ]

    outputs = test_classifier(inputs)

    assert len(outputs) == 7
    assert len(outputs[0]) == topk
    assert isinstance(outputs, list)