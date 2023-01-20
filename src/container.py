from dependency_injector import containers, providers

from service import service
from classifier import classifier
from preprocess import preprocess

class Container(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["../config.yml"])

    _preprocess = providers.Selector(
        config.preprocess.type,
        cv=None,
        nlp=None,
    )

    _classifier = providers.Selector(
        config.classifier.type,
        brothy=providers.Singleton(
            classifier.BrothyClassifier
        ),
        rice=providers.Singleton(
            classifier.RiceClassifier,
        ),
        noodle=providers.Singleton(
            classifier.NoodleClassifier,
        ),
        text=providers.Singleton(
            classifier.Textlassifier,
        ),
    )
    

    challenge_service = providers.Selector(
        config.service.type,
        cv=providers.Singleton(
            service.ComputerVisionChallengeService,
            classifier=_classifier,
        ),
        nlp=providers.Singleton(
            service.NLPChallengeService,
            classifier=_classifier,
        )
    )