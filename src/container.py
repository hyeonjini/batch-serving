from dependency_injector import containers, providers

from classifier import classifier
from preprocess import preprocess

class Container(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["../config.yml"])

    preprocess = providers.Selector(
        config.preprocess.type,
        cv=None,
        nlp=None,
    )

    classifier = providers.Selector(
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