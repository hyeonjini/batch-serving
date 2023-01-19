from dependency_injector import containers, providers

from service import service
from classifier import classifier

class Container(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["../config.yml"])

    cv_classifier = providers.Singleton(
        classifier.BrothyClassifier,
    )

    
    nlp_classifier = providers.Singleton(
        classifier.TextClassifier,
    )
    

    challenge_service = providers.Selector(
        config.service.type,
        cv=providers.Singleton(
            service.ComputerVisionChallengeService,
            classifier=cv_classifier,
        ),
        nlp=providers.Singleton(
            service.NLPChallengeService,
            classifier=nlp_classifier,
        )
    )