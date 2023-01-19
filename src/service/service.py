from collections import abc
from classifier.classifier import Classifier

class ChallengeService:
    
    def __init__(self, classifier: Classifier) -> None:
        self.classifier = classifier

    def service(self):
        return NotImplementedError()

class ComputerVisionChallengeService(ChallengeService):

    def __init__(self, classifier: Classifier) -> None:
        super().__init__(classifier)
    
    def service(self):
        return super().service()

class NLPChallengeService(ChallengeService):

    def __init__(self, classifier: Classifier) -> None:
        super().__init__(classifier)


    def service(self):
        return super().service()



