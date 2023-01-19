from dependency_injector.wiring import Provide, inject

from container import Container

@inject
def main() -> None:
    ...

if __name__ == "__main__":
    container = Container()
    container.wire(modules=[__name__])

    service = container.challenge_service()
    classifier = service.classifier

    print(service, classifier)
    main()