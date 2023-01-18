from dependency_injector.wiring import Provide, inject

from container import Container

@inject
def main() -> None:
    ...

if __name__ == "__main__":
    container = Container()
    print(container.config.__str__())
    main()