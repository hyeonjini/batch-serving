from dependency_injector import containers, providers

from service import service

class Container(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yml"])

    service = providers.Selector(

    )