import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits the live Hugging Face Hub")


def pytest_collection_modifyitems(config, items):
    if config.getoption("markexpr"):
        return
    skip = pytest.mark.skip(reason="needs -m network to run live Hub tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
