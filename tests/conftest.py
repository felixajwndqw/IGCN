import pytest


@pytest.fixture(scope='session')
def setup_func(request):
    example = ""

    return locals()
