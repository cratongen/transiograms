import sys

import numpy as np
import pytest

TEST_DATA_LIMIT = 4 * 1024**3


def pytest_configure(config):
    if sys.platform != "linux":
        return
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if soft == resource.RLIM_INFINITY or soft > TEST_DATA_LIMIT:
        new = TEST_DATA_LIMIT if hard == resource.RLIM_INFINITY else min(TEST_DATA_LIMIT, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (new, hard))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
