"""Scenario results are computed once per session and shared; each is a few seconds of simulation."""

import pytest

from kuka_table_sim import scenarios as sc


@pytest.fixture(scope="session")
def nominal():
    return sc.run_nominal(legacy_mixed=True)


@pytest.fixture(scope="session")
def free_air_static():
    return sc.run_free_air("static")


@pytest.fixture(scope="session")
def free_air_dynamic():
    return sc.run_free_air("dynamic")


@pytest.fixture(scope="session")
def fast_approach():
    return sc.run_free_air("fast")


@pytest.fixture(scope="session")
def loss_short():
    return sc.run_loss("short")


@pytest.fixture(scope="session")
def loss_reacquire():
    return sc.run_loss("reacquire")


@pytest.fixture(scope="session")
def loss_sustained():
    return sc.run_loss("sustained")


@pytest.fixture(scope="session")
def removal_stress():
    return sc.run_removal_stress()
