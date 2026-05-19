"""Feature-engineering math contracts."""

import numpy as np
import pytest

from wildfire.features import (
    kelvin_to_celsius,
    relative_humidity_pct,
    vpd_kpa,
    wind_speed_ms,
    wind_direction_deg,
)


def test_kelvin_conversion():
    assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert kelvin_to_celsius(300.0) == pytest.approx(26.85)


def test_relative_humidity_bounded():
    # RH must always be in [0, 100]
    temps = np.array([10.0, 20.0, 35.0, -5.0])
    dewps = np.array([-50.0, 19.0, 0.0, -10.0])
    rh = relative_humidity_pct(temps, dewps)
    assert (rh >= 0).all() and (rh <= 100).all()


def test_relative_humidity_saturation_when_dewpoint_equals_temp():
    # If dewpoint == temperature, air is fully saturated -> RH = 100%
    rh = relative_humidity_pct(np.array([20.0]), np.array([20.0]))
    assert rh[0] == pytest.approx(100.0, abs=0.01)


def test_vpd_is_nonneg_and_zero_at_saturation():
    # VPD = 0 when RH = 100
    assert vpd_kpa(20.0, 100.0) == pytest.approx(0.0)
    # VPD increases with temperature at fixed RH
    assert vpd_kpa(30.0, 50.0) > vpd_kpa(20.0, 50.0)


def test_wind_speed_from_components():
    # sqrt(3^2 + 4^2) = 5
    s = wind_speed_ms(np.array([3.0]), np.array([4.0]))
    assert s[0] == pytest.approx(5.0)


def test_wind_direction_meteorological_convention():
    # Wind blowing FROM the north (toward south) has u=0, v=-1 -> direction=0
    direction = wind_direction_deg(np.array([0.0]), np.array([-1.0]))
    assert direction[0] == pytest.approx(0.0, abs=0.01)
    # Wind blowing FROM the east -> direction=90
    direction = wind_direction_deg(np.array([-1.0]), np.array([0.0]))
    assert direction[0] == pytest.approx(90.0, abs=0.01)
