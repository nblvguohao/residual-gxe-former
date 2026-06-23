from __future__ import annotations

import pandas as pd

from residual_gxe.data.weather import prepare_weather_daily


def test_prepare_weather_daily_normalizes_and_derives_fields():
    raw = pd.DataFrame(
        {
            "yearsite_uid": ["E1", "E1", "E1"],
            "Date": ["2020-04-01", "2020-04-01", "2020-04-02"],
            "TMAX": [20.0, 22.0, 25.0],
            "TMIN": [10.0, 12.0, 15.0],
            "rainfall": [1.0, 2.0, 0.0],
            "Relative Humidity [%]": [50.0, 60.0, 55.0],
        }
    )
    env = pd.DataFrame({"environment_id": ["E1"], "planting_date": ["2020-04-01"]})

    result = prepare_weather_daily(raw, env, source_dataset="fip1")
    weather = result.weather

    assert len(weather) == 2
    first = weather.iloc[0]
    assert first["environment_id"] == "E1"
    assert first["day_after_planting"] == 0
    assert first["tmax"] == 22.0
    assert first["tmin"] == 10.0
    assert first["precipitation"] == 3.0
    assert first["tmean"] == 16.0
    assert first["gdd"] == 6.0
    assert first["vpd"] == first["vpd"]
    assert result.manifest["source_dataset"] == "fip1"

