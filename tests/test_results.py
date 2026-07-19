from database.results import save_result, get_result


def test_save_result_updates_existing():
    result_id = save_result(
        "PYTEST_SYSTEM",
        "energy",
        "exp",
        "demo",
        -1.0,
        uncertainty=0.1,
    )

    updated_id = save_result(
        "PYTEST_SYSTEM",
        "energy",
        "exp",
        "demo",
        -2.0,
        uncertainty=0.2,
    )

    result = get_result(updated_id)

    assert updated_id == result_id
    assert result["extrapolated_value"] == -2.0
    assert result["uncertainty"] == 0.2


def test_different_models_create_different_results():
    exp_id = save_result(
        "PYTEST_MODEL_SYSTEM",
        "energy",
        "exp",
        "demo",
        -1.0,
    )

    power_id = save_result(
        "PYTEST_MODEL_SYSTEM",
        "energy",
        "power",
        "demo",
        -2.0,
    )

    assert exp_id != power_id


def test_metadata_round_trip():
    result_id = save_result(
        "PYTEST_METADATA",
        "energy",
        "exp",
        "demo",
        -1.5,
        metadata={"r2": 0.999, "n_fit": 4},
    )

    result = get_result(result_id)

    assert result["metadata"] == {"r2": 0.999, "n_fit": 4}


def test_get_missing_result_returns_none():
    assert get_result(999999999) is None


import pytest


@pytest.mark.parametrize(
    "field,value",
    [
        ("system", ""),
        ("expectation_value", None),
        ("model", "   "),
        ("basis_family", ""),
    ],
)
def test_save_result_rejects_empty_identity_fields(field, value):
    kwargs = {
        "system": "SYS",
        "expectation_value": "energy",
        "model": "exp",
        "basis_family": "demo",
        "extrapolated_value": -1.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        save_result(**kwargs)
