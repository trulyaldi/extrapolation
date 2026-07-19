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
