import json

from database.connection import get_connection


def save_result(
    system,
    expectation_value,
    model,
    basis_family,
    extrapolated_value,
    uncertainty=None,
    metadata=None,
):
    required = {
        "system": system,
        "expectation_value": expectation_value,
        "model": model,
        "basis_family": basis_family,
    }

    for name, value in required.items():
        if value is None or not str(value).strip():
            raise ValueError(f"{name} must be non-empty")

    metadata_json = json.dumps(metadata) if metadata is not None else None

    with get_connection() as con:
        con.begin()

        existing = con.execute(
            """
            SELECT id
            FROM extrapolation_results
            WHERE system = ?
              AND expectation_value = ?
              AND model = ?
              AND basis_family = ?
            """,
            (
                system,
                expectation_value,
                model,
                basis_family,
            ),
        ).fetchone()

        if existing is not None:
            result_id = existing[0]

            con.execute(
                """
                UPDATE extrapolation_results
                SET
                    updated_at = CURRENT_TIMESTAMP,
                    extrapolated_value = ?,
                    uncertainty = ?,
                    metadata = ?
                WHERE id = ?
                """,
                (
                    extrapolated_value,
                    uncertainty,
                    metadata_json,
                    result_id,
                ),
            )
        else:
            result_id = con.execute(
                "SELECT nextval('extrapolation_result_id_seq')"
            ).fetchone()[0]

            con.execute(
                """
                INSERT INTO extrapolation_results (
                    id,
                    system,
                    expectation_value,
                    model,
                    basis_family,
                    extrapolated_value,
                    uncertainty,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    system,
                    expectation_value,
                    model,
                    basis_family,
                    extrapolated_value,
                    uncertainty,
                    metadata_json,
                ),
            )

        con.commit()

    return result_id


def get_result(result_id):
    with get_connection(read_only=True) as con:
        row = con.execute(
            """
            SELECT
                id,
                created_at,
                updated_at,
                system,
                expectation_value,
                model,
                basis_family,
                extrapolated_value,
                uncertainty,
                metadata
            FROM extrapolation_results
            WHERE id = ?
            """,
            [result_id],
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "created_at": row[1],
        "updated_at": row[2],
        "system": row[3],
        "expectation_value": row[4],
        "model": row[5],
        "basis_family": row[6],
        "extrapolated_value": row[7],
        "uncertainty": row[8],
        "metadata": json.loads(row[9]) if row[9] else None,
    }

def list_results(limit=20):
    """
    Return the most recent extrapolation results.
    """
    with get_connection(read_only=True) as con:
        rows = con.execute(
            """
            SELECT
                id,
                created_at,
                system,
                expectation_value,
                model,
                basis_family,
                extrapolated_value,
                uncertainty
            FROM extrapolation_results
            ORDER BY id DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    return rows

def find_results(system, expectation_value):
    """
    Return all extrapolation results for a given system and expectation value.
    """
    with get_connection(read_only=True) as con:
        rows = con.execute(
            """
            SELECT
                id,
                created_at,
                model,
                basis_family,
                extrapolated_value,
                uncertainty,
                metadata
            FROM extrapolation_results
            WHERE system = ?
              AND expectation_value = ?
            ORDER BY created_at DESC, id DESC
            """,
            [system, expectation_value],
        ).fetchall()

    return rows
