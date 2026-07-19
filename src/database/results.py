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
                    created_at = CURRENT_TIMESTAMP,
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
                "SELECT COALESCE(MAX(id), 0) + 1 FROM extrapolation_results"
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
        "system": row[2],
        "expectation_value": row[3],
        "model": row[4],
        "basis_family": row[5],
        "extrapolated_value": row[6],
        "uncertainty": row[7],
        "metadata": json.loads(row[8]) if row[8] else None,
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
