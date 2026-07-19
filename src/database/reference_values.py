from dataclasses import dataclass

from database.connection import get_connection


@dataclass(frozen=True)
class ReferenceValue:
    system: str
    expectation_value: str
    ref_value: float
    uncertainty: float | None
    source: str | None



def load_reference_values():
    """
    Return all reference values as a pandas DataFrame.
    """
    with get_connection(read_only=True) as con:
        return con.execute("""
            SELECT
                system,
                expectation_value,
                CAST(ref_value AS DOUBLE) AS ref_value,
                CAST(uncertainty AS DOUBLE) AS uncertainty,
                source
            FROM reference_values
            ORDER BY system, expectation_value
        """).fetchdf()

def get_reference_value(system: str, expectation_value: str):
    """
    Return one reference value as a dictionary, or None if it doesn't exist.
    """
    with get_connection(read_only=True) as con:
        row = con.execute(
            """
            SELECT
                system,
                expectation_value,
                CAST(ref_value AS DOUBLE) AS ref_value,
                CAST(uncertainty AS DOUBLE) AS uncertainty,
                source
            FROM reference_values
            WHERE system = ? AND expectation_value = ?
            """,
            [system, expectation_value],
        ).fetchone()

    if row is None:
        return None

    return ReferenceValue(
        system=row[0],
        expectation_value=row[1],
        ref_value=row[2],
        uncertainty=row[3],
        source=row[4],
    )
