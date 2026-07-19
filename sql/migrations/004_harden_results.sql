ALTER TABLE extrapolation_results
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

CREATE UNIQUE INDEX IF NOT EXISTS uq_extrapolation_result_identity
ON extrapolation_results (
    system,
    expectation_value,
    model,
    basis_family
);
