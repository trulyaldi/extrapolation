CREATE INDEX IF NOT EXISTS idx_results_system_expectation
ON extrapolation_results(system, expectation_value);

CREATE INDEX IF NOT EXISTS idx_results_created_at
ON extrapolation_results(created_at);
