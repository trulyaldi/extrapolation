CREATE TABLE IF NOT EXISTS extrapolation_results (
    id BIGINT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    system VARCHAR NOT NULL,
    expectation_value VARCHAR NOT NULL,

    model VARCHAR NOT NULL,
    basis_family VARCHAR,

    extrapolated_value DOUBLE NOT NULL,
    uncertainty DOUBLE,

    metadata JSON
);
