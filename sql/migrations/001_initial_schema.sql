CREATE TABLE IF NOT EXISTS reference_values (
    system VARCHAR NOT NULL,
    expectation_value VARCHAR NOT NULL,
    ref_value DECIMAL(38, 18) NOT NULL,
    uncertainty DECIMAL(38, 18),
    source VARCHAR,
    PRIMARY KEY (system, expectation_value)
);
