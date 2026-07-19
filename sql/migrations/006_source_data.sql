-- The source-data catalog deliberately stores rows in a normalized layout.
-- It supports the project's differently shaped CSV files without generating a
-- SQL table for every source file, while reconstructing wide DataFrames in
-- their original column order.
CREATE TABLE IF NOT EXISTS datasets (
    dataset_name VARCHAR PRIMARY KEY,
    data_role VARCHAR NOT NULL,
    source_path VARCHAR UNIQUE,
    source_hash VARCHAR,
    independent_column VARCHAR NOT NULL,
    schema_signature VARCHAR NOT NULL,
    row_count BIGINT NOT NULL,
    manual_modified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    imported_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dataset_columns (
    dataset_name VARCHAR NOT NULL,
    column_index BIGINT NOT NULL,
    column_name VARCHAR NOT NULL,
    PRIMARY KEY (dataset_name, column_index),
    UNIQUE (dataset_name, column_name)
);

CREATE TABLE IF NOT EXISTS dataset_rows (
    dataset_name VARCHAR NOT NULL,
    row_index BIGINT NOT NULL,
    PRIMARY KEY (dataset_name, row_index)
);

CREATE TABLE IF NOT EXISTS dataset_cells (
    dataset_name VARCHAR NOT NULL,
    row_index BIGINT NOT NULL,
    column_name VARCHAR NOT NULL,
    numeric_value DOUBLE,
    PRIMARY KEY (dataset_name, row_index, column_name)
);

CREATE INDEX IF NOT EXISTS idx_dataset_cells_load
ON dataset_cells (dataset_name, row_index);
