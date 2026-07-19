-- Older new_db databases may already contain the obsolete fit-result store.
-- Source data remain intact; only derived fitting output is removed.
DROP TABLE IF EXISTS extrapolation_results;
DROP SEQUENCE IF EXISTS extrapolation_result_id_seq;
