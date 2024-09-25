-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.7.3'" to load this file. \quit

CREATE FUNCTION load_centers(VarChar)
  RETURNS int
AS 'MODULE_PATHNAME', 'load_centers_from_csv'
LANGUAGE C STRICT;

