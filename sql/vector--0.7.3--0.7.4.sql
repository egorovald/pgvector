-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "ALTER EXTENSION vector UPDATE TO '0.7.4'" to load this file. \quit

DROP FUNCTION load_centers(int, int);

