## Pytest failures that are external or complex

- **tests/test_literature_live.py::test_literature_live_uses_openalex**  
  - **Summary**: Live integration test expects a non-empty literature inventory from the OpenAlex API.  
  - **Observed behavior**: The `LiteratureHarvester.search_openalex` call completed but returned an empty list, leaving `data/literature/index.json` as an empty list and causing the assertion `inventory and isinstance(inventory, list)` to fail.  
  - **Why not fully fixed**: This test depends on the external OpenAlex service (network availability, API behavior, dataset contents). Making the code “force” at least one result would hide real upstream issues and overfit to the current response; a robust solution would require investigating API status/quotas and potentially adding retry/backoff or alternate queries, which goes beyond unit-level changes. For now, treat this as an environment-dependent integration failure rather than a code bug.

