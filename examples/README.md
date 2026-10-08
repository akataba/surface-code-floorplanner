# Publication inputs

`corrected-v3` contains all 20 benchmark circuits across eight families, with
synthesis metadata and input hashes. Its manifest's `path` entries resolve in this
checkout. `source_path` entries identify earlier provenance locations, which are
not distributed here and are not required by the benchmark runner.

`benchmark-starts` contains the 140 layout/identity pairs used to initialize the
local-compilation benchmark. Its manifest preserves original content hashes.
The cost of the earlier optimization remains inherited, as explained in the docs.
