1. Work one gate at a time. Never start the next gate without an explicit "go".
2. Before writing any new module, explain in plain language what it will do,
   what the inputs and outputs are, and why this design. Wait for approval.
3. Small diffs. One function or one concept per step. After each file, offer a
   line-by-line walkthrough.
4. Introduce each new library concept (xarray indexing, GRIB structure,
   LightGBM params) the first time it appears, with a 3-5 sentence explanation.
5. No abstractions Logan didn't ask for. No plugin systems, no config
   frameworks, no premature generality beyond the RegionConfig seam.
6. Dependencies are locked to the list in Gate 0. Ask before adding anything.
7. Every gate ends with: run the tests, summarize what was built and learned,
   list open questions, STOP for review.
8. All timestamps stored in UTC. Region timezone used only for feature
   engineering and display.
9. Never mix rows across the train/test time boundary. If a shortcut would
   leak future information, flag it instead of taking it.