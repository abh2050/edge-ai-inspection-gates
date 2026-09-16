# ADR 0010 publishes the evidence dashboard as a static page.

Status: accepted by the project owner on 2026-09-16.

ADR 0008 bounded the dashboard to the local host, because a served review surface invites operator input and outward claims.
A reader of the repository could therefore not see the measured evidence without cloning it and running a local server.

The repository now publishes a copy of the dashboard through GitHub Pages from the docs directory.
The build step writes docs/index.html and docs/data.json alongside the local copy, so the published page and the local page render the same recorded values.
The build step also writes docs/.nojekyll, because the static site generator would otherwise process the page.

The published page carries no credential, no customer data, and no dataset archive.
Its data file contains recorded measurements, configured assumptions, the host model and software versions, and one attributed figure that ADR 0009 permits.
The page accepts no input, writes nothing, triggers no gate, and calls no model.
Its controls select among recorded values, so a reader cannot cause a computation by using it.

Rule 16 previously prohibited serving beyond the local host.
Rule 16 now permits a published copy that carries no credential and no customer data, and requires the published data file to contain only recorded evidence.

The published page is a reading surface and never acceptance evidence.
The gate records in the repository remain the only evidence, and every figure on the page names the gate that produced it.
A future deployment that carries customer measurements must not publish this page.
