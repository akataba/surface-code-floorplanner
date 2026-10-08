# Building and publishing the documentation

## Local build

From the project root, use Python 3.14 and an isolated environment:

```sh
python3.14 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.lock
python -m pip install --no-deps -e .
python scripts/check_docstrings.py
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b doctest docs docs/_build/doctest
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

Open `docs/_build/html/index.html`. `-W` turns warnings into build failures;
`-n` checks missing cross-references. External link checking needs network access.
The HTML build and executable examples do not run optimizations or require the
experiment archive. Optional Torch imports are mocked solely to render research
API docs; rendering is not a runtime test of those policies.

`docs/requirements.lock` records the resolved documentation environment. Update
its pins deliberately, then rebuild and review the generated API pages. The
package's Google-style docstrings are rendered using Napoleon and autodoc;
Markdown pages use MyST. Do not hand-edit generated HTML.

## Read the Docs deployment

The root `.readthedocs.yaml` configures Python, installation and warning failures.
The GitHub workflow builds docs and publishes an HTML build artifact for review.
Neither file creates a hosted project by itself.

1. Put the project source and documentation in the intended hosted Git repository.
2. In Read the Docs, add that repository as a project and choose the default branch.
3. Use `.readthedocs.yaml` and confirm the first remote build succeeds.
4. Configure the repository integration/webhook for subsequent builds.
5. Enable release-tag versions after releases exist; use `latest` for the development branch.

See the official [project setup guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html)
and [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).
A hosted URL is only established after the project is created and its build succeeds.

## Publication contents

Publish source, guides, API pages and the small curated files in `docs/_static`.
Keep environments, native build trees, `validation/`, `mlruns/`, run outputs,
local configuration and generated HTML out of Git. The site's local results use a
small checked-in summary; they do not require uploading the multi-gigabyte archive.

Historical internal notes and Gemini review screenshots are excluded from the
site. Before making the source repository public, review those files separately;
excluding a page from Sphinx does not hide it from a public Git repository.
Project ownership, repository visibility and software licensing must be chosen
by the maintainer rather than inferred from a working-folder name.
