# Hierarchical Job Ads Classifier

This repository now keeps two documented release tracks side by side so the historical workflow and the modernized workflow stay together in one place.

Available READMEs:

- [README-0.1.0.md](README-0.1.0.md): historical usage and dependency notes for the original release line
- [README-0.2.0.md](README-0.2.0.md): modernized release with current libraries, CPU/GPU handling, and tests

Versioned dependency files:

- `requirements-0.1.0.txt`
- `requirements-colab-0.1.0.txt`
- `requirements-0.2.0.txt`
- `requirements-colab-0.2.0.txt`

Default entry files point to the maintained line:

- `requirements.txt` -> `requirements-0.2.0.txt`
- `requirements-colab.txt` -> `requirements-colab-0.2.0.txt`

Citation:

- [CITATION.cff](CITATION.cff) contains the repository citation metadata used by GitHub
- the preferred paper citation is in both versioned READMEs

Suggested usage:

- use `0.1.0` when you want to compare with the original project layout and installation style
- use `0.2.0` when you want the refreshed install path, tests, and current dependency stack
