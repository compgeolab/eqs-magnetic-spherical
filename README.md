# Magnetic dual-layer equivalent sources on the sphere

by
[Arthur Siqueira-Macedo](https://orcid.org/0009-0006-5106-4356),
[Leonardo Uieda](https://orcid.org/0000-0001-6123-9515),
[India Uppal](https://orcid.org/0000-0003-3531-2656)

This repository contains the data and source code used to produce the results
presented in:

> Siqueira-Macedo, A., Uieda, L., Uppal, I. (2026). Magnetic dual-layer
> equivalent sources on the sphere. EarthArXiv.

|  | Info |
|-:|:-----|
| Version of record | https://doi.org/JOURNAL_DOI |
| Open-access version on EarthArXiv | https://doi.org/EARTHARXIV_DOI |
| Archive of this repository | https://doi.org/10.5281/zenodo.18509844 |
| Reproducing our results | [`REPRODUCING.md`](REPRODUCING.md) |

## About

The initial idea for this project emerged during a meeting between Arthur
and Prof. Leo on January 22, 2024. In this meeting, Leo presented his
preliminary thoughts on the topic, and Arthur immediately embraced the
proposal. Shortly afterward, Arthur moved from Formosa, Goiás, to São Paulo
to begin his master’s studies.

Since then, the project has represented a motivating challenge, as it
marked Arthur’s first experience working in a research area that had not
been part of his undergraduate research. Despite the initial
difficulties, the transition proved to be highly enriching. Working
closely with Leo and India throughout this project has been a valuable
and rewarding experience, contributing significantly to Arthur’s
academic and professional development.

## Abstract

The equivalent source method is widely used for processing and
interpolating magnetic data, particularly in airborne surveys. However,
implementations based on Cartesian coordinates present limitations at
regional and global scales, where Earth curvature introduces geometric
inconsistencies that affect data integration and modeling accuracy. To
address this problem, this study proposes an adaptation of the magnetic
equivalent source method to spherical coordinates, including revisions
to its mathematical formulation to account for spherical geometry. The
proposed framework enables consistent magnetic field modeling over large
geographic areas.
To improve the representation of magnetic sources, a dual-layer
configuration is adopted to separate long- and short-wavelength
components. Cross-validation is employed to determine optimal
hyperparameters for each layer, ensuring stable and balanced inversions.
To guarantee computational feasibility for large and high-resolution
datasets, a gradient-boosting strategy is incorporated into the inversion
process, significantly improving computational performance.
Synthetic experiments demonstrate that the method remains stable and accurate for large-scale datasets,
with tests conducted on synthetic data containing up to 500,000 observations and enables
the reliable recovery of magnetic field components from total-field
anomaly data. The approach was further applied to more than 1.5 million
real observations, confirming its scalability and robustness. The
recovered field amplitude provides additional constraints for data
interpretation and enhances the geological analysis.
The final implementation is released as open-source software to support
reproducibility and broader adoption.

## License

All Python source code (including `.py` and `.ipynb` files) is made available
under the MIT license. You can freely use and modify the code, without
warranty, so long as you provide attribution to the authors. See
`LICENSE-MIT.txt` for the full license text.

The manuscript text (including all LaTeX files), figures, and data/models
produced as part of this research are available under the [Creative Commons
Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full
license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/
