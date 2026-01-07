This repo contains code for my project on deep learning photometric redshifts with space-based images.

# Photometric Redshifts for High-Redshift Galaxies
Predicting redshifts of galaxies from their images (photo-z) is a crucial step in analyzing astronomical datasets. This is because obtaining accurate redshifts with spectroscopy (spec-z) is expensive, and as a result most imaged galaxies, especially in next-generation surveys (such as [Roman](https://roman.gsfc.nasa.gov])), will not have spec-z's. 

Empirical machine learning techniques offer a promising way to predict photo-z's, and state-of-the-art methods use convolutional neural networks on the images directly to leverage pixel-level information. However, these methods have been limited to low-redshift galaxies with ground-based imaging due to the lack of high-quality training sets for higher-redshift galaxies. 

In this work, we test the feasibility of these methods on higher-redshift galaxies with space-based imaging, using data from the Hubble Space Telescope [CANDELS survey](https://www.ipac.caltech.edu/project/candels). We find that a semi-supervised approach that leverages all available images (even ones which don't have redshift labels) performs best.

# Semi-Supervised Photo-z Algorithm
Our approach combines contrastive learning ([MoCo](https://github.com/facebookresearch/moco) implementation), color prediction, and redshift prediction, to learn a low-dimensional representation (latent space) of galaxy images that is ideal for redshift prediction across a wide range of redshifts (from 0 to ~3).

<img width="1578" height="869" alt="semi_supervised_architecture" src="https://github.com/user-attachments/assets/df847c21-aad5-4e2d-8e2d-6bd1bcd13a02" />
