<div align="center">
<img src="assets/93cc350bee4c4e43f38c9a078acdde29.png" width="680" alt="Teaser GENA3D" />
<h2>GENA3D: Generative Amodal 3D Modeling by Bridging 2D Priors and 3D Coherence</h2>
<p><b>ECCV 2026</b></p>
<p>
<a href="https://colezwhy.github.io/" target="_blank">Junwei Zhou</a>,
<a href="https://yuwingtai.github.io/" target="_blank">Yu-Wing Tai</a>
</p>
<p>Dartmouth College</p>
</div>


>**TL;DR**: <em>GENA3D bridges 2D amodal completion and 3D generative modeling to achieve amodal 3D objects generation from sparse and paritial-occluded observations.</em>

<p align="center">
  <a href="https://colezwhy.github.io/gena3d/">
    <img src="https://img.shields.io/badge/Project-Website-green">
  </a>
  <a href="https://arxiv.org/abs/2511.21945">
    <img src="https://img.shields.io/badge/ECCV'26-paper-orange">
  </a>
    <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Colezwhy.GENA3D" alt="Visitors">
  </a>
</p>


Official implementation for paper 'GENA3D: Generative Amodal 3D Modeling by Bridging 2D Priors and 3D Coherence'.

Intergrating 2D amodal completion prior and 3D generative modeling ability in the latent 3D space to achieve amodal 3D objects generation from sparse and partial-occluded views, under various scenarios, including single object-level, in-the-wild and in-the-scene.

https://github.com/user-attachments/assets/6f4b36e1-c50d-436a-9241-bd0e700c809e


## Updates and TODOs
- ✔️ 12/04/2025: Initialize the project page.
- 🎊 06/18/2026: GENA3D is accepted to ECCV 2026.
- 🔲 TODO: The code will soon be released. Please stay tuned!


## Method 
<p align="center">
  <img src="assets/11fee07ae661ec3e0f6e4941ea41e31e.png" width="800" alt="Overall pipeline" />
</p>

<p align="center">
  GENA3D bridges the 2D amodal completion with 3D generation using deliberaely designed View-Wise Cross Attention and Stereo-Conditioned Cross Attention in the Sparse Structure Generation Stage, with synthesized sparse-view 3D consistent occlusions as training data.
</p>

<p align="center">
  <img src="assets/308b495e4ed18e608a635ee2861486bb.png" width="540" alt="Module design" />
</p>

<p align="center">
  A detailed illustration of our proposed ViewWise Cross Attention and Stereo-Conditioned Cross Attention modules.
</p>

## Results
<p align="center">
  <img src="assets/4851b41a68ad377dfcef3a6ac9ad03a1.png" width="720" alt="Object-level" />
</p>

<p align="center">
  Results on GSO object-level synthetic dataset.
</p>

<p align="center">
  <img src="assets/d8797c493af58b11cb24b4fe7fa56e7e.png" width="580" alt="In-the-wild" />
</p>

<p align="center">
  Results on in-the-wild real-world captures.
</p>

## Citation
Here is the bibtex reference. If you find our work interesting or useful, please give us a :star: or cite our paper!
```
@misc{zhou2026gena3dgenerativeamodal3d,
      title={GENA3D: Generative Amodal 3D Modeling by Bridging 2D Priors and 3D Coherence}, 
      author={Junwei Zhou and Yu-Wing Tai},
      year={2026},
      eprint={2511.21945},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21945}, 
}
```
