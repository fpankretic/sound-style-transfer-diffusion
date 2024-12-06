# sound-style-transfer-diffusion

### Authors

- [Filip Pankretić](https://github.com/fpankretic)
- [Filip Perković](https://github.com/filip-perkovic)
- [Dominik Jambrović](https://github.com/DomJamb)
- [Velimir Kovačić](https://github.com/velimirkovacic)
- [Luka Glavinić](https://github.com/LukaGlavinic)
- [Fran Vučković](https://github.com/FranVuckovic)

### Description

Music style transfer using diffusion model.
In this project we try to recreate the paper
[Music Style Transfer with Time-Varying Inversion of Diffusion Models](https://lsfhuihuiff.github.io/MusicTI/) by
Sifei Li, Yuxin Zhang, Fan Tang, Chongyang Ma, Weiming Dong, Changsheng Xu.

### Environment

We are using Conda to manage our environments. Depending on if you are using CPU or GPU, you can create the environment
with one of the following commands:

CPU:
```bash
conda env create -f environment-cpu.yaml
```

GPU:
```bash
conda env create -f environment-gpu.yaml
```

### Dataset

To download the dataset, run the following command from root directory:

```bash
python dataset.py
```