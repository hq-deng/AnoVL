### Official Implementation of [AnoVL](https://arxiv.org/abs/2308.15939) (Updating)
AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization.
![AnoVL](/teaser/VLAD.png)
## Dataset Preparation 
### MVTec AD
- Download and extract [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) into `data/mvtec`
- run`python data/mvtec.py` to obtain `data/mvtec/meta.json`
```
data
├── mvtec
    ├── meta.json
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png
```

### VisA
- Download and extract [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) into `data/visa`
- run`python data/visa.py` to obtain `data/visa/meta.json`
```
data
├── visa
    ├── meta.json
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
```

## Test
  ```shell
  sh test_zero_shot.sh
  ```
## Acknowledgements
We thank [clip](https://github.com/openai/CLIP), [open_clip](https://github.com/mlfoundations/open_clip), [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814), [A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD](https://arxiv.org/abs/2305.17382) for providing assistance for our research.
## Citation
```
@article{anovl,
  title={AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization},
  author={Deng, Hanqiu and Zhang, Zhaoxiang and Bao, Jinan and Li, Xingyu},
  journal={arXiv preprint arXiv:2308.15939},
  year={2023}
}
```
