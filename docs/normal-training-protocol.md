# Conventional training before SISA

## Model choice

Start with **Torchvision ResNet-18, ImageNet1K_V1 initialization**, replacing its 1,000-class head with a two-class head and fine-tuning **all** parameters. One model sees all training images. There is no partitioning, deletion or SISA replay in this experiment.

| Candidate | Role in this project | Decision |
|---|---|---|
| ResNet-18, ImageNet weights | Established architecture; 11.18M parameters after replacing the head; practical CPU starting point | First conventional baseline |
| ResNet-50, random initialization | Architecture used for CheXpert in Nasirigerdeh et al. | Later medical-paper reproduction on the matching dataset/protocol |
| DenseNet-121, ImageNet or verified non-PadChest chest-X-ray weights | Chest imaging alternative; more convolutional operations than ResNet-18 | Optional follow-up after the pipeline works |
| TorchXRayVision PadChest or mixed `all` checkpoint | These initializations already include PadChest training | Do not use for this patient-unlearning evaluation |

The chosen ImageNet checkpoint is not trained on PadChest according to its documented provenance. This avoids known PadChest contamination in the initialization; it is not a claim of a comprehensive overlap audit. Code is BSD-3-Clause; pretrained-weight use is also subject to applicable dataset terms. Do not upload pretrained weights or dataset images to this repository.

## Fixed first-run protocol

| Item | Setting |
|---|---|
| Data | Downloaded `6.zip`; only patients whose complete metadata-listed image sets are present in this archive, followed by image eligibility filters |
| Eligible images | PA/AP only, nonempty labels, no `exclude` or `suboptimal study` label |
| Target | `1` if `normal` appears in the annotation list, `0` otherwise; absence does **not** establish an abnormal diagnosis |
| Split | Same patient-hash assignment and seed (20261021) as the previous SISA pilot; 435/123/95 train/validation/test patients and 459/128/99 images |
| Pixels | Original decoded PNGs; per-image 1st–99th percentile window, correct MONOCHROME1 inversion, aspect-preserving bilinear resize and zero padding to 224×224 |
| Model input | Repeat grayscale across RGB, normalize by ImageNet means `(0.485,0.456,0.406)` and SDs `(0.229,0.224,0.225)` |
| Optimization | Full fine-tuning, Adam, learning rate 0.0001, weight decay 0.0001, cross-entropy, batch 16, six epochs, no scheduler or augmentation |
| Selection | Highest validation AUROC; keep earliest epoch if tied. Test set is evaluated only after selection in this run |
| Threshold | Fixed 0.5; no threshold tuning on test |
| Execution | CPU, four PyTorch threads, deterministic algorithms, zero loader workers |
| Saved locally | Exact initialization, best and last model/optimizer/RNG checkpoints, configuration, per-epoch history, aggregate metrics |

We preserve the whole resized radiograph instead of using ImageNet's default center crop. The input intensities are adapted to radiographs, rather than treating 16-bit pixels as ordinary RGB photographs. This preprocessing is an explicit design choice, not a validated optimal preprocessing recipe.

Train, validation and test patients remain disjoint. Validation/test images never update model parameters or batch-normalization statistics. Batch normalization is trained normally on training images. The model is selected by validation AUROC, not by test performance.

## Reproduce

From the repository root, install `requirements-training.txt`. Download the official checkpoint into the ignored cache:

```powershell
New-Item -ItemType Directory -Force outputs/model-cache | Out-Null
Invoke-WebRequest -Uri https://download.pytorch.org/models/resnet18-f37072fd.pth -OutFile outputs/model-cache/resnet18-f37072fd.pth
python -m src.train_baseline --metadata data/raw/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz --archive "$env:USERPROFILE/Downloads/6.zip" --output outputs/normal-resnet18-run
```

The program verifies the full checkpoint SHA256 before loading. It prepares 224×224 pixels directly from the ZIP, not by upsampling the previous 64×64 cache. Choose a fresh output directory. The optional `--prepared-data` argument only accepts a cache with the same configuration; it checks metadata hash, archive size and image-array dimensions, but is not a full integrity audit of a subsequently modified local cache.

## Evaluation and interpretation

| Question | Measurement |
|---|---|
| Can the model rank the annotation classes? | Image-level AUROC on validation and test |
| Does the fixed classifier identify both classes? | Balanced accuracy, sensitivity for the normal label, specificity for the normal label, confusion matrix |
| What is overall predictive accuracy? | Accuracy; interpret alongside the majority-negative baseline (59/99 test images) |
| What is the computational cost? | Sum of training-epoch wall times; separately fit time including validation/checkpoint writes; preparation time separately |
| Did full fine-tuning happen? | All parameters trainable and initial convolution weights changed |

This is a small, single-seed convenience-sample experiment. The same exploratory test split was already inspected in the earlier pilot, so it is not a fresh confirmatory holdout. Repeated images from a patient are correlated; final uncertainty intervals must resample patients, not individual images. Report multiple seeds and use a larger, untouched evaluation cohort before making research claims.

**Next:** train patient-random SISA with the same ResNet-18 initialization, preprocessing, target and split. Then compare priority ordering with patient-random ordering under matched settings. The existing 64×64 TinyCNN SISA results are mechanics checks, not a fair performance comparison with this baseline. SISA checkpoints must restore the optimizer, RNG and batch-normalization buffers as well as weights. Validate replay against clean retained-data SISA; monolithic retained-data retraining is a separate utility/time comparator.

For patient placement, later training slices generally require less replay. A high-priority patient in the first trained slice would require replay of the subsequent stages; data lookup speed does not remove this training dependency.

## Sources

- [Official Torchvision 0.22 ResNet-18 model and weights](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)
- [Torchvision pretrained-model usage and terms note](https://docs.pytorch.org/vision/0.22/models.html)
- [Torchvision v0.22.1 code licence](https://github.com/pytorch/vision/blob/v0.22.1/LICENSE)
- [DenseNet-121 specifications](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.densenet121.html)
- [TorchXRayVision checkpoint training datasets](https://mlmed.org/torchxrayvision/models.html)
- [Nasirigerdeh et al., Machine Unlearning for Medical Imaging (2024)](https://arxiv.org/pdf/2407.07539)
- [Bourtoule et al., Machine Unlearning / SISA](https://arxiv.org/abs/1912.03817)
