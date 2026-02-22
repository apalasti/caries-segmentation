# caries-segmentation

## Key Models and Performance


| Model/System                                                                                 | Imaging Type    | Key Features                                                                                        | Performance Metrics                                                                                                                                                        |
| -------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CariesNet [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8736291/)              | Panoramic X-ray | U-shape network with the additional full-scale axial attention module to segment three caries types | Mean 93.64% Dice coefficient and 93.61% accuracy.                                                                                                                          |
| CariSeg [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405844024068671) | Panoramic X-ray | Ensemble of U-net, Feature Pyramid Network (FPN), and DeeplabV3                                     | This results in 94.895% accuracy and a Dice score of 88.5% for teeth segmentation, as well as a 99.42% accuracy and a mean 68.2% Dice coefficient for caries segmentation. |
| U-Net + ResNet-50 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12659894/)     | RVG radiographs | Pixel-wise segmentation (U-Net), multi-class (ResNet-50) (no/enamel/dentin caries)                  | Dice 0.89, Accuracy 93.2%                                                                                                                                                  |


Studies favor intraoral and panoramic X-rays, with CNNs enabling classification and segmentation across dentition stages.

---

1. **Class Imbalance:** A radiograph is 95% background/healthy tooth and only <5% caries. Loss functions like **Focal Loss** or **Dice Loss** are essential to prevent the model from just predicting "healthy" everywhere.
2. **The "Mach Band" Effect:** An optical illusion in X-rays that mimics decay at the enamel-dentin junction. Models often produce False Positives here.
3. **Restoration Artifacts:** Metallic fillings cause bright streaks (beam hardening) that obscure adjacent caries, confusing segmentation models.
4. **Enamel vs. Dentin:** Distinguishing between reversible (enamel) and irreversible (dentin) caries is clinically vital but computationally difficult due to low contrast.

