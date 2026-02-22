## 1. Children's Dental Panoramic Radiographs

- **Description:** Images mainly of children, which can make caries segmentation challenging.  
- **Annotations:** 
  - Teeth segmented  
  - Caries masks available  
- **Number of images:** ~100  
- **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset)  
- **GPU requirement:** Smaller CNNs (~U-Net) require 8–12 GB VRAM  
- **Suggested model:** U-Net / DoubleU-Net, CNN-based caries segmentation

---

## 2. DC1000

- **Description:** 597 high-resolution panoramic X-ray images  
- **Annotations:** 
  - Teeth segmented, but **not separated at instance level**  
  - Caries masks available at pixel level  
- **GPU requirement:** 
  - Published experiments used **NVIDIA V100 32GB**  
  - Smaller CNNs (~U-Net, DoubleU-Net) can run with 12–16 GB VRAM  
- **Link:** [DC1000 Drive](https://drive.google.com/file/d/1Xn1oGHvhGF9GbkcLEtCOV5QvWWqt1y62/view)  
- **Publication:** [arXiv:2511.14860](https://arxiv.org/pdf/2511.14860)  
- **Suggested model:** DoubleU-Net or U-Net, CNN-based caries segmentation

---

## 3. Panoramic Dental Dataset

- **Description:** Collection of panoramic dental images and their corresponding caries segmentation masks.  
- **Annotations:** 
  - Tooth boundaries and teeth segmented, in bounding box format  
  - Caries masks available  
- **Number of images:** ~100  
- **Publication:** No known publication has used this dataset  
- **Link:** [Kaggle](https://www.kaggle.com/datasets/thunderpede/panoramic-dental-dataset/data)  
- **GPU requirement:** Smaller CNNs require ~8–12 GB VRAM  
- **Suggested model:** U-Net / Mask R-CNN (if you want to ROI crop using bounding boxes)