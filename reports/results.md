| Model | Parameters | Epochs | Training time | Test accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 1,143,242 | 15 | 16.3 min (CPU) | **88.22%** | 0.878 |
| Improved CNN | 577,514 | 30 | 162.1 min (CPU) | **96.67%** | 0.966 |

Most frequent confusions (improved model):

- HerbaceousVegetation predicted as PermanentCrop: 16 images
- AnnualCrop predicted as PermanentCrop: 14 images
- PermanentCrop predicted as HerbaceousVegetation: 11 images
