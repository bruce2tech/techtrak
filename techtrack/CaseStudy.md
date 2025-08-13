## Model Comparison: YOLOv4-Tiny1 vs YOLOv4-Tiny2 

This document summarizes and compares the performance of two object detection models (yolov4-tiny1 and yolov4-tiny2) on the TechTrack logistics dataset using the following metrics: Precision, Recall, F1-Score at 0.5 confidence, ROC-AUC, 
Average Precision (AP) and mean Average Precision (mAP) both calculated at IoU = 0.5.

### Task 1: Compare Model 1 and Model 2 on the TechTrack dataset using standard object-detection metrics.

Average precision measures the model's ability to correctly identify and localize objects across different confidence thresholds. It summarizes the precision-recall curve in a single score that balances precision and recall.

The F1-score helps evaluate models at a fixed confidence threshold by ensuring a balance between catching actual cases (recall) and avoiding misidentifications (precision), especially in scenarios with imbalanced classes.

ROC-AUC uses true positive rate (TPR) vs false positive rate (FPR) across all thresholds to convey how often positives score above negatives. 

#### Per-Model Per-Class Metrics
| Model        | Class             |   AP@0.5 |   Precision@0.5 |   Recall@0.5 |   F1@0.5 |   ROC-AUC |
|:-------------|:------------------|---------:|----------------:|-------------:|---------:|----------:|
| yolov4-tiny1 | barcode           | 0.464827 |        0.880795 |     0.488971 | 0.628842 |  0.743587 |
| yolov4-tiny1 | car               | 0.720068 |        0.602086 |     0.808    | 0.690009 |  0.887176 |
| yolov4-tiny1 | cardboard box     | 0.631948 |        0.838806 |     0.634312 | 0.722365 |  0.815518 |
| yolov4-tiny1 | fire              | 0.217407 |        0.956835 |     0.117387 | 0.209119 |  0.558331 |
| yolov4-tiny1 | forklift          | 0.533984 |        0.957404 |     0.497366 | 0.654646 |  0.747624 |
| yolov4-tiny1 | freight container | 0.282805 |        0.782051 |     0.317708 | 0.451852 |  0.658018 |
| yolov4-tiny1 | gloves            | 0.512463 |        0.914062 |     0.517699 | 0.661017 |  0.758413 |
| yolov4-tiny1 | helmet            | 0.668605 |        0.899358 |     0.66561  | 0.765027 |  0.831176 |
| yolov4-tiny1 | ladder            | 0.214705 |        0.8      |     0.218579 | 0.343348 |  0.60881  |
| yolov4-tiny1 | license plate     | 0.329943 |        0.840708 |     0.325342 | 0.469136 |  0.661896 |
| yolov4-tiny1 | person            | 0.609001 |        0.907118 |     0.54061  | 0.677472 |  0.764718 |
| yolov4-tiny1 | qr code           | 0.820461 |        0.891156 |     0.876254 | 0.883642 |  0.936918 |
| yolov4-tiny1 | road sign         | 0.22329  |        0.709402 |     0.223118 | 0.339468 |  0.609868 |
| yolov4-tiny1 | safety vest       | 0.617752 |        0.916667 |     0.620042 | 0.739726 |  0.808929 |
| yolov4-tiny1 | smoke             | 0.275445 |        1        |     0.192821 | 0.323302 |  0.59641  |
| yolov4-tiny1 | traffic cone      | 0.803257 |        0.887218 |     0.810997 | 0.847397 |  0.904882 |
| yolov4-tiny1 | traffic light     | 0.727974 |        0.97619  |     0.710162 | 0.822193 |  0.85444  |
| yolov4-tiny1 | truck             | 0.560359 |        0.894855 |     0.566572 | 0.693842 |  0.781091 |
| yolov4-tiny1 | van               | 0.672072 |        0.705882 |     0.742446 | 0.723703 |  0.863328 |
| yolov4-tiny1 | wood pallet       | 0.745529 |        0.926523 |     0.750363 | 0.82919  |  0.873603 |
| yolov4-tiny2 | barcode           | 0.380633 |        0.913793 |     0.389706 | 0.546392 |  0.694337 |
| yolov4-tiny2 | car               | 0.725967 |        0.669484 |     0.792    | 0.725607 |  0.883367 |
| yolov4-tiny2 | cardboard box     | 0.702973 |        0.548282 |     0.756208 | 0.635674 |  0.87148  |
| yolov4-tiny2 | fire              | 0.306555 |        0.97619  |     0.217123 | 0.355235 |  0.608211 |
| yolov4-tiny2 | forklift          | 0.753178 |        0.892193 |     0.758693 | 0.820046 |  0.876405 |
| yolov4-tiny2 | freight container | 0.325954 |        0.804598 |     0.364583 | 0.501792 |  0.681494 |
| yolov4-tiny2 | gloves            | 0.560189 |        0.968992 |     0.553097 | 0.704225 |  0.776409 |
| yolov4-tiny2 | helmet            | 0.716259 |        0.894325 |     0.724247 | 0.80035  |  0.860401 |
| yolov4-tiny2 | ladder            | 0.364538 |        0.886076 |     0.382514 | 0.534351 |  0.690786 |
| yolov4-tiny2 | license plate     | 0.373165 |        0.88     |     0.376712 | 0.527578 |  0.687668 |
| yolov4-tiny2 | person            | 0.701473 |        0.888658 |     0.66477  | 0.76058  |  0.825097 |
| yolov4-tiny2 | qr code           | 0.822756 |        0.918728 |     0.869565 | 0.893471 |  0.933745 |
| yolov4-tiny2 | road sign         | 0.209442 |        0.47549  |     0.260753 | 0.336806 |  0.625074 |
| yolov4-tiny2 | safety vest       | 0.702284 |        0.92623  |     0.707724 | 0.802367 |  0.852862 |
| yolov4-tiny2 | smoke             | 0.35337  |        0.978571 |     0.281026 | 0.436653 |  0.640226 |
| yolov4-tiny2 | traffic cone      | 0.775637 |        0.923695 |     0.790378 | 0.851852 |  0.894566 |
| yolov4-tiny2 | traffic light     | 0.736803 |        0.979592 |     0.720554 | 0.830339 |  0.859636 |
| yolov4-tiny2 | truck             | 0.731656 |        0.756    |     0.803116 | 0.778846 |  0.89524  |
| yolov4-tiny2 | van               | 0.751309 |        0.693413 |     0.833094 | 0.756863 |  0.908528 |
| yolov4-tiny2 | wood pallet       | 0.788446 |        0.946181 |     0.791001 | 0.86166  |  0.894335 |

Table 1: The shows the resulting metrics for each model for each class.


Figures 1, 2 and 3 display side by side comparisons of the models for each class for Average-Precision, F1 Scores and ROC-AUC values respectively. 

#### Per-Class AP@05 plus overall mAP

![Alt text](modules/per_class_AP_with_mAP.png)
Figure 1: Bar graph displaying average precision per-class along with mAP for each model in a side-by-side comparison.


#### Per-Class F1 Score @ Confidence=0.5 

![Alt text](modules/per_class_F1.png)
Figure 2: Bar graph displaying F1-Scores per-class for each model in a side-by-side comparison.

#### Per-Class ROC-AUC

![Alt text](modules/per_class_ROC_AUC.png)
Figure 3: Bar graph displaying ROC-AUC per-class for each model in a side-by-side comparison.

### Task 2: Identify which classes Model 2 handles better than Model 1.

The Tables 2, 3 and 4 compare the average precesion, F1 scores, and ROC-AUC scores, respectively, listing the deltas between the values and identifying which model performed best for the metric.  Its important to note in object detection average precision at a specified IoU is the primary metric as it accounts for the localization factor and therefore reflects both classification and box quality.

For average precision model 1 performs better than model 2 detecting barcodes, road signs and traffic cones.

| Class Index                            | Class Name        | AP Tiny1     | AP Tiny2     | Δ = Tiny2 – Tiny1 | Winner       |
|:---------------------------------------|:------------------|:-------------|:-------------|:------------------|:-------------|
| 0                                      | barcode           | 0.4648       | 0.3806       | -0.0842           | Tiny1        |
| 1                                      | car               | 0.7201       | 0.726        | 0.0059            | Tiny2        |
| 2                                      | cardboard box     | 0.6319       | 0.703        | 0.0711            | Tiny2        |
| 3                                      | fire              | 0.2174       | 0.3066       | 0.0892            | Tiny2        |
| 4                                      | forklift          | 0.534        | 0.7532       | 0.2192            | Tiny2        |
| 5                                      | freight container | 0.2828       | 0.326        | 0.0432            | Tiny2        |
| 6                                      | gloves            | 0.5125       | 0.5602       | 0.0477            | Tiny2        |
| 7                                      | helmet            | 0.6686       | 0.7163       | 0.0477            | Tiny2        |
| 8                                      | ladder            | 0.2147       | 0.3645       | 0.1498            | Tiny2        |
| 9                                      | license plate     | 0.3299       | 0.3732       | 0.0433            | Tiny2        |
| 10                                     | person            | 0.609        | 0.7015       | 0.0925            | Tiny2        |
| 11                                     | qr code           | 0.8205       | 0.8228       | 0.0023            | Tiny2        |
| 12                                     | road sign         | 0.2233       | 0.2094       | -0.0139           | Tiny1        |
| 13                                     | safety vest       | 0.6178       | 0.7023       | 0.0845            | Tiny2        |
| 14                                     | smoke             | 0.2754       | 0.3534       | 0.078             | Tiny2        |
| 15                                     | traffic cone      | 0.8033       | 0.7756       | -0.0277           | Tiny1        |
| 16                                     | traffic light     | 0.728        | 0.7368       | 0.0088            | Tiny2        |
| 17                                     | truck             | 0.5604       | 0.7317       | 0.1713            | Tiny2        |
| 18                                     | van               | 0.6721       | 0.7513       | 0.0792            | Tiny2        |
| 19                                     | wood pallet       | 0.7455       | 0.7884       | 0.0429            | Tiny2        |

Table 2: Lists the difference/deltas between average precision values and identifies the best model which performs best as the winner.

Analyzing F1 scores at 0.5 confidence. Model 1 performs better than model 2 at detecting barcodes, cardboard boxes and road signs.

| Class Index                   | Class Name        | F1 Tiny1     | F1 Tiny2     | Δ = Tiny2 − Tiny1 | Winner       |
|:------------------------------|:------------------|:-------------|:-------------|:------------------|:-------------|
| 0                             | barcode           | 0.6288       | 0.5464       | -0.0824           | Tiny1        |
| 1                             | car               | 0.69         | 0.7256       | 0.0356            | Tiny2        |
| 2                             | cardboard box     | 0.7224       | 0.6357       | -0.0867           | Tiny1        |
| 3                             | fire              | 0.2091       | 0.3552       | 0.1461            | Tiny2        |
| 4                             | forklift          | 0.6546       | 0.82         | 0.1654            | Tiny2        |
| 5                             | freight container | 0.4519       | 0.5018       | 0.0499            | Tiny2        |
| 6                             | gloves            | 0.661        | 0.7042       | 0.0432            | Tiny2        |
| 7                             | helmet            | 0.765        | 0.8004       | 0.0354            | Tiny2        |
| 8                             | ladder            | 0.3433       | 0.5344       | 0.1911            | Tiny2        |
| 9                             | license plate     | 0.4691       | 0.5276       | 0.0585            | Tiny2        |
| 10                            | person            | 0.6775       | 0.7606       | 0.0831            | Tiny2        |
| 11                            | qr code           | 0.8836       | 0.8935       | 0.0099            | Tiny2        |
| 12                            | road sign         | 0.3395       | 0.3368       | -0.0027           | Tiny1        |
| 13                            | safety vest       | 0.7397       | 0.8024       | 0.0627            | Tiny2        |
| 14                            | smoke             | 0.3233       | 0.4367       | 0.1134            | Tiny2        |
| 15                            | traffic cone      | 0.8474       | 0.8519       | 0.0045            | Tiny2        |
| 16                            | traffic light     | 0.8222       | 0.8303       | 0.0081            | Tiny2        |
| 17                            | truck             | 0.6938       | 0.7788       | 0.085             | Tiny2        |
| 18                            | van               | 0.7237       | 0.7569       | 0.0332            | Tiny2        |
| 19                            | wood pallet       | 0.8292       | 0.8617       | 0.0325            | Tiny2        |

Table 3: Lists the difference/deltas between F1-Score and identifies the best model which performs best as the winner.


#### Model-Class ROC-AUC Deltas
According to ROC-AUC scores, model 1 outperforms model 2 in detecting the following classes: barcodes, cars, qr codes and traffic cones.

| Class Index                  | Class Name        | ROC-AUC Tiny1 | ROC-AUC Tiny2 | Δ = Tiny2 − Tiny1 | Winner       |
|:------------------------------|:------------------|:-------------|:-------------|:------------------|:-------------|
| 0                            | barcode           | 0.7436        | 0.6943        | -0.0493           | Tiny1        |
| 1                            | car               | 0.8872        | 0.8834        | -0.0038           | Tiny1        |
| 2                            | cardboard box     | 0.8155        | 0.8715        | 0.056             | Tiny2        |
| 3                            | fire              | 0.5583        | 0.6082        | 0.0499            | Tiny2        |
| 4                            | forklift          | 0.7476        | 0.8764        | 0.1288            | Tiny2        |
| 5                            | freight container | 0.658         | 0.6815        | 0.0235            | Tiny2        |
| 6                            | gloves            | 0.7584        | 0.7764        | 0.018             | Tiny2        |
| 7                            | helmet            | 0.8312        | 0.8604        | 0.0292            | Tiny2        |
| 8                            | ladder            | 0.6088        | 0.6908        | 0.082             | Tiny2        |
| 9                            | license plate     | 0.6619        | 0.6877        | 0.0258            | Tiny2        |
| 10                           | person            | 0.7647        | 0.8251        | 0.0604            | Tiny2        |
| 11                           | qr code           | 0.9369        | 0.9337        | -0.0032           | Tiny1        |
| 12                           | road sign         | 0.6099        | 0.6251        | 0.0152            | Tiny2        |
| 13                           | safety vest       | 0.8089        | 0.8529        | 0.044             | Tiny2        |
| 14                           | smoke             | 0.5964        | 0.6402        | 0.0438            | Tiny2        |
| 15                           | traffic cone      | 0.9049        | 0.8946        | -0.0103           | Tiny1        |
| 16                           | traffic light     | 0.8544        | 0.8596        | 0.0052            | Tiny2        |
| 17                           | truck             | 0.7811        | 0.8952        | 0.1141            | Tiny2        |
| 18                           | van               | 0.8633        | 0.9085        | 0.0452            | Tiny2        |
| 19                           | wood pallet       | 0.8736        | 0.8943        | 0.0207            | Tiny2        |

Table 4: Lists the difference/deltas between ROC-AUC and identifies the best model which performs best as the winner.

#### Overall mAP
However, overall, model 2 performed better than model 1.
| Model        |   mAP@0.5 |
|:-------------|----------:|
| yolov4-tiny1 |  0.531595 |
| yolov4-tiny2 |  0.589129 |

Table 4: Lists mean average precision for both models.
#### Important note on model tuning
Its important to consider tuning.
Figures 4 and 5 display F1 curves for the cardbox class only. This demonstrates how model performance varies with confidence thresholds. While model 1 out performs model 2 at 0.5 confidence, model 2 outperforms model 1 at 0.89 confidence. Analyzing per class curves and help determing optimal parameter values for given application of the models. 

![Alt text](modules/F1_curve_yolov4-tiny1_cls2.png)
Figure 4: F1 curve displaying model 1 performance over a range of confidence thresholds. 

![Alt text](modules/F1_curve_yolov4-tiny2_cls2.png)
Figure 5: F1 curve displaying model 2 performance over a range of confidence thresholds.

### Task 3 Measure the impact of Gaussian Noise, Vertical Flips, and Brightness adjustments on Model 1’s performance.

Figure 6 compares the mAP of model 1 with augmented data versus raw data. From the graph its clear that brghtness has some minor impact to performance while flips and gaussian noise reduce performance approximately half.

![Alt text](modules/viz_out/macro_map_bar.png "Bar Chart of Impact of Augmentation on mAP")
Figure 6: Bar graph comparing mAP when data is augmented versus original data.

Figure 7 Display the Precision-Recall curves of  the car class to further demonstrate the impact of the augmentations to model performance.

![Alt text](modules/viz_out/pr_curves_class_1.png "PR Curve: Car")
Figure 7: Precision-Recall curves of augmented data.

### Task 4: Analyze how different λ values in Hard Negative Mining (HNM) affect sampling of images with many ground-truth objects

As λ increases, the number of negatives kept rises in every bin, but the increase is much steeper for the most crowded images. By λ≈10 the top GT bin receives a dramatically larger share of negatives than the mid bins, meaning HNM is increasingly concentrating sampling on dense scenes rather than spreading it evenly.


![Alt text](../hnm_out/hnm_negkept_vs_gtbin.png)

Figure 8: The GT-bin axis goes from sparse → crowded. For each λ, the line shows how many negatives are kept on average in that bin.

![Alt text](../hnm_out/hnm_slope_vs_lambda.png)

Figure 9: The Slope vs. λ plot compresses that into one number per λ. If slope increases with λ, the HNM increasingly concentrates negatives on crowded scenes.

### Task 5: Analyze how different λ values in HNM affect sampling of images with many false negatives

With pos_ref=gt, larger λ also pushes more negatives into images that have many FNs: the high-FN bin grows far faster than the low-FN bin. In effect, the mining policy allocates more capacity to images the model currently struggles with, which can help recall—but at high λ it risks starving sparse/easy images and skewing learning toward crowded cases.

![Alt text](../hnm_out/hnm_negkept_vs_fn_bin.png)
Figure 10: Impact of λ on negatives kept vs FN Crowdedness. 