This is a super-resolution dataset containing paired LR-HR scene text images.

The LR images in TextZoom is much more challenging than synthetic LR images(BICUBIC).
![Synthetic LR vs Real LR](syn_real.jpg)

We allocate our dataset into 3 part following difficulty: easy, medium and hard subset. The misalignment and ambiguity increases as the difficulty increases.
![Example Images](easy_medium_hard.jpg)

For each pair of LR-HR images, we provide the annotation of the case sensitive character string (including punctuation), the type of the bounding box, and the original focal lengths.
