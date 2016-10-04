LITIV *datasets* Module
-----------------------
This folder contains standalone dataset parsing and result evaluation utilities with support for data precaching, async processing and on-the-fly custom work batch generation. The implementations here are largely based on template specialization and multiple interface inheritance. The only header required to use the module is [datasets.hpp](./include/litiv/datasets.hpp). The interfaces currently support image/video-based data parsing and saving, and evaluation for binary classification problems (including BSDS500 via a custom evaluator). As of v1.3.0, the interfaces also offer parallel stream processing (e.g. for cosegmentation or registration) and evaluation.

List of (non-LITIV) datasets with out-of-the-box specialization (more should be added over time):
  - Foreground-background video segmentation & background subtraction:
    - [ChangeDetection.net 2012](http://wordpress-jodoin.dmi.usherb.ca/cdw2012)
    - [ChangeDetection.net 2014](http://wordpress-jodoin.dmi.usherb.ca/cdw2014)
    - Wallflower
    - PETS2006 (dataset#3)
  - Boundary segmentation & edge detection:
    - [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
  - Multimodal video cosegmentation
    - [VAP trimodal segmentation dataset](http://www.vap.aau.dk/vap-trimodal-people-segmentation-dataset/)

These specializations are located in individual header files in the [impl directory](./include/litiv/datasets/impl/). For an example on how to parse a non-specialized (i.e. runtime-defined) pre-structured dataset using the classes implemented here, see the [*dataset_simple*](../../samples/dataset_simple/) sample project. In any case, if a dataset is instantiated with an evaluation strategy in its template, and if its groundtruth can be parsed by the data producer interface, its consumer interface will be able to evaluate the results directly based on the task type, and provide an evaluation report at the end. If no groundtruth is found (or if an evaluation strategy is not provided), the dataset will still be able to report useful metadata such as processed packet counts, processing speed, and framework version number/git commit id.

Apart from the dataset parsing utilities, this module introduces a data packet cacher standalone class ("DataPrecacher") which can be used to pre-fetch data samples for an algorithm, reducing I/O delays. Similarly, a data packet writer standalone class ("DataWriter") is also offered to allow queueing of data packets for an I/O device that is slower than the processing stream (ideal for, e.g., HD video capture applications).
