# RPITER
A hierarchical deep learning model for predicting ncRNA-protein interaction. 

The _sample_, _data_ and _result_ directories contain model codes, tested data sets and generated results, respectively.
The depended python packages are listed in _requirements.txt_. The package versions should be followed by users in their environments to achieve the supposed performance.

## How to run

The program is in Python 3.6 using [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) backends. Use the below bash command to run RPITER.

```bash
    python rpiter.py -d dataset
```

The parameter of _dataset_ could be RPI369, RPI488, RPI1807, RPI2241 or NPInter. Then, RPITER will perform 5-fold cross validation on the specific dataset.


## Five RPI datasets

The widely used RPI benchmark datasets are organized in the _data_ directory. 

Dataset | #Positive pairs | #Negative pairs | RNAs | Proteins | Reference
-------|----|------|----|-------|----------
RPI369 | 369 | 0 | 332 | 338 | [1]
RPI488 | 243 | 245 | 25 | 247 | [2]
RPI1807 | 1807 | 1436 | 1078 | 3131 | [3]
RPI2241 | 2241 | 0 | 841 | 2042 | [1]
NPInter | 10412 | 0 | 4636 | 449 | [4]

## Help

For any questions, feel free to contact me by chengpengeace@gmail.com or start an issue instead.



[1] Muppirala, U.K.; Honavar, V.G.; Dobbs, D. Predicting RNA-Protein Interactions Using Only Sequence Information. Bmc Bioinformatics 2011, 12. doi:Artn 489 10.1186/1471-2105-12-489.

[2] Pan, X.Y.; Fan, Y.X.; Yan, J.C.; Shen, H.B. IPMiner: hidden ncRNA-protein interaction sequential pattern mining with stacked autoencoder for accurate computational prediction. Bmc Genomics 2016, 17. doi:ARTN 582 10.1186/s12864-016-2931-8.

[3] Suresh, V.; Liu, L.; Adjeroh, D.; Zhou, X.B. RPI-Pred: predicting ncRNA-protein interaction using sequence and structural information. Nucleic Acids Research 2015, 43, 1370–1379. doi:10.1093/nar/gkv020.

[4] Yuan, J.;Wu,W.; Xie, C.Y.; Zhao, G.G.; Zhao, Y.; Chen, R.S. NPInter v2.0: an updated database of ncRNA interactions. Nucleic Acids Research 2014, 42, D104–D108. doi:10.1093/nar/gkt1057.
