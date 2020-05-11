### (Non)-Interaction pairs

Let *XXX* be a dataset name, i. e. *RPI369*, *RPI488*, *RPI1807*, *RPI2241* or *NPInter*.
In *XXX_pairs.txt*, each line means an (non)-interactive protein RNA pair.

```
Protein RNA Label
```

Here, `Protein` and `RNA` are corresponding sequence names. `Label` is 1 or 0 representing interaction or non-interaction.

### Primary sequence data

Directory *sequence* is about the primary sequence data of protein and RNA.
In *XXX_protein_seq.fa* or *XXX_rna_seq.fa*, each `Protein` or `RNA` name is followed by its sequence data in the next line.

For example, in *RPI1807_rna_seq.fa*, the initial two lines show the primary sequence data of RNA `3OVB-C`.

```
>3OVB-C
GGAAGUAGAUGGUUCAAGUCCAUUUACUUCCACCA
```

### Sequence structure data

Directory *structure* is about the predicted sequence structure data of protein and RNA.
In *XXX_protein_struct.fa* or *XXX_rna_struct.fa*, each `Protein` or `RNA` name is followed by its predicted sequence structure (3 states of Helix, Sheet and Coil for protein; two states of dot-bracket format for rna) in the next line.

For example, in *RPI1807_rna_struct.fa*, the initial two lines show the predicted sequence structure of RNA `3OVB-C`.

```
>3OVB-C
((((((((((((.......))))))))))))....
```
