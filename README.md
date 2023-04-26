# psmc-python
![psmc_1-05](https://user-images.githubusercontent.com/1506940/234715469-00f03580-0480-42e4-96af-ddbe39519c39.png)

**psmc-python** is a reimplementation of the PSMC (Pairwise Sequential Markovian Coalescent) method in Python, intended for educational purposes. The PSMC method is used to estimate the demographic history of a population by analyzing the patterns of mutations accumulated in the genome. For more details on the PSMC method, please refer to the original <a href='https://github.com/lh3/psmc'>C implementation</a> and the <a href='https://www.nature.com/articles/nature10231'>research paper</a>.

## Getting started
To use psmc-python, first obtain a psmcfa file by running the following commands on a genome:

```bash
samtools mpileup -C50 -uf GRCh38_full_analysis_set_plus_decoy_hla.fa -r chr1 \
SAMEA3302828.alt_bwamem_GRCh38DH.20200922.French.simons.cram | bcftools call -c \
-Ov -o French_chr1.vcf -

cat French_chr1.vcf | vcfutils.pl vcf2fq -d 10 -D 100 | gzip > French_chr1.diploid.fq.gz

path_to_original_psmc/utils/fq2psmcfa -q20 French_chr1-5.diploid.fq.gz > French_chr1.psmcfa
```

Then, run psmc-python using the following command:

```bash
python run.py French_chr1.psmcfa checkpoints/french_1.json 15 --t_max 15 --n_steps 64 --pattern '1*4+25*2+1*4+1*6' --batch_size 300000
```

For interactive examples, refer to the **example.ipynb** notebook. The trained model parameters can be found in the **checkpoints/** folder, psmcfa files of genome data in the **genomes/** folder, and simulations in the **simulations/** folder.

Dont forget to unzip using the following command:

```bash
gzip -d genomes/*.gz simulations/*.gz
```

