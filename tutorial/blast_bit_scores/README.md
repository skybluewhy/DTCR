extract files and set path
```
tar -zxvf ncbi-blast-2.16.0+-x64-linux.tar.gz
export PATH=/hy-tmp/ncbi-blast-2.16.0+/bin:$PATH
```

reference tcrs file: https://drive.google.com/file/d/1wFRogU8jLEhVfzferbItlVhWvTknu2ZF/view?usp=drive_link

process data for reference tcr database
```
python get_ref_database.py
```

construct reference tcr database
```
makeblastdb -in tcrdb_tcrs.fasta -dbtype prot -out tcrdb
```

process generated tcrs for align
```
python get_generated_tcrs_for_align.py
```

align & get results
```
blastp -query gen_tcrs.fasta -db tcrdb -out result -matrix BLOSUM62 -outfmt 6 -evalue 0.1 -max_target_seqs 100
```

draw blast bit scores distribution
```
python draw_line_chart.py
```

draw blast bit scores Cumulative distribution function (CDF)
```
python draw_correlations.py
```

