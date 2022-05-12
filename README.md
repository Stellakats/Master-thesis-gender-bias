# Measuring Gender Bias in Contextualized Embeddings

This repository includes the code necessary to reproduce the experiments used to measure gender bias in contextualized embeddings as part of this [Master Thesis Project](https://www.diva-portal.org/smash/record.jsf?dswid=3668&pid=diva2%3A1618310&c=1&searchType=SIMPLE&language=en&query=styliani+katsarou+&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all) which was done in collaboration with [KTH](https://www.kth.se/en) and [Peltarion](https://peltarion.com/).

This work has also been presented in the ["Artificial intelligence with Biased or Scarce Data" (AIBSD) Workshop](https://aibsdworkshop.github.io/2022/index.html) held In Conjunction with the 36th AAAI Conference on Artificial Intelligence 2022, the proceedings of which have also been published with MDPI. Please find the proceeding paper [here](https://www.mdpi.com/2813-0324/3/1/3).  

Methods: 
1. **Measuring gender bias through the downstream task of Semantic textual similarity.**<br />Languages used: English and Swedish.<br />Models used: T5 and mT5.<br />
1. **Measuring gender bias on contextualized embeddings using the gender polarity metric.**<br />Language used: English.<br />Model used: T5.<br />  

Datasets:

We create new datasets out of the [Swedish STS benchmark](https://github.com/timpal0l/sts-benchmark-swedish) and the [English STS benchmark](http://ixa2.si.ehu.eus/stswiki/index.php/Main_Page) (see ```dataloader/create_bias_dataset.py```)   

# Reproducibility


**1. Clone repository**

```
git clone git@github.com:Stellakats/Master-thesis-gender-bias.git
cd Master-thesis-gender-bias
```

**2. Create new virtual environment**
```
conda create --name <env_name> python=3.7
conda activate <env_name> 
pip install -r requirements.txt
```

**3. Reproduce first method**

To reproduce the results using the first method simply run the following script.

```
python run_bias_experiments.py
```

Once the experiments are complete, the results will be available in ```results/bias_experiments```.<br />
The  ```results/bias_experiments``` directory will include:
- graphs for all 50 occupations alongside with the corresponding ```.csv``` files for the small, base and large version of T5
- graphs for all 50 occupations alongside with the corresponding ```.csv``` files for the small and base version of mT5 for both Swedish and English
- one graph for all sizes of T5 vs a specific occupation of choice. <br />The default occupation used is ```technician```.<br /> If you want to plot another occupation, please run  ```python run_bias_experiments.py --occupation "<occupation>"``` instead. 

**4. Reproduce second method**

To reproduce the second method and visualize the results, please run the notebook ```gender_polarity_t5_embeds.ipynb```.
