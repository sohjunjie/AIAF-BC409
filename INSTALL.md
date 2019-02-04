# Project Installation

## Contents
- [Software installation and setup](#1-software-installation-and-setup)
- [Install python dependencies](#2-install-python-dependencies)
- [Code execution](#3-code-execution)
  - [Word2Vec model](#31-word2vec-model)
  - [Merge DJIA price dataset with news dataset](#32-merge-djia-price-dataset-with-news-dataset)
  - [Generate neural-network friendly dataset](#33-generate-neural-network-friendly-dataset)


### 1. Software installation and setup
- Get Mongodb [install](https://docs.mongodb.com/manual/installation/)
- Get Python v3.6.x


### 2. Install python dependencies
Change directory to the `AIAF-BC3409` project folder and install the required packages using the following command on `command line`
```
        $ pip install -r requirements.txt
```


### 3. Code execution
We will need to execute some python codes and commands to complete the project setup. The code excution are related to pre-processing of the raw dataset into one that meet our requirements.

#### 3.1 Word2Vec model
We are using `Gensim`, a python library for topic modelling, document indexing and similarity retrieval on large corpora. Amongst the extensive feature available in `Gensim`, we will be using the `word2vec` API to convert news text sequence to vectors sequence, which are then acceptable as input in deep neural network model.

`Gensim` `word2vec` model trains on a text corpus to derive word embeddings. We will be using the `text8` corpus which can be downloaded [here](http://mattmahoney.net/dc/text8.zip). Once downloaded, unzip the `text8` corpus file and move it into the `/model` folder. After doing that, run the following command on `command line` in the project folder directory.

```
        $ python create_word2vec.py
```

#### 3.2 Merge DJIA price dataset with news dataset
Our dataset source originates from [here](https://www.kaggle.com/aaron7sun/stocknews#Combined_News_DJIA.csv). There 3 given `csv` and their description are as follows.

- `Combined_News_DJIA.csv`: The dataset tells for a given trading day whether its adjusted closing price rose or stay the same, or decreased in value with the top 25 news headlines for that particular day
- `DJIA_table.csv`: The dataset tells the DJIA opening, closing, highest, lowest, adjusted closing price and trading volume for a given day
- `RedditNews.csv`: The dataset lists the top news headline from year 2008 to 2016

For our purpose, we need to have a dataset that combines DJIA trading price, trading volume and news features so that we can train a model that predict future price base on the mentioned features. We will need to merge the `Combined_News_DJIA.csv` that is missing the DJIA price and volume feature with the `DJIA_table.csv` dataset that do contain the features by a common trading date. Running the following command does exactly that.

```
        $ python create_dataset.py
```

Running the above command should generate a `DJIA_PRICE_NEWS.csv` dataset. This file should contain 1989 data points with 33 features.

#### 3.3 Generate neural-network friendly dataset
As mentioned previously, we will want to convert text sequences in news into vector sequence using the `gensim` `word2vec` model. This step will require running the following command.

```
        $ python create_dataset_db.py
```

By running the above command, we will be converting news features into its vector representation at a batch size of `50` items at a time. When the batch of 50 items completed the conversion process, they will be persisted into a `NOSQL` database. This step is necessary since we will encounter memory issue when generating large vectors representation of news text sequence for all the data points.
