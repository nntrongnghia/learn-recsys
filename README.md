# learn-recsys
Get started with Recommendation Systems in PyTorch. This repository covers the chapter 16 "Recommender Systems" in the book [Dive into Deep Learning](https://d2l.ai/chapter_recommender-systems/index.html).

## Setup
This repo is tested with Python 3.9 and PyTorch 1.10. For other packages, check `requirements.txt`.

For quick setup: `pip install -r requirements.txt`

## Dataset
Download these dataset and unzip in the root of this directory:
- [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- [CTR](http://d2l-data.s3-accelerate.amazonaws.com/ctr.zip)

## Corresponding scripts for each section in the book
1. Overview: no script for this section
2. The MovieLens Dataset: [`ml100k.py`](ml100k.py) contains dataloaders used in the following sections.
3. Matrix Factorization: [`mf.py`](mf.py)
4. AutoRec: Rating Prediction with Autoencoders: [`autorec.py`](autorec.py)
5. Personalized Ranking for Recommender Systems: [`utils.py`](utils.py). For now, only the BRP loss function is implemented.
6. Neural Collaborative Filtering for Personalized Ranking: [`neumf.py`](neumf.py)
7. Sequence-Aware Recommender Systems: [`caser.py`](caser.py)
8. Feature-Rich Recommender Systems: [`ctr.py`](ctr.py) contains dataloaders for CTR dataset
9. Factorization Machines: [`fm.py`](fm.py)
10. Deep Factorization Machines: [`deepfm.py`](deepfm.py)

## Usage
To train model in sections 3, 4, 6, 7, 9, 10, simply run `python corresponding_script.py`. Some hyperparameters like batch size, embedding dimension, etc. can be modified by arguments.

To check the training progress, run `tensorboard --logdir lightning_logs`
