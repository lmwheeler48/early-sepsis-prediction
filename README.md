# Early Sepsis Prediction

Team members: [Yang Mo](https://github.com/zetacaveman), [Sayantan Sarkar](https://github.com/Sayantan128), [Cristopher Thompson](), [Alexandria Wheeler](https://github.com/lmwheeler48)

--- Need link to Cristopher's Github page ---

# Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Dataset Preprocessing](#dataset-preprocessing)
4. [Baseline Models](#baseline-models)
5. [Deep Learning Models](#deep-learning-models)
6. [Final Model & Results](#final-model--results)
7. [Future Work](#future-work)
8. [Description of Repository](#description-of-repository)

## Introduction

Sepsis is a life-threatening condition that occurs when the body’s response to infection causes tissue damage, organ failure, or death. In the U.S. alone, 1.7 million cases and 350,000 deaths occur each year ([CDC](https://www.cdc.gov/mmwr/volumes/72/wr/mm7234a2.htm)), accounting for $24 billion annually ([Paoli et al., 2018](https://pubmed.ncbi.nlm.nih.gov/30048332/)). Early detection is critical, as each hour of delayed treatment increases mortality by 4–9% ([Seymour et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28528569/)). The goal of this project is to develop a deep learning predictive model to identify sepsis in ICU patients before clinical diagnosis, enabling earlier intervention to reduce mortality, decrease treatment costs, and improve resource allocation.

## Exploratory Data Analysis

The dataset consists of 40,000+ ICU patients' hourly time-series data across 40 clinical variables (vital signs, lab results) and demographics. Each patient was given a label identifying whether or not the patient was later diagnosed with sepsis. An initial EDA revealed three major challenges to address. First, the dataset is severely imbalanced, with only about 7% of patients actually developing sepsis across the two hospitals. Second, the scale of the dataset is massive, with over 1.4 million hourly patient records to work with. And third, there is significant missing data, with only about 20% density across the dataset. 

![EDA dataset by hosp](./Assets/EDA_dataset_by_hosp.png "EDA dataset by hosp")

The dataset density varied by feature, as seen in the figure below. Some features were more dense than others, such as the vital sign data which were typically entered hourly. However, lab results were especially sparse, with some patients having no entries, and others with entries once every few hours. 

![EDA feature density](./Assets/EDA_feature_density.png "EDA feature density")


## Dataset Preprocessing

## Baseline Models

We established a logistic regression baseline to benchmark deep learning performance. While quick to train on the large dataset, the model did not achieve high accuracy metrics, with a test AUROC of 66% and low precision and recall scores at 23% and 1% respectively.

![LR ROC PRC](./Assets/LR_roc_prc.png "LR ROC PRC")

At this time, it is clear that this baseline model is insufficient for clinical deployment and further work is needed to address issues impacting performance. We also attempted a K-Nearest Neighbors model with Dynamic Time Warping distance, which is specifically designed to handle time-series data. However, this model was not equipped to handle the scale of the dataset, nor the severe class imbalance. 

## Deep Learning Models

## Final Model & Results

## Future Work

## Description of Repository
