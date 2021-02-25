# Sigmund
Analysis of **depression features** in text-transcripts of **couple conversations**. 

We examine transcripts of couple conversations from a current [research project](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173246/) at [Heidelberg University Hospital](https://www.klinikum.uni-heidelberg.de/zentrum-fuer-psychosoziale-medizin-zpm/institut-fuer-medizinische-psychologie) to identify depression-related features and quantify the differences between couples, in which one partner is suffering from major depressive disorder, and couples, in which both partners lack a history of depression. 

### Project Members
* Julius Daub (3536557) | Applied Informatics (M.Sc.) | julius@daubweb.com
* Alexander Haas (3503540) | Applied Informatics (M.Sc.) | haas.alexanderjulian@gmail.com
* Ubeydullah Ilhan (3447661) | Applied Informatics (M.Sc.) | ubeydullah.il@gmail.com
* Benjamin Sparks (3664690) | Applied Informatics (M.Sc.) | benjamin.sparks@protonmail.com

### Getting Started

A prerequisite is installing the required dependencies with Pipenv.
Then, activate your virtual environment, and choose any of of these entrypoints to get started:

* [feature summary](feature_summary.ipynb): Summary of all features in one notebook
* [pipeline demo](pipeline_demo.ipynb): Demonstrates usage of Pipeline API, including utilisation of singular classification and voting classification
* [streamlit](streamlit_pipeline.py): Simple visual front-end for rendering plots. 
In order to execute this file, checkout the `streamlit-demo` branch and execute `streamlit run streamlit_pipeline.py`.

### Contributions for the Milestone
#### Fist milestone (16.12.2020)
* Data acquisition: speech-to-text and docx-extraction
* Implementation of the pipeline architecture
* Implementation of a demo text mining workflow contained in the branch *demonstration*
* Implementation of demo components
* [Collection of possible metrics](https://docs.google.com/spreadsheets/d/1z2vkU259P_5mGQCHb67HgyoEulPsd03LQv2z-SoTG4g/edit?usp=sharing) from literature 
* Basic statistics of the data set and first iteration of feature engineering (got the data on 15.12.2020)
#### Second milestone (04.02.2021)
* Implementation of features into the pipeline
* Implementation of classifiers into the pipeline
* Extensive feature testing and summary of the results in *feature_summary.ipynb*
* Summary of the feature insights in a [presentation](https://drive.google.com/file/d/11y0URs2Jyc4s6zUTcpzpSDF0oWK-ttOv/view?usp=sharing)


<!-- ### Existing Code Fragments
* LIWC (https://github.com/chbrown/liwc-python) -> library extended *src/utils/liwc.py* 
-->

### Utilized Libraries

We use several libraries for the project, including:
* Numpy 
* Pandas
* sklearn
* matplotlib
* Spacy + German News Dataset (https://spacy.io/models/de)
* NLTK
* SentiWS
* pyphen
* liwc-python
* gensim
* Stanford Parser Toolkit 
* Streamlit to build a simple front-end
* Seaborn for visualisation
* LIWC library with german dictionary (https://pypi.org/project/liwc/)

The libraries and versions are specified in the Pipfile.

## Project State


### Project Planning

* 15.01. Implementation of all features and first results to share with the Institute, to evaluate if further transcripts are possible. If that should not be the case, we evaluate on different datasets we discovered (see section data sources) ✅
* 04.02. Second "official" feedback round with supervisor ✅
* 05.02. Summary of results ✅ 
* 25.02. Second milestone: Code and presentation ✅
* 15.03. Third milestone: Report deadline 

### High Level Architecture Description 

For this project, we implemented a pipeline library, which is contained in [src.pipelinelib](src/pipelinelib).

The phase of importing and aggregating data is performed by two different classes:

* [Parser](src/pipelinelib/querying.py#L16):
Loads the provided transcripts into a [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), alongside the corresponding metadata, such as whether a transcript belongs to the depressive sample set.

* [Queryable](src/pipelinelib/querying.py#L192)
Provides a type-safe wrapper around queries that can be applied to the loaded DataFrame.
Its capabilities include being able to aggregate the data on [differing corpus levels](src/pipelinelib/text_body.py), loading only transcript data from the depressed group, etc.

Due to GDPR, the transcripts are not allowed to be uploaded to this repository.


Classifying data from the loaded documents is implemented by three more classes:

* [Pipeline](src/pipelinelib/pipeline.py#L13):
Represents a pipeline for training a classification model.

* [Component](src/pipelinelib/component.py#L13): 
Every step in a pipeline, be it preprocessing, feature extraction or classification, is implemented as a derivative of this abstract class.

* [Extension](src/pipelinelib/extension.py#L6): 
The results of a Component instance are stored within a lookup structure for later Components to reuse.
Each Extension is mapped to one result within said structure.

The aforementioned classes all work in conjunction to deliver the requested results.
Particularly, each Component declares in its constructor which Extensions it depends on.
This allows a Pipeline instance, prior to execution, to check whether a Component's dependencies are satisfied, or whether they will overwrite other calculated results.


The pipeline's steps for the [Sigmund](src/sigmund) project are implemented as Component derivates and can be split into 3 different parts:

* [Preprocessing](src/sigmund/preprocessing): 
As our features require different representations of the corpus, we provide a modular preprocessing pipeline. 
For that purpose, different aspects of the text can be queried, ranging from plain tokenizing and syllable extraction, to stemming and lemmatization. 

* [Feature Engineering](src/sigmund/features): 
Features can be added in a modular fashion as well.
Implemented features include Agreement Score, Talk Turn and TFIDF.
Their inputs depend on applied preprocessing Components.

* [Classification](src/sigmund/classification):
Lastly, we use a classification model in order to categorize the transcripts as depressed or non-depressed.
This is performed by combining select feature vectors from the aforementioned section and reporting a loss value.

The structure of the repository is as follow:

```
├── pipelinelib
│   ├── adapter.py
│   ├── component.py
│   ├── extension.py
│   ├── __init__.py
│   ├── pipeline.py
│   ├── querying.py
│   └── text_body.py
├── sigmund
│   ├── classification
│   │   ├── __init__.py
│   │   ├── linear_discriminant_analysis.py
│   │   ├── logistic_regression.py
│   │   ├── merger.py
│   │   ├── naive_bayes.py
│   │   └── pca.py
│   ├── extensions.py
│   ├── features
│   │   ├── agreement_score.py
│   │   ├── basic_statistics.py
│   │   ├── flesch_reading_ease.py
│   │   ├── __init__.py
│   │   ├── liwc.py
│   │   ├── pos.py
│   │   ├── talk_turn.py
│   │   ├── tfidf.py
│   │   └── vocabulary_size.py
│   ├── __init__.py
│   └── preprocessing
│       ├── __init__.py
│       └── words.py
```

We furthermore provide a simple front-end for the Institute of Medical Psychology to present the results and provide feature details. 

## Data Analysis

### Data Sources

* 10 transcripts of conversations between couples as part of the "Enhancing Social Interaction in Depression" (SIDE) [study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173246/).
* The structure of the entire dataset of the SIDE study is described in detail in the project proposal, which can be found in the repository as well.
* The format of the transcripts is as follows:
    * Docx format
    * Sequence of Speakers, separated by paragraph, starting with speaker label
    * Annotations of the transcriber are defined using parenthesis 
    * In our case, the depressive person is always female, however this is not necessary

### Basic Statistics

As of 16.12.2020, our data consists of:

* 10 transcripts (10 couples, 20 speakers; 5 pairs with depression, 5 pairs without depression; depressed partner always female)
* Word count: ~1000 words per transcript
* Word count total: ~13.000
* Utterances: ~60 per transcript

More detailed statistics of the transcripts are included in [feature_summary.ipynb](feature_summary.ipynb).

