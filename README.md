# Sigmund
Analysis of **depression features** in text-transcripts of **couple conversations**. 

We examine transcripts of couple conversations from a current [research project](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173246/) at [Heidelberg University Hospital](https://www.klinikum.uni-heidelberg.de/zentrum-fuer-psychosoziale-medizin-zpm/institut-fuer-medizinische-psychologie) to identify depression-related features and quantify the differences between couples, in which one partner is suffering from major depressive disorder, and couples, in which both partners lack a history of depression. 

### Project Members
* Julius Daub (3536557) | Applied Informatics (M.Sc.) | julius@daubweb.com
* Alexander Haas (3503540) | Applied Informatics (M.Sc.) | haas.alexanderjulian@gmail.com
* Ubeydullah Ilhan (3447661) | Applied Informatics (M.Sc.) | ubeydullah.il@gmail.com
* Benjamin Sparks (3664690) | Applied Informatics (M.Sc.) | benjamin.sparks@protonmail.com

### Contributions for the Milestone
* Data acquisition: speech-to-text and docx-extraction
* Implementation of the pipeline architecture
* Implementation of a demo text mining workflow
* Implementation of demo components
* [Collection of possible metrics](https://docs.google.com/spreadsheets/d/1z2vkU259P_5mGQCHb67HgyoEulPsd03LQv2z-SoTG4g/edit?usp=sharing) from literature 
* Basic statistics of the data set and first iteration of feature engineering (got the data on 15.12.2020)

### Existing Code Fragments
We do not use existing code fragments so far.

### Utilized Libraries
We use several libraries for the project, including:
* Spacy
* sklearn
* NLTK Stemmer
* SentiWS
* Pandas
* Stanford Parser Toolkit 
* Streamlit to build a simple front-end for the insitute
* Seaborn for visualisation
* LIWC library with german dictionary (https://pypi.org/project/liwc/)
* German news dataset (https://spacy.io/models/de)

The libraries and versions are specified in the Pipfile.

## Project State

### Planning State
We planned to get 150 transcripts of couple conversations from the Institute of Medical Psychology. As the audio quality was quite bad, they could not use automatic speech-to-text programs and had to create the transcripts by hand. We therefore spent a considerable amount of time in trying to get transcripts from the audio files, which turned out to be not possible with the given signal. We finally received 10 transcripts from the Institute of Medical Psychology on 15.12.2020, which is only a small subset of the 150 available conversations. In the case of promising results, they offered to provide resources for transcribing further conversations. Upon receiving the transcripts, we started with some basic statistics which are included in the *data_description.ipynb*. We additionally collected different feature engineering approaches from literature which are summarized [here](https://docs.google.com/spreadsheets/d/1z2vkU259P_5mGQCHb67HgyoEulPsd03LQv2z-SoTG4g/edit?usp=sharing). Furthermore, we set up a modular text analytics pipeline and implemented first components for every step of the pipeline.

### Future Planning
The next steps are to implement the features from our research and evaluate how well they are able to detect depression patterns and how well they serve towards classifying transcripts. 
* 15.01. Implementation of all features and first results to share with the Institute, to evaluate if further transcripts are possible. If that should not be the case, we evaluate on different datasets we discovered (see section data sources).
* 05.02. Summary of results
* 08.02. Second "official" feedback round with supervisor
* 25.02. Second milestone: Code and presentation due
* 15.03. Third milestone: Report deadline 

### High Level Architecture Description 
To structure our pipeline, we built a pipline library on the basis of Spacy's pipeline which is contained in pipelinelib. It is defined by three parts: 
* Component: every processing step of the pipeline is implemented as a component, inheriting from the abstract class Component. 
* Extension: a class to represent an extension member of Spacy's DocInstance type
* Pipeline: a class to represent a pipeline for training a model

The pipeline itself contains 3 main parts, **preprocessing**, **features** and **classification**.
* Data import: as the transcripts are not allowed on github-servers, we provide a config.json where the local path to the transcripts needs to be specified. We then use the path to create a document corpus with the raw text. 
* Preprocessing: as our features require different representations of the corpus, we provide a modular preprocessing pipeline. For that purpose, different layers of the text can be queried, ranging from character, to syllable, to word, to sentences, to paragraph, to document, to corpus. The preprocessing steps can be applied to the different layers. 
* Feature Engineering: features can be added in a modular fashion as well. 
* Classification: we finally use a classification model in order to classify the transcripts as depressed or non-depressed using the feature vector and report a loss value. The models can also be specified as components in the corresponding subfolder.

The structure of the repository is as follow:

```
├── pipelinelib
│   ├── component.py
│   ├── extension.py
│   ├── pipeline.py
├── sigmund
│   ├── classification
│   │   └── qda.py
│   ├── features
│   │   ├── pos.py
│   │   └── words.py
│   └── preprocessing
│       ├── syllables.py
│       └── words.py
└── utils
    ├── corpus_builder.py
    ├── corpus_manager.py
    ├── dialogue_parser.py
    ├── feature_annotator.py
    └── statistics.py
```

We furthermore provide a simple front-end for the Institute for Medical Psychology to present the results and provide feature details. 

## Data Analysis
* As our dataset was not available due to the date of the milestone, it was hard to do some Data-Analysis.

### Data Sources
* 10 Transcripts of couple conversations as part of the "Enhancing Social Interaction in Depression" (SIDE) [study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6173246/).
* The structure of the entire dataset of the SIDE study is described in detail in the project proposal, which can be found in the repository as well
* The format of the transcripts is as follows:
    * Docx format
    * Sequence of Speakers, separated by paragraph, starting with speaker label
    * Annotations of the transcriber are defined using parenthesis 
    * In Our case, the depressive person is always female, however this is not necessary
    * No further MetaData   

### Basic Statistic
At 16.12.2020 we had:
* 10 transcripts (10 couples, 20 speakers; 5 pairs with depression, 5 pairs without depression; depressed partner always female)
* word count ~ 1000 words per transcript
* word count total ~ 13.000
* utterances ~ 60 per transcript
More detailed statics of the transcripts are included in the data_description.ipynb.

### Pre-Processing

* Extracting Text-Data from docx
* Annotate Ground-Truth (Hamilton-Score, Depressive / non depressive)
* Removal of "annotations" like "(spricht unverständlich")
* Splitting into sentences
* Splitting into "utterances"
* Removal of Stop-Words
* Lemmatisation using German

### Feature Engineering
Our aim is to find features that allow to discriminate between text associated with depression and text not associated with depression. An overview with features derived from literature and own thoughts can be found [here](https://docs.google.com/spreadsheets/d/1z2vkU259P_5mGQCHb67HgyoEulPsd03LQv2z-SoTG4g/edit?usp=sharing) We started the implementation with the following features.

#### Structural Features

##### Complexity of speech | Flesch reading-ease score
Person suffering from MDD tend to structure their sentences with less complexity. Therefore the complexity of speech is an important feature to extract from the dialogs. For that, the Flesch-Reading-Ease needs to be calculated for each person. The score for the german language is calculated with the following formula,
where higher values indicate less complex speech:

<img src="https://render.githubusercontent.com/render/math?math=\text{FRE}_\text{german} = 180 - \frac{\text{total words}}{\text{total sentences}} - (58.5 \times \frac{\text{total syllables}}{\text{total words}})">

##### Talking Turns
To represent the talking turns each paragraph for each person is count together. 
The ratio of both numbers describes the dialog distribution. Is the ratio closer to 1, the dialog is
distributed more evenly between the partners.
However shorter sentences indicating only an agreement or disagreement are not counted in, as they are 
not really talking over the speech.

##### Agreement-Score
The Agreement-Score shows how often the partners agree oder disagree to each other. 
This feature is extracted by analizing the words in the first sentence of a paragraph. 
If the words show disagreement like in: "nein, trotzdem, aber" ; the paragraph is counted as 1 disagreement. 
At the end, the ratio of "Number of disagreements" to "Number of all paragraphs" is calculated.

#### Content Features

##### Part-of-Speech
Assigns a tag to each token (e.g. noun, adjective, ...). We use Spacy's POS feature.

##### Term-frequency Inverse-document-frequency (TFIDF)
To find the most important words, we use TFIDF.



