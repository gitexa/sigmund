# Sigmund - Readme
Detection of depression features in transcripts of couple conversations. 

We examine transcripts of couple conversations from a current research project at Heidelberg University Hospital to identify depression-related features and quantify the
differences between couples, in which one partner is suffering from MDD, and couples, in which both partners lack a history of depression. 

### Project Members
* Julius Daub (3536557) | Applied Informatics (M.Sc.) | julius@daubweb.com
* Alexander Haas (3503540) | Applied Informatics (M.Sc.) | haas.alexanderjulian@gmail.com
* Ubeydullah Ilhan (3447661) | Applied Informatics (M.Sc.) | ubeydullah.il@gmail.com
* Benjamin Sparks (3664690) | Applied Informatics (M.Sc.) | benjamin.sparks@protonmail.comGitHub:https://github.com/gitexa/sigmund


### Contributions
- DocX Extraction
- Experiments with Speech To Text
- General Architecture
- Feature Engineering

### Existing Code Fragments

### Utilized Libraries
* Spacy
* German News Dataset
* Streamlit
* Pandas
* Seaborn for visualisation
* NLTK Stemmer
* SentiWS
* LIWC library with german dictionary (https://pypi.org/project/liwc/)
* Stanford Parser Toolkit 
* sklearn


## Project State

### Planning State
We planned to get 150 transcripts of couple conversations from the Institute of Medical Psychology. As the audio quality was quite bad, they could not use automatic speech-to-text programs and had to create the transcripts by hand. We therefore spent a considerable amount of time in trying to get transcripts from the audio files, which turned out to be not possible with the given signal. We finally received 10 transcripts from the Institute for Medical Psychology on 15.12.2020, which is only a small subset of the 150 available conversations. In the case of promising results, they offered to provide resources for transcribing further conversations. Upon receiving the transcripts, we started with some basic statistics which are included in the data_description.ipynb. We additionally collected different feature engineering approaches from literature which are provided in the file metrics.csv. Furthermore, we set up a modular text analytics pipeline and implemented first components for every step of the pipeline.

### Future Planning
The next steps are to implement the features from our research and evaluate how well they are able to detect depression patterns and how well they serve towards classifying transcripts. 
- 15.01. Implementation of all features and first results to share with the Institute, to evaluate if further transcripts are possible. If that should not be the case, we evaluate on different datasets we discovered (see section data sources).
- 05.02. Summary of results
- 08.02. Second "official" feedback round with supervisor
- 25.02. Second milestone: Code and presentation due
- 15.03. Third milestone: Report deadline 

### High Level Architecture Description 
Our pipeline consists of 4 parts, which are connected using spacy. For every part of the pipeline, components can be designed manually and added in a modular fashion.
* Data import: as the transcripts are not allowed on github-servers, we provide a config.json where the local path to the transcripts needs to be specified. We then use the path to create a document corpus with the raw text. 
* Preprocessing: as our features require different representations of the corpus, we provide a modular preprocessing pipeline. For that purpose, different layers of the text can be queried, ranging from character, to syllable, to word, to sentences, to paragraph, to document, to corpus. The preprocessing steps can be applied to the different layers, currently implemented are:
    * Extracting Text-Data from docx
    * Annotate Ground-Truth (Hamilton-Score, Depressive / non depressive)
    * Removal of "annotations" like "(spricht unverständlich")
    * Splitting into sentences
    * Splitting into "utterances"
    * Removal of Stop-Words
    * Lemmatisation using German
* Feature Engineering: features can be added in a modular fashion as well. Currently implemented are:
    * Share of Speech
    * Talking Turns
    * Flesch reading-ease score
    * Aggreement-Score
    * Diminished cooperativeness
    * (Responding to negative-positive emotions) 
* Classification: we finally use a classification model in order to classify the transcripts as depressed or non-depressed using the feature vector and report a loss value. The models can also be specified as components in the corresponding subfolder.

We furthermore provide a simple front-end for the Institute for Medical Psychology to present the results and provide feature details. 

## Data Analysis
* As our dataset was not available due to the date of the milestone, it was hard to do some Data-Analysis.

### Data Sources
* (German) Transcripts of MDD-Pairs
* Assumption
    * Docx format
    * Sequence of Speakers, separated by paragraph, starting with speaker label
    * Annotations are made clear using parenthesis (Those should not be evaluated)
    * In Our case, the depressive person is always female, however this is not necessary
    * No further MetaData   
 

### Pre-Processing

## Basic Statistic
At the time of writing we had:
* Two Transcripts
* 2000 Words
* Unclear Specification
* ~ 120 Utterances (60 per Script)
## Feature Engineering
Our aim is to find features that allow to discriminate between text associated with depression and text not associated with depression.
### Structural Features
#### Complexity of speech
Person suffering from MDD tend to structure their sentences with less complexity. Therefore the complexity of speech is an important feature to extract from the dialogs. For that, the Flesch-Reading-Ease needs to be calculated for each person. The score for the german language is calculated with the following formula,
where higher values indicate less complex speech:

<img src="https://render.githubusercontent.com/render/math?math=\text{FRE}_\text{german} = 180 - \frac{\text{total words}}{\text{total sentences}} - (58.5 \times \frac{\text{total syllables}}{\text{total words}})">

### Content Features
## Current code State
* 16.12.2020: 10 Transkripte
* Total: Ca 13.000 Wörter
* 20 Sprecher
* Davon 5 D
* 5 weiblich, ND
* 10 Männlich, ND
* 



