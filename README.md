# Sigmund - Readme

Project Members
* Julius Daub (3536557) | Applied Informatics (M.Sc.)julius@daubweb.com
* Alexander Haas (3503540) | Applied Informatics (M.Sc.)haas.alexanderjulian@gmail.com
* Ubeydullah Ilhan (3447661) | Applied Informatics (M.Sc.)ubeydullah.il@gmail.com
* Benjamin Sparks (3664690) | Applied Informatics (M.Sc.)benjamin.sparks@protonmail.comGitHub:https://github.com/gitexa/sigmund

## Existing Code Fragments
 

## Utilized Libraries
* Spacy
* German News Dataset
* Streamlit
* Pandas
* Seaborn for visualisation
* NLTK Stemmer
* SentiWS
* LIWC library with german dictionary (https://pypi.org/project/liwc/)



### Contributions
- DocX Extraction
- Experiments with Speech To Text
- General Architecture
- Feature Engineering

## Project State


### Future Planning
* Currently: Feature engineering
### Further Information
Currently it is not sure if we can do our analysis on the data provided. We were assured to get 150 transcripts, now 10 are in the making. Those transcripts are quite short.

We consider restructuring our project using a different dataset (reddit mental health dataset) and hope to be able to apply our findings to the transcripts we got from the institute. If we don't get any response from the institute we cannot progress further.

### High-Level Architecture:
* Pre-Processing
* Feature-Engineering
    * Share of Speech
    * Talking Turns
    * Flesch reading-ease score
    * Aggreement-Score
    * Diminished cooperativeness
    * (Responding to negative-positive emotions) 

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
* Extracting Text-Data from docx
* Annotate Ground-Truth (Hamilton-Score, Depressive / non depressive)
* Removal of "annotations" like "(spricht unverständlich")
* Splitting into sentences
* Splitting into "utterances"
* Removal of Stop-Words
* Lemmatisation using German
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

### Content Features
## Current code State
* 16.12.2020: 10 Transkripte
* Total: Ca 13.000 Wörter
* 20 Sprecher
* Davon 5 D
* 5 weiblich, ND
* 10 Männlich, ND
* 



