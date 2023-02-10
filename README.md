### Problem Statement
Can reddit tell us what makes a question stupid? Using data from r/NoStupidQuestions and r/AskReddit I seek to answer this question. (Under the assumption that people going to r/NoStupidQuestions believe the question they are asking is stupid). Applying various classification models to garner best results I will attempt to pull out the features of such questions and report them here. 

### Methodology
- Combined text and titles to increase data population
- Lemmatized the text and used TFIDF vectorization for creating features
- Limited the document frequency to ensure features smaller than sample size
- Compared the output of several models to select best performing
- Conducted a hyperparameter grid search for each model option
- Then included review length (post lemmatization) for best model performance

### Model Performance

After grid searches Random Forest yielded the best results. All models were prone to overfitting, but in the end only Random Forest was able to deal with it (in particular thanks to the parameter `min_sample_leaf`). Shown in the table below are the train/test accuracies of the best performing parameters for each of the four model options. The only exception here is that the Random Forest model was updated to include the number of unique words in a post, as that proved to be a distinguishable trait amongst the subreddits. I didn't include it in the grid searches for all models because it vastly increased computation time.

|Model| Train Accuracy |Test Accuracy|
|---|---|---|
|kNN |0.994 |0.531|

|Naive Bayes |0.843 |0.741|

|Logistic Regression |0.888 |0.762|

|Random Forest |0.784 |0.767|

### Results
Words that were the strongest predictors of a stupid question:
##### reddit, good, thing, mean, say, really, use, favorite, work, want, think, try, movie, lot, time, wonder, life, look, able

##### Features of a stipid question:
- Too long; keep it concise
- Self-referential 
- Positivity/negativity
- Uncertainty
- Opinion-based

#### Some successes and failures (?)
Correctly predicted stupid question:
##### "Why do people watch Mukbangs?     I seriously get nauseous just looking at the thumbnails for these videos. And from what I know, the videos themselves are super messy. I just don't see what type of entertainment people get out of these videos."
Correctly predicted smart question:
##### "What's the funniest way to die?"
Incorrectly not labeled as a stupid question:
##### “When cheating happens, is it the fault of the married man or the mistress?”
Incorrectly labeled as a stupid question:
##### “What is your view on the male social hierarchy? (e.g, Sigma, beta and alpha male).”

Also included in this repo is an interactive streamlit app where you can ask a question and find out if it is a stupid one or not!