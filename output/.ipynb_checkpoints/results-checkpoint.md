## kNN Results
```python
pipe_knn = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])
params = {
    'knn__n_neighbors': [5, 20, 50, 100],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]
}
{'knn__n_neighbors': 5, 'knn__p': 2, 'knn__weights': 'distance'}
Wall time: 5min 45s
(0.9967112672883723, 0.5622381402892282)


pipe_knn = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(2,2))),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])

params = {
    'knn__n_neighbors': [5, 20, 50, 100],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]
}
{'knn__n_neighbors': 5, 'knn__p': 2, 'knn__weights': 'distance'}
Wall time: 5min 26s
(0.8147497409559851, 0.5331801594810109)


pipe_knn = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(1,2))),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])
params = {
    'knn__n_neighbors': [5, 20, 50, 100],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]
}
{'knn__n_neighbors': 5, 'knn__p': 2, 'knn__weights': 'distance'}
Wall time: 6min 17s

(0.9947290174347885, 0.5408839032301662)



pipe_knn = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])

params = {
    'knn__n_neighbors': [10, 20, 50, 100],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2, 3]
}
{'knn__n_neighbors': 5, 'knn__p': 2, 'knn__weights': 'distance'}
Wall time: 7min 20s
(0.9950443753660404, 0.5469658061900257)
```



## Naive Bayes Results
```python
pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
    ('nb', MultinomialNB())
])
params = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'nb__alpha': np.linspace(1e-10,1,100)
}
{'nb__alpha': 0.0707070708, 'tfidf__ngram_range': (1, 2)}
Wall time: 11min 26s
(0.9202144433932513, 0.7537505068252467)



pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(2,2))),
    ('nb', MultinomialNB())
])

params = {
    'tfidf__ngram_range': [(1,1), (1,2), (2,2)],
    'nb__alpha': np.linspace(1e-10,1,100)
}
{'nb__alpha': 0.050505050600000004, 'tfidf__ngram_range': (1, 2)}
Wall time: 9min 34s
(0.8718295265125918, 0.7460467630760914)



pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3)),
    ('nb', MultinomialNB())
])
params = {
    'tfidf__ngram_range': [(1,1), (2,2)],
    'nb__alpha': np.linspace(1e-10,1,100)
}
{'nb__alpha': 1e-10, 'tfidf__ngram_range': (1, 1)}
Wall time: 5min 26s
(0.8171374510068928, 0.738883632923368)



pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3)),
    ('nb', MultinomialNB())
])
params = {
    'tfidf__ngram_range': [(1,2)],
    'nb__alpha': np.linspace(1e-10,1,1000)
}
{'nb__alpha': 0.05105105114594595, 'tfidf__ngram_range': (1, 2)}
Wall time: 42min 7s
(0.8717844753795558, 0.7460467630760914)



pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB())
])
params = {
    'tfidf__min_df': [4, 7, 10],
    'tfidf__max_df': [1, .1,.09],
    'nb__alpha': np.linspace(1e-2,1,50)
}
{'nb__alpha': 0.05040816326530612, 'tfidf__max_df': 0.1, 'tfidf__min_df': 4}
Wall time: 16min 18s
(0.8428165968374105, 0.7410460873090958)
```



## Random Forest Results
```python
pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'rf__n_estimators': [10, 50, 100, 200],
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [10, 100, 250, 500, 1000],
    'rf__min_samples_split': [2, 10, 50]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_split': 2, 'rf__n_estimators': 200}
Wall time: 1h 33min 42s #caveat: closed computer for a while
(0.9815740865882777, 0.7860521692120557)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(2,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'rf__n_estimators': [10, 50, 100, 200],
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_depth': [10, 100, 250, 500, 1000],
    'rf__min_samples_split': [2, 10, 50]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 1000, 'rf__min_samples_split': 50, 'rf__n_estimators': 50}
Wall time: 23min 8s
(0.7783934765959364, 0.6717123935666982)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(2,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__criterion': ['gini', 'entropy', 'log_loss'],
    'rf__max_depth': [500, 1000, 1500, 2000],
    'rf__min_samples_split': [25, 50, 100]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 2000, 'rf__min_samples_split': 50, 'rf__n_estimators': 200}
Wall time: 1h 22min 51s
(0.8377258188043429, 0.6771185295310177)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3)),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'rf__n_estimators': [200, 300, 500],
    'rf__criterion': ['entropy'],
    'rf__max_depth': [500, 625, 750],
    'rf__min_samples_split': [2,10, 15, 50]
}
gs_rf = GridSearchCV(pipe_rf, params, n_jobs=-1, verbose=1)
gs_rf.fit(X_train, y_train)
print(gs_rf.best_params_)
gs_rf.score(X_train, y_train), gs_rf.score(X_test, y_test)
Fitting 5 folds for each of 72 candidates, totalling 360 fits
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_split': 10, 'rf__n_estimators': 300, 'tfidf__ngram_range': (1, 1)}
CPU times: user 1min 32s, sys: 2.45 s, total: 1min 35s
Wall time: 2h 4min 6s
(0.9707618146596387, 0.7848357886200837)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [1, .1],
    'rf__n_estimators': [200],
    'rf__criterion': ['entropy'],
    'rf__max_depth': [400,500],
    'rf__min_samples_leaf': [3, 5, 10]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_leaf': 3, 'rf__n_estimators': 200, 'tfidf__max_df': 0.1}
Wall time: 2min 28s
(0.8434923638329505, 0.7682119205298014)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.1],
    'rf__n_estimators': [200],
    'rf__criterion': ['entropy'],
    'rf__max_depth': [500],
    'rf__min_samples_leaf': [4]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_leaf': 4, 'rf__n_estimators': 200, 'tfidf__max_df': 0.1}
Wall time: 34.2 s
(0.8302022795873316, 0.7630760913636978)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.1],
    'rf__n_estimators': [200],
    'rf__criterion': ['entropy'],
    'rf__max_depth': [500],
    'rf__min_samples_leaf': [5]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_leaf': 5, 'rf__n_estimators': 200, 'tfidf__max_df': 0.1}
Wall time: 28.1 s
(0.8106050367166734, 0.7571293418029463)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.1],
    'rf__n_estimators': [200],
    'rf__criterion': ['entropy'],
    'rf__max_depth': [500, 600],
    'rf__min_samples_leaf': [6]
}
{'rf__criterion': 'entropy', 'rf__max_depth': 500, 'rf__min_samples_leaf': 6, 'rf__n_estimators': 200, 'tfidf__max_df': 0.1}
Wall time: 42.7 s
(0.7974501058701626, 0.7548317340181105)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.1],
    'rf__n_estimators': [200,300],
    'rf__criterion': ['gini','entropy'],
    'rf__max_depth': [500],
    'rf__min_samples_leaf': [7]
}
{'rf__criterion': 'gini', 'rf__max_depth': 500, 'rf__min_samples_leaf': 7, 'rf__n_estimators': 200, 'tfidf__max_df': 0.1}
Wall time: 1min 20s
(0.7808262377798801, 0.7537505068252467)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.1,.09],
    'rf__n_estimators': [200],
    'rf__criterion': ['gini'],
    'rf__max_depth': [500],
    'rf__min_samples_leaf': [6]
}
{'rf__criterion': 'gini', 'rf__max_depth': 500, 'rf__min_samples_leaf': 6, 'rf__n_estimators': 200, 'tfidf__max_df': 0.09}
Wall time: 38.8 s
(0.7903770779835113, 0.7567238816056224)



pipe_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=4, ngram_range=(1,2))),
    ('rf', RandomForestClassifier(random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__max_df': [.09],
    'rf__n_estimators': [200],
    'rf__criterion': ['gini'],
    'rf__max_depth': [500],
    'rf__min_samples_leaf': [7]
}
{'rf__criterion': 'gini', 'rf__max_depth': 500, 'rf__min_samples_leaf': 7, 'rf__n_estimators': 200, 'tfidf__max_df': 0.09}
Wall time: 22.3 s
(0.7807811866468441, 0.7548317340181105)
```



## Logistic Regression Results
```python
pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'logreg__C': [.0001, .01, .1, 10],
    'logreg__penalty': ['l1', 'l2']
}
{'logreg__C': 10, 'logreg__penalty': 'l2'}
Wall time: 19min 31s
(0.8930035590395099, 0.7621300175699419)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(2,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'logreg__C': [.0001, .01, .1, 10],
    'logreg__penalty': ['l1', 'l2']
}
{'logreg__C': 10, 'logreg__penalty': 'l2'}
Wall time: 18min 35s
(0.8394377618597108, 0.6827949722935532)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])# saga solver so can use l1, l2, and elasticnet
params = {
    'logreg__C': [.1, 10, 100],
    'logreg__penalty': ['l1', 'l2']
}
{'logreg__C': 10, 'logreg__penalty': 'l2'}
Wall time: 1h 35min 41s
(0.9364328512862099, 0.7629409379645898)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])# saga solver so can use l1, l2, and elasticnet
params = {
    'logreg__C': np.linspace(.1, 100, 10),
    'logreg__penalty': ['l2']
}
{'logreg__C': 11.200000000000001, 'logreg__penalty': 'l2'}
Wall time: 1min
(0.9396314817317656, 0.7632112447628058)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])# saga solver so can use l1, l2, and elasticnet
params = {
    'tfidf__min_df': [4],
    'tfidf__max_df': [1, .1,.09],
    'logreg__C': np.linspace(.01, 100, 20),
    'logreg__penalty': ['l2']
}
{'logreg__C': 5.272631578947368, 'logreg__penalty': 'l2', 'tfidf__max_df': 0.1, 'tfidf__min_df': 4}
Wall time: 7min 51s
(0.8979591836734694, 0.76023786998243)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__min_df': [4],
    'tfidf__max_df': [.1],
    'logreg__C': np.linspace(.1, 10, 20),
    'logreg__penalty': ['l2']
}
{'logreg__C': 3.747368421052632, 'logreg__penalty': 'l2', 'tfidf__max_df': 0.1, 'tfidf__min_df': 4}
Wall time: 1min 18s
(0.8876875253412624, 0.7622651709690499)



pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('logreg', LogisticRegression(max_iter=1e4, solver='saga', random_state=5, n_jobs=-1, warm_start=True))
])
params = {
    'tfidf__min_df': [4],
    'tfidf__max_df': [.1],
    'logreg__C': np.linspace(1, 5, 40),
    'logreg__penalty': ['l2']
}
{'logreg__C': 4.487179487179487, 'logreg__penalty': 'l2', 'tfidf__max_df': 0.1, 'tfidf__min_df': 4}
Wall time: 2min 30s
(0.8935441726359418, 0.76091363697797)
```