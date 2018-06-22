# proj1

Outstanding questions:

I don't understand the warning below.  My f-score has no predicted samples? Why is that?
\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.

I feel like my optimized model isn't very optimized (outputs below).  Is seeing such a small improvement common after tuning? Or did I simply tune poorly? How can I identify the relevant areas of sensitivity?

Unoptimized model
------
Accuracy score on testing data: 0.8576
F-score on testing data: 0.7246

Optimized Model
------
Final accuracy score on the testing data: 0.8600
Final F-score on the testing data: 0.7303
