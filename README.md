# haystacks.ai-anomaly-detection
Anomaly detection project with haystacks.ai

Project developed by Zach Stone and Dmitriy Popov-Velasco together with haystacks.ai

Some highlights of the project:

* Our main task was to use alternative data to identify and characterize anomalies across a statewide single-family rental housing market.  
* Pipelines to bring in alternative data were developed, specifically concerning schools, entertainment and attractions, upscale indicators (e.g. Whole Foods, wine bars, ...), walkability and transit, crime, and natural risk to property. 
* Alternative data has the benefit of being freely available, but can be noisy or course-grained in many cases.  Useful signals from this data were extracted using feature selection and engineering, scaling, principal component analysis, and other methods
* Traditional anomaly detection methods such as isolation forests and agglomerative clustering were tested first. Without further modifications, these methods detected statewide outliers, but failed to detect anomalies in many areas.  In addition, traditional anomaly detection methods were overly swayed by outliers in the estimated average loss ratio.
* To address these issues, a localized agglomerative clustering technique was developed.  The key idea was to break up the dataset into overlapping 'rectangular' geographic regions by lat/long, run the agglomerative clustering on the principal components, and combine clusters containing a house in common until clusters are disjoint.  This technique allows clusters with larger intra-cluster variance than the global agglomerative clustering model, while still isolating the same number of listings.  It was able to detect anomalies statewide and is less swayed by outliers.
*The localized agglomerative clustering technique allowed us to detect anomalies that are over $100K cheaper than houses in nearby clusters.  The global anomaly detection methods focused on statewide outliers, concentrating on properties that are likely to be of little interest to investors in single-family rental properties.

A complete writeup can be found at https://nycdatascience.com/blog/student-works/detecting-anomalies-in-a-statewide-housing-market-with-alternative-data/

This repo contains all the files used in the project, except for webscrapers (both those written by our team and haystacks.ai), the notebook creating the dataset, the raw data files, and initial EDA, as these used or generated information which is proprietary to haystacks.ai. 

Summary of scripts:
- pca_analyzer contains code for the PcaAnalyzer class which was used to create and analyze principal components
- geography_helper contains functions which were used to import and merge geographic data, create geographically based features, and visualize geographic information
- mapper_clusterer contains the original 'local' agglomerative clustering algorithm introduced in this research
- anomaly_analyzer defines the AnomalyAnalyzer class which was used for identifying anomalies, plotting them geographically and in feature space, extracting original listing information, and comparing anomalies to other clusters to evaluate them
- data_cluster_bundle allows the AnomalyAnalyzer to be used with the output of any clustering algorithm, as the original was designed with the local clustering algorithm in mind
- haystacks_importer is for extracting information from the results of Google Maps API calls
- GAboundary.txt contains the coordinates plotting the shape of GA, used regularly in visualization, and for filtering data by location

Summary of notebooks:
- 02 impute_and_PCA contains code for imputing, scaling, and PCAing the raw features, analyzing the PCs and their loadings, and visualizing the dataset both geographically and in feature space. Also includes basic application of k-means clustering for EDA purposes.
- 03 AC Local is the main notebook where (global/local w/ and w/o LL) AC and isolation forests are run. Includes visualizations as well.
- 03 GMM Local includes some tests with GMMs
- 04 Anomaly Analysis contains visualization of anomalies detected by all models, summary statistics about the anomalies, and case-studies of certain anomalies
