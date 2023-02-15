Scripts for running Poisson factorization Topic Modeling for creating initial (neutral) topic estimates per: Vafa, K., Naidu, S., & Blei, D. (2020, July). Text-Based Ideal Points. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 5345-5357). 

We want to use MALLET (LDA) topic modeling to initialize the topics for ideal point estimation. Poisson factorization is run ten times with different random seeds to get an expected mean value in order to _scale_ MALLET topic modeling output: this scaling is needed in order to use MALLET topic modeling results as input to text-based ideal point estimation. 

These scripts are part of the overall process described in `../README.md`. 