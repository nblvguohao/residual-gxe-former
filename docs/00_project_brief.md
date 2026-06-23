# Project Brief

## Working title

**ResidualGxE-Former: Mixed-model residual learning with weather time-series for genotype-by-environment phenotype prediction under unseen environments**

## Research question

Can a model that decomposes stable genotype/environment main effects and learns nonlinear G×E residuals improve phenotype prediction under unseen environments and future years?

## Single-crop entry point

This project starts with one crop. The recommended main dataset is G2F maize because it provides multi-year, multi-location genotype, phenotype, weather, soil, and environmental metadata. FIP1 wheat can be used as external validation.

## Core hypothesis

Direct phenotype prediction is suboptimal under environmental shift. A structured model that separates main effects from nonlinear G×E residuals will better generalize to unseen environments and future years.

## Expected contributions

1. A leakage-controlled G×E benchmark pipeline.
2. Strong baseline comparisons against genomic selection and machine-learning models.
3. A residual-learning architecture for environment extrapolation.
4. Weather time-series modeling with growth-stage-aware encoding.
5. Selection utility evaluation for breeding decisions.
