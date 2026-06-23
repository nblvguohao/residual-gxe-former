# FIP1 External Validation Notes

Use FIP1 after the main G2F pipeline is stable.

## Known useful structure

FIP1 is a winter wheat benchmark with genotype, environment, phenotype, and image/time-series resources. For this project, prioritize:

- genotype markers;
- environment/weather features;
- official train/validation/test splits;
- traits: yield_adjusted, protein, heading_date, plant_height.

## Existing internal benchmark context

The current project report describes FIP1 as having:

- 346 elite wheat lines;
- 7 site-year environments;
- 18,845 biallelic SNP markers;
- target traits: yield_adjusted, protein, heading_date, plant_height;
- official benchmark splits including Test P, Test G, Test E, and Test GE.

## Validation goal

Do not overclaim cross-crop generalization in this paper. Use FIP1 as external evidence that the proposed residual G×E learning design is not hard-coded to the G2F maize pipeline.

## Expected comparison

Compare against:

- BayesRidge;
- GBLUP or kernel ridge;
- existing GxETransformer;
- existing GxECrossAttn;
- ResidualGxE-Former.

Main emphasis:

- Test E;
- Test GE;
- SelectionGain@10% if applicable.
