# Human Abstention Sensitivity Analysis

- Dataset: 60 matched EN-AZ pairs, evaluated by 3 models.
- Total model outputs: 360.
- Resolved binary labels: 335.
- Unresolved labels: 25 (20 AZ, 5 EN).
- Initial inter-reviewer agreement: 52.78%; Cohen's kappa: 0.284.

HUMAN_ABSTENTION_BOUNDS_25_UNRESOLVED.csv:
Lower and upper abstention rates, treating unresolved labels
as no and yes, respectively.

PAIRED_MCNEMAR_SENSITIVITY.csv:
Two-sided exact paired McNemar p-value ranges across
possible binary completions of unresolved labels.

Limitations:
Reviewer independence and original judgment provenance
have not been independently verified. Results are exploratory
and conditional on annotation validity.

Original review files are retained outside this release folder.
