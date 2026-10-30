# Data

The dataset is **not** stored in this repository. Download it from the UCI Machine Learning
Repository and place the two files below in `data/raw/`:

| File | Content |
|------|---------|
| `adult.data` | official training file (32,561 rows) |
| `adult.test` | official test file (16,281 rows; first line is a comment, labels end with a full stop) |

Source: <https://archive.ics.uci.edu/dataset/2/adult>

Use the "Download" button on that page, unzip the archive and copy `adult.data` and `adult.test`
into `data/raw/`. (The expected final layout is `data/raw/adult.data` and `data/raw/adult.test`.)

`adult.names` (column descriptions) and `old_adult.names` are documentation only and are not used by
the code. Please read the terms of use / licence shown on the UCI dataset page before redistributing
the data; this is the reason the files are git-ignored.

## Citation

Becker, B. and Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository.
Extracted by Barry Becker from the 1994 US Census (Current Population Survey) database.

## Things to know about the data

- The extraction kept only "reasonably clean" records (age > 16, positive adjusted gross income,
  positive final weight, hours per week > 0), so it is not a random sample of the US population.
- Missing values are written as `?` (in `workclass`, `occupation` and `native-country`).
- The official train/test split is a random 2/3 - 1/3 split, which is why the project can safely pool
  the two files and make its own stratified split.
- `fnlwgt` is a survey weight; it is not used as a feature.
