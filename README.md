## FDIA Research Tool for the IEEE 118-Bus System

A Python desktop application for generating False Data Injection Attack (FDIA) datasets and evaluating how well machine-learning models distinguish attacked power-system measurements from normal measurements. The interface is built with PyQt6, the network model comes from pandapower, and the detection models use scikit-learn.

## Research background

This project relates to the included research article, [*Evaluating the Impact of Attack-Pattern Variability on False Data Injection Attack Detection in Power System State Estimation*](Arijit_FDIA_PES_IEEE_Conference.pdf).

The paper investigates a central question: **How does changing the attack location and magnitude affect FDIA detection?** A detector may recognize a repeated attack pattern well but perform differently when the attacked buses and perturbation strength change between samples.

The experiment uses normal operating measurements from the IEEE 118-bus system and constructs attacks using a linear DC state-estimation model. The underlying attack formulation is:

```text
a = Hc
z_attacked = z + a
```

Here, `z` is a measurement vector, `H` is the measurement Jacobian, `c` is a sparse state-perturbation vector, and `a` is the resulting measurement perturbation. In this implementation, the attack vector is then restricted to accessible active-power injection measurements. A residual check measures whether an attack exceeds the implemented detection threshold; constructing `Hc` alone does not guarantee stealth after this restriction.

The application lets you generate paired normal/attacked datasets, vary attack settings, and compare classification results. It uses an existing normal dataset; it does not generate the original Optimal Power Flow (OPF) dataset.

## Requirements and setup

- Python 3.10 or newer, with a desktop environment for the GUI.
- The normal dataset, `IEEE118Normal_Corrected.csv`, in the same folder as `app.py`.
- Python dependencies listed in **`requirement.txt.txt`**. This is the actual filename in this repository.

Open a terminal in the project folder. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirement.txt.txt
.\.venv\Scripts\python.exe app.py
```

On Linux or macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt.txt
.venv/bin/python app.py
```

Dependencies include NumPy, pandas, SciPy, scikit-learn, pandapower, and PyQt6. Run the application **from the project folder**, because the input CSV is loaded using a relative path.

## Run the experiment

### 1. Generate normal and attacked datasets

Start the application and enter the following settings on the generation page:

| Setting | Meaning | Example |
| --- | --- | --- |
| Dataset size | Number of normal rows sampled and corresponding attack rows generated; GUI limit is 70,000 | `5000` |
| Attack area | Check the buses eligible for selection during each attack | Buses `80` through `110` |
| Attack magnitude | Lower and upper bounds for the state perturbation; inputs are divided by 100 | From `10` to `20` |
| Number of meters | Despite the interface label, this controls how many target buses are sampled per row | From `10` to `20` |
| Authentic dataset name | Filename for the sampled normal data | `authentic_variable` |
| FDIA dataset name | Filename for the corresponding attacked data | `fdia_variable` |

Use a positive meter-count range whose maximum does not exceed the number of checked buses. Enter magnitude bounds in ascending order within the displayed 0-100 range. The magnitude is a value assigned to `c`; entering `20` gives a state shift of `0.20`, rather than multiplying each measurement by 20%.

Click **Generate**. The backend samples normal rows, constructs the DC measurement model, generates attacks, checks residuals, and saves two CSV files beside `app.py`. The `.csv` extension is added automatically if omitted. Use different output names for each experiment; existing files with the same names are overwritten.

For the example above, the outputs are:

```text
authentic_variable.csv   # 5,000 sampled normal rows
fdia_variable.csv        # 5,000 corresponding attacked rows, Label = 1
```

Only the `PG` and `PL` active-power columns are modified in the saved attack CSV. Other measurements are retained. Line flows are computed internally for the residual check and are not added to the output CSVs. The authentic file retains the original dataset's labels, which should be `0` for normal data.

The success dialog reports a **Stealth Count**: the number of attacked samples whose residual norm does not exceed the 95th percentile of the normal residual norms. All generated attack rows are saved, including rows that exceed that threshold.

Generation prints per-snapshot information in the terminal. Computation runs in the GUI thread, so the window may temporarily stop responding. Start with a smaller dataset, such as 100 rows, to check the workflow before generating a larger experiment.

### 2. Evaluate detection models

1. Click **Test Generated Data**.
2. Select the authentic CSV and its corresponding FDIA CSV.
3. Enter positive training and testing percentages that sum to 100, such as `80` and `20`.
4. Check one or more models: **Logistic Regression**, **SVM**, **Random Forest**, or **MLP**.
5. Select the buses whose measurements should be used as features, or check **Select All Buses**. Leaving all buses unchecked also uses all numeric feature columns.
6. Select **Accuracy**, **Precision**, and/or **Recall**. See the F1 limitation below.
7. Click **Generate** on the testing page to train the selected models and display the results.

Preprocessing combines the files, assigns FDIA rows to class `1`, adds normal labels of `0` if missing, and drops columns that are entirely zero in the authentic data. Bus selection includes numeric measurement columns associated with those bus numbers, rather than only `PG` and `PL` columns. A stratified train/test split uses `random_state=42`.

Logistic Regression, SVM, and MLP use standardization fitted on the training data. Random Forest uses the selected features directly. Accuracy, precision, and recall are displayed as fractions between `0` and `1`; FDIA is the positive class. Results appear in a dialog; there is no automatic report or trained-model export.

### 3. Compare attack-pattern variability

The paper compares three configurations. You can configure their selection rules through the GUI as follows:

| Configuration | Candidate buses | Meter-count range | Magnitude range |
| --- | --- | --- | --- |
| Fixed location and magnitude | `80` through `110` | `31` to `31` | `20` to `20` |
| Variable location, fixed magnitude | `80` through `110` | `10` to `20` | `20` to `20` |
| Variable location and magnitude | `80` through `110` | `10` to `20` | `10` to `20` |

For the fixed configuration, selecting all 31 candidate buses per sample keeps the bus set constant. For variable configurations, a new subset is sampled for each row.

Generate a separate file pair for each configuration, then evaluate each pair with the same models, feature selection, and split percentages. Record the settings, stealth count, and detection metrics to compare how performance changes. These settings mirror the paper's configuration rules; the implementation differences below mean they do not reproduce all reported results.

## Implementation notes and limitations

- **Paper coverage:** This folder implements four classifiers. The paper also evaluates models such as CNN and LSTM and includes additional similarity, distributional, and physical-consistency analyses that are not implemented here.
- **Residual criterion:** The code uses a single threshold at the 95th percentile of normal residuals. The article reports a paired criterion comparing each attacked residual with its matched normal residual.
- **Data partitioning:** The application provides a train/test split without a separate validation set. Normal and attacked counterparts are split as independent rows and may appear in different subsets.
- **F1 display:** The GUI sends `F1 Score`, while the metrics backend checks for `F1`. Selecting F1 Score currently produces no F1 result.
- **Bus indexing:** GUI buses are numbered 1-118, but state perturbations are looked up directly against pandapower's zero-based bus indices. This mapping needs review before interpreting attacks as targeting the exact bus IDs shown in the interface.
- **Randomness:** Normal-row sampling uses seed `1`, and the split and applicable classifiers use seed `42`. Attack subset and magnitude sampling are not seeded, so repeated generation can produce different attacks.
- **Model scope:** Attack construction uses a simplified linear DC model. The included code does not validate the generated data against nonlinear AC state estimation.

## Project files

```text
app.py                              Main application entry point and page navigation
fdia_generation_page.py             Dataset-generation interface
logic.py                            Connects the interface to generation logic
test.py                             Dataset loading, DC model, attacks, residuals, CSV export
testing_page_updated.py             Model-evaluation interface
testing/
    runner.py                       Runs the selected detection models
    pre_processing.py               Labeling, feature selection, zero-column removal, splitting
    metrics.py                      Classification metrics
    algorithms/                     Logistic Regression, SVM, Random Forest, and MLP
requirement.txt.txt                 Python dependency list
IEEE118Normal_Corrected.csv         Input normal operating dataset
Arijit_FDIA_PES_IEEE_Conference.pdf  Related research article
```

Despite its filename, `test.py` contains experiment-generation logic; launch the complete application with `app.py`.

## Dataset distribution

The input CSV is approximately 368 MB and is tracked with Git LFS through `.gitattributes`. Install Git LFS before cloning the repository. If you already cloned it, run these commands from the project folder to download the full dataset:

```bash
git lfs install
git lfs pull
```

The application requires the full CSV, rather than an LFS pointer file, with the exact filename `IEEE118Normal_Corrected.csv` beside `app.py` unless the loading path in `test.py` is changed.

Keep the source code, dependency file, and README together. Virtual environments, Python caches, and generated experiment CSVs do not need to be included in the source repository.
