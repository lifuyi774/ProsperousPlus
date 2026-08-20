# ProsperousPlus: a one-stop and comprehensive platform for accurate protease-specific substrate cleavage prediction and machine-learning model construction.
## Introduction

Proteases contribute to a broad spectrum of cellular functions. Given a relatively limited amount of experimental data, development of accurate sequence-based predictors of substrate cleavage sites facilitates better understanding of protease functions and substrate specificity. While many protease-specific predictors of substrate cleavage sites were developed, these efforts are outpaced by the growth of the protease substrate cleavage data. In particular, since data for 100+ protease types are available and this number continues to grow, it becomes impractical to publish predictors for new protease types, and instead it might be better to provide a computational platform that helps users to quickly and efficiently build predictors that address their specific needs. To this end, we conceptualized, developed, tested, and released a versatile bioinformatics platform, ProsperousPlus, that empowers users, even those with no programming or little bioinformatics background, to build fast and accurate predictors of substrate cleavage sites. ProsperousPlus facilitates the use of the rapidly accumulating substrate cleavage data to train, empirically assess and deploy predictive models for user-selected substrate types. Benchmarking tests on test datasets show that our platform produces predictors that on average exceed predictive performance of current state-of-the-art approaches. ProsperousPlus is available as a webserver and a stand-alone software package at http://prosperousplus.unimelb-biotools.cloud.edu.au/.

## Environment

> **Recommended operating system: Linux.** ProsperousPlus was originally developed and tested in a Linux environment. To minimize platform-specific errors caused by legacy Python packages, compiled libraries, OpenMP runtimes, and binary architectures, we recommend using Linux whenever possible.
>
> ProsperousPlus currently relies on **Python 3.7**, which reached end-of-life in 2023. The current release also depends on legacy machine-learning packages (especially PyCaret 2.3.10 and scikit-learn 0.23.2). Therefore, upgrading Python alone is not a drop-in replacement and would require broader dependency migration and software revalidation.

* Recommended OS: Linux
* Conda distribution: Anaconda, Miniconda, or another compatible Conda installation
* Python 3.7.x (original environment: Python 3.7.13)
* JDK 17
* R

## Dependency

* pandas        1.3.5
* numpy         1.19.5
* scikit-learn  0.23.2
* scipy         1.5.4
* pycaret       2.3.10
* shap          0.42.0
* biopython     1.81
* matplotlib    3.5.3
* weblogo       3.7.12
* catboost      1.2
* lightgbm      3.3.5
* xgboost       1.6.2
* Cython        0.29.36
* pymrmr        0.1.11
* redis         4.6.0

## Installation

### Recommended installation: Linux

ProsperousPlus was developed and tested under Linux, and this is the recommended platform for the stand-alone package.

1. Download and install Anaconda or Miniconda.

   Anaconda: https://www.anaconda.com/download; https://repo.anaconda.com/miniconda/

   Anaconda installation guide: https://docs.anaconda.com/free/anaconda/install/index.html

2. Create the ProsperousPlus environment.

   ```bash
   conda create -n prosperousplus python=3.7
   ```

3. Activate the environment and install the dependencies.

   ```bash
   conda activate prosperousplus
   python -m pip install -r requirements.txt
   python -m pip install pycaret==2.3.10 --no-deps
   ```

4. Install and configure JDK 17.

   See: https://docs.oracle.com/en/java/javase/17/install/overview-jdk-installation.html

5. Install and configure R.

   See: https://cran.r-project.org/manuals.html

6. Install any missing dependencies for PyCaret when prompted.

7. Verify the installation.

   ```bash
   python ProsperousPlus.py -h
   ```

### Apple Silicon macOS (M-series chips): compatibility installation

The command

```bash
conda create -n prosperousplus python=3.7
```

may fail on Apple Silicon Macs because Conda resolves packages for `osx-arm64`, while Python 3.7 is no longer available for that platform through the current conda-forge channel. ProsperousPlus has been successfully installed and launched on Apple Silicon macOS by running the Python environment as **x86_64 through Rosetta 2**.

This macOS procedure is provided as a compatibility solution. Because ProsperousPlus and several of its dependencies were developed for older Python/Linux environments, **Linux remains the recommended operating system** to reduce the risk of additional platform-specific errors.

#### 1. Install the macOS prerequisites

Install Rosetta 2:

```bash
softwareupdate --install-rosetta --agree-to-license
```

Install a current Miniconda/Conda distribution for macOS (e.g. Miniconda3-latest-MacOSX-arm64.pkg). Miniconda can be obtained from:

https://repo.anaconda.com/miniconda/

Install JDK 17 and R:

* JDK 17: https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html
* JDK installation guide: https://docs.oracle.com/en/java/javase/17/install/installation-jdk-macos.html
* R for macOS: https://cran.r-project.org/bin/macosx/

#### 2. Create an x86_64 (`osx-64`) Python 3.7 environment

```bash
conda create --platform osx-64 \
    -n prosperousplus \
    -c conda-forge \
    python=3.7 pip

conda activate prosperousplus
conda config --env --set subdir osx-64
```

Verify the Python version and architecture:

```bash
python --version
python -c "import platform; print(platform.machine())"
```

A successfully configured environment should report Python 3.7.x and:

```text
x86_64
```

The installation tested for this compatibility procedure used Python 3.7.12.

#### 3. Install ProsperousPlus Python dependencies

From the ProsperousPlus source directory, run:

```bash
python -m pip install "pip<24.1" wheel setuptools
python -m pip install -r requirements.txt
python -m pip install pycaret==2.3.10 --no-deps
```

PyCaret 2.3.10 requires `scikit-learn==0.23.2`. Check the installed version:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

If another version is installed, restore the required version:

```bash
python -m pip install "scikit-learn==0.23.2"
```

#### 4. Fix the LightGBM/OpenMP architecture mismatch on Apple Silicon

A pip-installed LightGBM build may attempt to load an ARM64 `libomp.dylib` while the ProsperousPlus Python environment is x86_64. A typical error contains:

```text
Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib
incompatible architecture (have 'arm64', need 'x86_64')
```

Replace the pip-installed LightGBM with the `osx-64` conda-forge build:

```bash
python -m pip uninstall -y lightgbm
conda install -c conda-forge "lightgbm=3.3.3"
```

Verify LightGBM and PyCaret:

```bash
python -c "import lightgbm; print(lightgbm.__version__)"
python -c "from pycaret.classification import setup,get_config,load_model,predict_model; print('PyCaret import OK')"
```

The verified macOS compatibility environment used LightGBM 3.3.3.

#### 5. Rebuild `pymrmr` for the x86_64 macOS environment

The precompiled `pymrmr==0.1.11` wheel may fail on recent macOS systems because it can be linked against an incompatible C++ runtime. One observed error was:

```text
Symbol not found: std::__cxx11::basic_string...
Expected in: /usr/lib/libstdc++.6.dylib
```

Install the x86_64 Conda Clang toolchain and LLVM OpenMP runtime:

```bash
conda install -c conda-forge \
    clang_osx-64 \
    clangxx_osx-64 \
    llvm-openmp
```

Reactivate the environment:

```bash
conda deactivate
conda activate prosperousplus
```

Locate the Conda x86_64 Clang compilers:

```bash
CLANG_CC=$(find "$CONDA_PREFIX/bin" -maxdepth 1 \
    -name 'x86_64-apple-darwin*-clang' -print -quit)

CLANG_CXX=$(find "$CONDA_PREFIX/bin" -maxdepth 1 \
    -name 'x86_64-apple-darwin*-clang++' -print -quit)

"$CLANG_CXX" --version
```

The compiler target should be `x86_64-apple-darwin...`.

Remove the existing `pymrmr` package and rebuild version 0.1.11 from source. The RPATH setting below ensures that the extension can find the x86_64 `libomp.dylib` in the Conda environment at runtime.

```bash
python -m pip uninstall -y pymrmr

CC="$CLANG_CC" \
CXX="$CLANG_CXX" \
LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib" \
CPPFLAGS="-I$CONDA_PREFIX/include" \
python -m pip install \
    --no-cache-dir \
    --no-binary=pymrmr \
    --no-build-isolation \
    "pymrmr==0.1.11"
```

Verify the rebuilt module:

```bash
python -c "import pymrmr; print('pymrmr OK:', pymrmr.__file__)"
python -c "import AutoML; print('AutoML import OK')"
```

If desired, the binary architecture and runtime links can also be checked with:

```bash
PYMRMR_SO=$(python -c "import pymrmr; print(pymrmr.__file__)")
file "$PYMRMR_SO"
otool -L "$PYMRMR_SO"
otool -l "$PYMRMR_SO" | grep -A2 LC_RPATH
```

The successfully rebuilt module should be an x86_64 Mach-O bundle, link to `@rpath/libomp.dylib` and `libc++`, and include the Conda environment's `lib` directory in its RPATH.

#### 6. Verify the complete ProsperousPlus installation

Run the checks below in order:

```bash
python -c "import platform; print('Python architecture:', platform.machine())"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "from pycaret.classification import setup,get_config,load_model,predict_model; print('PyCaret import OK')"
python -c "import pymrmr; print('pymrmr import OK')"
python -c "import AutoML; print('AutoML import OK')"
python ProsperousPlus.py -h
```

The Apple Silicon compatibility installation was considered successful when `ProsperousPlus.py -h` completed normally and displayed the command-line help information.

#### macOS troubleshooting summary

* **`PackagesNotFoundInChannelsError: python=3.7` on `osx-arm64`**: create the environment with `--platform osx-64` and Rosetta 2.
* **LightGBM reports `have 'arm64', need 'x86_64'` for `libomp.dylib`**: replace the pip LightGBM package with `conda-forge::lightgbm=3.3.3` inside the `osx-64` environment.
* **`pymrmr` reports a `libstdc++`/C++ symbol error**: rebuild `pymrmr==0.1.11` from source using Conda Clang/Clang++ and `llvm-openmp` with the RPATH flags shown above.
* **PyCaret reports that it requires `scikit-learn==0.23.2`**: install exactly `scikit-learn==0.23.2`.
* If additional binary or runtime errors occur on macOS, use Linux whenever possible, because Linux is the original and recommended ProsperousPlus environment.

## Usage

To get the information the user needs to enter for help, run:
    python ProsperousPlus.py --help
 or
    python ProsperousPlus.py -h

as follows:

```python ProsperousPlus.py -h```

>
usage: it's usage tip.
>
optional arguments:
“-h”, “--help”    Show this help message and exit
>
“--inputType”    fasta or peptide.
>
“--config”    The path to the config file.

“--trainfile”    The path to the training set file containing the sequences in fasta(peptide) format, where the length of the sequences is 8, 10, 12, 14, 16, 18 or 20.

“--protease”    The protease you want to predict cleavage to, eg: A01.001, Or if you want to build a new model, please create a name. There should no space in the model name.
                
“--outputpath”    The path of output.

“--testfile”    The path to the test set file containing the sequences in fasta(peptide) format, where the length of the sequences is 8, 10, 12, 14, 16, 18 or 20. If not, it will be divided from the training set.

“--predictfile”    The path to the prediction data file containing the sequences in fasta(peptide) format, where the length of the sequences is 8, 10, 12, 14, 16, 18 or 20.

“--mode”    Choose  the program module to run. Three modes can be used: prediction, TrainYourModel, UseYourOwnModel. Only select one mode each time.

“--modelfile”    The path to the trained model generated from the TrainYourModel module. eg 0_model

“--SHAP”    Select Yes or No to control the program to calculate SHAP.

“--PLOT”    Select Yes or No to control whether the program computes the visualization of cleavage sites.

“--processNum”   The number of processes in the program. Note: Integer values represent the number of processes. "processNum" setting can speed up the running efficiency of the program, but it also takes up more computing resources.

## Examples:

### Prediction:
```python ProsperousPlus.py --predictfile data/predict.fasta --outputpath results --inputType fasta --protease A01.001 --mode prediction --PLOT Yes --processNum 2```
### TrainYourModel:
```python ProsperousPlus.py --trainfile data/train.fasta --outputpath resultfile --inputType fasta --protease A01.001 --mode TrainYourModel --SHAP Yes --processNum 2```
### UseYourOwnModel:
```python ProsperousPlus.py --predictfile predict.fasta --outputpath resultfile --inputType fasta --protease A01.001 --mode UseYourOwnModel --modelfile modelfile --processNum 2```
## Output:

When the task is prediction or UseYourOwnModel, the result of the program is the test performance of the model; while when the task is TrainYourModel, the result of the program includes the model files, test results, matrix, ROC, SHAP(if selected), and the visualization of cleavage sites (if selected).

1. matrix: used to encode the features of the sequence.

## Note:

1. The config file contains the default base model for the program.
3. Under the source code of "shap.summary_plot", add two parameters to enable the saving of the SHAP plot.

```shap.summary_plot(shap_values, X_train,max_display=50,show=False, save=True,path='./shap/%s.png'%(d))```

Add to the bottom of the summary_plot function:
```
if save:
    pl.savefig(path)
    pl.close()
```

## Tips:

1. If you encounter a ”numpy.ndarray size changed “error, please do these:

   `pip uninstall pymrmr `

   Download the pymrmr source code from https://github.com/fbrundu/pymrmr

   `python setup.py build_ext --inplace `

   `python setup.py install `
