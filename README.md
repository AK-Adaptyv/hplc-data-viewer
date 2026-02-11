# HPLC-SEC Chromatogram Analysis

Interactive web app for visualizing and analyzing HPLC Size-Exclusion Chromatography data with peak detection, integration, and molecular weight estimation.

## Features

- **Multi-sequence browser** — select from multiple HPLC run folders
- **Three wavelength channels** — 220 nm, 280 nm, 395 nm (DAD1B, DAD1A, DAD1C)
- **Automated peak detection** — using `scipy.signal.find_peaks` with adjustable sensitivity
- **Peak integration** — trapezoidal area calculation with percentage breakdown
- **MW estimation** — molecular weight from retention time via calibration curve (`10^(-0.8316 * RT + 8.517)`)
- **Interactive controls** — prominence, minimum width, and minimum height sliders
- **Peak hover animation** — shaded peak areas brighten on hover
- **Dark theme** — black background with wavelength-specific gradient line colors

## Data Structure

The app expects the following directory hierarchy:

```
<DATA_ROOT>/
├── Sequence_Folder_1.rslt/
│   ├── Sample_A/
│   │   ├── <timestamp>_DAD1A.CSV    ← 280 nm
│   │   ├── <timestamp>_DAD1B.CSV    ← 220 nm
│   │   └── <timestamp>_DAD1C.CSV    ← 395 nm
│   ├── Sample_B/
│   │   └── ...
│   └── ...
├── Sequence_Folder_2.rslt/
│   └── ...
└── ...
```

Each CSV file has no header, with two columns:
- Column 1: retention time (minutes)
- Column 2: detector response (mAu)

## Setup

### Requirements

- Python 3.8+
- Dependencies: `dash`, `plotly`, `pandas`, `numpy`, `scipy`

### Installation

```bash
pip install dash plotly pandas numpy scipy
```

### Running

```bash
python hplc_sec_analysis.py
```

Then open http://127.0.0.1:8050 in your browser.

### Data Root Configuration

The app locates chromatogram data in this priority order:

1. **`DATA_ROOT` environment variable** — if set and valid
2. **Windows default** — `C:/CDSProjects/hplc_test_ak/csv_chromatograms` (on Windows only)
3. **Current working directory** — fallback for any platform

Override example:

```bash
# Linux/macOS
DATA_ROOT=/path/to/chromatograms python hplc_sec_analysis.py

# Windows
set DATA_ROOT=D:\hplc\data
python hplc_sec_analysis.py
```

## Network Access

The app binds to `0.0.0.0:8050`, making it accessible to other devices on the same network at `http://<server-ip>:8050`.

## Project Structure

```
post-processing/
├── hplc_sec_analysis.py    ← single-file Dash app (all logic included)
├── README.md
└── .gitignore
```

All data loading, peak detection, UI layout, and callbacks are in the single Python file — no additional modules or assets needed.
