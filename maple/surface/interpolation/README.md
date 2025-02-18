# Lunar Surface Interpolation Testing

A tool for testing different interpolation methods on lunar surface data.

## File Structure

```
.
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── interpolation_testing.py     # Main script for testing interpolation methods
├── surface_interpolation.py     # Implementation of interpolation methods
└── data/                        # Directory containing lunar surface data files
    └── Moon_Map_*.dat          # Lunar surface data files
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python interpolation_testing.py <data_file> [options]
```

### Command Line Arguments

```
Required arguments:
  data_file              Input data file path (e.g., Moon_Map_01_0_rep0)

Optional arguments:
  --sample-rate RATE     Sampling rate as fraction (0-1), default varies by mode
  --output-dir DIR       Output directory, default: 'interpolation_results'
  
Sampling method (choose one):
  --spiral-path          Use spiral path sampling
  --cluster             Use cluster sampling
  (if neither specified, uses random sampling)
  
Spiral path options:
  --path-width WIDTH     Width of sampling region around path (default: 0.02)
  --path-density DENS    Fraction of points along path vs random (default: 0.8)
  
Cluster options:
  --cluster-radius RAD   Radius of clusters (default: 0.1)
  --cluster-density DENS Fraction of points in clusters (default: 0.85)
  --n-clusters NUM       Number of clusters (default: auto)
  --center-bias BIAS     Bias towards center (0-1, default: 0.7)
  
Batch processing:
  --batch               Run batch processing
  --batch-mode MODE     Mode: sample_rates, path_widths, path_densities, or cluster_sizes
```

### Example Commands

1. Basic spiral path sampling:
```bash
python interpolation_testing.py Moon_Map_01_0_rep0 --spiral-path --sample-rate 0.2 --path-width 0.03 --path-density 0.8
```

2. Cluster sampling:
```bash
python interpolation_testing.py Moon_Map_01_0_rep0 --cluster --sample-rate 0.2 --cluster-radius 0.1 --cluster-density 0.85
```

3. Batch processing with different path densities:
```bash
python interpolation_testing.py Moon_Map_01_0_rep0 --batch --batch-mode path_densities --spiral-path --path-width 0.03 --sample-rate 0.2
```

## Output Files

The script generates output files in the following structure:

```
interpolation_results/
├── interactive_graphs/              # Interactive 3D visualizations
│   ├── interactive_linear_*.html    # Linear interpolation results
│   ├── interactive_nearest_*.html   # Nearest neighbor results
│   ├── interactive_regular_*.html   # Regular grid results
│   └── interactive_b-spline_*.html  # B-spline results
│
├── interpolation_comparison_*.png   # 3D surface comparison plots
├── height_comparison_*.png          # 2D height comparison maps
│
└── batch_results/                   # Generated during batch processing
    ├── metric_summary_*.png         # Summary of error metrics
    └── error_comparison_*.png       # Error metrics vs parameter plots
```

Each plot filename includes:
- Sampling method (spiral, cluster, random)
- Key parameters (sample rate, path width, path density)
- Timestamp

## Batch Processing

The script supports several batch processing modes to analyze how different parameters affect interpolation performance:

1. `sample_rates`: Test different sampling rates while keeping other parameters fixed
   ```bash
   python interpolation_testing.py data_file --batch --batch-mode sample_rates
   ```
   Tests sample rates: [5%, 10%, 20%, 30%, 50%]

2. `path_widths`: Test different path widths while keeping sample rate fixed
   ```bash
   python interpolation_testing.py data_file --batch --batch-mode path_widths --sample-rate 0.2
   ```
   Tests path widths: [0.01, 0.02, 0.03, 0.04, 0.05]

3. `path_densities`: Test different path densities while keeping sample rate and path width fixed
   ```bash
   python interpolation_testing.py data_file --batch --batch-mode path_densities --sample-rate 0.2 --path-width 0.03
   ```
   Tests path densities: [20%, 40%, 60%, 80%, 95%]

4. `cluster_sizes`: Test different cluster sizes while keeping sample rate fixed
   ```bash
   python interpolation_testing.py data_file --batch --batch-mode cluster_sizes --sample-rate 0.2
   ```
   Tests cluster radii: [0.05, 0.1, 0.15, 0.2, 0.25]

For each batch run, the script generates:
- Individual interpolation plots for each parameter value
- Interactive 3D plots (saved in the `interactive_graphs` subfolder)
- Summary plots comparing error metrics across all parameter values
- Error comparison plots showing how each metric varies with the parameter

Note: In batch mode, a fixed random seed (42) is used for consistent generation of sampling patterns.

## Interpolation Methods

The script compares four interpolation methods:
1. Linear interpolation
2. Nearest neighbor interpolation
3. Regular grid interpolation
4. B-spline interpolation

Each method is evaluated using multiple metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Maximum Error
