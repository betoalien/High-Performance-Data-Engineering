# High-Performance Data Engineering: Integrating Python and Rust

This repository contains a comprehensive course on building hybrid data engineering systems that combine Python's flexibility with Rust's speed and memory safety.

## Course Structure

```
dataengineering_rust_python/
├── index.md                 # Course overview
├── _config.yml              # GitHub Pages configuration
├── chapters/
│   ├── 01-introduction.md   # The Need for Speed
│   ├── 02-rust-core.md      # Building the Rust Engine
│   ├── 03-python-sdk.md     # Bridging the Gap (FFI)
│   ├── 04-etl-pipeline.md   # Building the ETL Pipeline
│   └── 05-jupyter-lab.md    # Interactive Data Engineering
├── examples/
│   ├── etl_pipeline.py      # Complete ETL example
│   └── benchmark.py         # Performance benchmarks
└── notebooks/
    ├── 01-getting-started.ipynb   # Interactive introduction
    ├── 02-etl-pipeline.ipynb      # Self-contained ETL demo (generates its own data)
    └── 02-etl-workflow.ipynb      # End-to-end workflow (requires a real CSV file)
```

## Quick Start

### Viewing the Course

The course is formatted for GitHub Pages with the "Just the Docs" theme. To view locally:

```bash
# Install Jekyll and the theme
gem install bundler
bundle install

# Start local server
bundle exec jekyll serve
```

Then open http://localhost:4000/dataengineering_rust_python/

### Running the Examples

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies (you'll need the hyperframe SDK)
pip install -e ../hyperframe_sdk/
pip install jupyterlab matplotlib pandas

# Run the ETL pipeline
python examples/etl_pipeline.py --input your_data.csv --output output.parquet

# Run benchmarks
python examples/benchmark.py --rows 100000

# Start Jupyter
jupyter lab notebooks/
```

## What You'll Learn

1. **Why Rust + Python**: Understanding the performance characteristics and use cases
2. **Rust Core Development**: Building columnar data structures, compute kernels, and FFI exports
3. **Python SDK Design**: Creating Pythonic wrappers with ctypes, memory management, error handling
4. **ETL Pipelines**: Building production-ready data pipelines with the hybrid engine
5. **Interactive Analysis**: Using Jupyter notebooks for exploratory data analysis

## Prerequisites

- Intermediate Python programming
- Basic understanding of data engineering concepts
- Familiarity with command line and package management

## Performance Highlights

The hybrid Rust+Python approach demonstrated in this course achieves:

| Operation | Speedup vs Pandas |
|-----------|------------------|
| CSV Load | 5-7x faster |
| Filtering | 8-10x faster |
| GroupBy Aggregation | 5-8x faster |
| Hash Joins | 7-10x faster |
| Memory Usage | 60-65% reduction |

## License

This course material is provided for educational purposes.

## Contributing

Contributions welcome! Please open an issue or submit a PR for:
- Corrections or clarifications
- Additional examples
- Performance improvements
- New chapters or topics
