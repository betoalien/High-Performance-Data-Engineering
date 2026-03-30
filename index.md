---
title: High-Performance Data Engineering
layout: home
nav_order: 1
description: A comprehensive course on integrating Python and Rust for high-performance data pipelines
---

# High-Performance Data Engineering: Integrating Python and Rust

This course teaches advanced data engineers how to build hybrid systems that combine **Python's flexibility** with **Rust's speed and memory safety** for high-performance data processing.

## Course Overview

Modern data engineering demands both rapid development cycles and extreme performance. Python excels at rapid prototyping and has a rich ecosystem of data science libraries. Rust provides memory safety without garbage collection, zero-cost abstractions, and performance comparable to C++.

By combining these languages, you can build data pipelines that are:

- **Fast**: Leverage Rust's zero-cost abstractions and SIMD optimizations
- **Safe**: Memory safety guaranteed at compile time, no null pointer exceptions
- **Flexible**: Python's expressive syntax for orchestration and analysis
- **Scalable**: Handle billions of rows with columnar storage and parallel processing

## Syllabus

| Chapter | Title | Description |
|---------|-------|-------------|
| [Chapter 1](chapters/01-introduction.md) | The Need for Speed | Introduction to high-performance data engineering and hybrid architectures |
| [Chapter 2](chapters/02-rust-core.md) | The Rust Core | Building memory-safe data structures and concurrent processing in Rust |
| [Chapter 3](chapters/03-python-sdk.md) | Bridging the Gap | FFI bindings, PyO3, and designing Pythonic wrappers |
| [Chapter 4](chapters/04-etl-pipeline.md) | Building the ETL Pipeline | Practical examples of extraction, transformation, and loading |
| [Chapter 5](chapters/05-jupyter-lab.md) | Interactive Data Engineering | Jupyter notebooks for exploratory data analysis |

## Prerequisites

- Intermediate Python programming
- Basic understanding of systems programming concepts
- Familiarity with data engineering concepts (ETL, dataframes, schemas)

## What You'll Build

By the end of this course, you will have:

1. A working Rust library with C-compatible FFI exports
2. A Python package that loads and uses the Rust library via `ctypes`
3. A complete ETL pipeline demonstrating the hybrid architecture
4. Interactive Jupyter notebooks for data exploration

## License

This course material is provided for educational purposes.
