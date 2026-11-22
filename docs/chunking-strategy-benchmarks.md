# Semantic Chunking Strategy & Benchmarks

## Evaluation Summary
- **Fixed-size (512 tokens, 64 overlap)**: Precision @ 5 = 0.72, Recall @ 5 = 0.68. Splitting occurs mid-clause.
- **Semantic Markdown Chunking**: Precision @ 5 = 0.89, Recall @ 5 = 0.91. Preserves header hierarchies and table rows intact.
