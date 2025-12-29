# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers with details of the vulnerability
3. Include steps to reproduce if possible
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

### Memory Safety

popcorn operates on GPU memory buffers provided by the caller. The library:

- Validates that pointers are non-null before use
- Does not allocate or free GPU memory (caller responsibility)
- Does not access memory beyond the specified element count
- Supports in-place operations (`out == in`) safely

### Input Validation

All public API functions validate inputs:

- Null pointer checks on all buffer arguments
- Early return for `n <= 0` (no-op, not an error)
- CUDA errors are captured and returned as status codes

### What This Library Does NOT Do

- **No file system access** - Pure computation only
- **No network access** - No external communication
- **No dynamic memory allocation** - Caller manages all buffers
- **No global state mutation** - Thread-safe by design

### Floating Point Considerations

Operations like `Log` and `Sqrt` will produce `NaN` or `Inf` for invalid inputs (negative numbers, zero). This matches standard IEEE 754 behavior and is not considered a bug. Callers should validate inputs if these edge cases matter.

## Acknowledgments

We appreciate responsible disclosure of security issues. Contributors who report valid security concerns will be acknowledged (with permission) in release notes.
