# Contributing to Project AGASTYA

Thank you for your interest in contributing to **Project AGASTYA (SIH26168)**! We welcome contributions that improve algorithmic precision, real-time deterministic performance, documentation, and hardware integration.

---

## 1. Core Principles: Zero-Leakage & Scientific Integrity

Project AGASTYA is an aerospace-grade navigation system governed by strict scientific principles. All contributions must adhere to the following rules:

1. **Strict Temporal Causality**:
   - Navigation features at epoch $k$ must strictly consume data from $t \le t_k$.
   - Any access to future samples ($t > t_k$) constitutes a critical failure.
2. **Quarantined Reference Streams**:
   - VBOX RTK GPS coordinates, velocities, and true headings must remain strictly isolated for label generation during training and ground-truth evaluation.
   - Reference streams must NEVER enter inference feature registries or online runtimes.
3. **Disjoint Trajectory Splitting**:
   - Do NOT perform random timestep shuffling across datasets.
   - `sync_01` is reserved for training, `v_standalone_03` for validation early stopping, and `sync_02` strictly as held-out test data.
4. **Train-Only Scaler Provenance**:
   - Feature normalizers (Z-score mean and variance) must be fitted solely on training trajectories.

---

## 2. Development Setup

### 2.1 Virtual Environment Setup
```bash
git clone https://github.com/krushnasaruk/Agastya.git
cd Agastya

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2.2 Running Automated Tests
Before submitting any changes, verify that all 181 automated tests pass:
```bash
python -m pytest
```

---

## 3. Code Standards & Style

1. **Python Version**: Compatible with Python 3.10 through 3.14.
2. **Type Hints**: Use standard type annotations (`typing` module) across public methods and interfaces.
3. **Docstrings**: Provide clear Google-style or NumPy-style docstrings explaining physical units (e.g., $\text{m/s}$, $\text{rad/s}$, $\text{deg}$, $\text{ms}$).
4. **Deterministic Reproducibility**: Always initialize random seeds (`seed=42`) for any stochastic processes or synthetic simulations.

---

## 4. Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

- `feat(...)`: Introduces a new feature or algorithmic capability.
- `fix(...)`: Fixes a bug or numerical inconsistency.
- `docs(...)`: Updates or adds documentation, reports, or diagrams.
- `test(...)`: Adds or improves unit / integration tests.
- `perf(...)`: Enhances real-time execution speed or memory efficiency.
- `refactor(...)`: Code restructuring without modifying behavioral semantics.

---

## 5. Pull Request Workflow

1. Fork the repository and create a descriptive feature branch:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. Implement changes and add corresponding pytest test cases in `tests/`.
3. Verify test suite:
   ```bash
   python -m pytest
   ```
4. Commit changes following conventional commit syntax.
5. Push to your fork and submit a Pull Request against `main`.
