# BERT Ablation Factory - Systematic Debugging Report

**Generated:** 2025-03-28
**Methodology:** Systematic root cause analysis across all components

## Executive Summary

This report documents the results of systematic debugging across all components of the BERT Ablation Factory project. The analysis identified **1 critical bug**, **3 medium-priority issues**, and **5 minor improvements**.

## Critical Issues (Must Fix)

### 1. 🚨 Device Mismatch in Evaluation Function

**File:** `bert_ablation_factory/trainer/engine.py`
**Function:** `evaluate()` (lines 90-113)
**Severity:** CRITICAL
**Root Cause:** Model not moved to device in evaluation function

**Problem:**
```python
def evaluate(model: torch.nn.Module, loader: DataLoader, ...):
    model.eval()  # Model NOT moved to device!
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Batch moved to device
        outputs = collate_loss_fn(batch)  # Model on CPU, batch on CUDA = ERROR
```

**Impact:**
- RuntimeError: "Expected all tensors to be on the same device"
- Evaluation completely broken when using CUDA
- Training appears to work but validation fails

**Fix Required:**
```python
def evaluate(model: torch.nn.Module, loader: DataLoader, ...):
    model.eval()
    model.to(device)  # ADD THIS LINE
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = collate_loss_fn(batch)
```

**Verification:**
- Test with CUDA-enabled environment
- Verify evaluation loss computes without errors
- Check that metrics are correctly computed

## Medium Priority Issues (Should Fix)

### 2. ⚠️ Streaming Data Reproducibility

**File:** `bert_ablation_factory/cli/pretrain.py`
**Function:** `build_books_wiki_stream()` (lines 25-64)
**Severity:** MEDIUM
**Root Cause:** Random seed not properly fixed in streaming data generation

**Problem:**
```python
def gen_examples():
    # ...
    for s in sents:
        if prev is None:
            prev = s
            continue
        import random  # Imported inside function
        if random.random() < 0.5:  # Random not seeded!
```

**Impact:**
- Non-reproducible pretraining runs
- Different NSP samples across runs with same seed
- Affects experimental comparability

**Fix Required:**
```python
def build_books_wiki_stream(tokenizer, max_len: int, seed: int = 42):
    # ...
    def gen_examples():
        rng = random.Random(seed)  # Use seeded random
        # ...
        if rng.random() < 0.5:
```

### 3. ⚠️ Gradient Accumulation Inconsistency

**File:** `bert_ablation_factory/trainer/engine.py`
**Function:** `train_loop()` (lines 66-70)
**Severity:** MEDIUM
**Root Cause:** Gradient accumulation steps not properly synchronized

**Problem:**
```python
if step % int(cfg.get("GRAD_ACCUM_STEPS", 1)) == 0:
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    sched.step()  # Scheduler steps every accumulation cycle
```

**Impact:**
- Learning rate schedule not aligned with actual updates
- Scheduler steps too frequently with gradient accumulation
- Affects convergence behavior

**Fix Required:**
```python
if step % int(cfg.get("GRAD_ACCUM_STEPS", 1)) == 0:
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)
    # Only step scheduler when optimizer steps
    sched.step()
```

### 4. ⚠️ Checkpoint Device Management

**File:** `bert_ablation_factory/trainer/checkpoint.py`
**Function:** `load_checkpoint()` (lines 57-80)
**Severity:** MEDIUM
**Root Cause:** Model not moved to device after loading

**Problem:**
```python
def load_checkpoint(path, model, optimizer, scheduler, scaler):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])  # Model stays on CPU!
    # ...
```

**Impact:**
- After resuming, model on CPU while optimizer on CUDA
- Device mismatch errors
- Training fails after checkpoint resume

**Fix Required:**
```python
def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    state = torch.load(path, map_location=device)  # Load directly to device
    model.load_state_dict(state["model"])
    model.to(device)  # Ensure model on correct device
    # ...
```

## Minor Issues (Nice to Fix)

### 5. 🔍 Type Hint Inconsistencies

**File:** Multiple files
**Severity:** LOW
**Root Cause:** Inconsistent type hints and missing annotations

**Examples:**
```python
# In registry.py
def register(self, key: str) -> Callable[[Any], Any]:  # Too broad

# In build.py
def build_pretrain_model(cfg: Dict[str, Any], ablation: str):  # Missing return type
```

**Impact:**
- Reduced IDE support
- Harder static analysis
- Potential runtime type errors

**Fix:** Add comprehensive type hints using modern Python typing

### 6. 🔍 Magic Numbers

**File:** Multiple files
**Severity:** LOW
**Root Cause:** Hardcoded values without explanation

**Examples:**
```python
for epoch in range(10_000_000):  # Magic large number
if random.random() < 0.5:  # Magic probability
torch.full(labels.shape, 0.8)  # Magic 80% value
```

**Impact:**
- Code harder to understand
- Difficult to modify parameters
- No clear documentation of choices

**Fix:** Define constants with descriptive names

### 7. 🔍 Missing Docstring Parameters

**File:** Multiple functions
**Severity:** LOW
**Root Cause:** Incomplete documentation

**Examples:**
```python
def build_optimizer(params: Iterable, cfg: Dict[str, Any]):
    """Build optimizer."""  # Missing parameter docs
    # ...
```

**Impact:**
- Poor developer experience
- Harder to use functions correctly
- Increased learning curve

**Fix:** Add comprehensive docstrings following Google/NumPy style

### 8. 🔍 Inconsistent Error Handling

**File:** Multiple files
**Severity:** LOW
**Root Cause:** Mix of exception types and handling patterns

**Examples:**
```python
try:
    scheduler.load_state_dict(state.get("scheduler", {}))
except Exception:  # Too broad
    pass

try:
    from transformers import BertLMHeadModel
except Exception as e:  # Specific but catches too much
    raise RuntimeError(...) from e
```

**Impact:**
- Hard to debug specific issues
- May hide important errors
- Inconsistent error reporting

**Fix:** Use specific exception types and proper error chains

### 9. 🔍 Performance Optimizations Missing

**File:** `bert_ablation_factory/data/squad.py`
**Severity:** LOW
**Root Cause:** Inefficient data processing

**Examples:**
```python
# Inefficient string splitting
sents = [s.strip() for s in text.split(".") if s.strip()]

# Repeated computation
while idx < len(sequence_ids) and sequence_ids[idx] != 1:
    idx += 1
```

**Impact:**
- Slower data loading
- Increased preprocessing time
- Higher CPU usage

**Fix:** Use more efficient algorithms and caching

## Integration Issues

### 10. 🔗 Component Interface Mismatches

**File:** `bert_ablation_factory/cli/pretrain.py`
**Severity:** MEDIUM
**Root Cause:** Collator output doesn't match model input expectations

**Problem:**
```python
# In pretrain.py
collator = MLMNSPCollator(...)  # Returns dict with specific keys
# But model might expect different key names
```

**Impact:**
- Potential key mismatches
- Silent failures if keys missing
- Hard to debug data flow issues

**Fix:** Add explicit key validation and better error messages

## Data Flow Analysis

### Pretraining Pipeline
```
Config → Model Building → Data Streaming → Collator → Training
     ✅           ❓            ⚠️         ✅        ❓
```

**Issues Found:**
1. ✅ Config loading: Validated
2. ❓ Model building: Device management issues
3. ⚠️ Data streaming: Reproducibility issues
4. ✅ Collator: Correct
5. ❓ Training: Gradient accumulation issues

### Finetuning Pipeline
```
Config → Task Building → Model Building → Data Loading → Training/Eval
     ✅          ✅            ✅            ✅           ❓
```

**Issues Found:**
1. ✅ Config loading: Validated
2. ✅ Task building: Correct
3. ✅ Model building: Correct
4. ✅ Data loading: Correct
5. ❓ Training/Eval: Device mismatch (CRITICAL)

### Data Pipeline
```
Raw Data → Tokenization → Collator → Batches → Model
      ✅         ✅         ✅        ✅       ❓
```

**Issues Found:**
1. ✅ Raw data: Correct
2. ✅ Tokenization: Correct
3. ✅ Collator: Correct
4. ✅ Batches: Correct
5. ❓ Model: Device issues

## Reproducibility Issues

### 1. Random Seed Propagation
- **Status:** ⚠️ PARTIAL
- **Issue:** Seeds not propagated to all random number generators
- **Impact:** Non-deterministic behavior in some components
- **Fix:** Ensure all RNGs seeded: torch, numpy, random, cuda

### 2. Data Order Determinism
- **Status:** ⚠️ PARTIAL
- **Issue:** Streaming data order not guaranteed
- **Impact:** Different data order across runs
- **Fix:** Use deterministic data loading with fixed seeds

### 3. CUDA Non-Determinism
- **Status:** ⚠️ PARTIAL
- **Issue:** CUDA operations can be non-deterministic
- **Impact:** Slight differences in results
- **Fix:** Set `torch.backends.cudnn.deterministic = True`

## Performance Issues

### 1. Memory Usage
- **Status:** ⚠️ CONCERN
- **Issue:** No gradient accumulation clearing
- **Impact:** Memory leaks during long training
- **Fix:** Ensure proper gradient zeroing

### 2. Data Loading
- **Status:** ⚠️ CONCERN
- **Issue:** Streaming data not buffered
- **Impact:** I/O bottlenecks
- **Fix:** Add buffering and prefetching

### 3. Computation
- **Status:** ✅ GOOD
- **Issue:** None identified
- **Impact:** N/A
- **Fix:** N/A

## Testing Gaps

### 1. Device-Specific Tests
- **Missing:** CUDA device placement tests
- **Impact:** Device bugs not caught
- **Needed:** Tests with mock CUDA devices

### 2. Integration Tests
- **Missing:** End-to-end workflow tests
- **Impact:** Integration issues not caught
- **Needed:** Full pipeline tests

### 3. Performance Tests
- **Missing:** Memory and speed benchmarks
- **Impact:** Performance regressions not caught
- **Needed:** Benchmarking suite

## Recommended Fix Priority

### Immediate (Before Any Training)
1. ✅ Fix device mismatch in evaluate() - CRITICAL
2. ✅ Fix checkpoint device management

### Soon (Before Production)
3. ⚠️ Fix streaming data reproducibility
4. ⚠️ Fix gradient accumulation scheduling
5. 🔍 Add comprehensive type hints

### Eventually (Nice to Have)
6. 🔍 Remove magic numbers
7. 🔍 Complete docstrings
8. 🔍 Performance optimizations
9. 🔍 Add integration tests
10. 🔍 Add performance benchmarks

## Verification Checklist

Before using in production, verify:

- [ ] Device mismatch fixed and tested
- [ ] Checkpoint resume works correctly
- [ ] Reproducibility verified across runs
- [ ] Gradient accumulation tested
- [ ] All tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete
- [ ] Error handling tested
- [ ] Memory usage acceptable

## Conclusion

The systematic debugging identified **1 critical bug** that must be fixed before any training, **3 medium-priority issues** that should be addressed for production use, and **5 minor improvements** for code quality.

**Critical Path:**
1. Fix device mismatch in evaluate()
2. Fix checkpoint device management
3. Verify reproducibility
4. Test gradient accumulation
5. Run integration tests

The codebase is well-structured but has some device management issues that need immediate attention.
