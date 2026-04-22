# Self-Pruning Neural Network

## Introduction
In real-world deep learning deployments, memory and computational constraints often require smaller and more efficient models. Traditional model pruning is usually done *after* training.

This project implements a **self-pruning neural network** that prunes connections *during training itself*.

The key idea:
- Each weight is paired with a learnable **gate parameter** (value between 0 and 1)
- Gates act as multipliers on weights
- A custom loss function pushes some gates toward **0**, effectively removing weak connections dynamically

---

## Implementation Details

### Part 1: Prunable Linear Layer

Standard `torch.nn.Linear` layers are replaced with a custom **PrunableLinear** layer.

**Key additions:**
- `gate_scores`: learnable parameters (same shape as weights)

**Forward Pass:**
```python
gates = sigmoid(gate_scores)
pruned_weights = weight * gates
output = F.linear(input, pruned_weights, bias)
```

✔ Gradients flow through both weights and gates.

---

### Part 2: Sparsity Regularization Loss

```math
Total Loss = ClassificationLoss + \lambda \times SparsityLoss
```

Where:
- **SparsityLoss** = sum of all gate values  
- Equivalent to **L1 regularization on gates**  
- **λ (lambda)** controls sparsity vs accuracy trade-off  

---

### Why L1 on Sigmoid Gates Works

- Constant gradient near zero → pushes values to exactly 0  
- Encourages true sparsity  
- Similar to Lasso regression (feature selection)  
- Gates lie between 0 and 1, making pruning stable  

Result:
- Gates behave almost binary:
  - ≈ 0 → pruned  
  - ≈ 1 → active  

---

### Part 3: Training & Evaluation

- Dataset: **CIFAR-10**  
- Optimizer: **Adam**  
- Loss: Classification + Sparsity penalty  

**Metrics tracked:**
- **Sparsity Level**: % of gates below threshold (e.g., 1e-2)  
- **Test Accuracy**

---

## Results

### Training Setup
- Batch Size: 128  
- Epochs: 10  
- Learning Rate: 1e-3  
- Threshold: 1e-2  

### Final Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|--------------|
| 1e-5   | 56.18            | 0.44         |
| 1e-4   | 55.77            | 1.46         |
| 1e-3   | 54.66            | 1.72         |

---

## Trade-off Analysis

- Increasing **λ** → higher sparsity (more pruning)  
- But → slight drop in accuracy  

Examples:
- **λ = 1e-5** → highest accuracy, lowest sparsity  
- **λ = 1e-3** → higher sparsity, lower accuracy  

---

## Gate Value Distribution

- Histogram saved as: `gate_distribution.png`

Expected:
- Peak near 0 → pruned weights  
- Cluster near 1 → active weights  

Observed:
- Many values near 0  
- Spread toward higher values  
✔ Indicates successful pruning behavior  

---

## How to Run

1. Open the provided **Colab notebook**
2. Run all cells sequentially
3. Outputs generated:
   - Training logs  
   - Results table  
   - `results.csv`  
   - `gate_distribution.png`  

---

## Outputs
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/e2c3ddd8-0157-433b-9fc3-34c1e4b5ced5" />
 
