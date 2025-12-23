# Credit Risk Probability Model for Alternative Data

## Business Understanding

### 1. Basel II Accord's Influence
Basel II requires banks to quantify credit risk for capital requirements. This necessitates:
- **Interpretable models**: Regulators must understand risk drivers
- **Documentation**: Full model development, validation, and monitoring records
- **Risk quantification**: Probability of Default (PD) must be measurable

### 2. Proxy Variable Necessity & Risks
**Why necessary**: E-commerce data lacks direct default labels. We use RFM (Recency, Frequency, Monetary) patterns as a proxy for creditworthiness.

**Business risks**:
1. **Proxy misalignment**: RFM measures engagement, not repayment ability
2. **Concept drift**: E-commerce behavior ≠ loan repayment behavior
3. **Regulatory scrutiny**: Proxy-based models require strong justification

### 3. Model Selection Trade-offs
| Model Type | Pros | Cons | Regulatory Fit |
|------------|------|------|----------------|
| **Logistic Regression with WoE** | Interpretable, transparent coefficients, well-understood | Limited non-linear patterns | ✅ Excellent |
| **Gradient Boosting (XGBoost)** | Higher accuracy, captures complex patterns | "Black box", harder to explain | ⚠️ Requires SHAP/LIME for explainability |

**Recommended approach**: Start with logistic regression for baseline, then compare with XGBoost using SHAP for explainability.