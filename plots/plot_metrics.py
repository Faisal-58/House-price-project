import matplotlib.pyplot as plt

# Hardcoded R² scores based on your manual metrics.txt
r2_scores = {
    "Linear Regression": 0.782,
    "Random Forest": 0.872,
    "XGBoost": 0.876
}

# Plot the bar chart
plt.figure(figsize=(8, 5))
plt.bar(r2_scores.keys(), r2_scores.values(), color='skyblue')
plt.ylabel("R² Score")
plt.title("Model Comparison - R² Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("plots/r2_scores.png")
