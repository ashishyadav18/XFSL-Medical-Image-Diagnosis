import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 1. Define the metrics (Labels)
labels = ['Mean Accuracy', 'Macro F1-Score', 'MCC']

# 2. Define the data from Table IV (Accuracies divided by 100 for a shared 0.0-1.0 scale)
svm_values = [0.7708, 0.7667, 0.7157]
proto_values = [0.7589, 0.7551, 0.7008]
proposed_values = [0.7664, 0.7634, 0.7105]


# 3. Close the loop for the radar chart math
svm_values += svm_values[:1]
proto_values += proto_values[:1]
proposed_values += proposed_values[:1]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 4. Set up the plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2) # Start top
ax.set_theta_direction(-1)     # Clockwise

# 5. Plot the models
# Linear SVM (Red, Dashed)
ax.plot(angles, svm_values, color='#d62728', linewidth=2, linestyle='dashed', label='Linear SVM')
ax.fill(angles, svm_values, color='#d62728', alpha=0.1)

# Vanilla ProtoNet (Blue, Dotted)
ax.plot(angles, proto_values, color='#1f77b4', linewidth=2, linestyle='dotted', label='Vanilla ProtoNet')
ax.fill(angles, proto_values, color='#1f77b4', alpha=0.1)

# Proposed XFSL (Green, Solid, Thick)
ax.plot(angles, proposed_values, color='#2ca02c', linewidth=3, label='Proposed XFSL')
ax.fill(angles, proposed_values, color='#2ca02c', alpha=0.25)

# 6. Formatting
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, weight='bold')

# Zoom in the axis limits to make the differences highly visible (0.68 to 0.80)
ax.set_ylim(0.68, 0.80)

# Add Legend and Title
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.title('Multi-Metric Performance at K=20', size=14, weight='bold', y=1.1)

# 7. Save and show
plt.tight_layout()
os.makedirs("outputs/graphs", exist_ok=True)
save_path = "outputs/graphs/radar_chart_k20.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print("Successfully saved 'radar_chart_k20.png'")
