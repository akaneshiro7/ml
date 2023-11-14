import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import multivariate_normal

# Given Values
m1 = [0,0,0]
m2 = [2,0,0]
m3 = [1,np.sqrt(3),0]
m4 = [1,np.sqrt(3) / 2, np.sqrt(3)]

CO1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
CO2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
CO3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
CO4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

w1 = 0.3
w2 = 0.3
w3 = 0.4

weight1 = weight2 = 0.5

np.random.seed(0)

n_samples = 10_000

# Generate Samples
samples_L1 = multivariate_normal.rvs(mean=m1, cov=CO1, size=int(n_samples * w1))
samples_L2 = multivariate_normal.rvs(mean=m2, cov=CO2, size=int(n_samples * w2))
samples_L31 = multivariate_normal.rvs(mean=m3, cov=CO3, size=int(n_samples * w3 * weight1))
samples_L32 = multivariate_normal.rvs(mean=m4, cov=CO4, size=int(n_samples * w3 * weight2))

samples_L3 = np.concatenate((samples_L31, samples_L32))

# Plot samples
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(samples_L1[:, 0], samples_L1[:, 1], samples_L1[:, 2], alpha=0.2, label='L1')
ax.scatter(samples_L2[:, 0], samples_L2[:, 1], samples_L2[:, 2], alpha=0.2, label='L2')
ax.scatter(samples_L3[:, 0], samples_L3[:, 1], samples_L3[:, 2], alpha=0.2, label='L3')
ax.set_title('Generated Sample Data')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

plt.show()


labels_L1 = np.array(['L1'] * len(samples_L1))
labels_L2 = np.array(['L2'] * len(samples_L2))
labels_L3 = np.array(['L3'] * len(samples_L3))

all_samples = np.concatenate((samples_L1, samples_L2, samples_L3))
all_labels = np.concatenate((labels_L1, labels_L2, labels_L3))

df = pd.DataFrame(all_samples, columns=['X', 'Y', 'Z'])
df['Label'] = all_labels

# Write to CSV
df.to_csv('question_2.csv', index=False)

# Part 2

# Bayesian classifier 
def bayes_classifier(row):
    x = row[["X", "Y", "Z"]].to_numpy()

    p_L1 = w1 * multivariate_normal.pdf(x, m1, CO1)
    p_L2 = w2 * multivariate_normal.pdf(x, m2, CO2)
    p_L3 = w3 * (multivariate_normal.pdf(x, m3, CO3) * weight1 + multivariate_normal.pdf(x, m3, CO4) * weight2)

    return 'L' + str(np.argmax([p_L1, p_L2, p_L3]) + 1)

df['Decision'] = df.apply(bayes_classifier, axis=1)

# Create Confusion Matrix
num_L1 = (df['Label'] == "L1").sum()
num_L2 = (df['Label'] == "L2").sum()
num_L3 = (df['Label'] == "L3").sum()

L11 = ((df['Label'] == "L1") & (df['Decision'] == 'L1')).sum() / num_L1
L12 = ((df['Label'] == "L1") & (df['Decision'] == "L2")).sum() / num_L1
L13 = ((df['Label'] == "L1") & (df['Decision'] == "L3")).sum() / num_L1

L21 = ((df['Label'] == "L2") & (df['Decision'] == "L1")).sum() / num_L2
L22 = ((df['Label'] == "L2") & (df['Decision'] == "L2")).sum() / num_L2
L23 = ((df['Label'] == "L2") & (df['Decision'] == "L3")).sum() / num_L2

L31 = ((df['Label'] == "L3") & (df['Decision'] == "L1")).sum() / num_L3
L32 = ((df['Label'] == "L3") & (df['Decision'] == "L2")).sum() / num_L3
L33 = ((df['Label'] == "L3") & (df['Decision'] == "L3")).sum() / num_L3

confusion_matrix = [[L11, L12, L13], [L21, L22, L23], [L31, L32, L33]]

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", 
            xticklabels=['Predicted L1', 'Predicted L2', 'Predicted L3'],
            yticklabels=['Actual L1', 'Actual L2', 'Actual L3'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


# Part 3

df['Accuracy'] = df['Label'] == df['Decision']
df.to_csv('data_2.csv', index=False)


marker_dict = {'L1': 'o', 'L2': 's', 'L3': '^'}
color_dict = {True: 'green', False: 'red'}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for index, row in df.iterrows():
    ax.scatter(row['X'], row['Y'], row['Z'], 
               marker=marker_dict[row['Label']], 
               color=color_dict[row['Accuracy']])

label_handles = [mlines.Line2D([], [], color='black', marker=marker_dict[label], linestyle='None', markersize=10, label=label) for label in marker_dict.keys()]

accuracy_handles = [mlines.Line2D([], [], color=color_dict[accuracy], marker='o', linestyle='None', markersize=10, label=str(accuracy)) for accuracy in color_dict.keys()]
ax.legend(handles=label_handles + accuracy_handles, loc='upper left', title='Labels & Accuracy')

ax.set_title('3D Scatter Plot of Labels with Accuracy Coloring')
ax.set_xlabel('X-Axis Label')
ax.set_ylabel('Y-Axis Label')
ax.set_zlabel('Z-Axis Label')

plt.show()

# Part B
loss_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
loss_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])

# Bayesian classifier with weighted losses
def bayes_classifier_with_loss(row, loss):
    x = row[["X", "Y", "Z"]].to_numpy()
    p_L1 = w1 * multivariate_normal.pdf(x, m1, CO1)
    p_L2 = w2 * multivariate_normal.pdf(x, m2, CO2)
    p_L3 = w3 * (multivariate_normal.pdf(x, m3, CO3) * weight1 + multivariate_normal.pdf(x, m3, CO4) * weight2)
    p = np.array([p_L1, p_L2, p_L3])
    risks = []
    for i in range(len(loss)):
        risk = 0
        for j in range(len(loss[i])):
            risk += loss[i][j] * p[j]
        risks.append(risk)
    
    return 'L' + str(np.argmin(risks) + 1)

df['Loss Decision: 10'] = df.apply(bayes_classifier_with_loss, args=(loss_10,), axis=1)
df['Loss Decision: 100'] = df.apply(bayes_classifier_with_loss, args=(loss_100,), axis=1)

L11 = ((df['Label'] == "L1") & (df['Loss Decision: 10'] == 'L1')).sum() / num_L1
L12 = ((df['Label'] == "L1") & (df['Loss Decision: 10'] == "L2")).sum() / num_L1
L13 = ((df['Label'] == "L1") & (df['Loss Decision: 10'] == "L3")).sum() / num_L1

L21 = ((df['Label'] == "L2") & (df['Loss Decision: 10'] == "L1")).sum() / num_L2
L22 = ((df['Label'] == "L2") & (df['Loss Decision: 10'] == "L2")).sum() / num_L2
L23 = ((df['Label'] == "L2") & (df['Loss Decision: 10'] == "L3")).sum() / num_L2

L31 = ((df['Label'] == "L3") & (df['Loss Decision: 10'] == "L1")).sum() / num_L3
L32 = ((df['Label'] == "L3") & (df['Loss Decision: 10'] == "L2")).sum() / num_L3
L33 = ((df['Label'] == "L3") & (df['Loss Decision: 10'] == "L3")).sum() / num_L3

confusion_matrix = [[L11, L12, L13], [L21, L22, L23], [L31, L32, L33]]

plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", 
            xticklabels=['Predicted L1', 'Predicted L2', 'Predicted L3'],
            yticklabels=['Actual L1', 'Actual L2', 'Actual L3'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap Loss 10')
plt.show()

L11 = ((df['Label'] == "L1") & (df['Loss Decision: 100'] == 'L1')).sum() / num_L1
L12 = ((df['Label'] == "L1") & (df['Loss Decision: 100'] == "L2")).sum() / num_L1
L13 = ((df['Label'] == "L1") & (df['Loss Decision: 100'] == "L3")).sum() / num_L1

L21 = ((df['Label'] == "L2") & (df['Loss Decision: 100'] == "L1")).sum() / num_L2
L22 = ((df['Label'] == "L2") & (df['Loss Decision: 100'] == "L2")).sum() / num_L2
L23 = ((df['Label'] == "L2") & (df['Loss Decision: 100'] == "L3")).sum() / num_L2

L31 = ((df['Label'] == "L3") & (df['Loss Decision: 100'] == "L1")).sum() / num_L3
L32 = ((df['Label'] == "L3") & (df['Loss Decision: 100'] == "L2")).sum() / num_L3
L33 = ((df['Label'] == "L3") & (df['Loss Decision: 100'] == "L3")).sum() / num_L3

confusion_matrix = [[L11, L12, L13], [L21, L22, L23], [L31, L32, L33]]

plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", 
            xticklabels=['Predicted L1', 'Predicted L2', 'Predicted L3'],
            yticklabels=['Actual L1', 'Actual L2', 'Actual L3'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap Loss 100')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
labels = ['L1', 'L2', 'L3']

for label, color in zip(labels, colors):
    subset = df[df['Loss Decision: 10'] == label]
    ax.scatter(subset['X'], subset['Y'], subset['Z'], label=label, s=50, c=color)
ax.set_title('Decisions with Loss 10')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for label, color in zip(labels, colors):
    subset = df[df['Loss Decision: 100'] == label]
    ax.scatter(subset['X'], subset['Y'], subset['Z'], label=label, s=50, c=color)
ax.set_title('Decisions with Loss 100')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

# Compute risk associated with each Decision
def compute_risks(row, loss):
    x = row[["X", "Y", "Z"]].to_numpy()

    p_L1 = w1 * multivariate_normal.pdf(x, m1, CO1)
    p_L2 = w2 * multivariate_normal.pdf(x, m2, CO2)
    p_L3 = w3 * (multivariate_normal.pdf(x, m3, CO3) * weight1 + multivariate_normal.pdf(x, m3, CO4) * weight2)
    p = np.array([p_L1, p_L2, p_L3])

    risks = []
    p_x = p_L1 + p_L2 + p_L3

    for i in range(len(loss)):
        risk = 0
        for j in range(len(loss[i])):
            risk += loss[i][j] * p[i]
        risks.append(risk / p_x)

    return min(risks)

df['Risk: Loss 10'] = df.apply(compute_risks, args=(loss_10,), axis=1)
average_risk = df['Risk: Loss 10'].mean()
print('Risk 10: ' + str(average_risk))


df['Risk: Loss 100'] = df.apply(compute_risks, args=(loss_100,), axis=1)
average_risk = df['Risk: Loss 100'].mean()
print('Risk 100: ' + str(average_risk))

