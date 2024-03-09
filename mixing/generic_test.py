import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Generate x and y coordinates
x = np.arange(512)
y = np.arange(512)

# Generate a grid of x and y coordinates
X, Y = np.meshgrid(x, y)

# Generate a single sinusoid in the x direction
sinusoid_x = np.sin(2 * np.pi * 0.1 * X)

# Generate aperiodic noise in the y direction
aperiodic_y = np.random.rand(512, 512)

# Combine the sinusoid and aperiodic noise to create the image
image = sinusoid_x + aperiodic_y

# Display the image
plt.imshow(image, cmap='gray')
plt.title('Sample Image:\nSinusoid in X Direction,\nAperiodic in Y Direction')
plt.colorbar()
plt.axis('off')
plt.show()

# subtract the mean from the image
mean_subtracted_image = image - np.mean(image)

frequencies_x, psd_x = signal.welch(mean_subtracted_image, axis=0, nperseg=256)

frequencies_y, psd_y = signal.welch(mean_subtracted_image, axis=1, nperseg=256)

# Plot the power spectral density along the X axis
plt.figure(figsize=(8, 4))
plt.plot(frequencies_x, psd_x.sum(axis=1))
plt.title('Power Spectral Density (X axis)')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()

# Plot the power spectral density along the Y axis
plt.figure(figsize=(8, 4))
plt.plot(frequencies_y, psd_y.sum(axis=0).T)
plt.title('Power Spectral Density (Y axis)')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()
