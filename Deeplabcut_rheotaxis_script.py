import pandas as pd
import numpy as np
from math import atan2, degrees, radians, cos, sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def angle_between(v1, v2):
    angle = atan2(np.linalg.det([v1,v2]), np.dot(v1,v2))
    return degrees(angle)

def rotate_vector(vector, angle_degrees):
    angle_radians = radians(angle_degrees)
    rotation_matrix = np.array([
        [cos(angle_radians), -sin(angle_radians)],
        [sin(angle_radians), cos(angle_radians)]
    ])
    return np.dot(rotation_matrix, vector)

def is_within_threshold(angle, threshold):
    return abs(angle) <= threshold

# Define the tilt correction angle (positive for counterclockwise rotation)
tilt_correction_angle = 6  # degrees

# Define column names
nose_likelihood_col = 3  # index of nose likelihood column
dorsal_likelihood_col = 12  # index of dorsal likelihood column
nose_x_col = 'x'
nose_y_col = 'y'
dorsal_x_col = 'x.3'
dorsal_y_col = 'y.3'

# Load the CSV file
file_path = "L:/Fish Tunnel Camera Data/Conversions/Converted/Ch262-TopDLC_resnet152_Chinook Sound StudyMar22shuffle1_100000.csv"
df = pd.read_csv(file_path, skiprows=2)

# Extract Fish ID from the file path
fish_id = os.path.basename(file_path)[:5]

# Check if required columns exist
required_cols = [nose_x_col, nose_y_col, dorsal_x_col, dorsal_y_col]
if not all(col in df.columns for col in required_cols):
    raise ValueError("One or more required columns are missing from the CSV file.")

# Define and rotate the reference vector
reference_vector = np.array([0, 1])  # Assuming forward is along the y-axis
rotated_reference_vector = rotate_vector(reference_vector, tilt_correction_angle)

results = []
alignment_frames = {45: None, 20: None, 10: None}
frames_within_threshold = {45: 0, 20: 0, 10: 0}
start_checking = False
frame_count = 0
total_frames = 0

for index, row in df.iterrows():
    nose_likelihood = row[df.columns[nose_likelihood_col]]
    dorsal_likelihood = row[df.columns[dorsal_likelihood_col]]
    
    if nose_likelihood > 0.60 and dorsal_likelihood > 0.60:
        nose_x, nose_y = row[nose_x_col], row[nose_y_col]
        dorsal_x, dorsal_y = row[dorsal_x_col], row[dorsal_y_col]
        
        # Rotate the body vector to correct for tilt
        body_vector = rotate_vector(np.array([dorsal_x - nose_x, dorsal_y - nose_y]), tilt_correction_angle)
        
        angle_difference = angle_between(body_vector, rotated_reference_vector)
        
        # Normalize the angle difference to be between -180 and 180 degrees
        if angle_difference > 180:
            angle_difference -= 360
        elif angle_difference < -180:
            angle_difference += 360
        
        results.append({
            'frame': index,
            'angle_difference': angle_difference + tilt_correction_angle
        })

        # Start checking for alignment after the 90th frame
        if index >= 90:
            start_checking = True

        if start_checking:
            frame_count += 1
            total_frames += 1
            for threshold in [45, 20, 10]:
                if alignment_frames[threshold] is None and is_within_threshold(angle_difference + tilt_correction_angle, threshold):
                    alignment_frames[threshold] = frame_count
                if is_within_threshold(angle_difference + tilt_correction_angle, threshold):
                    frames_within_threshold[threshold] += 1

results_df = pd.DataFrame(results)

# Calculate percentages
percentages = {threshold: (frames / total_frames) * 100 if total_frames > 0 else 0 
               for threshold, frames in frames_within_threshold.items()}

# Create the new DataFrame with alignment results
alignment_df = pd.DataFrame({
    'Fish_ID': [fish_id] * 3,
    'Threshold': [45, 20, 10],
    'Frames_to_Align': [
        alignment_frames[45] if alignment_frames[45] is not None else 'Not Aligned',
        alignment_frames[20] if alignment_frames[20] is not None else 'Not Aligned',
        alignment_frames[10] if alignment_frames[10] is not None else 'Not Aligned'
    ],
    'Frames_Within_Threshold': [
        frames_within_threshold[45],
        frames_within_threshold[20],
        frames_within_threshold[10]
    ],
    'Percentage_Within_Threshold': [
        f"{percentages[45]:.2f}%",
        f"{percentages[20]:.2f}%",
        f"{percentages[10]:.2f}%"
    ]
})

print("Results DataFrame:")
print(results_df)
print("\nAlignment DataFrame:")
print(alignment_df)

# Save the alignment DataFrame to a CSV file
alignment_df.to_csv('alignment_results.csv', index=False)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid(True)

# Create a line object for the body and a scatter object for the nose
line, = ax.plot([], [], 'ro-', lw=2)  # Red line for the body
nose_dot, = ax.plot([], [], 'bo', markersize=10)  # Blue dot for the nose

# Calculate the reference vector with tilt correction pointing downward
reference_angle = 90 + tilt_correction_angle
reference_radians = np.radians(reference_angle)
reference_x = np.cos(reference_radians)
reference_y = np.sin(reference_radians)

# Plot the reference vector as a gray line
ax.plot([0, reference_x], [0, reference_y], 'gray', linestyle='--', lw=2, label='Reference Vector')

# Initialize text object for displaying frame number and angle difference
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Animation update function
def update(frame):
    if frame < len(results_df):
        angle_difference = results_df.iloc[frame]['angle_difference']
        
        # Calculate the fish's body angle
        fish_angle = reference_angle + angle_difference
        
        radians = np.radians(fish_angle)
        x = np.cos(radians)
        y = np.sin(radians)
        
        # Update the line (body) and the nose dot
        line.set_data([0, x], [0, y])  # Line from nose to dorsal
        nose_dot.set_data(0, 0)  # Nose is at the origin (0,0)
        
        # Update the text for frame number and angle difference
        text.set_text(f'Frame: {results_df.iloc[frame]["frame"]}\nAngle Difference: {angle_difference:.2f}°')
    return line, nose_dot, text

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(results_df), 
                              interval=50, blit=True)

# Save the animation as a GIF
ani.save('fish_orientation_angle_difference.gif', writer='pillow', fps=30)

plt.show()