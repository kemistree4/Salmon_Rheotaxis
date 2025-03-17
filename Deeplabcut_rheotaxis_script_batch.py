import pandas as pd
import numpy as np
from math import atan2, degrees, radians, cos, sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from pathlib import Path
from datetime import datetime

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

def process_csv(file_path, tilt_correction_angle):
    # Load the CSV file
    df = pd.read_csv(file_path, skiprows=2)

    # Extract Fish ID from the file path
    fish_id = os.path.basename(file_path)[:5]

    # Define column names
    nose_likelihood_col = 3
    dorsal_likelihood_col = 12
    nose_x_col = 'x'
    nose_y_col = 'y'
    dorsal_x_col = 'x.3'
    dorsal_y_col = 'y.3'

    # Define and rotate the reference vector
    reference_vector = np.array([0, 1])
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
            
            body_vector = rotate_vector(np.array([dorsal_x - nose_x, dorsal_y - nose_y]), tilt_correction_angle)
            
            angle_difference = angle_between(body_vector, rotated_reference_vector)
            
            if angle_difference > 180:
                angle_difference -= 360
            elif angle_difference < -180:
                angle_difference += 360
            
            results.append({
                'frame': index,
                'angle_difference': angle_difference + tilt_correction_angle
            })

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

    percentages = {threshold: (frames / total_frames) * 100 if total_frames > 0 else 0 
                   for threshold, frames in frames_within_threshold.items()}

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

    # Create and save the GIF
    create_gif(results_df, tilt_correction_angle, fish_id)

    return alignment_df

def create_gif(results_df, tilt_correction_angle, fish_id):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'ro-', lw=2)
    nose_dot, = ax.plot([], [], 'bo', markersize=10)

    reference_angle = 90 + tilt_correction_angle
    reference_radians = np.radians(reference_angle)
    reference_x = np.cos(reference_radians)
    reference_y = np.sin(reference_radians)

    ax.plot([0, reference_x], [0, reference_y], 'gray', linestyle='--', lw=2, label='Reference Vector')

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        if frame < len(results_df):
            angle_difference = results_df.iloc[frame]['angle_difference']
            
            fish_angle = reference_angle + angle_difference
            
            radians = np.radians(fish_angle)
            x = np.cos(radians)
            y = np.sin(radians)
            
            line.set_data([0, x], [0, y])
            nose_dot.set_data(0, 0)
            
            text.set_text(f'Frame: {results_df.iloc[frame]["frame"]}\nAngle Difference: {angle_difference:.2f}°')
        return line, nose_dot, text

    ani = animation.FuncAnimation(fig, update, frames=len(results_df), 
                                  interval=50, blit=True)

    ani.save(f'{fish_id}_orientation_angle_difference.gif', writer='pillow', fps=30)
    plt.close(fig)

def main(folder_path, tilt_correction_angle):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"alignment_results_{timestamp}.xlsx"
    
    folder = Path(folder_path)
    all_dfs = []
    
    for csv_file in folder.glob("*.csv"):
        print(f"Processing {csv_file.name}")
        df = process_csv(csv_file, tilt_correction_angle)
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    output_path = folder / output_filename
    final_df.to_excel(output_path, index=False)
    print(f"All CSV files processed and results saved to {output_path}")

# Specify the folder path containing the CSV files and the tilt correction angle
folder_path = "L:/Fish Tunnel Camera Data/Diet Study Conversions/Converted/Rotation/R0"
tilt_correction_angle = 0  # degrees

main(folder_path, tilt_correction_angle)