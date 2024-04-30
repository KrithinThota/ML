import pandas as pd
import os

def load_preprocess_data(data_dir):


  all_data = pd.DataFrame()

  for filename in os.listdir(data_dir):
    if filename.endswith(".txt") and not filename.startswith("SHA"):  # Ignore non-data files
      filepath = os.path.join(data_dir, filename)

      # Load data
      data = pd.read_csv(filepath, header=None)
      time = data[0]
      force_data = data.iloc[:, 1:19]

      left_foot_force = force_data.iloc[:, :8]
      right_foot_force = force_data.iloc[:, 8:]

      total_left_force = data[18]
      total_right_force = data[19]

      subject_type = filename.split("_")[1] 

      stride_times = []
      for i in range(1, len(total_left_force)):
        if total_left_force.iloc[i-1] > 0 and total_left_force.iloc[i] == 0:
          stride_times.append(time.iloc[i] - time.iloc[i-1])

      subject_data = pd.DataFrame({
          "time": time,
          "left_foot_force": left_foot_force.sum(axis=1),  # Sum of all left foot sensors
          "right_foot_force": right_foot_force.sum(axis=1),  # Sum of all right foot sensors
          "total_left_force": total_left_force,
          "total_right_force": total_right_force,
          "stride_time": stride_times,
          "subject_type": subject_type
      })

      all_data = pd.concat([all_data, subject_data], ignore_index=True)

  return all_data

data_dir = "path/to/your/data/directory"
preprocessed_data = load_preprocess_data(data_dir)

print(preprocessed_data.head())
