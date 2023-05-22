import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
import gradio as gr
import os

def read_signals(file, num_signals):
    """
    This function reads a file and separates the signals into a list of arrays based on the number of
    signals specified.
    
    :param file: The file parameter is a file object that contains the data to be read
    :param num_signals: The number of signals to be read from the file
    :return: The function `read_signals` returns a list of signals, where each signal is a
    one-dimensional numpy array. The number of signals returned is determined by the `num_signals`
    parameter.
    """
    
    # Read the file
    data = np.genfromtxt(file.name, delimiter=',')

    # Separate the signals
    signals = []
    for i in range(int(num_signals)):
        signal = data[:, i]
        signals.append(signal)

    return signals

def apply_low_pass_filter(signal, sampling_frequency, cutoff_freq):
    """
    This function applies a Butterworth low-pass filter to a signal with a specified cutoff frequency
    and sampling frequency.
    
    :param signal: The input signal that needs to be filtered
    :param sampling_frequency: The sampling frequency is the number of samples per second taken from a
    continuous signal to convert it into a discrete signal. It is usually measured in Hertz (Hz)
    :param cutoff_freq: The cutoff frequency is the frequency at which the filter starts to attenuate
    the signal. It is the frequency below which the filter allows all frequencies to pass through and
    above which it starts to attenuate the signal
    :return: the filtered signal after applying a Butterworth low-pass filter to the input signal.
    """
    # Design the Butterworth low-pass filter
    nyquist_freq = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = sig.butter(4, normalized_cutoff, btype='low', analog=False)

    # Apply the filter to the signal
    filtered_signal = sig.filtfilt(b, a, signal)

    return filtered_signal

def convert_to_voltage(signal, adc_range, voltage_range):
    """
    This function converts signal values to voltage using the ADC range and voltage range as inputs.
    
    :param signal: The signal parameter is the input value that needs to be converted to voltage. It is
    usually a digital value obtained from an analog-to-digital converter (ADC)
    :param adc_range: The range of the analog-to-digital converter (ADC) used to convert the analog
    signal to a digital signal. This is typically specified in bits, such as 8-bit or 12-bit ADCs, and
    determines the number of discrete levels the signal can be quantized into. For example,
    :param voltage_range: The range of voltage that the analog-to-digital converter (ADC) can measure.
    For example, if the ADC has a voltage range of 0-5V, then voltage_range would be 5
    :return: the signal values converted to voltage using the formula signal_voltage = signal *
    (voltage_range / adc_range).
    """
    # Convert signal values to voltage
    signal_voltage = signal * (voltage_range / adc_range)
    return signal_voltage

def plot_signal(signal, sampling_frequency, title, ylabel):
    """
    This function plots a signal with its corresponding time array and labels.
    
    :param signal: The signal to be plotted
    :param sampling_frequency: The sampling frequency is the number of samples per second taken from a
    continuous signal to convert it into a discrete signal. It is usually measured in Hertz (Hz)
    :param title: The title of the plot that will be displayed. It should be a string
    :param ylabel: The label for the y-axis of the plot, indicating the units of the signal being
    plotted. For example, if the signal is measuring voltage, the ylabel could be "Voltage (V)"
    """
    # Calculate the time array in seconds
    time = np.arange(len(signal)) / sampling_frequency

    # Plot the signal
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_time_window(signal, sampling_frequency, start_time, end_time):
    """
    This function plots a signal within a specified time window.
    
    :param signal: The signal is a one-dimensional array of voltage values representing a
    continuous-time signal
    :param sampling_frequency: The number of samples per second in the signal. It is usually measured in
    Hertz (Hz)
    :param start_time: The start time of the time window in seconds. This is the time at which the
    window will begin in the signal
    :param end_time: The end time of the time window in seconds. It is used to calculate the sample
    indices and the time array for the window, and to extract the corresponding signal values from the
    input signal
    """
    # Calculate the sample indices for the given time window
    start_index = int(start_time * sampling_frequency)
    end_index = int(end_time * sampling_frequency)

    # Extract the signal values for the time window
    window_signal = signal[start_index:end_index]

    # Calculate the time array in seconds for the window
    window_time = np.linspace(start_time, end_time, len(window_signal))

    # Plot the signal window
    plt.figure(figsize=(10, 6))
    plt.plot(window_time, window_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.title('Signal Window')
    plt.grid(True)
    plt.show()

def normalize_signal(signal):
    # Calculate the minimum and maximum values of the signal
    min_val = np.min(signal)
    max_val = np.max(signal)

    # Normalize the signal to values between 0 and 1
    normalized_signal = (signal - min_val) / (max_val - min_val)

    return normalized_signal

def calculate_windows_std_dev(signal, sampling_frequency, window_size, step_size):
    """
    This function calculates the standard deviation of a signal in windows of a specified size and step
    size.
    
    :param signal: The input signal for which the standard deviation needs to be calculated in windows
    :param sampling_frequency: The frequency at which the signal is sampled, measured in Hz (Hertz)
    :param window_size: The size of the window in seconds
    :param step_size: The step size is the amount of time between the start of each window. It is
    measured in seconds and determines how much overlap there is between adjacent windows
    :return: a list of standard deviations calculated for each window of the input signal, as well as a
    list of the starting indices of each window.
    """
    # Calculate the number of samples in the window and step
    window_samples = int(window_size * sampling_frequency)
    step_samples = int(step_size * sampling_frequency)

    # Calculate the standard deviation for each window
    std_devs = []
    window_starts = np.arange(0, len(signal) - window_samples + 1, step_samples)
    for start in window_starts:
        window_signal = signal[start:start+window_samples]
        std_dev = np.std(window_signal)
        std_devs.append(std_dev)
    
    return std_devs, window_starts

def classify_behavior(std_devs, threshold):
    """
    The function takes a list of standard deviations and a threshold value, and returns a list of
    behavior labels based on whether the standard deviation is below or above the threshold.
    
    :param std_devs: A list of standard deviations calculated from a set of data points
    :param threshold: The threshold is a value that is used to classify the behavior of an animal based
    on the standard deviation of its movement. If the standard deviation is below the threshold, the
    animal is classified as "freezing" (not moving much), and if it is above the threshold, the animal
    is classified as
    :return: The function `classify_behavior` returns a list of behavior labels, where each label is
    either 0 (indicating freezing behavior) or 1 (indicating moving behavior). The labels are determined
    based on whether the standard deviation of a given data set is below or above a specified threshold
    value.
    """
    behavior_labels = []
    for std_dev in std_devs:
        if std_dev < threshold:
            behavior_labels.append(0)  # Freezing
        else:
            behavior_labels.append(1)  # Moving

    return behavior_labels

def calculate_behavior_times(behavior_labels, window_size, sampling_frequency):
    """
    The function calculates the total time spent in moving and freezing behaviors based on behavior
    labels, window size, and sampling frequency.
    
    :param behavior_labels: A list of binary labels indicating whether an animal is moving (1) or
    freezing (0) during each window of time
    :param window_size: The size of the time window used to calculate the behavior times, in seconds
    :param sampling_frequency: The sampling frequency is the rate at which data is collected or sampled
    in a given time period. It is usually measured in Hertz (Hz) and represents the number of samples
    per second. In the context of this function, it is used to convert the total time spent in each
    behavior (measured
    :return: two values: `moving_time` and `freezing_time`.
    """
    moving_time = 0
    freezing_time = 0

    for i in range(len(behavior_labels)):
        if behavior_labels[i] == 1:
            moving_time += window_size
        else:
            freezing_time += window_size

    moving_time = moving_time / sampling_frequency
    freezing_time = freezing_time / sampling_frequency

    return moving_time, freezing_time

def plot_behavior_labels(signal, behavior_labels, sampling_frequency, window_size, step_size):
    """
    This function plots a signal with behavior labels as colored rectangles on the plot.
    
    :param signal: The signal is a one-dimensional numpy array containing the values of the signal over
    time
    :param behavior_labels: A list of binary labels indicating the presence (1) or absence (0) of a
    certain behavior in each window of the signal
    :param sampling_frequency: The frequency at which the signal is sampled, measured in Hz (samples per
    second)
    :param window_size: The size of the window in seconds used to segment the signal for labeling
    :param step_size: The step size is the time interval between consecutive windows in seconds. It is
    used to determine the overlap between adjacent windows when analyzing a signal
    :return: a matplotlib figure object.
    """
    time = np.arange(len(signal)) / sampling_frequency
    window_samples = int(window_size * sampling_frequency)
    step_samples = int(step_size * sampling_frequency)

    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.set_title('Signal with Behavior Labels')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal Value')

    for i, behavior in enumerate(behavior_labels):
        if behavior == 0:
            start_time = i * step_samples / sampling_frequency
            end_time = start_time + window_size
            ax.axvspan(start_time, end_time, color='red', alpha=0.3)
        else:
            start_time = i * step_samples / sampling_frequency
            end_time = start_time + window_size
            ax.axvspan(start_time, end_time, color='green', alpha=0.3)

    return fig

def calculate_freezing_time(behavior_labels, step_size):
    """
    The function calculates the freezing and moving time based on behavior labels and step size.
    
    :param behavior_labels: A list of binary values representing the behavior of an animal during a
    certain period of time. The value 0 represents freezing behavior and the value 1 represents moving
    behavior
    :param step_size: The time interval (in seconds) between each behavior label in the behavior_labels
    list
    :return: a tuple containing two values: the total time spent moving and the total time spent
    freezing, both calculated based on the input behavior labels and step size.
    """
    freezing_time = 0
    moving_time = 0
    
    for behavior in behavior_labels:
        if behavior == 0:  # Freezing
            freezing_time += step_size
        else:  # Moving
            moving_time += step_size

    return freezing_time, moving_time

def analyze_signal(input_file, num_signals, sampling_frequency, cutoff_frequency, window_size, step_size, threshold):
    """
    The function analyzes signals by applying a low pass filter, converting to voltage, normalizing,
    calculating standard deviations and behavior labels, plotting behavior labels, and calculating
    freezing and moving times.
    
    :param input_file: The file path of the input file containing the signals data
    :param num_signals: The number of signals to be analyzed
    :param sampling_frequency: The frequency at which the signal is sampled, measured in Hz (Hertz)
    :param cutoff_frequency: The cutoff frequency is the frequency at which the low-pass filter
    attenuates the input signal by half (-3dB) of its original amplitude. It is used to remove
    high-frequency noise from the signal
    :param window_size: The size of the window used for calculating the standard deviation of the signal
    :param step_size: The step size is the amount of time between each window in seconds when
    calculating the standard deviation and behavior labels for the signal
    :param threshold: The threshold is a value used to classify the behavior of the signal. If the
    standard deviation of a window is below the threshold, the behavior is classified as freezing. If
    the standard deviation is above the threshold, the behavior is classified as moving
    :return: four values: fig_1, fig_2, freezing_time_avg, and moving_time_avg.
    """
    signals = read_signals(input_file, num_signals)
    filtered_signals = []
    converted_signals = []
    normalized_signals = []
    signals_behavior_labels = []
    
    for signal in signals:
        filtered_signal = apply_low_pass_filter(signal, sampling_frequency, cutoff_frequency)
        filtered_signals.append(filtered_signal)
        converted_signal = convert_to_voltage(filtered_signal, 1023, 5)
        converted_signals.append(converted_signal)
        normalized_signal = normalize_signal(converted_signal)
        normalized_signals.append(normalized_signal)
        std_devs, window_starts = calculate_windows_std_dev(normalized_signal, sampling_frequency, window_size, step_size)
        behavior_labels = classify_behavior(std_devs, threshold)
        signals_behavior_labels.append(behavior_labels)
        moving_time, freezing_time = calculate_behavior_times(behavior_labels, window_starts, sampling_frequency)
    
    fig_1 = plot_behavior_labels(normalized_signals[0], signals_behavior_labels[0], sampling_frequency, window_size, step_size)
    fig_2 = plot_behavior_labels(normalized_signals[1], signals_behavior_labels[1], sampling_frequency, window_size, step_size)
    freezing_time_1, moving_time_1 = calculate_freezing_time(signals_behavior_labels[0], step_size)
    freezing_time_2, moving_time_2 = calculate_freezing_time(signals_behavior_labels[1], step_size)
    
    freezing_time_avg = (freezing_time_1 + freezing_time_2) / 2
    moving_time_avg = (moving_time_1 + moving_time_2) / 2
    
    return fig_1, fig_2, freezing_time_avg, moving_time_avg

def read_experiment_parameters(input_file):
    
    column_names = ['hour', 'minute', 'seconds', 'day', 'month', 'year', 'experiment_duration', 'experiment_day', 'animal_id', 'exploration_duration', 'tone_frequency', 'tone_duration', 'shock_duration', 'motion_recording_duration', 'wait_interval_duration', 'number_of_repetitions', 'experiment_context']
    
    df = pd.read_csv(input_file.name, names=column_names)
    
    # Here we order the hour, minute and seconds columns into a single column called time
    # If any of the values are less than 10, we add a 0 to the beginning of the value
    df['hour'] = df['hour'].apply(lambda x: '0' + str(x) if x < 10 else str(x))
    df['minute'] = df['minute'].apply(lambda x: '0' + str(x) if x < 10 else str(x))
    df['seconds'] = df['seconds'].apply(lambda x: '0' + str(x) if x < 10 else str(x))
    df['time'] = df['hour'].astype(str) + ':' + df['minute'].astype(str) + ':' + df['seconds'].astype(str)
    df['date'] = df['day'].astype(str) + '/' + df['month'].astype(str) + '/' + df['year'].astype(str)
    df.drop(['hour', 'minute', 'seconds', 'day', 'month', 'year'], axis=1, inplace=True)
    
    # Here we convert the experiment_duration column from milliseconds to seconds
    df['experiment_duration'] = df['experiment_duration'] / 1000
    
    parameter_list = list(df.columns)
    values_list = df.values.tolist()
    
    data = {"Parameter": parameter_list, "Values": values_list[0]}
    dataframe = pd.DataFrame(data)
    
    return dataframe
    
# The above code is creating a graphical user interface (GUI) for a motion signals analysis tool using
# the `gradio` library in Python. The GUI allows the user to upload a text file containing motion
# signals from PIR sensors, select the number of signals in the file, and enter various parameters for
# low-pass filter design and classification. The user can then click the "Analyze" button to calculate
# the freezing and moving times and generate plots of the signals. The results are displayed in
# textboxes and plot windows within the GUI.
with gr.Blocks(title="CERA - OpenFCC", css="footer {visibility: hidden}", theme=gr.themes.Soft(primary_hue="cyan")) as interface:
    gr.Markdown(
    """
    # Conditioning Experiment Results Analysis (CERA) Tool
    
    A tool for analyzing outuput files from the conditioning experiment and movement signals from the PIR sensors.
    
    Developed by Daniel Fernández (@defdzg), 2023.
    """)
    with gr.Tab(label="Experiment configuration parameters"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                """
                ### Experiment Parameters File
                """)
                input_experiment_file = gr.File(label = "Input File", info="Upload a text file containing the experiment parameters")
                btn_1 = gr.Button(value="Process", label="Process", info="Click to process the input file")
            with gr.Column():
                gr.Markdown(
                """
                ### Experiment Information
                """)
                table = gr.Dataframe(headers=["Parameter", "Values"])
                
        gr.Examples(
            [
                os.path.join(os.path.dirname(__file__), "data/CONFIG.TXT"),
            ],
            input_experiment_file
        )
        
    with gr.Tab(label="Movement signals analysis"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                """
                ### Signal File and Sampling Parameters
                """)
                input_pir_file = gr.File(label = "Input File", info="Upload a text file containing the signals")
                num_signals = gr.Dropdown(["1", "2", "3"], label="Number of Signals", value="2", info="Select the number of signals in the file")
                sampling_frequency = gr.Number(label="Sampling Frequency", value=100, info="Enter the sampling frequency of the signal")
            with gr.Column():
                gr.Markdown(
                """
                ### Low-Pass Filter Design and Classification Parameters
                """)
                cutoff_frequency = gr.Number(label="Cutoff Frequency", value=10, info="Enter the cutoff frequency of the low-pass filter")
                window_size = gr.Number(label="Window Size", value=5, info="Enter the window for the standard deviation calculation")
                step_size = gr.Number(label="Step Size", value=5, info="Enter the step size for the standard deviation calculation")
                threshold = gr.Number(label="Standard Deviation Threshold", value=0.05, info="Enter the standard deviation threshold for the classification")
                btn_2 = gr.Button(value="Analyze", label="Analyze", info="Click to analyze the signals")
     
        gr.Markdown(
            """
            ### Motion results and signal plots
            """)
        with gr.Row():
            with gr.Tab(label="Moving and Freezing Time"):
                with gr.Row():
                    with gr.Column():
                        freezing_time = gr.Textbox(label="Freezing Time")
                    with gr.Column():
                        moving_time = gr.Textbox(label="Moving Time")
            with gr.Tab(label="Signal Plots"):
                with gr.Row():
                    with gr.Column():
                        fig_1 = gr.Plot(label="Signal 1")
                    with gr.Column():
                        fig_2 = gr.Plot(label="Signal 2")
        
        gr.Examples(
            [
                os.path.join(os.path.dirname(__file__), "data/PIR1.TXT"),
                os.path.join(os.path.dirname(__file__), "data/PIR2.TXT"),
                os.path.join(os.path.dirname(__file__), "data/PIR3.TXT"),
            ],
            input_pir_file
        )
        
        btn_1.click(read_experiment_parameters,
                    inputs=[input_experiment_file],
                    outputs=[table])
        
        btn_2.click(analyze_signal,
                    inputs=[input_pir_file, num_signals, sampling_frequency, cutoff_frequency, window_size, step_size, threshold],
                    outputs=[fig_1, fig_2, freezing_time, moving_time])
        
        
        
interface.launch(favicon_path=os.path.join(os.path.dirname(__file__), "icon.png"))