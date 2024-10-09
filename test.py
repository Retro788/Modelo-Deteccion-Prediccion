import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import math
import numpy as np
import obspy
import os
import scipy
import scipy.signal
# Version 2.2: Set default resolution to 900x600, renamed value1 to trigger_level

# Create the root window
root = tk.Tk()
root.title("Rapid Seismic App")  # Change window name
root.geometry("900x600")

# Set modern theme colors
bg_color = "#1F1F1F"  # Dark background color
primary_color = "#007ACC"  # Blue for buttons and active elements
text_color = "#FFFFFF"  # White text color
accent_color = "#3B3B3B"  # Darker accent

# Configure root window with dark background
root.configure(bg=bg_color)

# Global variables to store the paths of the selected files
selected_mseed_path = None
# Static PNG file path (set to "test.png")
static_png_path = "test.png"  # Update to your actual PNG file path if necessary
original_img = None
fhd_img = None
def load_mseed_file(filepath):
    st = obspy.read(filepath)
    trace = st.traces[0].copy()

    times = trace.times()
    data = trace.data
    stats = trace.stats

    return times, data, stats
# Function for calculating positions of quakes and showing .png representation


# Create the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

# Apply the filter to the signal
def apply_bandpass_filter(yaxis, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y_filtered = scipy.signal.filtfilt(b, a, yaxis)
    return y_filtered


def func1():
    global selected_mseed_path
    trigger_level = int(trigger_level_entry.get())  
    window_size = int(window_size_entry.get())
    lowfreq = float(lowfreq_entry.get())
    highfreq = float(highfreq_entry.get())
    p = os.path.normpath(selected_mseed_path).split(os.path.sep)
    print(f"Function 1 input value: {trigger_level}")
    global original_img, fhd_img
    print(p)
    print(p[-2]+'\\'+p[-1])
    xaxis = []
    yaxis = []
    stats = []
    xaxis,yaxis,stats = load_mseed_file(p[-2]+'\\'+p[-1])

    # Define the bandpass filter parameters
    lowcut = 0.3   # Lower frequency bound (Hz)
    highcut = 7.0 # Upper frequency bound (Hz)
    fs = len(xaxis) / (xaxis[-1] - xaxis[0])  # Sampling frequency based on the time array
    y_filtered = apply_bandpass_filter(yaxis, lowfreq, highfreq, fs)
    # Trigger algorithm on raw data
    
    # Trigger algorithm on)
    fig, axs = plt.subplots(4,figsize=(18,11))
    axs[0].plot(xaxis, yaxis)
    axs[0].set(xlabel='time (s)', ylabel='amplitude (m/s)',title='original_data') #c/s or m/s
    axs[0].grid()
    axs[0].axhline(y = trigger_level, color = 'r', linestyle = '-')
    for i in range(len(xaxis)):
        if yaxis[i]>=trigger_level:
            axs[0].axvline(x = xaxis[i],color = 'g', linestyle = '-')

    axs[1].plot(xaxis, y_filtered)
    axs[1].set(xlabel='time (s)', ylabel='amplitude (m/s)',title='filtered_data') #c/s or m/s
    axs[1].grid()
    axs[1].axhline(y = trigger_level, color = 'r', linestyle = '-')
    for i in range(len(xaxis)):
        if y_filtered[i]>=trigger_level:
            axs[1].axvline(x = xaxis[i],color = 'g', linestyle = '-')


    # Trigger algorithm on window avg
    yaxis_abs = [abs(x) for x in y_filtered]
    xaxis_wavg = []
    yaxis_wavg = []
    for i in range(0,len(xaxis)-window_size):
        xaxis_wavg.append(xaxis[i])
        yaxis_wavg.append(sum(yaxis_abs[i:i+window_size])/window_size)
    axs[2].plot(xaxis_wavg, yaxis_wavg)
    axs[2].set(xlabel='time (s)', ylabel='amplitude (m/s)',title='window averge') #c/s or m/s
    axs[2].grid()
    axs[2].axhline(y = trigger_level, color = 'r', linestyle = '-')
    for i in range(len(xaxis_wavg)):
        if yaxis_wavg[i]>=trigger_level:
            axs[2].axvline(x = xaxis[i],color = 'g', linestyle = '-')
    while(len(xaxis_wavg)<len(xaxis)):
        xaxis_wavg.append(2*xaxis_wavg[-1]-xaxis_wavg[-2])
        yaxis_wavg.append(0)

    # Trigger algorithm on derivative

    yaxis_deriv = []
    yaxis_deriv.append(0)
    for i in range(1,len(xaxis)):
        yaxis_deriv.append(y_filtered[i]-y_filtered[i-1])

    axs[3].plot(xaxis, yaxis_deriv)
    axs[3].set(xlabel='time (s)', ylabel='amplitude (m/s)',title='derivative') #c/s or m/s
    axs[3].grid()
    axs[3].axhline(y = trigger_level, color = 'r', linestyle = '-')
    for i in range(len(xaxis)):
        if yaxis_deriv[i]>=trigger_level:
            axs[3].axvline(x = xaxis[i],color = 'g', linestyle = '-')
        
    # add .csv file saving
    fig.savefig("test.png",dpi = 300)
    try:
        # Load the static .png file and store the original image
        original_img = Image.open(static_png_path)

        # Compress the image to FHD and store it
        fhd_img = compress_to_fhd(original_img)

        # Display the compressed image, resized to fit the window
        resize_image()

        label.config(text=f"Function 1 executed: Loaded and resized the image with input {trigger_level}")
        print(f"Function 1 executed: Loaded and resized the image with input {trigger_level}")

    except Exception as e:
        label.config(text=f"Error loading image in func1: {e}")
        print(f"Error loading image in func1: {e}")

# Placeholder function 2 (also handles loading/resizing the .png file)
def func2():
    value = window_size_entry.get()  # Get the value from the input field for func2 when the button is pressed
    print(f"Function 2 input value: {value}")
    global original_img, fhd_img

    try:
        # Load the static .png file and store the original image
        original_img = Image.open(static_png_path)

        # Compress the image to FHD and store it
        fhd_img = compress_to_fhd(original_img)

        # Display the compressed image, resized to fit the window
        resize_image()

        label.config(text=f"Function 2 executed: Loaded and resized the image with input {value}")
        print(f"Function 2 executed: Loaded and resized the image with input {value}")

    except Exception as e:
        label.config(text=f"Error loading image in func2: {e}")
        print(f"Error loading image in func2: {e}")

# Function to open file explorer and get CSV file path
# Function to open file explorer and get MSEED file path
def open_mseed_explorer():
    global selected_mseed_path
    selected_mseed_path = filedialog.askopenfilename(filetypes=[("MSEED files", "*.mseed")])  # Open file dialog for .mseed files

    if selected_mseed_path:
        label.config(text=f"Selected MSEED File: {selected_mseed_path}")  # Display selected file
        print(f"Selected MSEED File: {selected_mseed_path}")
    else:
        label.config(text="No MSEED file selected")
        print("No MSEED file selected")
# Function to compress image to FHD (1920x1080) while maintaining aspect ratio
def compress_to_fhd(image):
    max_width, max_height = 1920, 1080
    width, height = image.size

    # If the image is larger than FHD, compress it while maintaining aspect ratio
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    # If the image is already within FHD, return it unchanged
    return image

# Function to resize image dynamically based on window size while maintaining aspect ratio
def resize_image(event=None):
    if fhd_img:
        # Get the current window size
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        # Get the original size of the FHD image
        img_width, img_height = fhd_img.size

        # Calculate the appropriate ratio to fit the image into the window, preserving aspect ratio
        ratio = min((window_width - 100) / img_width, (window_height - 200) / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # Resize the image based on the calculated size while maintaining the aspect ratio
        resized_img = fhd_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_img)  # Convert to Tkinter-compatible image

        # Set the image to the label
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection

    # Update the size label with the current window dimensions
    update_size_label()

# Function to update size label
def update_size_label():
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    size_label.config(text=f"[{window_width}] x [{window_height}]")

# Create a frame for the main content
main_frame = tk.Frame(root, bg=bg_color)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Frame to hold the buttons in a horizontal line
button_frame = tk.Frame(main_frame, bg=bg_color)  # Set background color for frame
button_frame.pack(pady=10)

# Button to trigger MSEED file explorer
btn_mseed = tk.Button(button_frame, text="Choose MSEED File", command=open_mseed_explorer, bg=primary_color, fg=text_color, font=("Helvetica", 12), relief=tk.FLAT)
btn_mseed.pack(side=tk.LEFT, padx=10)

# Button to execute func1 (now loads and resizes the .png)
btn_func1 = tk.Button(button_frame, text="Execute Function 1", command=func1, bg=primary_color, fg=text_color, font=("Helvetica", 12), relief=tk.FLAT)
btn_func1.pack(side=tk.LEFT, padx=10)

# Button to execute func2 (now loads and resizes the .png)
btn_func2 = tk.Button(button_frame, text="Execute Function 2", command=func2, bg=primary_color, fg=text_color, font=("Helvetica", 12), relief=tk.FLAT)
btn_func2.pack(side=tk.LEFT, padx=10)

# Frame for input fields (to position below buttons in a single row)
input_frame = tk.Frame(main_frame, bg=bg_color)
input_frame.pack(pady=10)

# Create input fields with labels for function inputs (aligned horizontally)
def create_input_field(parent, label_text, default_value):
    label = tk.Label(parent, text=label_text, bg=bg_color, fg=text_color, font=("Helvetica", 10))
    label.pack(side=tk.LEFT, padx=10)
    entry = tk.Entry(parent, bg=accent_color, fg=text_color, font=("Helvetica", 12), width=10, relief=tk.FLAT)
    entry.insert(0, default_value)  # Set default value
    entry.pack(side=tk.LEFT, padx=10)
    return entry

# Label and Entry field for trigger_level (renamed from value1) with default value 750
trigger_level_entry = create_input_field(input_frame, "Trigger Level", 750)

# Label and Entry field for Function 2 input value (value2) with default value 5
window_size_entry = create_input_field(input_frame, "window size", 5)

# Label and Entry field for additional Function 3 input value (value3) with default value 10
lowfreq_entry = create_input_field(input_frame, "lowfreq", 0.1)

# Label and Entry field for additional Function 4 input value (value4) with default value 15
highfreq_entry = create_input_field(input_frame, "highfreq", 10)

# Label to show selected file
label = tk.Label(main_frame, text="No file selected", bg=bg_color, fg=text_color, font=("Helvetica", 14))
label.pack(pady=20)

# Label to display image (initially empty)
image_label = tk.Label(main_frame, bg=bg_color)
image_label.pack(pady=20)

# Label to show current window size
size_label = tk.Label(main_frame, text="[900] x [600]", bg=bg_color, fg=text_color, font=("Helvetica", 12))
size_label.pack(pady=10)

# Bind the window resize event to the resize_image function
root.bind("<Configure>", lambda event: [resize_image(event), update_size_label()])

# Run the application
root.mainloop()
