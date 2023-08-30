import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def randomwalk3D(n, angle_degrees, escape_radius=100):
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    angle_rad = np.radians(angle_degrees)
    current_direction = np.array([1, 0, 0])  # Initial direction (e.g., to the right)

    for i in range(1, n):
        # Calculate the distance from the origin (sun)
        distance = np.sqrt(x[i - 1]**2 + y[i - 1]**2 + z[i - 1]**2)

        if distance > escape_radius:
            # If outside the escape radius, move straight in the current direction
            x[i] = x[i - 1] + current_direction[0]
            y[i] = y[i - 1] + current_direction[1]
            z[i] = z[i - 1] + current_direction[2]
        else:
            # Generate a random reflection angle within the specified range
            reflection_angle = np.random.uniform(-angle_rad, angle_rad)

            # Generate a random axis of rotation (x, y, or z)
            axis = random.choice([0, 1, 2])

            # Create a 3D rotation matrix based on the chosen axis and angle
            rotation_matrix = np.identity(3)
            if axis == 0:
                rotation_matrix = np.dot(np.array([[1, 0, 0],
                                                   [0, np.cos(reflection_angle), -np.sin(reflection_angle)],
                                                   [0, np.sin(reflection_angle), np.cos(reflection_angle)]]), rotation_matrix)
            elif axis == 1:
                rotation_matrix = np.dot(np.array([[np.cos(reflection_angle), 0, np.sin(reflection_angle)],
                                                   [0, 1, 0],
                                                   [-np.sin(reflection_angle), 0, np.cos(reflection_angle)]]), rotation_matrix)
            else:
                rotation_matrix = np.dot(np.array([[np.cos(reflection_angle), -np.sin(reflection_angle), 0],
                                                   [np.sin(reflection_angle), np.cos(reflection_angle), 0],
                                                   [0, 0, 1]]), rotation_matrix)

            # Apply the rotation to the current direction
            current_direction = np.dot(rotation_matrix, current_direction)

            # Update the position
            x[i] = x[i - 1] + current_direction[0]
            y[i] = y[i - 1] + current_direction[1]
            z[i] = z[i - 1] + current_direction[2]

    return x, y, z

# Number of iterations
num_iterations = int(1e6)

# Lists to store escape times for each iteration
escape_times = []
iteration_numbers = []

# Define the sun_radius
sun_radius = int(1e2)

# 3D figure and axis for the entire plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# List to store the line objects for each frame
all_lines = []

# Function to initialize the animation
def init():
    for line in all_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return all_lines

# Function to calculate and update the average escape time
def update_average_escape(iteration):
    if len(escape_times) > 0:
        average_escape = sum(escape_times) / len(escape_times)
        average_escape_text.set_text(f'Average Counts to Escape: {average_escape:.2f}')
    else:
        average_escape_text.set_text('')
    return average_escape_text,

# Function to animate frames and define escape parameters
def animate(iteration):
    n_steps = int(1e3)
    reflection_angle_degrees = random.uniform(0, 180)
    x_data, y_data, z_data = randomwalk3D(n_steps, reflection_angle_degrees)

    distances = np.sqrt(x_data**2 + y_data**2 + z_data**2)
    escape_radius = int(1e2)
    escape_time = np.argmax(distances > escape_radius)

    escape_times.append(escape_time)
    iteration_numbers.append(iteration + 1)

    line, = ax.plot(x_data, y_data, z_data, '-', linewidth=0.5, alpha=0.5, color=np.random.rand(3,))
    all_lines.append(line)

    # Update the average escape text
    average_escape_text.set_text(f'Mean escape counts: {np.mean(escape_times):.2f}')

    return all_lines + [average_escape_text]

# Create a text annotation for displaying average escape time
average_escape_text = ax.text2D(0.005, 0.005, '', transform=ax.transAxes, fontsize=10, color='black')

# Create the animation
ani = FuncAnimation(fig, animate, frames=num_iterations, init_func=init, interval = 0.1, blit=True, repeat=False)

# Create a sphere to represent the Sun
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sun = sun_radius * np.outer(np.cos(u), np.sin(v))
y_sun = sun_radius * np.outer(np.sin(u), np.sin(v))
z_sun = sun_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sun, y_sun, z_sun, color='yellow', alpha=0.3)

# Define Plot
ax.set_title('Psudo Sun Simulator')

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_zlim(-200, 200)

plt.show()

# Create a histogram with 1000 bins
histogram = input("Histogram plot? (y or any other key for no): ")
if histogram == 'y':
    plt.figure()
    plt.hist(escape_times, bins=1000, color='blue', alpha=0.7)
    plt.xlabel('Escape Time')
    plt.ylabel('Frequency')
    plt.title('Escape Time Histogram')
    plt.show()
