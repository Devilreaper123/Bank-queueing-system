import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang
import queue
from mpl_toolkits.mplot3d import Axes3D

# Your existing functions here (bank_queue_simulation, find_optimal_servers, etc.)


def bank_queue_simulation(customer_arrivals, num_primary_servers, num_experienced_servers, primary_service_time, experienced_service_time, service_time_std):
    waiting_times = []
    server_available_times = [0] * \
        (num_primary_servers + num_experienced_servers)
    customer_queue = queue.Queue()

    for arrival_time in customer_arrivals:
        customer_queue.put(arrival_time)  # Enqueue the customer

    current_time = 0
    while not customer_queue.empty():
        # Peek at the next customer in queue
        next_customer = customer_queue.queue[0]

        # Check for available server
        available_server = next((i for i, t in enumerate(
            server_available_times) if t <= current_time), None)
        if available_server is not None and next_customer <= current_time:
            customer_queue.get()  # Remove the customer from the queue
            service_time = np.random.normal(experienced_service_time if available_server <
                                            num_experienced_servers else primary_service_time, service_time_std)
            service_start_time = max(
                next_customer, server_available_times[available_server])
            waiting_times.append(service_start_time - next_customer)
            server_available_times[available_server] = service_start_time + service_time
            continue

        # Advance the current time if no server is available
        current_time += 1

    # Calculate average waiting time and max queue length
    average_waiting_time = np.mean(waiting_times) if waiting_times else 0
    max_queue_length = len(waiting_times)

    return average_waiting_time, max_queue_length


def find_optimal_servers(target_waiting_time, customer_arrivals, primary_service_time, experienced_service_time, service_time_std, max_primary_servers, max_experienced_servers):
    optimal_primary_servers = max_primary_servers
    optimal_experienced_servers = max_experienced_servers
    found_optimal = False

    # Iterate over the number of servers
    for primary_servers in range(1, max_primary_servers + 1):
        for experienced_servers in range(1, max_experienced_servers + 1):
            avg_waiting_time, _ = bank_queue_simulation(
                customer_arrivals, primary_servers, experienced_servers, primary_service_time, experienced_service_time, service_time_std)
            if avg_waiting_time <= target_waiting_time:
                optimal_primary_servers = primary_servers
                optimal_experienced_servers = experienced_servers
                found_optimal = True
                break
        if found_optimal:
            break

    return optimal_primary_servers, optimal_experienced_servers


def main():
    st.title("Bank Queue Management System")

    # Sidebar for input parameters
    st.sidebar.header("Simulation Parameters")
    operational_hours = st.sidebar.number_input(
        "Operational Hours", min_value=1, max_value=24, value=8)
    lambda_per_hour = st.sidebar.number_input(
        "Arrival Rate (λ) per hour", min_value=1.0, max_value=10.0, value=1.0)
    k = st.sidebar.number_input(
        "Shape Parameter (k) for Erlang Distribution", min_value=1, max_value=10, value=3)
    scale = 1 / lambda_per_hour

    num_primary_servers = st.sidebar.number_input(
        "Number of Primary Servers", min_value=1, max_value=20, value=5)
    num_experienced_servers = st.sidebar.number_input(
        "Number of Experienced Servers", min_value=1, max_value=20, value=3)
    primary_service_time = st.sidebar.number_input(
        "Mean Service Time for Primary Servers (seconds)", min_value=10, max_value=300, value=150)
    experienced_service_time = st.sidebar.number_input(
        "Mean Service Time for Experienced Servers (seconds)", min_value=10, max_value=300, value=100)
    service_time_std = st.sidebar.number_input(
        "Standard Deviation of Service Time (seconds)", min_value=5, max_value=100, value=30)

    target_average_waiting_time = st.sidebar.number_input(
        "Target Average Waiting Time (seconds)", min_value=10, max_value=300, value=120)
    max_primary_servers = st.sidebar.number_input(
        "Maximum Number of Primary Servers", min_value=1, max_value=20, value=10)
    max_experienced_servers = st.sidebar.number_input(
        "Maximum Number of Experienced Servers", min_value=1, max_value=20, value=10)

    # Simulation button
    if st.sidebar.button('Run Simulation'):

        # Generate customer arrivals
        size = int(operational_hours * lambda_per_hour * k)
        customer_interarrival_times = erlang.rvs(k, scale=scale, size=size)
        customer_arrival_times = np.cumsum(customer_interarrival_times)
        customer_arrival_times = customer_arrival_times[
            customer_arrival_times <= operational_hours * 3600]

        # Running the simulation
        average_waiting_time, max_queue_length = bank_queue_simulation(
            customer_arrival_times, num_primary_servers, num_experienced_servers, primary_service_time, experienced_service_time, service_time_std)

        optimal_primary, optimal_experienced = find_optimal_servers(
            target_average_waiting_time, customer_arrival_times, primary_service_time, experienced_service_time, service_time_std, max_primary_servers, max_experienced_servers)
        st.snow()
        st.success(f"Average Waiting Time: {average_waiting_time} seconds\n")
        st.success(f"Maximum Queue Length: {max_queue_length} customers\n")
        st.success(f"Optimized number of primary servers: {optimal_primary}\n")
        st.success(
            f"Optimized number of experienced servers: {optimal_experienced}")

        # Plotting
        col1, col2 = st.columns(2)

        # Average Waiting Time Plot
        with col1:
            primary_servers_range = range(1, max_primary_servers + 1)
            experienced_servers_range = range(1, max_experienced_servers + 1)
            average_waiting_times = np.zeros(
                (max_primary_servers, max_experienced_servers))

            for i, num_primary_servers in enumerate(primary_servers_range):
                for j, num_experienced_servers in enumerate(experienced_servers_range):
                    avg_waiting_time, _ = bank_queue_simulation(
                        customer_arrival_times, num_primary_servers, num_experienced_servers, primary_service_time, experienced_service_time, service_time_std)
                    average_waiting_times[i, j] = avg_waiting_time

            X, Y = np.meshgrid(primary_servers_range,
                               experienced_servers_range)
            Z = average_waiting_times.T

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel('No of Primary Servers')
            ax.set_ylabel('No of Experienced Servers')
            ax.set_zlabel('Average Waiting Time (seconds)')
            plt.title('Average Waiting Time vs. Server Composition')
            st.pyplot(fig)

        # Optimal Server Numbers Plot
        with col2:
            plt.figure()
            plt.plot(optimal_primary, optimal_experienced,
                     'ro')  # Red circle markers
            plt.xlim(0, max_primary_servers + 1)
            plt.ylim(0, max_experienced_servers + 1)
            plt.xlabel('No of Primary Servers')
            plt.ylabel('No of Experienced Servers')
            plt.title('Optimized Server Numbers')
            plt.grid(True)
            st.pyplot(plt)


if __name__ == "__main__":
    main()
