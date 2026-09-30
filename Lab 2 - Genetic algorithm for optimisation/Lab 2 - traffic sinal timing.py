import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# TRAFFIC SIGNAL TIMING OPTIMIZATION
# USING GENETIC ALGORITHM
# ============================================================


# ============================================================
# 1. TRAFFIC SIGNAL DATA
# ============================================================

# Number of signal phases
NUM_PHASES = 4

# Names of the signal phases
PHASE_NAMES = [
    "North-South",
    "North-South Right Turn",
    "East-West",
    "East-West Right Turn"
]

# Number of vehicles arriving per hour
DEMAND = np.array([
    600,    # North-South
    300,    # North-South Right Turn
    800,    # East-West
    400     # East-West Right Turn
])

# Saturation flow rate
# Maximum vehicles that can pass per hour
# when the signal is continuously green
SATURATION_FLOW = np.array([
    1800,
    1800,
    1800,
    1800
])

# Minimum green time for each phase
MIN_GREEN = np.array([
    10,
    10,
    10,
    10
])

# Maximum green time for each phase
MAX_GREEN = np.array([
    60,
    60,
    60,
    60
])

# Yellow + all-red time per phase
LOST_TIME = 4

# Minimum and maximum allowed cycle time
MIN_CYCLE = 60
MAX_CYCLE = 180


# ============================================================
# 2. GENETIC ALGORITHM PARAMETERS
# ============================================================

POPULATION_SIZE = 100
GENERATIONS = 200

MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80


# ============================================================
# 3. CREATE ONE CHROMOSOME
# ============================================================

def create_individual():

    individual = []

    for i in range(NUM_PHASES):

        green_time = random.randint(
            MIN_GREEN[i],
            MAX_GREEN[i]
        )

        individual.append(green_time)

    return individual


# ============================================================
# 4. CREATE INITIAL POPULATION
# ============================================================

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        individual = create_individual()

        population.append(individual)

    return population


# ============================================================
# 5. CALCULATE CYCLE TIME
# ============================================================

def calculate_cycle_time(green_times):

    total_green = sum(green_times)

    total_lost_time = NUM_PHASES * LOST_TIME

    cycle_time = (
        total_green
        + total_lost_time
    )

    return cycle_time


# ============================================================
# 6. CALCULATE CAPACITY
# ============================================================

def calculate_capacity(
    green_time,
    cycle_time,
    saturation_flow
):

    capacity = (
        saturation_flow
        * green_time
        / cycle_time
    )

    return capacity


# ============================================================
# 7. CALCULATE TOTAL DELAY
# ============================================================

def calculate_delay(green_times):

    cycle_time = calculate_cycle_time(
        green_times
    )

    total_delay = 0

    for i in range(NUM_PHASES):

        green_time = green_times[i]

        demand = DEMAND[i]

        # ----------------------------------------------------
        # Calculate green ratio
        # ----------------------------------------------------

        green_ratio = (
            green_time / cycle_time
        )

        # ----------------------------------------------------
        # Calculate capacity
        # ----------------------------------------------------

        capacity = calculate_capacity(
            green_time,
            cycle_time,
            SATURATION_FLOW[i]
        )

        # ----------------------------------------------------
        # If demand exceeds capacity,
        # apply a very large penalty
        # ----------------------------------------------------

        if demand >= capacity:

            shortage = demand - capacity

            delay = (
                100000
                + shortage * 1000
            )

        else:

            # Degree of saturation
            x = demand / capacity

            # Approximate signal delay
            delay = (
                demand
                * cycle_time
                * (1 - green_ratio) ** 2
                /
                (
                    2
                    * (1 - min(x, 0.99))
                )
            )

        total_delay += delay

    # --------------------------------------------------------
    # Cycle time penalties
    # --------------------------------------------------------

    if cycle_time < MIN_CYCLE:

        penalty = (
            MIN_CYCLE - cycle_time
        ) * 10000

        total_delay += penalty

    if cycle_time > MAX_CYCLE:

        penalty = (
            cycle_time - MAX_CYCLE
        ) * 10000

        total_delay += penalty

    return total_delay


# ============================================================
# 8. TOURNAMENT SELECTION
# ============================================================

def tournament_selection(
    population,
    fitnesses,
    tournament_size=3
):

    # Select random individuals
    selected_indices = random.sample(
        range(len(population)),
        tournament_size
    )

    # Find the individual with
    # minimum delay
    winner_index = min(
        selected_indices,
        key=lambda i: fitnesses[i]
    )

    return population[winner_index].copy()


# ============================================================
# 9. CROSSOVER
# ============================================================

def crossover(parent1, parent2):

    # Select crossover point
    point = random.randint(
        1,
        NUM_PHASES - 1
    )

    # Create children
    child1 = (
        parent1[:point]
        + parent2[point:]
    )

    child2 = (
        parent2[:point]
        + parent1[point:]
    )

    return child1, child2


# ============================================================
# 10. MUTATION
# ============================================================

def mutation(
    individual,
    mutation_rate
):

    individual = individual.copy()

    for i in range(NUM_PHASES):

        if random.random() < mutation_rate:

            # Randomly change green time
            change = random.randint(
                -5,
                5
            )

            individual[i] += change

            # Keep within minimum
            # and maximum green time
            individual[i] = max(
                MIN_GREEN[i],
                individual[i]
            )

            individual[i] = min(
                MAX_GREEN[i],
                individual[i]
            )

    return individual


# ============================================================
# 11. GENETIC ALGORITHM
# ============================================================

def genetic_algorithm():

    # --------------------------------------------------------
    # Create initial population
    # --------------------------------------------------------

    population = create_population()

    # Best solution found so far
    best_solution = None

    best_fitness = float("inf")

    # Store best fitness of every generation
    history = []

    # --------------------------------------------------------
    # Start generations
    # --------------------------------------------------------

    for generation in range(GENERATIONS):

        # ----------------------------------------------------
        # Calculate fitness for every individual
        # ----------------------------------------------------

        fitnesses = []

        for individual in population:

            delay = calculate_delay(
                individual
            )

            fitnesses.append(delay)

        # ----------------------------------------------------
        # Find best individual in current generation
        # ----------------------------------------------------

        best_index = np.argmin(
            fitnesses
        )

        generation_best = (
            population[best_index]
        )

        generation_best_fitness = (
            fitnesses[best_index]
        )

        # ----------------------------------------------------
        # Update global best solution
        # ----------------------------------------------------

        if (
            generation_best_fitness
            < best_fitness
        ):

            best_fitness = (
                generation_best_fitness
            )

            best_solution = (
                generation_best.copy()
            )

        # Store best fitness
        history.append(
            best_fitness
        )

        # ----------------------------------------------------
        # Create next generation
        # ----------------------------------------------------

        new_population = []

        # ----------------------------------------------------
        # Elitism
        # Keep best solution
        # ----------------------------------------------------

        new_population.append(
            best_solution.copy()
        )

        # ----------------------------------------------------
        # Generate remaining individuals
        # ----------------------------------------------------

        while len(new_population) < POPULATION_SIZE:

            # Selection
            parent1 = tournament_selection(
                population,
                fitnesses
            )

            parent2 = tournament_selection(
                population,
                fitnesses
            )

            # Crossover
            if random.random() < CROSSOVER_RATE:

                child1, child2 = crossover(
                    parent1,
                    parent2
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            child1 = mutation(
                child1,
                MUTATION_RATE
            )

            child2 = mutation(
                child2,
                MUTATION_RATE
            )

            # Add child 1
            new_population.append(
                child1
            )

            # Add child 2
            if len(new_population) < POPULATION_SIZE:

                new_population.append(
                    child2
                )

        # Replace old population
        population = new_population

    return (
        best_solution,
        best_fitness,
        history
    )


# ============================================================
# 12. RUN THE GENETIC ALGORITHM
# ============================================================

best_solution, best_fitness, history = (
    genetic_algorithm()
)


# ============================================================
# 13. CALCULATE FINAL VALUES
# ============================================================

cycle_time = calculate_cycle_time(
    best_solution
)


# ============================================================
# 14. DISPLAY FINAL OUTPUT
# ============================================================

print()
print("=" * 90)

print(
    "          TRAFFIC SIGNAL TIMING OPTIMIZATION"
)

print(
    "                USING GENETIC ALGORITHM"
)

print("=" * 90)


print()
print("OPTIMIZED SIGNAL TIMING")

print("-" * 90)


# Table heading

print(
    f"{'Phase':<28}"
    f"{'Cars/hour':<15}"
    f"{'Green Time (sec)':<20}"
    f"{'Capacity (cars/hr)':<20}"
)

print("-" * 90)


# ------------------------------------------------------------
# Calculate totals
# ------------------------------------------------------------

total_demand = 0
total_capacity = 0


# ------------------------------------------------------------
# Display each phase
# ------------------------------------------------------------

for i in range(NUM_PHASES):

    green_time = best_solution[i]

    demand = DEMAND[i]

    capacity = calculate_capacity(
        green_time,
        cycle_time,
        SATURATION_FLOW[i]
    )

    total_demand += demand

    total_capacity += capacity

    print(
        f"{PHASE_NAMES[i]:<28}"
        f"{demand:<15}"
        f"{green_time:<20.2f}"
        f"{capacity:<20.2f}"
    )


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("-" * 90)

print(
    f"Total Traffic Demand       : "
    f"{total_demand} cars/hour"
)

print(
    f"Total Signal Cycle Time    : "
    f"{cycle_time:.2f} seconds"
)

print(
    f"Total Estimated Capacity   : "
    f"{total_capacity:.2f} cars/hour"
)

print(
    f"Estimated Total Delay      : "
    f"{best_fitness:.2f}"
)

print("=" * 90)


# ============================================================
# 15. DISPLAY INDIVIDUAL GREEN TIMES
# ============================================================

print()
print("GREEN TIME FOR EACH PHASE")
print("-" * 50)

for i in range(NUM_PHASES):

    print(
        f"{PHASE_NAMES[i]:<28}"
        f": {best_solution[i]:.2f} seconds"
    )

print("-" * 50)


# ============================================================
# 16. DISPLAY TRAFFIC LOAD
# ============================================================

print()
print("TRAFFIC LOAD ANALYSIS")
print("-" * 90)


for i in range(NUM_PHASES):

    green_time = best_solution[i]

    demand = DEMAND[i]

    capacity = calculate_capacity(
        green_time,
        cycle_time,
        SATURATION_FLOW[i]
    )

    utilization = (
        demand / capacity
    ) * 100

    print(
        f"{PHASE_NAMES[i]:<28}"
        f"Traffic = {demand:>4} cars/hr | "
        f"Green = {green_time:>6.2f} sec | "
        f"Capacity = {capacity:>8.2f} cars/hr | "
        f"Utilization = {utilization:>6.2f}%"
    )

print("-" * 90)


# ============================================================
# 17. CONVERGENCE GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    range(1, GENERATIONS + 1),
    history,
    color="blue",
    linewidth=2
)

plt.xlabel(
    "Generation"
)

plt.ylabel(
    "Best Total Delay"
)

plt.title(
    "Genetic Algorithm Convergence"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.6
)

plt.tight_layout()

plt.show()
