from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd

random.seed(24) #To ensure coherance between reruns

class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None

# Agents
class Otter(BaseAgent):
    def __init__(self, unique_id, model):
        BaseAgent.__init__(self, unique_id, model)
        self.energy = 16
        self.reproduce_threshold = 15
        self.base_reproduction_prob = 0.25  # Otter reproduction probability
        self.reproduction_prob = self.base_reproduction_prob  # Current adjusted probability
        self.sense_radius = 1
        # Temperature parameters (Fahrenheit)
        self.optimal_temp_min = 50.0 
        self.optimal_temp_max = 60.0  
        self.critical_temp = 65.0    # Critical high temp

    def update_reproduction_rate(self):
        '''
        Adjusts reproduction probability based on the sea temperature.
        '''
        current_temp = self.model.climate.temperature
        
        if current_temp <= self.optimal_temp_max:
            # Normal reproduction rate within optimal temperature range
            self.reproduction_prob = self.base_reproduction_prob
        else:
            # Linear decrease in reproduction as temperature rises above optimal
            temp_factor = max(0.1, 1.0 - 0.15 * (current_temp - self.optimal_temp_max) / 
                              (self.critical_temp - self.optimal_temp_max))
            self.reproduction_prob = self.base_reproduction_prob * temp_factor

    def move(self):
        """
        Move towards the nearest urchin if one is within range, otherwise move randomly.
        """
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=self.sense_radius)
        urchin_locations = [cell for cell in neighborhood if any(isinstance(obj, Urchin) for obj in self.model.grid.get_cell_list_contents([cell]))]
        
        if urchin_locations:
            # Move towards the nearest urchin
            new_position = min(urchin_locations, key=lambda pos: (pos[0] - self.pos[0])**2 + (pos[1] - self.pos[1])**2)
        else:
            # No urchins in range, move about randomly
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_steps)
        
        self.model.grid.move_agent(self, new_position)

    def eat(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        urchins = [obj for obj in cellmates if isinstance(obj, Urchin)]
        if urchins:
            urchin = random.choice(urchins)
            self.energy += 7
            self.model.grid.remove_agent(urchin)
            self.model.schedule.remove(urchin)
            self.model.num_urchins -= 1

    def reproduce(self):
        # Update reproduction rate based on current temperature
        self.update_reproduction_rate()
        
        if self.energy >= self.reproduce_threshold and random.random() < self.reproduction_prob:
            self.energy /= 2
            offspring = Otter(next(self.model.next_id), self.model)
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.append(offspring)
            self.model.num_otters += 1

    def step(self):
        self.move()
        self.eat()
        self.reproduce()
        
        # Increased energy expenditure in warmer temperatures
        current_temp = self.model.climate.temperature
        base_energy_cost = 1.2
        if current_temp > self.optimal_temp_max:
            # Increased metabolic cost in warmer water
            temp_factor = 1.0 + 0.1 * (current_temp - self.optimal_temp_max) / (self.critical_temp - self.optimal_temp_max)
            energy_cost = base_energy_cost * temp_factor
        else:
            energy_cost = base_energy_cost
            
        self.energy -= energy_cost
        
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            self.model.num_otters -= 1

class Urchin(BaseAgent):
    def __init__(self, unique_id, model):
        BaseAgent.__init__(self, unique_id, model)
        self.base_reproduce_prob = 0.3
        self.reproduce_prob = self.base_reproduce_prob 
        self.energy = 14
        self.reproduce_threshold = 4  
        
        # Temperature parameters
        self.optimal_temp_min = 55.0
        self.optimal_temp_max = 68.0  
        self.critical_temp = 75.0
        
    def update_reproduction_rate(self):
        """Adjusts reproduction rate based on water temperature."""
        current_temp = self.model.climate.temperature
        
        if current_temp < self.optimal_temp_min:
            # Below optimal temperature: reduced reproduction
            temp_factor = 0.5 + 0.5 * (current_temp - (self.optimal_temp_min - 10)) / 10
            self.reproduce_prob = self.base_reproduce_prob * max(0.2, temp_factor)
        elif self.optimal_temp_min <= current_temp <= self.optimal_temp_max:
            # Optimal range: enhanced reproduction
            # Curve with peak at middle of the optimal range
            optimal_mid = (self.optimal_temp_min + self.optimal_temp_max) / 2
            temp_factor = 1.0 + 0.5 * (1 - ((current_temp - optimal_mid) / 
                                           ((self.optimal_temp_max - self.optimal_temp_min) / 2))**2)
            self.reproduce_prob = self.base_reproduce_prob * temp_factor
        else:
            # Above optimal range: rapidly declining reproduction
            temp_factor = max(0.1, 1.0 - ((current_temp - self.optimal_temp_max) / 
                                         (self.critical_temp - self.optimal_temp_max)))
            self.reproduce_prob = self.base_reproduce_prob * temp_factor
        
    def eat(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        kelp_plants = [obj for obj in cellmates if isinstance(obj, Kelp)]
        if kelp_plants:
            kelp = random.choice(kelp_plants)
            self.energy += 7  # Gain energy from eating kelp
            self.model.grid.remove_agent(kelp)
            self.model.schedule.remove(kelp)
            self.model.num_kelp -= 1
    
    def reproduce(self):
        # Updates the reproduction rate based on the current temperature
        self.update_reproduction_rate()
        
        if self.energy >= self.reproduce_threshold and random.random() < self.reproduce_prob:
            self.energy /= 2  # Splits energy with offspring
            offspring = Urchin(next(self.model.next_id), self.model)
            offspring.energy = self.energy  # Gives offspring same energy
            self.model.grid.place_agent(offspring, self.pos)
            self.model.schedule.append(offspring)
            self.model.num_urchins += 1

    def step(self):
        # Adjust movement based on temperature (more active in warmer water)
        current_temp = self.model.climate.temperature
        move_probability = 0.8  # Base probability of movement
        
        if current_temp > self.optimal_temp_min:
            # More active in warmer water until the critical temperature
            if current_temp <= self.optimal_temp_max:
                move_probability = min(1.0, 0.8 + 0.2 * (current_temp - self.optimal_temp_min) / 
                                     (self.optimal_temp_max - self.optimal_temp_min))
            else:
                # Decline in movement as temperature becomes too high
                move_probability = max(0.5, 1.0 - 0.5 * (current_temp - self.optimal_temp_max) / 
                                     (self.critical_temp - self.optimal_temp_max))
        
        if random.random() < move_probability:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
        
        # Eating takes priority over reproduction
        self.eat()
        self.reproduce()
        
        # Adjust metabolism (energy consumption) based on temperature
        current_temp = self.model.climate.temperature
        base_energy_cost = 0.5
        
        if current_temp < self.optimal_temp_min:
            # Lower metabolism in cooler water
            energy_cost = base_energy_cost * 0.8
        elif current_temp <= self.optimal_temp_max:
            # Normal metabolism in optimal range
            energy_cost = base_energy_cost
        else:
            # Higher metabolism in warmer water
            energy_cost = base_energy_cost * (1.0 + 0.3 * (current_temp - self.optimal_temp_max) / 
                                            (self.critical_temp - self.optimal_temp_max))
        
        self.energy -= energy_cost
        
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            self.model.num_urchins -= 1

class Kelp(BaseAgent):
    def __init__(self, unique_id, model):
        BaseAgent.__init__(self, unique_id, model)
        self.base_growth_prob = 0.18  # Base growth probability
        self.growth_prob = self.base_growth_prob  # Current adjusted probability
        
        # Temperature parameters for kelp
        self.optimal_temp_min = 50.0
        self.optimal_temp_max = 60.0  
        self.critical_temp = 68.0     
    
    def update_growth_rate(self):
        """Adjusts growth probability based on the water temperature."""
        current_temp = self.model.climate.temperature
        
        if current_temp < self.optimal_temp_min:
            # Below optimal temp: reduced growth
            temp_factor = max(0.3, (current_temp - (self.optimal_temp_min - 10)) / 10)
            self.growth_prob = self.base_growth_prob * temp_factor
        elif self.optimal_temp_min <= current_temp <= self.optimal_temp_max:
            # Optimal temperature range: enhanced growth
            # Calculate position within the optimal range (0-1)
            range_position = (current_temp - self.optimal_temp_min) / (self.optimal_temp_max - self.optimal_temp_min)
            # Curve with peak at middle of optimal range
            temp_factor = 1.0 + 0.3 * (1 - 4 * (range_position - 0.5)**2)
            self.growth_prob = self.base_growth_prob * temp_factor
        else:
            # Above optimal: rapidly declining growth
            # Linear decrease to near-zero at critical temperature
            temp_factor = max(0.05, 1.0 - ((current_temp - self.optimal_temp_max) / 
                                         (self.critical_temp - self.optimal_temp_max))**1.5)
            self.growth_prob = self.base_growth_prob * temp_factor

    def step(self):
        # Updates the growth rate based on current temperature
        self.update_growth_rate()
        
        # Calculate the current kelp density
        total_cells = self.model.grid.width * self.model.grid.height
        current_density = self.model.num_kelp / total_cells
        
        # Adjust growth probability based on density
        adjusted_growth_prob = self.growth_prob
        if current_density > 0.2:  # If more than 20% of the grid is covered
            # Reduce probability as density increases
            reduction_factor = 1 - ((current_density - 0.15) / 0.85)
            adjusted_growth_prob *= max(0.1, reduction_factor)  # Ensure it doesn't go too close to zero
        
        # Checks if kelp dies due to extreme temperature
        current_temp = self.model.climate.temperature
        if current_temp > self.critical_temp:
            # Chance of death increases as temperature rises above critical
            death_prob = min(0.8, (current_temp - self.critical_temp) / 10)
            if random.random() < death_prob:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                self.model.num_kelp -= 1
                return  # Exit early if the kelp died
        
        # Use the adjusted probability for reproduction
        if random.random() < adjusted_growth_prob:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            empty_cells = [cell for cell in possible_steps if self.model.grid.is_cell_empty(cell)]
            if empty_cells:
                offspring = Kelp(next(self.model.next_id), self.model)
                pos = random.choice(empty_cells)
                self.model.grid.place_agent(offspring, pos)
                self.model.schedule.append(offspring)
                self.model.num_kelp += 1

class Climate:
    def __init__(self, initial_temp, warming_rate_per_year=0.036, step_per_year=365):
        """
        Initializes climate with:
        - initial_temp: Starting temperature from dataset.
        - warming_rate_per_year: °F increase per year (default 0.036°F/year).
        - step_per_year: Number of steps per simulated year (default 365 for daily steps for year).
        """
        self.temperature = initial_temp
        self.warming_rate = warming_rate_per_year / step_per_year  # Adjusts warming rate per step
        self.steps_per_year = step_per_year

    def step(self):
        """Increases temperature gradually based on the step scale."""
        self.temperature += self.warming_rate

class KelpForestModel(Model):
    def __init__(self, width, height, num_otters, num_urchins, num_kelp, climate):
        self.width = width
        self.height = height
        self.num_otters = num_otters
        self.num_urchins = num_urchins
        self.num_kelp = num_kelp
        self.climate = climate
        self.running = True
        
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = []
        
        # Creates a generator for unique IDs
        self.next_id = iter(range(1000000))
        
        # Create agents
        for _ in range(self.num_otters):
            otter = Otter(next(self.next_id), self)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(otter, (x, y))
            self.schedule.append(otter)
            
        for _ in range(self.num_urchins):
            urchin = Urchin(next(self.next_id), self)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(urchin, (x, y))
            self.schedule.append(urchin)
            
        for _ in range(self.num_kelp):
            kelp = Kelp(next(self.next_id), self)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(kelp, (x, y))
            self.schedule.append(kelp)

        self.datacollector = DataCollector(
            model_reporters={"Otters": lambda m: m.num_otters, 
                            "Urchins": lambda m: m.num_urchins, 
                            "Kelp": lambda m: m.num_kelp}
        )

    def step(self):
        self.climate.step() 
        self.datacollector.collect(self)
        agents = self.schedule.copy()
        random.shuffle(agents)
        for agent in agents:
            if agent in self.schedule:
                agent.step()
    
    def get_grid_state(self):
        """Return a grid representation for visualisation"""
        grid_state = np.zeros((self.height, self.width))
        
        for cell in self.grid.coord_iter():
            cell_content, (x, y) = cell
            # Priority: Otter > Urchin > Kelp
            for agent in cell_content:
                if isinstance(agent, Otter):
                    grid_state[y][x] = 3  
                    break
                elif isinstance(agent, Urchin):
                    grid_state[y][x] = 2  
                    break
                elif isinstance(agent, Kelp):
                    grid_state[y][x] = 1
        return grid_state

       
# Loading the real data from the CSV
data_path = "/Users/ellisdown/Downloads/MDM2.2/Finalish Stuff/MDM Data Summary.csv"
df = pd.read_csv(data_path)

#Dimensing the ABM grid
width = 120
height = 120

grid_area_m2 = width * height * 100  # Each cell is 10x10 meters
real_area_m2 = 184633000  # Total survey area in square meters
scaling_factor = grid_area_m2 / real_area_m2

year = 1994
num_otters = round(df.iloc[year-1985]['Otter Population'] * scaling_factor)
num_urchins = round(df.iloc[year-1985]['Urchins in Otter Area'] * scaling_factor * 0.015) #Assuming 1.5% of Urchins are present on grid at one time
num_kelp = round((df.iloc[year-1985]['Kelp Coverage'] * (width * height)) / 100) #Scale to grid size
sea_temperature = float(df.iloc[year-1985]['Sea Temp (F)'])

# Initialise climate with real-world temperature from dataset and apply warming
climate = Climate(initial_temp=sea_temperature, warming_rate_per_year=0.02)

# Creates and runs the model
model = KelpForestModel(width, height, num_otters, num_urchins, num_kelp, climate)

print("Starting values for species", num_otters, num_urchins, num_kelp)

# Sets up the figure for animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Two plots: Grid & Population Dynamics
fig.suptitle('Kelp Forest Ecosystem Simulation')

# Colourmap for the grid: ocean, kelp, urchin, otter
colours = ['lightskyblue', 'green', 'purple', 'brown']
cmap = ListedColormap(colours)

# Initialise the grid plot
grid_plot = ax1.imshow(model.get_grid_state(), cmap=cmap, vmin=0, vmax=3)
ax1.set_title('Ecosystem Grid')
ax1.set_xticks([])
ax1.set_yticks([])

# Create a legend with a placeholder for temperature
patches = [plt.Rectangle((0, 0), 1, 1, color=colours[i]) for i in range(len(colours))]
legend = ax1.legend(patches, ['Ocean', 'Kelp', 'Urchin', 'Otter', 'Temp: --°F'], 
                    loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# Population dynamics plot
max_steps = 1000
steps = np.arange(max_steps)
otter_line, = ax2.plot([], [], 'brown', label='Otters')
urchin_line, = ax2.plot([], [], 'purple', label='Urchins')
kelp_line, = ax2.plot([], [], 'green', label='Kelp')

ax2.set_xlim(0, max_steps)
ax2.set_ylim(0, 300)  # Adjust based on expected population ranges
ax2.set_xlabel('Steps')
ax2.set_ylabel('Population')
ax2.set_title('Population Dynamics')

legend1 = ax2.legend(loc="upper left")  # First legend for Otters, Urchins, and Kelp
ax2.add_artist(legend1)  # Add to figure

temperature_text = ax2.text(0.75, 0.95, f"Temp: {model.climate.temperature:.2f}°F", 
                            transform=ax2.transAxes, fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.7))

# Initialise the data storage
population_data = {
    'Otters': [model.num_otters],
    'Urchins': [model.num_urchins],
    'Kelp': [model.num_kelp],
    'Temperature': [model.climate.temperature]
}

# Animation function
def update(frame):
    # Step the model
    model.step()
    print(f"Step {frame}: Temperature = {model.climate.temperature:.2f}°F")
    # Updates the grid visualisation
    grid_plot.set_array(model.get_grid_state())
    
    # Update population data
    population_data['Otters'].append(model.num_otters)
    population_data['Urchins'].append(model.num_urchins)
    population_data['Kelp'].append(model.num_kelp)
    population_data['Temperature'].append(model.climate.temperature)
    
    # Updates the population plot
    current_step = len(population_data['Otters'])
    otter_line.set_data(range(current_step), population_data['Otters'])
    urchin_line.set_data(range(current_step), population_data['Urchins'])
    kelp_line.set_data(range(current_step), population_data['Kelp'])
    
    # Adjusts the y-axis limit
    max_pop = max(max(population_data['Otters']), max(population_data['Urchins']), max(population_data['Kelp']))
    if max_pop > ax2.get_ylim()[1]:
        ax2.set_ylim(0, max_pop * 1.1)  # Gives 10% headroom for readability
    
    temperature_text.set_text(f"Temp: {model.climate.temperature:.2f}°F")
    
    # Stops if any species goes extinct
    if model.num_otters == 0 or model.num_urchins == 0 or model.num_kelp == 0:
        ani.event_source.stop()
        plt.text(0.5, 0.5, "Extinction occurred!", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=15, color='red')
    
    return grid_plot, otter_line, urchin_line, kelp_line, temperature_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=max_steps-1, 
                              interval=1, blit=False)

plt.tight_layout()
plt.show()